// Lean compiler output
// Module: Mathlib.Tactic.Convert
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Notation public import Mathlib.Tactic.CongrExclamation meta import Mathlib.Tactic.CongrExclamation
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
extern lean_object* l_Lean_Elab_abortTermExceptionId;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_EvalExpr_instNat;
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_instOption___redArg(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_instEvalTermTransparencyMode_evalTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalOptionStx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_Meta_mkEqMP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lp_mathlib_Lean_MVarId_congrN_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqSymm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkSort(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Congr!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Convert"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "CheapConfig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 213, 32, 144, 36, 211, 146, 239)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(48, 235, 87, 242, 156, 8, 67, 27)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2;
static const lean_closure_object lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "TransparencyMode"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(245, 50, 227, 172, 92, 117, 235, 109)}};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "postTransparency"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "sameFun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "transparency"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "typeEqs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "useCongrSimp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(245, 138, 49, 219, 58, 250, 114, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(180, 115, 71, 170, 239, 203, 79, 56)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(232, 211, 119, 253, 240, 22, 19, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(171, 155, 9, 57, 171, 87, 163, 52)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "preTransparency"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "preferLHS"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11_value),LEAN_SCALAR_PTR_LITERAL(97, 172, 109, 102, 7, 221, 190, 19)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(126, 219, 101, 4, 164, 64, 157, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(147, 170, 13, 6, 29, 92, 170, 151)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "etaExpand"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "maxArgs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "partialApp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18_value),LEAN_SCALAR_PTR_LITERAL(135, 25, 130, 0, 77, 106, 143, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(190, 149, 172, 157, 219, 42, 233, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(240, 32, 215, 10, 151, 247, 98, 112)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "beqEq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "closePost"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "closePre"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(245, 77, 16, 123, 74, 227, 132, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(23, 238, 240, 239, 72, 122, 89, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(253, 203, 138, 62, 42, 242, 102, 101)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(102, 141, 186, 5, 225, 200, 16, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22_value),LEAN_SCALAR_PTR_LITERAL(127, 134, 159, 198, 204, 152, 186, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Convert_elabCheapConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Convert_elabCheapConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "ExpensiveConfig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 213, 32, 144, 36, 211, 146, 239)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(90, 173, 29, 124, 175, 90, 0, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___closed__0_value;
static lean_once_cell_t lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Convert_elabExpensiveConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Convert_elabExpensiveConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Convert_elabConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 16, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 1, 2, 3, 1, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Convert_elabConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Convert_elabConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(146, 109, 21, 40, 70, 113, 251, 6)}};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Lean_MVarId_convert___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(183, 66, 254, 161, 210, 133, 94, 78)}};
static const lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_MVarId_convert___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convert"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__2_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 234, 179, 92, 162, 248, 1)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = " ←"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__18_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__22_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__28_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__27_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__33;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__36_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__38_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rintroPat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__42_value),LEAN_SCALAR_PTR_LITERAL(105, 195, 203, 253, 3, 13, 142, 19)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__43_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__41_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__44_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__37_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__45_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__46_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__47_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__48_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__49;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert___closed__50;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_convert;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "convert!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(54, 209, 4, 240, 28, 59, 246, 129)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert_x21___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_convert_x21;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "using"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabTermForConvert___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__2_value),LEAN_SCALAR_PTR_LITERAL(108, 214, 37, 22, 78, 158, 100, 157)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabTermForConvert___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convertTo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "convertTo"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__0_value),LEAN_SCALAR_PTR_LITERAL(76, 171, 229, 230, 99, 95, 29, 250)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convertTo___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "convert_to"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convertTo___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convertTo___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convertTo___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convertTo___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convertTo___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_convertTo;
static const lean_string_object lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "convert_to!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 188, 149, 104, 171, 154, 220, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_convert__to_x21;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert__to_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert__to_x21__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "convert_to failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__4_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__9_value),LEAN_SCALAR_PTR_LITERAL(223, 78, 141, 85, 50, 255, 216, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___boxed(lean_object**);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_acChange___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "acChange"};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__0_value),LEAN_SCALAR_PTR_LITERAL(98, 233, 51, 132, 40, 8, 139, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_acChange___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "ac_change "};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_acChange = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "acChange!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 111, 1, 221, 157, 124, 81, 237)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ac_change! "};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_acChange_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_acChange_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_acChange_x21___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "acRfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ac_rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convert___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_6_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object* v_msgData_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(v_msgData_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_ref_51_; lean_object* v___x_52_; lean_object* v_a_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_61_; 
v_ref_51_ = lean_ctor_get(v___y_48_, 5);
v___x_52_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
return v_res_68_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = l_Lean_Elab_ConfigEval_EvalExpr_instNat;
v___x_70_ = l_Lean_Elab_ConfigEval_EvalExpr_instOption___redArg(v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__2));
v___x_74_ = l_Lean_stringToMessageData(v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_75_, lean_object* v_args_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_263_ = lean_string_dec_eq(v_ctor_75_, v___x_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_264_;
}
else
{
lean_object* v___x_265_; lean_object* v___x_266_; uint8_t v___x_267_; 
v___x_265_ = lean_array_get_size(v_args_76_);
v___x_266_ = lean_unsigned_to_nat(13u);
v___x_267_ = lean_nat_dec_eq(v___x_265_, v___x_266_);
if (v___x_267_ == 0)
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
v___x_268_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3);
v___x_269_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(v___x_268_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
v_a_270_ = lean_ctor_get(v___x_269_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v___x_269_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_269_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_275_; 
if (v_isShared_273_ == 0)
{
v___x_275_ = v___x_272_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_a_270_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
else
{
goto v___jp_82_;
}
}
v___jp_82_:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_83_ = l_Lean_instInhabitedExpr;
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_84_);
lean_inc(v___x_85_);
v___x_86_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_85_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_86_) == 0)
{
lean_object* v_a_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v_a_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc(v_a_87_);
lean_dec_ref_known(v___x_86_, 1);
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_88_);
lean_inc(v___x_89_);
v___x_90_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_89_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_90_) == 0)
{
lean_object* v_a_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v_a_91_ = lean_ctor_get(v___x_90_, 0);
lean_inc(v_a_91_);
lean_dec_ref_known(v___x_90_, 1);
v___x_92_ = lean_unsigned_to_nat(2u);
v___x_93_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_92_);
lean_inc(v___x_93_);
v___x_94_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v___x_93_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_94_) == 0)
{
lean_object* v_a_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v_a_95_ = lean_ctor_get(v___x_94_, 0);
lean_inc(v_a_95_);
lean_dec_ref_known(v___x_94_, 1);
v___x_96_ = lean_unsigned_to_nat(3u);
v___x_97_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_96_);
lean_inc(v___x_97_);
v___x_98_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v___x_97_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_98_) == 0)
{
lean_object* v_a_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_a_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_a_99_);
lean_dec_ref_known(v___x_98_, 1);
v___x_100_ = lean_unsigned_to_nat(4u);
v___x_101_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_100_);
lean_inc(v___x_101_);
v___x_102_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v___x_101_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_102_) == 0)
{
lean_object* v_a_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_a_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_a_103_);
lean_dec_ref_known(v___x_102_, 1);
v___x_104_ = lean_unsigned_to_nat(5u);
v___x_105_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_104_);
lean_inc(v___x_105_);
v___x_106_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_105_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_106_) == 0)
{
lean_object* v_a_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v_a_107_ = lean_ctor_get(v___x_106_, 0);
lean_inc(v_a_107_);
lean_dec_ref_known(v___x_106_, 1);
v___x_108_ = lean_unsigned_to_nat(6u);
v___x_109_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_108_);
lean_inc(v___x_109_);
v___x_110_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_109_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v_a_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v_a_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_a_111_);
lean_dec_ref_known(v___x_110_, 1);
v___x_112_ = lean_unsigned_to_nat(7u);
v___x_113_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_112_);
lean_inc(v___x_113_);
v___x_114_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_113_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_116_; lean_object* v_evalExpr_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_a_115_);
lean_dec_ref_known(v___x_114_, 1);
v___x_116_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0);
v_evalExpr_117_ = lean_ctor_get(v___x_116_, 0);
v___x_118_ = lean_unsigned_to_nat(8u);
v___x_119_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_118_);
lean_inc_ref(v_evalExpr_117_);
lean_inc(v___y_80_);
lean_inc_ref(v___y_79_);
lean_inc(v___y_78_);
lean_inc_ref(v___y_77_);
lean_inc(v___x_119_);
v___x_120_ = lean_apply_6(v_evalExpr_117_, v___x_119_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, lean_box(0));
if (lean_obj_tag(v___x_120_) == 0)
{
lean_object* v_a_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v_a_121_ = lean_ctor_get(v___x_120_, 0);
lean_inc(v_a_121_);
lean_dec_ref_known(v___x_120_, 1);
v___x_122_ = lean_unsigned_to_nat(9u);
v___x_123_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_122_);
lean_inc(v___x_123_);
v___x_124_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_123_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_124_) == 0)
{
lean_object* v_a_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v_a_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_a_125_);
lean_dec_ref_known(v___x_124_, 1);
v___x_126_ = lean_unsigned_to_nat(10u);
v___x_127_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_126_);
lean_inc(v___x_127_);
v___x_128_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_127_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_128_) == 0)
{
lean_object* v_a_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v_a_129_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_a_129_);
lean_dec_ref_known(v___x_128_, 1);
v___x_130_ = lean_unsigned_to_nat(11u);
v___x_131_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_130_);
lean_inc(v___x_131_);
v___x_132_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_131_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_132_) == 0)
{
lean_object* v_a_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v_a_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc(v_a_133_);
lean_dec_ref_known(v___x_132_, 1);
v___x_134_ = lean_unsigned_to_nat(12u);
v___x_135_ = lean_array_get_borrowed(v___x_83_, v_args_76_, v___x_134_);
lean_inc(v___x_135_);
v___x_136_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_135_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
if (lean_obj_tag(v___x_136_) == 0)
{
lean_object* v_a_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_157_; 
v_a_137_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_157_ == 0)
{
v___x_139_ = v___x_136_;
v_isShared_140_ = v_isSharedCheck_157_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_a_137_);
lean_dec(v___x_136_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_157_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v___x_141_; uint8_t v___x_142_; uint8_t v___x_143_; uint8_t v___x_144_; uint8_t v___x_145_; uint8_t v___x_146_; uint8_t v___x_147_; uint8_t v___x_148_; uint8_t v___x_149_; uint8_t v___x_150_; uint8_t v___x_151_; uint8_t v___x_152_; uint8_t v___x_153_; lean_object* v___x_155_; 
v___x_141_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v___x_141_, 0, v_a_121_);
v___x_142_ = lean_unbox(v_a_87_);
lean_dec(v_a_87_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1, v___x_142_);
v___x_143_ = lean_unbox(v_a_91_);
lean_dec(v_a_91_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 1, v___x_143_);
v___x_144_ = lean_unbox(v_a_95_);
lean_dec(v_a_95_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 2, v___x_144_);
v___x_145_ = lean_unbox(v_a_99_);
lean_dec(v_a_99_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 3, v___x_145_);
v___x_146_ = lean_unbox(v_a_103_);
lean_dec(v_a_103_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 4, v___x_146_);
v___x_147_ = lean_unbox(v_a_107_);
lean_dec(v_a_107_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 5, v___x_147_);
v___x_148_ = lean_unbox(v_a_111_);
lean_dec(v_a_111_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 6, v___x_148_);
v___x_149_ = lean_unbox(v_a_115_);
lean_dec(v_a_115_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 7, v___x_149_);
v___x_150_ = lean_unbox(v_a_125_);
lean_dec(v_a_125_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 8, v___x_150_);
v___x_151_ = lean_unbox(v_a_129_);
lean_dec(v_a_129_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 9, v___x_151_);
v___x_152_ = lean_unbox(v_a_133_);
lean_dec(v_a_133_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 10, v___x_152_);
v___x_153_ = lean_unbox(v_a_137_);
lean_dec(v_a_137_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*1 + 11, v___x_153_);
if (v_isShared_140_ == 0)
{
lean_ctor_set(v___x_139_, 0, v___x_141_);
v___x_155_ = v___x_139_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___x_141_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
else
{
lean_object* v_a_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_165_; 
lean_dec(v_a_133_);
lean_dec(v_a_129_);
lean_dec(v_a_125_);
lean_dec(v_a_121_);
lean_dec(v_a_115_);
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_158_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_165_ == 0)
{
v___x_160_ = v___x_136_;
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_a_158_);
lean_dec(v___x_136_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v___x_163_; 
if (v_isShared_161_ == 0)
{
v___x_163_ = v___x_160_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_a_158_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
}
}
else
{
lean_object* v_a_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_173_; 
lean_dec(v_a_129_);
lean_dec(v_a_125_);
lean_dec(v_a_121_);
lean_dec(v_a_115_);
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_166_ = lean_ctor_get(v___x_132_, 0);
v_isSharedCheck_173_ = !lean_is_exclusive(v___x_132_);
if (v_isSharedCheck_173_ == 0)
{
v___x_168_ = v___x_132_;
v_isShared_169_ = v_isSharedCheck_173_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_a_166_);
lean_dec(v___x_132_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_173_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_171_; 
if (v_isShared_169_ == 0)
{
v___x_171_ = v___x_168_;
goto v_reusejp_170_;
}
else
{
lean_object* v_reuseFailAlloc_172_; 
v_reuseFailAlloc_172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_172_, 0, v_a_166_);
v___x_171_ = v_reuseFailAlloc_172_;
goto v_reusejp_170_;
}
v_reusejp_170_:
{
return v___x_171_;
}
}
}
}
else
{
lean_object* v_a_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_181_; 
lean_dec(v_a_125_);
lean_dec(v_a_121_);
lean_dec(v_a_115_);
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_174_ = lean_ctor_get(v___x_128_, 0);
v_isSharedCheck_181_ = !lean_is_exclusive(v___x_128_);
if (v_isSharedCheck_181_ == 0)
{
v___x_176_ = v___x_128_;
v_isShared_177_ = v_isSharedCheck_181_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_a_174_);
lean_dec(v___x_128_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_181_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_179_; 
if (v_isShared_177_ == 0)
{
v___x_179_ = v___x_176_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_a_174_);
v___x_179_ = v_reuseFailAlloc_180_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
return v___x_179_;
}
}
}
}
else
{
lean_object* v_a_182_; lean_object* v___x_184_; uint8_t v_isShared_185_; uint8_t v_isSharedCheck_189_; 
lean_dec(v_a_121_);
lean_dec(v_a_115_);
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_182_ = lean_ctor_get(v___x_124_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_124_);
if (v_isSharedCheck_189_ == 0)
{
v___x_184_ = v___x_124_;
v_isShared_185_ = v_isSharedCheck_189_;
goto v_resetjp_183_;
}
else
{
lean_inc(v_a_182_);
lean_dec(v___x_124_);
v___x_184_ = lean_box(0);
v_isShared_185_ = v_isSharedCheck_189_;
goto v_resetjp_183_;
}
v_resetjp_183_:
{
lean_object* v___x_187_; 
if (v_isShared_185_ == 0)
{
v___x_187_ = v___x_184_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_a_182_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
}
}
else
{
lean_object* v_a_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_197_; 
lean_dec(v_a_115_);
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_190_ = lean_ctor_get(v___x_120_, 0);
v_isSharedCheck_197_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_197_ == 0)
{
v___x_192_ = v___x_120_;
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_a_190_);
lean_dec(v___x_120_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_195_; 
if (v_isShared_193_ == 0)
{
v___x_195_ = v___x_192_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v_a_190_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
else
{
lean_object* v_a_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_205_; 
lean_dec(v_a_111_);
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_198_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_205_ == 0)
{
v___x_200_ = v___x_114_;
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_a_198_);
lean_dec(v___x_114_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_203_; 
if (v_isShared_201_ == 0)
{
v___x_203_ = v___x_200_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_198_);
v___x_203_ = v_reuseFailAlloc_204_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
return v___x_203_;
}
}
}
}
else
{
lean_object* v_a_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_213_; 
lean_dec(v_a_107_);
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_206_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_213_ == 0)
{
v___x_208_ = v___x_110_;
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_a_206_);
lean_dec(v___x_110_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
if (v_isShared_209_ == 0)
{
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v_a_206_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
}
else
{
lean_object* v_a_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_221_; 
lean_dec(v_a_103_);
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_214_ = lean_ctor_get(v___x_106_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_221_ == 0)
{
v___x_216_ = v___x_106_;
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_a_214_);
lean_dec(v___x_106_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_219_; 
if (v_isShared_217_ == 0)
{
v___x_219_ = v___x_216_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_a_214_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
else
{
lean_object* v_a_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
lean_dec(v_a_99_);
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_222_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_229_ == 0)
{
v___x_224_ = v___x_102_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_a_222_);
lean_dec(v___x_102_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_222_);
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
else
{
lean_object* v_a_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_237_; 
lean_dec(v_a_95_);
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_230_ = lean_ctor_get(v___x_98_, 0);
v_isSharedCheck_237_ = !lean_is_exclusive(v___x_98_);
if (v_isSharedCheck_237_ == 0)
{
v___x_232_ = v___x_98_;
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_a_230_);
lean_dec(v___x_98_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_237_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_235_; 
if (v_isShared_233_ == 0)
{
v___x_235_ = v___x_232_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_a_230_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
}
}
else
{
lean_object* v_a_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_245_; 
lean_dec(v_a_91_);
lean_dec(v_a_87_);
v_a_238_ = lean_ctor_get(v___x_94_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_245_ == 0)
{
v___x_240_ = v___x_94_;
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_a_238_);
lean_dec(v___x_94_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___x_243_; 
if (v_isShared_241_ == 0)
{
v___x_243_ = v___x_240_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_a_238_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
else
{
lean_object* v_a_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_253_; 
lean_dec(v_a_87_);
v_a_246_ = lean_ctor_get(v___x_90_, 0);
v_isSharedCheck_253_ = !lean_is_exclusive(v___x_90_);
if (v_isSharedCheck_253_ == 0)
{
v___x_248_ = v___x_90_;
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_a_246_);
lean_dec(v___x_90_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
lean_object* v___x_251_; 
if (v_isShared_249_ == 0)
{
v___x_251_ = v___x_248_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v_a_246_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
return v___x_251_;
}
}
}
}
else
{
lean_object* v_a_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_261_; 
v_a_254_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_261_ == 0)
{
v___x_256_ = v___x_86_;
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_a_254_);
lean_dec(v___x_86_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_259_; 
if (v_isShared_257_ == 0)
{
v___x_259_ = v___x_256_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v_a_254_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_278_, lean_object* v_args_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0(v_ctor_278_, v_args_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec_ref(v_args_279_);
lean_dec_ref(v_ctor_278_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr(lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_){
_start:
{
lean_object* v___f_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___f_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__0));
v___x_299_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3));
v___x_300_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_299_, v___f_298_, v_a_292_, v_a_293_, v_a_294_, v_a_295_, v_a_296_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___boxed(lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr(v_a_301_, v_a_302_, v_a_303_, v_a_304_, v_a_305_);
lean_dec(v_a_305_);
lean_dec_ref(v_a_304_);
lean_dec(v_a_303_);
lean_dec_ref(v_a_302_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_308_, lean_object* v_msg_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_316_, lean_object* v_msg_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_316_, v_msg_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
return v_res_323_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_325_ = lean_box(0);
v___x_326_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___closed__3));
v___x_327_ = l_Lean_Expr_const___override(v___x_326_, v___x_325_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__1);
v___x_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
return v___x_329_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_330_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__2);
v___x_331_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__0));
v___x_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
lean_ctor_set(v___x_332_, 1, v___x_330_);
return v___x_332_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig(void){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig___closed__3);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0(lean_object* v_ctor_334_, lean_object* v_args_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
lean_object* v___x_362_; uint8_t v___x_363_; 
v___x_362_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_363_ = lean_string_dec_eq(v_ctor_334_, v___x_362_);
if (v___x_363_ == 0)
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_364_;
}
else
{
lean_object* v___x_365_; lean_object* v___x_366_; uint8_t v___x_367_; 
v___x_365_ = lean_array_get_size(v_args_335_);
v___x_366_ = lean_unsigned_to_nat(1u);
v___x_367_ = lean_nat_dec_eq(v___x_365_, v___x_366_);
if (v___x_367_ == 0)
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v_a_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_377_; 
v___x_368_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3);
v___x_369_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(v___x_368_, v___y_336_, v___y_337_, v___y_338_, v___y_339_);
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
v_reuseFailAlloc_376_ = lean_alloc_ctor(1, 1, 0);
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
goto v___jp_341_;
}
}
v___jp_341_:
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_342_ = l_Lean_instInhabitedExpr;
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = lean_array_get_borrowed(v___x_342_, v_args_335_, v___x_343_);
lean_inc(v___x_344_);
v___x_345_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr(v___x_344_, v___y_336_, v___y_337_, v___y_338_, v___y_339_);
if (lean_obj_tag(v___x_345_) == 0)
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_353_; 
v_a_346_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_353_ == 0)
{
v___x_348_ = v___x_345_;
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_345_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_351_; 
if (v_isShared_349_ == 0)
{
v___x_351_ = v___x_348_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_a_346_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
v_a_354_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_345_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_345_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v_a_354_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_378_, lean_object* v_args_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___lam__0(v_ctor_378_, v_args_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec_ref(v_args_379_);
lean_dec_ref(v_ctor_378_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr(lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_){
_start:
{
lean_object* v___f_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___f_398_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__0));
v___x_399_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3));
v___x_400_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_399_, v___f_398_, v_a_392_, v_a_393_, v_a_394_, v_a_395_, v_a_396_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___boxed(lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr(v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
lean_dec(v_a_405_);
lean_dec_ref(v_a_404_);
lean_dec(v_a_403_);
lean_dec_ref(v_a_402_);
return v_res_407_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_409_ = lean_box(0);
v___x_410_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3));
v___x_411_ = l_Lean_Expr_const___override(v___x_410_, v___x_409_);
return v___x_411_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1);
v___x_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_413_, 0, v___x_412_);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_414_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2);
v___x_415_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__0));
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_415_);
lean_ctor_set(v___x_416_, 1, v___x_414_);
return v___x_416_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig(void){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__3);
return v___x_417_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_418_ = lean_box(0);
v___x_419_ = l_Lean_Elab_abortTermExceptionId;
v___x_420_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
lean_ctor_set(v___x_420_, 1, v___x_418_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg(){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; 
v___x_422_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0);
v___x_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg___boxed(lean_object* v___y_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(lean_object* v_e_426_, lean_object* v___y_427_){
_start:
{
uint8_t v___x_429_; 
v___x_429_ = l_Lean_Expr_hasMVar(v_e_426_);
if (v___x_429_ == 0)
{
lean_object* v___x_430_; 
v___x_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_430_, 0, v_e_426_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v_mctx_432_; lean_object* v___x_433_; lean_object* v_fst_434_; lean_object* v_snd_435_; lean_object* v___x_436_; lean_object* v_cache_437_; lean_object* v_zetaDeltaFVarIds_438_; lean_object* v_postponed_439_; lean_object* v_diag_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_449_; 
v___x_431_ = lean_st_ref_get(v___y_427_);
v_mctx_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc_ref(v_mctx_432_);
lean_dec(v___x_431_);
v___x_433_ = l_Lean_instantiateMVarsCore(v_mctx_432_, v_e_426_);
v_fst_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_fst_434_);
v_snd_435_ = lean_ctor_get(v___x_433_, 1);
lean_inc(v_snd_435_);
lean_dec_ref(v___x_433_);
v___x_436_ = lean_st_ref_take(v___y_427_);
v_cache_437_ = lean_ctor_get(v___x_436_, 1);
v_zetaDeltaFVarIds_438_ = lean_ctor_get(v___x_436_, 2);
v_postponed_439_ = lean_ctor_get(v___x_436_, 3);
v_diag_440_ = lean_ctor_get(v___x_436_, 4);
v_isSharedCheck_449_ = !lean_is_exclusive(v___x_436_);
if (v_isSharedCheck_449_ == 0)
{
lean_object* v_unused_450_; 
v_unused_450_ = lean_ctor_get(v___x_436_, 0);
lean_dec(v_unused_450_);
v___x_442_ = v___x_436_;
v_isShared_443_ = v_isSharedCheck_449_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_diag_440_);
lean_inc(v_postponed_439_);
lean_inc(v_zetaDeltaFVarIds_438_);
lean_inc(v_cache_437_);
lean_dec(v___x_436_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_449_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 0, v_snd_435_);
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_snd_435_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v_cache_437_);
lean_ctor_set(v_reuseFailAlloc_448_, 2, v_zetaDeltaFVarIds_438_);
lean_ctor_set(v_reuseFailAlloc_448_, 3, v_postponed_439_);
lean_ctor_set(v_reuseFailAlloc_448_, 4, v_diag_440_);
v___x_445_ = v_reuseFailAlloc_448_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_446_ = lean_st_ref_set(v___y_427_, v___x_445_);
v___x_447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_447_, 0, v_fst_434_);
return v___x_447_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg___boxed(lean_object* v_e_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_e_451_, v___y_452_);
lean_dec(v___y_452_);
return v_res_454_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_455_ = lean_box(1);
v___x_456_ = l_Lean_MessageData_ofFormat(v___x_455_);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3(void){
_start:
{
lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_460_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2));
v___x_461_ = l_Lean_MessageData_ofFormat(v___x_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(lean_object* v_x_462_, lean_object* v_x_463_){
_start:
{
if (lean_obj_tag(v_x_463_) == 0)
{
return v_x_462_;
}
else
{
lean_object* v_head_464_; lean_object* v_tail_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_487_; 
v_head_464_ = lean_ctor_get(v_x_463_, 0);
v_tail_465_ = lean_ctor_get(v_x_463_, 1);
v_isSharedCheck_487_ = !lean_is_exclusive(v_x_463_);
if (v_isSharedCheck_487_ == 0)
{
v___x_467_ = v_x_463_;
v_isShared_468_ = v_isSharedCheck_487_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_tail_465_);
lean_inc(v_head_464_);
lean_dec(v_x_463_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_487_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v_before_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_485_; 
v_before_469_ = lean_ctor_get(v_head_464_, 0);
v_isSharedCheck_485_ = !lean_is_exclusive(v_head_464_);
if (v_isSharedCheck_485_ == 0)
{
lean_object* v_unused_486_; 
v_unused_486_ = lean_ctor_get(v_head_464_, 1);
lean_dec(v_unused_486_);
v___x_471_ = v_head_464_;
v_isShared_472_ = v_isSharedCheck_485_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_before_469_);
lean_dec(v_head_464_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_485_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_473_; lean_object* v___x_475_; 
v___x_473_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0);
if (v_isShared_472_ == 0)
{
lean_ctor_set_tag(v___x_471_, 7);
lean_ctor_set(v___x_471_, 1, v___x_473_);
lean_ctor_set(v___x_471_, 0, v_x_462_);
v___x_475_ = v___x_471_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_x_462_);
lean_ctor_set(v_reuseFailAlloc_484_, 1, v___x_473_);
v___x_475_ = v_reuseFailAlloc_484_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_476_; lean_object* v___x_478_; 
v___x_476_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3);
if (v_isShared_468_ == 0)
{
lean_ctor_set_tag(v___x_467_, 7);
lean_ctor_set(v___x_467_, 1, v___x_476_);
lean_ctor_set(v___x_467_, 0, v___x_475_);
v___x_478_ = v___x_467_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v___x_475_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v___x_476_);
v___x_478_ = v_reuseFailAlloc_483_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_479_ = l_Lean_MessageData_ofSyntax(v_before_469_);
v___x_480_ = l_Lean_indentD(v___x_479_);
v___x_481_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_478_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v_x_462_ = v___x_481_;
v_x_463_ = v_tail_465_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(lean_object* v_opts_488_, lean_object* v_opt_489_){
_start:
{
lean_object* v_name_490_; lean_object* v_defValue_491_; lean_object* v_map_492_; lean_object* v___x_493_; 
v_name_490_ = lean_ctor_get(v_opt_489_, 0);
v_defValue_491_ = lean_ctor_get(v_opt_489_, 1);
v_map_492_ = lean_ctor_get(v_opts_488_, 0);
v___x_493_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_492_, v_name_490_);
if (lean_obj_tag(v___x_493_) == 0)
{
uint8_t v___x_494_; 
v___x_494_ = lean_unbox(v_defValue_491_);
return v___x_494_;
}
else
{
lean_object* v_val_495_; 
v_val_495_ = lean_ctor_get(v___x_493_, 0);
lean_inc(v_val_495_);
lean_dec_ref_known(v___x_493_, 1);
if (lean_obj_tag(v_val_495_) == 1)
{
uint8_t v_v_496_; 
v_v_496_ = lean_ctor_get_uint8(v_val_495_, 0);
lean_dec_ref_known(v_val_495_, 0);
return v_v_496_;
}
else
{
uint8_t v___x_497_; 
lean_dec(v_val_495_);
v___x_497_ = lean_unbox(v_defValue_491_);
return v___x_497_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8___boxed(lean_object* v_opts_498_, lean_object* v_opt_499_){
_start:
{
uint8_t v_res_500_; lean_object* v_r_501_; 
v_res_500_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(v_opts_498_, v_opt_499_);
lean_dec_ref(v_opt_499_);
lean_dec_ref(v_opts_498_);
v_r_501_ = lean_box(v_res_500_);
return v_r_501_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_505_; lean_object* v___x_506_; 
v___x_505_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1));
v___x_506_ = l_Lean_MessageData_ofFormat(v___x_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(lean_object* v_msgData_507_, lean_object* v_macroStack_508_, lean_object* v___y_509_){
_start:
{
lean_object* v_options_511_; lean_object* v___x_512_; uint8_t v___x_513_; 
v_options_511_ = lean_ctor_get(v___y_509_, 2);
v___x_512_ = l_Lean_Elab_pp_macroStack;
v___x_513_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(v_options_511_, v___x_512_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; 
lean_dec(v_macroStack_508_);
v___x_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_514_, 0, v_msgData_507_);
return v___x_514_;
}
else
{
if (lean_obj_tag(v_macroStack_508_) == 0)
{
lean_object* v___x_515_; 
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v_msgData_507_);
return v___x_515_;
}
else
{
lean_object* v_head_516_; lean_object* v_after_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_532_; 
v_head_516_ = lean_ctor_get(v_macroStack_508_, 0);
lean_inc(v_head_516_);
v_after_517_ = lean_ctor_get(v_head_516_, 1);
v_isSharedCheck_532_ = !lean_is_exclusive(v_head_516_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; 
v_unused_533_ = lean_ctor_get(v_head_516_, 0);
lean_dec(v_unused_533_);
v___x_519_ = v_head_516_;
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_after_517_);
lean_dec(v_head_516_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_521_; lean_object* v___x_523_; 
v___x_521_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0);
if (v_isShared_520_ == 0)
{
lean_ctor_set_tag(v___x_519_, 7);
lean_ctor_set(v___x_519_, 1, v___x_521_);
lean_ctor_set(v___x_519_, 0, v_msgData_507_);
v___x_523_ = v___x_519_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_msgData_507_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v___x_521_);
v___x_523_ = v_reuseFailAlloc_531_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v_msgData_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_524_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2);
v___x_525_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_523_);
lean_ctor_set(v___x_525_, 1, v___x_524_);
v___x_526_ = l_Lean_MessageData_ofSyntax(v_after_517_);
v___x_527_ = l_Lean_indentD(v___x_526_);
v_msgData_528_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_528_, 0, v___x_525_);
lean_ctor_set(v_msgData_528_, 1, v___x_527_);
v___x_529_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(v_msgData_528_, v_macroStack_508_);
v___x_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_530_, 0, v___x_529_);
return v___x_530_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___boxed(lean_object* v_msgData_534_, lean_object* v_macroStack_535_, lean_object* v___y_536_, lean_object* v___y_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_msgData_534_, v_macroStack_535_, v___y_536_);
lean_dec_ref(v___y_536_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(lean_object* v_msg_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_ref_547_; lean_object* v___x_548_; lean_object* v_a_549_; lean_object* v_macroStack_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_561_; 
v_ref_547_ = lean_ctor_get(v___y_544_, 5);
v___x_548_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_539_, v___y_542_, v___y_543_, v___y_544_, v___y_545_);
v_a_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_a_549_);
lean_dec_ref(v___x_548_);
v_macroStack_550_ = lean_ctor_get(v___y_540_, 1);
v___x_551_ = l_Lean_Elab_getBetterRef(v_ref_547_, v_macroStack_550_);
lean_inc(v_macroStack_550_);
v___x_552_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_a_549_, v_macroStack_550_, v___y_544_);
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_561_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_561_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_561_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; lean_object* v___x_559_; 
v___x_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_557_, 0, v___x_551_);
lean_ctor_set(v___x_557_, 1, v_a_553_);
if (v_isShared_556_ == 0)
{
lean_ctor_set_tag(v___x_555_, 1);
lean_ctor_set(v___x_555_, 0, v___x_557_);
v___x_559_ = v___x_555_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v___x_557_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg___boxed(lean_object* v_msg_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v_msg_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec(v___y_564_);
lean_dec_ref(v___y_563_);
return v_res_570_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_572_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__0));
v___x_573_ = l_Lean_stringToMessageData(v___x_572_);
return v___x_573_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_575_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__2));
v___x_576_ = l_Lean_stringToMessageData(v___x_575_);
return v___x_576_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5(void){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_578_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__4));
v___x_579_ = l_Lean_stringToMessageData(v___x_578_);
return v___x_579_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7(void){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_581_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__6));
v___x_582_ = l_Lean_stringToMessageData(v___x_581_);
return v___x_582_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9(void){
_start:
{
lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_584_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__8));
v___x_585_ = l_Lean_stringToMessageData(v___x_584_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2(lean_object* v_stx_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_){
_start:
{
lean_object* v___x_594_; lean_object* v_evalExpr_595_; lean_object* v_expectedType_x3f_596_; uint8_t v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v_fileName_602_; lean_object* v_fileMap_603_; lean_object* v_options_604_; lean_object* v_currRecDepth_605_; lean_object* v_maxRecDepth_606_; lean_object* v_ref_607_; lean_object* v_currNamespace_608_; lean_object* v_openDecls_609_; lean_object* v_initHeartbeats_610_; lean_object* v_maxHeartbeats_611_; lean_object* v_quotContext_612_; lean_object* v_currMacroScope_613_; uint8_t v_diag_614_; lean_object* v_cancelTk_x3f_615_; uint8_t v_suppressElabErrors_616_; lean_object* v_inheritedTraceOptions_617_; uint8_t v___x_618_; lean_object* v_ref_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_594_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__0);
v_evalExpr_595_ = lean_ctor_get(v___x_594_, 0);
v_expectedType_x3f_596_ = lean_ctor_get(v___x_594_, 1);
v___x_597_ = 1;
v___x_598_ = lean_box(0);
v___x_599_ = lean_box(v___x_597_);
v___x_600_ = lean_box(v___x_597_);
lean_inc(v_expectedType_x3f_596_);
lean_inc(v_stx_586_);
v___x_601_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_601_, 0, v_stx_586_);
lean_closure_set(v___x_601_, 1, v_expectedType_x3f_596_);
lean_closure_set(v___x_601_, 2, v___x_599_);
lean_closure_set(v___x_601_, 3, v___x_600_);
lean_closure_set(v___x_601_, 4, v___x_598_);
v_fileName_602_ = lean_ctor_get(v_a_591_, 0);
v_fileMap_603_ = lean_ctor_get(v_a_591_, 1);
v_options_604_ = lean_ctor_get(v_a_591_, 2);
v_currRecDepth_605_ = lean_ctor_get(v_a_591_, 3);
v_maxRecDepth_606_ = lean_ctor_get(v_a_591_, 4);
v_ref_607_ = lean_ctor_get(v_a_591_, 5);
v_currNamespace_608_ = lean_ctor_get(v_a_591_, 6);
v_openDecls_609_ = lean_ctor_get(v_a_591_, 7);
v_initHeartbeats_610_ = lean_ctor_get(v_a_591_, 8);
v_maxHeartbeats_611_ = lean_ctor_get(v_a_591_, 9);
v_quotContext_612_ = lean_ctor_get(v_a_591_, 10);
v_currMacroScope_613_ = lean_ctor_get(v_a_591_, 11);
v_diag_614_ = lean_ctor_get_uint8(v_a_591_, sizeof(void*)*14);
v_cancelTk_x3f_615_ = lean_ctor_get(v_a_591_, 12);
v_suppressElabErrors_616_ = lean_ctor_get_uint8(v_a_591_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_617_ = lean_ctor_get(v_a_591_, 13);
v___x_618_ = 1;
v_ref_619_ = l_Lean_replaceRef(v_stx_586_, v_ref_607_);
lean_dec(v_stx_586_);
lean_inc_ref(v_inheritedTraceOptions_617_);
lean_inc(v_cancelTk_x3f_615_);
lean_inc(v_currMacroScope_613_);
lean_inc(v_quotContext_612_);
lean_inc(v_maxHeartbeats_611_);
lean_inc(v_initHeartbeats_610_);
lean_inc(v_openDecls_609_);
lean_inc(v_currNamespace_608_);
lean_inc(v_maxRecDepth_606_);
lean_inc(v_currRecDepth_605_);
lean_inc_ref(v_options_604_);
lean_inc_ref(v_fileMap_603_);
lean_inc_ref(v_fileName_602_);
v___x_620_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_620_, 0, v_fileName_602_);
lean_ctor_set(v___x_620_, 1, v_fileMap_603_);
lean_ctor_set(v___x_620_, 2, v_options_604_);
lean_ctor_set(v___x_620_, 3, v_currRecDepth_605_);
lean_ctor_set(v___x_620_, 4, v_maxRecDepth_606_);
lean_ctor_set(v___x_620_, 5, v_ref_619_);
lean_ctor_set(v___x_620_, 6, v_currNamespace_608_);
lean_ctor_set(v___x_620_, 7, v_openDecls_609_);
lean_ctor_set(v___x_620_, 8, v_initHeartbeats_610_);
lean_ctor_set(v___x_620_, 9, v_maxHeartbeats_611_);
lean_ctor_set(v___x_620_, 10, v_quotContext_612_);
lean_ctor_set(v___x_620_, 11, v_currMacroScope_613_);
lean_ctor_set(v___x_620_, 12, v_cancelTk_x3f_615_);
lean_ctor_set(v___x_620_, 13, v_inheritedTraceOptions_617_);
lean_ctor_set_uint8(v___x_620_, sizeof(void*)*14, v_diag_614_);
lean_ctor_set_uint8(v___x_620_, sizeof(void*)*14 + 1, v_suppressElabErrors_616_);
v___x_621_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_601_, v___x_618_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v___x_620_, v_a_592_);
if (lean_obj_tag(v___x_621_) == 0)
{
lean_object* v_a_622_; lean_object* v___x_623_; lean_object* v_a_624_; lean_object* v___y_626_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v___y_629_; lean_object* v___y_630_; lean_object* v___y_631_; lean_object* v___y_632_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; lean_object* v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; lean_object* v___y_647_; uint8_t v___y_648_; lean_object* v___y_666_; lean_object* v___y_667_; lean_object* v___y_668_; lean_object* v___y_669_; lean_object* v___y_670_; lean_object* v___y_671_; lean_object* v___y_678_; lean_object* v___y_679_; lean_object* v___y_680_; lean_object* v___y_681_; lean_object* v___y_682_; lean_object* v___y_683_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_717_; lean_object* v___y_718_; lean_object* v___y_719_; lean_object* v___y_720_; uint8_t v___x_733_; 
v_a_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc(v_a_622_);
lean_dec_ref_known(v___x_621_, 1);
v___x_623_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_622_, v_a_590_);
v_a_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc(v_a_624_);
lean_dec_ref(v___x_623_);
v___x_733_ = l_Lean_Expr_hasSorry(v_a_624_);
if (v___x_733_ == 0)
{
v___y_678_ = v_a_587_;
v___y_679_ = v_a_588_;
v___y_680_ = v_a_589_;
v___y_681_ = v_a_590_;
v___y_682_ = v___x_620_;
v___y_683_ = v_a_592_;
goto v___jp_677_;
}
else
{
uint8_t v___x_734_; 
v___x_734_ = l_Lean_Expr_hasSyntheticSorry(v_a_624_);
if (v___x_734_ == 0)
{
v___y_715_ = v_a_587_;
v___y_716_ = v_a_588_;
v___y_717_ = v_a_589_;
v___y_718_ = v_a_590_;
v___y_719_ = v___x_620_;
v___y_720_ = v_a_592_;
goto v___jp_714_;
}
else
{
lean_object* v___x_735_; lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_743_; 
lean_dec(v_a_624_);
lean_dec_ref_known(v___x_620_, 14);
v___x_735_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_736_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_735_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_735_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_741_; 
if (v_isShared_739_ == 0)
{
v___x_741_ = v___x_738_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_736_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
v___jp_625_:
{
lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_633_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1);
v___x_634_ = l_Lean_indentExpr(v_a_624_);
v___x_635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_635_, 0, v___x_633_);
lean_ctor_set(v___x_635_, 1, v___x_634_);
v___x_636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_635_);
lean_ctor_set(v___x_636_, 1, v___y_632_);
v___x_637_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_636_, v___y_631_, v___y_627_, v___y_630_, v___y_629_, v___y_626_, v___y_628_);
lean_dec_ref(v___y_626_);
return v___x_637_;
}
v___jp_638_:
{
if (v___y_648_ == 0)
{
if (lean_obj_tag(v___y_646_) == 0)
{
lean_dec_ref_known(v___y_646_, 2);
lean_dec_ref(v___y_640_);
lean_dec(v_a_624_);
return v___y_641_;
}
else
{
lean_object* v_id_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_663_; 
v_id_649_ = lean_ctor_get(v___y_646_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___y_646_);
if (v_isSharedCheck_663_ == 0)
{
lean_object* v_unused_664_; 
v_unused_664_ = lean_ctor_get(v___y_646_, 1);
lean_dec(v_unused_664_);
v___x_651_ = v___y_646_;
v_isShared_652_ = v_isSharedCheck_663_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_id_649_);
lean_dec(v___y_646_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_663_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
uint8_t v___x_653_; 
v___x_653_ = l_Lean_instBEqInternalExceptionId_beq(v___y_643_, v_id_649_);
lean_dec(v_id_649_);
if (v___x_653_ == 0)
{
lean_del_object(v___x_651_);
lean_dec_ref(v___y_640_);
lean_dec(v_a_624_);
return v___y_641_;
}
else
{
lean_dec_ref(v___y_641_);
if (lean_obj_tag(v_expectedType_x3f_596_) == 1)
{
lean_object* v_val_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_658_; 
v_val_654_ = lean_ctor_get(v_expectedType_x3f_596_, 0);
v___x_655_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3);
lean_inc(v_val_654_);
v___x_656_ = l_Lean_MessageData_ofExpr(v_val_654_);
if (v_isShared_652_ == 0)
{
lean_ctor_set_tag(v___x_651_, 7);
lean_ctor_set(v___x_651_, 1, v___x_656_);
lean_ctor_set(v___x_651_, 0, v___x_655_);
v___x_658_ = v___x_651_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_661_; 
v_reuseFailAlloc_661_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_661_, 0, v___x_655_);
lean_ctor_set(v_reuseFailAlloc_661_, 1, v___x_656_);
v___x_658_ = v_reuseFailAlloc_661_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_659_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5);
v___x_660_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_660_, 0, v___x_658_);
lean_ctor_set(v___x_660_, 1, v___x_659_);
v___y_626_ = v___y_640_;
v___y_627_ = v___y_639_;
v___y_628_ = v___y_642_;
v___y_629_ = v___y_645_;
v___y_630_ = v___y_644_;
v___y_631_ = v___y_647_;
v___y_632_ = v___x_660_;
goto v___jp_625_;
}
}
else
{
lean_object* v___x_662_; 
lean_del_object(v___x_651_);
v___x_662_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__7);
v___y_626_ = v___y_640_;
v___y_627_ = v___y_639_;
v___y_628_ = v___y_642_;
v___y_629_ = v___y_645_;
v___y_630_ = v___y_644_;
v___y_631_ = v___y_647_;
v___y_632_ = v___x_662_;
goto v___jp_625_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_646_);
lean_dec_ref(v___y_640_);
lean_dec(v_a_624_);
return v___y_641_;
}
}
v___jp_665_:
{
lean_object* v___x_672_; 
lean_inc_ref(v_evalExpr_595_);
lean_inc(v___y_671_);
lean_inc_ref(v___y_670_);
lean_inc(v___y_669_);
lean_inc_ref(v___y_668_);
lean_inc(v_a_624_);
v___x_672_ = lean_apply_6(v_evalExpr_595_, v_a_624_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, lean_box(0));
if (lean_obj_tag(v___x_672_) == 0)
{
lean_dec_ref(v___y_670_);
lean_dec(v_a_624_);
return v___x_672_;
}
else
{
lean_object* v_a_673_; lean_object* v___x_674_; uint8_t v___x_675_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
v___x_674_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_675_ = l_Lean_Exception_isInterrupt(v_a_673_);
if (v___x_675_ == 0)
{
uint8_t v___x_676_; 
lean_inc(v_a_673_);
v___x_676_ = l_Lean_Exception_isRuntime(v_a_673_);
v___y_639_ = v___y_667_;
v___y_640_ = v___y_670_;
v___y_641_ = v___x_672_;
v___y_642_ = v___y_671_;
v___y_643_ = v___x_674_;
v___y_644_ = v___y_668_;
v___y_645_ = v___y_669_;
v___y_646_ = v_a_673_;
v___y_647_ = v___y_666_;
v___y_648_ = v___x_676_;
goto v___jp_638_;
}
else
{
v___y_639_ = v___y_667_;
v___y_640_ = v___y_670_;
v___y_641_ = v___x_672_;
v___y_642_ = v___y_671_;
v___y_643_ = v___x_674_;
v___y_644_ = v___y_668_;
v___y_645_ = v___y_669_;
v___y_646_ = v_a_673_;
v___y_647_ = v___y_666_;
v___y_648_ = v___x_675_;
goto v___jp_638_;
}
}
}
v___jp_677_:
{
lean_object* v___x_684_; 
lean_inc(v_a_624_);
v___x_684_ = l_Lean_Meta_getMVars(v_a_624_, v___y_680_, v___y_681_, v___y_682_, v___y_683_);
if (lean_obj_tag(v___x_684_) == 0)
{
lean_object* v_a_685_; lean_object* v___x_686_; 
v_a_685_ = lean_ctor_get(v___x_684_, 0);
lean_inc(v_a_685_);
lean_dec_ref_known(v___x_684_, 1);
v___x_686_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_685_, v___x_598_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_);
lean_dec(v_a_685_);
if (lean_obj_tag(v___x_686_) == 0)
{
lean_object* v_a_687_; uint8_t v___x_688_; 
v_a_687_ = lean_ctor_get(v___x_686_, 0);
lean_inc(v_a_687_);
lean_dec_ref_known(v___x_686_, 1);
v___x_688_ = lean_unbox(v_a_687_);
lean_dec(v_a_687_);
if (v___x_688_ == 0)
{
v___y_666_ = v___y_678_;
v___y_667_ = v___y_679_;
v___y_668_ = v___y_680_;
v___y_669_ = v___y_681_;
v___y_670_ = v___y_682_;
v___y_671_ = v___y_683_;
goto v___jp_665_;
}
else
{
lean_object* v___x_689_; lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec_ref(v___y_682_);
lean_dec(v_a_624_);
v___x_689_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_689_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_695_; 
if (v_isShared_693_ == 0)
{
v___x_695_ = v___x_692_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_a_690_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
}
else
{
lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_705_; 
lean_dec_ref(v___y_682_);
lean_dec(v_a_624_);
v_a_698_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_705_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_705_ == 0)
{
v___x_700_ = v___x_686_;
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v___x_686_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_703_; 
if (v_isShared_701_ == 0)
{
v___x_703_ = v___x_700_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_a_698_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
return v___x_703_;
}
}
}
}
else
{
lean_object* v_a_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_713_; 
lean_dec_ref(v___y_682_);
lean_dec(v_a_624_);
v_a_706_ = lean_ctor_get(v___x_684_, 0);
v_isSharedCheck_713_ = !lean_is_exclusive(v___x_684_);
if (v_isSharedCheck_713_ == 0)
{
v___x_708_ = v___x_684_;
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_a_706_);
lean_dec(v___x_684_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v___x_711_; 
if (v_isShared_709_ == 0)
{
v___x_711_ = v___x_708_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v_a_706_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
return v___x_711_;
}
}
}
}
v___jp_714_:
{
lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v_a_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_732_; 
v___x_721_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_722_ = l_Lean_indentExpr(v_a_624_);
v___x_723_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_721_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_723_, v___y_715_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
lean_dec_ref(v___y_719_);
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
}
else
{
lean_object* v_a_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_751_; 
lean_dec_ref_known(v___x_620_, 14);
v_a_744_ = lean_ctor_get(v___x_621_, 0);
v_isSharedCheck_751_ = !lean_is_exclusive(v___x_621_);
if (v_isSharedCheck_751_ == 0)
{
v___x_746_ = v___x_621_;
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_a_744_);
lean_dec(v___x_621_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_749_; 
if (v_isShared_747_ == 0)
{
v___x_749_ = v___x_746_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_a_744_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object* v_stx_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_, lean_object* v_a_758_, lean_object* v_a_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2(v_stx_752_, v_a_753_, v_a_754_, v_a_755_, v_a_756_, v_a_757_, v_a_758_);
lean_dec(v_a_758_);
lean_dec_ref(v_a_757_);
lean_dec(v_a_756_);
lean_dec_ref(v_a_755_);
lean_dec(v_a_754_);
lean_dec_ref(v_a_753_);
return v_res_760_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2(void){
_start:
{
lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_764_ = lean_box(0);
v___x_765_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__1));
v___x_766_ = l_Lean_mkConst(v___x_765_, v___x_764_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1(lean_object* v_stx_768_, lean_object* v_a_769_, lean_object* v_a_770_, lean_object* v_a_771_, lean_object* v_a_772_, lean_object* v_a_773_, lean_object* v_a_774_){
_start:
{
lean_object* v_fileName_776_; lean_object* v_fileMap_777_; lean_object* v_options_778_; lean_object* v_currRecDepth_779_; lean_object* v_maxRecDepth_780_; lean_object* v_ref_781_; lean_object* v_currNamespace_782_; lean_object* v_openDecls_783_; lean_object* v_initHeartbeats_784_; lean_object* v_maxHeartbeats_785_; lean_object* v_quotContext_786_; lean_object* v_currMacroScope_787_; uint8_t v_diag_788_; lean_object* v_cancelTk_x3f_789_; uint8_t v_suppressElabErrors_790_; lean_object* v_inheritedTraceOptions_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v_ref_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v_fileName_776_ = lean_ctor_get(v_a_773_, 0);
v_fileMap_777_ = lean_ctor_get(v_a_773_, 1);
v_options_778_ = lean_ctor_get(v_a_773_, 2);
v_currRecDepth_779_ = lean_ctor_get(v_a_773_, 3);
v_maxRecDepth_780_ = lean_ctor_get(v_a_773_, 4);
v_ref_781_ = lean_ctor_get(v_a_773_, 5);
v_currNamespace_782_ = lean_ctor_get(v_a_773_, 6);
v_openDecls_783_ = lean_ctor_get(v_a_773_, 7);
v_initHeartbeats_784_ = lean_ctor_get(v_a_773_, 8);
v_maxHeartbeats_785_ = lean_ctor_get(v_a_773_, 9);
v_quotContext_786_ = lean_ctor_get(v_a_773_, 10);
v_currMacroScope_787_ = lean_ctor_get(v_a_773_, 11);
v_diag_788_ = lean_ctor_get_uint8(v_a_773_, sizeof(void*)*14);
v_cancelTk_x3f_789_ = lean_ctor_get(v_a_773_, 12);
v_suppressElabErrors_790_ = lean_ctor_get_uint8(v_a_773_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_791_ = lean_ctor_get(v_a_773_, 13);
v___x_792_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__2);
v___x_793_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___closed__3));
v_ref_794_ = l_Lean_replaceRef(v_stx_768_, v_ref_781_);
lean_inc_ref(v_inheritedTraceOptions_791_);
lean_inc(v_cancelTk_x3f_789_);
lean_inc(v_currMacroScope_787_);
lean_inc(v_quotContext_786_);
lean_inc(v_maxHeartbeats_785_);
lean_inc(v_initHeartbeats_784_);
lean_inc(v_openDecls_783_);
lean_inc(v_currNamespace_782_);
lean_inc(v_maxRecDepth_780_);
lean_inc(v_currRecDepth_779_);
lean_inc_ref(v_options_778_);
lean_inc_ref(v_fileMap_777_);
lean_inc_ref(v_fileName_776_);
v___x_795_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_795_, 0, v_fileName_776_);
lean_ctor_set(v___x_795_, 1, v_fileMap_777_);
lean_ctor_set(v___x_795_, 2, v_options_778_);
lean_ctor_set(v___x_795_, 3, v_currRecDepth_779_);
lean_ctor_set(v___x_795_, 4, v_maxRecDepth_780_);
lean_ctor_set(v___x_795_, 5, v_ref_794_);
lean_ctor_set(v___x_795_, 6, v_currNamespace_782_);
lean_ctor_set(v___x_795_, 7, v_openDecls_783_);
lean_ctor_set(v___x_795_, 8, v_initHeartbeats_784_);
lean_ctor_set(v___x_795_, 9, v_maxHeartbeats_785_);
lean_ctor_set(v___x_795_, 10, v_quotContext_786_);
lean_ctor_set(v___x_795_, 11, v_currMacroScope_787_);
lean_ctor_set(v___x_795_, 12, v_cancelTk_x3f_789_);
lean_ctor_set(v___x_795_, 13, v_inheritedTraceOptions_791_);
lean_ctor_set_uint8(v___x_795_, sizeof(void*)*14, v_diag_788_);
lean_ctor_set_uint8(v___x_795_, sizeof(void*)*14 + 1, v_suppressElabErrors_790_);
lean_inc(v_stx_768_);
v___x_796_ = l_Lean_Elab_ConfigEval_EvalTerm_evalOptionStx___redArg(v___x_792_, v___x_793_, v_stx_768_, v_a_769_, v_a_770_, v_a_771_, v_a_772_, v___x_795_, v_a_774_);
if (lean_obj_tag(v___x_796_) == 0)
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_805_; 
lean_dec_ref_known(v___x_795_, 14);
lean_dec(v_stx_768_);
v_a_797_ = lean_ctor_get(v___x_796_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_796_);
if (v_isSharedCheck_805_ == 0)
{
v___x_799_ = v___x_796_;
v_isShared_800_ = v_isSharedCheck_805_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_796_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_805_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v_fst_801_; lean_object* v___x_803_; 
v_fst_801_ = lean_ctor_get(v_a_797_, 0);
lean_inc(v_fst_801_);
lean_dec(v_a_797_);
if (v_isShared_800_ == 0)
{
lean_ctor_set(v___x_799_, 0, v_fst_801_);
v___x_803_ = v___x_799_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_fst_801_);
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
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_821_; 
v_a_806_ = lean_ctor_get(v___x_796_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_796_);
if (v_isSharedCheck_821_ == 0)
{
v___x_808_ = v___x_796_;
v_isShared_809_ = v_isSharedCheck_821_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_796_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_821_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_810_; lean_object* v___x_812_; 
v___x_810_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_806_);
if (v_isShared_809_ == 0)
{
v___x_812_ = v___x_808_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_a_806_);
v___x_812_ = v_reuseFailAlloc_820_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
uint8_t v___y_814_; uint8_t v___x_818_; 
v___x_818_ = l_Lean_Exception_isInterrupt(v_a_806_);
if (v___x_818_ == 0)
{
uint8_t v___x_819_; 
lean_inc(v_a_806_);
v___x_819_ = l_Lean_Exception_isRuntime(v_a_806_);
v___y_814_ = v___x_819_;
goto v___jp_813_;
}
else
{
v___y_814_ = v___x_818_;
goto v___jp_813_;
}
v___jp_813_:
{
if (v___y_814_ == 0)
{
if (lean_obj_tag(v_a_806_) == 0)
{
lean_dec_ref_known(v_a_806_, 2);
lean_dec_ref_known(v___x_795_, 14);
lean_dec(v_stx_768_);
return v___x_812_;
}
else
{
lean_object* v_id_815_; uint8_t v___x_816_; 
v_id_815_ = lean_ctor_get(v_a_806_, 0);
lean_inc(v_id_815_);
lean_dec_ref_known(v_a_806_, 2);
v___x_816_ = l_Lean_instBEqInternalExceptionId_beq(v___x_810_, v_id_815_);
lean_dec(v_id_815_);
if (v___x_816_ == 0)
{
lean_dec_ref_known(v___x_795_, 14);
lean_dec(v_stx_768_);
return v___x_812_;
}
else
{
lean_object* v___x_817_; 
lean_dec_ref(v___x_812_);
v___x_817_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2(v_stx_768_, v_a_769_, v_a_770_, v_a_771_, v_a_772_, v___x_795_, v_a_774_);
lean_dec_ref_known(v___x_795_, 14);
return v___x_817_;
}
}
}
else
{
lean_dec(v_a_806_);
lean_dec_ref_known(v___x_795_, 14);
lean_dec(v_stx_768_);
return v___x_812_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1___boxed(lean_object* v_stx_822_, lean_object* v_a_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1(v_stx_822_, v_a_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_, v_a_828_);
lean_dec(v_a_828_);
lean_dec_ref(v_a_827_);
lean_dec(v_a_826_);
lean_dec_ref(v_a_825_);
lean_dec(v_a_824_);
lean_dec_ref(v_a_823_);
return v_res_830_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0(void){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; 
v___x_831_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__1);
v___x_832_ = l_Lean_MessageData_ofExpr(v___x_831_);
return v___x_832_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1(void){
_start:
{
lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_833_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__0);
v___x_834_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3);
v___x_835_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_835_, 0, v___x_834_);
lean_ctor_set(v___x_835_, 1, v___x_833_);
return v___x_835_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2(void){
_start:
{
lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_836_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5);
v___x_837_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__1);
v___x_838_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
lean_ctor_set(v___x_838_, 1, v___x_836_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2(lean_object* v_stx_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_, lean_object* v_a_843_, lean_object* v_a_844_, lean_object* v_a_845_){
_start:
{
lean_object* v_ty_x3f_847_; uint8_t v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v_fileName_853_; lean_object* v_fileMap_854_; lean_object* v_options_855_; lean_object* v_currRecDepth_856_; lean_object* v_maxRecDepth_857_; lean_object* v_ref_858_; lean_object* v_currNamespace_859_; lean_object* v_openDecls_860_; lean_object* v_initHeartbeats_861_; lean_object* v_maxHeartbeats_862_; lean_object* v_quotContext_863_; lean_object* v_currMacroScope_864_; uint8_t v_diag_865_; lean_object* v_cancelTk_x3f_866_; uint8_t v_suppressElabErrors_867_; lean_object* v_inheritedTraceOptions_868_; uint8_t v___x_869_; lean_object* v_ref_870_; lean_object* v___x_871_; lean_object* v___x_872_; 
v_ty_x3f_847_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig___closed__2);
v___x_848_ = 1;
v___x_849_ = lean_box(0);
v___x_850_ = lean_box(v___x_848_);
v___x_851_ = lean_box(v___x_848_);
lean_inc(v_stx_839_);
v___x_852_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_852_, 0, v_stx_839_);
lean_closure_set(v___x_852_, 1, v_ty_x3f_847_);
lean_closure_set(v___x_852_, 2, v___x_850_);
lean_closure_set(v___x_852_, 3, v___x_851_);
lean_closure_set(v___x_852_, 4, v___x_849_);
v_fileName_853_ = lean_ctor_get(v_a_844_, 0);
v_fileMap_854_ = lean_ctor_get(v_a_844_, 1);
v_options_855_ = lean_ctor_get(v_a_844_, 2);
v_currRecDepth_856_ = lean_ctor_get(v_a_844_, 3);
v_maxRecDepth_857_ = lean_ctor_get(v_a_844_, 4);
v_ref_858_ = lean_ctor_get(v_a_844_, 5);
v_currNamespace_859_ = lean_ctor_get(v_a_844_, 6);
v_openDecls_860_ = lean_ctor_get(v_a_844_, 7);
v_initHeartbeats_861_ = lean_ctor_get(v_a_844_, 8);
v_maxHeartbeats_862_ = lean_ctor_get(v_a_844_, 9);
v_quotContext_863_ = lean_ctor_get(v_a_844_, 10);
v_currMacroScope_864_ = lean_ctor_get(v_a_844_, 11);
v_diag_865_ = lean_ctor_get_uint8(v_a_844_, sizeof(void*)*14);
v_cancelTk_x3f_866_ = lean_ctor_get(v_a_844_, 12);
v_suppressElabErrors_867_ = lean_ctor_get_uint8(v_a_844_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_868_ = lean_ctor_get(v_a_844_, 13);
v___x_869_ = 1;
v_ref_870_ = l_Lean_replaceRef(v_stx_839_, v_ref_858_);
lean_dec(v_stx_839_);
lean_inc_ref(v_inheritedTraceOptions_868_);
lean_inc(v_cancelTk_x3f_866_);
lean_inc(v_currMacroScope_864_);
lean_inc(v_quotContext_863_);
lean_inc(v_maxHeartbeats_862_);
lean_inc(v_initHeartbeats_861_);
lean_inc(v_openDecls_860_);
lean_inc(v_currNamespace_859_);
lean_inc(v_maxRecDepth_857_);
lean_inc(v_currRecDepth_856_);
lean_inc_ref(v_options_855_);
lean_inc_ref(v_fileMap_854_);
lean_inc_ref(v_fileName_853_);
v___x_871_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_871_, 0, v_fileName_853_);
lean_ctor_set(v___x_871_, 1, v_fileMap_854_);
lean_ctor_set(v___x_871_, 2, v_options_855_);
lean_ctor_set(v___x_871_, 3, v_currRecDepth_856_);
lean_ctor_set(v___x_871_, 4, v_maxRecDepth_857_);
lean_ctor_set(v___x_871_, 5, v_ref_870_);
lean_ctor_set(v___x_871_, 6, v_currNamespace_859_);
lean_ctor_set(v___x_871_, 7, v_openDecls_860_);
lean_ctor_set(v___x_871_, 8, v_initHeartbeats_861_);
lean_ctor_set(v___x_871_, 9, v_maxHeartbeats_862_);
lean_ctor_set(v___x_871_, 10, v_quotContext_863_);
lean_ctor_set(v___x_871_, 11, v_currMacroScope_864_);
lean_ctor_set(v___x_871_, 12, v_cancelTk_x3f_866_);
lean_ctor_set(v___x_871_, 13, v_inheritedTraceOptions_868_);
lean_ctor_set_uint8(v___x_871_, sizeof(void*)*14, v_diag_865_);
lean_ctor_set_uint8(v___x_871_, sizeof(void*)*14 + 1, v_suppressElabErrors_867_);
v___x_872_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_852_, v___x_869_, v_a_840_, v_a_841_, v_a_842_, v_a_843_, v___x_871_, v_a_845_);
if (lean_obj_tag(v___x_872_) == 0)
{
lean_object* v_a_873_; lean_object* v___x_874_; lean_object* v_a_875_; lean_object* v___y_877_; lean_object* v___y_878_; lean_object* v___y_879_; lean_object* v___y_880_; lean_object* v___y_881_; lean_object* v___y_882_; lean_object* v___y_883_; lean_object* v___y_884_; lean_object* v___y_885_; uint8_t v___y_886_; lean_object* v___y_903_; lean_object* v___y_904_; lean_object* v___y_905_; lean_object* v___y_906_; lean_object* v___y_907_; lean_object* v___y_908_; lean_object* v___y_915_; lean_object* v___y_916_; lean_object* v___y_917_; lean_object* v___y_918_; lean_object* v___y_919_; lean_object* v___y_920_; lean_object* v___y_952_; lean_object* v___y_953_; lean_object* v___y_954_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_957_; uint8_t v___x_970_; 
v_a_873_ = lean_ctor_get(v___x_872_, 0);
lean_inc(v_a_873_);
lean_dec_ref_known(v___x_872_, 1);
v___x_874_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_873_, v_a_843_);
v_a_875_ = lean_ctor_get(v___x_874_, 0);
lean_inc(v_a_875_);
lean_dec_ref(v___x_874_);
v___x_970_ = l_Lean_Expr_hasSorry(v_a_875_);
if (v___x_970_ == 0)
{
v___y_915_ = v_a_840_;
v___y_916_ = v_a_841_;
v___y_917_ = v_a_842_;
v___y_918_ = v_a_843_;
v___y_919_ = v___x_871_;
v___y_920_ = v_a_845_;
goto v___jp_914_;
}
else
{
uint8_t v___x_971_; 
v___x_971_ = l_Lean_Expr_hasSyntheticSorry(v_a_875_);
if (v___x_971_ == 0)
{
v___y_952_ = v_a_840_;
v___y_953_ = v_a_841_;
v___y_954_ = v_a_842_;
v___y_955_ = v_a_843_;
v___y_956_ = v___x_871_;
v___y_957_ = v_a_845_;
goto v___jp_951_;
}
else
{
lean_object* v___x_972_; lean_object* v_a_973_; lean_object* v___x_975_; uint8_t v_isShared_976_; uint8_t v_isSharedCheck_980_; 
lean_dec(v_a_875_);
lean_dec_ref_known(v___x_871_, 14);
v___x_972_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_973_ = lean_ctor_get(v___x_972_, 0);
v_isSharedCheck_980_ = !lean_is_exclusive(v___x_972_);
if (v_isSharedCheck_980_ == 0)
{
v___x_975_ = v___x_972_;
v_isShared_976_ = v_isSharedCheck_980_;
goto v_resetjp_974_;
}
else
{
lean_inc(v_a_973_);
lean_dec(v___x_972_);
v___x_975_ = lean_box(0);
v_isShared_976_ = v_isSharedCheck_980_;
goto v_resetjp_974_;
}
v_resetjp_974_:
{
lean_object* v___x_978_; 
if (v_isShared_976_ == 0)
{
v___x_978_ = v___x_975_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_979_; 
v_reuseFailAlloc_979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_979_, 0, v_a_973_);
v___x_978_ = v_reuseFailAlloc_979_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
return v___x_978_;
}
}
}
}
v___jp_876_:
{
if (v___y_886_ == 0)
{
if (lean_obj_tag(v___y_882_) == 0)
{
lean_dec_ref_known(v___y_882_, 2);
lean_dec_ref(v___y_884_);
lean_dec(v_a_875_);
return v___y_880_;
}
else
{
lean_object* v_id_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_900_; 
v_id_887_ = lean_ctor_get(v___y_882_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___y_882_);
if (v_isSharedCheck_900_ == 0)
{
lean_object* v_unused_901_; 
v_unused_901_ = lean_ctor_get(v___y_882_, 1);
lean_dec(v_unused_901_);
v___x_889_ = v___y_882_;
v_isShared_890_ = v_isSharedCheck_900_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_id_887_);
lean_dec(v___y_882_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_900_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
uint8_t v___x_891_; 
v___x_891_ = l_Lean_instBEqInternalExceptionId_beq(v___y_878_, v_id_887_);
lean_dec(v_id_887_);
if (v___x_891_ == 0)
{
lean_del_object(v___x_889_);
lean_dec_ref(v___y_884_);
lean_dec(v_a_875_);
return v___y_880_;
}
else
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_896_; 
lean_dec_ref(v___y_880_);
v___x_892_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___closed__2);
v___x_893_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1);
v___x_894_ = l_Lean_indentExpr(v_a_875_);
if (v_isShared_890_ == 0)
{
lean_ctor_set_tag(v___x_889_, 7);
lean_ctor_set(v___x_889_, 1, v___x_894_);
lean_ctor_set(v___x_889_, 0, v___x_893_);
v___x_896_ = v___x_889_;
goto v_reusejp_895_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v___x_893_);
lean_ctor_set(v_reuseFailAlloc_899_, 1, v___x_894_);
v___x_896_ = v_reuseFailAlloc_899_;
goto v_reusejp_895_;
}
v_reusejp_895_:
{
lean_object* v___x_897_; lean_object* v___x_898_; 
v___x_897_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_897_, 0, v___x_896_);
lean_ctor_set(v___x_897_, 1, v___x_892_);
v___x_898_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_897_, v___y_879_, v___y_877_, v___y_881_, v___y_885_, v___y_884_, v___y_883_);
lean_dec_ref(v___y_884_);
return v___x_898_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_884_);
lean_dec_ref(v___y_882_);
lean_dec(v_a_875_);
return v___y_880_;
}
}
v___jp_902_:
{
lean_object* v___x_909_; 
lean_inc(v_a_875_);
v___x_909_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr(v_a_875_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
if (lean_obj_tag(v___x_909_) == 0)
{
lean_dec_ref(v___y_907_);
lean_dec(v_a_875_);
return v___x_909_;
}
else
{
lean_object* v_a_910_; lean_object* v___x_911_; uint8_t v___x_912_; 
v_a_910_ = lean_ctor_get(v___x_909_, 0);
lean_inc(v_a_910_);
v___x_911_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_912_ = l_Lean_Exception_isInterrupt(v_a_910_);
if (v___x_912_ == 0)
{
uint8_t v___x_913_; 
lean_inc(v_a_910_);
v___x_913_ = l_Lean_Exception_isRuntime(v_a_910_);
v___y_877_ = v___y_904_;
v___y_878_ = v___x_911_;
v___y_879_ = v___y_903_;
v___y_880_ = v___x_909_;
v___y_881_ = v___y_905_;
v___y_882_ = v_a_910_;
v___y_883_ = v___y_908_;
v___y_884_ = v___y_907_;
v___y_885_ = v___y_906_;
v___y_886_ = v___x_913_;
goto v___jp_876_;
}
else
{
v___y_877_ = v___y_904_;
v___y_878_ = v___x_911_;
v___y_879_ = v___y_903_;
v___y_880_ = v___x_909_;
v___y_881_ = v___y_905_;
v___y_882_ = v_a_910_;
v___y_883_ = v___y_908_;
v___y_884_ = v___y_907_;
v___y_885_ = v___y_906_;
v___y_886_ = v___x_912_;
goto v___jp_876_;
}
}
}
v___jp_914_:
{
lean_object* v___x_921_; 
lean_inc(v_a_875_);
v___x_921_ = l_Lean_Meta_getMVars(v_a_875_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
if (lean_obj_tag(v___x_921_) == 0)
{
lean_object* v_a_922_; lean_object* v___x_923_; 
v_a_922_ = lean_ctor_get(v___x_921_, 0);
lean_inc(v_a_922_);
lean_dec_ref_known(v___x_921_, 1);
v___x_923_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_922_, v___x_849_, v___y_915_, v___y_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
lean_dec(v_a_922_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v_a_924_; uint8_t v___x_925_; 
v_a_924_ = lean_ctor_get(v___x_923_, 0);
lean_inc(v_a_924_);
lean_dec_ref_known(v___x_923_, 1);
v___x_925_ = lean_unbox(v_a_924_);
lean_dec(v_a_924_);
if (v___x_925_ == 0)
{
v___y_903_ = v___y_915_;
v___y_904_ = v___y_916_;
v___y_905_ = v___y_917_;
v___y_906_ = v___y_918_;
v___y_907_ = v___y_919_;
v___y_908_ = v___y_920_;
goto v___jp_902_;
}
else
{
lean_object* v___x_926_; lean_object* v_a_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_934_; 
lean_dec_ref(v___y_919_);
lean_dec(v_a_875_);
v___x_926_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_927_ = lean_ctor_get(v___x_926_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_926_);
if (v_isSharedCheck_934_ == 0)
{
v___x_929_ = v___x_926_;
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_a_927_);
lean_dec(v___x_926_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_a_927_);
v___x_932_ = v_reuseFailAlloc_933_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
return v___x_932_;
}
}
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
lean_dec_ref(v___y_919_);
lean_dec(v_a_875_);
v_a_935_ = lean_ctor_get(v___x_923_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_923_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_923_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_923_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
lean_object* v___x_940_; 
if (v_isShared_938_ == 0)
{
v___x_940_ = v___x_937_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_a_935_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
}
}
else
{
lean_object* v_a_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_950_; 
lean_dec_ref(v___y_919_);
lean_dec(v_a_875_);
v_a_943_ = lean_ctor_get(v___x_921_, 0);
v_isSharedCheck_950_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_950_ == 0)
{
v___x_945_ = v___x_921_;
v_isShared_946_ = v_isSharedCheck_950_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_a_943_);
lean_dec(v___x_921_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_950_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_948_; 
if (v_isShared_946_ == 0)
{
v___x_948_ = v___x_945_;
goto v_reusejp_947_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_949_, 0, v_a_943_);
v___x_948_ = v_reuseFailAlloc_949_;
goto v_reusejp_947_;
}
v_reusejp_947_:
{
return v___x_948_;
}
}
}
}
v___jp_951_:
{
lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_969_; 
v___x_958_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_959_ = l_Lean_indentExpr(v_a_875_);
v___x_960_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_960_, 0, v___x_958_);
lean_ctor_set(v___x_960_, 1, v___x_959_);
v___x_961_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_960_, v___y_952_, v___y_953_, v___y_954_, v___y_955_, v___y_956_, v___y_957_);
lean_dec_ref(v___y_956_);
v_a_962_ = lean_ctor_get(v___x_961_, 0);
v_isSharedCheck_969_ = !lean_is_exclusive(v___x_961_);
if (v_isSharedCheck_969_ == 0)
{
v___x_964_ = v___x_961_;
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_961_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___x_967_; 
if (v_isShared_965_ == 0)
{
v___x_967_ = v___x_964_;
goto v_reusejp_966_;
}
else
{
lean_object* v_reuseFailAlloc_968_; 
v_reuseFailAlloc_968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_968_, 0, v_a_962_);
v___x_967_ = v_reuseFailAlloc_968_;
goto v_reusejp_966_;
}
v_reusejp_966_:
{
return v___x_967_;
}
}
}
}
else
{
lean_object* v_a_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_988_; 
lean_dec_ref_known(v___x_871_, 14);
v_a_981_ = lean_ctor_get(v___x_872_, 0);
v_isSharedCheck_988_ = !lean_is_exclusive(v___x_872_);
if (v_isSharedCheck_988_ == 0)
{
v___x_983_ = v___x_872_;
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_a_981_);
lean_dec(v___x_872_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v___x_986_; 
if (v_isShared_984_ == 0)
{
v___x_986_ = v___x_983_;
goto v_reusejp_985_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v_a_981_);
v___x_986_ = v_reuseFailAlloc_987_;
goto v_reusejp_985_;
}
v_reusejp_985_:
{
return v___x_986_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2___boxed(lean_object* v_stx_989_, lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2(v_stx_989_, v_a_990_, v_a_991_, v_a_992_, v_a_993_, v_a_994_, v_a_995_);
lean_dec(v_a_995_);
lean_dec_ref(v_a_994_);
lean_dec(v_a_993_);
lean_dec_ref(v_a_992_);
lean_dec(v_a_991_);
lean_dec_ref(v_a_990_);
return v_res_997_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; 
v___x_1005_ = lean_box(0);
v___x_1006_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__3));
v___x_1007_ = l_Lean_Expr_const___override(v___x_1006_, v___x_1005_);
return v___x_1007_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_1008_; lean_object* v_ty_x3f_1009_; 
v___x_1008_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4);
v_ty_x3f_1009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_ty_x3f_1009_, 0, v___x_1008_);
return v_ty_x3f_1009_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6(void){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__4);
v___x_1011_ = l_Lean_MessageData_ofExpr(v___x_1010_);
return v___x_1011_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7(void){
_start:
{
lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1012_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__6);
v___x_1013_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3);
v___x_1014_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1014_, 0, v___x_1013_);
lean_ctor_set(v___x_1014_, 1, v___x_1012_);
return v___x_1014_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8(void){
_start:
{
lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1015_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5);
v___x_1016_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__7);
v___x_1017_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1017_, 0, v___x_1016_);
lean_ctor_set(v___x_1017_, 1, v___x_1015_);
return v___x_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0(lean_object* v_stx_1018_, lean_object* v_a_1019_, lean_object* v_a_1020_, lean_object* v_a_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_, lean_object* v_a_1024_){
_start:
{
lean_object* v_ty_x3f_1026_; uint8_t v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v_fileName_1032_; lean_object* v_fileMap_1033_; lean_object* v_options_1034_; lean_object* v_currRecDepth_1035_; lean_object* v_maxRecDepth_1036_; lean_object* v_ref_1037_; lean_object* v_currNamespace_1038_; lean_object* v_openDecls_1039_; lean_object* v_initHeartbeats_1040_; lean_object* v_maxHeartbeats_1041_; lean_object* v_quotContext_1042_; lean_object* v_currMacroScope_1043_; uint8_t v_diag_1044_; lean_object* v_cancelTk_x3f_1045_; uint8_t v_suppressElabErrors_1046_; lean_object* v_inheritedTraceOptions_1047_; uint8_t v___x_1048_; lean_object* v_ref_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; 
v_ty_x3f_1026_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_1027_ = 1;
v___x_1028_ = lean_box(0);
v___x_1029_ = lean_box(v___x_1027_);
v___x_1030_ = lean_box(v___x_1027_);
lean_inc(v_stx_1018_);
v___x_1031_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1031_, 0, v_stx_1018_);
lean_closure_set(v___x_1031_, 1, v_ty_x3f_1026_);
lean_closure_set(v___x_1031_, 2, v___x_1029_);
lean_closure_set(v___x_1031_, 3, v___x_1030_);
lean_closure_set(v___x_1031_, 4, v___x_1028_);
v_fileName_1032_ = lean_ctor_get(v_a_1023_, 0);
v_fileMap_1033_ = lean_ctor_get(v_a_1023_, 1);
v_options_1034_ = lean_ctor_get(v_a_1023_, 2);
v_currRecDepth_1035_ = lean_ctor_get(v_a_1023_, 3);
v_maxRecDepth_1036_ = lean_ctor_get(v_a_1023_, 4);
v_ref_1037_ = lean_ctor_get(v_a_1023_, 5);
v_currNamespace_1038_ = lean_ctor_get(v_a_1023_, 6);
v_openDecls_1039_ = lean_ctor_get(v_a_1023_, 7);
v_initHeartbeats_1040_ = lean_ctor_get(v_a_1023_, 8);
v_maxHeartbeats_1041_ = lean_ctor_get(v_a_1023_, 9);
v_quotContext_1042_ = lean_ctor_get(v_a_1023_, 10);
v_currMacroScope_1043_ = lean_ctor_get(v_a_1023_, 11);
v_diag_1044_ = lean_ctor_get_uint8(v_a_1023_, sizeof(void*)*14);
v_cancelTk_x3f_1045_ = lean_ctor_get(v_a_1023_, 12);
v_suppressElabErrors_1046_ = lean_ctor_get_uint8(v_a_1023_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1047_ = lean_ctor_get(v_a_1023_, 13);
v___x_1048_ = 1;
v_ref_1049_ = l_Lean_replaceRef(v_stx_1018_, v_ref_1037_);
lean_dec(v_stx_1018_);
lean_inc_ref(v_inheritedTraceOptions_1047_);
lean_inc(v_cancelTk_x3f_1045_);
lean_inc(v_currMacroScope_1043_);
lean_inc(v_quotContext_1042_);
lean_inc(v_maxHeartbeats_1041_);
lean_inc(v_initHeartbeats_1040_);
lean_inc(v_openDecls_1039_);
lean_inc(v_currNamespace_1038_);
lean_inc(v_maxRecDepth_1036_);
lean_inc(v_currRecDepth_1035_);
lean_inc_ref(v_options_1034_);
lean_inc_ref(v_fileMap_1033_);
lean_inc_ref(v_fileName_1032_);
v___x_1050_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1050_, 0, v_fileName_1032_);
lean_ctor_set(v___x_1050_, 1, v_fileMap_1033_);
lean_ctor_set(v___x_1050_, 2, v_options_1034_);
lean_ctor_set(v___x_1050_, 3, v_currRecDepth_1035_);
lean_ctor_set(v___x_1050_, 4, v_maxRecDepth_1036_);
lean_ctor_set(v___x_1050_, 5, v_ref_1049_);
lean_ctor_set(v___x_1050_, 6, v_currNamespace_1038_);
lean_ctor_set(v___x_1050_, 7, v_openDecls_1039_);
lean_ctor_set(v___x_1050_, 8, v_initHeartbeats_1040_);
lean_ctor_set(v___x_1050_, 9, v_maxHeartbeats_1041_);
lean_ctor_set(v___x_1050_, 10, v_quotContext_1042_);
lean_ctor_set(v___x_1050_, 11, v_currMacroScope_1043_);
lean_ctor_set(v___x_1050_, 12, v_cancelTk_x3f_1045_);
lean_ctor_set(v___x_1050_, 13, v_inheritedTraceOptions_1047_);
lean_ctor_set_uint8(v___x_1050_, sizeof(void*)*14, v_diag_1044_);
lean_ctor_set_uint8(v___x_1050_, sizeof(void*)*14 + 1, v_suppressElabErrors_1046_);
v___x_1051_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1031_, v___x_1048_, v_a_1019_, v_a_1020_, v_a_1021_, v_a_1022_, v___x_1050_, v_a_1024_);
if (lean_obj_tag(v___x_1051_) == 0)
{
lean_object* v_a_1052_; lean_object* v___x_1053_; lean_object* v_a_1054_; lean_object* v___y_1056_; lean_object* v___y_1057_; lean_object* v___y_1058_; lean_object* v___y_1059_; lean_object* v___y_1060_; lean_object* v___y_1061_; lean_object* v___y_1062_; lean_object* v___y_1063_; lean_object* v___y_1064_; uint8_t v___y_1065_; lean_object* v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; lean_object* v___y_1085_; lean_object* v___y_1086_; lean_object* v___y_1087_; lean_object* v___y_1094_; lean_object* v___y_1095_; lean_object* v___y_1096_; lean_object* v___y_1097_; lean_object* v___y_1098_; lean_object* v___y_1099_; lean_object* v___y_1131_; lean_object* v___y_1132_; lean_object* v___y_1133_; lean_object* v___y_1134_; lean_object* v___y_1135_; lean_object* v___y_1136_; uint8_t v___x_1149_; 
v_a_1052_ = lean_ctor_get(v___x_1051_, 0);
lean_inc(v_a_1052_);
lean_dec_ref_known(v___x_1051_, 1);
v___x_1053_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_1052_, v_a_1022_);
v_a_1054_ = lean_ctor_get(v___x_1053_, 0);
lean_inc(v_a_1054_);
lean_dec_ref(v___x_1053_);
v___x_1149_ = l_Lean_Expr_hasSorry(v_a_1054_);
if (v___x_1149_ == 0)
{
v___y_1094_ = v_a_1019_;
v___y_1095_ = v_a_1020_;
v___y_1096_ = v_a_1021_;
v___y_1097_ = v_a_1022_;
v___y_1098_ = v___x_1050_;
v___y_1099_ = v_a_1024_;
goto v___jp_1093_;
}
else
{
uint8_t v___x_1150_; 
v___x_1150_ = l_Lean_Expr_hasSyntheticSorry(v_a_1054_);
if (v___x_1150_ == 0)
{
v___y_1131_ = v_a_1019_;
v___y_1132_ = v_a_1020_;
v___y_1133_ = v_a_1021_;
v___y_1134_ = v_a_1022_;
v___y_1135_ = v___x_1050_;
v___y_1136_ = v_a_1024_;
goto v___jp_1130_;
}
else
{
lean_object* v___x_1151_; lean_object* v_a_1152_; lean_object* v___x_1154_; uint8_t v_isShared_1155_; uint8_t v_isSharedCheck_1159_; 
lean_dec(v_a_1054_);
lean_dec_ref_known(v___x_1050_, 14);
v___x_1151_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_1152_ = lean_ctor_get(v___x_1151_, 0);
v_isSharedCheck_1159_ = !lean_is_exclusive(v___x_1151_);
if (v_isSharedCheck_1159_ == 0)
{
v___x_1154_ = v___x_1151_;
v_isShared_1155_ = v_isSharedCheck_1159_;
goto v_resetjp_1153_;
}
else
{
lean_inc(v_a_1152_);
lean_dec(v___x_1151_);
v___x_1154_ = lean_box(0);
v_isShared_1155_ = v_isSharedCheck_1159_;
goto v_resetjp_1153_;
}
v_resetjp_1153_:
{
lean_object* v___x_1157_; 
if (v_isShared_1155_ == 0)
{
v___x_1157_ = v___x_1154_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1158_; 
v_reuseFailAlloc_1158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1158_, 0, v_a_1152_);
v___x_1157_ = v_reuseFailAlloc_1158_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
return v___x_1157_;
}
}
}
}
v___jp_1055_:
{
if (v___y_1065_ == 0)
{
if (lean_obj_tag(v___y_1064_) == 0)
{
lean_dec_ref_known(v___y_1064_, 2);
lean_dec_ref(v___y_1058_);
lean_dec(v_a_1054_);
return v___y_1057_;
}
else
{
lean_object* v_id_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1079_; 
v_id_1066_ = lean_ctor_get(v___y_1064_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___y_1064_);
if (v_isSharedCheck_1079_ == 0)
{
lean_object* v_unused_1080_; 
v_unused_1080_ = lean_ctor_get(v___y_1064_, 1);
lean_dec(v_unused_1080_);
v___x_1068_ = v___y_1064_;
v_isShared_1069_ = v_isSharedCheck_1079_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_id_1066_);
lean_dec(v___y_1064_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1079_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
uint8_t v___x_1070_; 
v___x_1070_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1056_, v_id_1066_);
lean_dec(v_id_1066_);
if (v___x_1070_ == 0)
{
lean_del_object(v___x_1068_);
lean_dec_ref(v___y_1058_);
lean_dec(v_a_1054_);
return v___y_1057_;
}
else
{
lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1075_; 
lean_dec_ref(v___y_1057_);
v___x_1071_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__8);
v___x_1072_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1);
v___x_1073_ = l_Lean_indentExpr(v_a_1054_);
if (v_isShared_1069_ == 0)
{
lean_ctor_set_tag(v___x_1068_, 7);
lean_ctor_set(v___x_1068_, 1, v___x_1073_);
lean_ctor_set(v___x_1068_, 0, v___x_1072_);
v___x_1075_ = v___x_1068_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v___x_1072_);
lean_ctor_set(v_reuseFailAlloc_1078_, 1, v___x_1073_);
v___x_1075_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1076_, 0, v___x_1075_);
lean_ctor_set(v___x_1076_, 1, v___x_1071_);
v___x_1077_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_1076_, v___y_1063_, v___y_1060_, v___y_1059_, v___y_1062_, v___y_1058_, v___y_1061_);
lean_dec_ref(v___y_1058_);
return v___x_1077_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1064_);
lean_dec_ref(v___y_1058_);
lean_dec(v_a_1054_);
return v___y_1057_;
}
}
v___jp_1081_:
{
lean_object* v___x_1088_; 
lean_inc(v_a_1054_);
v___x_1088_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v_a_1054_, v___y_1084_, v___y_1085_, v___y_1086_, v___y_1087_);
if (lean_obj_tag(v___x_1088_) == 0)
{
lean_dec_ref(v___y_1086_);
lean_dec(v_a_1054_);
return v___x_1088_;
}
else
{
lean_object* v_a_1089_; lean_object* v___x_1090_; uint8_t v___x_1091_; 
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc(v_a_1089_);
v___x_1090_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1091_ = l_Lean_Exception_isInterrupt(v_a_1089_);
if (v___x_1091_ == 0)
{
uint8_t v___x_1092_; 
lean_inc(v_a_1089_);
v___x_1092_ = l_Lean_Exception_isRuntime(v_a_1089_);
v___y_1056_ = v___x_1090_;
v___y_1057_ = v___x_1088_;
v___y_1058_ = v___y_1086_;
v___y_1059_ = v___y_1084_;
v___y_1060_ = v___y_1083_;
v___y_1061_ = v___y_1087_;
v___y_1062_ = v___y_1085_;
v___y_1063_ = v___y_1082_;
v___y_1064_ = v_a_1089_;
v___y_1065_ = v___x_1092_;
goto v___jp_1055_;
}
else
{
v___y_1056_ = v___x_1090_;
v___y_1057_ = v___x_1088_;
v___y_1058_ = v___y_1086_;
v___y_1059_ = v___y_1084_;
v___y_1060_ = v___y_1083_;
v___y_1061_ = v___y_1087_;
v___y_1062_ = v___y_1085_;
v___y_1063_ = v___y_1082_;
v___y_1064_ = v_a_1089_;
v___y_1065_ = v___x_1091_;
goto v___jp_1055_;
}
}
}
v___jp_1093_:
{
lean_object* v___x_1100_; 
lean_inc(v_a_1054_);
v___x_1100_ = l_Lean_Meta_getMVars(v_a_1054_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_);
if (lean_obj_tag(v___x_1100_) == 0)
{
lean_object* v_a_1101_; lean_object* v___x_1102_; 
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
lean_inc(v_a_1101_);
lean_dec_ref_known(v___x_1100_, 1);
v___x_1102_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1101_, v___x_1028_, v___y_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_);
lean_dec(v_a_1101_);
if (lean_obj_tag(v___x_1102_) == 0)
{
lean_object* v_a_1103_; uint8_t v___x_1104_; 
v_a_1103_ = lean_ctor_get(v___x_1102_, 0);
lean_inc(v_a_1103_);
lean_dec_ref_known(v___x_1102_, 1);
v___x_1104_ = lean_unbox(v_a_1103_);
lean_dec(v_a_1103_);
if (v___x_1104_ == 0)
{
v___y_1082_ = v___y_1094_;
v___y_1083_ = v___y_1095_;
v___y_1084_ = v___y_1096_;
v___y_1085_ = v___y_1097_;
v___y_1086_ = v___y_1098_;
v___y_1087_ = v___y_1099_;
goto v___jp_1081_;
}
else
{
lean_object* v___x_1105_; lean_object* v_a_1106_; lean_object* v___x_1108_; uint8_t v_isShared_1109_; uint8_t v_isSharedCheck_1113_; 
lean_dec_ref(v___y_1098_);
lean_dec(v_a_1054_);
v___x_1105_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_1106_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1113_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1113_ == 0)
{
v___x_1108_ = v___x_1105_;
v_isShared_1109_ = v_isSharedCheck_1113_;
goto v_resetjp_1107_;
}
else
{
lean_inc(v_a_1106_);
lean_dec(v___x_1105_);
v___x_1108_ = lean_box(0);
v_isShared_1109_ = v_isSharedCheck_1113_;
goto v_resetjp_1107_;
}
v_resetjp_1107_:
{
lean_object* v___x_1111_; 
if (v_isShared_1109_ == 0)
{
v___x_1111_ = v___x_1108_;
goto v_reusejp_1110_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v_a_1106_);
v___x_1111_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1110_;
}
v_reusejp_1110_:
{
return v___x_1111_;
}
}
}
}
else
{
lean_object* v_a_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1121_; 
lean_dec_ref(v___y_1098_);
lean_dec(v_a_1054_);
v_a_1114_ = lean_ctor_get(v___x_1102_, 0);
v_isSharedCheck_1121_ = !lean_is_exclusive(v___x_1102_);
if (v_isSharedCheck_1121_ == 0)
{
v___x_1116_ = v___x_1102_;
v_isShared_1117_ = v_isSharedCheck_1121_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_a_1114_);
lean_dec(v___x_1102_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1121_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1119_; 
if (v_isShared_1117_ == 0)
{
v___x_1119_ = v___x_1116_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v_a_1114_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
return v___x_1119_;
}
}
}
}
else
{
lean_object* v_a_1122_; lean_object* v___x_1124_; uint8_t v_isShared_1125_; uint8_t v_isSharedCheck_1129_; 
lean_dec_ref(v___y_1098_);
lean_dec(v_a_1054_);
v_a_1122_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1129_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1129_ == 0)
{
v___x_1124_ = v___x_1100_;
v_isShared_1125_ = v_isSharedCheck_1129_;
goto v_resetjp_1123_;
}
else
{
lean_inc(v_a_1122_);
lean_dec(v___x_1100_);
v___x_1124_ = lean_box(0);
v_isShared_1125_ = v_isSharedCheck_1129_;
goto v_resetjp_1123_;
}
v_resetjp_1123_:
{
lean_object* v___x_1127_; 
if (v_isShared_1125_ == 0)
{
v___x_1127_ = v___x_1124_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v_a_1122_);
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
v___jp_1130_:
{
lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v_a_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1148_; 
v___x_1137_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_1138_ = l_Lean_indentExpr(v_a_1054_);
v___x_1139_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1139_, 0, v___x_1137_);
lean_ctor_set(v___x_1139_, 1, v___x_1138_);
v___x_1140_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_1139_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_, v___y_1135_, v___y_1136_);
lean_dec_ref(v___y_1135_);
v_a_1141_ = lean_ctor_get(v___x_1140_, 0);
v_isSharedCheck_1148_ = !lean_is_exclusive(v___x_1140_);
if (v_isSharedCheck_1148_ == 0)
{
v___x_1143_ = v___x_1140_;
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_a_1141_);
lean_dec(v___x_1140_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1146_; 
if (v_isShared_1144_ == 0)
{
v___x_1146_ = v___x_1143_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1147_; 
v_reuseFailAlloc_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1147_, 0, v_a_1141_);
v___x_1146_ = v_reuseFailAlloc_1147_;
goto v_reusejp_1145_;
}
v_reusejp_1145_:
{
return v___x_1146_;
}
}
}
}
else
{
lean_object* v_a_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1167_; 
lean_dec_ref_known(v___x_1050_, 14);
v_a_1160_ = lean_ctor_get(v___x_1051_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1051_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1162_ = v___x_1051_;
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_a_1160_);
lean_dec(v___x_1051_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___x_1165_; 
if (v_isShared_1163_ == 0)
{
v___x_1165_ = v___x_1162_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_a_1160_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_stx_1168_, lean_object* v_a_1169_, lean_object* v_a_1170_, lean_object* v_a_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_){
_start:
{
lean_object* v_res_1176_; 
v_res_1176_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0(v_stx_1168_, v_a_1169_, v_a_1170_, v_a_1171_, v_a_1172_, v_a_1173_, v_a_1174_);
lean_dec(v_a_1174_);
lean_dec_ref(v_a_1173_);
lean_dec(v_a_1172_);
lean_dec_ref(v_a_1171_);
lean_dec(v_a_1170_);
lean_dec_ref(v_a_1169_);
return v_res_1176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(lean_object* v_stx_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_, lean_object* v_a_1183_){
_start:
{
lean_object* v_fileName_1185_; lean_object* v_fileMap_1186_; lean_object* v_options_1187_; lean_object* v_currRecDepth_1188_; lean_object* v_maxRecDepth_1189_; lean_object* v_ref_1190_; lean_object* v_currNamespace_1191_; lean_object* v_openDecls_1192_; lean_object* v_initHeartbeats_1193_; lean_object* v_maxHeartbeats_1194_; lean_object* v_quotContext_1195_; lean_object* v_currMacroScope_1196_; uint8_t v_diag_1197_; lean_object* v_cancelTk_x3f_1198_; uint8_t v_suppressElabErrors_1199_; lean_object* v_inheritedTraceOptions_1200_; lean_object* v_ref_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; 
v_fileName_1185_ = lean_ctor_get(v_a_1182_, 0);
v_fileMap_1186_ = lean_ctor_get(v_a_1182_, 1);
v_options_1187_ = lean_ctor_get(v_a_1182_, 2);
v_currRecDepth_1188_ = lean_ctor_get(v_a_1182_, 3);
v_maxRecDepth_1189_ = lean_ctor_get(v_a_1182_, 4);
v_ref_1190_ = lean_ctor_get(v_a_1182_, 5);
v_currNamespace_1191_ = lean_ctor_get(v_a_1182_, 6);
v_openDecls_1192_ = lean_ctor_get(v_a_1182_, 7);
v_initHeartbeats_1193_ = lean_ctor_get(v_a_1182_, 8);
v_maxHeartbeats_1194_ = lean_ctor_get(v_a_1182_, 9);
v_quotContext_1195_ = lean_ctor_get(v_a_1182_, 10);
v_currMacroScope_1196_ = lean_ctor_get(v_a_1182_, 11);
v_diag_1197_ = lean_ctor_get_uint8(v_a_1182_, sizeof(void*)*14);
v_cancelTk_x3f_1198_ = lean_ctor_get(v_a_1182_, 12);
v_suppressElabErrors_1199_ = lean_ctor_get_uint8(v_a_1182_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1200_ = lean_ctor_get(v_a_1182_, 13);
v_ref_1201_ = l_Lean_replaceRef(v_stx_1177_, v_ref_1190_);
lean_inc_ref(v_inheritedTraceOptions_1200_);
lean_inc(v_cancelTk_x3f_1198_);
lean_inc(v_currMacroScope_1196_);
lean_inc(v_quotContext_1195_);
lean_inc(v_maxHeartbeats_1194_);
lean_inc(v_initHeartbeats_1193_);
lean_inc(v_openDecls_1192_);
lean_inc(v_currNamespace_1191_);
lean_inc(v_maxRecDepth_1189_);
lean_inc(v_currRecDepth_1188_);
lean_inc_ref(v_options_1187_);
lean_inc_ref(v_fileMap_1186_);
lean_inc_ref(v_fileName_1185_);
v___x_1202_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1202_, 0, v_fileName_1185_);
lean_ctor_set(v___x_1202_, 1, v_fileMap_1186_);
lean_ctor_set(v___x_1202_, 2, v_options_1187_);
lean_ctor_set(v___x_1202_, 3, v_currRecDepth_1188_);
lean_ctor_set(v___x_1202_, 4, v_maxRecDepth_1189_);
lean_ctor_set(v___x_1202_, 5, v_ref_1201_);
lean_ctor_set(v___x_1202_, 6, v_currNamespace_1191_);
lean_ctor_set(v___x_1202_, 7, v_openDecls_1192_);
lean_ctor_set(v___x_1202_, 8, v_initHeartbeats_1193_);
lean_ctor_set(v___x_1202_, 9, v_maxHeartbeats_1194_);
lean_ctor_set(v___x_1202_, 10, v_quotContext_1195_);
lean_ctor_set(v___x_1202_, 11, v_currMacroScope_1196_);
lean_ctor_set(v___x_1202_, 12, v_cancelTk_x3f_1198_);
lean_ctor_set(v___x_1202_, 13, v_inheritedTraceOptions_1200_);
lean_ctor_set_uint8(v___x_1202_, sizeof(void*)*14, v_diag_1197_);
lean_ctor_set_uint8(v___x_1202_, sizeof(void*)*14 + 1, v_suppressElabErrors_1199_);
lean_inc(v_stx_1177_);
v___x_1203_ = l_Lean_Elab_ConfigEval_instEvalTermTransparencyMode_evalTerm(v_stx_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, v___x_1202_, v_a_1183_);
if (lean_obj_tag(v___x_1203_) == 0)
{
lean_object* v_a_1204_; lean_object* v___x_1206_; uint8_t v_isShared_1207_; uint8_t v_isSharedCheck_1212_; 
lean_dec_ref_known(v___x_1202_, 14);
lean_dec(v_stx_1177_);
v_a_1204_ = lean_ctor_get(v___x_1203_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_1203_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1206_ = v___x_1203_;
v_isShared_1207_ = v_isSharedCheck_1212_;
goto v_resetjp_1205_;
}
else
{
lean_inc(v_a_1204_);
lean_dec(v___x_1203_);
v___x_1206_ = lean_box(0);
v_isShared_1207_ = v_isSharedCheck_1212_;
goto v_resetjp_1205_;
}
v_resetjp_1205_:
{
lean_object* v_fst_1208_; lean_object* v___x_1210_; 
v_fst_1208_ = lean_ctor_get(v_a_1204_, 0);
lean_inc(v_fst_1208_);
lean_dec(v_a_1204_);
if (v_isShared_1207_ == 0)
{
lean_ctor_set(v___x_1206_, 0, v_fst_1208_);
v___x_1210_ = v___x_1206_;
goto v_reusejp_1209_;
}
else
{
lean_object* v_reuseFailAlloc_1211_; 
v_reuseFailAlloc_1211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1211_, 0, v_fst_1208_);
v___x_1210_ = v_reuseFailAlloc_1211_;
goto v_reusejp_1209_;
}
v_reusejp_1209_:
{
return v___x_1210_;
}
}
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1228_; 
v_a_1213_ = lean_ctor_get(v___x_1203_, 0);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1203_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1215_ = v___x_1203_;
v_isShared_1216_ = v_isSharedCheck_1228_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1203_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1228_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v___x_1217_; lean_object* v___x_1219_; 
v___x_1217_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_1213_);
if (v_isShared_1216_ == 0)
{
v___x_1219_ = v___x_1215_;
goto v_reusejp_1218_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_a_1213_);
v___x_1219_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1218_;
}
v_reusejp_1218_:
{
uint8_t v___y_1221_; uint8_t v___x_1225_; 
v___x_1225_ = l_Lean_Exception_isInterrupt(v_a_1213_);
if (v___x_1225_ == 0)
{
uint8_t v___x_1226_; 
lean_inc(v_a_1213_);
v___x_1226_ = l_Lean_Exception_isRuntime(v_a_1213_);
v___y_1221_ = v___x_1226_;
goto v___jp_1220_;
}
else
{
v___y_1221_ = v___x_1225_;
goto v___jp_1220_;
}
v___jp_1220_:
{
if (v___y_1221_ == 0)
{
if (lean_obj_tag(v_a_1213_) == 0)
{
lean_dec_ref_known(v_a_1213_, 2);
lean_dec_ref_known(v___x_1202_, 14);
lean_dec(v_stx_1177_);
return v___x_1219_;
}
else
{
lean_object* v_id_1222_; uint8_t v___x_1223_; 
v_id_1222_ = lean_ctor_get(v_a_1213_, 0);
lean_inc(v_id_1222_);
lean_dec_ref_known(v_a_1213_, 2);
v___x_1223_ = l_Lean_instBEqInternalExceptionId_beq(v___x_1217_, v_id_1222_);
lean_dec(v_id_1222_);
if (v___x_1223_ == 0)
{
lean_dec_ref_known(v___x_1202_, 14);
lean_dec(v_stx_1177_);
return v___x_1219_;
}
else
{
lean_object* v___x_1224_; 
lean_dec_ref(v___x_1219_);
v___x_1224_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0(v_stx_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, v___x_1202_, v_a_1183_);
lean_dec_ref_known(v___x_1202_, 14);
return v___x_1224_;
}
}
}
else
{
lean_dec(v_a_1213_);
lean_dec_ref_known(v___x_1202_, 14);
lean_dec(v_stx_1177_);
return v___x_1219_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_1229_, lean_object* v_a_1230_, lean_object* v_a_1231_, lean_object* v_a_1232_, lean_object* v_a_1233_, lean_object* v_a_1234_, lean_object* v_a_1235_, lean_object* v_a_1236_){
_start:
{
lean_object* v_res_1237_; 
v_res_1237_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_stx_1229_, v_a_1230_, v_a_1231_, v_a_1232_, v_a_1233_, v_a_1234_, v_a_1235_);
lean_dec(v_a_1235_);
lean_dec_ref(v_a_1234_);
lean_dec(v_a_1233_);
lean_dec_ref(v_a_1232_);
lean_dec(v_a_1231_);
lean_dec_ref(v_a_1230_);
return v_res_1237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0(lean_object* v_config_1306_, lean_object* v_item_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_){
_start:
{
lean_object* v_item_1316_; lean_object* v___y_1317_; lean_object* v___y_1318_; lean_object* v___y_1319_; lean_object* v___y_1320_; lean_object* v___y_1321_; lean_object* v___y_1322_; lean_object* v___x_1325_; lean_object* v___x_1326_; 
v___x_1325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3));
v___x_1326_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_1307_, v___x_1325_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1326_) == 0)
{
uint8_t v___x_1327_; 
lean_dec_ref_known(v___x_1326_, 1);
v___x_1327_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_1307_);
if (v___x_1327_ == 0)
{
lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; uint8_t v___x_1331_; 
v___x_1328_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_1307_);
lean_inc_ref(v_item_1307_);
v___x_1329_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_1307_);
v___x_1330_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1));
v___x_1331_ = lean_string_dec_lt(v___x_1328_, v___x_1330_);
if (v___x_1331_ == 0)
{
lean_object* v___x_1332_; uint8_t v___x_1333_; 
v___x_1332_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2));
v___x_1333_ = lean_string_dec_lt(v___x_1328_, v___x_1332_);
if (v___x_1333_ == 0)
{
uint8_t v___x_1334_; 
v___x_1334_ = lean_string_dec_eq(v___x_1328_, v___x_1332_);
if (v___x_1334_ == 0)
{
lean_object* v___x_1335_; uint8_t v___x_1336_; 
v___x_1335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3));
v___x_1336_ = lean_string_dec_eq(v___x_1328_, v___x_1335_);
if (v___x_1336_ == 0)
{
lean_object* v___x_1337_; uint8_t v___x_1338_; 
v___x_1337_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4));
v___x_1338_ = lean_string_dec_eq(v___x_1328_, v___x_1337_);
if (v___x_1338_ == 0)
{
lean_object* v___x_1339_; uint8_t v___x_1340_; 
v___x_1339_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5));
v___x_1340_ = lean_string_dec_eq(v___x_1328_, v___x_1339_);
lean_dec_ref(v___x_1328_);
if (v___x_1340_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1341_; lean_object* v___x_1342_; 
v___x_1341_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6));
v___x_1342_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1341_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1342_) == 0)
{
uint8_t v___x_1343_; 
lean_dec_ref_known(v___x_1342_, 1);
v___x_1343_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1343_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1344_; 
lean_dec_ref(v___x_1329_);
v___x_1344_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1344_) == 0)
{
lean_object* v_a_1345_; lean_object* v___x_1347_; uint8_t v_isShared_1348_; uint8_t v_isSharedCheck_1372_; 
v_a_1345_ = lean_ctor_get(v___x_1344_, 0);
v_isSharedCheck_1372_ = !lean_is_exclusive(v___x_1344_);
if (v_isSharedCheck_1372_ == 0)
{
v___x_1347_ = v___x_1344_;
v_isShared_1348_ = v_isSharedCheck_1372_;
goto v_resetjp_1346_;
}
else
{
lean_inc(v_a_1345_);
lean_dec(v___x_1344_);
v___x_1347_ = lean_box(0);
v_isShared_1348_ = v_isSharedCheck_1372_;
goto v_resetjp_1346_;
}
v_resetjp_1346_:
{
uint8_t v_closePre_1349_; uint8_t v_closePost_1350_; uint8_t v_transparency_1351_; uint8_t v_preTransparency_1352_; uint8_t v_postTransparency_1353_; uint8_t v_preferLHS_1354_; uint8_t v_partialApp_1355_; uint8_t v_sameFun_1356_; lean_object* v_maxArgs_1357_; uint8_t v_typeEqs_1358_; uint8_t v_etaExpand_1359_; uint8_t v_beqEq_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1371_; 
v_closePre_1349_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1350_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1351_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1352_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1353_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1354_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1355_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1356_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1357_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1358_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1359_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_beqEq_1360_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1371_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1371_ == 0)
{
v___x_1362_ = v_config_1306_;
v_isShared_1363_ = v_isSharedCheck_1371_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_maxArgs_1357_);
lean_dec(v_config_1306_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1371_;
goto v_resetjp_1361_;
}
v_resetjp_1361_:
{
lean_object* v___x_1365_; 
if (v_isShared_1363_ == 0)
{
v___x_1365_ = v___x_1362_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v_maxArgs_1357_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1, v_closePre_1349_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 1, v_closePost_1350_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 2, v_transparency_1351_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 3, v_preTransparency_1352_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 4, v_postTransparency_1353_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 5, v_preferLHS_1354_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 6, v_partialApp_1355_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 7, v_sameFun_1356_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 8, v_typeEqs_1358_);
lean_ctor_set_uint8(v_reuseFailAlloc_1370_, sizeof(void*)*1 + 9, v_etaExpand_1359_);
v___x_1365_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
uint8_t v___x_1366_; lean_object* v___x_1368_; 
v___x_1366_ = lean_unbox(v_a_1345_);
lean_dec(v_a_1345_);
lean_ctor_set_uint8(v___x_1365_, sizeof(void*)*1 + 10, v___x_1366_);
lean_ctor_set_uint8(v___x_1365_, sizeof(void*)*1 + 11, v_beqEq_1360_);
if (v_isShared_1348_ == 0)
{
lean_ctor_set(v___x_1347_, 0, v___x_1365_);
v___x_1368_ = v___x_1347_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v___x_1365_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
}
else
{
lean_object* v_a_1373_; lean_object* v___x_1375_; uint8_t v_isShared_1376_; uint8_t v_isSharedCheck_1380_; 
lean_dec_ref(v_config_1306_);
v_a_1373_ = lean_ctor_get(v___x_1344_, 0);
v_isSharedCheck_1380_ = !lean_is_exclusive(v___x_1344_);
if (v_isSharedCheck_1380_ == 0)
{
v___x_1375_ = v___x_1344_;
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
else
{
lean_inc(v_a_1373_);
lean_dec(v___x_1344_);
v___x_1375_ = lean_box(0);
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
v_resetjp_1374_:
{
lean_object* v___x_1378_; 
if (v_isShared_1376_ == 0)
{
v___x_1378_ = v___x_1375_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v_a_1373_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
}
else
{
lean_object* v_a_1381_; lean_object* v___x_1383_; uint8_t v_isShared_1384_; uint8_t v_isSharedCheck_1388_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1381_ = lean_ctor_get(v___x_1342_, 0);
v_isSharedCheck_1388_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1388_ == 0)
{
v___x_1383_ = v___x_1342_;
v_isShared_1384_ = v_isSharedCheck_1388_;
goto v_resetjp_1382_;
}
else
{
lean_inc(v_a_1381_);
lean_dec(v___x_1342_);
v___x_1383_ = lean_box(0);
v_isShared_1384_ = v_isSharedCheck_1388_;
goto v_resetjp_1382_;
}
v_resetjp_1382_:
{
lean_object* v___x_1386_; 
if (v_isShared_1384_ == 0)
{
v___x_1386_ = v___x_1383_;
goto v_reusejp_1385_;
}
else
{
lean_object* v_reuseFailAlloc_1387_; 
v_reuseFailAlloc_1387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1387_, 0, v_a_1381_);
v___x_1386_ = v_reuseFailAlloc_1387_;
goto v_reusejp_1385_;
}
v_reusejp_1385_:
{
return v___x_1386_;
}
}
}
}
}
else
{
lean_object* v___x_1389_; lean_object* v___x_1390_; 
lean_dec_ref(v___x_1328_);
v___x_1389_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7));
v___x_1390_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1389_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1390_) == 0)
{
uint8_t v___x_1391_; 
lean_dec_ref_known(v___x_1390_, 1);
v___x_1391_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1391_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1392_; 
lean_dec_ref(v___x_1329_);
v___x_1392_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1392_) == 0)
{
lean_object* v_a_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1420_; 
v_a_1393_ = lean_ctor_get(v___x_1392_, 0);
v_isSharedCheck_1420_ = !lean_is_exclusive(v___x_1392_);
if (v_isSharedCheck_1420_ == 0)
{
v___x_1395_ = v___x_1392_;
v_isShared_1396_ = v_isSharedCheck_1420_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_a_1393_);
lean_dec(v___x_1392_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1420_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
uint8_t v_closePre_1397_; uint8_t v_closePost_1398_; uint8_t v_transparency_1399_; uint8_t v_preTransparency_1400_; uint8_t v_postTransparency_1401_; uint8_t v_preferLHS_1402_; uint8_t v_partialApp_1403_; uint8_t v_sameFun_1404_; lean_object* v_maxArgs_1405_; uint8_t v_etaExpand_1406_; uint8_t v_useCongrSimp_1407_; uint8_t v_beqEq_1408_; lean_object* v___x_1410_; uint8_t v_isShared_1411_; uint8_t v_isSharedCheck_1419_; 
v_closePre_1397_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1398_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1399_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1400_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1401_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1402_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1403_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1404_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1405_ = lean_ctor_get(v_config_1306_, 0);
v_etaExpand_1406_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1407_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1408_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1419_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1410_ = v_config_1306_;
v_isShared_1411_ = v_isSharedCheck_1419_;
goto v_resetjp_1409_;
}
else
{
lean_inc(v_maxArgs_1405_);
lean_dec(v_config_1306_);
v___x_1410_ = lean_box(0);
v_isShared_1411_ = v_isSharedCheck_1419_;
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
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_maxArgs_1405_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1, v_closePre_1397_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 1, v_closePost_1398_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 2, v_transparency_1399_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 3, v_preTransparency_1400_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 4, v_postTransparency_1401_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 5, v_preferLHS_1402_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 6, v_partialApp_1403_);
lean_ctor_set_uint8(v_reuseFailAlloc_1418_, sizeof(void*)*1 + 7, v_sameFun_1404_);
v___x_1413_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
uint8_t v___x_1414_; lean_object* v___x_1416_; 
v___x_1414_ = lean_unbox(v_a_1393_);
lean_dec(v_a_1393_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*1 + 8, v___x_1414_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*1 + 9, v_etaExpand_1406_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*1 + 10, v_useCongrSimp_1407_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*1 + 11, v_beqEq_1408_);
if (v_isShared_1396_ == 0)
{
lean_ctor_set(v___x_1395_, 0, v___x_1413_);
v___x_1416_ = v___x_1395_;
goto v_reusejp_1415_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v___x_1413_);
v___x_1416_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1415_;
}
v_reusejp_1415_:
{
return v___x_1416_;
}
}
}
}
}
else
{
lean_object* v_a_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1428_; 
lean_dec_ref(v_config_1306_);
v_a_1421_ = lean_ctor_get(v___x_1392_, 0);
v_isSharedCheck_1428_ = !lean_is_exclusive(v___x_1392_);
if (v_isSharedCheck_1428_ == 0)
{
v___x_1423_ = v___x_1392_;
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_a_1421_);
lean_dec(v___x_1392_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v___x_1426_; 
if (v_isShared_1424_ == 0)
{
v___x_1426_ = v___x_1423_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v_a_1421_);
v___x_1426_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
return v___x_1426_;
}
}
}
}
}
else
{
lean_object* v_a_1429_; lean_object* v___x_1431_; uint8_t v_isShared_1432_; uint8_t v_isSharedCheck_1436_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1429_ = lean_ctor_get(v___x_1390_, 0);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1390_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1431_ = v___x_1390_;
v_isShared_1432_ = v_isSharedCheck_1436_;
goto v_resetjp_1430_;
}
else
{
lean_inc(v_a_1429_);
lean_dec(v___x_1390_);
v___x_1431_ = lean_box(0);
v_isShared_1432_ = v_isSharedCheck_1436_;
goto v_resetjp_1430_;
}
v_resetjp_1430_:
{
lean_object* v___x_1434_; 
if (v_isShared_1432_ == 0)
{
v___x_1434_ = v___x_1431_;
goto v_reusejp_1433_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v_a_1429_);
v___x_1434_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1433_;
}
v_reusejp_1433_:
{
return v___x_1434_;
}
}
}
}
}
else
{
lean_object* v___x_1437_; lean_object* v___x_1438_; 
lean_dec_ref(v___x_1328_);
v___x_1437_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8));
v___x_1438_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1437_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1438_) == 0)
{
uint8_t v___x_1439_; 
lean_dec_ref_known(v___x_1438_, 1);
v___x_1439_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1439_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1440_; 
lean_dec_ref(v___x_1329_);
lean_inc_ref(v_item_1307_);
v___x_1440_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1440_) == 0)
{
lean_object* v_value_1441_; lean_object* v___x_1442_; 
lean_dec_ref_known(v___x_1440_, 1);
v_value_1441_ = lean_ctor_get(v_item_1307_, 2);
lean_inc(v_value_1441_);
lean_dec_ref(v_item_1307_);
v___x_1442_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_1441_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1442_) == 0)
{
lean_object* v_a_1443_; lean_object* v___x_1445_; uint8_t v_isShared_1446_; uint8_t v_isSharedCheck_1470_; 
v_a_1443_ = lean_ctor_get(v___x_1442_, 0);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1445_ = v___x_1442_;
v_isShared_1446_ = v_isSharedCheck_1470_;
goto v_resetjp_1444_;
}
else
{
lean_inc(v_a_1443_);
lean_dec(v___x_1442_);
v___x_1445_ = lean_box(0);
v_isShared_1446_ = v_isSharedCheck_1470_;
goto v_resetjp_1444_;
}
v_resetjp_1444_:
{
uint8_t v_closePre_1447_; uint8_t v_closePost_1448_; uint8_t v_preTransparency_1449_; uint8_t v_postTransparency_1450_; uint8_t v_preferLHS_1451_; uint8_t v_partialApp_1452_; uint8_t v_sameFun_1453_; lean_object* v_maxArgs_1454_; uint8_t v_typeEqs_1455_; uint8_t v_etaExpand_1456_; uint8_t v_useCongrSimp_1457_; uint8_t v_beqEq_1458_; lean_object* v___x_1460_; uint8_t v_isShared_1461_; uint8_t v_isSharedCheck_1469_; 
v_closePre_1447_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1448_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_preTransparency_1449_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1450_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1451_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1452_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1453_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1454_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1455_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1456_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1457_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1458_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1469_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1460_ = v_config_1306_;
v_isShared_1461_ = v_isSharedCheck_1469_;
goto v_resetjp_1459_;
}
else
{
lean_inc(v_maxArgs_1454_);
lean_dec(v_config_1306_);
v___x_1460_ = lean_box(0);
v_isShared_1461_ = v_isSharedCheck_1469_;
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
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v_maxArgs_1454_);
lean_ctor_set_uint8(v_reuseFailAlloc_1468_, sizeof(void*)*1, v_closePre_1447_);
lean_ctor_set_uint8(v_reuseFailAlloc_1468_, sizeof(void*)*1 + 1, v_closePost_1448_);
v___x_1463_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1462_;
}
v_reusejp_1462_:
{
uint8_t v___x_1464_; lean_object* v___x_1466_; 
v___x_1464_ = lean_unbox(v_a_1443_);
lean_dec(v_a_1443_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 2, v___x_1464_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 3, v_preTransparency_1449_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 4, v_postTransparency_1450_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 5, v_preferLHS_1451_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 6, v_partialApp_1452_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 7, v_sameFun_1453_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 8, v_typeEqs_1455_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 9, v_etaExpand_1456_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 10, v_useCongrSimp_1457_);
lean_ctor_set_uint8(v___x_1463_, sizeof(void*)*1 + 11, v_beqEq_1458_);
if (v_isShared_1446_ == 0)
{
lean_ctor_set(v___x_1445_, 0, v___x_1463_);
v___x_1466_ = v___x_1445_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v___x_1463_);
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
lean_object* v_a_1471_; lean_object* v___x_1473_; uint8_t v_isShared_1474_; uint8_t v_isSharedCheck_1478_; 
lean_dec_ref(v_config_1306_);
v_a_1471_ = lean_ctor_get(v___x_1442_, 0);
v_isSharedCheck_1478_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1478_ == 0)
{
v___x_1473_ = v___x_1442_;
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
else
{
lean_inc(v_a_1471_);
lean_dec(v___x_1442_);
v___x_1473_ = lean_box(0);
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
v_resetjp_1472_:
{
lean_object* v___x_1476_; 
if (v_isShared_1474_ == 0)
{
v___x_1476_ = v___x_1473_;
goto v_reusejp_1475_;
}
else
{
lean_object* v_reuseFailAlloc_1477_; 
v_reuseFailAlloc_1477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1477_, 0, v_a_1471_);
v___x_1476_ = v_reuseFailAlloc_1477_;
goto v_reusejp_1475_;
}
v_reusejp_1475_:
{
return v___x_1476_;
}
}
}
}
else
{
lean_object* v_a_1479_; lean_object* v___x_1481_; uint8_t v_isShared_1482_; uint8_t v_isSharedCheck_1486_; 
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1479_ = lean_ctor_get(v___x_1440_, 0);
v_isSharedCheck_1486_ = !lean_is_exclusive(v___x_1440_);
if (v_isSharedCheck_1486_ == 0)
{
v___x_1481_ = v___x_1440_;
v_isShared_1482_ = v_isSharedCheck_1486_;
goto v_resetjp_1480_;
}
else
{
lean_inc(v_a_1479_);
lean_dec(v___x_1440_);
v___x_1481_ = lean_box(0);
v_isShared_1482_ = v_isSharedCheck_1486_;
goto v_resetjp_1480_;
}
v_resetjp_1480_:
{
lean_object* v___x_1484_; 
if (v_isShared_1482_ == 0)
{
v___x_1484_ = v___x_1481_;
goto v_reusejp_1483_;
}
else
{
lean_object* v_reuseFailAlloc_1485_; 
v_reuseFailAlloc_1485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1485_, 0, v_a_1479_);
v___x_1484_ = v_reuseFailAlloc_1485_;
goto v_reusejp_1483_;
}
v_reusejp_1483_:
{
return v___x_1484_;
}
}
}
}
}
else
{
lean_object* v_a_1487_; lean_object* v___x_1489_; uint8_t v_isShared_1490_; uint8_t v_isSharedCheck_1494_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1487_ = lean_ctor_get(v___x_1438_, 0);
v_isSharedCheck_1494_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1494_ == 0)
{
v___x_1489_ = v___x_1438_;
v_isShared_1490_ = v_isSharedCheck_1494_;
goto v_resetjp_1488_;
}
else
{
lean_inc(v_a_1487_);
lean_dec(v___x_1438_);
v___x_1489_ = lean_box(0);
v_isShared_1490_ = v_isSharedCheck_1494_;
goto v_resetjp_1488_;
}
v_resetjp_1488_:
{
lean_object* v___x_1492_; 
if (v_isShared_1490_ == 0)
{
v___x_1492_ = v___x_1489_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1493_; 
v_reuseFailAlloc_1493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1493_, 0, v_a_1487_);
v___x_1492_ = v_reuseFailAlloc_1493_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
return v___x_1492_;
}
}
}
}
}
else
{
lean_object* v___x_1495_; lean_object* v___x_1496_; 
lean_dec_ref(v___x_1328_);
v___x_1495_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9));
v___x_1496_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1495_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1496_) == 0)
{
uint8_t v___x_1497_; 
lean_dec_ref_known(v___x_1496_, 1);
v___x_1497_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1497_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1498_; 
lean_dec_ref(v___x_1329_);
v___x_1498_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1498_) == 0)
{
lean_object* v_a_1499_; lean_object* v___x_1501_; uint8_t v_isShared_1502_; uint8_t v_isSharedCheck_1526_; 
v_a_1499_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1526_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1526_ == 0)
{
v___x_1501_ = v___x_1498_;
v_isShared_1502_ = v_isSharedCheck_1526_;
goto v_resetjp_1500_;
}
else
{
lean_inc(v_a_1499_);
lean_dec(v___x_1498_);
v___x_1501_ = lean_box(0);
v_isShared_1502_ = v_isSharedCheck_1526_;
goto v_resetjp_1500_;
}
v_resetjp_1500_:
{
uint8_t v_closePre_1503_; uint8_t v_closePost_1504_; uint8_t v_transparency_1505_; uint8_t v_preTransparency_1506_; uint8_t v_postTransparency_1507_; uint8_t v_preferLHS_1508_; uint8_t v_partialApp_1509_; lean_object* v_maxArgs_1510_; uint8_t v_typeEqs_1511_; uint8_t v_etaExpand_1512_; uint8_t v_useCongrSimp_1513_; uint8_t v_beqEq_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1525_; 
v_closePre_1503_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1504_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1505_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1506_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1507_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1508_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1509_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_maxArgs_1510_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1511_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1512_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1513_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1514_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1525_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1525_ == 0)
{
v___x_1516_ = v_config_1306_;
v_isShared_1517_ = v_isSharedCheck_1525_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_maxArgs_1510_);
lean_dec(v_config_1306_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1525_;
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
lean_object* v_reuseFailAlloc_1524_; 
v_reuseFailAlloc_1524_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1524_, 0, v_maxArgs_1510_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1, v_closePre_1503_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 1, v_closePost_1504_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 2, v_transparency_1505_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 3, v_preTransparency_1506_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 4, v_postTransparency_1507_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 5, v_preferLHS_1508_);
lean_ctor_set_uint8(v_reuseFailAlloc_1524_, sizeof(void*)*1 + 6, v_partialApp_1509_);
v___x_1519_ = v_reuseFailAlloc_1524_;
goto v_reusejp_1518_;
}
v_reusejp_1518_:
{
uint8_t v___x_1520_; lean_object* v___x_1522_; 
v___x_1520_ = lean_unbox(v_a_1499_);
lean_dec(v_a_1499_);
lean_ctor_set_uint8(v___x_1519_, sizeof(void*)*1 + 7, v___x_1520_);
lean_ctor_set_uint8(v___x_1519_, sizeof(void*)*1 + 8, v_typeEqs_1511_);
lean_ctor_set_uint8(v___x_1519_, sizeof(void*)*1 + 9, v_etaExpand_1512_);
lean_ctor_set_uint8(v___x_1519_, sizeof(void*)*1 + 10, v_useCongrSimp_1513_);
lean_ctor_set_uint8(v___x_1519_, sizeof(void*)*1 + 11, v_beqEq_1514_);
if (v_isShared_1502_ == 0)
{
lean_ctor_set(v___x_1501_, 0, v___x_1519_);
v___x_1522_ = v___x_1501_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1523_; 
v_reuseFailAlloc_1523_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1523_, 0, v___x_1519_);
v___x_1522_ = v_reuseFailAlloc_1523_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
return v___x_1522_;
}
}
}
}
}
else
{
lean_object* v_a_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1534_; 
lean_dec_ref(v_config_1306_);
v_a_1527_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1534_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1534_ == 0)
{
v___x_1529_ = v___x_1498_;
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_a_1527_);
lean_dec(v___x_1498_);
v___x_1529_ = lean_box(0);
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
v_resetjp_1528_:
{
lean_object* v___x_1532_; 
if (v_isShared_1530_ == 0)
{
v___x_1532_ = v___x_1529_;
goto v_reusejp_1531_;
}
else
{
lean_object* v_reuseFailAlloc_1533_; 
v_reuseFailAlloc_1533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1533_, 0, v_a_1527_);
v___x_1532_ = v_reuseFailAlloc_1533_;
goto v_reusejp_1531_;
}
v_reusejp_1531_:
{
return v___x_1532_;
}
}
}
}
}
else
{
lean_object* v_a_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1542_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1535_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1537_ = v___x_1496_;
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_a_1535_);
lean_dec(v___x_1496_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1542_;
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
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v_a_1535_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
}
}
}
else
{
uint8_t v___x_1543_; 
v___x_1543_ = lean_string_dec_eq(v___x_1328_, v___x_1330_);
if (v___x_1543_ == 0)
{
lean_object* v___x_1544_; uint8_t v___x_1545_; 
v___x_1544_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10));
v___x_1545_ = lean_string_dec_eq(v___x_1328_, v___x_1544_);
if (v___x_1545_ == 0)
{
lean_object* v___x_1546_; uint8_t v___x_1547_; 
v___x_1546_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11));
v___x_1547_ = lean_string_dec_eq(v___x_1328_, v___x_1546_);
lean_dec_ref(v___x_1328_);
if (v___x_1547_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1548_; lean_object* v___x_1549_; 
v___x_1548_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12));
v___x_1549_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1548_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1549_) == 0)
{
uint8_t v___x_1550_; 
lean_dec_ref_known(v___x_1549_, 1);
v___x_1550_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1550_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1551_; 
lean_dec_ref(v___x_1329_);
v___x_1551_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1551_) == 0)
{
lean_object* v_a_1552_; lean_object* v___x_1554_; uint8_t v_isShared_1555_; uint8_t v_isSharedCheck_1579_; 
v_a_1552_ = lean_ctor_get(v___x_1551_, 0);
v_isSharedCheck_1579_ = !lean_is_exclusive(v___x_1551_);
if (v_isSharedCheck_1579_ == 0)
{
v___x_1554_ = v___x_1551_;
v_isShared_1555_ = v_isSharedCheck_1579_;
goto v_resetjp_1553_;
}
else
{
lean_inc(v_a_1552_);
lean_dec(v___x_1551_);
v___x_1554_ = lean_box(0);
v_isShared_1555_ = v_isSharedCheck_1579_;
goto v_resetjp_1553_;
}
v_resetjp_1553_:
{
uint8_t v_closePre_1556_; uint8_t v_closePost_1557_; uint8_t v_transparency_1558_; uint8_t v_preTransparency_1559_; uint8_t v_postTransparency_1560_; uint8_t v_partialApp_1561_; uint8_t v_sameFun_1562_; lean_object* v_maxArgs_1563_; uint8_t v_typeEqs_1564_; uint8_t v_etaExpand_1565_; uint8_t v_useCongrSimp_1566_; uint8_t v_beqEq_1567_; lean_object* v___x_1569_; uint8_t v_isShared_1570_; uint8_t v_isSharedCheck_1578_; 
v_closePre_1556_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1557_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1558_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1559_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1560_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_partialApp_1561_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1562_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1563_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1564_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1565_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1566_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1567_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1578_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1578_ == 0)
{
v___x_1569_ = v_config_1306_;
v_isShared_1570_ = v_isSharedCheck_1578_;
goto v_resetjp_1568_;
}
else
{
lean_inc(v_maxArgs_1563_);
lean_dec(v_config_1306_);
v___x_1569_ = lean_box(0);
v_isShared_1570_ = v_isSharedCheck_1578_;
goto v_resetjp_1568_;
}
v_resetjp_1568_:
{
lean_object* v___x_1572_; 
if (v_isShared_1570_ == 0)
{
v___x_1572_ = v___x_1569_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1577_; 
v_reuseFailAlloc_1577_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1577_, 0, v_maxArgs_1563_);
lean_ctor_set_uint8(v_reuseFailAlloc_1577_, sizeof(void*)*1, v_closePre_1556_);
lean_ctor_set_uint8(v_reuseFailAlloc_1577_, sizeof(void*)*1 + 1, v_closePost_1557_);
lean_ctor_set_uint8(v_reuseFailAlloc_1577_, sizeof(void*)*1 + 2, v_transparency_1558_);
lean_ctor_set_uint8(v_reuseFailAlloc_1577_, sizeof(void*)*1 + 3, v_preTransparency_1559_);
lean_ctor_set_uint8(v_reuseFailAlloc_1577_, sizeof(void*)*1 + 4, v_postTransparency_1560_);
v___x_1572_ = v_reuseFailAlloc_1577_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
uint8_t v___x_1573_; lean_object* v___x_1575_; 
v___x_1573_ = lean_unbox(v_a_1552_);
lean_dec(v_a_1552_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 5, v___x_1573_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 6, v_partialApp_1561_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 7, v_sameFun_1562_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 8, v_typeEqs_1564_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 9, v_etaExpand_1565_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 10, v_useCongrSimp_1566_);
lean_ctor_set_uint8(v___x_1572_, sizeof(void*)*1 + 11, v_beqEq_1567_);
if (v_isShared_1555_ == 0)
{
lean_ctor_set(v___x_1554_, 0, v___x_1572_);
v___x_1575_ = v___x_1554_;
goto v_reusejp_1574_;
}
else
{
lean_object* v_reuseFailAlloc_1576_; 
v_reuseFailAlloc_1576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1576_, 0, v___x_1572_);
v___x_1575_ = v_reuseFailAlloc_1576_;
goto v_reusejp_1574_;
}
v_reusejp_1574_:
{
return v___x_1575_;
}
}
}
}
}
else
{
lean_object* v_a_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1587_; 
lean_dec_ref(v_config_1306_);
v_a_1580_ = lean_ctor_get(v___x_1551_, 0);
v_isSharedCheck_1587_ = !lean_is_exclusive(v___x_1551_);
if (v_isSharedCheck_1587_ == 0)
{
v___x_1582_ = v___x_1551_;
v_isShared_1583_ = v_isSharedCheck_1587_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_a_1580_);
lean_dec(v___x_1551_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1587_;
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
lean_object* v_reuseFailAlloc_1586_; 
v_reuseFailAlloc_1586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1586_, 0, v_a_1580_);
v___x_1585_ = v_reuseFailAlloc_1586_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
return v___x_1585_;
}
}
}
}
}
else
{
lean_object* v_a_1588_; lean_object* v___x_1590_; uint8_t v_isShared_1591_; uint8_t v_isSharedCheck_1595_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1588_ = lean_ctor_get(v___x_1549_, 0);
v_isSharedCheck_1595_ = !lean_is_exclusive(v___x_1549_);
if (v_isSharedCheck_1595_ == 0)
{
v___x_1590_ = v___x_1549_;
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
else
{
lean_inc(v_a_1588_);
lean_dec(v___x_1549_);
v___x_1590_ = lean_box(0);
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
v_resetjp_1589_:
{
lean_object* v___x_1593_; 
if (v_isShared_1591_ == 0)
{
v___x_1593_ = v___x_1590_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v_a_1588_);
v___x_1593_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
return v___x_1593_;
}
}
}
}
}
else
{
lean_object* v___x_1596_; lean_object* v___x_1597_; 
lean_dec_ref(v___x_1328_);
v___x_1596_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13));
v___x_1597_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1596_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1597_) == 0)
{
uint8_t v___x_1598_; 
lean_dec_ref_known(v___x_1597_, 1);
v___x_1598_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1598_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1599_; 
lean_dec_ref(v___x_1329_);
lean_inc_ref(v_item_1307_);
v___x_1599_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1599_) == 0)
{
lean_object* v_value_1600_; lean_object* v___x_1601_; 
lean_dec_ref_known(v___x_1599_, 1);
v_value_1600_ = lean_ctor_get(v_item_1307_, 2);
lean_inc(v_value_1600_);
lean_dec_ref(v_item_1307_);
v___x_1601_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_1600_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1601_) == 0)
{
lean_object* v_a_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1629_; 
v_a_1602_ = lean_ctor_get(v___x_1601_, 0);
v_isSharedCheck_1629_ = !lean_is_exclusive(v___x_1601_);
if (v_isSharedCheck_1629_ == 0)
{
v___x_1604_ = v___x_1601_;
v_isShared_1605_ = v_isSharedCheck_1629_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_a_1602_);
lean_dec(v___x_1601_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1629_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
uint8_t v_closePre_1606_; uint8_t v_closePost_1607_; uint8_t v_transparency_1608_; uint8_t v_postTransparency_1609_; uint8_t v_preferLHS_1610_; uint8_t v_partialApp_1611_; uint8_t v_sameFun_1612_; lean_object* v_maxArgs_1613_; uint8_t v_typeEqs_1614_; uint8_t v_etaExpand_1615_; uint8_t v_useCongrSimp_1616_; uint8_t v_beqEq_1617_; lean_object* v___x_1619_; uint8_t v_isShared_1620_; uint8_t v_isSharedCheck_1628_; 
v_closePre_1606_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1607_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1608_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_postTransparency_1609_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1610_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1611_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1612_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1613_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1614_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1615_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1616_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1617_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1628_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1628_ == 0)
{
v___x_1619_ = v_config_1306_;
v_isShared_1620_ = v_isSharedCheck_1628_;
goto v_resetjp_1618_;
}
else
{
lean_inc(v_maxArgs_1613_);
lean_dec(v_config_1306_);
v___x_1619_ = lean_box(0);
v_isShared_1620_ = v_isSharedCheck_1628_;
goto v_resetjp_1618_;
}
v_resetjp_1618_:
{
lean_object* v___x_1622_; 
if (v_isShared_1620_ == 0)
{
v___x_1622_ = v___x_1619_;
goto v_reusejp_1621_;
}
else
{
lean_object* v_reuseFailAlloc_1627_; 
v_reuseFailAlloc_1627_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1627_, 0, v_maxArgs_1613_);
lean_ctor_set_uint8(v_reuseFailAlloc_1627_, sizeof(void*)*1, v_closePre_1606_);
lean_ctor_set_uint8(v_reuseFailAlloc_1627_, sizeof(void*)*1 + 1, v_closePost_1607_);
lean_ctor_set_uint8(v_reuseFailAlloc_1627_, sizeof(void*)*1 + 2, v_transparency_1608_);
v___x_1622_ = v_reuseFailAlloc_1627_;
goto v_reusejp_1621_;
}
v_reusejp_1621_:
{
uint8_t v___x_1623_; lean_object* v___x_1625_; 
v___x_1623_ = lean_unbox(v_a_1602_);
lean_dec(v_a_1602_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 3, v___x_1623_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 4, v_postTransparency_1609_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 5, v_preferLHS_1610_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 6, v_partialApp_1611_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 7, v_sameFun_1612_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 8, v_typeEqs_1614_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 9, v_etaExpand_1615_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 10, v_useCongrSimp_1616_);
lean_ctor_set_uint8(v___x_1622_, sizeof(void*)*1 + 11, v_beqEq_1617_);
if (v_isShared_1605_ == 0)
{
lean_ctor_set(v___x_1604_, 0, v___x_1622_);
v___x_1625_ = v___x_1604_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1626_; 
v_reuseFailAlloc_1626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1626_, 0, v___x_1622_);
v___x_1625_ = v_reuseFailAlloc_1626_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
return v___x_1625_;
}
}
}
}
}
else
{
lean_object* v_a_1630_; lean_object* v___x_1632_; uint8_t v_isShared_1633_; uint8_t v_isSharedCheck_1637_; 
lean_dec_ref(v_config_1306_);
v_a_1630_ = lean_ctor_get(v___x_1601_, 0);
v_isSharedCheck_1637_ = !lean_is_exclusive(v___x_1601_);
if (v_isSharedCheck_1637_ == 0)
{
v___x_1632_ = v___x_1601_;
v_isShared_1633_ = v_isSharedCheck_1637_;
goto v_resetjp_1631_;
}
else
{
lean_inc(v_a_1630_);
lean_dec(v___x_1601_);
v___x_1632_ = lean_box(0);
v_isShared_1633_ = v_isSharedCheck_1637_;
goto v_resetjp_1631_;
}
v_resetjp_1631_:
{
lean_object* v___x_1635_; 
if (v_isShared_1633_ == 0)
{
v___x_1635_ = v___x_1632_;
goto v_reusejp_1634_;
}
else
{
lean_object* v_reuseFailAlloc_1636_; 
v_reuseFailAlloc_1636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1636_, 0, v_a_1630_);
v___x_1635_ = v_reuseFailAlloc_1636_;
goto v_reusejp_1634_;
}
v_reusejp_1634_:
{
return v___x_1635_;
}
}
}
}
else
{
lean_object* v_a_1638_; lean_object* v___x_1640_; uint8_t v_isShared_1641_; uint8_t v_isSharedCheck_1645_; 
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1638_ = lean_ctor_get(v___x_1599_, 0);
v_isSharedCheck_1645_ = !lean_is_exclusive(v___x_1599_);
if (v_isSharedCheck_1645_ == 0)
{
v___x_1640_ = v___x_1599_;
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_a_1638_);
lean_dec(v___x_1599_);
v___x_1640_ = lean_box(0);
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
v_resetjp_1639_:
{
lean_object* v___x_1643_; 
if (v_isShared_1641_ == 0)
{
v___x_1643_ = v___x_1640_;
goto v_reusejp_1642_;
}
else
{
lean_object* v_reuseFailAlloc_1644_; 
v_reuseFailAlloc_1644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1644_, 0, v_a_1638_);
v___x_1643_ = v_reuseFailAlloc_1644_;
goto v_reusejp_1642_;
}
v_reusejp_1642_:
{
return v___x_1643_;
}
}
}
}
}
else
{
lean_object* v_a_1646_; lean_object* v___x_1648_; uint8_t v_isShared_1649_; uint8_t v_isSharedCheck_1653_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1646_ = lean_ctor_get(v___x_1597_, 0);
v_isSharedCheck_1653_ = !lean_is_exclusive(v___x_1597_);
if (v_isSharedCheck_1653_ == 0)
{
v___x_1648_ = v___x_1597_;
v_isShared_1649_ = v_isSharedCheck_1653_;
goto v_resetjp_1647_;
}
else
{
lean_inc(v_a_1646_);
lean_dec(v___x_1597_);
v___x_1648_ = lean_box(0);
v_isShared_1649_ = v_isSharedCheck_1653_;
goto v_resetjp_1647_;
}
v_resetjp_1647_:
{
lean_object* v___x_1651_; 
if (v_isShared_1649_ == 0)
{
v___x_1651_ = v___x_1648_;
goto v_reusejp_1650_;
}
else
{
lean_object* v_reuseFailAlloc_1652_; 
v_reuseFailAlloc_1652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1652_, 0, v_a_1646_);
v___x_1651_ = v_reuseFailAlloc_1652_;
goto v_reusejp_1650_;
}
v_reusejp_1650_:
{
return v___x_1651_;
}
}
}
}
}
else
{
lean_object* v___x_1654_; lean_object* v___x_1655_; 
lean_dec_ref(v___x_1328_);
v___x_1654_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14));
v___x_1655_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1654_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1655_) == 0)
{
uint8_t v___x_1656_; 
lean_dec_ref_known(v___x_1655_, 1);
v___x_1656_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1656_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1657_; 
lean_dec_ref(v___x_1329_);
lean_inc_ref(v_item_1307_);
v___x_1657_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1657_) == 0)
{
lean_object* v_value_1658_; lean_object* v___x_1659_; 
lean_dec_ref_known(v___x_1657_, 1);
v_value_1658_ = lean_ctor_get(v_item_1307_, 2);
lean_inc(v_value_1658_);
lean_dec_ref(v_item_1307_);
v___x_1659_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_1658_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_object* v_a_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1687_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1687_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1687_ == 0)
{
v___x_1662_ = v___x_1659_;
v_isShared_1663_ = v_isSharedCheck_1687_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_a_1660_);
lean_dec(v___x_1659_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1687_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
uint8_t v_closePre_1664_; uint8_t v_closePost_1665_; uint8_t v_transparency_1666_; uint8_t v_preTransparency_1667_; uint8_t v_preferLHS_1668_; uint8_t v_partialApp_1669_; uint8_t v_sameFun_1670_; lean_object* v_maxArgs_1671_; uint8_t v_typeEqs_1672_; uint8_t v_etaExpand_1673_; uint8_t v_useCongrSimp_1674_; uint8_t v_beqEq_1675_; lean_object* v___x_1677_; uint8_t v_isShared_1678_; uint8_t v_isSharedCheck_1686_; 
v_closePre_1664_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1665_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1666_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1667_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_preferLHS_1668_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1669_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1670_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1671_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1672_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1673_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1674_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1675_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1686_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1677_ = v_config_1306_;
v_isShared_1678_ = v_isSharedCheck_1686_;
goto v_resetjp_1676_;
}
else
{
lean_inc(v_maxArgs_1671_);
lean_dec(v_config_1306_);
v___x_1677_ = lean_box(0);
v_isShared_1678_ = v_isSharedCheck_1686_;
goto v_resetjp_1676_;
}
v_resetjp_1676_:
{
lean_object* v___x_1680_; 
if (v_isShared_1678_ == 0)
{
v___x_1680_ = v___x_1677_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_maxArgs_1671_);
lean_ctor_set_uint8(v_reuseFailAlloc_1685_, sizeof(void*)*1, v_closePre_1664_);
lean_ctor_set_uint8(v_reuseFailAlloc_1685_, sizeof(void*)*1 + 1, v_closePost_1665_);
lean_ctor_set_uint8(v_reuseFailAlloc_1685_, sizeof(void*)*1 + 2, v_transparency_1666_);
lean_ctor_set_uint8(v_reuseFailAlloc_1685_, sizeof(void*)*1 + 3, v_preTransparency_1667_);
v___x_1680_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
uint8_t v___x_1681_; lean_object* v___x_1683_; 
v___x_1681_ = lean_unbox(v_a_1660_);
lean_dec(v_a_1660_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 4, v___x_1681_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 5, v_preferLHS_1668_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 6, v_partialApp_1669_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 7, v_sameFun_1670_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 8, v_typeEqs_1672_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 9, v_etaExpand_1673_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 10, v_useCongrSimp_1674_);
lean_ctor_set_uint8(v___x_1680_, sizeof(void*)*1 + 11, v_beqEq_1675_);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v___x_1680_);
v___x_1683_ = v___x_1662_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1684_; 
v_reuseFailAlloc_1684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1684_, 0, v___x_1680_);
v___x_1683_ = v_reuseFailAlloc_1684_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
return v___x_1683_;
}
}
}
}
}
else
{
lean_object* v_a_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1695_; 
lean_dec_ref(v_config_1306_);
v_a_1688_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1695_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1695_ == 0)
{
v___x_1690_ = v___x_1659_;
v_isShared_1691_ = v_isSharedCheck_1695_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_a_1688_);
lean_dec(v___x_1659_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1695_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v___x_1693_; 
if (v_isShared_1691_ == 0)
{
v___x_1693_ = v___x_1690_;
goto v_reusejp_1692_;
}
else
{
lean_object* v_reuseFailAlloc_1694_; 
v_reuseFailAlloc_1694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1694_, 0, v_a_1688_);
v___x_1693_ = v_reuseFailAlloc_1694_;
goto v_reusejp_1692_;
}
v_reusejp_1692_:
{
return v___x_1693_;
}
}
}
}
else
{
lean_object* v_a_1696_; lean_object* v___x_1698_; uint8_t v_isShared_1699_; uint8_t v_isSharedCheck_1703_; 
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1696_ = lean_ctor_get(v___x_1657_, 0);
v_isSharedCheck_1703_ = !lean_is_exclusive(v___x_1657_);
if (v_isSharedCheck_1703_ == 0)
{
v___x_1698_ = v___x_1657_;
v_isShared_1699_ = v_isSharedCheck_1703_;
goto v_resetjp_1697_;
}
else
{
lean_inc(v_a_1696_);
lean_dec(v___x_1657_);
v___x_1698_ = lean_box(0);
v_isShared_1699_ = v_isSharedCheck_1703_;
goto v_resetjp_1697_;
}
v_resetjp_1697_:
{
lean_object* v___x_1701_; 
if (v_isShared_1699_ == 0)
{
v___x_1701_ = v___x_1698_;
goto v_reusejp_1700_;
}
else
{
lean_object* v_reuseFailAlloc_1702_; 
v_reuseFailAlloc_1702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1702_, 0, v_a_1696_);
v___x_1701_ = v_reuseFailAlloc_1702_;
goto v_reusejp_1700_;
}
v_reusejp_1700_:
{
return v___x_1701_;
}
}
}
}
}
else
{
lean_object* v_a_1704_; lean_object* v___x_1706_; uint8_t v_isShared_1707_; uint8_t v_isSharedCheck_1711_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1704_ = lean_ctor_get(v___x_1655_, 0);
v_isSharedCheck_1711_ = !lean_is_exclusive(v___x_1655_);
if (v_isSharedCheck_1711_ == 0)
{
v___x_1706_ = v___x_1655_;
v_isShared_1707_ = v_isSharedCheck_1711_;
goto v_resetjp_1705_;
}
else
{
lean_inc(v_a_1704_);
lean_dec(v___x_1655_);
v___x_1706_ = lean_box(0);
v_isShared_1707_ = v_isSharedCheck_1711_;
goto v_resetjp_1705_;
}
v_resetjp_1705_:
{
lean_object* v___x_1709_; 
if (v_isShared_1707_ == 0)
{
v___x_1709_ = v___x_1706_;
goto v_reusejp_1708_;
}
else
{
lean_object* v_reuseFailAlloc_1710_; 
v_reuseFailAlloc_1710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1710_, 0, v_a_1704_);
v___x_1709_ = v_reuseFailAlloc_1710_;
goto v_reusejp_1708_;
}
v_reusejp_1708_:
{
return v___x_1709_;
}
}
}
}
}
}
else
{
lean_object* v___x_1712_; uint8_t v___x_1713_; 
v___x_1712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__15));
v___x_1713_ = lean_string_dec_lt(v___x_1328_, v___x_1712_);
if (v___x_1713_ == 0)
{
uint8_t v___x_1714_; 
v___x_1714_ = lean_string_dec_eq(v___x_1328_, v___x_1712_);
if (v___x_1714_ == 0)
{
lean_object* v___x_1715_; uint8_t v___x_1716_; 
v___x_1715_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16));
v___x_1716_ = lean_string_dec_eq(v___x_1328_, v___x_1715_);
if (v___x_1716_ == 0)
{
lean_object* v___x_1717_; uint8_t v___x_1718_; 
v___x_1717_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17));
v___x_1718_ = lean_string_dec_eq(v___x_1328_, v___x_1717_);
if (v___x_1718_ == 0)
{
lean_object* v___x_1719_; uint8_t v___x_1720_; 
v___x_1719_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18));
v___x_1720_ = lean_string_dec_eq(v___x_1328_, v___x_1719_);
lean_dec_ref(v___x_1328_);
if (v___x_1720_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1721_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19));
v___x_1722_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1721_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1722_) == 0)
{
uint8_t v___x_1723_; 
lean_dec_ref_known(v___x_1722_, 1);
v___x_1723_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1723_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1724_; 
lean_dec_ref(v___x_1329_);
v___x_1724_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1724_) == 0)
{
lean_object* v_a_1725_; lean_object* v___x_1727_; uint8_t v_isShared_1728_; uint8_t v_isSharedCheck_1752_; 
v_a_1725_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1752_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1752_ == 0)
{
v___x_1727_ = v___x_1724_;
v_isShared_1728_ = v_isSharedCheck_1752_;
goto v_resetjp_1726_;
}
else
{
lean_inc(v_a_1725_);
lean_dec(v___x_1724_);
v___x_1727_ = lean_box(0);
v_isShared_1728_ = v_isSharedCheck_1752_;
goto v_resetjp_1726_;
}
v_resetjp_1726_:
{
uint8_t v_closePre_1729_; uint8_t v_closePost_1730_; uint8_t v_transparency_1731_; uint8_t v_preTransparency_1732_; uint8_t v_postTransparency_1733_; uint8_t v_preferLHS_1734_; uint8_t v_sameFun_1735_; lean_object* v_maxArgs_1736_; uint8_t v_typeEqs_1737_; uint8_t v_etaExpand_1738_; uint8_t v_useCongrSimp_1739_; uint8_t v_beqEq_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1751_; 
v_closePre_1729_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1730_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1731_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1732_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1733_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1734_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_sameFun_1735_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1736_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1737_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1738_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1739_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1740_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1751_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1751_ == 0)
{
v___x_1742_ = v_config_1306_;
v_isShared_1743_ = v_isSharedCheck_1751_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_maxArgs_1736_);
lean_dec(v_config_1306_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1751_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1745_; 
if (v_isShared_1743_ == 0)
{
v___x_1745_ = v___x_1742_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1750_; 
v_reuseFailAlloc_1750_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1750_, 0, v_maxArgs_1736_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1, v_closePre_1729_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1 + 1, v_closePost_1730_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1 + 2, v_transparency_1731_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1 + 3, v_preTransparency_1732_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1 + 4, v_postTransparency_1733_);
lean_ctor_set_uint8(v_reuseFailAlloc_1750_, sizeof(void*)*1 + 5, v_preferLHS_1734_);
v___x_1745_ = v_reuseFailAlloc_1750_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
uint8_t v___x_1746_; lean_object* v___x_1748_; 
v___x_1746_ = lean_unbox(v_a_1725_);
lean_dec(v_a_1725_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 6, v___x_1746_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 7, v_sameFun_1735_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 8, v_typeEqs_1737_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 9, v_etaExpand_1738_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 10, v_useCongrSimp_1739_);
lean_ctor_set_uint8(v___x_1745_, sizeof(void*)*1 + 11, v_beqEq_1740_);
if (v_isShared_1728_ == 0)
{
lean_ctor_set(v___x_1727_, 0, v___x_1745_);
v___x_1748_ = v___x_1727_;
goto v_reusejp_1747_;
}
else
{
lean_object* v_reuseFailAlloc_1749_; 
v_reuseFailAlloc_1749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1749_, 0, v___x_1745_);
v___x_1748_ = v_reuseFailAlloc_1749_;
goto v_reusejp_1747_;
}
v_reusejp_1747_:
{
return v___x_1748_;
}
}
}
}
}
else
{
lean_object* v_a_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1760_; 
lean_dec_ref(v_config_1306_);
v_a_1753_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1760_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1760_ == 0)
{
v___x_1755_ = v___x_1724_;
v_isShared_1756_ = v_isSharedCheck_1760_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_a_1753_);
lean_dec(v___x_1724_);
v___x_1755_ = lean_box(0);
v_isShared_1756_ = v_isSharedCheck_1760_;
goto v_resetjp_1754_;
}
v_resetjp_1754_:
{
lean_object* v___x_1758_; 
if (v_isShared_1756_ == 0)
{
v___x_1758_ = v___x_1755_;
goto v_reusejp_1757_;
}
else
{
lean_object* v_reuseFailAlloc_1759_; 
v_reuseFailAlloc_1759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1759_, 0, v_a_1753_);
v___x_1758_ = v_reuseFailAlloc_1759_;
goto v_reusejp_1757_;
}
v_reusejp_1757_:
{
return v___x_1758_;
}
}
}
}
}
else
{
lean_object* v_a_1761_; lean_object* v___x_1763_; uint8_t v_isShared_1764_; uint8_t v_isSharedCheck_1768_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1761_ = lean_ctor_get(v___x_1722_, 0);
v_isSharedCheck_1768_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1768_ == 0)
{
v___x_1763_ = v___x_1722_;
v_isShared_1764_ = v_isSharedCheck_1768_;
goto v_resetjp_1762_;
}
else
{
lean_inc(v_a_1761_);
lean_dec(v___x_1722_);
v___x_1763_ = lean_box(0);
v_isShared_1764_ = v_isSharedCheck_1768_;
goto v_resetjp_1762_;
}
v_resetjp_1762_:
{
lean_object* v___x_1766_; 
if (v_isShared_1764_ == 0)
{
v___x_1766_ = v___x_1763_;
goto v_reusejp_1765_;
}
else
{
lean_object* v_reuseFailAlloc_1767_; 
v_reuseFailAlloc_1767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1767_, 0, v_a_1761_);
v___x_1766_ = v_reuseFailAlloc_1767_;
goto v_reusejp_1765_;
}
v_reusejp_1765_:
{
return v___x_1766_;
}
}
}
}
}
else
{
lean_object* v___x_1769_; lean_object* v___x_1770_; 
lean_dec_ref(v___x_1328_);
v___x_1769_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20));
v___x_1770_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1769_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1770_) == 0)
{
uint8_t v___x_1771_; 
lean_dec_ref_known(v___x_1770_, 1);
v___x_1771_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1771_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1772_; 
lean_dec_ref(v___x_1329_);
lean_inc_ref(v_item_1307_);
v___x_1772_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v_value_1773_; lean_object* v___x_1774_; 
lean_dec_ref_known(v___x_1772_, 1);
v_value_1773_ = lean_ctor_get(v_item_1307_, 2);
lean_inc(v_value_1773_);
lean_dec_ref(v_item_1307_);
v___x_1774_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1(v_value_1773_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v_a_1775_; lean_object* v___x_1777_; uint8_t v_isShared_1778_; uint8_t v_isSharedCheck_1802_; 
v_a_1775_ = lean_ctor_get(v___x_1774_, 0);
v_isSharedCheck_1802_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1802_ == 0)
{
v___x_1777_ = v___x_1774_;
v_isShared_1778_ = v_isSharedCheck_1802_;
goto v_resetjp_1776_;
}
else
{
lean_inc(v_a_1775_);
lean_dec(v___x_1774_);
v___x_1777_ = lean_box(0);
v_isShared_1778_ = v_isSharedCheck_1802_;
goto v_resetjp_1776_;
}
v_resetjp_1776_:
{
uint8_t v_closePre_1779_; uint8_t v_closePost_1780_; uint8_t v_transparency_1781_; uint8_t v_preTransparency_1782_; uint8_t v_postTransparency_1783_; uint8_t v_preferLHS_1784_; uint8_t v_partialApp_1785_; uint8_t v_sameFun_1786_; uint8_t v_typeEqs_1787_; uint8_t v_etaExpand_1788_; uint8_t v_useCongrSimp_1789_; uint8_t v_beqEq_1790_; lean_object* v___x_1792_; uint8_t v_isShared_1793_; uint8_t v_isSharedCheck_1800_; 
v_closePre_1779_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1780_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1781_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1782_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1783_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1784_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1785_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1786_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_typeEqs_1787_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1788_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1789_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1790_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1800_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1800_ == 0)
{
lean_object* v_unused_1801_; 
v_unused_1801_ = lean_ctor_get(v_config_1306_, 0);
lean_dec(v_unused_1801_);
v___x_1792_ = v_config_1306_;
v_isShared_1793_ = v_isSharedCheck_1800_;
goto v_resetjp_1791_;
}
else
{
lean_dec(v_config_1306_);
v___x_1792_ = lean_box(0);
v_isShared_1793_ = v_isSharedCheck_1800_;
goto v_resetjp_1791_;
}
v_resetjp_1791_:
{
lean_object* v___x_1795_; 
if (v_isShared_1793_ == 0)
{
lean_ctor_set(v___x_1792_, 0, v_a_1775_);
v___x_1795_ = v___x_1792_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v_a_1775_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1, v_closePre_1779_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 1, v_closePost_1780_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 2, v_transparency_1781_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 3, v_preTransparency_1782_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 4, v_postTransparency_1783_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 5, v_preferLHS_1784_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 6, v_partialApp_1785_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 7, v_sameFun_1786_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 8, v_typeEqs_1787_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 9, v_etaExpand_1788_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 10, v_useCongrSimp_1789_);
lean_ctor_set_uint8(v_reuseFailAlloc_1799_, sizeof(void*)*1 + 11, v_beqEq_1790_);
v___x_1795_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
lean_object* v___x_1797_; 
if (v_isShared_1778_ == 0)
{
lean_ctor_set(v___x_1777_, 0, v___x_1795_);
v___x_1797_ = v___x_1777_;
goto v_reusejp_1796_;
}
else
{
lean_object* v_reuseFailAlloc_1798_; 
v_reuseFailAlloc_1798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1798_, 0, v___x_1795_);
v___x_1797_ = v_reuseFailAlloc_1798_;
goto v_reusejp_1796_;
}
v_reusejp_1796_:
{
return v___x_1797_;
}
}
}
}
}
else
{
lean_object* v_a_1803_; lean_object* v___x_1805_; uint8_t v_isShared_1806_; uint8_t v_isSharedCheck_1810_; 
lean_dec_ref(v_config_1306_);
v_a_1803_ = lean_ctor_get(v___x_1774_, 0);
v_isSharedCheck_1810_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1810_ == 0)
{
v___x_1805_ = v___x_1774_;
v_isShared_1806_ = v_isSharedCheck_1810_;
goto v_resetjp_1804_;
}
else
{
lean_inc(v_a_1803_);
lean_dec(v___x_1774_);
v___x_1805_ = lean_box(0);
v_isShared_1806_ = v_isSharedCheck_1810_;
goto v_resetjp_1804_;
}
v_resetjp_1804_:
{
lean_object* v___x_1808_; 
if (v_isShared_1806_ == 0)
{
v___x_1808_ = v___x_1805_;
goto v_reusejp_1807_;
}
else
{
lean_object* v_reuseFailAlloc_1809_; 
v_reuseFailAlloc_1809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1809_, 0, v_a_1803_);
v___x_1808_ = v_reuseFailAlloc_1809_;
goto v_reusejp_1807_;
}
v_reusejp_1807_:
{
return v___x_1808_;
}
}
}
}
else
{
lean_object* v_a_1811_; lean_object* v___x_1813_; uint8_t v_isShared_1814_; uint8_t v_isSharedCheck_1818_; 
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1811_ = lean_ctor_get(v___x_1772_, 0);
v_isSharedCheck_1818_ = !lean_is_exclusive(v___x_1772_);
if (v_isSharedCheck_1818_ == 0)
{
v___x_1813_ = v___x_1772_;
v_isShared_1814_ = v_isSharedCheck_1818_;
goto v_resetjp_1812_;
}
else
{
lean_inc(v_a_1811_);
lean_dec(v___x_1772_);
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
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1819_ = lean_ctor_get(v___x_1770_, 0);
v_isSharedCheck_1826_ = !lean_is_exclusive(v___x_1770_);
if (v_isSharedCheck_1826_ == 0)
{
v___x_1821_ = v___x_1770_;
v_isShared_1822_ = v_isSharedCheck_1826_;
goto v_resetjp_1820_;
}
else
{
lean_inc(v_a_1819_);
lean_dec(v___x_1770_);
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
}
else
{
lean_object* v___x_1827_; lean_object* v___x_1828_; 
lean_dec_ref(v___x_1328_);
v___x_1827_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21));
v___x_1828_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1827_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1828_) == 0)
{
uint8_t v___x_1829_; 
lean_dec_ref_known(v___x_1828_, 1);
v___x_1829_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1829_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1830_; 
lean_dec_ref(v___x_1329_);
v___x_1830_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1830_) == 0)
{
lean_object* v_a_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1858_; 
v_a_1831_ = lean_ctor_get(v___x_1830_, 0);
v_isSharedCheck_1858_ = !lean_is_exclusive(v___x_1830_);
if (v_isSharedCheck_1858_ == 0)
{
v___x_1833_ = v___x_1830_;
v_isShared_1834_ = v_isSharedCheck_1858_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_a_1831_);
lean_dec(v___x_1830_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1858_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
uint8_t v_closePre_1835_; uint8_t v_closePost_1836_; uint8_t v_transparency_1837_; uint8_t v_preTransparency_1838_; uint8_t v_postTransparency_1839_; uint8_t v_preferLHS_1840_; uint8_t v_partialApp_1841_; uint8_t v_sameFun_1842_; lean_object* v_maxArgs_1843_; uint8_t v_typeEqs_1844_; uint8_t v_useCongrSimp_1845_; uint8_t v_beqEq_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1857_; 
v_closePre_1835_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1836_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1837_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1838_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1839_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1840_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1841_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1842_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1843_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1844_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_useCongrSimp_1845_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1846_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1857_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1857_ == 0)
{
v___x_1848_ = v_config_1306_;
v_isShared_1849_ = v_isSharedCheck_1857_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_maxArgs_1843_);
lean_dec(v_config_1306_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1857_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v___x_1851_; 
if (v_isShared_1849_ == 0)
{
v___x_1851_ = v___x_1848_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1856_; 
v_reuseFailAlloc_1856_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1856_, 0, v_maxArgs_1843_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1, v_closePre_1835_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 1, v_closePost_1836_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 2, v_transparency_1837_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 3, v_preTransparency_1838_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 4, v_postTransparency_1839_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 5, v_preferLHS_1840_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 6, v_partialApp_1841_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 7, v_sameFun_1842_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*1 + 8, v_typeEqs_1844_);
v___x_1851_ = v_reuseFailAlloc_1856_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
uint8_t v___x_1852_; lean_object* v___x_1854_; 
v___x_1852_ = lean_unbox(v_a_1831_);
lean_dec(v_a_1831_);
lean_ctor_set_uint8(v___x_1851_, sizeof(void*)*1 + 9, v___x_1852_);
lean_ctor_set_uint8(v___x_1851_, sizeof(void*)*1 + 10, v_useCongrSimp_1845_);
lean_ctor_set_uint8(v___x_1851_, sizeof(void*)*1 + 11, v_beqEq_1846_);
if (v_isShared_1834_ == 0)
{
lean_ctor_set(v___x_1833_, 0, v___x_1851_);
v___x_1854_ = v___x_1833_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v___x_1851_);
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
lean_object* v_a_1859_; lean_object* v___x_1861_; uint8_t v_isShared_1862_; uint8_t v_isSharedCheck_1866_; 
lean_dec_ref(v_config_1306_);
v_a_1859_ = lean_ctor_get(v___x_1830_, 0);
v_isSharedCheck_1866_ = !lean_is_exclusive(v___x_1830_);
if (v_isSharedCheck_1866_ == 0)
{
v___x_1861_ = v___x_1830_;
v_isShared_1862_ = v_isSharedCheck_1866_;
goto v_resetjp_1860_;
}
else
{
lean_inc(v_a_1859_);
lean_dec(v___x_1830_);
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
}
else
{
lean_object* v_a_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1874_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1867_ = lean_ctor_get(v___x_1828_, 0);
v_isSharedCheck_1874_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1874_ == 0)
{
v___x_1869_ = v___x_1828_;
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_a_1867_);
lean_dec(v___x_1828_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
lean_object* v___x_1872_; 
if (v_isShared_1870_ == 0)
{
v___x_1872_ = v___x_1869_;
goto v_reusejp_1871_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v_a_1867_);
v___x_1872_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1871_;
}
v_reusejp_1871_:
{
return v___x_1872_;
}
}
}
}
}
else
{
uint8_t v___x_1875_; 
lean_dec_ref(v___x_1328_);
lean_dec_ref(v_config_1306_);
v___x_1875_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1875_ == 0)
{
lean_dec_ref(v_item_1307_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v_value_1876_; lean_object* v___x_1877_; 
lean_dec_ref(v___x_1329_);
v_value_1876_ = lean_ctor_get(v_item_1307_, 2);
lean_inc(v_value_1876_);
lean_dec_ref(v_item_1307_);
v___x_1877_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2(v_value_1876_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
return v___x_1877_;
}
}
}
else
{
lean_object* v___x_1878_; uint8_t v___x_1879_; 
v___x_1878_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22));
v___x_1879_ = lean_string_dec_eq(v___x_1328_, v___x_1878_);
if (v___x_1879_ == 0)
{
lean_object* v___x_1880_; uint8_t v___x_1881_; 
v___x_1880_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23));
v___x_1881_ = lean_string_dec_eq(v___x_1328_, v___x_1880_);
if (v___x_1881_ == 0)
{
lean_object* v___x_1882_; uint8_t v___x_1883_; 
v___x_1882_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24));
v___x_1883_ = lean_string_dec_eq(v___x_1328_, v___x_1882_);
lean_dec_ref(v___x_1328_);
if (v___x_1883_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1884_; lean_object* v___x_1885_; 
v___x_1884_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25));
v___x_1885_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1884_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1885_) == 0)
{
uint8_t v___x_1886_; 
lean_dec_ref_known(v___x_1885_, 1);
v___x_1886_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1886_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1887_; 
lean_dec_ref(v___x_1329_);
v___x_1887_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1887_) == 0)
{
lean_object* v_a_1888_; lean_object* v___x_1890_; uint8_t v_isShared_1891_; uint8_t v_isSharedCheck_1915_; 
v_a_1888_ = lean_ctor_get(v___x_1887_, 0);
v_isSharedCheck_1915_ = !lean_is_exclusive(v___x_1887_);
if (v_isSharedCheck_1915_ == 0)
{
v___x_1890_ = v___x_1887_;
v_isShared_1891_ = v_isSharedCheck_1915_;
goto v_resetjp_1889_;
}
else
{
lean_inc(v_a_1888_);
lean_dec(v___x_1887_);
v___x_1890_ = lean_box(0);
v_isShared_1891_ = v_isSharedCheck_1915_;
goto v_resetjp_1889_;
}
v_resetjp_1889_:
{
uint8_t v_closePost_1892_; uint8_t v_transparency_1893_; uint8_t v_preTransparency_1894_; uint8_t v_postTransparency_1895_; uint8_t v_preferLHS_1896_; uint8_t v_partialApp_1897_; uint8_t v_sameFun_1898_; lean_object* v_maxArgs_1899_; uint8_t v_typeEqs_1900_; uint8_t v_etaExpand_1901_; uint8_t v_useCongrSimp_1902_; uint8_t v_beqEq_1903_; lean_object* v___x_1905_; uint8_t v_isShared_1906_; uint8_t v_isSharedCheck_1914_; 
v_closePost_1892_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1893_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1894_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1895_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1896_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1897_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1898_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1899_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1900_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1901_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1902_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1903_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1914_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1914_ == 0)
{
v___x_1905_ = v_config_1306_;
v_isShared_1906_ = v_isSharedCheck_1914_;
goto v_resetjp_1904_;
}
else
{
lean_inc(v_maxArgs_1899_);
lean_dec(v_config_1306_);
v___x_1905_ = lean_box(0);
v_isShared_1906_ = v_isSharedCheck_1914_;
goto v_resetjp_1904_;
}
v_resetjp_1904_:
{
lean_object* v___x_1908_; 
if (v_isShared_1906_ == 0)
{
v___x_1908_ = v___x_1905_;
goto v_reusejp_1907_;
}
else
{
lean_object* v_reuseFailAlloc_1913_; 
v_reuseFailAlloc_1913_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1913_, 0, v_maxArgs_1899_);
v___x_1908_ = v_reuseFailAlloc_1913_;
goto v_reusejp_1907_;
}
v_reusejp_1907_:
{
uint8_t v___x_1909_; lean_object* v___x_1911_; 
v___x_1909_ = lean_unbox(v_a_1888_);
lean_dec(v_a_1888_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1, v___x_1909_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 1, v_closePost_1892_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 2, v_transparency_1893_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 3, v_preTransparency_1894_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 4, v_postTransparency_1895_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 5, v_preferLHS_1896_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 6, v_partialApp_1897_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 7, v_sameFun_1898_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 8, v_typeEqs_1900_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 9, v_etaExpand_1901_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 10, v_useCongrSimp_1902_);
lean_ctor_set_uint8(v___x_1908_, sizeof(void*)*1 + 11, v_beqEq_1903_);
if (v_isShared_1891_ == 0)
{
lean_ctor_set(v___x_1890_, 0, v___x_1908_);
v___x_1911_ = v___x_1890_;
goto v_reusejp_1910_;
}
else
{
lean_object* v_reuseFailAlloc_1912_; 
v_reuseFailAlloc_1912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1912_, 0, v___x_1908_);
v___x_1911_ = v_reuseFailAlloc_1912_;
goto v_reusejp_1910_;
}
v_reusejp_1910_:
{
return v___x_1911_;
}
}
}
}
}
else
{
lean_object* v_a_1916_; lean_object* v___x_1918_; uint8_t v_isShared_1919_; uint8_t v_isSharedCheck_1923_; 
lean_dec_ref(v_config_1306_);
v_a_1916_ = lean_ctor_get(v___x_1887_, 0);
v_isSharedCheck_1923_ = !lean_is_exclusive(v___x_1887_);
if (v_isSharedCheck_1923_ == 0)
{
v___x_1918_ = v___x_1887_;
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
else
{
lean_inc(v_a_1916_);
lean_dec(v___x_1887_);
v___x_1918_ = lean_box(0);
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
v_resetjp_1917_:
{
lean_object* v___x_1921_; 
if (v_isShared_1919_ == 0)
{
v___x_1921_ = v___x_1918_;
goto v_reusejp_1920_;
}
else
{
lean_object* v_reuseFailAlloc_1922_; 
v_reuseFailAlloc_1922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1922_, 0, v_a_1916_);
v___x_1921_ = v_reuseFailAlloc_1922_;
goto v_reusejp_1920_;
}
v_reusejp_1920_:
{
return v___x_1921_;
}
}
}
}
}
else
{
lean_object* v_a_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1931_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1924_ = lean_ctor_get(v___x_1885_, 0);
v_isSharedCheck_1931_ = !lean_is_exclusive(v___x_1885_);
if (v_isSharedCheck_1931_ == 0)
{
v___x_1926_ = v___x_1885_;
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_a_1924_);
lean_dec(v___x_1885_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1929_; 
if (v_isShared_1927_ == 0)
{
v___x_1929_ = v___x_1926_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v_a_1924_);
v___x_1929_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
return v___x_1929_;
}
}
}
}
}
else
{
lean_object* v___x_1932_; lean_object* v___x_1933_; 
lean_dec_ref(v___x_1328_);
v___x_1932_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26));
v___x_1933_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1932_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1933_) == 0)
{
uint8_t v___x_1934_; 
lean_dec_ref_known(v___x_1933_, 1);
v___x_1934_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1934_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1935_; 
lean_dec_ref(v___x_1329_);
v___x_1935_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1935_) == 0)
{
lean_object* v_a_1936_; lean_object* v___x_1938_; uint8_t v_isShared_1939_; uint8_t v_isSharedCheck_1963_; 
v_a_1936_ = lean_ctor_get(v___x_1935_, 0);
v_isSharedCheck_1963_ = !lean_is_exclusive(v___x_1935_);
if (v_isSharedCheck_1963_ == 0)
{
v___x_1938_ = v___x_1935_;
v_isShared_1939_ = v_isSharedCheck_1963_;
goto v_resetjp_1937_;
}
else
{
lean_inc(v_a_1936_);
lean_dec(v___x_1935_);
v___x_1938_ = lean_box(0);
v_isShared_1939_ = v_isSharedCheck_1963_;
goto v_resetjp_1937_;
}
v_resetjp_1937_:
{
uint8_t v_closePre_1940_; uint8_t v_transparency_1941_; uint8_t v_preTransparency_1942_; uint8_t v_postTransparency_1943_; uint8_t v_preferLHS_1944_; uint8_t v_partialApp_1945_; uint8_t v_sameFun_1946_; lean_object* v_maxArgs_1947_; uint8_t v_typeEqs_1948_; uint8_t v_etaExpand_1949_; uint8_t v_useCongrSimp_1950_; uint8_t v_beqEq_1951_; lean_object* v___x_1953_; uint8_t v_isShared_1954_; uint8_t v_isSharedCheck_1962_; 
v_closePre_1940_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_transparency_1941_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1942_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1943_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1944_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1945_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1946_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1947_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1948_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1949_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1950_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_beqEq_1951_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 11);
v_isSharedCheck_1962_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_1962_ == 0)
{
v___x_1953_ = v_config_1306_;
v_isShared_1954_ = v_isSharedCheck_1962_;
goto v_resetjp_1952_;
}
else
{
lean_inc(v_maxArgs_1947_);
lean_dec(v_config_1306_);
v___x_1953_ = lean_box(0);
v_isShared_1954_ = v_isSharedCheck_1962_;
goto v_resetjp_1952_;
}
v_resetjp_1952_:
{
lean_object* v___x_1956_; 
if (v_isShared_1954_ == 0)
{
v___x_1956_ = v___x_1953_;
goto v_reusejp_1955_;
}
else
{
lean_object* v_reuseFailAlloc_1961_; 
v_reuseFailAlloc_1961_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_1961_, 0, v_maxArgs_1947_);
lean_ctor_set_uint8(v_reuseFailAlloc_1961_, sizeof(void*)*1, v_closePre_1940_);
v___x_1956_ = v_reuseFailAlloc_1961_;
goto v_reusejp_1955_;
}
v_reusejp_1955_:
{
uint8_t v___x_1957_; lean_object* v___x_1959_; 
v___x_1957_ = lean_unbox(v_a_1936_);
lean_dec(v_a_1936_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 1, v___x_1957_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 2, v_transparency_1941_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 3, v_preTransparency_1942_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 4, v_postTransparency_1943_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 5, v_preferLHS_1944_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 6, v_partialApp_1945_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 7, v_sameFun_1946_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 8, v_typeEqs_1948_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 9, v_etaExpand_1949_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 10, v_useCongrSimp_1950_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1 + 11, v_beqEq_1951_);
if (v_isShared_1939_ == 0)
{
lean_ctor_set(v___x_1938_, 0, v___x_1956_);
v___x_1959_ = v___x_1938_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_1960_; 
v_reuseFailAlloc_1960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1960_, 0, v___x_1956_);
v___x_1959_ = v_reuseFailAlloc_1960_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
return v___x_1959_;
}
}
}
}
}
else
{
lean_object* v_a_1964_; lean_object* v___x_1966_; uint8_t v_isShared_1967_; uint8_t v_isSharedCheck_1971_; 
lean_dec_ref(v_config_1306_);
v_a_1964_ = lean_ctor_get(v___x_1935_, 0);
v_isSharedCheck_1971_ = !lean_is_exclusive(v___x_1935_);
if (v_isSharedCheck_1971_ == 0)
{
v___x_1966_ = v___x_1935_;
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
else
{
lean_inc(v_a_1964_);
lean_dec(v___x_1935_);
v___x_1966_ = lean_box(0);
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
v_resetjp_1965_:
{
lean_object* v___x_1969_; 
if (v_isShared_1967_ == 0)
{
v___x_1969_ = v___x_1966_;
goto v_reusejp_1968_;
}
else
{
lean_object* v_reuseFailAlloc_1970_; 
v_reuseFailAlloc_1970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1970_, 0, v_a_1964_);
v___x_1969_ = v_reuseFailAlloc_1970_;
goto v_reusejp_1968_;
}
v_reusejp_1968_:
{
return v___x_1969_;
}
}
}
}
}
else
{
lean_object* v_a_1972_; lean_object* v___x_1974_; uint8_t v_isShared_1975_; uint8_t v_isSharedCheck_1979_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_1972_ = lean_ctor_get(v___x_1933_, 0);
v_isSharedCheck_1979_ = !lean_is_exclusive(v___x_1933_);
if (v_isSharedCheck_1979_ == 0)
{
v___x_1974_ = v___x_1933_;
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
else
{
lean_inc(v_a_1972_);
lean_dec(v___x_1933_);
v___x_1974_ = lean_box(0);
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
v_resetjp_1973_:
{
lean_object* v___x_1977_; 
if (v_isShared_1975_ == 0)
{
v___x_1977_ = v___x_1974_;
goto v_reusejp_1976_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v_a_1972_);
v___x_1977_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1976_;
}
v_reusejp_1976_:
{
return v___x_1977_;
}
}
}
}
}
else
{
lean_object* v___x_1980_; lean_object* v___x_1981_; 
lean_dec_ref(v___x_1328_);
v___x_1980_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27));
v___x_1981_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1307_, v___x_1980_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1981_) == 0)
{
uint8_t v___x_1982_; 
lean_dec_ref_known(v___x_1981_, 1);
v___x_1982_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1329_);
if (v___x_1982_ == 0)
{
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_item_1316_ = v___x_1329_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
else
{
lean_object* v___x_1983_; 
lean_dec_ref(v___x_1329_);
v___x_1983_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
if (lean_obj_tag(v___x_1983_) == 0)
{
lean_object* v_a_1984_; lean_object* v___x_1986_; uint8_t v_isShared_1987_; uint8_t v_isSharedCheck_2011_; 
v_a_1984_ = lean_ctor_get(v___x_1983_, 0);
v_isSharedCheck_2011_ = !lean_is_exclusive(v___x_1983_);
if (v_isSharedCheck_2011_ == 0)
{
v___x_1986_ = v___x_1983_;
v_isShared_1987_ = v_isSharedCheck_2011_;
goto v_resetjp_1985_;
}
else
{
lean_inc(v_a_1984_);
lean_dec(v___x_1983_);
v___x_1986_ = lean_box(0);
v_isShared_1987_ = v_isSharedCheck_2011_;
goto v_resetjp_1985_;
}
v_resetjp_1985_:
{
uint8_t v_closePre_1988_; uint8_t v_closePost_1989_; uint8_t v_transparency_1990_; uint8_t v_preTransparency_1991_; uint8_t v_postTransparency_1992_; uint8_t v_preferLHS_1993_; uint8_t v_partialApp_1994_; uint8_t v_sameFun_1995_; lean_object* v_maxArgs_1996_; uint8_t v_typeEqs_1997_; uint8_t v_etaExpand_1998_; uint8_t v_useCongrSimp_1999_; lean_object* v___x_2001_; uint8_t v_isShared_2002_; uint8_t v_isSharedCheck_2010_; 
v_closePre_1988_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1);
v_closePost_1989_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 1);
v_transparency_1990_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 2);
v_preTransparency_1991_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 3);
v_postTransparency_1992_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 4);
v_preferLHS_1993_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 5);
v_partialApp_1994_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 6);
v_sameFun_1995_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 7);
v_maxArgs_1996_ = lean_ctor_get(v_config_1306_, 0);
v_typeEqs_1997_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 8);
v_etaExpand_1998_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 9);
v_useCongrSimp_1999_ = lean_ctor_get_uint8(v_config_1306_, sizeof(void*)*1 + 10);
v_isSharedCheck_2010_ = !lean_is_exclusive(v_config_1306_);
if (v_isSharedCheck_2010_ == 0)
{
v___x_2001_ = v_config_1306_;
v_isShared_2002_ = v_isSharedCheck_2010_;
goto v_resetjp_2000_;
}
else
{
lean_inc(v_maxArgs_1996_);
lean_dec(v_config_1306_);
v___x_2001_ = lean_box(0);
v_isShared_2002_ = v_isSharedCheck_2010_;
goto v_resetjp_2000_;
}
v_resetjp_2000_:
{
lean_object* v___x_2004_; 
if (v_isShared_2002_ == 0)
{
v___x_2004_ = v___x_2001_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2009_; 
v_reuseFailAlloc_2009_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2009_, 0, v_maxArgs_1996_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1, v_closePre_1988_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 1, v_closePost_1989_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 2, v_transparency_1990_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 3, v_preTransparency_1991_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 4, v_postTransparency_1992_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 5, v_preferLHS_1993_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 6, v_partialApp_1994_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 7, v_sameFun_1995_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 8, v_typeEqs_1997_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 9, v_etaExpand_1998_);
lean_ctor_set_uint8(v_reuseFailAlloc_2009_, sizeof(void*)*1 + 10, v_useCongrSimp_1999_);
v___x_2004_ = v_reuseFailAlloc_2009_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
uint8_t v___x_2005_; lean_object* v___x_2007_; 
v___x_2005_ = lean_unbox(v_a_1984_);
lean_dec(v_a_1984_);
lean_ctor_set_uint8(v___x_2004_, sizeof(void*)*1 + 11, v___x_2005_);
if (v_isShared_1987_ == 0)
{
lean_ctor_set(v___x_1986_, 0, v___x_2004_);
v___x_2007_ = v___x_1986_;
goto v_reusejp_2006_;
}
else
{
lean_object* v_reuseFailAlloc_2008_; 
v_reuseFailAlloc_2008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2008_, 0, v___x_2004_);
v___x_2007_ = v_reuseFailAlloc_2008_;
goto v_reusejp_2006_;
}
v_reusejp_2006_:
{
return v___x_2007_;
}
}
}
}
}
else
{
lean_object* v_a_2012_; lean_object* v___x_2014_; uint8_t v_isShared_2015_; uint8_t v_isSharedCheck_2019_; 
lean_dec_ref(v_config_1306_);
v_a_2012_ = lean_ctor_get(v___x_1983_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___x_1983_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_2014_ = v___x_1983_;
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
else
{
lean_inc(v_a_2012_);
lean_dec(v___x_1983_);
v___x_2014_ = lean_box(0);
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
v_resetjp_2013_:
{
lean_object* v___x_2017_; 
if (v_isShared_2015_ == 0)
{
v___x_2017_ = v___x_2014_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2018_; 
v_reuseFailAlloc_2018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2018_, 0, v_a_2012_);
v___x_2017_ = v_reuseFailAlloc_2018_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
return v___x_2017_;
}
}
}
}
}
else
{
lean_object* v_a_2020_; lean_object* v___x_2022_; uint8_t v_isShared_2023_; uint8_t v_isSharedCheck_2027_; 
lean_dec_ref(v___x_1329_);
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_2020_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_2027_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_2027_ == 0)
{
v___x_2022_ = v___x_1981_;
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
else
{
lean_inc(v_a_2020_);
lean_dec(v___x_1981_);
v___x_2022_ = lean_box(0);
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
v_resetjp_2021_:
{
lean_object* v___x_2025_; 
if (v_isShared_2023_ == 0)
{
v___x_2025_ = v___x_2022_;
goto v_reusejp_2024_;
}
else
{
lean_object* v_reuseFailAlloc_2026_; 
v_reuseFailAlloc_2026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2026_, 0, v_a_2020_);
v___x_2025_ = v_reuseFailAlloc_2026_;
goto v_reusejp_2024_;
}
v_reusejp_2024_:
{
return v___x_2025_;
}
}
}
}
}
}
}
else
{
lean_dec_ref(v_config_1306_);
v_item_1316_ = v_item_1307_;
v___y_1317_ = v___y_1308_;
v___y_1318_ = v___y_1309_;
v___y_1319_ = v___y_1310_;
v___y_1320_ = v___y_1311_;
v___y_1321_ = v___y_1312_;
v___y_1322_ = v___y_1313_;
goto v___jp_1315_;
}
}
else
{
lean_object* v_a_2028_; lean_object* v___x_2030_; uint8_t v_isShared_2031_; uint8_t v_isSharedCheck_2035_; 
lean_dec_ref(v_item_1307_);
lean_dec_ref(v_config_1306_);
v_a_2028_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_2035_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_2035_ == 0)
{
v___x_2030_ = v___x_1326_;
v_isShared_2031_ = v_isSharedCheck_2035_;
goto v_resetjp_2029_;
}
else
{
lean_inc(v_a_2028_);
lean_dec(v___x_1326_);
v___x_2030_ = lean_box(0);
v_isShared_2031_ = v_isSharedCheck_2035_;
goto v_resetjp_2029_;
}
v_resetjp_2029_:
{
lean_object* v___x_2033_; 
if (v_isShared_2031_ == 0)
{
v___x_2033_ = v___x_2030_;
goto v_reusejp_2032_;
}
else
{
lean_object* v_reuseFailAlloc_2034_; 
v_reuseFailAlloc_2034_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2034_, 0, v_a_2028_);
v___x_2033_ = v_reuseFailAlloc_2034_;
goto v_reusejp_2032_;
}
v_reusejp_2032_:
{
return v___x_2033_;
}
}
}
v___jp_1315_:
{
lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1323_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__0));
v___x_1324_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_1316_, v___x_1323_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
return v___x_1324_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_2036_, lean_object* v_item_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_){
_start:
{
lean_object* v_res_2045_; 
v_res_2045_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0(v_config_2036_, v_item_2037_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_);
lean_dec(v___y_2043_);
lean_dec_ref(v___y_2042_);
lean_dec(v___y_2041_);
lean_dec_ref(v___y_2040_);
lean_dec(v___y_2039_);
lean_dec_ref(v___y_2038_);
return v_res_2045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4(lean_object* v_e_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_){
_start:
{
lean_object* v___x_2056_; 
v___x_2056_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_e_2048_, v___y_2052_);
return v___x_2056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object* v_e_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_){
_start:
{
lean_object* v_res_2065_; 
v_res_2065_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4(v_e_2057_, v___y_2058_, v___y_2059_, v___y_2060_, v___y_2061_, v___y_2062_, v___y_2063_);
lean_dec(v___y_2063_);
lean_dec_ref(v___y_2062_);
lean_dec(v___y_2061_);
lean_dec_ref(v___y_2060_);
lean_dec(v___y_2059_);
lean_dec_ref(v___y_2058_);
return v_res_2065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6(lean_object* v_00_u03b1_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_){
_start:
{
lean_object* v___x_2074_; 
v___x_2074_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
return v___x_2074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___boxed(lean_object* v_00_u03b1_2075_, lean_object* v___y_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_){
_start:
{
lean_object* v_res_2083_; 
v_res_2083_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6(v_00_u03b1_2075_, v___y_2076_, v___y_2077_, v___y_2078_, v___y_2079_, v___y_2080_, v___y_2081_);
lean_dec(v___y_2081_);
lean_dec_ref(v___y_2080_);
lean_dec(v___y_2079_);
lean_dec_ref(v___y_2078_);
lean_dec(v___y_2077_);
lean_dec_ref(v___y_2076_);
return v_res_2083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5(lean_object* v_00_u03b1_2084_, lean_object* v_msg_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_, lean_object* v___y_2091_){
_start:
{
lean_object* v___x_2093_; 
v___x_2093_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v_msg_2085_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_, v___y_2090_, v___y_2091_);
return v___x_2093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___boxed(lean_object* v_00_u03b1_2094_, lean_object* v_msg_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_){
_start:
{
lean_object* v_res_2103_; 
v_res_2103_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5(v_00_u03b1_2094_, v_msg_2095_, v___y_2096_, v___y_2097_, v___y_2098_, v___y_2099_, v___y_2100_, v___y_2101_);
lean_dec(v___y_2101_);
lean_dec_ref(v___y_2100_);
lean_dec(v___y_2099_);
lean_dec_ref(v___y_2098_);
lean_dec(v___y_2097_);
lean_dec_ref(v___y_2096_);
return v_res_2103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6(lean_object* v_msgData_2104_, lean_object* v_macroStack_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_){
_start:
{
lean_object* v___x_2113_; 
v___x_2113_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_msgData_2104_, v_macroStack_2105_, v___y_2110_);
return v___x_2113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6___boxed(lean_object* v_msgData_2114_, lean_object* v_macroStack_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_){
_start:
{
lean_object* v_res_2123_; 
v_res_2123_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5_spec__6(v_msgData_2114_, v_macroStack_2115_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_);
lean_dec(v___y_2121_);
lean_dec_ref(v___y_2120_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec_ref(v___y_2116_);
return v_res_2123_;
}
}
static lean_object* _init_lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; 
v___x_2124_ = lean_box(0);
v___x_2125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr___closed__3));
v___x_2126_ = l_Lean_mkConst(v___x_2125_, v___x_2124_);
return v___x_2126_;
}
}
static lean_object* _init_lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2127_; lean_object* v___x_2128_; 
v___x_2127_ = lean_obj_once(&lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0, &lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__0);
v___x_2128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2128_, 0, v___x_2127_);
return v___x_2128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0(lean_object* v_cfg_2129_, lean_object* v_cfgItem_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_){
_start:
{
lean_object* v___x_2138_; lean_object* v___x_2139_; 
v___x_2138_ = lean_obj_once(&lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1, &lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___closed__1);
v___x_2139_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_2129_, v_cfgItem_2130_, v___x_2138_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_);
return v___x_2139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___lam__0___boxed(lean_object* v_cfg_2140_, lean_object* v_cfgItem_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_){
_start:
{
lean_object* v_res_2149_; 
v_res_2149_ = lp_mathlib_Convert_elabCheapConfig___redArg___lam__0(v_cfg_2140_, v_cfgItem_2141_, v___y_2142_, v___y_2143_, v___y_2144_, v___y_2145_, v___y_2146_, v___y_2147_);
lean_dec(v___y_2147_);
lean_dec_ref(v___y_2146_);
lean_dec(v___y_2145_);
lean_dec_ref(v___y_2144_);
lean_dec(v___y_2143_);
lean_dec_ref(v___y_2142_);
lean_dec(v_cfgItem_2141_);
return v_res_2149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg(lean_object* v_cfg_2151_, lean_object* v_init_2152_, uint8_t v_logExceptions_2153_, lean_object* v_a_2154_, lean_object* v_a_2155_, lean_object* v_a_2156_){
_start:
{
lean_object* v_onErr_2158_; lean_object* v_eval_2159_; 
v_onErr_2158_ = ((lean_object*)(lp_mathlib_Convert_elabCheapConfig___redArg___closed__0));
v_eval_2159_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___closed__0));
if (v_logExceptions_2153_ == 0)
{
lean_object* v___x_2160_; 
v___x_2160_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2159_, v_init_2152_, v_cfg_2151_, v_onErr_2158_, v_logExceptions_2153_, v_a_2155_, v_a_2156_);
return v___x_2160_;
}
else
{
uint8_t v_recover_2161_; lean_object* v___x_2162_; 
v_recover_2161_ = lean_ctor_get_uint8(v_a_2154_, sizeof(void*)*1);
v___x_2162_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2159_, v_init_2152_, v_cfg_2151_, v_onErr_2158_, v_recover_2161_, v_a_2155_, v_a_2156_);
return v___x_2162_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___redArg___boxed(lean_object* v_cfg_2163_, lean_object* v_init_2164_, lean_object* v_logExceptions_2165_, lean_object* v_a_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_, lean_object* v_a_2169_){
_start:
{
uint8_t v_logExceptions_boxed_2170_; lean_object* v_res_2171_; 
v_logExceptions_boxed_2170_ = lean_unbox(v_logExceptions_2165_);
v_res_2171_ = lp_mathlib_Convert_elabCheapConfig___redArg(v_cfg_2163_, v_init_2164_, v_logExceptions_boxed_2170_, v_a_2166_, v_a_2167_, v_a_2168_);
lean_dec(v_a_2168_);
lean_dec_ref(v_a_2167_);
lean_dec_ref(v_a_2166_);
return v_res_2171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig(lean_object* v_cfg_2172_, lean_object* v_init_2173_, uint8_t v_logExceptions_2174_, lean_object* v_a_2175_, lean_object* v_a_2176_, lean_object* v_a_2177_, lean_object* v_a_2178_, lean_object* v_a_2179_, lean_object* v_a_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_){
_start:
{
lean_object* v___x_2184_; 
v___x_2184_ = lp_mathlib_Convert_elabCheapConfig___redArg(v_cfg_2172_, v_init_2173_, v_logExceptions_2174_, v_a_2175_, v_a_2181_, v_a_2182_);
return v___x_2184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabCheapConfig___boxed(lean_object* v_cfg_2185_, lean_object* v_init_2186_, lean_object* v_logExceptions_2187_, lean_object* v_a_2188_, lean_object* v_a_2189_, lean_object* v_a_2190_, lean_object* v_a_2191_, lean_object* v_a_2192_, lean_object* v_a_2193_, lean_object* v_a_2194_, lean_object* v_a_2195_, lean_object* v_a_2196_){
_start:
{
uint8_t v_logExceptions_boxed_2197_; lean_object* v_res_2198_; 
v_logExceptions_boxed_2197_ = lean_unbox(v_logExceptions_2187_);
v_res_2198_ = lp_mathlib_Convert_elabCheapConfig(v_cfg_2185_, v_init_2186_, v_logExceptions_boxed_2197_, v_a_2188_, v_a_2189_, v_a_2190_, v_a_2191_, v_a_2192_, v_a_2193_, v_a_2194_, v_a_2195_);
lean_dec(v_a_2195_);
lean_dec_ref(v_a_2194_);
lean_dec(v_a_2193_);
lean_dec_ref(v_a_2192_);
lean_dec(v_a_2191_);
lean_dec_ref(v_a_2190_);
lean_dec(v_a_2189_);
lean_dec_ref(v_a_2188_);
return v_res_2198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0(lean_object* v_ctor_2199_, lean_object* v_args_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_){
_start:
{
lean_object* v___x_2227_; uint8_t v___x_2228_; 
v___x_2227_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_2228_ = lean_string_dec_eq(v_ctor_2199_, v___x_2227_);
if (v___x_2228_ == 0)
{
lean_object* v___x_2229_; 
v___x_2229_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_2229_;
}
else
{
lean_object* v___x_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
v___x_2230_ = lean_array_get_size(v_args_2200_);
v___x_2231_ = lean_unsigned_to_nat(1u);
v___x_2232_ = lean_nat_dec_eq(v___x_2230_, v___x_2231_);
if (v___x_2232_ == 0)
{
lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v_a_2235_; lean_object* v___x_2237_; uint8_t v_isShared_2238_; uint8_t v_isSharedCheck_2242_; 
v___x_2233_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr___lam__0___closed__3);
v___x_2234_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1___redArg(v___x_2233_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_);
v_a_2235_ = lean_ctor_get(v___x_2234_, 0);
v_isSharedCheck_2242_ = !lean_is_exclusive(v___x_2234_);
if (v_isSharedCheck_2242_ == 0)
{
v___x_2237_ = v___x_2234_;
v_isShared_2238_ = v_isSharedCheck_2242_;
goto v_resetjp_2236_;
}
else
{
lean_inc(v_a_2235_);
lean_dec(v___x_2234_);
v___x_2237_ = lean_box(0);
v_isShared_2238_ = v_isSharedCheck_2242_;
goto v_resetjp_2236_;
}
v_resetjp_2236_:
{
lean_object* v___x_2240_; 
if (v_isShared_2238_ == 0)
{
v___x_2240_ = v___x_2237_;
goto v_reusejp_2239_;
}
else
{
lean_object* v_reuseFailAlloc_2241_; 
v_reuseFailAlloc_2241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2241_, 0, v_a_2235_);
v___x_2240_ = v_reuseFailAlloc_2241_;
goto v_reusejp_2239_;
}
v_reusejp_2239_:
{
return v___x_2240_;
}
}
}
else
{
goto v___jp_2206_;
}
}
v___jp_2206_:
{
lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; 
v___x_2207_ = l_Lean_instInhabitedExpr;
v___x_2208_ = lean_unsigned_to_nat(0u);
v___x_2209_ = lean_array_get_borrowed(v___x_2207_, v_args_2200_, v___x_2208_);
lean_inc(v___x_2209_);
v___x_2210_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig_evalExpr(v___x_2209_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_);
if (lean_obj_tag(v___x_2210_) == 0)
{
lean_object* v_a_2211_; lean_object* v___x_2213_; uint8_t v_isShared_2214_; uint8_t v_isSharedCheck_2218_; 
v_a_2211_ = lean_ctor_get(v___x_2210_, 0);
v_isSharedCheck_2218_ = !lean_is_exclusive(v___x_2210_);
if (v_isSharedCheck_2218_ == 0)
{
v___x_2213_ = v___x_2210_;
v_isShared_2214_ = v_isSharedCheck_2218_;
goto v_resetjp_2212_;
}
else
{
lean_inc(v_a_2211_);
lean_dec(v___x_2210_);
v___x_2213_ = lean_box(0);
v_isShared_2214_ = v_isSharedCheck_2218_;
goto v_resetjp_2212_;
}
v_resetjp_2212_:
{
lean_object* v___x_2216_; 
if (v_isShared_2214_ == 0)
{
v___x_2216_ = v___x_2213_;
goto v_reusejp_2215_;
}
else
{
lean_object* v_reuseFailAlloc_2217_; 
v_reuseFailAlloc_2217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2217_, 0, v_a_2211_);
v___x_2216_ = v_reuseFailAlloc_2217_;
goto v_reusejp_2215_;
}
v_reusejp_2215_:
{
return v___x_2216_;
}
}
}
else
{
lean_object* v_a_2219_; lean_object* v___x_2221_; uint8_t v_isShared_2222_; uint8_t v_isSharedCheck_2226_; 
v_a_2219_ = lean_ctor_get(v___x_2210_, 0);
v_isSharedCheck_2226_ = !lean_is_exclusive(v___x_2210_);
if (v_isSharedCheck_2226_ == 0)
{
v___x_2221_ = v___x_2210_;
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
else
{
lean_inc(v_a_2219_);
lean_dec(v___x_2210_);
v___x_2221_ = lean_box(0);
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
v_resetjp_2220_:
{
lean_object* v___x_2224_; 
if (v_isShared_2222_ == 0)
{
v___x_2224_ = v___x_2221_;
goto v_reusejp_2223_;
}
else
{
lean_object* v_reuseFailAlloc_2225_; 
v_reuseFailAlloc_2225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2225_, 0, v_a_2219_);
v___x_2224_ = v_reuseFailAlloc_2225_;
goto v_reusejp_2223_;
}
v_reusejp_2223_:
{
return v___x_2224_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_2243_, lean_object* v_args_2244_, lean_object* v___y_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_, lean_object* v___y_2249_){
_start:
{
lean_object* v_res_2250_; 
v_res_2250_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___lam__0(v_ctor_2243_, v_args_2244_, v___y_2245_, v___y_2246_, v___y_2247_, v___y_2248_);
lean_dec(v___y_2248_);
lean_dec_ref(v___y_2247_);
lean_dec(v___y_2246_);
lean_dec_ref(v___y_2245_);
lean_dec_ref(v_args_2244_);
lean_dec_ref(v_ctor_2243_);
return v_res_2250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr(lean_object* v_a_2256_, lean_object* v_a_2257_, lean_object* v_a_2258_, lean_object* v_a_2259_, lean_object* v_a_2260_){
_start:
{
lean_object* v___f_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; 
v___f_2262_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__0));
v___x_2263_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2));
v___x_2264_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_2263_, v___f_2262_, v_a_2256_, v_a_2257_, v_a_2258_, v_a_2259_, v_a_2260_);
return v___x_2264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___boxed(lean_object* v_a_2265_, lean_object* v_a_2266_, lean_object* v_a_2267_, lean_object* v_a_2268_, lean_object* v_a_2269_, lean_object* v_a_2270_){
_start:
{
lean_object* v_res_2271_; 
v_res_2271_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr(v_a_2265_, v_a_2266_, v_a_2267_, v_a_2268_, v_a_2269_);
lean_dec(v_a_2269_);
lean_dec_ref(v_a_2268_);
lean_dec(v_a_2267_);
lean_dec_ref(v_a_2266_);
return v_res_2271_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1(void){
_start:
{
lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; 
v___x_2273_ = lean_box(0);
v___x_2274_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2));
v___x_2275_ = l_Lean_Expr_const___override(v___x_2274_, v___x_2273_);
return v___x_2275_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2(void){
_start:
{
lean_object* v___x_2276_; lean_object* v___x_2277_; 
v___x_2276_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1);
v___x_2277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2277_, 0, v___x_2276_);
return v___x_2277_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3(void){
_start:
{
lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; 
v___x_2278_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2);
v___x_2279_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__0));
v___x_2280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2280_, 0, v___x_2279_);
lean_ctor_set(v___x_2280_, 1, v___x_2278_);
return v___x_2280_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig(void){
_start:
{
lean_object* v___x_2281_; 
v___x_2281_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__3);
return v___x_2281_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2282_; lean_object* v___x_2283_; 
v___x_2282_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__1);
v___x_2283_ = l_Lean_MessageData_ofExpr(v___x_2282_);
return v___x_2283_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; 
v___x_2284_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__0);
v___x_2285_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__3);
v___x_2286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2286_, 0, v___x_2285_);
lean_ctor_set(v___x_2286_, 1, v___x_2284_);
return v___x_2286_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2(void){
_start:
{
lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; 
v___x_2287_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__5);
v___x_2288_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__1);
v___x_2289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2288_);
lean_ctor_set(v___x_2289_, 1, v___x_2287_);
return v___x_2289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0(lean_object* v_stx_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_, lean_object* v_a_2293_, lean_object* v_a_2294_, lean_object* v_a_2295_, lean_object* v_a_2296_){
_start:
{
lean_object* v_ty_x3f_2298_; uint8_t v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v_fileName_2304_; lean_object* v_fileMap_2305_; lean_object* v_options_2306_; lean_object* v_currRecDepth_2307_; lean_object* v_maxRecDepth_2308_; lean_object* v_ref_2309_; lean_object* v_currNamespace_2310_; lean_object* v_openDecls_2311_; lean_object* v_initHeartbeats_2312_; lean_object* v_maxHeartbeats_2313_; lean_object* v_quotContext_2314_; lean_object* v_currMacroScope_2315_; uint8_t v_diag_2316_; lean_object* v_cancelTk_x3f_2317_; uint8_t v_suppressElabErrors_2318_; lean_object* v_inheritedTraceOptions_2319_; uint8_t v___x_2320_; lean_object* v_ref_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; 
v_ty_x3f_2298_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig___closed__2);
v___x_2299_ = 1;
v___x_2300_ = lean_box(0);
v___x_2301_ = lean_box(v___x_2299_);
v___x_2302_ = lean_box(v___x_2299_);
lean_inc(v_stx_2290_);
v___x_2303_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_2303_, 0, v_stx_2290_);
lean_closure_set(v___x_2303_, 1, v_ty_x3f_2298_);
lean_closure_set(v___x_2303_, 2, v___x_2301_);
lean_closure_set(v___x_2303_, 3, v___x_2302_);
lean_closure_set(v___x_2303_, 4, v___x_2300_);
v_fileName_2304_ = lean_ctor_get(v_a_2295_, 0);
v_fileMap_2305_ = lean_ctor_get(v_a_2295_, 1);
v_options_2306_ = lean_ctor_get(v_a_2295_, 2);
v_currRecDepth_2307_ = lean_ctor_get(v_a_2295_, 3);
v_maxRecDepth_2308_ = lean_ctor_get(v_a_2295_, 4);
v_ref_2309_ = lean_ctor_get(v_a_2295_, 5);
v_currNamespace_2310_ = lean_ctor_get(v_a_2295_, 6);
v_openDecls_2311_ = lean_ctor_get(v_a_2295_, 7);
v_initHeartbeats_2312_ = lean_ctor_get(v_a_2295_, 8);
v_maxHeartbeats_2313_ = lean_ctor_get(v_a_2295_, 9);
v_quotContext_2314_ = lean_ctor_get(v_a_2295_, 10);
v_currMacroScope_2315_ = lean_ctor_get(v_a_2295_, 11);
v_diag_2316_ = lean_ctor_get_uint8(v_a_2295_, sizeof(void*)*14);
v_cancelTk_x3f_2317_ = lean_ctor_get(v_a_2295_, 12);
v_suppressElabErrors_2318_ = lean_ctor_get_uint8(v_a_2295_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2319_ = lean_ctor_get(v_a_2295_, 13);
v___x_2320_ = 1;
v_ref_2321_ = l_Lean_replaceRef(v_stx_2290_, v_ref_2309_);
lean_dec(v_stx_2290_);
lean_inc_ref(v_inheritedTraceOptions_2319_);
lean_inc(v_cancelTk_x3f_2317_);
lean_inc(v_currMacroScope_2315_);
lean_inc(v_quotContext_2314_);
lean_inc(v_maxHeartbeats_2313_);
lean_inc(v_initHeartbeats_2312_);
lean_inc(v_openDecls_2311_);
lean_inc(v_currNamespace_2310_);
lean_inc(v_maxRecDepth_2308_);
lean_inc(v_currRecDepth_2307_);
lean_inc_ref(v_options_2306_);
lean_inc_ref(v_fileMap_2305_);
lean_inc_ref(v_fileName_2304_);
v___x_2322_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2322_, 0, v_fileName_2304_);
lean_ctor_set(v___x_2322_, 1, v_fileMap_2305_);
lean_ctor_set(v___x_2322_, 2, v_options_2306_);
lean_ctor_set(v___x_2322_, 3, v_currRecDepth_2307_);
lean_ctor_set(v___x_2322_, 4, v_maxRecDepth_2308_);
lean_ctor_set(v___x_2322_, 5, v_ref_2321_);
lean_ctor_set(v___x_2322_, 6, v_currNamespace_2310_);
lean_ctor_set(v___x_2322_, 7, v_openDecls_2311_);
lean_ctor_set(v___x_2322_, 8, v_initHeartbeats_2312_);
lean_ctor_set(v___x_2322_, 9, v_maxHeartbeats_2313_);
lean_ctor_set(v___x_2322_, 10, v_quotContext_2314_);
lean_ctor_set(v___x_2322_, 11, v_currMacroScope_2315_);
lean_ctor_set(v___x_2322_, 12, v_cancelTk_x3f_2317_);
lean_ctor_set(v___x_2322_, 13, v_inheritedTraceOptions_2319_);
lean_ctor_set_uint8(v___x_2322_, sizeof(void*)*14, v_diag_2316_);
lean_ctor_set_uint8(v___x_2322_, sizeof(void*)*14 + 1, v_suppressElabErrors_2318_);
v___x_2323_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_2303_, v___x_2320_, v_a_2291_, v_a_2292_, v_a_2293_, v_a_2294_, v___x_2322_, v_a_2296_);
if (lean_obj_tag(v___x_2323_) == 0)
{
lean_object* v_a_2324_; lean_object* v___x_2325_; lean_object* v_a_2326_; lean_object* v___y_2328_; lean_object* v___y_2329_; lean_object* v___y_2330_; lean_object* v___y_2331_; lean_object* v___y_2332_; lean_object* v___y_2333_; lean_object* v___y_2334_; lean_object* v___y_2335_; lean_object* v___y_2336_; uint8_t v___y_2337_; lean_object* v___y_2354_; lean_object* v___y_2355_; lean_object* v___y_2356_; lean_object* v___y_2357_; lean_object* v___y_2358_; lean_object* v___y_2359_; lean_object* v___y_2366_; lean_object* v___y_2367_; lean_object* v___y_2368_; lean_object* v___y_2369_; lean_object* v___y_2370_; lean_object* v___y_2371_; lean_object* v___y_2403_; lean_object* v___y_2404_; lean_object* v___y_2405_; lean_object* v___y_2406_; lean_object* v___y_2407_; lean_object* v___y_2408_; uint8_t v___x_2421_; 
v_a_2324_ = lean_ctor_get(v___x_2323_, 0);
lean_inc(v_a_2324_);
lean_dec_ref_known(v___x_2323_, 1);
v___x_2325_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_2324_, v_a_2294_);
v_a_2326_ = lean_ctor_get(v___x_2325_, 0);
lean_inc(v_a_2326_);
lean_dec_ref(v___x_2325_);
v___x_2421_ = l_Lean_Expr_hasSorry(v_a_2326_);
if (v___x_2421_ == 0)
{
v___y_2366_ = v_a_2291_;
v___y_2367_ = v_a_2292_;
v___y_2368_ = v_a_2293_;
v___y_2369_ = v_a_2294_;
v___y_2370_ = v___x_2322_;
v___y_2371_ = v_a_2296_;
goto v___jp_2365_;
}
else
{
uint8_t v___x_2422_; 
v___x_2422_ = l_Lean_Expr_hasSyntheticSorry(v_a_2326_);
if (v___x_2422_ == 0)
{
v___y_2403_ = v_a_2291_;
v___y_2404_ = v_a_2292_;
v___y_2405_ = v_a_2293_;
v___y_2406_ = v_a_2294_;
v___y_2407_ = v___x_2322_;
v___y_2408_ = v_a_2296_;
goto v___jp_2402_;
}
else
{
lean_object* v___x_2423_; lean_object* v_a_2424_; lean_object* v___x_2426_; uint8_t v_isShared_2427_; uint8_t v_isSharedCheck_2431_; 
lean_dec(v_a_2326_);
lean_dec_ref_known(v___x_2322_, 14);
v___x_2423_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_2424_ = lean_ctor_get(v___x_2423_, 0);
v_isSharedCheck_2431_ = !lean_is_exclusive(v___x_2423_);
if (v_isSharedCheck_2431_ == 0)
{
v___x_2426_ = v___x_2423_;
v_isShared_2427_ = v_isSharedCheck_2431_;
goto v_resetjp_2425_;
}
else
{
lean_inc(v_a_2424_);
lean_dec(v___x_2423_);
v___x_2426_ = lean_box(0);
v_isShared_2427_ = v_isSharedCheck_2431_;
goto v_resetjp_2425_;
}
v_resetjp_2425_:
{
lean_object* v___x_2429_; 
if (v_isShared_2427_ == 0)
{
v___x_2429_ = v___x_2426_;
goto v_reusejp_2428_;
}
else
{
lean_object* v_reuseFailAlloc_2430_; 
v_reuseFailAlloc_2430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2430_, 0, v_a_2424_);
v___x_2429_ = v_reuseFailAlloc_2430_;
goto v_reusejp_2428_;
}
v_reusejp_2428_:
{
return v___x_2429_;
}
}
}
}
v___jp_2327_:
{
if (v___y_2337_ == 0)
{
if (lean_obj_tag(v___y_2332_) == 0)
{
lean_dec_ref_known(v___y_2332_, 2);
lean_dec_ref(v___y_2333_);
lean_dec(v_a_2326_);
return v___y_2331_;
}
else
{
lean_object* v_id_2338_; lean_object* v___x_2340_; uint8_t v_isShared_2341_; uint8_t v_isSharedCheck_2351_; 
v_id_2338_ = lean_ctor_get(v___y_2332_, 0);
v_isSharedCheck_2351_ = !lean_is_exclusive(v___y_2332_);
if (v_isSharedCheck_2351_ == 0)
{
lean_object* v_unused_2352_; 
v_unused_2352_ = lean_ctor_get(v___y_2332_, 1);
lean_dec(v_unused_2352_);
v___x_2340_ = v___y_2332_;
v_isShared_2341_ = v_isSharedCheck_2351_;
goto v_resetjp_2339_;
}
else
{
lean_inc(v_id_2338_);
lean_dec(v___y_2332_);
v___x_2340_ = lean_box(0);
v_isShared_2341_ = v_isSharedCheck_2351_;
goto v_resetjp_2339_;
}
v_resetjp_2339_:
{
uint8_t v___x_2342_; 
v___x_2342_ = l_Lean_instBEqInternalExceptionId_beq(v___y_2335_, v_id_2338_);
lean_dec(v_id_2338_);
if (v___x_2342_ == 0)
{
lean_del_object(v___x_2340_);
lean_dec_ref(v___y_2333_);
lean_dec(v_a_2326_);
return v___y_2331_;
}
else
{
lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2347_; 
lean_dec_ref(v___y_2331_);
v___x_2343_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___closed__2);
v___x_2344_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__1);
v___x_2345_ = l_Lean_indentExpr(v_a_2326_);
if (v_isShared_2341_ == 0)
{
lean_ctor_set_tag(v___x_2340_, 7);
lean_ctor_set(v___x_2340_, 1, v___x_2345_);
lean_ctor_set(v___x_2340_, 0, v___x_2344_);
v___x_2347_ = v___x_2340_;
goto v_reusejp_2346_;
}
else
{
lean_object* v_reuseFailAlloc_2350_; 
v_reuseFailAlloc_2350_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2350_, 0, v___x_2344_);
lean_ctor_set(v_reuseFailAlloc_2350_, 1, v___x_2345_);
v___x_2347_ = v_reuseFailAlloc_2350_;
goto v_reusejp_2346_;
}
v_reusejp_2346_:
{
lean_object* v___x_2348_; lean_object* v___x_2349_; 
v___x_2348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2348_, 0, v___x_2347_);
lean_ctor_set(v___x_2348_, 1, v___x_2343_);
v___x_2349_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_2348_, v___y_2334_, v___y_2330_, v___y_2328_, v___y_2329_, v___y_2333_, v___y_2336_);
lean_dec_ref(v___y_2333_);
return v___x_2349_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v_a_2326_);
return v___y_2331_;
}
}
v___jp_2353_:
{
lean_object* v___x_2360_; 
lean_inc(v_a_2326_);
v___x_2360_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr(v_a_2326_, v___y_2356_, v___y_2357_, v___y_2358_, v___y_2359_);
if (lean_obj_tag(v___x_2360_) == 0)
{
lean_dec_ref(v___y_2358_);
lean_dec(v_a_2326_);
return v___x_2360_;
}
else
{
lean_object* v_a_2361_; lean_object* v___x_2362_; uint8_t v___x_2363_; 
v_a_2361_ = lean_ctor_get(v___x_2360_, 0);
lean_inc(v_a_2361_);
v___x_2362_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2363_ = l_Lean_Exception_isInterrupt(v_a_2361_);
if (v___x_2363_ == 0)
{
uint8_t v___x_2364_; 
lean_inc(v_a_2361_);
v___x_2364_ = l_Lean_Exception_isRuntime(v_a_2361_);
v___y_2328_ = v___y_2356_;
v___y_2329_ = v___y_2357_;
v___y_2330_ = v___y_2355_;
v___y_2331_ = v___x_2360_;
v___y_2332_ = v_a_2361_;
v___y_2333_ = v___y_2358_;
v___y_2334_ = v___y_2354_;
v___y_2335_ = v___x_2362_;
v___y_2336_ = v___y_2359_;
v___y_2337_ = v___x_2364_;
goto v___jp_2327_;
}
else
{
v___y_2328_ = v___y_2356_;
v___y_2329_ = v___y_2357_;
v___y_2330_ = v___y_2355_;
v___y_2331_ = v___x_2360_;
v___y_2332_ = v_a_2361_;
v___y_2333_ = v___y_2358_;
v___y_2334_ = v___y_2354_;
v___y_2335_ = v___x_2362_;
v___y_2336_ = v___y_2359_;
v___y_2337_ = v___x_2363_;
goto v___jp_2327_;
}
}
}
v___jp_2365_:
{
lean_object* v___x_2372_; 
lean_inc(v_a_2326_);
v___x_2372_ = l_Lean_Meta_getMVars(v_a_2326_, v___y_2368_, v___y_2369_, v___y_2370_, v___y_2371_);
if (lean_obj_tag(v___x_2372_) == 0)
{
lean_object* v_a_2373_; lean_object* v___x_2374_; 
v_a_2373_ = lean_ctor_get(v___x_2372_, 0);
lean_inc(v_a_2373_);
lean_dec_ref_known(v___x_2372_, 1);
v___x_2374_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_2373_, v___x_2300_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_, v___y_2370_, v___y_2371_);
lean_dec(v_a_2373_);
if (lean_obj_tag(v___x_2374_) == 0)
{
lean_object* v_a_2375_; uint8_t v___x_2376_; 
v_a_2375_ = lean_ctor_get(v___x_2374_, 0);
lean_inc(v_a_2375_);
lean_dec_ref_known(v___x_2374_, 1);
v___x_2376_ = lean_unbox(v_a_2375_);
lean_dec(v_a_2375_);
if (v___x_2376_ == 0)
{
v___y_2354_ = v___y_2366_;
v___y_2355_ = v___y_2367_;
v___y_2356_ = v___y_2368_;
v___y_2357_ = v___y_2369_;
v___y_2358_ = v___y_2370_;
v___y_2359_ = v___y_2371_;
goto v___jp_2353_;
}
else
{
lean_object* v___x_2377_; lean_object* v_a_2378_; lean_object* v___x_2380_; uint8_t v_isShared_2381_; uint8_t v_isSharedCheck_2385_; 
lean_dec_ref(v___y_2370_);
lean_dec(v_a_2326_);
v___x_2377_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_2378_ = lean_ctor_get(v___x_2377_, 0);
v_isSharedCheck_2385_ = !lean_is_exclusive(v___x_2377_);
if (v_isSharedCheck_2385_ == 0)
{
v___x_2380_ = v___x_2377_;
v_isShared_2381_ = v_isSharedCheck_2385_;
goto v_resetjp_2379_;
}
else
{
lean_inc(v_a_2378_);
lean_dec(v___x_2377_);
v___x_2380_ = lean_box(0);
v_isShared_2381_ = v_isSharedCheck_2385_;
goto v_resetjp_2379_;
}
v_resetjp_2379_:
{
lean_object* v___x_2383_; 
if (v_isShared_2381_ == 0)
{
v___x_2383_ = v___x_2380_;
goto v_reusejp_2382_;
}
else
{
lean_object* v_reuseFailAlloc_2384_; 
v_reuseFailAlloc_2384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2384_, 0, v_a_2378_);
v___x_2383_ = v_reuseFailAlloc_2384_;
goto v_reusejp_2382_;
}
v_reusejp_2382_:
{
return v___x_2383_;
}
}
}
}
else
{
lean_object* v_a_2386_; lean_object* v___x_2388_; uint8_t v_isShared_2389_; uint8_t v_isSharedCheck_2393_; 
lean_dec_ref(v___y_2370_);
lean_dec(v_a_2326_);
v_a_2386_ = lean_ctor_get(v___x_2374_, 0);
v_isSharedCheck_2393_ = !lean_is_exclusive(v___x_2374_);
if (v_isSharedCheck_2393_ == 0)
{
v___x_2388_ = v___x_2374_;
v_isShared_2389_ = v_isSharedCheck_2393_;
goto v_resetjp_2387_;
}
else
{
lean_inc(v_a_2386_);
lean_dec(v___x_2374_);
v___x_2388_ = lean_box(0);
v_isShared_2389_ = v_isSharedCheck_2393_;
goto v_resetjp_2387_;
}
v_resetjp_2387_:
{
lean_object* v___x_2391_; 
if (v_isShared_2389_ == 0)
{
v___x_2391_ = v___x_2388_;
goto v_reusejp_2390_;
}
else
{
lean_object* v_reuseFailAlloc_2392_; 
v_reuseFailAlloc_2392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2392_, 0, v_a_2386_);
v___x_2391_ = v_reuseFailAlloc_2392_;
goto v_reusejp_2390_;
}
v_reusejp_2390_:
{
return v___x_2391_;
}
}
}
}
else
{
lean_object* v_a_2394_; lean_object* v___x_2396_; uint8_t v_isShared_2397_; uint8_t v_isSharedCheck_2401_; 
lean_dec_ref(v___y_2370_);
lean_dec(v_a_2326_);
v_a_2394_ = lean_ctor_get(v___x_2372_, 0);
v_isSharedCheck_2401_ = !lean_is_exclusive(v___x_2372_);
if (v_isSharedCheck_2401_ == 0)
{
v___x_2396_ = v___x_2372_;
v_isShared_2397_ = v_isSharedCheck_2401_;
goto v_resetjp_2395_;
}
else
{
lean_inc(v_a_2394_);
lean_dec(v___x_2372_);
v___x_2396_ = lean_box(0);
v_isShared_2397_ = v_isSharedCheck_2401_;
goto v_resetjp_2395_;
}
v_resetjp_2395_:
{
lean_object* v___x_2399_; 
if (v_isShared_2397_ == 0)
{
v___x_2399_ = v___x_2396_;
goto v_reusejp_2398_;
}
else
{
lean_object* v_reuseFailAlloc_2400_; 
v_reuseFailAlloc_2400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2400_, 0, v_a_2394_);
v___x_2399_ = v_reuseFailAlloc_2400_;
goto v_reusejp_2398_;
}
v_reusejp_2398_:
{
return v___x_2399_;
}
}
}
}
v___jp_2402_:
{
lean_object* v___x_2409_; lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v_a_2413_; lean_object* v___x_2415_; uint8_t v_isShared_2416_; uint8_t v_isSharedCheck_2420_; 
v___x_2409_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_2410_ = l_Lean_indentExpr(v_a_2326_);
v___x_2411_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2411_, 0, v___x_2409_);
lean_ctor_set(v___x_2411_, 1, v___x_2410_);
v___x_2412_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_2411_, v___y_2403_, v___y_2404_, v___y_2405_, v___y_2406_, v___y_2407_, v___y_2408_);
lean_dec_ref(v___y_2407_);
v_a_2413_ = lean_ctor_get(v___x_2412_, 0);
v_isSharedCheck_2420_ = !lean_is_exclusive(v___x_2412_);
if (v_isSharedCheck_2420_ == 0)
{
v___x_2415_ = v___x_2412_;
v_isShared_2416_ = v_isSharedCheck_2420_;
goto v_resetjp_2414_;
}
else
{
lean_inc(v_a_2413_);
lean_dec(v___x_2412_);
v___x_2415_ = lean_box(0);
v_isShared_2416_ = v_isSharedCheck_2420_;
goto v_resetjp_2414_;
}
v_resetjp_2414_:
{
lean_object* v___x_2418_; 
if (v_isShared_2416_ == 0)
{
v___x_2418_ = v___x_2415_;
goto v_reusejp_2417_;
}
else
{
lean_object* v_reuseFailAlloc_2419_; 
v_reuseFailAlloc_2419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2419_, 0, v_a_2413_);
v___x_2418_ = v_reuseFailAlloc_2419_;
goto v_reusejp_2417_;
}
v_reusejp_2417_:
{
return v___x_2418_;
}
}
}
}
else
{
lean_object* v_a_2432_; lean_object* v___x_2434_; uint8_t v_isShared_2435_; uint8_t v_isSharedCheck_2439_; 
lean_dec_ref_known(v___x_2322_, 14);
v_a_2432_ = lean_ctor_get(v___x_2323_, 0);
v_isSharedCheck_2439_ = !lean_is_exclusive(v___x_2323_);
if (v_isSharedCheck_2439_ == 0)
{
v___x_2434_ = v___x_2323_;
v_isShared_2435_ = v_isSharedCheck_2439_;
goto v_resetjp_2433_;
}
else
{
lean_inc(v_a_2432_);
lean_dec(v___x_2323_);
v___x_2434_ = lean_box(0);
v_isShared_2435_ = v_isSharedCheck_2439_;
goto v_resetjp_2433_;
}
v_resetjp_2433_:
{
lean_object* v___x_2437_; 
if (v_isShared_2435_ == 0)
{
v___x_2437_ = v___x_2434_;
goto v_reusejp_2436_;
}
else
{
lean_object* v_reuseFailAlloc_2438_; 
v_reuseFailAlloc_2438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2438_, 0, v_a_2432_);
v___x_2437_ = v_reuseFailAlloc_2438_;
goto v_reusejp_2436_;
}
v_reusejp_2436_:
{
return v___x_2437_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_2440_, lean_object* v_a_2441_, lean_object* v_a_2442_, lean_object* v_a_2443_, lean_object* v_a_2444_, lean_object* v_a_2445_, lean_object* v_a_2446_, lean_object* v_a_2447_){
_start:
{
lean_object* v_res_2448_; 
v_res_2448_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0(v_stx_2440_, v_a_2441_, v_a_2442_, v_a_2443_, v_a_2444_, v_a_2445_, v_a_2446_);
lean_dec(v_a_2446_);
lean_dec_ref(v_a_2445_);
lean_dec(v_a_2444_);
lean_dec_ref(v_a_2443_);
lean_dec(v_a_2442_);
lean_dec_ref(v_a_2441_);
return v_res_2448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0(lean_object* v_config_2451_, lean_object* v_item_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_){
_start:
{
lean_object* v_item_2461_; lean_object* v___y_2462_; lean_object* v___y_2463_; lean_object* v___y_2464_; lean_object* v___y_2465_; lean_object* v___y_2466_; lean_object* v___y_2467_; lean_object* v___x_2470_; lean_object* v___x_2471_; 
v___x_2470_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2));
v___x_2471_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_2452_, v___x_2470_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2471_) == 0)
{
uint8_t v___x_2472_; 
lean_dec_ref_known(v___x_2471_, 1);
v___x_2472_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_2452_);
if (v___x_2472_ == 0)
{
lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; uint8_t v___x_2476_; 
v___x_2473_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_2452_);
lean_inc_ref(v_item_2452_);
v___x_2474_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_2452_);
v___x_2475_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__1));
v___x_2476_ = lean_string_dec_lt(v___x_2473_, v___x_2475_);
if (v___x_2476_ == 0)
{
lean_object* v___x_2477_; uint8_t v___x_2478_; 
v___x_2477_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__2));
v___x_2478_ = lean_string_dec_lt(v___x_2473_, v___x_2477_);
if (v___x_2478_ == 0)
{
uint8_t v___x_2479_; 
v___x_2479_ = lean_string_dec_eq(v___x_2473_, v___x_2477_);
if (v___x_2479_ == 0)
{
lean_object* v___x_2480_; uint8_t v___x_2481_; 
v___x_2480_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__3));
v___x_2481_ = lean_string_dec_eq(v___x_2473_, v___x_2480_);
if (v___x_2481_ == 0)
{
lean_object* v___x_2482_; uint8_t v___x_2483_; 
v___x_2482_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__4));
v___x_2483_ = lean_string_dec_eq(v___x_2473_, v___x_2482_);
if (v___x_2483_ == 0)
{
lean_object* v___x_2484_; uint8_t v___x_2485_; 
v___x_2484_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__5));
v___x_2485_ = lean_string_dec_eq(v___x_2473_, v___x_2484_);
lean_dec_ref(v___x_2473_);
if (v___x_2485_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2486_; lean_object* v___x_2487_; 
v___x_2486_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__6));
v___x_2487_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2486_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2487_) == 0)
{
uint8_t v___x_2488_; 
lean_dec_ref_known(v___x_2487_, 1);
v___x_2488_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2488_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2489_; 
lean_dec_ref(v___x_2474_);
v___x_2489_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2489_) == 0)
{
lean_object* v_a_2490_; lean_object* v___x_2492_; uint8_t v_isShared_2493_; uint8_t v_isSharedCheck_2517_; 
v_a_2490_ = lean_ctor_get(v___x_2489_, 0);
v_isSharedCheck_2517_ = !lean_is_exclusive(v___x_2489_);
if (v_isSharedCheck_2517_ == 0)
{
v___x_2492_ = v___x_2489_;
v_isShared_2493_ = v_isSharedCheck_2517_;
goto v_resetjp_2491_;
}
else
{
lean_inc(v_a_2490_);
lean_dec(v___x_2489_);
v___x_2492_ = lean_box(0);
v_isShared_2493_ = v_isSharedCheck_2517_;
goto v_resetjp_2491_;
}
v_resetjp_2491_:
{
uint8_t v_closePre_2494_; uint8_t v_closePost_2495_; uint8_t v_transparency_2496_; uint8_t v_preTransparency_2497_; uint8_t v_postTransparency_2498_; uint8_t v_preferLHS_2499_; uint8_t v_partialApp_2500_; uint8_t v_sameFun_2501_; lean_object* v_maxArgs_2502_; uint8_t v_typeEqs_2503_; uint8_t v_etaExpand_2504_; uint8_t v_beqEq_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2516_; 
v_closePre_2494_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2495_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2496_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2497_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2498_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2499_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2500_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2501_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2502_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2503_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2504_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_beqEq_2505_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2516_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2516_ == 0)
{
v___x_2507_ = v_config_2451_;
v_isShared_2508_ = v_isSharedCheck_2516_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_maxArgs_2502_);
lean_dec(v_config_2451_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2516_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v___x_2510_; 
if (v_isShared_2508_ == 0)
{
v___x_2510_ = v___x_2507_;
goto v_reusejp_2509_;
}
else
{
lean_object* v_reuseFailAlloc_2515_; 
v_reuseFailAlloc_2515_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2515_, 0, v_maxArgs_2502_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1, v_closePre_2494_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 1, v_closePost_2495_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 2, v_transparency_2496_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 3, v_preTransparency_2497_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 4, v_postTransparency_2498_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 5, v_preferLHS_2499_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 6, v_partialApp_2500_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 7, v_sameFun_2501_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 8, v_typeEqs_2503_);
lean_ctor_set_uint8(v_reuseFailAlloc_2515_, sizeof(void*)*1 + 9, v_etaExpand_2504_);
v___x_2510_ = v_reuseFailAlloc_2515_;
goto v_reusejp_2509_;
}
v_reusejp_2509_:
{
uint8_t v___x_2511_; lean_object* v___x_2513_; 
v___x_2511_ = lean_unbox(v_a_2490_);
lean_dec(v_a_2490_);
lean_ctor_set_uint8(v___x_2510_, sizeof(void*)*1 + 10, v___x_2511_);
lean_ctor_set_uint8(v___x_2510_, sizeof(void*)*1 + 11, v_beqEq_2505_);
if (v_isShared_2493_ == 0)
{
lean_ctor_set(v___x_2492_, 0, v___x_2510_);
v___x_2513_ = v___x_2492_;
goto v_reusejp_2512_;
}
else
{
lean_object* v_reuseFailAlloc_2514_; 
v_reuseFailAlloc_2514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2514_, 0, v___x_2510_);
v___x_2513_ = v_reuseFailAlloc_2514_;
goto v_reusejp_2512_;
}
v_reusejp_2512_:
{
return v___x_2513_;
}
}
}
}
}
else
{
lean_object* v_a_2518_; lean_object* v___x_2520_; uint8_t v_isShared_2521_; uint8_t v_isSharedCheck_2525_; 
lean_dec_ref(v_config_2451_);
v_a_2518_ = lean_ctor_get(v___x_2489_, 0);
v_isSharedCheck_2525_ = !lean_is_exclusive(v___x_2489_);
if (v_isSharedCheck_2525_ == 0)
{
v___x_2520_ = v___x_2489_;
v_isShared_2521_ = v_isSharedCheck_2525_;
goto v_resetjp_2519_;
}
else
{
lean_inc(v_a_2518_);
lean_dec(v___x_2489_);
v___x_2520_ = lean_box(0);
v_isShared_2521_ = v_isSharedCheck_2525_;
goto v_resetjp_2519_;
}
v_resetjp_2519_:
{
lean_object* v___x_2523_; 
if (v_isShared_2521_ == 0)
{
v___x_2523_ = v___x_2520_;
goto v_reusejp_2522_;
}
else
{
lean_object* v_reuseFailAlloc_2524_; 
v_reuseFailAlloc_2524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2524_, 0, v_a_2518_);
v___x_2523_ = v_reuseFailAlloc_2524_;
goto v_reusejp_2522_;
}
v_reusejp_2522_:
{
return v___x_2523_;
}
}
}
}
}
else
{
lean_object* v_a_2526_; lean_object* v___x_2528_; uint8_t v_isShared_2529_; uint8_t v_isSharedCheck_2533_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2526_ = lean_ctor_get(v___x_2487_, 0);
v_isSharedCheck_2533_ = !lean_is_exclusive(v___x_2487_);
if (v_isSharedCheck_2533_ == 0)
{
v___x_2528_ = v___x_2487_;
v_isShared_2529_ = v_isSharedCheck_2533_;
goto v_resetjp_2527_;
}
else
{
lean_inc(v_a_2526_);
lean_dec(v___x_2487_);
v___x_2528_ = lean_box(0);
v_isShared_2529_ = v_isSharedCheck_2533_;
goto v_resetjp_2527_;
}
v_resetjp_2527_:
{
lean_object* v___x_2531_; 
if (v_isShared_2529_ == 0)
{
v___x_2531_ = v___x_2528_;
goto v_reusejp_2530_;
}
else
{
lean_object* v_reuseFailAlloc_2532_; 
v_reuseFailAlloc_2532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2532_, 0, v_a_2526_);
v___x_2531_ = v_reuseFailAlloc_2532_;
goto v_reusejp_2530_;
}
v_reusejp_2530_:
{
return v___x_2531_;
}
}
}
}
}
else
{
lean_object* v___x_2534_; lean_object* v___x_2535_; 
lean_dec_ref(v___x_2473_);
v___x_2534_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__7));
v___x_2535_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2534_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2535_) == 0)
{
uint8_t v___x_2536_; 
lean_dec_ref_known(v___x_2535_, 1);
v___x_2536_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2536_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2537_; 
lean_dec_ref(v___x_2474_);
v___x_2537_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2537_) == 0)
{
lean_object* v_a_2538_; lean_object* v___x_2540_; uint8_t v_isShared_2541_; uint8_t v_isSharedCheck_2565_; 
v_a_2538_ = lean_ctor_get(v___x_2537_, 0);
v_isSharedCheck_2565_ = !lean_is_exclusive(v___x_2537_);
if (v_isSharedCheck_2565_ == 0)
{
v___x_2540_ = v___x_2537_;
v_isShared_2541_ = v_isSharedCheck_2565_;
goto v_resetjp_2539_;
}
else
{
lean_inc(v_a_2538_);
lean_dec(v___x_2537_);
v___x_2540_ = lean_box(0);
v_isShared_2541_ = v_isSharedCheck_2565_;
goto v_resetjp_2539_;
}
v_resetjp_2539_:
{
uint8_t v_closePre_2542_; uint8_t v_closePost_2543_; uint8_t v_transparency_2544_; uint8_t v_preTransparency_2545_; uint8_t v_postTransparency_2546_; uint8_t v_preferLHS_2547_; uint8_t v_partialApp_2548_; uint8_t v_sameFun_2549_; lean_object* v_maxArgs_2550_; uint8_t v_etaExpand_2551_; uint8_t v_useCongrSimp_2552_; uint8_t v_beqEq_2553_; lean_object* v___x_2555_; uint8_t v_isShared_2556_; uint8_t v_isSharedCheck_2564_; 
v_closePre_2542_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2543_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2544_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2545_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2546_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2547_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2548_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2549_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2550_ = lean_ctor_get(v_config_2451_, 0);
v_etaExpand_2551_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2552_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2553_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2564_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2564_ == 0)
{
v___x_2555_ = v_config_2451_;
v_isShared_2556_ = v_isSharedCheck_2564_;
goto v_resetjp_2554_;
}
else
{
lean_inc(v_maxArgs_2550_);
lean_dec(v_config_2451_);
v___x_2555_ = lean_box(0);
v_isShared_2556_ = v_isSharedCheck_2564_;
goto v_resetjp_2554_;
}
v_resetjp_2554_:
{
lean_object* v___x_2558_; 
if (v_isShared_2556_ == 0)
{
v___x_2558_ = v___x_2555_;
goto v_reusejp_2557_;
}
else
{
lean_object* v_reuseFailAlloc_2563_; 
v_reuseFailAlloc_2563_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2563_, 0, v_maxArgs_2550_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1, v_closePre_2542_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 1, v_closePost_2543_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 2, v_transparency_2544_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 3, v_preTransparency_2545_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 4, v_postTransparency_2546_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 5, v_preferLHS_2547_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 6, v_partialApp_2548_);
lean_ctor_set_uint8(v_reuseFailAlloc_2563_, sizeof(void*)*1 + 7, v_sameFun_2549_);
v___x_2558_ = v_reuseFailAlloc_2563_;
goto v_reusejp_2557_;
}
v_reusejp_2557_:
{
uint8_t v___x_2559_; lean_object* v___x_2561_; 
v___x_2559_ = lean_unbox(v_a_2538_);
lean_dec(v_a_2538_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*1 + 8, v___x_2559_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*1 + 9, v_etaExpand_2551_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*1 + 10, v_useCongrSimp_2552_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*1 + 11, v_beqEq_2553_);
if (v_isShared_2541_ == 0)
{
lean_ctor_set(v___x_2540_, 0, v___x_2558_);
v___x_2561_ = v___x_2540_;
goto v_reusejp_2560_;
}
else
{
lean_object* v_reuseFailAlloc_2562_; 
v_reuseFailAlloc_2562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2562_, 0, v___x_2558_);
v___x_2561_ = v_reuseFailAlloc_2562_;
goto v_reusejp_2560_;
}
v_reusejp_2560_:
{
return v___x_2561_;
}
}
}
}
}
else
{
lean_object* v_a_2566_; lean_object* v___x_2568_; uint8_t v_isShared_2569_; uint8_t v_isSharedCheck_2573_; 
lean_dec_ref(v_config_2451_);
v_a_2566_ = lean_ctor_get(v___x_2537_, 0);
v_isSharedCheck_2573_ = !lean_is_exclusive(v___x_2537_);
if (v_isSharedCheck_2573_ == 0)
{
v___x_2568_ = v___x_2537_;
v_isShared_2569_ = v_isSharedCheck_2573_;
goto v_resetjp_2567_;
}
else
{
lean_inc(v_a_2566_);
lean_dec(v___x_2537_);
v___x_2568_ = lean_box(0);
v_isShared_2569_ = v_isSharedCheck_2573_;
goto v_resetjp_2567_;
}
v_resetjp_2567_:
{
lean_object* v___x_2571_; 
if (v_isShared_2569_ == 0)
{
v___x_2571_ = v___x_2568_;
goto v_reusejp_2570_;
}
else
{
lean_object* v_reuseFailAlloc_2572_; 
v_reuseFailAlloc_2572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2572_, 0, v_a_2566_);
v___x_2571_ = v_reuseFailAlloc_2572_;
goto v_reusejp_2570_;
}
v_reusejp_2570_:
{
return v___x_2571_;
}
}
}
}
}
else
{
lean_object* v_a_2574_; lean_object* v___x_2576_; uint8_t v_isShared_2577_; uint8_t v_isSharedCheck_2581_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2574_ = lean_ctor_get(v___x_2535_, 0);
v_isSharedCheck_2581_ = !lean_is_exclusive(v___x_2535_);
if (v_isSharedCheck_2581_ == 0)
{
v___x_2576_ = v___x_2535_;
v_isShared_2577_ = v_isSharedCheck_2581_;
goto v_resetjp_2575_;
}
else
{
lean_inc(v_a_2574_);
lean_dec(v___x_2535_);
v___x_2576_ = lean_box(0);
v_isShared_2577_ = v_isSharedCheck_2581_;
goto v_resetjp_2575_;
}
v_resetjp_2575_:
{
lean_object* v___x_2579_; 
if (v_isShared_2577_ == 0)
{
v___x_2579_ = v___x_2576_;
goto v_reusejp_2578_;
}
else
{
lean_object* v_reuseFailAlloc_2580_; 
v_reuseFailAlloc_2580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2580_, 0, v_a_2574_);
v___x_2579_ = v_reuseFailAlloc_2580_;
goto v_reusejp_2578_;
}
v_reusejp_2578_:
{
return v___x_2579_;
}
}
}
}
}
else
{
lean_object* v___x_2582_; lean_object* v___x_2583_; 
lean_dec_ref(v___x_2473_);
v___x_2582_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__8));
v___x_2583_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2582_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2583_) == 0)
{
uint8_t v___x_2584_; 
lean_dec_ref_known(v___x_2583_, 1);
v___x_2584_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2584_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2585_; 
lean_dec_ref(v___x_2474_);
lean_inc_ref(v_item_2452_);
v___x_2585_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2585_) == 0)
{
lean_object* v_value_2586_; lean_object* v___x_2587_; 
lean_dec_ref_known(v___x_2585_, 1);
v_value_2586_ = lean_ctor_get(v_item_2452_, 2);
lean_inc(v_value_2586_);
lean_dec_ref(v_item_2452_);
v___x_2587_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_2586_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2587_) == 0)
{
lean_object* v_a_2588_; lean_object* v___x_2590_; uint8_t v_isShared_2591_; uint8_t v_isSharedCheck_2615_; 
v_a_2588_ = lean_ctor_get(v___x_2587_, 0);
v_isSharedCheck_2615_ = !lean_is_exclusive(v___x_2587_);
if (v_isSharedCheck_2615_ == 0)
{
v___x_2590_ = v___x_2587_;
v_isShared_2591_ = v_isSharedCheck_2615_;
goto v_resetjp_2589_;
}
else
{
lean_inc(v_a_2588_);
lean_dec(v___x_2587_);
v___x_2590_ = lean_box(0);
v_isShared_2591_ = v_isSharedCheck_2615_;
goto v_resetjp_2589_;
}
v_resetjp_2589_:
{
uint8_t v_closePre_2592_; uint8_t v_closePost_2593_; uint8_t v_preTransparency_2594_; uint8_t v_postTransparency_2595_; uint8_t v_preferLHS_2596_; uint8_t v_partialApp_2597_; uint8_t v_sameFun_2598_; lean_object* v_maxArgs_2599_; uint8_t v_typeEqs_2600_; uint8_t v_etaExpand_2601_; uint8_t v_useCongrSimp_2602_; uint8_t v_beqEq_2603_; lean_object* v___x_2605_; uint8_t v_isShared_2606_; uint8_t v_isSharedCheck_2614_; 
v_closePre_2592_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2593_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_preTransparency_2594_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2595_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2596_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2597_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2598_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2599_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2600_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2601_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2602_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2603_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2614_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2614_ == 0)
{
v___x_2605_ = v_config_2451_;
v_isShared_2606_ = v_isSharedCheck_2614_;
goto v_resetjp_2604_;
}
else
{
lean_inc(v_maxArgs_2599_);
lean_dec(v_config_2451_);
v___x_2605_ = lean_box(0);
v_isShared_2606_ = v_isSharedCheck_2614_;
goto v_resetjp_2604_;
}
v_resetjp_2604_:
{
lean_object* v___x_2608_; 
if (v_isShared_2606_ == 0)
{
v___x_2608_ = v___x_2605_;
goto v_reusejp_2607_;
}
else
{
lean_object* v_reuseFailAlloc_2613_; 
v_reuseFailAlloc_2613_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2613_, 0, v_maxArgs_2599_);
lean_ctor_set_uint8(v_reuseFailAlloc_2613_, sizeof(void*)*1, v_closePre_2592_);
lean_ctor_set_uint8(v_reuseFailAlloc_2613_, sizeof(void*)*1 + 1, v_closePost_2593_);
v___x_2608_ = v_reuseFailAlloc_2613_;
goto v_reusejp_2607_;
}
v_reusejp_2607_:
{
uint8_t v___x_2609_; lean_object* v___x_2611_; 
v___x_2609_ = lean_unbox(v_a_2588_);
lean_dec(v_a_2588_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 2, v___x_2609_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 3, v_preTransparency_2594_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 4, v_postTransparency_2595_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 5, v_preferLHS_2596_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 6, v_partialApp_2597_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 7, v_sameFun_2598_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 8, v_typeEqs_2600_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 9, v_etaExpand_2601_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 10, v_useCongrSimp_2602_);
lean_ctor_set_uint8(v___x_2608_, sizeof(void*)*1 + 11, v_beqEq_2603_);
if (v_isShared_2591_ == 0)
{
lean_ctor_set(v___x_2590_, 0, v___x_2608_);
v___x_2611_ = v___x_2590_;
goto v_reusejp_2610_;
}
else
{
lean_object* v_reuseFailAlloc_2612_; 
v_reuseFailAlloc_2612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2612_, 0, v___x_2608_);
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
else
{
lean_object* v_a_2616_; lean_object* v___x_2618_; uint8_t v_isShared_2619_; uint8_t v_isSharedCheck_2623_; 
lean_dec_ref(v_config_2451_);
v_a_2616_ = lean_ctor_get(v___x_2587_, 0);
v_isSharedCheck_2623_ = !lean_is_exclusive(v___x_2587_);
if (v_isSharedCheck_2623_ == 0)
{
v___x_2618_ = v___x_2587_;
v_isShared_2619_ = v_isSharedCheck_2623_;
goto v_resetjp_2617_;
}
else
{
lean_inc(v_a_2616_);
lean_dec(v___x_2587_);
v___x_2618_ = lean_box(0);
v_isShared_2619_ = v_isSharedCheck_2623_;
goto v_resetjp_2617_;
}
v_resetjp_2617_:
{
lean_object* v___x_2621_; 
if (v_isShared_2619_ == 0)
{
v___x_2621_ = v___x_2618_;
goto v_reusejp_2620_;
}
else
{
lean_object* v_reuseFailAlloc_2622_; 
v_reuseFailAlloc_2622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2622_, 0, v_a_2616_);
v___x_2621_ = v_reuseFailAlloc_2622_;
goto v_reusejp_2620_;
}
v_reusejp_2620_:
{
return v___x_2621_;
}
}
}
}
else
{
lean_object* v_a_2624_; lean_object* v___x_2626_; uint8_t v_isShared_2627_; uint8_t v_isSharedCheck_2631_; 
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2624_ = lean_ctor_get(v___x_2585_, 0);
v_isSharedCheck_2631_ = !lean_is_exclusive(v___x_2585_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2626_ = v___x_2585_;
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
else
{
lean_inc(v_a_2624_);
lean_dec(v___x_2585_);
v___x_2626_ = lean_box(0);
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
v_resetjp_2625_:
{
lean_object* v___x_2629_; 
if (v_isShared_2627_ == 0)
{
v___x_2629_ = v___x_2626_;
goto v_reusejp_2628_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v_a_2624_);
v___x_2629_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2628_;
}
v_reusejp_2628_:
{
return v___x_2629_;
}
}
}
}
}
else
{
lean_object* v_a_2632_; lean_object* v___x_2634_; uint8_t v_isShared_2635_; uint8_t v_isSharedCheck_2639_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2632_ = lean_ctor_get(v___x_2583_, 0);
v_isSharedCheck_2639_ = !lean_is_exclusive(v___x_2583_);
if (v_isSharedCheck_2639_ == 0)
{
v___x_2634_ = v___x_2583_;
v_isShared_2635_ = v_isSharedCheck_2639_;
goto v_resetjp_2633_;
}
else
{
lean_inc(v_a_2632_);
lean_dec(v___x_2583_);
v___x_2634_ = lean_box(0);
v_isShared_2635_ = v_isSharedCheck_2639_;
goto v_resetjp_2633_;
}
v_resetjp_2633_:
{
lean_object* v___x_2637_; 
if (v_isShared_2635_ == 0)
{
v___x_2637_ = v___x_2634_;
goto v_reusejp_2636_;
}
else
{
lean_object* v_reuseFailAlloc_2638_; 
v_reuseFailAlloc_2638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2638_, 0, v_a_2632_);
v___x_2637_ = v_reuseFailAlloc_2638_;
goto v_reusejp_2636_;
}
v_reusejp_2636_:
{
return v___x_2637_;
}
}
}
}
}
else
{
lean_object* v___x_2640_; lean_object* v___x_2641_; 
lean_dec_ref(v___x_2473_);
v___x_2640_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__9));
v___x_2641_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2640_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2641_) == 0)
{
uint8_t v___x_2642_; 
lean_dec_ref_known(v___x_2641_, 1);
v___x_2642_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2642_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2643_; 
lean_dec_ref(v___x_2474_);
v___x_2643_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2643_) == 0)
{
lean_object* v_a_2644_; lean_object* v___x_2646_; uint8_t v_isShared_2647_; uint8_t v_isSharedCheck_2671_; 
v_a_2644_ = lean_ctor_get(v___x_2643_, 0);
v_isSharedCheck_2671_ = !lean_is_exclusive(v___x_2643_);
if (v_isSharedCheck_2671_ == 0)
{
v___x_2646_ = v___x_2643_;
v_isShared_2647_ = v_isSharedCheck_2671_;
goto v_resetjp_2645_;
}
else
{
lean_inc(v_a_2644_);
lean_dec(v___x_2643_);
v___x_2646_ = lean_box(0);
v_isShared_2647_ = v_isSharedCheck_2671_;
goto v_resetjp_2645_;
}
v_resetjp_2645_:
{
uint8_t v_closePre_2648_; uint8_t v_closePost_2649_; uint8_t v_transparency_2650_; uint8_t v_preTransparency_2651_; uint8_t v_postTransparency_2652_; uint8_t v_preferLHS_2653_; uint8_t v_partialApp_2654_; lean_object* v_maxArgs_2655_; uint8_t v_typeEqs_2656_; uint8_t v_etaExpand_2657_; uint8_t v_useCongrSimp_2658_; uint8_t v_beqEq_2659_; lean_object* v___x_2661_; uint8_t v_isShared_2662_; uint8_t v_isSharedCheck_2670_; 
v_closePre_2648_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2649_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2650_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2651_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2652_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2653_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2654_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_maxArgs_2655_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2656_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2657_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2658_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2659_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2670_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2670_ == 0)
{
v___x_2661_ = v_config_2451_;
v_isShared_2662_ = v_isSharedCheck_2670_;
goto v_resetjp_2660_;
}
else
{
lean_inc(v_maxArgs_2655_);
lean_dec(v_config_2451_);
v___x_2661_ = lean_box(0);
v_isShared_2662_ = v_isSharedCheck_2670_;
goto v_resetjp_2660_;
}
v_resetjp_2660_:
{
lean_object* v___x_2664_; 
if (v_isShared_2662_ == 0)
{
v___x_2664_ = v___x_2661_;
goto v_reusejp_2663_;
}
else
{
lean_object* v_reuseFailAlloc_2669_; 
v_reuseFailAlloc_2669_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2669_, 0, v_maxArgs_2655_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1, v_closePre_2648_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 1, v_closePost_2649_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 2, v_transparency_2650_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 3, v_preTransparency_2651_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 4, v_postTransparency_2652_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 5, v_preferLHS_2653_);
lean_ctor_set_uint8(v_reuseFailAlloc_2669_, sizeof(void*)*1 + 6, v_partialApp_2654_);
v___x_2664_ = v_reuseFailAlloc_2669_;
goto v_reusejp_2663_;
}
v_reusejp_2663_:
{
uint8_t v___x_2665_; lean_object* v___x_2667_; 
v___x_2665_ = lean_unbox(v_a_2644_);
lean_dec(v_a_2644_);
lean_ctor_set_uint8(v___x_2664_, sizeof(void*)*1 + 7, v___x_2665_);
lean_ctor_set_uint8(v___x_2664_, sizeof(void*)*1 + 8, v_typeEqs_2656_);
lean_ctor_set_uint8(v___x_2664_, sizeof(void*)*1 + 9, v_etaExpand_2657_);
lean_ctor_set_uint8(v___x_2664_, sizeof(void*)*1 + 10, v_useCongrSimp_2658_);
lean_ctor_set_uint8(v___x_2664_, sizeof(void*)*1 + 11, v_beqEq_2659_);
if (v_isShared_2647_ == 0)
{
lean_ctor_set(v___x_2646_, 0, v___x_2664_);
v___x_2667_ = v___x_2646_;
goto v_reusejp_2666_;
}
else
{
lean_object* v_reuseFailAlloc_2668_; 
v_reuseFailAlloc_2668_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2668_, 0, v___x_2664_);
v___x_2667_ = v_reuseFailAlloc_2668_;
goto v_reusejp_2666_;
}
v_reusejp_2666_:
{
return v___x_2667_;
}
}
}
}
}
else
{
lean_object* v_a_2672_; lean_object* v___x_2674_; uint8_t v_isShared_2675_; uint8_t v_isSharedCheck_2679_; 
lean_dec_ref(v_config_2451_);
v_a_2672_ = lean_ctor_get(v___x_2643_, 0);
v_isSharedCheck_2679_ = !lean_is_exclusive(v___x_2643_);
if (v_isSharedCheck_2679_ == 0)
{
v___x_2674_ = v___x_2643_;
v_isShared_2675_ = v_isSharedCheck_2679_;
goto v_resetjp_2673_;
}
else
{
lean_inc(v_a_2672_);
lean_dec(v___x_2643_);
v___x_2674_ = lean_box(0);
v_isShared_2675_ = v_isSharedCheck_2679_;
goto v_resetjp_2673_;
}
v_resetjp_2673_:
{
lean_object* v___x_2677_; 
if (v_isShared_2675_ == 0)
{
v___x_2677_ = v___x_2674_;
goto v_reusejp_2676_;
}
else
{
lean_object* v_reuseFailAlloc_2678_; 
v_reuseFailAlloc_2678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2678_, 0, v_a_2672_);
v___x_2677_ = v_reuseFailAlloc_2678_;
goto v_reusejp_2676_;
}
v_reusejp_2676_:
{
return v___x_2677_;
}
}
}
}
}
else
{
lean_object* v_a_2680_; lean_object* v___x_2682_; uint8_t v_isShared_2683_; uint8_t v_isSharedCheck_2687_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2680_ = lean_ctor_get(v___x_2641_, 0);
v_isSharedCheck_2687_ = !lean_is_exclusive(v___x_2641_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2682_ = v___x_2641_;
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
else
{
lean_inc(v_a_2680_);
lean_dec(v___x_2641_);
v___x_2682_ = lean_box(0);
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
v_resetjp_2681_:
{
lean_object* v___x_2685_; 
if (v_isShared_2683_ == 0)
{
v___x_2685_ = v___x_2682_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v_a_2680_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
}
}
else
{
uint8_t v___x_2688_; 
v___x_2688_ = lean_string_dec_eq(v___x_2473_, v___x_2475_);
if (v___x_2688_ == 0)
{
lean_object* v___x_2689_; uint8_t v___x_2690_; 
v___x_2689_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__10));
v___x_2690_ = lean_string_dec_eq(v___x_2473_, v___x_2689_);
if (v___x_2690_ == 0)
{
lean_object* v___x_2691_; uint8_t v___x_2692_; 
v___x_2691_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__11));
v___x_2692_ = lean_string_dec_eq(v___x_2473_, v___x_2691_);
lean_dec_ref(v___x_2473_);
if (v___x_2692_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2693_; lean_object* v___x_2694_; 
v___x_2693_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__12));
v___x_2694_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2693_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2694_) == 0)
{
uint8_t v___x_2695_; 
lean_dec_ref_known(v___x_2694_, 1);
v___x_2695_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2695_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2696_; 
lean_dec_ref(v___x_2474_);
v___x_2696_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2696_) == 0)
{
lean_object* v_a_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2724_; 
v_a_2697_ = lean_ctor_get(v___x_2696_, 0);
v_isSharedCheck_2724_ = !lean_is_exclusive(v___x_2696_);
if (v_isSharedCheck_2724_ == 0)
{
v___x_2699_ = v___x_2696_;
v_isShared_2700_ = v_isSharedCheck_2724_;
goto v_resetjp_2698_;
}
else
{
lean_inc(v_a_2697_);
lean_dec(v___x_2696_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2724_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
uint8_t v_closePre_2701_; uint8_t v_closePost_2702_; uint8_t v_transparency_2703_; uint8_t v_preTransparency_2704_; uint8_t v_postTransparency_2705_; uint8_t v_partialApp_2706_; uint8_t v_sameFun_2707_; lean_object* v_maxArgs_2708_; uint8_t v_typeEqs_2709_; uint8_t v_etaExpand_2710_; uint8_t v_useCongrSimp_2711_; uint8_t v_beqEq_2712_; lean_object* v___x_2714_; uint8_t v_isShared_2715_; uint8_t v_isSharedCheck_2723_; 
v_closePre_2701_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2702_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2703_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2704_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2705_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_partialApp_2706_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2707_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2708_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2709_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2710_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2711_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2712_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2723_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2723_ == 0)
{
v___x_2714_ = v_config_2451_;
v_isShared_2715_ = v_isSharedCheck_2723_;
goto v_resetjp_2713_;
}
else
{
lean_inc(v_maxArgs_2708_);
lean_dec(v_config_2451_);
v___x_2714_ = lean_box(0);
v_isShared_2715_ = v_isSharedCheck_2723_;
goto v_resetjp_2713_;
}
v_resetjp_2713_:
{
lean_object* v___x_2717_; 
if (v_isShared_2715_ == 0)
{
v___x_2717_ = v___x_2714_;
goto v_reusejp_2716_;
}
else
{
lean_object* v_reuseFailAlloc_2722_; 
v_reuseFailAlloc_2722_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2722_, 0, v_maxArgs_2708_);
lean_ctor_set_uint8(v_reuseFailAlloc_2722_, sizeof(void*)*1, v_closePre_2701_);
lean_ctor_set_uint8(v_reuseFailAlloc_2722_, sizeof(void*)*1 + 1, v_closePost_2702_);
lean_ctor_set_uint8(v_reuseFailAlloc_2722_, sizeof(void*)*1 + 2, v_transparency_2703_);
lean_ctor_set_uint8(v_reuseFailAlloc_2722_, sizeof(void*)*1 + 3, v_preTransparency_2704_);
lean_ctor_set_uint8(v_reuseFailAlloc_2722_, sizeof(void*)*1 + 4, v_postTransparency_2705_);
v___x_2717_ = v_reuseFailAlloc_2722_;
goto v_reusejp_2716_;
}
v_reusejp_2716_:
{
uint8_t v___x_2718_; lean_object* v___x_2720_; 
v___x_2718_ = lean_unbox(v_a_2697_);
lean_dec(v_a_2697_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 5, v___x_2718_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 6, v_partialApp_2706_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 7, v_sameFun_2707_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 8, v_typeEqs_2709_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 9, v_etaExpand_2710_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 10, v_useCongrSimp_2711_);
lean_ctor_set_uint8(v___x_2717_, sizeof(void*)*1 + 11, v_beqEq_2712_);
if (v_isShared_2700_ == 0)
{
lean_ctor_set(v___x_2699_, 0, v___x_2717_);
v___x_2720_ = v___x_2699_;
goto v_reusejp_2719_;
}
else
{
lean_object* v_reuseFailAlloc_2721_; 
v_reuseFailAlloc_2721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2721_, 0, v___x_2717_);
v___x_2720_ = v_reuseFailAlloc_2721_;
goto v_reusejp_2719_;
}
v_reusejp_2719_:
{
return v___x_2720_;
}
}
}
}
}
else
{
lean_object* v_a_2725_; lean_object* v___x_2727_; uint8_t v_isShared_2728_; uint8_t v_isSharedCheck_2732_; 
lean_dec_ref(v_config_2451_);
v_a_2725_ = lean_ctor_get(v___x_2696_, 0);
v_isSharedCheck_2732_ = !lean_is_exclusive(v___x_2696_);
if (v_isSharedCheck_2732_ == 0)
{
v___x_2727_ = v___x_2696_;
v_isShared_2728_ = v_isSharedCheck_2732_;
goto v_resetjp_2726_;
}
else
{
lean_inc(v_a_2725_);
lean_dec(v___x_2696_);
v___x_2727_ = lean_box(0);
v_isShared_2728_ = v_isSharedCheck_2732_;
goto v_resetjp_2726_;
}
v_resetjp_2726_:
{
lean_object* v___x_2730_; 
if (v_isShared_2728_ == 0)
{
v___x_2730_ = v___x_2727_;
goto v_reusejp_2729_;
}
else
{
lean_object* v_reuseFailAlloc_2731_; 
v_reuseFailAlloc_2731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2731_, 0, v_a_2725_);
v___x_2730_ = v_reuseFailAlloc_2731_;
goto v_reusejp_2729_;
}
v_reusejp_2729_:
{
return v___x_2730_;
}
}
}
}
}
else
{
lean_object* v_a_2733_; lean_object* v___x_2735_; uint8_t v_isShared_2736_; uint8_t v_isSharedCheck_2740_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2733_ = lean_ctor_get(v___x_2694_, 0);
v_isSharedCheck_2740_ = !lean_is_exclusive(v___x_2694_);
if (v_isSharedCheck_2740_ == 0)
{
v___x_2735_ = v___x_2694_;
v_isShared_2736_ = v_isSharedCheck_2740_;
goto v_resetjp_2734_;
}
else
{
lean_inc(v_a_2733_);
lean_dec(v___x_2694_);
v___x_2735_ = lean_box(0);
v_isShared_2736_ = v_isSharedCheck_2740_;
goto v_resetjp_2734_;
}
v_resetjp_2734_:
{
lean_object* v___x_2738_; 
if (v_isShared_2736_ == 0)
{
v___x_2738_ = v___x_2735_;
goto v_reusejp_2737_;
}
else
{
lean_object* v_reuseFailAlloc_2739_; 
v_reuseFailAlloc_2739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2739_, 0, v_a_2733_);
v___x_2738_ = v_reuseFailAlloc_2739_;
goto v_reusejp_2737_;
}
v_reusejp_2737_:
{
return v___x_2738_;
}
}
}
}
}
else
{
lean_object* v___x_2741_; lean_object* v___x_2742_; 
lean_dec_ref(v___x_2473_);
v___x_2741_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__13));
v___x_2742_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2741_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2742_) == 0)
{
uint8_t v___x_2743_; 
lean_dec_ref_known(v___x_2742_, 1);
v___x_2743_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2743_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2744_; 
lean_dec_ref(v___x_2474_);
lean_inc_ref(v_item_2452_);
v___x_2744_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2744_) == 0)
{
lean_object* v_value_2745_; lean_object* v___x_2746_; 
lean_dec_ref_known(v___x_2744_, 1);
v_value_2745_ = lean_ctor_get(v_item_2452_, 2);
lean_inc(v_value_2745_);
lean_dec_ref(v_item_2452_);
v___x_2746_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_2745_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2746_) == 0)
{
lean_object* v_a_2747_; lean_object* v___x_2749_; uint8_t v_isShared_2750_; uint8_t v_isSharedCheck_2774_; 
v_a_2747_ = lean_ctor_get(v___x_2746_, 0);
v_isSharedCheck_2774_ = !lean_is_exclusive(v___x_2746_);
if (v_isSharedCheck_2774_ == 0)
{
v___x_2749_ = v___x_2746_;
v_isShared_2750_ = v_isSharedCheck_2774_;
goto v_resetjp_2748_;
}
else
{
lean_inc(v_a_2747_);
lean_dec(v___x_2746_);
v___x_2749_ = lean_box(0);
v_isShared_2750_ = v_isSharedCheck_2774_;
goto v_resetjp_2748_;
}
v_resetjp_2748_:
{
uint8_t v_closePre_2751_; uint8_t v_closePost_2752_; uint8_t v_transparency_2753_; uint8_t v_postTransparency_2754_; uint8_t v_preferLHS_2755_; uint8_t v_partialApp_2756_; uint8_t v_sameFun_2757_; lean_object* v_maxArgs_2758_; uint8_t v_typeEqs_2759_; uint8_t v_etaExpand_2760_; uint8_t v_useCongrSimp_2761_; uint8_t v_beqEq_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2773_; 
v_closePre_2751_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2752_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2753_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_postTransparency_2754_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2755_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2756_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2757_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2758_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2759_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2760_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2761_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2762_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2773_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2773_ == 0)
{
v___x_2764_ = v_config_2451_;
v_isShared_2765_ = v_isSharedCheck_2773_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_maxArgs_2758_);
lean_dec(v_config_2451_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2773_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v___x_2767_; 
if (v_isShared_2765_ == 0)
{
v___x_2767_ = v___x_2764_;
goto v_reusejp_2766_;
}
else
{
lean_object* v_reuseFailAlloc_2772_; 
v_reuseFailAlloc_2772_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2772_, 0, v_maxArgs_2758_);
lean_ctor_set_uint8(v_reuseFailAlloc_2772_, sizeof(void*)*1, v_closePre_2751_);
lean_ctor_set_uint8(v_reuseFailAlloc_2772_, sizeof(void*)*1 + 1, v_closePost_2752_);
lean_ctor_set_uint8(v_reuseFailAlloc_2772_, sizeof(void*)*1 + 2, v_transparency_2753_);
v___x_2767_ = v_reuseFailAlloc_2772_;
goto v_reusejp_2766_;
}
v_reusejp_2766_:
{
uint8_t v___x_2768_; lean_object* v___x_2770_; 
v___x_2768_ = lean_unbox(v_a_2747_);
lean_dec(v_a_2747_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 3, v___x_2768_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 4, v_postTransparency_2754_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 5, v_preferLHS_2755_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 6, v_partialApp_2756_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 7, v_sameFun_2757_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 8, v_typeEqs_2759_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 9, v_etaExpand_2760_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 10, v_useCongrSimp_2761_);
lean_ctor_set_uint8(v___x_2767_, sizeof(void*)*1 + 11, v_beqEq_2762_);
if (v_isShared_2750_ == 0)
{
lean_ctor_set(v___x_2749_, 0, v___x_2767_);
v___x_2770_ = v___x_2749_;
goto v_reusejp_2769_;
}
else
{
lean_object* v_reuseFailAlloc_2771_; 
v_reuseFailAlloc_2771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2771_, 0, v___x_2767_);
v___x_2770_ = v_reuseFailAlloc_2771_;
goto v_reusejp_2769_;
}
v_reusejp_2769_:
{
return v___x_2770_;
}
}
}
}
}
else
{
lean_object* v_a_2775_; lean_object* v___x_2777_; uint8_t v_isShared_2778_; uint8_t v_isSharedCheck_2782_; 
lean_dec_ref(v_config_2451_);
v_a_2775_ = lean_ctor_get(v___x_2746_, 0);
v_isSharedCheck_2782_ = !lean_is_exclusive(v___x_2746_);
if (v_isSharedCheck_2782_ == 0)
{
v___x_2777_ = v___x_2746_;
v_isShared_2778_ = v_isSharedCheck_2782_;
goto v_resetjp_2776_;
}
else
{
lean_inc(v_a_2775_);
lean_dec(v___x_2746_);
v___x_2777_ = lean_box(0);
v_isShared_2778_ = v_isSharedCheck_2782_;
goto v_resetjp_2776_;
}
v_resetjp_2776_:
{
lean_object* v___x_2780_; 
if (v_isShared_2778_ == 0)
{
v___x_2780_ = v___x_2777_;
goto v_reusejp_2779_;
}
else
{
lean_object* v_reuseFailAlloc_2781_; 
v_reuseFailAlloc_2781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2781_, 0, v_a_2775_);
v___x_2780_ = v_reuseFailAlloc_2781_;
goto v_reusejp_2779_;
}
v_reusejp_2779_:
{
return v___x_2780_;
}
}
}
}
else
{
lean_object* v_a_2783_; lean_object* v___x_2785_; uint8_t v_isShared_2786_; uint8_t v_isSharedCheck_2790_; 
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2783_ = lean_ctor_get(v___x_2744_, 0);
v_isSharedCheck_2790_ = !lean_is_exclusive(v___x_2744_);
if (v_isSharedCheck_2790_ == 0)
{
v___x_2785_ = v___x_2744_;
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
else
{
lean_inc(v_a_2783_);
lean_dec(v___x_2744_);
v___x_2785_ = lean_box(0);
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
v_resetjp_2784_:
{
lean_object* v___x_2788_; 
if (v_isShared_2786_ == 0)
{
v___x_2788_ = v___x_2785_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v_a_2783_);
v___x_2788_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
return v___x_2788_;
}
}
}
}
}
else
{
lean_object* v_a_2791_; lean_object* v___x_2793_; uint8_t v_isShared_2794_; uint8_t v_isSharedCheck_2798_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2791_ = lean_ctor_get(v___x_2742_, 0);
v_isSharedCheck_2798_ = !lean_is_exclusive(v___x_2742_);
if (v_isSharedCheck_2798_ == 0)
{
v___x_2793_ = v___x_2742_;
v_isShared_2794_ = v_isSharedCheck_2798_;
goto v_resetjp_2792_;
}
else
{
lean_inc(v_a_2791_);
lean_dec(v___x_2742_);
v___x_2793_ = lean_box(0);
v_isShared_2794_ = v_isSharedCheck_2798_;
goto v_resetjp_2792_;
}
v_resetjp_2792_:
{
lean_object* v___x_2796_; 
if (v_isShared_2794_ == 0)
{
v___x_2796_ = v___x_2793_;
goto v_reusejp_2795_;
}
else
{
lean_object* v_reuseFailAlloc_2797_; 
v_reuseFailAlloc_2797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2797_, 0, v_a_2791_);
v___x_2796_ = v_reuseFailAlloc_2797_;
goto v_reusejp_2795_;
}
v_reusejp_2795_:
{
return v___x_2796_;
}
}
}
}
}
else
{
lean_object* v___x_2799_; lean_object* v___x_2800_; 
lean_dec_ref(v___x_2473_);
v___x_2799_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__14));
v___x_2800_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2799_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2800_) == 0)
{
uint8_t v___x_2801_; 
lean_dec_ref_known(v___x_2800_, 1);
v___x_2801_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2801_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2802_; 
lean_dec_ref(v___x_2474_);
lean_inc_ref(v_item_2452_);
v___x_2802_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2802_) == 0)
{
lean_object* v_value_2803_; lean_object* v___x_2804_; 
lean_dec_ref_known(v___x_2802_, 1);
v_value_2803_ = lean_ctor_get(v_item_2452_, 2);
lean_inc(v_value_2803_);
lean_dec_ref(v_item_2452_);
v___x_2804_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0(v_value_2803_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2804_) == 0)
{
lean_object* v_a_2805_; lean_object* v___x_2807_; uint8_t v_isShared_2808_; uint8_t v_isSharedCheck_2832_; 
v_a_2805_ = lean_ctor_get(v___x_2804_, 0);
v_isSharedCheck_2832_ = !lean_is_exclusive(v___x_2804_);
if (v_isSharedCheck_2832_ == 0)
{
v___x_2807_ = v___x_2804_;
v_isShared_2808_ = v_isSharedCheck_2832_;
goto v_resetjp_2806_;
}
else
{
lean_inc(v_a_2805_);
lean_dec(v___x_2804_);
v___x_2807_ = lean_box(0);
v_isShared_2808_ = v_isSharedCheck_2832_;
goto v_resetjp_2806_;
}
v_resetjp_2806_:
{
uint8_t v_closePre_2809_; uint8_t v_closePost_2810_; uint8_t v_transparency_2811_; uint8_t v_preTransparency_2812_; uint8_t v_preferLHS_2813_; uint8_t v_partialApp_2814_; uint8_t v_sameFun_2815_; lean_object* v_maxArgs_2816_; uint8_t v_typeEqs_2817_; uint8_t v_etaExpand_2818_; uint8_t v_useCongrSimp_2819_; uint8_t v_beqEq_2820_; lean_object* v___x_2822_; uint8_t v_isShared_2823_; uint8_t v_isSharedCheck_2831_; 
v_closePre_2809_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2810_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2811_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2812_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_preferLHS_2813_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2814_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2815_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2816_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2817_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2818_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2819_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2820_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2831_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2822_ = v_config_2451_;
v_isShared_2823_ = v_isSharedCheck_2831_;
goto v_resetjp_2821_;
}
else
{
lean_inc(v_maxArgs_2816_);
lean_dec(v_config_2451_);
v___x_2822_ = lean_box(0);
v_isShared_2823_ = v_isSharedCheck_2831_;
goto v_resetjp_2821_;
}
v_resetjp_2821_:
{
lean_object* v___x_2825_; 
if (v_isShared_2823_ == 0)
{
v___x_2825_ = v___x_2822_;
goto v_reusejp_2824_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v_maxArgs_2816_);
lean_ctor_set_uint8(v_reuseFailAlloc_2830_, sizeof(void*)*1, v_closePre_2809_);
lean_ctor_set_uint8(v_reuseFailAlloc_2830_, sizeof(void*)*1 + 1, v_closePost_2810_);
lean_ctor_set_uint8(v_reuseFailAlloc_2830_, sizeof(void*)*1 + 2, v_transparency_2811_);
lean_ctor_set_uint8(v_reuseFailAlloc_2830_, sizeof(void*)*1 + 3, v_preTransparency_2812_);
v___x_2825_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2824_;
}
v_reusejp_2824_:
{
uint8_t v___x_2826_; lean_object* v___x_2828_; 
v___x_2826_ = lean_unbox(v_a_2805_);
lean_dec(v_a_2805_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 4, v___x_2826_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 5, v_preferLHS_2813_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 6, v_partialApp_2814_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 7, v_sameFun_2815_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 8, v_typeEqs_2817_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 9, v_etaExpand_2818_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 10, v_useCongrSimp_2819_);
lean_ctor_set_uint8(v___x_2825_, sizeof(void*)*1 + 11, v_beqEq_2820_);
if (v_isShared_2808_ == 0)
{
lean_ctor_set(v___x_2807_, 0, v___x_2825_);
v___x_2828_ = v___x_2807_;
goto v_reusejp_2827_;
}
else
{
lean_object* v_reuseFailAlloc_2829_; 
v_reuseFailAlloc_2829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2829_, 0, v___x_2825_);
v___x_2828_ = v_reuseFailAlloc_2829_;
goto v_reusejp_2827_;
}
v_reusejp_2827_:
{
return v___x_2828_;
}
}
}
}
}
else
{
lean_object* v_a_2833_; lean_object* v___x_2835_; uint8_t v_isShared_2836_; uint8_t v_isSharedCheck_2840_; 
lean_dec_ref(v_config_2451_);
v_a_2833_ = lean_ctor_get(v___x_2804_, 0);
v_isSharedCheck_2840_ = !lean_is_exclusive(v___x_2804_);
if (v_isSharedCheck_2840_ == 0)
{
v___x_2835_ = v___x_2804_;
v_isShared_2836_ = v_isSharedCheck_2840_;
goto v_resetjp_2834_;
}
else
{
lean_inc(v_a_2833_);
lean_dec(v___x_2804_);
v___x_2835_ = lean_box(0);
v_isShared_2836_ = v_isSharedCheck_2840_;
goto v_resetjp_2834_;
}
v_resetjp_2834_:
{
lean_object* v___x_2838_; 
if (v_isShared_2836_ == 0)
{
v___x_2838_ = v___x_2835_;
goto v_reusejp_2837_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v_a_2833_);
v___x_2838_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2837_;
}
v_reusejp_2837_:
{
return v___x_2838_;
}
}
}
}
else
{
lean_object* v_a_2841_; lean_object* v___x_2843_; uint8_t v_isShared_2844_; uint8_t v_isSharedCheck_2848_; 
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2841_ = lean_ctor_get(v___x_2802_, 0);
v_isSharedCheck_2848_ = !lean_is_exclusive(v___x_2802_);
if (v_isSharedCheck_2848_ == 0)
{
v___x_2843_ = v___x_2802_;
v_isShared_2844_ = v_isSharedCheck_2848_;
goto v_resetjp_2842_;
}
else
{
lean_inc(v_a_2841_);
lean_dec(v___x_2802_);
v___x_2843_ = lean_box(0);
v_isShared_2844_ = v_isSharedCheck_2848_;
goto v_resetjp_2842_;
}
v_resetjp_2842_:
{
lean_object* v___x_2846_; 
if (v_isShared_2844_ == 0)
{
v___x_2846_ = v___x_2843_;
goto v_reusejp_2845_;
}
else
{
lean_object* v_reuseFailAlloc_2847_; 
v_reuseFailAlloc_2847_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2847_, 0, v_a_2841_);
v___x_2846_ = v_reuseFailAlloc_2847_;
goto v_reusejp_2845_;
}
v_reusejp_2845_:
{
return v___x_2846_;
}
}
}
}
}
else
{
lean_object* v_a_2849_; lean_object* v___x_2851_; uint8_t v_isShared_2852_; uint8_t v_isSharedCheck_2856_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2849_ = lean_ctor_get(v___x_2800_, 0);
v_isSharedCheck_2856_ = !lean_is_exclusive(v___x_2800_);
if (v_isSharedCheck_2856_ == 0)
{
v___x_2851_ = v___x_2800_;
v_isShared_2852_ = v_isSharedCheck_2856_;
goto v_resetjp_2850_;
}
else
{
lean_inc(v_a_2849_);
lean_dec(v___x_2800_);
v___x_2851_ = lean_box(0);
v_isShared_2852_ = v_isSharedCheck_2856_;
goto v_resetjp_2850_;
}
v_resetjp_2850_:
{
lean_object* v___x_2854_; 
if (v_isShared_2852_ == 0)
{
v___x_2854_ = v___x_2851_;
goto v_reusejp_2853_;
}
else
{
lean_object* v_reuseFailAlloc_2855_; 
v_reuseFailAlloc_2855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2855_, 0, v_a_2849_);
v___x_2854_ = v_reuseFailAlloc_2855_;
goto v_reusejp_2853_;
}
v_reusejp_2853_:
{
return v___x_2854_;
}
}
}
}
}
}
else
{
lean_object* v___x_2857_; uint8_t v___x_2858_; 
v___x_2857_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__15));
v___x_2858_ = lean_string_dec_lt(v___x_2473_, v___x_2857_);
if (v___x_2858_ == 0)
{
uint8_t v___x_2859_; 
v___x_2859_ = lean_string_dec_eq(v___x_2473_, v___x_2857_);
if (v___x_2859_ == 0)
{
lean_object* v___x_2860_; uint8_t v___x_2861_; 
v___x_2860_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__16));
v___x_2861_ = lean_string_dec_eq(v___x_2473_, v___x_2860_);
if (v___x_2861_ == 0)
{
lean_object* v___x_2862_; uint8_t v___x_2863_; 
v___x_2862_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__17));
v___x_2863_ = lean_string_dec_eq(v___x_2473_, v___x_2862_);
if (v___x_2863_ == 0)
{
lean_object* v___x_2864_; uint8_t v___x_2865_; 
v___x_2864_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__18));
v___x_2865_ = lean_string_dec_eq(v___x_2473_, v___x_2864_);
lean_dec_ref(v___x_2473_);
if (v___x_2865_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2866_; lean_object* v___x_2867_; 
v___x_2866_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__19));
v___x_2867_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2866_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2867_) == 0)
{
uint8_t v___x_2868_; 
lean_dec_ref_known(v___x_2867_, 1);
v___x_2868_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2868_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2869_; 
lean_dec_ref(v___x_2474_);
v___x_2869_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2869_) == 0)
{
lean_object* v_a_2870_; lean_object* v___x_2872_; uint8_t v_isShared_2873_; uint8_t v_isSharedCheck_2897_; 
v_a_2870_ = lean_ctor_get(v___x_2869_, 0);
v_isSharedCheck_2897_ = !lean_is_exclusive(v___x_2869_);
if (v_isSharedCheck_2897_ == 0)
{
v___x_2872_ = v___x_2869_;
v_isShared_2873_ = v_isSharedCheck_2897_;
goto v_resetjp_2871_;
}
else
{
lean_inc(v_a_2870_);
lean_dec(v___x_2869_);
v___x_2872_ = lean_box(0);
v_isShared_2873_ = v_isSharedCheck_2897_;
goto v_resetjp_2871_;
}
v_resetjp_2871_:
{
uint8_t v_closePre_2874_; uint8_t v_closePost_2875_; uint8_t v_transparency_2876_; uint8_t v_preTransparency_2877_; uint8_t v_postTransparency_2878_; uint8_t v_preferLHS_2879_; uint8_t v_sameFun_2880_; lean_object* v_maxArgs_2881_; uint8_t v_typeEqs_2882_; uint8_t v_etaExpand_2883_; uint8_t v_useCongrSimp_2884_; uint8_t v_beqEq_2885_; lean_object* v___x_2887_; uint8_t v_isShared_2888_; uint8_t v_isSharedCheck_2896_; 
v_closePre_2874_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2875_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2876_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2877_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2878_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2879_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_sameFun_2880_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2881_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2882_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2883_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2884_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2885_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2896_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2896_ == 0)
{
v___x_2887_ = v_config_2451_;
v_isShared_2888_ = v_isSharedCheck_2896_;
goto v_resetjp_2886_;
}
else
{
lean_inc(v_maxArgs_2881_);
lean_dec(v_config_2451_);
v___x_2887_ = lean_box(0);
v_isShared_2888_ = v_isSharedCheck_2896_;
goto v_resetjp_2886_;
}
v_resetjp_2886_:
{
lean_object* v___x_2890_; 
if (v_isShared_2888_ == 0)
{
v___x_2890_ = v___x_2887_;
goto v_reusejp_2889_;
}
else
{
lean_object* v_reuseFailAlloc_2895_; 
v_reuseFailAlloc_2895_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2895_, 0, v_maxArgs_2881_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1, v_closePre_2874_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1 + 1, v_closePost_2875_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1 + 2, v_transparency_2876_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1 + 3, v_preTransparency_2877_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1 + 4, v_postTransparency_2878_);
lean_ctor_set_uint8(v_reuseFailAlloc_2895_, sizeof(void*)*1 + 5, v_preferLHS_2879_);
v___x_2890_ = v_reuseFailAlloc_2895_;
goto v_reusejp_2889_;
}
v_reusejp_2889_:
{
uint8_t v___x_2891_; lean_object* v___x_2893_; 
v___x_2891_ = lean_unbox(v_a_2870_);
lean_dec(v_a_2870_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 6, v___x_2891_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 7, v_sameFun_2880_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 8, v_typeEqs_2882_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 9, v_etaExpand_2883_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 10, v_useCongrSimp_2884_);
lean_ctor_set_uint8(v___x_2890_, sizeof(void*)*1 + 11, v_beqEq_2885_);
if (v_isShared_2873_ == 0)
{
lean_ctor_set(v___x_2872_, 0, v___x_2890_);
v___x_2893_ = v___x_2872_;
goto v_reusejp_2892_;
}
else
{
lean_object* v_reuseFailAlloc_2894_; 
v_reuseFailAlloc_2894_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2894_, 0, v___x_2890_);
v___x_2893_ = v_reuseFailAlloc_2894_;
goto v_reusejp_2892_;
}
v_reusejp_2892_:
{
return v___x_2893_;
}
}
}
}
}
else
{
lean_object* v_a_2898_; lean_object* v___x_2900_; uint8_t v_isShared_2901_; uint8_t v_isSharedCheck_2905_; 
lean_dec_ref(v_config_2451_);
v_a_2898_ = lean_ctor_get(v___x_2869_, 0);
v_isSharedCheck_2905_ = !lean_is_exclusive(v___x_2869_);
if (v_isSharedCheck_2905_ == 0)
{
v___x_2900_ = v___x_2869_;
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
else
{
lean_inc(v_a_2898_);
lean_dec(v___x_2869_);
v___x_2900_ = lean_box(0);
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
v_resetjp_2899_:
{
lean_object* v___x_2903_; 
if (v_isShared_2901_ == 0)
{
v___x_2903_ = v___x_2900_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2904_; 
v_reuseFailAlloc_2904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2904_, 0, v_a_2898_);
v___x_2903_ = v_reuseFailAlloc_2904_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
return v___x_2903_;
}
}
}
}
}
else
{
lean_object* v_a_2906_; lean_object* v___x_2908_; uint8_t v_isShared_2909_; uint8_t v_isSharedCheck_2913_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2906_ = lean_ctor_get(v___x_2867_, 0);
v_isSharedCheck_2913_ = !lean_is_exclusive(v___x_2867_);
if (v_isSharedCheck_2913_ == 0)
{
v___x_2908_ = v___x_2867_;
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
else
{
lean_inc(v_a_2906_);
lean_dec(v___x_2867_);
v___x_2908_ = lean_box(0);
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
v_resetjp_2907_:
{
lean_object* v___x_2911_; 
if (v_isShared_2909_ == 0)
{
v___x_2911_ = v___x_2908_;
goto v_reusejp_2910_;
}
else
{
lean_object* v_reuseFailAlloc_2912_; 
v_reuseFailAlloc_2912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2912_, 0, v_a_2906_);
v___x_2911_ = v_reuseFailAlloc_2912_;
goto v_reusejp_2910_;
}
v_reusejp_2910_:
{
return v___x_2911_;
}
}
}
}
}
else
{
lean_object* v___x_2914_; lean_object* v___x_2915_; 
lean_dec_ref(v___x_2473_);
v___x_2914_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__20));
v___x_2915_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2914_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2915_) == 0)
{
uint8_t v___x_2916_; 
lean_dec_ref_known(v___x_2915_, 1);
v___x_2916_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2916_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2917_; 
lean_dec_ref(v___x_2474_);
lean_inc_ref(v_item_2452_);
v___x_2917_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2917_) == 0)
{
lean_object* v_value_2918_; lean_object* v___x_2919_; 
lean_dec_ref_known(v___x_2917_, 1);
v_value_2918_ = lean_ctor_get(v_item_2452_, 2);
lean_inc(v_value_2918_);
lean_dec_ref(v_item_2452_);
v___x_2919_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1(v_value_2918_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2919_) == 0)
{
lean_object* v_a_2920_; lean_object* v___x_2922_; uint8_t v_isShared_2923_; uint8_t v_isSharedCheck_2947_; 
v_a_2920_ = lean_ctor_get(v___x_2919_, 0);
v_isSharedCheck_2947_ = !lean_is_exclusive(v___x_2919_);
if (v_isSharedCheck_2947_ == 0)
{
v___x_2922_ = v___x_2919_;
v_isShared_2923_ = v_isSharedCheck_2947_;
goto v_resetjp_2921_;
}
else
{
lean_inc(v_a_2920_);
lean_dec(v___x_2919_);
v___x_2922_ = lean_box(0);
v_isShared_2923_ = v_isSharedCheck_2947_;
goto v_resetjp_2921_;
}
v_resetjp_2921_:
{
uint8_t v_closePre_2924_; uint8_t v_closePost_2925_; uint8_t v_transparency_2926_; uint8_t v_preTransparency_2927_; uint8_t v_postTransparency_2928_; uint8_t v_preferLHS_2929_; uint8_t v_partialApp_2930_; uint8_t v_sameFun_2931_; uint8_t v_typeEqs_2932_; uint8_t v_etaExpand_2933_; uint8_t v_useCongrSimp_2934_; uint8_t v_beqEq_2935_; lean_object* v___x_2937_; uint8_t v_isShared_2938_; uint8_t v_isSharedCheck_2945_; 
v_closePre_2924_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2925_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2926_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2927_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2928_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2929_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2930_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2931_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_typeEqs_2932_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_2933_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_2934_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2935_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_2945_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_2945_ == 0)
{
lean_object* v_unused_2946_; 
v_unused_2946_ = lean_ctor_get(v_config_2451_, 0);
lean_dec(v_unused_2946_);
v___x_2937_ = v_config_2451_;
v_isShared_2938_ = v_isSharedCheck_2945_;
goto v_resetjp_2936_;
}
else
{
lean_dec(v_config_2451_);
v___x_2937_ = lean_box(0);
v_isShared_2938_ = v_isSharedCheck_2945_;
goto v_resetjp_2936_;
}
v_resetjp_2936_:
{
lean_object* v___x_2940_; 
if (v_isShared_2938_ == 0)
{
lean_ctor_set(v___x_2937_, 0, v_a_2920_);
v___x_2940_ = v___x_2937_;
goto v_reusejp_2939_;
}
else
{
lean_object* v_reuseFailAlloc_2944_; 
v_reuseFailAlloc_2944_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_2944_, 0, v_a_2920_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1, v_closePre_2924_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 1, v_closePost_2925_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 2, v_transparency_2926_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 3, v_preTransparency_2927_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 4, v_postTransparency_2928_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 5, v_preferLHS_2929_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 6, v_partialApp_2930_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 7, v_sameFun_2931_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 8, v_typeEqs_2932_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 9, v_etaExpand_2933_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 10, v_useCongrSimp_2934_);
lean_ctor_set_uint8(v_reuseFailAlloc_2944_, sizeof(void*)*1 + 11, v_beqEq_2935_);
v___x_2940_ = v_reuseFailAlloc_2944_;
goto v_reusejp_2939_;
}
v_reusejp_2939_:
{
lean_object* v___x_2942_; 
if (v_isShared_2923_ == 0)
{
lean_ctor_set(v___x_2922_, 0, v___x_2940_);
v___x_2942_ = v___x_2922_;
goto v_reusejp_2941_;
}
else
{
lean_object* v_reuseFailAlloc_2943_; 
v_reuseFailAlloc_2943_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2943_, 0, v___x_2940_);
v___x_2942_ = v_reuseFailAlloc_2943_;
goto v_reusejp_2941_;
}
v_reusejp_2941_:
{
return v___x_2942_;
}
}
}
}
}
else
{
lean_object* v_a_2948_; lean_object* v___x_2950_; uint8_t v_isShared_2951_; uint8_t v_isSharedCheck_2955_; 
lean_dec_ref(v_config_2451_);
v_a_2948_ = lean_ctor_get(v___x_2919_, 0);
v_isSharedCheck_2955_ = !lean_is_exclusive(v___x_2919_);
if (v_isSharedCheck_2955_ == 0)
{
v___x_2950_ = v___x_2919_;
v_isShared_2951_ = v_isSharedCheck_2955_;
goto v_resetjp_2949_;
}
else
{
lean_inc(v_a_2948_);
lean_dec(v___x_2919_);
v___x_2950_ = lean_box(0);
v_isShared_2951_ = v_isSharedCheck_2955_;
goto v_resetjp_2949_;
}
v_resetjp_2949_:
{
lean_object* v___x_2953_; 
if (v_isShared_2951_ == 0)
{
v___x_2953_ = v___x_2950_;
goto v_reusejp_2952_;
}
else
{
lean_object* v_reuseFailAlloc_2954_; 
v_reuseFailAlloc_2954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2954_, 0, v_a_2948_);
v___x_2953_ = v_reuseFailAlloc_2954_;
goto v_reusejp_2952_;
}
v_reusejp_2952_:
{
return v___x_2953_;
}
}
}
}
else
{
lean_object* v_a_2956_; lean_object* v___x_2958_; uint8_t v_isShared_2959_; uint8_t v_isSharedCheck_2963_; 
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2956_ = lean_ctor_get(v___x_2917_, 0);
v_isSharedCheck_2963_ = !lean_is_exclusive(v___x_2917_);
if (v_isSharedCheck_2963_ == 0)
{
v___x_2958_ = v___x_2917_;
v_isShared_2959_ = v_isSharedCheck_2963_;
goto v_resetjp_2957_;
}
else
{
lean_inc(v_a_2956_);
lean_dec(v___x_2917_);
v___x_2958_ = lean_box(0);
v_isShared_2959_ = v_isSharedCheck_2963_;
goto v_resetjp_2957_;
}
v_resetjp_2957_:
{
lean_object* v___x_2961_; 
if (v_isShared_2959_ == 0)
{
v___x_2961_ = v___x_2958_;
goto v_reusejp_2960_;
}
else
{
lean_object* v_reuseFailAlloc_2962_; 
v_reuseFailAlloc_2962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2962_, 0, v_a_2956_);
v___x_2961_ = v_reuseFailAlloc_2962_;
goto v_reusejp_2960_;
}
v_reusejp_2960_:
{
return v___x_2961_;
}
}
}
}
}
else
{
lean_object* v_a_2964_; lean_object* v___x_2966_; uint8_t v_isShared_2967_; uint8_t v_isSharedCheck_2971_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_2964_ = lean_ctor_get(v___x_2915_, 0);
v_isSharedCheck_2971_ = !lean_is_exclusive(v___x_2915_);
if (v_isSharedCheck_2971_ == 0)
{
v___x_2966_ = v___x_2915_;
v_isShared_2967_ = v_isSharedCheck_2971_;
goto v_resetjp_2965_;
}
else
{
lean_inc(v_a_2964_);
lean_dec(v___x_2915_);
v___x_2966_ = lean_box(0);
v_isShared_2967_ = v_isSharedCheck_2971_;
goto v_resetjp_2965_;
}
v_resetjp_2965_:
{
lean_object* v___x_2969_; 
if (v_isShared_2967_ == 0)
{
v___x_2969_ = v___x_2966_;
goto v_reusejp_2968_;
}
else
{
lean_object* v_reuseFailAlloc_2970_; 
v_reuseFailAlloc_2970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2970_, 0, v_a_2964_);
v___x_2969_ = v_reuseFailAlloc_2970_;
goto v_reusejp_2968_;
}
v_reusejp_2968_:
{
return v___x_2969_;
}
}
}
}
}
else
{
lean_object* v___x_2972_; lean_object* v___x_2973_; 
lean_dec_ref(v___x_2473_);
v___x_2972_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__21));
v___x_2973_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_2972_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2973_) == 0)
{
uint8_t v___x_2974_; 
lean_dec_ref_known(v___x_2973_, 1);
v___x_2974_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_2974_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_2975_; 
lean_dec_ref(v___x_2474_);
v___x_2975_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_2975_) == 0)
{
lean_object* v_a_2976_; lean_object* v___x_2978_; uint8_t v_isShared_2979_; uint8_t v_isSharedCheck_3003_; 
v_a_2976_ = lean_ctor_get(v___x_2975_, 0);
v_isSharedCheck_3003_ = !lean_is_exclusive(v___x_2975_);
if (v_isSharedCheck_3003_ == 0)
{
v___x_2978_ = v___x_2975_;
v_isShared_2979_ = v_isSharedCheck_3003_;
goto v_resetjp_2977_;
}
else
{
lean_inc(v_a_2976_);
lean_dec(v___x_2975_);
v___x_2978_ = lean_box(0);
v_isShared_2979_ = v_isSharedCheck_3003_;
goto v_resetjp_2977_;
}
v_resetjp_2977_:
{
uint8_t v_closePre_2980_; uint8_t v_closePost_2981_; uint8_t v_transparency_2982_; uint8_t v_preTransparency_2983_; uint8_t v_postTransparency_2984_; uint8_t v_preferLHS_2985_; uint8_t v_partialApp_2986_; uint8_t v_sameFun_2987_; lean_object* v_maxArgs_2988_; uint8_t v_typeEqs_2989_; uint8_t v_useCongrSimp_2990_; uint8_t v_beqEq_2991_; lean_object* v___x_2993_; uint8_t v_isShared_2994_; uint8_t v_isSharedCheck_3002_; 
v_closePre_2980_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_2981_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_2982_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_2983_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_2984_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_2985_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_2986_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_2987_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_2988_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_2989_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_useCongrSimp_2990_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_2991_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_3002_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_3002_ == 0)
{
v___x_2993_ = v_config_2451_;
v_isShared_2994_ = v_isSharedCheck_3002_;
goto v_resetjp_2992_;
}
else
{
lean_inc(v_maxArgs_2988_);
lean_dec(v_config_2451_);
v___x_2993_ = lean_box(0);
v_isShared_2994_ = v_isSharedCheck_3002_;
goto v_resetjp_2992_;
}
v_resetjp_2992_:
{
lean_object* v___x_2996_; 
if (v_isShared_2994_ == 0)
{
v___x_2996_ = v___x_2993_;
goto v_reusejp_2995_;
}
else
{
lean_object* v_reuseFailAlloc_3001_; 
v_reuseFailAlloc_3001_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_3001_, 0, v_maxArgs_2988_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1, v_closePre_2980_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 1, v_closePost_2981_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 2, v_transparency_2982_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 3, v_preTransparency_2983_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 4, v_postTransparency_2984_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 5, v_preferLHS_2985_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 6, v_partialApp_2986_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 7, v_sameFun_2987_);
lean_ctor_set_uint8(v_reuseFailAlloc_3001_, sizeof(void*)*1 + 8, v_typeEqs_2989_);
v___x_2996_ = v_reuseFailAlloc_3001_;
goto v_reusejp_2995_;
}
v_reusejp_2995_:
{
uint8_t v___x_2997_; lean_object* v___x_2999_; 
v___x_2997_ = lean_unbox(v_a_2976_);
lean_dec(v_a_2976_);
lean_ctor_set_uint8(v___x_2996_, sizeof(void*)*1 + 9, v___x_2997_);
lean_ctor_set_uint8(v___x_2996_, sizeof(void*)*1 + 10, v_useCongrSimp_2990_);
lean_ctor_set_uint8(v___x_2996_, sizeof(void*)*1 + 11, v_beqEq_2991_);
if (v_isShared_2979_ == 0)
{
lean_ctor_set(v___x_2978_, 0, v___x_2996_);
v___x_2999_ = v___x_2978_;
goto v_reusejp_2998_;
}
else
{
lean_object* v_reuseFailAlloc_3000_; 
v_reuseFailAlloc_3000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3000_, 0, v___x_2996_);
v___x_2999_ = v_reuseFailAlloc_3000_;
goto v_reusejp_2998_;
}
v_reusejp_2998_:
{
return v___x_2999_;
}
}
}
}
}
else
{
lean_object* v_a_3004_; lean_object* v___x_3006_; uint8_t v_isShared_3007_; uint8_t v_isSharedCheck_3011_; 
lean_dec_ref(v_config_2451_);
v_a_3004_ = lean_ctor_get(v___x_2975_, 0);
v_isSharedCheck_3011_ = !lean_is_exclusive(v___x_2975_);
if (v_isSharedCheck_3011_ == 0)
{
v___x_3006_ = v___x_2975_;
v_isShared_3007_ = v_isSharedCheck_3011_;
goto v_resetjp_3005_;
}
else
{
lean_inc(v_a_3004_);
lean_dec(v___x_2975_);
v___x_3006_ = lean_box(0);
v_isShared_3007_ = v_isSharedCheck_3011_;
goto v_resetjp_3005_;
}
v_resetjp_3005_:
{
lean_object* v___x_3009_; 
if (v_isShared_3007_ == 0)
{
v___x_3009_ = v___x_3006_;
goto v_reusejp_3008_;
}
else
{
lean_object* v_reuseFailAlloc_3010_; 
v_reuseFailAlloc_3010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3010_, 0, v_a_3004_);
v___x_3009_ = v_reuseFailAlloc_3010_;
goto v_reusejp_3008_;
}
v_reusejp_3008_:
{
return v___x_3009_;
}
}
}
}
}
else
{
lean_object* v_a_3012_; lean_object* v___x_3014_; uint8_t v_isShared_3015_; uint8_t v_isSharedCheck_3019_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_3012_ = lean_ctor_get(v___x_2973_, 0);
v_isSharedCheck_3019_ = !lean_is_exclusive(v___x_2973_);
if (v_isSharedCheck_3019_ == 0)
{
v___x_3014_ = v___x_2973_;
v_isShared_3015_ = v_isSharedCheck_3019_;
goto v_resetjp_3013_;
}
else
{
lean_inc(v_a_3012_);
lean_dec(v___x_2973_);
v___x_3014_ = lean_box(0);
v_isShared_3015_ = v_isSharedCheck_3019_;
goto v_resetjp_3013_;
}
v_resetjp_3013_:
{
lean_object* v___x_3017_; 
if (v_isShared_3015_ == 0)
{
v___x_3017_ = v___x_3014_;
goto v_reusejp_3016_;
}
else
{
lean_object* v_reuseFailAlloc_3018_; 
v_reuseFailAlloc_3018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3018_, 0, v_a_3012_);
v___x_3017_ = v_reuseFailAlloc_3018_;
goto v_reusejp_3016_;
}
v_reusejp_3016_:
{
return v___x_3017_;
}
}
}
}
}
else
{
uint8_t v___x_3020_; 
lean_dec_ref(v___x_2473_);
lean_dec_ref(v_config_2451_);
v___x_3020_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_3020_ == 0)
{
lean_dec_ref(v_item_2452_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v_value_3021_; lean_object* v___x_3022_; 
lean_dec_ref(v___x_2474_);
v_value_3021_ = lean_ctor_get(v_item_2452_, 2);
lean_inc(v_value_3021_);
lean_dec_ref(v_item_2452_);
v___x_3022_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem_spec__0(v_value_3021_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
return v___x_3022_;
}
}
}
else
{
lean_object* v___x_3023_; uint8_t v___x_3024_; 
v___x_3023_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__22));
v___x_3024_ = lean_string_dec_eq(v___x_2473_, v___x_3023_);
if (v___x_3024_ == 0)
{
lean_object* v___x_3025_; uint8_t v___x_3026_; 
v___x_3025_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__23));
v___x_3026_ = lean_string_dec_eq(v___x_2473_, v___x_3025_);
if (v___x_3026_ == 0)
{
lean_object* v___x_3027_; uint8_t v___x_3028_; 
v___x_3027_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__24));
v___x_3028_ = lean_string_dec_eq(v___x_2473_, v___x_3027_);
lean_dec_ref(v___x_2473_);
if (v___x_3028_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_3029_; lean_object* v___x_3030_; 
v___x_3029_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__25));
v___x_3030_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_3029_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3030_) == 0)
{
uint8_t v___x_3031_; 
lean_dec_ref_known(v___x_3030_, 1);
v___x_3031_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_3031_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_3032_; 
lean_dec_ref(v___x_2474_);
v___x_3032_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3032_) == 0)
{
lean_object* v_a_3033_; lean_object* v___x_3035_; uint8_t v_isShared_3036_; uint8_t v_isSharedCheck_3060_; 
v_a_3033_ = lean_ctor_get(v___x_3032_, 0);
v_isSharedCheck_3060_ = !lean_is_exclusive(v___x_3032_);
if (v_isSharedCheck_3060_ == 0)
{
v___x_3035_ = v___x_3032_;
v_isShared_3036_ = v_isSharedCheck_3060_;
goto v_resetjp_3034_;
}
else
{
lean_inc(v_a_3033_);
lean_dec(v___x_3032_);
v___x_3035_ = lean_box(0);
v_isShared_3036_ = v_isSharedCheck_3060_;
goto v_resetjp_3034_;
}
v_resetjp_3034_:
{
uint8_t v_closePost_3037_; uint8_t v_transparency_3038_; uint8_t v_preTransparency_3039_; uint8_t v_postTransparency_3040_; uint8_t v_preferLHS_3041_; uint8_t v_partialApp_3042_; uint8_t v_sameFun_3043_; lean_object* v_maxArgs_3044_; uint8_t v_typeEqs_3045_; uint8_t v_etaExpand_3046_; uint8_t v_useCongrSimp_3047_; uint8_t v_beqEq_3048_; lean_object* v___x_3050_; uint8_t v_isShared_3051_; uint8_t v_isSharedCheck_3059_; 
v_closePost_3037_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_3038_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_3039_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_3040_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_3041_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_3042_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_3043_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_3044_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_3045_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_3046_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_3047_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_3048_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_3059_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_3059_ == 0)
{
v___x_3050_ = v_config_2451_;
v_isShared_3051_ = v_isSharedCheck_3059_;
goto v_resetjp_3049_;
}
else
{
lean_inc(v_maxArgs_3044_);
lean_dec(v_config_2451_);
v___x_3050_ = lean_box(0);
v_isShared_3051_ = v_isSharedCheck_3059_;
goto v_resetjp_3049_;
}
v_resetjp_3049_:
{
lean_object* v___x_3053_; 
if (v_isShared_3051_ == 0)
{
v___x_3053_ = v___x_3050_;
goto v_reusejp_3052_;
}
else
{
lean_object* v_reuseFailAlloc_3058_; 
v_reuseFailAlloc_3058_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_3058_, 0, v_maxArgs_3044_);
v___x_3053_ = v_reuseFailAlloc_3058_;
goto v_reusejp_3052_;
}
v_reusejp_3052_:
{
uint8_t v___x_3054_; lean_object* v___x_3056_; 
v___x_3054_ = lean_unbox(v_a_3033_);
lean_dec(v_a_3033_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1, v___x_3054_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 1, v_closePost_3037_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 2, v_transparency_3038_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 3, v_preTransparency_3039_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 4, v_postTransparency_3040_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 5, v_preferLHS_3041_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 6, v_partialApp_3042_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 7, v_sameFun_3043_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 8, v_typeEqs_3045_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 9, v_etaExpand_3046_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 10, v_useCongrSimp_3047_);
lean_ctor_set_uint8(v___x_3053_, sizeof(void*)*1 + 11, v_beqEq_3048_);
if (v_isShared_3036_ == 0)
{
lean_ctor_set(v___x_3035_, 0, v___x_3053_);
v___x_3056_ = v___x_3035_;
goto v_reusejp_3055_;
}
else
{
lean_object* v_reuseFailAlloc_3057_; 
v_reuseFailAlloc_3057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3057_, 0, v___x_3053_);
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
}
else
{
lean_object* v_a_3061_; lean_object* v___x_3063_; uint8_t v_isShared_3064_; uint8_t v_isSharedCheck_3068_; 
lean_dec_ref(v_config_2451_);
v_a_3061_ = lean_ctor_get(v___x_3032_, 0);
v_isSharedCheck_3068_ = !lean_is_exclusive(v___x_3032_);
if (v_isSharedCheck_3068_ == 0)
{
v___x_3063_ = v___x_3032_;
v_isShared_3064_ = v_isSharedCheck_3068_;
goto v_resetjp_3062_;
}
else
{
lean_inc(v_a_3061_);
lean_dec(v___x_3032_);
v___x_3063_ = lean_box(0);
v_isShared_3064_ = v_isSharedCheck_3068_;
goto v_resetjp_3062_;
}
v_resetjp_3062_:
{
lean_object* v___x_3066_; 
if (v_isShared_3064_ == 0)
{
v___x_3066_ = v___x_3063_;
goto v_reusejp_3065_;
}
else
{
lean_object* v_reuseFailAlloc_3067_; 
v_reuseFailAlloc_3067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3067_, 0, v_a_3061_);
v___x_3066_ = v_reuseFailAlloc_3067_;
goto v_reusejp_3065_;
}
v_reusejp_3065_:
{
return v___x_3066_;
}
}
}
}
}
else
{
lean_object* v_a_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3076_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_3069_ = lean_ctor_get(v___x_3030_, 0);
v_isSharedCheck_3076_ = !lean_is_exclusive(v___x_3030_);
if (v_isSharedCheck_3076_ == 0)
{
v___x_3071_ = v___x_3030_;
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
else
{
lean_inc(v_a_3069_);
lean_dec(v___x_3030_);
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
v_reuseFailAlloc_3075_ = lean_alloc_ctor(1, 1, 0);
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
}
}
else
{
lean_object* v___x_3077_; lean_object* v___x_3078_; 
lean_dec_ref(v___x_2473_);
v___x_3077_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__26));
v___x_3078_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_3077_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3078_) == 0)
{
uint8_t v___x_3079_; 
lean_dec_ref_known(v___x_3078_, 1);
v___x_3079_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_3079_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_3080_; 
lean_dec_ref(v___x_2474_);
v___x_3080_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3080_) == 0)
{
lean_object* v_a_3081_; lean_object* v___x_3083_; uint8_t v_isShared_3084_; uint8_t v_isSharedCheck_3108_; 
v_a_3081_ = lean_ctor_get(v___x_3080_, 0);
v_isSharedCheck_3108_ = !lean_is_exclusive(v___x_3080_);
if (v_isSharedCheck_3108_ == 0)
{
v___x_3083_ = v___x_3080_;
v_isShared_3084_ = v_isSharedCheck_3108_;
goto v_resetjp_3082_;
}
else
{
lean_inc(v_a_3081_);
lean_dec(v___x_3080_);
v___x_3083_ = lean_box(0);
v_isShared_3084_ = v_isSharedCheck_3108_;
goto v_resetjp_3082_;
}
v_resetjp_3082_:
{
uint8_t v_closePre_3085_; uint8_t v_transparency_3086_; uint8_t v_preTransparency_3087_; uint8_t v_postTransparency_3088_; uint8_t v_preferLHS_3089_; uint8_t v_partialApp_3090_; uint8_t v_sameFun_3091_; lean_object* v_maxArgs_3092_; uint8_t v_typeEqs_3093_; uint8_t v_etaExpand_3094_; uint8_t v_useCongrSimp_3095_; uint8_t v_beqEq_3096_; lean_object* v___x_3098_; uint8_t v_isShared_3099_; uint8_t v_isSharedCheck_3107_; 
v_closePre_3085_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_transparency_3086_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_3087_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_3088_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_3089_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_3090_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_3091_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_3092_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_3093_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_3094_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_3095_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_beqEq_3096_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 11);
v_isSharedCheck_3107_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_3107_ == 0)
{
v___x_3098_ = v_config_2451_;
v_isShared_3099_ = v_isSharedCheck_3107_;
goto v_resetjp_3097_;
}
else
{
lean_inc(v_maxArgs_3092_);
lean_dec(v_config_2451_);
v___x_3098_ = lean_box(0);
v_isShared_3099_ = v_isSharedCheck_3107_;
goto v_resetjp_3097_;
}
v_resetjp_3097_:
{
lean_object* v___x_3101_; 
if (v_isShared_3099_ == 0)
{
v___x_3101_ = v___x_3098_;
goto v_reusejp_3100_;
}
else
{
lean_object* v_reuseFailAlloc_3106_; 
v_reuseFailAlloc_3106_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_3106_, 0, v_maxArgs_3092_);
lean_ctor_set_uint8(v_reuseFailAlloc_3106_, sizeof(void*)*1, v_closePre_3085_);
v___x_3101_ = v_reuseFailAlloc_3106_;
goto v_reusejp_3100_;
}
v_reusejp_3100_:
{
uint8_t v___x_3102_; lean_object* v___x_3104_; 
v___x_3102_ = lean_unbox(v_a_3081_);
lean_dec(v_a_3081_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 1, v___x_3102_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 2, v_transparency_3086_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 3, v_preTransparency_3087_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 4, v_postTransparency_3088_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 5, v_preferLHS_3089_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 6, v_partialApp_3090_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 7, v_sameFun_3091_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 8, v_typeEqs_3093_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 9, v_etaExpand_3094_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 10, v_useCongrSimp_3095_);
lean_ctor_set_uint8(v___x_3101_, sizeof(void*)*1 + 11, v_beqEq_3096_);
if (v_isShared_3084_ == 0)
{
lean_ctor_set(v___x_3083_, 0, v___x_3101_);
v___x_3104_ = v___x_3083_;
goto v_reusejp_3103_;
}
else
{
lean_object* v_reuseFailAlloc_3105_; 
v_reuseFailAlloc_3105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3105_, 0, v___x_3101_);
v___x_3104_ = v_reuseFailAlloc_3105_;
goto v_reusejp_3103_;
}
v_reusejp_3103_:
{
return v___x_3104_;
}
}
}
}
}
else
{
lean_object* v_a_3109_; lean_object* v___x_3111_; uint8_t v_isShared_3112_; uint8_t v_isSharedCheck_3116_; 
lean_dec_ref(v_config_2451_);
v_a_3109_ = lean_ctor_get(v___x_3080_, 0);
v_isSharedCheck_3116_ = !lean_is_exclusive(v___x_3080_);
if (v_isSharedCheck_3116_ == 0)
{
v___x_3111_ = v___x_3080_;
v_isShared_3112_ = v_isSharedCheck_3116_;
goto v_resetjp_3110_;
}
else
{
lean_inc(v_a_3109_);
lean_dec(v___x_3080_);
v___x_3111_ = lean_box(0);
v_isShared_3112_ = v_isSharedCheck_3116_;
goto v_resetjp_3110_;
}
v_resetjp_3110_:
{
lean_object* v___x_3114_; 
if (v_isShared_3112_ == 0)
{
v___x_3114_ = v___x_3111_;
goto v_reusejp_3113_;
}
else
{
lean_object* v_reuseFailAlloc_3115_; 
v_reuseFailAlloc_3115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3115_, 0, v_a_3109_);
v___x_3114_ = v_reuseFailAlloc_3115_;
goto v_reusejp_3113_;
}
v_reusejp_3113_:
{
return v___x_3114_;
}
}
}
}
}
else
{
lean_object* v_a_3117_; lean_object* v___x_3119_; uint8_t v_isShared_3120_; uint8_t v_isSharedCheck_3124_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_3117_ = lean_ctor_get(v___x_3078_, 0);
v_isSharedCheck_3124_ = !lean_is_exclusive(v___x_3078_);
if (v_isSharedCheck_3124_ == 0)
{
v___x_3119_ = v___x_3078_;
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
else
{
lean_inc(v_a_3117_);
lean_dec(v___x_3078_);
v___x_3119_ = lean_box(0);
v_isShared_3120_ = v_isSharedCheck_3124_;
goto v_resetjp_3118_;
}
v_resetjp_3118_:
{
lean_object* v___x_3122_; 
if (v_isShared_3120_ == 0)
{
v___x_3122_ = v___x_3119_;
goto v_reusejp_3121_;
}
else
{
lean_object* v_reuseFailAlloc_3123_; 
v_reuseFailAlloc_3123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3123_, 0, v_a_3117_);
v___x_3122_ = v_reuseFailAlloc_3123_;
goto v_reusejp_3121_;
}
v_reusejp_3121_:
{
return v___x_3122_;
}
}
}
}
}
else
{
lean_object* v___x_3125_; lean_object* v___x_3126_; 
lean_dec_ref(v___x_2473_);
v___x_3125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem___lam__0___closed__27));
v___x_3126_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2452_, v___x_3125_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3126_) == 0)
{
uint8_t v___x_3127_; 
lean_dec_ref_known(v___x_3126_, 1);
v___x_3127_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2474_);
if (v___x_3127_ == 0)
{
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_item_2461_ = v___x_2474_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
else
{
lean_object* v___x_3128_; 
lean_dec_ref(v___x_2474_);
v___x_3128_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_);
if (lean_obj_tag(v___x_3128_) == 0)
{
lean_object* v_a_3129_; lean_object* v___x_3131_; uint8_t v_isShared_3132_; uint8_t v_isSharedCheck_3156_; 
v_a_3129_ = lean_ctor_get(v___x_3128_, 0);
v_isSharedCheck_3156_ = !lean_is_exclusive(v___x_3128_);
if (v_isSharedCheck_3156_ == 0)
{
v___x_3131_ = v___x_3128_;
v_isShared_3132_ = v_isSharedCheck_3156_;
goto v_resetjp_3130_;
}
else
{
lean_inc(v_a_3129_);
lean_dec(v___x_3128_);
v___x_3131_ = lean_box(0);
v_isShared_3132_ = v_isSharedCheck_3156_;
goto v_resetjp_3130_;
}
v_resetjp_3130_:
{
uint8_t v_closePre_3133_; uint8_t v_closePost_3134_; uint8_t v_transparency_3135_; uint8_t v_preTransparency_3136_; uint8_t v_postTransparency_3137_; uint8_t v_preferLHS_3138_; uint8_t v_partialApp_3139_; uint8_t v_sameFun_3140_; lean_object* v_maxArgs_3141_; uint8_t v_typeEqs_3142_; uint8_t v_etaExpand_3143_; uint8_t v_useCongrSimp_3144_; lean_object* v___x_3146_; uint8_t v_isShared_3147_; uint8_t v_isSharedCheck_3155_; 
v_closePre_3133_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1);
v_closePost_3134_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 1);
v_transparency_3135_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 2);
v_preTransparency_3136_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 3);
v_postTransparency_3137_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 4);
v_preferLHS_3138_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 5);
v_partialApp_3139_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 6);
v_sameFun_3140_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 7);
v_maxArgs_3141_ = lean_ctor_get(v_config_2451_, 0);
v_typeEqs_3142_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 8);
v_etaExpand_3143_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 9);
v_useCongrSimp_3144_ = lean_ctor_get_uint8(v_config_2451_, sizeof(void*)*1 + 10);
v_isSharedCheck_3155_ = !lean_is_exclusive(v_config_2451_);
if (v_isSharedCheck_3155_ == 0)
{
v___x_3146_ = v_config_2451_;
v_isShared_3147_ = v_isSharedCheck_3155_;
goto v_resetjp_3145_;
}
else
{
lean_inc(v_maxArgs_3141_);
lean_dec(v_config_2451_);
v___x_3146_ = lean_box(0);
v_isShared_3147_ = v_isSharedCheck_3155_;
goto v_resetjp_3145_;
}
v_resetjp_3145_:
{
lean_object* v___x_3149_; 
if (v_isShared_3147_ == 0)
{
v___x_3149_ = v___x_3146_;
goto v_reusejp_3148_;
}
else
{
lean_object* v_reuseFailAlloc_3154_; 
v_reuseFailAlloc_3154_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v_reuseFailAlloc_3154_, 0, v_maxArgs_3141_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1, v_closePre_3133_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 1, v_closePost_3134_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 2, v_transparency_3135_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 3, v_preTransparency_3136_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 4, v_postTransparency_3137_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 5, v_preferLHS_3138_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 6, v_partialApp_3139_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 7, v_sameFun_3140_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 8, v_typeEqs_3142_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 9, v_etaExpand_3143_);
lean_ctor_set_uint8(v_reuseFailAlloc_3154_, sizeof(void*)*1 + 10, v_useCongrSimp_3144_);
v___x_3149_ = v_reuseFailAlloc_3154_;
goto v_reusejp_3148_;
}
v_reusejp_3148_:
{
uint8_t v___x_3150_; lean_object* v___x_3152_; 
v___x_3150_ = lean_unbox(v_a_3129_);
lean_dec(v_a_3129_);
lean_ctor_set_uint8(v___x_3149_, sizeof(void*)*1 + 11, v___x_3150_);
if (v_isShared_3132_ == 0)
{
lean_ctor_set(v___x_3131_, 0, v___x_3149_);
v___x_3152_ = v___x_3131_;
goto v_reusejp_3151_;
}
else
{
lean_object* v_reuseFailAlloc_3153_; 
v_reuseFailAlloc_3153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3153_, 0, v___x_3149_);
v___x_3152_ = v_reuseFailAlloc_3153_;
goto v_reusejp_3151_;
}
v_reusejp_3151_:
{
return v___x_3152_;
}
}
}
}
}
else
{
lean_object* v_a_3157_; lean_object* v___x_3159_; uint8_t v_isShared_3160_; uint8_t v_isSharedCheck_3164_; 
lean_dec_ref(v_config_2451_);
v_a_3157_ = lean_ctor_get(v___x_3128_, 0);
v_isSharedCheck_3164_ = !lean_is_exclusive(v___x_3128_);
if (v_isSharedCheck_3164_ == 0)
{
v___x_3159_ = v___x_3128_;
v_isShared_3160_ = v_isSharedCheck_3164_;
goto v_resetjp_3158_;
}
else
{
lean_inc(v_a_3157_);
lean_dec(v___x_3128_);
v___x_3159_ = lean_box(0);
v_isShared_3160_ = v_isSharedCheck_3164_;
goto v_resetjp_3158_;
}
v_resetjp_3158_:
{
lean_object* v___x_3162_; 
if (v_isShared_3160_ == 0)
{
v___x_3162_ = v___x_3159_;
goto v_reusejp_3161_;
}
else
{
lean_object* v_reuseFailAlloc_3163_; 
v_reuseFailAlloc_3163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3163_, 0, v_a_3157_);
v___x_3162_ = v_reuseFailAlloc_3163_;
goto v_reusejp_3161_;
}
v_reusejp_3161_:
{
return v___x_3162_;
}
}
}
}
}
else
{
lean_object* v_a_3165_; lean_object* v___x_3167_; uint8_t v_isShared_3168_; uint8_t v_isSharedCheck_3172_; 
lean_dec_ref(v___x_2474_);
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_3165_ = lean_ctor_get(v___x_3126_, 0);
v_isSharedCheck_3172_ = !lean_is_exclusive(v___x_3126_);
if (v_isSharedCheck_3172_ == 0)
{
v___x_3167_ = v___x_3126_;
v_isShared_3168_ = v_isSharedCheck_3172_;
goto v_resetjp_3166_;
}
else
{
lean_inc(v_a_3165_);
lean_dec(v___x_3126_);
v___x_3167_ = lean_box(0);
v_isShared_3168_ = v_isSharedCheck_3172_;
goto v_resetjp_3166_;
}
v_resetjp_3166_:
{
lean_object* v___x_3170_; 
if (v_isShared_3168_ == 0)
{
v___x_3170_ = v___x_3167_;
goto v_reusejp_3169_;
}
else
{
lean_object* v_reuseFailAlloc_3171_; 
v_reuseFailAlloc_3171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3171_, 0, v_a_3165_);
v___x_3170_ = v_reuseFailAlloc_3171_;
goto v_reusejp_3169_;
}
v_reusejp_3169_:
{
return v___x_3170_;
}
}
}
}
}
}
}
else
{
lean_dec_ref(v_config_2451_);
v_item_2461_ = v_item_2452_;
v___y_2462_ = v___y_2453_;
v___y_2463_ = v___y_2454_;
v___y_2464_ = v___y_2455_;
v___y_2465_ = v___y_2456_;
v___y_2466_ = v___y_2457_;
v___y_2467_ = v___y_2458_;
goto v___jp_2460_;
}
}
else
{
lean_object* v_a_3173_; lean_object* v___x_3175_; uint8_t v_isShared_3176_; uint8_t v_isSharedCheck_3180_; 
lean_dec_ref(v_item_2452_);
lean_dec_ref(v_config_2451_);
v_a_3173_ = lean_ctor_get(v___x_2471_, 0);
v_isSharedCheck_3180_ = !lean_is_exclusive(v___x_2471_);
if (v_isSharedCheck_3180_ == 0)
{
v___x_3175_ = v___x_2471_;
v_isShared_3176_ = v_isSharedCheck_3180_;
goto v_resetjp_3174_;
}
else
{
lean_inc(v_a_3173_);
lean_dec(v___x_2471_);
v___x_3175_ = lean_box(0);
v_isShared_3176_ = v_isSharedCheck_3180_;
goto v_resetjp_3174_;
}
v_resetjp_3174_:
{
lean_object* v___x_3178_; 
if (v_isShared_3176_ == 0)
{
v___x_3178_ = v___x_3175_;
goto v_reusejp_3177_;
}
else
{
lean_object* v_reuseFailAlloc_3179_; 
v_reuseFailAlloc_3179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3179_, 0, v_a_3173_);
v___x_3178_ = v_reuseFailAlloc_3179_;
goto v_reusejp_3177_;
}
v_reusejp_3177_:
{
return v___x_3178_;
}
}
}
v___jp_2460_:
{
lean_object* v___x_2468_; lean_object* v___x_2469_; 
v___x_2468_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___closed__0));
v___x_2469_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_2461_, v___x_2468_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_, v___y_2466_, v___y_2467_);
return v___x_2469_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_3181_, lean_object* v_item_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_){
_start:
{
lean_object* v_res_3190_; 
v_res_3190_ = lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___lam__0(v_config_3181_, v_item_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec(v___y_3184_);
lean_dec_ref(v___y_3183_);
return v_res_3190_;
}
}
static lean_object* _init_lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; 
v___x_3193_ = lean_box(0);
v___x_3194_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig_evalExpr___closed__2));
v___x_3195_ = l_Lean_mkConst(v___x_3194_, v___x_3193_);
return v___x_3195_;
}
}
static lean_object* _init_lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3196_; lean_object* v___x_3197_; 
v___x_3196_ = lean_obj_once(&lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0, &lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__0);
v___x_3197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3197_, 0, v___x_3196_);
return v___x_3197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0(lean_object* v_cfg_3198_, lean_object* v_cfgItem_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_){
_start:
{
lean_object* v___x_3207_; lean_object* v___x_3208_; 
v___x_3207_ = lean_obj_once(&lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1, &lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___closed__1);
v___x_3208_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_3198_, v_cfgItem_3199_, v___x_3207_, v___y_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_, v___y_3205_);
return v___x_3208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0___boxed(lean_object* v_cfg_3209_, lean_object* v_cfgItem_3210_, lean_object* v___y_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_){
_start:
{
lean_object* v_res_3218_; 
v_res_3218_ = lp_mathlib_Convert_elabExpensiveConfig___redArg___lam__0(v_cfg_3209_, v_cfgItem_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
lean_dec(v___y_3216_);
lean_dec_ref(v___y_3215_);
lean_dec(v___y_3214_);
lean_dec_ref(v___y_3213_);
lean_dec(v___y_3212_);
lean_dec_ref(v___y_3211_);
lean_dec(v_cfgItem_3210_);
return v_res_3218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg(lean_object* v_cfg_3220_, lean_object* v_init_3221_, uint8_t v_logExceptions_3222_, lean_object* v_a_3223_, lean_object* v_a_3224_, lean_object* v_a_3225_){
_start:
{
lean_object* v_onErr_3227_; lean_object* v_eval_3228_; 
v_onErr_3227_ = ((lean_object*)(lp_mathlib_Convert_elabExpensiveConfig___redArg___closed__0));
v_eval_3228_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Convert_0__Convert_elabExpensiveConfig_evalConfigItem___closed__0));
if (v_logExceptions_3222_ == 0)
{
lean_object* v___x_3229_; 
v___x_3229_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3228_, v_init_3221_, v_cfg_3220_, v_onErr_3227_, v_logExceptions_3222_, v_a_3224_, v_a_3225_);
return v___x_3229_;
}
else
{
uint8_t v_recover_3230_; lean_object* v___x_3231_; 
v_recover_3230_ = lean_ctor_get_uint8(v_a_3223_, sizeof(void*)*1);
v___x_3231_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3228_, v_init_3221_, v_cfg_3220_, v_onErr_3227_, v_recover_3230_, v_a_3224_, v_a_3225_);
return v___x_3231_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___redArg___boxed(lean_object* v_cfg_3232_, lean_object* v_init_3233_, lean_object* v_logExceptions_3234_, lean_object* v_a_3235_, lean_object* v_a_3236_, lean_object* v_a_3237_, lean_object* v_a_3238_){
_start:
{
uint8_t v_logExceptions_boxed_3239_; lean_object* v_res_3240_; 
v_logExceptions_boxed_3239_ = lean_unbox(v_logExceptions_3234_);
v_res_3240_ = lp_mathlib_Convert_elabExpensiveConfig___redArg(v_cfg_3232_, v_init_3233_, v_logExceptions_boxed_3239_, v_a_3235_, v_a_3236_, v_a_3237_);
lean_dec(v_a_3237_);
lean_dec_ref(v_a_3236_);
lean_dec_ref(v_a_3235_);
return v_res_3240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig(lean_object* v_cfg_3241_, lean_object* v_init_3242_, uint8_t v_logExceptions_3243_, lean_object* v_a_3244_, lean_object* v_a_3245_, lean_object* v_a_3246_, lean_object* v_a_3247_, lean_object* v_a_3248_, lean_object* v_a_3249_, lean_object* v_a_3250_, lean_object* v_a_3251_){
_start:
{
lean_object* v___x_3253_; 
v___x_3253_ = lp_mathlib_Convert_elabExpensiveConfig___redArg(v_cfg_3241_, v_init_3242_, v_logExceptions_3243_, v_a_3244_, v_a_3250_, v_a_3251_);
return v___x_3253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabExpensiveConfig___boxed(lean_object* v_cfg_3254_, lean_object* v_init_3255_, lean_object* v_logExceptions_3256_, lean_object* v_a_3257_, lean_object* v_a_3258_, lean_object* v_a_3259_, lean_object* v_a_3260_, lean_object* v_a_3261_, lean_object* v_a_3262_, lean_object* v_a_3263_, lean_object* v_a_3264_, lean_object* v_a_3265_){
_start:
{
uint8_t v_logExceptions_boxed_3266_; lean_object* v_res_3267_; 
v_logExceptions_boxed_3266_ = lean_unbox(v_logExceptions_3256_);
v_res_3267_ = lp_mathlib_Convert_elabExpensiveConfig(v_cfg_3254_, v_init_3255_, v_logExceptions_boxed_3266_, v_a_3257_, v_a_3258_, v_a_3259_, v_a_3260_, v_a_3261_, v_a_3262_, v_a_3263_, v_a_3264_);
lean_dec(v_a_3264_);
lean_dec_ref(v_a_3263_);
lean_dec(v_a_3262_);
lean_dec_ref(v_a_3261_);
lean_dec(v_a_3260_);
lean_dec_ref(v_a_3259_);
lean_dec(v_a_3258_);
lean_dec_ref(v_a_3257_);
return v_res_3267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___redArg(uint8_t v_expensive_3275_, lean_object* v_stx_3276_, lean_object* v_a_3277_, lean_object* v_a_3278_, lean_object* v_a_3279_){
_start:
{
uint8_t v___x_3281_; 
v___x_3281_ = 1;
if (v_expensive_3275_ == 0)
{
uint8_t v___x_3282_; uint8_t v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3285_; lean_object* v___x_3286_; 
v___x_3282_ = 2;
v___x_3283_ = 3;
v___x_3284_ = lean_box(0);
v___x_3285_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v___x_3285_, 0, v___x_3284_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1, v___x_3281_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 1, v___x_3281_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 2, v___x_3282_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 3, v___x_3283_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 4, v___x_3282_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 5, v___x_3281_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 6, v___x_3281_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 7, v___x_3281_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 8, v_expensive_3275_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 9, v_expensive_3275_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 10, v_expensive_3275_);
lean_ctor_set_uint8(v___x_3285_, sizeof(void*)*1 + 11, v___x_3281_);
v___x_3286_ = lp_mathlib_Convert_elabCheapConfig___redArg(v_stx_3276_, v___x_3285_, v___x_3281_, v_a_3277_, v_a_3278_, v_a_3279_);
if (lean_obj_tag(v___x_3286_) == 0)
{
lean_object* v_a_3287_; lean_object* v___x_3289_; uint8_t v_isShared_3290_; uint8_t v_isSharedCheck_3294_; 
v_a_3287_ = lean_ctor_get(v___x_3286_, 0);
v_isSharedCheck_3294_ = !lean_is_exclusive(v___x_3286_);
if (v_isSharedCheck_3294_ == 0)
{
v___x_3289_ = v___x_3286_;
v_isShared_3290_ = v_isSharedCheck_3294_;
goto v_resetjp_3288_;
}
else
{
lean_inc(v_a_3287_);
lean_dec(v___x_3286_);
v___x_3289_ = lean_box(0);
v_isShared_3290_ = v_isSharedCheck_3294_;
goto v_resetjp_3288_;
}
v_resetjp_3288_:
{
lean_object* v___x_3292_; 
if (v_isShared_3290_ == 0)
{
v___x_3292_ = v___x_3289_;
goto v_reusejp_3291_;
}
else
{
lean_object* v_reuseFailAlloc_3293_; 
v_reuseFailAlloc_3293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3293_, 0, v_a_3287_);
v___x_3292_ = v_reuseFailAlloc_3293_;
goto v_reusejp_3291_;
}
v_reusejp_3291_:
{
return v___x_3292_;
}
}
}
else
{
lean_object* v_a_3295_; lean_object* v___x_3297_; uint8_t v_isShared_3298_; uint8_t v_isSharedCheck_3302_; 
v_a_3295_ = lean_ctor_get(v___x_3286_, 0);
v_isSharedCheck_3302_ = !lean_is_exclusive(v___x_3286_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3297_ = v___x_3286_;
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
else
{
lean_inc(v_a_3295_);
lean_dec(v___x_3286_);
v___x_3297_ = lean_box(0);
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
v_resetjp_3296_:
{
lean_object* v___x_3300_; 
if (v_isShared_3298_ == 0)
{
v___x_3300_ = v___x_3297_;
goto v_reusejp_3299_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v_a_3295_);
v___x_3300_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3299_;
}
v_reusejp_3299_:
{
return v___x_3300_;
}
}
}
}
else
{
lean_object* v___x_3303_; lean_object* v___x_3304_; 
v___x_3303_ = ((lean_object*)(lp_mathlib_Convert_elabConfig___redArg___closed__0));
v___x_3304_ = lp_mathlib_Convert_elabExpensiveConfig___redArg(v_stx_3276_, v___x_3303_, v___x_3281_, v_a_3277_, v_a_3278_, v_a_3279_);
if (lean_obj_tag(v___x_3304_) == 0)
{
lean_object* v_a_3305_; lean_object* v___x_3307_; uint8_t v_isShared_3308_; uint8_t v_isSharedCheck_3312_; 
v_a_3305_ = lean_ctor_get(v___x_3304_, 0);
v_isSharedCheck_3312_ = !lean_is_exclusive(v___x_3304_);
if (v_isSharedCheck_3312_ == 0)
{
v___x_3307_ = v___x_3304_;
v_isShared_3308_ = v_isSharedCheck_3312_;
goto v_resetjp_3306_;
}
else
{
lean_inc(v_a_3305_);
lean_dec(v___x_3304_);
v___x_3307_ = lean_box(0);
v_isShared_3308_ = v_isSharedCheck_3312_;
goto v_resetjp_3306_;
}
v_resetjp_3306_:
{
lean_object* v___x_3310_; 
if (v_isShared_3308_ == 0)
{
v___x_3310_ = v___x_3307_;
goto v_reusejp_3309_;
}
else
{
lean_object* v_reuseFailAlloc_3311_; 
v_reuseFailAlloc_3311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3311_, 0, v_a_3305_);
v___x_3310_ = v_reuseFailAlloc_3311_;
goto v_reusejp_3309_;
}
v_reusejp_3309_:
{
return v___x_3310_;
}
}
}
else
{
lean_object* v_a_3313_; lean_object* v___x_3315_; uint8_t v_isShared_3316_; uint8_t v_isSharedCheck_3320_; 
v_a_3313_ = lean_ctor_get(v___x_3304_, 0);
v_isSharedCheck_3320_ = !lean_is_exclusive(v___x_3304_);
if (v_isSharedCheck_3320_ == 0)
{
v___x_3315_ = v___x_3304_;
v_isShared_3316_ = v_isSharedCheck_3320_;
goto v_resetjp_3314_;
}
else
{
lean_inc(v_a_3313_);
lean_dec(v___x_3304_);
v___x_3315_ = lean_box(0);
v_isShared_3316_ = v_isSharedCheck_3320_;
goto v_resetjp_3314_;
}
v_resetjp_3314_:
{
lean_object* v___x_3318_; 
if (v_isShared_3316_ == 0)
{
v___x_3318_ = v___x_3315_;
goto v_reusejp_3317_;
}
else
{
lean_object* v_reuseFailAlloc_3319_; 
v_reuseFailAlloc_3319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3319_, 0, v_a_3313_);
v___x_3318_ = v_reuseFailAlloc_3319_;
goto v_reusejp_3317_;
}
v_reusejp_3317_:
{
return v___x_3318_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___redArg___boxed(lean_object* v_expensive_3321_, lean_object* v_stx_3322_, lean_object* v_a_3323_, lean_object* v_a_3324_, lean_object* v_a_3325_, lean_object* v_a_3326_){
_start:
{
uint8_t v_expensive_boxed_3327_; lean_object* v_res_3328_; 
v_expensive_boxed_3327_ = lean_unbox(v_expensive_3321_);
v_res_3328_ = lp_mathlib_Convert_elabConfig___redArg(v_expensive_boxed_3327_, v_stx_3322_, v_a_3323_, v_a_3324_, v_a_3325_);
lean_dec(v_a_3325_);
lean_dec_ref(v_a_3324_);
lean_dec_ref(v_a_3323_);
return v_res_3328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig(uint8_t v_expensive_3329_, lean_object* v_stx_3330_, lean_object* v_a_3331_, lean_object* v_a_3332_, lean_object* v_a_3333_, lean_object* v_a_3334_, lean_object* v_a_3335_, lean_object* v_a_3336_, lean_object* v_a_3337_, lean_object* v_a_3338_){
_start:
{
lean_object* v___x_3340_; 
v___x_3340_ = lp_mathlib_Convert_elabConfig___redArg(v_expensive_3329_, v_stx_3330_, v_a_3331_, v_a_3337_, v_a_3338_);
return v___x_3340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Convert_elabConfig___boxed(lean_object* v_expensive_3341_, lean_object* v_stx_3342_, lean_object* v_a_3343_, lean_object* v_a_3344_, lean_object* v_a_3345_, lean_object* v_a_3346_, lean_object* v_a_3347_, lean_object* v_a_3348_, lean_object* v_a_3349_, lean_object* v_a_3350_, lean_object* v_a_3351_){
_start:
{
uint8_t v_expensive_boxed_3352_; lean_object* v_res_3353_; 
v_expensive_boxed_3352_ = lean_unbox(v_expensive_3341_);
v_res_3353_ = lp_mathlib_Convert_elabConfig(v_expensive_boxed_3352_, v_stx_3342_, v_a_3343_, v_a_3344_, v_a_3345_, v_a_3346_, v_a_3347_, v_a_3348_, v_a_3349_, v_a_3350_);
lean_dec(v_a_3350_);
lean_dec_ref(v_a_3349_);
lean_dec(v_a_3348_);
lean_dec_ref(v_a_3347_);
lean_dec(v_a_3346_);
lean_dec_ref(v_a_3345_);
lean_dec(v_a_3344_);
lean_dec_ref(v_a_3343_);
return v_res_3353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(lean_object* v_mvarId_3354_, lean_object* v_x_3355_, lean_object* v___y_3356_, lean_object* v___y_3357_, lean_object* v___y_3358_, lean_object* v___y_3359_){
_start:
{
lean_object* v___x_3361_; 
v___x_3361_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_3354_, v_x_3355_, v___y_3356_, v___y_3357_, v___y_3358_, v___y_3359_);
if (lean_obj_tag(v___x_3361_) == 0)
{
lean_object* v_a_3362_; lean_object* v___x_3364_; uint8_t v_isShared_3365_; uint8_t v_isSharedCheck_3369_; 
v_a_3362_ = lean_ctor_get(v___x_3361_, 0);
v_isSharedCheck_3369_ = !lean_is_exclusive(v___x_3361_);
if (v_isSharedCheck_3369_ == 0)
{
v___x_3364_ = v___x_3361_;
v_isShared_3365_ = v_isSharedCheck_3369_;
goto v_resetjp_3363_;
}
else
{
lean_inc(v_a_3362_);
lean_dec(v___x_3361_);
v___x_3364_ = lean_box(0);
v_isShared_3365_ = v_isSharedCheck_3369_;
goto v_resetjp_3363_;
}
v_resetjp_3363_:
{
lean_object* v___x_3367_; 
if (v_isShared_3365_ == 0)
{
v___x_3367_ = v___x_3364_;
goto v_reusejp_3366_;
}
else
{
lean_object* v_reuseFailAlloc_3368_; 
v_reuseFailAlloc_3368_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3368_, 0, v_a_3362_);
v___x_3367_ = v_reuseFailAlloc_3368_;
goto v_reusejp_3366_;
}
v_reusejp_3366_:
{
return v___x_3367_;
}
}
}
else
{
lean_object* v_a_3370_; lean_object* v___x_3372_; uint8_t v_isShared_3373_; uint8_t v_isSharedCheck_3377_; 
v_a_3370_ = lean_ctor_get(v___x_3361_, 0);
v_isSharedCheck_3377_ = !lean_is_exclusive(v___x_3361_);
if (v_isSharedCheck_3377_ == 0)
{
v___x_3372_ = v___x_3361_;
v_isShared_3373_ = v_isSharedCheck_3377_;
goto v_resetjp_3371_;
}
else
{
lean_inc(v_a_3370_);
lean_dec(v___x_3361_);
v___x_3372_ = lean_box(0);
v_isShared_3373_ = v_isSharedCheck_3377_;
goto v_resetjp_3371_;
}
v_resetjp_3371_:
{
lean_object* v___x_3375_; 
if (v_isShared_3373_ == 0)
{
v___x_3375_ = v___x_3372_;
goto v_reusejp_3374_;
}
else
{
lean_object* v_reuseFailAlloc_3376_; 
v_reuseFailAlloc_3376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3376_, 0, v_a_3370_);
v___x_3375_ = v_reuseFailAlloc_3376_;
goto v_reusejp_3374_;
}
v_reusejp_3374_:
{
return v___x_3375_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg___boxed(lean_object* v_mvarId_3378_, lean_object* v_x_3379_, lean_object* v___y_3380_, lean_object* v___y_3381_, lean_object* v___y_3382_, lean_object* v___y_3383_, lean_object* v___y_3384_){
_start:
{
lean_object* v_res_3385_; 
v_res_3385_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(v_mvarId_3378_, v_x_3379_, v___y_3380_, v___y_3381_, v___y_3382_, v___y_3383_);
lean_dec(v___y_3383_);
lean_dec_ref(v___y_3382_);
lean_dec(v___y_3381_);
lean_dec_ref(v___y_3380_);
return v_res_3385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1(lean_object* v_00_u03b1_3386_, lean_object* v_mvarId_3387_, lean_object* v_x_3388_, lean_object* v___y_3389_, lean_object* v___y_3390_, lean_object* v___y_3391_, lean_object* v___y_3392_){
_start:
{
lean_object* v___x_3394_; 
v___x_3394_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(v_mvarId_3387_, v_x_3388_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_);
return v___x_3394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___boxed(lean_object* v_00_u03b1_3395_, lean_object* v_mvarId_3396_, lean_object* v_x_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_){
_start:
{
lean_object* v_res_3403_; 
v_res_3403_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1(v_00_u03b1_3395_, v_mvarId_3396_, v_x_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_);
lean_dec(v___y_3401_);
lean_dec_ref(v___y_3400_);
lean_dec(v___y_3399_);
lean_dec_ref(v___y_3398_);
return v_res_3403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(lean_object* v_x_3404_, lean_object* v_x_3405_, lean_object* v_x_3406_, lean_object* v_x_3407_){
_start:
{
lean_object* v_ks_3408_; lean_object* v_vs_3409_; lean_object* v___x_3411_; uint8_t v_isShared_3412_; uint8_t v_isSharedCheck_3433_; 
v_ks_3408_ = lean_ctor_get(v_x_3404_, 0);
v_vs_3409_ = lean_ctor_get(v_x_3404_, 1);
v_isSharedCheck_3433_ = !lean_is_exclusive(v_x_3404_);
if (v_isSharedCheck_3433_ == 0)
{
v___x_3411_ = v_x_3404_;
v_isShared_3412_ = v_isSharedCheck_3433_;
goto v_resetjp_3410_;
}
else
{
lean_inc(v_vs_3409_);
lean_inc(v_ks_3408_);
lean_dec(v_x_3404_);
v___x_3411_ = lean_box(0);
v_isShared_3412_ = v_isSharedCheck_3433_;
goto v_resetjp_3410_;
}
v_resetjp_3410_:
{
lean_object* v___x_3413_; uint8_t v___x_3414_; 
v___x_3413_ = lean_array_get_size(v_ks_3408_);
v___x_3414_ = lean_nat_dec_lt(v_x_3405_, v___x_3413_);
if (v___x_3414_ == 0)
{
lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v___x_3418_; 
lean_dec(v_x_3405_);
v___x_3415_ = lean_array_push(v_ks_3408_, v_x_3406_);
v___x_3416_ = lean_array_push(v_vs_3409_, v_x_3407_);
if (v_isShared_3412_ == 0)
{
lean_ctor_set(v___x_3411_, 1, v___x_3416_);
lean_ctor_set(v___x_3411_, 0, v___x_3415_);
v___x_3418_ = v___x_3411_;
goto v_reusejp_3417_;
}
else
{
lean_object* v_reuseFailAlloc_3419_; 
v_reuseFailAlloc_3419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3419_, 0, v___x_3415_);
lean_ctor_set(v_reuseFailAlloc_3419_, 1, v___x_3416_);
v___x_3418_ = v_reuseFailAlloc_3419_;
goto v_reusejp_3417_;
}
v_reusejp_3417_:
{
return v___x_3418_;
}
}
else
{
lean_object* v_k_x27_3420_; uint8_t v___x_3421_; 
v_k_x27_3420_ = lean_array_fget_borrowed(v_ks_3408_, v_x_3405_);
v___x_3421_ = l_Lean_instBEqMVarId_beq(v_x_3406_, v_k_x27_3420_);
if (v___x_3421_ == 0)
{
lean_object* v___x_3423_; 
if (v_isShared_3412_ == 0)
{
v___x_3423_ = v___x_3411_;
goto v_reusejp_3422_;
}
else
{
lean_object* v_reuseFailAlloc_3427_; 
v_reuseFailAlloc_3427_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3427_, 0, v_ks_3408_);
lean_ctor_set(v_reuseFailAlloc_3427_, 1, v_vs_3409_);
v___x_3423_ = v_reuseFailAlloc_3427_;
goto v_reusejp_3422_;
}
v_reusejp_3422_:
{
lean_object* v___x_3424_; lean_object* v___x_3425_; 
v___x_3424_ = lean_unsigned_to_nat(1u);
v___x_3425_ = lean_nat_add(v_x_3405_, v___x_3424_);
lean_dec(v_x_3405_);
v_x_3404_ = v___x_3423_;
v_x_3405_ = v___x_3425_;
goto _start;
}
}
else
{
lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3431_; 
v___x_3428_ = lean_array_fset(v_ks_3408_, v_x_3405_, v_x_3406_);
v___x_3429_ = lean_array_fset(v_vs_3409_, v_x_3405_, v_x_3407_);
lean_dec(v_x_3405_);
if (v_isShared_3412_ == 0)
{
lean_ctor_set(v___x_3411_, 1, v___x_3429_);
lean_ctor_set(v___x_3411_, 0, v___x_3428_);
v___x_3431_ = v___x_3411_;
goto v_reusejp_3430_;
}
else
{
lean_object* v_reuseFailAlloc_3432_; 
v_reuseFailAlloc_3432_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3432_, 0, v___x_3428_);
lean_ctor_set(v_reuseFailAlloc_3432_, 1, v___x_3429_);
v___x_3431_ = v_reuseFailAlloc_3432_;
goto v_reusejp_3430_;
}
v_reusejp_3430_:
{
return v___x_3431_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_n_3434_, lean_object* v_k_3435_, lean_object* v_v_3436_){
_start:
{
lean_object* v___x_3437_; lean_object* v___x_3438_; 
v___x_3437_ = lean_unsigned_to_nat(0u);
v___x_3438_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(v_n_3434_, v___x_3437_, v_k_3435_, v_v_3436_);
return v___x_3438_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_3439_; 
v___x_3439_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_3439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(lean_object* v_x_3440_, size_t v_x_3441_, size_t v_x_3442_, lean_object* v_x_3443_, lean_object* v_x_3444_){
_start:
{
if (lean_obj_tag(v_x_3440_) == 0)
{
lean_object* v_es_3445_; size_t v___x_3446_; size_t v___x_3447_; lean_object* v_j_3448_; lean_object* v___x_3449_; uint8_t v___x_3450_; 
v_es_3445_ = lean_ctor_get(v_x_3440_, 0);
v___x_3446_ = ((size_t)31ULL);
v___x_3447_ = lean_usize_land(v_x_3441_, v___x_3446_);
v_j_3448_ = lean_usize_to_nat(v___x_3447_);
v___x_3449_ = lean_array_get_size(v_es_3445_);
v___x_3450_ = lean_nat_dec_lt(v_j_3448_, v___x_3449_);
if (v___x_3450_ == 0)
{
lean_dec(v_j_3448_);
lean_dec(v_x_3444_);
lean_dec(v_x_3443_);
return v_x_3440_;
}
else
{
lean_object* v___x_3452_; uint8_t v_isShared_3453_; uint8_t v_isSharedCheck_3489_; 
lean_inc_ref(v_es_3445_);
v_isSharedCheck_3489_ = !lean_is_exclusive(v_x_3440_);
if (v_isSharedCheck_3489_ == 0)
{
lean_object* v_unused_3490_; 
v_unused_3490_ = lean_ctor_get(v_x_3440_, 0);
lean_dec(v_unused_3490_);
v___x_3452_ = v_x_3440_;
v_isShared_3453_ = v_isSharedCheck_3489_;
goto v_resetjp_3451_;
}
else
{
lean_dec(v_x_3440_);
v___x_3452_ = lean_box(0);
v_isShared_3453_ = v_isSharedCheck_3489_;
goto v_resetjp_3451_;
}
v_resetjp_3451_:
{
lean_object* v_v_3454_; lean_object* v___x_3455_; lean_object* v_xs_x27_3456_; lean_object* v___y_3458_; 
v_v_3454_ = lean_array_fget(v_es_3445_, v_j_3448_);
v___x_3455_ = lean_box(0);
v_xs_x27_3456_ = lean_array_fset(v_es_3445_, v_j_3448_, v___x_3455_);
switch(lean_obj_tag(v_v_3454_))
{
case 0:
{
lean_object* v_key_3463_; lean_object* v_val_3464_; lean_object* v___x_3466_; uint8_t v_isShared_3467_; uint8_t v_isSharedCheck_3474_; 
v_key_3463_ = lean_ctor_get(v_v_3454_, 0);
v_val_3464_ = lean_ctor_get(v_v_3454_, 1);
v_isSharedCheck_3474_ = !lean_is_exclusive(v_v_3454_);
if (v_isSharedCheck_3474_ == 0)
{
v___x_3466_ = v_v_3454_;
v_isShared_3467_ = v_isSharedCheck_3474_;
goto v_resetjp_3465_;
}
else
{
lean_inc(v_val_3464_);
lean_inc(v_key_3463_);
lean_dec(v_v_3454_);
v___x_3466_ = lean_box(0);
v_isShared_3467_ = v_isSharedCheck_3474_;
goto v_resetjp_3465_;
}
v_resetjp_3465_:
{
uint8_t v___x_3468_; 
v___x_3468_ = l_Lean_instBEqMVarId_beq(v_x_3443_, v_key_3463_);
if (v___x_3468_ == 0)
{
lean_object* v___x_3469_; lean_object* v___x_3470_; 
lean_del_object(v___x_3466_);
v___x_3469_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_3463_, v_val_3464_, v_x_3443_, v_x_3444_);
v___x_3470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3470_, 0, v___x_3469_);
v___y_3458_ = v___x_3470_;
goto v___jp_3457_;
}
else
{
lean_object* v___x_3472_; 
lean_dec(v_val_3464_);
lean_dec(v_key_3463_);
if (v_isShared_3467_ == 0)
{
lean_ctor_set(v___x_3466_, 1, v_x_3444_);
lean_ctor_set(v___x_3466_, 0, v_x_3443_);
v___x_3472_ = v___x_3466_;
goto v_reusejp_3471_;
}
else
{
lean_object* v_reuseFailAlloc_3473_; 
v_reuseFailAlloc_3473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3473_, 0, v_x_3443_);
lean_ctor_set(v_reuseFailAlloc_3473_, 1, v_x_3444_);
v___x_3472_ = v_reuseFailAlloc_3473_;
goto v_reusejp_3471_;
}
v_reusejp_3471_:
{
v___y_3458_ = v___x_3472_;
goto v___jp_3457_;
}
}
}
}
case 1:
{
lean_object* v_node_3475_; lean_object* v___x_3477_; uint8_t v_isShared_3478_; uint8_t v_isSharedCheck_3487_; 
v_node_3475_ = lean_ctor_get(v_v_3454_, 0);
v_isSharedCheck_3487_ = !lean_is_exclusive(v_v_3454_);
if (v_isSharedCheck_3487_ == 0)
{
v___x_3477_ = v_v_3454_;
v_isShared_3478_ = v_isSharedCheck_3487_;
goto v_resetjp_3476_;
}
else
{
lean_inc(v_node_3475_);
lean_dec(v_v_3454_);
v___x_3477_ = lean_box(0);
v_isShared_3478_ = v_isSharedCheck_3487_;
goto v_resetjp_3476_;
}
v_resetjp_3476_:
{
size_t v___x_3479_; size_t v___x_3480_; size_t v___x_3481_; size_t v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3485_; 
v___x_3479_ = ((size_t)5ULL);
v___x_3480_ = lean_usize_shift_right(v_x_3441_, v___x_3479_);
v___x_3481_ = ((size_t)1ULL);
v___x_3482_ = lean_usize_add(v_x_3442_, v___x_3481_);
v___x_3483_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(v_node_3475_, v___x_3480_, v___x_3482_, v_x_3443_, v_x_3444_);
if (v_isShared_3478_ == 0)
{
lean_ctor_set(v___x_3477_, 0, v___x_3483_);
v___x_3485_ = v___x_3477_;
goto v_reusejp_3484_;
}
else
{
lean_object* v_reuseFailAlloc_3486_; 
v_reuseFailAlloc_3486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3486_, 0, v___x_3483_);
v___x_3485_ = v_reuseFailAlloc_3486_;
goto v_reusejp_3484_;
}
v_reusejp_3484_:
{
v___y_3458_ = v___x_3485_;
goto v___jp_3457_;
}
}
}
default: 
{
lean_object* v___x_3488_; 
v___x_3488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3488_, 0, v_x_3443_);
lean_ctor_set(v___x_3488_, 1, v_x_3444_);
v___y_3458_ = v___x_3488_;
goto v___jp_3457_;
}
}
v___jp_3457_:
{
lean_object* v___x_3459_; lean_object* v___x_3461_; 
v___x_3459_ = lean_array_fset(v_xs_x27_3456_, v_j_3448_, v___y_3458_);
lean_dec(v_j_3448_);
if (v_isShared_3453_ == 0)
{
lean_ctor_set(v___x_3452_, 0, v___x_3459_);
v___x_3461_ = v___x_3452_;
goto v_reusejp_3460_;
}
else
{
lean_object* v_reuseFailAlloc_3462_; 
v_reuseFailAlloc_3462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3462_, 0, v___x_3459_);
v___x_3461_ = v_reuseFailAlloc_3462_;
goto v_reusejp_3460_;
}
v_reusejp_3460_:
{
return v___x_3461_;
}
}
}
}
}
else
{
lean_object* v_ks_3491_; lean_object* v_vs_3492_; lean_object* v___x_3494_; uint8_t v_isShared_3495_; uint8_t v_isSharedCheck_3512_; 
v_ks_3491_ = lean_ctor_get(v_x_3440_, 0);
v_vs_3492_ = lean_ctor_get(v_x_3440_, 1);
v_isSharedCheck_3512_ = !lean_is_exclusive(v_x_3440_);
if (v_isSharedCheck_3512_ == 0)
{
v___x_3494_ = v_x_3440_;
v_isShared_3495_ = v_isSharedCheck_3512_;
goto v_resetjp_3493_;
}
else
{
lean_inc(v_vs_3492_);
lean_inc(v_ks_3491_);
lean_dec(v_x_3440_);
v___x_3494_ = lean_box(0);
v_isShared_3495_ = v_isSharedCheck_3512_;
goto v_resetjp_3493_;
}
v_resetjp_3493_:
{
lean_object* v___x_3497_; 
if (v_isShared_3495_ == 0)
{
v___x_3497_ = v___x_3494_;
goto v_reusejp_3496_;
}
else
{
lean_object* v_reuseFailAlloc_3511_; 
v_reuseFailAlloc_3511_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3511_, 0, v_ks_3491_);
lean_ctor_set(v_reuseFailAlloc_3511_, 1, v_vs_3492_);
v___x_3497_ = v_reuseFailAlloc_3511_;
goto v_reusejp_3496_;
}
v_reusejp_3496_:
{
lean_object* v_newNode_3498_; uint8_t v___y_3500_; size_t v___x_3506_; uint8_t v___x_3507_; 
v_newNode_3498_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3___redArg(v___x_3497_, v_x_3443_, v_x_3444_);
v___x_3506_ = ((size_t)7ULL);
v___x_3507_ = lean_usize_dec_le(v___x_3506_, v_x_3442_);
if (v___x_3507_ == 0)
{
lean_object* v___x_3508_; lean_object* v___x_3509_; uint8_t v___x_3510_; 
v___x_3508_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_3498_);
v___x_3509_ = lean_unsigned_to_nat(4u);
v___x_3510_ = lean_nat_dec_lt(v___x_3508_, v___x_3509_);
lean_dec(v___x_3508_);
v___y_3500_ = v___x_3510_;
goto v___jp_3499_;
}
else
{
v___y_3500_ = v___x_3507_;
goto v___jp_3499_;
}
v___jp_3499_:
{
if (v___y_3500_ == 0)
{
lean_object* v_ks_3501_; lean_object* v_vs_3502_; lean_object* v___x_3503_; lean_object* v___x_3504_; lean_object* v___x_3505_; 
v_ks_3501_ = lean_ctor_get(v_newNode_3498_, 0);
lean_inc_ref(v_ks_3501_);
v_vs_3502_ = lean_ctor_get(v_newNode_3498_, 1);
lean_inc_ref(v_vs_3502_);
lean_dec_ref(v_newNode_3498_);
v___x_3503_ = lean_unsigned_to_nat(0u);
v___x_3504_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___closed__0);
v___x_3505_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg(v_x_3442_, v_ks_3501_, v_vs_3502_, v___x_3503_, v___x_3504_);
lean_dec_ref(v_vs_3502_);
lean_dec_ref(v_ks_3501_);
return v___x_3505_;
}
else
{
return v_newNode_3498_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg(size_t v_depth_3513_, lean_object* v_keys_3514_, lean_object* v_vals_3515_, lean_object* v_i_3516_, lean_object* v_entries_3517_){
_start:
{
lean_object* v___x_3518_; uint8_t v___x_3519_; 
v___x_3518_ = lean_array_get_size(v_keys_3514_);
v___x_3519_ = lean_nat_dec_lt(v_i_3516_, v___x_3518_);
if (v___x_3519_ == 0)
{
lean_dec(v_i_3516_);
return v_entries_3517_;
}
else
{
lean_object* v_k_3520_; lean_object* v_v_3521_; uint64_t v___x_3522_; size_t v_h_3523_; size_t v___x_3524_; lean_object* v___x_3525_; size_t v___x_3526_; size_t v___x_3527_; size_t v___x_3528_; size_t v_h_3529_; lean_object* v___x_3530_; lean_object* v___x_3531_; 
v_k_3520_ = lean_array_fget_borrowed(v_keys_3514_, v_i_3516_);
v_v_3521_ = lean_array_fget_borrowed(v_vals_3515_, v_i_3516_);
v___x_3522_ = l_Lean_instHashableMVarId_hash(v_k_3520_);
v_h_3523_ = lean_uint64_to_usize(v___x_3522_);
v___x_3524_ = ((size_t)5ULL);
v___x_3525_ = lean_unsigned_to_nat(1u);
v___x_3526_ = ((size_t)1ULL);
v___x_3527_ = lean_usize_sub(v_depth_3513_, v___x_3526_);
v___x_3528_ = lean_usize_mul(v___x_3524_, v___x_3527_);
v_h_3529_ = lean_usize_shift_right(v_h_3523_, v___x_3528_);
v___x_3530_ = lean_nat_add(v_i_3516_, v___x_3525_);
lean_dec(v_i_3516_);
lean_inc(v_v_3521_);
lean_inc(v_k_3520_);
v___x_3531_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(v_entries_3517_, v_h_3529_, v_depth_3513_, v_k_3520_, v_v_3521_);
v_i_3516_ = v___x_3530_;
v_entries_3517_ = v___x_3531_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object* v_depth_3533_, lean_object* v_keys_3534_, lean_object* v_vals_3535_, lean_object* v_i_3536_, lean_object* v_entries_3537_){
_start:
{
size_t v_depth_boxed_3538_; lean_object* v_res_3539_; 
v_depth_boxed_3538_ = lean_unbox_usize(v_depth_3533_);
lean_dec(v_depth_3533_);
v_res_3539_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg(v_depth_boxed_3538_, v_keys_3534_, v_vals_3535_, v_i_3536_, v_entries_3537_);
lean_dec_ref(v_vals_3535_);
lean_dec_ref(v_keys_3534_);
return v_res_3539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_x_3540_, lean_object* v_x_3541_, lean_object* v_x_3542_, lean_object* v_x_3543_, lean_object* v_x_3544_){
_start:
{
size_t v_x_1554__boxed_3545_; size_t v_x_1555__boxed_3546_; lean_object* v_res_3547_; 
v_x_1554__boxed_3545_ = lean_unbox_usize(v_x_3541_);
lean_dec(v_x_3541_);
v_x_1555__boxed_3546_ = lean_unbox_usize(v_x_3542_);
lean_dec(v_x_3542_);
v_res_3547_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(v_x_3540_, v_x_1554__boxed_3545_, v_x_1555__boxed_3546_, v_x_3543_, v_x_3544_);
return v_res_3547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0___redArg(lean_object* v_x_3548_, lean_object* v_x_3549_, lean_object* v_x_3550_){
_start:
{
uint64_t v___x_3551_; size_t v___x_3552_; size_t v___x_3553_; lean_object* v___x_3554_; 
v___x_3551_ = l_Lean_instHashableMVarId_hash(v_x_3549_);
v___x_3552_ = lean_uint64_to_usize(v___x_3551_);
v___x_3553_ = ((size_t)1ULL);
v___x_3554_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(v_x_3548_, v___x_3552_, v___x_3553_, v_x_3549_, v_x_3550_);
return v___x_3554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg(lean_object* v_mvarId_3555_, lean_object* v_val_3556_, lean_object* v___y_3557_){
_start:
{
lean_object* v___x_3559_; lean_object* v_mctx_3560_; lean_object* v_cache_3561_; lean_object* v_zetaDeltaFVarIds_3562_; lean_object* v_postponed_3563_; lean_object* v_diag_3564_; lean_object* v___x_3566_; uint8_t v_isShared_3567_; uint8_t v_isSharedCheck_3592_; 
v___x_3559_ = lean_st_ref_take(v___y_3557_);
v_mctx_3560_ = lean_ctor_get(v___x_3559_, 0);
v_cache_3561_ = lean_ctor_get(v___x_3559_, 1);
v_zetaDeltaFVarIds_3562_ = lean_ctor_get(v___x_3559_, 2);
v_postponed_3563_ = lean_ctor_get(v___x_3559_, 3);
v_diag_3564_ = lean_ctor_get(v___x_3559_, 4);
v_isSharedCheck_3592_ = !lean_is_exclusive(v___x_3559_);
if (v_isSharedCheck_3592_ == 0)
{
v___x_3566_ = v___x_3559_;
v_isShared_3567_ = v_isSharedCheck_3592_;
goto v_resetjp_3565_;
}
else
{
lean_inc(v_diag_3564_);
lean_inc(v_postponed_3563_);
lean_inc(v_zetaDeltaFVarIds_3562_);
lean_inc(v_cache_3561_);
lean_inc(v_mctx_3560_);
lean_dec(v___x_3559_);
v___x_3566_ = lean_box(0);
v_isShared_3567_ = v_isSharedCheck_3592_;
goto v_resetjp_3565_;
}
v_resetjp_3565_:
{
lean_object* v_depth_3568_; lean_object* v_levelAssignDepth_3569_; lean_object* v_lmvarCounter_3570_; lean_object* v_mvarCounter_3571_; lean_object* v_lDecls_3572_; lean_object* v_decls_3573_; lean_object* v_userNames_3574_; lean_object* v_lAssignment_3575_; lean_object* v_eAssignment_3576_; lean_object* v_dAssignment_3577_; lean_object* v___x_3579_; uint8_t v_isShared_3580_; uint8_t v_isSharedCheck_3591_; 
v_depth_3568_ = lean_ctor_get(v_mctx_3560_, 0);
v_levelAssignDepth_3569_ = lean_ctor_get(v_mctx_3560_, 1);
v_lmvarCounter_3570_ = lean_ctor_get(v_mctx_3560_, 2);
v_mvarCounter_3571_ = lean_ctor_get(v_mctx_3560_, 3);
v_lDecls_3572_ = lean_ctor_get(v_mctx_3560_, 4);
v_decls_3573_ = lean_ctor_get(v_mctx_3560_, 5);
v_userNames_3574_ = lean_ctor_get(v_mctx_3560_, 6);
v_lAssignment_3575_ = lean_ctor_get(v_mctx_3560_, 7);
v_eAssignment_3576_ = lean_ctor_get(v_mctx_3560_, 8);
v_dAssignment_3577_ = lean_ctor_get(v_mctx_3560_, 9);
v_isSharedCheck_3591_ = !lean_is_exclusive(v_mctx_3560_);
if (v_isSharedCheck_3591_ == 0)
{
v___x_3579_ = v_mctx_3560_;
v_isShared_3580_ = v_isSharedCheck_3591_;
goto v_resetjp_3578_;
}
else
{
lean_inc(v_dAssignment_3577_);
lean_inc(v_eAssignment_3576_);
lean_inc(v_lAssignment_3575_);
lean_inc(v_userNames_3574_);
lean_inc(v_decls_3573_);
lean_inc(v_lDecls_3572_);
lean_inc(v_mvarCounter_3571_);
lean_inc(v_lmvarCounter_3570_);
lean_inc(v_levelAssignDepth_3569_);
lean_inc(v_depth_3568_);
lean_dec(v_mctx_3560_);
v___x_3579_ = lean_box(0);
v_isShared_3580_ = v_isSharedCheck_3591_;
goto v_resetjp_3578_;
}
v_resetjp_3578_:
{
lean_object* v___x_3581_; lean_object* v___x_3583_; 
v___x_3581_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0___redArg(v_eAssignment_3576_, v_mvarId_3555_, v_val_3556_);
if (v_isShared_3580_ == 0)
{
lean_ctor_set(v___x_3579_, 8, v___x_3581_);
v___x_3583_ = v___x_3579_;
goto v_reusejp_3582_;
}
else
{
lean_object* v_reuseFailAlloc_3590_; 
v_reuseFailAlloc_3590_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3590_, 0, v_depth_3568_);
lean_ctor_set(v_reuseFailAlloc_3590_, 1, v_levelAssignDepth_3569_);
lean_ctor_set(v_reuseFailAlloc_3590_, 2, v_lmvarCounter_3570_);
lean_ctor_set(v_reuseFailAlloc_3590_, 3, v_mvarCounter_3571_);
lean_ctor_set(v_reuseFailAlloc_3590_, 4, v_lDecls_3572_);
lean_ctor_set(v_reuseFailAlloc_3590_, 5, v_decls_3573_);
lean_ctor_set(v_reuseFailAlloc_3590_, 6, v_userNames_3574_);
lean_ctor_set(v_reuseFailAlloc_3590_, 7, v_lAssignment_3575_);
lean_ctor_set(v_reuseFailAlloc_3590_, 8, v___x_3581_);
lean_ctor_set(v_reuseFailAlloc_3590_, 9, v_dAssignment_3577_);
v___x_3583_ = v_reuseFailAlloc_3590_;
goto v_reusejp_3582_;
}
v_reusejp_3582_:
{
lean_object* v___x_3585_; 
if (v_isShared_3567_ == 0)
{
lean_ctor_set(v___x_3566_, 0, v___x_3583_);
v___x_3585_ = v___x_3566_;
goto v_reusejp_3584_;
}
else
{
lean_object* v_reuseFailAlloc_3589_; 
v_reuseFailAlloc_3589_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3589_, 0, v___x_3583_);
lean_ctor_set(v_reuseFailAlloc_3589_, 1, v_cache_3561_);
lean_ctor_set(v_reuseFailAlloc_3589_, 2, v_zetaDeltaFVarIds_3562_);
lean_ctor_set(v_reuseFailAlloc_3589_, 3, v_postponed_3563_);
lean_ctor_set(v_reuseFailAlloc_3589_, 4, v_diag_3564_);
v___x_3585_ = v_reuseFailAlloc_3589_;
goto v_reusejp_3584_;
}
v_reusejp_3584_:
{
lean_object* v___x_3586_; lean_object* v___x_3587_; lean_object* v___x_3588_; 
v___x_3586_ = lean_st_ref_set(v___y_3557_, v___x_3585_);
v___x_3587_ = lean_box(0);
v___x_3588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3588_, 0, v___x_3587_);
return v___x_3588_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg___boxed(lean_object* v_mvarId_3593_, lean_object* v_val_3594_, lean_object* v___y_3595_, lean_object* v___y_3596_){
_start:
{
lean_object* v_res_3597_; 
v_res_3597_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg(v_mvarId_3593_, v_val_3594_, v___y_3595_);
lean_dec(v___y_3595_);
return v_res_3597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___lam__0(lean_object* v_e_3609_, lean_object* v_g_3610_, lean_object* v_depth_3611_, lean_object* v_config_3612_, lean_object* v_patterns_3613_, uint8_t v_symm_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_, lean_object* v___y_3618_){
_start:
{
lean_object* v___y_3621_; lean_object* v___y_3622_; lean_object* v___x_3640_; 
lean_inc(v___y_3618_);
lean_inc_ref(v___y_3617_);
lean_inc(v___y_3616_);
lean_inc_ref(v___y_3615_);
lean_inc_ref(v_e_3609_);
v___x_3640_ = lean_infer_type(v_e_3609_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
if (lean_obj_tag(v___x_3640_) == 0)
{
lean_object* v_a_3641_; lean_object* v___x_3642_; 
v_a_3641_ = lean_ctor_get(v___x_3640_, 0);
lean_inc(v_a_3641_);
lean_dec_ref_known(v___x_3640_, 1);
lean_inc(v_g_3610_);
v___x_3642_ = l_Lean_MVarId_getType(v_g_3610_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
if (lean_obj_tag(v___x_3642_) == 0)
{
lean_object* v_a_3643_; lean_object* v___x_3644_; lean_object* v___y_3646_; 
v_a_3643_ = lean_ctor_get(v___x_3642_, 0);
lean_inc(v_a_3643_);
lean_dec_ref_known(v___x_3642_, 1);
v___x_3644_ = ((lean_object*)(lp_mathlib_Lean_MVarId_convert___lam__0___closed__1));
if (v_symm_3614_ == 0)
{
lean_object* v___x_3673_; lean_object* v___x_3674_; lean_object* v___x_3675_; lean_object* v___x_3676_; 
v___x_3673_ = lean_unsigned_to_nat(2u);
v___x_3674_ = lean_mk_empty_array_with_capacity(v___x_3673_);
v___x_3675_ = lean_array_push(v___x_3674_, v_a_3643_);
v___x_3676_ = lean_array_push(v___x_3675_, v_a_3641_);
v___y_3646_ = v___x_3676_;
goto v___jp_3645_;
}
else
{
lean_object* v___x_3677_; lean_object* v___x_3678_; lean_object* v___x_3679_; lean_object* v___x_3680_; 
v___x_3677_ = lean_unsigned_to_nat(2u);
v___x_3678_ = lean_mk_empty_array_with_capacity(v___x_3677_);
v___x_3679_ = lean_array_push(v___x_3678_, v_a_3641_);
v___x_3680_ = lean_array_push(v___x_3679_, v_a_3643_);
v___y_3646_ = v___x_3680_;
goto v___jp_3645_;
}
v___jp_3645_:
{
lean_object* v___x_3647_; 
v___x_3647_ = l_Lean_Meta_mkAppM(v___x_3644_, v___y_3646_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
if (lean_obj_tag(v___x_3647_) == 0)
{
lean_object* v_a_3648_; lean_object* v___x_3649_; uint8_t v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; 
v_a_3648_ = lean_ctor_get(v___x_3647_, 0);
lean_inc(v_a_3648_);
lean_dec_ref_known(v___x_3647_, 1);
v___x_3649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3649_, 0, v_a_3648_);
v___x_3650_ = 0;
v___x_3651_ = lean_box(0);
v___x_3652_ = l_Lean_Meta_mkFreshExprMVar(v___x_3649_, v___x_3650_, v___x_3651_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
if (lean_obj_tag(v___x_3652_) == 0)
{
if (v_symm_3614_ == 0)
{
lean_object* v_a_3653_; lean_object* v___x_3654_; 
v_a_3653_ = lean_ctor_get(v___x_3652_, 0);
lean_inc(v_a_3653_);
lean_dec_ref_known(v___x_3652_, 1);
v___x_3654_ = ((lean_object*)(lp_mathlib_Lean_MVarId_convert___lam__0___closed__3));
v___y_3621_ = v_a_3653_;
v___y_3622_ = v___x_3654_;
goto v___jp_3620_;
}
else
{
lean_object* v_a_3655_; lean_object* v___x_3656_; 
v_a_3655_ = lean_ctor_get(v___x_3652_, 0);
lean_inc(v_a_3655_);
lean_dec_ref_known(v___x_3652_, 1);
v___x_3656_ = ((lean_object*)(lp_mathlib_Lean_MVarId_convert___lam__0___closed__5));
v___y_3621_ = v_a_3655_;
v___y_3622_ = v___x_3656_;
goto v___jp_3620_;
}
}
else
{
lean_object* v_a_3657_; lean_object* v___x_3659_; uint8_t v_isShared_3660_; uint8_t v_isSharedCheck_3664_; 
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v_patterns_3613_);
lean_dec_ref(v_config_3612_);
lean_dec(v_depth_3611_);
lean_dec(v_g_3610_);
lean_dec_ref(v_e_3609_);
v_a_3657_ = lean_ctor_get(v___x_3652_, 0);
v_isSharedCheck_3664_ = !lean_is_exclusive(v___x_3652_);
if (v_isSharedCheck_3664_ == 0)
{
v___x_3659_ = v___x_3652_;
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
else
{
lean_inc(v_a_3657_);
lean_dec(v___x_3652_);
v___x_3659_ = lean_box(0);
v_isShared_3660_ = v_isSharedCheck_3664_;
goto v_resetjp_3658_;
}
v_resetjp_3658_:
{
lean_object* v___x_3662_; 
if (v_isShared_3660_ == 0)
{
v___x_3662_ = v___x_3659_;
goto v_reusejp_3661_;
}
else
{
lean_object* v_reuseFailAlloc_3663_; 
v_reuseFailAlloc_3663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3663_, 0, v_a_3657_);
v___x_3662_ = v_reuseFailAlloc_3663_;
goto v_reusejp_3661_;
}
v_reusejp_3661_:
{
return v___x_3662_;
}
}
}
}
else
{
lean_object* v_a_3665_; lean_object* v___x_3667_; uint8_t v_isShared_3668_; uint8_t v_isSharedCheck_3672_; 
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v_patterns_3613_);
lean_dec_ref(v_config_3612_);
lean_dec(v_depth_3611_);
lean_dec(v_g_3610_);
lean_dec_ref(v_e_3609_);
v_a_3665_ = lean_ctor_get(v___x_3647_, 0);
v_isSharedCheck_3672_ = !lean_is_exclusive(v___x_3647_);
if (v_isSharedCheck_3672_ == 0)
{
v___x_3667_ = v___x_3647_;
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
else
{
lean_inc(v_a_3665_);
lean_dec(v___x_3647_);
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
else
{
lean_object* v_a_3681_; lean_object* v___x_3683_; uint8_t v_isShared_3684_; uint8_t v_isSharedCheck_3688_; 
lean_dec(v_a_3641_);
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v_patterns_3613_);
lean_dec_ref(v_config_3612_);
lean_dec(v_depth_3611_);
lean_dec(v_g_3610_);
lean_dec_ref(v_e_3609_);
v_a_3681_ = lean_ctor_get(v___x_3642_, 0);
v_isSharedCheck_3688_ = !lean_is_exclusive(v___x_3642_);
if (v_isSharedCheck_3688_ == 0)
{
v___x_3683_ = v___x_3642_;
v_isShared_3684_ = v_isSharedCheck_3688_;
goto v_resetjp_3682_;
}
else
{
lean_inc(v_a_3681_);
lean_dec(v___x_3642_);
v___x_3683_ = lean_box(0);
v_isShared_3684_ = v_isSharedCheck_3688_;
goto v_resetjp_3682_;
}
v_resetjp_3682_:
{
lean_object* v___x_3686_; 
if (v_isShared_3684_ == 0)
{
v___x_3686_ = v___x_3683_;
goto v_reusejp_3685_;
}
else
{
lean_object* v_reuseFailAlloc_3687_; 
v_reuseFailAlloc_3687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3687_, 0, v_a_3681_);
v___x_3686_ = v_reuseFailAlloc_3687_;
goto v_reusejp_3685_;
}
v_reusejp_3685_:
{
return v___x_3686_;
}
}
}
}
else
{
lean_object* v_a_3689_; lean_object* v___x_3691_; uint8_t v_isShared_3692_; uint8_t v_isSharedCheck_3696_; 
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v_patterns_3613_);
lean_dec_ref(v_config_3612_);
lean_dec(v_depth_3611_);
lean_dec(v_g_3610_);
lean_dec_ref(v_e_3609_);
v_a_3689_ = lean_ctor_get(v___x_3640_, 0);
v_isSharedCheck_3696_ = !lean_is_exclusive(v___x_3640_);
if (v_isSharedCheck_3696_ == 0)
{
v___x_3691_ = v___x_3640_;
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
else
{
lean_inc(v_a_3689_);
lean_dec(v___x_3640_);
v___x_3691_ = lean_box(0);
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
v_resetjp_3690_:
{
lean_object* v___x_3694_; 
if (v_isShared_3692_ == 0)
{
v___x_3694_ = v___x_3691_;
goto v_reusejp_3693_;
}
else
{
lean_object* v_reuseFailAlloc_3695_; 
v_reuseFailAlloc_3695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3695_, 0, v_a_3689_);
v___x_3694_ = v_reuseFailAlloc_3695_;
goto v_reusejp_3693_;
}
v_reusejp_3693_:
{
return v___x_3694_;
}
}
}
v___jp_3620_:
{
lean_object* v___x_3623_; lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3626_; lean_object* v___x_3627_; 
v___x_3623_ = lean_unsigned_to_nat(2u);
v___x_3624_ = lean_mk_empty_array_with_capacity(v___x_3623_);
lean_inc_ref(v___y_3621_);
v___x_3625_ = lean_array_push(v___x_3624_, v___y_3621_);
v___x_3626_ = lean_array_push(v___x_3625_, v_e_3609_);
lean_inc(v___y_3622_);
v___x_3627_ = l_Lean_Meta_mkAppM(v___y_3622_, v___x_3626_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
if (lean_obj_tag(v___x_3627_) == 0)
{
lean_object* v_a_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3631_; 
v_a_3628_ = lean_ctor_get(v___x_3627_, 0);
lean_inc(v_a_3628_);
lean_dec_ref_known(v___x_3627_, 1);
v___x_3629_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg(v_g_3610_, v_a_3628_, v___y_3616_);
lean_dec_ref(v___x_3629_);
v___x_3630_ = l_Lean_Expr_mvarId_x21(v___y_3621_);
lean_dec_ref(v___y_3621_);
v___x_3631_ = lp_mathlib_Lean_MVarId_congrN_x21(v___x_3630_, v_depth_3611_, v_config_3612_, v_patterns_3613_, v___y_3615_, v___y_3616_, v___y_3617_, v___y_3618_);
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
return v___x_3631_;
}
else
{
lean_object* v_a_3632_; lean_object* v___x_3634_; uint8_t v_isShared_3635_; uint8_t v_isSharedCheck_3639_; 
lean_dec_ref(v___y_3621_);
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v_patterns_3613_);
lean_dec_ref(v_config_3612_);
lean_dec(v_depth_3611_);
lean_dec(v_g_3610_);
v_a_3632_ = lean_ctor_get(v___x_3627_, 0);
v_isSharedCheck_3639_ = !lean_is_exclusive(v___x_3627_);
if (v_isSharedCheck_3639_ == 0)
{
v___x_3634_ = v___x_3627_;
v_isShared_3635_ = v_isSharedCheck_3639_;
goto v_resetjp_3633_;
}
else
{
lean_inc(v_a_3632_);
lean_dec(v___x_3627_);
v___x_3634_ = lean_box(0);
v_isShared_3635_ = v_isSharedCheck_3639_;
goto v_resetjp_3633_;
}
v_resetjp_3633_:
{
lean_object* v___x_3637_; 
if (v_isShared_3635_ == 0)
{
v___x_3637_ = v___x_3634_;
goto v_reusejp_3636_;
}
else
{
lean_object* v_reuseFailAlloc_3638_; 
v_reuseFailAlloc_3638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3638_, 0, v_a_3632_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___lam__0___boxed(lean_object* v_e_3697_, lean_object* v_g_3698_, lean_object* v_depth_3699_, lean_object* v_config_3700_, lean_object* v_patterns_3701_, lean_object* v_symm_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_, lean_object* v___y_3706_, lean_object* v___y_3707_){
_start:
{
uint8_t v_symm_boxed_3708_; lean_object* v_res_3709_; 
v_symm_boxed_3708_ = lean_unbox(v_symm_3702_);
v_res_3709_ = lp_mathlib_Lean_MVarId_convert___lam__0(v_e_3697_, v_g_3698_, v_depth_3699_, v_config_3700_, v_patterns_3701_, v_symm_boxed_3708_, v___y_3703_, v___y_3704_, v___y_3705_, v___y_3706_);
return v_res_3709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert(lean_object* v_e_3710_, uint8_t v_symm_3711_, lean_object* v_depth_3712_, lean_object* v_config_3713_, lean_object* v_patterns_3714_, lean_object* v_g_3715_, lean_object* v_a_3716_, lean_object* v_a_3717_, lean_object* v_a_3718_, lean_object* v_a_3719_){
_start:
{
lean_object* v___x_3721_; lean_object* v___f_3722_; lean_object* v___x_3723_; 
v___x_3721_ = lean_box(v_symm_3711_);
lean_inc(v_g_3715_);
v___f_3722_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_convert___lam__0___boxed), 11, 6);
lean_closure_set(v___f_3722_, 0, v_e_3710_);
lean_closure_set(v___f_3722_, 1, v_g_3715_);
lean_closure_set(v___f_3722_, 2, v_depth_3712_);
lean_closure_set(v___f_3722_, 3, v_config_3713_);
lean_closure_set(v___f_3722_, 4, v_patterns_3714_);
lean_closure_set(v___f_3722_, 5, v___x_3721_);
v___x_3723_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(v_g_3715_, v___f_3722_, v_a_3716_, v_a_3717_, v_a_3718_, v_a_3719_);
return v___x_3723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convert___boxed(lean_object* v_e_3724_, lean_object* v_symm_3725_, lean_object* v_depth_3726_, lean_object* v_config_3727_, lean_object* v_patterns_3728_, lean_object* v_g_3729_, lean_object* v_a_3730_, lean_object* v_a_3731_, lean_object* v_a_3732_, lean_object* v_a_3733_, lean_object* v_a_3734_){
_start:
{
uint8_t v_symm_boxed_3735_; lean_object* v_res_3736_; 
v_symm_boxed_3735_ = lean_unbox(v_symm_3725_);
v_res_3736_ = lp_mathlib_Lean_MVarId_convert(v_e_3724_, v_symm_boxed_3735_, v_depth_3726_, v_config_3727_, v_patterns_3728_, v_g_3729_, v_a_3730_, v_a_3731_, v_a_3732_, v_a_3733_);
lean_dec(v_a_3733_);
lean_dec_ref(v_a_3732_);
lean_dec(v_a_3731_);
lean_dec_ref(v_a_3730_);
return v_res_3736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0(lean_object* v_mvarId_3737_, lean_object* v_val_3738_, lean_object* v___y_3739_, lean_object* v___y_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_){
_start:
{
lean_object* v___x_3744_; 
v___x_3744_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___redArg(v_mvarId_3737_, v_val_3738_, v___y_3740_);
return v___x_3744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0___boxed(lean_object* v_mvarId_3745_, lean_object* v_val_3746_, lean_object* v___y_3747_, lean_object* v___y_3748_, lean_object* v___y_3749_, lean_object* v___y_3750_, lean_object* v___y_3751_){
_start:
{
lean_object* v_res_3752_; 
v_res_3752_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0(v_mvarId_3745_, v_val_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
lean_dec(v___y_3750_);
lean_dec_ref(v___y_3749_);
lean_dec(v___y_3748_);
lean_dec_ref(v___y_3747_);
return v_res_3752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0(lean_object* v_00_u03b2_3753_, lean_object* v_x_3754_, lean_object* v_x_3755_, lean_object* v_x_3756_){
_start:
{
lean_object* v___x_3757_; 
v___x_3757_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0___redArg(v_x_3754_, v_x_3755_, v_x_3756_);
return v___x_3757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_3758_, lean_object* v_x_3759_, size_t v_x_3760_, size_t v_x_3761_, lean_object* v_x_3762_, lean_object* v_x_3763_){
_start:
{
lean_object* v___x_3764_; 
v___x_3764_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___redArg(v_x_3759_, v_x_3760_, v_x_3761_, v_x_3762_, v_x_3763_);
return v___x_3764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_3765_, lean_object* v_x_3766_, lean_object* v_x_3767_, lean_object* v_x_3768_, lean_object* v_x_3769_, lean_object* v_x_3770_){
_start:
{
size_t v_x_2006__boxed_3771_; size_t v_x_2007__boxed_3772_; lean_object* v_res_3773_; 
v_x_2006__boxed_3771_ = lean_unbox_usize(v_x_3767_);
lean_dec(v_x_3767_);
v_x_2007__boxed_3772_ = lean_unbox_usize(v_x_3768_);
lean_dec(v_x_3768_);
v_res_3773_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2(v_00_u03b2_3765_, v_x_3766_, v_x_2006__boxed_3771_, v_x_2007__boxed_3772_, v_x_3769_, v_x_3770_);
return v_res_3773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3(lean_object* v_00_u03b2_3774_, lean_object* v_n_3775_, lean_object* v_k_3776_, lean_object* v_v_3777_){
_start:
{
lean_object* v___x_3778_; 
v___x_3778_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3___redArg(v_n_3775_, v_k_3776_, v_v_3777_);
return v___x_3778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4(lean_object* v_00_u03b2_3779_, size_t v_depth_3780_, lean_object* v_keys_3781_, lean_object* v_vals_3782_, lean_object* v_heq_3783_, lean_object* v_i_3784_, lean_object* v_entries_3785_){
_start:
{
lean_object* v___x_3786_; 
v___x_3786_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___redArg(v_depth_3780_, v_keys_3781_, v_vals_3782_, v_i_3784_, v_entries_3785_);
return v___x_3786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_00_u03b2_3787_, lean_object* v_depth_3788_, lean_object* v_keys_3789_, lean_object* v_vals_3790_, lean_object* v_heq_3791_, lean_object* v_i_3792_, lean_object* v_entries_3793_){
_start:
{
size_t v_depth_boxed_3794_; lean_object* v_res_3795_; 
v_depth_boxed_3794_ = lean_unbox_usize(v_depth_3788_);
lean_dec(v_depth_3788_);
v_res_3795_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__4(v_00_u03b2_3787_, v_depth_boxed_3794_, v_keys_3789_, v_vals_3790_, v_heq_3791_, v_i_3792_, v_entries_3793_);
lean_dec_ref(v_vals_3790_);
lean_dec_ref(v_keys_3789_);
return v_res_3795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_3796_, lean_object* v_x_3797_, lean_object* v_x_3798_, lean_object* v_x_3799_, lean_object* v_x_3800_){
_start:
{
lean_object* v___x_3801_; 
v___x_3801_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_convert_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(v_x_3797_, v_x_3798_, v_x_3799_, v_x_3800_);
return v___x_3801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0(lean_object* v_pf_3802_, lean_object* v___x_3803_, lean_object* v_typeNew_3804_, lean_object* v_g_3805_, lean_object* v_fvarId_3806_, lean_object* v___y_3807_, lean_object* v___y_3808_, lean_object* v___y_3809_, lean_object* v___y_3810_){
_start:
{
lean_object* v___x_3812_; 
v___x_3812_ = l_Lean_Meta_mkEqMP(v_pf_3802_, v___x_3803_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
if (lean_obj_tag(v___x_3812_) == 0)
{
lean_object* v_a_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; 
v_a_3813_ = lean_ctor_get(v___x_3812_, 0);
lean_inc(v_a_3813_);
lean_dec_ref_known(v___x_3812_, 1);
v___x_3814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3814_, 0, v_typeNew_3804_);
v___x_3815_ = lean_box(0);
v___x_3816_ = l_Lean_MVarId_replace(v_g_3805_, v_fvarId_3806_, v_a_3813_, v___x_3814_, v___x_3815_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
return v___x_3816_;
}
else
{
lean_object* v_a_3817_; lean_object* v___x_3819_; uint8_t v_isShared_3820_; uint8_t v_isSharedCheck_3824_; 
lean_dec(v_fvarId_3806_);
lean_dec(v_g_3805_);
lean_dec_ref(v_typeNew_3804_);
v_a_3817_ = lean_ctor_get(v___x_3812_, 0);
v_isSharedCheck_3824_ = !lean_is_exclusive(v___x_3812_);
if (v_isSharedCheck_3824_ == 0)
{
v___x_3819_ = v___x_3812_;
v_isShared_3820_ = v_isSharedCheck_3824_;
goto v_resetjp_3818_;
}
else
{
lean_inc(v_a_3817_);
lean_dec(v___x_3812_);
v___x_3819_ = lean_box(0);
v_isShared_3820_ = v_isSharedCheck_3824_;
goto v_resetjp_3818_;
}
v_resetjp_3818_:
{
lean_object* v___x_3822_; 
if (v_isShared_3820_ == 0)
{
v___x_3822_ = v___x_3819_;
goto v_reusejp_3821_;
}
else
{
lean_object* v_reuseFailAlloc_3823_; 
v_reuseFailAlloc_3823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3823_, 0, v_a_3817_);
v___x_3822_ = v_reuseFailAlloc_3823_;
goto v_reusejp_3821_;
}
v_reusejp_3821_:
{
return v___x_3822_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0___boxed(lean_object* v_pf_3825_, lean_object* v___x_3826_, lean_object* v_typeNew_3827_, lean_object* v_g_3828_, lean_object* v_fvarId_3829_, lean_object* v___y_3830_, lean_object* v___y_3831_, lean_object* v___y_3832_, lean_object* v___y_3833_, lean_object* v___y_3834_){
_start:
{
lean_object* v_res_3835_; 
v_res_3835_ = lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0(v_pf_3825_, v___x_3826_, v_typeNew_3827_, v_g_3828_, v_fvarId_3829_, v___y_3830_, v___y_3831_, v___y_3832_, v___y_3833_);
lean_dec(v___y_3833_);
lean_dec_ref(v___y_3832_);
lean_dec(v___y_3831_);
lean_dec_ref(v___y_3830_);
return v_res_3835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1(lean_object* v_fvarId_3836_, lean_object* v_typeNew_3837_, lean_object* v_g_3838_, lean_object* v_depth_3839_, lean_object* v_config_3840_, lean_object* v_patterns_3841_, uint8_t v_symm_3842_, lean_object* v___y_3843_, lean_object* v___y_3844_, lean_object* v___y_3845_, lean_object* v___y_3846_){
_start:
{
lean_object* v___y_3849_; lean_object* v_pf_3850_; lean_object* v___y_3851_; lean_object* v___y_3852_; lean_object* v___y_3853_; lean_object* v___y_3854_; lean_object* v___x_3887_; 
lean_inc(v_fvarId_3836_);
v___x_3887_ = l_Lean_FVarId_getType___redArg(v_fvarId_3836_, v___y_3843_, v___y_3845_, v___y_3846_);
if (lean_obj_tag(v___x_3887_) == 0)
{
lean_object* v_a_3888_; lean_object* v___x_3889_; lean_object* v___y_3891_; 
v_a_3888_ = lean_ctor_get(v___x_3887_, 0);
lean_inc(v_a_3888_);
lean_dec_ref_known(v___x_3887_, 1);
v___x_3889_ = ((lean_object*)(lp_mathlib_Lean_MVarId_convert___lam__0___closed__1));
if (v_symm_3842_ == 0)
{
lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; 
v___x_3926_ = lean_unsigned_to_nat(2u);
v___x_3927_ = lean_mk_empty_array_with_capacity(v___x_3926_);
v___x_3928_ = lean_array_push(v___x_3927_, v_a_3888_);
lean_inc_ref(v_typeNew_3837_);
v___x_3929_ = lean_array_push(v___x_3928_, v_typeNew_3837_);
v___y_3891_ = v___x_3929_;
goto v___jp_3890_;
}
else
{
lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; 
v___x_3930_ = lean_unsigned_to_nat(2u);
v___x_3931_ = lean_mk_empty_array_with_capacity(v___x_3930_);
lean_inc_ref(v_typeNew_3837_);
v___x_3932_ = lean_array_push(v___x_3931_, v_typeNew_3837_);
v___x_3933_ = lean_array_push(v___x_3932_, v_a_3888_);
v___y_3891_ = v___x_3933_;
goto v___jp_3890_;
}
v___jp_3890_:
{
lean_object* v___x_3892_; 
v___x_3892_ = l_Lean_Meta_mkAppM(v___x_3889_, v___y_3891_, v___y_3843_, v___y_3844_, v___y_3845_, v___y_3846_);
if (lean_obj_tag(v___x_3892_) == 0)
{
lean_object* v_a_3893_; lean_object* v___x_3894_; uint8_t v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; 
v_a_3893_ = lean_ctor_get(v___x_3892_, 0);
lean_inc(v_a_3893_);
lean_dec_ref_known(v___x_3892_, 1);
v___x_3894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3894_, 0, v_a_3893_);
v___x_3895_ = 0;
v___x_3896_ = lean_box(0);
v___x_3897_ = l_Lean_Meta_mkFreshExprMVar(v___x_3894_, v___x_3895_, v___x_3896_, v___y_3843_, v___y_3844_, v___y_3845_, v___y_3846_);
if (lean_obj_tag(v___x_3897_) == 0)
{
if (v_symm_3842_ == 0)
{
lean_object* v_a_3898_; 
v_a_3898_ = lean_ctor_get(v___x_3897_, 0);
lean_inc_n(v_a_3898_, 2);
lean_dec_ref_known(v___x_3897_, 1);
v___y_3849_ = v_a_3898_;
v_pf_3850_ = v_a_3898_;
v___y_3851_ = v___y_3843_;
v___y_3852_ = v___y_3844_;
v___y_3853_ = v___y_3845_;
v___y_3854_ = v___y_3846_;
goto v___jp_3848_;
}
else
{
lean_object* v_a_3899_; lean_object* v___x_3900_; 
v_a_3899_ = lean_ctor_get(v___x_3897_, 0);
lean_inc_n(v_a_3899_, 2);
lean_dec_ref_known(v___x_3897_, 1);
v___x_3900_ = l_Lean_Meta_mkEqSymm(v_a_3899_, v___y_3843_, v___y_3844_, v___y_3845_, v___y_3846_);
if (lean_obj_tag(v___x_3900_) == 0)
{
lean_object* v_a_3901_; 
v_a_3901_ = lean_ctor_get(v___x_3900_, 0);
lean_inc(v_a_3901_);
lean_dec_ref_known(v___x_3900_, 1);
v___y_3849_ = v_a_3899_;
v_pf_3850_ = v_a_3901_;
v___y_3851_ = v___y_3843_;
v___y_3852_ = v___y_3844_;
v___y_3853_ = v___y_3845_;
v___y_3854_ = v___y_3846_;
goto v___jp_3848_;
}
else
{
lean_object* v_a_3902_; lean_object* v___x_3904_; uint8_t v_isShared_3905_; uint8_t v_isSharedCheck_3909_; 
lean_dec(v_a_3899_);
lean_dec(v_patterns_3841_);
lean_dec_ref(v_config_3840_);
lean_dec(v_depth_3839_);
lean_dec(v_g_3838_);
lean_dec_ref(v_typeNew_3837_);
lean_dec(v_fvarId_3836_);
v_a_3902_ = lean_ctor_get(v___x_3900_, 0);
v_isSharedCheck_3909_ = !lean_is_exclusive(v___x_3900_);
if (v_isSharedCheck_3909_ == 0)
{
v___x_3904_ = v___x_3900_;
v_isShared_3905_ = v_isSharedCheck_3909_;
goto v_resetjp_3903_;
}
else
{
lean_inc(v_a_3902_);
lean_dec(v___x_3900_);
v___x_3904_ = lean_box(0);
v_isShared_3905_ = v_isSharedCheck_3909_;
goto v_resetjp_3903_;
}
v_resetjp_3903_:
{
lean_object* v___x_3907_; 
if (v_isShared_3905_ == 0)
{
v___x_3907_ = v___x_3904_;
goto v_reusejp_3906_;
}
else
{
lean_object* v_reuseFailAlloc_3908_; 
v_reuseFailAlloc_3908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3908_, 0, v_a_3902_);
v___x_3907_ = v_reuseFailAlloc_3908_;
goto v_reusejp_3906_;
}
v_reusejp_3906_:
{
return v___x_3907_;
}
}
}
}
}
else
{
lean_object* v_a_3910_; lean_object* v___x_3912_; uint8_t v_isShared_3913_; uint8_t v_isSharedCheck_3917_; 
lean_dec(v_patterns_3841_);
lean_dec_ref(v_config_3840_);
lean_dec(v_depth_3839_);
lean_dec(v_g_3838_);
lean_dec_ref(v_typeNew_3837_);
lean_dec(v_fvarId_3836_);
v_a_3910_ = lean_ctor_get(v___x_3897_, 0);
v_isSharedCheck_3917_ = !lean_is_exclusive(v___x_3897_);
if (v_isSharedCheck_3917_ == 0)
{
v___x_3912_ = v___x_3897_;
v_isShared_3913_ = v_isSharedCheck_3917_;
goto v_resetjp_3911_;
}
else
{
lean_inc(v_a_3910_);
lean_dec(v___x_3897_);
v___x_3912_ = lean_box(0);
v_isShared_3913_ = v_isSharedCheck_3917_;
goto v_resetjp_3911_;
}
v_resetjp_3911_:
{
lean_object* v___x_3915_; 
if (v_isShared_3913_ == 0)
{
v___x_3915_ = v___x_3912_;
goto v_reusejp_3914_;
}
else
{
lean_object* v_reuseFailAlloc_3916_; 
v_reuseFailAlloc_3916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3916_, 0, v_a_3910_);
v___x_3915_ = v_reuseFailAlloc_3916_;
goto v_reusejp_3914_;
}
v_reusejp_3914_:
{
return v___x_3915_;
}
}
}
}
else
{
lean_object* v_a_3918_; lean_object* v___x_3920_; uint8_t v_isShared_3921_; uint8_t v_isSharedCheck_3925_; 
lean_dec(v_patterns_3841_);
lean_dec_ref(v_config_3840_);
lean_dec(v_depth_3839_);
lean_dec(v_g_3838_);
lean_dec_ref(v_typeNew_3837_);
lean_dec(v_fvarId_3836_);
v_a_3918_ = lean_ctor_get(v___x_3892_, 0);
v_isSharedCheck_3925_ = !lean_is_exclusive(v___x_3892_);
if (v_isSharedCheck_3925_ == 0)
{
v___x_3920_ = v___x_3892_;
v_isShared_3921_ = v_isSharedCheck_3925_;
goto v_resetjp_3919_;
}
else
{
lean_inc(v_a_3918_);
lean_dec(v___x_3892_);
v___x_3920_ = lean_box(0);
v_isShared_3921_ = v_isSharedCheck_3925_;
goto v_resetjp_3919_;
}
v_resetjp_3919_:
{
lean_object* v___x_3923_; 
if (v_isShared_3921_ == 0)
{
v___x_3923_ = v___x_3920_;
goto v_reusejp_3922_;
}
else
{
lean_object* v_reuseFailAlloc_3924_; 
v_reuseFailAlloc_3924_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3924_, 0, v_a_3918_);
v___x_3923_ = v_reuseFailAlloc_3924_;
goto v_reusejp_3922_;
}
v_reusejp_3922_:
{
return v___x_3923_;
}
}
}
}
}
else
{
lean_object* v_a_3934_; lean_object* v___x_3936_; uint8_t v_isShared_3937_; uint8_t v_isSharedCheck_3941_; 
lean_dec(v_patterns_3841_);
lean_dec_ref(v_config_3840_);
lean_dec(v_depth_3839_);
lean_dec(v_g_3838_);
lean_dec_ref(v_typeNew_3837_);
lean_dec(v_fvarId_3836_);
v_a_3934_ = lean_ctor_get(v___x_3887_, 0);
v_isSharedCheck_3941_ = !lean_is_exclusive(v___x_3887_);
if (v_isSharedCheck_3941_ == 0)
{
v___x_3936_ = v___x_3887_;
v_isShared_3937_ = v_isSharedCheck_3941_;
goto v_resetjp_3935_;
}
else
{
lean_inc(v_a_3934_);
lean_dec(v___x_3887_);
v___x_3936_ = lean_box(0);
v_isShared_3937_ = v_isSharedCheck_3941_;
goto v_resetjp_3935_;
}
v_resetjp_3935_:
{
lean_object* v___x_3939_; 
if (v_isShared_3937_ == 0)
{
v___x_3939_ = v___x_3936_;
goto v_reusejp_3938_;
}
else
{
lean_object* v_reuseFailAlloc_3940_; 
v_reuseFailAlloc_3940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3940_, 0, v_a_3934_);
v___x_3939_ = v_reuseFailAlloc_3940_;
goto v_reusejp_3938_;
}
v_reusejp_3938_:
{
return v___x_3939_;
}
}
}
v___jp_3848_:
{
lean_object* v___x_3855_; lean_object* v___f_3856_; lean_object* v___x_3857_; 
lean_inc(v_fvarId_3836_);
v___x_3855_ = l_Lean_mkFVar(v_fvarId_3836_);
lean_inc(v_g_3838_);
v___f_3856_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_convertLocalDecl___lam__0___boxed), 10, 5);
lean_closure_set(v___f_3856_, 0, v_pf_3850_);
lean_closure_set(v___f_3856_, 1, v___x_3855_);
lean_closure_set(v___f_3856_, 2, v_typeNew_3837_);
lean_closure_set(v___f_3856_, 3, v_g_3838_);
lean_closure_set(v___f_3856_, 4, v_fvarId_3836_);
v___x_3857_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(v_g_3838_, v___f_3856_, v___y_3851_, v___y_3852_, v___y_3853_, v___y_3854_);
if (lean_obj_tag(v___x_3857_) == 0)
{
lean_object* v_a_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; 
v_a_3858_ = lean_ctor_get(v___x_3857_, 0);
lean_inc(v_a_3858_);
lean_dec_ref_known(v___x_3857_, 1);
v___x_3859_ = l_Lean_Expr_mvarId_x21(v___y_3849_);
lean_dec_ref(v___y_3849_);
v___x_3860_ = lp_mathlib_Lean_MVarId_congrN_x21(v___x_3859_, v_depth_3839_, v_config_3840_, v_patterns_3841_, v___y_3851_, v___y_3852_, v___y_3853_, v___y_3854_);
if (lean_obj_tag(v___x_3860_) == 0)
{
lean_object* v_a_3861_; lean_object* v___x_3863_; uint8_t v_isShared_3864_; uint8_t v_isSharedCheck_3870_; 
v_a_3861_ = lean_ctor_get(v___x_3860_, 0);
v_isSharedCheck_3870_ = !lean_is_exclusive(v___x_3860_);
if (v_isSharedCheck_3870_ == 0)
{
v___x_3863_ = v___x_3860_;
v_isShared_3864_ = v_isSharedCheck_3870_;
goto v_resetjp_3862_;
}
else
{
lean_inc(v_a_3861_);
lean_dec(v___x_3860_);
v___x_3863_ = lean_box(0);
v_isShared_3864_ = v_isSharedCheck_3870_;
goto v_resetjp_3862_;
}
v_resetjp_3862_:
{
lean_object* v_mvarId_3865_; lean_object* v___x_3866_; lean_object* v___x_3868_; 
v_mvarId_3865_ = lean_ctor_get(v_a_3858_, 1);
lean_inc(v_mvarId_3865_);
lean_dec(v_a_3858_);
v___x_3866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3866_, 0, v_mvarId_3865_);
lean_ctor_set(v___x_3866_, 1, v_a_3861_);
if (v_isShared_3864_ == 0)
{
lean_ctor_set(v___x_3863_, 0, v___x_3866_);
v___x_3868_ = v___x_3863_;
goto v_reusejp_3867_;
}
else
{
lean_object* v_reuseFailAlloc_3869_; 
v_reuseFailAlloc_3869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3869_, 0, v___x_3866_);
v___x_3868_ = v_reuseFailAlloc_3869_;
goto v_reusejp_3867_;
}
v_reusejp_3867_:
{
return v___x_3868_;
}
}
}
else
{
lean_object* v_a_3871_; lean_object* v___x_3873_; uint8_t v_isShared_3874_; uint8_t v_isSharedCheck_3878_; 
lean_dec(v_a_3858_);
v_a_3871_ = lean_ctor_get(v___x_3860_, 0);
v_isSharedCheck_3878_ = !lean_is_exclusive(v___x_3860_);
if (v_isSharedCheck_3878_ == 0)
{
v___x_3873_ = v___x_3860_;
v_isShared_3874_ = v_isSharedCheck_3878_;
goto v_resetjp_3872_;
}
else
{
lean_inc(v_a_3871_);
lean_dec(v___x_3860_);
v___x_3873_ = lean_box(0);
v_isShared_3874_ = v_isSharedCheck_3878_;
goto v_resetjp_3872_;
}
v_resetjp_3872_:
{
lean_object* v___x_3876_; 
if (v_isShared_3874_ == 0)
{
v___x_3876_ = v___x_3873_;
goto v_reusejp_3875_;
}
else
{
lean_object* v_reuseFailAlloc_3877_; 
v_reuseFailAlloc_3877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3877_, 0, v_a_3871_);
v___x_3876_ = v_reuseFailAlloc_3877_;
goto v_reusejp_3875_;
}
v_reusejp_3875_:
{
return v___x_3876_;
}
}
}
}
else
{
lean_object* v_a_3879_; lean_object* v___x_3881_; uint8_t v_isShared_3882_; uint8_t v_isSharedCheck_3886_; 
lean_dec_ref(v___y_3849_);
lean_dec(v_patterns_3841_);
lean_dec_ref(v_config_3840_);
lean_dec(v_depth_3839_);
v_a_3879_ = lean_ctor_get(v___x_3857_, 0);
v_isSharedCheck_3886_ = !lean_is_exclusive(v___x_3857_);
if (v_isSharedCheck_3886_ == 0)
{
v___x_3881_ = v___x_3857_;
v_isShared_3882_ = v_isSharedCheck_3886_;
goto v_resetjp_3880_;
}
else
{
lean_inc(v_a_3879_);
lean_dec(v___x_3857_);
v___x_3881_ = lean_box(0);
v_isShared_3882_ = v_isSharedCheck_3886_;
goto v_resetjp_3880_;
}
v_resetjp_3880_:
{
lean_object* v___x_3884_; 
if (v_isShared_3882_ == 0)
{
v___x_3884_ = v___x_3881_;
goto v_reusejp_3883_;
}
else
{
lean_object* v_reuseFailAlloc_3885_; 
v_reuseFailAlloc_3885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3885_, 0, v_a_3879_);
v___x_3884_ = v_reuseFailAlloc_3885_;
goto v_reusejp_3883_;
}
v_reusejp_3883_:
{
return v___x_3884_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1___boxed(lean_object* v_fvarId_3942_, lean_object* v_typeNew_3943_, lean_object* v_g_3944_, lean_object* v_depth_3945_, lean_object* v_config_3946_, lean_object* v_patterns_3947_, lean_object* v_symm_3948_, lean_object* v___y_3949_, lean_object* v___y_3950_, lean_object* v___y_3951_, lean_object* v___y_3952_, lean_object* v___y_3953_){
_start:
{
uint8_t v_symm_boxed_3954_; lean_object* v_res_3955_; 
v_symm_boxed_3954_ = lean_unbox(v_symm_3948_);
v_res_3955_ = lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1(v_fvarId_3942_, v_typeNew_3943_, v_g_3944_, v_depth_3945_, v_config_3946_, v_patterns_3947_, v_symm_boxed_3954_, v___y_3949_, v___y_3950_, v___y_3951_, v___y_3952_);
lean_dec(v___y_3952_);
lean_dec_ref(v___y_3951_);
lean_dec(v___y_3950_);
lean_dec_ref(v___y_3949_);
return v_res_3955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl(lean_object* v_g_3956_, lean_object* v_fvarId_3957_, lean_object* v_typeNew_3958_, uint8_t v_symm_3959_, lean_object* v_depth_3960_, lean_object* v_config_3961_, lean_object* v_patterns_3962_, lean_object* v_a_3963_, lean_object* v_a_3964_, lean_object* v_a_3965_, lean_object* v_a_3966_){
_start:
{
lean_object* v___x_3968_; lean_object* v___f_3969_; lean_object* v___x_3970_; 
v___x_3968_ = lean_box(v_symm_3959_);
lean_inc(v_g_3956_);
v___f_3969_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_convertLocalDecl___lam__1___boxed), 12, 7);
lean_closure_set(v___f_3969_, 0, v_fvarId_3957_);
lean_closure_set(v___f_3969_, 1, v_typeNew_3958_);
lean_closure_set(v___f_3969_, 2, v_g_3956_);
lean_closure_set(v___f_3969_, 3, v_depth_3960_);
lean_closure_set(v___f_3969_, 4, v_config_3961_);
lean_closure_set(v___f_3969_, 5, v_patterns_3962_);
lean_closure_set(v___f_3969_, 6, v___x_3968_);
v___x_3970_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_convert_spec__1___redArg(v_g_3956_, v___f_3969_, v_a_3963_, v_a_3964_, v_a_3965_, v_a_3966_);
return v___x_3970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_convertLocalDecl___boxed(lean_object* v_g_3971_, lean_object* v_fvarId_3972_, lean_object* v_typeNew_3973_, lean_object* v_symm_3974_, lean_object* v_depth_3975_, lean_object* v_config_3976_, lean_object* v_patterns_3977_, lean_object* v_a_3978_, lean_object* v_a_3979_, lean_object* v_a_3980_, lean_object* v_a_3981_, lean_object* v_a_3982_){
_start:
{
uint8_t v_symm_boxed_3983_; lean_object* v_res_3984_; 
v_symm_boxed_3983_ = lean_unbox(v_symm_3974_);
v_res_3984_ = lp_mathlib_Lean_MVarId_convertLocalDecl(v_g_3971_, v_fvarId_3972_, v_typeNew_3973_, v_symm_boxed_3983_, v_depth_3975_, v_config_3976_, v_patterns_3977_, v_a_3978_, v_a_3979_, v_a_3980_, v_a_3981_);
lean_dec(v_a_3981_);
lean_dec_ref(v_a_3980_);
lean_dec(v_a_3979_);
lean_dec_ref(v_a_3978_);
return v_res_3984_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__13(void){
_start:
{
lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4014_; 
v___x_4011_ = l_Lean_Parser_Tactic_optConfig;
v___x_4012_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__12));
v___x_4013_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4014_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4014_, 0, v___x_4013_);
lean_ctor_set(v___x_4014_, 1, v___x_4012_);
lean_ctor_set(v___x_4014_, 2, v___x_4011_);
return v___x_4014_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__17(void){
_start:
{
lean_object* v___x_4021_; lean_object* v___x_4022_; lean_object* v___x_4023_; lean_object* v___x_4024_; 
v___x_4021_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__16));
v___x_4022_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__13, &lp_mathlib_Mathlib_Tactic_convert___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__13);
v___x_4023_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4024_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4024_, 0, v___x_4023_);
lean_ctor_set(v___x_4024_, 1, v___x_4022_);
lean_ctor_set(v___x_4024_, 2, v___x_4021_);
return v___x_4024_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__21(void){
_start:
{
lean_object* v___x_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; lean_object* v___x_4033_; 
v___x_4030_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__20));
v___x_4031_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__17, &lp_mathlib_Mathlib_Tactic_convert___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__17);
v___x_4032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4033_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4033_, 0, v___x_4032_);
lean_ctor_set(v___x_4033_, 1, v___x_4031_);
lean_ctor_set(v___x_4033_, 2, v___x_4030_);
return v___x_4033_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__25(void){
_start:
{
lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; 
v___x_4040_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__24));
v___x_4041_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__21, &lp_mathlib_Mathlib_Tactic_convert___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__21);
v___x_4042_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4043_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4043_, 0, v___x_4042_);
lean_ctor_set(v___x_4043_, 1, v___x_4041_);
lean_ctor_set(v___x_4043_, 2, v___x_4040_);
return v___x_4043_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__33(void){
_start:
{
lean_object* v___x_4059_; lean_object* v___x_4060_; lean_object* v___x_4061_; lean_object* v___x_4062_; 
v___x_4059_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__32));
v___x_4060_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__25, &lp_mathlib_Mathlib_Tactic_convert___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__25);
v___x_4061_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4062_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4062_, 0, v___x_4061_);
lean_ctor_set(v___x_4062_, 1, v___x_4060_);
lean_ctor_set(v___x_4062_, 2, v___x_4059_);
return v___x_4062_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__49(void){
_start:
{
lean_object* v___x_4098_; lean_object* v___x_4099_; lean_object* v___x_4100_; lean_object* v___x_4101_; 
v___x_4098_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__48));
v___x_4099_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__33, &lp_mathlib_Mathlib_Tactic_convert___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__33);
v___x_4100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4101_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4101_, 0, v___x_4100_);
lean_ctor_set(v___x_4101_, 1, v___x_4099_);
lean_ctor_set(v___x_4101_, 2, v___x_4098_);
return v___x_4101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert___closed__50(void){
_start:
{
lean_object* v___x_4102_; lean_object* v___x_4103_; lean_object* v___x_4104_; lean_object* v___x_4105_; 
v___x_4102_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__49, &lp_mathlib_Mathlib_Tactic_convert___closed__49_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__49);
v___x_4103_ = lean_unsigned_to_nat(1022u);
v___x_4104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__3));
v___x_4105_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4105_, 0, v___x_4104_);
lean_ctor_set(v___x_4105_, 1, v___x_4103_);
lean_ctor_set(v___x_4105_, 2, v___x_4102_);
return v___x_4105_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert(void){
_start:
{
lean_object* v___x_4106_; 
v___x_4106_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert___closed__50, &lp_mathlib_Mathlib_Tactic_convert___closed__50_once, _init_lp_mathlib_Mathlib_Tactic_convert___closed__50);
return v___x_4106_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__3(void){
_start:
{
lean_object* v___x_4115_; lean_object* v___x_4116_; lean_object* v___x_4117_; lean_object* v___x_4118_; 
v___x_4115_ = l_Lean_Parser_Tactic_optConfig;
v___x_4116_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert_x21___closed__2));
v___x_4117_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4118_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4118_, 0, v___x_4117_);
lean_ctor_set(v___x_4118_, 1, v___x_4116_);
lean_ctor_set(v___x_4118_, 2, v___x_4115_);
return v___x_4118_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__4(void){
_start:
{
lean_object* v___x_4119_; lean_object* v___x_4120_; lean_object* v___x_4121_; lean_object* v___x_4122_; 
v___x_4119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__16));
v___x_4120_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__3, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__3);
v___x_4121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4122_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4122_, 0, v___x_4121_);
lean_ctor_set(v___x_4122_, 1, v___x_4120_);
lean_ctor_set(v___x_4122_, 2, v___x_4119_);
return v___x_4122_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__5(void){
_start:
{
lean_object* v___x_4123_; lean_object* v___x_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; 
v___x_4123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__20));
v___x_4124_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__4, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__4);
v___x_4125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4126_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4126_, 0, v___x_4125_);
lean_ctor_set(v___x_4126_, 1, v___x_4124_);
lean_ctor_set(v___x_4126_, 2, v___x_4123_);
return v___x_4126_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__6(void){
_start:
{
lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; lean_object* v___x_4130_; 
v___x_4127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__24));
v___x_4128_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__5, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__5);
v___x_4129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4130_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4130_, 0, v___x_4129_);
lean_ctor_set(v___x_4130_, 1, v___x_4128_);
lean_ctor_set(v___x_4130_, 2, v___x_4127_);
return v___x_4130_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__7(void){
_start:
{
lean_object* v___x_4131_; lean_object* v___x_4132_; lean_object* v___x_4133_; lean_object* v___x_4134_; 
v___x_4131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__32));
v___x_4132_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__6, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__6);
v___x_4133_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4134_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4134_, 0, v___x_4133_);
lean_ctor_set(v___x_4134_, 1, v___x_4132_);
lean_ctor_set(v___x_4134_, 2, v___x_4131_);
return v___x_4134_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__8(void){
_start:
{
lean_object* v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; lean_object* v___x_4138_; 
v___x_4135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__48));
v___x_4136_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__7, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__7);
v___x_4137_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4138_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4138_, 0, v___x_4137_);
lean_ctor_set(v___x_4138_, 1, v___x_4136_);
lean_ctor_set(v___x_4138_, 2, v___x_4135_);
return v___x_4138_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__9(void){
_start:
{
lean_object* v___x_4139_; lean_object* v___x_4140_; lean_object* v___x_4141_; lean_object* v___x_4142_; 
v___x_4139_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__8, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__8);
v___x_4140_ = lean_unsigned_to_nat(1022u);
v___x_4141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert_x21___closed__1));
v___x_4142_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4142_, 0, v___x_4141_);
lean_ctor_set(v___x_4142_, 1, v___x_4140_);
lean_ctor_set(v___x_4142_, 2, v___x_4139_);
return v___x_4142_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert_x21(void){
_start:
{
lean_object* v___x_4143_; 
v___x_4143_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert_x21___closed__9, &lp_mathlib_Mathlib_Tactic_convert_x21___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_convert_x21___closed__9);
return v___x_4143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1(size_t v_sz_4144_, size_t v_i_4145_, lean_object* v_bs_4146_){
_start:
{
uint8_t v___x_4147_; 
v___x_4147_ = lean_usize_dec_lt(v_i_4145_, v_sz_4144_);
if (v___x_4147_ == 0)
{
lean_object* v___x_4148_; 
v___x_4148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4148_, 0, v_bs_4146_);
return v___x_4148_;
}
else
{
lean_object* v_v_4149_; lean_object* v___x_4150_; lean_object* v_bs_x27_4151_; size_t v___x_4152_; size_t v___x_4153_; lean_object* v___x_4154_; 
v_v_4149_ = lean_array_uget(v_bs_4146_, v_i_4145_);
v___x_4150_ = lean_unsigned_to_nat(0u);
v_bs_x27_4151_ = lean_array_uset(v_bs_4146_, v_i_4145_, v___x_4150_);
v___x_4152_ = ((size_t)1ULL);
v___x_4153_ = lean_usize_add(v_i_4145_, v___x_4152_);
v___x_4154_ = lean_array_uset(v_bs_x27_4151_, v_i_4145_, v_v_4149_);
v_i_4145_ = v___x_4153_;
v_bs_4146_ = v___x_4154_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1___boxed(lean_object* v_sz_4156_, lean_object* v_i_4157_, lean_object* v_bs_4158_){
_start:
{
size_t v_sz_boxed_4159_; size_t v_i_boxed_4160_; lean_object* v_res_4161_; 
v_sz_boxed_4159_ = lean_unbox_usize(v_sz_4156_);
lean_dec(v_sz_4156_);
v_i_boxed_4160_ = lean_unbox_usize(v_i_4157_);
lean_dec(v_i_4157_);
v_res_4161_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1(v_sz_boxed_4159_, v_i_boxed_4160_, v_bs_4158_);
return v_res_4161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0(size_t v_sz_4162_, size_t v_i_4163_, lean_object* v_bs_4164_){
_start:
{
uint8_t v___x_4165_; 
v___x_4165_ = lean_usize_dec_lt(v_i_4163_, v_sz_4162_);
if (v___x_4165_ == 0)
{
return v_bs_4164_;
}
else
{
lean_object* v_v_4166_; lean_object* v___x_4167_; lean_object* v_bs_x27_4168_; size_t v___x_4169_; size_t v___x_4170_; lean_object* v___x_4171_; 
v_v_4166_ = lean_array_uget(v_bs_4164_, v_i_4163_);
v___x_4167_ = lean_unsigned_to_nat(0u);
v_bs_x27_4168_ = lean_array_uset(v_bs_4164_, v_i_4163_, v___x_4167_);
v___x_4169_ = ((size_t)1ULL);
v___x_4170_ = lean_usize_add(v_i_4163_, v___x_4169_);
v___x_4171_ = lean_array_uset(v_bs_x27_4168_, v_i_4163_, v_v_4166_);
v_i_4163_ = v___x_4170_;
v_bs_4164_ = v___x_4171_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0___boxed(lean_object* v_sz_4173_, lean_object* v_i_4174_, lean_object* v_bs_4175_){
_start:
{
size_t v_sz_boxed_4176_; size_t v_i_boxed_4177_; lean_object* v_res_4178_; 
v_sz_boxed_4176_ = lean_unbox_usize(v_sz_4173_);
lean_dec(v_sz_4173_);
v_i_boxed_4177_ = lean_unbox_usize(v_i_4174_);
lean_dec(v_i_4174_);
v_res_4178_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0(v_sz_boxed_4176_, v_i_boxed_4177_, v_bs_4175_);
return v_res_4178_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5(void){
_start:
{
lean_object* v___x_4186_; 
v___x_4186_ = l_Array_mkArray0(lean_box(0));
return v___x_4186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1(lean_object* v_x_4188_, lean_object* v_a_4189_, lean_object* v_a_4190_){
_start:
{
lean_object* v___x_4191_; uint8_t v___x_4192_; 
v___x_4191_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert_x21___closed__1));
lean_inc(v_x_4188_);
v___x_4192_ = l_Lean_Syntax_isOfKind(v_x_4188_, v___x_4191_);
if (v___x_4192_ == 0)
{
lean_object* v___x_4193_; lean_object* v___x_4194_; 
lean_dec(v_x_4188_);
v___x_4193_ = lean_box(1);
v___x_4194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4194_, 0, v___x_4193_);
lean_ctor_set(v___x_4194_, 1, v_a_4190_);
return v___x_4194_;
}
else
{
lean_object* v___x_4195_; lean_object* v___x_4196_; lean_object* v___y_4198_; lean_object* v___y_4199_; lean_object* v___y_4200_; lean_object* v___y_4201_; lean_object* v___y_4202_; lean_object* v___y_4203_; lean_object* v___y_4204_; lean_object* v___y_4205_; lean_object* v___y_4206_; lean_object* v___y_4207_; lean_object* v___y_4208_; lean_object* v___y_4214_; lean_object* v___y_4215_; lean_object* v___y_4216_; lean_object* v___y_4217_; lean_object* v___y_4218_; lean_object* v___y_4219_; lean_object* v___y_4220_; lean_object* v___y_4221_; lean_object* v___y_4222_; lean_object* v___y_4223_; lean_object* v___y_4224_; lean_object* v___y_4238_; lean_object* v___y_4239_; lean_object* v___y_4240_; lean_object* v___y_4241_; lean_object* v___y_4242_; lean_object* v___y_4243_; lean_object* v___y_4244_; lean_object* v___y_4245_; lean_object* v___y_4246_; lean_object* v___y_4247_; lean_object* v___y_4248_; lean_object* v___y_4257_; lean_object* v___y_4258_; lean_object* v___y_4259_; lean_object* v_w_4260_; lean_object* v___y_4261_; lean_object* v___y_4262_; lean_object* v___x_4280_; lean_object* v___y_4282_; lean_object* v___y_4283_; lean_object* v_n_4284_; lean_object* v___y_4285_; lean_object* v___y_4286_; lean_object* v_l_4302_; lean_object* v___y_4303_; lean_object* v___y_4304_; lean_object* v___x_4316_; uint8_t v___x_4317_; 
v___x_4195_ = lean_unsigned_to_nat(1u);
v___x_4196_ = l_Lean_Syntax_getArg(v_x_4188_, v___x_4195_);
v___x_4280_ = lean_unsigned_to_nat(2u);
v___x_4316_ = l_Lean_Syntax_getArg(v_x_4188_, v___x_4280_);
v___x_4317_ = l_Lean_Syntax_isNone(v___x_4316_);
if (v___x_4317_ == 0)
{
uint8_t v___x_4318_; 
lean_inc(v___x_4316_);
v___x_4318_ = l_Lean_Syntax_matchesNull(v___x_4316_, v___x_4195_);
if (v___x_4318_ == 0)
{
lean_object* v___x_4319_; lean_object* v___x_4320_; 
lean_dec(v___x_4316_);
lean_dec(v___x_4196_);
lean_dec(v_x_4188_);
v___x_4319_ = lean_box(1);
v___x_4320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4320_, 0, v___x_4319_);
lean_ctor_set(v___x_4320_, 1, v_a_4190_);
return v___x_4320_;
}
else
{
lean_object* v___x_4321_; lean_object* v_l_4322_; lean_object* v___x_4323_; 
v___x_4321_ = lean_unsigned_to_nat(0u);
v_l_4322_ = l_Lean_Syntax_getArg(v___x_4316_, v___x_4321_);
lean_dec(v___x_4316_);
v___x_4323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4323_, 0, v_l_4322_);
v_l_4302_ = v___x_4323_;
v___y_4303_ = v_a_4189_;
v___y_4304_ = v_a_4190_;
goto v___jp_4301_;
}
}
else
{
lean_object* v___x_4324_; 
lean_dec(v___x_4316_);
v___x_4324_ = lean_box(0);
v_l_4302_ = v___x_4324_;
v___y_4303_ = v_a_4189_;
v___y_4304_ = v_a_4190_;
goto v___jp_4301_;
}
v___jp_4197_:
{
lean_object* v___x_4209_; lean_object* v___x_4210_; lean_object* v___x_4211_; lean_object* v___x_4212_; 
lean_inc_ref(v___y_4199_);
v___x_4209_ = l_Array_append___redArg(v___y_4199_, v___y_4208_);
lean_dec_ref(v___y_4208_);
lean_inc(v___y_4202_);
v___x_4210_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4210_, 0, v___y_4202_);
lean_ctor_set(v___x_4210_, 1, v___y_4198_);
lean_ctor_set(v___x_4210_, 2, v___x_4209_);
lean_inc(v___y_4201_);
v___x_4211_ = l_Lean_Syntax_node7(v___y_4202_, v___y_4201_, v___y_4205_, v___y_4200_, v___x_4196_, v___y_4206_, v___y_4204_, v___y_4203_, v___x_4210_);
v___x_4212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4212_, 0, v___x_4211_);
lean_ctor_set(v___x_4212_, 1, v___y_4207_);
return v___x_4212_;
}
v___jp_4213_:
{
lean_object* v___x_4225_; lean_object* v___x_4226_; 
lean_inc_ref(v___y_4215_);
v___x_4225_ = l_Array_append___redArg(v___y_4215_, v___y_4224_);
lean_dec_ref(v___y_4224_);
lean_inc(v___y_4214_);
lean_inc(v___y_4218_);
v___x_4226_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4226_, 0, v___y_4218_);
lean_ctor_set(v___x_4226_, 1, v___y_4214_);
lean_ctor_set(v___x_4226_, 2, v___x_4225_);
if (lean_obj_tag(v___y_4221_) == 1)
{
lean_object* v_val_4227_; lean_object* v___x_4228_; lean_object* v___x_4229_; size_t v_sz_4230_; size_t v___x_4231_; lean_object* v___x_4232_; lean_object* v___x_4233_; lean_object* v___x_4234_; lean_object* v___x_4235_; 
v_val_4227_ = lean_ctor_get(v___y_4221_, 0);
lean_inc(v_val_4227_);
lean_dec_ref_known(v___y_4221_, 1);
v___x_4228_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__0));
lean_inc_n(v___y_4218_, 2);
v___x_4229_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4229_, 0, v___y_4218_);
lean_ctor_set(v___x_4229_, 1, v___x_4228_);
v_sz_4230_ = lean_array_size(v_val_4227_);
v___x_4231_ = ((size_t)0ULL);
v___x_4232_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__0(v_sz_4230_, v___x_4231_, v_val_4227_);
lean_inc_ref(v___y_4215_);
v___x_4233_ = l_Array_append___redArg(v___y_4215_, v___x_4232_);
lean_dec_ref(v___x_4232_);
lean_inc(v___y_4214_);
v___x_4234_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4234_, 0, v___y_4218_);
lean_ctor_set(v___x_4234_, 1, v___y_4214_);
lean_ctor_set(v___x_4234_, 2, v___x_4233_);
v___x_4235_ = l_Array_mkArray2___redArg(v___x_4229_, v___x_4234_);
v___y_4198_ = v___y_4214_;
v___y_4199_ = v___y_4215_;
v___y_4200_ = v___y_4217_;
v___y_4201_ = v___y_4216_;
v___y_4202_ = v___y_4218_;
v___y_4203_ = v___x_4226_;
v___y_4204_ = v___y_4220_;
v___y_4205_ = v___y_4219_;
v___y_4206_ = v___y_4222_;
v___y_4207_ = v___y_4223_;
v___y_4208_ = v___x_4235_;
goto v___jp_4197_;
}
else
{
lean_object* v___x_4236_; 
lean_dec(v___y_4221_);
v___x_4236_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4198_ = v___y_4214_;
v___y_4199_ = v___y_4215_;
v___y_4200_ = v___y_4217_;
v___y_4201_ = v___y_4216_;
v___y_4202_ = v___y_4218_;
v___y_4203_ = v___x_4226_;
v___y_4204_ = v___y_4220_;
v___y_4205_ = v___y_4219_;
v___y_4206_ = v___y_4222_;
v___y_4207_ = v___y_4223_;
v___y_4208_ = v___x_4236_;
goto v___jp_4197_;
}
}
v___jp_4237_:
{
lean_object* v___x_4249_; lean_object* v___x_4250_; 
lean_inc_ref(v___y_4239_);
v___x_4249_ = l_Array_append___redArg(v___y_4239_, v___y_4248_);
lean_dec_ref(v___y_4248_);
lean_inc(v___y_4238_);
lean_inc(v___y_4242_);
v___x_4250_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4250_, 0, v___y_4242_);
lean_ctor_set(v___x_4250_, 1, v___y_4238_);
lean_ctor_set(v___x_4250_, 2, v___x_4249_);
if (lean_obj_tag(v___y_4243_) == 1)
{
lean_object* v_val_4251_; lean_object* v___x_4252_; lean_object* v___x_4253_; lean_object* v___x_4254_; 
v_val_4251_ = lean_ctor_get(v___y_4243_, 0);
lean_inc(v_val_4251_);
lean_dec_ref_known(v___y_4243_, 1);
v___x_4252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2));
lean_inc(v___y_4242_);
v___x_4253_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4253_, 0, v___y_4242_);
lean_ctor_set(v___x_4253_, 1, v___x_4252_);
v___x_4254_ = l_Array_mkArray2___redArg(v___x_4253_, v_val_4251_);
v___y_4214_ = v___y_4238_;
v___y_4215_ = v___y_4239_;
v___y_4216_ = v___y_4241_;
v___y_4217_ = v___y_4240_;
v___y_4218_ = v___y_4242_;
v___y_4219_ = v___y_4245_;
v___y_4220_ = v___y_4244_;
v___y_4221_ = v___y_4246_;
v___y_4222_ = v___x_4250_;
v___y_4223_ = v___y_4247_;
v___y_4224_ = v___x_4254_;
goto v___jp_4213_;
}
else
{
lean_object* v___x_4255_; 
lean_dec(v___y_4243_);
v___x_4255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4214_ = v___y_4238_;
v___y_4215_ = v___y_4239_;
v___y_4216_ = v___y_4241_;
v___y_4217_ = v___y_4240_;
v___y_4218_ = v___y_4242_;
v___y_4219_ = v___y_4245_;
v___y_4220_ = v___y_4244_;
v___y_4221_ = v___y_4246_;
v___y_4222_ = v___x_4250_;
v___y_4223_ = v___y_4247_;
v___y_4224_ = v___x_4255_;
goto v___jp_4213_;
}
}
v___jp_4256_:
{
lean_object* v_ref_4263_; uint8_t v___x_4264_; lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; lean_object* v___x_4268_; lean_object* v___x_4269_; lean_object* v___x_4270_; lean_object* v___x_4271_; lean_object* v___x_4272_; lean_object* v___x_4273_; 
v_ref_4263_ = lean_ctor_get(v___y_4261_, 5);
v___x_4264_ = 0;
v___x_4265_ = l_Lean_SourceInfo_fromRef(v_ref_4263_, v___x_4264_);
v___x_4266_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__2));
v___x_4267_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__3));
lean_inc_n(v___x_4265_, 3);
v___x_4268_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4268_, 0, v___x_4265_);
lean_ctor_set(v___x_4268_, 1, v___x_4266_);
v___x_4269_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4));
v___x_4270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__9));
v___x_4271_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4271_, 0, v___x_4265_);
lean_ctor_set(v___x_4271_, 1, v___x_4270_);
v___x_4272_ = l_Lean_Syntax_node1(v___x_4265_, v___x_4269_, v___x_4271_);
v___x_4273_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5);
if (lean_obj_tag(v___y_4257_) == 1)
{
lean_object* v_val_4274_; lean_object* v___x_4275_; lean_object* v___x_4276_; lean_object* v___x_4277_; lean_object* v___x_4278_; 
v_val_4274_ = lean_ctor_get(v___y_4257_, 0);
lean_inc(v_val_4274_);
lean_dec_ref_known(v___y_4257_, 1);
v___x_4275_ = l_Lean_SourceInfo_fromRef(v_val_4274_, v___x_4192_);
lean_dec(v_val_4274_);
v___x_4276_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__6));
v___x_4277_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4277_, 0, v___x_4275_);
lean_ctor_set(v___x_4277_, 1, v___x_4276_);
v___x_4278_ = l_Array_mkArray1___redArg(v___x_4277_);
v___y_4238_ = v___x_4269_;
v___y_4239_ = v___x_4273_;
v___y_4240_ = v___x_4272_;
v___y_4241_ = v___x_4267_;
v___y_4242_ = v___x_4265_;
v___y_4243_ = v___y_4258_;
v___y_4244_ = v___y_4259_;
v___y_4245_ = v___x_4268_;
v___y_4246_ = v_w_4260_;
v___y_4247_ = v___y_4262_;
v___y_4248_ = v___x_4278_;
goto v___jp_4237_;
}
else
{
lean_object* v___x_4279_; 
lean_dec(v___y_4257_);
v___x_4279_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4238_ = v___x_4269_;
v___y_4239_ = v___x_4273_;
v___y_4240_ = v___x_4272_;
v___y_4241_ = v___x_4267_;
v___y_4242_ = v___x_4265_;
v___y_4243_ = v___y_4258_;
v___y_4244_ = v___y_4259_;
v___y_4245_ = v___x_4268_;
v___y_4246_ = v_w_4260_;
v___y_4247_ = v___y_4262_;
v___y_4248_ = v___x_4279_;
goto v___jp_4237_;
}
}
v___jp_4281_:
{
lean_object* v___x_4287_; lean_object* v___x_4288_; uint8_t v___x_4289_; 
v___x_4287_ = lean_unsigned_to_nat(5u);
v___x_4288_ = l_Lean_Syntax_getArg(v_x_4188_, v___x_4287_);
lean_dec(v_x_4188_);
v___x_4289_ = l_Lean_Syntax_isNone(v___x_4288_);
if (v___x_4289_ == 0)
{
uint8_t v___x_4290_; 
lean_inc(v___x_4288_);
v___x_4290_ = l_Lean_Syntax_matchesNull(v___x_4288_, v___x_4280_);
if (v___x_4290_ == 0)
{
lean_object* v___x_4291_; lean_object* v___x_4292_; 
lean_dec(v___x_4288_);
lean_dec(v_n_4284_);
lean_dec(v___y_4283_);
lean_dec(v___y_4282_);
lean_dec(v___x_4196_);
v___x_4291_ = lean_box(1);
v___x_4292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4292_, 0, v___x_4291_);
lean_ctor_set(v___x_4292_, 1, v___y_4286_);
return v___x_4292_;
}
else
{
lean_object* v___x_4293_; lean_object* v___x_4294_; size_t v_sz_4295_; size_t v___x_4296_; lean_object* v___x_4297_; 
v___x_4293_ = l_Lean_Syntax_getArg(v___x_4288_, v___x_4195_);
lean_dec(v___x_4288_);
v___x_4294_ = l_Lean_Syntax_getArgs(v___x_4293_);
lean_dec(v___x_4293_);
v_sz_4295_ = lean_array_size(v___x_4294_);
v___x_4296_ = ((size_t)0ULL);
v___x_4297_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1_spec__1(v_sz_4295_, v___x_4296_, v___x_4294_);
if (lean_obj_tag(v___x_4297_) == 0)
{
lean_object* v___x_4298_; lean_object* v___x_4299_; 
lean_dec(v_n_4284_);
lean_dec(v___y_4283_);
lean_dec(v___y_4282_);
lean_dec(v___x_4196_);
v___x_4298_ = lean_box(1);
v___x_4299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4299_, 0, v___x_4298_);
lean_ctor_set(v___x_4299_, 1, v___y_4286_);
return v___x_4299_;
}
else
{
v___y_4257_ = v___y_4282_;
v___y_4258_ = v_n_4284_;
v___y_4259_ = v___y_4283_;
v_w_4260_ = v___x_4297_;
v___y_4261_ = v___y_4285_;
v___y_4262_ = v___y_4286_;
goto v___jp_4256_;
}
}
}
else
{
lean_object* v___x_4300_; 
lean_dec(v___x_4288_);
v___x_4300_ = lean_box(0);
v___y_4257_ = v___y_4282_;
v___y_4258_ = v_n_4284_;
v___y_4259_ = v___y_4283_;
v_w_4260_ = v___x_4300_;
v___y_4261_ = v___y_4285_;
v___y_4262_ = v___y_4286_;
goto v___jp_4256_;
}
}
v___jp_4301_:
{
lean_object* v___x_4305_; lean_object* v___x_4306_; lean_object* v___x_4307_; lean_object* v___x_4308_; uint8_t v___x_4309_; 
v___x_4305_ = lean_unsigned_to_nat(3u);
v___x_4306_ = l_Lean_Syntax_getArg(v_x_4188_, v___x_4305_);
v___x_4307_ = lean_unsigned_to_nat(4u);
v___x_4308_ = l_Lean_Syntax_getArg(v_x_4188_, v___x_4307_);
v___x_4309_ = l_Lean_Syntax_isNone(v___x_4308_);
if (v___x_4309_ == 0)
{
uint8_t v___x_4310_; 
lean_inc(v___x_4308_);
v___x_4310_ = l_Lean_Syntax_matchesNull(v___x_4308_, v___x_4280_);
if (v___x_4310_ == 0)
{
lean_object* v___x_4311_; lean_object* v___x_4312_; 
lean_dec(v___x_4308_);
lean_dec(v___x_4306_);
lean_dec(v_l_4302_);
lean_dec(v___x_4196_);
lean_dec(v_x_4188_);
v___x_4311_ = lean_box(1);
v___x_4312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4312_, 0, v___x_4311_);
lean_ctor_set(v___x_4312_, 1, v___y_4304_);
return v___x_4312_;
}
else
{
lean_object* v_n_4313_; lean_object* v___x_4314_; 
v_n_4313_ = l_Lean_Syntax_getArg(v___x_4308_, v___x_4195_);
lean_dec(v___x_4308_);
v___x_4314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4314_, 0, v_n_4313_);
v___y_4282_ = v_l_4302_;
v___y_4283_ = v___x_4306_;
v_n_4284_ = v___x_4314_;
v___y_4285_ = v___y_4303_;
v___y_4286_ = v___y_4304_;
goto v___jp_4281_;
}
}
else
{
lean_object* v___x_4315_; 
lean_dec(v___x_4308_);
v___x_4315_ = lean_box(0);
v___y_4282_ = v_l_4302_;
v___y_4283_ = v___x_4306_;
v_n_4284_ = v___x_4315_;
v___y_4285_ = v___y_4303_;
v___y_4286_ = v___y_4304_;
goto v___jp_4281_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___boxed(lean_object* v_x_4325_, lean_object* v_a_4326_, lean_object* v_a_4327_){
_start:
{
lean_object* v_res_4328_; 
v_res_4328_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1(v_x_4325_, v_a_4326_, v_a_4327_);
lean_dec_ref(v_a_4326_);
return v_res_4328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0(uint8_t v___x_4329_, lean_object* v_term_4330_, lean_object* v_expectedType_x3f_4331_, lean_object* v___y_4332_, lean_object* v___y_4333_, lean_object* v___y_4334_, lean_object* v___y_4335_, lean_object* v___y_4336_, lean_object* v___y_4337_, lean_object* v___y_4338_, lean_object* v___y_4339_){
_start:
{
lean_object* v_declName_x3f_4341_; lean_object* v_macroStack_4342_; uint8_t v_mayPostpone_4343_; uint8_t v_errToSorry_4344_; lean_object* v_autoBoundImplicitContext_4345_; lean_object* v_autoBoundImplicitForbidden_4346_; lean_object* v_sectionVars_4347_; lean_object* v_sectionFVars_4348_; uint8_t v_implicitLambda_4349_; uint8_t v_heedElabAsElim_4350_; uint8_t v_isNoncomputableSection_4351_; uint8_t v_isMetaSection_4352_; uint8_t v_inPattern_4353_; lean_object* v_tacSnap_x3f_4354_; uint8_t v_saveRecAppSyntax_4355_; uint8_t v_holesAsSyntheticOpaque_4356_; uint8_t v_checkDeprecated_4357_; lean_object* v_fixedTermElabs_4358_; lean_object* v___x_4359_; lean_object* v___x_4360_; 
v_declName_x3f_4341_ = lean_ctor_get(v___y_4334_, 0);
v_macroStack_4342_ = lean_ctor_get(v___y_4334_, 1);
v_mayPostpone_4343_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8);
v_errToSorry_4344_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 1);
v_autoBoundImplicitContext_4345_ = lean_ctor_get(v___y_4334_, 2);
v_autoBoundImplicitForbidden_4346_ = lean_ctor_get(v___y_4334_, 3);
v_sectionVars_4347_ = lean_ctor_get(v___y_4334_, 4);
v_sectionFVars_4348_ = lean_ctor_get(v___y_4334_, 5);
v_implicitLambda_4349_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 2);
v_heedElabAsElim_4350_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_4351_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 4);
v_isMetaSection_4352_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 5);
v_inPattern_4353_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_4354_ = lean_ctor_get(v___y_4334_, 6);
v_saveRecAppSyntax_4355_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_4356_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 9);
v_checkDeprecated_4357_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*8 + 10);
v_fixedTermElabs_4358_ = lean_ctor_get(v___y_4334_, 7);
lean_inc_ref(v_fixedTermElabs_4358_);
lean_inc(v_tacSnap_x3f_4354_);
lean_inc(v_sectionFVars_4348_);
lean_inc(v_sectionVars_4347_);
lean_inc_ref(v_autoBoundImplicitForbidden_4346_);
lean_inc(v_autoBoundImplicitContext_4345_);
lean_inc(v_macroStack_4342_);
lean_inc(v_declName_x3f_4341_);
v___x_4359_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_4359_, 0, v_declName_x3f_4341_);
lean_ctor_set(v___x_4359_, 1, v_macroStack_4342_);
lean_ctor_set(v___x_4359_, 2, v_autoBoundImplicitContext_4345_);
lean_ctor_set(v___x_4359_, 3, v_autoBoundImplicitForbidden_4346_);
lean_ctor_set(v___x_4359_, 4, v_sectionVars_4347_);
lean_ctor_set(v___x_4359_, 5, v_sectionFVars_4348_);
lean_ctor_set(v___x_4359_, 6, v_tacSnap_x3f_4354_);
lean_ctor_set(v___x_4359_, 7, v_fixedTermElabs_4358_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8, v_mayPostpone_4343_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 1, v_errToSorry_4344_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 2, v_implicitLambda_4349_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 3, v_heedElabAsElim_4350_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 4, v_isNoncomputableSection_4351_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 5, v_isMetaSection_4352_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 6, v___x_4329_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 7, v_inPattern_4353_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_4355_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_4356_);
lean_ctor_set_uint8(v___x_4359_, sizeof(void*)*8 + 10, v_checkDeprecated_4357_);
v___x_4360_ = l_Lean_Elab_Tactic_elabTermEnsuringType(v_term_4330_, v_expectedType_x3f_4331_, v___x_4329_, v___y_4332_, v___y_4333_, v___x_4359_, v___y_4335_, v___y_4336_, v___y_4337_, v___y_4338_, v___y_4339_);
if (lean_obj_tag(v___x_4360_) == 0)
{
lean_object* v_a_4361_; uint8_t v___x_4362_; lean_object* v___x_4363_; 
v_a_4361_ = lean_ctor_get(v___x_4360_, 0);
lean_inc(v_a_4361_);
lean_dec_ref_known(v___x_4360_, 1);
v___x_4362_ = 1;
v___x_4363_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_4362_, v___x_4329_, v___x_4359_, v___y_4335_, v___y_4336_, v___y_4337_, v___y_4338_, v___y_4339_);
lean_dec_ref_known(v___x_4359_, 8);
if (lean_obj_tag(v___x_4363_) == 0)
{
lean_object* v___x_4365_; uint8_t v_isShared_4366_; uint8_t v_isSharedCheck_4370_; 
v_isSharedCheck_4370_ = !lean_is_exclusive(v___x_4363_);
if (v_isSharedCheck_4370_ == 0)
{
lean_object* v_unused_4371_; 
v_unused_4371_ = lean_ctor_get(v___x_4363_, 0);
lean_dec(v_unused_4371_);
v___x_4365_ = v___x_4363_;
v_isShared_4366_ = v_isSharedCheck_4370_;
goto v_resetjp_4364_;
}
else
{
lean_dec(v___x_4363_);
v___x_4365_ = lean_box(0);
v_isShared_4366_ = v_isSharedCheck_4370_;
goto v_resetjp_4364_;
}
v_resetjp_4364_:
{
lean_object* v___x_4368_; 
if (v_isShared_4366_ == 0)
{
lean_ctor_set(v___x_4365_, 0, v_a_4361_);
v___x_4368_ = v___x_4365_;
goto v_reusejp_4367_;
}
else
{
lean_object* v_reuseFailAlloc_4369_; 
v_reuseFailAlloc_4369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4369_, 0, v_a_4361_);
v___x_4368_ = v_reuseFailAlloc_4369_;
goto v_reusejp_4367_;
}
v_reusejp_4367_:
{
return v___x_4368_;
}
}
}
else
{
lean_object* v_a_4372_; lean_object* v___x_4374_; uint8_t v_isShared_4375_; uint8_t v_isSharedCheck_4379_; 
lean_dec(v_a_4361_);
v_a_4372_ = lean_ctor_get(v___x_4363_, 0);
v_isSharedCheck_4379_ = !lean_is_exclusive(v___x_4363_);
if (v_isSharedCheck_4379_ == 0)
{
v___x_4374_ = v___x_4363_;
v_isShared_4375_ = v_isSharedCheck_4379_;
goto v_resetjp_4373_;
}
else
{
lean_inc(v_a_4372_);
lean_dec(v___x_4363_);
v___x_4374_ = lean_box(0);
v_isShared_4375_ = v_isSharedCheck_4379_;
goto v_resetjp_4373_;
}
v_resetjp_4373_:
{
lean_object* v___x_4377_; 
if (v_isShared_4375_ == 0)
{
v___x_4377_ = v___x_4374_;
goto v_reusejp_4376_;
}
else
{
lean_object* v_reuseFailAlloc_4378_; 
v_reuseFailAlloc_4378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4378_, 0, v_a_4372_);
v___x_4377_ = v_reuseFailAlloc_4378_;
goto v_reusejp_4376_;
}
v_reusejp_4376_:
{
return v___x_4377_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_4359_, 8);
return v___x_4360_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0___boxed(lean_object* v___x_4380_, lean_object* v_term_4381_, lean_object* v_expectedType_x3f_4382_, lean_object* v___y_4383_, lean_object* v___y_4384_, lean_object* v___y_4385_, lean_object* v___y_4386_, lean_object* v___y_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_){
_start:
{
uint8_t v___x_701__boxed_4392_; lean_object* v_res_4393_; 
v___x_701__boxed_4392_ = lean_unbox(v___x_4380_);
v_res_4393_ = lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0(v___x_701__boxed_4392_, v_term_4381_, v_expectedType_x3f_4382_, v___y_4383_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_);
lean_dec(v___y_4390_);
lean_dec_ref(v___y_4389_);
lean_dec(v___y_4388_);
lean_dec_ref(v___y_4387_);
lean_dec(v___y_4386_);
lean_dec_ref(v___y_4385_);
lean_dec(v___y_4384_);
lean_dec_ref(v___y_4383_);
return v_res_4393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert(lean_object* v_term_4396_, lean_object* v_expectedType_x3f_4397_, lean_object* v_a_4398_, lean_object* v_a_4399_, lean_object* v_a_4400_, lean_object* v_a_4401_, lean_object* v_a_4402_, lean_object* v_a_4403_, lean_object* v_a_4404_, lean_object* v_a_4405_){
_start:
{
lean_object* v___x_4407_; 
v___x_4407_ = l_Lean_Elab_Tactic_getMainTag___redArg(v_a_4399_, v_a_4402_, v_a_4403_, v_a_4404_, v_a_4405_);
if (lean_obj_tag(v___x_4407_) == 0)
{
lean_object* v_a_4408_; uint8_t v___x_4409_; lean_object* v___x_4410_; lean_object* v___f_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; 
v_a_4408_ = lean_ctor_get(v___x_4407_, 0);
lean_inc(v_a_4408_);
lean_dec_ref_known(v___x_4407_, 1);
v___x_4409_ = 1;
v___x_4410_ = lean_box(v___x_4409_);
v___f_4411_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_elabTermForConvert___lam__0___boxed), 12, 3);
lean_closure_set(v___f_4411_, 0, v___x_4410_);
lean_closure_set(v___f_4411_, 1, v_term_4396_);
lean_closure_set(v___f_4411_, 2, v_expectedType_x3f_4397_);
v___x_4412_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabTermForConvert___closed__0));
v___x_4413_ = l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(v___f_4411_, v_a_4408_, v___x_4412_, v___x_4409_, v_a_4398_, v_a_4399_, v_a_4400_, v_a_4401_, v_a_4402_, v_a_4403_, v_a_4404_, v_a_4405_);
return v___x_4413_;
}
else
{
lean_object* v_a_4414_; lean_object* v___x_4416_; uint8_t v_isShared_4417_; uint8_t v_isSharedCheck_4421_; 
lean_dec(v_expectedType_x3f_4397_);
lean_dec(v_term_4396_);
v_a_4414_ = lean_ctor_get(v___x_4407_, 0);
v_isSharedCheck_4421_ = !lean_is_exclusive(v___x_4407_);
if (v_isSharedCheck_4421_ == 0)
{
v___x_4416_ = v___x_4407_;
v_isShared_4417_ = v_isSharedCheck_4421_;
goto v_resetjp_4415_;
}
else
{
lean_inc(v_a_4414_);
lean_dec(v___x_4407_);
v___x_4416_ = lean_box(0);
v_isShared_4417_ = v_isSharedCheck_4421_;
goto v_resetjp_4415_;
}
v_resetjp_4415_:
{
lean_object* v___x_4419_; 
if (v_isShared_4417_ == 0)
{
v___x_4419_ = v___x_4416_;
goto v_reusejp_4418_;
}
else
{
lean_object* v_reuseFailAlloc_4420_; 
v_reuseFailAlloc_4420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4420_, 0, v_a_4414_);
v___x_4419_ = v_reuseFailAlloc_4420_;
goto v_reusejp_4418_;
}
v_reusejp_4418_:
{
return v___x_4419_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabTermForConvert___boxed(lean_object* v_term_4422_, lean_object* v_expectedType_x3f_4423_, lean_object* v_a_4424_, lean_object* v_a_4425_, lean_object* v_a_4426_, lean_object* v_a_4427_, lean_object* v_a_4428_, lean_object* v_a_4429_, lean_object* v_a_4430_, lean_object* v_a_4431_, lean_object* v_a_4432_){
_start:
{
lean_object* v_res_4433_; 
v_res_4433_ = lp_mathlib_Mathlib_Tactic_elabTermForConvert(v_term_4422_, v_expectedType_x3f_4423_, v_a_4424_, v_a_4425_, v_a_4426_, v_a_4427_, v_a_4428_, v_a_4429_, v_a_4430_, v_a_4431_);
lean_dec(v_a_4431_);
lean_dec_ref(v_a_4430_);
lean_dec(v_a_4429_);
lean_dec_ref(v_a_4428_);
lean_dec(v_a_4427_);
lean_dec_ref(v_a_4426_);
lean_dec(v_a_4425_);
lean_dec_ref(v_a_4424_);
return v_res_4433_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4434_; lean_object* v___x_4435_; lean_object* v___x_4436_; 
v___x_4434_ = lean_box(0);
v___x_4435_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4436_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4436_, 0, v___x_4435_);
lean_ctor_set(v___x_4436_, 1, v___x_4434_);
return v___x_4436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg(){
_start:
{
lean_object* v___x_4438_; lean_object* v___x_4439_; 
v___x_4438_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___closed__0);
v___x_4439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4439_, 0, v___x_4438_);
return v___x_4439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg___boxed(lean_object* v___y_4440_){
_start:
{
lean_object* v_res_4441_; 
v_res_4441_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v_res_4441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0(lean_object* v_00_u03b1_4442_, lean_object* v___y_4443_, lean_object* v___y_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_, lean_object* v___y_4447_, lean_object* v___y_4448_, lean_object* v___y_4449_, lean_object* v___y_4450_){
_start:
{
lean_object* v___x_4452_; 
v___x_4452_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___boxed(lean_object* v_00_u03b1_4453_, lean_object* v___y_4454_, lean_object* v___y_4455_, lean_object* v___y_4456_, lean_object* v___y_4457_, lean_object* v___y_4458_, lean_object* v___y_4459_, lean_object* v___y_4460_, lean_object* v___y_4461_, lean_object* v___y_4462_){
_start:
{
lean_object* v_res_4463_; 
v_res_4463_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0(v_00_u03b1_4453_, v___y_4454_, v___y_4455_, v___y_4456_, v___y_4457_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
lean_dec(v___y_4461_);
lean_dec_ref(v___y_4460_);
lean_dec(v___y_4459_);
lean_dec_ref(v___y_4458_);
lean_dec(v___y_4457_);
lean_dec_ref(v___y_4456_);
lean_dec(v___y_4455_);
lean_dec_ref(v___y_4454_);
return v_res_4463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0(lean_object* v_fst_4464_, lean_object* v_a_4465_, lean_object* v___x_4466_, lean_object* v_snd_4467_, lean_object* v_n_4468_, lean_object* v_sym_4469_, uint8_t v___x_4470_, lean_object* v___y_4471_, lean_object* v___y_4472_, lean_object* v___y_4473_, lean_object* v___y_4474_, lean_object* v___y_4475_, lean_object* v___y_4476_, lean_object* v___y_4477_, lean_object* v___y_4478_){
_start:
{
lean_object* v_a_4481_; lean_object* v___x_4492_; 
v___x_4492_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4472_, v___y_4475_, v___y_4476_, v___y_4477_, v___y_4478_);
if (lean_obj_tag(v___x_4492_) == 0)
{
lean_object* v_a_4493_; uint8_t v___y_4495_; lean_object* v___y_4496_; uint8_t v___y_4510_; 
v_a_4493_ = lean_ctor_get(v___x_4492_, 0);
lean_inc(v_a_4493_);
lean_dec_ref_known(v___x_4492_, 1);
if (lean_obj_tag(v_sym_4469_) == 0)
{
uint8_t v___x_4521_; 
v___x_4521_ = 0;
v___y_4510_ = v___x_4521_;
goto v___jp_4509_;
}
else
{
v___y_4510_ = v___x_4470_;
goto v___jp_4509_;
}
v___jp_4494_:
{
lean_object* v___x_4497_; 
v___x_4497_ = lp_mathlib_Lean_MVarId_convert(v_fst_4464_, v___y_4495_, v___y_4496_, v_a_4465_, v___x_4466_, v_a_4493_, v___y_4475_, v___y_4476_, v___y_4477_, v___y_4478_);
if (lean_obj_tag(v___x_4497_) == 0)
{
lean_object* v_a_4498_; lean_object* v___x_4499_; 
v_a_4498_ = lean_ctor_get(v___x_4497_, 0);
lean_inc(v_a_4498_);
lean_dec_ref_known(v___x_4497_, 1);
v___x_4499_ = l_List_appendTR___redArg(v_a_4498_, v_snd_4467_);
v_a_4481_ = v___x_4499_;
goto v___jp_4480_;
}
else
{
lean_dec(v_snd_4467_);
if (lean_obj_tag(v___x_4497_) == 0)
{
lean_object* v_a_4500_; 
v_a_4500_ = lean_ctor_get(v___x_4497_, 0);
lean_inc(v_a_4500_);
lean_dec_ref_known(v___x_4497_, 1);
v_a_4481_ = v_a_4500_;
goto v___jp_4480_;
}
else
{
lean_object* v_a_4501_; lean_object* v___x_4503_; uint8_t v_isShared_4504_; uint8_t v_isSharedCheck_4508_; 
v_a_4501_ = lean_ctor_get(v___x_4497_, 0);
v_isSharedCheck_4508_ = !lean_is_exclusive(v___x_4497_);
if (v_isSharedCheck_4508_ == 0)
{
v___x_4503_ = v___x_4497_;
v_isShared_4504_ = v_isSharedCheck_4508_;
goto v_resetjp_4502_;
}
else
{
lean_inc(v_a_4501_);
lean_dec(v___x_4497_);
v___x_4503_ = lean_box(0);
v_isShared_4504_ = v_isSharedCheck_4508_;
goto v_resetjp_4502_;
}
v_resetjp_4502_:
{
lean_object* v___x_4506_; 
if (v_isShared_4504_ == 0)
{
v___x_4506_ = v___x_4503_;
goto v_reusejp_4505_;
}
else
{
lean_object* v_reuseFailAlloc_4507_; 
v_reuseFailAlloc_4507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4507_, 0, v_a_4501_);
v___x_4506_ = v_reuseFailAlloc_4507_;
goto v_reusejp_4505_;
}
v_reusejp_4505_:
{
return v___x_4506_;
}
}
}
}
}
v___jp_4509_:
{
if (lean_obj_tag(v_n_4468_) == 0)
{
lean_object* v___x_4511_; 
v___x_4511_ = lean_box(0);
v___y_4495_ = v___y_4510_;
v___y_4496_ = v___x_4511_;
goto v___jp_4494_;
}
else
{
lean_object* v_val_4512_; lean_object* v___x_4514_; uint8_t v_isShared_4515_; uint8_t v_isSharedCheck_4520_; 
v_val_4512_ = lean_ctor_get(v_n_4468_, 0);
v_isSharedCheck_4520_ = !lean_is_exclusive(v_n_4468_);
if (v_isSharedCheck_4520_ == 0)
{
v___x_4514_ = v_n_4468_;
v_isShared_4515_ = v_isSharedCheck_4520_;
goto v_resetjp_4513_;
}
else
{
lean_inc(v_val_4512_);
lean_dec(v_n_4468_);
v___x_4514_ = lean_box(0);
v_isShared_4515_ = v_isSharedCheck_4520_;
goto v_resetjp_4513_;
}
v_resetjp_4513_:
{
lean_object* v___x_4516_; lean_object* v___x_4518_; 
v___x_4516_ = l_Lean_TSyntax_getNat(v_val_4512_);
lean_dec(v_val_4512_);
if (v_isShared_4515_ == 0)
{
lean_ctor_set(v___x_4514_, 0, v___x_4516_);
v___x_4518_ = v___x_4514_;
goto v_reusejp_4517_;
}
else
{
lean_object* v_reuseFailAlloc_4519_; 
v_reuseFailAlloc_4519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4519_, 0, v___x_4516_);
v___x_4518_ = v_reuseFailAlloc_4519_;
goto v_reusejp_4517_;
}
v_reusejp_4517_:
{
v___y_4495_ = v___y_4510_;
v___y_4496_ = v___x_4518_;
goto v___jp_4494_;
}
}
}
}
}
else
{
lean_object* v_a_4522_; lean_object* v___x_4524_; uint8_t v_isShared_4525_; uint8_t v_isSharedCheck_4529_; 
lean_dec(v_n_4468_);
lean_dec(v_snd_4467_);
lean_dec(v___x_4466_);
lean_dec_ref(v_a_4465_);
lean_dec_ref(v_fst_4464_);
v_a_4522_ = lean_ctor_get(v___x_4492_, 0);
v_isSharedCheck_4529_ = !lean_is_exclusive(v___x_4492_);
if (v_isSharedCheck_4529_ == 0)
{
v___x_4524_ = v___x_4492_;
v_isShared_4525_ = v_isSharedCheck_4529_;
goto v_resetjp_4523_;
}
else
{
lean_inc(v_a_4522_);
lean_dec(v___x_4492_);
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
v___jp_4480_:
{
lean_object* v___x_4482_; 
v___x_4482_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_4481_, v___y_4472_, v___y_4475_, v___y_4476_, v___y_4477_, v___y_4478_);
if (lean_obj_tag(v___x_4482_) == 0)
{
lean_object* v___x_4484_; uint8_t v_isShared_4485_; uint8_t v_isSharedCheck_4490_; 
v_isSharedCheck_4490_ = !lean_is_exclusive(v___x_4482_);
if (v_isSharedCheck_4490_ == 0)
{
lean_object* v_unused_4491_; 
v_unused_4491_ = lean_ctor_get(v___x_4482_, 0);
lean_dec(v_unused_4491_);
v___x_4484_ = v___x_4482_;
v_isShared_4485_ = v_isSharedCheck_4490_;
goto v_resetjp_4483_;
}
else
{
lean_dec(v___x_4482_);
v___x_4484_ = lean_box(0);
v_isShared_4485_ = v_isSharedCheck_4490_;
goto v_resetjp_4483_;
}
v_resetjp_4483_:
{
lean_object* v___x_4486_; lean_object* v___x_4488_; 
v___x_4486_ = lean_box(0);
if (v_isShared_4485_ == 0)
{
lean_ctor_set(v___x_4484_, 0, v___x_4486_);
v___x_4488_ = v___x_4484_;
goto v_reusejp_4487_;
}
else
{
lean_object* v_reuseFailAlloc_4489_; 
v_reuseFailAlloc_4489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4489_, 0, v___x_4486_);
v___x_4488_ = v_reuseFailAlloc_4489_;
goto v_reusejp_4487_;
}
v_reusejp_4487_:
{
return v___x_4488_;
}
}
}
else
{
return v___x_4482_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0___boxed(lean_object* v_fst_4530_, lean_object* v_a_4531_, lean_object* v___x_4532_, lean_object* v_snd_4533_, lean_object* v_n_4534_, lean_object* v_sym_4535_, lean_object* v___x_4536_, lean_object* v___y_4537_, lean_object* v___y_4538_, lean_object* v___y_4539_, lean_object* v___y_4540_, lean_object* v___y_4541_, lean_object* v___y_4542_, lean_object* v___y_4543_, lean_object* v___y_4544_, lean_object* v___y_4545_){
_start:
{
uint8_t v___x_3564__boxed_4546_; lean_object* v_res_4547_; 
v___x_3564__boxed_4546_ = lean_unbox(v___x_4536_);
v_res_4547_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0(v_fst_4530_, v_a_4531_, v___x_4532_, v_snd_4533_, v_n_4534_, v_sym_4535_, v___x_3564__boxed_4546_, v___y_4537_, v___y_4538_, v___y_4539_, v___y_4540_, v___y_4541_, v___y_4542_, v___y_4543_, v___y_4544_);
lean_dec(v___y_4544_);
lean_dec_ref(v___y_4543_);
lean_dec(v___y_4542_);
lean_dec_ref(v___y_4541_);
lean_dec(v___y_4540_);
lean_dec_ref(v___y_4539_);
lean_dec(v___y_4538_);
lean_dec_ref(v___y_4537_);
lean_dec(v_sym_4535_);
return v_res_4547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1(uint8_t v___y_4548_, lean_object* v___x_4549_, lean_object* v___x_4550_, lean_object* v_n_4551_, lean_object* v_sym_4552_, uint8_t v___x_4553_, lean_object* v_ps_x3f_4554_, lean_object* v___x_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_, lean_object* v___y_4560_, lean_object* v___y_4561_, lean_object* v___y_4562_, lean_object* v___y_4563_){
_start:
{
lean_object* v___x_4565_; 
v___x_4565_ = lp_mathlib_Convert_elabConfig___redArg(v___y_4548_, v___x_4549_, v___y_4556_, v___y_4562_, v___y_4563_);
if (lean_obj_tag(v___x_4565_) == 0)
{
lean_object* v_a_4566_; lean_object* v___y_4568_; 
v_a_4566_ = lean_ctor_get(v___x_4565_, 0);
lean_inc(v_a_4566_);
lean_dec_ref_known(v___x_4565_, 1);
if (lean_obj_tag(v_ps_x3f_4554_) == 0)
{
lean_object* v___x_4620_; 
v___x_4620_ = lean_mk_empty_array_with_capacity(v___x_4555_);
v___y_4568_ = v___x_4620_;
goto v___jp_4567_;
}
else
{
lean_object* v_val_4621_; 
v_val_4621_ = lean_ctor_get(v_ps_x3f_4554_, 0);
lean_inc(v_val_4621_);
lean_dec_ref_known(v_ps_x3f_4554_, 1);
v___y_4568_ = v_val_4621_;
goto v___jp_4567_;
}
v___jp_4567_:
{
lean_object* v___x_4569_; 
v___x_4569_ = l_Lean_Elab_Tactic_getMainTarget(v___y_4556_, v___y_4557_, v___y_4558_, v___y_4559_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_);
if (lean_obj_tag(v___x_4569_) == 0)
{
lean_object* v_a_4570_; lean_object* v___x_4571_; 
v_a_4570_ = lean_ctor_get(v___x_4569_, 0);
lean_inc(v_a_4570_);
lean_dec_ref_known(v___x_4569_, 1);
v___x_4571_ = l_Lean_Meta_getLevel(v_a_4570_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_);
if (lean_obj_tag(v___x_4571_) == 0)
{
lean_object* v_a_4572_; lean_object* v___x_4573_; lean_object* v___x_4574_; uint8_t v___x_4575_; lean_object* v___x_4576_; lean_object* v___x_4577_; 
v_a_4572_ = lean_ctor_get(v___x_4571_, 0);
lean_inc(v_a_4572_);
lean_dec_ref_known(v___x_4571_, 1);
v___x_4573_ = l_Lean_mkSort(v_a_4572_);
v___x_4574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4574_, 0, v___x_4573_);
v___x_4575_ = 0;
v___x_4576_ = lean_box(0);
v___x_4577_ = l_Lean_Meta_mkFreshExprMVar(v___x_4574_, v___x_4575_, v___x_4576_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_);
if (lean_obj_tag(v___x_4577_) == 0)
{
lean_object* v_a_4578_; lean_object* v___x_4579_; lean_object* v___x_4580_; 
v_a_4578_ = lean_ctor_get(v___x_4577_, 0);
lean_inc(v_a_4578_);
lean_dec_ref_known(v___x_4577_, 1);
v___x_4579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4579_, 0, v_a_4578_);
v___x_4580_ = lp_mathlib_Mathlib_Tactic_elabTermForConvert(v___x_4550_, v___x_4579_, v___y_4556_, v___y_4557_, v___y_4558_, v___y_4559_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_);
if (lean_obj_tag(v___x_4580_) == 0)
{
lean_object* v_a_4581_; lean_object* v_fst_4582_; lean_object* v_snd_4583_; lean_object* v___x_4584_; lean_object* v___x_4585_; lean_object* v___f_4586_; lean_object* v___x_4587_; 
v_a_4581_ = lean_ctor_get(v___x_4580_, 0);
lean_inc(v_a_4581_);
lean_dec_ref_known(v___x_4580_, 1);
v_fst_4582_ = lean_ctor_get(v_a_4581_, 0);
lean_inc(v_fst_4582_);
v_snd_4583_ = lean_ctor_get(v_a_4581_, 1);
lean_inc(v_snd_4583_);
lean_dec(v_a_4581_);
v___x_4584_ = lean_array_to_list(v___y_4568_);
v___x_4585_ = lean_box(v___x_4553_);
v___f_4586_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__0___boxed), 16, 7);
lean_closure_set(v___f_4586_, 0, v_fst_4582_);
lean_closure_set(v___f_4586_, 1, v_a_4566_);
lean_closure_set(v___f_4586_, 2, v___x_4584_);
lean_closure_set(v___f_4586_, 3, v_snd_4583_);
lean_closure_set(v___f_4586_, 4, v_n_4551_);
lean_closure_set(v___f_4586_, 5, v_sym_4552_);
lean_closure_set(v___f_4586_, 6, v___x_4585_);
v___x_4587_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4586_, v___y_4556_, v___y_4557_, v___y_4558_, v___y_4559_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_);
return v___x_4587_;
}
else
{
lean_object* v_a_4588_; lean_object* v___x_4590_; uint8_t v_isShared_4591_; uint8_t v_isSharedCheck_4595_; 
lean_dec_ref(v___y_4568_);
lean_dec(v_a_4566_);
lean_dec(v_sym_4552_);
lean_dec(v_n_4551_);
v_a_4588_ = lean_ctor_get(v___x_4580_, 0);
v_isSharedCheck_4595_ = !lean_is_exclusive(v___x_4580_);
if (v_isSharedCheck_4595_ == 0)
{
v___x_4590_ = v___x_4580_;
v_isShared_4591_ = v_isSharedCheck_4595_;
goto v_resetjp_4589_;
}
else
{
lean_inc(v_a_4588_);
lean_dec(v___x_4580_);
v___x_4590_ = lean_box(0);
v_isShared_4591_ = v_isSharedCheck_4595_;
goto v_resetjp_4589_;
}
v_resetjp_4589_:
{
lean_object* v___x_4593_; 
if (v_isShared_4591_ == 0)
{
v___x_4593_ = v___x_4590_;
goto v_reusejp_4592_;
}
else
{
lean_object* v_reuseFailAlloc_4594_; 
v_reuseFailAlloc_4594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4594_, 0, v_a_4588_);
v___x_4593_ = v_reuseFailAlloc_4594_;
goto v_reusejp_4592_;
}
v_reusejp_4592_:
{
return v___x_4593_;
}
}
}
}
else
{
lean_object* v_a_4596_; lean_object* v___x_4598_; uint8_t v_isShared_4599_; uint8_t v_isSharedCheck_4603_; 
lean_dec_ref(v___y_4568_);
lean_dec(v_a_4566_);
lean_dec(v_sym_4552_);
lean_dec(v_n_4551_);
lean_dec(v___x_4550_);
v_a_4596_ = lean_ctor_get(v___x_4577_, 0);
v_isSharedCheck_4603_ = !lean_is_exclusive(v___x_4577_);
if (v_isSharedCheck_4603_ == 0)
{
v___x_4598_ = v___x_4577_;
v_isShared_4599_ = v_isSharedCheck_4603_;
goto v_resetjp_4597_;
}
else
{
lean_inc(v_a_4596_);
lean_dec(v___x_4577_);
v___x_4598_ = lean_box(0);
v_isShared_4599_ = v_isSharedCheck_4603_;
goto v_resetjp_4597_;
}
v_resetjp_4597_:
{
lean_object* v___x_4601_; 
if (v_isShared_4599_ == 0)
{
v___x_4601_ = v___x_4598_;
goto v_reusejp_4600_;
}
else
{
lean_object* v_reuseFailAlloc_4602_; 
v_reuseFailAlloc_4602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4602_, 0, v_a_4596_);
v___x_4601_ = v_reuseFailAlloc_4602_;
goto v_reusejp_4600_;
}
v_reusejp_4600_:
{
return v___x_4601_;
}
}
}
}
else
{
lean_object* v_a_4604_; lean_object* v___x_4606_; uint8_t v_isShared_4607_; uint8_t v_isSharedCheck_4611_; 
lean_dec_ref(v___y_4568_);
lean_dec(v_a_4566_);
lean_dec(v_sym_4552_);
lean_dec(v_n_4551_);
lean_dec(v___x_4550_);
v_a_4604_ = lean_ctor_get(v___x_4571_, 0);
v_isSharedCheck_4611_ = !lean_is_exclusive(v___x_4571_);
if (v_isSharedCheck_4611_ == 0)
{
v___x_4606_ = v___x_4571_;
v_isShared_4607_ = v_isSharedCheck_4611_;
goto v_resetjp_4605_;
}
else
{
lean_inc(v_a_4604_);
lean_dec(v___x_4571_);
v___x_4606_ = lean_box(0);
v_isShared_4607_ = v_isSharedCheck_4611_;
goto v_resetjp_4605_;
}
v_resetjp_4605_:
{
lean_object* v___x_4609_; 
if (v_isShared_4607_ == 0)
{
v___x_4609_ = v___x_4606_;
goto v_reusejp_4608_;
}
else
{
lean_object* v_reuseFailAlloc_4610_; 
v_reuseFailAlloc_4610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4610_, 0, v_a_4604_);
v___x_4609_ = v_reuseFailAlloc_4610_;
goto v_reusejp_4608_;
}
v_reusejp_4608_:
{
return v___x_4609_;
}
}
}
}
else
{
lean_object* v_a_4612_; lean_object* v___x_4614_; uint8_t v_isShared_4615_; uint8_t v_isSharedCheck_4619_; 
lean_dec_ref(v___y_4568_);
lean_dec(v_a_4566_);
lean_dec(v_sym_4552_);
lean_dec(v_n_4551_);
lean_dec(v___x_4550_);
v_a_4612_ = lean_ctor_get(v___x_4569_, 0);
v_isSharedCheck_4619_ = !lean_is_exclusive(v___x_4569_);
if (v_isSharedCheck_4619_ == 0)
{
v___x_4614_ = v___x_4569_;
v_isShared_4615_ = v_isSharedCheck_4619_;
goto v_resetjp_4613_;
}
else
{
lean_inc(v_a_4612_);
lean_dec(v___x_4569_);
v___x_4614_ = lean_box(0);
v_isShared_4615_ = v_isSharedCheck_4619_;
goto v_resetjp_4613_;
}
v_resetjp_4613_:
{
lean_object* v___x_4617_; 
if (v_isShared_4615_ == 0)
{
v___x_4617_ = v___x_4614_;
goto v_reusejp_4616_;
}
else
{
lean_object* v_reuseFailAlloc_4618_; 
v_reuseFailAlloc_4618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4618_, 0, v_a_4612_);
v___x_4617_ = v_reuseFailAlloc_4618_;
goto v_reusejp_4616_;
}
v_reusejp_4616_:
{
return v___x_4617_;
}
}
}
}
}
else
{
lean_object* v_a_4622_; lean_object* v___x_4624_; uint8_t v_isShared_4625_; uint8_t v_isSharedCheck_4629_; 
lean_dec(v_ps_x3f_4554_);
lean_dec(v_sym_4552_);
lean_dec(v_n_4551_);
lean_dec(v___x_4550_);
v_a_4622_ = lean_ctor_get(v___x_4565_, 0);
v_isSharedCheck_4629_ = !lean_is_exclusive(v___x_4565_);
if (v_isSharedCheck_4629_ == 0)
{
v___x_4624_ = v___x_4565_;
v_isShared_4625_ = v_isSharedCheck_4629_;
goto v_resetjp_4623_;
}
else
{
lean_inc(v_a_4622_);
lean_dec(v___x_4565_);
v___x_4624_ = lean_box(0);
v_isShared_4625_ = v_isSharedCheck_4629_;
goto v_resetjp_4623_;
}
v_resetjp_4623_:
{
lean_object* v___x_4627_; 
if (v_isShared_4625_ == 0)
{
v___x_4627_ = v___x_4624_;
goto v_reusejp_4626_;
}
else
{
lean_object* v_reuseFailAlloc_4628_; 
v_reuseFailAlloc_4628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4628_, 0, v_a_4622_);
v___x_4627_ = v_reuseFailAlloc_4628_;
goto v_reusejp_4626_;
}
v_reusejp_4626_:
{
return v___x_4627_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1___boxed(lean_object** _args){
lean_object* v___y_4630_ = _args[0];
lean_object* v___x_4631_ = _args[1];
lean_object* v___x_4632_ = _args[2];
lean_object* v_n_4633_ = _args[3];
lean_object* v_sym_4634_ = _args[4];
lean_object* v___x_4635_ = _args[5];
lean_object* v_ps_x3f_4636_ = _args[6];
lean_object* v___x_4637_ = _args[7];
lean_object* v___y_4638_ = _args[8];
lean_object* v___y_4639_ = _args[9];
lean_object* v___y_4640_ = _args[10];
lean_object* v___y_4641_ = _args[11];
lean_object* v___y_4642_ = _args[12];
lean_object* v___y_4643_ = _args[13];
lean_object* v___y_4644_ = _args[14];
lean_object* v___y_4645_ = _args[15];
lean_object* v___y_4646_ = _args[16];
_start:
{
uint8_t v___y_3705__boxed_4647_; uint8_t v___x_3708__boxed_4648_; lean_object* v_res_4649_; 
v___y_3705__boxed_4647_ = lean_unbox(v___y_4630_);
v___x_3708__boxed_4648_ = lean_unbox(v___x_4635_);
v_res_4649_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1(v___y_3705__boxed_4647_, v___x_4631_, v___x_4632_, v_n_4633_, v_sym_4634_, v___x_3708__boxed_4648_, v_ps_x3f_4636_, v___x_4637_, v___y_4638_, v___y_4639_, v___y_4640_, v___y_4641_, v___y_4642_, v___y_4643_, v___y_4644_, v___y_4645_);
lean_dec(v___y_4645_);
lean_dec_ref(v___y_4644_);
lean_dec(v___y_4643_);
lean_dec_ref(v___y_4642_);
lean_dec(v___y_4641_);
lean_dec_ref(v___y_4640_);
lean_dec(v___y_4639_);
lean_dec_ref(v___y_4638_);
lean_dec(v___x_4637_);
return v_res_4649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1(lean_object* v_x_4650_, lean_object* v_a_4651_, lean_object* v_a_4652_, lean_object* v_a_4653_, lean_object* v_a_4654_, lean_object* v_a_4655_, lean_object* v_a_4656_, lean_object* v_a_4657_, lean_object* v_a_4658_){
_start:
{
lean_object* v___x_4660_; uint8_t v___x_4661_; 
v___x_4660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__3));
lean_inc(v_x_4650_);
v___x_4661_ = l_Lean_Syntax_isOfKind(v_x_4650_, v___x_4660_);
if (v___x_4661_ == 0)
{
lean_object* v___x_4662_; 
lean_dec(v_x_4650_);
v___x_4662_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4662_;
}
else
{
lean_object* v___x_4663_; lean_object* v___y_4665_; lean_object* v___y_4666_; lean_object* v___y_4667_; lean_object* v___y_4668_; lean_object* v___y_4669_; lean_object* v___y_4670_; lean_object* v___y_4671_; lean_object* v___y_4672_; lean_object* v___y_4673_; lean_object* v___y_4674_; lean_object* v___y_4675_; lean_object* v___y_4676_; lean_object* v___y_4677_; uint8_t v___y_4678_; lean_object* v___y_4684_; lean_object* v___y_4685_; lean_object* v___y_4686_; lean_object* v___y_4687_; lean_object* v___y_4688_; lean_object* v___y_4689_; lean_object* v___y_4690_; lean_object* v___y_4691_; lean_object* v___y_4692_; lean_object* v___y_4693_; lean_object* v___y_4694_; lean_object* v___y_4695_; lean_object* v___y_4696_; lean_object* v_ps_x3f_4697_; lean_object* v___x_4699_; lean_object* v___y_4701_; lean_object* v___y_4702_; lean_object* v___y_4703_; lean_object* v___y_4704_; lean_object* v___y_4705_; lean_object* v___y_4706_; lean_object* v___y_4707_; lean_object* v___y_4708_; lean_object* v___y_4709_; lean_object* v___y_4710_; lean_object* v___y_4711_; lean_object* v___y_4712_; lean_object* v___y_4713_; lean_object* v_n_4714_; lean_object* v___y_4725_; lean_object* v___y_4726_; lean_object* v___y_4727_; lean_object* v___y_4728_; lean_object* v___y_4729_; lean_object* v___y_4730_; lean_object* v___y_4731_; lean_object* v___y_4732_; lean_object* v___y_4733_; lean_object* v___y_4734_; lean_object* v___y_4735_; lean_object* v_sym_4736_; lean_object* v_expensive_4748_; lean_object* v___y_4749_; lean_object* v___y_4750_; lean_object* v___y_4751_; lean_object* v___y_4752_; lean_object* v___y_4753_; lean_object* v___y_4754_; lean_object* v___y_4755_; lean_object* v___y_4756_; lean_object* v___x_4767_; uint8_t v___x_4768_; 
v___x_4663_ = lean_unsigned_to_nat(0u);
v___x_4699_ = lean_unsigned_to_nat(1u);
v___x_4767_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4699_);
v___x_4768_ = l_Lean_Syntax_isNone(v___x_4767_);
if (v___x_4768_ == 0)
{
uint8_t v___x_4769_; 
lean_inc(v___x_4767_);
v___x_4769_ = l_Lean_Syntax_matchesNull(v___x_4767_, v___x_4699_);
if (v___x_4769_ == 0)
{
lean_object* v___x_4770_; 
lean_dec(v___x_4767_);
lean_dec(v_x_4650_);
v___x_4770_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4770_;
}
else
{
lean_object* v_expensive_4771_; lean_object* v___x_4772_; 
v_expensive_4771_ = l_Lean_Syntax_getArg(v___x_4767_, v___x_4663_);
lean_dec(v___x_4767_);
v___x_4772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4772_, 0, v_expensive_4771_);
v_expensive_4748_ = v___x_4772_;
v___y_4749_ = v_a_4651_;
v___y_4750_ = v_a_4652_;
v___y_4751_ = v_a_4653_;
v___y_4752_ = v_a_4654_;
v___y_4753_ = v_a_4655_;
v___y_4754_ = v_a_4656_;
v___y_4755_ = v_a_4657_;
v___y_4756_ = v_a_4658_;
goto v___jp_4747_;
}
}
else
{
lean_object* v___x_4773_; 
lean_dec(v___x_4767_);
v___x_4773_ = lean_box(0);
v_expensive_4748_ = v___x_4773_;
v___y_4749_ = v_a_4651_;
v___y_4750_ = v_a_4652_;
v___y_4751_ = v_a_4653_;
v___y_4752_ = v_a_4654_;
v___y_4753_ = v_a_4655_;
v___y_4754_ = v_a_4656_;
v___y_4755_ = v_a_4657_;
v___y_4756_ = v_a_4658_;
goto v___jp_4747_;
}
v___jp_4664_:
{
lean_object* v___x_4679_; lean_object* v___x_4680_; lean_object* v___f_4681_; lean_object* v___x_4682_; 
v___x_4679_ = lean_box(v___y_4678_);
v___x_4680_ = lean_box(v___x_4661_);
v___f_4681_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___lam__1___boxed), 17, 8);
lean_closure_set(v___f_4681_, 0, v___x_4679_);
lean_closure_set(v___f_4681_, 1, v___y_4669_);
lean_closure_set(v___f_4681_, 2, v___y_4673_);
lean_closure_set(v___f_4681_, 3, v___y_4665_);
lean_closure_set(v___f_4681_, 4, v___y_4677_);
lean_closure_set(v___f_4681_, 5, v___x_4680_);
lean_closure_set(v___f_4681_, 6, v___y_4666_);
lean_closure_set(v___f_4681_, 7, v___x_4663_);
v___x_4682_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4681_, v___y_4671_, v___y_4675_, v___y_4668_, v___y_4672_, v___y_4667_, v___y_4676_, v___y_4674_, v___y_4670_);
return v___x_4682_;
}
v___jp_4683_:
{
if (lean_obj_tag(v___y_4687_) == 0)
{
uint8_t v___x_4698_; 
v___x_4698_ = 0;
v___y_4665_ = v___y_4684_;
v___y_4666_ = v_ps_x3f_4697_;
v___y_4667_ = v___y_4685_;
v___y_4668_ = v___y_4686_;
v___y_4669_ = v___y_4688_;
v___y_4670_ = v___y_4689_;
v___y_4671_ = v___y_4690_;
v___y_4672_ = v___y_4691_;
v___y_4673_ = v___y_4692_;
v___y_4674_ = v___y_4693_;
v___y_4675_ = v___y_4694_;
v___y_4676_ = v___y_4695_;
v___y_4677_ = v___y_4696_;
v___y_4678_ = v___x_4698_;
goto v___jp_4664_;
}
else
{
lean_dec_ref_known(v___y_4687_, 1);
v___y_4665_ = v___y_4684_;
v___y_4666_ = v_ps_x3f_4697_;
v___y_4667_ = v___y_4685_;
v___y_4668_ = v___y_4686_;
v___y_4669_ = v___y_4688_;
v___y_4670_ = v___y_4689_;
v___y_4671_ = v___y_4690_;
v___y_4672_ = v___y_4691_;
v___y_4673_ = v___y_4692_;
v___y_4674_ = v___y_4693_;
v___y_4675_ = v___y_4694_;
v___y_4676_ = v___y_4695_;
v___y_4677_ = v___y_4696_;
v___y_4678_ = v___x_4661_;
goto v___jp_4664_;
}
}
v___jp_4700_:
{
lean_object* v___x_4715_; lean_object* v___x_4716_; uint8_t v___x_4717_; 
v___x_4715_ = lean_unsigned_to_nat(6u);
v___x_4716_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4715_);
lean_dec(v_x_4650_);
v___x_4717_ = l_Lean_Syntax_isNone(v___x_4716_);
if (v___x_4717_ == 0)
{
uint8_t v___x_4718_; 
lean_inc(v___x_4716_);
v___x_4718_ = l_Lean_Syntax_matchesNull(v___x_4716_, v___y_4703_);
if (v___x_4718_ == 0)
{
lean_object* v___x_4719_; 
lean_dec(v___x_4716_);
lean_dec(v_n_4714_);
lean_dec(v___y_4713_);
lean_dec(v___y_4709_);
lean_dec(v___y_4705_);
lean_dec(v___y_4704_);
v___x_4719_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4719_;
}
else
{
lean_object* v___x_4720_; lean_object* v_ps_x3f_4721_; lean_object* v___x_4722_; 
v___x_4720_ = l_Lean_Syntax_getArg(v___x_4716_, v___x_4699_);
lean_dec(v___x_4716_);
v_ps_x3f_4721_ = l_Lean_Syntax_getArgs(v___x_4720_);
lean_dec(v___x_4720_);
v___x_4722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4722_, 0, v_ps_x3f_4721_);
v___y_4684_ = v_n_4714_;
v___y_4685_ = v___y_4701_;
v___y_4686_ = v___y_4702_;
v___y_4687_ = v___y_4704_;
v___y_4688_ = v___y_4705_;
v___y_4689_ = v___y_4706_;
v___y_4690_ = v___y_4707_;
v___y_4691_ = v___y_4708_;
v___y_4692_ = v___y_4709_;
v___y_4693_ = v___y_4710_;
v___y_4694_ = v___y_4711_;
v___y_4695_ = v___y_4712_;
v___y_4696_ = v___y_4713_;
v_ps_x3f_4697_ = v___x_4722_;
goto v___jp_4683_;
}
}
else
{
lean_object* v___x_4723_; 
lean_dec(v___x_4716_);
v___x_4723_ = lean_box(0);
v___y_4684_ = v_n_4714_;
v___y_4685_ = v___y_4701_;
v___y_4686_ = v___y_4702_;
v___y_4687_ = v___y_4704_;
v___y_4688_ = v___y_4705_;
v___y_4689_ = v___y_4706_;
v___y_4690_ = v___y_4707_;
v___y_4691_ = v___y_4708_;
v___y_4692_ = v___y_4709_;
v___y_4693_ = v___y_4710_;
v___y_4694_ = v___y_4711_;
v___y_4695_ = v___y_4712_;
v___y_4696_ = v___y_4713_;
v_ps_x3f_4697_ = v___x_4723_;
goto v___jp_4683_;
}
}
v___jp_4724_:
{
lean_object* v___x_4737_; lean_object* v___x_4738_; lean_object* v___x_4739_; lean_object* v___x_4740_; uint8_t v___x_4741_; 
v___x_4737_ = lean_unsigned_to_nat(4u);
v___x_4738_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4737_);
v___x_4739_ = lean_unsigned_to_nat(5u);
v___x_4740_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4739_);
v___x_4741_ = l_Lean_Syntax_isNone(v___x_4740_);
if (v___x_4741_ == 0)
{
uint8_t v___x_4742_; 
lean_inc(v___x_4740_);
v___x_4742_ = l_Lean_Syntax_matchesNull(v___x_4740_, v___y_4731_);
if (v___x_4742_ == 0)
{
lean_object* v___x_4743_; 
lean_dec(v___x_4740_);
lean_dec(v___x_4738_);
lean_dec(v_sym_4736_);
lean_dec(v___y_4733_);
lean_dec(v___y_4730_);
lean_dec(v_x_4650_);
v___x_4743_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4743_;
}
else
{
lean_object* v_n_4744_; lean_object* v___x_4745_; 
v_n_4744_ = l_Lean_Syntax_getArg(v___x_4740_, v___x_4699_);
lean_dec(v___x_4740_);
v___x_4745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4745_, 0, v_n_4744_);
v___y_4701_ = v___y_4728_;
v___y_4702_ = v___y_4729_;
v___y_4703_ = v___y_4731_;
v___y_4704_ = v___y_4730_;
v___y_4705_ = v___y_4733_;
v___y_4706_ = v___y_4725_;
v___y_4707_ = v___y_4726_;
v___y_4708_ = v___y_4727_;
v___y_4709_ = v___x_4738_;
v___y_4710_ = v___y_4732_;
v___y_4711_ = v___y_4735_;
v___y_4712_ = v___y_4734_;
v___y_4713_ = v_sym_4736_;
v_n_4714_ = v___x_4745_;
goto v___jp_4700_;
}
}
else
{
lean_object* v___x_4746_; 
lean_dec(v___x_4740_);
v___x_4746_ = lean_box(0);
v___y_4701_ = v___y_4728_;
v___y_4702_ = v___y_4729_;
v___y_4703_ = v___y_4731_;
v___y_4704_ = v___y_4730_;
v___y_4705_ = v___y_4733_;
v___y_4706_ = v___y_4725_;
v___y_4707_ = v___y_4726_;
v___y_4708_ = v___y_4727_;
v___y_4709_ = v___x_4738_;
v___y_4710_ = v___y_4732_;
v___y_4711_ = v___y_4735_;
v___y_4712_ = v___y_4734_;
v___y_4713_ = v_sym_4736_;
v_n_4714_ = v___x_4746_;
goto v___jp_4700_;
}
}
v___jp_4747_:
{
lean_object* v___x_4757_; lean_object* v___x_4758_; lean_object* v___x_4759_; lean_object* v___x_4760_; uint8_t v___x_4761_; 
v___x_4757_ = lean_unsigned_to_nat(2u);
v___x_4758_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4757_);
v___x_4759_ = lean_unsigned_to_nat(3u);
v___x_4760_ = l_Lean_Syntax_getArg(v_x_4650_, v___x_4759_);
v___x_4761_ = l_Lean_Syntax_isNone(v___x_4760_);
if (v___x_4761_ == 0)
{
uint8_t v___x_4762_; 
lean_inc(v___x_4760_);
v___x_4762_ = l_Lean_Syntax_matchesNull(v___x_4760_, v___x_4699_);
if (v___x_4762_ == 0)
{
lean_object* v___x_4763_; 
lean_dec(v___x_4760_);
lean_dec(v___x_4758_);
lean_dec(v_expensive_4748_);
lean_dec(v_x_4650_);
v___x_4763_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_4763_;
}
else
{
lean_object* v_sym_4764_; lean_object* v___x_4765_; 
v_sym_4764_ = l_Lean_Syntax_getArg(v___x_4760_, v___x_4663_);
lean_dec(v___x_4760_);
v___x_4765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4765_, 0, v_sym_4764_);
v___y_4725_ = v___y_4756_;
v___y_4726_ = v___y_4749_;
v___y_4727_ = v___y_4752_;
v___y_4728_ = v___y_4753_;
v___y_4729_ = v___y_4751_;
v___y_4730_ = v_expensive_4748_;
v___y_4731_ = v___x_4757_;
v___y_4732_ = v___y_4755_;
v___y_4733_ = v___x_4758_;
v___y_4734_ = v___y_4754_;
v___y_4735_ = v___y_4750_;
v_sym_4736_ = v___x_4765_;
goto v___jp_4724_;
}
}
else
{
lean_object* v___x_4766_; 
lean_dec(v___x_4760_);
v___x_4766_ = lean_box(0);
v___y_4725_ = v___y_4756_;
v___y_4726_ = v___y_4749_;
v___y_4727_ = v___y_4752_;
v___y_4728_ = v___y_4753_;
v___y_4729_ = v___y_4751_;
v___y_4730_ = v_expensive_4748_;
v___y_4731_ = v___x_4757_;
v___y_4732_ = v___y_4755_;
v___y_4733_ = v___x_4758_;
v___y_4734_ = v___y_4754_;
v___y_4735_ = v___y_4750_;
v_sym_4736_ = v___x_4766_;
goto v___jp_4724_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1___boxed(lean_object* v_x_4774_, lean_object* v_a_4775_, lean_object* v_a_4776_, lean_object* v_a_4777_, lean_object* v_a_4778_, lean_object* v_a_4779_, lean_object* v_a_4780_, lean_object* v_a_4781_, lean_object* v_a_4782_, lean_object* v_a_4783_){
_start:
{
lean_object* v_res_4784_; 
v_res_4784_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1(v_x_4774_, v_a_4775_, v_a_4776_, v_a_4777_, v_a_4778_, v_a_4779_, v_a_4780_, v_a_4781_, v_a_4782_);
lean_dec(v_a_4782_);
lean_dec_ref(v_a_4781_);
lean_dec(v_a_4780_);
lean_dec_ref(v_a_4779_);
lean_dec(v_a_4778_);
lean_dec_ref(v_a_4777_);
lean_dec(v_a_4776_);
lean_dec_ref(v_a_4775_);
return v_res_4784_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__5(void){
_start:
{
lean_object* v___x_4798_; lean_object* v___x_4799_; lean_object* v___x_4800_; lean_object* v___x_4801_; 
v___x_4798_ = l_Lean_Parser_Tactic_optConfig;
v___x_4799_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__4));
v___x_4800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4801_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4801_, 0, v___x_4800_);
lean_ctor_set(v___x_4801_, 1, v___x_4799_);
lean_ctor_set(v___x_4801_, 2, v___x_4798_);
return v___x_4801_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__6(void){
_start:
{
lean_object* v___x_4802_; lean_object* v___x_4803_; lean_object* v___x_4804_; lean_object* v___x_4805_; 
v___x_4802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__16));
v___x_4803_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__5, &lp_mathlib_Mathlib_Tactic_convertTo___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__5);
v___x_4804_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4805_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4805_, 0, v___x_4804_);
lean_ctor_set(v___x_4805_, 1, v___x_4803_);
lean_ctor_set(v___x_4805_, 2, v___x_4802_);
return v___x_4805_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__7(void){
_start:
{
lean_object* v___x_4806_; lean_object* v___x_4807_; lean_object* v___x_4808_; lean_object* v___x_4809_; 
v___x_4806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__20));
v___x_4807_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__6, &lp_mathlib_Mathlib_Tactic_convertTo___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__6);
v___x_4808_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4809_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4809_, 0, v___x_4808_);
lean_ctor_set(v___x_4809_, 1, v___x_4807_);
lean_ctor_set(v___x_4809_, 2, v___x_4806_);
return v___x_4809_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__8(void){
_start:
{
lean_object* v___x_4810_; lean_object* v___x_4811_; lean_object* v___x_4812_; lean_object* v___x_4813_; 
v___x_4810_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__24));
v___x_4811_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__7, &lp_mathlib_Mathlib_Tactic_convertTo___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__7);
v___x_4812_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4813_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4813_, 0, v___x_4812_);
lean_ctor_set(v___x_4813_, 1, v___x_4811_);
lean_ctor_set(v___x_4813_, 2, v___x_4810_);
return v___x_4813_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__9(void){
_start:
{
lean_object* v___x_4814_; lean_object* v___x_4815_; lean_object* v___x_4816_; lean_object* v___x_4817_; 
v___x_4814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__32));
v___x_4815_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__8, &lp_mathlib_Mathlib_Tactic_convertTo___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__8);
v___x_4816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4817_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4817_, 0, v___x_4816_);
lean_ctor_set(v___x_4817_, 1, v___x_4815_);
lean_ctor_set(v___x_4817_, 2, v___x_4814_);
return v___x_4817_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__10(void){
_start:
{
lean_object* v___x_4818_; lean_object* v___x_4819_; lean_object* v___x_4820_; lean_object* v___x_4821_; 
v___x_4818_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__48));
v___x_4819_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__9, &lp_mathlib_Mathlib_Tactic_convertTo___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__9);
v___x_4820_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4821_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4821_, 0, v___x_4820_);
lean_ctor_set(v___x_4821_, 1, v___x_4819_);
lean_ctor_set(v___x_4821_, 2, v___x_4818_);
return v___x_4821_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__11(void){
_start:
{
lean_object* v___x_4822_; lean_object* v___x_4823_; lean_object* v___x_4824_; 
v___x_4822_ = l_Lean_Parser_Tactic_location;
v___x_4823_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__8));
v___x_4824_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4824_, 0, v___x_4823_);
lean_ctor_set(v___x_4824_, 1, v___x_4822_);
return v___x_4824_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__12(void){
_start:
{
lean_object* v___x_4825_; lean_object* v___x_4826_; lean_object* v___x_4827_; lean_object* v___x_4828_; 
v___x_4825_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__11, &lp_mathlib_Mathlib_Tactic_convertTo___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__11);
v___x_4826_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__10, &lp_mathlib_Mathlib_Tactic_convertTo___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__10);
v___x_4827_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4828_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4828_, 0, v___x_4827_);
lean_ctor_set(v___x_4828_, 1, v___x_4826_);
lean_ctor_set(v___x_4828_, 2, v___x_4825_);
return v___x_4828_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__13(void){
_start:
{
lean_object* v___x_4829_; lean_object* v___x_4830_; lean_object* v___x_4831_; lean_object* v___x_4832_; 
v___x_4829_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__12, &lp_mathlib_Mathlib_Tactic_convertTo___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__12);
v___x_4830_ = lean_unsigned_to_nat(1022u);
v___x_4831_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__1));
v___x_4832_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4832_, 0, v___x_4831_);
lean_ctor_set(v___x_4832_, 1, v___x_4830_);
lean_ctor_set(v___x_4832_, 2, v___x_4829_);
return v___x_4832_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convertTo(void){
_start:
{
lean_object* v___x_4833_; 
v___x_4833_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__13, &lp_mathlib_Mathlib_Tactic_convertTo___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__13);
return v___x_4833_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3(void){
_start:
{
lean_object* v___x_4842_; lean_object* v___x_4843_; lean_object* v___x_4844_; lean_object* v___x_4845_; 
v___x_4842_ = l_Lean_Parser_Tactic_optConfig;
v___x_4843_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__2));
v___x_4844_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4845_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4845_, 0, v___x_4844_);
lean_ctor_set(v___x_4845_, 1, v___x_4843_);
lean_ctor_set(v___x_4845_, 2, v___x_4842_);
return v___x_4845_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4(void){
_start:
{
lean_object* v___x_4846_; lean_object* v___x_4847_; lean_object* v___x_4848_; lean_object* v___x_4849_; 
v___x_4846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__16));
v___x_4847_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__3);
v___x_4848_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4849_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4849_, 0, v___x_4848_);
lean_ctor_set(v___x_4849_, 1, v___x_4847_);
lean_ctor_set(v___x_4849_, 2, v___x_4846_);
return v___x_4849_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5(void){
_start:
{
lean_object* v___x_4850_; lean_object* v___x_4851_; lean_object* v___x_4852_; lean_object* v___x_4853_; 
v___x_4850_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__20));
v___x_4851_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__4);
v___x_4852_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4853_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4853_, 0, v___x_4852_);
lean_ctor_set(v___x_4853_, 1, v___x_4851_);
lean_ctor_set(v___x_4853_, 2, v___x_4850_);
return v___x_4853_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6(void){
_start:
{
lean_object* v___x_4854_; lean_object* v___x_4855_; lean_object* v___x_4856_; lean_object* v___x_4857_; 
v___x_4854_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__24));
v___x_4855_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__5);
v___x_4856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4857_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4857_, 0, v___x_4856_);
lean_ctor_set(v___x_4857_, 1, v___x_4855_);
lean_ctor_set(v___x_4857_, 2, v___x_4854_);
return v___x_4857_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7(void){
_start:
{
lean_object* v___x_4858_; lean_object* v___x_4859_; lean_object* v___x_4860_; lean_object* v___x_4861_; 
v___x_4858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__32));
v___x_4859_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__6);
v___x_4860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4861_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4861_, 0, v___x_4860_);
lean_ctor_set(v___x_4861_, 1, v___x_4859_);
lean_ctor_set(v___x_4861_, 2, v___x_4858_);
return v___x_4861_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8(void){
_start:
{
lean_object* v___x_4862_; lean_object* v___x_4863_; lean_object* v___x_4864_; lean_object* v___x_4865_; 
v___x_4862_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__48));
v___x_4863_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__7);
v___x_4864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4865_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4865_, 0, v___x_4864_);
lean_ctor_set(v___x_4865_, 1, v___x_4863_);
lean_ctor_set(v___x_4865_, 2, v___x_4862_);
return v___x_4865_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9(void){
_start:
{
lean_object* v___x_4866_; lean_object* v___x_4867_; lean_object* v___x_4868_; lean_object* v___x_4869_; 
v___x_4866_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convertTo___closed__11, &lp_mathlib_Mathlib_Tactic_convertTo___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_convertTo___closed__11);
v___x_4867_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__8);
v___x_4868_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__5));
v___x_4869_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4869_, 0, v___x_4868_);
lean_ctor_set(v___x_4869_, 1, v___x_4867_);
lean_ctor_set(v___x_4869_, 2, v___x_4866_);
return v___x_4869_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10(void){
_start:
{
lean_object* v___x_4870_; lean_object* v___x_4871_; lean_object* v___x_4872_; lean_object* v___x_4873_; 
v___x_4870_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__9);
v___x_4871_ = lean_unsigned_to_nat(1022u);
v___x_4872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1));
v___x_4873_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4873_, 0, v___x_4872_);
lean_ctor_set(v___x_4873_, 1, v___x_4871_);
lean_ctor_set(v___x_4873_, 2, v___x_4870_);
return v___x_4873_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_convert__to_x21(void){
_start:
{
lean_object* v___x_4874_; 
v___x_4874_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10, &lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__10);
return v___x_4874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert__to_x21__1(lean_object* v_x_4875_, lean_object* v_a_4876_, lean_object* v_a_4877_){
_start:
{
lean_object* v___x_4878_; uint8_t v___x_4879_; 
v___x_4878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1));
lean_inc(v_x_4875_);
v___x_4879_ = l_Lean_Syntax_isOfKind(v_x_4875_, v___x_4878_);
if (v___x_4879_ == 0)
{
lean_object* v___x_4880_; lean_object* v___x_4881_; 
lean_dec(v_x_4875_);
v___x_4880_ = lean_box(1);
v___x_4881_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4881_, 0, v___x_4880_);
lean_ctor_set(v___x_4881_, 1, v_a_4877_);
return v___x_4881_;
}
else
{
lean_object* v___x_4882_; lean_object* v___x_4883_; lean_object* v___x_4884_; lean_object* v___y_4886_; lean_object* v___y_4887_; lean_object* v___y_4888_; lean_object* v___y_4889_; lean_object* v___y_4890_; lean_object* v___y_4891_; lean_object* v___y_4892_; lean_object* v___y_4893_; lean_object* v___y_4894_; lean_object* v___y_4895_; lean_object* v___y_4896_; lean_object* v___y_4897_; lean_object* v___y_4903_; lean_object* v___y_4904_; lean_object* v___y_4905_; lean_object* v___y_4906_; lean_object* v___y_4907_; lean_object* v___y_4908_; lean_object* v___y_4909_; lean_object* v___y_4910_; lean_object* v___y_4911_; lean_object* v___y_4912_; lean_object* v___y_4913_; lean_object* v___y_4914_; lean_object* v___y_4921_; lean_object* v___y_4922_; lean_object* v___y_4923_; lean_object* v___y_4924_; lean_object* v___y_4925_; lean_object* v___y_4926_; lean_object* v___y_4927_; lean_object* v___y_4928_; lean_object* v___y_4929_; lean_object* v___y_4930_; lean_object* v___y_4931_; lean_object* v___y_4932_; lean_object* v___y_4942_; lean_object* v___y_4943_; lean_object* v___y_4944_; lean_object* v___y_4945_; lean_object* v___y_4946_; lean_object* v___y_4947_; lean_object* v___y_4948_; lean_object* v___y_4949_; lean_object* v___y_4950_; lean_object* v___y_4951_; lean_object* v___y_4952_; lean_object* v___y_4953_; lean_object* v___y_4962_; lean_object* v___y_4963_; lean_object* v___y_4964_; lean_object* v___y_4965_; lean_object* v_loc_4966_; lean_object* v___y_4967_; lean_object* v___y_4968_; lean_object* v___y_4987_; lean_object* v___y_4988_; lean_object* v___y_4989_; lean_object* v_w_4990_; lean_object* v___y_4991_; lean_object* v___y_4992_; lean_object* v___x_5002_; lean_object* v___y_5004_; lean_object* v___y_5005_; lean_object* v_n_5006_; lean_object* v___y_5007_; lean_object* v___y_5008_; lean_object* v_l_5023_; lean_object* v___y_5024_; lean_object* v___y_5025_; lean_object* v___x_5037_; uint8_t v___x_5038_; 
v___x_4882_ = lean_unsigned_to_nat(0u);
v___x_4883_ = lean_unsigned_to_nat(1u);
v___x_4884_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_4883_);
v___x_5002_ = lean_unsigned_to_nat(2u);
v___x_5037_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_5002_);
v___x_5038_ = l_Lean_Syntax_isNone(v___x_5037_);
if (v___x_5038_ == 0)
{
uint8_t v___x_5039_; 
lean_inc(v___x_5037_);
v___x_5039_ = l_Lean_Syntax_matchesNull(v___x_5037_, v___x_4883_);
if (v___x_5039_ == 0)
{
lean_object* v___x_5040_; lean_object* v___x_5041_; 
lean_dec(v___x_5037_);
lean_dec(v___x_4884_);
lean_dec(v_x_4875_);
v___x_5040_ = lean_box(1);
v___x_5041_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5041_, 0, v___x_5040_);
lean_ctor_set(v___x_5041_, 1, v_a_4877_);
return v___x_5041_;
}
else
{
lean_object* v_l_5042_; lean_object* v___x_5043_; 
v_l_5042_ = l_Lean_Syntax_getArg(v___x_5037_, v___x_4882_);
lean_dec(v___x_5037_);
v___x_5043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5043_, 0, v_l_5042_);
v_l_5023_ = v___x_5043_;
v___y_5024_ = v_a_4876_;
v___y_5025_ = v_a_4877_;
goto v___jp_5022_;
}
}
else
{
lean_object* v___x_5044_; 
lean_dec(v___x_5037_);
v___x_5044_ = lean_box(0);
v_l_5023_ = v___x_5044_;
v___y_5024_ = v_a_4876_;
v___y_5025_ = v_a_4877_;
goto v___jp_5022_;
}
v___jp_4885_:
{
lean_object* v___x_4898_; lean_object* v___x_4899_; lean_object* v___x_4900_; lean_object* v___x_4901_; 
lean_inc_ref(v___y_4894_);
v___x_4898_ = l_Array_append___redArg(v___y_4894_, v___y_4897_);
lean_dec_ref(v___y_4897_);
lean_inc(v___y_4889_);
v___x_4899_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4899_, 0, v___y_4889_);
lean_ctor_set(v___x_4899_, 1, v___y_4891_);
lean_ctor_set(v___x_4899_, 2, v___x_4898_);
lean_inc(v___y_4890_);
v___x_4900_ = l_Lean_Syntax_node8(v___y_4889_, v___y_4890_, v___y_4893_, v___y_4886_, v___x_4884_, v___y_4888_, v___y_4896_, v___y_4892_, v___y_4895_, v___x_4899_);
v___x_4901_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4901_, 0, v___x_4900_);
lean_ctor_set(v___x_4901_, 1, v___y_4887_);
return v___x_4901_;
}
v___jp_4902_:
{
lean_object* v___x_4915_; lean_object* v___x_4916_; 
lean_inc_ref(v___y_4912_);
v___x_4915_ = l_Array_append___redArg(v___y_4912_, v___y_4914_);
lean_dec_ref(v___y_4914_);
lean_inc(v___y_4908_);
lean_inc(v___y_4906_);
v___x_4916_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4916_, 0, v___y_4906_);
lean_ctor_set(v___x_4916_, 1, v___y_4908_);
lean_ctor_set(v___x_4916_, 2, v___x_4915_);
if (lean_obj_tag(v___y_4909_) == 1)
{
lean_object* v_val_4917_; lean_object* v___x_4918_; 
v_val_4917_ = lean_ctor_get(v___y_4909_, 0);
lean_inc(v_val_4917_);
lean_dec_ref_known(v___y_4909_, 1);
v___x_4918_ = l_Array_mkArray1___redArg(v_val_4917_);
v___y_4886_ = v___y_4903_;
v___y_4887_ = v___y_4904_;
v___y_4888_ = v___y_4905_;
v___y_4889_ = v___y_4906_;
v___y_4890_ = v___y_4907_;
v___y_4891_ = v___y_4908_;
v___y_4892_ = v___y_4910_;
v___y_4893_ = v___y_4911_;
v___y_4894_ = v___y_4912_;
v___y_4895_ = v___x_4916_;
v___y_4896_ = v___y_4913_;
v___y_4897_ = v___x_4918_;
goto v___jp_4885_;
}
else
{
lean_object* v___x_4919_; 
lean_dec(v___y_4909_);
v___x_4919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4886_ = v___y_4903_;
v___y_4887_ = v___y_4904_;
v___y_4888_ = v___y_4905_;
v___y_4889_ = v___y_4906_;
v___y_4890_ = v___y_4907_;
v___y_4891_ = v___y_4908_;
v___y_4892_ = v___y_4910_;
v___y_4893_ = v___y_4911_;
v___y_4894_ = v___y_4912_;
v___y_4895_ = v___x_4916_;
v___y_4896_ = v___y_4913_;
v___y_4897_ = v___x_4919_;
goto v___jp_4885_;
}
}
v___jp_4920_:
{
lean_object* v___x_4933_; lean_object* v___x_4934_; 
lean_inc_ref(v___y_4930_);
v___x_4933_ = l_Array_append___redArg(v___y_4930_, v___y_4932_);
lean_dec_ref(v___y_4932_);
lean_inc(v___y_4926_);
lean_inc(v___y_4924_);
v___x_4934_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4934_, 0, v___y_4924_);
lean_ctor_set(v___x_4934_, 1, v___y_4926_);
lean_ctor_set(v___x_4934_, 2, v___x_4933_);
if (lean_obj_tag(v___y_4927_) == 1)
{
lean_object* v_val_4935_; lean_object* v___x_4936_; lean_object* v___x_4937_; lean_object* v___x_4938_; lean_object* v___x_4939_; 
v_val_4935_ = lean_ctor_get(v___y_4927_, 0);
lean_inc(v_val_4935_);
lean_dec_ref_known(v___y_4927_, 1);
v___x_4936_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__0));
lean_inc_n(v___y_4924_, 2);
v___x_4937_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4937_, 0, v___y_4924_);
lean_ctor_set(v___x_4937_, 1, v___x_4936_);
lean_inc(v___y_4926_);
v___x_4938_ = l_Lean_Syntax_node1(v___y_4924_, v___y_4926_, v_val_4935_);
v___x_4939_ = l_Array_mkArray2___redArg(v___x_4937_, v___x_4938_);
v___y_4903_ = v___y_4921_;
v___y_4904_ = v___y_4922_;
v___y_4905_ = v___y_4923_;
v___y_4906_ = v___y_4924_;
v___y_4907_ = v___y_4925_;
v___y_4908_ = v___y_4926_;
v___y_4909_ = v___y_4928_;
v___y_4910_ = v___x_4934_;
v___y_4911_ = v___y_4929_;
v___y_4912_ = v___y_4930_;
v___y_4913_ = v___y_4931_;
v___y_4914_ = v___x_4939_;
goto v___jp_4902_;
}
else
{
lean_object* v___x_4940_; 
lean_dec(v___y_4927_);
v___x_4940_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4903_ = v___y_4921_;
v___y_4904_ = v___y_4922_;
v___y_4905_ = v___y_4923_;
v___y_4906_ = v___y_4924_;
v___y_4907_ = v___y_4925_;
v___y_4908_ = v___y_4926_;
v___y_4909_ = v___y_4928_;
v___y_4910_ = v___x_4934_;
v___y_4911_ = v___y_4929_;
v___y_4912_ = v___y_4930_;
v___y_4913_ = v___y_4931_;
v___y_4914_ = v___x_4940_;
goto v___jp_4902_;
}
}
v___jp_4941_:
{
lean_object* v___x_4954_; lean_object* v___x_4955_; 
lean_inc_ref(v___y_4951_);
v___x_4954_ = l_Array_append___redArg(v___y_4951_, v___y_4953_);
lean_dec_ref(v___y_4953_);
lean_inc(v___y_4946_);
lean_inc(v___y_4944_);
v___x_4955_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4955_, 0, v___y_4944_);
lean_ctor_set(v___x_4955_, 1, v___y_4946_);
lean_ctor_set(v___x_4955_, 2, v___x_4954_);
if (lean_obj_tag(v___y_4949_) == 1)
{
lean_object* v_val_4956_; lean_object* v___x_4957_; lean_object* v___x_4958_; lean_object* v___x_4959_; 
v_val_4956_ = lean_ctor_get(v___y_4949_, 0);
lean_inc(v_val_4956_);
lean_dec_ref_known(v___y_4949_, 1);
v___x_4957_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2));
lean_inc(v___y_4944_);
v___x_4958_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4958_, 0, v___y_4944_);
lean_ctor_set(v___x_4958_, 1, v___x_4957_);
v___x_4959_ = l_Array_mkArray2___redArg(v___x_4958_, v_val_4956_);
v___y_4921_ = v___y_4942_;
v___y_4922_ = v___y_4943_;
v___y_4923_ = v___x_4955_;
v___y_4924_ = v___y_4944_;
v___y_4925_ = v___y_4945_;
v___y_4926_ = v___y_4946_;
v___y_4927_ = v___y_4947_;
v___y_4928_ = v___y_4948_;
v___y_4929_ = v___y_4950_;
v___y_4930_ = v___y_4951_;
v___y_4931_ = v___y_4952_;
v___y_4932_ = v___x_4959_;
goto v___jp_4920_;
}
else
{
lean_object* v___x_4960_; 
lean_dec(v___y_4949_);
v___x_4960_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4921_ = v___y_4942_;
v___y_4922_ = v___y_4943_;
v___y_4923_ = v___x_4955_;
v___y_4924_ = v___y_4944_;
v___y_4925_ = v___y_4945_;
v___y_4926_ = v___y_4946_;
v___y_4927_ = v___y_4947_;
v___y_4928_ = v___y_4948_;
v___y_4929_ = v___y_4950_;
v___y_4930_ = v___y_4951_;
v___y_4931_ = v___y_4952_;
v___y_4932_ = v___x_4960_;
goto v___jp_4920_;
}
}
v___jp_4961_:
{
lean_object* v_ref_4969_; uint8_t v___x_4970_; lean_object* v___x_4971_; lean_object* v___x_4972_; lean_object* v___x_4973_; lean_object* v___x_4974_; lean_object* v___x_4975_; lean_object* v___x_4976_; lean_object* v___x_4977_; lean_object* v___x_4978_; lean_object* v___x_4979_; 
v_ref_4969_ = lean_ctor_get(v___y_4967_, 5);
v___x_4970_ = 0;
v___x_4971_ = l_Lean_SourceInfo_fromRef(v_ref_4969_, v___x_4970_);
v___x_4972_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__1));
v___x_4973_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__2));
lean_inc_n(v___x_4971_, 3);
v___x_4974_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4974_, 0, v___x_4971_);
lean_ctor_set(v___x_4974_, 1, v___x_4973_);
v___x_4975_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4));
v___x_4976_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__9));
v___x_4977_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4977_, 0, v___x_4971_);
lean_ctor_set(v___x_4977_, 1, v___x_4976_);
v___x_4978_ = l_Lean_Syntax_node1(v___x_4971_, v___x_4975_, v___x_4977_);
v___x_4979_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5);
if (lean_obj_tag(v___y_4962_) == 1)
{
lean_object* v_val_4980_; lean_object* v___x_4981_; lean_object* v___x_4982_; lean_object* v___x_4983_; lean_object* v___x_4984_; 
v_val_4980_ = lean_ctor_get(v___y_4962_, 0);
lean_inc(v_val_4980_);
lean_dec_ref_known(v___y_4962_, 1);
v___x_4981_ = l_Lean_SourceInfo_fromRef(v_val_4980_, v___x_4879_);
lean_dec(v_val_4980_);
v___x_4982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__6));
v___x_4983_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4983_, 0, v___x_4981_);
lean_ctor_set(v___x_4983_, 1, v___x_4982_);
v___x_4984_ = l_Array_mkArray1___redArg(v___x_4983_);
v___y_4942_ = v___x_4978_;
v___y_4943_ = v___y_4968_;
v___y_4944_ = v___x_4971_;
v___y_4945_ = v___x_4972_;
v___y_4946_ = v___x_4975_;
v___y_4947_ = v___y_4963_;
v___y_4948_ = v_loc_4966_;
v___y_4949_ = v___y_4964_;
v___y_4950_ = v___x_4974_;
v___y_4951_ = v___x_4979_;
v___y_4952_ = v___y_4965_;
v___y_4953_ = v___x_4984_;
goto v___jp_4941_;
}
else
{
lean_object* v___x_4985_; 
lean_dec(v___y_4962_);
v___x_4985_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_4942_ = v___x_4978_;
v___y_4943_ = v___y_4968_;
v___y_4944_ = v___x_4971_;
v___y_4945_ = v___x_4972_;
v___y_4946_ = v___x_4975_;
v___y_4947_ = v___y_4963_;
v___y_4948_ = v_loc_4966_;
v___y_4949_ = v___y_4964_;
v___y_4950_ = v___x_4974_;
v___y_4951_ = v___x_4979_;
v___y_4952_ = v___y_4965_;
v___y_4953_ = v___x_4985_;
goto v___jp_4941_;
}
}
v___jp_4986_:
{
lean_object* v___x_4993_; lean_object* v___x_4994_; uint8_t v___x_4995_; 
v___x_4993_ = lean_unsigned_to_nat(6u);
v___x_4994_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_4993_);
lean_dec(v_x_4875_);
v___x_4995_ = l_Lean_Syntax_isNone(v___x_4994_);
if (v___x_4995_ == 0)
{
uint8_t v___x_4996_; 
lean_inc(v___x_4994_);
v___x_4996_ = l_Lean_Syntax_matchesNull(v___x_4994_, v___x_4883_);
if (v___x_4996_ == 0)
{
lean_object* v___x_4997_; lean_object* v___x_4998_; 
lean_dec(v___x_4994_);
lean_dec(v_w_4990_);
lean_dec(v___y_4989_);
lean_dec(v___y_4988_);
lean_dec(v___y_4987_);
lean_dec(v___x_4884_);
v___x_4997_ = lean_box(1);
v___x_4998_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4998_, 0, v___x_4997_);
lean_ctor_set(v___x_4998_, 1, v___y_4992_);
return v___x_4998_;
}
else
{
lean_object* v_loc_4999_; lean_object* v___x_5000_; 
v_loc_4999_ = l_Lean_Syntax_getArg(v___x_4994_, v___x_4882_);
lean_dec(v___x_4994_);
v___x_5000_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5000_, 0, v_loc_4999_);
v___y_4962_ = v___y_4987_;
v___y_4963_ = v_w_4990_;
v___y_4964_ = v___y_4988_;
v___y_4965_ = v___y_4989_;
v_loc_4966_ = v___x_5000_;
v___y_4967_ = v___y_4991_;
v___y_4968_ = v___y_4992_;
goto v___jp_4961_;
}
}
else
{
lean_object* v___x_5001_; 
lean_dec(v___x_4994_);
v___x_5001_ = lean_box(0);
v___y_4962_ = v___y_4987_;
v___y_4963_ = v_w_4990_;
v___y_4964_ = v___y_4988_;
v___y_4965_ = v___y_4989_;
v_loc_4966_ = v___x_5001_;
v___y_4967_ = v___y_4991_;
v___y_4968_ = v___y_4992_;
goto v___jp_4961_;
}
}
v___jp_5003_:
{
lean_object* v___x_5009_; lean_object* v___x_5010_; uint8_t v___x_5011_; 
v___x_5009_ = lean_unsigned_to_nat(5u);
v___x_5010_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_5009_);
v___x_5011_ = l_Lean_Syntax_isNone(v___x_5010_);
if (v___x_5011_ == 0)
{
uint8_t v___x_5012_; 
lean_inc(v___x_5010_);
v___x_5012_ = l_Lean_Syntax_matchesNull(v___x_5010_, v___x_5002_);
if (v___x_5012_ == 0)
{
lean_object* v___x_5013_; lean_object* v___x_5014_; 
lean_dec(v___x_5010_);
lean_dec(v_n_5006_);
lean_dec(v___y_5005_);
lean_dec(v___y_5004_);
lean_dec(v___x_4884_);
lean_dec(v_x_4875_);
v___x_5013_ = lean_box(1);
v___x_5014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5014_, 0, v___x_5013_);
lean_ctor_set(v___x_5014_, 1, v___y_5008_);
return v___x_5014_;
}
else
{
lean_object* v___x_5015_; uint8_t v___x_5016_; 
v___x_5015_ = l_Lean_Syntax_getArg(v___x_5010_, v___x_4883_);
lean_dec(v___x_5010_);
lean_inc(v___x_5015_);
v___x_5016_ = l_Lean_Syntax_matchesNull(v___x_5015_, v___x_4883_);
if (v___x_5016_ == 0)
{
lean_object* v___x_5017_; lean_object* v___x_5018_; 
lean_dec(v___x_5015_);
lean_dec(v_n_5006_);
lean_dec(v___y_5005_);
lean_dec(v___y_5004_);
lean_dec(v___x_4884_);
lean_dec(v_x_4875_);
v___x_5017_ = lean_box(1);
v___x_5018_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5018_, 0, v___x_5017_);
lean_ctor_set(v___x_5018_, 1, v___y_5008_);
return v___x_5018_;
}
else
{
lean_object* v_w_5019_; lean_object* v___x_5020_; 
v_w_5019_ = l_Lean_Syntax_getArg(v___x_5015_, v___x_4882_);
lean_dec(v___x_5015_);
v___x_5020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5020_, 0, v_w_5019_);
v___y_4987_ = v___y_5004_;
v___y_4988_ = v_n_5006_;
v___y_4989_ = v___y_5005_;
v_w_4990_ = v___x_5020_;
v___y_4991_ = v___y_5007_;
v___y_4992_ = v___y_5008_;
goto v___jp_4986_;
}
}
}
else
{
lean_object* v___x_5021_; 
lean_dec(v___x_5010_);
v___x_5021_ = lean_box(0);
v___y_4987_ = v___y_5004_;
v___y_4988_ = v_n_5006_;
v___y_4989_ = v___y_5005_;
v_w_4990_ = v___x_5021_;
v___y_4991_ = v___y_5007_;
v___y_4992_ = v___y_5008_;
goto v___jp_4986_;
}
}
v___jp_5022_:
{
lean_object* v___x_5026_; lean_object* v___x_5027_; lean_object* v___x_5028_; lean_object* v___x_5029_; uint8_t v___x_5030_; 
v___x_5026_ = lean_unsigned_to_nat(3u);
v___x_5027_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_5026_);
v___x_5028_ = lean_unsigned_to_nat(4u);
v___x_5029_ = l_Lean_Syntax_getArg(v_x_4875_, v___x_5028_);
v___x_5030_ = l_Lean_Syntax_isNone(v___x_5029_);
if (v___x_5030_ == 0)
{
uint8_t v___x_5031_; 
lean_inc(v___x_5029_);
v___x_5031_ = l_Lean_Syntax_matchesNull(v___x_5029_, v___x_5002_);
if (v___x_5031_ == 0)
{
lean_object* v___x_5032_; lean_object* v___x_5033_; 
lean_dec(v___x_5029_);
lean_dec(v___x_5027_);
lean_dec(v_l_5023_);
lean_dec(v___x_4884_);
lean_dec(v_x_4875_);
v___x_5032_ = lean_box(1);
v___x_5033_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5033_, 0, v___x_5032_);
lean_ctor_set(v___x_5033_, 1, v___y_5025_);
return v___x_5033_;
}
else
{
lean_object* v_n_5034_; lean_object* v___x_5035_; 
v_n_5034_ = l_Lean_Syntax_getArg(v___x_5029_, v___x_4883_);
lean_dec(v___x_5029_);
v___x_5035_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5035_, 0, v_n_5034_);
v___y_5004_ = v_l_5023_;
v___y_5005_ = v___x_5027_;
v_n_5006_ = v___x_5035_;
v___y_5007_ = v___y_5024_;
v___y_5008_ = v___y_5025_;
goto v___jp_5003_;
}
}
else
{
lean_object* v___x_5036_; 
lean_dec(v___x_5029_);
v___x_5036_ = lean_box(0);
v___y_5004_ = v_l_5023_;
v___y_5005_ = v___x_5027_;
v_n_5006_ = v___x_5036_;
v___y_5007_ = v___y_5024_;
v___y_5008_ = v___y_5025_;
goto v___jp_5003_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert__to_x21__1___boxed(lean_object* v_x_5045_, lean_object* v_a_5046_, lean_object* v_a_5047_){
_start:
{
lean_object* v_res_5048_; 
v_res_5048_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert__to_x21__1(v_x_5045_, v_a_5046_, v_a_5047_);
lean_dec_ref(v_a_5046_);
return v_res_5048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg(lean_object* v_msg_5049_, lean_object* v___y_5050_, lean_object* v___y_5051_, lean_object* v___y_5052_, lean_object* v___y_5053_){
_start:
{
lean_object* v_ref_5055_; lean_object* v___x_5056_; lean_object* v_a_5057_; lean_object* v___x_5059_; uint8_t v_isShared_5060_; uint8_t v_isSharedCheck_5065_; 
v_ref_5055_ = lean_ctor_get(v___y_5052_, 5);
v___x_5056_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Convert_0__instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_5049_, v___y_5050_, v___y_5051_, v___y_5052_, v___y_5053_);
v_a_5057_ = lean_ctor_get(v___x_5056_, 0);
v_isSharedCheck_5065_ = !lean_is_exclusive(v___x_5056_);
if (v_isSharedCheck_5065_ == 0)
{
v___x_5059_ = v___x_5056_;
v_isShared_5060_ = v_isSharedCheck_5065_;
goto v_resetjp_5058_;
}
else
{
lean_inc(v_a_5057_);
lean_dec(v___x_5056_);
v___x_5059_ = lean_box(0);
v_isShared_5060_ = v_isSharedCheck_5065_;
goto v_resetjp_5058_;
}
v_resetjp_5058_:
{
lean_object* v___x_5061_; lean_object* v___x_5063_; 
lean_inc(v_ref_5055_);
v___x_5061_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5061_, 0, v_ref_5055_);
lean_ctor_set(v___x_5061_, 1, v_a_5057_);
if (v_isShared_5060_ == 0)
{
lean_ctor_set_tag(v___x_5059_, 1);
lean_ctor_set(v___x_5059_, 0, v___x_5061_);
v___x_5063_ = v___x_5059_;
goto v_reusejp_5062_;
}
else
{
lean_object* v_reuseFailAlloc_5064_; 
v_reuseFailAlloc_5064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5064_, 0, v___x_5061_);
v___x_5063_ = v_reuseFailAlloc_5064_;
goto v_reusejp_5062_;
}
v_reusejp_5062_:
{
return v___x_5063_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg___boxed(lean_object* v_msg_5066_, lean_object* v___y_5067_, lean_object* v___y_5068_, lean_object* v___y_5069_, lean_object* v___y_5070_, lean_object* v___y_5071_){
_start:
{
lean_object* v_res_5072_; 
v_res_5072_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg(v_msg_5066_, v___y_5067_, v___y_5068_, v___y_5069_, v___y_5070_);
lean_dec(v___y_5070_);
lean_dec_ref(v___y_5069_);
lean_dec(v___y_5068_);
lean_dec_ref(v___y_5067_);
return v_res_5072_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_5074_; lean_object* v___x_5075_; 
v___x_5074_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__0));
v___x_5075_ = l_Lean_stringToMessageData(v___x_5074_);
return v___x_5075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0(lean_object* v_x_5076_, lean_object* v___y_5077_, lean_object* v___y_5078_, lean_object* v___y_5079_, lean_object* v___y_5080_, lean_object* v___y_5081_, lean_object* v___y_5082_, lean_object* v___y_5083_, lean_object* v___y_5084_){
_start:
{
lean_object* v___x_5086_; lean_object* v___x_5087_; 
v___x_5086_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___closed__1);
v___x_5087_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg(v___x_5086_, v___y_5081_, v___y_5082_, v___y_5083_, v___y_5084_);
return v___x_5087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0___boxed(lean_object* v_x_5088_, lean_object* v___y_5089_, lean_object* v___y_5090_, lean_object* v___y_5091_, lean_object* v___y_5092_, lean_object* v___y_5093_, lean_object* v___y_5094_, lean_object* v___y_5095_, lean_object* v___y_5096_, lean_object* v___y_5097_){
_start:
{
lean_object* v_res_5098_; 
v_res_5098_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__0(v_x_5088_, v___y_5089_, v___y_5090_, v___y_5091_, v___y_5092_, v___y_5093_, v___y_5094_, v___y_5095_, v___y_5096_);
lean_dec(v___y_5096_);
lean_dec_ref(v___y_5095_);
lean_dec(v___y_5094_);
lean_dec_ref(v___y_5093_);
lean_dec(v___y_5092_);
lean_dec_ref(v___y_5091_);
lean_dec(v___y_5090_);
lean_dec_ref(v___y_5089_);
lean_dec(v_x_5088_);
return v_res_5098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1(lean_object* v___y_5099_, lean_object* v_fvarId_5100_, lean_object* v_fst_5101_, lean_object* v_a_5102_, lean_object* v___x_5103_, lean_object* v_snd_5104_, lean_object* v_sym_5105_, uint8_t v___x_5106_, lean_object* v___y_5107_, lean_object* v___y_5108_, lean_object* v___y_5109_, lean_object* v___y_5110_, lean_object* v___y_5111_, lean_object* v___y_5112_, lean_object* v___y_5113_, lean_object* v___y_5114_){
_start:
{
lean_object* v___x_5116_; 
v___x_5116_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5108_, v___y_5111_, v___y_5112_, v___y_5113_, v___y_5114_);
if (lean_obj_tag(v___x_5116_) == 0)
{
lean_object* v_a_5117_; uint8_t v___y_5119_; 
v_a_5117_ = lean_ctor_get(v___x_5116_, 0);
lean_inc(v_a_5117_);
lean_dec_ref_known(v___x_5116_, 1);
if (lean_obj_tag(v_sym_5105_) == 0)
{
uint8_t v___x_5151_; 
v___x_5151_ = 0;
v___y_5119_ = v___x_5151_;
goto v___jp_5118_;
}
else
{
v___y_5119_ = v___x_5106_;
goto v___jp_5118_;
}
v___jp_5118_:
{
lean_object* v___x_5120_; lean_object* v___x_5121_; 
v___x_5120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5120_, 0, v___y_5099_);
v___x_5121_ = lp_mathlib_Lean_MVarId_convertLocalDecl(v_a_5117_, v_fvarId_5100_, v_fst_5101_, v___y_5119_, v___x_5120_, v_a_5102_, v___x_5103_, v___y_5111_, v___y_5112_, v___y_5113_, v___y_5114_);
if (lean_obj_tag(v___x_5121_) == 0)
{
lean_object* v_a_5122_; lean_object* v_fst_5123_; lean_object* v_snd_5124_; lean_object* v___x_5126_; uint8_t v_isShared_5127_; uint8_t v_isSharedCheck_5142_; 
v_a_5122_ = lean_ctor_get(v___x_5121_, 0);
lean_inc(v_a_5122_);
lean_dec_ref_known(v___x_5121_, 1);
v_fst_5123_ = lean_ctor_get(v_a_5122_, 0);
v_snd_5124_ = lean_ctor_get(v_a_5122_, 1);
v_isSharedCheck_5142_ = !lean_is_exclusive(v_a_5122_);
if (v_isSharedCheck_5142_ == 0)
{
v___x_5126_ = v_a_5122_;
v_isShared_5127_ = v_isSharedCheck_5142_;
goto v_resetjp_5125_;
}
else
{
lean_inc(v_snd_5124_);
lean_inc(v_fst_5123_);
lean_dec(v_a_5122_);
v___x_5126_ = lean_box(0);
v_isShared_5127_ = v_isSharedCheck_5142_;
goto v_resetjp_5125_;
}
v_resetjp_5125_:
{
lean_object* v___x_5129_; 
if (v_isShared_5127_ == 0)
{
lean_ctor_set_tag(v___x_5126_, 1);
lean_ctor_set(v___x_5126_, 1, v_snd_5104_);
v___x_5129_ = v___x_5126_;
goto v_reusejp_5128_;
}
else
{
lean_object* v_reuseFailAlloc_5141_; 
v_reuseFailAlloc_5141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5141_, 0, v_fst_5123_);
lean_ctor_set(v_reuseFailAlloc_5141_, 1, v_snd_5104_);
v___x_5129_ = v_reuseFailAlloc_5141_;
goto v_reusejp_5128_;
}
v_reusejp_5128_:
{
lean_object* v___x_5130_; lean_object* v___x_5131_; 
v___x_5130_ = l_List_appendTR___redArg(v_snd_5124_, v___x_5129_);
v___x_5131_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_5130_, v___y_5108_, v___y_5111_, v___y_5112_, v___y_5113_, v___y_5114_);
if (lean_obj_tag(v___x_5131_) == 0)
{
lean_object* v___x_5133_; uint8_t v_isShared_5134_; uint8_t v_isSharedCheck_5139_; 
v_isSharedCheck_5139_ = !lean_is_exclusive(v___x_5131_);
if (v_isSharedCheck_5139_ == 0)
{
lean_object* v_unused_5140_; 
v_unused_5140_ = lean_ctor_get(v___x_5131_, 0);
lean_dec(v_unused_5140_);
v___x_5133_ = v___x_5131_;
v_isShared_5134_ = v_isSharedCheck_5139_;
goto v_resetjp_5132_;
}
else
{
lean_dec(v___x_5131_);
v___x_5133_ = lean_box(0);
v_isShared_5134_ = v_isSharedCheck_5139_;
goto v_resetjp_5132_;
}
v_resetjp_5132_:
{
lean_object* v___x_5135_; lean_object* v___x_5137_; 
v___x_5135_ = lean_box(0);
if (v_isShared_5134_ == 0)
{
lean_ctor_set(v___x_5133_, 0, v___x_5135_);
v___x_5137_ = v___x_5133_;
goto v_reusejp_5136_;
}
else
{
lean_object* v_reuseFailAlloc_5138_; 
v_reuseFailAlloc_5138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5138_, 0, v___x_5135_);
v___x_5137_ = v_reuseFailAlloc_5138_;
goto v_reusejp_5136_;
}
v_reusejp_5136_:
{
return v___x_5137_;
}
}
}
else
{
return v___x_5131_;
}
}
}
}
else
{
lean_object* v_a_5143_; lean_object* v___x_5145_; uint8_t v_isShared_5146_; uint8_t v_isSharedCheck_5150_; 
lean_dec(v_snd_5104_);
v_a_5143_ = lean_ctor_get(v___x_5121_, 0);
v_isSharedCheck_5150_ = !lean_is_exclusive(v___x_5121_);
if (v_isSharedCheck_5150_ == 0)
{
v___x_5145_ = v___x_5121_;
v_isShared_5146_ = v_isSharedCheck_5150_;
goto v_resetjp_5144_;
}
else
{
lean_inc(v_a_5143_);
lean_dec(v___x_5121_);
v___x_5145_ = lean_box(0);
v_isShared_5146_ = v_isSharedCheck_5150_;
goto v_resetjp_5144_;
}
v_resetjp_5144_:
{
lean_object* v___x_5148_; 
if (v_isShared_5146_ == 0)
{
v___x_5148_ = v___x_5145_;
goto v_reusejp_5147_;
}
else
{
lean_object* v_reuseFailAlloc_5149_; 
v_reuseFailAlloc_5149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5149_, 0, v_a_5143_);
v___x_5148_ = v_reuseFailAlloc_5149_;
goto v_reusejp_5147_;
}
v_reusejp_5147_:
{
return v___x_5148_;
}
}
}
}
}
else
{
lean_object* v_a_5152_; lean_object* v___x_5154_; uint8_t v_isShared_5155_; uint8_t v_isSharedCheck_5159_; 
lean_dec(v_snd_5104_);
lean_dec(v___x_5103_);
lean_dec_ref(v_a_5102_);
lean_dec_ref(v_fst_5101_);
lean_dec(v_fvarId_5100_);
lean_dec(v___y_5099_);
v_a_5152_ = lean_ctor_get(v___x_5116_, 0);
v_isSharedCheck_5159_ = !lean_is_exclusive(v___x_5116_);
if (v_isSharedCheck_5159_ == 0)
{
v___x_5154_ = v___x_5116_;
v_isShared_5155_ = v_isSharedCheck_5159_;
goto v_resetjp_5153_;
}
else
{
lean_inc(v_a_5152_);
lean_dec(v___x_5116_);
v___x_5154_ = lean_box(0);
v_isShared_5155_ = v_isSharedCheck_5159_;
goto v_resetjp_5153_;
}
v_resetjp_5153_:
{
lean_object* v___x_5157_; 
if (v_isShared_5155_ == 0)
{
v___x_5157_ = v___x_5154_;
goto v_reusejp_5156_;
}
else
{
lean_object* v_reuseFailAlloc_5158_; 
v_reuseFailAlloc_5158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5158_, 0, v_a_5152_);
v___x_5157_ = v_reuseFailAlloc_5158_;
goto v_reusejp_5156_;
}
v_reusejp_5156_:
{
return v___x_5157_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1___boxed(lean_object** _args){
lean_object* v___y_5160_ = _args[0];
lean_object* v_fvarId_5161_ = _args[1];
lean_object* v_fst_5162_ = _args[2];
lean_object* v_a_5163_ = _args[3];
lean_object* v___x_5164_ = _args[4];
lean_object* v_snd_5165_ = _args[5];
lean_object* v_sym_5166_ = _args[6];
lean_object* v___x_5167_ = _args[7];
lean_object* v___y_5168_ = _args[8];
lean_object* v___y_5169_ = _args[9];
lean_object* v___y_5170_ = _args[10];
lean_object* v___y_5171_ = _args[11];
lean_object* v___y_5172_ = _args[12];
lean_object* v___y_5173_ = _args[13];
lean_object* v___y_5174_ = _args[14];
lean_object* v___y_5175_ = _args[15];
lean_object* v___y_5176_ = _args[16];
_start:
{
uint8_t v___x_10807__boxed_5177_; lean_object* v_res_5178_; 
v___x_10807__boxed_5177_ = lean_unbox(v___x_5167_);
v_res_5178_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1(v___y_5160_, v_fvarId_5161_, v_fst_5162_, v_a_5163_, v___x_5164_, v_snd_5165_, v_sym_5166_, v___x_10807__boxed_5177_, v___y_5168_, v___y_5169_, v___y_5170_, v___y_5171_, v___y_5172_, v___y_5173_, v___y_5174_, v___y_5175_);
lean_dec(v___y_5175_);
lean_dec_ref(v___y_5174_);
lean_dec(v___y_5173_);
lean_dec_ref(v___y_5172_);
lean_dec(v___y_5171_);
lean_dec_ref(v___y_5170_);
lean_dec(v___y_5169_);
lean_dec_ref(v___y_5168_);
lean_dec(v_sym_5166_);
return v_res_5178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2(lean_object* v___x_5179_, lean_object* v___y_5180_, lean_object* v_a_5181_, lean_object* v___x_5182_, lean_object* v_sym_5183_, uint8_t v___x_5184_, lean_object* v_fvarId_5185_, lean_object* v___y_5186_, lean_object* v___y_5187_, lean_object* v___y_5188_, lean_object* v___y_5189_, lean_object* v___y_5190_, lean_object* v___y_5191_, lean_object* v___y_5192_, lean_object* v___y_5193_){
_start:
{
lean_object* v___x_5195_; 
lean_inc(v_fvarId_5185_);
v___x_5195_ = l_Lean_FVarId_getType___redArg(v_fvarId_5185_, v___y_5190_, v___y_5192_, v___y_5193_);
if (lean_obj_tag(v___x_5195_) == 0)
{
lean_object* v_a_5196_; lean_object* v___x_5197_; 
v_a_5196_ = lean_ctor_get(v___x_5195_, 0);
lean_inc(v_a_5196_);
lean_dec_ref_known(v___x_5195_, 1);
lean_inc(v___y_5193_);
lean_inc_ref(v___y_5192_);
lean_inc(v___y_5191_);
lean_inc_ref(v___y_5190_);
v___x_5197_ = lean_infer_type(v_a_5196_, v___y_5190_, v___y_5191_, v___y_5192_, v___y_5193_);
if (lean_obj_tag(v___x_5197_) == 0)
{
lean_object* v_a_5198_; lean_object* v___x_5199_; lean_object* v___x_5200_; 
v_a_5198_ = lean_ctor_get(v___x_5197_, 0);
lean_inc(v_a_5198_);
lean_dec_ref_known(v___x_5197_, 1);
v___x_5199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5199_, 0, v_a_5198_);
v___x_5200_ = lp_mathlib_Mathlib_Tactic_elabTermForConvert(v___x_5179_, v___x_5199_, v___y_5186_, v___y_5187_, v___y_5188_, v___y_5189_, v___y_5190_, v___y_5191_, v___y_5192_, v___y_5193_);
if (lean_obj_tag(v___x_5200_) == 0)
{
lean_object* v_a_5201_; lean_object* v_fst_5202_; lean_object* v_snd_5203_; lean_object* v___x_5204_; lean_object* v___f_5205_; lean_object* v___x_5206_; 
v_a_5201_ = lean_ctor_get(v___x_5200_, 0);
lean_inc(v_a_5201_);
lean_dec_ref_known(v___x_5200_, 1);
v_fst_5202_ = lean_ctor_get(v_a_5201_, 0);
lean_inc(v_fst_5202_);
v_snd_5203_ = lean_ctor_get(v_a_5201_, 1);
lean_inc(v_snd_5203_);
lean_dec(v_a_5201_);
v___x_5204_ = lean_box(v___x_5184_);
v___f_5205_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__1___boxed), 17, 8);
lean_closure_set(v___f_5205_, 0, v___y_5180_);
lean_closure_set(v___f_5205_, 1, v_fvarId_5185_);
lean_closure_set(v___f_5205_, 2, v_fst_5202_);
lean_closure_set(v___f_5205_, 3, v_a_5181_);
lean_closure_set(v___f_5205_, 4, v___x_5182_);
lean_closure_set(v___f_5205_, 5, v_snd_5203_);
lean_closure_set(v___f_5205_, 6, v_sym_5183_);
lean_closure_set(v___f_5205_, 7, v___x_5204_);
v___x_5206_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5205_, v___y_5186_, v___y_5187_, v___y_5188_, v___y_5189_, v___y_5190_, v___y_5191_, v___y_5192_, v___y_5193_);
return v___x_5206_;
}
else
{
lean_object* v_a_5207_; lean_object* v___x_5209_; uint8_t v_isShared_5210_; uint8_t v_isSharedCheck_5214_; 
lean_dec(v_fvarId_5185_);
lean_dec(v_sym_5183_);
lean_dec(v___x_5182_);
lean_dec_ref(v_a_5181_);
lean_dec(v___y_5180_);
v_a_5207_ = lean_ctor_get(v___x_5200_, 0);
v_isSharedCheck_5214_ = !lean_is_exclusive(v___x_5200_);
if (v_isSharedCheck_5214_ == 0)
{
v___x_5209_ = v___x_5200_;
v_isShared_5210_ = v_isSharedCheck_5214_;
goto v_resetjp_5208_;
}
else
{
lean_inc(v_a_5207_);
lean_dec(v___x_5200_);
v___x_5209_ = lean_box(0);
v_isShared_5210_ = v_isSharedCheck_5214_;
goto v_resetjp_5208_;
}
v_resetjp_5208_:
{
lean_object* v___x_5212_; 
if (v_isShared_5210_ == 0)
{
v___x_5212_ = v___x_5209_;
goto v_reusejp_5211_;
}
else
{
lean_object* v_reuseFailAlloc_5213_; 
v_reuseFailAlloc_5213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5213_, 0, v_a_5207_);
v___x_5212_ = v_reuseFailAlloc_5213_;
goto v_reusejp_5211_;
}
v_reusejp_5211_:
{
return v___x_5212_;
}
}
}
}
else
{
lean_object* v_a_5215_; lean_object* v___x_5217_; uint8_t v_isShared_5218_; uint8_t v_isSharedCheck_5222_; 
lean_dec(v_fvarId_5185_);
lean_dec(v_sym_5183_);
lean_dec(v___x_5182_);
lean_dec_ref(v_a_5181_);
lean_dec(v___y_5180_);
lean_dec(v___x_5179_);
v_a_5215_ = lean_ctor_get(v___x_5197_, 0);
v_isSharedCheck_5222_ = !lean_is_exclusive(v___x_5197_);
if (v_isSharedCheck_5222_ == 0)
{
v___x_5217_ = v___x_5197_;
v_isShared_5218_ = v_isSharedCheck_5222_;
goto v_resetjp_5216_;
}
else
{
lean_inc(v_a_5215_);
lean_dec(v___x_5197_);
v___x_5217_ = lean_box(0);
v_isShared_5218_ = v_isSharedCheck_5222_;
goto v_resetjp_5216_;
}
v_resetjp_5216_:
{
lean_object* v___x_5220_; 
if (v_isShared_5218_ == 0)
{
v___x_5220_ = v___x_5217_;
goto v_reusejp_5219_;
}
else
{
lean_object* v_reuseFailAlloc_5221_; 
v_reuseFailAlloc_5221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5221_, 0, v_a_5215_);
v___x_5220_ = v_reuseFailAlloc_5221_;
goto v_reusejp_5219_;
}
v_reusejp_5219_:
{
return v___x_5220_;
}
}
}
}
else
{
lean_object* v_a_5223_; lean_object* v___x_5225_; uint8_t v_isShared_5226_; uint8_t v_isSharedCheck_5230_; 
lean_dec(v_fvarId_5185_);
lean_dec(v_sym_5183_);
lean_dec(v___x_5182_);
lean_dec_ref(v_a_5181_);
lean_dec(v___y_5180_);
lean_dec(v___x_5179_);
v_a_5223_ = lean_ctor_get(v___x_5195_, 0);
v_isSharedCheck_5230_ = !lean_is_exclusive(v___x_5195_);
if (v_isSharedCheck_5230_ == 0)
{
v___x_5225_ = v___x_5195_;
v_isShared_5226_ = v_isSharedCheck_5230_;
goto v_resetjp_5224_;
}
else
{
lean_inc(v_a_5223_);
lean_dec(v___x_5195_);
v___x_5225_ = lean_box(0);
v_isShared_5226_ = v_isSharedCheck_5230_;
goto v_resetjp_5224_;
}
v_resetjp_5224_:
{
lean_object* v___x_5228_; 
if (v_isShared_5226_ == 0)
{
v___x_5228_ = v___x_5225_;
goto v_reusejp_5227_;
}
else
{
lean_object* v_reuseFailAlloc_5229_; 
v_reuseFailAlloc_5229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5229_, 0, v_a_5223_);
v___x_5228_ = v_reuseFailAlloc_5229_;
goto v_reusejp_5227_;
}
v_reusejp_5227_:
{
return v___x_5228_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2___boxed(lean_object* v___x_5231_, lean_object* v___y_5232_, lean_object* v_a_5233_, lean_object* v___x_5234_, lean_object* v_sym_5235_, lean_object* v___x_5236_, lean_object* v_fvarId_5237_, lean_object* v___y_5238_, lean_object* v___y_5239_, lean_object* v___y_5240_, lean_object* v___y_5241_, lean_object* v___y_5242_, lean_object* v___y_5243_, lean_object* v___y_5244_, lean_object* v___y_5245_, lean_object* v___y_5246_){
_start:
{
uint8_t v___x_10942__boxed_5247_; lean_object* v_res_5248_; 
v___x_10942__boxed_5247_ = lean_unbox(v___x_5236_);
v_res_5248_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2(v___x_5231_, v___y_5232_, v_a_5233_, v___x_5234_, v_sym_5235_, v___x_10942__boxed_5247_, v_fvarId_5237_, v___y_5238_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_, v___y_5243_, v___y_5244_, v___y_5245_);
lean_dec(v___y_5245_);
lean_dec_ref(v___y_5244_);
lean_dec(v___y_5243_);
lean_dec_ref(v___y_5242_);
lean_dec(v___y_5241_);
lean_dec_ref(v___y_5240_);
lean_dec(v___y_5239_);
lean_dec_ref(v___y_5238_);
return v_res_5248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3(lean_object* v___y_5249_, lean_object* v_fst_5250_, lean_object* v_a_5251_, lean_object* v___x_5252_, lean_object* v_snd_5253_, lean_object* v_sym_5254_, uint8_t v___x_5255_, uint8_t v___x_5256_, lean_object* v___y_5257_, lean_object* v___y_5258_, lean_object* v___y_5259_, lean_object* v___y_5260_, lean_object* v___y_5261_, lean_object* v___y_5262_, lean_object* v___y_5263_, lean_object* v___y_5264_){
_start:
{
lean_object* v_a_5267_; lean_object* v___x_5278_; 
v___x_5278_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5258_, v___y_5261_, v___y_5262_, v___y_5263_, v___y_5264_);
if (lean_obj_tag(v___x_5278_) == 0)
{
lean_object* v_a_5279_; uint8_t v___y_5281_; 
v_a_5279_ = lean_ctor_get(v___x_5278_, 0);
lean_inc(v_a_5279_);
lean_dec_ref_known(v___x_5278_, 1);
if (lean_obj_tag(v_sym_5254_) == 0)
{
v___y_5281_ = v___x_5255_;
goto v___jp_5280_;
}
else
{
v___y_5281_ = v___x_5256_;
goto v___jp_5280_;
}
v___jp_5280_:
{
lean_object* v___x_5282_; lean_object* v___x_5283_; 
v___x_5282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5282_, 0, v___y_5249_);
v___x_5283_ = lp_mathlib_Lean_MVarId_convert(v_fst_5250_, v___y_5281_, v___x_5282_, v_a_5251_, v___x_5252_, v_a_5279_, v___y_5261_, v___y_5262_, v___y_5263_, v___y_5264_);
if (lean_obj_tag(v___x_5283_) == 0)
{
lean_object* v_a_5284_; lean_object* v___x_5285_; 
v_a_5284_ = lean_ctor_get(v___x_5283_, 0);
lean_inc(v_a_5284_);
lean_dec_ref_known(v___x_5283_, 1);
v___x_5285_ = l_List_appendTR___redArg(v_a_5284_, v_snd_5253_);
v_a_5267_ = v___x_5285_;
goto v___jp_5266_;
}
else
{
lean_dec(v_snd_5253_);
if (lean_obj_tag(v___x_5283_) == 0)
{
lean_object* v_a_5286_; 
v_a_5286_ = lean_ctor_get(v___x_5283_, 0);
lean_inc(v_a_5286_);
lean_dec_ref_known(v___x_5283_, 1);
v_a_5267_ = v_a_5286_;
goto v___jp_5266_;
}
else
{
lean_object* v_a_5287_; lean_object* v___x_5289_; uint8_t v_isShared_5290_; uint8_t v_isSharedCheck_5294_; 
v_a_5287_ = lean_ctor_get(v___x_5283_, 0);
v_isSharedCheck_5294_ = !lean_is_exclusive(v___x_5283_);
if (v_isSharedCheck_5294_ == 0)
{
v___x_5289_ = v___x_5283_;
v_isShared_5290_ = v_isSharedCheck_5294_;
goto v_resetjp_5288_;
}
else
{
lean_inc(v_a_5287_);
lean_dec(v___x_5283_);
v___x_5289_ = lean_box(0);
v_isShared_5290_ = v_isSharedCheck_5294_;
goto v_resetjp_5288_;
}
v_resetjp_5288_:
{
lean_object* v___x_5292_; 
if (v_isShared_5290_ == 0)
{
v___x_5292_ = v___x_5289_;
goto v_reusejp_5291_;
}
else
{
lean_object* v_reuseFailAlloc_5293_; 
v_reuseFailAlloc_5293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5293_, 0, v_a_5287_);
v___x_5292_ = v_reuseFailAlloc_5293_;
goto v_reusejp_5291_;
}
v_reusejp_5291_:
{
return v___x_5292_;
}
}
}
}
}
}
else
{
lean_object* v_a_5295_; lean_object* v___x_5297_; uint8_t v_isShared_5298_; uint8_t v_isSharedCheck_5302_; 
lean_dec(v_snd_5253_);
lean_dec(v___x_5252_);
lean_dec_ref(v_a_5251_);
lean_dec_ref(v_fst_5250_);
lean_dec(v___y_5249_);
v_a_5295_ = lean_ctor_get(v___x_5278_, 0);
v_isSharedCheck_5302_ = !lean_is_exclusive(v___x_5278_);
if (v_isSharedCheck_5302_ == 0)
{
v___x_5297_ = v___x_5278_;
v_isShared_5298_ = v_isSharedCheck_5302_;
goto v_resetjp_5296_;
}
else
{
lean_inc(v_a_5295_);
lean_dec(v___x_5278_);
v___x_5297_ = lean_box(0);
v_isShared_5298_ = v_isSharedCheck_5302_;
goto v_resetjp_5296_;
}
v_resetjp_5296_:
{
lean_object* v___x_5300_; 
if (v_isShared_5298_ == 0)
{
v___x_5300_ = v___x_5297_;
goto v_reusejp_5299_;
}
else
{
lean_object* v_reuseFailAlloc_5301_; 
v_reuseFailAlloc_5301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5301_, 0, v_a_5295_);
v___x_5300_ = v_reuseFailAlloc_5301_;
goto v_reusejp_5299_;
}
v_reusejp_5299_:
{
return v___x_5300_;
}
}
}
v___jp_5266_:
{
lean_object* v___x_5268_; 
v___x_5268_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_5267_, v___y_5258_, v___y_5261_, v___y_5262_, v___y_5263_, v___y_5264_);
if (lean_obj_tag(v___x_5268_) == 0)
{
lean_object* v___x_5270_; uint8_t v_isShared_5271_; uint8_t v_isSharedCheck_5276_; 
v_isSharedCheck_5276_ = !lean_is_exclusive(v___x_5268_);
if (v_isSharedCheck_5276_ == 0)
{
lean_object* v_unused_5277_; 
v_unused_5277_ = lean_ctor_get(v___x_5268_, 0);
lean_dec(v_unused_5277_);
v___x_5270_ = v___x_5268_;
v_isShared_5271_ = v_isSharedCheck_5276_;
goto v_resetjp_5269_;
}
else
{
lean_dec(v___x_5268_);
v___x_5270_ = lean_box(0);
v_isShared_5271_ = v_isSharedCheck_5276_;
goto v_resetjp_5269_;
}
v_resetjp_5269_:
{
lean_object* v___x_5272_; lean_object* v___x_5274_; 
v___x_5272_ = lean_box(0);
if (v_isShared_5271_ == 0)
{
lean_ctor_set(v___x_5270_, 0, v___x_5272_);
v___x_5274_ = v___x_5270_;
goto v_reusejp_5273_;
}
else
{
lean_object* v_reuseFailAlloc_5275_; 
v_reuseFailAlloc_5275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5275_, 0, v___x_5272_);
v___x_5274_ = v_reuseFailAlloc_5275_;
goto v_reusejp_5273_;
}
v_reusejp_5273_:
{
return v___x_5274_;
}
}
}
else
{
return v___x_5268_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3___boxed(lean_object** _args){
lean_object* v___y_5303_ = _args[0];
lean_object* v_fst_5304_ = _args[1];
lean_object* v_a_5305_ = _args[2];
lean_object* v___x_5306_ = _args[3];
lean_object* v_snd_5307_ = _args[4];
lean_object* v_sym_5308_ = _args[5];
lean_object* v___x_5309_ = _args[6];
lean_object* v___x_5310_ = _args[7];
lean_object* v___y_5311_ = _args[8];
lean_object* v___y_5312_ = _args[9];
lean_object* v___y_5313_ = _args[10];
lean_object* v___y_5314_ = _args[11];
lean_object* v___y_5315_ = _args[12];
lean_object* v___y_5316_ = _args[13];
lean_object* v___y_5317_ = _args[14];
lean_object* v___y_5318_ = _args[15];
lean_object* v___y_5319_ = _args[16];
_start:
{
uint8_t v___x_11057__boxed_5320_; uint8_t v___x_11058__boxed_5321_; lean_object* v_res_5322_; 
v___x_11057__boxed_5320_ = lean_unbox(v___x_5309_);
v___x_11058__boxed_5321_ = lean_unbox(v___x_5310_);
v_res_5322_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3(v___y_5303_, v_fst_5304_, v_a_5305_, v___x_5306_, v_snd_5307_, v_sym_5308_, v___x_11057__boxed_5320_, v___x_11058__boxed_5321_, v___y_5311_, v___y_5312_, v___y_5313_, v___y_5314_, v___y_5315_, v___y_5316_, v___y_5317_, v___y_5318_);
lean_dec(v___y_5318_);
lean_dec_ref(v___y_5317_);
lean_dec(v___y_5316_);
lean_dec_ref(v___y_5315_);
lean_dec(v___y_5314_);
lean_dec_ref(v___y_5313_);
lean_dec(v___y_5312_);
lean_dec_ref(v___y_5311_);
lean_dec(v_sym_5308_);
return v_res_5322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6(void){
_start:
{
lean_object* v___x_5330_; lean_object* v___x_5331_; 
v___x_5330_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__1_spec__2___closed__6));
v___x_5331_ = l_String_toRawSubstring_x27(v___x_5330_);
return v___x_5331_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10(void){
_start:
{
lean_object* v___x_5335_; lean_object* v___x_5336_; 
v___x_5335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__9));
v___x_5336_ = l_String_toRawSubstring_x27(v___x_5335_);
return v___x_5336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4(lean_object* v___x_5350_, lean_object* v___x_5351_, lean_object* v___x_5352_, lean_object* v___x_5353_, lean_object* v___x_5354_, lean_object* v___y_5355_, lean_object* v_a_5356_, lean_object* v___x_5357_, lean_object* v_sym_5358_, uint8_t v___x_5359_, lean_object* v___y_5360_, lean_object* v___y_5361_, lean_object* v___y_5362_, lean_object* v___y_5363_, lean_object* v___y_5364_, lean_object* v___y_5365_, lean_object* v___y_5366_, lean_object* v___y_5367_){
_start:
{
lean_object* v___x_5369_; 
v___x_5369_ = l_Lean_Elab_Tactic_getMainTarget(v___y_5360_, v___y_5361_, v___y_5362_, v___y_5363_, v___y_5364_, v___y_5365_, v___y_5366_, v___y_5367_);
if (lean_obj_tag(v___x_5369_) == 0)
{
lean_object* v_a_5370_; lean_object* v___x_5371_; 
v_a_5370_ = lean_ctor_get(v___x_5369_, 0);
lean_inc(v_a_5370_);
lean_dec_ref_known(v___x_5369_, 1);
v___x_5371_ = l_Lean_Meta_getLevel(v_a_5370_, v___y_5364_, v___y_5365_, v___y_5366_, v___y_5367_);
if (lean_obj_tag(v___x_5371_) == 0)
{
lean_object* v_a_5372_; lean_object* v___x_5373_; lean_object* v___x_5374_; uint8_t v___x_5375_; lean_object* v___x_5376_; lean_object* v___x_5377_; 
v_a_5372_ = lean_ctor_get(v___x_5371_, 0);
lean_inc(v_a_5372_);
lean_dec_ref_known(v___x_5371_, 1);
v___x_5373_ = l_Lean_mkSort(v_a_5372_);
v___x_5374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5374_, 0, v___x_5373_);
v___x_5375_ = 0;
v___x_5376_ = lean_box(0);
v___x_5377_ = l_Lean_Meta_mkFreshExprMVar(v___x_5374_, v___x_5375_, v___x_5376_, v___y_5364_, v___y_5365_, v___y_5366_, v___y_5367_);
if (lean_obj_tag(v___x_5377_) == 0)
{
lean_object* v_a_5378_; lean_object* v_ref_5379_; lean_object* v_quotContext_5380_; lean_object* v_currMacroScope_5381_; uint8_t v___x_5382_; lean_object* v___x_5383_; lean_object* v___x_5384_; lean_object* v___x_5385_; lean_object* v___x_5386_; lean_object* v___x_5387_; lean_object* v___x_5388_; lean_object* v___x_5389_; lean_object* v___x_5390_; lean_object* v___x_5391_; lean_object* v___x_5392_; lean_object* v___x_5393_; lean_object* v___x_5394_; lean_object* v___x_5395_; lean_object* v___x_5396_; lean_object* v___x_5397_; lean_object* v___x_5398_; lean_object* v___x_5399_; lean_object* v___x_5400_; lean_object* v___x_5401_; lean_object* v___x_5402_; lean_object* v___x_5403_; lean_object* v___x_5404_; lean_object* v___x_5405_; lean_object* v___x_5406_; lean_object* v___x_5407_; lean_object* v___x_5408_; lean_object* v___x_5409_; lean_object* v___x_5410_; lean_object* v___x_5411_; lean_object* v___x_5412_; lean_object* v___x_5413_; lean_object* v___x_5414_; lean_object* v___x_5415_; lean_object* v___x_5416_; lean_object* v___x_5417_; lean_object* v___x_5418_; lean_object* v___x_5419_; lean_object* v___x_5420_; lean_object* v___x_5421_; lean_object* v___x_5422_; lean_object* v___x_5423_; lean_object* v___x_5424_; lean_object* v___x_5425_; lean_object* v___x_5426_; lean_object* v___x_5427_; lean_object* v___x_5428_; lean_object* v___x_5429_; lean_object* v___x_5430_; lean_object* v___x_5431_; lean_object* v___x_5432_; lean_object* v___x_5433_; lean_object* v___x_5434_; lean_object* v___x_5435_; lean_object* v___x_5436_; lean_object* v___x_5437_; lean_object* v___x_5438_; lean_object* v___x_5439_; lean_object* v___x_5440_; lean_object* v___x_5441_; lean_object* v___x_5442_; 
v_a_5378_ = lean_ctor_get(v___x_5377_, 0);
lean_inc(v_a_5378_);
lean_dec_ref_known(v___x_5377_, 1);
v_ref_5379_ = lean_ctor_get(v___y_5366_, 5);
v_quotContext_5380_ = lean_ctor_get(v___y_5366_, 10);
v_currMacroScope_5381_ = lean_ctor_get(v___y_5366_, 11);
v___x_5382_ = 0;
v___x_5383_ = l_Lean_SourceInfo_fromRef(v_ref_5379_, v___x_5382_);
v___x_5384_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__0));
v___x_5385_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__1));
lean_inc_ref_n(v___x_5351_, 3);
lean_inc_ref_n(v___x_5350_, 8);
v___x_5386_ = l_Lean_Name_mkStr4(v___x_5350_, v___x_5351_, v___x_5384_, v___x_5385_);
v___x_5387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__2));
v___x_5388_ = l_Lean_Name_mkStr4(v___x_5350_, v___x_5351_, v___x_5384_, v___x_5387_);
v___x_5389_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__3));
lean_inc_n(v___x_5383_, 13);
v___x_5390_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5390_, 0, v___x_5383_);
lean_ctor_set(v___x_5390_, 1, v___x_5389_);
v___x_5391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__5));
v___x_5392_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__6);
lean_inc_n(v_currMacroScope_5381_, 2);
lean_inc_n(v_quotContext_5380_, 2);
v___x_5393_ = l_Lean_addMacroScope(v_quotContext_5380_, v___x_5376_, v_currMacroScope_5381_);
lean_inc_ref_n(v___x_5353_, 2);
v___x_5394_ = l_Lean_Name_mkStr2(v___x_5352_, v___x_5353_);
v___x_5395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5395_, 0, v___x_5394_);
v___x_5396_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__1));
v___x_5397_ = l_Lean_Name_mkStr3(v___x_5350_, v___x_5396_, v___x_5353_);
v___x_5398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5398_, 0, v___x_5397_);
v___x_5399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__7));
v___x_5400_ = l_Lean_Name_mkStr3(v___x_5350_, v___x_5399_, v___x_5353_);
v___x_5401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5401_, 0, v___x_5400_);
v___x_5402_ = l_Lean_Name_mkStr2(v___x_5350_, v___x_5399_);
v___x_5403_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5403_, 0, v___x_5402_);
v___x_5404_ = l_Lean_Name_mkStr2(v___x_5350_, v___x_5396_);
v___x_5405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5405_, 0, v___x_5404_);
v___x_5406_ = l_Lean_Name_mkStr1(v___x_5350_);
v___x_5407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5407_, 0, v___x_5406_);
v___x_5408_ = lean_box(0);
v___x_5409_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5409_, 0, v___x_5407_);
lean_ctor_set(v___x_5409_, 1, v___x_5408_);
v___x_5410_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5410_, 0, v___x_5405_);
lean_ctor_set(v___x_5410_, 1, v___x_5409_);
v___x_5411_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5411_, 0, v___x_5403_);
lean_ctor_set(v___x_5411_, 1, v___x_5410_);
v___x_5412_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5412_, 0, v___x_5401_);
lean_ctor_set(v___x_5412_, 1, v___x_5411_);
v___x_5413_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5413_, 0, v___x_5398_);
lean_ctor_set(v___x_5413_, 1, v___x_5412_);
v___x_5414_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5414_, 0, v___x_5395_);
lean_ctor_set(v___x_5414_, 1, v___x_5413_);
v___x_5415_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_5415_, 0, v___x_5383_);
lean_ctor_set(v___x_5415_, 1, v___x_5392_);
lean_ctor_set(v___x_5415_, 2, v___x_5393_);
lean_ctor_set(v___x_5415_, 3, v___x_5414_);
v___x_5416_ = l_Lean_Syntax_node1(v___x_5383_, v___x_5391_, v___x_5415_);
v___x_5417_ = l_Lean_Syntax_node2(v___x_5383_, v___x_5388_, v___x_5390_, v___x_5416_);
v___x_5418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__8));
v___x_5419_ = l_Lean_Name_mkStr4(v___x_5350_, v___x_5351_, v___x_5384_, v___x_5418_);
v___x_5420_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__10);
v___x_5421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__11));
v___x_5422_ = l_Lean_addMacroScope(v_quotContext_5380_, v___x_5421_, v_currMacroScope_5381_);
v___x_5423_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__13));
v___x_5424_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_5424_, 0, v___x_5383_);
lean_ctor_set(v___x_5424_, 1, v___x_5420_);
lean_ctor_set(v___x_5424_, 2, v___x_5422_);
lean_ctor_set(v___x_5424_, 3, v___x_5423_);
v___x_5425_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4));
v___x_5426_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__14));
v___x_5427_ = l_Lean_Name_mkStr4(v___x_5350_, v___x_5351_, v___x_5384_, v___x_5426_);
v___x_5428_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__15));
v___x_5429_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5429_, 0, v___x_5383_);
lean_ctor_set(v___x_5429_, 1, v___x_5428_);
v___x_5430_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__16));
v___x_5431_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5431_, 0, v___x_5383_);
lean_ctor_set(v___x_5431_, 1, v___x_5430_);
v___x_5432_ = l_Lean_Syntax_node2(v___x_5383_, v___x_5427_, v___x_5429_, v___x_5431_);
v___x_5433_ = l_Lean_Syntax_node1(v___x_5383_, v___x_5425_, v___x_5432_);
v___x_5434_ = l_Lean_Syntax_node2(v___x_5383_, v___x_5419_, v___x_5424_, v___x_5433_);
v___x_5435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__17));
v___x_5436_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5436_, 0, v___x_5383_);
lean_ctor_set(v___x_5436_, 1, v___x_5435_);
v___x_5437_ = l_Lean_Syntax_node1(v___x_5383_, v___x_5425_, v___x_5354_);
v___x_5438_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___closed__18));
v___x_5439_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5439_, 0, v___x_5383_);
lean_ctor_set(v___x_5439_, 1, v___x_5438_);
v___x_5440_ = l_Lean_Syntax_node5(v___x_5383_, v___x_5386_, v___x_5417_, v___x_5434_, v___x_5436_, v___x_5437_, v___x_5439_);
v___x_5441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5441_, 0, v_a_5378_);
v___x_5442_ = lp_mathlib_Mathlib_Tactic_elabTermForConvert(v___x_5440_, v___x_5441_, v___y_5360_, v___y_5361_, v___y_5362_, v___y_5363_, v___y_5364_, v___y_5365_, v___y_5366_, v___y_5367_);
if (lean_obj_tag(v___x_5442_) == 0)
{
lean_object* v_a_5443_; lean_object* v_fst_5444_; lean_object* v_snd_5445_; lean_object* v___x_5446_; lean_object* v___x_5447_; lean_object* v___f_5448_; lean_object* v___x_5449_; 
v_a_5443_ = lean_ctor_get(v___x_5442_, 0);
lean_inc(v_a_5443_);
lean_dec_ref_known(v___x_5442_, 1);
v_fst_5444_ = lean_ctor_get(v_a_5443_, 0);
lean_inc(v_fst_5444_);
v_snd_5445_ = lean_ctor_get(v_a_5443_, 1);
lean_inc(v_snd_5445_);
lean_dec(v_a_5443_);
v___x_5446_ = lean_box(v___x_5382_);
v___x_5447_ = lean_box(v___x_5359_);
v___f_5448_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__3___boxed), 17, 8);
lean_closure_set(v___f_5448_, 0, v___y_5355_);
lean_closure_set(v___f_5448_, 1, v_fst_5444_);
lean_closure_set(v___f_5448_, 2, v_a_5356_);
lean_closure_set(v___f_5448_, 3, v___x_5357_);
lean_closure_set(v___f_5448_, 4, v_snd_5445_);
lean_closure_set(v___f_5448_, 5, v_sym_5358_);
lean_closure_set(v___f_5448_, 6, v___x_5446_);
lean_closure_set(v___f_5448_, 7, v___x_5447_);
v___x_5449_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5448_, v___y_5360_, v___y_5361_, v___y_5362_, v___y_5363_, v___y_5364_, v___y_5365_, v___y_5366_, v___y_5367_);
lean_dec_ref(v___y_5366_);
return v___x_5449_;
}
else
{
lean_object* v_a_5450_; lean_object* v___x_5452_; uint8_t v_isShared_5453_; uint8_t v_isSharedCheck_5457_; 
lean_dec_ref(v___y_5366_);
lean_dec(v_sym_5358_);
lean_dec(v___x_5357_);
lean_dec_ref(v_a_5356_);
lean_dec(v___y_5355_);
v_a_5450_ = lean_ctor_get(v___x_5442_, 0);
v_isSharedCheck_5457_ = !lean_is_exclusive(v___x_5442_);
if (v_isSharedCheck_5457_ == 0)
{
v___x_5452_ = v___x_5442_;
v_isShared_5453_ = v_isSharedCheck_5457_;
goto v_resetjp_5451_;
}
else
{
lean_inc(v_a_5450_);
lean_dec(v___x_5442_);
v___x_5452_ = lean_box(0);
v_isShared_5453_ = v_isSharedCheck_5457_;
goto v_resetjp_5451_;
}
v_resetjp_5451_:
{
lean_object* v___x_5455_; 
if (v_isShared_5453_ == 0)
{
v___x_5455_ = v___x_5452_;
goto v_reusejp_5454_;
}
else
{
lean_object* v_reuseFailAlloc_5456_; 
v_reuseFailAlloc_5456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5456_, 0, v_a_5450_);
v___x_5455_ = v_reuseFailAlloc_5456_;
goto v_reusejp_5454_;
}
v_reusejp_5454_:
{
return v___x_5455_;
}
}
}
}
else
{
lean_object* v_a_5458_; lean_object* v___x_5460_; uint8_t v_isShared_5461_; uint8_t v_isSharedCheck_5465_; 
lean_dec_ref(v___y_5366_);
lean_dec(v_sym_5358_);
lean_dec(v___x_5357_);
lean_dec_ref(v_a_5356_);
lean_dec(v___y_5355_);
lean_dec(v___x_5354_);
lean_dec_ref(v___x_5353_);
lean_dec_ref(v___x_5352_);
lean_dec_ref(v___x_5351_);
lean_dec_ref(v___x_5350_);
v_a_5458_ = lean_ctor_get(v___x_5377_, 0);
v_isSharedCheck_5465_ = !lean_is_exclusive(v___x_5377_);
if (v_isSharedCheck_5465_ == 0)
{
v___x_5460_ = v___x_5377_;
v_isShared_5461_ = v_isSharedCheck_5465_;
goto v_resetjp_5459_;
}
else
{
lean_inc(v_a_5458_);
lean_dec(v___x_5377_);
v___x_5460_ = lean_box(0);
v_isShared_5461_ = v_isSharedCheck_5465_;
goto v_resetjp_5459_;
}
v_resetjp_5459_:
{
lean_object* v___x_5463_; 
if (v_isShared_5461_ == 0)
{
v___x_5463_ = v___x_5460_;
goto v_reusejp_5462_;
}
else
{
lean_object* v_reuseFailAlloc_5464_; 
v_reuseFailAlloc_5464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5464_, 0, v_a_5458_);
v___x_5463_ = v_reuseFailAlloc_5464_;
goto v_reusejp_5462_;
}
v_reusejp_5462_:
{
return v___x_5463_;
}
}
}
}
else
{
lean_object* v_a_5466_; lean_object* v___x_5468_; uint8_t v_isShared_5469_; uint8_t v_isSharedCheck_5473_; 
lean_dec_ref(v___y_5366_);
lean_dec(v_sym_5358_);
lean_dec(v___x_5357_);
lean_dec_ref(v_a_5356_);
lean_dec(v___y_5355_);
lean_dec(v___x_5354_);
lean_dec_ref(v___x_5353_);
lean_dec_ref(v___x_5352_);
lean_dec_ref(v___x_5351_);
lean_dec_ref(v___x_5350_);
v_a_5466_ = lean_ctor_get(v___x_5371_, 0);
v_isSharedCheck_5473_ = !lean_is_exclusive(v___x_5371_);
if (v_isSharedCheck_5473_ == 0)
{
v___x_5468_ = v___x_5371_;
v_isShared_5469_ = v_isSharedCheck_5473_;
goto v_resetjp_5467_;
}
else
{
lean_inc(v_a_5466_);
lean_dec(v___x_5371_);
v___x_5468_ = lean_box(0);
v_isShared_5469_ = v_isSharedCheck_5473_;
goto v_resetjp_5467_;
}
v_resetjp_5467_:
{
lean_object* v___x_5471_; 
if (v_isShared_5469_ == 0)
{
v___x_5471_ = v___x_5468_;
goto v_reusejp_5470_;
}
else
{
lean_object* v_reuseFailAlloc_5472_; 
v_reuseFailAlloc_5472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5472_, 0, v_a_5466_);
v___x_5471_ = v_reuseFailAlloc_5472_;
goto v_reusejp_5470_;
}
v_reusejp_5470_:
{
return v___x_5471_;
}
}
}
}
else
{
lean_object* v_a_5474_; lean_object* v___x_5476_; uint8_t v_isShared_5477_; uint8_t v_isSharedCheck_5481_; 
lean_dec_ref(v___y_5366_);
lean_dec(v_sym_5358_);
lean_dec(v___x_5357_);
lean_dec_ref(v_a_5356_);
lean_dec(v___y_5355_);
lean_dec(v___x_5354_);
lean_dec_ref(v___x_5353_);
lean_dec_ref(v___x_5352_);
lean_dec_ref(v___x_5351_);
lean_dec_ref(v___x_5350_);
v_a_5474_ = lean_ctor_get(v___x_5369_, 0);
v_isSharedCheck_5481_ = !lean_is_exclusive(v___x_5369_);
if (v_isSharedCheck_5481_ == 0)
{
v___x_5476_ = v___x_5369_;
v_isShared_5477_ = v_isSharedCheck_5481_;
goto v_resetjp_5475_;
}
else
{
lean_inc(v_a_5474_);
lean_dec(v___x_5369_);
v___x_5476_ = lean_box(0);
v_isShared_5477_ = v_isSharedCheck_5481_;
goto v_resetjp_5475_;
}
v_resetjp_5475_:
{
lean_object* v___x_5479_; 
if (v_isShared_5477_ == 0)
{
v___x_5479_ = v___x_5476_;
goto v_reusejp_5478_;
}
else
{
lean_object* v_reuseFailAlloc_5480_; 
v_reuseFailAlloc_5480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5480_, 0, v_a_5474_);
v___x_5479_ = v_reuseFailAlloc_5480_;
goto v_reusejp_5478_;
}
v_reusejp_5478_:
{
return v___x_5479_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___boxed(lean_object** _args){
lean_object* v___x_5482_ = _args[0];
lean_object* v___x_5483_ = _args[1];
lean_object* v___x_5484_ = _args[2];
lean_object* v___x_5485_ = _args[3];
lean_object* v___x_5486_ = _args[4];
lean_object* v___y_5487_ = _args[5];
lean_object* v_a_5488_ = _args[6];
lean_object* v___x_5489_ = _args[7];
lean_object* v_sym_5490_ = _args[8];
lean_object* v___x_5491_ = _args[9];
lean_object* v___y_5492_ = _args[10];
lean_object* v___y_5493_ = _args[11];
lean_object* v___y_5494_ = _args[12];
lean_object* v___y_5495_ = _args[13];
lean_object* v___y_5496_ = _args[14];
lean_object* v___y_5497_ = _args[15];
lean_object* v___y_5498_ = _args[16];
lean_object* v___y_5499_ = _args[17];
lean_object* v___y_5500_ = _args[18];
_start:
{
uint8_t v___x_11244__boxed_5501_; lean_object* v_res_5502_; 
v___x_11244__boxed_5501_ = lean_unbox(v___x_5491_);
v_res_5502_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4(v___x_5482_, v___x_5483_, v___x_5484_, v___x_5485_, v___x_5486_, v___y_5487_, v_a_5488_, v___x_5489_, v_sym_5490_, v___x_11244__boxed_5501_, v___y_5492_, v___y_5493_, v___y_5494_, v___y_5495_, v___y_5496_, v___y_5497_, v___y_5498_, v___y_5499_);
lean_dec(v___y_5499_);
lean_dec(v___y_5497_);
lean_dec_ref(v___y_5496_);
lean_dec(v___y_5495_);
lean_dec_ref(v___y_5494_);
lean_dec(v___y_5493_);
lean_dec_ref(v___y_5492_);
return v_res_5502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1(lean_object* v_x_5513_, lean_object* v_a_5514_, lean_object* v_a_5515_, lean_object* v_a_5516_, lean_object* v_a_5517_, lean_object* v_a_5518_, lean_object* v_a_5519_, lean_object* v_a_5520_, lean_object* v_a_5521_){
_start:
{
lean_object* v___y_5524_; lean_object* v___y_5525_; lean_object* v___y_5526_; lean_object* v___y_5527_; lean_object* v___y_5528_; lean_object* v___y_5529_; lean_object* v___y_5530_; lean_object* v___y_5531_; lean_object* v___y_5532_; lean_object* v___y_5533_; lean_object* v___y_5534_; lean_object* v___y_5535_; lean_object* v___x_5539_; lean_object* v___x_5540_; lean_object* v___x_5541_; uint8_t v___x_5542_; lean_object* v___y_5544_; lean_object* v___y_5545_; lean_object* v___y_5546_; lean_object* v___y_5547_; lean_object* v___y_5548_; lean_object* v___y_5549_; lean_object* v___y_5550_; lean_object* v___y_5551_; lean_object* v___y_5552_; lean_object* v___y_5553_; lean_object* v___y_5554_; lean_object* v___y_5555_; lean_object* v___y_5556_; lean_object* v___y_5557_; lean_object* v___y_5558_; lean_object* v___y_5559_; lean_object* v___y_5560_; 
v___x_5539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__0));
v___x_5540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__1));
v___x_5541_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__1));
lean_inc(v_x_5513_);
v___x_5542_ = l_Lean_Syntax_isOfKind(v_x_5513_, v___x_5541_);
if (v___x_5542_ == 0)
{
lean_object* v___x_5575_; 
lean_dec(v_x_5513_);
v___x_5575_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5575_;
}
else
{
lean_object* v___f_5576_; lean_object* v___x_5577_; lean_object* v___y_5579_; lean_object* v___y_5580_; lean_object* v___y_5581_; lean_object* v___y_5582_; lean_object* v___y_5583_; lean_object* v___y_5584_; lean_object* v___y_5585_; lean_object* v___y_5586_; lean_object* v___y_5587_; lean_object* v___y_5588_; lean_object* v___y_5589_; lean_object* v___y_5590_; lean_object* v___y_5591_; lean_object* v___y_5592_; lean_object* v___y_5593_; lean_object* v___y_5594_; lean_object* v___y_5595_; uint8_t v___y_5596_; lean_object* v___y_5611_; lean_object* v___y_5612_; lean_object* v___y_5613_; lean_object* v___y_5614_; lean_object* v___y_5615_; lean_object* v___y_5616_; lean_object* v___y_5617_; lean_object* v___y_5618_; lean_object* v___y_5619_; lean_object* v___y_5620_; lean_object* v___y_5621_; lean_object* v___y_5622_; lean_object* v___y_5623_; lean_object* v___y_5624_; lean_object* v___y_5625_; lean_object* v___y_5626_; lean_object* v___y_5627_; lean_object* v___x_5629_; lean_object* v___y_5631_; lean_object* v___y_5632_; lean_object* v___y_5633_; lean_object* v___y_5634_; lean_object* v___y_5635_; lean_object* v___y_5636_; lean_object* v_loc_x3f_5637_; lean_object* v___y_5638_; lean_object* v___y_5639_; lean_object* v___y_5640_; lean_object* v___y_5641_; lean_object* v___y_5642_; lean_object* v___y_5643_; lean_object* v___y_5644_; lean_object* v___y_5645_; lean_object* v___y_5651_; lean_object* v___y_5652_; lean_object* v___y_5653_; lean_object* v___y_5654_; lean_object* v___y_5655_; lean_object* v_ps_x3f_5656_; lean_object* v___y_5657_; lean_object* v___y_5658_; lean_object* v___y_5659_; lean_object* v___y_5660_; lean_object* v___y_5661_; lean_object* v___y_5662_; lean_object* v___y_5663_; lean_object* v___y_5664_; lean_object* v___y_5677_; lean_object* v___y_5678_; lean_object* v___y_5679_; lean_object* v___y_5680_; lean_object* v___y_5681_; lean_object* v___y_5682_; lean_object* v___y_5683_; lean_object* v___y_5684_; lean_object* v___y_5685_; lean_object* v___y_5686_; lean_object* v___y_5687_; lean_object* v___y_5688_; lean_object* v___y_5689_; lean_object* v_n_5690_; lean_object* v___y_5701_; lean_object* v___y_5702_; lean_object* v___y_5703_; lean_object* v_sym_5704_; lean_object* v___y_5705_; lean_object* v___y_5706_; lean_object* v___y_5707_; lean_object* v___y_5708_; lean_object* v___y_5709_; lean_object* v___y_5710_; lean_object* v___y_5711_; lean_object* v___y_5712_; lean_object* v_expensive_5724_; lean_object* v___y_5725_; lean_object* v___y_5726_; lean_object* v___y_5727_; lean_object* v___y_5728_; lean_object* v___y_5729_; lean_object* v___y_5730_; lean_object* v___y_5731_; lean_object* v___y_5732_; lean_object* v___x_5743_; uint8_t v___x_5744_; 
v___f_5576_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__0));
v___x_5577_ = lean_unsigned_to_nat(0u);
v___x_5629_ = lean_unsigned_to_nat(1u);
v___x_5743_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5629_);
v___x_5744_ = l_Lean_Syntax_isNone(v___x_5743_);
if (v___x_5744_ == 0)
{
uint8_t v___x_5745_; 
lean_inc(v___x_5743_);
v___x_5745_ = l_Lean_Syntax_matchesNull(v___x_5743_, v___x_5629_);
if (v___x_5745_ == 0)
{
lean_object* v___x_5746_; 
lean_dec(v___x_5743_);
lean_dec(v_x_5513_);
v___x_5746_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5746_;
}
else
{
lean_object* v_expensive_5747_; lean_object* v___x_5748_; 
v_expensive_5747_ = l_Lean_Syntax_getArg(v___x_5743_, v___x_5577_);
lean_dec(v___x_5743_);
v___x_5748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5748_, 0, v_expensive_5747_);
v_expensive_5724_ = v___x_5748_;
v___y_5725_ = v_a_5514_;
v___y_5726_ = v_a_5515_;
v___y_5727_ = v_a_5516_;
v___y_5728_ = v_a_5517_;
v___y_5729_ = v_a_5518_;
v___y_5730_ = v_a_5519_;
v___y_5731_ = v_a_5520_;
v___y_5732_ = v_a_5521_;
goto v___jp_5723_;
}
}
else
{
lean_object* v___x_5749_; 
lean_dec(v___x_5743_);
v___x_5749_ = lean_box(0);
v_expensive_5724_ = v___x_5749_;
v___y_5725_ = v_a_5514_;
v___y_5726_ = v_a_5515_;
v___y_5727_ = v_a_5516_;
v___y_5728_ = v_a_5517_;
v___y_5729_ = v_a_5518_;
v___y_5730_ = v_a_5519_;
v___y_5731_ = v_a_5520_;
v___y_5732_ = v_a_5521_;
goto v___jp_5723_;
}
v___jp_5578_:
{
lean_object* v___x_5597_; 
v___x_5597_ = lp_mathlib_Convert_elabConfig___redArg(v___y_5596_, v___y_5587_, v___y_5579_, v___y_5581_, v___y_5590_);
if (lean_obj_tag(v___x_5597_) == 0)
{
if (lean_obj_tag(v___y_5592_) == 0)
{
lean_object* v_a_5598_; lean_object* v___x_5599_; 
v_a_5598_ = lean_ctor_get(v___x_5597_, 0);
lean_inc(v_a_5598_);
lean_dec_ref_known(v___x_5597_, 1);
v___x_5599_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__1));
v___y_5544_ = v___y_5588_;
v___y_5545_ = v___y_5580_;
v___y_5546_ = v___y_5591_;
v___y_5547_ = v_a_5598_;
v___y_5548_ = v___y_5583_;
v___y_5549_ = v___y_5593_;
v___y_5550_ = v___y_5579_;
v___y_5551_ = v___y_5586_;
v___y_5552_ = v___y_5589_;
v___y_5553_ = v___y_5590_;
v___y_5554_ = v___y_5595_;
v___y_5555_ = v___y_5581_;
v___y_5556_ = v___y_5582_;
v___y_5557_ = v___y_5584_;
v___y_5558_ = v___y_5585_;
v___y_5559_ = v___y_5594_;
v___y_5560_ = v___x_5599_;
goto v___jp_5543_;
}
else
{
lean_object* v_a_5600_; lean_object* v_val_5601_; 
v_a_5600_ = lean_ctor_get(v___x_5597_, 0);
lean_inc(v_a_5600_);
lean_dec_ref_known(v___x_5597_, 1);
v_val_5601_ = lean_ctor_get(v___y_5592_, 0);
lean_inc(v_val_5601_);
lean_dec_ref_known(v___y_5592_, 1);
v___y_5544_ = v___y_5588_;
v___y_5545_ = v___y_5580_;
v___y_5546_ = v___y_5591_;
v___y_5547_ = v_a_5600_;
v___y_5548_ = v___y_5583_;
v___y_5549_ = v___y_5593_;
v___y_5550_ = v___y_5579_;
v___y_5551_ = v___y_5586_;
v___y_5552_ = v___y_5589_;
v___y_5553_ = v___y_5590_;
v___y_5554_ = v___y_5595_;
v___y_5555_ = v___y_5581_;
v___y_5556_ = v___y_5582_;
v___y_5557_ = v___y_5584_;
v___y_5558_ = v___y_5585_;
v___y_5559_ = v___y_5594_;
v___y_5560_ = v_val_5601_;
goto v___jp_5543_;
}
}
else
{
lean_object* v_a_5602_; lean_object* v___x_5604_; uint8_t v_isShared_5605_; uint8_t v_isSharedCheck_5609_; 
lean_dec(v___y_5593_);
lean_dec(v___y_5592_);
lean_dec(v___y_5589_);
lean_dec(v___y_5583_);
lean_dec(v___y_5580_);
v_a_5602_ = lean_ctor_get(v___x_5597_, 0);
v_isSharedCheck_5609_ = !lean_is_exclusive(v___x_5597_);
if (v_isSharedCheck_5609_ == 0)
{
v___x_5604_ = v___x_5597_;
v_isShared_5605_ = v_isSharedCheck_5609_;
goto v_resetjp_5603_;
}
else
{
lean_inc(v_a_5602_);
lean_dec(v___x_5597_);
v___x_5604_ = lean_box(0);
v_isShared_5605_ = v_isSharedCheck_5609_;
goto v_resetjp_5603_;
}
v_resetjp_5603_:
{
lean_object* v___x_5607_; 
if (v_isShared_5605_ == 0)
{
v___x_5607_ = v___x_5604_;
goto v_reusejp_5606_;
}
else
{
lean_object* v_reuseFailAlloc_5608_; 
v_reuseFailAlloc_5608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5608_, 0, v_a_5602_);
v___x_5607_ = v_reuseFailAlloc_5608_;
goto v_reusejp_5606_;
}
v_reusejp_5606_:
{
return v___x_5607_;
}
}
}
}
v___jp_5610_:
{
if (lean_obj_tag(v___y_5615_) == 0)
{
uint8_t v___x_5628_; 
v___x_5628_ = 0;
v___y_5579_ = v___y_5611_;
v___y_5580_ = v___y_5627_;
v___y_5581_ = v___y_5612_;
v___y_5582_ = v___y_5613_;
v___y_5583_ = v___y_5614_;
v___y_5584_ = v___y_5616_;
v___y_5585_ = v___y_5617_;
v___y_5586_ = v___y_5618_;
v___y_5587_ = v___y_5619_;
v___y_5588_ = v___y_5620_;
v___y_5589_ = v___y_5621_;
v___y_5590_ = v___y_5622_;
v___y_5591_ = v___y_5623_;
v___y_5592_ = v___y_5624_;
v___y_5593_ = v___y_5625_;
v___y_5594_ = v___y_5626_;
v___y_5595_ = v___f_5576_;
v___y_5596_ = v___x_5628_;
goto v___jp_5578_;
}
else
{
lean_dec_ref_known(v___y_5615_, 1);
v___y_5579_ = v___y_5611_;
v___y_5580_ = v___y_5627_;
v___y_5581_ = v___y_5612_;
v___y_5582_ = v___y_5613_;
v___y_5583_ = v___y_5614_;
v___y_5584_ = v___y_5616_;
v___y_5585_ = v___y_5617_;
v___y_5586_ = v___y_5618_;
v___y_5587_ = v___y_5619_;
v___y_5588_ = v___y_5620_;
v___y_5589_ = v___y_5621_;
v___y_5590_ = v___y_5622_;
v___y_5591_ = v___y_5623_;
v___y_5592_ = v___y_5624_;
v___y_5593_ = v___y_5625_;
v___y_5594_ = v___y_5626_;
v___y_5595_ = v___f_5576_;
v___y_5596_ = v___x_5542_;
goto v___jp_5578_;
}
}
v___jp_5630_:
{
lean_object* v___x_5646_; lean_object* v___x_5647_; 
v___x_5646_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0));
v___x_5647_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2));
if (lean_obj_tag(v___y_5636_) == 0)
{
v___y_5611_ = v___y_5638_;
v___y_5612_ = v___y_5644_;
v___y_5613_ = v___y_5643_;
v___y_5614_ = v___y_5632_;
v___y_5615_ = v___y_5634_;
v___y_5616_ = v___y_5641_;
v___y_5617_ = v___y_5640_;
v___y_5618_ = v___y_5639_;
v___y_5619_ = v___y_5635_;
v___y_5620_ = v___x_5647_;
v___y_5621_ = v_loc_x3f_5637_;
v___y_5622_ = v___y_5645_;
v___y_5623_ = v___x_5646_;
v___y_5624_ = v___y_5633_;
v___y_5625_ = v___y_5631_;
v___y_5626_ = v___y_5642_;
v___y_5627_ = v___x_5629_;
goto v___jp_5610_;
}
else
{
lean_object* v_val_5648_; lean_object* v___x_5649_; 
v_val_5648_ = lean_ctor_get(v___y_5636_, 0);
lean_inc(v_val_5648_);
lean_dec_ref_known(v___y_5636_, 1);
v___x_5649_ = l_Lean_TSyntax_getNat(v_val_5648_);
lean_dec(v_val_5648_);
v___y_5611_ = v___y_5638_;
v___y_5612_ = v___y_5644_;
v___y_5613_ = v___y_5643_;
v___y_5614_ = v___y_5632_;
v___y_5615_ = v___y_5634_;
v___y_5616_ = v___y_5641_;
v___y_5617_ = v___y_5640_;
v___y_5618_ = v___y_5639_;
v___y_5619_ = v___y_5635_;
v___y_5620_ = v___x_5647_;
v___y_5621_ = v_loc_x3f_5637_;
v___y_5622_ = v___y_5645_;
v___y_5623_ = v___x_5646_;
v___y_5624_ = v___y_5633_;
v___y_5625_ = v___y_5631_;
v___y_5626_ = v___y_5642_;
v___y_5627_ = v___x_5649_;
goto v___jp_5610_;
}
}
v___jp_5650_:
{
lean_object* v___x_5665_; lean_object* v___x_5666_; uint8_t v___x_5667_; 
v___x_5665_ = lean_unsigned_to_nat(7u);
v___x_5666_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5665_);
lean_dec(v_x_5513_);
v___x_5667_ = l_Lean_Syntax_isNone(v___x_5666_);
if (v___x_5667_ == 0)
{
uint8_t v___x_5668_; 
lean_inc(v___x_5666_);
v___x_5668_ = l_Lean_Syntax_matchesNull(v___x_5666_, v___x_5629_);
if (v___x_5668_ == 0)
{
lean_object* v___x_5669_; 
lean_dec(v___x_5666_);
lean_dec(v_ps_x3f_5656_);
lean_dec(v___y_5655_);
lean_dec(v___y_5654_);
lean_dec(v___y_5653_);
lean_dec(v___y_5652_);
lean_dec(v___y_5651_);
v___x_5669_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5669_;
}
else
{
lean_object* v_loc_x3f_5670_; lean_object* v___x_5671_; uint8_t v___x_5672_; 
v_loc_x3f_5670_ = l_Lean_Syntax_getArg(v___x_5666_, v___x_5577_);
lean_dec(v___x_5666_);
v___x_5671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__4));
lean_inc(v_loc_x3f_5670_);
v___x_5672_ = l_Lean_Syntax_isOfKind(v_loc_x3f_5670_, v___x_5671_);
if (v___x_5672_ == 0)
{
lean_object* v___x_5673_; 
lean_dec(v_loc_x3f_5670_);
lean_dec(v_ps_x3f_5656_);
lean_dec(v___y_5655_);
lean_dec(v___y_5654_);
lean_dec(v___y_5653_);
lean_dec(v___y_5652_);
lean_dec(v___y_5651_);
v___x_5673_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5673_;
}
else
{
lean_object* v___x_5674_; 
v___x_5674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5674_, 0, v_loc_x3f_5670_);
v___y_5631_ = v___y_5652_;
v___y_5632_ = v___y_5651_;
v___y_5633_ = v_ps_x3f_5656_;
v___y_5634_ = v___y_5653_;
v___y_5635_ = v___y_5654_;
v___y_5636_ = v___y_5655_;
v_loc_x3f_5637_ = v___x_5674_;
v___y_5638_ = v___y_5657_;
v___y_5639_ = v___y_5658_;
v___y_5640_ = v___y_5659_;
v___y_5641_ = v___y_5660_;
v___y_5642_ = v___y_5661_;
v___y_5643_ = v___y_5662_;
v___y_5644_ = v___y_5663_;
v___y_5645_ = v___y_5664_;
goto v___jp_5630_;
}
}
}
else
{
lean_object* v___x_5675_; 
lean_dec(v___x_5666_);
v___x_5675_ = lean_box(0);
v___y_5631_ = v___y_5652_;
v___y_5632_ = v___y_5651_;
v___y_5633_ = v_ps_x3f_5656_;
v___y_5634_ = v___y_5653_;
v___y_5635_ = v___y_5654_;
v___y_5636_ = v___y_5655_;
v_loc_x3f_5637_ = v___x_5675_;
v___y_5638_ = v___y_5657_;
v___y_5639_ = v___y_5658_;
v___y_5640_ = v___y_5659_;
v___y_5641_ = v___y_5660_;
v___y_5642_ = v___y_5661_;
v___y_5643_ = v___y_5662_;
v___y_5644_ = v___y_5663_;
v___y_5645_ = v___y_5664_;
goto v___jp_5630_;
}
}
v___jp_5676_:
{
lean_object* v___x_5691_; lean_object* v___x_5692_; uint8_t v___x_5693_; 
v___x_5691_ = lean_unsigned_to_nat(6u);
v___x_5692_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5691_);
v___x_5693_ = l_Lean_Syntax_isNone(v___x_5692_);
if (v___x_5693_ == 0)
{
uint8_t v___x_5694_; 
lean_inc(v___x_5692_);
v___x_5694_ = l_Lean_Syntax_matchesNull(v___x_5692_, v___y_5678_);
if (v___x_5694_ == 0)
{
lean_object* v___x_5695_; 
lean_dec(v___x_5692_);
lean_dec(v_n_5690_);
lean_dec(v___y_5686_);
lean_dec(v___y_5685_);
lean_dec(v___y_5680_);
lean_dec(v___y_5677_);
lean_dec(v_x_5513_);
v___x_5695_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5695_;
}
else
{
lean_object* v___x_5696_; lean_object* v_ps_x3f_5697_; lean_object* v___x_5698_; 
v___x_5696_ = l_Lean_Syntax_getArg(v___x_5692_, v___x_5629_);
lean_dec(v___x_5692_);
v_ps_x3f_5697_ = l_Lean_Syntax_getArgs(v___x_5696_);
lean_dec(v___x_5696_);
v___x_5698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5698_, 0, v_ps_x3f_5697_);
v___y_5651_ = v___y_5680_;
v___y_5652_ = v___y_5686_;
v___y_5653_ = v___y_5685_;
v___y_5654_ = v___y_5677_;
v___y_5655_ = v_n_5690_;
v_ps_x3f_5656_ = v___x_5698_;
v___y_5657_ = v___y_5682_;
v___y_5658_ = v___y_5687_;
v___y_5659_ = v___y_5683_;
v___y_5660_ = v___y_5688_;
v___y_5661_ = v___y_5684_;
v___y_5662_ = v___y_5689_;
v___y_5663_ = v___y_5681_;
v___y_5664_ = v___y_5679_;
goto v___jp_5650_;
}
}
else
{
lean_object* v___x_5699_; 
lean_dec(v___x_5692_);
v___x_5699_ = lean_box(0);
v___y_5651_ = v___y_5680_;
v___y_5652_ = v___y_5686_;
v___y_5653_ = v___y_5685_;
v___y_5654_ = v___y_5677_;
v___y_5655_ = v_n_5690_;
v_ps_x3f_5656_ = v___x_5699_;
v___y_5657_ = v___y_5682_;
v___y_5658_ = v___y_5687_;
v___y_5659_ = v___y_5683_;
v___y_5660_ = v___y_5688_;
v___y_5661_ = v___y_5684_;
v___y_5662_ = v___y_5689_;
v___y_5663_ = v___y_5681_;
v___y_5664_ = v___y_5679_;
goto v___jp_5650_;
}
}
v___jp_5700_:
{
lean_object* v___x_5713_; lean_object* v___x_5714_; lean_object* v___x_5715_; lean_object* v___x_5716_; uint8_t v___x_5717_; 
v___x_5713_ = lean_unsigned_to_nat(4u);
v___x_5714_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5713_);
v___x_5715_ = lean_unsigned_to_nat(5u);
v___x_5716_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5715_);
v___x_5717_ = l_Lean_Syntax_isNone(v___x_5716_);
if (v___x_5717_ == 0)
{
uint8_t v___x_5718_; 
lean_inc(v___x_5716_);
v___x_5718_ = l_Lean_Syntax_matchesNull(v___x_5716_, v___y_5702_);
if (v___x_5718_ == 0)
{
lean_object* v___x_5719_; 
lean_dec(v___x_5716_);
lean_dec(v___x_5714_);
lean_dec(v_sym_5704_);
lean_dec(v___y_5703_);
lean_dec(v___y_5701_);
lean_dec(v_x_5513_);
v___x_5719_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5719_;
}
else
{
lean_object* v_n_5720_; lean_object* v___x_5721_; 
v_n_5720_ = l_Lean_Syntax_getArg(v___x_5716_, v___x_5629_);
lean_dec(v___x_5716_);
v___x_5721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5721_, 0, v_n_5720_);
v___y_5677_ = v___y_5703_;
v___y_5678_ = v___y_5702_;
v___y_5679_ = v___y_5712_;
v___y_5680_ = v_sym_5704_;
v___y_5681_ = v___y_5711_;
v___y_5682_ = v___y_5705_;
v___y_5683_ = v___y_5707_;
v___y_5684_ = v___y_5709_;
v___y_5685_ = v___y_5701_;
v___y_5686_ = v___x_5714_;
v___y_5687_ = v___y_5706_;
v___y_5688_ = v___y_5708_;
v___y_5689_ = v___y_5710_;
v_n_5690_ = v___x_5721_;
goto v___jp_5676_;
}
}
else
{
lean_object* v___x_5722_; 
lean_dec(v___x_5716_);
v___x_5722_ = lean_box(0);
v___y_5677_ = v___y_5703_;
v___y_5678_ = v___y_5702_;
v___y_5679_ = v___y_5712_;
v___y_5680_ = v_sym_5704_;
v___y_5681_ = v___y_5711_;
v___y_5682_ = v___y_5705_;
v___y_5683_ = v___y_5707_;
v___y_5684_ = v___y_5709_;
v___y_5685_ = v___y_5701_;
v___y_5686_ = v___x_5714_;
v___y_5687_ = v___y_5706_;
v___y_5688_ = v___y_5708_;
v___y_5689_ = v___y_5710_;
v_n_5690_ = v___x_5722_;
goto v___jp_5676_;
}
}
v___jp_5723_:
{
lean_object* v___x_5733_; lean_object* v___x_5734_; lean_object* v___x_5735_; lean_object* v___x_5736_; uint8_t v___x_5737_; 
v___x_5733_ = lean_unsigned_to_nat(2u);
v___x_5734_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5733_);
v___x_5735_ = lean_unsigned_to_nat(3u);
v___x_5736_ = l_Lean_Syntax_getArg(v_x_5513_, v___x_5735_);
v___x_5737_ = l_Lean_Syntax_isNone(v___x_5736_);
if (v___x_5737_ == 0)
{
uint8_t v___x_5738_; 
lean_inc(v___x_5736_);
v___x_5738_ = l_Lean_Syntax_matchesNull(v___x_5736_, v___x_5629_);
if (v___x_5738_ == 0)
{
lean_object* v___x_5739_; 
lean_dec(v___x_5736_);
lean_dec(v___x_5734_);
lean_dec(v_expensive_5724_);
lean_dec(v_x_5513_);
v___x_5739_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convert__1_spec__0___redArg();
return v___x_5739_;
}
else
{
lean_object* v_sym_5740_; lean_object* v___x_5741_; 
v_sym_5740_ = l_Lean_Syntax_getArg(v___x_5736_, v___x_5577_);
lean_dec(v___x_5736_);
v___x_5741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5741_, 0, v_sym_5740_);
v___y_5701_ = v_expensive_5724_;
v___y_5702_ = v___x_5733_;
v___y_5703_ = v___x_5734_;
v_sym_5704_ = v___x_5741_;
v___y_5705_ = v___y_5725_;
v___y_5706_ = v___y_5726_;
v___y_5707_ = v___y_5727_;
v___y_5708_ = v___y_5728_;
v___y_5709_ = v___y_5729_;
v___y_5710_ = v___y_5730_;
v___y_5711_ = v___y_5731_;
v___y_5712_ = v___y_5732_;
goto v___jp_5700_;
}
}
else
{
lean_object* v___x_5742_; 
lean_dec(v___x_5736_);
v___x_5742_ = lean_box(0);
v___y_5701_ = v_expensive_5724_;
v___y_5702_ = v___x_5733_;
v___y_5703_ = v___x_5734_;
v_sym_5704_ = v___x_5742_;
v___y_5705_ = v___y_5725_;
v___y_5706_ = v___y_5726_;
v___y_5707_ = v___y_5727_;
v___y_5708_ = v___y_5728_;
v___y_5709_ = v___y_5729_;
v___y_5710_ = v___y_5730_;
v___y_5711_ = v___y_5731_;
v___y_5712_ = v___y_5732_;
goto v___jp_5700_;
}
}
}
v___jp_5523_:
{
lean_object* v___x_5536_; lean_object* v___x_5537_; lean_object* v___x_5538_; 
v___x_5536_ = l_Lean_mkOptionalNode(v___y_5535_);
v___x_5537_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_5536_);
lean_dec(v___x_5536_);
lean_inc_ref(v___y_5528_);
v___x_5538_ = l_Lean_Elab_Tactic_withLocation(v___x_5537_, v___y_5533_, v___y_5531_, v___y_5528_, v___y_5525_, v___y_5524_, v___y_5532_, v___y_5530_, v___y_5534_, v___y_5529_, v___y_5527_, v___y_5526_);
lean_dec(v___x_5537_);
return v___x_5538_;
}
v___jp_5543_:
{
lean_object* v___x_5561_; lean_object* v___x_5562_; lean_object* v___f_5563_; lean_object* v___x_5564_; lean_object* v___f_5565_; 
v___x_5561_ = lean_array_to_list(v___y_5560_);
v___x_5562_ = lean_box(v___x_5542_);
lean_inc(v___y_5548_);
lean_inc(v___x_5561_);
lean_inc_ref(v___y_5547_);
lean_inc(v___y_5545_);
lean_inc(v___y_5549_);
v___f_5563_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__2___boxed), 16, 6);
lean_closure_set(v___f_5563_, 0, v___y_5549_);
lean_closure_set(v___f_5563_, 1, v___y_5545_);
lean_closure_set(v___f_5563_, 2, v___y_5547_);
lean_closure_set(v___f_5563_, 3, v___x_5561_);
lean_closure_set(v___f_5563_, 4, v___y_5548_);
lean_closure_set(v___f_5563_, 5, v___x_5562_);
v___x_5564_ = lean_box(v___x_5542_);
lean_inc_ref(v___y_5544_);
lean_inc_ref(v___y_5546_);
v___f_5565_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___lam__4___boxed), 19, 10);
lean_closure_set(v___f_5565_, 0, v___y_5546_);
lean_closure_set(v___f_5565_, 1, v___y_5544_);
lean_closure_set(v___f_5565_, 2, v___x_5539_);
lean_closure_set(v___f_5565_, 3, v___x_5540_);
lean_closure_set(v___f_5565_, 4, v___y_5549_);
lean_closure_set(v___f_5565_, 5, v___y_5545_);
lean_closure_set(v___f_5565_, 6, v___y_5547_);
lean_closure_set(v___f_5565_, 7, v___x_5561_);
lean_closure_set(v___f_5565_, 8, v___y_5548_);
lean_closure_set(v___f_5565_, 9, v___x_5564_);
if (lean_obj_tag(v___y_5552_) == 0)
{
lean_object* v___x_5566_; 
v___x_5566_ = lean_box(0);
v___y_5524_ = v___y_5551_;
v___y_5525_ = v___y_5550_;
v___y_5526_ = v___y_5553_;
v___y_5527_ = v___y_5555_;
v___y_5528_ = v___y_5554_;
v___y_5529_ = v___y_5556_;
v___y_5530_ = v___y_5557_;
v___y_5531_ = v___f_5565_;
v___y_5532_ = v___y_5558_;
v___y_5533_ = v___f_5563_;
v___y_5534_ = v___y_5559_;
v___y_5535_ = v___x_5566_;
goto v___jp_5523_;
}
else
{
lean_object* v_val_5567_; lean_object* v___x_5569_; uint8_t v_isShared_5570_; uint8_t v_isSharedCheck_5574_; 
v_val_5567_ = lean_ctor_get(v___y_5552_, 0);
v_isSharedCheck_5574_ = !lean_is_exclusive(v___y_5552_);
if (v_isSharedCheck_5574_ == 0)
{
v___x_5569_ = v___y_5552_;
v_isShared_5570_ = v_isSharedCheck_5574_;
goto v_resetjp_5568_;
}
else
{
lean_inc(v_val_5567_);
lean_dec(v___y_5552_);
v___x_5569_ = lean_box(0);
v_isShared_5570_ = v_isSharedCheck_5574_;
goto v_resetjp_5568_;
}
v_resetjp_5568_:
{
lean_object* v___x_5572_; 
if (v_isShared_5570_ == 0)
{
v___x_5572_ = v___x_5569_;
goto v_reusejp_5571_;
}
else
{
lean_object* v_reuseFailAlloc_5573_; 
v_reuseFailAlloc_5573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5573_, 0, v_val_5567_);
v___x_5572_ = v_reuseFailAlloc_5573_;
goto v_reusejp_5571_;
}
v_reusejp_5571_:
{
v___y_5524_ = v___y_5551_;
v___y_5525_ = v___y_5550_;
v___y_5526_ = v___y_5553_;
v___y_5527_ = v___y_5555_;
v___y_5528_ = v___y_5554_;
v___y_5529_ = v___y_5556_;
v___y_5530_ = v___y_5557_;
v___y_5531_ = v___f_5565_;
v___y_5532_ = v___y_5558_;
v___y_5533_ = v___f_5563_;
v___y_5534_ = v___y_5559_;
v___y_5535_ = v___x_5572_;
goto v___jp_5523_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___boxed(lean_object* v_x_5750_, lean_object* v_a_5751_, lean_object* v_a_5752_, lean_object* v_a_5753_, lean_object* v_a_5754_, lean_object* v_a_5755_, lean_object* v_a_5756_, lean_object* v_a_5757_, lean_object* v_a_5758_, lean_object* v_a_5759_){
_start:
{
lean_object* v_res_5760_; 
v_res_5760_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1(v_x_5750_, v_a_5751_, v_a_5752_, v_a_5753_, v_a_5754_, v_a_5755_, v_a_5756_, v_a_5757_, v_a_5758_);
lean_dec(v_a_5758_);
lean_dec_ref(v_a_5757_);
lean_dec(v_a_5756_);
lean_dec_ref(v_a_5755_);
lean_dec(v_a_5754_);
lean_dec_ref(v_a_5753_);
lean_dec(v_a_5752_);
lean_dec_ref(v_a_5751_);
return v_res_5760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0(lean_object* v_00_u03b1_5761_, lean_object* v_msg_5762_, lean_object* v___y_5763_, lean_object* v___y_5764_, lean_object* v___y_5765_, lean_object* v___y_5766_, lean_object* v___y_5767_, lean_object* v___y_5768_, lean_object* v___y_5769_, lean_object* v___y_5770_){
_start:
{
lean_object* v___x_5772_; 
v___x_5772_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___redArg(v_msg_5762_, v___y_5767_, v___y_5768_, v___y_5769_, v___y_5770_);
return v___x_5772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0___boxed(lean_object* v_00_u03b1_5773_, lean_object* v_msg_5774_, lean_object* v___y_5775_, lean_object* v___y_5776_, lean_object* v___y_5777_, lean_object* v___y_5778_, lean_object* v___y_5779_, lean_object* v___y_5780_, lean_object* v___y_5781_, lean_object* v___y_5782_, lean_object* v___y_5783_){
_start:
{
lean_object* v_res_5784_; 
v_res_5784_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1_spec__0(v_00_u03b1_5773_, v_msg_5774_, v___y_5775_, v___y_5776_, v___y_5777_, v___y_5778_, v___y_5779_, v___y_5780_, v___y_5781_, v___y_5782_);
lean_dec(v___y_5782_);
lean_dec_ref(v___y_5781_);
lean_dec(v___y_5780_);
lean_dec_ref(v___y_5779_);
lean_dec(v___y_5778_);
lean_dec_ref(v___y_5777_);
lean_dec(v___y_5776_);
lean_dec_ref(v___y_5775_);
return v_res_5784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1(lean_object* v_x_5848_, lean_object* v_a_5849_, lean_object* v_a_5850_){
_start:
{
lean_object* v___x_5851_; lean_object* v___x_5852_; uint8_t v___x_5853_; 
v___x_5851_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__1));
v___x_5852_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_acChange___closed__1));
lean_inc(v_x_5848_);
v___x_5853_ = l_Lean_Syntax_isOfKind(v_x_5848_, v___x_5852_);
if (v___x_5853_ == 0)
{
lean_object* v___x_5854_; lean_object* v___x_5855_; 
lean_dec(v_x_5848_);
v___x_5854_ = lean_box(1);
v___x_5855_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5855_, 0, v___x_5854_);
lean_ctor_set(v___x_5855_, 1, v_a_5850_);
return v___x_5855_;
}
else
{
lean_object* v___x_5856_; lean_object* v___x_5857_; lean_object* v___y_5859_; lean_object* v___y_5860_; lean_object* v___y_5861_; lean_object* v___y_5862_; lean_object* v___y_5863_; lean_object* v___y_5864_; lean_object* v___y_5865_; lean_object* v___y_5866_; lean_object* v___y_5867_; lean_object* v___y_5868_; lean_object* v___y_5869_; lean_object* v___y_5870_; lean_object* v_n_5896_; lean_object* v___y_5897_; lean_object* v___y_5898_; lean_object* v___x_5918_; lean_object* v___x_5919_; uint8_t v___x_5920_; 
v___x_5856_ = lean_unsigned_to_nat(1u);
v___x_5857_ = l_Lean_Syntax_getArg(v_x_5848_, v___x_5856_);
v___x_5918_ = lean_unsigned_to_nat(2u);
v___x_5919_ = l_Lean_Syntax_getArg(v_x_5848_, v___x_5918_);
lean_dec(v_x_5848_);
v___x_5920_ = l_Lean_Syntax_isNone(v___x_5919_);
if (v___x_5920_ == 0)
{
uint8_t v___x_5921_; 
lean_inc(v___x_5919_);
v___x_5921_ = l_Lean_Syntax_matchesNull(v___x_5919_, v___x_5918_);
if (v___x_5921_ == 0)
{
lean_object* v___x_5922_; lean_object* v___x_5923_; 
lean_dec(v___x_5919_);
lean_dec(v___x_5857_);
v___x_5922_ = lean_box(1);
v___x_5923_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5923_, 0, v___x_5922_);
lean_ctor_set(v___x_5923_, 1, v_a_5850_);
return v___x_5923_;
}
else
{
lean_object* v_n_5924_; lean_object* v___x_5925_; 
v_n_5924_ = l_Lean_Syntax_getArg(v___x_5919_, v___x_5856_);
lean_dec(v___x_5919_);
v___x_5925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5925_, 0, v_n_5924_);
v_n_5896_ = v___x_5925_;
v___y_5897_ = v_a_5849_;
v___y_5898_ = v_a_5850_;
goto v___jp_5895_;
}
}
else
{
lean_object* v___x_5926_; 
lean_dec(v___x_5919_);
v___x_5926_ = lean_box(0);
v_n_5896_ = v___x_5926_;
v___y_5897_ = v_a_5849_;
v___y_5898_ = v_a_5850_;
goto v___jp_5895_;
}
v___jp_5858_:
{
lean_object* v___x_5871_; lean_object* v___x_5872_; lean_object* v___x_5873_; lean_object* v___x_5874_; lean_object* v___x_5875_; lean_object* v___x_5876_; lean_object* v___x_5877_; lean_object* v___x_5878_; lean_object* v___x_5879_; lean_object* v___x_5880_; lean_object* v___x_5881_; lean_object* v___x_5882_; lean_object* v___x_5883_; lean_object* v___x_5884_; lean_object* v___x_5885_; lean_object* v___x_5886_; lean_object* v___x_5887_; lean_object* v___x_5888_; lean_object* v___x_5889_; lean_object* v___x_5890_; lean_object* v___x_5891_; lean_object* v___x_5892_; lean_object* v___x_5893_; lean_object* v___x_5894_; 
lean_inc_ref(v___y_5862_);
v___x_5871_ = l_Array_append___redArg(v___y_5862_, v___y_5870_);
lean_dec_ref(v___y_5870_);
lean_inc_n(v___y_5866_, 2);
lean_inc_n(v___y_5861_, 10);
v___x_5872_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5872_, 0, v___y_5861_);
lean_ctor_set(v___x_5872_, 1, v___y_5866_);
lean_ctor_set(v___x_5872_, 2, v___x_5871_);
lean_inc_n(v___y_5864_, 3);
lean_inc(v___y_5860_);
v___x_5873_ = l_Lean_Syntax_node8(v___y_5861_, v___y_5860_, v___y_5865_, v___y_5864_, v___y_5867_, v___y_5864_, v___x_5857_, v___x_5872_, v___y_5864_, v___y_5864_);
v___x_5874_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__0));
v___x_5875_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5875_, 0, v___y_5861_);
lean_ctor_set(v___x_5875_, 1, v___x_5874_);
v___x_5876_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__1));
lean_inc_ref_n(v___y_5863_, 4);
lean_inc_ref_n(v___y_5868_, 4);
v___x_5877_ = l_Lean_Name_mkStr4(v___y_5868_, v___y_5863_, v___x_5851_, v___x_5876_);
v___x_5878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__2));
v___x_5879_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5879_, 0, v___y_5861_);
lean_ctor_set(v___x_5879_, 1, v___x_5878_);
v___x_5880_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__3));
v___x_5881_ = l_Lean_Name_mkStr4(v___y_5868_, v___y_5863_, v___x_5851_, v___x_5880_);
v___x_5882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__4));
v___x_5883_ = l_Lean_Name_mkStr4(v___y_5868_, v___y_5863_, v___x_5851_, v___x_5882_);
v___x_5884_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__5));
v___x_5885_ = l_Lean_Name_mkStr4(v___y_5868_, v___y_5863_, v___x_5851_, v___x_5884_);
v___x_5886_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__6));
v___x_5887_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5887_, 0, v___y_5861_);
lean_ctor_set(v___x_5887_, 1, v___x_5886_);
v___x_5888_ = l_Lean_Syntax_node1(v___y_5861_, v___x_5885_, v___x_5887_);
v___x_5889_ = l_Lean_Syntax_node1(v___y_5861_, v___y_5866_, v___x_5888_);
v___x_5890_ = l_Lean_Syntax_node1(v___y_5861_, v___x_5883_, v___x_5889_);
v___x_5891_ = l_Lean_Syntax_node1(v___y_5861_, v___x_5881_, v___x_5890_);
v___x_5892_ = l_Lean_Syntax_node2(v___y_5861_, v___x_5877_, v___x_5879_, v___x_5891_);
lean_inc(v___y_5859_);
v___x_5893_ = l_Lean_Syntax_node3(v___y_5861_, v___y_5859_, v___x_5873_, v___x_5875_, v___x_5892_);
v___x_5894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5894_, 0, v___x_5893_);
lean_ctor_set(v___x_5894_, 1, v___y_5869_);
return v___x_5894_;
}
v___jp_5895_:
{
lean_object* v_ref_5899_; uint8_t v___x_5900_; lean_object* v___x_5901_; lean_object* v___x_5902_; lean_object* v___x_5903_; lean_object* v___x_5904_; lean_object* v___x_5905_; lean_object* v___x_5906_; lean_object* v___x_5907_; lean_object* v___x_5908_; lean_object* v___x_5909_; lean_object* v___x_5910_; lean_object* v___x_5911_; lean_object* v___x_5912_; 
v_ref_5899_ = lean_ctor_get(v___y_5897_, 5);
v___x_5900_ = 0;
v___x_5901_ = l_Lean_SourceInfo_fromRef(v_ref_5899_, v___x_5900_);
v___x_5902_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0));
v___x_5903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2));
v___x_5904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8));
v___x_5905_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__1));
v___x_5906_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convertTo___closed__2));
lean_inc_n(v___x_5901_, 3);
v___x_5907_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5907_, 0, v___x_5901_);
lean_ctor_set(v___x_5907_, 1, v___x_5906_);
v___x_5908_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4));
v___x_5909_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5);
v___x_5910_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5910_, 0, v___x_5901_);
lean_ctor_set(v___x_5910_, 1, v___x_5908_);
lean_ctor_set(v___x_5910_, 2, v___x_5909_);
v___x_5911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10));
lean_inc_ref(v___x_5910_);
v___x_5912_ = l_Lean_Syntax_node1(v___x_5901_, v___x_5911_, v___x_5910_);
if (lean_obj_tag(v_n_5896_) == 1)
{
lean_object* v_val_5913_; lean_object* v___x_5914_; lean_object* v___x_5915_; lean_object* v___x_5916_; 
v_val_5913_ = lean_ctor_get(v_n_5896_, 0);
lean_inc(v_val_5913_);
lean_dec_ref_known(v_n_5896_, 1);
v___x_5914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2));
lean_inc(v___x_5901_);
v___x_5915_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5915_, 0, v___x_5901_);
lean_ctor_set(v___x_5915_, 1, v___x_5914_);
v___x_5916_ = l_Array_mkArray2___redArg(v___x_5915_, v_val_5913_);
v___y_5859_ = v___x_5904_;
v___y_5860_ = v___x_5905_;
v___y_5861_ = v___x_5901_;
v___y_5862_ = v___x_5909_;
v___y_5863_ = v___x_5903_;
v___y_5864_ = v___x_5910_;
v___y_5865_ = v___x_5907_;
v___y_5866_ = v___x_5908_;
v___y_5867_ = v___x_5912_;
v___y_5868_ = v___x_5902_;
v___y_5869_ = v___y_5898_;
v___y_5870_ = v___x_5916_;
goto v___jp_5858_;
}
else
{
lean_object* v___x_5917_; 
lean_dec(v_n_5896_);
v___x_5917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_5859_ = v___x_5904_;
v___y_5860_ = v___x_5905_;
v___y_5861_ = v___x_5901_;
v___y_5862_ = v___x_5909_;
v___y_5863_ = v___x_5903_;
v___y_5864_ = v___x_5910_;
v___y_5865_ = v___x_5907_;
v___y_5866_ = v___x_5908_;
v___y_5867_ = v___x_5912_;
v___y_5868_ = v___x_5902_;
v___y_5869_ = v___y_5898_;
v___y_5870_ = v___x_5917_;
goto v___jp_5858_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___boxed(lean_object* v_x_5927_, lean_object* v_a_5928_, lean_object* v_a_5929_){
_start:
{
lean_object* v_res_5930_; 
v_res_5930_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1(v_x_5927_, v_a_5928_, v_a_5929_);
lean_dec_ref(v_a_5928_);
return v_res_5930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange_x21__1(lean_object* v_x_5931_, lean_object* v_a_5932_, lean_object* v_a_5933_){
_start:
{
lean_object* v___x_5934_; lean_object* v___x_5935_; uint8_t v___x_5936_; 
v___x_5934_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert___closed__1));
v___x_5935_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_acChange_x21___closed__1));
lean_inc(v_x_5931_);
v___x_5936_ = l_Lean_Syntax_isOfKind(v_x_5931_, v___x_5935_);
if (v___x_5936_ == 0)
{
lean_object* v___x_5937_; lean_object* v___x_5938_; 
lean_dec(v_x_5931_);
v___x_5937_ = lean_box(1);
v___x_5938_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5938_, 0, v___x_5937_);
lean_ctor_set(v___x_5938_, 1, v_a_5933_);
return v___x_5938_;
}
else
{
lean_object* v___x_5939_; lean_object* v___x_5940_; lean_object* v___y_5942_; lean_object* v___y_5943_; lean_object* v___y_5944_; lean_object* v___y_5945_; lean_object* v___y_5946_; lean_object* v___y_5947_; lean_object* v___y_5948_; lean_object* v___y_5949_; lean_object* v___y_5950_; lean_object* v___y_5951_; lean_object* v___y_5952_; lean_object* v___y_5953_; lean_object* v_n_5979_; lean_object* v___y_5980_; lean_object* v___y_5981_; lean_object* v___x_6001_; lean_object* v___x_6002_; uint8_t v___x_6003_; 
v___x_5939_ = lean_unsigned_to_nat(1u);
v___x_5940_ = l_Lean_Syntax_getArg(v_x_5931_, v___x_5939_);
v___x_6001_ = lean_unsigned_to_nat(2u);
v___x_6002_ = l_Lean_Syntax_getArg(v_x_5931_, v___x_6001_);
lean_dec(v_x_5931_);
v___x_6003_ = l_Lean_Syntax_isNone(v___x_6002_);
if (v___x_6003_ == 0)
{
uint8_t v___x_6004_; 
lean_inc(v___x_6002_);
v___x_6004_ = l_Lean_Syntax_matchesNull(v___x_6002_, v___x_6001_);
if (v___x_6004_ == 0)
{
lean_object* v___x_6005_; lean_object* v___x_6006_; 
lean_dec(v___x_6002_);
lean_dec(v___x_5940_);
v___x_6005_ = lean_box(1);
v___x_6006_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_6006_, 0, v___x_6005_);
lean_ctor_set(v___x_6006_, 1, v_a_5933_);
return v___x_6006_;
}
else
{
lean_object* v_n_6007_; lean_object* v___x_6008_; 
v_n_6007_ = l_Lean_Syntax_getArg(v___x_6002_, v___x_5939_);
lean_dec(v___x_6002_);
v___x_6008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6008_, 0, v_n_6007_);
v_n_5979_ = v___x_6008_;
v___y_5980_ = v_a_5932_;
v___y_5981_ = v_a_5933_;
goto v___jp_5978_;
}
}
else
{
lean_object* v___x_6009_; 
lean_dec(v___x_6002_);
v___x_6009_ = lean_box(0);
v_n_5979_ = v___x_6009_;
v___y_5980_ = v_a_5932_;
v___y_5981_ = v_a_5933_;
goto v___jp_5978_;
}
v___jp_5941_:
{
lean_object* v___x_5954_; lean_object* v___x_5955_; lean_object* v___x_5956_; lean_object* v___x_5957_; lean_object* v___x_5958_; lean_object* v___x_5959_; lean_object* v___x_5960_; lean_object* v___x_5961_; lean_object* v___x_5962_; lean_object* v___x_5963_; lean_object* v___x_5964_; lean_object* v___x_5965_; lean_object* v___x_5966_; lean_object* v___x_5967_; lean_object* v___x_5968_; lean_object* v___x_5969_; lean_object* v___x_5970_; lean_object* v___x_5971_; lean_object* v___x_5972_; lean_object* v___x_5973_; lean_object* v___x_5974_; lean_object* v___x_5975_; lean_object* v___x_5976_; lean_object* v___x_5977_; 
lean_inc_ref(v___y_5952_);
v___x_5954_ = l_Array_append___redArg(v___y_5952_, v___y_5953_);
lean_dec_ref(v___y_5953_);
lean_inc_n(v___y_5944_, 2);
lean_inc_n(v___y_5946_, 10);
v___x_5955_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5955_, 0, v___y_5946_);
lean_ctor_set(v___x_5955_, 1, v___y_5944_);
lean_ctor_set(v___x_5955_, 2, v___x_5954_);
lean_inc_n(v___y_5949_, 2);
lean_inc(v___y_5943_);
v___x_5956_ = l_Lean_Syntax_node7(v___y_5946_, v___y_5943_, v___y_5948_, v___y_5942_, v___y_5949_, v___x_5940_, v___x_5955_, v___y_5949_, v___y_5949_);
v___x_5957_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__0));
v___x_5958_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5958_, 0, v___y_5946_);
lean_ctor_set(v___x_5958_, 1, v___x_5957_);
v___x_5959_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__1));
lean_inc_ref_n(v___y_5945_, 4);
lean_inc_ref_n(v___y_5947_, 4);
v___x_5960_ = l_Lean_Name_mkStr4(v___y_5947_, v___y_5945_, v___x_5934_, v___x_5959_);
v___x_5961_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__2));
v___x_5962_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5962_, 0, v___y_5946_);
lean_ctor_set(v___x_5962_, 1, v___x_5961_);
v___x_5963_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__3));
v___x_5964_ = l_Lean_Name_mkStr4(v___y_5947_, v___y_5945_, v___x_5934_, v___x_5963_);
v___x_5965_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__4));
v___x_5966_ = l_Lean_Name_mkStr4(v___y_5947_, v___y_5945_, v___x_5934_, v___x_5965_);
v___x_5967_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__5));
v___x_5968_ = l_Lean_Name_mkStr4(v___y_5947_, v___y_5945_, v___x_5934_, v___x_5967_);
v___x_5969_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__6));
v___x_5970_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5970_, 0, v___y_5946_);
lean_ctor_set(v___x_5970_, 1, v___x_5969_);
v___x_5971_ = l_Lean_Syntax_node1(v___y_5946_, v___x_5968_, v___x_5970_);
v___x_5972_ = l_Lean_Syntax_node1(v___y_5946_, v___y_5944_, v___x_5971_);
v___x_5973_ = l_Lean_Syntax_node1(v___y_5946_, v___x_5966_, v___x_5972_);
v___x_5974_ = l_Lean_Syntax_node1(v___y_5946_, v___x_5964_, v___x_5973_);
v___x_5975_ = l_Lean_Syntax_node2(v___y_5946_, v___x_5960_, v___x_5962_, v___x_5974_);
lean_inc(v___y_5950_);
v___x_5976_ = l_Lean_Syntax_node3(v___y_5946_, v___y_5950_, v___x_5956_, v___x_5958_, v___x_5975_);
v___x_5977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5977_, 0, v___x_5976_);
lean_ctor_set(v___x_5977_, 1, v___y_5951_);
return v___x_5977_;
}
v___jp_5978_:
{
lean_object* v_ref_5982_; uint8_t v___x_5983_; lean_object* v___x_5984_; lean_object* v___x_5985_; lean_object* v___x_5986_; lean_object* v___x_5987_; lean_object* v___x_5988_; lean_object* v___x_5989_; lean_object* v___x_5990_; lean_object* v___x_5991_; lean_object* v___x_5992_; lean_object* v___x_5993_; lean_object* v___x_5994_; lean_object* v___x_5995_; 
v_ref_5982_ = lean_ctor_get(v___y_5980_, 5);
v___x_5983_ = 0;
v___x_5984_ = l_Lean_SourceInfo_fromRef(v_ref_5982_, v___x_5983_);
v___x_5985_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Convert_0__Convert_elabCheapConfig_evalConfigItem_spec__0_spec__0___closed__0));
v___x_5986_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______elabRules__Mathlib__Tactic__convertTo__1___closed__2));
v___x_5987_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__8));
v___x_5988_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__0));
v___x_5989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convert__to_x21___closed__1));
lean_inc_n(v___x_5984_, 3);
v___x_5990_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5990_, 0, v___x_5984_);
lean_ctor_set(v___x_5990_, 1, v___x_5988_);
v___x_5991_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange__1___closed__10));
v___x_5992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__4));
v___x_5993_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__5);
v___x_5994_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5994_, 0, v___x_5984_);
lean_ctor_set(v___x_5994_, 1, v___x_5992_);
lean_ctor_set(v___x_5994_, 2, v___x_5993_);
lean_inc_ref(v___x_5994_);
v___x_5995_ = l_Lean_Syntax_node1(v___x_5984_, v___x_5991_, v___x_5994_);
if (lean_obj_tag(v_n_5979_) == 1)
{
lean_object* v_val_5996_; lean_object* v___x_5997_; lean_object* v___x_5998_; lean_object* v___x_5999_; 
v_val_5996_ = lean_ctor_get(v_n_5979_, 0);
lean_inc(v_val_5996_);
lean_dec_ref_known(v_n_5979_, 1);
v___x_5997_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__2));
lean_inc(v___x_5984_);
v___x_5998_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5998_, 0, v___x_5984_);
lean_ctor_set(v___x_5998_, 1, v___x_5997_);
v___x_5999_ = l_Array_mkArray2___redArg(v___x_5998_, v_val_5996_);
v___y_5942_ = v___x_5995_;
v___y_5943_ = v___x_5989_;
v___y_5944_ = v___x_5992_;
v___y_5945_ = v___x_5986_;
v___y_5946_ = v___x_5984_;
v___y_5947_ = v___x_5985_;
v___y_5948_ = v___x_5990_;
v___y_5949_ = v___x_5994_;
v___y_5950_ = v___x_5987_;
v___y_5951_ = v___y_5981_;
v___y_5952_ = v___x_5993_;
v___y_5953_ = v___x_5999_;
goto v___jp_5941_;
}
else
{
lean_object* v___x_6000_; 
lean_dec(v_n_5979_);
v___x_6000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__convert_x21__1___closed__1));
v___y_5942_ = v___x_5995_;
v___y_5943_ = v___x_5989_;
v___y_5944_ = v___x_5992_;
v___y_5945_ = v___x_5986_;
v___y_5946_ = v___x_5984_;
v___y_5947_ = v___x_5985_;
v___y_5948_ = v___x_5990_;
v___y_5949_ = v___x_5994_;
v___y_5950_ = v___x_5987_;
v___y_5951_ = v___y_5981_;
v___y_5952_ = v___x_5993_;
v___y_5953_ = v___x_6000_;
goto v___jp_5941_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange_x21__1___boxed(lean_object* v_x_6010_, lean_object* v_a_6011_, lean_object* v_a_6012_){
_start:
{
lean_object* v_res_6013_; 
v_res_6013_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Convert______macroRules__Mathlib__Tactic__acChange_x21__1(v_x_6010_, v_a_6011_, v_a_6012_);
lean_dec_ref(v_a_6011_);
return v_res_6013_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprConfig);
lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig = _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprCheapConfig);
lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig = _init_lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Convert_0__instEvalExprExpensiveConfig);
lp_mathlib_Mathlib_Tactic_convert = _init_lp_mathlib_Mathlib_Tactic_convert();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_convert);
lp_mathlib_Mathlib_Tactic_convert_x21 = _init_lp_mathlib_Mathlib_Tactic_convert_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_convert_x21);
lp_mathlib_Mathlib_Tactic_convertTo = _init_lp_mathlib_Mathlib_Tactic_convertTo();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_convertTo);
lp_mathlib_Mathlib_Tactic_convert__to_x21 = _init_lp_mathlib_Mathlib_Tactic_convert__to_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_convert__to_x21);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Convert(builtin);
}
#ifdef __cplusplus
}
#endif
