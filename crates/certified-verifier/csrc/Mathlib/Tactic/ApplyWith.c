// Lean compiler output
// Module: Mathlib.Tactic.ApplyWith
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Eval public meta import Lean.Elab.Tactic.ElabTerm public meta import Lean.Elab.ConfigEval public import Lean.Elab.ConfigEval
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_instEvalTermApplyNewGoals_evalTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_instEvalExprApplyNewGoals_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_configItem;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalApplyLikeTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "manyConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 217, 207, 163, 23, 252, 246, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__4_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_manyConfig___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_manyConfig___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_manyConfig___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_manyConfig___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_manyConfig___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_manyConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ApplyConfig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(160, 10, 102, 119, 166, 230, 192, 129)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ApplyNewGoals"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 184, 156, 67, 64, 216, 140, 26)}};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "allowSynthFailures"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "approx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "newGoals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "synthAssignedInstances"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(160, 10, 102, 119, 166, 230, 192, 129)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(90, 193, 0, 190, 186, 169, 206, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(160, 10, 102, 119, 166, 230, 192, 129)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(88, 179, 125, 206, 32, 93, 161, 129)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(160, 10, 102, 119, 166, 230, 192, 129)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(6, 168, 186, 145, 204, 47, 162, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(160, 10, 102, 119, 166, 230, 192, 129)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(39, 183, 221, 177, 173, 55, 205, 95)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyWith___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "applyWith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_manyConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 196, 43, 219, 46, 98, 149, 210)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyWith___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyWith___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyWith___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__5_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyWith___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__7_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyWith___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyWith___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__12_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyWith___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyWith___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyWith___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyWith___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyWith___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyWith;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__11(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_19_ = l_Lean_Parser_Tactic_configItem;
v___x_20_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__10));
v___x_21_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__7));
v___x_22_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v___x_20_);
lean_ctor_set(v___x_22_, 2, v___x_19_);
return v___x_22_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__12(void){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_23_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_manyConfig___closed__11, &lp_mathlib_Mathlib_Tactic_manyConfig___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__11);
v___x_24_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__5));
v___x_25_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_25_, 0, v___x_24_);
lean_ctor_set(v___x_25_, 1, v___x_23_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__13(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_26_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_manyConfig___closed__12, &lp_mathlib_Mathlib_Tactic_manyConfig___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__12);
v___x_27_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__3));
v___x_28_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__0));
v___x_29_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
lean_ctor_set(v___x_29_, 1, v___x_27_);
lean_ctor_set(v___x_29_, 2, v___x_26_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_manyConfig(void){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_manyConfig___closed__13, &lp_mathlib_Mathlib_Tactic_manyConfig___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_manyConfig___closed__13);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_31_ = lean_box(0);
v___x_32_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_33_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___closed__0);
v___x_36_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg___boxed(lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg();
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0(lean_object* v_00_u03b1_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg();
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0(v_00_u03b1_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1(lean_object* v_msgData_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v___x_59_; lean_object* v_env_60_; lean_object* v___x_61_; lean_object* v_mctx_62_; lean_object* v_lctx_63_; lean_object* v_options_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_59_ = lean_st_ref_get(v___y_57_);
v_env_60_ = lean_ctor_get(v___x_59_, 0);
lean_inc_ref(v_env_60_);
lean_dec(v___x_59_);
v___x_61_ = lean_st_ref_get(v___y_55_);
v_mctx_62_ = lean_ctor_get(v___x_61_, 0);
lean_inc_ref(v_mctx_62_);
lean_dec(v___x_61_);
v_lctx_63_ = lean_ctor_get(v___y_54_, 2);
v_options_64_ = lean_ctor_get(v___y_56_, 2);
lean_inc_ref(v_options_64_);
lean_inc_ref(v_lctx_63_);
v___x_65_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_65_, 0, v_env_60_);
lean_ctor_set(v___x_65_, 1, v_mctx_62_);
lean_ctor_set(v___x_65_, 2, v_lctx_63_);
lean_ctor_set(v___x_65_, 3, v_options_64_);
v___x_66_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v_msgData_53_);
v___x_67_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1(v_msgData_68_, v___y_69_, v___y_70_, v___y_71_, v___y_72_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
lean_dec(v___y_70_);
lean_dec_ref(v___y_69_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg(lean_object* v_msg_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
lean_object* v_ref_81_; lean_object* v___x_82_; lean_object* v_a_83_; lean_object* v___x_85_; uint8_t v_isShared_86_; uint8_t v_isSharedCheck_91_; 
v_ref_81_ = lean_ctor_get(v___y_78_, 5);
v___x_82_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1(v_msg_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
v_a_83_ = lean_ctor_get(v___x_82_, 0);
v_isSharedCheck_91_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_91_ == 0)
{
v___x_85_ = v___x_82_;
v_isShared_86_ = v_isSharedCheck_91_;
goto v_resetjp_84_;
}
else
{
lean_inc(v_a_83_);
lean_dec(v___x_82_);
v___x_85_ = lean_box(0);
v_isShared_86_ = v_isSharedCheck_91_;
goto v_resetjp_84_;
}
v_resetjp_84_:
{
lean_object* v___x_87_; lean_object* v___x_89_; 
lean_inc(v_ref_81_);
v___x_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_87_, 0, v_ref_81_);
lean_ctor_set(v___x_87_, 1, v_a_83_);
if (v_isShared_86_ == 0)
{
lean_ctor_set_tag(v___x_85_, 1);
lean_ctor_set(v___x_85_, 0, v___x_87_);
v___x_89_ = v___x_85_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v___x_87_);
v___x_89_ = v_reuseFailAlloc_90_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
return v___x_89_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg(v_msg_92_, v___y_93_, v___y_94_, v___y_95_, v___y_96_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
return v_res_98_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__1));
v___x_102_ = l_Lean_stringToMessageData(v___x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0(lean_object* v_ctor_103_, lean_object* v_args_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
lean_object* v___x_172_; uint8_t v___x_173_; 
v___x_172_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__0));
v___x_173_ = lean_string_dec_eq(v_ctor_103_, v___x_172_);
if (v___x_173_ == 0)
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__0___redArg();
return v___x_174_;
}
else
{
lean_object* v___x_175_; lean_object* v___x_176_; uint8_t v___x_177_; 
v___x_175_ = lean_array_get_size(v_args_104_);
v___x_176_ = lean_unsigned_to_nat(4u);
v___x_177_ = lean_nat_dec_eq(v___x_175_, v___x_176_);
if (v___x_177_ == 0)
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v_a_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_187_; 
v___x_178_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___closed__2);
v___x_179_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg(v___x_178_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
v_a_180_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_187_ == 0)
{
v___x_182_ = v___x_179_;
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_a_180_);
lean_dec(v___x_179_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_185_; 
if (v_isShared_183_ == 0)
{
v___x_185_ = v___x_182_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_a_180_);
v___x_185_ = v_reuseFailAlloc_186_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
return v___x_185_;
}
}
}
else
{
goto v___jp_110_;
}
}
v___jp_110_:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_111_ = l_Lean_instInhabitedExpr;
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = lean_array_get_borrowed(v___x_111_, v_args_104_, v___x_112_);
lean_inc(v___x_113_);
v___x_114_ = l_Lean_Elab_ConfigEval_instEvalExprApplyNewGoals_evalExpr(v___x_113_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_a_115_);
lean_dec_ref_known(v___x_114_, 1);
v___x_116_ = lean_unsigned_to_nat(1u);
v___x_117_ = lean_array_get_borrowed(v___x_111_, v_args_104_, v___x_116_);
lean_inc(v___x_117_);
v___x_118_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_117_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
lean_inc(v_a_119_);
lean_dec_ref_known(v___x_118_, 1);
v___x_120_ = lean_unsigned_to_nat(2u);
v___x_121_ = lean_array_get_borrowed(v___x_111_, v_args_104_, v___x_120_);
lean_inc(v___x_121_);
v___x_122_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_121_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_object* v_a_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v_a_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc(v_a_123_);
lean_dec_ref_known(v___x_122_, 1);
v___x_124_ = lean_unsigned_to_nat(3u);
v___x_125_ = lean_array_get_borrowed(v___x_111_, v_args_104_, v___x_124_);
lean_inc(v___x_125_);
v___x_126_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_125_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_139_; 
v_a_127_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_139_ == 0)
{
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_139_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_139_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_131_; uint8_t v___x_132_; uint8_t v___x_133_; uint8_t v___x_134_; uint8_t v___x_135_; lean_object* v___x_137_; 
v___x_131_ = lean_alloc_ctor(0, 0, 4);
v___x_132_ = lean_unbox(v_a_115_);
lean_dec(v_a_115_);
lean_ctor_set_uint8(v___x_131_, 0, v___x_132_);
v___x_133_ = lean_unbox(v_a_119_);
lean_dec(v_a_119_);
lean_ctor_set_uint8(v___x_131_, 1, v___x_133_);
v___x_134_ = lean_unbox(v_a_123_);
lean_dec(v_a_123_);
lean_ctor_set_uint8(v___x_131_, 2, v___x_134_);
v___x_135_ = lean_unbox(v_a_127_);
lean_dec(v_a_127_);
lean_ctor_set_uint8(v___x_131_, 3, v___x_135_);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v___x_131_);
v___x_137_ = v___x_129_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_131_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
else
{
lean_object* v_a_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_147_; 
lean_dec(v_a_123_);
lean_dec(v_a_119_);
lean_dec(v_a_115_);
v_a_140_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_147_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_147_ == 0)
{
v___x_142_ = v___x_126_;
v_isShared_143_ = v_isSharedCheck_147_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_a_140_);
lean_dec(v___x_126_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_147_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
lean_object* v___x_145_; 
if (v_isShared_143_ == 0)
{
v___x_145_ = v___x_142_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_a_140_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
}
}
else
{
lean_object* v_a_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_155_; 
lean_dec(v_a_119_);
lean_dec(v_a_115_);
v_a_148_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_155_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_155_ == 0)
{
v___x_150_ = v___x_122_;
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_a_148_);
lean_dec(v___x_122_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_148_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
else
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
lean_dec(v_a_115_);
v_a_156_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_118_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_118_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
else
{
lean_object* v_a_164_; lean_object* v___x_166_; uint8_t v_isShared_167_; uint8_t v_isSharedCheck_171_; 
v_a_164_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_171_ == 0)
{
v___x_166_ = v___x_114_;
v_isShared_167_ = v_isSharedCheck_171_;
goto v_resetjp_165_;
}
else
{
lean_inc(v_a_164_);
lean_dec(v___x_114_);
v___x_166_ = lean_box(0);
v_isShared_167_ = v_isSharedCheck_171_;
goto v_resetjp_165_;
}
v_resetjp_165_:
{
lean_object* v___x_169_; 
if (v_isShared_167_ == 0)
{
v___x_169_ = v___x_166_;
goto v_reusejp_168_;
}
else
{
lean_object* v_reuseFailAlloc_170_; 
v_reuseFailAlloc_170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_170_, 0, v_a_164_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0___boxed(lean_object* v_ctor_188_, lean_object* v_args_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___lam__0(v_ctor_188_, v_args_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec_ref(v_args_189_);
lean_dec_ref(v_ctor_188_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr(lean_object* v_a_204_, lean_object* v_a_205_, lean_object* v_a_206_, lean_object* v_a_207_, lean_object* v_a_208_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___f_210_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__0));
v___x_211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4));
v___x_212_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_211_, v___f_210_, v_a_204_, v_a_205_, v_a_206_, v_a_207_, v_a_208_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___boxed(lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr(v_a_213_, v_a_214_, v_a_215_, v_a_216_, v_a_217_);
lean_dec(v_a_217_);
lean_dec_ref(v_a_216_);
lean_dec(v_a_215_);
lean_dec_ref(v_a_214_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1(lean_object* v_00_u03b1_220_, lean_object* v_msg_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___redArg(v_msg_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_228_, lean_object* v_msg_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1(v_00_u03b1_228_, v_msg_229_, v___y_230_, v___y_231_, v___y_232_, v___y_233_);
lean_dec(v___y_233_);
lean_dec_ref(v___y_232_);
lean_dec(v___y_231_);
lean_dec_ref(v___y_230_);
return v_res_235_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_237_ = lean_box(0);
v___x_238_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4));
v___x_239_ = l_Lean_Expr_const___override(v___x_238_, v___x_237_);
return v___x_239_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1);
v___x_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
return v___x_241_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3(void){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_242_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2);
v___x_243_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__0));
v___x_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v___x_242_);
return v___x_244_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib(void){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__3);
return v___x_245_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(lean_object* v_opts_246_, lean_object* v_opt_247_){
_start:
{
lean_object* v_name_248_; lean_object* v_defValue_249_; lean_object* v_map_250_; lean_object* v___x_251_; 
v_name_248_ = lean_ctor_get(v_opt_247_, 0);
v_defValue_249_ = lean_ctor_get(v_opt_247_, 1);
v_map_250_ = lean_ctor_get(v_opts_246_, 0);
v___x_251_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_250_, v_name_248_);
if (lean_obj_tag(v___x_251_) == 0)
{
uint8_t v___x_252_; 
v___x_252_ = lean_unbox(v_defValue_249_);
return v___x_252_;
}
else
{
lean_object* v_val_253_; 
v_val_253_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_val_253_);
lean_dec_ref_known(v___x_251_, 1);
if (lean_obj_tag(v_val_253_) == 1)
{
uint8_t v_v_254_; 
v_v_254_ = lean_ctor_get_uint8(v_val_253_, 0);
lean_dec_ref_known(v_val_253_, 0);
return v_v_254_;
}
else
{
uint8_t v___x_255_; 
lean_dec(v_val_253_);
v___x_255_ = lean_unbox(v_defValue_249_);
return v___x_255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6___boxed(lean_object* v_opts_256_, lean_object* v_opt_257_){
_start:
{
uint8_t v_res_258_; lean_object* v_r_259_; 
v_res_258_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_opts_256_, v_opt_257_);
lean_dec_ref(v_opt_257_);
lean_dec_ref(v_opts_256_);
v_r_259_ = lean_box(v_res_258_);
return v_r_259_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = lean_box(1);
v___x_261_ = l_Lean_MessageData_ofFormat(v___x_260_);
return v___x_261_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2));
v___x_266_ = l_Lean_MessageData_ofFormat(v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(lean_object* v_x_267_, lean_object* v_x_268_){
_start:
{
if (lean_obj_tag(v_x_268_) == 0)
{
return v_x_267_;
}
else
{
lean_object* v_head_269_; lean_object* v_tail_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_292_; 
v_head_269_ = lean_ctor_get(v_x_268_, 0);
v_tail_270_ = lean_ctor_get(v_x_268_, 1);
v_isSharedCheck_292_ = !lean_is_exclusive(v_x_268_);
if (v_isSharedCheck_292_ == 0)
{
v___x_272_ = v_x_268_;
v_isShared_273_ = v_isSharedCheck_292_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_tail_270_);
lean_inc(v_head_269_);
lean_dec(v_x_268_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_292_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v_before_274_; lean_object* v___x_276_; uint8_t v_isShared_277_; uint8_t v_isSharedCheck_290_; 
v_before_274_ = lean_ctor_get(v_head_269_, 0);
v_isSharedCheck_290_ = !lean_is_exclusive(v_head_269_);
if (v_isSharedCheck_290_ == 0)
{
lean_object* v_unused_291_; 
v_unused_291_ = lean_ctor_get(v_head_269_, 1);
lean_dec(v_unused_291_);
v___x_276_ = v_head_269_;
v_isShared_277_ = v_isSharedCheck_290_;
goto v_resetjp_275_;
}
else
{
lean_inc(v_before_274_);
lean_dec(v_head_269_);
v___x_276_ = lean_box(0);
v_isShared_277_ = v_isSharedCheck_290_;
goto v_resetjp_275_;
}
v_resetjp_275_:
{
lean_object* v___x_278_; lean_object* v___x_280_; 
v___x_278_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0);
if (v_isShared_277_ == 0)
{
lean_ctor_set_tag(v___x_276_, 7);
lean_ctor_set(v___x_276_, 1, v___x_278_);
lean_ctor_set(v___x_276_, 0, v_x_267_);
v___x_280_ = v___x_276_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v_x_267_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v___x_278_);
v___x_280_ = v_reuseFailAlloc_289_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
lean_object* v___x_281_; lean_object* v___x_283_; 
v___x_281_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3);
if (v_isShared_273_ == 0)
{
lean_ctor_set_tag(v___x_272_, 7);
lean_ctor_set(v___x_272_, 1, v___x_281_);
lean_ctor_set(v___x_272_, 0, v___x_280_);
v___x_283_ = v___x_272_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v___x_280_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v___x_281_);
v___x_283_ = v_reuseFailAlloc_288_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_284_ = l_Lean_MessageData_ofSyntax(v_before_274_);
v___x_285_ = l_Lean_indentD(v___x_284_);
v___x_286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_283_);
lean_ctor_set(v___x_286_, 1, v___x_285_);
v_x_267_ = v___x_286_;
v_x_268_ = v_tail_270_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1));
v___x_297_ = l_Lean_MessageData_ofFormat(v___x_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(lean_object* v_msgData_298_, lean_object* v_macroStack_299_, lean_object* v___y_300_){
_start:
{
lean_object* v_options_302_; lean_object* v___x_303_; uint8_t v___x_304_; 
v_options_302_ = lean_ctor_get(v___y_300_, 2);
v___x_303_ = l_Lean_Elab_pp_macroStack;
v___x_304_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_options_302_, v___x_303_);
if (v___x_304_ == 0)
{
lean_object* v___x_305_; 
lean_dec(v_macroStack_299_);
v___x_305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_305_, 0, v_msgData_298_);
return v___x_305_;
}
else
{
if (lean_obj_tag(v_macroStack_299_) == 0)
{
lean_object* v___x_306_; 
v___x_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_306_, 0, v_msgData_298_);
return v___x_306_;
}
else
{
lean_object* v_head_307_; lean_object* v_after_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_323_; 
v_head_307_ = lean_ctor_get(v_macroStack_299_, 0);
lean_inc(v_head_307_);
v_after_308_ = lean_ctor_get(v_head_307_, 1);
v_isSharedCheck_323_ = !lean_is_exclusive(v_head_307_);
if (v_isSharedCheck_323_ == 0)
{
lean_object* v_unused_324_; 
v_unused_324_ = lean_ctor_get(v_head_307_, 0);
lean_dec(v_unused_324_);
v___x_310_ = v_head_307_;
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_after_308_);
lean_dec(v_head_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v___x_314_; 
v___x_312_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0);
if (v_isShared_311_ == 0)
{
lean_ctor_set_tag(v___x_310_, 7);
lean_ctor_set(v___x_310_, 1, v___x_312_);
lean_ctor_set(v___x_310_, 0, v_msgData_298_);
v___x_314_ = v___x_310_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_msgData_298_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v___x_312_);
v___x_314_ = v_reuseFailAlloc_322_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v_msgData_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_315_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2);
v___x_316_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_314_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = l_Lean_MessageData_ofSyntax(v_after_308_);
v___x_318_ = l_Lean_indentD(v___x_317_);
v_msgData_319_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_319_, 0, v___x_316_);
lean_ctor_set(v_msgData_319_, 1, v___x_318_);
v___x_320_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(v_msgData_319_, v_macroStack_299_);
v___x_321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_321_, 0, v___x_320_);
return v___x_321_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_325_, lean_object* v_macroStack_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_msgData_325_, v_macroStack_326_, v___y_327_);
lean_dec_ref(v___y_327_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(lean_object* v_msg_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v_ref_338_; lean_object* v___x_339_; lean_object* v_a_340_; lean_object* v_macroStack_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v_a_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_352_; 
v_ref_338_ = lean_ctor_get(v___y_335_, 5);
v___x_339_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr_spec__1_spec__1(v_msg_330_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
v_a_340_ = lean_ctor_get(v___x_339_, 0);
lean_inc(v_a_340_);
lean_dec_ref(v___x_339_);
v_macroStack_341_ = lean_ctor_get(v___y_331_, 1);
v___x_342_ = l_Lean_Elab_getBetterRef(v_ref_338_, v_macroStack_341_);
lean_inc(v_macroStack_341_);
v___x_343_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_a_340_, v_macroStack_341_, v___y_335_);
v_a_344_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_352_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_352_ == 0)
{
v___x_346_ = v___x_343_;
v_isShared_347_ = v_isSharedCheck_352_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_a_344_);
lean_dec(v___x_343_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_352_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_350_; 
v___x_348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_342_);
lean_ctor_set(v___x_348_, 1, v_a_344_);
if (v_isShared_347_ == 0)
{
lean_ctor_set_tag(v___x_346_, 1);
lean_ctor_set(v___x_346_, 0, v___x_348_);
v___x_350_ = v___x_346_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v___x_348_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg___boxed(lean_object* v_msg_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v_msg_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
lean_dec(v___y_357_);
lean_dec_ref(v___y_356_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(lean_object* v_e_362_, lean_object* v___y_363_){
_start:
{
uint8_t v___x_365_; 
v___x_365_ = l_Lean_Expr_hasMVar(v_e_362_);
if (v___x_365_ == 0)
{
lean_object* v___x_366_; 
v___x_366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_366_, 0, v_e_362_);
return v___x_366_;
}
else
{
lean_object* v___x_367_; lean_object* v_mctx_368_; lean_object* v___x_369_; lean_object* v_fst_370_; lean_object* v_snd_371_; lean_object* v___x_372_; lean_object* v_cache_373_; lean_object* v_zetaDeltaFVarIds_374_; lean_object* v_postponed_375_; lean_object* v_diag_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_385_; 
v___x_367_ = lean_st_ref_get(v___y_363_);
v_mctx_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc_ref(v_mctx_368_);
lean_dec(v___x_367_);
v___x_369_ = l_Lean_instantiateMVarsCore(v_mctx_368_, v_e_362_);
v_fst_370_ = lean_ctor_get(v___x_369_, 0);
lean_inc(v_fst_370_);
v_snd_371_ = lean_ctor_get(v___x_369_, 1);
lean_inc(v_snd_371_);
lean_dec_ref(v___x_369_);
v___x_372_ = lean_st_ref_take(v___y_363_);
v_cache_373_ = lean_ctor_get(v___x_372_, 1);
v_zetaDeltaFVarIds_374_ = lean_ctor_get(v___x_372_, 2);
v_postponed_375_ = lean_ctor_get(v___x_372_, 3);
v_diag_376_ = lean_ctor_get(v___x_372_, 4);
v_isSharedCheck_385_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_385_ == 0)
{
lean_object* v_unused_386_; 
v_unused_386_ = lean_ctor_get(v___x_372_, 0);
lean_dec(v_unused_386_);
v___x_378_ = v___x_372_;
v_isShared_379_ = v_isSharedCheck_385_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_diag_376_);
lean_inc(v_postponed_375_);
lean_inc(v_zetaDeltaFVarIds_374_);
lean_inc(v_cache_373_);
lean_dec(v___x_372_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_385_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
lean_ctor_set(v___x_378_, 0, v_snd_371_);
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v_snd_371_);
lean_ctor_set(v_reuseFailAlloc_384_, 1, v_cache_373_);
lean_ctor_set(v_reuseFailAlloc_384_, 2, v_zetaDeltaFVarIds_374_);
lean_ctor_set(v_reuseFailAlloc_384_, 3, v_postponed_375_);
lean_ctor_set(v_reuseFailAlloc_384_, 4, v_diag_376_);
v___x_381_ = v_reuseFailAlloc_384_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_382_; lean_object* v___x_383_; 
v___x_382_ = lean_st_ref_set(v___y_363_, v___x_381_);
v___x_383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_383_, 0, v_fst_370_);
return v___x_383_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg___boxed(lean_object* v_e_387_, lean_object* v___y_388_, lean_object* v___y_389_){
_start:
{
lean_object* v_res_390_; 
v_res_390_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(v_e_387_, v___y_388_);
lean_dec(v___y_388_);
return v_res_390_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v___x_391_ = lean_box(0);
v___x_392_ = l_Lean_Elab_abortTermExceptionId;
v___x_393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_392_);
lean_ctor_set(v___x_393_, 1, v___x_391_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg(){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0);
v___x_396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg___boxed(lean_object* v___y_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
return v_res_398_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_404_ = lean_box(0);
v___x_405_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__1));
v___x_406_ = l_Lean_Expr_const___override(v___x_405_, v___x_404_);
return v___x_406_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_407_; lean_object* v_ty_x3f_408_; 
v___x_407_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2);
v_ty_x3f_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_ty_x3f_408_, 0, v___x_407_);
return v_ty_x3f_408_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__4));
v___x_411_ = l_Lean_stringToMessageData(v___x_410_);
return v___x_411_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__2);
v___x_413_ = l_Lean_MessageData_ofExpr(v___x_412_);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_414_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__6);
v___x_415_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_416_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_415_);
lean_ctor_set(v___x_416_, 1, v___x_414_);
return v___x_416_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9(void){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_418_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__8));
v___x_419_ = l_Lean_stringToMessageData(v___x_418_);
return v___x_419_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10(void){
_start:
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_420_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_421_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__7);
v___x_422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v___x_420_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12(void){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_424_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__11));
v___x_425_ = l_Lean_stringToMessageData(v___x_424_);
return v___x_425_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14(void){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_427_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__13));
v___x_428_ = l_Lean_stringToMessageData(v___x_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0(lean_object* v_stx_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_){
_start:
{
lean_object* v_ty_x3f_437_; uint8_t v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v_fileName_443_; lean_object* v_fileMap_444_; lean_object* v_options_445_; lean_object* v_currRecDepth_446_; lean_object* v_maxRecDepth_447_; lean_object* v_ref_448_; lean_object* v_currNamespace_449_; lean_object* v_openDecls_450_; lean_object* v_initHeartbeats_451_; lean_object* v_maxHeartbeats_452_; lean_object* v_quotContext_453_; lean_object* v_currMacroScope_454_; uint8_t v_diag_455_; lean_object* v_cancelTk_x3f_456_; uint8_t v_suppressElabErrors_457_; lean_object* v_inheritedTraceOptions_458_; uint8_t v___x_459_; lean_object* v_ref_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v_ty_x3f_437_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__3);
v___x_438_ = 1;
v___x_439_ = lean_box(0);
v___x_440_ = lean_box(v___x_438_);
v___x_441_ = lean_box(v___x_438_);
lean_inc(v_stx_429_);
v___x_442_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_442_, 0, v_stx_429_);
lean_closure_set(v___x_442_, 1, v_ty_x3f_437_);
lean_closure_set(v___x_442_, 2, v___x_440_);
lean_closure_set(v___x_442_, 3, v___x_441_);
lean_closure_set(v___x_442_, 4, v___x_439_);
v_fileName_443_ = lean_ctor_get(v_a_434_, 0);
v_fileMap_444_ = lean_ctor_get(v_a_434_, 1);
v_options_445_ = lean_ctor_get(v_a_434_, 2);
v_currRecDepth_446_ = lean_ctor_get(v_a_434_, 3);
v_maxRecDepth_447_ = lean_ctor_get(v_a_434_, 4);
v_ref_448_ = lean_ctor_get(v_a_434_, 5);
v_currNamespace_449_ = lean_ctor_get(v_a_434_, 6);
v_openDecls_450_ = lean_ctor_get(v_a_434_, 7);
v_initHeartbeats_451_ = lean_ctor_get(v_a_434_, 8);
v_maxHeartbeats_452_ = lean_ctor_get(v_a_434_, 9);
v_quotContext_453_ = lean_ctor_get(v_a_434_, 10);
v_currMacroScope_454_ = lean_ctor_get(v_a_434_, 11);
v_diag_455_ = lean_ctor_get_uint8(v_a_434_, sizeof(void*)*14);
v_cancelTk_x3f_456_ = lean_ctor_get(v_a_434_, 12);
v_suppressElabErrors_457_ = lean_ctor_get_uint8(v_a_434_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_458_ = lean_ctor_get(v_a_434_, 13);
v___x_459_ = 1;
v_ref_460_ = l_Lean_replaceRef(v_stx_429_, v_ref_448_);
lean_dec(v_stx_429_);
lean_inc_ref(v_inheritedTraceOptions_458_);
lean_inc(v_cancelTk_x3f_456_);
lean_inc(v_currMacroScope_454_);
lean_inc(v_quotContext_453_);
lean_inc(v_maxHeartbeats_452_);
lean_inc(v_initHeartbeats_451_);
lean_inc(v_openDecls_450_);
lean_inc(v_currNamespace_449_);
lean_inc(v_maxRecDepth_447_);
lean_inc(v_currRecDepth_446_);
lean_inc_ref(v_options_445_);
lean_inc_ref(v_fileMap_444_);
lean_inc_ref(v_fileName_443_);
v___x_461_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_461_, 0, v_fileName_443_);
lean_ctor_set(v___x_461_, 1, v_fileMap_444_);
lean_ctor_set(v___x_461_, 2, v_options_445_);
lean_ctor_set(v___x_461_, 3, v_currRecDepth_446_);
lean_ctor_set(v___x_461_, 4, v_maxRecDepth_447_);
lean_ctor_set(v___x_461_, 5, v_ref_460_);
lean_ctor_set(v___x_461_, 6, v_currNamespace_449_);
lean_ctor_set(v___x_461_, 7, v_openDecls_450_);
lean_ctor_set(v___x_461_, 8, v_initHeartbeats_451_);
lean_ctor_set(v___x_461_, 9, v_maxHeartbeats_452_);
lean_ctor_set(v___x_461_, 10, v_quotContext_453_);
lean_ctor_set(v___x_461_, 11, v_currMacroScope_454_);
lean_ctor_set(v___x_461_, 12, v_cancelTk_x3f_456_);
lean_ctor_set(v___x_461_, 13, v_inheritedTraceOptions_458_);
lean_ctor_set_uint8(v___x_461_, sizeof(void*)*14, v_diag_455_);
lean_ctor_set_uint8(v___x_461_, sizeof(void*)*14 + 1, v_suppressElabErrors_457_);
v___x_462_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_442_, v___x_459_, v_a_430_, v_a_431_, v_a_432_, v_a_433_, v___x_461_, v_a_435_);
if (lean_obj_tag(v___x_462_) == 0)
{
lean_object* v_a_463_; lean_object* v___x_464_; lean_object* v_a_465_; lean_object* v___y_467_; lean_object* v___y_468_; lean_object* v___y_469_; lean_object* v___y_470_; lean_object* v___y_471_; lean_object* v___y_472_; lean_object* v___y_473_; lean_object* v___y_474_; lean_object* v___y_475_; uint8_t v___y_476_; lean_object* v___y_493_; lean_object* v___y_494_; lean_object* v___y_495_; lean_object* v___y_496_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___y_508_; lean_object* v___y_509_; lean_object* v___y_510_; lean_object* v___y_542_; lean_object* v___y_543_; lean_object* v___y_544_; lean_object* v___y_545_; lean_object* v___y_546_; lean_object* v___y_547_; uint8_t v___x_560_; 
v_a_463_ = lean_ctor_get(v___x_462_, 0);
lean_inc(v_a_463_);
lean_dec_ref_known(v___x_462_, 1);
v___x_464_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(v_a_463_, v_a_433_);
v_a_465_ = lean_ctor_get(v___x_464_, 0);
lean_inc(v_a_465_);
lean_dec_ref(v___x_464_);
v___x_560_ = l_Lean_Expr_hasSorry(v_a_465_);
if (v___x_560_ == 0)
{
v___y_505_ = v_a_430_;
v___y_506_ = v_a_431_;
v___y_507_ = v_a_432_;
v___y_508_ = v_a_433_;
v___y_509_ = v___x_461_;
v___y_510_ = v_a_435_;
goto v___jp_504_;
}
else
{
uint8_t v___x_561_; 
v___x_561_ = l_Lean_Expr_hasSyntheticSorry(v_a_465_);
if (v___x_561_ == 0)
{
v___y_542_ = v_a_430_;
v___y_543_ = v_a_431_;
v___y_544_ = v_a_432_;
v___y_545_ = v_a_433_;
v___y_546_ = v___x_461_;
v___y_547_ = v_a_435_;
goto v___jp_541_;
}
else
{
lean_object* v___x_562_; lean_object* v_a_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_570_; 
lean_dec(v_a_465_);
lean_dec_ref_known(v___x_461_, 14);
v___x_562_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_563_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_570_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_570_ == 0)
{
v___x_565_ = v___x_562_;
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_a_563_);
lean_dec(v___x_562_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_568_; 
if (v_isShared_566_ == 0)
{
v___x_568_ = v___x_565_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_a_563_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
v___jp_466_:
{
if (v___y_476_ == 0)
{
if (lean_obj_tag(v___y_470_) == 0)
{
lean_dec_ref_known(v___y_470_, 2);
lean_dec_ref(v___y_467_);
lean_dec(v_a_465_);
return v___y_469_;
}
else
{
lean_object* v_id_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_490_; 
v_id_477_ = lean_ctor_get(v___y_470_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___y_470_);
if (v_isSharedCheck_490_ == 0)
{
lean_object* v_unused_491_; 
v_unused_491_ = lean_ctor_get(v___y_470_, 1);
lean_dec(v_unused_491_);
v___x_479_ = v___y_470_;
v_isShared_480_ = v_isSharedCheck_490_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_id_477_);
lean_dec(v___y_470_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_490_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
uint8_t v___x_481_; 
v___x_481_ = l_Lean_instBEqInternalExceptionId_beq(v___y_468_, v_id_477_);
lean_dec(v_id_477_);
if (v___x_481_ == 0)
{
lean_del_object(v___x_479_);
lean_dec_ref(v___y_467_);
lean_dec(v_a_465_);
return v___y_469_;
}
else
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_486_; 
lean_dec_ref(v___y_469_);
v___x_482_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__10);
v___x_483_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12);
v___x_484_ = l_Lean_indentExpr(v_a_465_);
if (v_isShared_480_ == 0)
{
lean_ctor_set_tag(v___x_479_, 7);
lean_ctor_set(v___x_479_, 1, v___x_484_);
lean_ctor_set(v___x_479_, 0, v___x_483_);
v___x_486_ = v___x_479_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_489_, 1, v___x_484_);
v___x_486_ = v_reuseFailAlloc_489_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_487_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v___x_482_);
v___x_488_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_487_, v___y_472_, v___y_471_, v___y_474_, v___y_473_, v___y_467_, v___y_475_);
lean_dec_ref(v___y_467_);
return v___x_488_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_470_);
lean_dec_ref(v___y_467_);
lean_dec(v_a_465_);
return v___y_469_;
}
}
v___jp_492_:
{
lean_object* v___x_499_; 
lean_inc(v_a_465_);
v___x_499_ = l_Lean_Elab_ConfigEval_instEvalExprApplyNewGoals_evalExpr(v_a_465_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_dec_ref(v___y_497_);
lean_dec(v_a_465_);
return v___x_499_;
}
else
{
lean_object* v_a_500_; lean_object* v___x_501_; uint8_t v___x_502_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_a_500_);
v___x_501_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_502_ = l_Lean_Exception_isInterrupt(v_a_500_);
if (v___x_502_ == 0)
{
uint8_t v___x_503_; 
lean_inc(v_a_500_);
v___x_503_ = l_Lean_Exception_isRuntime(v_a_500_);
v___y_467_ = v___y_497_;
v___y_468_ = v___x_501_;
v___y_469_ = v___x_499_;
v___y_470_ = v_a_500_;
v___y_471_ = v___y_494_;
v___y_472_ = v___y_493_;
v___y_473_ = v___y_496_;
v___y_474_ = v___y_495_;
v___y_475_ = v___y_498_;
v___y_476_ = v___x_503_;
goto v___jp_466_;
}
else
{
v___y_467_ = v___y_497_;
v___y_468_ = v___x_501_;
v___y_469_ = v___x_499_;
v___y_470_ = v_a_500_;
v___y_471_ = v___y_494_;
v___y_472_ = v___y_493_;
v___y_473_ = v___y_496_;
v___y_474_ = v___y_495_;
v___y_475_ = v___y_498_;
v___y_476_ = v___x_502_;
goto v___jp_466_;
}
}
}
v___jp_504_:
{
lean_object* v___x_511_; 
lean_inc(v_a_465_);
v___x_511_ = l_Lean_Meta_getMVars(v_a_465_, v___y_507_, v___y_508_, v___y_509_, v___y_510_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; lean_object* v___x_513_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
lean_dec_ref_known(v___x_511_, 1);
v___x_513_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_512_, v___x_439_, v___y_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_);
lean_dec(v_a_512_);
if (lean_obj_tag(v___x_513_) == 0)
{
lean_object* v_a_514_; uint8_t v___x_515_; 
v_a_514_ = lean_ctor_get(v___x_513_, 0);
lean_inc(v_a_514_);
lean_dec_ref_known(v___x_513_, 1);
v___x_515_ = lean_unbox(v_a_514_);
lean_dec(v_a_514_);
if (v___x_515_ == 0)
{
v___y_493_ = v___y_505_;
v___y_494_ = v___y_506_;
v___y_495_ = v___y_507_;
v___y_496_ = v___y_508_;
v___y_497_ = v___y_509_;
v___y_498_ = v___y_510_;
goto v___jp_492_;
}
else
{
lean_object* v___x_516_; lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_524_; 
lean_dec_ref(v___y_509_);
lean_dec(v_a_465_);
v___x_516_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_517_ = lean_ctor_get(v___x_516_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_516_);
if (v_isSharedCheck_524_ == 0)
{
v___x_519_ = v___x_516_;
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_516_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_522_; 
if (v_isShared_520_ == 0)
{
v___x_522_ = v___x_519_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v_a_517_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
else
{
lean_object* v_a_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_532_; 
lean_dec_ref(v___y_509_);
lean_dec(v_a_465_);
v_a_525_ = lean_ctor_get(v___x_513_, 0);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_532_ == 0)
{
v___x_527_ = v___x_513_;
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_a_525_);
lean_dec(v___x_513_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___x_530_; 
if (v_isShared_528_ == 0)
{
v___x_530_ = v___x_527_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_a_525_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
return v___x_530_;
}
}
}
}
else
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_540_; 
lean_dec_ref(v___y_509_);
lean_dec(v_a_465_);
v_a_533_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_540_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_540_ == 0)
{
v___x_535_ = v___x_511_;
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_511_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v___x_538_; 
if (v_isShared_536_ == 0)
{
v___x_538_ = v___x_535_;
goto v_reusejp_537_;
}
else
{
lean_object* v_reuseFailAlloc_539_; 
v_reuseFailAlloc_539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_539_, 0, v_a_533_);
v___x_538_ = v_reuseFailAlloc_539_;
goto v_reusejp_537_;
}
v_reusejp_537_:
{
return v___x_538_;
}
}
}
}
v___jp_541_:
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v_a_552_; lean_object* v___x_554_; uint8_t v_isShared_555_; uint8_t v_isSharedCheck_559_; 
v___x_548_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14);
v___x_549_ = l_Lean_indentExpr(v_a_465_);
v___x_550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_550_, 0, v___x_548_);
lean_ctor_set(v___x_550_, 1, v___x_549_);
v___x_551_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_550_, v___y_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_);
lean_dec_ref(v___y_546_);
v_a_552_ = lean_ctor_get(v___x_551_, 0);
v_isSharedCheck_559_ = !lean_is_exclusive(v___x_551_);
if (v_isSharedCheck_559_ == 0)
{
v___x_554_ = v___x_551_;
v_isShared_555_ = v_isSharedCheck_559_;
goto v_resetjp_553_;
}
else
{
lean_inc(v_a_552_);
lean_dec(v___x_551_);
v___x_554_ = lean_box(0);
v_isShared_555_ = v_isSharedCheck_559_;
goto v_resetjp_553_;
}
v_resetjp_553_:
{
lean_object* v___x_557_; 
if (v_isShared_555_ == 0)
{
v___x_557_ = v___x_554_;
goto v_reusejp_556_;
}
else
{
lean_object* v_reuseFailAlloc_558_; 
v_reuseFailAlloc_558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_558_, 0, v_a_552_);
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
else
{
lean_object* v_a_571_; lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_578_; 
lean_dec_ref_known(v___x_461_, 14);
v_a_571_ = lean_ctor_get(v___x_462_, 0);
v_isSharedCheck_578_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_578_ == 0)
{
v___x_573_ = v___x_462_;
v_isShared_574_ = v_isSharedCheck_578_;
goto v_resetjp_572_;
}
else
{
lean_inc(v_a_571_);
lean_dec(v___x_462_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_578_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_576_; 
if (v_isShared_574_ == 0)
{
v___x_576_ = v___x_573_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v_a_571_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_stx_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0(v_stx_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_, v_a_584_, v_a_585_);
lean_dec(v_a_585_);
lean_dec_ref(v_a_584_);
lean_dec(v_a_583_);
lean_dec_ref(v_a_582_);
lean_dec(v_a_581_);
lean_dec_ref(v_a_580_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0(lean_object* v_stx_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_, lean_object* v_a_594_){
_start:
{
lean_object* v_fileName_596_; lean_object* v_fileMap_597_; lean_object* v_options_598_; lean_object* v_currRecDepth_599_; lean_object* v_maxRecDepth_600_; lean_object* v_ref_601_; lean_object* v_currNamespace_602_; lean_object* v_openDecls_603_; lean_object* v_initHeartbeats_604_; lean_object* v_maxHeartbeats_605_; lean_object* v_quotContext_606_; lean_object* v_currMacroScope_607_; uint8_t v_diag_608_; lean_object* v_cancelTk_x3f_609_; uint8_t v_suppressElabErrors_610_; lean_object* v_inheritedTraceOptions_611_; lean_object* v_ref_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v_fileName_596_ = lean_ctor_get(v_a_593_, 0);
v_fileMap_597_ = lean_ctor_get(v_a_593_, 1);
v_options_598_ = lean_ctor_get(v_a_593_, 2);
v_currRecDepth_599_ = lean_ctor_get(v_a_593_, 3);
v_maxRecDepth_600_ = lean_ctor_get(v_a_593_, 4);
v_ref_601_ = lean_ctor_get(v_a_593_, 5);
v_currNamespace_602_ = lean_ctor_get(v_a_593_, 6);
v_openDecls_603_ = lean_ctor_get(v_a_593_, 7);
v_initHeartbeats_604_ = lean_ctor_get(v_a_593_, 8);
v_maxHeartbeats_605_ = lean_ctor_get(v_a_593_, 9);
v_quotContext_606_ = lean_ctor_get(v_a_593_, 10);
v_currMacroScope_607_ = lean_ctor_get(v_a_593_, 11);
v_diag_608_ = lean_ctor_get_uint8(v_a_593_, sizeof(void*)*14);
v_cancelTk_x3f_609_ = lean_ctor_get(v_a_593_, 12);
v_suppressElabErrors_610_ = lean_ctor_get_uint8(v_a_593_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_611_ = lean_ctor_get(v_a_593_, 13);
v_ref_612_ = l_Lean_replaceRef(v_stx_588_, v_ref_601_);
lean_inc_ref(v_inheritedTraceOptions_611_);
lean_inc(v_cancelTk_x3f_609_);
lean_inc(v_currMacroScope_607_);
lean_inc(v_quotContext_606_);
lean_inc(v_maxHeartbeats_605_);
lean_inc(v_initHeartbeats_604_);
lean_inc(v_openDecls_603_);
lean_inc(v_currNamespace_602_);
lean_inc(v_maxRecDepth_600_);
lean_inc(v_currRecDepth_599_);
lean_inc_ref(v_options_598_);
lean_inc_ref(v_fileMap_597_);
lean_inc_ref(v_fileName_596_);
v___x_613_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_613_, 0, v_fileName_596_);
lean_ctor_set(v___x_613_, 1, v_fileMap_597_);
lean_ctor_set(v___x_613_, 2, v_options_598_);
lean_ctor_set(v___x_613_, 3, v_currRecDepth_599_);
lean_ctor_set(v___x_613_, 4, v_maxRecDepth_600_);
lean_ctor_set(v___x_613_, 5, v_ref_612_);
lean_ctor_set(v___x_613_, 6, v_currNamespace_602_);
lean_ctor_set(v___x_613_, 7, v_openDecls_603_);
lean_ctor_set(v___x_613_, 8, v_initHeartbeats_604_);
lean_ctor_set(v___x_613_, 9, v_maxHeartbeats_605_);
lean_ctor_set(v___x_613_, 10, v_quotContext_606_);
lean_ctor_set(v___x_613_, 11, v_currMacroScope_607_);
lean_ctor_set(v___x_613_, 12, v_cancelTk_x3f_609_);
lean_ctor_set(v___x_613_, 13, v_inheritedTraceOptions_611_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*14, v_diag_608_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*14 + 1, v_suppressElabErrors_610_);
lean_inc(v_stx_588_);
v___x_614_ = l_Lean_Elab_ConfigEval_instEvalTermApplyNewGoals_evalTerm(v_stx_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_, v___x_613_, v_a_594_);
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_623_; 
lean_dec_ref_known(v___x_613_, 14);
lean_dec(v_stx_588_);
v_a_615_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_623_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_623_ == 0)
{
v___x_617_ = v___x_614_;
v_isShared_618_ = v_isSharedCheck_623_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_614_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_623_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v_fst_619_; lean_object* v___x_621_; 
v_fst_619_ = lean_ctor_get(v_a_615_, 0);
lean_inc(v_fst_619_);
lean_dec(v_a_615_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v_fst_619_);
v___x_621_ = v___x_617_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v_fst_619_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
else
{
lean_object* v_a_624_; lean_object* v___x_626_; uint8_t v_isShared_627_; uint8_t v_isSharedCheck_639_; 
v_a_624_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_639_ == 0)
{
v___x_626_ = v___x_614_;
v_isShared_627_ = v_isSharedCheck_639_;
goto v_resetjp_625_;
}
else
{
lean_inc(v_a_624_);
lean_dec(v___x_614_);
v___x_626_ = lean_box(0);
v_isShared_627_ = v_isSharedCheck_639_;
goto v_resetjp_625_;
}
v_resetjp_625_:
{
lean_object* v___x_628_; lean_object* v___x_630_; 
v___x_628_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_624_);
if (v_isShared_627_ == 0)
{
v___x_630_ = v___x_626_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_624_);
v___x_630_ = v_reuseFailAlloc_638_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
uint8_t v___y_632_; uint8_t v___x_636_; 
v___x_636_ = l_Lean_Exception_isInterrupt(v_a_624_);
if (v___x_636_ == 0)
{
uint8_t v___x_637_; 
lean_inc(v_a_624_);
v___x_637_ = l_Lean_Exception_isRuntime(v_a_624_);
v___y_632_ = v___x_637_;
goto v___jp_631_;
}
else
{
v___y_632_ = v___x_636_;
goto v___jp_631_;
}
v___jp_631_:
{
if (v___y_632_ == 0)
{
if (lean_obj_tag(v_a_624_) == 0)
{
lean_dec_ref_known(v_a_624_, 2);
lean_dec_ref_known(v___x_613_, 14);
lean_dec(v_stx_588_);
return v___x_630_;
}
else
{
lean_object* v_id_633_; uint8_t v___x_634_; 
v_id_633_ = lean_ctor_get(v_a_624_, 0);
lean_inc(v_id_633_);
lean_dec_ref_known(v_a_624_, 2);
v___x_634_ = l_Lean_instBEqInternalExceptionId_beq(v___x_628_, v_id_633_);
lean_dec(v_id_633_);
if (v___x_634_ == 0)
{
lean_dec_ref_known(v___x_613_, 14);
lean_dec(v_stx_588_);
return v___x_630_;
}
else
{
lean_object* v___x_635_; 
lean_dec_ref(v___x_630_);
v___x_635_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0(v_stx_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_, v___x_613_, v_a_594_);
lean_dec_ref_known(v___x_613_, 14);
return v___x_635_;
}
}
}
else
{
lean_dec(v_a_624_);
lean_dec_ref_known(v___x_613_, 14);
lean_dec(v_stx_588_);
return v___x_630_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_640_, lean_object* v_a_641_, lean_object* v_a_642_, lean_object* v_a_643_, lean_object* v_a_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0(v_stx_640_, v_a_641_, v_a_642_, v_a_643_, v_a_644_, v_a_645_, v_a_646_);
lean_dec(v_a_646_);
lean_dec_ref(v_a_645_);
lean_dec(v_a_644_);
lean_dec_ref(v_a_643_);
lean_dec(v_a_642_);
lean_dec_ref(v_a_641_);
return v_res_648_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_649_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__1);
v___x_650_ = l_Lean_MessageData_ofExpr(v___x_649_);
return v___x_650_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1(void){
_start:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_651_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__0);
v___x_652_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_653_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_653_, 0, v___x_652_);
lean_ctor_set(v___x_653_, 1, v___x_651_);
return v___x_653_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2(void){
_start:
{
lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_654_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_655_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__1);
v___x_656_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_656_, 0, v___x_655_);
lean_ctor_set(v___x_656_, 1, v___x_654_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1(lean_object* v_stx_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_){
_start:
{
lean_object* v_ty_x3f_665_; uint8_t v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v_fileName_671_; lean_object* v_fileMap_672_; lean_object* v_options_673_; lean_object* v_currRecDepth_674_; lean_object* v_maxRecDepth_675_; lean_object* v_ref_676_; lean_object* v_currNamespace_677_; lean_object* v_openDecls_678_; lean_object* v_initHeartbeats_679_; lean_object* v_maxHeartbeats_680_; lean_object* v_quotContext_681_; lean_object* v_currMacroScope_682_; uint8_t v_diag_683_; lean_object* v_cancelTk_x3f_684_; uint8_t v_suppressElabErrors_685_; lean_object* v_inheritedTraceOptions_686_; uint8_t v___x_687_; lean_object* v_ref_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v_ty_x3f_665_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2, &lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib___closed__2);
v___x_666_ = 1;
v___x_667_ = lean_box(0);
v___x_668_ = lean_box(v___x_666_);
v___x_669_ = lean_box(v___x_666_);
lean_inc(v_stx_657_);
v___x_670_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_670_, 0, v_stx_657_);
lean_closure_set(v___x_670_, 1, v_ty_x3f_665_);
lean_closure_set(v___x_670_, 2, v___x_668_);
lean_closure_set(v___x_670_, 3, v___x_669_);
lean_closure_set(v___x_670_, 4, v___x_667_);
v_fileName_671_ = lean_ctor_get(v_a_662_, 0);
v_fileMap_672_ = lean_ctor_get(v_a_662_, 1);
v_options_673_ = lean_ctor_get(v_a_662_, 2);
v_currRecDepth_674_ = lean_ctor_get(v_a_662_, 3);
v_maxRecDepth_675_ = lean_ctor_get(v_a_662_, 4);
v_ref_676_ = lean_ctor_get(v_a_662_, 5);
v_currNamespace_677_ = lean_ctor_get(v_a_662_, 6);
v_openDecls_678_ = lean_ctor_get(v_a_662_, 7);
v_initHeartbeats_679_ = lean_ctor_get(v_a_662_, 8);
v_maxHeartbeats_680_ = lean_ctor_get(v_a_662_, 9);
v_quotContext_681_ = lean_ctor_get(v_a_662_, 10);
v_currMacroScope_682_ = lean_ctor_get(v_a_662_, 11);
v_diag_683_ = lean_ctor_get_uint8(v_a_662_, sizeof(void*)*14);
v_cancelTk_x3f_684_ = lean_ctor_get(v_a_662_, 12);
v_suppressElabErrors_685_ = lean_ctor_get_uint8(v_a_662_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_686_ = lean_ctor_get(v_a_662_, 13);
v___x_687_ = 1;
v_ref_688_ = l_Lean_replaceRef(v_stx_657_, v_ref_676_);
lean_dec(v_stx_657_);
lean_inc_ref(v_inheritedTraceOptions_686_);
lean_inc(v_cancelTk_x3f_684_);
lean_inc(v_currMacroScope_682_);
lean_inc(v_quotContext_681_);
lean_inc(v_maxHeartbeats_680_);
lean_inc(v_initHeartbeats_679_);
lean_inc(v_openDecls_678_);
lean_inc(v_currNamespace_677_);
lean_inc(v_maxRecDepth_675_);
lean_inc(v_currRecDepth_674_);
lean_inc_ref(v_options_673_);
lean_inc_ref(v_fileMap_672_);
lean_inc_ref(v_fileName_671_);
v___x_689_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_689_, 0, v_fileName_671_);
lean_ctor_set(v___x_689_, 1, v_fileMap_672_);
lean_ctor_set(v___x_689_, 2, v_options_673_);
lean_ctor_set(v___x_689_, 3, v_currRecDepth_674_);
lean_ctor_set(v___x_689_, 4, v_maxRecDepth_675_);
lean_ctor_set(v___x_689_, 5, v_ref_688_);
lean_ctor_set(v___x_689_, 6, v_currNamespace_677_);
lean_ctor_set(v___x_689_, 7, v_openDecls_678_);
lean_ctor_set(v___x_689_, 8, v_initHeartbeats_679_);
lean_ctor_set(v___x_689_, 9, v_maxHeartbeats_680_);
lean_ctor_set(v___x_689_, 10, v_quotContext_681_);
lean_ctor_set(v___x_689_, 11, v_currMacroScope_682_);
lean_ctor_set(v___x_689_, 12, v_cancelTk_x3f_684_);
lean_ctor_set(v___x_689_, 13, v_inheritedTraceOptions_686_);
lean_ctor_set_uint8(v___x_689_, sizeof(void*)*14, v_diag_683_);
lean_ctor_set_uint8(v___x_689_, sizeof(void*)*14 + 1, v_suppressElabErrors_685_);
v___x_690_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_670_, v___x_687_, v_a_658_, v_a_659_, v_a_660_, v_a_661_, v___x_689_, v_a_663_);
if (lean_obj_tag(v___x_690_) == 0)
{
lean_object* v_a_691_; lean_object* v___x_692_; lean_object* v_a_693_; lean_object* v___y_695_; lean_object* v___y_696_; lean_object* v___y_697_; lean_object* v___y_698_; lean_object* v___y_699_; lean_object* v___y_700_; lean_object* v___y_701_; lean_object* v___y_702_; lean_object* v___y_703_; uint8_t v___y_704_; lean_object* v___y_721_; lean_object* v___y_722_; lean_object* v___y_723_; lean_object* v___y_724_; lean_object* v___y_725_; lean_object* v___y_726_; lean_object* v___y_733_; lean_object* v___y_734_; lean_object* v___y_735_; lean_object* v___y_736_; lean_object* v___y_737_; lean_object* v___y_738_; lean_object* v___y_770_; lean_object* v___y_771_; lean_object* v___y_772_; lean_object* v___y_773_; lean_object* v___y_774_; lean_object* v___y_775_; uint8_t v___x_788_; 
v_a_691_ = lean_ctor_get(v___x_690_, 0);
lean_inc(v_a_691_);
lean_dec_ref_known(v___x_690_, 1);
v___x_692_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(v_a_691_, v_a_661_);
v_a_693_ = lean_ctor_get(v___x_692_, 0);
lean_inc(v_a_693_);
lean_dec_ref(v___x_692_);
v___x_788_ = l_Lean_Expr_hasSorry(v_a_693_);
if (v___x_788_ == 0)
{
v___y_733_ = v_a_658_;
v___y_734_ = v_a_659_;
v___y_735_ = v_a_660_;
v___y_736_ = v_a_661_;
v___y_737_ = v___x_689_;
v___y_738_ = v_a_663_;
goto v___jp_732_;
}
else
{
uint8_t v___x_789_; 
v___x_789_ = l_Lean_Expr_hasSyntheticSorry(v_a_693_);
if (v___x_789_ == 0)
{
v___y_770_ = v_a_658_;
v___y_771_ = v_a_659_;
v___y_772_ = v_a_660_;
v___y_773_ = v_a_661_;
v___y_774_ = v___x_689_;
v___y_775_ = v_a_663_;
goto v___jp_769_;
}
else
{
lean_object* v___x_790_; lean_object* v_a_791_; lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_798_; 
lean_dec(v_a_693_);
lean_dec_ref_known(v___x_689_, 14);
v___x_790_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_791_ = lean_ctor_get(v___x_790_, 0);
v_isSharedCheck_798_ = !lean_is_exclusive(v___x_790_);
if (v_isSharedCheck_798_ == 0)
{
v___x_793_ = v___x_790_;
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
else
{
lean_inc(v_a_791_);
lean_dec(v___x_790_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_796_; 
if (v_isShared_794_ == 0)
{
v___x_796_ = v___x_793_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v_a_791_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
return v___x_796_;
}
}
}
}
v___jp_694_:
{
if (v___y_704_ == 0)
{
if (lean_obj_tag(v___y_701_) == 0)
{
lean_dec_ref_known(v___y_701_, 2);
lean_dec_ref(v___y_703_);
lean_dec(v_a_693_);
return v___y_702_;
}
else
{
lean_object* v_id_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_718_; 
v_id_705_ = lean_ctor_get(v___y_701_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___y_701_);
if (v_isSharedCheck_718_ == 0)
{
lean_object* v_unused_719_; 
v_unused_719_ = lean_ctor_get(v___y_701_, 1);
lean_dec(v_unused_719_);
v___x_707_ = v___y_701_;
v_isShared_708_ = v_isSharedCheck_718_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_id_705_);
lean_dec(v___y_701_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_718_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
uint8_t v___x_709_; 
v___x_709_ = l_Lean_instBEqInternalExceptionId_beq(v___y_699_, v_id_705_);
lean_dec(v_id_705_);
if (v___x_709_ == 0)
{
lean_del_object(v___x_707_);
lean_dec_ref(v___y_703_);
lean_dec(v_a_693_);
return v___y_702_;
}
else
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_714_; 
lean_dec_ref(v___y_702_);
v___x_710_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___closed__2);
v___x_711_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__12);
v___x_712_ = l_Lean_indentExpr(v_a_693_);
if (v_isShared_708_ == 0)
{
lean_ctor_set_tag(v___x_707_, 7);
lean_ctor_set(v___x_707_, 1, v___x_712_);
lean_ctor_set(v___x_707_, 0, v___x_711_);
v___x_714_ = v___x_707_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v___x_711_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v___x_712_);
v___x_714_ = v_reuseFailAlloc_717_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_715_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
lean_ctor_set(v___x_715_, 1, v___x_710_);
v___x_716_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_715_, v___y_700_, v___y_697_, v___y_698_, v___y_695_, v___y_703_, v___y_696_);
lean_dec_ref(v___y_703_);
return v___x_716_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_703_);
lean_dec_ref(v___y_701_);
lean_dec(v_a_693_);
return v___y_702_;
}
}
v___jp_720_:
{
lean_object* v___x_727_; 
lean_inc(v_a_693_);
v___x_727_ = lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr(v_a_693_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_dec_ref(v___y_725_);
lean_dec(v_a_693_);
return v___x_727_;
}
else
{
lean_object* v_a_728_; lean_object* v___x_729_; uint8_t v___x_730_; 
v_a_728_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_a_728_);
v___x_729_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_730_ = l_Lean_Exception_isInterrupt(v_a_728_);
if (v___x_730_ == 0)
{
uint8_t v___x_731_; 
lean_inc(v_a_728_);
v___x_731_ = l_Lean_Exception_isRuntime(v_a_728_);
v___y_695_ = v___y_724_;
v___y_696_ = v___y_726_;
v___y_697_ = v___y_722_;
v___y_698_ = v___y_723_;
v___y_699_ = v___x_729_;
v___y_700_ = v___y_721_;
v___y_701_ = v_a_728_;
v___y_702_ = v___x_727_;
v___y_703_ = v___y_725_;
v___y_704_ = v___x_731_;
goto v___jp_694_;
}
else
{
v___y_695_ = v___y_724_;
v___y_696_ = v___y_726_;
v___y_697_ = v___y_722_;
v___y_698_ = v___y_723_;
v___y_699_ = v___x_729_;
v___y_700_ = v___y_721_;
v___y_701_ = v_a_728_;
v___y_702_ = v___x_727_;
v___y_703_ = v___y_725_;
v___y_704_ = v___x_730_;
goto v___jp_694_;
}
}
}
v___jp_732_:
{
lean_object* v___x_739_; 
lean_inc(v_a_693_);
v___x_739_ = l_Lean_Meta_getMVars(v_a_693_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; lean_object* v___x_741_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_a_740_);
lean_dec_ref_known(v___x_739_, 1);
v___x_741_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_740_, v___x_667_, v___y_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
lean_dec(v_a_740_);
if (lean_obj_tag(v___x_741_) == 0)
{
lean_object* v_a_742_; uint8_t v___x_743_; 
v_a_742_ = lean_ctor_get(v___x_741_, 0);
lean_inc(v_a_742_);
lean_dec_ref_known(v___x_741_, 1);
v___x_743_ = lean_unbox(v_a_742_);
lean_dec(v_a_742_);
if (v___x_743_ == 0)
{
v___y_721_ = v___y_733_;
v___y_722_ = v___y_734_;
v___y_723_ = v___y_735_;
v___y_724_ = v___y_736_;
v___y_725_ = v___y_737_;
v___y_726_ = v___y_738_;
goto v___jp_720_;
}
else
{
lean_object* v___x_744_; lean_object* v_a_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_752_; 
lean_dec_ref(v___y_737_);
lean_dec(v_a_693_);
v___x_744_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_745_ = lean_ctor_get(v___x_744_, 0);
v_isSharedCheck_752_ = !lean_is_exclusive(v___x_744_);
if (v_isSharedCheck_752_ == 0)
{
v___x_747_ = v___x_744_;
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_a_745_);
lean_dec(v___x_744_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_750_; 
if (v_isShared_748_ == 0)
{
v___x_750_ = v___x_747_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v_a_745_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
else
{
lean_object* v_a_753_; lean_object* v___x_755_; uint8_t v_isShared_756_; uint8_t v_isSharedCheck_760_; 
lean_dec_ref(v___y_737_);
lean_dec(v_a_693_);
v_a_753_ = lean_ctor_get(v___x_741_, 0);
v_isSharedCheck_760_ = !lean_is_exclusive(v___x_741_);
if (v_isSharedCheck_760_ == 0)
{
v___x_755_ = v___x_741_;
v_isShared_756_ = v_isSharedCheck_760_;
goto v_resetjp_754_;
}
else
{
lean_inc(v_a_753_);
lean_dec(v___x_741_);
v___x_755_ = lean_box(0);
v_isShared_756_ = v_isSharedCheck_760_;
goto v_resetjp_754_;
}
v_resetjp_754_:
{
lean_object* v___x_758_; 
if (v_isShared_756_ == 0)
{
v___x_758_ = v___x_755_;
goto v_reusejp_757_;
}
else
{
lean_object* v_reuseFailAlloc_759_; 
v_reuseFailAlloc_759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_759_, 0, v_a_753_);
v___x_758_ = v_reuseFailAlloc_759_;
goto v_reusejp_757_;
}
v_reusejp_757_:
{
return v___x_758_;
}
}
}
}
else
{
lean_object* v_a_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_768_; 
lean_dec_ref(v___y_737_);
lean_dec(v_a_693_);
v_a_761_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_768_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_768_ == 0)
{
v___x_763_ = v___x_739_;
v_isShared_764_ = v_isSharedCheck_768_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_a_761_);
lean_dec(v___x_739_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_768_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_766_; 
if (v_isShared_764_ == 0)
{
v___x_766_ = v___x_763_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_a_761_);
v___x_766_ = v_reuseFailAlloc_767_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
return v___x_766_;
}
}
}
}
v___jp_769_:
{
lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v_a_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_787_; 
v___x_776_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0_spec__0___closed__14);
v___x_777_ = l_Lean_indentExpr(v_a_693_);
v___x_778_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_778_, 0, v___x_776_);
lean_ctor_set(v___x_778_, 1, v___x_777_);
v___x_779_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_778_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_);
lean_dec_ref(v___y_774_);
v_a_780_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_787_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_787_ == 0)
{
v___x_782_ = v___x_779_;
v_isShared_783_ = v_isSharedCheck_787_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_a_780_);
lean_dec(v___x_779_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_787_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_785_; 
if (v_isShared_783_ == 0)
{
v___x_785_ = v___x_782_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_a_780_);
v___x_785_ = v_reuseFailAlloc_786_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
return v___x_785_;
}
}
}
}
else
{
lean_object* v_a_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_806_; 
lean_dec_ref_known(v___x_689_, 14);
v_a_799_ = lean_ctor_get(v___x_690_, 0);
v_isSharedCheck_806_ = !lean_is_exclusive(v___x_690_);
if (v_isSharedCheck_806_ == 0)
{
v___x_801_ = v___x_690_;
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_a_799_);
lean_dec(v___x_690_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_804_; 
if (v_isShared_802_ == 0)
{
v___x_804_ = v___x_801_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v_a_799_);
v___x_804_ = v_reuseFailAlloc_805_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
return v___x_804_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1___boxed(lean_object* v_stx_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1(v_stx_807_, v_a_808_, v_a_809_, v_a_810_, v_a_811_, v_a_812_, v_a_813_);
lean_dec(v_a_813_);
lean_dec_ref(v_a_812_);
lean_dec(v_a_811_);
lean_dec_ref(v_a_810_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0(lean_object* v_config_843_, lean_object* v_item_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_){
_start:
{
lean_object* v_item_853_; lean_object* v___y_854_; lean_object* v___y_855_; lean_object* v___y_856_; lean_object* v___y_857_; lean_object* v___y_858_; lean_object* v___y_859_; lean_object* v___x_862_; lean_object* v___x_863_; 
v___x_862_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4));
v___x_863_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_844_, v___x_862_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_863_) == 0)
{
uint8_t v___x_864_; 
lean_dec_ref_known(v___x_863_, 1);
v___x_864_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_844_);
if (v___x_864_ == 0)
{
lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; uint8_t v___x_868_; 
v___x_865_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_844_);
lean_inc_ref(v_item_844_);
v___x_866_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_844_);
v___x_867_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__1));
v___x_868_ = lean_string_dec_eq(v___x_865_, v___x_867_);
if (v___x_868_ == 0)
{
lean_object* v___x_869_; uint8_t v___x_870_; 
v___x_869_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__2));
v___x_870_ = lean_string_dec_eq(v___x_865_, v___x_869_);
if (v___x_870_ == 0)
{
lean_object* v___x_871_; uint8_t v___x_872_; 
v___x_871_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__3));
v___x_872_ = lean_string_dec_eq(v___x_865_, v___x_871_);
if (v___x_872_ == 0)
{
lean_object* v___x_873_; uint8_t v___x_874_; 
v___x_873_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__4));
v___x_874_ = lean_string_dec_eq(v___x_865_, v___x_873_);
if (v___x_874_ == 0)
{
lean_object* v___x_875_; uint8_t v___x_876_; 
v___x_875_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__5));
v___x_876_ = lean_string_dec_eq(v___x_865_, v___x_875_);
lean_dec_ref(v___x_865_);
if (v___x_876_ == 0)
{
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v___x_877_; lean_object* v___x_878_; 
v___x_877_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__6));
v___x_878_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_844_, v___x_877_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_878_) == 0)
{
uint8_t v___x_879_; 
lean_dec_ref_known(v___x_878_, 1);
v___x_879_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_866_);
if (v___x_879_ == 0)
{
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v___x_880_; 
lean_dec_ref(v___x_866_);
v___x_880_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_880_) == 0)
{
lean_object* v_a_881_; lean_object* v___x_883_; uint8_t v_isShared_884_; uint8_t v_isSharedCheck_899_; 
v_a_881_ = lean_ctor_get(v___x_880_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_880_);
if (v_isSharedCheck_899_ == 0)
{
v___x_883_ = v___x_880_;
v_isShared_884_ = v_isSharedCheck_899_;
goto v_resetjp_882_;
}
else
{
lean_inc(v_a_881_);
lean_dec(v___x_880_);
v___x_883_ = lean_box(0);
v_isShared_884_ = v_isSharedCheck_899_;
goto v_resetjp_882_;
}
v_resetjp_882_:
{
uint8_t v_newGoals_885_; uint8_t v_allowSynthFailures_886_; uint8_t v_approx_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_898_; 
v_newGoals_885_ = lean_ctor_get_uint8(v_config_843_, 0);
v_allowSynthFailures_886_ = lean_ctor_get_uint8(v_config_843_, 2);
v_approx_887_ = lean_ctor_get_uint8(v_config_843_, 3);
v_isSharedCheck_898_ = !lean_is_exclusive(v_config_843_);
if (v_isSharedCheck_898_ == 0)
{
v___x_889_ = v_config_843_;
v_isShared_890_ = v_isSharedCheck_898_;
goto v_resetjp_888_;
}
else
{
lean_dec(v_config_843_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_898_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v_reuseFailAlloc_897_, 0, v_newGoals_885_);
v___x_892_ = v_reuseFailAlloc_897_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
uint8_t v___x_893_; lean_object* v___x_895_; 
v___x_893_ = lean_unbox(v_a_881_);
lean_dec(v_a_881_);
lean_ctor_set_uint8(v___x_892_, 1, v___x_893_);
lean_ctor_set_uint8(v___x_892_, 2, v_allowSynthFailures_886_);
lean_ctor_set_uint8(v___x_892_, 3, v_approx_887_);
if (v_isShared_884_ == 0)
{
lean_ctor_set(v___x_883_, 0, v___x_892_);
v___x_895_ = v___x_883_;
goto v_reusejp_894_;
}
else
{
lean_object* v_reuseFailAlloc_896_; 
v_reuseFailAlloc_896_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_896_, 0, v___x_892_);
v___x_895_ = v_reuseFailAlloc_896_;
goto v_reusejp_894_;
}
v_reusejp_894_:
{
return v___x_895_;
}
}
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
lean_dec_ref(v_config_843_);
v_a_900_ = lean_ctor_get(v___x_880_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_880_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_880_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_880_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
}
else
{
lean_object* v_a_908_; lean_object* v___x_910_; uint8_t v_isShared_911_; uint8_t v_isSharedCheck_915_; 
lean_dec_ref(v___x_866_);
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_908_ = lean_ctor_get(v___x_878_, 0);
v_isSharedCheck_915_ = !lean_is_exclusive(v___x_878_);
if (v_isSharedCheck_915_ == 0)
{
v___x_910_ = v___x_878_;
v_isShared_911_ = v_isSharedCheck_915_;
goto v_resetjp_909_;
}
else
{
lean_inc(v_a_908_);
lean_dec(v___x_878_);
v___x_910_ = lean_box(0);
v_isShared_911_ = v_isSharedCheck_915_;
goto v_resetjp_909_;
}
v_resetjp_909_:
{
lean_object* v___x_913_; 
if (v_isShared_911_ == 0)
{
v___x_913_ = v___x_910_;
goto v_reusejp_912_;
}
else
{
lean_object* v_reuseFailAlloc_914_; 
v_reuseFailAlloc_914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_914_, 0, v_a_908_);
v___x_913_ = v_reuseFailAlloc_914_;
goto v_reusejp_912_;
}
v_reusejp_912_:
{
return v___x_913_;
}
}
}
}
}
else
{
lean_object* v___x_916_; lean_object* v___x_917_; 
lean_dec_ref(v___x_865_);
v___x_916_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__7));
v___x_917_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_844_, v___x_916_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_917_) == 0)
{
uint8_t v___x_918_; 
lean_dec_ref_known(v___x_917_, 1);
v___x_918_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_866_);
if (v___x_918_ == 0)
{
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v___x_919_; 
lean_dec_ref(v___x_866_);
lean_inc_ref(v_item_844_);
v___x_919_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_919_) == 0)
{
lean_object* v_value_920_; lean_object* v___x_921_; 
lean_dec_ref_known(v___x_919_, 1);
v_value_920_ = lean_ctor_get(v_item_844_, 2);
lean_inc(v_value_920_);
lean_dec_ref(v_item_844_);
v___x_921_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__0(v_value_920_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_921_) == 0)
{
lean_object* v_a_922_; lean_object* v___x_924_; uint8_t v_isShared_925_; uint8_t v_isSharedCheck_940_; 
v_a_922_ = lean_ctor_get(v___x_921_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_940_ == 0)
{
v___x_924_ = v___x_921_;
v_isShared_925_ = v_isSharedCheck_940_;
goto v_resetjp_923_;
}
else
{
lean_inc(v_a_922_);
lean_dec(v___x_921_);
v___x_924_ = lean_box(0);
v_isShared_925_ = v_isSharedCheck_940_;
goto v_resetjp_923_;
}
v_resetjp_923_:
{
uint8_t v_synthAssignedInstances_926_; uint8_t v_allowSynthFailures_927_; uint8_t v_approx_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_939_; 
v_synthAssignedInstances_926_ = lean_ctor_get_uint8(v_config_843_, 1);
v_allowSynthFailures_927_ = lean_ctor_get_uint8(v_config_843_, 2);
v_approx_928_ = lean_ctor_get_uint8(v_config_843_, 3);
v_isSharedCheck_939_ = !lean_is_exclusive(v_config_843_);
if (v_isSharedCheck_939_ == 0)
{
v___x_930_ = v_config_843_;
v_isShared_931_ = v_isSharedCheck_939_;
goto v_resetjp_929_;
}
else
{
lean_dec(v_config_843_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_939_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v___x_933_; 
if (v_isShared_931_ == 0)
{
v___x_933_ = v___x_930_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(0, 0, 4);
v___x_933_ = v_reuseFailAlloc_938_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
uint8_t v___x_934_; lean_object* v___x_936_; 
v___x_934_ = lean_unbox(v_a_922_);
lean_dec(v_a_922_);
lean_ctor_set_uint8(v___x_933_, 0, v___x_934_);
lean_ctor_set_uint8(v___x_933_, 1, v_synthAssignedInstances_926_);
lean_ctor_set_uint8(v___x_933_, 2, v_allowSynthFailures_927_);
lean_ctor_set_uint8(v___x_933_, 3, v_approx_928_);
if (v_isShared_925_ == 0)
{
lean_ctor_set(v___x_924_, 0, v___x_933_);
v___x_936_ = v___x_924_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v___x_933_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
}
}
}
else
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_948_; 
lean_dec_ref(v_config_843_);
v_a_941_ = lean_ctor_get(v___x_921_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_948_ == 0)
{
v___x_943_ = v___x_921_;
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_921_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_946_; 
if (v_isShared_944_ == 0)
{
v___x_946_ = v___x_943_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_a_941_);
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
else
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_956_; 
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_949_ = lean_ctor_get(v___x_919_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_919_);
if (v_isSharedCheck_956_ == 0)
{
v___x_951_ = v___x_919_;
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_919_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v___x_954_; 
if (v_isShared_952_ == 0)
{
v___x_954_ = v___x_951_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_a_949_);
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
else
{
lean_object* v_a_957_; lean_object* v___x_959_; uint8_t v_isShared_960_; uint8_t v_isSharedCheck_964_; 
lean_dec_ref(v___x_866_);
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_957_ = lean_ctor_get(v___x_917_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v___x_917_);
if (v_isSharedCheck_964_ == 0)
{
v___x_959_ = v___x_917_;
v_isShared_960_ = v_isSharedCheck_964_;
goto v_resetjp_958_;
}
else
{
lean_inc(v_a_957_);
lean_dec(v___x_917_);
v___x_959_ = lean_box(0);
v_isShared_960_ = v_isSharedCheck_964_;
goto v_resetjp_958_;
}
v_resetjp_958_:
{
lean_object* v___x_962_; 
if (v_isShared_960_ == 0)
{
v___x_962_ = v___x_959_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_963_; 
v_reuseFailAlloc_963_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_963_, 0, v_a_957_);
v___x_962_ = v_reuseFailAlloc_963_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
return v___x_962_;
}
}
}
}
}
else
{
uint8_t v___x_965_; 
lean_dec_ref(v___x_865_);
lean_dec_ref(v_config_843_);
v___x_965_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_866_);
if (v___x_965_ == 0)
{
lean_dec_ref(v_item_844_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v_value_966_; lean_object* v___x_967_; 
lean_dec_ref(v___x_866_);
v_value_966_ = lean_ctor_get(v_item_844_, 2);
lean_inc(v_value_966_);
lean_dec_ref(v_item_844_);
v___x_967_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1(v_value_966_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
return v___x_967_;
}
}
}
else
{
lean_object* v___x_968_; lean_object* v___x_969_; 
lean_dec_ref(v___x_865_);
v___x_968_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__8));
v___x_969_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_844_, v___x_968_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_969_) == 0)
{
uint8_t v___x_970_; 
lean_dec_ref_known(v___x_969_, 1);
v___x_970_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_866_);
if (v___x_970_ == 0)
{
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v___x_971_; 
lean_dec_ref(v___x_866_);
v___x_971_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_971_) == 0)
{
lean_object* v_a_972_; lean_object* v___x_974_; uint8_t v_isShared_975_; uint8_t v_isSharedCheck_990_; 
v_a_972_ = lean_ctor_get(v___x_971_, 0);
v_isSharedCheck_990_ = !lean_is_exclusive(v___x_971_);
if (v_isSharedCheck_990_ == 0)
{
v___x_974_ = v___x_971_;
v_isShared_975_ = v_isSharedCheck_990_;
goto v_resetjp_973_;
}
else
{
lean_inc(v_a_972_);
lean_dec(v___x_971_);
v___x_974_ = lean_box(0);
v_isShared_975_ = v_isSharedCheck_990_;
goto v_resetjp_973_;
}
v_resetjp_973_:
{
uint8_t v_newGoals_976_; uint8_t v_synthAssignedInstances_977_; uint8_t v_allowSynthFailures_978_; lean_object* v___x_980_; uint8_t v_isShared_981_; uint8_t v_isSharedCheck_989_; 
v_newGoals_976_ = lean_ctor_get_uint8(v_config_843_, 0);
v_synthAssignedInstances_977_ = lean_ctor_get_uint8(v_config_843_, 1);
v_allowSynthFailures_978_ = lean_ctor_get_uint8(v_config_843_, 2);
v_isSharedCheck_989_ = !lean_is_exclusive(v_config_843_);
if (v_isSharedCheck_989_ == 0)
{
v___x_980_ = v_config_843_;
v_isShared_981_ = v_isSharedCheck_989_;
goto v_resetjp_979_;
}
else
{
lean_dec(v_config_843_);
v___x_980_ = lean_box(0);
v_isShared_981_ = v_isSharedCheck_989_;
goto v_resetjp_979_;
}
v_resetjp_979_:
{
lean_object* v___x_983_; 
if (v_isShared_981_ == 0)
{
v___x_983_ = v___x_980_;
goto v_reusejp_982_;
}
else
{
lean_object* v_reuseFailAlloc_988_; 
v_reuseFailAlloc_988_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v_reuseFailAlloc_988_, 0, v_newGoals_976_);
lean_ctor_set_uint8(v_reuseFailAlloc_988_, 1, v_synthAssignedInstances_977_);
lean_ctor_set_uint8(v_reuseFailAlloc_988_, 2, v_allowSynthFailures_978_);
v___x_983_ = v_reuseFailAlloc_988_;
goto v_reusejp_982_;
}
v_reusejp_982_:
{
uint8_t v___x_984_; lean_object* v___x_986_; 
v___x_984_ = lean_unbox(v_a_972_);
lean_dec(v_a_972_);
lean_ctor_set_uint8(v___x_983_, 3, v___x_984_);
if (v_isShared_975_ == 0)
{
lean_ctor_set(v___x_974_, 0, v___x_983_);
v___x_986_ = v___x_974_;
goto v_reusejp_985_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v___x_983_);
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
else
{
lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
lean_dec_ref(v_config_843_);
v_a_991_ = lean_ctor_get(v___x_971_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v___x_971_);
if (v_isSharedCheck_998_ == 0)
{
v___x_993_ = v___x_971_;
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_971_);
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
}
else
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
lean_dec_ref(v___x_866_);
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_999_ = lean_ctor_get(v___x_969_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_969_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_1001_ = v___x_969_;
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v___x_969_);
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
}
else
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
lean_dec_ref(v___x_865_);
v___x_1007_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__9));
v___x_1008_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_844_, v___x_1007_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_1008_) == 0)
{
uint8_t v___x_1009_; 
lean_dec_ref_known(v___x_1008_, 1);
v___x_1009_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_866_);
if (v___x_1009_ == 0)
{
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_item_853_ = v___x_866_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
else
{
lean_object* v___x_1010_; 
lean_dec_ref(v___x_866_);
v___x_1010_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_1010_) == 0)
{
lean_object* v_a_1011_; lean_object* v___x_1013_; uint8_t v_isShared_1014_; uint8_t v_isSharedCheck_1029_; 
v_a_1011_ = lean_ctor_get(v___x_1010_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1010_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1013_ = v___x_1010_;
v_isShared_1014_ = v_isSharedCheck_1029_;
goto v_resetjp_1012_;
}
else
{
lean_inc(v_a_1011_);
lean_dec(v___x_1010_);
v___x_1013_ = lean_box(0);
v_isShared_1014_ = v_isSharedCheck_1029_;
goto v_resetjp_1012_;
}
v_resetjp_1012_:
{
uint8_t v_newGoals_1015_; uint8_t v_synthAssignedInstances_1016_; uint8_t v_approx_1017_; lean_object* v___x_1019_; uint8_t v_isShared_1020_; uint8_t v_isSharedCheck_1028_; 
v_newGoals_1015_ = lean_ctor_get_uint8(v_config_843_, 0);
v_synthAssignedInstances_1016_ = lean_ctor_get_uint8(v_config_843_, 1);
v_approx_1017_ = lean_ctor_get_uint8(v_config_843_, 3);
v_isSharedCheck_1028_ = !lean_is_exclusive(v_config_843_);
if (v_isSharedCheck_1028_ == 0)
{
v___x_1019_ = v_config_843_;
v_isShared_1020_ = v_isSharedCheck_1028_;
goto v_resetjp_1018_;
}
else
{
lean_dec(v_config_843_);
v___x_1019_ = lean_box(0);
v_isShared_1020_ = v_isSharedCheck_1028_;
goto v_resetjp_1018_;
}
v_resetjp_1018_:
{
lean_object* v___x_1022_; 
if (v_isShared_1020_ == 0)
{
v___x_1022_ = v___x_1019_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v_reuseFailAlloc_1027_, 0, v_newGoals_1015_);
lean_ctor_set_uint8(v_reuseFailAlloc_1027_, 1, v_synthAssignedInstances_1016_);
v___x_1022_ = v_reuseFailAlloc_1027_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
uint8_t v___x_1023_; lean_object* v___x_1025_; 
v___x_1023_ = lean_unbox(v_a_1011_);
lean_dec(v_a_1011_);
lean_ctor_set_uint8(v___x_1022_, 2, v___x_1023_);
lean_ctor_set_uint8(v___x_1022_, 3, v_approx_1017_);
if (v_isShared_1014_ == 0)
{
lean_ctor_set(v___x_1013_, 0, v___x_1022_);
v___x_1025_ = v___x_1013_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v___x_1022_);
v___x_1025_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
return v___x_1025_;
}
}
}
}
}
else
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1037_; 
lean_dec_ref(v_config_843_);
v_a_1030_ = lean_ctor_get(v___x_1010_, 0);
v_isSharedCheck_1037_ = !lean_is_exclusive(v___x_1010_);
if (v_isSharedCheck_1037_ == 0)
{
v___x_1032_ = v___x_1010_;
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_1010_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1035_; 
if (v_isShared_1033_ == 0)
{
v___x_1035_ = v___x_1032_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_a_1030_);
v___x_1035_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
return v___x_1035_;
}
}
}
}
}
else
{
lean_object* v_a_1038_; lean_object* v___x_1040_; uint8_t v_isShared_1041_; uint8_t v_isSharedCheck_1045_; 
lean_dec_ref(v___x_866_);
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_1038_ = lean_ctor_get(v___x_1008_, 0);
v_isSharedCheck_1045_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1045_ == 0)
{
v___x_1040_ = v___x_1008_;
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
else
{
lean_inc(v_a_1038_);
lean_dec(v___x_1008_);
v___x_1040_ = lean_box(0);
v_isShared_1041_ = v_isSharedCheck_1045_;
goto v_resetjp_1039_;
}
v_resetjp_1039_:
{
lean_object* v___x_1043_; 
if (v_isShared_1041_ == 0)
{
v___x_1043_ = v___x_1040_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1044_; 
v_reuseFailAlloc_1044_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1044_, 0, v_a_1038_);
v___x_1043_ = v_reuseFailAlloc_1044_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
return v___x_1043_;
}
}
}
}
}
else
{
lean_dec_ref(v_config_843_);
v_item_853_ = v_item_844_;
v___y_854_ = v___y_845_;
v___y_855_ = v___y_846_;
v___y_856_ = v___y_847_;
v___y_857_ = v___y_848_;
v___y_858_ = v___y_849_;
v___y_859_ = v___y_850_;
goto v___jp_852_;
}
}
else
{
lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1053_; 
lean_dec_ref(v_item_844_);
lean_dec_ref(v_config_843_);
v_a_1046_ = lean_ctor_get(v___x_863_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v___x_863_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1048_ = v___x_863_;
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_863_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1051_; 
if (v_isShared_1049_ == 0)
{
v___x_1051_ = v___x_1048_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v_a_1046_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
v___jp_852_:
{
lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_860_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___closed__0));
v___x_861_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_853_, v___x_860_, v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
return v___x_861_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_1054_, lean_object* v_item_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___lam__0(v_config_1054_, v_item_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2(lean_object* v_e_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v___x_1074_; 
v___x_1074_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___redArg(v_e_1066_, v___y_1070_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object* v_e_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__2(v_e_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
return v_res_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4(lean_object* v_00_u03b1_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___redArg();
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_){
_start:
{
lean_object* v_res_1101_; 
v_res_1101_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__4(v_00_u03b1_1093_, v___y_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
lean_dec(v___y_1097_);
lean_dec_ref(v___y_1096_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
return v_res_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3(lean_object* v_00_u03b1_1102_, lean_object* v_msg_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
lean_object* v___x_1111_; 
v___x_1111_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___redArg(v_msg_1103_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
return v___x_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1112_, lean_object* v_msg_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
lean_object* v_res_1121_; 
v_res_1121_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3(v_00_u03b1_1112_, v_msg_1113_, v___y_1114_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
lean_dec(v___y_1117_);
lean_dec_ref(v___y_1116_);
lean_dec(v___y_1115_);
lean_dec_ref(v___y_1114_);
return v_res_1121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4(lean_object* v_msgData_1122_, lean_object* v_macroStack_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
lean_object* v___x_1131_; 
v___x_1131_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_msgData_1122_, v_macroStack_1123_, v___y_1128_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4___boxed(lean_object* v_msgData_1132_, lean_object* v_macroStack_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_){
_start:
{
lean_object* v_res_1141_; 
v_res_1141_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem_spec__1_spec__3_spec__4(v_msgData_1132_, v_macroStack_1133_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
lean_dec(v___y_1137_);
lean_dec_ref(v___y_1136_);
lean_dec(v___y_1135_);
lean_dec_ref(v___y_1134_);
return v_res_1141_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; 
v___x_1142_ = lean_box(0);
v___x_1143_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib_evalExpr___closed__4));
v___x_1144_ = l_Lean_mkConst(v___x_1143_, v___x_1142_);
return v___x_1144_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1145_; lean_object* v___x_1146_; 
v___x_1145_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__0);
v___x_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1146_, 0, v___x_1145_);
return v___x_1146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0(lean_object* v_cfg_1147_, lean_object* v_cfgItem_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_){
_start:
{
lean_object* v___x_1156_; lean_object* v___x_1157_; 
v___x_1156_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___closed__1);
v___x_1157_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_1147_, v_cfgItem_1148_, v___x_1156_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_);
return v___x_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0___boxed(lean_object* v_cfg_1158_, lean_object* v_cfgItem_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___lam__0(v_cfg_1158_, v_cfgItem_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
lean_dec(v_cfgItem_1159_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg(lean_object* v_cfg_1169_, lean_object* v_init_1170_, uint8_t v_logExceptions_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_, lean_object* v_a_1174_){
_start:
{
lean_object* v_onErr_1176_; lean_object* v_eval_1177_; 
v_onErr_1176_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___closed__0));
v_eval_1177_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_elabApplyConfig_evalConfigItem___closed__0));
if (v_logExceptions_1171_ == 0)
{
lean_object* v___x_1178_; 
v___x_1178_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_1177_, v_init_1170_, v_cfg_1169_, v_onErr_1176_, v_logExceptions_1171_, v_a_1173_, v_a_1174_);
return v___x_1178_;
}
else
{
uint8_t v_recover_1179_; lean_object* v___x_1180_; 
v_recover_1179_ = lean_ctor_get_uint8(v_a_1172_, sizeof(void*)*1);
v___x_1180_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_1177_, v_init_1170_, v_cfg_1169_, v_onErr_1176_, v_recover_1179_, v_a_1173_, v_a_1174_);
return v___x_1180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg___boxed(lean_object* v_cfg_1181_, lean_object* v_init_1182_, lean_object* v_logExceptions_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_){
_start:
{
uint8_t v_logExceptions_boxed_1188_; lean_object* v_res_1189_; 
v_logExceptions_boxed_1188_ = lean_unbox(v_logExceptions_1183_);
v_res_1189_ = lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg(v_cfg_1181_, v_init_1182_, v_logExceptions_boxed_1188_, v_a_1184_, v_a_1185_, v_a_1186_);
lean_dec(v_a_1186_);
lean_dec_ref(v_a_1185_);
lean_dec_ref(v_a_1184_);
return v_res_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig(lean_object* v_cfg_1190_, lean_object* v_init_1191_, uint8_t v_logExceptions_1192_, lean_object* v_a_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_, lean_object* v_a_1199_, lean_object* v_a_1200_){
_start:
{
lean_object* v___x_1202_; 
v___x_1202_ = lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg(v_cfg_1190_, v_init_1191_, v_logExceptions_1192_, v_a_1193_, v_a_1199_, v_a_1200_);
return v___x_1202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabApplyConfig___boxed(lean_object* v_cfg_1203_, lean_object* v_init_1204_, lean_object* v_logExceptions_1205_, lean_object* v_a_1206_, lean_object* v_a_1207_, lean_object* v_a_1208_, lean_object* v_a_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_, lean_object* v_a_1213_, lean_object* v_a_1214_){
_start:
{
uint8_t v_logExceptions_boxed_1215_; lean_object* v_res_1216_; 
v_logExceptions_boxed_1215_ = lean_unbox(v_logExceptions_1205_);
v_res_1216_ = lp_mathlib_Mathlib_Tactic_elabApplyConfig(v_cfg_1203_, v_init_1204_, v_logExceptions_boxed_1215_, v_a_1206_, v_a_1207_, v_a_1208_, v_a_1209_, v_a_1210_, v_a_1211_, v_a_1212_, v_a_1213_);
lean_dec(v_a_1213_);
lean_dec_ref(v_a_1212_);
lean_dec(v_a_1211_);
lean_dec_ref(v_a_1210_);
lean_dec(v_a_1209_);
lean_dec_ref(v_a_1208_);
lean_dec(v_a_1207_);
lean_dec_ref(v_a_1206_);
return v_res_1216_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__4(void){
_start:
{
lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; 
v___x_1226_ = lp_mathlib_Mathlib_Tactic_manyConfig;
v___x_1227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyWith___closed__3));
v___x_1228_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__7));
v___x_1229_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1229_, 0, v___x_1228_);
lean_ctor_set(v___x_1229_, 1, v___x_1227_);
lean_ctor_set(v___x_1229_, 2, v___x_1226_);
return v___x_1229_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__11(void){
_start:
{
lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; 
v___x_1241_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyWith___closed__10));
v___x_1242_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyWith___closed__4, &lp_mathlib_Mathlib_Tactic_applyWith___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__4);
v___x_1243_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__7));
v___x_1244_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1243_);
lean_ctor_set(v___x_1244_, 1, v___x_1242_);
lean_ctor_set(v___x_1244_, 2, v___x_1241_);
return v___x_1244_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__15(void){
_start:
{
lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; 
v___x_1251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyWith___closed__14));
v___x_1252_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyWith___closed__11, &lp_mathlib_Mathlib_Tactic_applyWith___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__11);
v___x_1253_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_manyConfig___closed__7));
v___x_1254_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1254_, 0, v___x_1253_);
lean_ctor_set(v___x_1254_, 1, v___x_1252_);
lean_ctor_set(v___x_1254_, 2, v___x_1251_);
return v___x_1254_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__16(void){
_start:
{
lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; 
v___x_1255_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyWith___closed__15, &lp_mathlib_Mathlib_Tactic_applyWith___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__15);
v___x_1256_ = lean_unsigned_to_nat(1022u);
v___x_1257_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyWith___closed__1));
v___x_1258_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1258_, 0, v___x_1257_);
lean_ctor_set(v___x_1258_, 1, v___x_1256_);
lean_ctor_set(v___x_1258_, 2, v___x_1255_);
return v___x_1258_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyWith(void){
_start:
{
lean_object* v___x_1259_; 
v___x_1259_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyWith___closed__16, &lp_mathlib_Mathlib_Tactic_applyWith___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_applyWith___closed__16);
return v___x_1259_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; 
v___x_1260_ = lean_box(0);
v___x_1261_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1262_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1262_, 0, v___x_1261_);
lean_ctor_set(v___x_1262_, 1, v___x_1260_);
return v___x_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; 
v___x_1264_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___closed__0);
v___x_1265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1265_, 0, v___x_1264_);
return v___x_1265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg___boxed(lean_object* v___y_1266_){
_start:
{
lean_object* v_res_1267_; 
v_res_1267_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg();
return v_res_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0(lean_object* v_00_u03b1_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_){
_start:
{
lean_object* v___x_1278_; 
v___x_1278_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg();
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___boxed(lean_object* v_00_u03b1_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
lean_object* v_res_1289_; 
v_res_1289_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0(v_00_u03b1_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_);
lean_dec(v___y_1287_);
lean_dec_ref(v___y_1286_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
return v_res_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0(lean_object* v_a_1290_, lean_object* v_x1_1291_, lean_object* v_x2_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_){
_start:
{
lean_object* v___x_1298_; lean_object* v___x_1299_; 
v___x_1298_ = lean_box(0);
v___x_1299_ = l_Lean_MVarId_apply(v_x1_1291_, v_x2_1292_, v_a_1290_, v___x_1298_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_);
return v___x_1299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0___boxed(lean_object* v_a_1300_, lean_object* v_x1_1301_, lean_object* v_x2_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_){
_start:
{
lean_object* v_res_1308_; 
v_res_1308_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0(v_a_1300_, v_x1_1301_, v_x2_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_);
lean_dec(v___y_1306_);
lean_dec_ref(v___y_1305_);
lean_dec(v___y_1304_);
lean_dec_ref(v___y_1303_);
return v_res_1308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1(lean_object* v_x_1309_, lean_object* v_a_1310_, lean_object* v_a_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v_a_1314_, lean_object* v_a_1315_, lean_object* v_a_1316_, lean_object* v_a_1317_){
_start:
{
lean_object* v___x_1319_; uint8_t v___x_1320_; 
v___x_1319_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyWith___closed__1));
lean_inc(v_x_1309_);
v___x_1320_ = l_Lean_Syntax_isOfKind(v_x_1309_, v___x_1319_);
if (v___x_1320_ == 0)
{
lean_object* v___x_1321_; 
lean_dec(v_x_1309_);
v___x_1321_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1_spec__0___redArg();
return v___x_1321_;
}
else
{
lean_object* v___x_1322_; lean_object* v___x_1323_; uint8_t v___x_1324_; uint8_t v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1322_ = lean_unsigned_to_nat(1u);
v___x_1323_ = l_Lean_Syntax_getArg(v_x_1309_, v___x_1322_);
v___x_1324_ = 0;
v___x_1325_ = 0;
v___x_1326_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_1326_, 0, v___x_1324_);
lean_ctor_set_uint8(v___x_1326_, 1, v___x_1320_);
lean_ctor_set_uint8(v___x_1326_, 2, v___x_1325_);
lean_ctor_set_uint8(v___x_1326_, 3, v___x_1320_);
v___x_1327_ = lp_mathlib_Mathlib_Tactic_elabApplyConfig___redArg(v___x_1323_, v___x_1326_, v___x_1320_, v_a_1310_, v_a_1316_, v_a_1317_);
if (lean_obj_tag(v___x_1327_) == 0)
{
lean_object* v_a_1328_; lean_object* v___f_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; 
v_a_1328_ = lean_ctor_get(v___x_1327_, 0);
lean_inc(v_a_1328_);
lean_dec_ref_known(v___x_1327_, 1);
v___f_1329_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1329_, 0, v_a_1328_);
v___x_1330_ = lean_unsigned_to_nat(3u);
v___x_1331_ = l_Lean_Syntax_getArg(v_x_1309_, v___x_1330_);
lean_dec(v_x_1309_);
v___x_1332_ = l_Lean_Elab_Tactic_evalApplyLikeTactic(v___f_1329_, v___x_1331_, v_a_1310_, v_a_1311_, v_a_1312_, v_a_1313_, v_a_1314_, v_a_1315_, v_a_1316_, v_a_1317_);
return v___x_1332_;
}
else
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1340_; 
lean_dec(v_x_1309_);
v_a_1333_ = lean_ctor_get(v___x_1327_, 0);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1327_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1335_ = v___x_1327_;
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1327_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1338_; 
if (v_isShared_1336_ == 0)
{
v___x_1338_ = v___x_1335_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_a_1333_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1___boxed(lean_object* v_x_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_, lean_object* v_a_1349_, lean_object* v_a_1350_){
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyWith______elabRules__Mathlib__Tactic__applyWith__1(v_x_1341_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
lean_dec(v_a_1349_);
lean_dec_ref(v_a_1348_);
lean_dec(v_a_1347_);
lean_dec_ref(v_a_1346_);
lean_dec(v_a_1345_);
lean_dec_ref(v_a_1344_);
lean_dec(v_a_1343_);
lean_dec_ref(v_a_1342_);
return v_res_1351_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Eval(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_manyConfig = _init_lp_mathlib_Mathlib_Tactic_manyConfig();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_manyConfig);
lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib = _init_lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_ApplyWith_0__Mathlib_Tactic_instEvalExprApplyConfig__mathlib);
lp_mathlib_Mathlib_Tactic_applyWith = _init_lp_mathlib_Mathlib_Tactic_applyWith();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_applyWith);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Eval(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
}
#ifdef __cplusplus
}
#endif
