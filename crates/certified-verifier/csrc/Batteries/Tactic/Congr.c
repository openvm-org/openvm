// Lean compiler output
// Module: Batteries.Tactic.Congr
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Congr public meta import Lean.Elab.Tactic.Config public meta import Lean.Elab.Tactic.Ext public meta import Lean.Elab.Tactic.RCases public meta import Lean.Elab.ConfigEval public import Lean.Elab.ConfigEval
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_config;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Term_saveState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_congrN(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_SavedState_restore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Ext_extCore(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_RCases_expandRIntroPats(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Congr"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4_value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(178, 169, 109, 36, 185, 120, 77, 252)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 129, 6, 218, 64, 4, 26, 24)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig;
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0;
static const lean_string_object lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1 = (const lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value;
static const lean_ctor_object lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value)}};
static const lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2 = (const lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__0 = (const lean_object*)&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3;
static const lean_string_object lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__4 = (const lean_object*)&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6;
static const lean_string_object lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__7 = (const lean_object*)&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__7_value;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8;
static const lean_string_object lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__9 = (const lean_object*)&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__9_value;
static lean_once_cell_t lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5_value)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "closePost"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "closePre"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(178, 169, 109, 36, 185, 120, 77, 252)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 129, 6, 218, 64, 4, 26, 24)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(122, 26, 66, 106, 100, 160, 235, 205)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(178, 169, 109, 36, 185, 120, 77, 252)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(149, 129, 6, 218, 64, 4, 26, 24)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_3),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(168, 137, 202, 215, 132, 54, 251, 178)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "congrConfig"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 247, 253, 74, 222, 62, 63, 240)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "congr"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfig___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__6;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__9_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfig___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__12_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__11_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfig___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__16_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfig___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__17;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfig___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfig___closed__18;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_congrConfig;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "congrConfigWith"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__0_value),LEAN_SCALAR_PTR_LITERAL(238, 29, 226, 27, 90, 212, 205, 209)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__2;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__11_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__9_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__10;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__12_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__13;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__14_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__15_value;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rintroPat"};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__16_value),LEAN_SCALAR_PTR_LITERAL(105, 195, 203, 253, 3, 13, 142, 19)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__15_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__20_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__21;
static const lean_string_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__22 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__22_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__22_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__23 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__23_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__23_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__24 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__24_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_congrConfigWith___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__24_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__25 = (const lean_object*)&lp_batteries_Batteries_Tactic_congrConfigWith___closed__25_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__26;
static lean_once_cell_t lp_batteries_Batteries_Tactic_congrConfigWith___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_congrConfigWith___closed__27;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_congrConfigWith;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value_aux_2),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(230, 254, 59, 95, 54, 234, 162, 220)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ext"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__4_value;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_congrConfig___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 88, 242, 177, 210, 111, 166, 107)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_rcongrCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_rcongrCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_rcongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rcongr"};
static const lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_rcongr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_rcongr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_rcongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 159, 235, 254, 240, 194, 134, 234)}};
static const lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_rcongr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_rcongr___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_rcongr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__3;
static lean_once_cell_t lp_batteries_Batteries_Tactic_rcongr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__4;
static lean_once_cell_t lp_batteries_Batteries_Tactic_rcongr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_rcongr___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_rcongr;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
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
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_6_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object* v_msgData_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
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
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msgData_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_ref_51_; lean_object* v___x_52_; lean_object* v_a_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_61_; 
v_ref_51_ = lean_ctor_get(v___y_48_, 5);
v___x_52_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
return v_res_68_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_72_ = l_Lean_stringToMessageData(v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_73_, lean_object* v_args_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_117_ = lean_string_dec_eq(v_ctor_73_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; 
v___x_118_ = lp_batteries_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_119_ = lean_array_get_size(v_args_74_);
v___x_120_ = lean_unsigned_to_nat(2u);
v___x_121_ = lean_nat_dec_eq(v___x_119_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_131_; 
v___x_122_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_123_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_122_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
v_a_124_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_131_ == 0)
{
v___x_126_ = v___x_123_;
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_123_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_129_; 
if (v_isShared_127_ == 0)
{
v___x_129_ = v___x_126_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v_a_124_);
v___x_129_ = v_reuseFailAlloc_130_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
return v___x_129_;
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
v___x_84_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_83_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
if (lean_obj_tag(v___x_84_) == 0)
{
lean_object* v_a_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v_a_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_a_85_);
lean_dec_ref_known(v___x_84_, 1);
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = lean_array_get_borrowed(v___x_81_, v_args_74_, v___x_86_);
lean_inc(v___x_87_);
v___x_88_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_87_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_99_; 
v_a_89_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_99_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_99_ == 0)
{
v___x_91_ = v___x_88_;
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_88_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; uint8_t v___x_94_; uint8_t v___x_95_; lean_object* v___x_97_; 
v___x_93_ = lean_alloc_ctor(0, 0, 2);
v___x_94_ = lean_unbox(v_a_85_);
lean_dec(v_a_85_);
lean_ctor_set_uint8(v___x_93_, 0, v___x_94_);
v___x_95_ = lean_unbox(v_a_89_);
lean_dec(v_a_89_);
lean_ctor_set_uint8(v___x_93_, 1, v___x_95_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_97_ = v___x_91_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v___x_93_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
else
{
lean_object* v_a_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_107_; 
lean_dec(v_a_85_);
v_a_100_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_107_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_107_ == 0)
{
v___x_102_ = v___x_88_;
v_isShared_103_ = v_isSharedCheck_107_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_a_100_);
lean_dec(v___x_88_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_107_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___x_105_; 
if (v_isShared_103_ == 0)
{
v___x_105_ = v___x_102_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_a_100_);
v___x_105_ = v_reuseFailAlloc_106_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
return v___x_105_;
}
}
}
}
else
{
lean_object* v_a_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_115_; 
v_a_108_ = lean_ctor_get(v___x_84_, 0);
v_isSharedCheck_115_ = !lean_is_exclusive(v___x_84_);
if (v_isSharedCheck_115_ == 0)
{
v___x_110_ = v___x_84_;
v_isShared_111_ = v_isSharedCheck_115_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_a_108_);
lean_dec(v___x_84_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_115_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___x_113_; 
if (v_isShared_111_ == 0)
{
v___x_113_ = v___x_110_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v_a_108_);
v___x_113_ = v_reuseFailAlloc_114_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
return v___x_113_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_132_, lean_object* v_args_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___lam__0(v_ctor_132_, v_args_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec_ref(v_args_133_);
lean_dec_ref(v_ctor_132_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr(lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_, lean_object* v_a_154_){
_start:
{
lean_object* v___f_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___f_156_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__0));
v___x_157_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5));
v___x_158_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_157_, v___f_156_, v_a_150_, v_a_151_, v_a_152_, v_a_153_, v_a_154_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr(v_a_159_, v_a_160_, v_a_161_, v_a_162_, v_a_163_);
lean_dec(v_a_163_);
lean_dec_ref(v_a_162_);
lean_dec(v_a_161_);
lean_dec_ref(v_a_160_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_166_, lean_object* v_msg_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_167_, v___y_168_, v___y_169_, v___y_170_, v___y_171_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_174_, lean_object* v_msg_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_174_, v_msg_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
return v_res_181_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_183_ = lean_box(0);
v___x_184_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5));
v___x_185_ = l_Lean_Expr_const___override(v___x_184_, v___x_183_);
return v___x_185_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1);
v___x_187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_187_, 0, v___x_186_);
return v___x_187_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_188_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2);
v___x_189_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__0));
v___x_190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
lean_ctor_set(v___x_190_, 1, v___x_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig(void){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__3);
return v___x_191_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object* v_opts_192_, lean_object* v_opt_193_){
_start:
{
lean_object* v_name_194_; lean_object* v_defValue_195_; lean_object* v_map_196_; lean_object* v___x_197_; 
v_name_194_ = lean_ctor_get(v_opt_193_, 0);
v_defValue_195_ = lean_ctor_get(v_opt_193_, 1);
v_map_196_ = lean_ctor_get(v_opts_192_, 0);
v___x_197_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_196_, v_name_194_);
if (lean_obj_tag(v___x_197_) == 0)
{
uint8_t v___x_198_; 
v___x_198_ = lean_unbox(v_defValue_195_);
return v___x_198_;
}
else
{
lean_object* v_val_199_; 
v_val_199_ = lean_ctor_get(v___x_197_, 0);
lean_inc(v_val_199_);
lean_dec_ref_known(v___x_197_, 1);
if (lean_obj_tag(v_val_199_) == 1)
{
uint8_t v_v_200_; 
v_v_200_ = lean_ctor_get_uint8(v_val_199_, 0);
lean_dec_ref_known(v_val_199_, 0);
return v_v_200_;
}
else
{
uint8_t v___x_201_; 
lean_dec(v_val_199_);
v___x_201_ = lean_unbox(v_defValue_195_);
return v___x_201_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_opts_202_, lean_object* v_opt_203_){
_start:
{
uint8_t v_res_204_; lean_object* v_r_205_; 
v_res_204_ = lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_opts_202_, v_opt_203_);
lean_dec_ref(v_opt_203_);
lean_dec_ref(v_opts_202_);
v_r_205_ = lean_box(v_res_204_);
return v_r_205_;
}
}
static lean_object* _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = lean_box(1);
v___x_207_ = l_Lean_MessageData_ofFormat(v___x_206_);
return v___x_207_;
}
}
static lean_object* _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = ((lean_object*)(lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2));
v___x_212_ = l_Lean_MessageData_ofFormat(v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object* v_x_213_, lean_object* v_x_214_){
_start:
{
if (lean_obj_tag(v_x_214_) == 0)
{
return v_x_213_;
}
else
{
lean_object* v_head_215_; lean_object* v_tail_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_238_; 
v_head_215_ = lean_ctor_get(v_x_214_, 0);
v_tail_216_ = lean_ctor_get(v_x_214_, 1);
v_isSharedCheck_238_ = !lean_is_exclusive(v_x_214_);
if (v_isSharedCheck_238_ == 0)
{
v___x_218_ = v_x_214_;
v_isShared_219_ = v_isSharedCheck_238_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_tail_216_);
lean_inc(v_head_215_);
lean_dec(v_x_214_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_238_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v_before_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_236_; 
v_before_220_ = lean_ctor_get(v_head_215_, 0);
v_isSharedCheck_236_ = !lean_is_exclusive(v_head_215_);
if (v_isSharedCheck_236_ == 0)
{
lean_object* v_unused_237_; 
v_unused_237_ = lean_ctor_get(v_head_215_, 1);
lean_dec(v_unused_237_);
v___x_222_ = v_head_215_;
v_isShared_223_ = v_isSharedCheck_236_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_before_220_);
lean_dec(v_head_215_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_236_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_224_; lean_object* v___x_226_; 
v___x_224_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_223_ == 0)
{
lean_ctor_set_tag(v___x_222_, 7);
lean_ctor_set(v___x_222_, 1, v___x_224_);
lean_ctor_set(v___x_222_, 0, v_x_213_);
v___x_226_ = v___x_222_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v_x_213_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v___x_224_);
v___x_226_ = v_reuseFailAlloc_235_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
lean_object* v___x_227_; lean_object* v___x_229_; 
v___x_227_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3);
if (v_isShared_219_ == 0)
{
lean_ctor_set_tag(v___x_218_, 7);
lean_ctor_set(v___x_218_, 1, v___x_227_);
lean_ctor_set(v___x_218_, 0, v___x_226_);
v___x_229_ = v___x_218_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v___x_227_);
v___x_229_ = v_reuseFailAlloc_234_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_230_ = l_Lean_MessageData_ofSyntax(v_before_220_);
v___x_231_ = l_Lean_indentD(v___x_230_);
v___x_232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_229_);
lean_ctor_set(v___x_232_, 1, v___x_231_);
v_x_213_ = v___x_232_;
v_x_214_ = v_tail_216_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = ((lean_object*)(lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1));
v___x_243_ = l_Lean_MessageData_ofFormat(v___x_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object* v_msgData_244_, lean_object* v_macroStack_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_options_248_; lean_object* v___x_249_; uint8_t v___x_250_; 
v_options_248_ = lean_ctor_get(v___y_246_, 2);
v___x_249_ = l_Lean_Elab_pp_macroStack;
v___x_250_ = lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_options_248_, v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; 
lean_dec(v_macroStack_245_);
v___x_251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_251_, 0, v_msgData_244_);
return v___x_251_;
}
else
{
if (lean_obj_tag(v_macroStack_245_) == 0)
{
lean_object* v___x_252_; 
v___x_252_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_252_, 0, v_msgData_244_);
return v___x_252_;
}
else
{
lean_object* v_head_253_; lean_object* v_after_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_269_; 
v_head_253_ = lean_ctor_get(v_macroStack_245_, 0);
lean_inc(v_head_253_);
v_after_254_ = lean_ctor_get(v_head_253_, 1);
v_isSharedCheck_269_ = !lean_is_exclusive(v_head_253_);
if (v_isSharedCheck_269_ == 0)
{
lean_object* v_unused_270_; 
v_unused_270_ = lean_ctor_get(v_head_253_, 0);
lean_dec(v_unused_270_);
v___x_256_ = v_head_253_;
v_isShared_257_ = v_isSharedCheck_269_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_after_254_);
lean_dec(v_head_253_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_269_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_258_; lean_object* v___x_260_; 
v___x_258_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_257_ == 0)
{
lean_ctor_set_tag(v___x_256_, 7);
lean_ctor_set(v___x_256_, 1, v___x_258_);
lean_ctor_set(v___x_256_, 0, v_msgData_244_);
v___x_260_ = v___x_256_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v_msgData_244_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v___x_258_);
v___x_260_ = v_reuseFailAlloc_268_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v_msgData_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_261_ = lean_obj_once(&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2, &lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once, _init_lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2);
v___x_262_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_260_);
lean_ctor_set(v___x_262_, 1, v___x_261_);
v___x_263_ = l_Lean_MessageData_ofSyntax(v_after_254_);
v___x_264_ = l_Lean_indentD(v___x_263_);
v_msgData_265_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_265_, 0, v___x_262_);
lean_ctor_set(v_msgData_265_, 1, v___x_264_);
v___x_266_ = lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(v_msgData_265_, v_macroStack_245_);
v___x_267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
return v___x_267_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_271_, lean_object* v_macroStack_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_271_, v_macroStack_272_, v___y_273_);
lean_dec_ref(v___y_273_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object* v_msg_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v_ref_284_; lean_object* v___x_285_; lean_object* v_a_286_; lean_object* v_macroStack_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v_a_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_298_; 
v_ref_284_ = lean_ctor_get(v___y_281_, 5);
v___x_285_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_276_, v___y_279_, v___y_280_, v___y_281_, v___y_282_);
v_a_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_a_286_);
lean_dec_ref(v___x_285_);
v_macroStack_287_ = lean_ctor_get(v___y_277_, 1);
v___x_288_ = l_Lean_Elab_getBetterRef(v_ref_284_, v_macroStack_287_);
lean_inc(v_macroStack_287_);
v___x_289_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_a_286_, v_macroStack_287_, v___y_281_);
v_a_290_ = lean_ctor_get(v___x_289_, 0);
v_isSharedCheck_298_ = !lean_is_exclusive(v___x_289_);
if (v_isSharedCheck_298_ == 0)
{
v___x_292_ = v___x_289_;
v_isShared_293_ = v_isSharedCheck_298_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_a_290_);
lean_dec(v___x_289_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_298_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v___x_294_; lean_object* v___x_296_; 
v___x_294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_288_);
lean_ctor_set(v___x_294_, 1, v_a_290_);
if (v_isShared_293_ == 0)
{
lean_ctor_set_tag(v___x_292_, 1);
lean_ctor_set(v___x_292_, 0, v___x_294_);
v___x_296_ = v___x_292_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_294_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object* v_msg_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
return v_res_307_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_308_ = lean_box(0);
v___x_309_ = l_Lean_Elab_abortTermExceptionId;
v___x_310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v___x_308_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg(){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_312_ = lean_obj_once(&lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0, &lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0);
v___x_313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object* v___y_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object* v_e_316_, lean_object* v___y_317_){
_start:
{
uint8_t v___x_319_; 
v___x_319_ = l_Lean_Expr_hasMVar(v_e_316_);
if (v___x_319_ == 0)
{
lean_object* v___x_320_; 
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v_e_316_);
return v___x_320_;
}
else
{
lean_object* v___x_321_; lean_object* v_mctx_322_; lean_object* v___x_323_; lean_object* v_fst_324_; lean_object* v_snd_325_; lean_object* v___x_326_; lean_object* v_cache_327_; lean_object* v_zetaDeltaFVarIds_328_; lean_object* v_postponed_329_; lean_object* v_diag_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_339_; 
v___x_321_ = lean_st_ref_get(v___y_317_);
v_mctx_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc_ref(v_mctx_322_);
lean_dec(v___x_321_);
v___x_323_ = l_Lean_instantiateMVarsCore(v_mctx_322_, v_e_316_);
v_fst_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc(v_fst_324_);
v_snd_325_ = lean_ctor_get(v___x_323_, 1);
lean_inc(v_snd_325_);
lean_dec_ref(v___x_323_);
v___x_326_ = lean_st_ref_take(v___y_317_);
v_cache_327_ = lean_ctor_get(v___x_326_, 1);
v_zetaDeltaFVarIds_328_ = lean_ctor_get(v___x_326_, 2);
v_postponed_329_ = lean_ctor_get(v___x_326_, 3);
v_diag_330_ = lean_ctor_get(v___x_326_, 4);
v_isSharedCheck_339_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_339_ == 0)
{
lean_object* v_unused_340_; 
v_unused_340_ = lean_ctor_get(v___x_326_, 0);
lean_dec(v_unused_340_);
v___x_332_ = v___x_326_;
v_isShared_333_ = v_isSharedCheck_339_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_diag_330_);
lean_inc(v_postponed_329_);
lean_inc(v_zetaDeltaFVarIds_328_);
lean_inc(v_cache_327_);
lean_dec(v___x_326_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_339_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_335_; 
if (v_isShared_333_ == 0)
{
lean_ctor_set(v___x_332_, 0, v_snd_325_);
v___x_335_ = v___x_332_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_snd_325_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v_cache_327_);
lean_ctor_set(v_reuseFailAlloc_338_, 2, v_zetaDeltaFVarIds_328_);
lean_ctor_set(v_reuseFailAlloc_338_, 3, v_postponed_329_);
lean_ctor_set(v_reuseFailAlloc_338_, 4, v_diag_330_);
v___x_335_ = v_reuseFailAlloc_338_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_336_ = lean_st_ref_set(v___y_317_, v___x_335_);
v___x_337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_337_, 0, v_fst_324_);
return v___x_337_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object* v_e_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_341_, v___y_342_);
lean_dec(v___y_342_);
return v_res_344_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1(void){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_346_ = ((lean_object*)(lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__0));
v___x_347_ = l_Lean_stringToMessageData(v___x_346_);
return v___x_347_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2(void){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_348_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__1);
v___x_349_ = l_Lean_MessageData_ofExpr(v___x_348_);
return v___x_349_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_350_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__2);
v___x_351_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__1);
v___x_352_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_350_);
return v___x_352_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = ((lean_object*)(lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__4));
v___x_355_ = l_Lean_stringToMessageData(v___x_354_);
return v___x_355_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_356_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__5);
v___x_357_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__3);
v___x_358_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v___x_356_);
return v___x_358_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_360_ = ((lean_object*)(lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__7));
v___x_361_ = l_Lean_stringToMessageData(v___x_360_);
return v___x_361_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_363_ = ((lean_object*)(lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__9));
v___x_364_ = l_Lean_stringToMessageData(v___x_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0(lean_object* v_stx_365_, lean_object* v_a_366_, lean_object* v_a_367_, lean_object* v_a_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_){
_start:
{
lean_object* v_ty_x3f_373_; uint8_t v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v_fileName_379_; lean_object* v_fileMap_380_; lean_object* v_options_381_; lean_object* v_currRecDepth_382_; lean_object* v_maxRecDepth_383_; lean_object* v_ref_384_; lean_object* v_currNamespace_385_; lean_object* v_openDecls_386_; lean_object* v_initHeartbeats_387_; lean_object* v_maxHeartbeats_388_; lean_object* v_quotContext_389_; lean_object* v_currMacroScope_390_; uint8_t v_diag_391_; lean_object* v_cancelTk_x3f_392_; uint8_t v_suppressElabErrors_393_; lean_object* v_inheritedTraceOptions_394_; uint8_t v___x_395_; lean_object* v_ref_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
v_ty_x3f_373_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2, &lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2_once, _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig___closed__2);
v___x_374_ = 1;
v___x_375_ = lean_box(0);
v___x_376_ = lean_box(v___x_374_);
v___x_377_ = lean_box(v___x_374_);
lean_inc(v_stx_365_);
v___x_378_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_378_, 0, v_stx_365_);
lean_closure_set(v___x_378_, 1, v_ty_x3f_373_);
lean_closure_set(v___x_378_, 2, v___x_376_);
lean_closure_set(v___x_378_, 3, v___x_377_);
lean_closure_set(v___x_378_, 4, v___x_375_);
v_fileName_379_ = lean_ctor_get(v_a_370_, 0);
v_fileMap_380_ = lean_ctor_get(v_a_370_, 1);
v_options_381_ = lean_ctor_get(v_a_370_, 2);
v_currRecDepth_382_ = lean_ctor_get(v_a_370_, 3);
v_maxRecDepth_383_ = lean_ctor_get(v_a_370_, 4);
v_ref_384_ = lean_ctor_get(v_a_370_, 5);
v_currNamespace_385_ = lean_ctor_get(v_a_370_, 6);
v_openDecls_386_ = lean_ctor_get(v_a_370_, 7);
v_initHeartbeats_387_ = lean_ctor_get(v_a_370_, 8);
v_maxHeartbeats_388_ = lean_ctor_get(v_a_370_, 9);
v_quotContext_389_ = lean_ctor_get(v_a_370_, 10);
v_currMacroScope_390_ = lean_ctor_get(v_a_370_, 11);
v_diag_391_ = lean_ctor_get_uint8(v_a_370_, sizeof(void*)*14);
v_cancelTk_x3f_392_ = lean_ctor_get(v_a_370_, 12);
v_suppressElabErrors_393_ = lean_ctor_get_uint8(v_a_370_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_394_ = lean_ctor_get(v_a_370_, 13);
v___x_395_ = 1;
v_ref_396_ = l_Lean_replaceRef(v_stx_365_, v_ref_384_);
lean_dec(v_stx_365_);
lean_inc_ref(v_inheritedTraceOptions_394_);
lean_inc(v_cancelTk_x3f_392_);
lean_inc(v_currMacroScope_390_);
lean_inc(v_quotContext_389_);
lean_inc(v_maxHeartbeats_388_);
lean_inc(v_initHeartbeats_387_);
lean_inc(v_openDecls_386_);
lean_inc(v_currNamespace_385_);
lean_inc(v_maxRecDepth_383_);
lean_inc(v_currRecDepth_382_);
lean_inc_ref(v_options_381_);
lean_inc_ref(v_fileMap_380_);
lean_inc_ref(v_fileName_379_);
v___x_397_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_397_, 0, v_fileName_379_);
lean_ctor_set(v___x_397_, 1, v_fileMap_380_);
lean_ctor_set(v___x_397_, 2, v_options_381_);
lean_ctor_set(v___x_397_, 3, v_currRecDepth_382_);
lean_ctor_set(v___x_397_, 4, v_maxRecDepth_383_);
lean_ctor_set(v___x_397_, 5, v_ref_396_);
lean_ctor_set(v___x_397_, 6, v_currNamespace_385_);
lean_ctor_set(v___x_397_, 7, v_openDecls_386_);
lean_ctor_set(v___x_397_, 8, v_initHeartbeats_387_);
lean_ctor_set(v___x_397_, 9, v_maxHeartbeats_388_);
lean_ctor_set(v___x_397_, 10, v_quotContext_389_);
lean_ctor_set(v___x_397_, 11, v_currMacroScope_390_);
lean_ctor_set(v___x_397_, 12, v_cancelTk_x3f_392_);
lean_ctor_set(v___x_397_, 13, v_inheritedTraceOptions_394_);
lean_ctor_set_uint8(v___x_397_, sizeof(void*)*14, v_diag_391_);
lean_ctor_set_uint8(v___x_397_, sizeof(void*)*14 + 1, v_suppressElabErrors_393_);
v___x_398_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_378_, v___x_395_, v_a_366_, v_a_367_, v_a_368_, v_a_369_, v___x_397_, v_a_371_);
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v_a_399_; lean_object* v___x_400_; lean_object* v_a_401_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v___y_405_; lean_object* v___y_406_; lean_object* v___y_407_; lean_object* v___y_408_; lean_object* v___y_409_; lean_object* v___y_410_; lean_object* v___y_411_; uint8_t v___y_412_; lean_object* v___y_429_; lean_object* v___y_430_; lean_object* v___y_431_; lean_object* v___y_432_; lean_object* v___y_433_; lean_object* v___y_434_; lean_object* v___y_441_; lean_object* v___y_442_; lean_object* v___y_443_; lean_object* v___y_444_; lean_object* v___y_445_; lean_object* v___y_446_; lean_object* v___y_478_; lean_object* v___y_479_; lean_object* v___y_480_; lean_object* v___y_481_; lean_object* v___y_482_; lean_object* v___y_483_; uint8_t v___x_496_; 
v_a_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_a_399_);
lean_dec_ref_known(v___x_398_, 1);
v___x_400_ = lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_a_399_, v_a_369_);
v_a_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_a_401_);
lean_dec_ref(v___x_400_);
v___x_496_ = l_Lean_Expr_hasSorry(v_a_401_);
if (v___x_496_ == 0)
{
v___y_441_ = v_a_366_;
v___y_442_ = v_a_367_;
v___y_443_ = v_a_368_;
v___y_444_ = v_a_369_;
v___y_445_ = v___x_397_;
v___y_446_ = v_a_371_;
goto v___jp_440_;
}
else
{
uint8_t v___x_497_; 
v___x_497_ = l_Lean_Expr_hasSyntheticSorry(v_a_401_);
if (v___x_497_ == 0)
{
v___y_478_ = v_a_366_;
v___y_479_ = v_a_367_;
v___y_480_ = v_a_368_;
v___y_481_ = v_a_369_;
v___y_482_ = v___x_397_;
v___y_483_ = v_a_371_;
goto v___jp_477_;
}
else
{
lean_object* v___x_498_; lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_506_; 
lean_dec(v_a_401_);
lean_dec_ref_known(v___x_397_, 14);
v___x_498_ = lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_499_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_506_ == 0)
{
v___x_501_ = v___x_498_;
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___x_498_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v___x_504_; 
if (v_isShared_502_ == 0)
{
v___x_504_ = v___x_501_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_a_499_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
}
v___jp_402_:
{
if (v___y_412_ == 0)
{
if (lean_obj_tag(v___y_408_) == 0)
{
lean_dec_ref_known(v___y_408_, 2);
lean_dec_ref(v___y_409_);
lean_dec(v_a_401_);
return v___y_406_;
}
else
{
lean_object* v_id_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_426_; 
v_id_413_ = lean_ctor_get(v___y_408_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___y_408_);
if (v_isSharedCheck_426_ == 0)
{
lean_object* v_unused_427_; 
v_unused_427_ = lean_ctor_get(v___y_408_, 1);
lean_dec(v_unused_427_);
v___x_415_ = v___y_408_;
v_isShared_416_ = v_isSharedCheck_426_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_id_413_);
lean_dec(v___y_408_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_426_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
uint8_t v___x_417_; 
v___x_417_ = l_Lean_instBEqInternalExceptionId_beq(v___y_405_, v_id_413_);
lean_dec(v_id_413_);
if (v___x_417_ == 0)
{
lean_del_object(v___x_415_);
lean_dec_ref(v___y_409_);
lean_dec(v_a_401_);
return v___y_406_;
}
else
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_422_; 
lean_dec_ref(v___y_406_);
v___x_418_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__6);
v___x_419_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__8);
v___x_420_ = l_Lean_indentExpr(v_a_401_);
if (v_isShared_416_ == 0)
{
lean_ctor_set_tag(v___x_415_, 7);
lean_ctor_set(v___x_415_, 1, v___x_420_);
lean_ctor_set(v___x_415_, 0, v___x_419_);
v___x_422_ = v___x_415_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v___x_419_);
lean_ctor_set(v_reuseFailAlloc_425_, 1, v___x_420_);
v___x_422_ = v_reuseFailAlloc_425_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_423_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
lean_ctor_set(v___x_423_, 1, v___x_418_);
v___x_424_ = lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_423_, v___y_403_, v___y_404_, v___y_411_, v___y_410_, v___y_409_, v___y_407_);
lean_dec_ref(v___y_409_);
return v___x_424_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_409_);
lean_dec_ref(v___y_408_);
lean_dec(v_a_401_);
return v___y_406_;
}
}
v___jp_428_:
{
lean_object* v___x_435_; 
lean_inc(v_a_401_);
v___x_435_ = lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr(v_a_401_, v___y_431_, v___y_432_, v___y_433_, v___y_434_);
if (lean_obj_tag(v___x_435_) == 0)
{
lean_dec_ref(v___y_433_);
lean_dec(v_a_401_);
return v___x_435_;
}
else
{
lean_object* v_a_436_; lean_object* v___x_437_; uint8_t v___x_438_; 
v_a_436_ = lean_ctor_get(v___x_435_, 0);
lean_inc(v_a_436_);
v___x_437_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_438_ = l_Lean_Exception_isInterrupt(v_a_436_);
if (v___x_438_ == 0)
{
uint8_t v___x_439_; 
lean_inc(v_a_436_);
v___x_439_ = l_Lean_Exception_isRuntime(v_a_436_);
v___y_403_ = v___y_429_;
v___y_404_ = v___y_430_;
v___y_405_ = v___x_437_;
v___y_406_ = v___x_435_;
v___y_407_ = v___y_434_;
v___y_408_ = v_a_436_;
v___y_409_ = v___y_433_;
v___y_410_ = v___y_432_;
v___y_411_ = v___y_431_;
v___y_412_ = v___x_439_;
goto v___jp_402_;
}
else
{
v___y_403_ = v___y_429_;
v___y_404_ = v___y_430_;
v___y_405_ = v___x_437_;
v___y_406_ = v___x_435_;
v___y_407_ = v___y_434_;
v___y_408_ = v_a_436_;
v___y_409_ = v___y_433_;
v___y_410_ = v___y_432_;
v___y_411_ = v___y_431_;
v___y_412_ = v___x_438_;
goto v___jp_402_;
}
}
}
v___jp_440_:
{
lean_object* v___x_447_; 
lean_inc(v_a_401_);
v___x_447_ = l_Lean_Meta_getMVars(v_a_401_, v___y_443_, v___y_444_, v___y_445_, v___y_446_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v_a_448_; lean_object* v___x_449_; 
v_a_448_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_448_);
lean_dec_ref_known(v___x_447_, 1);
v___x_449_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_448_, v___x_375_, v___y_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_);
lean_dec(v_a_448_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; uint8_t v___x_451_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_449_, 1);
v___x_451_ = lean_unbox(v_a_450_);
lean_dec(v_a_450_);
if (v___x_451_ == 0)
{
v___y_429_ = v___y_441_;
v___y_430_ = v___y_442_;
v___y_431_ = v___y_443_;
v___y_432_ = v___y_444_;
v___y_433_ = v___y_445_;
v___y_434_ = v___y_446_;
goto v___jp_428_;
}
else
{
lean_object* v___x_452_; lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
lean_dec_ref(v___y_445_);
lean_dec(v_a_401_);
v___x_452_ = lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_453_ = lean_ctor_get(v___x_452_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_452_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_452_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_a_453_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
else
{
lean_object* v_a_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_468_; 
lean_dec_ref(v___y_445_);
lean_dec(v_a_401_);
v_a_461_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_468_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_468_ == 0)
{
v___x_463_ = v___x_449_;
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_a_461_);
lean_dec(v___x_449_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_466_; 
if (v_isShared_464_ == 0)
{
v___x_466_ = v___x_463_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v_a_461_);
v___x_466_ = v_reuseFailAlloc_467_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
return v___x_466_;
}
}
}
}
else
{
lean_object* v_a_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_476_; 
lean_dec_ref(v___y_445_);
lean_dec(v_a_401_);
v_a_469_ = lean_ctor_get(v___x_447_, 0);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_447_);
if (v_isSharedCheck_476_ == 0)
{
v___x_471_ = v___x_447_;
v_isShared_472_ = v_isSharedCheck_476_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_a_469_);
lean_dec(v___x_447_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_476_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_474_; 
if (v_isShared_472_ == 0)
{
v___x_474_ = v___x_471_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_475_; 
v_reuseFailAlloc_475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_475_, 0, v_a_469_);
v___x_474_ = v_reuseFailAlloc_475_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
return v___x_474_;
}
}
}
}
v___jp_477_:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v_a_488_; lean_object* v___x_490_; uint8_t v_isShared_491_; uint8_t v_isSharedCheck_495_; 
v___x_484_ = lean_obj_once(&lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10, &lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10_once, _init_lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___closed__10);
v___x_485_ = l_Lean_indentExpr(v_a_401_);
v___x_486_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_484_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_486_, v___y_478_, v___y_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
lean_dec_ref(v___y_482_);
v_a_488_ = lean_ctor_get(v___x_487_, 0);
v_isSharedCheck_495_ = !lean_is_exclusive(v___x_487_);
if (v_isSharedCheck_495_ == 0)
{
v___x_490_ = v___x_487_;
v_isShared_491_ = v_isSharedCheck_495_;
goto v_resetjp_489_;
}
else
{
lean_inc(v_a_488_);
lean_dec(v___x_487_);
v___x_490_ = lean_box(0);
v_isShared_491_ = v_isSharedCheck_495_;
goto v_resetjp_489_;
}
v_resetjp_489_:
{
lean_object* v___x_493_; 
if (v_isShared_491_ == 0)
{
v___x_493_ = v___x_490_;
goto v_reusejp_492_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v_a_488_);
v___x_493_ = v_reuseFailAlloc_494_;
goto v_reusejp_492_;
}
v_reusejp_492_:
{
return v___x_493_;
}
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
lean_dec_ref_known(v___x_397_, 14);
v_a_507_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_398_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_398_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0(v_stx_515_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_, v_a_521_);
lean_dec(v_a_521_);
lean_dec_ref(v_a_520_);
lean_dec(v_a_519_);
lean_dec_ref(v_a_518_);
lean_dec(v_a_517_);
lean_dec_ref(v_a_516_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0(lean_object* v_config_541_, lean_object* v_item_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_){
_start:
{
lean_object* v_item_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_560_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5));
v___x_561_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_542_, v___x_560_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_561_) == 0)
{
uint8_t v___x_562_; 
lean_dec_ref_known(v___x_561_, 1);
v___x_562_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_542_);
if (v___x_562_ == 0)
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; uint8_t v___x_566_; 
v___x_563_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_542_);
lean_inc_ref(v_item_542_);
v___x_564_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_542_);
v___x_565_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__1));
v___x_566_ = lean_string_dec_eq(v___x_563_, v___x_565_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; uint8_t v___x_568_; 
v___x_567_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__2));
v___x_568_ = lean_string_dec_eq(v___x_563_, v___x_567_);
if (v___x_568_ == 0)
{
lean_object* v___x_569_; uint8_t v___x_570_; 
lean_dec_ref(v_config_541_);
v___x_569_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__3));
v___x_570_ = lean_string_dec_eq(v___x_563_, v___x_569_);
lean_dec_ref(v___x_563_);
if (v___x_570_ == 0)
{
lean_dec_ref(v_item_542_);
v_item_551_ = v___x_564_;
v___y_552_ = v___y_543_;
v___y_553_ = v___y_544_;
v___y_554_ = v___y_545_;
v___y_555_ = v___y_546_;
v___y_556_ = v___y_547_;
v___y_557_ = v___y_548_;
goto v___jp_550_;
}
else
{
uint8_t v___x_571_; 
v___x_571_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_564_);
if (v___x_571_ == 0)
{
lean_dec_ref(v_item_542_);
v_item_551_ = v___x_564_;
v___y_552_ = v___y_543_;
v___y_553_ = v___y_544_;
v___y_554_ = v___y_545_;
v___y_555_ = v___y_546_;
v___y_556_ = v___y_547_;
v___y_557_ = v___y_548_;
goto v___jp_550_;
}
else
{
lean_object* v_value_572_; lean_object* v___x_573_; 
lean_dec_ref(v___x_564_);
v_value_572_ = lean_ctor_get(v_item_542_, 2);
lean_inc(v_value_572_);
lean_dec_ref(v_item_542_);
v___x_573_ = lp_batteries_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0(v_value_572_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
return v___x_573_;
}
}
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; 
lean_dec_ref(v___x_563_);
v___x_574_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__4));
v___x_575_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_542_, v___x_574_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_575_) == 0)
{
uint8_t v___x_576_; 
lean_dec_ref_known(v___x_575_, 1);
v___x_576_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_564_);
if (v___x_576_ == 0)
{
lean_dec_ref(v_item_542_);
lean_dec_ref(v_config_541_);
v_item_551_ = v___x_564_;
v___y_552_ = v___y_543_;
v___y_553_ = v___y_544_;
v___y_554_ = v___y_545_;
v___y_555_ = v___y_546_;
v___y_556_ = v___y_547_;
v___y_557_ = v___y_548_;
goto v___jp_550_;
}
else
{
lean_object* v___x_577_; 
lean_dec_ref(v___x_564_);
v___x_577_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_577_) == 0)
{
lean_object* v_a_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_594_; 
v_a_578_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_594_ == 0)
{
v___x_580_ = v___x_577_;
v_isShared_581_ = v_isSharedCheck_594_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_a_578_);
lean_dec(v___x_577_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_594_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
uint8_t v_closePost_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_593_; 
v_closePost_582_ = lean_ctor_get_uint8(v_config_541_, 1);
v_isSharedCheck_593_ = !lean_is_exclusive(v_config_541_);
if (v_isSharedCheck_593_ == 0)
{
v___x_584_ = v_config_541_;
v_isShared_585_ = v_isSharedCheck_593_;
goto v_resetjp_583_;
}
else
{
lean_dec(v_config_541_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_593_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v___x_587_; 
if (v_isShared_585_ == 0)
{
v___x_587_ = v___x_584_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_592_; 
v_reuseFailAlloc_592_ = lean_alloc_ctor(0, 0, 2);
v___x_587_ = v_reuseFailAlloc_592_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
uint8_t v___x_588_; lean_object* v___x_590_; 
v___x_588_ = lean_unbox(v_a_578_);
lean_dec(v_a_578_);
lean_ctor_set_uint8(v___x_587_, 0, v___x_588_);
lean_ctor_set_uint8(v___x_587_, 1, v_closePost_582_);
if (v_isShared_581_ == 0)
{
lean_ctor_set(v___x_580_, 0, v___x_587_);
v___x_590_ = v___x_580_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v___x_587_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
}
else
{
lean_object* v_a_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_602_; 
lean_dec_ref(v_config_541_);
v_a_595_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_602_ == 0)
{
v___x_597_ = v___x_577_;
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_a_595_);
lean_dec(v___x_577_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_600_; 
if (v_isShared_598_ == 0)
{
v___x_600_ = v___x_597_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_a_595_);
v___x_600_ = v_reuseFailAlloc_601_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
return v___x_600_;
}
}
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_dec_ref(v___x_564_);
lean_dec_ref(v_item_542_);
lean_dec_ref(v_config_541_);
v_a_603_ = lean_ctor_get(v___x_575_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_575_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_575_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_575_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
}
else
{
lean_object* v___x_611_; lean_object* v___x_612_; 
lean_dec_ref(v___x_563_);
v___x_611_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__5));
v___x_612_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_542_, v___x_611_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_612_) == 0)
{
uint8_t v___x_613_; 
lean_dec_ref_known(v___x_612_, 1);
v___x_613_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_564_);
if (v___x_613_ == 0)
{
lean_dec_ref(v_item_542_);
lean_dec_ref(v_config_541_);
v_item_551_ = v___x_564_;
v___y_552_ = v___y_543_;
v___y_553_ = v___y_544_;
v___y_554_ = v___y_545_;
v___y_555_ = v___y_546_;
v___y_556_ = v___y_547_;
v___y_557_ = v___y_548_;
goto v___jp_550_;
}
else
{
lean_object* v___x_614_; 
lean_dec_ref(v___x_564_);
v___x_614_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_631_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_631_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_631_ == 0)
{
v___x_617_ = v___x_614_;
v_isShared_618_ = v_isSharedCheck_631_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_614_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_631_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
uint8_t v_closePre_619_; lean_object* v___x_621_; uint8_t v_isShared_622_; uint8_t v_isSharedCheck_630_; 
v_closePre_619_ = lean_ctor_get_uint8(v_config_541_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v_config_541_);
if (v_isSharedCheck_630_ == 0)
{
v___x_621_ = v_config_541_;
v_isShared_622_ = v_isSharedCheck_630_;
goto v_resetjp_620_;
}
else
{
lean_dec(v_config_541_);
v___x_621_ = lean_box(0);
v_isShared_622_ = v_isSharedCheck_630_;
goto v_resetjp_620_;
}
v_resetjp_620_:
{
lean_object* v___x_624_; 
if (v_isShared_622_ == 0)
{
v___x_624_ = v___x_621_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(0, 0, 2);
lean_ctor_set_uint8(v_reuseFailAlloc_629_, 0, v_closePre_619_);
v___x_624_ = v_reuseFailAlloc_629_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
uint8_t v___x_625_; lean_object* v___x_627_; 
v___x_625_ = lean_unbox(v_a_615_);
lean_dec(v_a_615_);
lean_ctor_set_uint8(v___x_624_, 1, v___x_625_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v___x_624_);
v___x_627_ = v___x_617_;
goto v_reusejp_626_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v___x_624_);
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
}
else
{
lean_object* v_a_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
lean_dec_ref(v_config_541_);
v_a_632_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_639_ == 0)
{
v___x_634_ = v___x_614_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_a_632_);
lean_dec(v___x_614_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_632_);
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
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_dec_ref(v___x_564_);
lean_dec_ref(v_item_542_);
lean_dec_ref(v_config_541_);
v_a_640_ = lean_ctor_get(v___x_612_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_612_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_612_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_612_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
}
}
else
{
lean_dec_ref(v_config_541_);
v_item_551_ = v_item_542_;
v___y_552_ = v___y_543_;
v___y_553_ = v___y_544_;
v___y_554_ = v___y_545_;
v___y_555_ = v___y_546_;
v___y_556_ = v___y_547_;
v___y_557_ = v___y_548_;
goto v___jp_550_;
}
}
else
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
lean_dec_ref(v_item_542_);
lean_dec_ref(v_config_541_);
v_a_648_ = lean_ctor_get(v___x_561_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_561_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_561_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_561_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
v___jp_550_:
{
lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_558_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___closed__0));
v___x_559_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_551_, v___x_558_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
return v___x_559_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_656_, lean_object* v_item_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___lam__0(v_config_656_, v_item_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, v___y_663_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0(lean_object* v_e_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_668_, v___y_672_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_e_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_batteries_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__0(v_e_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_);
lean_dec(v___y_683_);
lean_dec_ref(v___y_682_);
lean_dec(v___y_681_);
lean_dec_ref(v___y_680_);
lean_dec(v___y_679_);
lean_dec_ref(v___y_678_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2(lean_object* v_00_u03b1_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object* v_00_u03b1_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_batteries_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__2(v_00_u03b1_695_, v___y_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_, v___y_701_);
lean_dec(v___y_701_);
lean_dec_ref(v___y_700_);
lean_dec(v___y_699_);
lean_dec_ref(v___y_698_);
lean_dec(v___y_697_);
lean_dec_ref(v___y_696_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1(lean_object* v_00_u03b1_704_, lean_object* v_msg_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v___x_713_; 
v___x_713_ = lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_, v___y_711_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object* v_00_u03b1_714_, lean_object* v_msg_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_batteries_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1(v_00_u03b1_714_, v_msg_715_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
lean_dec(v___y_721_);
lean_dec_ref(v___y_720_);
lean_dec(v___y_719_);
lean_dec_ref(v___y_718_);
lean_dec(v___y_717_);
lean_dec_ref(v___y_716_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object* v_msgData_724_, lean_object* v_macroStack_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_724_, v_macroStack_725_, v___y_730_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object* v_msgData_734_, lean_object* v_macroStack_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(v_msgData_734_, v_macroStack_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
return v_res_743_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_744_ = lean_box(0);
v___x_745_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__5));
v___x_746_ = l_Lean_mkConst(v___x_745_, v___x_744_);
return v___x_746_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_747_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0, &lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0_once, _init_lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__0);
v___x_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_748_, 0, v___x_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0(lean_object* v_cfg_749_, lean_object* v_cfgItem_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_758_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1, &lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1_once, _init_lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___closed__1);
v___x_759_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_749_, v_cfgItem_750_, v___x_758_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0___boxed(lean_object* v_cfg_760_, lean_object* v_cfgItem_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___lam__0(v_cfg_760_, v_cfgItem_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
lean_dec(v___y_767_);
lean_dec_ref(v___y_766_);
lean_dec(v___y_765_);
lean_dec_ref(v___y_764_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
lean_dec(v_cfgItem_761_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(lean_object* v_cfg_771_, lean_object* v_init_772_, uint8_t v_logExceptions_773_, lean_object* v_a_774_, lean_object* v_a_775_, lean_object* v_a_776_){
_start:
{
lean_object* v_onErr_778_; lean_object* v_eval_779_; 
v_onErr_778_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___closed__0));
v_eval_779_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_Congr_elabConfig_evalConfigItem___closed__0));
if (v_logExceptions_773_ == 0)
{
lean_object* v___x_780_; 
v___x_780_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_779_, v_init_772_, v_cfg_771_, v_onErr_778_, v_logExceptions_773_, v_a_775_, v_a_776_);
return v___x_780_;
}
else
{
uint8_t v_recover_781_; lean_object* v___x_782_; 
v_recover_781_ = lean_ctor_get_uint8(v_a_774_, sizeof(void*)*1);
v___x_782_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_779_, v_init_772_, v_cfg_771_, v_onErr_778_, v_recover_781_, v_a_775_, v_a_776_);
return v___x_782_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg___boxed(lean_object* v_cfg_783_, lean_object* v_init_784_, lean_object* v_logExceptions_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_){
_start:
{
uint8_t v_logExceptions_boxed_790_; lean_object* v_res_791_; 
v_logExceptions_boxed_790_ = lean_unbox(v_logExceptions_785_);
v_res_791_ = lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(v_cfg_783_, v_init_784_, v_logExceptions_boxed_790_, v_a_786_, v_a_787_, v_a_788_);
lean_dec(v_a_788_);
lean_dec_ref(v_a_787_);
lean_dec_ref(v_a_786_);
return v_res_791_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig(lean_object* v_cfg_792_, lean_object* v_init_793_, uint8_t v_logExceptions_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(v_cfg_792_, v_init_793_, v_logExceptions_794_, v_a_795_, v_a_801_, v_a_802_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Congr_elabConfig___boxed(lean_object* v_cfg_805_, lean_object* v_init_806_, lean_object* v_logExceptions_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_, lean_object* v_a_813_, lean_object* v_a_814_, lean_object* v_a_815_, lean_object* v_a_816_){
_start:
{
uint8_t v_logExceptions_boxed_817_; lean_object* v_res_818_; 
v_logExceptions_boxed_817_ = lean_unbox(v_logExceptions_807_);
v_res_818_ = lp_batteries_Batteries_Tactic_Congr_elabConfig(v_cfg_805_, v_init_806_, v_logExceptions_boxed_817_, v_a_808_, v_a_809_, v_a_810_, v_a_811_, v_a_812_, v_a_813_, v_a_814_, v_a_815_);
lean_dec(v_a_815_);
lean_dec_ref(v_a_814_);
lean_dec(v_a_813_);
lean_dec_ref(v_a_812_);
lean_dec(v_a_811_);
lean_dec_ref(v_a_810_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
return v_res_818_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfig___closed__6(void){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_831_ = l_Lean_Parser_Tactic_config;
v___x_832_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__5));
v___x_833_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_834_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_834_, 0, v___x_833_);
lean_ctor_set(v___x_834_, 1, v___x_832_);
lean_ctor_set(v___x_834_, 2, v___x_831_);
return v___x_834_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfig___closed__17(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; 
v___x_855_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__16));
v___x_856_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfig___closed__6, &lp_batteries_Batteries_Tactic_congrConfig___closed__6_once, _init_lp_batteries_Batteries_Tactic_congrConfig___closed__6);
v___x_857_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_858_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_858_, 0, v___x_857_);
lean_ctor_set(v___x_858_, 1, v___x_856_);
lean_ctor_set(v___x_858_, 2, v___x_855_);
return v___x_858_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfig___closed__18(void){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_859_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfig___closed__17, &lp_batteries_Batteries_Tactic_congrConfig___closed__17_once, _init_lp_batteries_Batteries_Tactic_congrConfig___closed__17);
v___x_860_ = lean_unsigned_to_nat(1022u);
v___x_861_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__1));
v___x_862_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_862_, 0, v___x_861_);
lean_ctor_set(v___x_862_, 1, v___x_860_);
lean_ctor_set(v___x_862_, 2, v___x_859_);
return v___x_862_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfig(void){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfig___closed__18, &lp_batteries_Batteries_Tactic_congrConfig___closed__18_once, _init_lp_batteries_Batteries_Tactic_congrConfig___closed__18);
return v___x_863_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__2(void){
_start:
{
lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; 
v___x_869_ = l_Lean_Parser_Tactic_config;
v___x_870_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__8));
v___x_871_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
lean_ctor_set(v___x_871_, 1, v___x_869_);
return v___x_871_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__3(void){
_start:
{
lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; 
v___x_872_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__2, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__2_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__2);
v___x_873_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__5));
v___x_874_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_875_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_875_, 0, v___x_874_);
lean_ctor_set(v___x_875_, 1, v___x_873_);
lean_ctor_set(v___x_875_, 2, v___x_872_);
return v___x_875_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__10(void){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_892_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__9));
v___x_893_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__3, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__3_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__3);
v___x_894_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_895_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_895_, 0, v___x_894_);
lean_ctor_set(v___x_895_, 1, v___x_893_);
lean_ctor_set(v___x_895_, 2, v___x_892_);
return v___x_895_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__13(void){
_start:
{
lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_899_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__12));
v___x_900_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__10, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__10_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__10);
v___x_901_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_902_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_902_, 0, v___x_901_);
lean_ctor_set(v___x_902_, 1, v___x_900_);
lean_ctor_set(v___x_902_, 2, v___x_899_);
return v___x_902_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__21(void){
_start:
{
lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v___x_919_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__20));
v___x_920_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__13, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__13_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__13);
v___x_921_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_922_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_922_, 0, v___x_921_);
lean_ctor_set(v___x_922_, 1, v___x_920_);
lean_ctor_set(v___x_922_, 2, v___x_919_);
return v___x_922_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__26(void){
_start:
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; 
v___x_933_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__25));
v___x_934_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__21, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__21_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__21);
v___x_935_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_936_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_936_, 0, v___x_935_);
lean_ctor_set(v___x_936_, 1, v___x_934_);
lean_ctor_set(v___x_936_, 2, v___x_933_);
return v___x_936_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__27(void){
_start:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; 
v___x_937_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__26, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__26_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__26);
v___x_938_ = lean_unsigned_to_nat(1022u);
v___x_939_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__1));
v___x_940_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_940_, 0, v___x_939_);
lean_ctor_set(v___x_940_, 1, v___x_938_);
lean_ctor_set(v___x_940_, 2, v___x_937_);
return v___x_940_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_congrConfigWith(void){
_start:
{
lean_object* v___x_941_; 
v___x_941_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__27, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__27_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__27);
return v___x_941_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; 
v___x_942_ = lean_box(0);
v___x_943_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_944_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_944_, 0, v___x_943_);
lean_ctor_set(v___x_944_, 1, v___x_942_);
return v___x_944_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg(){
_start:
{
lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_946_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___closed__0);
v___x_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_947_, 0, v___x_946_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg___boxed(lean_object* v___y_948_){
_start:
{
lean_object* v_res_949_; 
v_res_949_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v_res_949_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0(lean_object* v_00_u03b1_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___boxed(lean_object* v_00_u03b1_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_){
_start:
{
lean_object* v_res_971_; 
v_res_971_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0(v_00_u03b1_961_, v___y_962_, v___y_963_, v___y_964_, v___y_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
lean_dec(v___y_965_);
lean_dec_ref(v___y_964_);
lean_dec(v___y_963_);
lean_dec_ref(v___y_962_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0(lean_object* v_a_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
lean_object* v___x_983_; 
v___x_983_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_975_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; uint8_t v_closePre_985_; uint8_t v_closePost_986_; lean_object* v___x_987_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v___x_983_, 1);
v_closePre_985_ = lean_ctor_get_uint8(v_a_972_, 0);
v_closePost_986_ = lean_ctor_get_uint8(v_a_972_, 1);
v___x_987_ = l_Lean_MVarId_congrN(v_a_984_, v___y_973_, v_closePre_985_, v_closePost_986_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; lean_object* v___x_989_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
lean_inc(v_a_988_);
lean_dec_ref_known(v___x_987_, 1);
v___x_989_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_988_, v___y_975_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v___x_991_; uint8_t v_isShared_992_; uint8_t v_isSharedCheck_997_; 
v_isSharedCheck_997_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_997_ == 0)
{
lean_object* v_unused_998_; 
v_unused_998_ = lean_ctor_get(v___x_989_, 0);
lean_dec(v_unused_998_);
v___x_991_ = v___x_989_;
v_isShared_992_ = v_isSharedCheck_997_;
goto v_resetjp_990_;
}
else
{
lean_dec(v___x_989_);
v___x_991_ = lean_box(0);
v_isShared_992_ = v_isSharedCheck_997_;
goto v_resetjp_990_;
}
v_resetjp_990_:
{
lean_object* v___x_993_; lean_object* v___x_995_; 
v___x_993_ = lean_box(0);
if (v_isShared_992_ == 0)
{
lean_ctor_set(v___x_991_, 0, v___x_993_);
v___x_995_ = v___x_991_;
goto v_reusejp_994_;
}
else
{
lean_object* v_reuseFailAlloc_996_; 
v_reuseFailAlloc_996_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_996_, 0, v___x_993_);
v___x_995_ = v_reuseFailAlloc_996_;
goto v_reusejp_994_;
}
v_reusejp_994_:
{
return v___x_995_;
}
}
}
else
{
return v___x_989_;
}
}
else
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
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
v_a_1007_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1014_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_1009_ = v___x_983_;
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_a_1007_);
lean_dec(v___x_983_);
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
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0___boxed(lean_object* v_a_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_){
_start:
{
lean_object* v_res_1026_; 
v_res_1026_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0(v_a_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_, v___y_1024_);
lean_dec(v___y_1024_);
lean_dec_ref(v___y_1023_);
lean_dec(v___y_1022_);
lean_dec_ref(v___y_1021_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
lean_dec(v___y_1018_);
lean_dec_ref(v___y_1017_);
lean_dec(v___y_1016_);
lean_dec_ref(v_a_1015_);
return v_res_1026_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1(lean_object* v_x_1034_, lean_object* v_a_1035_, lean_object* v_a_1036_, lean_object* v_a_1037_, lean_object* v_a_1038_, lean_object* v_a_1039_, lean_object* v_a_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_){
_start:
{
lean_object* v___y_1045_; lean_object* v___y_1046_; lean_object* v___y_1047_; lean_object* v___y_1048_; lean_object* v___y_1049_; lean_object* v___y_1050_; lean_object* v___y_1051_; lean_object* v___y_1052_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___x_1057_; uint8_t v___x_1058_; 
v___x_1057_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__1));
lean_inc(v_x_1034_);
v___x_1058_ = l_Lean_Syntax_isOfKind(v_x_1034_, v___x_1057_);
if (v___x_1058_ == 0)
{
lean_object* v___x_1059_; 
lean_dec(v_x_1034_);
v___x_1059_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v___x_1059_;
}
else
{
lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; uint8_t v___x_1063_; lean_object* v_n_x3f_1065_; lean_object* v___y_1066_; lean_object* v___y_1067_; lean_object* v___y_1068_; lean_object* v___y_1069_; lean_object* v___y_1070_; lean_object* v___y_1071_; lean_object* v___y_1072_; lean_object* v___y_1073_; 
v___x_1060_ = lean_unsigned_to_nat(1u);
v___x_1061_ = l_Lean_Syntax_getArg(v_x_1034_, v___x_1060_);
v___x_1062_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__2));
lean_inc(v___x_1061_);
v___x_1063_ = l_Lean_Syntax_isOfKind(v___x_1061_, v___x_1062_);
if (v___x_1063_ == 0)
{
lean_object* v___x_1091_; 
lean_dec(v___x_1061_);
lean_dec(v_x_1034_);
v___x_1091_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v___x_1091_;
}
else
{
lean_object* v___x_1092_; lean_object* v___x_1093_; uint8_t v___x_1094_; 
v___x_1092_ = lean_unsigned_to_nat(2u);
v___x_1093_ = l_Lean_Syntax_getArg(v_x_1034_, v___x_1092_);
lean_dec(v_x_1034_);
v___x_1094_ = l_Lean_Syntax_isNone(v___x_1093_);
if (v___x_1094_ == 0)
{
uint8_t v___x_1095_; 
lean_inc(v___x_1093_);
v___x_1095_ = l_Lean_Syntax_matchesNull(v___x_1093_, v___x_1060_);
if (v___x_1095_ == 0)
{
lean_object* v___x_1096_; 
lean_dec(v___x_1093_);
lean_dec(v___x_1061_);
v___x_1096_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v___x_1096_;
}
else
{
lean_object* v___x_1097_; lean_object* v_n_x3f_1098_; lean_object* v___x_1099_; 
v___x_1097_ = lean_unsigned_to_nat(0u);
v_n_x3f_1098_ = l_Lean_Syntax_getArg(v___x_1093_, v___x_1097_);
lean_dec(v___x_1093_);
v___x_1099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1099_, 0, v_n_x3f_1098_);
v_n_x3f_1065_ = v___x_1099_;
v___y_1066_ = v_a_1035_;
v___y_1067_ = v_a_1036_;
v___y_1068_ = v_a_1037_;
v___y_1069_ = v_a_1038_;
v___y_1070_ = v_a_1039_;
v___y_1071_ = v_a_1040_;
v___y_1072_ = v_a_1041_;
v___y_1073_ = v_a_1042_;
goto v___jp_1064_;
}
}
else
{
lean_object* v___x_1100_; 
lean_dec(v___x_1093_);
v___x_1100_ = lean_box(0);
v_n_x3f_1065_ = v___x_1100_;
v___y_1066_ = v_a_1035_;
v___y_1067_ = v_a_1036_;
v___y_1068_ = v_a_1037_;
v___y_1069_ = v_a_1038_;
v___y_1070_ = v_a_1039_;
v___y_1071_ = v_a_1040_;
v___y_1072_ = v_a_1041_;
v___y_1073_ = v_a_1042_;
goto v___jp_1064_;
}
}
v___jp_1064_:
{
lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1074_, 0, v___x_1061_);
v___x_1075_ = l_Lean_mkOptionalNode(v___x_1074_);
v___x_1076_ = lean_alloc_ctor(0, 0, 2);
lean_ctor_set_uint8(v___x_1076_, 0, v___x_1063_);
lean_ctor_set_uint8(v___x_1076_, 1, v___x_1063_);
v___x_1077_ = lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(v___x_1075_, v___x_1076_, v___x_1063_, v___y_1066_, v___y_1072_, v___y_1073_);
if (lean_obj_tag(v___x_1077_) == 0)
{
if (lean_obj_tag(v_n_x3f_1065_) == 0)
{
lean_object* v_a_1078_; lean_object* v___x_1079_; 
v_a_1078_ = lean_ctor_get(v___x_1077_, 0);
lean_inc(v_a_1078_);
lean_dec_ref_known(v___x_1077_, 1);
v___x_1079_ = lean_unsigned_to_nat(1000000u);
v___y_1045_ = v_a_1078_;
v___y_1046_ = v___y_1070_;
v___y_1047_ = v___y_1067_;
v___y_1048_ = v___y_1068_;
v___y_1049_ = v___y_1072_;
v___y_1050_ = v___y_1066_;
v___y_1051_ = v___y_1069_;
v___y_1052_ = v___y_1073_;
v___y_1053_ = v___y_1071_;
v___y_1054_ = v___x_1079_;
goto v___jp_1044_;
}
else
{
lean_object* v_a_1080_; lean_object* v_val_1081_; lean_object* v___x_1082_; 
v_a_1080_ = lean_ctor_get(v___x_1077_, 0);
lean_inc(v_a_1080_);
lean_dec_ref_known(v___x_1077_, 1);
v_val_1081_ = lean_ctor_get(v_n_x3f_1065_, 0);
lean_inc(v_val_1081_);
lean_dec_ref_known(v_n_x3f_1065_, 1);
v___x_1082_ = l_Lean_TSyntax_getNat(v_val_1081_);
lean_dec(v_val_1081_);
v___y_1045_ = v_a_1080_;
v___y_1046_ = v___y_1070_;
v___y_1047_ = v___y_1067_;
v___y_1048_ = v___y_1068_;
v___y_1049_ = v___y_1072_;
v___y_1050_ = v___y_1066_;
v___y_1051_ = v___y_1069_;
v___y_1052_ = v___y_1073_;
v___y_1053_ = v___y_1071_;
v___y_1054_ = v___x_1082_;
goto v___jp_1044_;
}
}
else
{
lean_object* v_a_1083_; lean_object* v___x_1085_; uint8_t v_isShared_1086_; uint8_t v_isSharedCheck_1090_; 
lean_dec(v_n_x3f_1065_);
v_a_1083_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1090_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1090_ == 0)
{
v___x_1085_ = v___x_1077_;
v_isShared_1086_ = v_isSharedCheck_1090_;
goto v_resetjp_1084_;
}
else
{
lean_inc(v_a_1083_);
lean_dec(v___x_1077_);
v___x_1085_ = lean_box(0);
v_isShared_1086_ = v_isSharedCheck_1090_;
goto v_resetjp_1084_;
}
v_resetjp_1084_:
{
lean_object* v___x_1088_; 
if (v_isShared_1086_ == 0)
{
v___x_1088_ = v___x_1085_;
goto v_reusejp_1087_;
}
else
{
lean_object* v_reuseFailAlloc_1089_; 
v_reuseFailAlloc_1089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1089_, 0, v_a_1083_);
v___x_1088_ = v_reuseFailAlloc_1089_;
goto v_reusejp_1087_;
}
v_reusejp_1087_:
{
return v___x_1088_;
}
}
}
}
}
v___jp_1044_:
{
lean_object* v___f_1055_; lean_object* v___x_1056_; 
v___f_1055_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_1055_, 0, v___y_1045_);
lean_closure_set(v___f_1055_, 1, v___y_1054_);
v___x_1056_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1055_, v___y_1050_, v___y_1047_, v___y_1048_, v___y_1051_, v___y_1046_, v___y_1053_, v___y_1049_, v___y_1052_);
return v___x_1056_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___boxed(lean_object* v_x_1101_, lean_object* v_a_1102_, lean_object* v_a_1103_, lean_object* v_a_1104_, lean_object* v_a_1105_, lean_object* v_a_1106_, lean_object* v_a_1107_, lean_object* v_a_1108_, lean_object* v_a_1109_, lean_object* v_a_1110_){
_start:
{
lean_object* v_res_1111_; 
v_res_1111_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1(v_x_1101_, v_a_1102_, v_a_1103_, v_a_1104_, v_a_1105_, v_a_1106_, v_a_1107_, v_a_1108_, v_a_1109_);
lean_dec(v_a_1109_);
lean_dec_ref(v_a_1108_);
lean_dec(v_a_1107_);
lean_dec_ref(v_a_1106_);
lean_dec(v_a_1105_);
lean_dec_ref(v_a_1104_);
lean_dec(v_a_1103_);
lean_dec_ref(v_a_1102_);
return v_res_1111_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11(void){
_start:
{
lean_object* v___x_1133_; 
v___x_1133_ = l_Array_mkArray0(lean_box(0));
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1(lean_object* v_x_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_){
_start:
{
lean_object* v___y_1138_; lean_object* v___y_1139_; lean_object* v___y_1140_; lean_object* v___y_1141_; lean_object* v___y_1142_; lean_object* v___y_1143_; lean_object* v___y_1144_; lean_object* v___y_1145_; lean_object* v___y_1146_; lean_object* v___y_1147_; lean_object* v___y_1148_; lean_object* v___y_1155_; lean_object* v___y_1156_; lean_object* v___y_1157_; lean_object* v___y_1158_; lean_object* v___y_1159_; lean_object* v___y_1160_; lean_object* v___y_1161_; lean_object* v___y_1162_; lean_object* v___y_1163_; lean_object* v___y_1164_; lean_object* v___y_1165_; lean_object* v___x_1171_; lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; lean_object* v___y_1182_; lean_object* v___y_1183_; lean_object* v___y_1202_; lean_object* v___y_1203_; lean_object* v___y_1204_; lean_object* v___y_1205_; lean_object* v___y_1206_; lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v_val_1212_; lean_object* v___y_1216_; lean_object* v___y_1217_; lean_object* v___y_1218_; lean_object* v___y_1219_; lean_object* v___y_1220_; lean_object* v___y_1221_; lean_object* v___y_1222_; lean_object* v___y_1223_; lean_object* v___y_1224_; lean_object* v___y_1225_; lean_object* v___y_1226_; lean_object* v___y_1227_; lean_object* v___x_1245_; uint8_t v___x_1246_; 
v___x_1171_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig_evalExpr___closed__2));
v___x_1245_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__1));
lean_inc(v_x_1134_);
v___x_1246_ = l_Lean_Syntax_isOfKind(v_x_1134_, v___x_1245_);
if (v___x_1246_ == 0)
{
lean_object* v___x_1247_; lean_object* v___x_1248_; 
lean_dec(v_x_1134_);
v___x_1247_ = lean_box(1);
v___x_1248_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
lean_ctor_set(v___x_1248_, 1, v_a_1136_);
return v___x_1248_;
}
else
{
lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___y_1252_; lean_object* v___y_1253_; lean_object* v___y_1254_; lean_object* v___y_1255_; lean_object* v___y_1256_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v_n_1291_; lean_object* v___y_1292_; lean_object* v___y_1293_; lean_object* v___x_1305_; lean_object* v___x_1306_; uint8_t v___x_1307_; 
v___x_1249_ = lean_unsigned_to_nat(1u);
v___x_1250_ = l_Lean_Syntax_getArg(v_x_1134_, v___x_1249_);
v___x_1286_ = lean_unsigned_to_nat(2u);
v___x_1287_ = l_Lean_Syntax_getArg(v_x_1134_, v___x_1286_);
v___x_1288_ = lean_unsigned_to_nat(4u);
v___x_1289_ = l_Lean_Syntax_getArg(v_x_1134_, v___x_1288_);
v___x_1305_ = lean_unsigned_to_nat(5u);
v___x_1306_ = l_Lean_Syntax_getArg(v_x_1134_, v___x_1305_);
lean_dec(v_x_1134_);
v___x_1307_ = l_Lean_Syntax_isNone(v___x_1306_);
if (v___x_1307_ == 0)
{
uint8_t v___x_1308_; 
lean_inc(v___x_1306_);
v___x_1308_ = l_Lean_Syntax_matchesNull(v___x_1306_, v___x_1286_);
if (v___x_1308_ == 0)
{
lean_object* v___x_1309_; lean_object* v___x_1310_; 
lean_dec(v___x_1306_);
lean_dec(v___x_1289_);
lean_dec(v___x_1287_);
lean_dec(v___x_1250_);
v___x_1309_ = lean_box(1);
v___x_1310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1310_, 0, v___x_1309_);
lean_ctor_set(v___x_1310_, 1, v_a_1136_);
return v___x_1310_;
}
else
{
lean_object* v_n_1311_; lean_object* v___x_1312_; 
v_n_1311_ = l_Lean_Syntax_getArg(v___x_1306_, v___x_1249_);
lean_dec(v___x_1306_);
v___x_1312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1312_, 0, v_n_1311_);
v_n_1291_ = v___x_1312_;
v___y_1292_ = v_a_1135_;
v___y_1293_ = v_a_1136_;
goto v___jp_1290_;
}
}
else
{
lean_object* v___x_1313_; 
lean_dec(v___x_1306_);
v___x_1313_ = lean_box(0);
v_n_1291_ = v___x_1313_;
v___y_1292_ = v_a_1135_;
v___y_1293_ = v_a_1136_;
goto v___jp_1290_;
}
v___jp_1251_:
{
lean_object* v___x_1257_; 
v___x_1257_ = l_Lean_Syntax_getOptional_x3f(v___x_1250_);
lean_dec(v___x_1250_);
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_object* v_ref_1258_; uint8_t v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; 
v_ref_1258_ = lean_ctor_get(v___y_1255_, 5);
v___x_1259_ = 0;
v___x_1260_ = l_Lean_SourceInfo_fromRef(v_ref_1258_, v___x_1259_);
v___x_1261_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0));
v___x_1262_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7));
v___x_1263_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__4));
v___x_1264_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__8));
lean_inc(v___x_1260_);
v___x_1265_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1265_, 0, v___x_1260_);
lean_ctor_set(v___x_1265_, 1, v___x_1263_);
v___x_1266_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__10));
v___x_1267_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11);
if (lean_obj_tag(v___y_1256_) == 0)
{
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_object* v___x_1268_; 
v___x_1268_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___y_1173_ = v___y_1252_;
v___y_1174_ = v___x_1265_;
v___y_1175_ = v___x_1266_;
v___y_1176_ = v___y_1253_;
v___y_1177_ = v___x_1262_;
v___y_1178_ = v___y_1254_;
v___y_1179_ = v___x_1264_;
v___y_1180_ = v___x_1267_;
v___y_1181_ = v___x_1260_;
v___y_1182_ = v___x_1261_;
v___y_1183_ = v___x_1268_;
goto v___jp_1172_;
}
else
{
lean_object* v_val_1269_; 
v_val_1269_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_val_1269_);
lean_dec_ref_known(v___x_1257_, 1);
v___y_1202_ = v___x_1265_;
v___y_1203_ = v___y_1252_;
v___y_1204_ = v___x_1266_;
v___y_1205_ = v___x_1262_;
v___y_1206_ = v___y_1253_;
v___y_1207_ = v___x_1264_;
v___y_1208_ = v___y_1254_;
v___y_1209_ = v___x_1267_;
v___y_1210_ = v___x_1260_;
v___y_1211_ = v___x_1261_;
v_val_1212_ = v_val_1269_;
goto v___jp_1201_;
}
}
else
{
lean_object* v_val_1270_; 
v_val_1270_ = lean_ctor_get(v___y_1256_, 0);
lean_inc(v_val_1270_);
lean_dec_ref_known(v___y_1256_, 1);
v___y_1202_ = v___x_1265_;
v___y_1203_ = v___y_1252_;
v___y_1204_ = v___x_1266_;
v___y_1205_ = v___x_1262_;
v___y_1206_ = v___y_1253_;
v___y_1207_ = v___x_1264_;
v___y_1208_ = v___y_1254_;
v___y_1209_ = v___x_1267_;
v___y_1210_ = v___x_1260_;
v___y_1211_ = v___x_1261_;
v_val_1212_ = v_val_1270_;
goto v___jp_1201_;
}
}
else
{
lean_object* v_val_1271_; lean_object* v_ref_1272_; uint8_t v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; 
v_val_1271_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_val_1271_);
lean_dec_ref_known(v___x_1257_, 1);
v_ref_1272_ = lean_ctor_get(v___y_1255_, 5);
v___x_1273_ = 0;
v___x_1274_ = l_Lean_SourceInfo_fromRef(v_ref_1272_, v___x_1273_);
v___x_1275_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1___closed__0));
v___x_1276_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__7));
v___x_1277_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__1));
v___x_1278_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__4));
lean_inc(v___x_1274_);
v___x_1279_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1274_);
lean_ctor_set(v___x_1279_, 1, v___x_1278_);
v___x_1280_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__10));
v___x_1281_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__11);
if (lean_obj_tag(v___y_1256_) == 0)
{
lean_object* v___x_1282_; 
v___x_1282_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___y_1216_ = v___y_1252_;
v___y_1217_ = v___x_1276_;
v___y_1218_ = v___x_1279_;
v___y_1219_ = v_val_1271_;
v___y_1220_ = v___x_1280_;
v___y_1221_ = v___x_1274_;
v___y_1222_ = v___x_1275_;
v___y_1223_ = v___y_1253_;
v___y_1224_ = v___x_1277_;
v___y_1225_ = v___y_1254_;
v___y_1226_ = v___x_1281_;
v___y_1227_ = v___x_1282_;
goto v___jp_1215_;
}
else
{
lean_object* v_val_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; 
v_val_1283_ = lean_ctor_get(v___y_1256_, 0);
lean_inc(v_val_1283_);
lean_dec_ref_known(v___y_1256_, 1);
v___x_1284_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___x_1285_ = lean_array_push(v___x_1284_, v_val_1283_);
v___y_1216_ = v___y_1252_;
v___y_1217_ = v___x_1276_;
v___y_1218_ = v___x_1279_;
v___y_1219_ = v_val_1271_;
v___y_1220_ = v___x_1280_;
v___y_1221_ = v___x_1274_;
v___y_1222_ = v___x_1275_;
v___y_1223_ = v___y_1253_;
v___y_1224_ = v___x_1277_;
v___y_1225_ = v___y_1254_;
v___y_1226_ = v___x_1281_;
v___y_1227_ = v___x_1285_;
goto v___jp_1215_;
}
}
}
v___jp_1290_:
{
lean_object* v_ps_1294_; lean_object* v___x_1295_; 
v_ps_1294_ = l_Lean_Syntax_getArgs(v___x_1289_);
lean_dec(v___x_1289_);
v___x_1295_ = l_Lean_Syntax_getOptional_x3f(v___x_1287_);
lean_dec(v___x_1287_);
if (lean_obj_tag(v___x_1295_) == 0)
{
lean_object* v___x_1296_; 
v___x_1296_ = lean_box(0);
v___y_1252_ = v___y_1293_;
v___y_1253_ = v_n_1291_;
v___y_1254_ = v_ps_1294_;
v___y_1255_ = v___y_1292_;
v___y_1256_ = v___x_1296_;
goto v___jp_1251_;
}
else
{
lean_object* v_val_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1304_; 
v_val_1297_ = lean_ctor_get(v___x_1295_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1295_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1299_ = v___x_1295_;
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_val_1297_);
lean_dec(v___x_1295_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
lean_object* v___x_1302_; 
if (v_isShared_1300_ == 0)
{
v___x_1302_ = v___x_1299_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_val_1297_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
v___y_1252_ = v___y_1293_;
v___y_1253_ = v_n_1291_;
v___y_1254_ = v_ps_1294_;
v___y_1255_ = v___y_1292_;
v___y_1256_ = v___x_1302_;
goto v___jp_1251_;
}
}
}
}
}
v___jp_1137_:
{
lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
lean_inc_ref(v___y_1143_);
v___x_1149_ = l_Array_append___redArg(v___y_1143_, v___y_1148_);
lean_dec_ref(v___y_1148_);
lean_inc(v___y_1139_);
lean_inc_n(v___y_1144_, 2);
v___x_1150_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1150_, 0, v___y_1144_);
lean_ctor_set(v___x_1150_, 1, v___y_1139_);
lean_ctor_set(v___x_1150_, 2, v___x_1149_);
v___x_1151_ = l_Lean_Syntax_node3(v___y_1144_, v___y_1146_, v___y_1140_, v___y_1142_, v___x_1150_);
lean_inc(v___y_1141_);
v___x_1152_ = l_Lean_Syntax_node3(v___y_1144_, v___y_1141_, v___y_1147_, v___y_1145_, v___x_1151_);
v___x_1153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1153_, 0, v___x_1152_);
lean_ctor_set(v___x_1153_, 1, v___y_1138_);
return v___x_1153_;
}
v___jp_1154_:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; 
lean_inc_ref(v___y_1164_);
v___x_1166_ = l_Array_append___redArg(v___y_1164_, v___y_1165_);
lean_dec_ref(v___y_1165_);
lean_inc(v___y_1160_);
lean_inc_n(v___y_1161_, 2);
v___x_1167_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1167_, 0, v___y_1161_);
lean_ctor_set(v___x_1167_, 1, v___y_1160_);
lean_ctor_set(v___x_1167_, 2, v___x_1166_);
v___x_1168_ = l_Lean_Syntax_node3(v___y_1161_, v___y_1162_, v___y_1157_, v___y_1159_, v___x_1167_);
lean_inc(v___y_1156_);
v___x_1169_ = l_Lean_Syntax_node3(v___y_1161_, v___y_1156_, v___y_1158_, v___y_1163_, v___x_1168_);
v___x_1170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1169_);
lean_ctor_set(v___x_1170_, 1, v___y_1155_);
return v___x_1170_;
}
v___jp_1172_:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; 
lean_inc_ref_n(v___y_1180_, 2);
v___x_1184_ = l_Array_append___redArg(v___y_1180_, v___y_1183_);
lean_dec_ref(v___y_1183_);
lean_inc_n(v___y_1175_, 2);
lean_inc_n(v___y_1181_, 5);
v___x_1185_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1185_, 0, v___y_1181_);
lean_ctor_set(v___x_1185_, 1, v___y_1175_);
lean_ctor_set(v___x_1185_, 2, v___x_1184_);
lean_inc(v___y_1179_);
v___x_1186_ = l_Lean_Syntax_node2(v___y_1181_, v___y_1179_, v___y_1174_, v___x_1185_);
v___x_1187_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__0));
v___x_1188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1188_, 0, v___y_1181_);
lean_ctor_set(v___x_1188_, 1, v___x_1187_);
v___x_1189_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__1));
v___x_1190_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__2));
v___x_1191_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__3));
lean_inc_ref(v___y_1182_);
v___x_1192_ = l_Lean_Name_mkStr5(v___y_1182_, v___x_1189_, v___x_1171_, v___x_1190_, v___x_1191_);
v___x_1193_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___y_1181_);
lean_ctor_set(v___x_1193_, 1, v___x_1191_);
v___x_1194_ = l_Array_append___redArg(v___y_1180_, v___y_1178_);
lean_dec_ref(v___y_1178_);
v___x_1195_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1195_, 0, v___y_1181_);
lean_ctor_set(v___x_1195_, 1, v___y_1175_);
lean_ctor_set(v___x_1195_, 2, v___x_1194_);
if (lean_obj_tag(v___y_1176_) == 1)
{
lean_object* v_val_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v_val_1196_ = lean_ctor_get(v___y_1176_, 0);
lean_inc(v_val_1196_);
lean_dec_ref_known(v___y_1176_, 1);
v___x_1197_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__4));
lean_inc(v___y_1181_);
v___x_1198_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1198_, 0, v___y_1181_);
lean_ctor_set(v___x_1198_, 1, v___x_1197_);
v___x_1199_ = l_Array_mkArray2___redArg(v___x_1198_, v_val_1196_);
v___y_1138_ = v___y_1173_;
v___y_1139_ = v___y_1175_;
v___y_1140_ = v___x_1193_;
v___y_1141_ = v___y_1177_;
v___y_1142_ = v___x_1195_;
v___y_1143_ = v___y_1180_;
v___y_1144_ = v___y_1181_;
v___y_1145_ = v___x_1188_;
v___y_1146_ = v___x_1192_;
v___y_1147_ = v___x_1186_;
v___y_1148_ = v___x_1199_;
goto v___jp_1137_;
}
else
{
lean_object* v___x_1200_; 
lean_dec(v___y_1176_);
v___x_1200_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___y_1138_ = v___y_1173_;
v___y_1139_ = v___y_1175_;
v___y_1140_ = v___x_1193_;
v___y_1141_ = v___y_1177_;
v___y_1142_ = v___x_1195_;
v___y_1143_ = v___y_1180_;
v___y_1144_ = v___y_1181_;
v___y_1145_ = v___x_1188_;
v___y_1146_ = v___x_1192_;
v___y_1147_ = v___x_1186_;
v___y_1148_ = v___x_1200_;
goto v___jp_1137_;
}
}
v___jp_1201_:
{
lean_object* v___x_1213_; lean_object* v___x_1214_; 
v___x_1213_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___x_1214_ = lean_array_push(v___x_1213_, v_val_1212_);
v___y_1173_ = v___y_1203_;
v___y_1174_ = v___y_1202_;
v___y_1175_ = v___y_1204_;
v___y_1176_ = v___y_1206_;
v___y_1177_ = v___y_1205_;
v___y_1178_ = v___y_1208_;
v___y_1179_ = v___y_1207_;
v___y_1180_ = v___y_1209_;
v___y_1181_ = v___y_1210_;
v___y_1182_ = v___y_1211_;
v___y_1183_ = v___x_1214_;
goto v___jp_1172_;
}
v___jp_1215_:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; 
lean_inc_ref_n(v___y_1226_, 2);
v___x_1228_ = l_Array_append___redArg(v___y_1226_, v___y_1227_);
lean_dec_ref(v___y_1227_);
lean_inc_n(v___y_1220_, 2);
lean_inc_n(v___y_1221_, 5);
v___x_1229_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1229_, 0, v___y_1221_);
lean_ctor_set(v___x_1229_, 1, v___y_1220_);
lean_ctor_set(v___x_1229_, 2, v___x_1228_);
lean_inc(v___y_1224_);
v___x_1230_ = l_Lean_Syntax_node3(v___y_1221_, v___y_1224_, v___y_1218_, v___y_1219_, v___x_1229_);
v___x_1231_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__0));
v___x_1232_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1232_, 0, v___y_1221_);
lean_ctor_set(v___x_1232_, 1, v___x_1231_);
v___x_1233_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__1));
v___x_1234_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__2));
v___x_1235_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__3));
lean_inc_ref(v___y_1222_);
v___x_1236_ = l_Lean_Name_mkStr5(v___y_1222_, v___x_1233_, v___x_1171_, v___x_1234_, v___x_1235_);
v___x_1237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1237_, 0, v___y_1221_);
lean_ctor_set(v___x_1237_, 1, v___x_1235_);
v___x_1238_ = l_Array_append___redArg(v___y_1226_, v___y_1225_);
lean_dec_ref(v___y_1225_);
v___x_1239_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1239_, 0, v___y_1221_);
lean_ctor_set(v___x_1239_, 1, v___y_1220_);
lean_ctor_set(v___x_1239_, 2, v___x_1238_);
if (lean_obj_tag(v___y_1223_) == 1)
{
lean_object* v_val_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; 
v_val_1240_ = lean_ctor_get(v___y_1223_, 0);
lean_inc(v_val_1240_);
lean_dec_ref_known(v___y_1223_, 1);
v___x_1241_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__4));
lean_inc(v___y_1221_);
v___x_1242_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1242_, 0, v___y_1221_);
lean_ctor_set(v___x_1242_, 1, v___x_1241_);
v___x_1243_ = l_Array_mkArray2___redArg(v___x_1242_, v_val_1240_);
v___y_1155_ = v___y_1216_;
v___y_1156_ = v___y_1217_;
v___y_1157_ = v___x_1237_;
v___y_1158_ = v___x_1230_;
v___y_1159_ = v___x_1239_;
v___y_1160_ = v___y_1220_;
v___y_1161_ = v___y_1221_;
v___y_1162_ = v___x_1236_;
v___y_1163_ = v___x_1232_;
v___y_1164_ = v___y_1226_;
v___y_1165_ = v___x_1243_;
goto v___jp_1154_;
}
else
{
lean_object* v___x_1244_; 
lean_dec(v___y_1223_);
v___x_1244_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___closed__5));
v___y_1155_ = v___y_1216_;
v___y_1156_ = v___y_1217_;
v___y_1157_ = v___x_1237_;
v___y_1158_ = v___x_1230_;
v___y_1159_ = v___x_1239_;
v___y_1160_ = v___y_1220_;
v___y_1161_ = v___y_1221_;
v___y_1162_ = v___x_1236_;
v___y_1163_ = v___x_1232_;
v___y_1164_ = v___y_1226_;
v___y_1165_ = v___x_1244_;
goto v___jp_1154_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1___boxed(lean_object* v_x_1314_, lean_object* v_a_1315_, lean_object* v_a_1316_){
_start:
{
lean_object* v_res_1317_; 
v_res_1317_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______macroRules__Batteries__Tactic__congrConfigWith__1(v_x_1314_, v_a_1315_, v_a_1316_);
lean_dec_ref(v_a_1315_);
return v_res_1317_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg(lean_object* v_fst_1318_, lean_object* v_x_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_){
_start:
{
if (lean_obj_tag(v_x_1319_) == 0)
{
uint8_t v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; 
lean_dec(v_fst_1318_);
v___x_1325_ = 0;
v___x_1326_ = lean_box(v___x_1325_);
v___x_1327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
return v___x_1327_;
}
else
{
lean_object* v_head_1328_; lean_object* v_tail_1329_; lean_object* v___x_1330_; 
v_head_1328_ = lean_ctor_get(v_x_1319_, 0);
lean_inc(v_head_1328_);
v_tail_1329_ = lean_ctor_get(v_x_1319_, 1);
lean_inc(v_tail_1329_);
lean_dec_ref_known(v_x_1319_, 2);
v___x_1330_ = l_Lean_MVarId_getType(v_head_1328_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_);
if (lean_obj_tag(v___x_1330_) == 0)
{
lean_object* v_a_1331_; lean_object* v___x_1332_; 
v_a_1331_ = lean_ctor_get(v___x_1330_, 0);
lean_inc(v_a_1331_);
lean_dec_ref_known(v___x_1330_, 1);
lean_inc(v_fst_1318_);
v___x_1332_ = l_Lean_MVarId_getType(v_fst_1318_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_);
if (lean_obj_tag(v___x_1332_) == 0)
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1343_; 
v_a_1333_ = lean_ctor_get(v___x_1332_, 0);
v_isSharedCheck_1343_ = !lean_is_exclusive(v___x_1332_);
if (v_isSharedCheck_1343_ == 0)
{
v___x_1335_ = v___x_1332_;
v_isShared_1336_ = v_isSharedCheck_1343_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1332_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1343_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
uint8_t v___x_1337_; 
v___x_1337_ = lean_expr_eqv(v_a_1331_, v_a_1333_);
lean_dec(v_a_1333_);
lean_dec(v_a_1331_);
if (v___x_1337_ == 0)
{
lean_del_object(v___x_1335_);
v_x_1319_ = v_tail_1329_;
goto _start;
}
else
{
lean_object* v___x_1339_; lean_object* v___x_1341_; 
lean_dec(v_tail_1329_);
lean_dec(v_fst_1318_);
v___x_1339_ = lean_box(v___x_1337_);
if (v_isShared_1336_ == 0)
{
lean_ctor_set(v___x_1335_, 0, v___x_1339_);
v___x_1341_ = v___x_1335_;
goto v_reusejp_1340_;
}
else
{
lean_object* v_reuseFailAlloc_1342_; 
v_reuseFailAlloc_1342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1342_, 0, v___x_1339_);
v___x_1341_ = v_reuseFailAlloc_1342_;
goto v_reusejp_1340_;
}
v_reusejp_1340_:
{
return v___x_1341_;
}
}
}
}
else
{
lean_object* v_a_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1351_; 
lean_dec(v_a_1331_);
lean_dec(v_tail_1329_);
lean_dec(v_fst_1318_);
v_a_1344_ = lean_ctor_get(v___x_1332_, 0);
v_isSharedCheck_1351_ = !lean_is_exclusive(v___x_1332_);
if (v_isSharedCheck_1351_ == 0)
{
v___x_1346_ = v___x_1332_;
v_isShared_1347_ = v_isSharedCheck_1351_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_a_1344_);
lean_dec(v___x_1332_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1351_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
lean_object* v___x_1349_; 
if (v_isShared_1347_ == 0)
{
v___x_1349_ = v___x_1346_;
goto v_reusejp_1348_;
}
else
{
lean_object* v_reuseFailAlloc_1350_; 
v_reuseFailAlloc_1350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1350_, 0, v_a_1344_);
v___x_1349_ = v_reuseFailAlloc_1350_;
goto v_reusejp_1348_;
}
v_reusejp_1348_:
{
return v___x_1349_;
}
}
}
}
else
{
lean_object* v_a_1352_; lean_object* v___x_1354_; uint8_t v_isShared_1355_; uint8_t v_isSharedCheck_1359_; 
lean_dec(v_tail_1329_);
lean_dec(v_fst_1318_);
v_a_1352_ = lean_ctor_get(v___x_1330_, 0);
v_isSharedCheck_1359_ = !lean_is_exclusive(v___x_1330_);
if (v_isSharedCheck_1359_ == 0)
{
v___x_1354_ = v___x_1330_;
v_isShared_1355_ = v_isSharedCheck_1359_;
goto v_resetjp_1353_;
}
else
{
lean_inc(v_a_1352_);
lean_dec(v___x_1330_);
v___x_1354_ = lean_box(0);
v_isShared_1355_ = v_isSharedCheck_1359_;
goto v_resetjp_1353_;
}
v_resetjp_1353_:
{
lean_object* v___x_1357_; 
if (v_isShared_1355_ == 0)
{
v___x_1357_ = v___x_1354_;
goto v_reusejp_1356_;
}
else
{
lean_object* v_reuseFailAlloc_1358_; 
v_reuseFailAlloc_1358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1358_, 0, v_a_1352_);
v___x_1357_ = v_reuseFailAlloc_1358_;
goto v_reusejp_1356_;
}
v_reusejp_1356_:
{
return v___x_1357_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg___boxed(lean_object* v_fst_1360_, lean_object* v_x_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_){
_start:
{
lean_object* v_res_1367_; 
v_res_1367_ = lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg(v_fst_1360_, v_x_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_);
lean_dec(v___y_1365_);
lean_dec_ref(v___y_1364_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
return v_res_1367_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg(lean_object* v_keys_1368_, lean_object* v_i_1369_, lean_object* v_k_1370_){
_start:
{
lean_object* v___x_1371_; uint8_t v___x_1372_; 
v___x_1371_ = lean_array_get_size(v_keys_1368_);
v___x_1372_ = lean_nat_dec_lt(v_i_1369_, v___x_1371_);
if (v___x_1372_ == 0)
{
lean_dec(v_i_1369_);
return v___x_1372_;
}
else
{
lean_object* v_k_x27_1373_; uint8_t v___x_1374_; 
v_k_x27_1373_ = lean_array_fget_borrowed(v_keys_1368_, v_i_1369_);
v___x_1374_ = l_Lean_instBEqMVarId_beq(v_k_1370_, v_k_x27_1373_);
if (v___x_1374_ == 0)
{
lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1375_ = lean_unsigned_to_nat(1u);
v___x_1376_ = lean_nat_add(v_i_1369_, v___x_1375_);
lean_dec(v_i_1369_);
v_i_1369_ = v___x_1376_;
goto _start;
}
else
{
lean_dec(v_i_1369_);
return v___x_1374_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_keys_1378_, lean_object* v_i_1379_, lean_object* v_k_1380_){
_start:
{
uint8_t v_res_1381_; lean_object* v_r_1382_; 
v_res_1381_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg(v_keys_1378_, v_i_1379_, v_k_1380_);
lean_dec(v_k_1380_);
lean_dec_ref(v_keys_1378_);
v_r_1382_ = lean_box(v_res_1381_);
return v_r_1382_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg(lean_object* v_x_1383_, size_t v_x_1384_, lean_object* v_x_1385_){
_start:
{
if (lean_obj_tag(v_x_1383_) == 0)
{
lean_object* v_es_1386_; lean_object* v___x_1387_; size_t v___x_1388_; size_t v___x_1389_; lean_object* v_j_1390_; lean_object* v___x_1391_; 
v_es_1386_ = lean_ctor_get(v_x_1383_, 0);
v___x_1387_ = lean_box(2);
v___x_1388_ = ((size_t)31ULL);
v___x_1389_ = lean_usize_land(v_x_1384_, v___x_1388_);
v_j_1390_ = lean_usize_to_nat(v___x_1389_);
v___x_1391_ = lean_array_get_borrowed(v___x_1387_, v_es_1386_, v_j_1390_);
lean_dec(v_j_1390_);
switch(lean_obj_tag(v___x_1391_))
{
case 0:
{
lean_object* v_key_1392_; uint8_t v___x_1393_; 
v_key_1392_ = lean_ctor_get(v___x_1391_, 0);
v___x_1393_ = l_Lean_instBEqMVarId_beq(v_x_1385_, v_key_1392_);
return v___x_1393_;
}
case 1:
{
lean_object* v_node_1394_; size_t v___x_1395_; size_t v___x_1396_; 
v_node_1394_ = lean_ctor_get(v___x_1391_, 0);
v___x_1395_ = ((size_t)5ULL);
v___x_1396_ = lean_usize_shift_right(v_x_1384_, v___x_1395_);
v_x_1383_ = v_node_1394_;
v_x_1384_ = v___x_1396_;
goto _start;
}
default: 
{
uint8_t v___x_1398_; 
v___x_1398_ = 0;
return v___x_1398_;
}
}
}
else
{
lean_object* v_ks_1399_; lean_object* v___x_1400_; uint8_t v___x_1401_; 
v_ks_1399_ = lean_ctor_get(v_x_1383_, 0);
v___x_1400_ = lean_unsigned_to_nat(0u);
v___x_1401_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg(v_ks_1399_, v___x_1400_, v_x_1385_);
return v___x_1401_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_x_1402_, lean_object* v_x_1403_, lean_object* v_x_1404_){
_start:
{
size_t v_x_4759__boxed_1405_; uint8_t v_res_1406_; lean_object* v_r_1407_; 
v_x_4759__boxed_1405_ = lean_unbox_usize(v_x_1403_);
lean_dec(v_x_1403_);
v_res_1406_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg(v_x_1402_, v_x_4759__boxed_1405_, v_x_1404_);
lean_dec(v_x_1404_);
lean_dec_ref(v_x_1402_);
v_r_1407_ = lean_box(v_res_1406_);
return v_r_1407_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg(lean_object* v_x_1408_, lean_object* v_x_1409_){
_start:
{
uint64_t v___x_1410_; size_t v___x_1411_; uint8_t v___x_1412_; 
v___x_1410_ = l_Lean_instHashableMVarId_hash(v_x_1409_);
v___x_1411_ = lean_uint64_to_usize(v___x_1410_);
v___x_1412_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg(v_x_1408_, v___x_1411_, v_x_1409_);
return v___x_1412_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg___boxed(lean_object* v_x_1413_, lean_object* v_x_1414_){
_start:
{
uint8_t v_res_1415_; lean_object* v_r_1416_; 
v_res_1415_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg(v_x_1413_, v_x_1414_);
lean_dec(v_x_1414_);
lean_dec_ref(v_x_1413_);
v_r_1416_ = lean_box(v_res_1415_);
return v_r_1416_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg(lean_object* v_mvarId_1417_, lean_object* v___y_1418_){
_start:
{
lean_object* v___x_1420_; lean_object* v_mctx_1421_; lean_object* v_eAssignment_1422_; uint8_t v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; 
v___x_1420_ = lean_st_ref_get(v___y_1418_);
v_mctx_1421_ = lean_ctor_get(v___x_1420_, 0);
lean_inc_ref(v_mctx_1421_);
lean_dec(v___x_1420_);
v_eAssignment_1422_ = lean_ctor_get(v_mctx_1421_, 8);
lean_inc_ref(v_eAssignment_1422_);
lean_dec_ref(v_mctx_1421_);
v___x_1423_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg(v_eAssignment_1422_, v_mvarId_1417_);
lean_dec_ref(v_eAssignment_1422_);
v___x_1424_ = lean_box(v___x_1423_);
v___x_1425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1424_);
return v___x_1425_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg___boxed(lean_object* v_mvarId_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_){
_start:
{
lean_object* v_res_1429_; 
v_res_1429_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg(v_mvarId_1426_, v___y_1427_);
lean_dec(v___y_1427_);
lean_dec(v_mvarId_1426_);
return v_res_1429_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3(lean_object* v_config_1430_, lean_object* v_as_1431_, size_t v_sz_1432_, size_t v_i_1433_, lean_object* v_b_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_){
_start:
{
lean_object* v_a_1443_; uint8_t v___x_1447_; 
v___x_1447_ = lean_usize_dec_lt(v_i_1433_, v_sz_1432_);
if (v___x_1447_ == 0)
{
lean_object* v___x_1448_; 
v___x_1448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1448_, 0, v_b_1434_);
return v___x_1448_;
}
else
{
lean_object* v_a_1449_; lean_object* v_fst_1450_; lean_object* v_snd_1451_; lean_object* v___x_1452_; 
v_a_1449_ = lean_array_uget_borrowed(v_as_1431_, v_i_1433_);
v_fst_1450_ = lean_ctor_get(v_a_1449_, 0);
v_snd_1451_ = lean_ctor_get(v_a_1449_, 1);
v___x_1452_ = l_Lean_Elab_Term_saveState___redArg(v___y_1436_, v___y_1438_, v___y_1440_);
if (lean_obj_tag(v___x_1452_) == 0)
{
lean_object* v_a_1453_; uint8_t v_closePre_1454_; uint8_t v_closePost_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; 
v_a_1453_ = lean_ctor_get(v___x_1452_, 0);
lean_inc(v_a_1453_);
lean_dec_ref_known(v___x_1452_, 1);
v_closePre_1454_ = lean_ctor_get_uint8(v_config_1430_, 0);
v_closePost_1455_ = lean_ctor_get_uint8(v_config_1430_, 1);
v___x_1456_ = lean_unsigned_to_nat(1000000u);
lean_inc(v_fst_1450_);
v___x_1457_ = l_Lean_MVarId_congrN(v_fst_1450_, v___x_1456_, v_closePre_1454_, v_closePost_1455_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
if (lean_obj_tag(v___x_1457_) == 0)
{
lean_object* v_a_1458_; uint8_t v___x_1459_; lean_object* v___x_1485_; 
v_a_1458_ = lean_ctor_get(v___x_1457_, 0);
lean_inc(v_a_1458_);
lean_dec_ref_known(v___x_1457_, 1);
v___x_1459_ = 0;
v___x_1485_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg(v_fst_1450_, v___y_1438_);
if (lean_obj_tag(v___x_1485_) == 0)
{
lean_object* v_a_1486_; uint8_t v___x_1487_; 
v_a_1486_ = lean_ctor_get(v___x_1485_, 0);
lean_inc(v_a_1486_);
lean_dec_ref_known(v___x_1485_, 1);
v___x_1487_ = lean_unbox(v_a_1486_);
lean_dec(v_a_1486_);
if (v___x_1487_ == 0)
{
lean_dec(v_a_1458_);
goto v___jp_1460_;
}
else
{
goto v___jp_1471_;
}
}
else
{
if (lean_obj_tag(v___x_1485_) == 0)
{
lean_object* v_a_1488_; uint8_t v___x_1489_; 
v_a_1488_ = lean_ctor_get(v___x_1485_, 0);
lean_inc(v_a_1488_);
lean_dec_ref_known(v___x_1485_, 1);
v___x_1489_ = lean_unbox(v_a_1488_);
lean_dec(v_a_1488_);
if (v___x_1489_ == 0)
{
goto v___jp_1471_;
}
else
{
lean_dec(v_a_1458_);
goto v___jp_1460_;
}
}
else
{
lean_object* v_a_1490_; lean_object* v___x_1492_; uint8_t v_isShared_1493_; uint8_t v_isSharedCheck_1497_; 
lean_dec(v_a_1458_);
lean_dec(v_a_1453_);
lean_dec_ref(v_b_1434_);
v_a_1490_ = lean_ctor_get(v___x_1485_, 0);
v_isSharedCheck_1497_ = !lean_is_exclusive(v___x_1485_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1492_ = v___x_1485_;
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
else
{
lean_inc(v_a_1490_);
lean_dec(v___x_1485_);
v___x_1492_ = lean_box(0);
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
v_resetjp_1491_:
{
lean_object* v___x_1495_; 
if (v_isShared_1493_ == 0)
{
v___x_1495_ = v___x_1492_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_a_1490_);
v___x_1495_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1494_;
}
v_reusejp_1494_:
{
return v___x_1495_;
}
}
}
}
v___jp_1460_:
{
lean_object* v___x_1461_; 
v___x_1461_ = l_Lean_Elab_Term_SavedState_restore(v_a_1453_, v___x_1459_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
if (lean_obj_tag(v___x_1461_) == 0)
{
lean_object* v___x_1462_; 
lean_dec_ref_known(v___x_1461_, 1);
lean_inc(v_fst_1450_);
v___x_1462_ = lean_array_push(v_b_1434_, v_fst_1450_);
v_a_1443_ = v___x_1462_;
goto v___jp_1442_;
}
else
{
lean_object* v_a_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1470_; 
lean_dec_ref(v_b_1434_);
v_a_1463_ = lean_ctor_get(v___x_1461_, 0);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1461_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1465_ = v___x_1461_;
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_a_1463_);
lean_dec(v___x_1461_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v___x_1468_; 
if (v_isShared_1466_ == 0)
{
v___x_1468_ = v___x_1465_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1469_; 
v_reuseFailAlloc_1469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1469_, 0, v_a_1463_);
v___x_1468_ = v_reuseFailAlloc_1469_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
return v___x_1468_;
}
}
}
}
v___jp_1471_:
{
lean_object* v___x_1472_; 
lean_inc(v_a_1458_);
lean_inc(v_fst_1450_);
v___x_1472_ = lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg(v_fst_1450_, v_a_1458_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
if (lean_obj_tag(v___x_1472_) == 0)
{
lean_object* v_a_1473_; uint8_t v___x_1474_; 
v_a_1473_ = lean_ctor_get(v___x_1472_, 0);
lean_inc(v_a_1473_);
lean_dec_ref_known(v___x_1472_, 1);
v___x_1474_ = lean_unbox(v_a_1473_);
lean_dec(v_a_1473_);
if (v___x_1474_ == 0)
{
lean_object* v___x_1475_; 
lean_dec(v_a_1453_);
lean_inc(v_snd_1451_);
v___x_1475_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg(v_config_1430_, v_snd_1451_, v_a_1458_, v_b_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
lean_dec(v_a_1458_);
if (lean_obj_tag(v___x_1475_) == 0)
{
lean_object* v_a_1476_; 
v_a_1476_ = lean_ctor_get(v___x_1475_, 0);
lean_inc(v_a_1476_);
lean_dec_ref_known(v___x_1475_, 1);
v_a_1443_ = v_a_1476_;
goto v___jp_1442_;
}
else
{
return v___x_1475_;
}
}
else
{
lean_dec(v_a_1458_);
goto v___jp_1460_;
}
}
else
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1484_; 
lean_dec(v_a_1458_);
lean_dec(v_a_1453_);
lean_dec_ref(v_b_1434_);
v_a_1477_ = lean_ctor_get(v___x_1472_, 0);
v_isSharedCheck_1484_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1479_ = v___x_1472_;
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1472_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1482_; 
if (v_isShared_1480_ == 0)
{
v___x_1482_ = v___x_1479_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v_a_1477_);
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
else
{
lean_object* v_a_1498_; lean_object* v___x_1500_; uint8_t v_isShared_1501_; uint8_t v_isSharedCheck_1505_; 
lean_dec(v_a_1453_);
lean_dec_ref(v_b_1434_);
v_a_1498_ = lean_ctor_get(v___x_1457_, 0);
v_isSharedCheck_1505_ = !lean_is_exclusive(v___x_1457_);
if (v_isSharedCheck_1505_ == 0)
{
v___x_1500_ = v___x_1457_;
v_isShared_1501_ = v_isSharedCheck_1505_;
goto v_resetjp_1499_;
}
else
{
lean_inc(v_a_1498_);
lean_dec(v___x_1457_);
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
else
{
lean_object* v_a_1506_; lean_object* v___x_1508_; uint8_t v_isShared_1509_; uint8_t v_isSharedCheck_1513_; 
lean_dec_ref(v_b_1434_);
v_a_1506_ = lean_ctor_get(v___x_1452_, 0);
v_isSharedCheck_1513_ = !lean_is_exclusive(v___x_1452_);
if (v_isSharedCheck_1513_ == 0)
{
v___x_1508_ = v___x_1452_;
v_isShared_1509_ = v_isSharedCheck_1513_;
goto v_resetjp_1507_;
}
else
{
lean_inc(v_a_1506_);
lean_dec(v___x_1452_);
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
v___jp_1442_:
{
size_t v___x_1444_; size_t v___x_1445_; 
v___x_1444_ = ((size_t)1ULL);
v___x_1445_ = lean_usize_add(v_i_1433_, v___x_1444_);
v_i_1433_ = v___x_1445_;
v_b_1434_ = v_a_1443_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_rcongrCore(lean_object* v_g_1514_, lean_object* v_config_1515_, lean_object* v_pats_1516_, lean_object* v_acc_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_){
_start:
{
lean_object* v___x_1525_; uint8_t v___x_1526_; lean_object* v___x_1527_; 
v___x_1525_ = lean_unsigned_to_nat(100u);
v___x_1526_ = 0;
v___x_1527_ = l_Lean_Elab_Tactic_Ext_extCore(v_g_1514_, v_pats_1516_, v___x_1525_, v___x_1526_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_);
if (lean_obj_tag(v___x_1527_) == 0)
{
lean_object* v_a_1528_; lean_object* v_snd_1529_; size_t v_sz_1530_; size_t v___x_1531_; lean_object* v___x_1532_; 
v_a_1528_ = lean_ctor_get(v___x_1527_, 0);
lean_inc(v_a_1528_);
lean_dec_ref_known(v___x_1527_, 1);
v_snd_1529_ = lean_ctor_get(v_a_1528_, 1);
lean_inc(v_snd_1529_);
lean_dec(v_a_1528_);
v_sz_1530_ = lean_array_size(v_snd_1529_);
v___x_1531_ = ((size_t)0ULL);
v___x_1532_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3(v_config_1515_, v_snd_1529_, v_sz_1530_, v___x_1531_, v_acc_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_);
lean_dec(v_snd_1529_);
return v___x_1532_;
}
else
{
lean_object* v_a_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1540_; 
lean_dec_ref(v_acc_1517_);
v_a_1533_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1540_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1540_ == 0)
{
v___x_1535_ = v___x_1527_;
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
else
{
lean_inc(v_a_1533_);
lean_dec(v___x_1527_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v___x_1538_; 
if (v_isShared_1536_ == 0)
{
v___x_1538_ = v___x_1535_;
goto v_reusejp_1537_;
}
else
{
lean_object* v_reuseFailAlloc_1539_; 
v_reuseFailAlloc_1539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1539_, 0, v_a_1533_);
v___x_1538_ = v_reuseFailAlloc_1539_;
goto v_reusejp_1537_;
}
v_reusejp_1537_:
{
return v___x_1538_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg(lean_object* v_config_1541_, lean_object* v_snd_1542_, lean_object* v_as_x27_1543_, lean_object* v_b_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_){
_start:
{
if (lean_obj_tag(v_as_x27_1543_) == 0)
{
lean_object* v___x_1552_; 
lean_dec(v_snd_1542_);
v___x_1552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1552_, 0, v_b_1544_);
return v___x_1552_;
}
else
{
lean_object* v_head_1553_; lean_object* v_tail_1554_; lean_object* v___x_1555_; 
v_head_1553_ = lean_ctor_get(v_as_x27_1543_, 0);
v_tail_1554_ = lean_ctor_get(v_as_x27_1543_, 1);
lean_inc(v_snd_1542_);
lean_inc(v_head_1553_);
v___x_1555_ = lp_batteries_Batteries_Tactic_rcongrCore(v_head_1553_, v_config_1541_, v_snd_1542_, v_b_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_);
if (lean_obj_tag(v___x_1555_) == 0)
{
lean_object* v_a_1556_; 
v_a_1556_ = lean_ctor_get(v___x_1555_, 0);
lean_inc(v_a_1556_);
lean_dec_ref_known(v___x_1555_, 1);
v_as_x27_1543_ = v_tail_1554_;
v_b_1544_ = v_a_1556_;
goto _start;
}
else
{
lean_dec(v_snd_1542_);
return v___x_1555_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg___boxed(lean_object* v_config_1558_, lean_object* v_snd_1559_, lean_object* v_as_x27_1560_, lean_object* v_b_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_){
_start:
{
lean_object* v_res_1569_; 
v_res_1569_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg(v_config_1558_, v_snd_1559_, v_as_x27_1560_, v_b_1561_, v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_);
lean_dec(v___y_1567_);
lean_dec_ref(v___y_1566_);
lean_dec(v___y_1565_);
lean_dec_ref(v___y_1564_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1562_);
lean_dec(v_as_x27_1560_);
lean_dec_ref(v_config_1558_);
return v_res_1569_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_rcongrCore___boxed(lean_object* v_g_1570_, lean_object* v_config_1571_, lean_object* v_pats_1572_, lean_object* v_acc_1573_, lean_object* v_a_1574_, lean_object* v_a_1575_, lean_object* v_a_1576_, lean_object* v_a_1577_, lean_object* v_a_1578_, lean_object* v_a_1579_, lean_object* v_a_1580_){
_start:
{
lean_object* v_res_1581_; 
v_res_1581_ = lp_batteries_Batteries_Tactic_rcongrCore(v_g_1570_, v_config_1571_, v_pats_1572_, v_acc_1573_, v_a_1574_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_, v_a_1579_);
lean_dec(v_a_1579_);
lean_dec_ref(v_a_1578_);
lean_dec(v_a_1577_);
lean_dec_ref(v_a_1576_);
lean_dec(v_a_1575_);
lean_dec_ref(v_a_1574_);
lean_dec_ref(v_config_1571_);
return v_res_1581_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3___boxed(lean_object* v_config_1582_, lean_object* v_as_1583_, lean_object* v_sz_1584_, lean_object* v_i_1585_, lean_object* v_b_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_, lean_object* v___y_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_){
_start:
{
size_t v_sz_boxed_1594_; size_t v_i_boxed_1595_; lean_object* v_res_1596_; 
v_sz_boxed_1594_ = lean_unbox_usize(v_sz_1584_);
lean_dec(v_sz_1584_);
v_i_boxed_1595_ = lean_unbox_usize(v_i_1585_);
lean_dec(v_i_1585_);
v_res_1596_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_rcongrCore_spec__3(v_config_1582_, v_as_1583_, v_sz_boxed_1594_, v_i_boxed_1595_, v_b_1586_, v___y_1587_, v___y_1588_, v___y_1589_, v___y_1590_, v___y_1591_, v___y_1592_);
lean_dec(v___y_1592_);
lean_dec_ref(v___y_1591_);
lean_dec(v___y_1590_);
lean_dec_ref(v___y_1589_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec_ref(v_as_1583_);
lean_dec_ref(v_config_1582_);
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0(lean_object* v_fst_1597_, lean_object* v_x_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_){
_start:
{
lean_object* v___x_1606_; 
v___x_1606_ = lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___redArg(v_fst_1597_, v_x_1598_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_);
return v___x_1606_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0___boxed(lean_object* v_fst_1607_, lean_object* v_x_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_){
_start:
{
lean_object* v_res_1616_; 
v_res_1616_ = lp_batteries_List_anyM___at___00Batteries_Tactic_rcongrCore_spec__0(v_fst_1607_, v_x_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
lean_dec(v___y_1614_);
lean_dec_ref(v___y_1613_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
return v_res_1616_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1(lean_object* v_config_1617_, lean_object* v_snd_1618_, lean_object* v_as_1619_, lean_object* v_as_x27_1620_, lean_object* v_b_1621_, lean_object* v_a_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_, lean_object* v___y_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_){
_start:
{
lean_object* v___x_1630_; 
v___x_1630_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___redArg(v_config_1617_, v_snd_1618_, v_as_x27_1620_, v_b_1621_, v___y_1623_, v___y_1624_, v___y_1625_, v___y_1626_, v___y_1627_, v___y_1628_);
return v___x_1630_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1___boxed(lean_object* v_config_1631_, lean_object* v_snd_1632_, lean_object* v_as_1633_, lean_object* v_as_x27_1634_, lean_object* v_b_1635_, lean_object* v_a_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_){
_start:
{
lean_object* v_res_1644_; 
v_res_1644_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_rcongrCore_spec__1(v_config_1631_, v_snd_1632_, v_as_1633_, v_as_x27_1634_, v_b_1635_, v_a_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_);
lean_dec(v___y_1642_);
lean_dec_ref(v___y_1641_);
lean_dec(v___y_1640_);
lean_dec_ref(v___y_1639_);
lean_dec(v___y_1638_);
lean_dec_ref(v___y_1637_);
lean_dec(v_as_x27_1634_);
lean_dec(v_as_1633_);
lean_dec_ref(v_config_1631_);
return v_res_1644_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2(lean_object* v_mvarId_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_){
_start:
{
lean_object* v___x_1653_; 
v___x_1653_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___redArg(v_mvarId_1645_, v___y_1649_);
return v___x_1653_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2___boxed(lean_object* v_mvarId_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_){
_start:
{
lean_object* v_res_1662_; 
v_res_1662_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2(v_mvarId_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_);
lean_dec(v___y_1660_);
lean_dec_ref(v___y_1659_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec(v___y_1656_);
lean_dec_ref(v___y_1655_);
lean_dec(v_mvarId_1654_);
return v_res_1662_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2(lean_object* v_00_u03b2_1663_, lean_object* v_x_1664_, lean_object* v_x_1665_){
_start:
{
uint8_t v___x_1666_; 
v___x_1666_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___redArg(v_x_1664_, v_x_1665_);
return v___x_1666_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2___boxed(lean_object* v_00_u03b2_1667_, lean_object* v_x_1668_, lean_object* v_x_1669_){
_start:
{
uint8_t v_res_1670_; lean_object* v_r_1671_; 
v_res_1670_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2(v_00_u03b2_1667_, v_x_1668_, v_x_1669_);
lean_dec(v_x_1669_);
lean_dec_ref(v_x_1668_);
v_r_1671_ = lean_box(v_res_1670_);
return v_r_1671_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3(lean_object* v_00_u03b2_1672_, lean_object* v_x_1673_, size_t v_x_1674_, lean_object* v_x_1675_){
_start:
{
uint8_t v___x_1676_; 
v___x_1676_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___redArg(v_x_1673_, v_x_1674_, v_x_1675_);
return v___x_1676_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3___boxed(lean_object* v_00_u03b2_1677_, lean_object* v_x_1678_, lean_object* v_x_1679_, lean_object* v_x_1680_){
_start:
{
size_t v_x_5144__boxed_1681_; uint8_t v_res_1682_; lean_object* v_r_1683_; 
v_x_5144__boxed_1681_ = lean_unbox_usize(v_x_1679_);
lean_dec(v_x_1679_);
v_res_1682_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3(v_00_u03b2_1677_, v_x_1678_, v_x_5144__boxed_1681_, v_x_1680_);
lean_dec(v_x_1680_);
lean_dec_ref(v_x_1678_);
v_r_1683_ = lean_box(v_res_1682_);
return v_r_1683_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_1684_, lean_object* v_keys_1685_, lean_object* v_vals_1686_, lean_object* v_heq_1687_, lean_object* v_i_1688_, lean_object* v_k_1689_){
_start:
{
uint8_t v___x_1690_; 
v___x_1690_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___redArg(v_keys_1685_, v_i_1688_, v_k_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1691_, lean_object* v_keys_1692_, lean_object* v_vals_1693_, lean_object* v_heq_1694_, lean_object* v_i_1695_, lean_object* v_k_1696_){
_start:
{
uint8_t v_res_1697_; lean_object* v_r_1698_; 
v_res_1697_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic_rcongrCore_spec__2_spec__2_spec__3_spec__5(v_00_u03b2_1691_, v_keys_1692_, v_vals_1693_, v_heq_1694_, v_i_1695_, v_k_1696_);
lean_dec(v_k_1696_);
lean_dec_ref(v_vals_1693_);
lean_dec_ref(v_keys_1692_);
v_r_1698_ = lean_box(v_res_1697_);
return v_r_1698_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_rcongr___closed__3(void){
_start:
{
lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; 
v___x_1707_ = lean_obj_once(&lp_batteries_Batteries_Tactic_congrConfigWith___closed__2, &lp_batteries_Batteries_Tactic_congrConfigWith___closed__2_once, _init_lp_batteries_Batteries_Tactic_congrConfigWith___closed__2);
v___x_1708_ = ((lean_object*)(lp_batteries_Batteries_Tactic_rcongr___closed__2));
v___x_1709_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_1710_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1710_, 0, v___x_1709_);
lean_ctor_set(v___x_1710_, 1, v___x_1708_);
lean_ctor_set(v___x_1710_, 2, v___x_1707_);
return v___x_1710_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_rcongr___closed__4(void){
_start:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; 
v___x_1711_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfigWith___closed__20));
v___x_1712_ = lean_obj_once(&lp_batteries_Batteries_Tactic_rcongr___closed__3, &lp_batteries_Batteries_Tactic_rcongr___closed__3_once, _init_lp_batteries_Batteries_Tactic_rcongr___closed__3);
v___x_1713_ = ((lean_object*)(lp_batteries_Batteries_Tactic_congrConfig___closed__3));
v___x_1714_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1714_, 0, v___x_1713_);
lean_ctor_set(v___x_1714_, 1, v___x_1712_);
lean_ctor_set(v___x_1714_, 2, v___x_1711_);
return v___x_1714_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_rcongr___closed__5(void){
_start:
{
lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1715_ = lean_obj_once(&lp_batteries_Batteries_Tactic_rcongr___closed__4, &lp_batteries_Batteries_Tactic_rcongr___closed__4_once, _init_lp_batteries_Batteries_Tactic_rcongr___closed__4);
v___x_1716_ = lean_unsigned_to_nat(1022u);
v___x_1717_ = ((lean_object*)(lp_batteries_Batteries_Tactic_rcongr___closed__1));
v___x_1718_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1718_, 0, v___x_1717_);
lean_ctor_set(v___x_1718_, 1, v___x_1716_);
lean_ctor_set(v___x_1718_, 2, v___x_1715_);
return v___x_1718_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_rcongr(void){
_start:
{
lean_object* v___x_1719_; 
v___x_1719_ = lean_obj_once(&lp_batteries_Batteries_Tactic_rcongr___closed__5, &lp_batteries_Batteries_Tactic_rcongr___closed__5_once, _init_lp_batteries_Batteries_Tactic_rcongr___closed__5);
return v___x_1719_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1(lean_object* v_x_1722_, lean_object* v_a_1723_, lean_object* v_a_1724_, lean_object* v_a_1725_, lean_object* v_a_1726_, lean_object* v_a_1727_, lean_object* v_a_1728_, lean_object* v_a_1729_, lean_object* v_a_1730_){
_start:
{
lean_object* v___x_1732_; uint8_t v___x_1733_; 
v___x_1732_ = ((lean_object*)(lp_batteries_Batteries_Tactic_rcongr___closed__1));
lean_inc(v_x_1722_);
v___x_1733_ = l_Lean_Syntax_isOfKind(v_x_1722_, v___x_1732_);
if (v___x_1733_ == 0)
{
lean_object* v___x_1734_; 
lean_dec(v_x_1722_);
v___x_1734_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__congrConfig__1_spec__0___redArg();
return v___x_1734_;
}
else
{
lean_object* v___x_1735_; 
v___x_1735_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1724_, v_a_1727_, v_a_1728_, v_a_1729_, v_a_1730_);
if (lean_obj_tag(v___x_1735_) == 0)
{
lean_object* v_a_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; 
v_a_1736_ = lean_ctor_get(v___x_1735_, 0);
lean_inc(v_a_1736_);
lean_dec_ref_known(v___x_1735_, 1);
v___x_1737_ = lean_unsigned_to_nat(1u);
v___x_1738_ = l_Lean_Syntax_getArg(v_x_1722_, v___x_1737_);
v___x_1739_ = lean_alloc_ctor(0, 0, 2);
lean_ctor_set_uint8(v___x_1739_, 0, v___x_1733_);
lean_ctor_set_uint8(v___x_1739_, 1, v___x_1733_);
v___x_1740_ = lp_batteries_Batteries_Tactic_Congr_elabConfig___redArg(v___x_1738_, v___x_1739_, v___x_1733_, v_a_1723_, v_a_1729_, v_a_1730_);
if (lean_obj_tag(v___x_1740_) == 0)
{
lean_object* v_a_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v_ps_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; 
v_a_1741_ = lean_ctor_get(v___x_1740_, 0);
lean_inc(v_a_1741_);
lean_dec_ref_known(v___x_1740_, 1);
v___x_1742_ = lean_unsigned_to_nat(2u);
v___x_1743_ = l_Lean_Syntax_getArg(v_x_1722_, v___x_1742_);
lean_dec(v_x_1722_);
v_ps_1744_ = l_Lean_Syntax_getArgs(v___x_1743_);
lean_dec(v___x_1743_);
v___x_1745_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___closed__0));
v___x_1746_ = lean_box(0);
v___x_1747_ = l_Lean_Elab_Tactic_RCases_expandRIntroPats(v_ps_1744_, v___x_1745_, v___x_1746_);
lean_dec_ref(v_ps_1744_);
v___x_1748_ = lean_array_to_list(v___x_1747_);
v___x_1749_ = lp_batteries_Batteries_Tactic_rcongrCore(v_a_1736_, v_a_1741_, v___x_1748_, v___x_1745_, v_a_1725_, v_a_1726_, v_a_1727_, v_a_1728_, v_a_1729_, v_a_1730_);
lean_dec(v_a_1741_);
if (lean_obj_tag(v___x_1749_) == 0)
{
lean_object* v_a_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; 
v_a_1750_ = lean_ctor_get(v___x_1749_, 0);
lean_inc(v_a_1750_);
lean_dec_ref_known(v___x_1749_, 1);
v___x_1751_ = lean_array_to_list(v_a_1750_);
v___x_1752_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1751_, v_a_1724_, v_a_1727_, v_a_1728_, v_a_1729_, v_a_1730_);
return v___x_1752_;
}
else
{
lean_object* v_a_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1760_; 
v_a_1753_ = lean_ctor_get(v___x_1749_, 0);
v_isSharedCheck_1760_ = !lean_is_exclusive(v___x_1749_);
if (v_isSharedCheck_1760_ == 0)
{
v___x_1755_ = v___x_1749_;
v_isShared_1756_ = v_isSharedCheck_1760_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_a_1753_);
lean_dec(v___x_1749_);
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
else
{
lean_object* v_a_1761_; lean_object* v___x_1763_; uint8_t v_isShared_1764_; uint8_t v_isSharedCheck_1768_; 
lean_dec(v_a_1736_);
lean_dec(v_x_1722_);
v_a_1761_ = lean_ctor_get(v___x_1740_, 0);
v_isSharedCheck_1768_ = !lean_is_exclusive(v___x_1740_);
if (v_isSharedCheck_1768_ == 0)
{
v___x_1763_ = v___x_1740_;
v_isShared_1764_ = v_isSharedCheck_1768_;
goto v_resetjp_1762_;
}
else
{
lean_inc(v_a_1761_);
lean_dec(v___x_1740_);
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
else
{
lean_object* v_a_1769_; lean_object* v___x_1771_; uint8_t v_isShared_1772_; uint8_t v_isSharedCheck_1776_; 
lean_dec(v_x_1722_);
v_a_1769_ = lean_ctor_get(v___x_1735_, 0);
v_isSharedCheck_1776_ = !lean_is_exclusive(v___x_1735_);
if (v_isSharedCheck_1776_ == 0)
{
v___x_1771_ = v___x_1735_;
v_isShared_1772_ = v_isSharedCheck_1776_;
goto v_resetjp_1770_;
}
else
{
lean_inc(v_a_1769_);
lean_dec(v___x_1735_);
v___x_1771_ = lean_box(0);
v_isShared_1772_ = v_isSharedCheck_1776_;
goto v_resetjp_1770_;
}
v_resetjp_1770_:
{
lean_object* v___x_1774_; 
if (v_isShared_1772_ == 0)
{
v___x_1774_ = v___x_1771_;
goto v_reusejp_1773_;
}
else
{
lean_object* v_reuseFailAlloc_1775_; 
v_reuseFailAlloc_1775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1775_, 0, v_a_1769_);
v___x_1774_ = v_reuseFailAlloc_1775_;
goto v_reusejp_1773_;
}
v_reusejp_1773_:
{
return v___x_1774_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1___boxed(lean_object* v_x_1777_, lean_object* v_a_1778_, lean_object* v_a_1779_, lean_object* v_a_1780_, lean_object* v_a_1781_, lean_object* v_a_1782_, lean_object* v_a_1783_, lean_object* v_a_1784_, lean_object* v_a_1785_, lean_object* v_a_1786_){
_start:
{
lean_object* v_res_1787_; 
v_res_1787_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Congr______elabRules__Batteries__Tactic__rcongr__1(v_x_1777_, v_a_1778_, v_a_1779_, v_a_1780_, v_a_1781_, v_a_1782_, v_a_1783_, v_a_1784_, v_a_1785_);
lean_dec(v_a_1785_);
lean_dec_ref(v_a_1784_);
lean_dec(v_a_1783_);
lean_dec_ref(v_a_1782_);
lean_dec(v_a_1781_);
lean_dec_ref(v_a_1780_);
lean_dec(v_a_1779_);
lean_dec_ref(v_a_1778_);
return v_res_1787_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Congr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Congr(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Config(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Ext(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Congr(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Congr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig = _init_lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig();
lean_mark_persistent(lp_batteries___private_Batteries_Tactic_Congr_0__Batteries_Tactic_instEvalExprConfig);
lp_batteries_Batteries_Tactic_congrConfig = _init_lp_batteries_Batteries_Tactic_congrConfig();
lean_mark_persistent(lp_batteries_Batteries_Tactic_congrConfig);
lp_batteries_Batteries_Tactic_congrConfigWith = _init_lp_batteries_Batteries_Tactic_congrConfigWith();
lean_mark_persistent(lp_batteries_Batteries_Tactic_congrConfigWith);
lp_batteries_Batteries_Tactic_rcongr = _init_lp_batteries_Batteries_Tactic_rcongr();
lean_mark_persistent(lp_batteries_Batteries_Tactic_rcongr);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Congr(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Config(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Ext(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Congr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Congr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Congr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Congr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Congr(builtin);
}
#ifdef __cplusplus
}
#endif
