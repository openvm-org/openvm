// Lean compiler output
// Module: Mathlib.Tactic.Ring.RingNF
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Ring.Basic public import Mathlib.Tactic.TryThis public import Mathlib.Util.AtomM.Recurse public meta import Mathlib.Util.AtomM.Recurse
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_instEvalTermTransparencyMode_evalTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
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
lean_object* l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_withSimpleEvalStx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_Simp_mkDefaultMethodsCore(lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompute(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_isAtomOrDerivable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_expandLocation(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default;
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr(uint8_t, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq(lean_object*, lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq(uint8_t, uint8_t);
lean_object* l_Lean_Elab_Tactic_Conv_getLhs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_applySimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Mathlib.Tactic.RingNF.RingMode.SOP"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Mathlib.Tactic.RingNF.RingMode.raw"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_RingNF_instReprConfig_repr_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ifUnchanged"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mode"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "SOP"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "raw"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RingNF"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "RingMode"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__3_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___boxed, .m_arity = 13, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__3_value),LEAN_SCALAR_PTR_LITERAL(100, 240, 215, 129, 254, 178, 89, 255)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "error"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "silent"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "warning"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "BehaviorIfUnchanged"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___boxed, .m_arity = 12, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(242, 91, 122, 18, 194, 100, 66, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "AtomM"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Recurse"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(178, 60, 12, 156, 136, 63, 145, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(84, 56, 114, 227, 157, 71, 143, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(115, 34, 204, 144, 160, 78, 191, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(3, 54, 97, 25, 108, 186, 112, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "TransparencyMode"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(245, 50, 227, 172, 92, 117, 235, 109)}};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "red"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "zetaDelta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(178, 60, 12, 156, 136, 63, 145, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(84, 56, 114, 227, 157, 71, 143, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(115, 34, 204, 144, 160, 78, 191, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 39, 77, 200, 132, 102, 196, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(178, 60, 12, 156, 136, 63, 145, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(84, 56, 114, 227, 157, 71, 143, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(115, 34, 204, 144, 160, 78, 191, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(31, 134, 213, 220, 253, 220, 228, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(3, 54, 97, 25, 108, 186, 112, 237)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(184, 101, 221, 95, 253, 175, 55, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "contextual"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(3, 54, 97, 25, 108, 186, 112, 237)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(4, 164, 135, 162, 165, 125, 19, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(178, 60, 12, 156, 136, 63, 145, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(84, 56, 114, 227, 157, 71, 143, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(115, 34, 204, 144, 160, 78, 191, 235)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(77, 135, 35, 209, 204, 67, 139, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "add_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__6_value),LEAN_SCALAR_PTR_LITERAL(188, 217, 59, 250, 243, 223, 216, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 178, 196, 247, 70, 46, 81, 207)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "pow_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__10_value),LEAN_SCALAR_PTR_LITERAL(14, 233, 85, 229, 191, 195, 180, 155)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__12_value),LEAN_SCALAR_PTR_LITERAL(168, 48, 81, 124, 253, 23, 128, 19)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "add_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__14_value),LEAN_SCALAR_PTR_LITERAL(161, 136, 196, 227, 102, 232, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nat_rawCast_0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__21_value),LEAN_SCALAR_PTR_LITERAL(52, 112, 15, 243, 98, 121, 163, 232)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nat_rawCast_1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__23_value),LEAN_SCALAR_PTR_LITERAL(157, 20, 224, 236, 49, 175, 161, 201)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nat_rawCast_2"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__25_value),LEAN_SCALAR_PTR_LITERAL(249, 73, 71, 163, 70, 179, 9, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "int_rawCast_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__27_value),LEAN_SCALAR_PTR_LITERAL(130, 84, 174, 186, 254, 97, 60, 217)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nnrat_rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__29_value),LEAN_SCALAR_PTR_LITERAL(111, 140, 59, 254, 188, 169, 89, 4)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "rat_rawCast_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__31_value),LEAN_SCALAR_PTR_LITERAL(72, 137, 161, 89, 117, 118, 134, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "add_assoc_rev"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__33_value),LEAN_SCALAR_PTR_LITERAL(135, 208, 124, 244, 68, 210, 188, 14)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "mul_assoc_rev"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__35_value),LEAN_SCALAR_PTR_LITERAL(143, 133, 53, 90, 219, 182, 251, 204)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__36_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__32_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__38_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__30_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__28_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__26_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__42_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__43_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__44_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49;
static const lean_array_object lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__50_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(2, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(2, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ringNF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 58, 229, 242, 41, 102, 20, 168)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ring_nf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNF;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_RingNF_evalExpr___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticRing_nf!__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 23, 172, 43, 132, 39, 215, 40)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ring_nf!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21____;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2;
static const lean_array_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "ringNFConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 92, 8, 246, 89, 184, 194, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ring1NF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__0_value),LEAN_SCALAR_PTR_LITERAL(32, 222, 8, 155, 240, 149, 242, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ring1_nf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring1NF;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticRing1_nf!_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 161, 146, 145, 83, 28, 29, 216)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "ring1_nf!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21__;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing1__nf_x21____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing1__nf_x21____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__1_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "convRing_nf!_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 234, 169, 93, 255, 142, 239, 4)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21__;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing__nf_x21____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing__nf_x21____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 86, 114, 51, 188, 197, 85, 98)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ring = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ring1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(221, 141, 62, 226, 100, 80, 9, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticTry_this__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(40, 25, 74, 169, 214, 21, 160, 3)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "try_this"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 277, .m_capacity = 277, .m_length = 276, .m_data = "\"\\n\\nThe `ring` tactic failed to close the goal. Use `ring_nf` to obtain a normal form.\n  \\nNote that `ring` works primarily in *commutative* rings. \\\n  If you have a noncommutative ring, abelian group or module, consider using \\\n  `noncomm_ring`, `abel` or `module` instead.\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticRing!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 30, 68, 192, 235, 62, 193, 146)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ring!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticRing1!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 132, 48, 112, 76, 186, 197, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ring1!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 280, .m_capacity = 280, .m_length = 279, .m_data = "\"\\n\\nThe `ring!` tactic failed to close the goal. Use `ring_nf!` to obtain a normal form.\n  \\nNote that `ring!` works primarily in *commutative* rings. \\\n  If you have a noncommutative ring, abelian group or module, consider using \\\n  `noncomm_ring`, `abel` or `module` instead.\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ringConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(80, 232, 180, 110, 37, 240, 202, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_ringConv = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 113, 181, 219, 229, 158, 34, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(202, 81, 30, 13, 252, 23, 29, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "convSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(249, 35, 202, 76, 198, 168, 114, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "dischargeConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(108, 125, 66, 83, 245, 64, 99, 127)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "discharge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "convTry_this__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(0, 180, 235, 220, 39, 148, 50, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "convRing!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(91, 182, 220, 92, 192, 16, 72, 80)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing_x21__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx(uint8_t v_x_1_){
_start:
{
if (v_x_1_ == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
uint8_t v_x_boxed_5_; lean_object* v_res_6_; 
v_x_boxed_5_ = lean_unbox(v_x_4_);
v_res_6_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx(v_x_boxed_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___redArg(lean_object* v_k_7_){
_start:
{
lean_inc(v_k_7_);
return v_k_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___redArg___boxed(lean_object* v_k_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___redArg(v_k_8_);
lean_dec(v_k_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim(lean_object* v_motive_10_, lean_object* v_ctorIdx_11_, uint8_t v_t_12_, lean_object* v_h_13_, lean_object* v_k_14_){
_start:
{
lean_inc(v_k_14_);
return v_k_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim___boxed(lean_object* v_motive_15_, lean_object* v_ctorIdx_16_, lean_object* v_t_17_, lean_object* v_h_18_, lean_object* v_k_19_){
_start:
{
uint8_t v_t_boxed_20_; lean_object* v_res_21_; 
v_t_boxed_20_ = lean_unbox(v_t_17_);
v_res_21_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorElim(v_motive_15_, v_ctorIdx_16_, v_t_boxed_20_, v_h_18_, v_k_19_);
lean_dec(v_k_19_);
lean_dec(v_ctorIdx_16_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___redArg(lean_object* v_SOP_22_){
_start:
{
lean_inc(v_SOP_22_);
return v_SOP_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___redArg___boxed(lean_object* v_SOP_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___redArg(v_SOP_23_);
lean_dec(v_SOP_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim(lean_object* v_motive_25_, uint8_t v_t_26_, lean_object* v_h_27_, lean_object* v_SOP_28_){
_start:
{
lean_inc(v_SOP_28_);
return v_SOP_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim___boxed(lean_object* v_motive_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_SOP_32_){
_start:
{
uint8_t v_t_boxed_33_; lean_object* v_res_34_; 
v_t_boxed_33_ = lean_unbox(v_t_30_);
v_res_34_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_SOP_elim(v_motive_29_, v_t_boxed_33_, v_h_31_, v_SOP_32_);
lean_dec(v_SOP_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___redArg(lean_object* v_raw_35_){
_start:
{
lean_inc(v_raw_35_);
return v_raw_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___redArg___boxed(lean_object* v_raw_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___redArg(v_raw_36_);
lean_dec(v_raw_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim(lean_object* v_motive_38_, uint8_t v_t_39_, lean_object* v_h_40_, lean_object* v_raw_41_){
_start:
{
lean_inc(v_raw_41_);
return v_raw_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim___boxed(lean_object* v_motive_42_, lean_object* v_t_43_, lean_object* v_h_44_, lean_object* v_raw_45_){
_start:
{
uint8_t v_t_boxed_46_; lean_object* v_res_47_; 
v_t_boxed_46_ = lean_unbox(v_t_43_);
v_res_47_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_raw_elim(v_motive_42_, v_t_boxed_46_, v_h_44_, v_raw_45_);
lean_dec(v_raw_45_);
return v_res_47_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode_default(void){
_start:
{
uint8_t v___x_48_; 
v___x_48_ = 0;
return v___x_48_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode(void){
_start:
{
uint8_t v___x_49_; 
v___x_49_ = 0;
return v___x_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq(uint8_t v_x_50_, uint8_t v_y_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; uint8_t v___x_54_; 
v___x_52_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx(v_x_50_);
v___x_53_ = lp_mathlib_Mathlib_Tactic_RingNF_RingMode_ctorIdx(v_y_51_);
v___x_54_ = lean_nat_dec_eq(v___x_52_, v___x_53_);
lean_dec(v___x_53_);
lean_dec(v___x_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq___boxed(lean_object* v_x_55_, lean_object* v_y_56_){
_start:
{
uint8_t v_x_17__boxed_57_; uint8_t v_y_18__boxed_58_; uint8_t v_res_59_; lean_object* v_r_60_; 
v_x_17__boxed_57_ = lean_unbox(v_x_55_);
v_y_18__boxed_58_ = lean_unbox(v_y_56_);
v_res_59_ = lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq(v_x_17__boxed_57_, v_y_18__boxed_58_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lean_unsigned_to_nat(2u);
v___x_70_ = lean_nat_to_int(v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = lean_unsigned_to_nat(1u);
v___x_72_ = lean_nat_to_int(v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr(uint8_t v_x_73_, lean_object* v_prec_74_){
_start:
{
lean_object* v___y_76_; lean_object* v___y_83_; 
if (v_x_73_ == 0)
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = lean_unsigned_to_nat(1024u);
v___x_90_ = lean_nat_dec_le(v___x_89_, v_prec_74_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; 
v___x_91_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4);
v___y_76_ = v___x_91_;
goto v___jp_75_;
}
else
{
lean_object* v___x_92_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5);
v___y_76_ = v___x_92_;
goto v___jp_75_;
}
}
else
{
lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_93_ = lean_unsigned_to_nat(1024u);
v___x_94_ = lean_nat_dec_le(v___x_93_, v_prec_74_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; 
v___x_95_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__4);
v___y_83_ = v___x_95_;
goto v___jp_82_;
}
else
{
lean_object* v___x_96_; 
v___x_96_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__5);
v___y_83_ = v___x_96_;
goto v___jp_82_;
}
}
v___jp_75_:
{
lean_object* v___x_77_; lean_object* v___x_78_; uint8_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_77_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__1));
lean_inc(v___y_76_);
v___x_78_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_78_, 0, v___y_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v___x_79_ = 0;
v___x_80_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_80_, 0, v___x_78_);
lean_ctor_set_uint8(v___x_80_, sizeof(void*)*1, v___x_79_);
v___x_81_ = l_Repr_addAppParen(v___x_80_, v_prec_74_);
return v___x_81_;
}
v___jp_82_:
{
lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_84_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___closed__3));
lean_inc(v___y_83_);
v___x_85_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_85_, 0, v___y_83_);
lean_ctor_set(v___x_85_, 1, v___x_84_);
v___x_86_ = 0;
v___x_87_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_87_, 0, v___x_85_);
lean_ctor_set_uint8(v___x_87_, sizeof(void*)*1, v___x_86_);
v___x_88_ = l_Repr_addAppParen(v___x_87_, v_prec_74_);
return v___x_88_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr___boxed(lean_object* v_x_97_, lean_object* v_prec_98_){
_start:
{
uint8_t v_x_121__boxed_99_; lean_object* v_res_100_; 
v_x_121__boxed_99_ = lean_unbox(v_x_97_);
v_res_100_ = lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr(v_x_121__boxed_99_, v_prec_98_);
lean_dec(v_prec_98_);
return v_res_100_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0(void){
_start:
{
uint8_t v___x_103_; uint8_t v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_103_ = 0;
v___x_104_ = 2;
v___x_105_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default;
v___x_106_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set_uint8(v___x_106_, sizeof(void*)*1, v___x_104_);
lean_ctor_set_uint8(v___x_106_, sizeof(void*)*1 + 1, v___x_103_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default(void){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0, &lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default___closed__0);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig(void){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default;
return v___x_108_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq(lean_object* v_x_109_, lean_object* v_x_110_){
_start:
{
lean_object* v_toConfig_111_; uint8_t v_ifUnchanged_112_; uint8_t v_mode_113_; lean_object* v_toConfig_114_; uint8_t v_ifUnchanged_115_; uint8_t v_mode_116_; uint8_t v___x_117_; 
v_toConfig_111_ = lean_ctor_get(v_x_109_, 0);
v_ifUnchanged_112_ = lean_ctor_get_uint8(v_x_109_, sizeof(void*)*1);
v_mode_113_ = lean_ctor_get_uint8(v_x_109_, sizeof(void*)*1 + 1);
v_toConfig_114_ = lean_ctor_get(v_x_110_, 0);
v_ifUnchanged_115_ = lean_ctor_get_uint8(v_x_110_, sizeof(void*)*1);
v_mode_116_ = lean_ctor_get_uint8(v_x_110_, sizeof(void*)*1 + 1);
v___x_117_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq(v_toConfig_111_, v_toConfig_114_);
if (v___x_117_ == 0)
{
return v___x_117_;
}
else
{
uint8_t v___x_118_; 
v___x_118_ = lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq(v_ifUnchanged_112_, v_ifUnchanged_115_);
if (v___x_118_ == 0)
{
return v___x_118_;
}
else
{
uint8_t v___x_119_; 
v___x_119_ = lp_mathlib_Mathlib_Tactic_RingNF_instBEqRingMode_beq(v_mode_113_, v_mode_116_);
return v___x_119_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq___boxed(lean_object* v_x_120_, lean_object* v_x_121_){
_start:
{
uint8_t v_res_122_; lean_object* v_r_123_; 
v_res_122_ = lp_mathlib_Mathlib_Tactic_RingNF_instBEqConfig_beq(v_x_120_, v_x_121_);
lean_dec_ref(v_x_121_);
lean_dec_ref(v_x_120_);
v_r_123_ = lean_box(v_res_122_);
return v_r_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_RingNF_instReprConfig_repr_spec__0(lean_object* v_a_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_nat_to_int(v_a_126_);
return v___x_127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_141_ = lean_unsigned_to_nat(12u);
v___x_142_ = lean_nat_to_int(v___x_141_);
return v___x_142_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lean_unsigned_to_nat(15u);
v___x_150_ = lean_nat_to_int(v___x_149_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_154_ = lean_unsigned_to_nat(8u);
v___x_155_ = lean_nat_to_int(v___x_154_);
return v___x_155_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__0));
v___x_158_ = lean_string_length(v___x_157_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__17);
v___x_160_ = lean_nat_to_int(v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg(lean_object* v_x_165_){
_start:
{
lean_object* v_toConfig_166_; uint8_t v_ifUnchanged_167_; uint8_t v_mode_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; uint8_t v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v_toConfig_166_ = lean_ctor_get(v_x_165_, 0);
v_ifUnchanged_167_ = lean_ctor_get_uint8(v_x_165_, sizeof(void*)*1);
v_mode_168_ = lean_ctor_get_uint8(v_x_165_, sizeof(void*)*1 + 1);
v___x_169_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__5));
v___x_170_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__6));
v___x_171_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__7);
v___x_172_ = lean_unsigned_to_nat(0u);
v___x_173_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(v_toConfig_166_);
v___x_174_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_171_);
lean_ctor_set(v___x_174_, 1, v___x_173_);
v___x_175_ = 0;
v___x_176_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_176_, 0, v___x_174_);
lean_ctor_set_uint8(v___x_176_, sizeof(void*)*1, v___x_175_);
v___x_177_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_170_);
lean_ctor_set(v___x_177_, 1, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__9));
v___x_179_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_177_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = lean_box(1);
v___x_181_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_179_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
v___x_182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__11));
v___x_183_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_181_);
lean_ctor_set(v___x_183_, 1, v___x_182_);
v___x_184_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_169_);
v___x_185_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__12);
v___x_186_ = lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr(v_ifUnchanged_167_, v___x_172_);
v___x_187_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_187_, 0, v___x_185_);
lean_ctor_set(v___x_187_, 1, v___x_186_);
v___x_188_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set_uint8(v___x_188_, sizeof(void*)*1, v___x_175_);
v___x_189_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_184_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
lean_ctor_set(v___x_190_, 1, v___x_178_);
v___x_191_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_180_);
v___x_192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__14));
v___x_193_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_191_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v___x_169_);
v___x_195_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__15);
v___x_196_ = lp_mathlib_Mathlib_Tactic_RingNF_instReprRingMode_repr(v_mode_168_, v___x_172_);
v___x_197_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_195_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
v___x_198_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set_uint8(v___x_198_, sizeof(void*)*1, v___x_175_);
v___x_199_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_194_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v___x_200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18, &lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__18);
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__19));
v___x_202_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v___x_199_);
v___x_203_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__20));
v___x_204_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_202_);
lean_ctor_set(v___x_204_, 1, v___x_203_);
v___x_205_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_200_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_206_, 0, v___x_205_);
lean_ctor_set_uint8(v___x_206_, sizeof(void*)*1, v___x_175_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___boxed(lean_object* v_x_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg(v_x_207_);
lean_dec_ref(v_x_207_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr(lean_object* v_x_209_, lean_object* v_prec_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg(v_x_209_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___boxed(lean_object* v_x_212_, lean_object* v_prec_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr(v_x_212_, v_prec_213_);
lean_dec(v_prec_213_);
lean_dec_ref(v_x_212_);
return v_res_214_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_217_ = lean_box(0);
v___x_218_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v___x_217_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg(){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_221_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0);
v___x_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___boxed(lean_object* v___y_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg();
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0(lean_object* v_00_u03b1_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg();
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___boxed(lean_object* v_00_u03b1_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0(v_00_u03b1_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0(lean_object* v___x_245_, lean_object* v___x_246_, lean_object* v___x_247_, lean_object* v___x_248_, lean_object* v_ctor_249_, lean_object* v_args_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
lean_object* v___x_258_; uint8_t v___x_259_; 
v___x_258_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__0));
v___x_259_ = lean_string_dec_eq(v_ctor_249_, v___x_258_);
if (v___x_259_ == 0)
{
lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_260_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__1));
v___x_261_ = lean_string_dec_eq(v_ctor_249_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; 
lean_dec_ref(v___x_248_);
lean_dec_ref(v___x_247_);
lean_dec_ref(v___x_246_);
lean_dec_ref(v___x_245_);
v___x_262_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg();
return v___x_262_;
}
else
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_263_ = l_Lean_Name_mkStr5(v___x_245_, v___x_246_, v___x_247_, v___x_248_, v___x_260_);
v___x_264_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_263_);
v___x_265_ = l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(v___x_263_, v___x_264_, v_args_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_);
if (lean_obj_tag(v___x_265_) == 0)
{
lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_277_; 
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_265_);
if (v_isSharedCheck_277_ == 0)
{
lean_object* v_unused_278_; 
v_unused_278_ = lean_ctor_get(v___x_265_, 0);
lean_dec(v_unused_278_);
v___x_267_ = v___x_265_;
v_isShared_268_ = v_isSharedCheck_277_;
goto v_resetjp_266_;
}
else
{
lean_dec(v___x_265_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_277_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
uint8_t v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_275_; 
v___x_269_ = 1;
v___x_270_ = lean_box(0);
v___x_271_ = l_Lean_Expr_const___override(v___x_263_, v___x_270_);
v___x_272_ = lean_box(v___x_269_);
v___x_273_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
lean_ctor_set(v___x_273_, 1, v___x_271_);
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 0, v___x_273_);
v___x_275_ = v___x_267_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v___x_273_);
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
lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_286_; 
lean_dec(v___x_263_);
v_a_279_ = lean_ctor_get(v___x_265_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_265_);
if (v_isSharedCheck_286_ == 0)
{
v___x_281_ = v___x_265_;
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_265_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_284_; 
if (v_isShared_282_ == 0)
{
v___x_284_ = v___x_281_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_a_279_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
}
}
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_287_ = l_Lean_Name_mkStr5(v___x_245_, v___x_246_, v___x_247_, v___x_248_, v___x_258_);
v___x_288_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_287_);
v___x_289_ = l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(v___x_287_, v___x_288_, v_args_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_);
if (lean_obj_tag(v___x_289_) == 0)
{
lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_301_; 
v_isSharedCheck_301_ = !lean_is_exclusive(v___x_289_);
if (v_isSharedCheck_301_ == 0)
{
lean_object* v_unused_302_; 
v_unused_302_ = lean_ctor_get(v___x_289_, 0);
lean_dec(v_unused_302_);
v___x_291_ = v___x_289_;
v_isShared_292_ = v_isSharedCheck_301_;
goto v_resetjp_290_;
}
else
{
lean_dec(v___x_289_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_301_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
uint8_t v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_299_; 
v___x_293_ = 0;
v___x_294_ = lean_box(0);
v___x_295_ = l_Lean_Expr_const___override(v___x_287_, v___x_294_);
v___x_296_ = lean_box(v___x_293_);
v___x_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v___x_295_);
if (v_isShared_292_ == 0)
{
lean_ctor_set(v___x_291_, 0, v___x_297_);
v___x_299_ = v___x_291_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_300_; 
v_reuseFailAlloc_300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_300_, 0, v___x_297_);
v___x_299_ = v_reuseFailAlloc_300_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
return v___x_299_;
}
}
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
lean_dec(v___x_287_);
v_a_303_ = lean_ctor_get(v___x_289_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_289_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_289_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_289_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___boxed(lean_object* v___x_311_, lean_object* v___x_312_, lean_object* v___x_313_, lean_object* v___x_314_, lean_object* v_ctor_315_, lean_object* v_args_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0(v___x_311_, v___x_312_, v___x_313_, v___x_314_, v_ctor_315_, v_args_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec_ref(v_args_316_);
lean_dec_ref(v_ctor_315_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm(lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_){
_start:
{
lean_object* v___f_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___f_347_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__4));
v___x_348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5));
v___x_349_ = l_Lean_Elab_ConfigEval_EvalTerm_withSimpleEvalStx___redArg(v___x_348_, v___f_347_, v_a_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_, v_a_344_, v_a_345_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___boxed(lean_object* v_a_350_, lean_object* v_a_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm(v_a_350_, v_a_351_, v_a_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_);
lean_dec(v_a_356_);
lean_dec_ref(v_a_355_);
lean_dec(v_a_354_);
lean_dec_ref(v_a_353_);
lean_dec(v_a_352_);
lean_dec_ref(v_a_351_);
return v_res_358_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
v___x_360_ = lean_box(0);
v___x_361_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5));
v___x_362_ = l_Lean_Expr_const___override(v___x_361_, v___x_360_);
return v___x_362_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1);
v___x_364_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__0));
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
lean_ctor_set(v___x_365_, 1, v___x_363_);
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode(void){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__2);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0(lean_object* v___x_370_, lean_object* v___x_371_, lean_object* v___x_372_, lean_object* v_ctor_373_, lean_object* v_args_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_){
_start:
{
lean_object* v___x_382_; uint8_t v___x_383_; 
v___x_382_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__0));
v___x_383_ = lean_string_dec_eq(v_ctor_373_, v___x_382_);
if (v___x_383_ == 0)
{
lean_object* v___x_384_; uint8_t v___x_385_; 
v___x_384_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__1));
v___x_385_ = lean_string_dec_eq(v_ctor_373_, v___x_384_);
if (v___x_385_ == 0)
{
lean_object* v___x_386_; uint8_t v___x_387_; 
v___x_386_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__2));
v___x_387_ = lean_string_dec_eq(v_ctor_373_, v___x_386_);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; 
lean_dec_ref(v___x_372_);
lean_dec_ref(v___x_371_);
lean_dec_ref(v___x_370_);
v___x_388_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg();
return v___x_388_;
}
else
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_389_ = l_Lean_Name_mkStr4(v___x_370_, v___x_371_, v___x_372_, v___x_386_);
v___x_390_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_389_);
v___x_391_ = l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(v___x_389_, v___x_390_, v_args_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
if (lean_obj_tag(v___x_391_) == 0)
{
lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_403_; 
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_403_ == 0)
{
lean_object* v_unused_404_; 
v_unused_404_ = lean_ctor_get(v___x_391_, 0);
lean_dec(v_unused_404_);
v___x_393_ = v___x_391_;
v_isShared_394_ = v_isSharedCheck_403_;
goto v_resetjp_392_;
}
else
{
lean_dec(v___x_391_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_403_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
uint8_t v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_401_; 
v___x_395_ = 1;
v___x_396_ = lean_box(0);
v___x_397_ = l_Lean_Expr_const___override(v___x_389_, v___x_396_);
v___x_398_ = lean_box(v___x_395_);
v___x_399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
lean_ctor_set(v___x_399_, 1, v___x_397_);
if (v_isShared_394_ == 0)
{
lean_ctor_set(v___x_393_, 0, v___x_399_);
v___x_401_ = v___x_393_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v___x_399_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v___x_389_);
v_a_405_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_391_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_391_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
}
else
{
lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_413_ = l_Lean_Name_mkStr4(v___x_370_, v___x_371_, v___x_372_, v___x_384_);
v___x_414_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_413_);
v___x_415_ = l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(v___x_413_, v___x_414_, v_args_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
if (lean_obj_tag(v___x_415_) == 0)
{
lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_427_; 
v_isSharedCheck_427_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_427_ == 0)
{
lean_object* v_unused_428_; 
v_unused_428_ = lean_ctor_get(v___x_415_, 0);
lean_dec(v_unused_428_);
v___x_417_ = v___x_415_;
v_isShared_418_ = v_isSharedCheck_427_;
goto v_resetjp_416_;
}
else
{
lean_dec(v___x_415_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_427_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
uint8_t v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_425_; 
v___x_419_ = 0;
v___x_420_ = lean_box(0);
v___x_421_ = l_Lean_Expr_const___override(v___x_413_, v___x_420_);
v___x_422_ = lean_box(v___x_419_);
v___x_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
lean_ctor_set(v___x_423_, 1, v___x_421_);
if (v_isShared_418_ == 0)
{
lean_ctor_set(v___x_417_, 0, v___x_423_);
v___x_425_ = v___x_417_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v___x_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
else
{
lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_436_; 
lean_dec(v___x_413_);
v_a_429_ = lean_ctor_get(v___x_415_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_436_ == 0)
{
v___x_431_ = v___x_415_;
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_415_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_434_; 
if (v_isShared_432_ == 0)
{
v___x_434_ = v___x_431_;
goto v_reusejp_433_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v_a_429_);
v___x_434_ = v_reuseFailAlloc_435_;
goto v_reusejp_433_;
}
v_reusejp_433_:
{
return v___x_434_;
}
}
}
}
}
else
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; 
v___x_437_ = l_Lean_Name_mkStr4(v___x_370_, v___x_371_, v___x_372_, v___x_382_);
v___x_438_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_437_);
v___x_439_ = l_Lean_Elab_ConfigEval_EvalTerm_checkExpectedNumberOfArguments(v___x_437_, v___x_438_, v_args_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
if (lean_obj_tag(v___x_439_) == 0)
{
lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_451_; 
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_439_);
if (v_isSharedCheck_451_ == 0)
{
lean_object* v_unused_452_; 
v_unused_452_ = lean_ctor_get(v___x_439_, 0);
lean_dec(v_unused_452_);
v___x_441_ = v___x_439_;
v_isShared_442_ = v_isSharedCheck_451_;
goto v_resetjp_440_;
}
else
{
lean_dec(v___x_439_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_451_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
uint8_t v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_449_; 
v___x_443_ = 2;
v___x_444_ = lean_box(0);
v___x_445_ = l_Lean_Expr_const___override(v___x_437_, v___x_444_);
v___x_446_ = lean_box(v___x_443_);
v___x_447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_447_, 0, v___x_446_);
lean_ctor_set(v___x_447_, 1, v___x_445_);
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 0, v___x_447_);
v___x_449_ = v___x_441_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_447_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
else
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
lean_dec(v___x_437_);
v_a_453_ = lean_ctor_get(v___x_439_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_439_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_439_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_439_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___boxed(lean_object* v___x_461_, lean_object* v___x_462_, lean_object* v___x_463_, lean_object* v_ctor_464_, lean_object* v_args_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0(v___x_461_, v___x_462_, v___x_463_, v_ctor_464_, v_args_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec(v___y_467_);
lean_dec_ref(v___y_466_);
lean_dec_ref(v_args_465_);
lean_dec_ref(v_ctor_464_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm(lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v___f_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v___f_491_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__1));
v___x_492_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2));
v___x_493_ = l_Lean_Elab_ConfigEval_EvalTerm_withSimpleEvalStx___redArg(v___x_492_, v___f_491_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___boxed(lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_){
_start:
{
lean_object* v_res_502_; 
v_res_502_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm(v_a_494_, v_a_495_, v_a_496_, v_a_497_, v_a_498_, v_a_499_, v_a_500_);
lean_dec(v_a_500_);
lean_dec_ref(v_a_499_);
lean_dec(v_a_498_);
lean_dec_ref(v_a_497_);
lean_dec(v_a_496_);
lean_dec_ref(v_a_495_);
return v_res_502_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1(void){
_start:
{
lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; 
v___x_504_ = lean_box(0);
v___x_505_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2));
v___x_506_ = l_Lean_Expr_const___override(v___x_505_, v___x_504_);
return v___x_506_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2(void){
_start:
{
lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_507_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1);
v___x_508_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__0));
v___x_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged(void){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__2);
return v___x_510_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_box(0);
v___x_512_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v___x_511_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_515_; lean_object* v___x_516_; 
v___x_515_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_516_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_);
lean_dec(v___y_530_);
lean_dec_ref(v___y_529_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object* v_msgData_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_){
_start:
{
lean_object* v___x_539_; lean_object* v_env_540_; lean_object* v___x_541_; lean_object* v_mctx_542_; lean_object* v_lctx_543_; lean_object* v_options_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_539_ = lean_st_ref_get(v___y_537_);
v_env_540_ = lean_ctor_get(v___x_539_, 0);
lean_inc_ref(v_env_540_);
lean_dec(v___x_539_);
v___x_541_ = lean_st_ref_get(v___y_535_);
v_mctx_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc_ref(v_mctx_542_);
lean_dec(v___x_541_);
v_lctx_543_ = lean_ctor_get(v___y_534_, 2);
v_options_544_ = lean_ctor_get(v___y_536_, 2);
lean_inc_ref(v_options_544_);
lean_inc_ref(v_lctx_543_);
v___x_545_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_545_, 0, v_env_540_);
lean_ctor_set(v___x_545_, 1, v_mctx_542_);
lean_ctor_set(v___x_545_, 2, v_lctx_543_);
lean_ctor_set(v___x_545_, 3, v_options_544_);
v___x_546_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
lean_ctor_set(v___x_546_, 1, v_msgData_533_);
v___x_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_547_, 0, v___x_546_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msgData_548_, v___y_549_, v___y_550_, v___y_551_, v___y_552_);
lean_dec(v___y_552_);
lean_dec_ref(v___y_551_);
lean_dec(v___y_550_);
lean_dec_ref(v___y_549_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
lean_object* v_ref_561_; lean_object* v___x_562_; lean_object* v_a_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_571_; 
v_ref_561_ = lean_ctor_get(v___y_558_, 5);
v___x_562_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_555_, v___y_556_, v___y_557_, v___y_558_, v___y_559_);
v_a_563_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_571_ == 0)
{
v___x_565_ = v___x_562_;
v_isShared_566_ = v_isSharedCheck_571_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_a_563_);
lean_dec(v___x_562_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_571_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_567_; lean_object* v___x_569_; 
lean_inc(v_ref_561_);
v___x_567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_567_, 0, v_ref_561_);
lean_ctor_set(v___x_567_, 1, v_a_563_);
if (v_isShared_566_ == 0)
{
lean_ctor_set_tag(v___x_565_, 1);
lean_ctor_set(v___x_565_, 0, v___x_567_);
v___x_569_ = v___x_565_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
lean_object* v_res_578_; 
v_res_578_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_572_, v___y_573_, v___y_574_, v___y_575_, v___y_576_);
lean_dec(v___y_576_);
lean_dec_ref(v___y_575_);
lean_dec(v___y_574_);
lean_dec_ref(v___y_573_);
return v_res_578_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_581_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_582_ = l_Lean_stringToMessageData(v___x_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_583_, lean_object* v_args_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
lean_object* v___x_639_; uint8_t v___x_640_; 
v___x_639_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_640_ = lean_string_dec_eq(v_ctor_583_, v___x_639_);
if (v___x_640_ == 0)
{
lean_object* v___x_641_; 
v___x_641_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_641_;
}
else
{
lean_object* v___x_642_; lean_object* v___x_643_; uint8_t v___x_644_; 
v___x_642_ = lean_array_get_size(v_args_584_);
v___x_643_ = lean_unsigned_to_nat(3u);
v___x_644_ = lean_nat_dec_eq(v___x_642_, v___x_643_);
if (v___x_644_ == 0)
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v_a_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_654_; 
v___x_645_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_646_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_645_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
v_a_647_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_654_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_654_ == 0)
{
v___x_649_ = v___x_646_;
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_a_647_);
lean_dec(v___x_646_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v___x_652_; 
if (v_isShared_650_ == 0)
{
v___x_652_ = v___x_649_;
goto v_reusejp_651_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v_a_647_);
v___x_652_ = v_reuseFailAlloc_653_;
goto v_reusejp_651_;
}
v_reusejp_651_:
{
return v___x_652_;
}
}
}
else
{
goto v___jp_590_;
}
}
v___jp_590_:
{
lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_591_ = l_Lean_instInhabitedExpr;
v___x_592_ = lean_unsigned_to_nat(0u);
v___x_593_ = lean_array_get_borrowed(v___x_591_, v_args_584_, v___x_592_);
lean_inc(v___x_593_);
v___x_594_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v___x_593_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_594_) == 0)
{
lean_object* v_a_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v_a_595_ = lean_ctor_get(v___x_594_, 0);
lean_inc(v_a_595_);
lean_dec_ref_known(v___x_594_, 1);
v___x_596_ = lean_unsigned_to_nat(1u);
v___x_597_ = lean_array_get_borrowed(v___x_591_, v_args_584_, v___x_596_);
lean_inc(v___x_597_);
v___x_598_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_597_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_598_) == 0)
{
lean_object* v_a_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v_a_599_ = lean_ctor_get(v___x_598_, 0);
lean_inc(v_a_599_);
lean_dec_ref_known(v___x_598_, 1);
v___x_600_ = lean_unsigned_to_nat(2u);
v___x_601_ = lean_array_get_borrowed(v___x_591_, v_args_584_, v___x_600_);
lean_inc(v___x_601_);
v___x_602_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_601_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_614_; 
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_614_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_614_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_614_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_607_; uint8_t v___x_608_; uint8_t v___x_609_; uint8_t v___x_610_; lean_object* v___x_612_; 
v___x_607_ = lean_alloc_ctor(0, 0, 3);
v___x_608_ = lean_unbox(v_a_595_);
lean_dec(v_a_595_);
lean_ctor_set_uint8(v___x_607_, 0, v___x_608_);
v___x_609_ = lean_unbox(v_a_599_);
lean_dec(v_a_599_);
lean_ctor_set_uint8(v___x_607_, 1, v___x_609_);
v___x_610_ = lean_unbox(v_a_603_);
lean_dec(v_a_603_);
lean_ctor_set_uint8(v___x_607_, 2, v___x_610_);
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v___x_607_);
v___x_612_ = v___x_605_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_607_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_a_599_);
lean_dec(v_a_595_);
v_a_615_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_602_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_602_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec(v_a_595_);
v_a_623_ = lean_ctor_get(v___x_598_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_598_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_598_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_598_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
else
{
lean_object* v_a_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_638_; 
v_a_631_ = lean_ctor_get(v___x_594_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_638_ == 0)
{
v___x_633_ = v___x_594_;
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_a_631_);
lean_dec(v___x_594_);
v___x_633_ = lean_box(0);
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
v_resetjp_632_:
{
lean_object* v___x_636_; 
if (v_isShared_634_ == 0)
{
v___x_636_ = v___x_633_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v_a_631_);
v___x_636_ = v_reuseFailAlloc_637_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
return v___x_636_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_655_, lean_object* v_args_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0(v_ctor_655_, v_args_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec_ref(v_args_656_);
lean_dec_ref(v_ctor_655_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr(lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_){
_start:
{
lean_object* v___f_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
v___f_679_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__0));
v___x_680_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4));
v___x_681_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_680_, v___f_679_, v_a_673_, v_a_674_, v_a_675_, v_a_676_, v_a_677_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr(v_a_682_, v_a_683_, v_a_684_, v_a_685_, v_a_686_);
lean_dec(v_a_686_);
lean_dec_ref(v_a_685_);
lean_dec(v_a_684_);
lean_dec_ref(v_a_683_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_689_, lean_object* v_msg_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_){
_start:
{
lean_object* v___x_696_; 
v___x_696_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_697_, lean_object* v_msg_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_697_, v_msg_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
return v_res_704_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; 
v___x_706_ = lean_box(0);
v___x_707_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___closed__4));
v___x_708_ = l_Lean_Expr_const___override(v___x_707_, v___x_706_);
return v___x_708_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_709_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__1);
v___x_710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; 
v___x_711_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__2);
v___x_712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__0));
v___x_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_713_, 0, v___x_712_);
lean_ctor_set(v___x_713_, 1, v___x_711_);
return v___x_713_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig(void){
_start:
{
lean_object* v___x_714_; 
v___x_714_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig___closed__3);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0(lean_object* v_ctor_715_, lean_object* v_args_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_){
_start:
{
lean_object* v___x_734_; uint8_t v___x_735_; 
v___x_734_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__0));
v___x_735_ = lean_string_dec_eq(v_ctor_715_, v___x_734_);
if (v___x_735_ == 0)
{
lean_object* v___x_736_; uint8_t v___x_737_; 
v___x_736_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__1));
v___x_737_ = lean_string_dec_eq(v_ctor_715_, v___x_736_);
if (v___x_737_ == 0)
{
lean_object* v___x_738_; uint8_t v___x_739_; 
v___x_738_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___lam__0___closed__2));
v___x_739_ = lean_string_dec_eq(v_ctor_715_, v___x_738_);
if (v___x_739_ == 0)
{
lean_object* v___x_740_; 
v___x_740_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_740_;
}
else
{
lean_object* v___x_741_; lean_object* v___x_742_; uint8_t v___x_743_; 
v___x_741_ = lean_array_get_size(v_args_716_);
v___x_742_ = lean_unsigned_to_nat(0u);
v___x_743_ = lean_nat_dec_eq(v___x_741_, v___x_742_);
if (v___x_743_ == 0)
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v_a_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_753_; 
v___x_744_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_745_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_744_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
v_a_746_ = lean_ctor_get(v___x_745_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_745_);
if (v_isSharedCheck_753_ == 0)
{
v___x_748_ = v___x_745_;
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_a_746_);
lean_dec(v___x_745_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_751_; 
if (v_isShared_749_ == 0)
{
v___x_751_ = v___x_748_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_a_746_);
v___x_751_ = v_reuseFailAlloc_752_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
return v___x_751_;
}
}
}
else
{
goto v___jp_722_;
}
}
}
else
{
lean_object* v___x_754_; lean_object* v___x_755_; uint8_t v___x_756_; 
v___x_754_ = lean_array_get_size(v_args_716_);
v___x_755_ = lean_unsigned_to_nat(0u);
v___x_756_ = lean_nat_dec_eq(v___x_754_, v___x_755_);
if (v___x_756_ == 0)
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v_a_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_766_; 
v___x_757_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_758_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_757_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
v_a_759_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_766_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_766_ == 0)
{
v___x_761_ = v___x_758_;
v_isShared_762_ = v_isSharedCheck_766_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_a_759_);
lean_dec(v___x_758_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_766_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v___x_764_; 
if (v_isShared_762_ == 0)
{
v___x_764_ = v___x_761_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_a_759_);
v___x_764_ = v_reuseFailAlloc_765_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
return v___x_764_;
}
}
}
else
{
goto v___jp_726_;
}
}
}
else
{
lean_object* v___x_767_; lean_object* v___x_768_; uint8_t v___x_769_; 
v___x_767_ = lean_array_get_size(v_args_716_);
v___x_768_ = lean_unsigned_to_nat(0u);
v___x_769_ = lean_nat_dec_eq(v___x_767_, v___x_768_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v_a_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_779_; 
v___x_770_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_771_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_770_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
v_a_772_ = lean_ctor_get(v___x_771_, 0);
v_isSharedCheck_779_ = !lean_is_exclusive(v___x_771_);
if (v_isSharedCheck_779_ == 0)
{
v___x_774_ = v___x_771_;
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_a_772_);
lean_dec(v___x_771_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
lean_object* v___x_777_; 
if (v_isShared_775_ == 0)
{
v___x_777_ = v___x_774_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v_a_772_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
}
else
{
goto v___jp_730_;
}
}
v___jp_722_:
{
uint8_t v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_723_ = 1;
v___x_724_ = lean_box(v___x_723_);
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
v___jp_726_:
{
uint8_t v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_727_ = 0;
v___x_728_ = lean_box(v___x_727_);
v___x_729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_729_, 0, v___x_728_);
return v___x_729_;
}
v___jp_730_:
{
uint8_t v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_731_ = 2;
v___x_732_ = lean_box(v___x_731_);
v___x_733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_733_, 0, v___x_732_);
return v___x_733_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0___boxed(lean_object* v_ctor_780_, lean_object* v_args_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___lam__0(v_ctor_780_, v_args_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
lean_dec(v___y_783_);
lean_dec_ref(v___y_782_);
lean_dec_ref(v_args_781_);
lean_dec_ref(v_ctor_780_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr(lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_){
_start:
{
lean_object* v___f_795_; lean_object* v___x_796_; lean_object* v___x_797_; 
v___f_795_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___closed__0));
v___x_796_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm___closed__2));
v___x_797_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_796_, v___f_795_, v_a_789_, v_a_790_, v_a_791_, v_a_792_, v_a_793_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr___boxed(lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_){
_start:
{
lean_object* v_res_804_; 
v_res_804_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr(v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_);
lean_dec(v_a_802_);
lean_dec_ref(v_a_801_);
lean_dec(v_a_800_);
lean_dec_ref(v_a_799_);
return v_res_804_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1(void){
_start:
{
lean_object* v___x_806_; lean_object* v___x_807_; 
v___x_806_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1);
v___x_807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_807_, 0, v___x_806_);
return v___x_807_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2(void){
_start:
{
lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; 
v___x_808_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1);
v___x_809_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__0));
v___x_810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_810_, 0, v___x_809_);
lean_ctor_set(v___x_810_, 1, v___x_808_);
return v___x_810_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged(void){
_start:
{
lean_object* v___x_811_; 
v___x_811_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__2);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0(lean_object* v_ctor_812_, lean_object* v_args_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
lean_object* v___x_827_; uint8_t v___x_828_; 
v___x_827_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__0));
v___x_828_ = lean_string_dec_eq(v_ctor_812_, v___x_827_);
if (v___x_828_ == 0)
{
lean_object* v___x_829_; uint8_t v___x_830_; 
v___x_829_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___lam__0___closed__1));
v___x_830_ = lean_string_dec_eq(v_ctor_812_, v___x_829_);
if (v___x_830_ == 0)
{
lean_object* v___x_831_; 
v___x_831_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_831_;
}
else
{
lean_object* v___x_832_; lean_object* v___x_833_; uint8_t v___x_834_; 
v___x_832_ = lean_array_get_size(v_args_813_);
v___x_833_ = lean_unsigned_to_nat(0u);
v___x_834_ = lean_nat_dec_eq(v___x_832_, v___x_833_);
if (v___x_834_ == 0)
{
lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v_a_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_844_; 
v___x_835_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_836_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_835_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
v_a_837_ = lean_ctor_get(v___x_836_, 0);
v_isSharedCheck_844_ = !lean_is_exclusive(v___x_836_);
if (v_isSharedCheck_844_ == 0)
{
v___x_839_ = v___x_836_;
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_a_837_);
lean_dec(v___x_836_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_842_; 
if (v_isShared_840_ == 0)
{
v___x_842_ = v___x_839_;
goto v_reusejp_841_;
}
else
{
lean_object* v_reuseFailAlloc_843_; 
v_reuseFailAlloc_843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_843_, 0, v_a_837_);
v___x_842_ = v_reuseFailAlloc_843_;
goto v_reusejp_841_;
}
v_reusejp_841_:
{
return v___x_842_;
}
}
}
else
{
goto v___jp_819_;
}
}
}
else
{
lean_object* v___x_845_; lean_object* v___x_846_; uint8_t v___x_847_; 
v___x_845_ = lean_array_get_size(v_args_813_);
v___x_846_ = lean_unsigned_to_nat(0u);
v___x_847_ = lean_nat_dec_eq(v___x_845_, v___x_846_);
if (v___x_847_ == 0)
{
lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v_a_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_857_; 
v___x_848_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_849_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_848_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
v_a_850_ = lean_ctor_get(v___x_849_, 0);
v_isSharedCheck_857_ = !lean_is_exclusive(v___x_849_);
if (v_isSharedCheck_857_ == 0)
{
v___x_852_ = v___x_849_;
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_a_850_);
lean_dec(v___x_849_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v___x_855_; 
if (v_isShared_853_ == 0)
{
v___x_855_ = v___x_852_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_a_850_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
}
else
{
goto v___jp_823_;
}
}
v___jp_819_:
{
uint8_t v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_820_ = 1;
v___x_821_ = lean_box(v___x_820_);
v___x_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_822_, 0, v___x_821_);
return v___x_822_;
}
v___jp_823_:
{
uint8_t v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_824_ = 0;
v___x_825_ = lean_box(v___x_824_);
v___x_826_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_826_, 0, v___x_825_);
return v___x_826_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0___boxed(lean_object* v_ctor_858_, lean_object* v_args_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_){
_start:
{
lean_object* v_res_865_; 
v_res_865_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___lam__0(v_ctor_858_, v_args_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
lean_dec_ref(v_args_859_);
lean_dec_ref(v_ctor_858_);
return v_res_865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr(lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_a_869_, lean_object* v_a_870_, lean_object* v_a_871_){
_start:
{
lean_object* v___f_873_; lean_object* v___x_874_; lean_object* v___x_875_; 
v___f_873_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___closed__0));
v___x_874_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm___closed__5));
v___x_875_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_874_, v___f_873_, v_a_867_, v_a_868_, v_a_869_, v_a_870_, v_a_871_);
return v___x_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr___boxed(lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr(v_a_876_, v_a_877_, v_a_878_, v_a_879_, v_a_880_);
lean_dec(v_a_880_);
lean_dec_ref(v_a_879_);
lean_dec(v_a_878_);
lean_dec_ref(v_a_877_);
return v_res_882_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1(void){
_start:
{
lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_884_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1);
v___x_885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_885_, 0, v___x_884_);
return v___x_885_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2(void){
_start:
{
lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; 
v___x_886_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1);
v___x_887_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__0));
v___x_888_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_888_, 0, v___x_887_);
lean_ctor_set(v___x_888_, 1, v___x_886_);
return v___x_888_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode(void){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__2);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0(lean_object* v_ctor_890_, lean_object* v_args_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_){
_start:
{
lean_object* v___x_945_; uint8_t v___x_946_; 
v___x_945_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_946_ = lean_string_dec_eq(v_ctor_890_, v___x_945_);
if (v___x_946_ == 0)
{
lean_object* v___x_947_; 
v___x_947_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_947_;
}
else
{
lean_object* v___x_948_; lean_object* v___x_949_; uint8_t v___x_950_; 
v___x_948_ = lean_array_get_size(v_args_891_);
v___x_949_ = lean_unsigned_to_nat(3u);
v___x_950_ = lean_nat_dec_eq(v___x_948_, v___x_949_);
if (v___x_950_ == 0)
{
lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v_a_953_; lean_object* v___x_955_; uint8_t v_isShared_956_; uint8_t v_isSharedCheck_960_; 
v___x_951_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_952_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_951_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
v_a_953_ = lean_ctor_get(v___x_952_, 0);
v_isSharedCheck_960_ = !lean_is_exclusive(v___x_952_);
if (v_isSharedCheck_960_ == 0)
{
v___x_955_ = v___x_952_;
v_isShared_956_ = v_isSharedCheck_960_;
goto v_resetjp_954_;
}
else
{
lean_inc(v_a_953_);
lean_dec(v___x_952_);
v___x_955_ = lean_box(0);
v_isShared_956_ = v_isSharedCheck_960_;
goto v_resetjp_954_;
}
v_resetjp_954_:
{
lean_object* v___x_958_; 
if (v_isShared_956_ == 0)
{
v___x_958_ = v___x_955_;
goto v_reusejp_957_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v_a_953_);
v___x_958_ = v_reuseFailAlloc_959_;
goto v_reusejp_957_;
}
v_reusejp_957_:
{
return v___x_958_;
}
}
}
else
{
goto v___jp_897_;
}
}
v___jp_897_:
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; 
v___x_898_ = l_Lean_instInhabitedExpr;
v___x_899_ = lean_unsigned_to_nat(0u);
v___x_900_ = lean_array_get_borrowed(v___x_898_, v_args_891_, v___x_899_);
lean_inc(v___x_900_);
v___x_901_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr(v___x_900_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_901_) == 0)
{
lean_object* v_a_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; 
v_a_902_ = lean_ctor_get(v___x_901_, 0);
lean_inc(v_a_902_);
lean_dec_ref_known(v___x_901_, 1);
v___x_903_ = lean_unsigned_to_nat(1u);
v___x_904_ = lean_array_get_borrowed(v___x_898_, v_args_891_, v___x_903_);
lean_inc(v___x_904_);
v___x_905_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr(v___x_904_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_905_) == 0)
{
lean_object* v_a_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v_a_906_ = lean_ctor_get(v___x_905_, 0);
lean_inc(v_a_906_);
lean_dec_ref_known(v___x_905_, 1);
v___x_907_ = lean_unsigned_to_nat(2u);
v___x_908_ = lean_array_get_borrowed(v___x_898_, v_args_891_, v___x_907_);
lean_inc(v___x_908_);
v___x_909_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr(v___x_908_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_909_) == 0)
{
lean_object* v_a_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_920_; 
v_a_910_ = lean_ctor_get(v___x_909_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_909_);
if (v_isSharedCheck_920_ == 0)
{
v___x_912_ = v___x_909_;
v_isShared_913_ = v_isSharedCheck_920_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_a_910_);
lean_dec(v___x_909_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_920_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
lean_object* v___x_914_; uint8_t v___x_915_; uint8_t v___x_916_; lean_object* v___x_918_; 
v___x_914_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_914_, 0, v_a_902_);
v___x_915_ = lean_unbox(v_a_906_);
lean_dec(v_a_906_);
lean_ctor_set_uint8(v___x_914_, sizeof(void*)*1, v___x_915_);
v___x_916_ = lean_unbox(v_a_910_);
lean_dec(v_a_910_);
lean_ctor_set_uint8(v___x_914_, sizeof(void*)*1 + 1, v___x_916_);
if (v_isShared_913_ == 0)
{
lean_ctor_set(v___x_912_, 0, v___x_914_);
v___x_918_ = v___x_912_;
goto v_reusejp_917_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v___x_914_);
v___x_918_ = v_reuseFailAlloc_919_;
goto v_reusejp_917_;
}
v_reusejp_917_:
{
return v___x_918_;
}
}
}
else
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_928_; 
lean_dec(v_a_906_);
lean_dec(v_a_902_);
v_a_921_ = lean_ctor_get(v___x_909_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_909_);
if (v_isSharedCheck_928_ == 0)
{
v___x_923_ = v___x_909_;
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_909_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_924_ == 0)
{
v___x_926_ = v___x_923_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v_a_921_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
}
else
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_936_; 
lean_dec(v_a_902_);
v_a_929_ = lean_ctor_get(v___x_905_, 0);
v_isSharedCheck_936_ = !lean_is_exclusive(v___x_905_);
if (v_isSharedCheck_936_ == 0)
{
v___x_931_ = v___x_905_;
v_isShared_932_ = v_isSharedCheck_936_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v___x_905_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_936_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
lean_object* v___x_934_; 
if (v_isShared_932_ == 0)
{
v___x_934_ = v___x_931_;
goto v_reusejp_933_;
}
else
{
lean_object* v_reuseFailAlloc_935_; 
v_reuseFailAlloc_935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_935_, 0, v_a_929_);
v___x_934_ = v_reuseFailAlloc_935_;
goto v_reusejp_933_;
}
v_reusejp_933_:
{
return v___x_934_;
}
}
}
}
else
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_944_; 
v_a_937_ = lean_ctor_get(v___x_901_, 0);
v_isSharedCheck_944_ = !lean_is_exclusive(v___x_901_);
if (v_isSharedCheck_944_ == 0)
{
v___x_939_ = v___x_901_;
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_901_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_942_; 
if (v_isShared_940_ == 0)
{
v___x_942_ = v___x_939_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_a_937_);
v___x_942_ = v_reuseFailAlloc_943_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
return v___x_942_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0___boxed(lean_object* v_ctor_961_, lean_object* v_args_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_){
_start:
{
lean_object* v_res_968_; 
v_res_968_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___lam__0(v_ctor_961_, v_args_962_, v___y_963_, v___y_964_, v___y_965_, v___y_966_);
lean_dec(v___y_966_);
lean_dec_ref(v___y_965_);
lean_dec(v___y_964_);
lean_dec_ref(v___y_963_);
lean_dec_ref(v_args_962_);
lean_dec_ref(v_ctor_961_);
return v_res_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr(lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_, lean_object* v_a_978_, lean_object* v_a_979_){
_start:
{
lean_object* v___f_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___f_981_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__0));
v___x_982_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1));
v___x_983_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_982_, v___f_981_, v_a_975_, v_a_976_, v_a_977_, v_a_978_, v_a_979_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___boxed(lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_, lean_object* v_a_988_, lean_object* v_a_989_){
_start:
{
lean_object* v_res_990_; 
v_res_990_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr(v_a_984_, v_a_985_, v_a_986_, v_a_987_, v_a_988_);
lean_dec(v_a_988_);
lean_dec_ref(v_a_987_);
lean_dec(v_a_986_);
lean_dec_ref(v_a_985_);
return v_res_990_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; 
v___x_992_ = lean_box(0);
v___x_993_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1));
v___x_994_ = l_Lean_Expr_const___override(v___x_993_, v___x_992_);
return v___x_994_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2(void){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_995_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1);
v___x_996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_996_, 0, v___x_995_);
return v___x_996_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3(void){
_start:
{
lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; 
v___x_997_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2);
v___x_998_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__0));
v___x_999_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_999_, 0, v___x_998_);
lean_ctor_set(v___x_999_, 1, v___x_997_);
return v___x_999_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1(void){
_start:
{
lean_object* v___x_1000_; 
v___x_1000_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__3);
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(lean_object* v_e_1001_, lean_object* v___y_1002_){
_start:
{
uint8_t v___x_1004_; 
v___x_1004_ = l_Lean_Expr_hasMVar(v_e_1001_);
if (v___x_1004_ == 0)
{
lean_object* v___x_1005_; 
v___x_1005_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1005_, 0, v_e_1001_);
return v___x_1005_;
}
else
{
lean_object* v___x_1006_; lean_object* v_mctx_1007_; lean_object* v___x_1008_; lean_object* v_fst_1009_; lean_object* v_snd_1010_; lean_object* v___x_1011_; lean_object* v_cache_1012_; lean_object* v_zetaDeltaFVarIds_1013_; lean_object* v_postponed_1014_; lean_object* v_diag_1015_; lean_object* v___x_1017_; uint8_t v_isShared_1018_; uint8_t v_isSharedCheck_1024_; 
v___x_1006_ = lean_st_ref_get(v___y_1002_);
v_mctx_1007_ = lean_ctor_get(v___x_1006_, 0);
lean_inc_ref(v_mctx_1007_);
lean_dec(v___x_1006_);
v___x_1008_ = l_Lean_instantiateMVarsCore(v_mctx_1007_, v_e_1001_);
v_fst_1009_ = lean_ctor_get(v___x_1008_, 0);
lean_inc(v_fst_1009_);
v_snd_1010_ = lean_ctor_get(v___x_1008_, 1);
lean_inc(v_snd_1010_);
lean_dec_ref(v___x_1008_);
v___x_1011_ = lean_st_ref_take(v___y_1002_);
v_cache_1012_ = lean_ctor_get(v___x_1011_, 1);
v_zetaDeltaFVarIds_1013_ = lean_ctor_get(v___x_1011_, 2);
v_postponed_1014_ = lean_ctor_get(v___x_1011_, 3);
v_diag_1015_ = lean_ctor_get(v___x_1011_, 4);
v_isSharedCheck_1024_ = !lean_is_exclusive(v___x_1011_);
if (v_isSharedCheck_1024_ == 0)
{
lean_object* v_unused_1025_; 
v_unused_1025_ = lean_ctor_get(v___x_1011_, 0);
lean_dec(v_unused_1025_);
v___x_1017_ = v___x_1011_;
v_isShared_1018_ = v_isSharedCheck_1024_;
goto v_resetjp_1016_;
}
else
{
lean_inc(v_diag_1015_);
lean_inc(v_postponed_1014_);
lean_inc(v_zetaDeltaFVarIds_1013_);
lean_inc(v_cache_1012_);
lean_dec(v___x_1011_);
v___x_1017_ = lean_box(0);
v_isShared_1018_ = v_isSharedCheck_1024_;
goto v_resetjp_1016_;
}
v_resetjp_1016_:
{
lean_object* v___x_1020_; 
if (v_isShared_1018_ == 0)
{
lean_ctor_set(v___x_1017_, 0, v_snd_1010_);
v___x_1020_ = v___x_1017_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v_snd_1010_);
lean_ctor_set(v_reuseFailAlloc_1023_, 1, v_cache_1012_);
lean_ctor_set(v_reuseFailAlloc_1023_, 2, v_zetaDeltaFVarIds_1013_);
lean_ctor_set(v_reuseFailAlloc_1023_, 3, v_postponed_1014_);
lean_ctor_set(v_reuseFailAlloc_1023_, 4, v_diag_1015_);
v___x_1020_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; 
v___x_1021_ = lean_st_ref_set(v___y_1002_, v___x_1020_);
v___x_1022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1022_, 0, v_fst_1009_);
return v___x_1022_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg___boxed(lean_object* v_e_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_){
_start:
{
lean_object* v_res_1029_; 
v_res_1029_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_e_1026_, v___y_1027_);
lean_dec(v___y_1027_);
return v_res_1029_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0(void){
_start:
{
lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1030_ = lean_box(1);
v___x_1031_ = l_Lean_MessageData_ofFormat(v___x_1030_);
return v___x_1031_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3(void){
_start:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; 
v___x_1035_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__2));
v___x_1036_ = l_Lean_MessageData_ofFormat(v___x_1035_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11(lean_object* v_x_1037_, lean_object* v_x_1038_){
_start:
{
if (lean_obj_tag(v_x_1038_) == 0)
{
return v_x_1037_;
}
else
{
lean_object* v_head_1039_; lean_object* v_tail_1040_; lean_object* v___x_1042_; uint8_t v_isShared_1043_; uint8_t v_isSharedCheck_1062_; 
v_head_1039_ = lean_ctor_get(v_x_1038_, 0);
v_tail_1040_ = lean_ctor_get(v_x_1038_, 1);
v_isSharedCheck_1062_ = !lean_is_exclusive(v_x_1038_);
if (v_isSharedCheck_1062_ == 0)
{
v___x_1042_ = v_x_1038_;
v_isShared_1043_ = v_isSharedCheck_1062_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_tail_1040_);
lean_inc(v_head_1039_);
lean_dec(v_x_1038_);
v___x_1042_ = lean_box(0);
v_isShared_1043_ = v_isSharedCheck_1062_;
goto v_resetjp_1041_;
}
v_resetjp_1041_:
{
lean_object* v_before_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1060_; 
v_before_1044_ = lean_ctor_get(v_head_1039_, 0);
v_isSharedCheck_1060_ = !lean_is_exclusive(v_head_1039_);
if (v_isSharedCheck_1060_ == 0)
{
lean_object* v_unused_1061_; 
v_unused_1061_ = lean_ctor_get(v_head_1039_, 1);
lean_dec(v_unused_1061_);
v___x_1046_ = v_head_1039_;
v_isShared_1047_ = v_isSharedCheck_1060_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_before_1044_);
lean_dec(v_head_1039_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1060_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v___x_1048_; lean_object* v___x_1050_; 
v___x_1048_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0);
if (v_isShared_1047_ == 0)
{
lean_ctor_set_tag(v___x_1046_, 7);
lean_ctor_set(v___x_1046_, 1, v___x_1048_);
lean_ctor_set(v___x_1046_, 0, v_x_1037_);
v___x_1050_ = v___x_1046_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v_x_1037_);
lean_ctor_set(v_reuseFailAlloc_1059_, 1, v___x_1048_);
v___x_1050_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
lean_object* v___x_1051_; lean_object* v___x_1053_; 
v___x_1051_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__3);
if (v_isShared_1043_ == 0)
{
lean_ctor_set_tag(v___x_1042_, 7);
lean_ctor_set(v___x_1042_, 1, v___x_1051_);
lean_ctor_set(v___x_1042_, 0, v___x_1050_);
v___x_1053_ = v___x_1042_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v___x_1050_);
lean_ctor_set(v_reuseFailAlloc_1058_, 1, v___x_1051_);
v___x_1053_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; 
v___x_1054_ = l_Lean_MessageData_ofSyntax(v_before_1044_);
v___x_1055_ = l_Lean_indentD(v___x_1054_);
v___x_1056_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1053_);
lean_ctor_set(v___x_1056_, 1, v___x_1055_);
v_x_1037_ = v___x_1056_;
v_x_1038_ = v_tail_1040_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10(lean_object* v_opts_1063_, lean_object* v_opt_1064_){
_start:
{
lean_object* v_name_1065_; lean_object* v_defValue_1066_; lean_object* v_map_1067_; lean_object* v___x_1068_; 
v_name_1065_ = lean_ctor_get(v_opt_1064_, 0);
v_defValue_1066_ = lean_ctor_get(v_opt_1064_, 1);
v_map_1067_ = lean_ctor_get(v_opts_1063_, 0);
v___x_1068_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1067_, v_name_1065_);
if (lean_obj_tag(v___x_1068_) == 0)
{
uint8_t v___x_1069_; 
v___x_1069_ = lean_unbox(v_defValue_1066_);
return v___x_1069_;
}
else
{
lean_object* v_val_1070_; 
v_val_1070_ = lean_ctor_get(v___x_1068_, 0);
lean_inc(v_val_1070_);
lean_dec_ref_known(v___x_1068_, 1);
if (lean_obj_tag(v_val_1070_) == 1)
{
uint8_t v_v_1071_; 
v_v_1071_ = lean_ctor_get_uint8(v_val_1070_, 0);
lean_dec_ref_known(v_val_1070_, 0);
return v_v_1071_;
}
else
{
uint8_t v___x_1072_; 
lean_dec(v_val_1070_);
v___x_1072_ = lean_unbox(v_defValue_1066_);
return v___x_1072_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10___boxed(lean_object* v_opts_1073_, lean_object* v_opt_1074_){
_start:
{
uint8_t v_res_1075_; lean_object* v_r_1076_; 
v_res_1075_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10(v_opts_1073_, v_opt_1074_);
lean_dec_ref(v_opt_1074_);
lean_dec_ref(v_opts_1073_);
v_r_1076_ = lean_box(v_res_1075_);
return v_r_1076_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2(void){
_start:
{
lean_object* v___x_1080_; lean_object* v___x_1081_; 
v___x_1080_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__1));
v___x_1081_ = l_Lean_MessageData_ofFormat(v___x_1080_);
return v___x_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg(lean_object* v_msgData_1082_, lean_object* v_macroStack_1083_, lean_object* v___y_1084_){
_start:
{
lean_object* v_options_1086_; lean_object* v___x_1087_; uint8_t v___x_1088_; 
v_options_1086_ = lean_ctor_get(v___y_1084_, 2);
v___x_1087_ = l_Lean_Elab_pp_macroStack;
v___x_1088_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__10(v_options_1086_, v___x_1087_);
if (v___x_1088_ == 0)
{
lean_object* v___x_1089_; 
lean_dec(v_macroStack_1083_);
v___x_1089_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1089_, 0, v_msgData_1082_);
return v___x_1089_;
}
else
{
if (lean_obj_tag(v_macroStack_1083_) == 0)
{
lean_object* v___x_1090_; 
v___x_1090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1090_, 0, v_msgData_1082_);
return v___x_1090_;
}
else
{
lean_object* v_head_1091_; lean_object* v_after_1092_; lean_object* v___x_1094_; uint8_t v_isShared_1095_; uint8_t v_isSharedCheck_1107_; 
v_head_1091_ = lean_ctor_get(v_macroStack_1083_, 0);
lean_inc(v_head_1091_);
v_after_1092_ = lean_ctor_get(v_head_1091_, 1);
v_isSharedCheck_1107_ = !lean_is_exclusive(v_head_1091_);
if (v_isSharedCheck_1107_ == 0)
{
lean_object* v_unused_1108_; 
v_unused_1108_ = lean_ctor_get(v_head_1091_, 0);
lean_dec(v_unused_1108_);
v___x_1094_ = v_head_1091_;
v_isShared_1095_ = v_isSharedCheck_1107_;
goto v_resetjp_1093_;
}
else
{
lean_inc(v_after_1092_);
lean_dec(v_head_1091_);
v___x_1094_ = lean_box(0);
v_isShared_1095_ = v_isSharedCheck_1107_;
goto v_resetjp_1093_;
}
v_resetjp_1093_:
{
lean_object* v___x_1096_; lean_object* v___x_1098_; 
v___x_1096_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11___closed__0);
if (v_isShared_1095_ == 0)
{
lean_ctor_set_tag(v___x_1094_, 7);
lean_ctor_set(v___x_1094_, 1, v___x_1096_);
lean_ctor_set(v___x_1094_, 0, v_msgData_1082_);
v___x_1098_ = v___x_1094_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1106_; 
v_reuseFailAlloc_1106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1106_, 0, v_msgData_1082_);
lean_ctor_set(v_reuseFailAlloc_1106_, 1, v___x_1096_);
v___x_1098_ = v_reuseFailAlloc_1106_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v_msgData_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1099_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___closed__2);
v___x_1100_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1098_);
lean_ctor_set(v___x_1100_, 1, v___x_1099_);
v___x_1101_ = l_Lean_MessageData_ofSyntax(v_after_1092_);
v___x_1102_ = l_Lean_indentD(v___x_1101_);
v_msgData_1103_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1103_, 0, v___x_1100_);
lean_ctor_set(v_msgData_1103_, 1, v___x_1102_);
v___x_1104_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8_spec__11(v_msgData_1103_, v_macroStack_1083_);
v___x_1105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1104_);
return v___x_1105_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg___boxed(lean_object* v_msgData_1109_, lean_object* v_macroStack_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
lean_object* v_res_1113_; 
v_res_1113_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg(v_msgData_1109_, v_macroStack_1110_, v___y_1111_);
lean_dec_ref(v___y_1111_);
return v_res_1113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(lean_object* v_msg_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
lean_object* v_ref_1122_; lean_object* v___x_1123_; lean_object* v_a_1124_; lean_object* v_macroStack_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v_a_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1136_; 
v_ref_1122_ = lean_ctor_get(v___y_1119_, 5);
v___x_1123_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_1114_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_);
v_a_1124_ = lean_ctor_get(v___x_1123_, 0);
lean_inc(v_a_1124_);
lean_dec_ref(v___x_1123_);
v_macroStack_1125_ = lean_ctor_get(v___y_1115_, 1);
v___x_1126_ = l_Lean_Elab_getBetterRef(v_ref_1122_, v_macroStack_1125_);
lean_inc(v_macroStack_1125_);
v___x_1127_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg(v_a_1124_, v_macroStack_1125_, v___y_1119_);
v_a_1128_ = lean_ctor_get(v___x_1127_, 0);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_1127_);
if (v_isSharedCheck_1136_ == 0)
{
v___x_1130_ = v___x_1127_;
v_isShared_1131_ = v_isSharedCheck_1136_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_a_1128_);
lean_dec(v___x_1127_);
v___x_1130_ = lean_box(0);
v_isShared_1131_ = v_isSharedCheck_1136_;
goto v_resetjp_1129_;
}
v_resetjp_1129_:
{
lean_object* v___x_1132_; lean_object* v___x_1134_; 
v___x_1132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1132_, 0, v___x_1126_);
lean_ctor_set(v___x_1132_, 1, v_a_1128_);
if (v_isShared_1131_ == 0)
{
lean_ctor_set_tag(v___x_1130_, 1);
lean_ctor_set(v___x_1130_, 0, v___x_1132_);
v___x_1134_ = v___x_1130_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v___x_1132_);
v___x_1134_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1133_;
}
v_reusejp_1133_:
{
return v___x_1134_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg___boxed(lean_object* v_msg_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v_msg_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
return v_res_1145_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; 
v___x_1146_ = lean_box(0);
v___x_1147_ = l_Lean_Elab_abortTermExceptionId;
v___x_1148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1148_, 0, v___x_1147_);
lean_ctor_set(v___x_1148_, 1, v___x_1146_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg(){
_start:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___closed__0);
v___x_1151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1151_, 0, v___x_1150_);
return v___x_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg___boxed(lean_object* v___y_1152_){
_start:
{
lean_object* v_res_1153_; 
v_res_1153_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
return v_res_1153_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1155_; lean_object* v___x_1156_; 
v___x_1155_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__0));
v___x_1156_ = l_Lean_stringToMessageData(v___x_1155_);
return v___x_1156_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2(void){
_start:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1157_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__1);
v___x_1158_ = l_Lean_MessageData_ofExpr(v___x_1157_);
return v___x_1158_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; 
v___x_1159_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__2);
v___x_1160_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1);
v___x_1161_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1160_);
lean_ctor_set(v___x_1161_, 1, v___x_1159_);
return v___x_1161_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5(void){
_start:
{
lean_object* v___x_1163_; lean_object* v___x_1164_; 
v___x_1163_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__4));
v___x_1164_ = l_Lean_stringToMessageData(v___x_1163_);
return v___x_1164_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6(void){
_start:
{
lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; 
v___x_1165_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5);
v___x_1166_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__3);
v___x_1167_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1166_);
lean_ctor_set(v___x_1167_, 1, v___x_1165_);
return v___x_1167_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8(void){
_start:
{
lean_object* v___x_1169_; lean_object* v___x_1170_; 
v___x_1169_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__7));
v___x_1170_ = l_Lean_stringToMessageData(v___x_1169_);
return v___x_1170_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10(void){
_start:
{
lean_object* v___x_1172_; lean_object* v___x_1173_; 
v___x_1172_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__9));
v___x_1173_ = l_Lean_stringToMessageData(v___x_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3(lean_object* v_stx_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_){
_start:
{
lean_object* v_ty_x3f_1182_; uint8_t v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v_fileName_1188_; lean_object* v_fileMap_1189_; lean_object* v_options_1190_; lean_object* v_currRecDepth_1191_; lean_object* v_maxRecDepth_1192_; lean_object* v_ref_1193_; lean_object* v_currNamespace_1194_; lean_object* v_openDecls_1195_; lean_object* v_initHeartbeats_1196_; lean_object* v_maxHeartbeats_1197_; lean_object* v_quotContext_1198_; lean_object* v_currMacroScope_1199_; uint8_t v_diag_1200_; lean_object* v_cancelTk_x3f_1201_; uint8_t v_suppressElabErrors_1202_; lean_object* v_inheritedTraceOptions_1203_; uint8_t v___x_1204_; lean_object* v_ref_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
v_ty_x3f_1182_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1___closed__2);
v___x_1183_ = 1;
v___x_1184_ = lean_box(0);
v___x_1185_ = lean_box(v___x_1183_);
v___x_1186_ = lean_box(v___x_1183_);
lean_inc(v_stx_1174_);
v___x_1187_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1187_, 0, v_stx_1174_);
lean_closure_set(v___x_1187_, 1, v_ty_x3f_1182_);
lean_closure_set(v___x_1187_, 2, v___x_1185_);
lean_closure_set(v___x_1187_, 3, v___x_1186_);
lean_closure_set(v___x_1187_, 4, v___x_1184_);
v_fileName_1188_ = lean_ctor_get(v_a_1179_, 0);
v_fileMap_1189_ = lean_ctor_get(v_a_1179_, 1);
v_options_1190_ = lean_ctor_get(v_a_1179_, 2);
v_currRecDepth_1191_ = lean_ctor_get(v_a_1179_, 3);
v_maxRecDepth_1192_ = lean_ctor_get(v_a_1179_, 4);
v_ref_1193_ = lean_ctor_get(v_a_1179_, 5);
v_currNamespace_1194_ = lean_ctor_get(v_a_1179_, 6);
v_openDecls_1195_ = lean_ctor_get(v_a_1179_, 7);
v_initHeartbeats_1196_ = lean_ctor_get(v_a_1179_, 8);
v_maxHeartbeats_1197_ = lean_ctor_get(v_a_1179_, 9);
v_quotContext_1198_ = lean_ctor_get(v_a_1179_, 10);
v_currMacroScope_1199_ = lean_ctor_get(v_a_1179_, 11);
v_diag_1200_ = lean_ctor_get_uint8(v_a_1179_, sizeof(void*)*14);
v_cancelTk_x3f_1201_ = lean_ctor_get(v_a_1179_, 12);
v_suppressElabErrors_1202_ = lean_ctor_get_uint8(v_a_1179_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1203_ = lean_ctor_get(v_a_1179_, 13);
v___x_1204_ = 1;
v_ref_1205_ = l_Lean_replaceRef(v_stx_1174_, v_ref_1193_);
lean_dec(v_stx_1174_);
lean_inc_ref(v_inheritedTraceOptions_1203_);
lean_inc(v_cancelTk_x3f_1201_);
lean_inc(v_currMacroScope_1199_);
lean_inc(v_quotContext_1198_);
lean_inc(v_maxHeartbeats_1197_);
lean_inc(v_initHeartbeats_1196_);
lean_inc(v_openDecls_1195_);
lean_inc(v_currNamespace_1194_);
lean_inc(v_maxRecDepth_1192_);
lean_inc(v_currRecDepth_1191_);
lean_inc_ref(v_options_1190_);
lean_inc_ref(v_fileMap_1189_);
lean_inc_ref(v_fileName_1188_);
v___x_1206_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1206_, 0, v_fileName_1188_);
lean_ctor_set(v___x_1206_, 1, v_fileMap_1189_);
lean_ctor_set(v___x_1206_, 2, v_options_1190_);
lean_ctor_set(v___x_1206_, 3, v_currRecDepth_1191_);
lean_ctor_set(v___x_1206_, 4, v_maxRecDepth_1192_);
lean_ctor_set(v___x_1206_, 5, v_ref_1205_);
lean_ctor_set(v___x_1206_, 6, v_currNamespace_1194_);
lean_ctor_set(v___x_1206_, 7, v_openDecls_1195_);
lean_ctor_set(v___x_1206_, 8, v_initHeartbeats_1196_);
lean_ctor_set(v___x_1206_, 9, v_maxHeartbeats_1197_);
lean_ctor_set(v___x_1206_, 10, v_quotContext_1198_);
lean_ctor_set(v___x_1206_, 11, v_currMacroScope_1199_);
lean_ctor_set(v___x_1206_, 12, v_cancelTk_x3f_1201_);
lean_ctor_set(v___x_1206_, 13, v_inheritedTraceOptions_1203_);
lean_ctor_set_uint8(v___x_1206_, sizeof(void*)*14, v_diag_1200_);
lean_ctor_set_uint8(v___x_1206_, sizeof(void*)*14 + 1, v_suppressElabErrors_1202_);
v___x_1207_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1187_, v___x_1204_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v___x_1206_, v_a_1180_);
if (lean_obj_tag(v___x_1207_) == 0)
{
lean_object* v_a_1208_; lean_object* v___x_1209_; lean_object* v_a_1210_; lean_object* v___y_1212_; lean_object* v___y_1213_; lean_object* v___y_1214_; lean_object* v___y_1215_; lean_object* v___y_1216_; lean_object* v___y_1217_; lean_object* v___y_1218_; lean_object* v___y_1219_; lean_object* v___y_1220_; uint8_t v___y_1221_; lean_object* v___y_1238_; lean_object* v___y_1239_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1250_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___y_1253_; lean_object* v___y_1254_; lean_object* v___y_1255_; lean_object* v___y_1287_; lean_object* v___y_1288_; lean_object* v___y_1289_; lean_object* v___y_1290_; lean_object* v___y_1291_; lean_object* v___y_1292_; uint8_t v___x_1305_; 
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
lean_inc(v_a_1208_);
lean_dec_ref_known(v___x_1207_, 1);
v___x_1209_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_a_1208_, v_a_1178_);
v_a_1210_ = lean_ctor_get(v___x_1209_, 0);
lean_inc(v_a_1210_);
lean_dec_ref(v___x_1209_);
v___x_1305_ = l_Lean_Expr_hasSorry(v_a_1210_);
if (v___x_1305_ == 0)
{
v___y_1250_ = v_a_1175_;
v___y_1251_ = v_a_1176_;
v___y_1252_ = v_a_1177_;
v___y_1253_ = v_a_1178_;
v___y_1254_ = v___x_1206_;
v___y_1255_ = v_a_1180_;
goto v___jp_1249_;
}
else
{
uint8_t v___x_1306_; 
v___x_1306_ = l_Lean_Expr_hasSyntheticSorry(v_a_1210_);
if (v___x_1306_ == 0)
{
v___y_1287_ = v_a_1175_;
v___y_1288_ = v_a_1176_;
v___y_1289_ = v_a_1177_;
v___y_1290_ = v_a_1178_;
v___y_1291_ = v___x_1206_;
v___y_1292_ = v_a_1180_;
goto v___jp_1286_;
}
else
{
lean_object* v___x_1307_; lean_object* v_a_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1315_; 
lean_dec(v_a_1210_);
lean_dec_ref_known(v___x_1206_, 14);
v___x_1307_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1308_ = lean_ctor_get(v___x_1307_, 0);
v_isSharedCheck_1315_ = !lean_is_exclusive(v___x_1307_);
if (v_isSharedCheck_1315_ == 0)
{
v___x_1310_ = v___x_1307_;
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_a_1308_);
lean_dec(v___x_1307_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v___x_1313_; 
if (v_isShared_1311_ == 0)
{
v___x_1313_ = v___x_1310_;
goto v_reusejp_1312_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v_a_1308_);
v___x_1313_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1312_;
}
v_reusejp_1312_:
{
return v___x_1313_;
}
}
}
}
v___jp_1211_:
{
if (v___y_1221_ == 0)
{
if (lean_obj_tag(v___y_1220_) == 0)
{
lean_dec_ref_known(v___y_1220_, 2);
lean_dec_ref(v___y_1215_);
lean_dec(v_a_1210_);
return v___y_1213_;
}
else
{
lean_object* v_id_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1235_; 
v_id_1222_ = lean_ctor_get(v___y_1220_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___y_1220_);
if (v_isSharedCheck_1235_ == 0)
{
lean_object* v_unused_1236_; 
v_unused_1236_ = lean_ctor_get(v___y_1220_, 1);
lean_dec(v_unused_1236_);
v___x_1224_ = v___y_1220_;
v_isShared_1225_ = v_isSharedCheck_1235_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_id_1222_);
lean_dec(v___y_1220_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1235_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
uint8_t v___x_1226_; 
v___x_1226_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1219_, v_id_1222_);
lean_dec(v_id_1222_);
if (v___x_1226_ == 0)
{
lean_del_object(v___x_1224_);
lean_dec_ref(v___y_1215_);
lean_dec(v_a_1210_);
return v___y_1213_;
}
else
{
lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1231_; 
lean_dec_ref(v___y_1213_);
v___x_1227_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__6);
v___x_1228_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8);
v___x_1229_ = l_Lean_indentExpr(v_a_1210_);
if (v_isShared_1225_ == 0)
{
lean_ctor_set_tag(v___x_1224_, 7);
lean_ctor_set(v___x_1224_, 1, v___x_1229_);
lean_ctor_set(v___x_1224_, 0, v___x_1228_);
v___x_1231_ = v___x_1224_;
goto v_reusejp_1230_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v___x_1228_);
lean_ctor_set(v_reuseFailAlloc_1234_, 1, v___x_1229_);
v___x_1231_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1230_;
}
v_reusejp_1230_:
{
lean_object* v___x_1232_; lean_object* v___x_1233_; 
v___x_1232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1232_, 0, v___x_1231_);
lean_ctor_set(v___x_1232_, 1, v___x_1227_);
v___x_1233_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1232_, v___y_1217_, v___y_1212_, v___y_1214_, v___y_1216_, v___y_1215_, v___y_1218_);
lean_dec_ref(v___y_1215_);
return v___x_1233_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1220_);
lean_dec_ref(v___y_1215_);
lean_dec(v_a_1210_);
return v___y_1213_;
}
}
v___jp_1237_:
{
lean_object* v___x_1244_; 
lean_inc(v_a_1210_);
v___x_1244_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr(v_a_1210_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_);
if (lean_obj_tag(v___x_1244_) == 0)
{
lean_dec_ref(v___y_1242_);
lean_dec(v_a_1210_);
return v___x_1244_;
}
else
{
lean_object* v_a_1245_; lean_object* v___x_1246_; uint8_t v___x_1247_; 
v_a_1245_ = lean_ctor_get(v___x_1244_, 0);
lean_inc(v_a_1245_);
v___x_1246_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1247_ = l_Lean_Exception_isInterrupt(v_a_1245_);
if (v___x_1247_ == 0)
{
uint8_t v___x_1248_; 
lean_inc(v_a_1245_);
v___x_1248_ = l_Lean_Exception_isRuntime(v_a_1245_);
v___y_1212_ = v___y_1239_;
v___y_1213_ = v___x_1244_;
v___y_1214_ = v___y_1240_;
v___y_1215_ = v___y_1242_;
v___y_1216_ = v___y_1241_;
v___y_1217_ = v___y_1238_;
v___y_1218_ = v___y_1243_;
v___y_1219_ = v___x_1246_;
v___y_1220_ = v_a_1245_;
v___y_1221_ = v___x_1248_;
goto v___jp_1211_;
}
else
{
v___y_1212_ = v___y_1239_;
v___y_1213_ = v___x_1244_;
v___y_1214_ = v___y_1240_;
v___y_1215_ = v___y_1242_;
v___y_1216_ = v___y_1241_;
v___y_1217_ = v___y_1238_;
v___y_1218_ = v___y_1243_;
v___y_1219_ = v___x_1246_;
v___y_1220_ = v_a_1245_;
v___y_1221_ = v___x_1247_;
goto v___jp_1211_;
}
}
}
v___jp_1249_:
{
lean_object* v___x_1256_; 
lean_inc(v_a_1210_);
v___x_1256_ = l_Lean_Meta_getMVars(v_a_1210_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_);
if (lean_obj_tag(v___x_1256_) == 0)
{
lean_object* v_a_1257_; lean_object* v___x_1258_; 
v_a_1257_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_a_1257_);
lean_dec_ref_known(v___x_1256_, 1);
v___x_1258_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1257_, v___x_1184_, v___y_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_);
lean_dec(v_a_1257_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; uint8_t v___x_1260_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
lean_inc(v_a_1259_);
lean_dec_ref_known(v___x_1258_, 1);
v___x_1260_ = lean_unbox(v_a_1259_);
lean_dec(v_a_1259_);
if (v___x_1260_ == 0)
{
v___y_1238_ = v___y_1250_;
v___y_1239_ = v___y_1251_;
v___y_1240_ = v___y_1252_;
v___y_1241_ = v___y_1253_;
v___y_1242_ = v___y_1254_;
v___y_1243_ = v___y_1255_;
goto v___jp_1237_;
}
else
{
lean_object* v___x_1261_; lean_object* v_a_1262_; lean_object* v___x_1264_; uint8_t v_isShared_1265_; uint8_t v_isSharedCheck_1269_; 
lean_dec_ref(v___y_1254_);
lean_dec(v_a_1210_);
v___x_1261_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1262_ = lean_ctor_get(v___x_1261_, 0);
v_isSharedCheck_1269_ = !lean_is_exclusive(v___x_1261_);
if (v_isSharedCheck_1269_ == 0)
{
v___x_1264_ = v___x_1261_;
v_isShared_1265_ = v_isSharedCheck_1269_;
goto v_resetjp_1263_;
}
else
{
lean_inc(v_a_1262_);
lean_dec(v___x_1261_);
v___x_1264_ = lean_box(0);
v_isShared_1265_ = v_isSharedCheck_1269_;
goto v_resetjp_1263_;
}
v_resetjp_1263_:
{
lean_object* v___x_1267_; 
if (v_isShared_1265_ == 0)
{
v___x_1267_ = v___x_1264_;
goto v_reusejp_1266_;
}
else
{
lean_object* v_reuseFailAlloc_1268_; 
v_reuseFailAlloc_1268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1268_, 0, v_a_1262_);
v___x_1267_ = v_reuseFailAlloc_1268_;
goto v_reusejp_1266_;
}
v_reusejp_1266_:
{
return v___x_1267_;
}
}
}
}
else
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1277_; 
lean_dec_ref(v___y_1254_);
lean_dec(v_a_1210_);
v_a_1270_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1277_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1277_ == 0)
{
v___x_1272_ = v___x_1258_;
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1258_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1275_; 
if (v_isShared_1273_ == 0)
{
v___x_1275_ = v___x_1272_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v_a_1270_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
else
{
lean_object* v_a_1278_; lean_object* v___x_1280_; uint8_t v_isShared_1281_; uint8_t v_isSharedCheck_1285_; 
lean_dec_ref(v___y_1254_);
lean_dec(v_a_1210_);
v_a_1278_ = lean_ctor_get(v___x_1256_, 0);
v_isSharedCheck_1285_ = !lean_is_exclusive(v___x_1256_);
if (v_isSharedCheck_1285_ == 0)
{
v___x_1280_ = v___x_1256_;
v_isShared_1281_ = v_isSharedCheck_1285_;
goto v_resetjp_1279_;
}
else
{
lean_inc(v_a_1278_);
lean_dec(v___x_1256_);
v___x_1280_ = lean_box(0);
v_isShared_1281_ = v_isSharedCheck_1285_;
goto v_resetjp_1279_;
}
v_resetjp_1279_:
{
lean_object* v___x_1283_; 
if (v_isShared_1281_ == 0)
{
v___x_1283_ = v___x_1280_;
goto v_reusejp_1282_;
}
else
{
lean_object* v_reuseFailAlloc_1284_; 
v_reuseFailAlloc_1284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1284_, 0, v_a_1278_);
v___x_1283_ = v_reuseFailAlloc_1284_;
goto v_reusejp_1282_;
}
v_reusejp_1282_:
{
return v___x_1283_;
}
}
}
}
v___jp_1286_:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v_a_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1304_; 
v___x_1293_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10);
v___x_1294_ = l_Lean_indentExpr(v_a_1210_);
v___x_1295_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1295_, 0, v___x_1293_);
lean_ctor_set(v___x_1295_, 1, v___x_1294_);
v___x_1296_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1295_, v___y_1287_, v___y_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_);
lean_dec_ref(v___y_1291_);
v_a_1297_ = lean_ctor_get(v___x_1296_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1296_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1299_ = v___x_1296_;
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_a_1297_);
lean_dec(v___x_1296_);
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
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_a_1297_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
}
else
{
lean_object* v_a_1316_; lean_object* v___x_1318_; uint8_t v_isShared_1319_; uint8_t v_isSharedCheck_1323_; 
lean_dec_ref_known(v___x_1206_, 14);
v_a_1316_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1323_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1323_ == 0)
{
v___x_1318_ = v___x_1207_;
v_isShared_1319_ = v_isSharedCheck_1323_;
goto v_resetjp_1317_;
}
else
{
lean_inc(v_a_1316_);
lean_dec(v___x_1207_);
v___x_1318_ = lean_box(0);
v_isShared_1319_ = v_isSharedCheck_1323_;
goto v_resetjp_1317_;
}
v_resetjp_1317_:
{
lean_object* v___x_1321_; 
if (v_isShared_1319_ == 0)
{
v___x_1321_ = v___x_1318_;
goto v_reusejp_1320_;
}
else
{
lean_object* v_reuseFailAlloc_1322_; 
v_reuseFailAlloc_1322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1322_, 0, v_a_1316_);
v___x_1321_ = v_reuseFailAlloc_1322_;
goto v_reusejp_1320_;
}
v_reusejp_1320_:
{
return v___x_1321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___boxed(lean_object* v_stx_1324_, lean_object* v_a_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_, lean_object* v_a_1328_, lean_object* v_a_1329_, lean_object* v_a_1330_, lean_object* v_a_1331_){
_start:
{
lean_object* v_res_1332_; 
v_res_1332_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3(v_stx_1324_, v_a_1325_, v_a_1326_, v_a_1327_, v_a_1328_, v_a_1329_, v_a_1330_);
lean_dec(v_a_1330_);
lean_dec_ref(v_a_1329_);
lean_dec(v_a_1328_);
lean_dec_ref(v_a_1327_);
lean_dec(v_a_1326_);
lean_dec_ref(v_a_1325_);
return v_res_1332_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1333_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode___closed__1);
v___x_1334_ = l_Lean_MessageData_ofExpr(v___x_1333_);
return v___x_1334_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1335_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__0);
v___x_1336_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1);
v___x_1337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1337_, 0, v___x_1336_);
lean_ctor_set(v___x_1337_, 1, v___x_1335_);
return v___x_1337_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; 
v___x_1338_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5);
v___x_1339_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__1);
v___x_1340_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1340_, 0, v___x_1339_);
lean_ctor_set(v___x_1340_, 1, v___x_1338_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2(lean_object* v_stx_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_){
_start:
{
lean_object* v_ty_x3f_1349_; uint8_t v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v_fileName_1355_; lean_object* v_fileMap_1356_; lean_object* v_options_1357_; lean_object* v_currRecDepth_1358_; lean_object* v_maxRecDepth_1359_; lean_object* v_ref_1360_; lean_object* v_currNamespace_1361_; lean_object* v_openDecls_1362_; lean_object* v_initHeartbeats_1363_; lean_object* v_maxHeartbeats_1364_; lean_object* v_quotContext_1365_; lean_object* v_currMacroScope_1366_; uint8_t v_diag_1367_; lean_object* v_cancelTk_x3f_1368_; uint8_t v_suppressElabErrors_1369_; lean_object* v_inheritedTraceOptions_1370_; uint8_t v___x_1371_; lean_object* v_ref_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; 
v_ty_x3f_1349_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode___closed__1);
v___x_1350_ = 1;
v___x_1351_ = lean_box(0);
v___x_1352_ = lean_box(v___x_1350_);
v___x_1353_ = lean_box(v___x_1350_);
lean_inc(v_stx_1341_);
v___x_1354_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1354_, 0, v_stx_1341_);
lean_closure_set(v___x_1354_, 1, v_ty_x3f_1349_);
lean_closure_set(v___x_1354_, 2, v___x_1352_);
lean_closure_set(v___x_1354_, 3, v___x_1353_);
lean_closure_set(v___x_1354_, 4, v___x_1351_);
v_fileName_1355_ = lean_ctor_get(v_a_1346_, 0);
v_fileMap_1356_ = lean_ctor_get(v_a_1346_, 1);
v_options_1357_ = lean_ctor_get(v_a_1346_, 2);
v_currRecDepth_1358_ = lean_ctor_get(v_a_1346_, 3);
v_maxRecDepth_1359_ = lean_ctor_get(v_a_1346_, 4);
v_ref_1360_ = lean_ctor_get(v_a_1346_, 5);
v_currNamespace_1361_ = lean_ctor_get(v_a_1346_, 6);
v_openDecls_1362_ = lean_ctor_get(v_a_1346_, 7);
v_initHeartbeats_1363_ = lean_ctor_get(v_a_1346_, 8);
v_maxHeartbeats_1364_ = lean_ctor_get(v_a_1346_, 9);
v_quotContext_1365_ = lean_ctor_get(v_a_1346_, 10);
v_currMacroScope_1366_ = lean_ctor_get(v_a_1346_, 11);
v_diag_1367_ = lean_ctor_get_uint8(v_a_1346_, sizeof(void*)*14);
v_cancelTk_x3f_1368_ = lean_ctor_get(v_a_1346_, 12);
v_suppressElabErrors_1369_ = lean_ctor_get_uint8(v_a_1346_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1370_ = lean_ctor_get(v_a_1346_, 13);
v___x_1371_ = 1;
v_ref_1372_ = l_Lean_replaceRef(v_stx_1341_, v_ref_1360_);
lean_dec(v_stx_1341_);
lean_inc_ref(v_inheritedTraceOptions_1370_);
lean_inc(v_cancelTk_x3f_1368_);
lean_inc(v_currMacroScope_1366_);
lean_inc(v_quotContext_1365_);
lean_inc(v_maxHeartbeats_1364_);
lean_inc(v_initHeartbeats_1363_);
lean_inc(v_openDecls_1362_);
lean_inc(v_currNamespace_1361_);
lean_inc(v_maxRecDepth_1359_);
lean_inc(v_currRecDepth_1358_);
lean_inc_ref(v_options_1357_);
lean_inc_ref(v_fileMap_1356_);
lean_inc_ref(v_fileName_1355_);
v___x_1373_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1373_, 0, v_fileName_1355_);
lean_ctor_set(v___x_1373_, 1, v_fileMap_1356_);
lean_ctor_set(v___x_1373_, 2, v_options_1357_);
lean_ctor_set(v___x_1373_, 3, v_currRecDepth_1358_);
lean_ctor_set(v___x_1373_, 4, v_maxRecDepth_1359_);
lean_ctor_set(v___x_1373_, 5, v_ref_1372_);
lean_ctor_set(v___x_1373_, 6, v_currNamespace_1361_);
lean_ctor_set(v___x_1373_, 7, v_openDecls_1362_);
lean_ctor_set(v___x_1373_, 8, v_initHeartbeats_1363_);
lean_ctor_set(v___x_1373_, 9, v_maxHeartbeats_1364_);
lean_ctor_set(v___x_1373_, 10, v_quotContext_1365_);
lean_ctor_set(v___x_1373_, 11, v_currMacroScope_1366_);
lean_ctor_set(v___x_1373_, 12, v_cancelTk_x3f_1368_);
lean_ctor_set(v___x_1373_, 13, v_inheritedTraceOptions_1370_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*14, v_diag_1367_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*14 + 1, v_suppressElabErrors_1369_);
v___x_1374_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1354_, v___x_1371_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_, v___x_1373_, v_a_1347_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; lean_object* v___x_1376_; lean_object* v_a_1377_; lean_object* v___y_1379_; lean_object* v___y_1380_; lean_object* v___y_1381_; lean_object* v___y_1382_; lean_object* v___y_1383_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v___y_1387_; uint8_t v___y_1388_; lean_object* v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1407_; lean_object* v___y_1408_; lean_object* v___y_1409_; lean_object* v___y_1410_; lean_object* v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___y_1421_; lean_object* v___y_1422_; lean_object* v___y_1454_; lean_object* v___y_1455_; lean_object* v___y_1456_; lean_object* v___y_1457_; lean_object* v___y_1458_; lean_object* v___y_1459_; uint8_t v___x_1472_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1375_);
lean_dec_ref_known(v___x_1374_, 1);
v___x_1376_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_a_1375_, v_a_1345_);
v_a_1377_ = lean_ctor_get(v___x_1376_, 0);
lean_inc(v_a_1377_);
lean_dec_ref(v___x_1376_);
v___x_1472_ = l_Lean_Expr_hasSorry(v_a_1377_);
if (v___x_1472_ == 0)
{
v___y_1417_ = v_a_1342_;
v___y_1418_ = v_a_1343_;
v___y_1419_ = v_a_1344_;
v___y_1420_ = v_a_1345_;
v___y_1421_ = v___x_1373_;
v___y_1422_ = v_a_1347_;
goto v___jp_1416_;
}
else
{
uint8_t v___x_1473_; 
v___x_1473_ = l_Lean_Expr_hasSyntheticSorry(v_a_1377_);
if (v___x_1473_ == 0)
{
v___y_1454_ = v_a_1342_;
v___y_1455_ = v_a_1343_;
v___y_1456_ = v_a_1344_;
v___y_1457_ = v_a_1345_;
v___y_1458_ = v___x_1373_;
v___y_1459_ = v_a_1347_;
goto v___jp_1453_;
}
else
{
lean_object* v___x_1474_; lean_object* v_a_1475_; lean_object* v___x_1477_; uint8_t v_isShared_1478_; uint8_t v_isSharedCheck_1482_; 
lean_dec(v_a_1377_);
lean_dec_ref_known(v___x_1373_, 14);
v___x_1474_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1475_ = lean_ctor_get(v___x_1474_, 0);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1474_);
if (v_isSharedCheck_1482_ == 0)
{
v___x_1477_ = v___x_1474_;
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
else
{
lean_inc(v_a_1475_);
lean_dec(v___x_1474_);
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
v___jp_1378_:
{
if (v___y_1388_ == 0)
{
if (lean_obj_tag(v___y_1385_) == 0)
{
lean_dec_ref_known(v___y_1385_, 2);
lean_dec_ref(v___y_1379_);
lean_dec(v_a_1377_);
return v___y_1384_;
}
else
{
lean_object* v_id_1389_; lean_object* v___x_1391_; uint8_t v_isShared_1392_; uint8_t v_isSharedCheck_1402_; 
v_id_1389_ = lean_ctor_get(v___y_1385_, 0);
v_isSharedCheck_1402_ = !lean_is_exclusive(v___y_1385_);
if (v_isSharedCheck_1402_ == 0)
{
lean_object* v_unused_1403_; 
v_unused_1403_ = lean_ctor_get(v___y_1385_, 1);
lean_dec(v_unused_1403_);
v___x_1391_ = v___y_1385_;
v_isShared_1392_ = v_isSharedCheck_1402_;
goto v_resetjp_1390_;
}
else
{
lean_inc(v_id_1389_);
lean_dec(v___y_1385_);
v___x_1391_ = lean_box(0);
v_isShared_1392_ = v_isSharedCheck_1402_;
goto v_resetjp_1390_;
}
v_resetjp_1390_:
{
uint8_t v___x_1393_; 
v___x_1393_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1382_, v_id_1389_);
lean_dec(v_id_1389_);
if (v___x_1393_ == 0)
{
lean_del_object(v___x_1391_);
lean_dec_ref(v___y_1379_);
lean_dec(v_a_1377_);
return v___y_1384_;
}
else
{
lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1398_; 
lean_dec_ref(v___y_1384_);
v___x_1394_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___closed__2);
v___x_1395_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8);
v___x_1396_ = l_Lean_indentExpr(v_a_1377_);
if (v_isShared_1392_ == 0)
{
lean_ctor_set_tag(v___x_1391_, 7);
lean_ctor_set(v___x_1391_, 1, v___x_1396_);
lean_ctor_set(v___x_1391_, 0, v___x_1395_);
v___x_1398_ = v___x_1391_;
goto v_reusejp_1397_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v___x_1395_);
lean_ctor_set(v_reuseFailAlloc_1401_, 1, v___x_1396_);
v___x_1398_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1397_;
}
v_reusejp_1397_:
{
lean_object* v___x_1399_; lean_object* v___x_1400_; 
v___x_1399_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1399_, 0, v___x_1398_);
lean_ctor_set(v___x_1399_, 1, v___x_1394_);
v___x_1400_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1399_, v___y_1381_, v___y_1380_, v___y_1387_, v___y_1383_, v___y_1379_, v___y_1386_);
lean_dec_ref(v___y_1379_);
return v___x_1400_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1385_);
lean_dec_ref(v___y_1379_);
lean_dec(v_a_1377_);
return v___y_1384_;
}
}
v___jp_1404_:
{
lean_object* v___x_1411_; 
lean_inc(v_a_1377_);
v___x_1411_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode_evalExpr(v_a_1377_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
if (lean_obj_tag(v___x_1411_) == 0)
{
lean_dec_ref(v___y_1409_);
lean_dec(v_a_1377_);
return v___x_1411_;
}
else
{
lean_object* v_a_1412_; lean_object* v___x_1413_; uint8_t v___x_1414_; 
v_a_1412_ = lean_ctor_get(v___x_1411_, 0);
lean_inc(v_a_1412_);
v___x_1413_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1414_ = l_Lean_Exception_isInterrupt(v_a_1412_);
if (v___x_1414_ == 0)
{
uint8_t v___x_1415_; 
lean_inc(v_a_1412_);
v___x_1415_ = l_Lean_Exception_isRuntime(v_a_1412_);
v___y_1379_ = v___y_1409_;
v___y_1380_ = v___y_1406_;
v___y_1381_ = v___y_1405_;
v___y_1382_ = v___x_1413_;
v___y_1383_ = v___y_1408_;
v___y_1384_ = v___x_1411_;
v___y_1385_ = v_a_1412_;
v___y_1386_ = v___y_1410_;
v___y_1387_ = v___y_1407_;
v___y_1388_ = v___x_1415_;
goto v___jp_1378_;
}
else
{
v___y_1379_ = v___y_1409_;
v___y_1380_ = v___y_1406_;
v___y_1381_ = v___y_1405_;
v___y_1382_ = v___x_1413_;
v___y_1383_ = v___y_1408_;
v___y_1384_ = v___x_1411_;
v___y_1385_ = v_a_1412_;
v___y_1386_ = v___y_1410_;
v___y_1387_ = v___y_1407_;
v___y_1388_ = v___x_1414_;
goto v___jp_1378_;
}
}
}
v___jp_1416_:
{
lean_object* v___x_1423_; 
lean_inc(v_a_1377_);
v___x_1423_ = l_Lean_Meta_getMVars(v_a_1377_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_);
if (lean_obj_tag(v___x_1423_) == 0)
{
lean_object* v_a_1424_; lean_object* v___x_1425_; 
v_a_1424_ = lean_ctor_get(v___x_1423_, 0);
lean_inc(v_a_1424_);
lean_dec_ref_known(v___x_1423_, 1);
v___x_1425_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1424_, v___x_1351_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_);
lean_dec(v_a_1424_);
if (lean_obj_tag(v___x_1425_) == 0)
{
lean_object* v_a_1426_; uint8_t v___x_1427_; 
v_a_1426_ = lean_ctor_get(v___x_1425_, 0);
lean_inc(v_a_1426_);
lean_dec_ref_known(v___x_1425_, 1);
v___x_1427_ = lean_unbox(v_a_1426_);
lean_dec(v_a_1426_);
if (v___x_1427_ == 0)
{
v___y_1405_ = v___y_1417_;
v___y_1406_ = v___y_1418_;
v___y_1407_ = v___y_1419_;
v___y_1408_ = v___y_1420_;
v___y_1409_ = v___y_1421_;
v___y_1410_ = v___y_1422_;
goto v___jp_1404_;
}
else
{
lean_object* v___x_1428_; lean_object* v_a_1429_; lean_object* v___x_1431_; uint8_t v_isShared_1432_; uint8_t v_isSharedCheck_1436_; 
lean_dec_ref(v___y_1421_);
lean_dec(v_a_1377_);
v___x_1428_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1429_ = lean_ctor_get(v___x_1428_, 0);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1428_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1431_ = v___x_1428_;
v_isShared_1432_ = v_isSharedCheck_1436_;
goto v_resetjp_1430_;
}
else
{
lean_inc(v_a_1429_);
lean_dec(v___x_1428_);
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
else
{
lean_object* v_a_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1444_; 
lean_dec_ref(v___y_1421_);
lean_dec(v_a_1377_);
v_a_1437_ = lean_ctor_get(v___x_1425_, 0);
v_isSharedCheck_1444_ = !lean_is_exclusive(v___x_1425_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1439_ = v___x_1425_;
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
else
{
lean_inc(v_a_1437_);
lean_dec(v___x_1425_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1442_; 
if (v_isShared_1440_ == 0)
{
v___x_1442_ = v___x_1439_;
goto v_reusejp_1441_;
}
else
{
lean_object* v_reuseFailAlloc_1443_; 
v_reuseFailAlloc_1443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1443_, 0, v_a_1437_);
v___x_1442_ = v_reuseFailAlloc_1443_;
goto v_reusejp_1441_;
}
v_reusejp_1441_:
{
return v___x_1442_;
}
}
}
}
else
{
lean_object* v_a_1445_; lean_object* v___x_1447_; uint8_t v_isShared_1448_; uint8_t v_isSharedCheck_1452_; 
lean_dec_ref(v___y_1421_);
lean_dec(v_a_1377_);
v_a_1445_ = lean_ctor_get(v___x_1423_, 0);
v_isSharedCheck_1452_ = !lean_is_exclusive(v___x_1423_);
if (v_isSharedCheck_1452_ == 0)
{
v___x_1447_ = v___x_1423_;
v_isShared_1448_ = v_isSharedCheck_1452_;
goto v_resetjp_1446_;
}
else
{
lean_inc(v_a_1445_);
lean_dec(v___x_1423_);
v___x_1447_ = lean_box(0);
v_isShared_1448_ = v_isSharedCheck_1452_;
goto v_resetjp_1446_;
}
v_resetjp_1446_:
{
lean_object* v___x_1450_; 
if (v_isShared_1448_ == 0)
{
v___x_1450_ = v___x_1447_;
goto v_reusejp_1449_;
}
else
{
lean_object* v_reuseFailAlloc_1451_; 
v_reuseFailAlloc_1451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1451_, 0, v_a_1445_);
v___x_1450_ = v_reuseFailAlloc_1451_;
goto v_reusejp_1449_;
}
v_reusejp_1449_:
{
return v___x_1450_;
}
}
}
}
v___jp_1453_:
{
lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v_a_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1471_; 
v___x_1460_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10);
v___x_1461_ = l_Lean_indentExpr(v_a_1377_);
v___x_1462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1462_, 0, v___x_1460_);
lean_ctor_set(v___x_1462_, 1, v___x_1461_);
v___x_1463_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1462_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_, v___y_1459_);
lean_dec_ref(v___y_1458_);
v_a_1464_ = lean_ctor_get(v___x_1463_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1463_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1466_ = v___x_1463_;
v_isShared_1467_ = v_isSharedCheck_1471_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_a_1464_);
lean_dec(v___x_1463_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1471_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v___x_1469_; 
if (v_isShared_1467_ == 0)
{
v___x_1469_ = v___x_1466_;
goto v_reusejp_1468_;
}
else
{
lean_object* v_reuseFailAlloc_1470_; 
v_reuseFailAlloc_1470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1470_, 0, v_a_1464_);
v___x_1469_ = v_reuseFailAlloc_1470_;
goto v_reusejp_1468_;
}
v_reusejp_1468_:
{
return v___x_1469_;
}
}
}
}
else
{
lean_object* v_a_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1490_; 
lean_dec_ref_known(v___x_1373_, 14);
v_a_1483_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1490_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1490_ == 0)
{
v___x_1485_ = v___x_1374_;
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_a_1483_);
lean_dec(v___x_1374_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1488_; 
if (v_isShared_1486_ == 0)
{
v___x_1488_ = v___x_1485_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v_a_1483_);
v___x_1488_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
return v___x_1488_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object* v_stx_1491_, lean_object* v_a_1492_, lean_object* v_a_1493_, lean_object* v_a_1494_, lean_object* v_a_1495_, lean_object* v_a_1496_, lean_object* v_a_1497_, lean_object* v_a_1498_){
_start:
{
lean_object* v_res_1499_; 
v_res_1499_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2(v_stx_1491_, v_a_1492_, v_a_1493_, v_a_1494_, v_a_1495_, v_a_1496_, v_a_1497_);
lean_dec(v_a_1497_);
lean_dec_ref(v_a_1496_);
lean_dec(v_a_1495_);
lean_dec_ref(v_a_1494_);
lean_dec(v_a_1493_);
lean_dec_ref(v_a_1492_);
return v_res_1499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1(lean_object* v_stx_1500_, lean_object* v_a_1501_, lean_object* v_a_1502_, lean_object* v_a_1503_, lean_object* v_a_1504_, lean_object* v_a_1505_, lean_object* v_a_1506_){
_start:
{
lean_object* v_fileName_1508_; lean_object* v_fileMap_1509_; lean_object* v_options_1510_; lean_object* v_currRecDepth_1511_; lean_object* v_maxRecDepth_1512_; lean_object* v_ref_1513_; lean_object* v_currNamespace_1514_; lean_object* v_openDecls_1515_; lean_object* v_initHeartbeats_1516_; lean_object* v_maxHeartbeats_1517_; lean_object* v_quotContext_1518_; lean_object* v_currMacroScope_1519_; uint8_t v_diag_1520_; lean_object* v_cancelTk_x3f_1521_; uint8_t v_suppressElabErrors_1522_; lean_object* v_inheritedTraceOptions_1523_; lean_object* v_ref_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; 
v_fileName_1508_ = lean_ctor_get(v_a_1505_, 0);
v_fileMap_1509_ = lean_ctor_get(v_a_1505_, 1);
v_options_1510_ = lean_ctor_get(v_a_1505_, 2);
v_currRecDepth_1511_ = lean_ctor_get(v_a_1505_, 3);
v_maxRecDepth_1512_ = lean_ctor_get(v_a_1505_, 4);
v_ref_1513_ = lean_ctor_get(v_a_1505_, 5);
v_currNamespace_1514_ = lean_ctor_get(v_a_1505_, 6);
v_openDecls_1515_ = lean_ctor_get(v_a_1505_, 7);
v_initHeartbeats_1516_ = lean_ctor_get(v_a_1505_, 8);
v_maxHeartbeats_1517_ = lean_ctor_get(v_a_1505_, 9);
v_quotContext_1518_ = lean_ctor_get(v_a_1505_, 10);
v_currMacroScope_1519_ = lean_ctor_get(v_a_1505_, 11);
v_diag_1520_ = lean_ctor_get_uint8(v_a_1505_, sizeof(void*)*14);
v_cancelTk_x3f_1521_ = lean_ctor_get(v_a_1505_, 12);
v_suppressElabErrors_1522_ = lean_ctor_get_uint8(v_a_1505_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1523_ = lean_ctor_get(v_a_1505_, 13);
v_ref_1524_ = l_Lean_replaceRef(v_stx_1500_, v_ref_1513_);
lean_inc_ref(v_inheritedTraceOptions_1523_);
lean_inc(v_cancelTk_x3f_1521_);
lean_inc(v_currMacroScope_1519_);
lean_inc(v_quotContext_1518_);
lean_inc(v_maxHeartbeats_1517_);
lean_inc(v_initHeartbeats_1516_);
lean_inc(v_openDecls_1515_);
lean_inc(v_currNamespace_1514_);
lean_inc(v_maxRecDepth_1512_);
lean_inc(v_currRecDepth_1511_);
lean_inc_ref(v_options_1510_);
lean_inc_ref(v_fileMap_1509_);
lean_inc_ref(v_fileName_1508_);
v___x_1525_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1525_, 0, v_fileName_1508_);
lean_ctor_set(v___x_1525_, 1, v_fileMap_1509_);
lean_ctor_set(v___x_1525_, 2, v_options_1510_);
lean_ctor_set(v___x_1525_, 3, v_currRecDepth_1511_);
lean_ctor_set(v___x_1525_, 4, v_maxRecDepth_1512_);
lean_ctor_set(v___x_1525_, 5, v_ref_1524_);
lean_ctor_set(v___x_1525_, 6, v_currNamespace_1514_);
lean_ctor_set(v___x_1525_, 7, v_openDecls_1515_);
lean_ctor_set(v___x_1525_, 8, v_initHeartbeats_1516_);
lean_ctor_set(v___x_1525_, 9, v_maxHeartbeats_1517_);
lean_ctor_set(v___x_1525_, 10, v_quotContext_1518_);
lean_ctor_set(v___x_1525_, 11, v_currMacroScope_1519_);
lean_ctor_set(v___x_1525_, 12, v_cancelTk_x3f_1521_);
lean_ctor_set(v___x_1525_, 13, v_inheritedTraceOptions_1523_);
lean_ctor_set_uint8(v___x_1525_, sizeof(void*)*14, v_diag_1520_);
lean_ctor_set_uint8(v___x_1525_, sizeof(void*)*14 + 1, v_suppressElabErrors_1522_);
lean_inc(v_stx_1500_);
v___x_1526_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm(v_stx_1500_, v_a_1501_, v_a_1502_, v_a_1503_, v_a_1504_, v___x_1525_, v_a_1506_);
if (lean_obj_tag(v___x_1526_) == 0)
{
lean_object* v_a_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1535_; 
lean_dec_ref_known(v___x_1525_, 14);
lean_dec(v_stx_1500_);
v_a_1527_ = lean_ctor_get(v___x_1526_, 0);
v_isSharedCheck_1535_ = !lean_is_exclusive(v___x_1526_);
if (v_isSharedCheck_1535_ == 0)
{
v___x_1529_ = v___x_1526_;
v_isShared_1530_ = v_isSharedCheck_1535_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_a_1527_);
lean_dec(v___x_1526_);
v___x_1529_ = lean_box(0);
v_isShared_1530_ = v_isSharedCheck_1535_;
goto v_resetjp_1528_;
}
v_resetjp_1528_:
{
lean_object* v_fst_1531_; lean_object* v___x_1533_; 
v_fst_1531_ = lean_ctor_get(v_a_1527_, 0);
lean_inc(v_fst_1531_);
lean_dec(v_a_1527_);
if (v_isShared_1530_ == 0)
{
lean_ctor_set(v___x_1529_, 0, v_fst_1531_);
v___x_1533_ = v___x_1529_;
goto v_reusejp_1532_;
}
else
{
lean_object* v_reuseFailAlloc_1534_; 
v_reuseFailAlloc_1534_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1534_, 0, v_fst_1531_);
v___x_1533_ = v_reuseFailAlloc_1534_;
goto v_reusejp_1532_;
}
v_reusejp_1532_:
{
return v___x_1533_;
}
}
}
else
{
lean_object* v_a_1536_; lean_object* v___x_1538_; uint8_t v_isShared_1539_; uint8_t v_isSharedCheck_1551_; 
v_a_1536_ = lean_ctor_get(v___x_1526_, 0);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1526_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1538_ = v___x_1526_;
v_isShared_1539_ = v_isSharedCheck_1551_;
goto v_resetjp_1537_;
}
else
{
lean_inc(v_a_1536_);
lean_dec(v___x_1526_);
v___x_1538_ = lean_box(0);
v_isShared_1539_ = v_isSharedCheck_1551_;
goto v_resetjp_1537_;
}
v_resetjp_1537_:
{
lean_object* v___x_1540_; lean_object* v___x_1542_; 
v___x_1540_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_1536_);
if (v_isShared_1539_ == 0)
{
v___x_1542_ = v___x_1538_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_a_1536_);
v___x_1542_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
uint8_t v___y_1544_; uint8_t v___x_1548_; 
v___x_1548_ = l_Lean_Exception_isInterrupt(v_a_1536_);
if (v___x_1548_ == 0)
{
uint8_t v___x_1549_; 
lean_inc(v_a_1536_);
v___x_1549_ = l_Lean_Exception_isRuntime(v_a_1536_);
v___y_1544_ = v___x_1549_;
goto v___jp_1543_;
}
else
{
v___y_1544_ = v___x_1548_;
goto v___jp_1543_;
}
v___jp_1543_:
{
if (v___y_1544_ == 0)
{
if (lean_obj_tag(v_a_1536_) == 0)
{
lean_dec_ref_known(v_a_1536_, 2);
lean_dec_ref_known(v___x_1525_, 14);
lean_dec(v_stx_1500_);
return v___x_1542_;
}
else
{
lean_object* v_id_1545_; uint8_t v___x_1546_; 
v_id_1545_ = lean_ctor_get(v_a_1536_, 0);
lean_inc(v_id_1545_);
lean_dec_ref_known(v_a_1536_, 2);
v___x_1546_ = l_Lean_instBEqInternalExceptionId_beq(v___x_1540_, v_id_1545_);
lean_dec(v_id_1545_);
if (v___x_1546_ == 0)
{
lean_dec_ref_known(v___x_1525_, 14);
lean_dec(v_stx_1500_);
return v___x_1542_;
}
else
{
lean_object* v___x_1547_; 
lean_dec_ref(v___x_1542_);
v___x_1547_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1_spec__2(v_stx_1500_, v_a_1501_, v_a_1502_, v_a_1503_, v_a_1504_, v___x_1525_, v_a_1506_);
lean_dec_ref_known(v___x_1525_, 14);
return v___x_1547_;
}
}
}
else
{
lean_dec(v_a_1536_);
lean_dec_ref_known(v___x_1525_, 14);
lean_dec(v_stx_1500_);
return v___x_1542_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1___boxed(lean_object* v_stx_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_, lean_object* v_a_1555_, lean_object* v_a_1556_, lean_object* v_a_1557_, lean_object* v_a_1558_, lean_object* v_a_1559_){
_start:
{
lean_object* v_res_1560_; 
v_res_1560_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1(v_stx_1552_, v_a_1553_, v_a_1554_, v_a_1555_, v_a_1556_, v_a_1557_, v_a_1558_);
lean_dec(v_a_1558_);
lean_dec_ref(v_a_1557_);
lean_dec(v_a_1556_);
lean_dec_ref(v_a_1555_);
lean_dec(v_a_1554_);
lean_dec_ref(v_a_1553_);
return v_res_1560_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; 
v___x_1568_ = lean_box(0);
v___x_1569_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__3));
v___x_1570_ = l_Lean_Expr_const___override(v___x_1569_, v___x_1568_);
return v___x_1570_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_1571_; lean_object* v_ty_x3f_1572_; 
v___x_1571_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4);
v_ty_x3f_1572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_ty_x3f_1572_, 0, v___x_1571_);
return v_ty_x3f_1572_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6(void){
_start:
{
lean_object* v___x_1573_; lean_object* v___x_1574_; 
v___x_1573_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__4);
v___x_1574_ = l_Lean_MessageData_ofExpr(v___x_1573_);
return v___x_1574_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7(void){
_start:
{
lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; 
v___x_1575_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__6);
v___x_1576_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1);
v___x_1577_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1577_, 0, v___x_1576_);
lean_ctor_set(v___x_1577_, 1, v___x_1575_);
return v___x_1577_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8(void){
_start:
{
lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1578_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5);
v___x_1579_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__7);
v___x_1580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1580_, 0, v___x_1579_);
lean_ctor_set(v___x_1580_, 1, v___x_1578_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0(lean_object* v_stx_1581_, lean_object* v_a_1582_, lean_object* v_a_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_, lean_object* v_a_1586_, lean_object* v_a_1587_){
_start:
{
lean_object* v_ty_x3f_1589_; uint8_t v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v_fileName_1595_; lean_object* v_fileMap_1596_; lean_object* v_options_1597_; lean_object* v_currRecDepth_1598_; lean_object* v_maxRecDepth_1599_; lean_object* v_ref_1600_; lean_object* v_currNamespace_1601_; lean_object* v_openDecls_1602_; lean_object* v_initHeartbeats_1603_; lean_object* v_maxHeartbeats_1604_; lean_object* v_quotContext_1605_; lean_object* v_currMacroScope_1606_; uint8_t v_diag_1607_; lean_object* v_cancelTk_x3f_1608_; uint8_t v_suppressElabErrors_1609_; lean_object* v_inheritedTraceOptions_1610_; uint8_t v___x_1611_; lean_object* v_ref_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; 
v_ty_x3f_1589_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_1590_ = 1;
v___x_1591_ = lean_box(0);
v___x_1592_ = lean_box(v___x_1590_);
v___x_1593_ = lean_box(v___x_1590_);
lean_inc(v_stx_1581_);
v___x_1594_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1594_, 0, v_stx_1581_);
lean_closure_set(v___x_1594_, 1, v_ty_x3f_1589_);
lean_closure_set(v___x_1594_, 2, v___x_1592_);
lean_closure_set(v___x_1594_, 3, v___x_1593_);
lean_closure_set(v___x_1594_, 4, v___x_1591_);
v_fileName_1595_ = lean_ctor_get(v_a_1586_, 0);
v_fileMap_1596_ = lean_ctor_get(v_a_1586_, 1);
v_options_1597_ = lean_ctor_get(v_a_1586_, 2);
v_currRecDepth_1598_ = lean_ctor_get(v_a_1586_, 3);
v_maxRecDepth_1599_ = lean_ctor_get(v_a_1586_, 4);
v_ref_1600_ = lean_ctor_get(v_a_1586_, 5);
v_currNamespace_1601_ = lean_ctor_get(v_a_1586_, 6);
v_openDecls_1602_ = lean_ctor_get(v_a_1586_, 7);
v_initHeartbeats_1603_ = lean_ctor_get(v_a_1586_, 8);
v_maxHeartbeats_1604_ = lean_ctor_get(v_a_1586_, 9);
v_quotContext_1605_ = lean_ctor_get(v_a_1586_, 10);
v_currMacroScope_1606_ = lean_ctor_get(v_a_1586_, 11);
v_diag_1607_ = lean_ctor_get_uint8(v_a_1586_, sizeof(void*)*14);
v_cancelTk_x3f_1608_ = lean_ctor_get(v_a_1586_, 12);
v_suppressElabErrors_1609_ = lean_ctor_get_uint8(v_a_1586_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1610_ = lean_ctor_get(v_a_1586_, 13);
v___x_1611_ = 1;
v_ref_1612_ = l_Lean_replaceRef(v_stx_1581_, v_ref_1600_);
lean_dec(v_stx_1581_);
lean_inc_ref(v_inheritedTraceOptions_1610_);
lean_inc(v_cancelTk_x3f_1608_);
lean_inc(v_currMacroScope_1606_);
lean_inc(v_quotContext_1605_);
lean_inc(v_maxHeartbeats_1604_);
lean_inc(v_initHeartbeats_1603_);
lean_inc(v_openDecls_1602_);
lean_inc(v_currNamespace_1601_);
lean_inc(v_maxRecDepth_1599_);
lean_inc(v_currRecDepth_1598_);
lean_inc_ref(v_options_1597_);
lean_inc_ref(v_fileMap_1596_);
lean_inc_ref(v_fileName_1595_);
v___x_1613_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1613_, 0, v_fileName_1595_);
lean_ctor_set(v___x_1613_, 1, v_fileMap_1596_);
lean_ctor_set(v___x_1613_, 2, v_options_1597_);
lean_ctor_set(v___x_1613_, 3, v_currRecDepth_1598_);
lean_ctor_set(v___x_1613_, 4, v_maxRecDepth_1599_);
lean_ctor_set(v___x_1613_, 5, v_ref_1612_);
lean_ctor_set(v___x_1613_, 6, v_currNamespace_1601_);
lean_ctor_set(v___x_1613_, 7, v_openDecls_1602_);
lean_ctor_set(v___x_1613_, 8, v_initHeartbeats_1603_);
lean_ctor_set(v___x_1613_, 9, v_maxHeartbeats_1604_);
lean_ctor_set(v___x_1613_, 10, v_quotContext_1605_);
lean_ctor_set(v___x_1613_, 11, v_currMacroScope_1606_);
lean_ctor_set(v___x_1613_, 12, v_cancelTk_x3f_1608_);
lean_ctor_set(v___x_1613_, 13, v_inheritedTraceOptions_1610_);
lean_ctor_set_uint8(v___x_1613_, sizeof(void*)*14, v_diag_1607_);
lean_ctor_set_uint8(v___x_1613_, sizeof(void*)*14 + 1, v_suppressElabErrors_1609_);
v___x_1614_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1594_, v___x_1611_, v_a_1582_, v_a_1583_, v_a_1584_, v_a_1585_, v___x_1613_, v_a_1587_);
if (lean_obj_tag(v___x_1614_) == 0)
{
lean_object* v_a_1615_; lean_object* v___x_1616_; lean_object* v_a_1617_; lean_object* v___y_1619_; lean_object* v___y_1620_; lean_object* v___y_1621_; lean_object* v___y_1622_; lean_object* v___y_1623_; lean_object* v___y_1624_; lean_object* v___y_1625_; lean_object* v___y_1626_; lean_object* v___y_1627_; uint8_t v___y_1628_; lean_object* v___y_1645_; lean_object* v___y_1646_; lean_object* v___y_1647_; lean_object* v___y_1648_; lean_object* v___y_1649_; lean_object* v___y_1650_; lean_object* v___y_1657_; lean_object* v___y_1658_; lean_object* v___y_1659_; lean_object* v___y_1660_; lean_object* v___y_1661_; lean_object* v___y_1662_; lean_object* v___y_1694_; lean_object* v___y_1695_; lean_object* v___y_1696_; lean_object* v___y_1697_; lean_object* v___y_1698_; lean_object* v___y_1699_; uint8_t v___x_1712_; 
v_a_1615_ = lean_ctor_get(v___x_1614_, 0);
lean_inc(v_a_1615_);
lean_dec_ref_known(v___x_1614_, 1);
v___x_1616_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_a_1615_, v_a_1585_);
v_a_1617_ = lean_ctor_get(v___x_1616_, 0);
lean_inc(v_a_1617_);
lean_dec_ref(v___x_1616_);
v___x_1712_ = l_Lean_Expr_hasSorry(v_a_1617_);
if (v___x_1712_ == 0)
{
v___y_1657_ = v_a_1582_;
v___y_1658_ = v_a_1583_;
v___y_1659_ = v_a_1584_;
v___y_1660_ = v_a_1585_;
v___y_1661_ = v___x_1613_;
v___y_1662_ = v_a_1587_;
goto v___jp_1656_;
}
else
{
uint8_t v___x_1713_; 
v___x_1713_ = l_Lean_Expr_hasSyntheticSorry(v_a_1617_);
if (v___x_1713_ == 0)
{
v___y_1694_ = v_a_1582_;
v___y_1695_ = v_a_1583_;
v___y_1696_ = v_a_1584_;
v___y_1697_ = v_a_1585_;
v___y_1698_ = v___x_1613_;
v___y_1699_ = v_a_1587_;
goto v___jp_1693_;
}
else
{
lean_object* v___x_1714_; lean_object* v_a_1715_; lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1722_; 
lean_dec(v_a_1617_);
lean_dec_ref_known(v___x_1613_, 14);
v___x_1714_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1714_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1717_ = v___x_1714_;
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
else
{
lean_inc(v_a_1715_);
lean_dec(v___x_1714_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
lean_object* v___x_1720_; 
if (v_isShared_1718_ == 0)
{
v___x_1720_ = v___x_1717_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v_a_1715_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
}
v___jp_1618_:
{
if (v___y_1628_ == 0)
{
if (lean_obj_tag(v___y_1626_) == 0)
{
lean_dec_ref_known(v___y_1626_, 2);
lean_dec_ref(v___y_1620_);
lean_dec(v_a_1617_);
return v___y_1624_;
}
else
{
lean_object* v_id_1629_; lean_object* v___x_1631_; uint8_t v_isShared_1632_; uint8_t v_isSharedCheck_1642_; 
v_id_1629_ = lean_ctor_get(v___y_1626_, 0);
v_isSharedCheck_1642_ = !lean_is_exclusive(v___y_1626_);
if (v_isSharedCheck_1642_ == 0)
{
lean_object* v_unused_1643_; 
v_unused_1643_ = lean_ctor_get(v___y_1626_, 1);
lean_dec(v_unused_1643_);
v___x_1631_ = v___y_1626_;
v_isShared_1632_ = v_isSharedCheck_1642_;
goto v_resetjp_1630_;
}
else
{
lean_inc(v_id_1629_);
lean_dec(v___y_1626_);
v___x_1631_ = lean_box(0);
v_isShared_1632_ = v_isSharedCheck_1642_;
goto v_resetjp_1630_;
}
v_resetjp_1630_:
{
uint8_t v___x_1633_; 
v___x_1633_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1619_, v_id_1629_);
lean_dec(v_id_1629_);
if (v___x_1633_ == 0)
{
lean_del_object(v___x_1631_);
lean_dec_ref(v___y_1620_);
lean_dec(v_a_1617_);
return v___y_1624_;
}
else
{
lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1638_; 
lean_dec_ref(v___y_1624_);
v___x_1634_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___closed__8);
v___x_1635_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8);
v___x_1636_ = l_Lean_indentExpr(v_a_1617_);
if (v_isShared_1632_ == 0)
{
lean_ctor_set_tag(v___x_1631_, 7);
lean_ctor_set(v___x_1631_, 1, v___x_1636_);
lean_ctor_set(v___x_1631_, 0, v___x_1635_);
v___x_1638_ = v___x_1631_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1641_; 
v_reuseFailAlloc_1641_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1641_, 0, v___x_1635_);
lean_ctor_set(v_reuseFailAlloc_1641_, 1, v___x_1636_);
v___x_1638_ = v_reuseFailAlloc_1641_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
lean_object* v___x_1639_; lean_object* v___x_1640_; 
v___x_1639_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1639_, 0, v___x_1638_);
lean_ctor_set(v___x_1639_, 1, v___x_1634_);
v___x_1640_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1639_, v___y_1621_, v___y_1623_, v___y_1627_, v___y_1622_, v___y_1620_, v___y_1625_);
lean_dec_ref(v___y_1620_);
return v___x_1640_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1626_);
lean_dec_ref(v___y_1620_);
lean_dec(v_a_1617_);
return v___y_1624_;
}
}
v___jp_1644_:
{
lean_object* v___x_1651_; 
lean_inc(v_a_1617_);
v___x_1651_ = l_Lean_Elab_ConfigEval_instEvalExprTransparencyMode_evalExpr(v_a_1617_, v___y_1647_, v___y_1648_, v___y_1649_, v___y_1650_);
if (lean_obj_tag(v___x_1651_) == 0)
{
lean_dec_ref(v___y_1649_);
lean_dec(v_a_1617_);
return v___x_1651_;
}
else
{
lean_object* v_a_1652_; lean_object* v___x_1653_; uint8_t v___x_1654_; 
v_a_1652_ = lean_ctor_get(v___x_1651_, 0);
lean_inc(v_a_1652_);
v___x_1653_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1654_ = l_Lean_Exception_isInterrupt(v_a_1652_);
if (v___x_1654_ == 0)
{
uint8_t v___x_1655_; 
lean_inc(v_a_1652_);
v___x_1655_ = l_Lean_Exception_isRuntime(v_a_1652_);
v___y_1619_ = v___x_1653_;
v___y_1620_ = v___y_1649_;
v___y_1621_ = v___y_1645_;
v___y_1622_ = v___y_1648_;
v___y_1623_ = v___y_1646_;
v___y_1624_ = v___x_1651_;
v___y_1625_ = v___y_1650_;
v___y_1626_ = v_a_1652_;
v___y_1627_ = v___y_1647_;
v___y_1628_ = v___x_1655_;
goto v___jp_1618_;
}
else
{
v___y_1619_ = v___x_1653_;
v___y_1620_ = v___y_1649_;
v___y_1621_ = v___y_1645_;
v___y_1622_ = v___y_1648_;
v___y_1623_ = v___y_1646_;
v___y_1624_ = v___x_1651_;
v___y_1625_ = v___y_1650_;
v___y_1626_ = v_a_1652_;
v___y_1627_ = v___y_1647_;
v___y_1628_ = v___x_1654_;
goto v___jp_1618_;
}
}
}
v___jp_1656_:
{
lean_object* v___x_1663_; 
lean_inc(v_a_1617_);
v___x_1663_ = l_Lean_Meta_getMVars(v_a_1617_, v___y_1659_, v___y_1660_, v___y_1661_, v___y_1662_);
if (lean_obj_tag(v___x_1663_) == 0)
{
lean_object* v_a_1664_; lean_object* v___x_1665_; 
v_a_1664_ = lean_ctor_get(v___x_1663_, 0);
lean_inc(v_a_1664_);
lean_dec_ref_known(v___x_1663_, 1);
v___x_1665_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1664_, v___x_1591_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_, v___y_1661_, v___y_1662_);
lean_dec(v_a_1664_);
if (lean_obj_tag(v___x_1665_) == 0)
{
lean_object* v_a_1666_; uint8_t v___x_1667_; 
v_a_1666_ = lean_ctor_get(v___x_1665_, 0);
lean_inc(v_a_1666_);
lean_dec_ref_known(v___x_1665_, 1);
v___x_1667_ = lean_unbox(v_a_1666_);
lean_dec(v_a_1666_);
if (v___x_1667_ == 0)
{
v___y_1645_ = v___y_1657_;
v___y_1646_ = v___y_1658_;
v___y_1647_ = v___y_1659_;
v___y_1648_ = v___y_1660_;
v___y_1649_ = v___y_1661_;
v___y_1650_ = v___y_1662_;
goto v___jp_1644_;
}
else
{
lean_object* v___x_1668_; lean_object* v_a_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1676_; 
lean_dec_ref(v___y_1661_);
lean_dec(v_a_1617_);
v___x_1668_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1669_ = lean_ctor_get(v___x_1668_, 0);
v_isSharedCheck_1676_ = !lean_is_exclusive(v___x_1668_);
if (v_isSharedCheck_1676_ == 0)
{
v___x_1671_ = v___x_1668_;
v_isShared_1672_ = v_isSharedCheck_1676_;
goto v_resetjp_1670_;
}
else
{
lean_inc(v_a_1669_);
lean_dec(v___x_1668_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1676_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v___x_1674_; 
if (v_isShared_1672_ == 0)
{
v___x_1674_ = v___x_1671_;
goto v_reusejp_1673_;
}
else
{
lean_object* v_reuseFailAlloc_1675_; 
v_reuseFailAlloc_1675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1675_, 0, v_a_1669_);
v___x_1674_ = v_reuseFailAlloc_1675_;
goto v_reusejp_1673_;
}
v_reusejp_1673_:
{
return v___x_1674_;
}
}
}
}
else
{
lean_object* v_a_1677_; lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1684_; 
lean_dec_ref(v___y_1661_);
lean_dec(v_a_1617_);
v_a_1677_ = lean_ctor_get(v___x_1665_, 0);
v_isSharedCheck_1684_ = !lean_is_exclusive(v___x_1665_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1679_ = v___x_1665_;
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
else
{
lean_inc(v_a_1677_);
lean_dec(v___x_1665_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1684_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1682_; 
if (v_isShared_1680_ == 0)
{
v___x_1682_ = v___x_1679_;
goto v_reusejp_1681_;
}
else
{
lean_object* v_reuseFailAlloc_1683_; 
v_reuseFailAlloc_1683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1683_, 0, v_a_1677_);
v___x_1682_ = v_reuseFailAlloc_1683_;
goto v_reusejp_1681_;
}
v_reusejp_1681_:
{
return v___x_1682_;
}
}
}
}
else
{
lean_object* v_a_1685_; lean_object* v___x_1687_; uint8_t v_isShared_1688_; uint8_t v_isSharedCheck_1692_; 
lean_dec_ref(v___y_1661_);
lean_dec(v_a_1617_);
v_a_1685_ = lean_ctor_get(v___x_1663_, 0);
v_isSharedCheck_1692_ = !lean_is_exclusive(v___x_1663_);
if (v_isSharedCheck_1692_ == 0)
{
v___x_1687_ = v___x_1663_;
v_isShared_1688_ = v_isSharedCheck_1692_;
goto v_resetjp_1686_;
}
else
{
lean_inc(v_a_1685_);
lean_dec(v___x_1663_);
v___x_1687_ = lean_box(0);
v_isShared_1688_ = v_isSharedCheck_1692_;
goto v_resetjp_1686_;
}
v_resetjp_1686_:
{
lean_object* v___x_1690_; 
if (v_isShared_1688_ == 0)
{
v___x_1690_ = v___x_1687_;
goto v_reusejp_1689_;
}
else
{
lean_object* v_reuseFailAlloc_1691_; 
v_reuseFailAlloc_1691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1691_, 0, v_a_1685_);
v___x_1690_ = v_reuseFailAlloc_1691_;
goto v_reusejp_1689_;
}
v_reusejp_1689_:
{
return v___x_1690_;
}
}
}
}
v___jp_1693_:
{
lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v_a_1704_; lean_object* v___x_1706_; uint8_t v_isShared_1707_; uint8_t v_isSharedCheck_1711_; 
v___x_1700_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10);
v___x_1701_ = l_Lean_indentExpr(v_a_1617_);
v___x_1702_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1700_);
lean_ctor_set(v___x_1702_, 1, v___x_1701_);
v___x_1703_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1702_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_, v___y_1699_);
lean_dec_ref(v___y_1698_);
v_a_1704_ = lean_ctor_get(v___x_1703_, 0);
v_isSharedCheck_1711_ = !lean_is_exclusive(v___x_1703_);
if (v_isSharedCheck_1711_ == 0)
{
v___x_1706_ = v___x_1703_;
v_isShared_1707_ = v_isSharedCheck_1711_;
goto v_resetjp_1705_;
}
else
{
lean_inc(v_a_1704_);
lean_dec(v___x_1703_);
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
else
{
lean_object* v_a_1723_; lean_object* v___x_1725_; uint8_t v_isShared_1726_; uint8_t v_isSharedCheck_1730_; 
lean_dec_ref_known(v___x_1613_, 14);
v_a_1723_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1730_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1730_ == 0)
{
v___x_1725_ = v___x_1614_;
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
else
{
lean_inc(v_a_1723_);
lean_dec(v___x_1614_);
v___x_1725_ = lean_box(0);
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
v_resetjp_1724_:
{
lean_object* v___x_1728_; 
if (v_isShared_1726_ == 0)
{
v___x_1728_ = v___x_1725_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_a_1723_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_stx_1731_, lean_object* v_a_1732_, lean_object* v_a_1733_, lean_object* v_a_1734_, lean_object* v_a_1735_, lean_object* v_a_1736_, lean_object* v_a_1737_, lean_object* v_a_1738_){
_start:
{
lean_object* v_res_1739_; 
v_res_1739_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0(v_stx_1731_, v_a_1732_, v_a_1733_, v_a_1734_, v_a_1735_, v_a_1736_, v_a_1737_);
lean_dec(v_a_1737_);
lean_dec_ref(v_a_1736_);
lean_dec(v_a_1735_);
lean_dec_ref(v_a_1734_);
lean_dec(v_a_1733_);
lean_dec_ref(v_a_1732_);
return v_res_1739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0(lean_object* v_stx_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_){
_start:
{
lean_object* v_fileName_1748_; lean_object* v_fileMap_1749_; lean_object* v_options_1750_; lean_object* v_currRecDepth_1751_; lean_object* v_maxRecDepth_1752_; lean_object* v_ref_1753_; lean_object* v_currNamespace_1754_; lean_object* v_openDecls_1755_; lean_object* v_initHeartbeats_1756_; lean_object* v_maxHeartbeats_1757_; lean_object* v_quotContext_1758_; lean_object* v_currMacroScope_1759_; uint8_t v_diag_1760_; lean_object* v_cancelTk_x3f_1761_; uint8_t v_suppressElabErrors_1762_; lean_object* v_inheritedTraceOptions_1763_; lean_object* v_ref_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; 
v_fileName_1748_ = lean_ctor_get(v_a_1745_, 0);
v_fileMap_1749_ = lean_ctor_get(v_a_1745_, 1);
v_options_1750_ = lean_ctor_get(v_a_1745_, 2);
v_currRecDepth_1751_ = lean_ctor_get(v_a_1745_, 3);
v_maxRecDepth_1752_ = lean_ctor_get(v_a_1745_, 4);
v_ref_1753_ = lean_ctor_get(v_a_1745_, 5);
v_currNamespace_1754_ = lean_ctor_get(v_a_1745_, 6);
v_openDecls_1755_ = lean_ctor_get(v_a_1745_, 7);
v_initHeartbeats_1756_ = lean_ctor_get(v_a_1745_, 8);
v_maxHeartbeats_1757_ = lean_ctor_get(v_a_1745_, 9);
v_quotContext_1758_ = lean_ctor_get(v_a_1745_, 10);
v_currMacroScope_1759_ = lean_ctor_get(v_a_1745_, 11);
v_diag_1760_ = lean_ctor_get_uint8(v_a_1745_, sizeof(void*)*14);
v_cancelTk_x3f_1761_ = lean_ctor_get(v_a_1745_, 12);
v_suppressElabErrors_1762_ = lean_ctor_get_uint8(v_a_1745_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1763_ = lean_ctor_get(v_a_1745_, 13);
v_ref_1764_ = l_Lean_replaceRef(v_stx_1740_, v_ref_1753_);
lean_inc_ref(v_inheritedTraceOptions_1763_);
lean_inc(v_cancelTk_x3f_1761_);
lean_inc(v_currMacroScope_1759_);
lean_inc(v_quotContext_1758_);
lean_inc(v_maxHeartbeats_1757_);
lean_inc(v_initHeartbeats_1756_);
lean_inc(v_openDecls_1755_);
lean_inc(v_currNamespace_1754_);
lean_inc(v_maxRecDepth_1752_);
lean_inc(v_currRecDepth_1751_);
lean_inc_ref(v_options_1750_);
lean_inc_ref(v_fileMap_1749_);
lean_inc_ref(v_fileName_1748_);
v___x_1765_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1765_, 0, v_fileName_1748_);
lean_ctor_set(v___x_1765_, 1, v_fileMap_1749_);
lean_ctor_set(v___x_1765_, 2, v_options_1750_);
lean_ctor_set(v___x_1765_, 3, v_currRecDepth_1751_);
lean_ctor_set(v___x_1765_, 4, v_maxRecDepth_1752_);
lean_ctor_set(v___x_1765_, 5, v_ref_1764_);
lean_ctor_set(v___x_1765_, 6, v_currNamespace_1754_);
lean_ctor_set(v___x_1765_, 7, v_openDecls_1755_);
lean_ctor_set(v___x_1765_, 8, v_initHeartbeats_1756_);
lean_ctor_set(v___x_1765_, 9, v_maxHeartbeats_1757_);
lean_ctor_set(v___x_1765_, 10, v_quotContext_1758_);
lean_ctor_set(v___x_1765_, 11, v_currMacroScope_1759_);
lean_ctor_set(v___x_1765_, 12, v_cancelTk_x3f_1761_);
lean_ctor_set(v___x_1765_, 13, v_inheritedTraceOptions_1763_);
lean_ctor_set_uint8(v___x_1765_, sizeof(void*)*14, v_diag_1760_);
lean_ctor_set_uint8(v___x_1765_, sizeof(void*)*14 + 1, v_suppressElabErrors_1762_);
lean_inc(v_stx_1740_);
v___x_1766_ = l_Lean_Elab_ConfigEval_instEvalTermTransparencyMode_evalTerm(v_stx_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v___x_1765_, v_a_1746_);
if (lean_obj_tag(v___x_1766_) == 0)
{
lean_object* v_a_1767_; lean_object* v___x_1769_; uint8_t v_isShared_1770_; uint8_t v_isSharedCheck_1775_; 
lean_dec_ref_known(v___x_1765_, 14);
lean_dec(v_stx_1740_);
v_a_1767_ = lean_ctor_get(v___x_1766_, 0);
v_isSharedCheck_1775_ = !lean_is_exclusive(v___x_1766_);
if (v_isSharedCheck_1775_ == 0)
{
v___x_1769_ = v___x_1766_;
v_isShared_1770_ = v_isSharedCheck_1775_;
goto v_resetjp_1768_;
}
else
{
lean_inc(v_a_1767_);
lean_dec(v___x_1766_);
v___x_1769_ = lean_box(0);
v_isShared_1770_ = v_isSharedCheck_1775_;
goto v_resetjp_1768_;
}
v_resetjp_1768_:
{
lean_object* v_fst_1771_; lean_object* v___x_1773_; 
v_fst_1771_ = lean_ctor_get(v_a_1767_, 0);
lean_inc(v_fst_1771_);
lean_dec(v_a_1767_);
if (v_isShared_1770_ == 0)
{
lean_ctor_set(v___x_1769_, 0, v_fst_1771_);
v___x_1773_ = v___x_1769_;
goto v_reusejp_1772_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v_fst_1771_);
v___x_1773_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1772_;
}
v_reusejp_1772_:
{
return v___x_1773_;
}
}
}
else
{
lean_object* v_a_1776_; lean_object* v___x_1778_; uint8_t v_isShared_1779_; uint8_t v_isSharedCheck_1791_; 
v_a_1776_ = lean_ctor_get(v___x_1766_, 0);
v_isSharedCheck_1791_ = !lean_is_exclusive(v___x_1766_);
if (v_isSharedCheck_1791_ == 0)
{
v___x_1778_ = v___x_1766_;
v_isShared_1779_ = v_isSharedCheck_1791_;
goto v_resetjp_1777_;
}
else
{
lean_inc(v_a_1776_);
lean_dec(v___x_1766_);
v___x_1778_ = lean_box(0);
v_isShared_1779_ = v_isSharedCheck_1791_;
goto v_resetjp_1777_;
}
v_resetjp_1777_:
{
lean_object* v___x_1780_; lean_object* v___x_1782_; 
v___x_1780_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_1776_);
if (v_isShared_1779_ == 0)
{
v___x_1782_ = v___x_1778_;
goto v_reusejp_1781_;
}
else
{
lean_object* v_reuseFailAlloc_1790_; 
v_reuseFailAlloc_1790_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1790_, 0, v_a_1776_);
v___x_1782_ = v_reuseFailAlloc_1790_;
goto v_reusejp_1781_;
}
v_reusejp_1781_:
{
uint8_t v___y_1784_; uint8_t v___x_1788_; 
v___x_1788_ = l_Lean_Exception_isInterrupt(v_a_1776_);
if (v___x_1788_ == 0)
{
uint8_t v___x_1789_; 
lean_inc(v_a_1776_);
v___x_1789_ = l_Lean_Exception_isRuntime(v_a_1776_);
v___y_1784_ = v___x_1789_;
goto v___jp_1783_;
}
else
{
v___y_1784_ = v___x_1788_;
goto v___jp_1783_;
}
v___jp_1783_:
{
if (v___y_1784_ == 0)
{
if (lean_obj_tag(v_a_1776_) == 0)
{
lean_dec_ref_known(v_a_1776_, 2);
lean_dec_ref_known(v___x_1765_, 14);
lean_dec(v_stx_1740_);
return v___x_1782_;
}
else
{
lean_object* v_id_1785_; uint8_t v___x_1786_; 
v_id_1785_ = lean_ctor_get(v_a_1776_, 0);
lean_inc(v_id_1785_);
lean_dec_ref_known(v_a_1776_, 2);
v___x_1786_ = l_Lean_instBEqInternalExceptionId_beq(v___x_1780_, v_id_1785_);
lean_dec(v_id_1785_);
if (v___x_1786_ == 0)
{
lean_dec_ref_known(v___x_1765_, 14);
lean_dec(v_stx_1740_);
return v___x_1782_;
}
else
{
lean_object* v___x_1787_; 
lean_dec_ref(v___x_1782_);
v___x_1787_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0_spec__0(v_stx_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v___x_1765_, v_a_1746_);
lean_dec_ref_known(v___x_1765_, 14);
return v___x_1787_;
}
}
}
else
{
lean_dec(v_a_1776_);
lean_dec_ref_known(v___x_1765_, 14);
lean_dec(v_stx_1740_);
return v___x_1782_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_1792_, lean_object* v_a_1793_, lean_object* v_a_1794_, lean_object* v_a_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_, lean_object* v_a_1798_, lean_object* v_a_1799_){
_start:
{
lean_object* v_res_1800_; 
v_res_1800_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0(v_stx_1792_, v_a_1793_, v_a_1794_, v_a_1795_, v_a_1796_, v_a_1797_, v_a_1798_);
lean_dec(v_a_1798_);
lean_dec_ref(v_a_1797_);
lean_dec(v_a_1796_);
lean_dec_ref(v_a_1795_);
lean_dec(v_a_1794_);
lean_dec_ref(v_a_1793_);
return v_res_1800_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_1801_; lean_object* v___x_1802_; 
v___x_1801_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged___closed__1);
v___x_1802_ = l_Lean_MessageData_ofExpr(v___x_1801_);
return v___x_1802_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1(void){
_start:
{
lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; 
v___x_1803_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__0);
v___x_1804_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__1);
v___x_1805_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1805_, 0, v___x_1804_);
lean_ctor_set(v___x_1805_, 1, v___x_1803_);
return v___x_1805_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2(void){
_start:
{
lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; 
v___x_1806_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__5);
v___x_1807_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__1);
v___x_1808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1807_);
lean_ctor_set(v___x_1808_, 1, v___x_1806_);
return v___x_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4(lean_object* v_stx_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_, lean_object* v_a_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_, lean_object* v_a_1815_){
_start:
{
lean_object* v_ty_x3f_1817_; uint8_t v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v_fileName_1823_; lean_object* v_fileMap_1824_; lean_object* v_options_1825_; lean_object* v_currRecDepth_1826_; lean_object* v_maxRecDepth_1827_; lean_object* v_ref_1828_; lean_object* v_currNamespace_1829_; lean_object* v_openDecls_1830_; lean_object* v_initHeartbeats_1831_; lean_object* v_maxHeartbeats_1832_; lean_object* v_quotContext_1833_; lean_object* v_currMacroScope_1834_; uint8_t v_diag_1835_; lean_object* v_cancelTk_x3f_1836_; uint8_t v_suppressElabErrors_1837_; lean_object* v_inheritedTraceOptions_1838_; uint8_t v___x_1839_; lean_object* v_ref_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; 
v_ty_x3f_1817_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged___closed__1);
v___x_1818_ = 1;
v___x_1819_ = lean_box(0);
v___x_1820_ = lean_box(v___x_1818_);
v___x_1821_ = lean_box(v___x_1818_);
lean_inc(v_stx_1809_);
v___x_1822_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1822_, 0, v_stx_1809_);
lean_closure_set(v___x_1822_, 1, v_ty_x3f_1817_);
lean_closure_set(v___x_1822_, 2, v___x_1820_);
lean_closure_set(v___x_1822_, 3, v___x_1821_);
lean_closure_set(v___x_1822_, 4, v___x_1819_);
v_fileName_1823_ = lean_ctor_get(v_a_1814_, 0);
v_fileMap_1824_ = lean_ctor_get(v_a_1814_, 1);
v_options_1825_ = lean_ctor_get(v_a_1814_, 2);
v_currRecDepth_1826_ = lean_ctor_get(v_a_1814_, 3);
v_maxRecDepth_1827_ = lean_ctor_get(v_a_1814_, 4);
v_ref_1828_ = lean_ctor_get(v_a_1814_, 5);
v_currNamespace_1829_ = lean_ctor_get(v_a_1814_, 6);
v_openDecls_1830_ = lean_ctor_get(v_a_1814_, 7);
v_initHeartbeats_1831_ = lean_ctor_get(v_a_1814_, 8);
v_maxHeartbeats_1832_ = lean_ctor_get(v_a_1814_, 9);
v_quotContext_1833_ = lean_ctor_get(v_a_1814_, 10);
v_currMacroScope_1834_ = lean_ctor_get(v_a_1814_, 11);
v_diag_1835_ = lean_ctor_get_uint8(v_a_1814_, sizeof(void*)*14);
v_cancelTk_x3f_1836_ = lean_ctor_get(v_a_1814_, 12);
v_suppressElabErrors_1837_ = lean_ctor_get_uint8(v_a_1814_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1838_ = lean_ctor_get(v_a_1814_, 13);
v___x_1839_ = 1;
v_ref_1840_ = l_Lean_replaceRef(v_stx_1809_, v_ref_1828_);
lean_dec(v_stx_1809_);
lean_inc_ref(v_inheritedTraceOptions_1838_);
lean_inc(v_cancelTk_x3f_1836_);
lean_inc(v_currMacroScope_1834_);
lean_inc(v_quotContext_1833_);
lean_inc(v_maxHeartbeats_1832_);
lean_inc(v_initHeartbeats_1831_);
lean_inc(v_openDecls_1830_);
lean_inc(v_currNamespace_1829_);
lean_inc(v_maxRecDepth_1827_);
lean_inc(v_currRecDepth_1826_);
lean_inc_ref(v_options_1825_);
lean_inc_ref(v_fileMap_1824_);
lean_inc_ref(v_fileName_1823_);
v___x_1841_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1841_, 0, v_fileName_1823_);
lean_ctor_set(v___x_1841_, 1, v_fileMap_1824_);
lean_ctor_set(v___x_1841_, 2, v_options_1825_);
lean_ctor_set(v___x_1841_, 3, v_currRecDepth_1826_);
lean_ctor_set(v___x_1841_, 4, v_maxRecDepth_1827_);
lean_ctor_set(v___x_1841_, 5, v_ref_1840_);
lean_ctor_set(v___x_1841_, 6, v_currNamespace_1829_);
lean_ctor_set(v___x_1841_, 7, v_openDecls_1830_);
lean_ctor_set(v___x_1841_, 8, v_initHeartbeats_1831_);
lean_ctor_set(v___x_1841_, 9, v_maxHeartbeats_1832_);
lean_ctor_set(v___x_1841_, 10, v_quotContext_1833_);
lean_ctor_set(v___x_1841_, 11, v_currMacroScope_1834_);
lean_ctor_set(v___x_1841_, 12, v_cancelTk_x3f_1836_);
lean_ctor_set(v___x_1841_, 13, v_inheritedTraceOptions_1838_);
lean_ctor_set_uint8(v___x_1841_, sizeof(void*)*14, v_diag_1835_);
lean_ctor_set_uint8(v___x_1841_, sizeof(void*)*14 + 1, v_suppressElabErrors_1837_);
v___x_1842_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1822_, v___x_1839_, v_a_1810_, v_a_1811_, v_a_1812_, v_a_1813_, v___x_1841_, v_a_1815_);
if (lean_obj_tag(v___x_1842_) == 0)
{
lean_object* v_a_1843_; lean_object* v___x_1844_; lean_object* v_a_1845_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; lean_object* v___y_1851_; lean_object* v___y_1852_; lean_object* v___y_1853_; lean_object* v___y_1854_; lean_object* v___y_1855_; uint8_t v___y_1856_; lean_object* v___y_1873_; lean_object* v___y_1874_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1878_; lean_object* v___y_1885_; lean_object* v___y_1886_; lean_object* v___y_1887_; lean_object* v___y_1888_; lean_object* v___y_1889_; lean_object* v___y_1890_; lean_object* v___y_1922_; lean_object* v___y_1923_; lean_object* v___y_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; uint8_t v___x_1940_; 
v_a_1843_ = lean_ctor_get(v___x_1842_, 0);
lean_inc(v_a_1843_);
lean_dec_ref_known(v___x_1842_, 1);
v___x_1844_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_a_1843_, v_a_1813_);
v_a_1845_ = lean_ctor_get(v___x_1844_, 0);
lean_inc(v_a_1845_);
lean_dec_ref(v___x_1844_);
v___x_1940_ = l_Lean_Expr_hasSorry(v_a_1845_);
if (v___x_1940_ == 0)
{
v___y_1885_ = v_a_1810_;
v___y_1886_ = v_a_1811_;
v___y_1887_ = v_a_1812_;
v___y_1888_ = v_a_1813_;
v___y_1889_ = v___x_1841_;
v___y_1890_ = v_a_1815_;
goto v___jp_1884_;
}
else
{
uint8_t v___x_1941_; 
v___x_1941_ = l_Lean_Expr_hasSyntheticSorry(v_a_1845_);
if (v___x_1941_ == 0)
{
v___y_1922_ = v_a_1810_;
v___y_1923_ = v_a_1811_;
v___y_1924_ = v_a_1812_;
v___y_1925_ = v_a_1813_;
v___y_1926_ = v___x_1841_;
v___y_1927_ = v_a_1815_;
goto v___jp_1921_;
}
else
{
lean_object* v___x_1942_; lean_object* v_a_1943_; lean_object* v___x_1945_; uint8_t v_isShared_1946_; uint8_t v_isSharedCheck_1950_; 
lean_dec(v_a_1845_);
lean_dec_ref_known(v___x_1841_, 14);
v___x_1942_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1943_ = lean_ctor_get(v___x_1942_, 0);
v_isSharedCheck_1950_ = !lean_is_exclusive(v___x_1942_);
if (v_isSharedCheck_1950_ == 0)
{
v___x_1945_ = v___x_1942_;
v_isShared_1946_ = v_isSharedCheck_1950_;
goto v_resetjp_1944_;
}
else
{
lean_inc(v_a_1943_);
lean_dec(v___x_1942_);
v___x_1945_ = lean_box(0);
v_isShared_1946_ = v_isSharedCheck_1950_;
goto v_resetjp_1944_;
}
v_resetjp_1944_:
{
lean_object* v___x_1948_; 
if (v_isShared_1946_ == 0)
{
v___x_1948_ = v___x_1945_;
goto v_reusejp_1947_;
}
else
{
lean_object* v_reuseFailAlloc_1949_; 
v_reuseFailAlloc_1949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1949_, 0, v_a_1943_);
v___x_1948_ = v_reuseFailAlloc_1949_;
goto v_reusejp_1947_;
}
v_reusejp_1947_:
{
return v___x_1948_;
}
}
}
}
v___jp_1846_:
{
if (v___y_1856_ == 0)
{
if (lean_obj_tag(v___y_1855_) == 0)
{
lean_dec_ref_known(v___y_1855_, 2);
lean_dec_ref(v___y_1853_);
lean_dec(v_a_1845_);
return v___y_1850_;
}
else
{
lean_object* v_id_1857_; lean_object* v___x_1859_; uint8_t v_isShared_1860_; uint8_t v_isSharedCheck_1870_; 
v_id_1857_ = lean_ctor_get(v___y_1855_, 0);
v_isSharedCheck_1870_ = !lean_is_exclusive(v___y_1855_);
if (v_isSharedCheck_1870_ == 0)
{
lean_object* v_unused_1871_; 
v_unused_1871_ = lean_ctor_get(v___y_1855_, 1);
lean_dec(v_unused_1871_);
v___x_1859_ = v___y_1855_;
v_isShared_1860_ = v_isSharedCheck_1870_;
goto v_resetjp_1858_;
}
else
{
lean_inc(v_id_1857_);
lean_dec(v___y_1855_);
v___x_1859_ = lean_box(0);
v_isShared_1860_ = v_isSharedCheck_1870_;
goto v_resetjp_1858_;
}
v_resetjp_1858_:
{
uint8_t v___x_1861_; 
v___x_1861_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1852_, v_id_1857_);
lean_dec(v_id_1857_);
if (v___x_1861_ == 0)
{
lean_del_object(v___x_1859_);
lean_dec_ref(v___y_1853_);
lean_dec(v_a_1845_);
return v___y_1850_;
}
else
{
lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1866_; 
lean_dec_ref(v___y_1850_);
v___x_1862_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___closed__2);
v___x_1863_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__8);
v___x_1864_ = l_Lean_indentExpr(v_a_1845_);
if (v_isShared_1860_ == 0)
{
lean_ctor_set_tag(v___x_1859_, 7);
lean_ctor_set(v___x_1859_, 1, v___x_1864_);
lean_ctor_set(v___x_1859_, 0, v___x_1863_);
v___x_1866_ = v___x_1859_;
goto v_reusejp_1865_;
}
else
{
lean_object* v_reuseFailAlloc_1869_; 
v_reuseFailAlloc_1869_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1869_, 0, v___x_1863_);
lean_ctor_set(v_reuseFailAlloc_1869_, 1, v___x_1864_);
v___x_1866_ = v_reuseFailAlloc_1869_;
goto v_reusejp_1865_;
}
v_reusejp_1865_:
{
lean_object* v___x_1867_; lean_object* v___x_1868_; 
v___x_1867_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___x_1866_);
lean_ctor_set(v___x_1867_, 1, v___x_1862_);
v___x_1868_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1867_, v___y_1848_, v___y_1847_, v___y_1851_, v___y_1849_, v___y_1853_, v___y_1854_);
lean_dec_ref(v___y_1853_);
return v___x_1868_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1855_);
lean_dec_ref(v___y_1853_);
lean_dec(v_a_1845_);
return v___y_1850_;
}
}
v___jp_1872_:
{
lean_object* v___x_1879_; 
lean_inc(v_a_1845_);
v___x_1879_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged_evalExpr(v_a_1845_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_);
if (lean_obj_tag(v___x_1879_) == 0)
{
lean_dec_ref(v___y_1877_);
lean_dec(v_a_1845_);
return v___x_1879_;
}
else
{
lean_object* v_a_1880_; lean_object* v___x_1881_; uint8_t v___x_1882_; 
v_a_1880_ = lean_ctor_get(v___x_1879_, 0);
lean_inc(v_a_1880_);
v___x_1881_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1882_ = l_Lean_Exception_isInterrupt(v_a_1880_);
if (v___x_1882_ == 0)
{
uint8_t v___x_1883_; 
lean_inc(v_a_1880_);
v___x_1883_ = l_Lean_Exception_isRuntime(v_a_1880_);
v___y_1847_ = v___y_1874_;
v___y_1848_ = v___y_1873_;
v___y_1849_ = v___y_1876_;
v___y_1850_ = v___x_1879_;
v___y_1851_ = v___y_1875_;
v___y_1852_ = v___x_1881_;
v___y_1853_ = v___y_1877_;
v___y_1854_ = v___y_1878_;
v___y_1855_ = v_a_1880_;
v___y_1856_ = v___x_1883_;
goto v___jp_1846_;
}
else
{
v___y_1847_ = v___y_1874_;
v___y_1848_ = v___y_1873_;
v___y_1849_ = v___y_1876_;
v___y_1850_ = v___x_1879_;
v___y_1851_ = v___y_1875_;
v___y_1852_ = v___x_1881_;
v___y_1853_ = v___y_1877_;
v___y_1854_ = v___y_1878_;
v___y_1855_ = v_a_1880_;
v___y_1856_ = v___x_1882_;
goto v___jp_1846_;
}
}
}
v___jp_1884_:
{
lean_object* v___x_1891_; 
lean_inc(v_a_1845_);
v___x_1891_ = l_Lean_Meta_getMVars(v_a_1845_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_);
if (lean_obj_tag(v___x_1891_) == 0)
{
lean_object* v_a_1892_; lean_object* v___x_1893_; 
v_a_1892_ = lean_ctor_get(v___x_1891_, 0);
lean_inc(v_a_1892_);
lean_dec_ref_known(v___x_1891_, 1);
v___x_1893_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1892_, v___x_1819_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_);
lean_dec(v_a_1892_);
if (lean_obj_tag(v___x_1893_) == 0)
{
lean_object* v_a_1894_; uint8_t v___x_1895_; 
v_a_1894_ = lean_ctor_get(v___x_1893_, 0);
lean_inc(v_a_1894_);
lean_dec_ref_known(v___x_1893_, 1);
v___x_1895_ = lean_unbox(v_a_1894_);
lean_dec(v_a_1894_);
if (v___x_1895_ == 0)
{
v___y_1873_ = v___y_1885_;
v___y_1874_ = v___y_1886_;
v___y_1875_ = v___y_1887_;
v___y_1876_ = v___y_1888_;
v___y_1877_ = v___y_1889_;
v___y_1878_ = v___y_1890_;
goto v___jp_1872_;
}
else
{
lean_object* v___x_1896_; lean_object* v_a_1897_; lean_object* v___x_1899_; uint8_t v_isShared_1900_; uint8_t v_isSharedCheck_1904_; 
lean_dec_ref(v___y_1889_);
lean_dec(v_a_1845_);
v___x_1896_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
v_a_1897_ = lean_ctor_get(v___x_1896_, 0);
v_isSharedCheck_1904_ = !lean_is_exclusive(v___x_1896_);
if (v_isSharedCheck_1904_ == 0)
{
v___x_1899_ = v___x_1896_;
v_isShared_1900_ = v_isSharedCheck_1904_;
goto v_resetjp_1898_;
}
else
{
lean_inc(v_a_1897_);
lean_dec(v___x_1896_);
v___x_1899_ = lean_box(0);
v_isShared_1900_ = v_isSharedCheck_1904_;
goto v_resetjp_1898_;
}
v_resetjp_1898_:
{
lean_object* v___x_1902_; 
if (v_isShared_1900_ == 0)
{
v___x_1902_ = v___x_1899_;
goto v_reusejp_1901_;
}
else
{
lean_object* v_reuseFailAlloc_1903_; 
v_reuseFailAlloc_1903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1903_, 0, v_a_1897_);
v___x_1902_ = v_reuseFailAlloc_1903_;
goto v_reusejp_1901_;
}
v_reusejp_1901_:
{
return v___x_1902_;
}
}
}
}
else
{
lean_object* v_a_1905_; lean_object* v___x_1907_; uint8_t v_isShared_1908_; uint8_t v_isSharedCheck_1912_; 
lean_dec_ref(v___y_1889_);
lean_dec(v_a_1845_);
v_a_1905_ = lean_ctor_get(v___x_1893_, 0);
v_isSharedCheck_1912_ = !lean_is_exclusive(v___x_1893_);
if (v_isSharedCheck_1912_ == 0)
{
v___x_1907_ = v___x_1893_;
v_isShared_1908_ = v_isSharedCheck_1912_;
goto v_resetjp_1906_;
}
else
{
lean_inc(v_a_1905_);
lean_dec(v___x_1893_);
v___x_1907_ = lean_box(0);
v_isShared_1908_ = v_isSharedCheck_1912_;
goto v_resetjp_1906_;
}
v_resetjp_1906_:
{
lean_object* v___x_1910_; 
if (v_isShared_1908_ == 0)
{
v___x_1910_ = v___x_1907_;
goto v_reusejp_1909_;
}
else
{
lean_object* v_reuseFailAlloc_1911_; 
v_reuseFailAlloc_1911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1911_, 0, v_a_1905_);
v___x_1910_ = v_reuseFailAlloc_1911_;
goto v_reusejp_1909_;
}
v_reusejp_1909_:
{
return v___x_1910_;
}
}
}
}
else
{
lean_object* v_a_1913_; lean_object* v___x_1915_; uint8_t v_isShared_1916_; uint8_t v_isSharedCheck_1920_; 
lean_dec_ref(v___y_1889_);
lean_dec(v_a_1845_);
v_a_1913_ = lean_ctor_get(v___x_1891_, 0);
v_isSharedCheck_1920_ = !lean_is_exclusive(v___x_1891_);
if (v_isSharedCheck_1920_ == 0)
{
v___x_1915_ = v___x_1891_;
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
else
{
lean_inc(v_a_1913_);
lean_dec(v___x_1891_);
v___x_1915_ = lean_box(0);
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
v_resetjp_1914_:
{
lean_object* v___x_1918_; 
if (v_isShared_1916_ == 0)
{
v___x_1918_ = v___x_1915_;
goto v_reusejp_1917_;
}
else
{
lean_object* v_reuseFailAlloc_1919_; 
v_reuseFailAlloc_1919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1919_, 0, v_a_1913_);
v___x_1918_ = v_reuseFailAlloc_1919_;
goto v_reusejp_1917_;
}
v_reusejp_1917_:
{
return v___x_1918_;
}
}
}
}
v___jp_1921_:
{
lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v_a_1932_; lean_object* v___x_1934_; uint8_t v_isShared_1935_; uint8_t v_isSharedCheck_1939_; 
v___x_1928_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3___closed__10);
v___x_1929_ = l_Lean_indentExpr(v_a_1845_);
v___x_1930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1930_, 0, v___x_1928_);
lean_ctor_set(v___x_1930_, 1, v___x_1929_);
v___x_1931_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v___x_1930_, v___y_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_);
lean_dec_ref(v___y_1926_);
v_a_1932_ = lean_ctor_get(v___x_1931_, 0);
v_isSharedCheck_1939_ = !lean_is_exclusive(v___x_1931_);
if (v_isSharedCheck_1939_ == 0)
{
v___x_1934_ = v___x_1931_;
v_isShared_1935_ = v_isSharedCheck_1939_;
goto v_resetjp_1933_;
}
else
{
lean_inc(v_a_1932_);
lean_dec(v___x_1931_);
v___x_1934_ = lean_box(0);
v_isShared_1935_ = v_isSharedCheck_1939_;
goto v_resetjp_1933_;
}
v_resetjp_1933_:
{
lean_object* v___x_1937_; 
if (v_isShared_1935_ == 0)
{
v___x_1937_ = v___x_1934_;
goto v_reusejp_1936_;
}
else
{
lean_object* v_reuseFailAlloc_1938_; 
v_reuseFailAlloc_1938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1938_, 0, v_a_1932_);
v___x_1937_ = v_reuseFailAlloc_1938_;
goto v_reusejp_1936_;
}
v_reusejp_1936_:
{
return v___x_1937_;
}
}
}
}
else
{
lean_object* v_a_1951_; lean_object* v___x_1953_; uint8_t v_isShared_1954_; uint8_t v_isSharedCheck_1958_; 
lean_dec_ref_known(v___x_1841_, 14);
v_a_1951_ = lean_ctor_get(v___x_1842_, 0);
v_isSharedCheck_1958_ = !lean_is_exclusive(v___x_1842_);
if (v_isSharedCheck_1958_ == 0)
{
v___x_1953_ = v___x_1842_;
v_isShared_1954_ = v_isSharedCheck_1958_;
goto v_resetjp_1952_;
}
else
{
lean_inc(v_a_1951_);
lean_dec(v___x_1842_);
v___x_1953_ = lean_box(0);
v_isShared_1954_ = v_isSharedCheck_1958_;
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
lean_object* v_reuseFailAlloc_1957_; 
v_reuseFailAlloc_1957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1957_, 0, v_a_1951_);
v___x_1956_ = v_reuseFailAlloc_1957_;
goto v_reusejp_1955_;
}
v_reusejp_1955_:
{
return v___x_1956_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object* v_stx_1959_, lean_object* v_a_1960_, lean_object* v_a_1961_, lean_object* v_a_1962_, lean_object* v_a_1963_, lean_object* v_a_1964_, lean_object* v_a_1965_, lean_object* v_a_1966_){
_start:
{
lean_object* v_res_1967_; 
v_res_1967_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4(v_stx_1959_, v_a_1960_, v_a_1961_, v_a_1962_, v_a_1963_, v_a_1964_, v_a_1965_);
lean_dec(v_a_1965_);
lean_dec_ref(v_a_1964_);
lean_dec(v_a_1963_);
lean_dec_ref(v_a_1962_);
lean_dec(v_a_1961_);
lean_dec_ref(v_a_1960_);
return v_res_1967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2(lean_object* v_stx_1968_, lean_object* v_a_1969_, lean_object* v_a_1970_, lean_object* v_a_1971_, lean_object* v_a_1972_, lean_object* v_a_1973_, lean_object* v_a_1974_){
_start:
{
lean_object* v_fileName_1976_; lean_object* v_fileMap_1977_; lean_object* v_options_1978_; lean_object* v_currRecDepth_1979_; lean_object* v_maxRecDepth_1980_; lean_object* v_ref_1981_; lean_object* v_currNamespace_1982_; lean_object* v_openDecls_1983_; lean_object* v_initHeartbeats_1984_; lean_object* v_maxHeartbeats_1985_; lean_object* v_quotContext_1986_; lean_object* v_currMacroScope_1987_; uint8_t v_diag_1988_; lean_object* v_cancelTk_x3f_1989_; uint8_t v_suppressElabErrors_1990_; lean_object* v_inheritedTraceOptions_1991_; lean_object* v_ref_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; 
v_fileName_1976_ = lean_ctor_get(v_a_1973_, 0);
v_fileMap_1977_ = lean_ctor_get(v_a_1973_, 1);
v_options_1978_ = lean_ctor_get(v_a_1973_, 2);
v_currRecDepth_1979_ = lean_ctor_get(v_a_1973_, 3);
v_maxRecDepth_1980_ = lean_ctor_get(v_a_1973_, 4);
v_ref_1981_ = lean_ctor_get(v_a_1973_, 5);
v_currNamespace_1982_ = lean_ctor_get(v_a_1973_, 6);
v_openDecls_1983_ = lean_ctor_get(v_a_1973_, 7);
v_initHeartbeats_1984_ = lean_ctor_get(v_a_1973_, 8);
v_maxHeartbeats_1985_ = lean_ctor_get(v_a_1973_, 9);
v_quotContext_1986_ = lean_ctor_get(v_a_1973_, 10);
v_currMacroScope_1987_ = lean_ctor_get(v_a_1973_, 11);
v_diag_1988_ = lean_ctor_get_uint8(v_a_1973_, sizeof(void*)*14);
v_cancelTk_x3f_1989_ = lean_ctor_get(v_a_1973_, 12);
v_suppressElabErrors_1990_ = lean_ctor_get_uint8(v_a_1973_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1991_ = lean_ctor_get(v_a_1973_, 13);
v_ref_1992_ = l_Lean_replaceRef(v_stx_1968_, v_ref_1981_);
lean_inc_ref(v_inheritedTraceOptions_1991_);
lean_inc(v_cancelTk_x3f_1989_);
lean_inc(v_currMacroScope_1987_);
lean_inc(v_quotContext_1986_);
lean_inc(v_maxHeartbeats_1985_);
lean_inc(v_initHeartbeats_1984_);
lean_inc(v_openDecls_1983_);
lean_inc(v_currNamespace_1982_);
lean_inc(v_maxRecDepth_1980_);
lean_inc(v_currRecDepth_1979_);
lean_inc_ref(v_options_1978_);
lean_inc_ref(v_fileMap_1977_);
lean_inc_ref(v_fileName_1976_);
v___x_1993_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1993_, 0, v_fileName_1976_);
lean_ctor_set(v___x_1993_, 1, v_fileMap_1977_);
lean_ctor_set(v___x_1993_, 2, v_options_1978_);
lean_ctor_set(v___x_1993_, 3, v_currRecDepth_1979_);
lean_ctor_set(v___x_1993_, 4, v_maxRecDepth_1980_);
lean_ctor_set(v___x_1993_, 5, v_ref_1992_);
lean_ctor_set(v___x_1993_, 6, v_currNamespace_1982_);
lean_ctor_set(v___x_1993_, 7, v_openDecls_1983_);
lean_ctor_set(v___x_1993_, 8, v_initHeartbeats_1984_);
lean_ctor_set(v___x_1993_, 9, v_maxHeartbeats_1985_);
lean_ctor_set(v___x_1993_, 10, v_quotContext_1986_);
lean_ctor_set(v___x_1993_, 11, v_currMacroScope_1987_);
lean_ctor_set(v___x_1993_, 12, v_cancelTk_x3f_1989_);
lean_ctor_set(v___x_1993_, 13, v_inheritedTraceOptions_1991_);
lean_ctor_set_uint8(v___x_1993_, sizeof(void*)*14, v_diag_1988_);
lean_ctor_set_uint8(v___x_1993_, sizeof(void*)*14 + 1, v_suppressElabErrors_1990_);
lean_inc(v_stx_1968_);
v___x_1994_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged_evalTerm(v_stx_1968_, v_a_1969_, v_a_1970_, v_a_1971_, v_a_1972_, v___x_1993_, v_a_1974_);
if (lean_obj_tag(v___x_1994_) == 0)
{
lean_object* v_a_1995_; lean_object* v___x_1997_; uint8_t v_isShared_1998_; uint8_t v_isSharedCheck_2003_; 
lean_dec_ref_known(v___x_1993_, 14);
lean_dec(v_stx_1968_);
v_a_1995_ = lean_ctor_get(v___x_1994_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1994_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1997_ = v___x_1994_;
v_isShared_1998_ = v_isSharedCheck_2003_;
goto v_resetjp_1996_;
}
else
{
lean_inc(v_a_1995_);
lean_dec(v___x_1994_);
v___x_1997_ = lean_box(0);
v_isShared_1998_ = v_isSharedCheck_2003_;
goto v_resetjp_1996_;
}
v_resetjp_1996_:
{
lean_object* v_fst_1999_; lean_object* v___x_2001_; 
v_fst_1999_ = lean_ctor_get(v_a_1995_, 0);
lean_inc(v_fst_1999_);
lean_dec(v_a_1995_);
if (v_isShared_1998_ == 0)
{
lean_ctor_set(v___x_1997_, 0, v_fst_1999_);
v___x_2001_ = v___x_1997_;
goto v_reusejp_2000_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_fst_1999_);
v___x_2001_ = v_reuseFailAlloc_2002_;
goto v_reusejp_2000_;
}
v_reusejp_2000_:
{
return v___x_2001_;
}
}
}
else
{
lean_object* v_a_2004_; lean_object* v___x_2006_; uint8_t v_isShared_2007_; uint8_t v_isSharedCheck_2019_; 
v_a_2004_ = lean_ctor_get(v___x_1994_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___x_1994_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_2006_ = v___x_1994_;
v_isShared_2007_ = v_isSharedCheck_2019_;
goto v_resetjp_2005_;
}
else
{
lean_inc(v_a_2004_);
lean_dec(v___x_1994_);
v___x_2006_ = lean_box(0);
v_isShared_2007_ = v_isSharedCheck_2019_;
goto v_resetjp_2005_;
}
v_resetjp_2005_:
{
lean_object* v___x_2008_; lean_object* v___x_2010_; 
v___x_2008_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_2004_);
if (v_isShared_2007_ == 0)
{
v___x_2010_ = v___x_2006_;
goto v_reusejp_2009_;
}
else
{
lean_object* v_reuseFailAlloc_2018_; 
v_reuseFailAlloc_2018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2018_, 0, v_a_2004_);
v___x_2010_ = v_reuseFailAlloc_2018_;
goto v_reusejp_2009_;
}
v_reusejp_2009_:
{
uint8_t v___y_2012_; uint8_t v___x_2016_; 
v___x_2016_ = l_Lean_Exception_isInterrupt(v_a_2004_);
if (v___x_2016_ == 0)
{
uint8_t v___x_2017_; 
lean_inc(v_a_2004_);
v___x_2017_ = l_Lean_Exception_isRuntime(v_a_2004_);
v___y_2012_ = v___x_2017_;
goto v___jp_2011_;
}
else
{
v___y_2012_ = v___x_2016_;
goto v___jp_2011_;
}
v___jp_2011_:
{
if (v___y_2012_ == 0)
{
if (lean_obj_tag(v_a_2004_) == 0)
{
lean_dec_ref_known(v_a_2004_, 2);
lean_dec_ref_known(v___x_1993_, 14);
lean_dec(v_stx_1968_);
return v___x_2010_;
}
else
{
lean_object* v_id_2013_; uint8_t v___x_2014_; 
v_id_2013_ = lean_ctor_get(v_a_2004_, 0);
lean_inc(v_id_2013_);
lean_dec_ref_known(v_a_2004_, 2);
v___x_2014_ = l_Lean_instBEqInternalExceptionId_beq(v___x_2008_, v_id_2013_);
lean_dec(v_id_2013_);
if (v___x_2014_ == 0)
{
lean_dec_ref_known(v___x_1993_, 14);
lean_dec(v_stx_1968_);
return v___x_2010_;
}
else
{
lean_object* v___x_2015_; 
lean_dec_ref(v___x_2010_);
v___x_2015_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2_spec__4(v_stx_1968_, v_a_1969_, v_a_1970_, v_a_1971_, v_a_1972_, v___x_1993_, v_a_1974_);
lean_dec_ref_known(v___x_1993_, 14);
return v___x_2015_;
}
}
}
else
{
lean_dec(v_a_2004_);
lean_dec_ref_known(v___x_1993_, 14);
lean_dec(v_stx_1968_);
return v___x_2010_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2___boxed(lean_object* v_stx_2020_, lean_object* v_a_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_, lean_object* v_a_2026_, lean_object* v_a_2027_){
_start:
{
lean_object* v_res_2028_; 
v_res_2028_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2(v_stx_2020_, v_a_2021_, v_a_2022_, v_a_2023_, v_a_2024_, v_a_2025_, v_a_2026_);
lean_dec(v_a_2026_);
lean_dec_ref(v_a_2025_);
lean_dec(v_a_2024_);
lean_dec_ref(v_a_2023_);
lean_dec(v_a_2022_);
lean_dec_ref(v_a_2021_);
return v_res_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0(lean_object* v_config_2068_, lean_object* v_item_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_){
_start:
{
lean_object* v_item_2078_; lean_object* v___y_2079_; lean_object* v___y_2080_; lean_object* v___y_2081_; lean_object* v___y_2082_; lean_object* v___y_2083_; lean_object* v___y_2084_; lean_object* v___x_2087_; lean_object* v___x_2088_; 
v___x_2087_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1));
v___x_2088_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_2069_, v___x_2087_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2088_) == 0)
{
uint8_t v___x_2089_; 
lean_dec_ref_known(v___x_2088_, 1);
v___x_2089_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_2069_);
if (v___x_2089_ == 0)
{
lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; uint8_t v___x_2093_; 
v___x_2090_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_2069_);
lean_inc_ref(v_item_2069_);
v___x_2091_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_2069_);
v___x_2092_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__13));
v___x_2093_ = lean_string_dec_lt(v___x_2090_, v___x_2092_);
if (v___x_2093_ == 0)
{
uint8_t v___x_2094_; 
v___x_2094_ = lean_string_dec_eq(v___x_2090_, v___x_2092_);
if (v___x_2094_ == 0)
{
lean_object* v___x_2095_; uint8_t v___x_2096_; 
v___x_2095_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__1));
v___x_2096_ = lean_string_dec_eq(v___x_2090_, v___x_2095_);
if (v___x_2096_ == 0)
{
lean_object* v___x_2097_; uint8_t v___x_2098_; 
v___x_2097_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__2));
v___x_2098_ = lean_string_dec_eq(v___x_2090_, v___x_2097_);
lean_dec_ref(v___x_2090_);
if (v___x_2098_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2099_; lean_object* v___x_2100_; 
v___x_2099_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__3));
v___x_2100_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2069_, v___x_2099_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2100_) == 0)
{
uint8_t v___x_2101_; 
lean_dec_ref_known(v___x_2100_, 1);
v___x_2101_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2101_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2102_; 
lean_dec_ref(v___x_2091_);
v___x_2102_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2102_) == 0)
{
lean_object* v_toConfig_2103_; lean_object* v_a_2104_; lean_object* v___x_2106_; uint8_t v_isShared_2107_; uint8_t v_isSharedCheck_2131_; 
v_toConfig_2103_ = lean_ctor_get(v_config_2068_, 0);
lean_inc_ref(v_toConfig_2103_);
v_a_2104_ = lean_ctor_get(v___x_2102_, 0);
v_isSharedCheck_2131_ = !lean_is_exclusive(v___x_2102_);
if (v_isSharedCheck_2131_ == 0)
{
v___x_2106_ = v___x_2102_;
v_isShared_2107_ = v_isSharedCheck_2131_;
goto v_resetjp_2105_;
}
else
{
lean_inc(v_a_2104_);
lean_dec(v___x_2102_);
v___x_2106_ = lean_box(0);
v_isShared_2107_ = v_isSharedCheck_2131_;
goto v_resetjp_2105_;
}
v_resetjp_2105_:
{
uint8_t v_ifUnchanged_2108_; uint8_t v_mode_2109_; lean_object* v___x_2111_; uint8_t v_isShared_2112_; uint8_t v_isSharedCheck_2129_; 
v_ifUnchanged_2108_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1);
v_mode_2109_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1 + 1);
v_isSharedCheck_2129_ = !lean_is_exclusive(v_config_2068_);
if (v_isSharedCheck_2129_ == 0)
{
lean_object* v_unused_2130_; 
v_unused_2130_ = lean_ctor_get(v_config_2068_, 0);
lean_dec(v_unused_2130_);
v___x_2111_ = v_config_2068_;
v_isShared_2112_ = v_isSharedCheck_2129_;
goto v_resetjp_2110_;
}
else
{
lean_dec(v_config_2068_);
v___x_2111_ = lean_box(0);
v_isShared_2112_ = v_isSharedCheck_2129_;
goto v_resetjp_2110_;
}
v_resetjp_2110_:
{
uint8_t v_red_2113_; uint8_t v_contextual_2114_; lean_object* v___x_2116_; uint8_t v_isShared_2117_; uint8_t v_isSharedCheck_2128_; 
v_red_2113_ = lean_ctor_get_uint8(v_toConfig_2103_, 0);
v_contextual_2114_ = lean_ctor_get_uint8(v_toConfig_2103_, 2);
v_isSharedCheck_2128_ = !lean_is_exclusive(v_toConfig_2103_);
if (v_isSharedCheck_2128_ == 0)
{
v___x_2116_ = v_toConfig_2103_;
v_isShared_2117_ = v_isSharedCheck_2128_;
goto v_resetjp_2115_;
}
else
{
lean_dec(v_toConfig_2103_);
v___x_2116_ = lean_box(0);
v_isShared_2117_ = v_isSharedCheck_2128_;
goto v_resetjp_2115_;
}
v_resetjp_2115_:
{
lean_object* v___x_2119_; 
if (v_isShared_2117_ == 0)
{
v___x_2119_ = v___x_2116_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2127_; 
v_reuseFailAlloc_2127_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v_reuseFailAlloc_2127_, 0, v_red_2113_);
v___x_2119_ = v_reuseFailAlloc_2127_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
uint8_t v___x_2120_; lean_object* v___x_2122_; 
v___x_2120_ = lean_unbox(v_a_2104_);
lean_dec(v_a_2104_);
lean_ctor_set_uint8(v___x_2119_, 1, v___x_2120_);
lean_ctor_set_uint8(v___x_2119_, 2, v_contextual_2114_);
if (v_isShared_2112_ == 0)
{
lean_ctor_set(v___x_2111_, 0, v___x_2119_);
v___x_2122_ = v___x_2111_;
goto v_reusejp_2121_;
}
else
{
lean_object* v_reuseFailAlloc_2126_; 
v_reuseFailAlloc_2126_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_2126_, 0, v___x_2119_);
lean_ctor_set_uint8(v_reuseFailAlloc_2126_, sizeof(void*)*1, v_ifUnchanged_2108_);
lean_ctor_set_uint8(v_reuseFailAlloc_2126_, sizeof(void*)*1 + 1, v_mode_2109_);
v___x_2122_ = v_reuseFailAlloc_2126_;
goto v_reusejp_2121_;
}
v_reusejp_2121_:
{
lean_object* v___x_2124_; 
if (v_isShared_2107_ == 0)
{
lean_ctor_set(v___x_2106_, 0, v___x_2122_);
v___x_2124_ = v___x_2106_;
goto v_reusejp_2123_;
}
else
{
lean_object* v_reuseFailAlloc_2125_; 
v_reuseFailAlloc_2125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2125_, 0, v___x_2122_);
v___x_2124_ = v_reuseFailAlloc_2125_;
goto v_reusejp_2123_;
}
v_reusejp_2123_:
{
return v___x_2124_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2132_; lean_object* v___x_2134_; uint8_t v_isShared_2135_; uint8_t v_isSharedCheck_2139_; 
lean_dec_ref(v_config_2068_);
v_a_2132_ = lean_ctor_get(v___x_2102_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_2102_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_2134_ = v___x_2102_;
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
else
{
lean_inc(v_a_2132_);
lean_dec(v___x_2102_);
v___x_2134_ = lean_box(0);
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
v_resetjp_2133_:
{
lean_object* v___x_2137_; 
if (v_isShared_2135_ == 0)
{
v___x_2137_ = v___x_2134_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v_a_2132_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
return v___x_2137_;
}
}
}
}
}
else
{
lean_object* v_a_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2147_; 
lean_dec_ref(v___x_2091_);
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2140_ = lean_ctor_get(v___x_2100_, 0);
v_isSharedCheck_2147_ = !lean_is_exclusive(v___x_2100_);
if (v_isSharedCheck_2147_ == 0)
{
v___x_2142_ = v___x_2100_;
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_a_2140_);
lean_dec(v___x_2100_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v___x_2145_; 
if (v_isShared_2143_ == 0)
{
v___x_2145_ = v___x_2142_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2146_; 
v_reuseFailAlloc_2146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2146_, 0, v_a_2140_);
v___x_2145_ = v_reuseFailAlloc_2146_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
return v___x_2145_;
}
}
}
}
}
else
{
lean_object* v___x_2148_; lean_object* v___x_2149_; 
lean_dec_ref(v___x_2090_);
v___x_2148_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__4));
v___x_2149_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2069_, v___x_2148_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2149_) == 0)
{
uint8_t v___x_2150_; 
lean_dec_ref_known(v___x_2149_, 1);
v___x_2150_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2150_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2151_; 
lean_dec_ref(v___x_2091_);
lean_inc_ref(v_item_2069_);
v___x_2151_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2151_) == 0)
{
lean_object* v_value_2152_; lean_object* v___x_2153_; 
lean_dec_ref_known(v___x_2151_, 1);
v_value_2152_ = lean_ctor_get(v_item_2069_, 2);
lean_inc(v_value_2152_);
lean_dec_ref(v_item_2069_);
v___x_2153_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__0(v_value_2152_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_object* v_toConfig_2154_; lean_object* v_a_2155_; lean_object* v___x_2157_; uint8_t v_isShared_2158_; uint8_t v_isSharedCheck_2182_; 
v_toConfig_2154_ = lean_ctor_get(v_config_2068_, 0);
lean_inc_ref(v_toConfig_2154_);
v_a_2155_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2182_ == 0)
{
v___x_2157_ = v___x_2153_;
v_isShared_2158_ = v_isSharedCheck_2182_;
goto v_resetjp_2156_;
}
else
{
lean_inc(v_a_2155_);
lean_dec(v___x_2153_);
v___x_2157_ = lean_box(0);
v_isShared_2158_ = v_isSharedCheck_2182_;
goto v_resetjp_2156_;
}
v_resetjp_2156_:
{
uint8_t v_ifUnchanged_2159_; uint8_t v_mode_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2180_; 
v_ifUnchanged_2159_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1);
v_mode_2160_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1 + 1);
v_isSharedCheck_2180_ = !lean_is_exclusive(v_config_2068_);
if (v_isSharedCheck_2180_ == 0)
{
lean_object* v_unused_2181_; 
v_unused_2181_ = lean_ctor_get(v_config_2068_, 0);
lean_dec(v_unused_2181_);
v___x_2162_ = v_config_2068_;
v_isShared_2163_ = v_isSharedCheck_2180_;
goto v_resetjp_2161_;
}
else
{
lean_dec(v_config_2068_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2180_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
uint8_t v_zetaDelta_2164_; uint8_t v_contextual_2165_; lean_object* v___x_2167_; uint8_t v_isShared_2168_; uint8_t v_isSharedCheck_2179_; 
v_zetaDelta_2164_ = lean_ctor_get_uint8(v_toConfig_2154_, 1);
v_contextual_2165_ = lean_ctor_get_uint8(v_toConfig_2154_, 2);
v_isSharedCheck_2179_ = !lean_is_exclusive(v_toConfig_2154_);
if (v_isSharedCheck_2179_ == 0)
{
v___x_2167_ = v_toConfig_2154_;
v_isShared_2168_ = v_isSharedCheck_2179_;
goto v_resetjp_2166_;
}
else
{
lean_dec(v_toConfig_2154_);
v___x_2167_ = lean_box(0);
v_isShared_2168_ = v_isSharedCheck_2179_;
goto v_resetjp_2166_;
}
v_resetjp_2166_:
{
lean_object* v___x_2170_; 
if (v_isShared_2168_ == 0)
{
v___x_2170_ = v___x_2167_;
goto v_reusejp_2169_;
}
else
{
lean_object* v_reuseFailAlloc_2178_; 
v_reuseFailAlloc_2178_ = lean_alloc_ctor(0, 0, 3);
v___x_2170_ = v_reuseFailAlloc_2178_;
goto v_reusejp_2169_;
}
v_reusejp_2169_:
{
uint8_t v___x_2171_; lean_object* v___x_2173_; 
v___x_2171_ = lean_unbox(v_a_2155_);
lean_dec(v_a_2155_);
lean_ctor_set_uint8(v___x_2170_, 0, v___x_2171_);
lean_ctor_set_uint8(v___x_2170_, 1, v_zetaDelta_2164_);
lean_ctor_set_uint8(v___x_2170_, 2, v_contextual_2165_);
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 0, v___x_2170_);
v___x_2173_ = v___x_2162_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2177_; 
v_reuseFailAlloc_2177_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_2177_, 0, v___x_2170_);
lean_ctor_set_uint8(v_reuseFailAlloc_2177_, sizeof(void*)*1, v_ifUnchanged_2159_);
lean_ctor_set_uint8(v_reuseFailAlloc_2177_, sizeof(void*)*1 + 1, v_mode_2160_);
v___x_2173_ = v_reuseFailAlloc_2177_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
lean_object* v___x_2175_; 
if (v_isShared_2158_ == 0)
{
lean_ctor_set(v___x_2157_, 0, v___x_2173_);
v___x_2175_ = v___x_2157_;
goto v_reusejp_2174_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v___x_2173_);
v___x_2175_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2174_;
}
v_reusejp_2174_:
{
return v___x_2175_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2183_; lean_object* v___x_2185_; uint8_t v_isShared_2186_; uint8_t v_isSharedCheck_2190_; 
lean_dec_ref(v_config_2068_);
v_a_2183_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2190_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2190_ == 0)
{
v___x_2185_ = v___x_2153_;
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
else
{
lean_inc(v_a_2183_);
lean_dec(v___x_2153_);
v___x_2185_ = lean_box(0);
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
v_resetjp_2184_:
{
lean_object* v___x_2188_; 
if (v_isShared_2186_ == 0)
{
v___x_2188_ = v___x_2185_;
goto v_reusejp_2187_;
}
else
{
lean_object* v_reuseFailAlloc_2189_; 
v_reuseFailAlloc_2189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2189_, 0, v_a_2183_);
v___x_2188_ = v_reuseFailAlloc_2189_;
goto v_reusejp_2187_;
}
v_reusejp_2187_:
{
return v___x_2188_;
}
}
}
}
else
{
lean_object* v_a_2191_; lean_object* v___x_2193_; uint8_t v_isShared_2194_; uint8_t v_isSharedCheck_2198_; 
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2191_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2198_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2198_ == 0)
{
v___x_2193_ = v___x_2151_;
v_isShared_2194_ = v_isSharedCheck_2198_;
goto v_resetjp_2192_;
}
else
{
lean_inc(v_a_2191_);
lean_dec(v___x_2151_);
v___x_2193_ = lean_box(0);
v_isShared_2194_ = v_isSharedCheck_2198_;
goto v_resetjp_2192_;
}
v_resetjp_2192_:
{
lean_object* v___x_2196_; 
if (v_isShared_2194_ == 0)
{
v___x_2196_ = v___x_2193_;
goto v_reusejp_2195_;
}
else
{
lean_object* v_reuseFailAlloc_2197_; 
v_reuseFailAlloc_2197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2197_, 0, v_a_2191_);
v___x_2196_ = v_reuseFailAlloc_2197_;
goto v_reusejp_2195_;
}
v_reusejp_2195_:
{
return v___x_2196_;
}
}
}
}
}
else
{
lean_object* v_a_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2206_; 
lean_dec_ref(v___x_2091_);
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2199_ = lean_ctor_get(v___x_2149_, 0);
v_isSharedCheck_2206_ = !lean_is_exclusive(v___x_2149_);
if (v_isSharedCheck_2206_ == 0)
{
v___x_2201_ = v___x_2149_;
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_a_2199_);
lean_dec(v___x_2149_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v___x_2204_; 
if (v_isShared_2202_ == 0)
{
v___x_2204_ = v___x_2201_;
goto v_reusejp_2203_;
}
else
{
lean_object* v_reuseFailAlloc_2205_; 
v_reuseFailAlloc_2205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2205_, 0, v_a_2199_);
v___x_2204_ = v_reuseFailAlloc_2205_;
goto v_reusejp_2203_;
}
v_reusejp_2203_:
{
return v___x_2204_;
}
}
}
}
}
else
{
lean_object* v___x_2207_; lean_object* v___x_2208_; 
lean_dec_ref(v___x_2090_);
v___x_2207_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__5));
v___x_2208_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2069_, v___x_2207_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2208_) == 0)
{
uint8_t v___x_2209_; 
lean_dec_ref_known(v___x_2208_, 1);
v___x_2209_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2209_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2210_; 
lean_dec_ref(v___x_2091_);
lean_inc_ref(v_item_2069_);
v___x_2210_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2210_) == 0)
{
lean_object* v_value_2211_; lean_object* v___x_2212_; 
lean_dec_ref_known(v___x_2210_, 1);
v_value_2211_ = lean_ctor_get(v_item_2069_, 2);
lean_inc(v_value_2211_);
lean_dec_ref(v_item_2069_);
v___x_2212_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__1(v_value_2211_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2212_) == 0)
{
lean_object* v_a_2213_; lean_object* v___x_2215_; uint8_t v_isShared_2216_; uint8_t v_isSharedCheck_2230_; 
v_a_2213_ = lean_ctor_get(v___x_2212_, 0);
v_isSharedCheck_2230_ = !lean_is_exclusive(v___x_2212_);
if (v_isSharedCheck_2230_ == 0)
{
v___x_2215_ = v___x_2212_;
v_isShared_2216_ = v_isSharedCheck_2230_;
goto v_resetjp_2214_;
}
else
{
lean_inc(v_a_2213_);
lean_dec(v___x_2212_);
v___x_2215_ = lean_box(0);
v_isShared_2216_ = v_isSharedCheck_2230_;
goto v_resetjp_2214_;
}
v_resetjp_2214_:
{
lean_object* v_toConfig_2217_; uint8_t v_ifUnchanged_2218_; lean_object* v___x_2220_; uint8_t v_isShared_2221_; uint8_t v_isSharedCheck_2229_; 
v_toConfig_2217_ = lean_ctor_get(v_config_2068_, 0);
v_ifUnchanged_2218_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1);
v_isSharedCheck_2229_ = !lean_is_exclusive(v_config_2068_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2220_ = v_config_2068_;
v_isShared_2221_ = v_isSharedCheck_2229_;
goto v_resetjp_2219_;
}
else
{
lean_inc(v_toConfig_2217_);
lean_dec(v_config_2068_);
v___x_2220_ = lean_box(0);
v_isShared_2221_ = v_isSharedCheck_2229_;
goto v_resetjp_2219_;
}
v_resetjp_2219_:
{
lean_object* v___x_2223_; 
if (v_isShared_2221_ == 0)
{
v___x_2223_ = v___x_2220_;
goto v_reusejp_2222_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_toConfig_2217_);
lean_ctor_set_uint8(v_reuseFailAlloc_2228_, sizeof(void*)*1, v_ifUnchanged_2218_);
v___x_2223_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2222_;
}
v_reusejp_2222_:
{
uint8_t v___x_2224_; lean_object* v___x_2226_; 
v___x_2224_ = lean_unbox(v_a_2213_);
lean_dec(v_a_2213_);
lean_ctor_set_uint8(v___x_2223_, sizeof(void*)*1 + 1, v___x_2224_);
if (v_isShared_2216_ == 0)
{
lean_ctor_set(v___x_2215_, 0, v___x_2223_);
v___x_2226_ = v___x_2215_;
goto v_reusejp_2225_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v___x_2223_);
v___x_2226_ = v_reuseFailAlloc_2227_;
goto v_reusejp_2225_;
}
v_reusejp_2225_:
{
return v___x_2226_;
}
}
}
}
}
else
{
lean_object* v_a_2231_; lean_object* v___x_2233_; uint8_t v_isShared_2234_; uint8_t v_isSharedCheck_2238_; 
lean_dec_ref(v_config_2068_);
v_a_2231_ = lean_ctor_get(v___x_2212_, 0);
v_isSharedCheck_2238_ = !lean_is_exclusive(v___x_2212_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2233_ = v___x_2212_;
v_isShared_2234_ = v_isSharedCheck_2238_;
goto v_resetjp_2232_;
}
else
{
lean_inc(v_a_2231_);
lean_dec(v___x_2212_);
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
v_reuseFailAlloc_2237_ = lean_alloc_ctor(1, 1, 0);
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
}
else
{
lean_object* v_a_2239_; lean_object* v___x_2241_; uint8_t v_isShared_2242_; uint8_t v_isSharedCheck_2246_; 
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2239_ = lean_ctor_get(v___x_2210_, 0);
v_isSharedCheck_2246_ = !lean_is_exclusive(v___x_2210_);
if (v_isSharedCheck_2246_ == 0)
{
v___x_2241_ = v___x_2210_;
v_isShared_2242_ = v_isSharedCheck_2246_;
goto v_resetjp_2240_;
}
else
{
lean_inc(v_a_2239_);
lean_dec(v___x_2210_);
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
else
{
lean_object* v_a_2247_; lean_object* v___x_2249_; uint8_t v_isShared_2250_; uint8_t v_isSharedCheck_2254_; 
lean_dec_ref(v___x_2091_);
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2247_ = lean_ctor_get(v___x_2208_, 0);
v_isSharedCheck_2254_ = !lean_is_exclusive(v___x_2208_);
if (v_isSharedCheck_2254_ == 0)
{
v___x_2249_ = v___x_2208_;
v_isShared_2250_ = v_isSharedCheck_2254_;
goto v_resetjp_2248_;
}
else
{
lean_inc(v_a_2247_);
lean_dec(v___x_2208_);
v___x_2249_ = lean_box(0);
v_isShared_2250_ = v_isSharedCheck_2254_;
goto v_resetjp_2248_;
}
v_resetjp_2248_:
{
lean_object* v___x_2252_; 
if (v_isShared_2250_ == 0)
{
v___x_2252_ = v___x_2249_;
goto v_reusejp_2251_;
}
else
{
lean_object* v_reuseFailAlloc_2253_; 
v_reuseFailAlloc_2253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2253_, 0, v_a_2247_);
v___x_2252_ = v_reuseFailAlloc_2253_;
goto v_reusejp_2251_;
}
v_reusejp_2251_:
{
return v___x_2252_;
}
}
}
}
}
else
{
lean_object* v___x_2255_; uint8_t v___x_2256_; 
v___x_2255_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__6));
v___x_2256_ = lean_string_dec_eq(v___x_2090_, v___x_2255_);
if (v___x_2256_ == 0)
{
lean_object* v___x_2257_; uint8_t v___x_2258_; 
v___x_2257_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__7));
v___x_2258_ = lean_string_dec_eq(v___x_2090_, v___x_2257_);
if (v___x_2258_ == 0)
{
lean_object* v___x_2259_; uint8_t v___x_2260_; 
v___x_2259_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_instReprConfig_repr___redArg___closed__10));
v___x_2260_ = lean_string_dec_eq(v___x_2090_, v___x_2259_);
lean_dec_ref(v___x_2090_);
if (v___x_2260_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2261_; lean_object* v___x_2262_; 
v___x_2261_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__8));
v___x_2262_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2069_, v___x_2261_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2262_) == 0)
{
uint8_t v___x_2263_; 
lean_dec_ref_known(v___x_2262_, 1);
v___x_2263_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2263_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2264_; 
lean_dec_ref(v___x_2091_);
lean_inc_ref(v_item_2069_);
v___x_2264_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2264_) == 0)
{
lean_object* v_value_2265_; lean_object* v___x_2266_; 
lean_dec_ref_known(v___x_2264_, 1);
v_value_2265_ = lean_ctor_get(v_item_2069_, 2);
lean_inc(v_value_2265_);
lean_dec_ref(v_item_2069_);
v___x_2266_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__2(v_value_2265_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2266_) == 0)
{
lean_object* v_a_2267_; lean_object* v___x_2269_; uint8_t v_isShared_2270_; uint8_t v_isSharedCheck_2284_; 
v_a_2267_ = lean_ctor_get(v___x_2266_, 0);
v_isSharedCheck_2284_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2284_ == 0)
{
v___x_2269_ = v___x_2266_;
v_isShared_2270_ = v_isSharedCheck_2284_;
goto v_resetjp_2268_;
}
else
{
lean_inc(v_a_2267_);
lean_dec(v___x_2266_);
v___x_2269_ = lean_box(0);
v_isShared_2270_ = v_isSharedCheck_2284_;
goto v_resetjp_2268_;
}
v_resetjp_2268_:
{
lean_object* v_toConfig_2271_; uint8_t v_mode_2272_; lean_object* v___x_2274_; uint8_t v_isShared_2275_; uint8_t v_isSharedCheck_2283_; 
v_toConfig_2271_ = lean_ctor_get(v_config_2068_, 0);
v_mode_2272_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1 + 1);
v_isSharedCheck_2283_ = !lean_is_exclusive(v_config_2068_);
if (v_isSharedCheck_2283_ == 0)
{
v___x_2274_ = v_config_2068_;
v_isShared_2275_ = v_isSharedCheck_2283_;
goto v_resetjp_2273_;
}
else
{
lean_inc(v_toConfig_2271_);
lean_dec(v_config_2068_);
v___x_2274_ = lean_box(0);
v_isShared_2275_ = v_isSharedCheck_2283_;
goto v_resetjp_2273_;
}
v_resetjp_2273_:
{
lean_object* v___x_2277_; 
if (v_isShared_2275_ == 0)
{
v___x_2277_ = v___x_2274_;
goto v_reusejp_2276_;
}
else
{
lean_object* v_reuseFailAlloc_2282_; 
v_reuseFailAlloc_2282_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_2282_, 0, v_toConfig_2271_);
v___x_2277_ = v_reuseFailAlloc_2282_;
goto v_reusejp_2276_;
}
v_reusejp_2276_:
{
uint8_t v___x_2278_; lean_object* v___x_2280_; 
v___x_2278_ = lean_unbox(v_a_2267_);
lean_dec(v_a_2267_);
lean_ctor_set_uint8(v___x_2277_, sizeof(void*)*1, v___x_2278_);
lean_ctor_set_uint8(v___x_2277_, sizeof(void*)*1 + 1, v_mode_2272_);
if (v_isShared_2270_ == 0)
{
lean_ctor_set(v___x_2269_, 0, v___x_2277_);
v___x_2280_ = v___x_2269_;
goto v_reusejp_2279_;
}
else
{
lean_object* v_reuseFailAlloc_2281_; 
v_reuseFailAlloc_2281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2281_, 0, v___x_2277_);
v___x_2280_ = v_reuseFailAlloc_2281_;
goto v_reusejp_2279_;
}
v_reusejp_2279_:
{
return v___x_2280_;
}
}
}
}
}
else
{
lean_object* v_a_2285_; lean_object* v___x_2287_; uint8_t v_isShared_2288_; uint8_t v_isSharedCheck_2292_; 
lean_dec_ref(v_config_2068_);
v_a_2285_ = lean_ctor_get(v___x_2266_, 0);
v_isSharedCheck_2292_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2292_ == 0)
{
v___x_2287_ = v___x_2266_;
v_isShared_2288_ = v_isSharedCheck_2292_;
goto v_resetjp_2286_;
}
else
{
lean_inc(v_a_2285_);
lean_dec(v___x_2266_);
v___x_2287_ = lean_box(0);
v_isShared_2288_ = v_isSharedCheck_2292_;
goto v_resetjp_2286_;
}
v_resetjp_2286_:
{
lean_object* v___x_2290_; 
if (v_isShared_2288_ == 0)
{
v___x_2290_ = v___x_2287_;
goto v_reusejp_2289_;
}
else
{
lean_object* v_reuseFailAlloc_2291_; 
v_reuseFailAlloc_2291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2291_, 0, v_a_2285_);
v___x_2290_ = v_reuseFailAlloc_2291_;
goto v_reusejp_2289_;
}
v_reusejp_2289_:
{
return v___x_2290_;
}
}
}
}
else
{
lean_object* v_a_2293_; lean_object* v___x_2295_; uint8_t v_isShared_2296_; uint8_t v_isSharedCheck_2300_; 
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2293_ = lean_ctor_get(v___x_2264_, 0);
v_isSharedCheck_2300_ = !lean_is_exclusive(v___x_2264_);
if (v_isSharedCheck_2300_ == 0)
{
v___x_2295_ = v___x_2264_;
v_isShared_2296_ = v_isSharedCheck_2300_;
goto v_resetjp_2294_;
}
else
{
lean_inc(v_a_2293_);
lean_dec(v___x_2264_);
v___x_2295_ = lean_box(0);
v_isShared_2296_ = v_isSharedCheck_2300_;
goto v_resetjp_2294_;
}
v_resetjp_2294_:
{
lean_object* v___x_2298_; 
if (v_isShared_2296_ == 0)
{
v___x_2298_ = v___x_2295_;
goto v_reusejp_2297_;
}
else
{
lean_object* v_reuseFailAlloc_2299_; 
v_reuseFailAlloc_2299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2299_, 0, v_a_2293_);
v___x_2298_ = v_reuseFailAlloc_2299_;
goto v_reusejp_2297_;
}
v_reusejp_2297_:
{
return v___x_2298_;
}
}
}
}
}
else
{
lean_object* v_a_2301_; lean_object* v___x_2303_; uint8_t v_isShared_2304_; uint8_t v_isSharedCheck_2308_; 
lean_dec_ref(v___x_2091_);
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2301_ = lean_ctor_get(v___x_2262_, 0);
v_isSharedCheck_2308_ = !lean_is_exclusive(v___x_2262_);
if (v_isSharedCheck_2308_ == 0)
{
v___x_2303_ = v___x_2262_;
v_isShared_2304_ = v_isSharedCheck_2308_;
goto v_resetjp_2302_;
}
else
{
lean_inc(v_a_2301_);
lean_dec(v___x_2262_);
v___x_2303_ = lean_box(0);
v_isShared_2304_ = v_isSharedCheck_2308_;
goto v_resetjp_2302_;
}
v_resetjp_2302_:
{
lean_object* v___x_2306_; 
if (v_isShared_2304_ == 0)
{
v___x_2306_ = v___x_2303_;
goto v_reusejp_2305_;
}
else
{
lean_object* v_reuseFailAlloc_2307_; 
v_reuseFailAlloc_2307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2307_, 0, v_a_2301_);
v___x_2306_ = v_reuseFailAlloc_2307_;
goto v_reusejp_2305_;
}
v_reusejp_2305_:
{
return v___x_2306_;
}
}
}
}
}
else
{
lean_object* v___x_2309_; lean_object* v___x_2310_; 
lean_dec_ref(v___x_2090_);
v___x_2309_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__9));
v___x_2310_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_2069_, v___x_2309_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2310_) == 0)
{
uint8_t v___x_2311_; 
lean_dec_ref_known(v___x_2310_, 1);
v___x_2311_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2311_ == 0)
{
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2312_; 
lean_dec_ref(v___x_2091_);
v___x_2312_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
if (lean_obj_tag(v___x_2312_) == 0)
{
lean_object* v_toConfig_2313_; lean_object* v_a_2314_; lean_object* v___x_2316_; uint8_t v_isShared_2317_; uint8_t v_isSharedCheck_2341_; 
v_toConfig_2313_ = lean_ctor_get(v_config_2068_, 0);
lean_inc_ref(v_toConfig_2313_);
v_a_2314_ = lean_ctor_get(v___x_2312_, 0);
v_isSharedCheck_2341_ = !lean_is_exclusive(v___x_2312_);
if (v_isSharedCheck_2341_ == 0)
{
v___x_2316_ = v___x_2312_;
v_isShared_2317_ = v_isSharedCheck_2341_;
goto v_resetjp_2315_;
}
else
{
lean_inc(v_a_2314_);
lean_dec(v___x_2312_);
v___x_2316_ = lean_box(0);
v_isShared_2317_ = v_isSharedCheck_2341_;
goto v_resetjp_2315_;
}
v_resetjp_2315_:
{
uint8_t v_ifUnchanged_2318_; uint8_t v_mode_2319_; lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2339_; 
v_ifUnchanged_2318_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1);
v_mode_2319_ = lean_ctor_get_uint8(v_config_2068_, sizeof(void*)*1 + 1);
v_isSharedCheck_2339_ = !lean_is_exclusive(v_config_2068_);
if (v_isSharedCheck_2339_ == 0)
{
lean_object* v_unused_2340_; 
v_unused_2340_ = lean_ctor_get(v_config_2068_, 0);
lean_dec(v_unused_2340_);
v___x_2321_ = v_config_2068_;
v_isShared_2322_ = v_isSharedCheck_2339_;
goto v_resetjp_2320_;
}
else
{
lean_dec(v_config_2068_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2339_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
uint8_t v_red_2323_; uint8_t v_zetaDelta_2324_; lean_object* v___x_2326_; uint8_t v_isShared_2327_; uint8_t v_isSharedCheck_2338_; 
v_red_2323_ = lean_ctor_get_uint8(v_toConfig_2313_, 0);
v_zetaDelta_2324_ = lean_ctor_get_uint8(v_toConfig_2313_, 1);
v_isSharedCheck_2338_ = !lean_is_exclusive(v_toConfig_2313_);
if (v_isSharedCheck_2338_ == 0)
{
v___x_2326_ = v_toConfig_2313_;
v_isShared_2327_ = v_isSharedCheck_2338_;
goto v_resetjp_2325_;
}
else
{
lean_dec(v_toConfig_2313_);
v___x_2326_ = lean_box(0);
v_isShared_2327_ = v_isSharedCheck_2338_;
goto v_resetjp_2325_;
}
v_resetjp_2325_:
{
lean_object* v___x_2329_; 
if (v_isShared_2327_ == 0)
{
v___x_2329_ = v___x_2326_;
goto v_reusejp_2328_;
}
else
{
lean_object* v_reuseFailAlloc_2337_; 
v_reuseFailAlloc_2337_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v_reuseFailAlloc_2337_, 0, v_red_2323_);
lean_ctor_set_uint8(v_reuseFailAlloc_2337_, 1, v_zetaDelta_2324_);
v___x_2329_ = v_reuseFailAlloc_2337_;
goto v_reusejp_2328_;
}
v_reusejp_2328_:
{
uint8_t v___x_2330_; lean_object* v___x_2332_; 
v___x_2330_ = lean_unbox(v_a_2314_);
lean_dec(v_a_2314_);
lean_ctor_set_uint8(v___x_2329_, 2, v___x_2330_);
if (v_isShared_2322_ == 0)
{
lean_ctor_set(v___x_2321_, 0, v___x_2329_);
v___x_2332_ = v___x_2321_;
goto v_reusejp_2331_;
}
else
{
lean_object* v_reuseFailAlloc_2336_; 
v_reuseFailAlloc_2336_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_2336_, 0, v___x_2329_);
lean_ctor_set_uint8(v_reuseFailAlloc_2336_, sizeof(void*)*1, v_ifUnchanged_2318_);
lean_ctor_set_uint8(v_reuseFailAlloc_2336_, sizeof(void*)*1 + 1, v_mode_2319_);
v___x_2332_ = v_reuseFailAlloc_2336_;
goto v_reusejp_2331_;
}
v_reusejp_2331_:
{
lean_object* v___x_2334_; 
if (v_isShared_2317_ == 0)
{
lean_ctor_set(v___x_2316_, 0, v___x_2332_);
v___x_2334_ = v___x_2316_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v___x_2332_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2342_; lean_object* v___x_2344_; uint8_t v_isShared_2345_; uint8_t v_isSharedCheck_2349_; 
lean_dec_ref(v_config_2068_);
v_a_2342_ = lean_ctor_get(v___x_2312_, 0);
v_isSharedCheck_2349_ = !lean_is_exclusive(v___x_2312_);
if (v_isSharedCheck_2349_ == 0)
{
v___x_2344_ = v___x_2312_;
v_isShared_2345_ = v_isSharedCheck_2349_;
goto v_resetjp_2343_;
}
else
{
lean_inc(v_a_2342_);
lean_dec(v___x_2312_);
v___x_2344_ = lean_box(0);
v_isShared_2345_ = v_isSharedCheck_2349_;
goto v_resetjp_2343_;
}
v_resetjp_2343_:
{
lean_object* v___x_2347_; 
if (v_isShared_2345_ == 0)
{
v___x_2347_ = v___x_2344_;
goto v_reusejp_2346_;
}
else
{
lean_object* v_reuseFailAlloc_2348_; 
v_reuseFailAlloc_2348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2348_, 0, v_a_2342_);
v___x_2347_ = v_reuseFailAlloc_2348_;
goto v_reusejp_2346_;
}
v_reusejp_2346_:
{
return v___x_2347_;
}
}
}
}
}
else
{
lean_object* v_a_2350_; lean_object* v___x_2352_; uint8_t v_isShared_2353_; uint8_t v_isSharedCheck_2357_; 
lean_dec_ref(v___x_2091_);
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2350_ = lean_ctor_get(v___x_2310_, 0);
v_isSharedCheck_2357_ = !lean_is_exclusive(v___x_2310_);
if (v_isSharedCheck_2357_ == 0)
{
v___x_2352_ = v___x_2310_;
v_isShared_2353_ = v_isSharedCheck_2357_;
goto v_resetjp_2351_;
}
else
{
lean_inc(v_a_2350_);
lean_dec(v___x_2310_);
v___x_2352_ = lean_box(0);
v_isShared_2353_ = v_isSharedCheck_2357_;
goto v_resetjp_2351_;
}
v_resetjp_2351_:
{
lean_object* v___x_2355_; 
if (v_isShared_2353_ == 0)
{
v___x_2355_ = v___x_2352_;
goto v_reusejp_2354_;
}
else
{
lean_object* v_reuseFailAlloc_2356_; 
v_reuseFailAlloc_2356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2356_, 0, v_a_2350_);
v___x_2355_ = v_reuseFailAlloc_2356_;
goto v_reusejp_2354_;
}
v_reusejp_2354_:
{
return v___x_2355_;
}
}
}
}
}
else
{
uint8_t v___x_2358_; 
lean_dec_ref(v___x_2090_);
lean_dec_ref(v_config_2068_);
v___x_2358_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_2091_);
if (v___x_2358_ == 0)
{
lean_dec_ref(v_item_2069_);
v_item_2078_ = v___x_2091_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
else
{
lean_object* v_value_2359_; lean_object* v___x_2360_; 
lean_dec_ref(v___x_2091_);
v_value_2359_ = lean_ctor_get(v_item_2069_, 2);
lean_inc(v_value_2359_);
lean_dec_ref(v_item_2069_);
v___x_2360_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3(v_value_2359_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
return v___x_2360_;
}
}
}
}
else
{
lean_dec_ref(v_config_2068_);
v_item_2078_ = v_item_2069_;
v___y_2079_ = v___y_2070_;
v___y_2080_ = v___y_2071_;
v___y_2081_ = v___y_2072_;
v___y_2082_ = v___y_2073_;
v___y_2083_ = v___y_2074_;
v___y_2084_ = v___y_2075_;
goto v___jp_2077_;
}
}
else
{
lean_object* v_a_2361_; lean_object* v___x_2363_; uint8_t v_isShared_2364_; uint8_t v_isSharedCheck_2368_; 
lean_dec_ref(v_item_2069_);
lean_dec_ref(v_config_2068_);
v_a_2361_ = lean_ctor_get(v___x_2088_, 0);
v_isSharedCheck_2368_ = !lean_is_exclusive(v___x_2088_);
if (v_isSharedCheck_2368_ == 0)
{
v___x_2363_ = v___x_2088_;
v_isShared_2364_ = v_isSharedCheck_2368_;
goto v_resetjp_2362_;
}
else
{
lean_inc(v_a_2361_);
lean_dec(v___x_2088_);
v___x_2363_ = lean_box(0);
v_isShared_2364_ = v_isSharedCheck_2368_;
goto v_resetjp_2362_;
}
v_resetjp_2362_:
{
lean_object* v___x_2366_; 
if (v_isShared_2364_ == 0)
{
v___x_2366_ = v___x_2363_;
goto v_reusejp_2365_;
}
else
{
lean_object* v_reuseFailAlloc_2367_; 
v_reuseFailAlloc_2367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2367_, 0, v_a_2361_);
v___x_2366_ = v_reuseFailAlloc_2367_;
goto v_reusejp_2365_;
}
v_reusejp_2365_:
{
return v___x_2366_;
}
}
}
v___jp_2077_:
{
lean_object* v___x_2085_; lean_object* v___x_2086_; 
v___x_2085_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___closed__0));
v___x_2086_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_2078_, v___x_2085_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_, v___y_2084_);
return v___x_2086_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_2369_, lean_object* v_item_2370_, lean_object* v___y_2371_, lean_object* v___y_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_){
_start:
{
lean_object* v_res_2378_; 
v_res_2378_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___lam__0(v_config_2369_, v_item_2370_, v___y_2371_, v___y_2372_, v___y_2373_, v___y_2374_, v___y_2375_, v___y_2376_);
lean_dec(v___y_2376_);
lean_dec_ref(v___y_2375_);
lean_dec(v___y_2374_);
lean_dec_ref(v___y_2373_);
lean_dec(v___y_2372_);
lean_dec_ref(v___y_2371_);
return v_res_2378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6(lean_object* v_e_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_){
_start:
{
lean_object* v___x_2389_; 
v___x_2389_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___redArg(v_e_2381_, v___y_2385_);
return v___x_2389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6___boxed(lean_object* v_e_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_, lean_object* v___y_2393_, lean_object* v___y_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_){
_start:
{
lean_object* v_res_2398_; 
v_res_2398_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__6(v_e_2390_, v___y_2391_, v___y_2392_, v___y_2393_, v___y_2394_, v___y_2395_, v___y_2396_);
lean_dec(v___y_2396_);
lean_dec_ref(v___y_2395_);
lean_dec(v___y_2394_);
lean_dec_ref(v___y_2393_);
lean_dec(v___y_2392_);
lean_dec_ref(v___y_2391_);
return v_res_2398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8(lean_object* v_00_u03b1_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_){
_start:
{
lean_object* v___x_2407_; 
v___x_2407_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___redArg();
return v___x_2407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8___boxed(lean_object* v_00_u03b1_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_, lean_object* v___y_2415_){
_start:
{
lean_object* v_res_2416_; 
v_res_2416_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__8(v_00_u03b1_2408_, v___y_2409_, v___y_2410_, v___y_2411_, v___y_2412_, v___y_2413_, v___y_2414_);
lean_dec(v___y_2414_);
lean_dec_ref(v___y_2413_);
lean_dec(v___y_2412_);
lean_dec_ref(v___y_2411_);
lean_dec(v___y_2410_);
lean_dec_ref(v___y_2409_);
return v_res_2416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7(lean_object* v_00_u03b1_2417_, lean_object* v_msg_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_){
_start:
{
lean_object* v___x_2426_; 
v___x_2426_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___redArg(v_msg_2418_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
return v___x_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7___boxed(lean_object* v_00_u03b1_2427_, lean_object* v_msg_2428_, lean_object* v___y_2429_, lean_object* v___y_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_, lean_object* v___y_2435_){
_start:
{
lean_object* v_res_2436_; 
v_res_2436_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7(v_00_u03b1_2427_, v_msg_2428_, v___y_2429_, v___y_2430_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_);
lean_dec(v___y_2434_);
lean_dec_ref(v___y_2433_);
lean_dec(v___y_2432_);
lean_dec_ref(v___y_2431_);
lean_dec(v___y_2430_);
lean_dec_ref(v___y_2429_);
return v_res_2436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8(lean_object* v_msgData_2437_, lean_object* v_macroStack_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_){
_start:
{
lean_object* v___x_2446_; 
v___x_2446_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___redArg(v_msgData_2437_, v_macroStack_2438_, v___y_2443_);
return v___x_2446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8___boxed(lean_object* v_msgData_2447_, lean_object* v_macroStack_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_){
_start:
{
lean_object* v_res_2456_; 
v_res_2456_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem_spec__3_spec__7_spec__8(v_msgData_2447_, v_macroStack_2448_, v___y_2449_, v___y_2450_, v___y_2451_, v___y_2452_, v___y_2453_, v___y_2454_);
lean_dec(v___y_2454_);
lean_dec_ref(v___y_2453_);
lean_dec(v___y_2452_);
lean_dec_ref(v___y_2451_);
lean_dec(v___y_2450_);
lean_dec_ref(v___y_2449_);
return v_res_2456_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; 
v___x_2457_ = lean_box(0);
v___x_2458_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1_evalExpr___closed__1));
v___x_2459_ = l_Lean_mkConst(v___x_2458_, v___x_2457_);
return v___x_2459_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2460_; lean_object* v___x_2461_; 
v___x_2460_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__0);
v___x_2461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2461_, 0, v___x_2460_);
return v___x_2461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0(lean_object* v_cfg_2462_, lean_object* v_cfgItem_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_){
_start:
{
lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2471_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___closed__1);
v___x_2472_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_2462_, v_cfgItem_2463_, v___x_2471_, v___y_2464_, v___y_2465_, v___y_2466_, v___y_2467_, v___y_2468_, v___y_2469_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0___boxed(lean_object* v_cfg_2473_, lean_object* v_cfgItem_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_){
_start:
{
lean_object* v_res_2482_; 
v_res_2482_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___lam__0(v_cfg_2473_, v_cfgItem_2474_, v___y_2475_, v___y_2476_, v___y_2477_, v___y_2478_, v___y_2479_, v___y_2480_);
lean_dec(v___y_2480_);
lean_dec_ref(v___y_2479_);
lean_dec(v___y_2478_);
lean_dec_ref(v___y_2477_);
lean_dec(v___y_2476_);
lean_dec_ref(v___y_2475_);
lean_dec(v_cfgItem_2474_);
return v_res_2482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(lean_object* v_cfg_2484_, lean_object* v_init_2485_, uint8_t v_logExceptions_2486_, lean_object* v_a_2487_, lean_object* v_a_2488_, lean_object* v_a_2489_){
_start:
{
lean_object* v_onErr_2491_; lean_object* v_eval_2492_; 
v_onErr_2491_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___closed__0));
v_eval_2492_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_elabConfig_evalConfigItem___closed__0));
if (v_logExceptions_2486_ == 0)
{
lean_object* v___x_2493_; 
v___x_2493_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2492_, v_init_2485_, v_cfg_2484_, v_onErr_2491_, v_logExceptions_2486_, v_a_2488_, v_a_2489_);
return v___x_2493_;
}
else
{
uint8_t v_recover_2494_; lean_object* v___x_2495_; 
v_recover_2494_ = lean_ctor_get_uint8(v_a_2487_, sizeof(void*)*1);
v___x_2495_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2492_, v_init_2485_, v_cfg_2484_, v_onErr_2491_, v_recover_2494_, v_a_2488_, v_a_2489_);
return v___x_2495_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg___boxed(lean_object* v_cfg_2496_, lean_object* v_init_2497_, lean_object* v_logExceptions_2498_, lean_object* v_a_2499_, lean_object* v_a_2500_, lean_object* v_a_2501_, lean_object* v_a_2502_){
_start:
{
uint8_t v_logExceptions_boxed_2503_; lean_object* v_res_2504_; 
v_logExceptions_boxed_2503_ = lean_unbox(v_logExceptions_2498_);
v_res_2504_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(v_cfg_2496_, v_init_2497_, v_logExceptions_boxed_2503_, v_a_2499_, v_a_2500_, v_a_2501_);
lean_dec(v_a_2501_);
lean_dec_ref(v_a_2500_);
lean_dec_ref(v_a_2499_);
return v_res_2504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig(lean_object* v_cfg_2505_, lean_object* v_init_2506_, uint8_t v_logExceptions_2507_, lean_object* v_a_2508_, lean_object* v_a_2509_, lean_object* v_a_2510_, lean_object* v_a_2511_, lean_object* v_a_2512_, lean_object* v_a_2513_, lean_object* v_a_2514_, lean_object* v_a_2515_){
_start:
{
lean_object* v___x_2517_; 
v___x_2517_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(v_cfg_2505_, v_init_2506_, v_logExceptions_2507_, v_a_2508_, v_a_2514_, v_a_2515_);
return v___x_2517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___boxed(lean_object* v_cfg_2518_, lean_object* v_init_2519_, lean_object* v_logExceptions_2520_, lean_object* v_a_2521_, lean_object* v_a_2522_, lean_object* v_a_2523_, lean_object* v_a_2524_, lean_object* v_a_2525_, lean_object* v_a_2526_, lean_object* v_a_2527_, lean_object* v_a_2528_, lean_object* v_a_2529_){
_start:
{
uint8_t v_logExceptions_boxed_2530_; lean_object* v_res_2531_; 
v_logExceptions_boxed_2530_ = lean_unbox(v_logExceptions_2520_);
v_res_2531_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig(v_cfg_2518_, v_init_2519_, v_logExceptions_boxed_2530_, v_a_2521_, v_a_2522_, v_a_2523_, v_a_2524_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_);
lean_dec(v_a_2528_);
lean_dec_ref(v_a_2527_);
lean_dec(v_a_2526_);
lean_dec_ref(v_a_2525_);
lean_dec(v_a_2524_);
lean_dec_ref(v_a_2523_);
lean_dec(v_a_2522_);
lean_dec_ref(v_a_2521_);
return v_res_2531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f(lean_object* v_e_2535_, lean_object* v_a_2536_, lean_object* v_a_2537_, lean_object* v_a_2538_, lean_object* v_a_2539_, lean_object* v_a_2540_, lean_object* v_a_2541_){
_start:
{
uint8_t v___y_2544_; lean_object* v_____x_2545_; lean_object* v_a_2553_; lean_object* v_keyedConfig_2642_; uint8_t v_trackZetaDelta_2643_; lean_object* v_zetaDeltaSet_2644_; lean_object* v_lctx_2645_; lean_object* v_localInstances_2646_; lean_object* v_defEqCtx_x3f_2647_; lean_object* v_synthPendingDepth_2648_; lean_object* v_customCanUnfoldPredicate_x3f_2649_; uint8_t v_univApprox_2650_; uint8_t v_inTypeClassResolution_2651_; uint8_t v_cacheInferType_2652_; uint8_t v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; 
v_keyedConfig_2642_ = lean_ctor_get(v_a_2538_, 0);
v_trackZetaDelta_2643_ = lean_ctor_get_uint8(v_a_2538_, sizeof(void*)*7);
v_zetaDeltaSet_2644_ = lean_ctor_get(v_a_2538_, 1);
v_lctx_2645_ = lean_ctor_get(v_a_2538_, 2);
v_localInstances_2646_ = lean_ctor_get(v_a_2538_, 3);
v_defEqCtx_x3f_2647_ = lean_ctor_get(v_a_2538_, 4);
v_synthPendingDepth_2648_ = lean_ctor_get(v_a_2538_, 5);
v_customCanUnfoldPredicate_x3f_2649_ = lean_ctor_get(v_a_2538_, 6);
v_univApprox_2650_ = lean_ctor_get_uint8(v_a_2538_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2651_ = lean_ctor_get_uint8(v_a_2538_, sizeof(void*)*7 + 2);
v_cacheInferType_2652_ = lean_ctor_get_uint8(v_a_2538_, sizeof(void*)*7 + 3);
v___x_2653_ = 2;
lean_inc_ref(v_keyedConfig_2642_);
v___x_2654_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2653_, v_keyedConfig_2642_);
lean_inc(v_customCanUnfoldPredicate_x3f_2649_);
lean_inc(v_synthPendingDepth_2648_);
lean_inc(v_defEqCtx_x3f_2647_);
lean_inc_ref(v_localInstances_2646_);
lean_inc_ref(v_lctx_2645_);
lean_inc(v_zetaDeltaSet_2644_);
v___x_2655_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2655_, 0, v___x_2654_);
lean_ctor_set(v___x_2655_, 1, v_zetaDeltaSet_2644_);
lean_ctor_set(v___x_2655_, 2, v_lctx_2645_);
lean_ctor_set(v___x_2655_, 3, v_localInstances_2646_);
lean_ctor_set(v___x_2655_, 4, v_defEqCtx_x3f_2647_);
lean_ctor_set(v___x_2655_, 5, v_synthPendingDepth_2648_);
lean_ctor_set(v___x_2655_, 6, v_customCanUnfoldPredicate_x3f_2649_);
lean_ctor_set_uint8(v___x_2655_, sizeof(void*)*7, v_trackZetaDelta_2643_);
lean_ctor_set_uint8(v___x_2655_, sizeof(void*)*7 + 1, v_univApprox_2650_);
lean_ctor_set_uint8(v___x_2655_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2651_);
lean_ctor_set_uint8(v___x_2655_, sizeof(void*)*7 + 3, v_cacheInferType_2652_);
lean_inc(v_a_2541_);
lean_inc_ref(v_a_2540_);
lean_inc(v_a_2539_);
v___x_2656_ = lean_whnf(v_e_2535_, v___x_2655_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2656_) == 0)
{
lean_object* v_a_2657_; 
v_a_2657_ = lean_ctor_get(v___x_2656_, 0);
lean_inc(v_a_2657_);
lean_dec_ref_known(v___x_2656_, 1);
v_a_2553_ = v_a_2657_;
goto v___jp_2552_;
}
else
{
if (lean_obj_tag(v___x_2656_) == 0)
{
lean_object* v_a_2658_; 
v_a_2658_ = lean_ctor_get(v___x_2656_, 0);
lean_inc(v_a_2658_);
lean_dec_ref_known(v___x_2656_, 1);
v_a_2553_ = v_a_2658_;
goto v___jp_2552_;
}
else
{
lean_object* v_a_2659_; lean_object* v___x_2661_; uint8_t v_isShared_2662_; uint8_t v_isSharedCheck_2666_; 
v_a_2659_ = lean_ctor_get(v___x_2656_, 0);
v_isSharedCheck_2666_ = !lean_is_exclusive(v___x_2656_);
if (v_isSharedCheck_2666_ == 0)
{
v___x_2661_ = v___x_2656_;
v_isShared_2662_ = v_isSharedCheck_2666_;
goto v_resetjp_2660_;
}
else
{
lean_inc(v_a_2659_);
lean_dec(v___x_2656_);
v___x_2661_ = lean_box(0);
v_isShared_2662_ = v_isSharedCheck_2666_;
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
lean_object* v_reuseFailAlloc_2665_; 
v_reuseFailAlloc_2665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2665_, 0, v_a_2659_);
v___x_2664_ = v_reuseFailAlloc_2665_;
goto v_reusejp_2663_;
}
v_reusejp_2663_:
{
return v___x_2664_;
}
}
}
}
v___jp_2543_:
{
lean_object* v_expr_2546_; lean_object* v_proof_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; 
v_expr_2546_ = lean_ctor_get(v_____x_2545_, 0);
lean_inc_ref(v_expr_2546_);
v_proof_2547_ = lean_ctor_get(v_____x_2545_, 2);
lean_inc_ref(v_proof_2547_);
lean_dec_ref(v_____x_2545_);
v___x_2548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2548_, 0, v_proof_2547_);
v___x_2549_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2549_, 0, v_expr_2546_);
lean_ctor_set(v___x_2549_, 1, v___x_2548_);
lean_ctor_set_uint8(v___x_2549_, sizeof(void*)*2, v___y_2544_);
v___x_2550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2550_, 0, v___x_2549_);
v___x_2551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2551_, 0, v___x_2550_);
return v___x_2551_;
}
v___jp_2552_:
{
uint8_t v___x_2554_; 
v___x_2554_ = l_Lean_Expr_isApp(v_a_2553_);
if (v___x_2554_ == 0)
{
lean_object* v___x_2555_; lean_object* v___x_2556_; 
lean_dec_ref(v_a_2553_);
v___x_2555_ = lean_box(0);
v___x_2556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2556_, 0, v___x_2555_);
return v___x_2556_;
}
else
{
lean_object* v___x_2557_; 
v___x_2557_ = lp_mathlib_Qq_inferTypeQ_x27(v_a_2553_, v_a_2538_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2557_) == 0)
{
lean_object* v_a_2558_; lean_object* v_snd_2559_; lean_object* v_fst_2560_; lean_object* v_fst_2561_; lean_object* v_snd_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2633_; 
v_a_2558_ = lean_ctor_get(v___x_2557_, 0);
lean_inc(v_a_2558_);
lean_dec_ref_known(v___x_2557_, 1);
v_snd_2559_ = lean_ctor_get(v_a_2558_, 1);
lean_inc(v_snd_2559_);
v_fst_2560_ = lean_ctor_get(v_a_2558_, 0);
lean_inc(v_fst_2560_);
lean_dec(v_a_2558_);
v_fst_2561_ = lean_ctor_get(v_snd_2559_, 0);
v_snd_2562_ = lean_ctor_get(v_snd_2559_, 1);
v_isSharedCheck_2633_ = !lean_is_exclusive(v_snd_2559_);
if (v_isSharedCheck_2633_ == 0)
{
v___x_2564_ = v_snd_2559_;
v_isShared_2565_ = v_isSharedCheck_2633_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_snd_2562_);
lean_inc(v_fst_2561_);
lean_dec(v_snd_2559_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2633_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2569_; 
v___x_2566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___closed__1));
v___x_2567_ = lean_box(0);
lean_inc(v_fst_2560_);
if (v_isShared_2565_ == 0)
{
lean_ctor_set_tag(v___x_2564_, 1);
lean_ctor_set(v___x_2564_, 1, v___x_2567_);
lean_ctor_set(v___x_2564_, 0, v_fst_2560_);
v___x_2569_ = v___x_2564_;
goto v_reusejp_2568_;
}
else
{
lean_object* v_reuseFailAlloc_2632_; 
v_reuseFailAlloc_2632_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2632_, 0, v_fst_2560_);
lean_ctor_set(v_reuseFailAlloc_2632_, 1, v___x_2567_);
v___x_2569_ = v_reuseFailAlloc_2632_;
goto v_reusejp_2568_;
}
v_reusejp_2568_:
{
lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; 
v___x_2570_ = l_Lean_Expr_const___override(v___x_2566_, v___x_2569_);
lean_inc(v_fst_2561_);
v___x_2571_ = l_Lean_Expr_app___override(v___x_2570_, v_fst_2561_);
v___x_2572_ = lean_box(0);
v___x_2573_ = l_Lean_Meta_trySynthInstance(v___x_2571_, v___x_2572_, v_a_2538_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2573_) == 0)
{
lean_object* v_a_2574_; lean_object* v___x_2576_; uint8_t v_isShared_2577_; uint8_t v_isSharedCheck_2623_; 
v_a_2574_ = lean_ctor_get(v___x_2573_, 0);
v_isSharedCheck_2623_ = !lean_is_exclusive(v___x_2573_);
if (v_isSharedCheck_2623_ == 0)
{
v___x_2576_ = v___x_2573_;
v_isShared_2577_ = v_isSharedCheck_2623_;
goto v_resetjp_2575_;
}
else
{
lean_inc(v_a_2574_);
lean_dec(v___x_2573_);
v___x_2576_ = lean_box(0);
v_isShared_2577_ = v_isSharedCheck_2623_;
goto v_resetjp_2575_;
}
v_resetjp_2575_:
{
if (lean_obj_tag(v_a_2574_) == 1)
{
lean_object* v_a_2578_; lean_object* v___x_2579_; 
lean_del_object(v___x_2576_);
v_a_2578_ = lean_ctor_get(v_a_2574_, 0);
lean_inc_n(v_a_2578_, 2);
lean_dec_ref_known(v_a_2574_, 1);
lean_inc(v_fst_2561_);
lean_inc(v_fst_2560_);
v___x_2579_ = lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(v_fst_2560_, v_fst_2561_, v_a_2578_, v_a_2538_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2579_) == 0)
{
lean_object* v_a_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; 
v_a_2580_ = lean_ctor_get(v___x_2579_, 0);
lean_inc_n(v_a_2580_, 2);
lean_dec_ref_known(v___x_2579_, 1);
lean_inc(v_a_2578_);
lean_inc(v_fst_2561_);
lean_inc(v_fst_2560_);
v___x_2581_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_fst_2560_, v_fst_2561_, v_a_2578_, v_a_2580_);
lean_inc(v_snd_2562_);
lean_inc_ref(v___x_2581_);
v___x_2582_ = lp_mathlib_Mathlib_Tactic_Ring_Common_isAtomOrDerivable___redArg(v___x_2581_, v_a_2580_, v_snd_2562_, v_a_2538_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2582_) == 0)
{
lean_object* v_a_2583_; lean_object* v___x_2585_; uint8_t v_isShared_2586_; uint8_t v_isSharedCheck_2603_; 
v_a_2583_ = lean_ctor_get(v___x_2582_, 0);
v_isSharedCheck_2603_ = !lean_is_exclusive(v___x_2582_);
if (v_isSharedCheck_2603_ == 0)
{
v___x_2585_ = v___x_2582_;
v_isShared_2586_ = v_isSharedCheck_2603_;
goto v_resetjp_2584_;
}
else
{
lean_inc(v_a_2583_);
lean_dec(v___x_2582_);
v___x_2585_ = lean_box(0);
v_isShared_2586_ = v_isSharedCheck_2603_;
goto v_resetjp_2584_;
}
v_resetjp_2584_:
{
if (lean_obj_tag(v_a_2583_) == 0)
{
lean_object* v___x_2587_; lean_object* v___x_2588_; 
lean_del_object(v___x_2585_);
v___x_2587_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
v___x_2588_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_2587_, v_fst_2560_, v_fst_2561_, v_a_2578_, v___x_2581_, v_a_2580_, v_snd_2562_, v_a_2536_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_, v_a_2541_);
if (lean_obj_tag(v___x_2588_) == 0)
{
lean_object* v_a_2589_; 
v_a_2589_ = lean_ctor_get(v___x_2588_, 0);
lean_inc(v_a_2589_);
lean_dec_ref_known(v___x_2588_, 1);
v___y_2544_ = v___x_2554_;
v_____x_2545_ = v_a_2589_;
goto v___jp_2543_;
}
else
{
lean_object* v_a_2590_; lean_object* v___x_2592_; uint8_t v_isShared_2593_; uint8_t v_isSharedCheck_2597_; 
v_a_2590_ = lean_ctor_get(v___x_2588_, 0);
v_isSharedCheck_2597_ = !lean_is_exclusive(v___x_2588_);
if (v_isSharedCheck_2597_ == 0)
{
v___x_2592_ = v___x_2588_;
v_isShared_2593_ = v_isSharedCheck_2597_;
goto v_resetjp_2591_;
}
else
{
lean_inc(v_a_2590_);
lean_dec(v___x_2588_);
v___x_2592_ = lean_box(0);
v_isShared_2593_ = v_isSharedCheck_2597_;
goto v_resetjp_2591_;
}
v_resetjp_2591_:
{
lean_object* v___x_2595_; 
if (v_isShared_2593_ == 0)
{
v___x_2595_ = v___x_2592_;
goto v_reusejp_2594_;
}
else
{
lean_object* v_reuseFailAlloc_2596_; 
v_reuseFailAlloc_2596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2596_, 0, v_a_2590_);
v___x_2595_ = v_reuseFailAlloc_2596_;
goto v_reusejp_2594_;
}
v_reusejp_2594_:
{
return v___x_2595_;
}
}
}
}
else
{
lean_object* v_val_2598_; 
lean_dec_ref(v___x_2581_);
lean_dec(v_a_2580_);
lean_dec(v_a_2578_);
lean_dec(v_snd_2562_);
lean_dec(v_fst_2561_);
lean_dec(v_fst_2560_);
v_val_2598_ = lean_ctor_get(v_a_2583_, 0);
lean_inc(v_val_2598_);
lean_dec_ref_known(v_a_2583_, 1);
if (lean_obj_tag(v_val_2598_) == 0)
{
lean_object* v___x_2600_; 
if (v_isShared_2586_ == 0)
{
lean_ctor_set(v___x_2585_, 0, v___x_2572_);
v___x_2600_ = v___x_2585_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2601_; 
v_reuseFailAlloc_2601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2601_, 0, v___x_2572_);
v___x_2600_ = v_reuseFailAlloc_2601_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
return v___x_2600_;
}
}
else
{
lean_object* v_val_2602_; 
lean_del_object(v___x_2585_);
v_val_2602_ = lean_ctor_get(v_val_2598_, 0);
lean_inc(v_val_2602_);
lean_dec_ref_known(v_val_2598_, 1);
v___y_2544_ = v___x_2554_;
v_____x_2545_ = v_val_2602_;
goto v___jp_2543_;
}
}
}
}
else
{
lean_object* v_a_2604_; lean_object* v___x_2606_; uint8_t v_isShared_2607_; uint8_t v_isSharedCheck_2611_; 
lean_dec_ref(v___x_2581_);
lean_dec(v_a_2580_);
lean_dec(v_a_2578_);
lean_dec(v_snd_2562_);
lean_dec(v_fst_2561_);
lean_dec(v_fst_2560_);
v_a_2604_ = lean_ctor_get(v___x_2582_, 0);
v_isSharedCheck_2611_ = !lean_is_exclusive(v___x_2582_);
if (v_isSharedCheck_2611_ == 0)
{
v___x_2606_ = v___x_2582_;
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
else
{
lean_inc(v_a_2604_);
lean_dec(v___x_2582_);
v___x_2606_ = lean_box(0);
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
v_resetjp_2605_:
{
lean_object* v___x_2609_; 
if (v_isShared_2607_ == 0)
{
v___x_2609_ = v___x_2606_;
goto v_reusejp_2608_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v_a_2604_);
v___x_2609_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2608_;
}
v_reusejp_2608_:
{
return v___x_2609_;
}
}
}
}
else
{
lean_object* v_a_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2619_; 
lean_dec(v_a_2578_);
lean_dec(v_snd_2562_);
lean_dec(v_fst_2561_);
lean_dec(v_fst_2560_);
v_a_2612_ = lean_ctor_get(v___x_2579_, 0);
v_isSharedCheck_2619_ = !lean_is_exclusive(v___x_2579_);
if (v_isSharedCheck_2619_ == 0)
{
v___x_2614_ = v___x_2579_;
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_a_2612_);
lean_dec(v___x_2579_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2617_; 
if (v_isShared_2615_ == 0)
{
v___x_2617_ = v___x_2614_;
goto v_reusejp_2616_;
}
else
{
lean_object* v_reuseFailAlloc_2618_; 
v_reuseFailAlloc_2618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2618_, 0, v_a_2612_);
v___x_2617_ = v_reuseFailAlloc_2618_;
goto v_reusejp_2616_;
}
v_reusejp_2616_:
{
return v___x_2617_;
}
}
}
}
else
{
lean_object* v___x_2621_; 
lean_dec(v_a_2574_);
lean_dec(v_snd_2562_);
lean_dec(v_fst_2561_);
lean_dec(v_fst_2560_);
if (v_isShared_2577_ == 0)
{
lean_ctor_set(v___x_2576_, 0, v___x_2572_);
v___x_2621_ = v___x_2576_;
goto v_reusejp_2620_;
}
else
{
lean_object* v_reuseFailAlloc_2622_; 
v_reuseFailAlloc_2622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2622_, 0, v___x_2572_);
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
lean_dec(v_snd_2562_);
lean_dec(v_fst_2561_);
lean_dec(v_fst_2560_);
v_a_2624_ = lean_ctor_get(v___x_2573_, 0);
v_isSharedCheck_2631_ = !lean_is_exclusive(v___x_2573_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2626_ = v___x_2573_;
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
else
{
lean_inc(v_a_2624_);
lean_dec(v___x_2573_);
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
}
else
{
lean_object* v_a_2634_; lean_object* v___x_2636_; uint8_t v_isShared_2637_; uint8_t v_isSharedCheck_2641_; 
v_a_2634_ = lean_ctor_get(v___x_2557_, 0);
v_isSharedCheck_2641_ = !lean_is_exclusive(v___x_2557_);
if (v_isSharedCheck_2641_ == 0)
{
v___x_2636_ = v___x_2557_;
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
else
{
lean_inc(v_a_2634_);
lean_dec(v___x_2557_);
v___x_2636_ = lean_box(0);
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
v_resetjp_2635_:
{
lean_object* v___x_2639_; 
if (v_isShared_2637_ == 0)
{
v___x_2639_ = v___x_2636_;
goto v_reusejp_2638_;
}
else
{
lean_object* v_reuseFailAlloc_2640_; 
v_reuseFailAlloc_2640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2640_, 0, v_a_2634_);
v___x_2639_ = v_reuseFailAlloc_2640_;
goto v_reusejp_2638_;
}
v_reusejp_2638_:
{
return v___x_2639_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f___boxed(lean_object* v_e_2667_, lean_object* v_a_2668_, lean_object* v_a_2669_, lean_object* v_a_2670_, lean_object* v_a_2671_, lean_object* v_a_2672_, lean_object* v_a_2673_, lean_object* v_a_2674_){
_start:
{
lean_object* v_res_2675_; 
v_res_2675_ = lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f(v_e_2667_, v_a_2668_, v_a_2669_, v_a_2670_, v_a_2671_, v_a_2672_, v_a_2673_);
lean_dec(v_a_2673_);
lean_dec_ref(v_a_2672_);
lean_dec(v_a_2671_);
lean_dec_ref(v_a_2670_);
lean_dec(v_a_2669_);
lean_dec_ref(v_a_2668_);
return v_res_2675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr(lean_object* v_e_2676_, lean_object* v_a_2677_, lean_object* v_a_2678_, lean_object* v_a_2679_, lean_object* v_a_2680_, lean_object* v_a_2681_, lean_object* v_a_2682_){
_start:
{
lean_object* v___x_2684_; 
v___x_2684_ = lp_mathlib_Mathlib_Tactic_RingNF_evalExpr_x3f(v_e_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_, v_a_2682_);
if (lean_obj_tag(v___x_2684_) == 0)
{
lean_object* v_a_2685_; lean_object* v___x_2687_; uint8_t v_isShared_2688_; uint8_t v_isSharedCheck_2695_; 
v_a_2685_ = lean_ctor_get(v___x_2684_, 0);
v_isSharedCheck_2695_ = !lean_is_exclusive(v___x_2684_);
if (v_isSharedCheck_2695_ == 0)
{
v___x_2687_ = v___x_2684_;
v_isShared_2688_ = v_isSharedCheck_2695_;
goto v_resetjp_2686_;
}
else
{
lean_inc(v_a_2685_);
lean_dec(v___x_2684_);
v___x_2687_ = lean_box(0);
v_isShared_2688_ = v_isSharedCheck_2695_;
goto v_resetjp_2686_;
}
v_resetjp_2686_:
{
if (lean_obj_tag(v_a_2685_) == 1)
{
lean_object* v_val_2689_; lean_object* v___x_2691_; 
v_val_2689_ = lean_ctor_get(v_a_2685_, 0);
lean_inc(v_val_2689_);
lean_dec_ref_known(v_a_2685_, 1);
if (v_isShared_2688_ == 0)
{
lean_ctor_set(v___x_2687_, 0, v_val_2689_);
v___x_2691_ = v___x_2687_;
goto v_reusejp_2690_;
}
else
{
lean_object* v_reuseFailAlloc_2692_; 
v_reuseFailAlloc_2692_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2692_, 0, v_val_2689_);
v___x_2691_ = v_reuseFailAlloc_2692_;
goto v_reusejp_2690_;
}
v_reusejp_2690_:
{
return v___x_2691_;
}
}
else
{
lean_object* v___x_2693_; lean_object* v___x_2694_; 
lean_del_object(v___x_2687_);
lean_dec(v_a_2685_);
v___x_2693_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_2694_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_2693_, v_a_2679_, v_a_2680_, v_a_2681_, v_a_2682_);
return v___x_2694_;
}
}
}
else
{
lean_object* v_a_2696_; lean_object* v___x_2698_; uint8_t v_isShared_2699_; uint8_t v_isSharedCheck_2703_; 
v_a_2696_ = lean_ctor_get(v___x_2684_, 0);
v_isSharedCheck_2703_ = !lean_is_exclusive(v___x_2684_);
if (v_isSharedCheck_2703_ == 0)
{
v___x_2698_ = v___x_2684_;
v_isShared_2699_ = v_isSharedCheck_2703_;
goto v_resetjp_2697_;
}
else
{
lean_inc(v_a_2696_);
lean_dec(v___x_2684_);
v___x_2698_ = lean_box(0);
v_isShared_2699_ = v_isSharedCheck_2703_;
goto v_resetjp_2697_;
}
v_resetjp_2697_:
{
lean_object* v___x_2701_; 
if (v_isShared_2699_ == 0)
{
v___x_2701_ = v___x_2698_;
goto v_reusejp_2700_;
}
else
{
lean_object* v_reuseFailAlloc_2702_; 
v_reuseFailAlloc_2702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2702_, 0, v_a_2696_);
v___x_2701_ = v_reuseFailAlloc_2702_;
goto v_reusejp_2700_;
}
v_reusejp_2700_:
{
return v___x_2701_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_evalExpr___boxed(lean_object* v_e_2704_, lean_object* v_a_2705_, lean_object* v_a_2706_, lean_object* v_a_2707_, lean_object* v_a_2708_, lean_object* v_a_2709_, lean_object* v_a_2710_, lean_object* v_a_2711_){
_start:
{
lean_object* v_res_2712_; 
v_res_2712_ = lp_mathlib_Mathlib_Tactic_RingNF_evalExpr(v_e_2704_, v_a_2705_, v_a_2706_, v_a_2707_, v_a_2708_, v_a_2709_, v_a_2710_);
lean_dec(v_a_2710_);
lean_dec_ref(v_a_2709_);
lean_dec(v_a_2708_);
lean_dec_ref(v_a_2707_);
lean_dec(v_a_2706_);
lean_dec_ref(v_a_2705_);
return v_res_2712_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2713_; 
v___x_2713_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2713_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2714_; lean_object* v___x_2715_; 
v___x_2714_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__0);
v___x_2715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2715_, 0, v___x_2714_);
return v___x_2715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0(lean_object* v_00_u03b2_2716_){
_start:
{
lean_object* v___x_2717_; 
v___x_2717_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0___closed__1);
return v___x_2717_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0(void){
_start:
{
lean_object* v___x_2718_; 
v___x_2718_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2718_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1(void){
_start:
{
lean_object* v___x_2719_; lean_object* v___x_2720_; 
v___x_2719_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__0);
v___x_2720_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2720_, 0, v___x_2719_);
return v___x_2720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1(lean_object* v_00_u03b2_2721_){
_start:
{
lean_object* v___x_2722_; 
v___x_2722_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1___closed__1);
return v___x_2722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2(lean_object* v_x_2723_, lean_object* v_x_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_){
_start:
{
if (lean_obj_tag(v_x_2724_) == 0)
{
lean_object* v___x_2730_; 
v___x_2730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2730_, 0, v_x_2723_);
return v___x_2730_;
}
else
{
lean_object* v_head_2731_; lean_object* v_tail_2732_; uint8_t v___x_2733_; uint8_t v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; 
v_head_2731_ = lean_ctor_get(v_x_2724_, 0);
lean_inc(v_head_2731_);
v_tail_2732_ = lean_ctor_get(v_x_2724_, 1);
lean_inc(v_tail_2732_);
lean_dec_ref_known(v_x_2724_, 2);
v___x_2733_ = 1;
v___x_2734_ = 0;
v___x_2735_ = lean_unsigned_to_nat(1000u);
v___x_2736_ = l_Lean_Meta_SimpTheorems_addConst(v_x_2723_, v_head_2731_, v___x_2733_, v___x_2734_, v___x_2735_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_);
if (lean_obj_tag(v___x_2736_) == 0)
{
lean_object* v_a_2737_; 
v_a_2737_ = lean_ctor_get(v___x_2736_, 0);
lean_inc(v_a_2737_);
lean_dec_ref_known(v___x_2736_, 1);
v_x_2723_ = v_a_2737_;
v_x_2724_ = v_tail_2732_;
goto _start;
}
else
{
lean_dec(v_tail_2732_);
return v___x_2736_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2___boxed(lean_object* v_x_2739_, lean_object* v_x_2740_, lean_object* v___y_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_){
_start:
{
lean_object* v_res_2746_; 
v_res_2746_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2(v_x_2739_, v_x_2740_, v___y_2741_, v___y_2742_, v___y_2743_, v___y_2744_);
lean_dec(v___y_2744_);
lean_dec_ref(v___y_2743_);
lean_dec(v___y_2742_);
lean_dec_ref(v___y_2741_);
return v_res_2746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3(lean_object* v_x_2747_, lean_object* v_x_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_){
_start:
{
if (lean_obj_tag(v_x_2748_) == 0)
{
lean_object* v___x_2754_; 
v___x_2754_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2754_, 0, v_x_2747_);
return v___x_2754_;
}
else
{
lean_object* v_head_2755_; lean_object* v_tail_2756_; uint8_t v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; 
v_head_2755_ = lean_ctor_get(v_x_2748_, 0);
lean_inc(v_head_2755_);
v_tail_2756_ = lean_ctor_get(v_x_2748_, 1);
lean_inc(v_tail_2756_);
lean_dec_ref_known(v_x_2748_, 2);
v___x_2757_ = 0;
v___x_2758_ = lean_unsigned_to_nat(1000u);
v___x_2759_ = l_Lean_Meta_SimpTheorems_addConst(v_x_2747_, v_head_2755_, v___x_2757_, v___x_2757_, v___x_2758_, v___y_2749_, v___y_2750_, v___y_2751_, v___y_2752_);
if (lean_obj_tag(v___x_2759_) == 0)
{
lean_object* v_a_2760_; 
v_a_2760_ = lean_ctor_get(v___x_2759_, 0);
lean_inc(v_a_2760_);
lean_dec_ref_known(v___x_2759_, 1);
v_x_2747_ = v_a_2760_;
v_x_2748_ = v_tail_2756_;
goto _start;
}
else
{
lean_dec(v_tail_2756_);
return v___x_2759_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3___boxed(lean_object* v_x_2762_, lean_object* v_x_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_, lean_object* v___y_2768_){
_start:
{
lean_object* v_res_2769_; 
v_res_2769_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3(v_x_2762_, v_x_2763_, v___y_2764_, v___y_2765_, v___y_2766_, v___y_2767_);
lean_dec(v___y_2767_);
lean_dec_ref(v___y_2766_);
lean_dec(v___y_2765_);
lean_dec_ref(v___y_2764_);
return v_res_2769_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0(void){
_start:
{
lean_object* v___x_2770_; 
v___x_2770_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_2770_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1(void){
_start:
{
lean_object* v___x_2771_; 
v___x_2771_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__0(lean_box(0));
return v___x_2771_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2(void){
_start:
{
lean_object* v___x_2772_; 
v___x_2772_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_RingNF_cleanup_spec__1(lean_box(0));
return v___x_2772_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3(void){
_start:
{
lean_object* v___x_2773_; 
v___x_2773_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2773_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4(void){
_start:
{
lean_object* v___x_2774_; lean_object* v___x_2775_; 
v___x_2774_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__3);
v___x_2775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2775_, 0, v___x_2774_);
return v___x_2775_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5(void){
_start:
{
lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v_thms_2780_; 
v___x_2776_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4);
v___x_2777_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__2);
v___x_2778_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__1);
v___x_2779_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__0);
v_thms_2780_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_thms_2780_, 0, v___x_2779_);
lean_ctor_set(v_thms_2780_, 1, v___x_2779_);
lean_ctor_set(v_thms_2780_, 2, v___x_2778_);
lean_ctor_set(v_thms_2780_, 3, v___x_2777_);
lean_ctor_set(v_thms_2780_, 4, v___x_2778_);
lean_ctor_set(v_thms_2780_, 5, v___x_2776_);
return v_thms_2780_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45(void){
_start:
{
lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; 
v___x_2889_ = lean_unsigned_to_nat(0u);
v___x_2890_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4);
v___x_2891_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2890_);
lean_ctor_set(v___x_2891_, 1, v___x_2889_);
return v___x_2891_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46(void){
_start:
{
lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; 
v___x_2892_ = lean_unsigned_to_nat(32u);
v___x_2893_ = lean_mk_empty_array_with_capacity(v___x_2892_);
v___x_2894_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2894_, 0, v___x_2893_);
return v___x_2894_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47(void){
_start:
{
size_t v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; 
v___x_2895_ = ((size_t)5ULL);
v___x_2896_ = lean_unsigned_to_nat(0u);
v___x_2897_ = lean_unsigned_to_nat(32u);
v___x_2898_ = lean_mk_empty_array_with_capacity(v___x_2897_);
v___x_2899_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__46);
v___x_2900_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2900_, 0, v___x_2899_);
lean_ctor_set(v___x_2900_, 1, v___x_2898_);
lean_ctor_set(v___x_2900_, 2, v___x_2896_);
lean_ctor_set(v___x_2900_, 3, v___x_2896_);
lean_ctor_set_usize(v___x_2900_, 4, v___x_2895_);
return v___x_2900_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48(void){
_start:
{
lean_object* v___x_2901_; lean_object* v___x_2902_; lean_object* v___x_2903_; 
v___x_2901_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__47);
v___x_2902_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__4);
v___x_2903_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2903_, 0, v___x_2902_);
lean_ctor_set(v___x_2903_, 1, v___x_2902_);
lean_ctor_set(v___x_2903_, 2, v___x_2902_);
lean_ctor_set(v___x_2903_, 3, v___x_2901_);
return v___x_2903_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49(void){
_start:
{
lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; 
v___x_2904_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__48);
v___x_2905_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__45);
v___x_2906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2906_, 0, v___x_2905_);
lean_ctor_set(v___x_2906_, 1, v___x_2904_);
return v___x_2906_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51(void){
_start:
{
lean_object* v___x_2909_; lean_object* v___x_2910_; 
v___x_2909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__50));
v___x_2910_ = l_Lean_Meta_Simp_mkDefaultMethodsCore(v___x_2909_);
return v___x_2910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup(lean_object* v_cfg_2911_, lean_object* v_r_2912_, lean_object* v_a_2913_, lean_object* v_a_2914_, lean_object* v_a_2915_, lean_object* v_a_2916_){
_start:
{
uint8_t v_mode_2918_; 
v_mode_2918_ = lean_ctor_get_uint8(v_cfg_2911_, sizeof(void*)*1 + 1);
if (v_mode_2918_ == 0)
{
lean_object* v_toConfig_2919_; lean_object* v_thms_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; 
v_toConfig_2919_ = lean_ctor_get(v_cfg_2911_, 0);
v_thms_2920_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__5);
v___x_2921_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__20));
v___x_2922_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__2(v_thms_2920_, v___x_2921_, v_a_2913_, v_a_2914_, v_a_2915_, v_a_2916_);
if (lean_obj_tag(v___x_2922_) == 0)
{
lean_object* v_a_2923_; lean_object* v___x_2924_; lean_object* v___x_2925_; 
v_a_2923_ = lean_ctor_get(v___x_2922_, 0);
lean_inc(v_a_2923_);
lean_dec_ref_known(v___x_2922_, 1);
v___x_2924_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__44));
v___x_2925_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_RingNF_cleanup_spec__3(v_a_2923_, v___x_2924_, v_a_2913_, v_a_2914_, v_a_2915_, v_a_2916_);
if (lean_obj_tag(v___x_2925_) == 0)
{
lean_object* v_a_2926_; lean_object* v___x_2927_; 
v_a_2926_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_a_2926_);
lean_dec_ref_known(v___x_2925_, 1);
v___x_2927_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_2916_);
if (lean_obj_tag(v___x_2927_) == 0)
{
lean_object* v_a_2928_; uint8_t v_zetaDelta_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; uint8_t v___x_2932_; uint8_t v___x_2933_; uint8_t v___x_2934_; lean_object* v___x_2935_; lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; 
v_a_2928_ = lean_ctor_get(v___x_2927_, 0);
lean_inc(v_a_2928_);
lean_dec_ref_known(v___x_2927_, 1);
v_zetaDelta_2929_ = lean_ctor_get_uint8(v_toConfig_2919_, 1);
v___x_2930_ = lean_unsigned_to_nat(100000u);
v___x_2931_ = lean_unsigned_to_nat(2u);
v___x_2932_ = 0;
v___x_2933_ = 1;
v___x_2934_ = 0;
v___x_2935_ = lean_box(0);
v___x_2936_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_2936_, 0, v___x_2930_);
lean_ctor_set(v___x_2936_, 1, v___x_2931_);
lean_ctor_set(v___x_2936_, 2, v___x_2935_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 1, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 2, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 3, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 4, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 5, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 6, v___x_2934_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 7, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 8, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 9, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 10, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 11, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 12, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 13, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 14, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 15, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 16, v_zetaDelta_2929_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 17, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 18, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 19, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 20, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 21, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 22, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 23, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 24, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 25, v___x_2933_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 26, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 27, v___x_2932_);
lean_ctor_set_uint8(v___x_2936_, sizeof(void*)*3 + 28, v___x_2932_);
v___x_2937_ = lean_unsigned_to_nat(1u);
v___x_2938_ = lean_mk_empty_array_with_capacity(v___x_2937_);
v___x_2939_ = lean_array_push(v___x_2938_, v_a_2926_);
v___x_2940_ = l_Lean_Options_empty;
v___x_2941_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_2936_, v___x_2939_, v_a_2928_, v___x_2940_, v_a_2913_, v_a_2915_, v_a_2916_);
if (lean_obj_tag(v___x_2941_) == 0)
{
lean_object* v_a_2942_; lean_object* v_expr_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; 
v_a_2942_ = lean_ctor_get(v___x_2941_, 0);
lean_inc(v_a_2942_);
lean_dec_ref_known(v___x_2941_, 1);
v_expr_2943_ = lean_ctor_get(v_r_2912_, 0);
v___x_2944_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__49);
v___x_2945_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51, &lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_cleanup___closed__51);
lean_inc_ref(v_expr_2943_);
v___x_2946_ = l_Lean_Meta_Simp_main(v_expr_2943_, v_a_2942_, v___x_2944_, v___x_2945_, v_a_2913_, v_a_2914_, v_a_2915_, v_a_2916_);
if (lean_obj_tag(v___x_2946_) == 0)
{
lean_object* v_a_2947_; lean_object* v_fst_2948_; lean_object* v___x_2949_; 
v_a_2947_ = lean_ctor_get(v___x_2946_, 0);
lean_inc(v_a_2947_);
lean_dec_ref_known(v___x_2946_, 1);
v_fst_2948_ = lean_ctor_get(v_a_2947_, 0);
lean_inc(v_fst_2948_);
lean_dec(v_a_2947_);
v___x_2949_ = l_Lean_Meta_Simp_Result_mkEqTrans(v_r_2912_, v_fst_2948_, v_a_2913_, v_a_2914_, v_a_2915_, v_a_2916_);
return v___x_2949_;
}
else
{
lean_object* v_a_2950_; lean_object* v___x_2952_; uint8_t v_isShared_2953_; uint8_t v_isSharedCheck_2957_; 
lean_dec_ref(v_r_2912_);
v_a_2950_ = lean_ctor_get(v___x_2946_, 0);
v_isSharedCheck_2957_ = !lean_is_exclusive(v___x_2946_);
if (v_isSharedCheck_2957_ == 0)
{
v___x_2952_ = v___x_2946_;
v_isShared_2953_ = v_isSharedCheck_2957_;
goto v_resetjp_2951_;
}
else
{
lean_inc(v_a_2950_);
lean_dec(v___x_2946_);
v___x_2952_ = lean_box(0);
v_isShared_2953_ = v_isSharedCheck_2957_;
goto v_resetjp_2951_;
}
v_resetjp_2951_:
{
lean_object* v___x_2955_; 
if (v_isShared_2953_ == 0)
{
v___x_2955_ = v___x_2952_;
goto v_reusejp_2954_;
}
else
{
lean_object* v_reuseFailAlloc_2956_; 
v_reuseFailAlloc_2956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2956_, 0, v_a_2950_);
v___x_2955_ = v_reuseFailAlloc_2956_;
goto v_reusejp_2954_;
}
v_reusejp_2954_:
{
return v___x_2955_;
}
}
}
}
else
{
lean_object* v_a_2958_; lean_object* v___x_2960_; uint8_t v_isShared_2961_; uint8_t v_isSharedCheck_2965_; 
lean_dec_ref(v_r_2912_);
v_a_2958_ = lean_ctor_get(v___x_2941_, 0);
v_isSharedCheck_2965_ = !lean_is_exclusive(v___x_2941_);
if (v_isSharedCheck_2965_ == 0)
{
v___x_2960_ = v___x_2941_;
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
else
{
lean_inc(v_a_2958_);
lean_dec(v___x_2941_);
v___x_2960_ = lean_box(0);
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
v_resetjp_2959_:
{
lean_object* v___x_2963_; 
if (v_isShared_2961_ == 0)
{
v___x_2963_ = v___x_2960_;
goto v_reusejp_2962_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v_a_2958_);
v___x_2963_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2962_;
}
v_reusejp_2962_:
{
return v___x_2963_;
}
}
}
}
else
{
lean_object* v_a_2966_; lean_object* v___x_2968_; uint8_t v_isShared_2969_; uint8_t v_isSharedCheck_2973_; 
lean_dec(v_a_2926_);
lean_dec_ref(v_r_2912_);
v_a_2966_ = lean_ctor_get(v___x_2927_, 0);
v_isSharedCheck_2973_ = !lean_is_exclusive(v___x_2927_);
if (v_isSharedCheck_2973_ == 0)
{
v___x_2968_ = v___x_2927_;
v_isShared_2969_ = v_isSharedCheck_2973_;
goto v_resetjp_2967_;
}
else
{
lean_inc(v_a_2966_);
lean_dec(v___x_2927_);
v___x_2968_ = lean_box(0);
v_isShared_2969_ = v_isSharedCheck_2973_;
goto v_resetjp_2967_;
}
v_resetjp_2967_:
{
lean_object* v___x_2971_; 
if (v_isShared_2969_ == 0)
{
v___x_2971_ = v___x_2968_;
goto v_reusejp_2970_;
}
else
{
lean_object* v_reuseFailAlloc_2972_; 
v_reuseFailAlloc_2972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2972_, 0, v_a_2966_);
v___x_2971_ = v_reuseFailAlloc_2972_;
goto v_reusejp_2970_;
}
v_reusejp_2970_:
{
return v___x_2971_;
}
}
}
}
else
{
lean_object* v_a_2974_; lean_object* v___x_2976_; uint8_t v_isShared_2977_; uint8_t v_isSharedCheck_2981_; 
lean_dec_ref(v_r_2912_);
v_a_2974_ = lean_ctor_get(v___x_2925_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2925_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2976_ = v___x_2925_;
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
else
{
lean_inc(v_a_2974_);
lean_dec(v___x_2925_);
v___x_2976_ = lean_box(0);
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
v_resetjp_2975_:
{
lean_object* v___x_2979_; 
if (v_isShared_2977_ == 0)
{
v___x_2979_ = v___x_2976_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_a_2974_);
v___x_2979_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
return v___x_2979_;
}
}
}
}
else
{
lean_object* v_a_2982_; lean_object* v___x_2984_; uint8_t v_isShared_2985_; uint8_t v_isSharedCheck_2989_; 
lean_dec_ref(v_r_2912_);
v_a_2982_ = lean_ctor_get(v___x_2922_, 0);
v_isSharedCheck_2989_ = !lean_is_exclusive(v___x_2922_);
if (v_isSharedCheck_2989_ == 0)
{
v___x_2984_ = v___x_2922_;
v_isShared_2985_ = v_isSharedCheck_2989_;
goto v_resetjp_2983_;
}
else
{
lean_inc(v_a_2982_);
lean_dec(v___x_2922_);
v___x_2984_ = lean_box(0);
v_isShared_2985_ = v_isSharedCheck_2989_;
goto v_resetjp_2983_;
}
v_resetjp_2983_:
{
lean_object* v___x_2987_; 
if (v_isShared_2985_ == 0)
{
v___x_2987_ = v___x_2984_;
goto v_reusejp_2986_;
}
else
{
lean_object* v_reuseFailAlloc_2988_; 
v_reuseFailAlloc_2988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2988_, 0, v_a_2982_);
v___x_2987_ = v_reuseFailAlloc_2988_;
goto v_reusejp_2986_;
}
v_reusejp_2986_:
{
return v___x_2987_;
}
}
}
}
else
{
lean_object* v___x_2990_; 
v___x_2990_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2990_, 0, v_r_2912_);
return v___x_2990_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_cleanup___boxed(lean_object* v_cfg_2991_, lean_object* v_r_2992_, lean_object* v_a_2993_, lean_object* v_a_2994_, lean_object* v_a_2995_, lean_object* v_a_2996_, lean_object* v_a_2997_){
_start:
{
lean_object* v_res_2998_; 
v_res_2998_ = lp_mathlib_Mathlib_Tactic_RingNF_cleanup(v_cfg_2991_, v_r_2992_, v_a_2993_, v_a_2994_, v_a_2995_, v_a_2996_);
lean_dec(v_a_2996_);
lean_dec_ref(v_a_2995_);
lean_dec(v_a_2994_);
lean_dec_ref(v_a_2993_);
lean_dec_ref(v_cfg_2991_);
return v_res_2998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_(lean_object* v_e_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_){
_start:
{
lean_object* v___x_3012_; lean_object* v___x_3013_; uint8_t v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; 
v___x_3012_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_));
v___x_3013_ = lean_box(0);
v___x_3014_ = 1;
v___x_3015_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_3015_, 0, v_e_3006_);
lean_ctor_set(v___x_3015_, 1, v___x_3013_);
lean_ctor_set_uint8(v___x_3015_, sizeof(void*)*2, v___x_3014_);
v___x_3016_ = lp_mathlib_Mathlib_Tactic_RingNF_cleanup(v___x_3012_, v___x_3015_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_);
if (lean_obj_tag(v___x_3016_) == 0)
{
lean_object* v_a_3017_; lean_object* v___x_3019_; uint8_t v_isShared_3020_; uint8_t v_isSharedCheck_3025_; 
v_a_3017_ = lean_ctor_get(v___x_3016_, 0);
v_isSharedCheck_3025_ = !lean_is_exclusive(v___x_3016_);
if (v_isSharedCheck_3025_ == 0)
{
v___x_3019_ = v___x_3016_;
v_isShared_3020_ = v_isSharedCheck_3025_;
goto v_resetjp_3018_;
}
else
{
lean_inc(v_a_3017_);
lean_dec(v___x_3016_);
v___x_3019_ = lean_box(0);
v_isShared_3020_ = v_isSharedCheck_3025_;
goto v_resetjp_3018_;
}
v_resetjp_3018_:
{
lean_object* v_expr_3021_; lean_object* v___x_3023_; 
v_expr_3021_ = lean_ctor_get(v_a_3017_, 0);
lean_inc_ref(v_expr_3021_);
lean_dec(v_a_3017_);
if (v_isShared_3020_ == 0)
{
lean_ctor_set(v___x_3019_, 0, v_expr_3021_);
v___x_3023_ = v___x_3019_;
goto v_reusejp_3022_;
}
else
{
lean_object* v_reuseFailAlloc_3024_; 
v_reuseFailAlloc_3024_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3024_, 0, v_expr_3021_);
v___x_3023_ = v_reuseFailAlloc_3024_;
goto v_reusejp_3022_;
}
v_reusejp_3022_:
{
return v___x_3023_;
}
}
}
else
{
lean_object* v_a_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3033_; 
v_a_3026_ = lean_ctor_get(v___x_3016_, 0);
v_isSharedCheck_3033_ = !lean_is_exclusive(v___x_3016_);
if (v_isSharedCheck_3033_ == 0)
{
v___x_3028_ = v___x_3016_;
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_a_3026_);
lean_dec(v___x_3016_);
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
v_reuseFailAlloc_3032_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2____boxed(lean_object* v_e_3034_, lean_object* v___y_3035_, lean_object* v___y_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_){
_start:
{
lean_object* v_res_3040_; 
v_res_3040_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_(v_e_3034_, v___y_3035_, v___y_3036_, v___y_3037_, v___y_3038_);
lean_dec(v___y_3038_);
lean_dec_ref(v___y_3037_);
lean_dec(v___y_3036_);
lean_dec_ref(v___y_3035_);
return v_res_3040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; 
v___f_3043_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_));
v___x_3044_ = lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
v___x_3045_ = lean_st_ref_set(v___x_3044_, v___f_3043_);
v___x_3046_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3046_, 0, v___x_3045_);
return v___x_3046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2____boxed(lean_object* v_a_3047_){
_start:
{
lean_object* v_res_3048_; 
v_res_3048_ = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_();
return v_res_3048_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12(void){
_start:
{
lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; 
v___x_3075_ = l_Lean_Parser_Tactic_optConfig;
v___x_3076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__11));
v___x_3077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3078_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3078_, 0, v___x_3077_);
lean_ctor_set(v___x_3078_, 1, v___x_3076_);
lean_ctor_set(v___x_3078_, 2, v___x_3075_);
return v___x_3078_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13(void){
_start:
{
lean_object* v___x_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; 
v___x_3079_ = l_Lean_Parser_Tactic_location;
v___x_3080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__7));
v___x_3081_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3081_, 0, v___x_3080_);
lean_ctor_set(v___x_3081_, 1, v___x_3079_);
return v___x_3081_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14(void){
_start:
{
lean_object* v___x_3082_; lean_object* v___x_3083_; lean_object* v___x_3084_; lean_object* v___x_3085_; 
v___x_3082_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13);
v___x_3083_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12);
v___x_3084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3085_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3085_, 0, v___x_3084_);
lean_ctor_set(v___x_3085_, 1, v___x_3083_);
lean_ctor_set(v___x_3085_, 2, v___x_3082_);
return v___x_3085_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15(void){
_start:
{
lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; 
v___x_3086_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__14);
v___x_3087_ = lean_unsigned_to_nat(1022u);
v___x_3088_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1));
v___x_3089_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3089_, 0, v___x_3088_);
lean_ctor_set(v___x_3089_, 1, v___x_3087_);
lean_ctor_set(v___x_3089_, 2, v___x_3086_);
return v___x_3089_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF(void){
_start:
{
lean_object* v___x_3090_; 
v___x_3090_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__15);
return v___x_3090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3092_; lean_object* v___x_3093_; 
v___x_3092_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode_evalTerm_spec__0___redArg___closed__0);
v___x_3093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3093_, 0, v___x_3092_);
return v___x_3093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg___boxed(lean_object* v___y_3094_){
_start:
{
lean_object* v_res_3095_; 
v_res_3095_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v_res_3095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0(lean_object* v_00_u03b1_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_){
_start:
{
lean_object* v___x_3106_; 
v___x_3106_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___boxed(lean_object* v_00_u03b1_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_){
_start:
{
lean_object* v_res_3117_; 
v_res_3117_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0(v_00_u03b1_3107_, v___y_3108_, v___y_3109_, v___y_3110_, v___y_3111_, v___y_3112_, v___y_3113_, v___y_3114_, v___y_3115_);
lean_dec(v___y_3115_);
lean_dec_ref(v___y_3114_);
lean_dec(v___y_3113_);
lean_dec_ref(v___y_3112_);
lean_dec(v___y_3111_);
lean_dec_ref(v___y_3110_);
lean_dec(v___y_3109_);
lean_dec_ref(v___y_3108_);
return v_res_3117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0(lean_object* v_val_3118_, lean_object* v_toConfig_3119_, uint8_t v___x_3120_, lean_object* v___x_3121_, lean_object* v___x_3122_, lean_object* v_x_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_){
_start:
{
lean_object* v___x_3130_; 
v___x_3130_ = lp_mathlib_Mathlib_Tactic_AtomM_recurse(v_val_3118_, v_toConfig_3119_, v___x_3120_, v___x_3121_, v___x_3122_, v_x_3123_, v___y_3125_, v___y_3126_, v___y_3127_, v___y_3128_);
return v___x_3130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0___boxed(lean_object* v_val_3131_, lean_object* v_toConfig_3132_, lean_object* v___x_3133_, lean_object* v___x_3134_, lean_object* v___x_3135_, lean_object* v_x_3136_, lean_object* v___y_3137_, lean_object* v___y_3138_, lean_object* v___y_3139_, lean_object* v___y_3140_, lean_object* v___y_3141_, lean_object* v___y_3142_){
_start:
{
uint8_t v___x_1697__boxed_3143_; lean_object* v_res_3144_; 
v___x_1697__boxed_3143_ = lean_unbox(v___x_3133_);
v_res_3144_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0(v_val_3131_, v_toConfig_3132_, v___x_1697__boxed_3143_, v___x_3134_, v___x_3135_, v_x_3136_, v___y_3137_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
lean_dec(v___y_3141_);
lean_dec_ref(v___y_3140_);
lean_dec(v___y_3139_);
lean_dec_ref(v___y_3138_);
lean_dec_ref(v___y_3137_);
return v_res_3144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1(lean_object* v_x_3150_, lean_object* v_a_3151_, lean_object* v_a_3152_, lean_object* v_a_3153_, lean_object* v_a_3154_, lean_object* v_a_3155_, lean_object* v_a_3156_, lean_object* v_a_3157_, lean_object* v_a_3158_){
_start:
{
lean_object* v___x_3160_; uint8_t v___x_3161_; lean_object* v___y_3163_; lean_object* v___y_3164_; lean_object* v___y_3165_; uint8_t v___y_3166_; lean_object* v___y_3167_; lean_object* v___y_3168_; lean_object* v___y_3169_; lean_object* v___y_3170_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; uint8_t v___y_3186_; lean_object* v___y_3187_; lean_object* v_cfg_3188_; lean_object* v___y_3189_; lean_object* v___y_3190_; lean_object* v___y_3191_; lean_object* v___y_3192_; lean_object* v___y_3193_; lean_object* v___y_3194_; lean_object* v___y_3195_; lean_object* v___y_3196_; 
v___x_3160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1));
lean_inc(v_x_3150_);
v___x_3161_ = l_Lean_Syntax_isOfKind(v_x_3150_, v___x_3160_);
if (v___x_3161_ == 0)
{
lean_object* v___x_3201_; 
lean_dec(v_x_3150_);
v___x_3201_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3201_;
}
else
{
lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___y_3207_; lean_object* v___y_3208_; lean_object* v___y_3243_; lean_object* v___x_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; 
v___x_3202_ = lean_unsigned_to_nat(1u);
v___x_3203_ = l_Lean_Syntax_getArg(v_x_3150_, v___x_3202_);
v___x_3204_ = lean_unsigned_to_nat(2u);
v___x_3205_ = l_Lean_Syntax_getArg(v_x_3150_, v___x_3204_);
v___x_3254_ = lean_unsigned_to_nat(3u);
v___x_3255_ = l_Lean_Syntax_getArg(v_x_3150_, v___x_3254_);
lean_dec(v_x_3150_);
v___x_3256_ = l_Lean_Syntax_getOptional_x3f(v___x_3255_);
lean_dec(v___x_3255_);
if (lean_obj_tag(v___x_3256_) == 0)
{
lean_object* v___x_3257_; 
v___x_3257_ = lean_box(0);
v___y_3243_ = v___x_3257_;
goto v___jp_3242_;
}
else
{
lean_object* v_val_3258_; lean_object* v___x_3260_; uint8_t v_isShared_3261_; uint8_t v_isSharedCheck_3265_; 
v_val_3258_ = lean_ctor_get(v___x_3256_, 0);
v_isSharedCheck_3265_ = !lean_is_exclusive(v___x_3256_);
if (v_isSharedCheck_3265_ == 0)
{
v___x_3260_ = v___x_3256_;
v_isShared_3261_ = v_isSharedCheck_3265_;
goto v_resetjp_3259_;
}
else
{
lean_inc(v_val_3258_);
lean_dec(v___x_3256_);
v___x_3260_ = lean_box(0);
v_isShared_3261_ = v_isSharedCheck_3265_;
goto v_resetjp_3259_;
}
v_resetjp_3259_:
{
lean_object* v___x_3263_; 
if (v_isShared_3261_ == 0)
{
v___x_3263_ = v___x_3260_;
goto v_reusejp_3262_;
}
else
{
lean_object* v_reuseFailAlloc_3264_; 
v_reuseFailAlloc_3264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3264_, 0, v_val_3258_);
v___x_3263_ = v_reuseFailAlloc_3264_;
goto v_reusejp_3262_;
}
v_reusejp_3262_:
{
v___y_3243_ = v___x_3263_;
goto v___jp_3242_;
}
}
}
v___jp_3206_:
{
uint8_t v___x_3209_; lean_object* v___x_3210_; lean_object* v___x_3211_; 
v___x_3209_ = 0;
v___x_3210_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_));
v___x_3211_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(v___x_3205_, v___x_3210_, v___x_3161_, v_a_3151_, v_a_3157_, v_a_3158_);
if (lean_obj_tag(v___x_3211_) == 0)
{
if (lean_obj_tag(v___y_3208_) == 0)
{
lean_object* v_a_3212_; 
v_a_3212_ = lean_ctor_get(v___x_3211_, 0);
lean_inc(v_a_3212_);
lean_dec_ref_known(v___x_3211_, 1);
v___y_3186_ = v___x_3209_;
v___y_3187_ = v___y_3207_;
v_cfg_3188_ = v_a_3212_;
v___y_3189_ = v_a_3151_;
v___y_3190_ = v_a_3152_;
v___y_3191_ = v_a_3153_;
v___y_3192_ = v_a_3154_;
v___y_3193_ = v_a_3155_;
v___y_3194_ = v_a_3156_;
v___y_3195_ = v_a_3157_;
v___y_3196_ = v_a_3158_;
goto v___jp_3185_;
}
else
{
lean_dec_ref_known(v___y_3208_, 1);
if (v___x_3161_ == 0)
{
lean_object* v_a_3213_; 
v_a_3213_ = lean_ctor_get(v___x_3211_, 0);
lean_inc(v_a_3213_);
lean_dec_ref_known(v___x_3211_, 1);
v___y_3186_ = v___x_3209_;
v___y_3187_ = v___y_3207_;
v_cfg_3188_ = v_a_3213_;
v___y_3189_ = v_a_3151_;
v___y_3190_ = v_a_3152_;
v___y_3191_ = v_a_3153_;
v___y_3192_ = v_a_3154_;
v___y_3193_ = v_a_3155_;
v___y_3194_ = v_a_3156_;
v___y_3195_ = v_a_3157_;
v___y_3196_ = v_a_3158_;
goto v___jp_3185_;
}
else
{
lean_object* v_a_3214_; lean_object* v_toConfig_3215_; uint8_t v_ifUnchanged_3216_; uint8_t v_mode_3217_; lean_object* v___x_3219_; uint8_t v_isShared_3220_; uint8_t v_isSharedCheck_3233_; 
v_a_3214_ = lean_ctor_get(v___x_3211_, 0);
lean_inc(v_a_3214_);
lean_dec_ref_known(v___x_3211_, 1);
v_toConfig_3215_ = lean_ctor_get(v_a_3214_, 0);
v_ifUnchanged_3216_ = lean_ctor_get_uint8(v_a_3214_, sizeof(void*)*1);
v_mode_3217_ = lean_ctor_get_uint8(v_a_3214_, sizeof(void*)*1 + 1);
v_isSharedCheck_3233_ = !lean_is_exclusive(v_a_3214_);
if (v_isSharedCheck_3233_ == 0)
{
v___x_3219_ = v_a_3214_;
v_isShared_3220_ = v_isSharedCheck_3233_;
goto v_resetjp_3218_;
}
else
{
lean_inc(v_toConfig_3215_);
lean_dec(v_a_3214_);
v___x_3219_ = lean_box(0);
v_isShared_3220_ = v_isSharedCheck_3233_;
goto v_resetjp_3218_;
}
v_resetjp_3218_:
{
uint8_t v_contextual_3221_; lean_object* v___x_3223_; uint8_t v_isShared_3224_; uint8_t v_isSharedCheck_3232_; 
v_contextual_3221_ = lean_ctor_get_uint8(v_toConfig_3215_, 2);
v_isSharedCheck_3232_ = !lean_is_exclusive(v_toConfig_3215_);
if (v_isSharedCheck_3232_ == 0)
{
v___x_3223_ = v_toConfig_3215_;
v_isShared_3224_ = v_isSharedCheck_3232_;
goto v_resetjp_3222_;
}
else
{
lean_dec(v_toConfig_3215_);
v___x_3223_ = lean_box(0);
v_isShared_3224_ = v_isSharedCheck_3232_;
goto v_resetjp_3222_;
}
v_resetjp_3222_:
{
uint8_t v___x_3225_; lean_object* v___x_3227_; 
v___x_3225_ = 1;
if (v_isShared_3224_ == 0)
{
v___x_3227_ = v___x_3223_;
goto v_reusejp_3226_;
}
else
{
lean_object* v_reuseFailAlloc_3231_; 
v_reuseFailAlloc_3231_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v_reuseFailAlloc_3231_, 2, v_contextual_3221_);
v___x_3227_ = v_reuseFailAlloc_3231_;
goto v_reusejp_3226_;
}
v_reusejp_3226_:
{
lean_object* v___x_3229_; 
lean_ctor_set_uint8(v___x_3227_, 0, v___x_3225_);
lean_ctor_set_uint8(v___x_3227_, 1, v___x_3161_);
if (v_isShared_3220_ == 0)
{
lean_ctor_set(v___x_3219_, 0, v___x_3227_);
v___x_3229_ = v___x_3219_;
goto v_reusejp_3228_;
}
else
{
lean_object* v_reuseFailAlloc_3230_; 
v_reuseFailAlloc_3230_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_3230_, 0, v___x_3227_);
lean_ctor_set_uint8(v_reuseFailAlloc_3230_, sizeof(void*)*1, v_ifUnchanged_3216_);
lean_ctor_set_uint8(v_reuseFailAlloc_3230_, sizeof(void*)*1 + 1, v_mode_3217_);
v___x_3229_ = v_reuseFailAlloc_3230_;
goto v_reusejp_3228_;
}
v_reusejp_3228_:
{
v___y_3186_ = v___x_3209_;
v___y_3187_ = v___y_3207_;
v_cfg_3188_ = v___x_3229_;
v___y_3189_ = v_a_3151_;
v___y_3190_ = v_a_3152_;
v___y_3191_ = v_a_3153_;
v___y_3192_ = v_a_3154_;
v___y_3193_ = v_a_3155_;
v___y_3194_ = v_a_3156_;
v___y_3195_ = v_a_3157_;
v___y_3196_ = v_a_3158_;
goto v___jp_3185_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3234_; lean_object* v___x_3236_; uint8_t v_isShared_3237_; uint8_t v_isSharedCheck_3241_; 
lean_dec(v___y_3208_);
lean_dec(v___y_3207_);
v_a_3234_ = lean_ctor_get(v___x_3211_, 0);
v_isSharedCheck_3241_ = !lean_is_exclusive(v___x_3211_);
if (v_isSharedCheck_3241_ == 0)
{
v___x_3236_ = v___x_3211_;
v_isShared_3237_ = v_isSharedCheck_3241_;
goto v_resetjp_3235_;
}
else
{
lean_inc(v_a_3234_);
lean_dec(v___x_3211_);
v___x_3236_ = lean_box(0);
v_isShared_3237_ = v_isSharedCheck_3241_;
goto v_resetjp_3235_;
}
v_resetjp_3235_:
{
lean_object* v___x_3239_; 
if (v_isShared_3237_ == 0)
{
v___x_3239_ = v___x_3236_;
goto v_reusejp_3238_;
}
else
{
lean_object* v_reuseFailAlloc_3240_; 
v_reuseFailAlloc_3240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3240_, 0, v_a_3234_);
v___x_3239_ = v_reuseFailAlloc_3240_;
goto v_reusejp_3238_;
}
v_reusejp_3238_:
{
return v___x_3239_;
}
}
}
}
v___jp_3242_:
{
lean_object* v___x_3244_; 
v___x_3244_ = l_Lean_Syntax_getOptional_x3f(v___x_3203_);
lean_dec(v___x_3203_);
if (lean_obj_tag(v___x_3244_) == 0)
{
lean_object* v___x_3245_; 
v___x_3245_ = lean_box(0);
v___y_3207_ = v___y_3243_;
v___y_3208_ = v___x_3245_;
goto v___jp_3206_;
}
else
{
lean_object* v_val_3246_; lean_object* v___x_3248_; uint8_t v_isShared_3249_; uint8_t v_isSharedCheck_3253_; 
v_val_3246_ = lean_ctor_get(v___x_3244_, 0);
v_isSharedCheck_3253_ = !lean_is_exclusive(v___x_3244_);
if (v_isSharedCheck_3253_ == 0)
{
v___x_3248_ = v___x_3244_;
v_isShared_3249_ = v_isSharedCheck_3253_;
goto v_resetjp_3247_;
}
else
{
lean_inc(v_val_3246_);
lean_dec(v___x_3244_);
v___x_3248_ = lean_box(0);
v_isShared_3249_ = v_isSharedCheck_3253_;
goto v_resetjp_3247_;
}
v_resetjp_3247_:
{
lean_object* v___x_3251_; 
if (v_isShared_3249_ == 0)
{
v___x_3251_ = v___x_3248_;
goto v_reusejp_3250_;
}
else
{
lean_object* v_reuseFailAlloc_3252_; 
v_reuseFailAlloc_3252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3252_, 0, v_val_3246_);
v___x_3251_ = v_reuseFailAlloc_3252_;
goto v_reusejp_3250_;
}
v_reusejp_3250_:
{
v___y_3207_ = v___y_3243_;
v___y_3208_ = v___x_3251_;
goto v___jp_3206_;
}
}
}
}
}
v___jp_3162_:
{
lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v_toConfig_3176_; uint8_t v_ifUnchanged_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; lean_object* v___f_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; 
v___x_3174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0));
v___x_3175_ = lean_st_mk_ref(v___x_3174_);
v_toConfig_3176_ = lean_ctor_get(v___y_3167_, 0);
lean_inc_ref(v_toConfig_3176_);
v_ifUnchanged_3177_ = lean_ctor_get_uint8(v___y_3167_, sizeof(void*)*1);
v___x_3178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1));
v___x_3179_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___boxed), 7, 1);
lean_closure_set(v___x_3179_, 0, v___y_3167_);
v___x_3180_ = lean_box(v___x_3161_);
v___f_3181_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___lam__0___boxed), 12, 5);
lean_closure_set(v___f_3181_, 0, v___x_3175_);
lean_closure_set(v___f_3181_, 1, v_toConfig_3176_);
lean_closure_set(v___f_3181_, 2, v___x_3180_);
lean_closure_set(v___f_3181_, 3, v___x_3178_);
lean_closure_set(v___f_3181_, 4, v___x_3179_);
v___x_3182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4));
v___x_3183_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3184_ = lp_mathlib_Mathlib_Tactic_transformAtLocation(v___f_3181_, v___x_3182_, v___y_3173_, v_ifUnchanged_3177_, v___y_3166_, v___x_3183_, v___y_3170_, v___y_3172_, v___y_3164_, v___y_3169_, v___y_3168_, v___y_3171_, v___y_3165_, v___y_3163_);
lean_dec(v___y_3173_);
return v___x_3184_;
}
v___jp_3185_:
{
if (lean_obj_tag(v___y_3187_) == 0)
{
lean_object* v___x_3197_; lean_object* v___x_3198_; 
v___x_3197_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__2));
v___x_3198_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_3198_, 0, v___x_3197_);
lean_ctor_set_uint8(v___x_3198_, sizeof(void*)*1, v___x_3161_);
v___y_3163_ = v___y_3196_;
v___y_3164_ = v___y_3191_;
v___y_3165_ = v___y_3195_;
v___y_3166_ = v___y_3186_;
v___y_3167_ = v_cfg_3188_;
v___y_3168_ = v___y_3193_;
v___y_3169_ = v___y_3192_;
v___y_3170_ = v___y_3189_;
v___y_3171_ = v___y_3194_;
v___y_3172_ = v___y_3190_;
v___y_3173_ = v___x_3198_;
goto v___jp_3162_;
}
else
{
lean_object* v_val_3199_; lean_object* v___x_3200_; 
v_val_3199_ = lean_ctor_get(v___y_3187_, 0);
lean_inc(v_val_3199_);
lean_dec_ref_known(v___y_3187_, 1);
v___x_3200_ = l_Lean_Elab_Tactic_expandLocation(v_val_3199_);
lean_dec(v_val_3199_);
v___y_3163_ = v___y_3196_;
v___y_3164_ = v___y_3191_;
v___y_3165_ = v___y_3195_;
v___y_3166_ = v___y_3186_;
v___y_3167_ = v_cfg_3188_;
v___y_3168_ = v___y_3193_;
v___y_3169_ = v___y_3192_;
v___y_3170_ = v___y_3189_;
v___y_3171_ = v___y_3194_;
v___y_3172_ = v___y_3190_;
v___y_3173_ = v___x_3200_;
goto v___jp_3162_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___boxed(lean_object* v_x_3266_, lean_object* v_a_3267_, lean_object* v_a_3268_, lean_object* v_a_3269_, lean_object* v_a_3270_, lean_object* v_a_3271_, lean_object* v_a_3272_, lean_object* v_a_3273_, lean_object* v_a_3274_, lean_object* v_a_3275_){
_start:
{
lean_object* v_res_3276_; 
v_res_3276_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1(v_x_3266_, v_a_3267_, v_a_3268_, v_a_3269_, v_a_3270_, v_a_3271_, v_a_3272_, v_a_3273_, v_a_3274_);
lean_dec(v_a_3274_);
lean_dec_ref(v_a_3273_);
lean_dec(v_a_3272_);
lean_dec_ref(v_a_3271_);
lean_dec(v_a_3270_);
lean_dec_ref(v_a_3269_);
lean_dec(v_a_3268_);
lean_dec_ref(v_a_3267_);
return v_res_3276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4(void){
_start:
{
lean_object* v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; lean_object* v___x_3290_; 
v___x_3287_ = l_Lean_Parser_Tactic_optConfig;
v___x_3288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__3));
v___x_3289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3290_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3290_, 0, v___x_3289_);
lean_ctor_set(v___x_3290_, 1, v___x_3288_);
lean_ctor_set(v___x_3290_, 2, v___x_3287_);
return v___x_3290_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5(void){
_start:
{
lean_object* v___x_3291_; lean_object* v___x_3292_; lean_object* v___x_3293_; lean_object* v___x_3294_; 
v___x_3291_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__13);
v___x_3292_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4);
v___x_3293_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3294_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3294_, 0, v___x_3293_);
lean_ctor_set(v___x_3294_, 1, v___x_3292_);
lean_ctor_set(v___x_3294_, 2, v___x_3291_);
return v___x_3294_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6(void){
_start:
{
lean_object* v___x_3295_; lean_object* v___x_3296_; lean_object* v___x_3297_; lean_object* v___x_3298_; 
v___x_3295_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__5);
v___x_3296_ = lean_unsigned_to_nat(1022u);
v___x_3297_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1));
v___x_3298_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3298_, 0, v___x_3297_);
lean_ctor_set(v___x_3298_, 1, v___x_3296_);
lean_ctor_set(v___x_3298_, 2, v___x_3295_);
return v___x_3298_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21____(void){
_start:
{
lean_object* v___x_3299_; 
v___x_3299_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__6);
return v___x_3299_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2(void){
_start:
{
lean_object* v___x_3303_; 
v___x_3303_ = l_Array_mkArray0(lean_box(0));
return v___x_3303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1(lean_object* v_x_3306_, lean_object* v_a_3307_, lean_object* v_a_3308_){
_start:
{
lean_object* v___x_3309_; uint8_t v___x_3310_; 
v___x_3309_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1));
lean_inc(v_x_3306_);
v___x_3310_ = l_Lean_Syntax_isOfKind(v_x_3306_, v___x_3309_);
if (v___x_3310_ == 0)
{
lean_object* v___x_3311_; lean_object* v___x_3312_; 
lean_dec(v_x_3306_);
v___x_3311_ = lean_box(1);
v___x_3312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3312_, 0, v___x_3311_);
lean_ctor_set(v___x_3312_, 1, v_a_3308_);
return v___x_3312_;
}
else
{
lean_object* v___x_3313_; lean_object* v___x_3314_; lean_object* v___y_3316_; lean_object* v___y_3317_; lean_object* v___y_3318_; lean_object* v___y_3319_; lean_object* v___y_3320_; lean_object* v___y_3321_; lean_object* v___y_3322_; lean_object* v___y_3328_; lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3346_; 
v___x_3313_ = lean_unsigned_to_nat(1u);
v___x_3314_ = l_Lean_Syntax_getArg(v_x_3306_, v___x_3313_);
v___x_3344_ = lean_unsigned_to_nat(2u);
v___x_3345_ = l_Lean_Syntax_getArg(v_x_3306_, v___x_3344_);
lean_dec(v_x_3306_);
v___x_3346_ = l_Lean_Syntax_getOptional_x3f(v___x_3345_);
lean_dec(v___x_3345_);
if (lean_obj_tag(v___x_3346_) == 0)
{
lean_object* v___x_3347_; 
v___x_3347_ = lean_box(0);
v___y_3328_ = v___x_3347_;
goto v___jp_3327_;
}
else
{
lean_object* v_val_3348_; lean_object* v___x_3350_; uint8_t v_isShared_3351_; uint8_t v_isSharedCheck_3355_; 
v_val_3348_ = lean_ctor_get(v___x_3346_, 0);
v_isSharedCheck_3355_ = !lean_is_exclusive(v___x_3346_);
if (v_isSharedCheck_3355_ == 0)
{
v___x_3350_ = v___x_3346_;
v_isShared_3351_ = v_isSharedCheck_3355_;
goto v_resetjp_3349_;
}
else
{
lean_inc(v_val_3348_);
lean_dec(v___x_3346_);
v___x_3350_ = lean_box(0);
v_isShared_3351_ = v_isSharedCheck_3355_;
goto v_resetjp_3349_;
}
v_resetjp_3349_:
{
lean_object* v___x_3353_; 
if (v_isShared_3351_ == 0)
{
v___x_3353_ = v___x_3350_;
goto v_reusejp_3352_;
}
else
{
lean_object* v_reuseFailAlloc_3354_; 
v_reuseFailAlloc_3354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3354_, 0, v_val_3348_);
v___x_3353_ = v_reuseFailAlloc_3354_;
goto v_reusejp_3352_;
}
v_reusejp_3352_:
{
v___y_3328_ = v___x_3353_;
goto v___jp_3327_;
}
}
}
v___jp_3315_:
{
lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; 
lean_inc_ref(v___y_3319_);
v___x_3323_ = l_Array_append___redArg(v___y_3319_, v___y_3322_);
lean_dec_ref(v___y_3322_);
lean_inc(v___y_3316_);
v___x_3324_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3324_, 0, v___y_3316_);
lean_ctor_set(v___x_3324_, 1, v___y_3320_);
lean_ctor_set(v___x_3324_, 2, v___x_3323_);
lean_inc(v___y_3318_);
v___x_3325_ = l_Lean_Syntax_node4(v___y_3316_, v___y_3318_, v___y_3317_, v___y_3321_, v___x_3314_, v___x_3324_);
v___x_3326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3326_, 0, v___x_3325_);
lean_ctor_set(v___x_3326_, 1, v_a_3308_);
return v___x_3326_;
}
v___jp_3327_:
{
lean_object* v_ref_3329_; uint8_t v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; 
v_ref_3329_ = lean_ctor_get(v_a_3307_, 5);
v___x_3330_ = 0;
v___x_3331_ = l_Lean_SourceInfo_fromRef(v_ref_3329_, v___x_3330_);
v___x_3332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1));
v___x_3333_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4));
lean_inc_n(v___x_3331_, 3);
v___x_3334_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3334_, 0, v___x_3331_);
lean_ctor_set(v___x_3334_, 1, v___x_3333_);
v___x_3335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_3336_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8));
v___x_3337_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3337_, 0, v___x_3331_);
lean_ctor_set(v___x_3337_, 1, v___x_3336_);
v___x_3338_ = l_Lean_Syntax_node1(v___x_3331_, v___x_3335_, v___x_3337_);
v___x_3339_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2);
if (lean_obj_tag(v___y_3328_) == 0)
{
lean_object* v___x_3340_; 
v___x_3340_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__3));
v___y_3316_ = v___x_3331_;
v___y_3317_ = v___x_3334_;
v___y_3318_ = v___x_3332_;
v___y_3319_ = v___x_3339_;
v___y_3320_ = v___x_3335_;
v___y_3321_ = v___x_3338_;
v___y_3322_ = v___x_3340_;
goto v___jp_3315_;
}
else
{
lean_object* v_val_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; 
v_val_3341_ = lean_ctor_get(v___y_3328_, 0);
lean_inc(v_val_3341_);
lean_dec_ref_known(v___y_3328_, 1);
v___x_3342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__3));
v___x_3343_ = lean_array_push(v___x_3342_, v_val_3341_);
v___y_3316_ = v___x_3331_;
v___y_3317_ = v___x_3334_;
v___y_3318_ = v___x_3332_;
v___y_3319_ = v___x_3339_;
v___y_3320_ = v___x_3335_;
v___y_3321_ = v___x_3338_;
v___y_3322_ = v___x_3343_;
goto v___jp_3315_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___boxed(lean_object* v_x_3356_, lean_object* v_a_3357_, lean_object* v_a_3358_){
_start:
{
lean_object* v_res_3359_; 
v_res_3359_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1(v_x_3356_, v_a_3357_, v_a_3358_);
lean_dec_ref(v_a_3357_);
return v_res_3359_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2(void){
_start:
{
lean_object* v___x_3366_; lean_object* v___x_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; 
v___x_3366_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12, &lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__12);
v___x_3367_ = lean_unsigned_to_nat(1022u);
v___x_3368_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1));
v___x_3369_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3369_, 0, v___x_3368_);
lean_ctor_set(v___x_3369_, 1, v___x_3367_);
lean_ctor_set(v___x_3369_, 2, v___x_3366_);
return v___x_3369_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv(void){
_start:
{
lean_object* v___x_3370_; 
v___x_3370_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__2);
return v___x_3370_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5(void){
_start:
{
lean_object* v___x_3385_; lean_object* v___x_3386_; lean_object* v___x_3387_; lean_object* v___x_3388_; 
v___x_3385_ = l_Lean_Parser_Tactic_optConfig;
v___x_3386_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__4));
v___x_3387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3388_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3388_, 0, v___x_3387_);
lean_ctor_set(v___x_3388_, 1, v___x_3386_);
lean_ctor_set(v___x_3388_, 2, v___x_3385_);
return v___x_3388_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6(void){
_start:
{
lean_object* v___x_3389_; lean_object* v___x_3390_; lean_object* v___x_3391_; lean_object* v___x_3392_; 
v___x_3389_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__5);
v___x_3390_ = lean_unsigned_to_nat(1022u);
v___x_3391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1));
v___x_3392_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3392_, 0, v___x_3391_);
lean_ctor_set(v___x_3392_, 1, v___x_3390_);
lean_ctor_set(v___x_3392_, 2, v___x_3389_);
return v___x_3392_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF(void){
_start:
{
lean_object* v___x_3393_; 
v___x_3393_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6, &lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__6);
return v___x_3393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0(lean_object* v_a_3394_, lean_object* v___y_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_){
_start:
{
lean_object* v___x_3403_; 
v___x_3403_ = lp_mathlib_Mathlib_Tactic_Ring_proveEq(v_a_3394_, v___y_3396_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_);
return v___x_3403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0___boxed(lean_object* v_a_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_, lean_object* v___y_3408_, lean_object* v___y_3409_, lean_object* v___y_3410_, lean_object* v___y_3411_, lean_object* v___y_3412_){
_start:
{
lean_object* v_res_3413_; 
v_res_3413_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0(v_a_3404_, v___y_3405_, v___y_3406_, v___y_3407_, v___y_3408_, v___y_3409_, v___y_3410_, v___y_3411_);
lean_dec(v___y_3411_);
lean_dec_ref(v___y_3410_);
lean_dec(v___y_3409_);
lean_dec_ref(v___y_3408_);
lean_dec(v___y_3407_);
lean_dec_ref(v___y_3406_);
lean_dec_ref(v___y_3405_);
return v_res_3413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1(lean_object* v_cfg_3414_, lean_object* v_val_3415_, uint8_t v___x_3416_, lean_object* v___y_3417_, lean_object* v___y_3418_, lean_object* v___y_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_, lean_object* v___y_3422_, lean_object* v___y_3423_, lean_object* v___y_3424_){
_start:
{
lean_object* v___x_3426_; 
v___x_3426_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_3418_, v___y_3421_, v___y_3422_, v___y_3423_, v___y_3424_);
if (lean_obj_tag(v___x_3426_) == 0)
{
lean_object* v_a_3427_; lean_object* v_toConfig_3428_; lean_object* v___f_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; 
v_a_3427_ = lean_ctor_get(v___x_3426_, 0);
lean_inc(v_a_3427_);
lean_dec_ref_known(v___x_3426_, 1);
v_toConfig_3428_ = lean_ctor_get(v_cfg_3414_, 0);
lean_inc_ref(v_toConfig_3428_);
v___f_3429_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__0___boxed), 9, 1);
lean_closure_set(v___f_3429_, 0, v_a_3427_);
v___x_3430_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1));
v___x_3431_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___boxed), 7, 1);
lean_closure_set(v___x_3431_, 0, v_cfg_3414_);
v___x_3432_ = lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(v_val_3415_, v_toConfig_3428_, v___x_3416_, v___x_3430_, v___x_3431_, v___f_3429_, v___y_3421_, v___y_3422_, v___y_3423_, v___y_3424_);
return v___x_3432_;
}
else
{
lean_object* v_a_3433_; lean_object* v___x_3435_; uint8_t v_isShared_3436_; uint8_t v_isSharedCheck_3440_; 
lean_dec(v_val_3415_);
lean_dec_ref(v_cfg_3414_);
v_a_3433_ = lean_ctor_get(v___x_3426_, 0);
v_isSharedCheck_3440_ = !lean_is_exclusive(v___x_3426_);
if (v_isSharedCheck_3440_ == 0)
{
v___x_3435_ = v___x_3426_;
v_isShared_3436_ = v_isSharedCheck_3440_;
goto v_resetjp_3434_;
}
else
{
lean_inc(v_a_3433_);
lean_dec(v___x_3426_);
v___x_3435_ = lean_box(0);
v_isShared_3436_ = v_isSharedCheck_3440_;
goto v_resetjp_3434_;
}
v_resetjp_3434_:
{
lean_object* v___x_3438_; 
if (v_isShared_3436_ == 0)
{
v___x_3438_ = v___x_3435_;
goto v_reusejp_3437_;
}
else
{
lean_object* v_reuseFailAlloc_3439_; 
v_reuseFailAlloc_3439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3439_, 0, v_a_3433_);
v___x_3438_ = v_reuseFailAlloc_3439_;
goto v_reusejp_3437_;
}
v_reusejp_3437_:
{
return v___x_3438_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1___boxed(lean_object* v_cfg_3441_, lean_object* v_val_3442_, lean_object* v___x_3443_, lean_object* v___y_3444_, lean_object* v___y_3445_, lean_object* v___y_3446_, lean_object* v___y_3447_, lean_object* v___y_3448_, lean_object* v___y_3449_, lean_object* v___y_3450_, lean_object* v___y_3451_, lean_object* v___y_3452_){
_start:
{
uint8_t v___x_1265__boxed_3453_; lean_object* v_res_3454_; 
v___x_1265__boxed_3453_ = lean_unbox(v___x_3443_);
v_res_3454_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1(v_cfg_3441_, v_val_3442_, v___x_1265__boxed_3453_, v___y_3444_, v___y_3445_, v___y_3446_, v___y_3447_, v___y_3448_, v___y_3449_, v___y_3450_, v___y_3451_);
lean_dec(v___y_3451_);
lean_dec_ref(v___y_3450_);
lean_dec(v___y_3449_);
lean_dec_ref(v___y_3448_);
lean_dec(v___y_3447_);
lean_dec_ref(v___y_3446_);
lean_dec(v___y_3445_);
lean_dec_ref(v___y_3444_);
return v_res_3454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1(lean_object* v_x_3455_, lean_object* v_a_3456_, lean_object* v_a_3457_, lean_object* v_a_3458_, lean_object* v_a_3459_, lean_object* v_a_3460_, lean_object* v_a_3461_, lean_object* v_a_3462_, lean_object* v_a_3463_){
_start:
{
lean_object* v___x_3465_; uint8_t v___x_3466_; lean_object* v_cfg_3468_; lean_object* v___y_3469_; lean_object* v___y_3470_; lean_object* v___y_3471_; lean_object* v___y_3472_; lean_object* v___y_3473_; lean_object* v___y_3474_; lean_object* v___y_3475_; lean_object* v___y_3476_; 
v___x_3465_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1));
lean_inc(v_x_3455_);
v___x_3466_ = l_Lean_Syntax_isOfKind(v_x_3455_, v___x_3465_);
if (v___x_3466_ == 0)
{
lean_object* v___x_3482_; 
lean_dec(v_x_3455_);
v___x_3482_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3482_;
}
else
{
lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; lean_object* v___y_3488_; lean_object* v___x_3521_; 
v___x_3483_ = lean_unsigned_to_nat(1u);
v___x_3484_ = l_Lean_Syntax_getArg(v_x_3455_, v___x_3483_);
v___x_3485_ = lean_unsigned_to_nat(2u);
v___x_3486_ = l_Lean_Syntax_getArg(v_x_3455_, v___x_3485_);
lean_dec(v_x_3455_);
v___x_3521_ = l_Lean_Syntax_getOptional_x3f(v___x_3484_);
lean_dec(v___x_3484_);
if (lean_obj_tag(v___x_3521_) == 0)
{
lean_object* v___x_3522_; 
v___x_3522_ = lean_box(0);
v___y_3488_ = v___x_3522_;
goto v___jp_3487_;
}
else
{
lean_object* v_val_3523_; lean_object* v___x_3525_; uint8_t v_isShared_3526_; uint8_t v_isSharedCheck_3530_; 
v_val_3523_ = lean_ctor_get(v___x_3521_, 0);
v_isSharedCheck_3530_ = !lean_is_exclusive(v___x_3521_);
if (v_isSharedCheck_3530_ == 0)
{
v___x_3525_ = v___x_3521_;
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
else
{
lean_inc(v_val_3523_);
lean_dec(v___x_3521_);
v___x_3525_ = lean_box(0);
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
v_resetjp_3524_:
{
lean_object* v___x_3528_; 
if (v_isShared_3526_ == 0)
{
v___x_3528_ = v___x_3525_;
goto v_reusejp_3527_;
}
else
{
lean_object* v_reuseFailAlloc_3529_; 
v_reuseFailAlloc_3529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3529_, 0, v_val_3523_);
v___x_3528_ = v_reuseFailAlloc_3529_;
goto v_reusejp_3527_;
}
v_reusejp_3527_:
{
v___y_3488_ = v___x_3528_;
goto v___jp_3487_;
}
}
}
v___jp_3487_:
{
lean_object* v___x_3489_; lean_object* v___x_3490_; 
v___x_3489_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_));
v___x_3490_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(v___x_3486_, v___x_3489_, v___x_3466_, v_a_3456_, v_a_3462_, v_a_3463_);
if (lean_obj_tag(v___x_3490_) == 0)
{
if (lean_obj_tag(v___y_3488_) == 0)
{
lean_object* v_a_3491_; 
v_a_3491_ = lean_ctor_get(v___x_3490_, 0);
lean_inc(v_a_3491_);
lean_dec_ref_known(v___x_3490_, 1);
v_cfg_3468_ = v_a_3491_;
v___y_3469_ = v_a_3456_;
v___y_3470_ = v_a_3457_;
v___y_3471_ = v_a_3458_;
v___y_3472_ = v_a_3459_;
v___y_3473_ = v_a_3460_;
v___y_3474_ = v_a_3461_;
v___y_3475_ = v_a_3462_;
v___y_3476_ = v_a_3463_;
goto v___jp_3467_;
}
else
{
lean_dec_ref_known(v___y_3488_, 1);
if (v___x_3466_ == 0)
{
lean_object* v_a_3492_; 
v_a_3492_ = lean_ctor_get(v___x_3490_, 0);
lean_inc(v_a_3492_);
lean_dec_ref_known(v___x_3490_, 1);
v_cfg_3468_ = v_a_3492_;
v___y_3469_ = v_a_3456_;
v___y_3470_ = v_a_3457_;
v___y_3471_ = v_a_3458_;
v___y_3472_ = v_a_3459_;
v___y_3473_ = v_a_3460_;
v___y_3474_ = v_a_3461_;
v___y_3475_ = v_a_3462_;
v___y_3476_ = v_a_3463_;
goto v___jp_3467_;
}
else
{
lean_object* v_a_3493_; lean_object* v_toConfig_3494_; uint8_t v_ifUnchanged_3495_; uint8_t v_mode_3496_; lean_object* v___x_3498_; uint8_t v_isShared_3499_; uint8_t v_isSharedCheck_3512_; 
v_a_3493_ = lean_ctor_get(v___x_3490_, 0);
lean_inc(v_a_3493_);
lean_dec_ref_known(v___x_3490_, 1);
v_toConfig_3494_ = lean_ctor_get(v_a_3493_, 0);
v_ifUnchanged_3495_ = lean_ctor_get_uint8(v_a_3493_, sizeof(void*)*1);
v_mode_3496_ = lean_ctor_get_uint8(v_a_3493_, sizeof(void*)*1 + 1);
v_isSharedCheck_3512_ = !lean_is_exclusive(v_a_3493_);
if (v_isSharedCheck_3512_ == 0)
{
v___x_3498_ = v_a_3493_;
v_isShared_3499_ = v_isSharedCheck_3512_;
goto v_resetjp_3497_;
}
else
{
lean_inc(v_toConfig_3494_);
lean_dec(v_a_3493_);
v___x_3498_ = lean_box(0);
v_isShared_3499_ = v_isSharedCheck_3512_;
goto v_resetjp_3497_;
}
v_resetjp_3497_:
{
uint8_t v_contextual_3500_; lean_object* v___x_3502_; uint8_t v_isShared_3503_; uint8_t v_isSharedCheck_3511_; 
v_contextual_3500_ = lean_ctor_get_uint8(v_toConfig_3494_, 2);
v_isSharedCheck_3511_ = !lean_is_exclusive(v_toConfig_3494_);
if (v_isSharedCheck_3511_ == 0)
{
v___x_3502_ = v_toConfig_3494_;
v_isShared_3503_ = v_isSharedCheck_3511_;
goto v_resetjp_3501_;
}
else
{
lean_dec(v_toConfig_3494_);
v___x_3502_ = lean_box(0);
v_isShared_3503_ = v_isSharedCheck_3511_;
goto v_resetjp_3501_;
}
v_resetjp_3501_:
{
uint8_t v___x_3504_; lean_object* v___x_3506_; 
v___x_3504_ = 1;
if (v_isShared_3503_ == 0)
{
v___x_3506_ = v___x_3502_;
goto v_reusejp_3505_;
}
else
{
lean_object* v_reuseFailAlloc_3510_; 
v_reuseFailAlloc_3510_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v_reuseFailAlloc_3510_, 2, v_contextual_3500_);
v___x_3506_ = v_reuseFailAlloc_3510_;
goto v_reusejp_3505_;
}
v_reusejp_3505_:
{
lean_object* v___x_3508_; 
lean_ctor_set_uint8(v___x_3506_, 0, v___x_3504_);
lean_ctor_set_uint8(v___x_3506_, 1, v___x_3466_);
if (v_isShared_3499_ == 0)
{
lean_ctor_set(v___x_3498_, 0, v___x_3506_);
v___x_3508_ = v___x_3498_;
goto v_reusejp_3507_;
}
else
{
lean_object* v_reuseFailAlloc_3509_; 
v_reuseFailAlloc_3509_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_3509_, 0, v___x_3506_);
lean_ctor_set_uint8(v_reuseFailAlloc_3509_, sizeof(void*)*1, v_ifUnchanged_3495_);
lean_ctor_set_uint8(v_reuseFailAlloc_3509_, sizeof(void*)*1 + 1, v_mode_3496_);
v___x_3508_ = v_reuseFailAlloc_3509_;
goto v_reusejp_3507_;
}
v_reusejp_3507_:
{
v_cfg_3468_ = v___x_3508_;
v___y_3469_ = v_a_3456_;
v___y_3470_ = v_a_3457_;
v___y_3471_ = v_a_3458_;
v___y_3472_ = v_a_3459_;
v___y_3473_ = v_a_3460_;
v___y_3474_ = v_a_3461_;
v___y_3475_ = v_a_3462_;
v___y_3476_ = v_a_3463_;
goto v___jp_3467_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3513_; lean_object* v___x_3515_; uint8_t v_isShared_3516_; uint8_t v_isSharedCheck_3520_; 
lean_dec(v___y_3488_);
v_a_3513_ = lean_ctor_get(v___x_3490_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v___x_3490_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3515_ = v___x_3490_;
v_isShared_3516_ = v_isSharedCheck_3520_;
goto v_resetjp_3514_;
}
else
{
lean_inc(v_a_3513_);
lean_dec(v___x_3490_);
v___x_3515_ = lean_box(0);
v_isShared_3516_ = v_isSharedCheck_3520_;
goto v_resetjp_3514_;
}
v_resetjp_3514_:
{
lean_object* v___x_3518_; 
if (v_isShared_3516_ == 0)
{
v___x_3518_ = v___x_3515_;
goto v_reusejp_3517_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v_a_3513_);
v___x_3518_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3517_;
}
v_reusejp_3517_:
{
return v___x_3518_;
}
}
}
}
}
v___jp_3467_:
{
lean_object* v___x_3477_; lean_object* v___x_3478_; lean_object* v___x_3479_; lean_object* v___f_3480_; lean_object* v___x_3481_; 
v___x_3477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0));
v___x_3478_ = lean_st_mk_ref(v___x_3477_);
v___x_3479_ = lean_box(v___x_3466_);
v___f_3480_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___lam__1___boxed), 12, 3);
lean_closure_set(v___f_3480_, 0, v_cfg_3468_);
lean_closure_set(v___f_3480_, 1, v___x_3478_);
lean_closure_set(v___f_3480_, 2, v___x_3479_);
v___x_3481_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3480_, v___y_3469_, v___y_3470_, v___y_3471_, v___y_3472_, v___y_3473_, v___y_3474_, v___y_3475_, v___y_3476_);
return v___x_3481_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1___boxed(lean_object* v_x_3531_, lean_object* v_a_3532_, lean_object* v_a_3533_, lean_object* v_a_3534_, lean_object* v_a_3535_, lean_object* v_a_3536_, lean_object* v_a_3537_, lean_object* v_a_3538_, lean_object* v_a_3539_, lean_object* v_a_3540_){
_start:
{
lean_object* v_res_3541_; 
v_res_3541_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ring1NF__1(v_x_3531_, v_a_3532_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, v_a_3537_, v_a_3538_, v_a_3539_);
lean_dec(v_a_3539_);
lean_dec_ref(v_a_3538_);
lean_dec(v_a_3537_);
lean_dec_ref(v_a_3536_);
lean_dec(v_a_3535_);
lean_dec_ref(v_a_3534_);
lean_dec(v_a_3533_);
lean_dec_ref(v_a_3532_);
return v_res_3541_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4(void){
_start:
{
lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3554_; lean_object* v___x_3555_; 
v___x_3552_ = l_Lean_Parser_Tactic_optConfig;
v___x_3553_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__3));
v___x_3554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__3));
v___x_3555_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3555_, 0, v___x_3554_);
lean_ctor_set(v___x_3555_, 1, v___x_3553_);
lean_ctor_set(v___x_3555_, 2, v___x_3552_);
return v___x_3555_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5(void){
_start:
{
lean_object* v___x_3556_; lean_object* v___x_3557_; lean_object* v___x_3558_; lean_object* v___x_3559_; 
v___x_3556_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__4);
v___x_3557_ = lean_unsigned_to_nat(1022u);
v___x_3558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1));
v___x_3559_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3559_, 0, v___x_3558_);
lean_ctor_set(v___x_3559_, 1, v___x_3557_);
lean_ctor_set(v___x_3559_, 2, v___x_3556_);
return v___x_3559_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21__(void){
_start:
{
lean_object* v___x_3560_; 
v___x_3560_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__5);
return v___x_3560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing1__nf_x21____1(lean_object* v_x_3561_, lean_object* v_a_3562_, lean_object* v_a_3563_){
_start:
{
lean_object* v___x_3564_; uint8_t v___x_3565_; 
v___x_3564_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21___00__closed__1));
lean_inc(v_x_3561_);
v___x_3565_ = l_Lean_Syntax_isOfKind(v_x_3561_, v___x_3564_);
if (v___x_3565_ == 0)
{
lean_object* v___x_3566_; lean_object* v___x_3567_; 
lean_dec(v_x_3561_);
v___x_3566_ = lean_box(1);
v___x_3567_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3567_, 0, v___x_3566_);
lean_ctor_set(v___x_3567_, 1, v_a_3563_);
return v___x_3567_;
}
else
{
lean_object* v_ref_3568_; lean_object* v___x_3569_; lean_object* v___x_3570_; uint8_t v___x_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3574_; lean_object* v___x_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; 
v_ref_3568_ = lean_ctor_get(v_a_3562_, 5);
v___x_3569_ = lean_unsigned_to_nat(1u);
v___x_3570_ = l_Lean_Syntax_getArg(v_x_3561_, v___x_3569_);
lean_dec(v_x_3561_);
v___x_3571_ = 0;
v___x_3572_ = l_Lean_SourceInfo_fromRef(v_ref_3568_, v___x_3571_);
v___x_3573_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__1));
v___x_3574_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF___closed__2));
lean_inc_n(v___x_3572_, 3);
v___x_3575_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3575_, 0, v___x_3572_);
lean_ctor_set(v___x_3575_, 1, v___x_3574_);
v___x_3576_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_3577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8));
v___x_3578_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3578_, 0, v___x_3572_);
lean_ctor_set(v___x_3578_, 1, v___x_3577_);
v___x_3579_ = l_Lean_Syntax_node1(v___x_3572_, v___x_3576_, v___x_3578_);
v___x_3580_ = l_Lean_Syntax_node3(v___x_3572_, v___x_3573_, v___x_3575_, v___x_3579_, v___x_3570_);
v___x_3581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3581_, 0, v___x_3580_);
lean_ctor_set(v___x_3581_, 1, v_a_3563_);
return v___x_3581_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing1__nf_x21____1___boxed(lean_object* v_x_3582_, lean_object* v_a_3583_, lean_object* v_a_3584_){
_start:
{
lean_object* v_res_3585_; 
v_res_3585_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing1__nf_x21____1(v_x_3582_, v_a_3583_, v_a_3584_);
lean_dec_ref(v_a_3583_);
return v_res_3585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg(lean_object* v_e_3586_, lean_object* v___y_3587_){
_start:
{
uint8_t v___x_3589_; 
v___x_3589_ = l_Lean_Expr_hasMVar(v_e_3586_);
if (v___x_3589_ == 0)
{
lean_object* v___x_3590_; 
v___x_3590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3590_, 0, v_e_3586_);
return v___x_3590_;
}
else
{
lean_object* v___x_3591_; lean_object* v_mctx_3592_; lean_object* v___x_3593_; lean_object* v_fst_3594_; lean_object* v_snd_3595_; lean_object* v___x_3596_; lean_object* v_cache_3597_; lean_object* v_zetaDeltaFVarIds_3598_; lean_object* v_postponed_3599_; lean_object* v_diag_3600_; lean_object* v___x_3602_; uint8_t v_isShared_3603_; uint8_t v_isSharedCheck_3609_; 
v___x_3591_ = lean_st_ref_get(v___y_3587_);
v_mctx_3592_ = lean_ctor_get(v___x_3591_, 0);
lean_inc_ref(v_mctx_3592_);
lean_dec(v___x_3591_);
v___x_3593_ = l_Lean_instantiateMVarsCore(v_mctx_3592_, v_e_3586_);
v_fst_3594_ = lean_ctor_get(v___x_3593_, 0);
lean_inc(v_fst_3594_);
v_snd_3595_ = lean_ctor_get(v___x_3593_, 1);
lean_inc(v_snd_3595_);
lean_dec_ref(v___x_3593_);
v___x_3596_ = lean_st_ref_take(v___y_3587_);
v_cache_3597_ = lean_ctor_get(v___x_3596_, 1);
v_zetaDeltaFVarIds_3598_ = lean_ctor_get(v___x_3596_, 2);
v_postponed_3599_ = lean_ctor_get(v___x_3596_, 3);
v_diag_3600_ = lean_ctor_get(v___x_3596_, 4);
v_isSharedCheck_3609_ = !lean_is_exclusive(v___x_3596_);
if (v_isSharedCheck_3609_ == 0)
{
lean_object* v_unused_3610_; 
v_unused_3610_ = lean_ctor_get(v___x_3596_, 0);
lean_dec(v_unused_3610_);
v___x_3602_ = v___x_3596_;
v_isShared_3603_ = v_isSharedCheck_3609_;
goto v_resetjp_3601_;
}
else
{
lean_inc(v_diag_3600_);
lean_inc(v_postponed_3599_);
lean_inc(v_zetaDeltaFVarIds_3598_);
lean_inc(v_cache_3597_);
lean_dec(v___x_3596_);
v___x_3602_ = lean_box(0);
v_isShared_3603_ = v_isSharedCheck_3609_;
goto v_resetjp_3601_;
}
v_resetjp_3601_:
{
lean_object* v___x_3605_; 
if (v_isShared_3603_ == 0)
{
lean_ctor_set(v___x_3602_, 0, v_snd_3595_);
v___x_3605_ = v___x_3602_;
goto v_reusejp_3604_;
}
else
{
lean_object* v_reuseFailAlloc_3608_; 
v_reuseFailAlloc_3608_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3608_, 0, v_snd_3595_);
lean_ctor_set(v_reuseFailAlloc_3608_, 1, v_cache_3597_);
lean_ctor_set(v_reuseFailAlloc_3608_, 2, v_zetaDeltaFVarIds_3598_);
lean_ctor_set(v_reuseFailAlloc_3608_, 3, v_postponed_3599_);
lean_ctor_set(v_reuseFailAlloc_3608_, 4, v_diag_3600_);
v___x_3605_ = v_reuseFailAlloc_3608_;
goto v_reusejp_3604_;
}
v_reusejp_3604_:
{
lean_object* v___x_3606_; lean_object* v___x_3607_; 
v___x_3606_ = lean_st_ref_set(v___y_3587_, v___x_3605_);
v___x_3607_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3607_, 0, v_fst_3594_);
return v___x_3607_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg___boxed(lean_object* v_e_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_){
_start:
{
lean_object* v_res_3614_; 
v_res_3614_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg(v_e_3611_, v___y_3612_);
lean_dec(v___y_3612_);
return v_res_3614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0(lean_object* v_e_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_, lean_object* v___y_3618_, lean_object* v___y_3619_, lean_object* v___y_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_, lean_object* v___y_3623_){
_start:
{
lean_object* v___x_3625_; 
v___x_3625_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg(v_e_3615_, v___y_3621_);
return v___x_3625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___boxed(lean_object* v_e_3626_, lean_object* v___y_3627_, lean_object* v___y_3628_, lean_object* v___y_3629_, lean_object* v___y_3630_, lean_object* v___y_3631_, lean_object* v___y_3632_, lean_object* v___y_3633_, lean_object* v___y_3634_, lean_object* v___y_3635_){
_start:
{
lean_object* v_res_3636_; 
v_res_3636_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0(v_e_3626_, v___y_3627_, v___y_3628_, v___y_3629_, v___y_3630_, v___y_3631_, v___y_3632_, v___y_3633_, v___y_3634_);
lean_dec(v___y_3634_);
lean_dec_ref(v___y_3633_);
lean_dec(v___y_3632_);
lean_dec_ref(v___y_3631_);
lean_dec(v___y_3630_);
lean_dec_ref(v___y_3629_);
lean_dec(v___y_3628_);
lean_dec_ref(v___y_3627_);
return v_res_3636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0(lean_object* v___x_3637_, lean_object* v___x_3638_, uint8_t v___x_3639_, lean_object* v_tk_3640_, uint8_t v___x_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_){
_start:
{
lean_object* v___x_3651_; 
v___x_3651_ = lp_mathlib_Mathlib_Tactic_RingNF_elabConfig___redArg(v___x_3637_, v___x_3638_, v___x_3639_, v___y_3642_, v___y_3648_, v___y_3649_);
if (lean_obj_tag(v___x_3651_) == 0)
{
lean_object* v_a_3652_; lean_object* v_cfg_3654_; lean_object* v___y_3655_; lean_object* v___y_3656_; lean_object* v___y_3657_; lean_object* v___y_3658_; lean_object* v___y_3659_; lean_object* v___y_3660_; lean_object* v___y_3661_; lean_object* v___y_3662_; 
v_a_3652_ = lean_ctor_get(v___x_3651_, 0);
lean_inc(v_a_3652_);
lean_dec_ref_known(v___x_3651_, 1);
if (lean_obj_tag(v_tk_3640_) == 0)
{
v_cfg_3654_ = v_a_3652_;
v___y_3655_ = v___y_3642_;
v___y_3656_ = v___y_3643_;
v___y_3657_ = v___y_3644_;
v___y_3658_ = v___y_3645_;
v___y_3659_ = v___y_3646_;
v___y_3660_ = v___y_3647_;
v___y_3661_ = v___y_3648_;
v___y_3662_ = v___y_3649_;
goto v___jp_3653_;
}
else
{
if (v___x_3641_ == 0)
{
v_cfg_3654_ = v_a_3652_;
v___y_3655_ = v___y_3642_;
v___y_3656_ = v___y_3643_;
v___y_3657_ = v___y_3644_;
v___y_3658_ = v___y_3645_;
v___y_3659_ = v___y_3646_;
v___y_3660_ = v___y_3647_;
v___y_3661_ = v___y_3648_;
v___y_3662_ = v___y_3649_;
goto v___jp_3653_;
}
else
{
lean_object* v_toConfig_3691_; uint8_t v_ifUnchanged_3692_; uint8_t v_mode_3693_; lean_object* v___x_3695_; uint8_t v_isShared_3696_; uint8_t v_isSharedCheck_3709_; 
v_toConfig_3691_ = lean_ctor_get(v_a_3652_, 0);
v_ifUnchanged_3692_ = lean_ctor_get_uint8(v_a_3652_, sizeof(void*)*1);
v_mode_3693_ = lean_ctor_get_uint8(v_a_3652_, sizeof(void*)*1 + 1);
v_isSharedCheck_3709_ = !lean_is_exclusive(v_a_3652_);
if (v_isSharedCheck_3709_ == 0)
{
v___x_3695_ = v_a_3652_;
v_isShared_3696_ = v_isSharedCheck_3709_;
goto v_resetjp_3694_;
}
else
{
lean_inc(v_toConfig_3691_);
lean_dec(v_a_3652_);
v___x_3695_ = lean_box(0);
v_isShared_3696_ = v_isSharedCheck_3709_;
goto v_resetjp_3694_;
}
v_resetjp_3694_:
{
uint8_t v_contextual_3697_; lean_object* v___x_3699_; uint8_t v_isShared_3700_; uint8_t v_isSharedCheck_3708_; 
v_contextual_3697_ = lean_ctor_get_uint8(v_toConfig_3691_, 2);
v_isSharedCheck_3708_ = !lean_is_exclusive(v_toConfig_3691_);
if (v_isSharedCheck_3708_ == 0)
{
v___x_3699_ = v_toConfig_3691_;
v_isShared_3700_ = v_isSharedCheck_3708_;
goto v_resetjp_3698_;
}
else
{
lean_dec(v_toConfig_3691_);
v___x_3699_ = lean_box(0);
v_isShared_3700_ = v_isSharedCheck_3708_;
goto v_resetjp_3698_;
}
v_resetjp_3698_:
{
uint8_t v___x_3701_; lean_object* v___x_3703_; 
v___x_3701_ = 1;
if (v_isShared_3700_ == 0)
{
v___x_3703_ = v___x_3699_;
goto v_reusejp_3702_;
}
else
{
lean_object* v_reuseFailAlloc_3707_; 
v_reuseFailAlloc_3707_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v_reuseFailAlloc_3707_, 2, v_contextual_3697_);
v___x_3703_ = v_reuseFailAlloc_3707_;
goto v_reusejp_3702_;
}
v_reusejp_3702_:
{
lean_object* v___x_3705_; 
lean_ctor_set_uint8(v___x_3703_, 0, v___x_3701_);
lean_ctor_set_uint8(v___x_3703_, 1, v___x_3639_);
if (v_isShared_3696_ == 0)
{
lean_ctor_set(v___x_3695_, 0, v___x_3703_);
v___x_3705_ = v___x_3695_;
goto v_reusejp_3704_;
}
else
{
lean_object* v_reuseFailAlloc_3706_; 
v_reuseFailAlloc_3706_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_reuseFailAlloc_3706_, 0, v___x_3703_);
lean_ctor_set_uint8(v_reuseFailAlloc_3706_, sizeof(void*)*1, v_ifUnchanged_3692_);
lean_ctor_set_uint8(v_reuseFailAlloc_3706_, sizeof(void*)*1 + 1, v_mode_3693_);
v___x_3705_ = v_reuseFailAlloc_3706_;
goto v_reusejp_3704_;
}
v_reusejp_3704_:
{
v_cfg_3654_ = v___x_3705_;
v___y_3655_ = v___y_3642_;
v___y_3656_ = v___y_3643_;
v___y_3657_ = v___y_3644_;
v___y_3658_ = v___y_3645_;
v___y_3659_ = v___y_3646_;
v___y_3660_ = v___y_3647_;
v___y_3661_ = v___y_3648_;
v___y_3662_ = v___y_3649_;
goto v___jp_3653_;
}
}
}
}
}
}
v___jp_3653_:
{
lean_object* v___x_3663_; lean_object* v___x_3664_; lean_object* v___x_3665_; 
v___x_3663_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__0));
v___x_3664_ = lean_st_mk_ref(v___x_3663_);
v___x_3665_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_3656_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
if (lean_obj_tag(v___x_3665_) == 0)
{
lean_object* v_a_3666_; lean_object* v___x_3667_; lean_object* v_a_3668_; lean_object* v_toConfig_3669_; lean_object* v___x_3670_; lean_object* v___x_3671_; lean_object* v___x_3672_; 
v_a_3666_ = lean_ctor_get(v___x_3665_, 0);
lean_inc(v_a_3666_);
lean_dec_ref_known(v___x_3665_, 1);
v___x_3667_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_RingNF_elabRingNFConv_spec__0___redArg(v_a_3666_, v___y_3660_);
v_a_3668_ = lean_ctor_get(v___x_3667_, 0);
lean_inc(v_a_3668_);
lean_dec_ref(v___x_3667_);
v_toConfig_3669_ = lean_ctor_get(v_cfg_3654_, 0);
lean_inc_ref(v_toConfig_3669_);
v___x_3670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1___closed__1));
v___x_3671_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF_cleanup___boxed), 7, 1);
lean_closure_set(v___x_3671_, 0, v_cfg_3654_);
v___x_3672_ = lp_mathlib_Mathlib_Tactic_AtomM_recurse(v___x_3664_, v_toConfig_3669_, v___x_3639_, v___x_3670_, v___x_3671_, v_a_3668_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
if (lean_obj_tag(v___x_3672_) == 0)
{
lean_object* v_a_3673_; lean_object* v___x_3674_; 
v_a_3673_ = lean_ctor_get(v___x_3672_, 0);
lean_inc(v_a_3673_);
lean_dec_ref_known(v___x_3672_, 1);
v___x_3674_ = l_Lean_Elab_Tactic_Conv_applySimpResult(v_a_3673_, v___y_3655_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
return v___x_3674_;
}
else
{
lean_object* v_a_3675_; lean_object* v___x_3677_; uint8_t v_isShared_3678_; uint8_t v_isSharedCheck_3682_; 
v_a_3675_ = lean_ctor_get(v___x_3672_, 0);
v_isSharedCheck_3682_ = !lean_is_exclusive(v___x_3672_);
if (v_isSharedCheck_3682_ == 0)
{
v___x_3677_ = v___x_3672_;
v_isShared_3678_ = v_isSharedCheck_3682_;
goto v_resetjp_3676_;
}
else
{
lean_inc(v_a_3675_);
lean_dec(v___x_3672_);
v___x_3677_ = lean_box(0);
v_isShared_3678_ = v_isSharedCheck_3682_;
goto v_resetjp_3676_;
}
v_resetjp_3676_:
{
lean_object* v___x_3680_; 
if (v_isShared_3678_ == 0)
{
v___x_3680_ = v___x_3677_;
goto v_reusejp_3679_;
}
else
{
lean_object* v_reuseFailAlloc_3681_; 
v_reuseFailAlloc_3681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3681_, 0, v_a_3675_);
v___x_3680_ = v_reuseFailAlloc_3681_;
goto v_reusejp_3679_;
}
v_reusejp_3679_:
{
return v___x_3680_;
}
}
}
}
else
{
lean_object* v_a_3683_; lean_object* v___x_3685_; uint8_t v_isShared_3686_; uint8_t v_isSharedCheck_3690_; 
lean_dec(v___x_3664_);
lean_dec_ref(v_cfg_3654_);
v_a_3683_ = lean_ctor_get(v___x_3665_, 0);
v_isSharedCheck_3690_ = !lean_is_exclusive(v___x_3665_);
if (v_isSharedCheck_3690_ == 0)
{
v___x_3685_ = v___x_3665_;
v_isShared_3686_ = v_isSharedCheck_3690_;
goto v_resetjp_3684_;
}
else
{
lean_inc(v_a_3683_);
lean_dec(v___x_3665_);
v___x_3685_ = lean_box(0);
v_isShared_3686_ = v_isSharedCheck_3690_;
goto v_resetjp_3684_;
}
v_resetjp_3684_:
{
lean_object* v___x_3688_; 
if (v_isShared_3686_ == 0)
{
v___x_3688_ = v___x_3685_;
goto v_reusejp_3687_;
}
else
{
lean_object* v_reuseFailAlloc_3689_; 
v_reuseFailAlloc_3689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3689_, 0, v_a_3683_);
v___x_3688_ = v_reuseFailAlloc_3689_;
goto v_reusejp_3687_;
}
v_reusejp_3687_:
{
return v___x_3688_;
}
}
}
}
}
else
{
lean_object* v_a_3710_; lean_object* v___x_3712_; uint8_t v_isShared_3713_; uint8_t v_isSharedCheck_3717_; 
v_a_3710_ = lean_ctor_get(v___x_3651_, 0);
v_isSharedCheck_3717_ = !lean_is_exclusive(v___x_3651_);
if (v_isSharedCheck_3717_ == 0)
{
v___x_3712_ = v___x_3651_;
v_isShared_3713_ = v_isSharedCheck_3717_;
goto v_resetjp_3711_;
}
else
{
lean_inc(v_a_3710_);
lean_dec(v___x_3651_);
v___x_3712_ = lean_box(0);
v_isShared_3713_ = v_isSharedCheck_3717_;
goto v_resetjp_3711_;
}
v_resetjp_3711_:
{
lean_object* v___x_3715_; 
if (v_isShared_3713_ == 0)
{
v___x_3715_ = v___x_3712_;
goto v_reusejp_3714_;
}
else
{
lean_object* v_reuseFailAlloc_3716_; 
v_reuseFailAlloc_3716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3716_, 0, v_a_3710_);
v___x_3715_ = v_reuseFailAlloc_3716_;
goto v_reusejp_3714_;
}
v_reusejp_3714_:
{
return v___x_3715_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0___boxed(lean_object* v___x_3718_, lean_object* v___x_3719_, lean_object* v___x_3720_, lean_object* v_tk_3721_, lean_object* v___x_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_, lean_object* v___y_3729_, lean_object* v___y_3730_, lean_object* v___y_3731_){
_start:
{
uint8_t v___x_2737__boxed_3732_; uint8_t v___x_2738__boxed_3733_; lean_object* v_res_3734_; 
v___x_2737__boxed_3732_ = lean_unbox(v___x_3720_);
v___x_2738__boxed_3733_ = lean_unbox(v___x_3722_);
v_res_3734_ = lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0(v___x_3718_, v___x_3719_, v___x_2737__boxed_3732_, v_tk_3721_, v___x_2738__boxed_3733_, v___y_3723_, v___y_3724_, v___y_3725_, v___y_3726_, v___y_3727_, v___y_3728_, v___y_3729_, v___y_3730_);
lean_dec(v___y_3730_);
lean_dec_ref(v___y_3729_);
lean_dec(v___y_3728_);
lean_dec_ref(v___y_3727_);
lean_dec(v___y_3726_);
lean_dec_ref(v___y_3725_);
lean_dec(v___y_3724_);
lean_dec_ref(v___y_3723_);
lean_dec(v_tk_3721_);
return v_res_3734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv(lean_object* v_stx_3742_, lean_object* v_a_3743_, lean_object* v_a_3744_, lean_object* v_a_3745_, lean_object* v_a_3746_, lean_object* v_a_3747_, lean_object* v_a_3748_, lean_object* v_a_3749_, lean_object* v_a_3750_){
_start:
{
lean_object* v___x_3752_; uint8_t v___x_3753_; lean_object* v_tk_3755_; lean_object* v___y_3756_; lean_object* v___y_3757_; lean_object* v___y_3758_; lean_object* v___y_3759_; lean_object* v___y_3760_; lean_object* v___y_3761_; lean_object* v___y_3762_; lean_object* v___y_3763_; 
v___x_3752_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1));
lean_inc(v_stx_3742_);
v___x_3753_ = l_Lean_Syntax_isOfKind(v_stx_3742_, v___x_3752_);
if (v___x_3753_ == 0)
{
lean_object* v___x_3774_; 
lean_dec(v_stx_3742_);
v___x_3774_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3774_;
}
else
{
lean_object* v___x_3775_; lean_object* v___x_3776_; uint8_t v___x_3777_; 
v___x_3775_ = lean_unsigned_to_nat(1u);
v___x_3776_ = l_Lean_Syntax_getArg(v_stx_3742_, v___x_3775_);
v___x_3777_ = l_Lean_Syntax_isNone(v___x_3776_);
if (v___x_3777_ == 0)
{
uint8_t v___x_3778_; 
lean_inc(v___x_3776_);
v___x_3778_ = l_Lean_Syntax_matchesNull(v___x_3776_, v___x_3775_);
if (v___x_3778_ == 0)
{
lean_object* v___x_3779_; 
lean_dec(v___x_3776_);
lean_dec(v_stx_3742_);
v___x_3779_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3779_;
}
else
{
lean_object* v___x_3780_; lean_object* v_tk_3781_; lean_object* v___x_3782_; 
v___x_3780_ = lean_unsigned_to_nat(0u);
v_tk_3781_ = l_Lean_Syntax_getArg(v___x_3776_, v___x_3780_);
lean_dec(v___x_3776_);
v___x_3782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3782_, 0, v_tk_3781_);
v_tk_3755_ = v___x_3782_;
v___y_3756_ = v_a_3743_;
v___y_3757_ = v_a_3744_;
v___y_3758_ = v_a_3745_;
v___y_3759_ = v_a_3746_;
v___y_3760_ = v_a_3747_;
v___y_3761_ = v_a_3748_;
v___y_3762_ = v_a_3749_;
v___y_3763_ = v_a_3750_;
goto v___jp_3754_;
}
}
else
{
lean_object* v___x_3783_; 
lean_dec(v___x_3776_);
v___x_3783_ = lean_box(0);
v_tk_3755_ = v___x_3783_;
v___y_3756_ = v_a_3743_;
v___y_3757_ = v_a_3744_;
v___y_3758_ = v_a_3745_;
v___y_3759_ = v_a_3746_;
v___y_3760_ = v_a_3747_;
v___y_3761_ = v_a_3748_;
v___y_3762_ = v_a_3749_;
v___y_3763_ = v_a_3750_;
goto v___jp_3754_;
}
}
v___jp_3754_:
{
lean_object* v___x_3764_; lean_object* v___x_3765_; lean_object* v___x_3766_; uint8_t v___x_3767_; 
v___x_3764_ = lean_unsigned_to_nat(2u);
v___x_3765_ = l_Lean_Syntax_getArg(v_stx_3742_, v___x_3764_);
lean_dec(v_stx_3742_);
v___x_3766_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2));
lean_inc(v___x_3765_);
v___x_3767_ = l_Lean_Syntax_isOfKind(v___x_3765_, v___x_3766_);
if (v___x_3767_ == 0)
{
lean_object* v___x_3768_; 
lean_dec(v___x_3765_);
lean_dec(v_tk_3755_);
v___x_3768_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______elabRules__Mathlib__Tactic__RingNF__ringNF__1_spec__0___redArg();
return v___x_3768_;
}
else
{
lean_object* v___x_3769_; lean_object* v___x_3770_; lean_object* v___x_3771_; lean_object* v___f_3772_; lean_object* v___x_3773_; 
v___x_3769_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_));
v___x_3770_ = lean_box(v___x_3753_);
v___x_3771_ = lean_box(v___x_3767_);
v___f_3772_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___lam__0___boxed), 14, 5);
lean_closure_set(v___f_3772_, 0, v___x_3765_);
lean_closure_set(v___f_3772_, 1, v___x_3769_);
lean_closure_set(v___f_3772_, 2, v___x_3770_);
lean_closure_set(v___f_3772_, 3, v_tk_3755_);
lean_closure_set(v___f_3772_, 4, v___x_3771_);
v___x_3773_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3772_, v___y_3756_, v___y_3757_, v___y_3758_, v___y_3759_, v___y_3760_, v___y_3761_, v___y_3762_, v___y_3763_);
return v___x_3773_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___boxed(lean_object* v_stx_3784_, lean_object* v_a_3785_, lean_object* v_a_3786_, lean_object* v_a_3787_, lean_object* v_a_3788_, lean_object* v_a_3789_, lean_object* v_a_3790_, lean_object* v_a_3791_, lean_object* v_a_3792_, lean_object* v_a_3793_){
_start:
{
lean_object* v_res_3794_; 
v_res_3794_ = lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv(v_stx_3784_, v_a_3785_, v_a_3786_, v_a_3787_, v_a_3788_, v_a_3789_, v_a_3790_, v_a_3791_, v_a_3792_);
lean_dec(v_a_3792_);
lean_dec_ref(v_a_3791_);
lean_dec(v_a_3790_);
lean_dec_ref(v_a_3789_);
lean_dec(v_a_3788_);
lean_dec_ref(v_a_3787_);
lean_dec(v_a_3786_);
lean_dec_ref(v_a_3785_);
return v_res_3794_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2(void){
_start:
{
lean_object* v___x_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; 
v___x_3801_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4, &lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__4);
v___x_3802_ = lean_unsigned_to_nat(1022u);
v___x_3803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1));
v___x_3804_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3804_, 0, v___x_3803_);
lean_ctor_set(v___x_3804_, 1, v___x_3802_);
lean_ctor_set(v___x_3804_, 2, v___x_3801_);
return v___x_3804_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21__(void){
_start:
{
lean_object* v___x_3805_; 
v___x_3805_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2, &lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__2);
return v___x_3805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing__nf_x21____1(lean_object* v_x_3806_, lean_object* v_a_3807_, lean_object* v_a_3808_){
_start:
{
lean_object* v___x_3809_; uint8_t v___x_3810_; 
v___x_3809_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1));
lean_inc(v_x_3806_);
v___x_3810_ = l_Lean_Syntax_isOfKind(v_x_3806_, v___x_3809_);
if (v___x_3810_ == 0)
{
lean_object* v___x_3811_; lean_object* v___x_3812_; 
lean_dec(v_x_3806_);
v___x_3811_ = lean_box(1);
v___x_3812_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3812_, 0, v___x_3811_);
lean_ctor_set(v___x_3812_, 1, v_a_3808_);
return v___x_3812_;
}
else
{
lean_object* v_ref_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; uint8_t v___x_3816_; lean_object* v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3819_; lean_object* v___x_3820_; lean_object* v___x_3821_; lean_object* v___x_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___x_3825_; lean_object* v___x_3826_; 
v_ref_3813_ = lean_ctor_get(v_a_3807_, 5);
v___x_3814_ = lean_unsigned_to_nat(1u);
v___x_3815_ = l_Lean_Syntax_getArg(v_x_3806_, v___x_3814_);
lean_dec(v_x_3806_);
v___x_3816_ = 0;
v___x_3817_ = l_Lean_SourceInfo_fromRef(v_ref_3813_, v___x_3816_);
v___x_3818_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1));
v___x_3819_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4));
lean_inc_n(v___x_3817_, 3);
v___x_3820_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3820_, 0, v___x_3817_);
lean_ctor_set(v___x_3820_, 1, v___x_3819_);
v___x_3821_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_3822_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__8));
v___x_3823_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3823_, 0, v___x_3817_);
lean_ctor_set(v___x_3823_, 1, v___x_3822_);
v___x_3824_ = l_Lean_Syntax_node1(v___x_3817_, v___x_3821_, v___x_3823_);
v___x_3825_ = l_Lean_Syntax_node3(v___x_3817_, v___x_3818_, v___x_3820_, v___x_3824_, v___x_3815_);
v___x_3826_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3826_, 0, v___x_3825_);
lean_ctor_set(v___x_3826_, 1, v_a_3808_);
return v___x_3826_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing__nf_x21____1___boxed(lean_object* v_x_3827_, lean_object* v_a_3828_, lean_object* v_a_3829_){
_start:
{
lean_object* v_res_3830_; 
v_res_3830_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing__nf_x21____1(v_x_3827_, v_a_3828_, v_a_3829_);
lean_dec_ref(v_a_3828_);
return v_res_3830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1(lean_object* v_x_3884_, lean_object* v_a_3885_, lean_object* v_a_3886_){
_start:
{
lean_object* v___x_3887_; uint8_t v___x_3888_; 
v___x_3887_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ring___closed__1));
v___x_3888_ = l_Lean_Syntax_isOfKind(v_x_3884_, v___x_3887_);
if (v___x_3888_ == 0)
{
lean_object* v___x_3889_; lean_object* v___x_3890_; 
v___x_3889_ = lean_box(1);
v___x_3890_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3890_, 0, v___x_3889_);
lean_ctor_set(v___x_3890_, 1, v_a_3886_);
return v___x_3890_;
}
else
{
lean_object* v_ref_3891_; uint8_t v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; 
v_ref_3891_ = lean_ctor_get(v_a_3885_, 5);
v___x_3892_ = 0;
v___x_3893_ = l_Lean_SourceInfo_fromRef(v_ref_3891_, v___x_3892_);
v___x_3894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0));
v___x_3895_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1));
lean_inc_n(v___x_3893_, 22);
v___x_3896_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3896_, 0, v___x_3893_);
lean_ctor_set(v___x_3896_, 1, v___x_3894_);
v___x_3897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_3898_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3));
v___x_3899_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4));
v___x_3900_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3900_, 0, v___x_3893_);
lean_ctor_set(v___x_3900_, 1, v___x_3899_);
v___x_3901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6));
v___x_3902_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8));
v___x_3903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10));
v___x_3904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11));
v___x_3905_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3905_, 0, v___x_3893_);
lean_ctor_set(v___x_3905_, 1, v___x_3903_);
v___x_3906_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2);
v___x_3907_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3907_, 0, v___x_3893_);
lean_ctor_set(v___x_3907_, 1, v___x_3897_);
lean_ctor_set(v___x_3907_, 2, v___x_3906_);
lean_inc_ref_n(v___x_3907_, 3);
v___x_3908_ = l_Lean_Syntax_node2(v___x_3893_, v___x_3904_, v___x_3905_, v___x_3907_);
v___x_3909_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3897_, v___x_3908_);
v___x_3910_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3902_, v___x_3909_);
v___x_3911_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3901_, v___x_3910_);
lean_inc_ref(v___x_3900_);
v___x_3912_ = l_Lean_Syntax_node2(v___x_3893_, v___x_3898_, v___x_3900_, v___x_3911_);
v___x_3913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13));
v___x_3914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14));
v___x_3915_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3915_, 0, v___x_3893_);
lean_ctor_set(v___x_3915_, 1, v___x_3914_);
v___x_3916_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__1));
v___x_3917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4));
v___x_3918_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3918_, 0, v___x_3893_);
lean_ctor_set(v___x_3918_, 1, v___x_3917_);
v___x_3919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2));
v___x_3920_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3919_, v___x_3907_);
v___x_3921_ = l_Lean_Syntax_node4(v___x_3893_, v___x_3916_, v___x_3918_, v___x_3907_, v___x_3920_, v___x_3907_);
v___x_3922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16));
v___x_3923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__17));
v___x_3924_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3924_, 0, v___x_3893_);
lean_ctor_set(v___x_3924_, 1, v___x_3923_);
v___x_3925_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3922_, v___x_3924_);
v___x_3926_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3897_, v___x_3925_);
v___x_3927_ = l_Lean_Syntax_node3(v___x_3893_, v___x_3913_, v___x_3915_, v___x_3921_, v___x_3926_);
v___x_3928_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3897_, v___x_3927_);
v___x_3929_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3902_, v___x_3928_);
v___x_3930_ = l_Lean_Syntax_node1(v___x_3893_, v___x_3901_, v___x_3929_);
v___x_3931_ = l_Lean_Syntax_node2(v___x_3893_, v___x_3898_, v___x_3900_, v___x_3930_);
v___x_3932_ = l_Lean_Syntax_node2(v___x_3893_, v___x_3897_, v___x_3912_, v___x_3931_);
v___x_3933_ = l_Lean_Syntax_node2(v___x_3893_, v___x_3895_, v___x_3896_, v___x_3932_);
v___x_3934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3934_, 0, v___x_3933_);
lean_ctor_set(v___x_3934_, 1, v_a_3886_);
return v___x_3934_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___boxed(lean_object* v_x_3935_, lean_object* v_a_3936_, lean_object* v_a_3937_){
_start:
{
lean_object* v_res_3938_; 
v_res_3938_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1(v_x_3935_, v_a_3936_, v_a_3937_);
lean_dec_ref(v_a_3936_);
return v_res_3938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1(lean_object* v_x_3962_, lean_object* v_a_3963_, lean_object* v_a_3964_){
_start:
{
lean_object* v___x_3965_; uint8_t v___x_3966_; 
v___x_3965_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing_x21___closed__1));
v___x_3966_ = l_Lean_Syntax_isOfKind(v_x_3962_, v___x_3965_);
if (v___x_3966_ == 0)
{
lean_object* v___x_3967_; lean_object* v___x_3968_; 
v___x_3967_ = lean_box(1);
v___x_3968_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3968_, 0, v___x_3967_);
lean_ctor_set(v___x_3968_, 1, v_a_3964_);
return v___x_3968_;
}
else
{
lean_object* v_ref_3969_; uint8_t v___x_3970_; lean_object* v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; lean_object* v___x_3981_; lean_object* v___x_3982_; lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v___x_3985_; lean_object* v___x_3986_; lean_object* v___x_3987_; lean_object* v___x_3988_; lean_object* v___x_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; lean_object* v___x_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; lean_object* v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; lean_object* v___x_4009_; lean_object* v___x_4010_; lean_object* v___x_4011_; lean_object* v___x_4012_; 
v_ref_3969_ = lean_ctor_get(v_a_3963_, 5);
v___x_3970_ = 0;
v___x_3971_ = l_Lean_SourceInfo_fromRef(v_ref_3969_, v___x_3970_);
v___x_3972_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0));
v___x_3973_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__1));
lean_inc_n(v___x_3971_, 22);
v___x_3974_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3974_, 0, v___x_3971_);
lean_ctor_set(v___x_3974_, 1, v___x_3972_);
v___x_3975_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_3976_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3));
v___x_3977_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4));
v___x_3978_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3978_, 0, v___x_3971_);
lean_ctor_set(v___x_3978_, 1, v___x_3977_);
v___x_3979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6));
v___x_3980_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8));
v___x_3981_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1));
v___x_3982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__2));
v___x_3983_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3983_, 0, v___x_3971_);
lean_ctor_set(v___x_3983_, 1, v___x_3982_);
v___x_3984_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3981_, v___x_3983_);
v___x_3985_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3975_, v___x_3984_);
v___x_3986_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3980_, v___x_3985_);
v___x_3987_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3979_, v___x_3986_);
lean_inc_ref(v___x_3978_);
v___x_3988_ = l_Lean_Syntax_node2(v___x_3971_, v___x_3976_, v___x_3978_, v___x_3987_);
v___x_3989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__13));
v___x_3990_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14));
v___x_3991_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3991_, 0, v___x_3971_);
lean_ctor_set(v___x_3991_, 1, v___x_3990_);
v___x_3992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__1));
v___x_3993_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2));
v___x_3994_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3994_, 0, v___x_3971_);
lean_ctor_set(v___x_3994_, 1, v___x_3993_);
v___x_3995_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2));
v___x_3996_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2);
v___x_3997_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3997_, 0, v___x_3971_);
lean_ctor_set(v___x_3997_, 1, v___x_3975_);
lean_ctor_set(v___x_3997_, 2, v___x_3996_);
lean_inc_ref(v___x_3997_);
v___x_3998_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3995_, v___x_3997_);
v___x_3999_ = l_Lean_Syntax_node3(v___x_3971_, v___x_3992_, v___x_3994_, v___x_3998_, v___x_3997_);
v___x_4000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16));
v___x_4001_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__3));
v___x_4002_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4002_, 0, v___x_3971_);
lean_ctor_set(v___x_4002_, 1, v___x_4001_);
v___x_4003_ = l_Lean_Syntax_node1(v___x_3971_, v___x_4000_, v___x_4002_);
v___x_4004_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3975_, v___x_4003_);
v___x_4005_ = l_Lean_Syntax_node3(v___x_3971_, v___x_3989_, v___x_3991_, v___x_3999_, v___x_4004_);
v___x_4006_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3975_, v___x_4005_);
v___x_4007_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3980_, v___x_4006_);
v___x_4008_ = l_Lean_Syntax_node1(v___x_3971_, v___x_3979_, v___x_4007_);
v___x_4009_ = l_Lean_Syntax_node2(v___x_3971_, v___x_3976_, v___x_3978_, v___x_4008_);
v___x_4010_ = l_Lean_Syntax_node2(v___x_3971_, v___x_3975_, v___x_3988_, v___x_4009_);
v___x_4011_ = l_Lean_Syntax_node2(v___x_3971_, v___x_3973_, v___x_3974_, v___x_4010_);
v___x_4012_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4012_, 0, v___x_4011_);
lean_ctor_set(v___x_4012_, 1, v_a_3964_);
return v___x_4012_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___boxed(lean_object* v_x_4013_, lean_object* v_a_4014_, lean_object* v_a_4015_){
_start:
{
lean_object* v_res_4016_; 
v_res_4016_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1(v_x_4013_, v_a_4014_, v_a_4015_);
lean_dec_ref(v_a_4014_);
return v_res_4016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1(lean_object* v_x_4062_, lean_object* v_a_4063_, lean_object* v_a_4064_){
_start:
{
lean_object* v___x_4065_; uint8_t v___x_4066_; 
v___x_4065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringConv___closed__1));
v___x_4066_ = l_Lean_Syntax_isOfKind(v_x_4062_, v___x_4065_);
if (v___x_4066_ == 0)
{
lean_object* v___x_4067_; lean_object* v___x_4068_; 
v___x_4067_ = lean_box(1);
v___x_4068_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4068_, 0, v___x_4067_);
lean_ctor_set(v___x_4068_, 1, v_a_4064_);
return v___x_4068_;
}
else
{
lean_object* v_ref_4069_; uint8_t v___x_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; lean_object* v___x_4073_; lean_object* v___x_4074_; lean_object* v___x_4075_; lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v___x_4078_; lean_object* v___x_4079_; lean_object* v___x_4080_; lean_object* v___x_4081_; lean_object* v___x_4082_; lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v___x_4091_; lean_object* v___x_4092_; lean_object* v___x_4093_; lean_object* v___x_4094_; lean_object* v___x_4095_; lean_object* v___x_4096_; lean_object* v___x_4097_; lean_object* v___x_4098_; lean_object* v___x_4099_; lean_object* v___x_4100_; lean_object* v___x_4101_; lean_object* v___x_4102_; lean_object* v___x_4103_; lean_object* v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; lean_object* v___x_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; lean_object* v___x_4110_; lean_object* v___x_4111_; lean_object* v___x_4112_; lean_object* v___x_4113_; lean_object* v___x_4114_; lean_object* v___x_4115_; lean_object* v___x_4116_; lean_object* v___x_4117_; lean_object* v___x_4118_; lean_object* v___x_4119_; lean_object* v___x_4120_; lean_object* v___x_4121_; lean_object* v___x_4122_; lean_object* v___x_4123_; lean_object* v___x_4124_; 
v_ref_4069_ = lean_ctor_get(v_a_4063_, 5);
v___x_4070_ = 0;
v___x_4071_ = l_Lean_SourceInfo_fromRef(v_ref_4069_, v___x_4070_);
v___x_4072_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0));
v___x_4073_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1));
lean_inc_n(v___x_4071_, 29);
v___x_4074_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4074_, 0, v___x_4071_);
lean_ctor_set(v___x_4074_, 1, v___x_4072_);
v___x_4075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_4076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3));
v___x_4077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4));
v___x_4078_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4078_, 0, v___x_4071_);
lean_ctor_set(v___x_4078_, 1, v___x_4077_);
v___x_4079_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3));
v___x_4080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5));
v___x_4081_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7));
v___x_4082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__8));
v___x_4083_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4083_, 0, v___x_4071_);
lean_ctor_set(v___x_4083_, 1, v___x_4082_);
v___x_4084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__9));
v___x_4085_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4085_, 0, v___x_4071_);
lean_ctor_set(v___x_4085_, 1, v___x_4084_);
v___x_4086_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6));
v___x_4087_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8));
v___x_4088_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__10));
v___x_4089_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__11));
v___x_4090_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4090_, 0, v___x_4071_);
lean_ctor_set(v___x_4090_, 1, v___x_4088_);
v___x_4091_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2);
v___x_4092_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4092_, 0, v___x_4071_);
lean_ctor_set(v___x_4092_, 1, v___x_4075_);
lean_ctor_set(v___x_4092_, 2, v___x_4091_);
lean_inc_ref_n(v___x_4092_, 2);
v___x_4093_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4089_, v___x_4090_, v___x_4092_);
v___x_4094_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4075_, v___x_4093_);
v___x_4095_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4087_, v___x_4094_);
v___x_4096_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4086_, v___x_4095_);
v___x_4097_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4075_, v___x_4085_, v___x_4096_);
v___x_4098_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4081_, v___x_4083_, v___x_4097_);
v___x_4099_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4075_, v___x_4098_);
v___x_4100_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4080_, v___x_4099_);
v___x_4101_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4079_, v___x_4100_);
lean_inc_ref(v___x_4078_);
v___x_4102_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4076_, v___x_4078_, v___x_4101_);
v___x_4103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11));
v___x_4104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14));
v___x_4105_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4105_, 0, v___x_4071_);
lean_ctor_set(v___x_4105_, 1, v___x_4104_);
v___x_4106_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv___closed__1));
v___x_4107_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_ringNF___closed__4));
v___x_4108_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4108_, 0, v___x_4071_);
lean_ctor_set(v___x_4108_, 1, v___x_4107_);
v___x_4109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2));
v___x_4110_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4109_, v___x_4092_);
v___x_4111_ = l_Lean_Syntax_node3(v___x_4071_, v___x_4106_, v___x_4108_, v___x_4092_, v___x_4110_);
v___x_4112_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16));
v___x_4113_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__17));
v___x_4114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4114_, 0, v___x_4071_);
lean_ctor_set(v___x_4114_, 1, v___x_4113_);
v___x_4115_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4112_, v___x_4114_);
v___x_4116_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4075_, v___x_4115_);
v___x_4117_ = l_Lean_Syntax_node3(v___x_4071_, v___x_4103_, v___x_4105_, v___x_4111_, v___x_4116_);
v___x_4118_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4075_, v___x_4117_);
v___x_4119_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4080_, v___x_4118_);
v___x_4120_ = l_Lean_Syntax_node1(v___x_4071_, v___x_4079_, v___x_4119_);
v___x_4121_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4076_, v___x_4078_, v___x_4120_);
v___x_4122_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4075_, v___x_4102_, v___x_4121_);
v___x_4123_ = l_Lean_Syntax_node2(v___x_4071_, v___x_4073_, v___x_4074_, v___x_4122_);
v___x_4124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4124_, 0, v___x_4123_);
lean_ctor_set(v___x_4124_, 1, v_a_4064_);
return v___x_4124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___boxed(lean_object* v_x_4125_, lean_object* v_a_4126_, lean_object* v_a_4127_){
_start:
{
lean_object* v_res_4128_; 
v_res_4128_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1(v_x_4125_, v_a_4126_, v_a_4127_);
lean_dec_ref(v_a_4126_);
return v_res_4128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing_x21__1(lean_object* v_x_4140_, lean_object* v_a_4141_, lean_object* v_a_4142_){
_start:
{
lean_object* v___x_4143_; uint8_t v___x_4144_; 
v___x_4143_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_convRing_x21___closed__1));
v___x_4144_ = l_Lean_Syntax_isOfKind(v_x_4140_, v___x_4143_);
if (v___x_4144_ == 0)
{
lean_object* v___x_4145_; lean_object* v___x_4146_; 
v___x_4145_ = lean_box(1);
v___x_4146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4146_, 0, v___x_4145_);
lean_ctor_set(v___x_4146_, 1, v_a_4142_);
return v___x_4146_;
}
else
{
lean_object* v_ref_4147_; uint8_t v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; lean_object* v___x_4152_; lean_object* v___x_4153_; lean_object* v___x_4154_; lean_object* v___x_4155_; lean_object* v___x_4156_; lean_object* v___x_4157_; lean_object* v___x_4158_; lean_object* v___x_4159_; lean_object* v___x_4160_; lean_object* v___x_4161_; lean_object* v___x_4162_; lean_object* v___x_4163_; lean_object* v___x_4164_; lean_object* v___x_4165_; lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_4168_; lean_object* v___x_4169_; lean_object* v___x_4170_; lean_object* v___x_4171_; lean_object* v___x_4172_; lean_object* v___x_4173_; lean_object* v___x_4174_; lean_object* v___x_4175_; lean_object* v___x_4176_; lean_object* v___x_4177_; lean_object* v___x_4178_; lean_object* v___x_4179_; lean_object* v___x_4180_; lean_object* v___x_4181_; lean_object* v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; lean_object* v___x_4191_; lean_object* v___x_4192_; lean_object* v___x_4193_; lean_object* v___x_4194_; lean_object* v___x_4195_; lean_object* v___x_4196_; lean_object* v___x_4197_; lean_object* v___x_4198_; lean_object* v___x_4199_; lean_object* v___x_4200_; lean_object* v___x_4201_; lean_object* v___x_4202_; 
v_ref_4147_ = lean_ctor_get(v_a_4141_, 5);
v___x_4148_ = 0;
v___x_4149_ = l_Lean_SourceInfo_fromRef(v_ref_4147_, v___x_4148_);
v___x_4150_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__0));
v___x_4151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__1));
lean_inc_n(v___x_4149_, 29);
v___x_4152_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4152_, 0, v___x_4149_);
lean_ctor_set(v___x_4152_, 1, v___x_4150_);
v___x_4153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__1));
v___x_4154_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__3));
v___x_4155_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__4));
v___x_4156_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4156_, 0, v___x_4149_);
lean_ctor_set(v___x_4156_, 1, v___x_4155_);
v___x_4157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__3));
v___x_4158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__5));
v___x_4159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__7));
v___x_4160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__8));
v___x_4161_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4161_, 0, v___x_4149_);
lean_ctor_set(v___x_4161_, 1, v___x_4160_);
v___x_4162_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__9));
v___x_4163_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4163_, 0, v___x_4149_);
lean_ctor_set(v___x_4163_, 1, v___x_4162_);
v___x_4164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__6));
v___x_4165_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__8));
v___x_4166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__1));
v___x_4167_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__2));
v___x_4168_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4168_, 0, v___x_4149_);
lean_ctor_set(v___x_4168_, 1, v___x_4167_);
v___x_4169_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4166_, v___x_4168_);
v___x_4170_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4153_, v___x_4169_);
v___x_4171_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4165_, v___x_4170_);
v___x_4172_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4164_, v___x_4171_);
v___x_4173_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4153_, v___x_4163_, v___x_4172_);
v___x_4174_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4159_, v___x_4161_, v___x_4173_);
v___x_4175_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4153_, v___x_4174_);
v___x_4176_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4158_, v___x_4175_);
v___x_4177_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4157_, v___x_4176_);
lean_inc_ref(v___x_4156_);
v___x_4178_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4154_, v___x_4156_, v___x_4177_);
v___x_4179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ringConv__1___closed__11));
v___x_4180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__14));
v___x_4181_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4181_, 0, v___x_4149_);
lean_ctor_set(v___x_4181_, 1, v___x_4180_);
v___x_4182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21___00__closed__1));
v___x_4183_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21_____00__closed__2));
v___x_4184_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4184_, 0, v___x_4149_);
lean_ctor_set(v___x_4184_, 1, v___x_4183_);
v___x_4185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF_elabRingNFConv___closed__2));
v___x_4186_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2, &lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing__nf_x21______1___closed__2);
v___x_4187_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4187_, 0, v___x_4149_);
lean_ctor_set(v___x_4187_, 1, v___x_4153_);
lean_ctor_set(v___x_4187_, 2, v___x_4186_);
v___x_4188_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4185_, v___x_4187_);
v___x_4189_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4182_, v___x_4184_, v___x_4188_);
v___x_4190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__ring__1___closed__16));
v___x_4191_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__tacticRing_x21__1___closed__3));
v___x_4192_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4192_, 0, v___x_4149_);
lean_ctor_set(v___x_4192_, 1, v___x_4191_);
v___x_4193_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4190_, v___x_4192_);
v___x_4194_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4153_, v___x_4193_);
v___x_4195_ = l_Lean_Syntax_node3(v___x_4149_, v___x_4179_, v___x_4181_, v___x_4189_, v___x_4194_);
v___x_4196_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4153_, v___x_4195_);
v___x_4197_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4158_, v___x_4196_);
v___x_4198_ = l_Lean_Syntax_node1(v___x_4149_, v___x_4157_, v___x_4197_);
v___x_4199_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4154_, v___x_4156_, v___x_4198_);
v___x_4200_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4153_, v___x_4178_, v___x_4199_);
v___x_4201_ = l_Lean_Syntax_node2(v___x_4149_, v___x_4151_, v___x_4152_, v___x_4200_);
v___x_4202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4202_, 0, v___x_4201_);
lean_ctor_set(v___x_4202_, 1, v_a_4142_);
return v___x_4202_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing_x21__1___boxed(lean_object* v_x_4203_, lean_object* v_a_4204_, lean_object* v_a_4205_){
_start:
{
lean_object* v_res_4206_; 
v_res_4206_ = lp_mathlib_Mathlib_Tactic_RingNF___aux__Mathlib__Tactic__Ring__RingNF______macroRules__Mathlib__Tactic__RingNF__convRing_x21__1(v_x_4203_, v_a_4204_, v_a_4205_);
lean_dec_ref(v_a_4204_);
return v_res_4206_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode_default = _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode_default();
lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode = _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedRingMode();
lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default = _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig_default);
lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig = _init_lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_instInhabitedConfig);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermRingMode);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalTermBehaviorIfUnchanged);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprBehaviorIfUnchanged);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprRingMode);
lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1 = _init_lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_instEvalExprConfig__1);
res = lp_mathlib___private_Mathlib_Tactic_Ring_RingNF_0__Mathlib_Tactic_RingNF_initFn_00___x40_Mathlib_Tactic_Ring_RingNF_1278461645____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_RingNF_ringNF = _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNF();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_ringNF);
lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21____ = _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing__nf_x21____);
lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv = _init_lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_ringNFConv);
lp_mathlib_Mathlib_Tactic_RingNF_ring1NF = _init_lp_mathlib_Mathlib_Tactic_RingNF_ring1NF();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_ring1NF);
lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21__ = _init_lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_tacticRing1__nf_x21__);
lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21__ = _init_lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_RingNF_convRing__nf_x21__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Ring_RingNF(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Ring_RingNF(builtin);
}
#ifdef __cplusplus
}
#endif
