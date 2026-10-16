// Lean compiler output
// Module: Mathlib.Tactic.Push
// Imports: public import Init public meta import Init public meta import Lean.Elab.ConfigEval public meta import Lean.Elab.Tactic.Conv.Simp public import Mathlib.Basic.Logic.Basic public import Mathlib.Tactic.Conv public import Mathlib.Tactic.Push.Attr public import Mathlib.Util.AtLocation public import Lean.Elab.ConfigEval
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_expandMacros(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_expandMacroImpl_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
lean_object* l_Lean_mkPrivateName(lean_object*, lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_privateToUserName(lean_object*);
lean_object* l_Lean_ResolveName_resolveNamespace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Environment_header(lean_object*);
extern lean_object* l_Lean_instInhabitedEffectiveImport_default;
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instHashableExtraModUse_hash___boxed(lean_object*);
lean_object* l_Lean_instBEqExtraModUse_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_extraModUses;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableExtraModUse_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_instBEqExtraModUse_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Std_HashMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
extern lean_object* l_Lean_indirectModUseExt;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
uint8_t l_Lean_isMarkedMeta(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_resolveId_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_Head_ofExpr_x3f(lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_Push_instBEqHead_beq(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Push_pushExt;
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_rewrite_x3f(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* l_Lean_mkNot(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOr(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_Head_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_expandLocation(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Elab_Tactic_tacticToDischarge___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_tryTheoremWithExtraArgs_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Push_pullExt;
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatchWithExtra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_getLhs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_applySimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_format(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Std_Format_join(lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_Nat_reprFast(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_discharger;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Elab_Command_runTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "push_neg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__1_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "use_distrib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__1_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__1_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(191, 76, 189, 91, 207, 132, 16, 122)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__1_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(223, 62, 84, 15, 184, 4, 150, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__3_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Set `distrib` to true in `push_neg` and related tactics."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__3_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__3_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__4_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__3_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__4_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__4_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Push"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(103, 224, 1, 132, 229, 106, 48, 221)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__1_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(151, 155, 142, 7, 136, 38, 28, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg_use__distrib;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 44, 53, 159, 128, 198, 125, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "distrib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 44, 53, 159, 128, 198, 125, 242)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(198, 37, 187, 195, 69, 120, 82, 243)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_mkSimpStep(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__2_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not_and_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__4_value),LEAN_SCALAR_PTR_LITERAL(181, 83, 236, 178, 76, 99, 47, 11)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "not_and_or_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__7_value),LEAN_SCALAR_PTR_LITERAL(156, 147, 193, 67, 13, 91, 215, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__10_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "not_forall_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__12_value),LEAN_SCALAR_PTR_LITERAL(127, 163, 137, 27, 184, 219, 49, 1)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 0, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "push"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__3_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__3_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__5_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__7_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Push_isUnderscore(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 150, 238, 148, 228, 221, 116, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__0 = (const lean_object*)&lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__0_value;
static const lean_ctor_object lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__1 = (const lean_object*)&lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqExtraModUse_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__0_value;
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableExtraModUse_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__7 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__8 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__8_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " extra mod use "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__9 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " of "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__11 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recording "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__15 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__17 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "regular"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__19 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__19_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__20 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__20_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__21 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__21_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__22 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2;
static const lean_array_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__3 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 158, .m_capacity = 158, .m_length = 157, .m_data = "maximum recursion depth has been reached\nuse `set_option maxRecDepth <num>` to increase limit\nuse `set_option diagnostics true` to get diagnostic information"};
static const lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "binop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(47, 253, 231, 222, 243, 42, 73, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "binop_lazy"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__5_value),LEAN_SCALAR_PTR_LITERAL(35, 222, 106, 2, 24, 145, 254, 163)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "leftact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__7_value),LEAN_SCALAR_PTR_LITERAL(235, 190, 22, 58, 115, 190, 112, 107)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rightact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__9_value),LEAN_SCALAR_PTR_LITERAL(147, 6, 66, 125, 81, 188, 156, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "binrel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__11_value),LEAN_SCALAR_PTR_LITERAL(81, 238, 75, 93, 70, 164, 233, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "binrel_no_prop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__13_value),LEAN_SCALAR_PTR_LITERAL(90, 122, 90, 92, 171, 187, 176, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "unop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__15_value),LEAN_SCALAR_PTR_LITERAL(68, 253, 109, 39, 185, 175, 169, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "forall"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__0_value),LEAN_SCALAR_PTR_LITERAL(195, 142, 115, 15, 55, 103, 31, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Could not resolve `push` argument `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 95, .m_capacity = 95, .m_length = 93, .m_data = "`. Expected either a constant, e.g. `push Not`, or notation with underscores, e.g. `push ¬ _`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_push___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "push "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_push___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "pushStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 92, 154, 179, 191, 15, 216, 138)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__13_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStx;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(103, 224, 1, 132, 229, 106, 48, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__0_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push__neg;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 298, .m_capacity = 298, .m_length = 297, .m_data = "`push_neg` has been deprecated. Prefer using `push Not` instead.\nIf you'd rather continue using `push_neg` in your project, you can implement it as follows:\n```\nopen Lean.Parser.Tactic in\nmacro \"push_neg\" cfg:optConfig loc:(location)\? : tactic =>\n  `(tactic| push $cfg:optConfig Not $[$loc]\?)\n```\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "pull"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__0_value),LEAN_SCALAR_PTR_LITERAL(48, 248, 114, 168, 40, 148, 90, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pull___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pull___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pull___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pull___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pull___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pull___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pull___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pull;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pushFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pushFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pullFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_pullFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "convPush_____"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 171, 32, 67, 230, 15, 23, 107)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush__________;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "convPush_neg_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 254, 82, 107, 50, 19, 65, 125)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_convPush__neg__;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 267, .m_capacity = 267, .m_length = 266, .m_data = "`push_neg` has been deprecated. Prefer using `push Not` instead.\nIf you'd rather continue using `push_neg` in your project, you can implement it as follows:\n```\nopen Lean.Parser.Tactic in\nmacro \"push_neg\" cfg:optConfig : conv => `(conv| push $cfg:optConfig Not)\n```\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "pushCommand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 116, 253, 237, 51, 72, 199, 132)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#push"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__6_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCommand;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "command#conv_=>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(254, 184, 104, 200, 83, 192, 77, 132)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__4_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "convPull____"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(206, 3, 165, 8, 222, 63, 91, 53)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_convPull________;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "pullCommand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 149, 239, 161, 68, 158, 143, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#pull"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCommand;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pullCommand__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pullCommand__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pushTree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__5_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__7_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(179, 237, 76, 44, 143, 69, 172, 56)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 214, 202, 117, 243, 77, 17, 126)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "#push_discr_tree "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Push_pushTree = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushTree___closed__6_value;
static const lean_string_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__0_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__0_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1_value;
static const lean_string_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = ":perm"};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__0_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__0_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__1 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__1_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__2 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__2_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__2_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__3 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__3_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__4 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__4_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__5 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__5_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__6 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__6_value;
static lean_once_cell_t lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7;
static lean_once_cell_t lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__5_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__9 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__9_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__6_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__10 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2(lean_object*);
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__11_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__0_value;
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__1_value;
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__1_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__5 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__5_value;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__2_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__6 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "node"};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__17_value)}};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__3_value)}};
static const lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "DiscrTree branch for "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "There are no `push` theorems for `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__4_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__8_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_58_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4__spec__0(v___x_55_, v___x_56_, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4____boxed(lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_();
return v_res_60_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_box(0);
v___x_62_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_63_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v___x_61_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_65_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_66_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object* v_msgData_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v___x_89_; lean_object* v_env_90_; lean_object* v___x_91_; lean_object* v_mctx_92_; lean_object* v_lctx_93_; lean_object* v_options_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_89_ = lean_st_ref_get(v___y_87_);
v_env_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc_ref(v_env_90_);
lean_dec(v___x_89_);
v___x_91_ = lean_st_ref_get(v___y_85_);
v_mctx_92_ = lean_ctor_get(v___x_91_, 0);
lean_inc_ref(v_mctx_92_);
lean_dec(v___x_91_);
v_lctx_93_ = lean_ctor_get(v___y_84_, 2);
v_options_94_ = lean_ctor_get(v___y_86_, 2);
lean_inc_ref(v_options_94_);
lean_inc_ref(v_lctx_93_);
v___x_95_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_95_, 0, v_env_90_);
lean_ctor_set(v___x_95_, 1, v_mctx_92_);
lean_ctor_set(v___x_95_, 2, v_lctx_93_);
lean_ctor_set(v___x_95_, 3, v_options_94_);
v___x_96_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_msgData_83_);
v___x_97_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msgData_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_){
_start:
{
lean_object* v_ref_111_; lean_object* v___x_112_; lean_object* v_a_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_121_; 
v_ref_111_ = lean_ctor_get(v___y_108_, 5);
v___x_112_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_);
v_a_113_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_121_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_121_ == 0)
{
v___x_115_ = v___x_112_;
v_isShared_116_ = v_isSharedCheck_121_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_a_113_);
lean_dec(v___x_112_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_121_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v___x_117_; lean_object* v___x_119_; 
lean_inc(v_ref_111_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v_ref_111_);
lean_ctor_set(v___x_117_, 1, v_a_113_);
if (v_isShared_116_ == 0)
{
lean_ctor_set_tag(v___x_115_, 1);
lean_ctor_set(v___x_115_, 0, v___x_117_);
v___x_119_ = v___x_115_;
goto v_reusejp_118_;
}
else
{
lean_object* v_reuseFailAlloc_120_; 
v_reuseFailAlloc_120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_120_, 0, v___x_117_);
v___x_119_ = v_reuseFailAlloc_120_;
goto v_reusejp_118_;
}
v_reusejp_118_:
{
return v___x_119_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
return v_res_128_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_132_ = l_Lean_stringToMessageData(v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_133_, lean_object* v_args_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_162_ = lean_string_dec_eq(v_ctor_133_, v___x_161_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_163_;
}
else
{
lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_164_ = lean_array_get_size(v_args_134_);
v___x_165_ = lean_unsigned_to_nat(1u);
v___x_166_ = lean_nat_dec_eq(v___x_164_, v___x_165_);
if (v___x_166_ == 0)
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_176_; 
v___x_167_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_168_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_167_, v___y_135_, v___y_136_, v___y_137_, v___y_138_);
v_a_169_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_176_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_176_ == 0)
{
v___x_171_ = v___x_168_;
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_dec(v___x_168_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_172_ == 0)
{
v___x_174_ = v___x_171_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_a_169_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
else
{
goto v___jp_140_;
}
}
v___jp_140_:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_141_ = l_Lean_instInhabitedExpr;
v___x_142_ = lean_unsigned_to_nat(0u);
v___x_143_ = lean_array_get_borrowed(v___x_141_, v_args_134_, v___x_142_);
lean_inc(v___x_143_);
v___x_144_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_143_, v___y_135_, v___y_136_, v___y_137_, v___y_138_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v_a_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_152_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_152_ == 0)
{
v___x_147_ = v___x_144_;
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_a_145_);
lean_dec(v___x_144_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_150_; 
if (v_isShared_148_ == 0)
{
v___x_150_ = v___x_147_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_a_145_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
else
{
lean_object* v_a_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
v_a_153_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_160_ == 0)
{
v___x_155_ = v___x_144_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_a_153_);
lean_dec(v___x_144_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_177_, lean_object* v_args_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___lam__0(v_ctor_177_, v_args_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec_ref(v_args_178_);
lean_dec_ref(v_ctor_177_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr(lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_){
_start:
{
lean_object* v___f_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___f_198_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__0));
v___x_199_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2));
v___x_200_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_199_, v___f_198_, v_a_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_, lean_object* v_a_205_, lean_object* v_a_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr(v_a_201_, v_a_202_, v_a_203_, v_a_204_, v_a_205_);
lean_dec(v_a_205_);
lean_dec_ref(v_a_204_);
lean_dec(v_a_203_);
lean_dec_ref(v_a_202_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_208_, lean_object* v_msg_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_216_, lean_object* v_msg_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_216_, v_msg_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
return v_res_223_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_225_ = lean_box(0);
v___x_226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2));
v___x_227_ = l_Lean_Expr_const___override(v___x_226_, v___x_225_);
return v___x_227_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1);
v___x_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
return v___x_229_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_230_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2);
v___x_231_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__0));
v___x_232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v___x_230_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig(void){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__3);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_234_ = lean_box(0);
v___x_235_ = l_Lean_Elab_abortTermExceptionId;
v___x_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
lean_ctor_set(v___x_236_, 1, v___x_234_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg(){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0);
v___x_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object* v___y_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg();
return v_res_241_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object* v_opts_242_, lean_object* v_opt_243_){
_start:
{
lean_object* v_name_244_; lean_object* v_defValue_245_; lean_object* v_map_246_; lean_object* v___x_247_; 
v_name_244_ = lean_ctor_get(v_opt_243_, 0);
v_defValue_245_ = lean_ctor_get(v_opt_243_, 1);
v_map_246_ = lean_ctor_get(v_opts_242_, 0);
v___x_247_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_246_, v_name_244_);
if (lean_obj_tag(v___x_247_) == 0)
{
uint8_t v___x_248_; 
v___x_248_ = lean_unbox(v_defValue_245_);
return v___x_248_;
}
else
{
lean_object* v_val_249_; 
v_val_249_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_val_249_);
lean_dec_ref_known(v___x_247_, 1);
if (lean_obj_tag(v_val_249_) == 1)
{
uint8_t v_v_250_; 
v_v_250_ = lean_ctor_get_uint8(v_val_249_, 0);
lean_dec_ref_known(v_val_249_, 0);
return v_v_250_;
}
else
{
uint8_t v___x_251_; 
lean_dec(v_val_249_);
v___x_251_ = lean_unbox(v_defValue_245_);
return v___x_251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_opts_252_, lean_object* v_opt_253_){
_start:
{
uint8_t v_res_254_; lean_object* v_r_255_; 
v_res_254_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_opts_252_, v_opt_253_);
lean_dec_ref(v_opt_253_);
lean_dec_ref(v_opts_252_);
v_r_255_ = lean_box(v_res_254_);
return v_r_255_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = lean_box(1);
v___x_257_ = l_Lean_MessageData_ofFormat(v___x_256_);
return v___x_257_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2));
v___x_262_ = l_Lean_MessageData_ofFormat(v___x_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object* v_x_263_, lean_object* v_x_264_){
_start:
{
if (lean_obj_tag(v_x_264_) == 0)
{
return v_x_263_;
}
else
{
lean_object* v_head_265_; lean_object* v_tail_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_288_; 
v_head_265_ = lean_ctor_get(v_x_264_, 0);
v_tail_266_ = lean_ctor_get(v_x_264_, 1);
v_isSharedCheck_288_ = !lean_is_exclusive(v_x_264_);
if (v_isSharedCheck_288_ == 0)
{
v___x_268_ = v_x_264_;
v_isShared_269_ = v_isSharedCheck_288_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_tail_266_);
lean_inc(v_head_265_);
lean_dec(v_x_264_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_288_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v_before_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_286_; 
v_before_270_ = lean_ctor_get(v_head_265_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v_head_265_);
if (v_isSharedCheck_286_ == 0)
{
lean_object* v_unused_287_; 
v_unused_287_ = lean_ctor_get(v_head_265_, 1);
lean_dec(v_unused_287_);
v___x_272_ = v_head_265_;
v_isShared_273_ = v_isSharedCheck_286_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_before_270_);
lean_dec(v_head_265_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_286_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_274_; lean_object* v___x_276_; 
v___x_274_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_273_ == 0)
{
lean_ctor_set_tag(v___x_272_, 7);
lean_ctor_set(v___x_272_, 1, v___x_274_);
lean_ctor_set(v___x_272_, 0, v_x_263_);
v___x_276_ = v___x_272_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_x_263_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v___x_274_);
v___x_276_ = v_reuseFailAlloc_285_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
lean_object* v___x_277_; lean_object* v___x_279_; 
v___x_277_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3);
if (v_isShared_269_ == 0)
{
lean_ctor_set_tag(v___x_268_, 7);
lean_ctor_set(v___x_268_, 1, v___x_277_);
lean_ctor_set(v___x_268_, 0, v___x_276_);
v___x_279_ = v___x_268_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_284_, 1, v___x_277_);
v___x_279_ = v_reuseFailAlloc_284_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_280_ = l_Lean_MessageData_ofSyntax(v_before_270_);
v___x_281_ = l_Lean_indentD(v___x_280_);
v___x_282_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_282_, 0, v___x_279_);
lean_ctor_set(v___x_282_, 1, v___x_281_);
v_x_263_ = v___x_282_;
v_x_264_ = v_tail_266_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_292_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1));
v___x_293_ = l_Lean_MessageData_ofFormat(v___x_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object* v_msgData_294_, lean_object* v_macroStack_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_options_298_; lean_object* v___x_299_; uint8_t v___x_300_; 
v_options_298_ = lean_ctor_get(v___y_296_, 2);
v___x_299_ = l_Lean_Elab_pp_macroStack;
v___x_300_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_options_298_, v___x_299_);
if (v___x_300_ == 0)
{
lean_object* v___x_301_; 
lean_dec(v_macroStack_295_);
v___x_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_301_, 0, v_msgData_294_);
return v___x_301_;
}
else
{
if (lean_obj_tag(v_macroStack_295_) == 0)
{
lean_object* v___x_302_; 
v___x_302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_302_, 0, v_msgData_294_);
return v___x_302_;
}
else
{
lean_object* v_head_303_; lean_object* v_after_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_319_; 
v_head_303_ = lean_ctor_get(v_macroStack_295_, 0);
lean_inc(v_head_303_);
v_after_304_ = lean_ctor_get(v_head_303_, 1);
v_isSharedCheck_319_ = !lean_is_exclusive(v_head_303_);
if (v_isSharedCheck_319_ == 0)
{
lean_object* v_unused_320_; 
v_unused_320_ = lean_ctor_get(v_head_303_, 0);
lean_dec(v_unused_320_);
v___x_306_ = v_head_303_;
v_isShared_307_ = v_isSharedCheck_319_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_after_304_);
lean_dec(v_head_303_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_319_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___x_308_; lean_object* v___x_310_; 
v___x_308_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_307_ == 0)
{
lean_ctor_set_tag(v___x_306_, 7);
lean_ctor_set(v___x_306_, 1, v___x_308_);
lean_ctor_set(v___x_306_, 0, v_msgData_294_);
v___x_310_ = v___x_306_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v_msgData_294_);
lean_ctor_set(v_reuseFailAlloc_318_, 1, v___x_308_);
v___x_310_ = v_reuseFailAlloc_318_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v_msgData_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_311_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2);
v___x_312_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_310_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
v___x_313_ = l_Lean_MessageData_ofSyntax(v_after_304_);
v___x_314_ = l_Lean_indentD(v___x_313_);
v_msgData_315_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_315_, 0, v___x_312_);
lean_ctor_set(v_msgData_315_, 1, v___x_314_);
v___x_316_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(v_msgData_315_, v_macroStack_295_);
v___x_317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
return v___x_317_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_321_, lean_object* v_macroStack_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_321_, v_macroStack_322_, v___y_323_);
lean_dec_ref(v___y_323_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object* v_msg_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
lean_object* v_ref_334_; lean_object* v___x_335_; lean_object* v_a_336_; lean_object* v_macroStack_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_348_; 
v_ref_334_ = lean_ctor_get(v___y_331_, 5);
v___x_335_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_326_, v___y_329_, v___y_330_, v___y_331_, v___y_332_);
v_a_336_ = lean_ctor_get(v___x_335_, 0);
lean_inc(v_a_336_);
lean_dec_ref(v___x_335_);
v_macroStack_337_ = lean_ctor_get(v___y_327_, 1);
v___x_338_ = l_Lean_Elab_getBetterRef(v_ref_334_, v_macroStack_337_);
lean_inc(v_macroStack_337_);
v___x_339_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_a_336_, v_macroStack_337_, v___y_331_);
v_a_340_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_348_ == 0)
{
v___x_342_ = v___x_339_;
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_339_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_344_; lean_object* v___x_346_; 
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_338_);
lean_ctor_set(v___x_344_, 1, v_a_340_);
if (v_isShared_343_ == 0)
{
lean_ctor_set_tag(v___x_342_, 1);
lean_ctor_set(v___x_342_, 0, v___x_344_);
v___x_346_ = v___x_342_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_344_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object* v_msg_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object* v_e_358_, lean_object* v___y_359_){
_start:
{
uint8_t v___x_361_; 
v___x_361_ = l_Lean_Expr_hasMVar(v_e_358_);
if (v___x_361_ == 0)
{
lean_object* v___x_362_; 
v___x_362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_362_, 0, v_e_358_);
return v___x_362_;
}
else
{
lean_object* v___x_363_; lean_object* v_mctx_364_; lean_object* v___x_365_; lean_object* v_fst_366_; lean_object* v_snd_367_; lean_object* v___x_368_; lean_object* v_cache_369_; lean_object* v_zetaDeltaFVarIds_370_; lean_object* v_postponed_371_; lean_object* v_diag_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_381_; 
v___x_363_ = lean_st_ref_get(v___y_359_);
v_mctx_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc_ref(v_mctx_364_);
lean_dec(v___x_363_);
v___x_365_ = l_Lean_instantiateMVarsCore(v_mctx_364_, v_e_358_);
v_fst_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_fst_366_);
v_snd_367_ = lean_ctor_get(v___x_365_, 1);
lean_inc(v_snd_367_);
lean_dec_ref(v___x_365_);
v___x_368_ = lean_st_ref_take(v___y_359_);
v_cache_369_ = lean_ctor_get(v___x_368_, 1);
v_zetaDeltaFVarIds_370_ = lean_ctor_get(v___x_368_, 2);
v_postponed_371_ = lean_ctor_get(v___x_368_, 3);
v_diag_372_ = lean_ctor_get(v___x_368_, 4);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_381_ == 0)
{
lean_object* v_unused_382_; 
v_unused_382_ = lean_ctor_get(v___x_368_, 0);
lean_dec(v_unused_382_);
v___x_374_ = v___x_368_;
v_isShared_375_ = v_isSharedCheck_381_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_diag_372_);
lean_inc(v_postponed_371_);
lean_inc(v_zetaDeltaFVarIds_370_);
lean_inc(v_cache_369_);
lean_dec(v___x_368_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_381_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v___x_377_; 
if (v_isShared_375_ == 0)
{
lean_ctor_set(v___x_374_, 0, v_snd_367_);
v___x_377_ = v___x_374_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_snd_367_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v_cache_369_);
lean_ctor_set(v_reuseFailAlloc_380_, 2, v_zetaDeltaFVarIds_370_);
lean_ctor_set(v_reuseFailAlloc_380_, 3, v_postponed_371_);
lean_ctor_set(v_reuseFailAlloc_380_, 4, v_diag_372_);
v___x_377_ = v_reuseFailAlloc_380_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = lean_st_ref_set(v___y_359_, v___x_377_);
v___x_379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_379_, 0, v_fst_366_);
return v___x_379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object* v_e_383_, lean_object* v___y_384_, lean_object* v___y_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_383_, v___y_384_);
lean_dec(v___y_384_);
return v_res_386_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1(void){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_388_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__0));
v___x_389_ = l_Lean_stringToMessageData(v___x_388_);
return v___x_389_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__1);
v___x_391_ = l_Lean_MessageData_ofExpr(v___x_390_);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3(void){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_392_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__2);
v___x_393_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__1);
v___x_394_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v___x_392_);
return v___x_394_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5(void){
_start:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__4));
v___x_397_ = l_Lean_stringToMessageData(v___x_396_);
return v___x_397_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_398_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5);
v___x_399_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__3);
v___x_400_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v___x_398_);
return v___x_400_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8(void){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_402_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__7));
v___x_403_ = l_Lean_stringToMessageData(v___x_402_);
return v___x_403_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_405_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__9));
v___x_406_ = l_Lean_stringToMessageData(v___x_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0(lean_object* v_stx_407_, lean_object* v_a_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
lean_object* v_ty_x3f_415_; uint8_t v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v_fileName_421_; lean_object* v_fileMap_422_; lean_object* v_options_423_; lean_object* v_currRecDepth_424_; lean_object* v_maxRecDepth_425_; lean_object* v_ref_426_; lean_object* v_currNamespace_427_; lean_object* v_openDecls_428_; lean_object* v_initHeartbeats_429_; lean_object* v_maxHeartbeats_430_; lean_object* v_quotContext_431_; lean_object* v_currMacroScope_432_; uint8_t v_diag_433_; lean_object* v_cancelTk_x3f_434_; uint8_t v_suppressElabErrors_435_; lean_object* v_inheritedTraceOptions_436_; uint8_t v___x_437_; lean_object* v_ref_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v_ty_x3f_415_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig___closed__2);
v___x_416_ = 1;
v___x_417_ = lean_box(0);
v___x_418_ = lean_box(v___x_416_);
v___x_419_ = lean_box(v___x_416_);
lean_inc(v_stx_407_);
v___x_420_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_420_, 0, v_stx_407_);
lean_closure_set(v___x_420_, 1, v_ty_x3f_415_);
lean_closure_set(v___x_420_, 2, v___x_418_);
lean_closure_set(v___x_420_, 3, v___x_419_);
lean_closure_set(v___x_420_, 4, v___x_417_);
v_fileName_421_ = lean_ctor_get(v_a_412_, 0);
v_fileMap_422_ = lean_ctor_get(v_a_412_, 1);
v_options_423_ = lean_ctor_get(v_a_412_, 2);
v_currRecDepth_424_ = lean_ctor_get(v_a_412_, 3);
v_maxRecDepth_425_ = lean_ctor_get(v_a_412_, 4);
v_ref_426_ = lean_ctor_get(v_a_412_, 5);
v_currNamespace_427_ = lean_ctor_get(v_a_412_, 6);
v_openDecls_428_ = lean_ctor_get(v_a_412_, 7);
v_initHeartbeats_429_ = lean_ctor_get(v_a_412_, 8);
v_maxHeartbeats_430_ = lean_ctor_get(v_a_412_, 9);
v_quotContext_431_ = lean_ctor_get(v_a_412_, 10);
v_currMacroScope_432_ = lean_ctor_get(v_a_412_, 11);
v_diag_433_ = lean_ctor_get_uint8(v_a_412_, sizeof(void*)*14);
v_cancelTk_x3f_434_ = lean_ctor_get(v_a_412_, 12);
v_suppressElabErrors_435_ = lean_ctor_get_uint8(v_a_412_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_436_ = lean_ctor_get(v_a_412_, 13);
v___x_437_ = 1;
v_ref_438_ = l_Lean_replaceRef(v_stx_407_, v_ref_426_);
lean_dec(v_stx_407_);
lean_inc_ref(v_inheritedTraceOptions_436_);
lean_inc(v_cancelTk_x3f_434_);
lean_inc(v_currMacroScope_432_);
lean_inc(v_quotContext_431_);
lean_inc(v_maxHeartbeats_430_);
lean_inc(v_initHeartbeats_429_);
lean_inc(v_openDecls_428_);
lean_inc(v_currNamespace_427_);
lean_inc(v_maxRecDepth_425_);
lean_inc(v_currRecDepth_424_);
lean_inc_ref(v_options_423_);
lean_inc_ref(v_fileMap_422_);
lean_inc_ref(v_fileName_421_);
v___x_439_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_439_, 0, v_fileName_421_);
lean_ctor_set(v___x_439_, 1, v_fileMap_422_);
lean_ctor_set(v___x_439_, 2, v_options_423_);
lean_ctor_set(v___x_439_, 3, v_currRecDepth_424_);
lean_ctor_set(v___x_439_, 4, v_maxRecDepth_425_);
lean_ctor_set(v___x_439_, 5, v_ref_438_);
lean_ctor_set(v___x_439_, 6, v_currNamespace_427_);
lean_ctor_set(v___x_439_, 7, v_openDecls_428_);
lean_ctor_set(v___x_439_, 8, v_initHeartbeats_429_);
lean_ctor_set(v___x_439_, 9, v_maxHeartbeats_430_);
lean_ctor_set(v___x_439_, 10, v_quotContext_431_);
lean_ctor_set(v___x_439_, 11, v_currMacroScope_432_);
lean_ctor_set(v___x_439_, 12, v_cancelTk_x3f_434_);
lean_ctor_set(v___x_439_, 13, v_inheritedTraceOptions_436_);
lean_ctor_set_uint8(v___x_439_, sizeof(void*)*14, v_diag_433_);
lean_ctor_set_uint8(v___x_439_, sizeof(void*)*14 + 1, v_suppressElabErrors_435_);
v___x_440_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_420_, v___x_437_, v_a_408_, v_a_409_, v_a_410_, v_a_411_, v___x_439_, v_a_413_);
if (lean_obj_tag(v___x_440_) == 0)
{
lean_object* v_a_441_; lean_object* v___x_442_; lean_object* v_a_443_; lean_object* v___y_445_; lean_object* v___y_446_; lean_object* v___y_447_; lean_object* v___y_448_; lean_object* v___y_449_; lean_object* v___y_450_; lean_object* v___y_451_; lean_object* v___y_452_; lean_object* v___y_453_; uint8_t v___y_454_; lean_object* v___y_471_; lean_object* v___y_472_; lean_object* v___y_473_; lean_object* v___y_474_; lean_object* v___y_475_; lean_object* v___y_476_; lean_object* v___y_483_; lean_object* v___y_484_; lean_object* v___y_485_; lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v___y_488_; lean_object* v___y_520_; lean_object* v___y_521_; lean_object* v___y_522_; lean_object* v___y_523_; lean_object* v___y_524_; lean_object* v___y_525_; uint8_t v___x_538_; 
v_a_441_ = lean_ctor_get(v___x_440_, 0);
lean_inc(v_a_441_);
lean_dec_ref_known(v___x_440_, 1);
v___x_442_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg(v_a_441_, v_a_411_);
v_a_443_ = lean_ctor_get(v___x_442_, 0);
lean_inc(v_a_443_);
lean_dec_ref(v___x_442_);
v___x_538_ = l_Lean_Expr_hasSorry(v_a_443_);
if (v___x_538_ == 0)
{
v___y_483_ = v_a_408_;
v___y_484_ = v_a_409_;
v___y_485_ = v_a_410_;
v___y_486_ = v_a_411_;
v___y_487_ = v___x_439_;
v___y_488_ = v_a_413_;
goto v___jp_482_;
}
else
{
uint8_t v___x_539_; 
v___x_539_ = l_Lean_Expr_hasSyntheticSorry(v_a_443_);
if (v___x_539_ == 0)
{
v___y_520_ = v_a_408_;
v___y_521_ = v_a_409_;
v___y_522_ = v_a_410_;
v___y_523_ = v_a_411_;
v___y_524_ = v___x_439_;
v___y_525_ = v_a_413_;
goto v___jp_519_;
}
else
{
lean_object* v___x_540_; lean_object* v_a_541_; lean_object* v___x_543_; uint8_t v_isShared_544_; uint8_t v_isSharedCheck_548_; 
lean_dec(v_a_443_);
lean_dec_ref_known(v___x_439_, 14);
v___x_540_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_541_ = lean_ctor_get(v___x_540_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v___x_540_);
if (v_isSharedCheck_548_ == 0)
{
v___x_543_ = v___x_540_;
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
else
{
lean_inc(v_a_541_);
lean_dec(v___x_540_);
v___x_543_ = lean_box(0);
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
v_resetjp_542_:
{
lean_object* v___x_546_; 
if (v_isShared_544_ == 0)
{
v___x_546_ = v___x_543_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v_a_541_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
return v___x_546_;
}
}
}
}
v___jp_444_:
{
if (v___y_454_ == 0)
{
if (lean_obj_tag(v___y_449_) == 0)
{
lean_dec_ref_known(v___y_449_, 2);
lean_dec_ref(v___y_445_);
lean_dec(v_a_443_);
return v___y_446_;
}
else
{
lean_object* v_id_455_; lean_object* v___x_457_; uint8_t v_isShared_458_; uint8_t v_isSharedCheck_468_; 
v_id_455_ = lean_ctor_get(v___y_449_, 0);
v_isSharedCheck_468_ = !lean_is_exclusive(v___y_449_);
if (v_isSharedCheck_468_ == 0)
{
lean_object* v_unused_469_; 
v_unused_469_ = lean_ctor_get(v___y_449_, 1);
lean_dec(v_unused_469_);
v___x_457_ = v___y_449_;
v_isShared_458_ = v_isSharedCheck_468_;
goto v_resetjp_456_;
}
else
{
lean_inc(v_id_455_);
lean_dec(v___y_449_);
v___x_457_ = lean_box(0);
v_isShared_458_ = v_isSharedCheck_468_;
goto v_resetjp_456_;
}
v_resetjp_456_:
{
uint8_t v___x_459_; 
v___x_459_ = l_Lean_instBEqInternalExceptionId_beq(v___y_450_, v_id_455_);
lean_dec(v_id_455_);
if (v___x_459_ == 0)
{
lean_del_object(v___x_457_);
lean_dec_ref(v___y_445_);
lean_dec(v_a_443_);
return v___y_446_;
}
else
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_464_; 
lean_dec_ref(v___y_446_);
v___x_460_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__6);
v___x_461_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__8);
v___x_462_ = l_Lean_indentExpr(v_a_443_);
if (v_isShared_458_ == 0)
{
lean_ctor_set_tag(v___x_457_, 7);
lean_ctor_set(v___x_457_, 1, v___x_462_);
lean_ctor_set(v___x_457_, 0, v___x_461_);
v___x_464_ = v___x_457_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v___x_462_);
v___x_464_ = v_reuseFailAlloc_467_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_465_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v___x_460_);
v___x_466_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_465_, v___y_451_, v___y_452_, v___y_448_, v___y_453_, v___y_445_, v___y_447_);
lean_dec_ref(v___y_445_);
return v___x_466_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_449_);
lean_dec_ref(v___y_445_);
lean_dec(v_a_443_);
return v___y_446_;
}
}
v___jp_470_:
{
lean_object* v___x_477_; 
lean_inc(v_a_443_);
v___x_477_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr(v_a_443_, v___y_473_, v___y_474_, v___y_475_, v___y_476_);
if (lean_obj_tag(v___x_477_) == 0)
{
lean_dec_ref(v___y_475_);
lean_dec(v_a_443_);
return v___x_477_;
}
else
{
lean_object* v_a_478_; lean_object* v___x_479_; uint8_t v___x_480_; 
v_a_478_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_a_478_);
v___x_479_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_480_ = l_Lean_Exception_isInterrupt(v_a_478_);
if (v___x_480_ == 0)
{
uint8_t v___x_481_; 
lean_inc(v_a_478_);
v___x_481_ = l_Lean_Exception_isRuntime(v_a_478_);
v___y_445_ = v___y_475_;
v___y_446_ = v___x_477_;
v___y_447_ = v___y_476_;
v___y_448_ = v___y_473_;
v___y_449_ = v_a_478_;
v___y_450_ = v___x_479_;
v___y_451_ = v___y_471_;
v___y_452_ = v___y_472_;
v___y_453_ = v___y_474_;
v___y_454_ = v___x_481_;
goto v___jp_444_;
}
else
{
v___y_445_ = v___y_475_;
v___y_446_ = v___x_477_;
v___y_447_ = v___y_476_;
v___y_448_ = v___y_473_;
v___y_449_ = v_a_478_;
v___y_450_ = v___x_479_;
v___y_451_ = v___y_471_;
v___y_452_ = v___y_472_;
v___y_453_ = v___y_474_;
v___y_454_ = v___x_480_;
goto v___jp_444_;
}
}
}
v___jp_482_:
{
lean_object* v___x_489_; 
lean_inc(v_a_443_);
v___x_489_ = l_Lean_Meta_getMVars(v_a_443_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
if (lean_obj_tag(v___x_489_) == 0)
{
lean_object* v_a_490_; lean_object* v___x_491_; 
v_a_490_ = lean_ctor_get(v___x_489_, 0);
lean_inc(v_a_490_);
lean_dec_ref_known(v___x_489_, 1);
v___x_491_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_490_, v___x_417_, v___y_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
lean_dec(v_a_490_);
if (lean_obj_tag(v___x_491_) == 0)
{
lean_object* v_a_492_; uint8_t v___x_493_; 
v_a_492_ = lean_ctor_get(v___x_491_, 0);
lean_inc(v_a_492_);
lean_dec_ref_known(v___x_491_, 1);
v___x_493_ = lean_unbox(v_a_492_);
lean_dec(v_a_492_);
if (v___x_493_ == 0)
{
v___y_471_ = v___y_483_;
v___y_472_ = v___y_484_;
v___y_473_ = v___y_485_;
v___y_474_ = v___y_486_;
v___y_475_ = v___y_487_;
v___y_476_ = v___y_488_;
goto v___jp_470_;
}
else
{
lean_object* v___x_494_; lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_502_; 
lean_dec_ref(v___y_487_);
lean_dec(v_a_443_);
v___x_494_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_495_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_502_ == 0)
{
v___x_497_ = v___x_494_;
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_494_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_500_; 
if (v_isShared_498_ == 0)
{
v___x_500_ = v___x_497_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_a_495_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
else
{
lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_510_; 
lean_dec_ref(v___y_487_);
lean_dec(v_a_443_);
v_a_503_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_510_ == 0)
{
v___x_505_ = v___x_491_;
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_491_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_508_; 
if (v_isShared_506_ == 0)
{
v___x_508_ = v___x_505_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_a_503_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
else
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_dec_ref(v___y_487_);
lean_dec(v_a_443_);
v_a_511_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_489_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_489_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
v___jp_519_:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v_a_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_537_; 
v___x_526_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__10);
v___x_527_ = l_Lean_indentExpr(v_a_443_);
v___x_528_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_528_, 0, v___x_526_);
lean_ctor_set(v___x_528_, 1, v___x_527_);
v___x_529_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_528_, v___y_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_);
lean_dec_ref(v___y_524_);
v_a_530_ = lean_ctor_get(v___x_529_, 0);
v_isSharedCheck_537_ = !lean_is_exclusive(v___x_529_);
if (v_isSharedCheck_537_ == 0)
{
v___x_532_ = v___x_529_;
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_a_530_);
lean_dec(v___x_529_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v___x_535_; 
if (v_isShared_533_ == 0)
{
v___x_535_ = v___x_532_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_536_; 
v_reuseFailAlloc_536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_536_, 0, v_a_530_);
v___x_535_ = v_reuseFailAlloc_536_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
return v___x_535_;
}
}
}
}
else
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec_ref_known(v___x_439_, 14);
v_a_549_ = lean_ctor_get(v___x_440_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_440_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___x_440_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_549_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0(v_stx_557_, v_a_558_, v_a_559_, v_a_560_, v_a_561_, v_a_562_, v_a_563_);
lean_dec(v_a_563_);
lean_dec_ref(v_a_562_);
lean_dec(v_a_561_);
lean_dec_ref(v_a_560_);
lean_dec(v_a_559_);
lean_dec_ref(v_a_558_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0(uint8_t v_config_576_, lean_object* v_item_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_){
_start:
{
lean_object* v_item_586_; lean_object* v___y_587_; lean_object* v___y_588_; lean_object* v___y_589_; lean_object* v___y_590_; lean_object* v___y_591_; lean_object* v___y_592_; lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_595_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2));
v___x_596_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_577_, v___x_595_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
if (lean_obj_tag(v___x_596_) == 0)
{
uint8_t v___x_597_; 
lean_dec_ref_known(v___x_596_, 1);
v___x_597_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_577_);
if (v___x_597_ == 0)
{
lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; uint8_t v___x_601_; 
v___x_598_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_577_);
lean_inc_ref(v_item_577_);
v___x_599_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_577_);
v___x_600_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__1));
v___x_601_ = lean_string_dec_eq(v___x_598_, v___x_600_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; uint8_t v___x_603_; 
v___x_602_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__2));
v___x_603_ = lean_string_dec_eq(v___x_598_, v___x_602_);
lean_dec_ref(v___x_598_);
if (v___x_603_ == 0)
{
lean_dec_ref(v_item_577_);
v_item_586_ = v___x_599_;
v___y_587_ = v___y_578_;
v___y_588_ = v___y_579_;
v___y_589_ = v___y_580_;
v___y_590_ = v___y_581_;
v___y_591_ = v___y_582_;
v___y_592_ = v___y_583_;
goto v___jp_585_;
}
else
{
lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_604_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__3));
v___x_605_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_577_, v___x_604_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
if (lean_obj_tag(v___x_605_) == 0)
{
uint8_t v___x_606_; 
lean_dec_ref_known(v___x_605_, 1);
v___x_606_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_599_);
if (v___x_606_ == 0)
{
lean_dec_ref(v_item_577_);
v_item_586_ = v___x_599_;
v___y_587_ = v___y_578_;
v___y_588_ = v___y_579_;
v___y_589_ = v___y_580_;
v___y_590_ = v___y_581_;
v___y_591_ = v___y_582_;
v___y_592_ = v___y_583_;
goto v___jp_585_;
}
else
{
lean_object* v___x_607_; 
lean_dec_ref(v___x_599_);
v___x_607_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_577_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
if (lean_obj_tag(v___x_607_) == 0)
{
lean_object* v_a_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_615_; 
v_a_608_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_615_ == 0)
{
v___x_610_ = v___x_607_;
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_a_608_);
lean_dec(v___x_607_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_613_; 
if (v_isShared_611_ == 0)
{
v___x_613_ = v___x_610_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v_a_608_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
}
}
}
else
{
lean_object* v_a_616_; lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_623_; 
v_a_616_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_623_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_623_ == 0)
{
v___x_618_ = v___x_607_;
v_isShared_619_ = v_isSharedCheck_623_;
goto v_resetjp_617_;
}
else
{
lean_inc(v_a_616_);
lean_dec(v___x_607_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_623_;
goto v_resetjp_617_;
}
v_resetjp_617_:
{
lean_object* v___x_621_; 
if (v_isShared_619_ == 0)
{
v___x_621_ = v___x_618_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v_a_616_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
}
}
else
{
lean_object* v_a_624_; lean_object* v___x_626_; uint8_t v_isShared_627_; uint8_t v_isSharedCheck_631_; 
lean_dec_ref(v___x_599_);
lean_dec_ref(v_item_577_);
v_a_624_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_631_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_631_ == 0)
{
v___x_626_ = v___x_605_;
v_isShared_627_ = v_isSharedCheck_631_;
goto v_resetjp_625_;
}
else
{
lean_inc(v_a_624_);
lean_dec(v___x_605_);
v___x_626_ = lean_box(0);
v_isShared_627_ = v_isSharedCheck_631_;
goto v_resetjp_625_;
}
v_resetjp_625_:
{
lean_object* v___x_629_; 
if (v_isShared_627_ == 0)
{
v___x_629_ = v___x_626_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v_a_624_);
v___x_629_ = v_reuseFailAlloc_630_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
return v___x_629_;
}
}
}
}
}
else
{
uint8_t v___x_632_; 
lean_dec_ref(v___x_598_);
v___x_632_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_599_);
if (v___x_632_ == 0)
{
lean_dec_ref(v_item_577_);
v_item_586_ = v___x_599_;
v___y_587_ = v___y_578_;
v___y_588_ = v___y_579_;
v___y_589_ = v___y_580_;
v___y_590_ = v___y_581_;
v___y_591_ = v___y_582_;
v___y_592_ = v___y_583_;
goto v___jp_585_;
}
else
{
lean_object* v_value_633_; lean_object* v___x_634_; 
lean_dec_ref(v___x_599_);
v_value_633_ = lean_ctor_get(v_item_577_, 2);
lean_inc(v_value_633_);
lean_dec_ref(v_item_577_);
v___x_634_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0(v_value_633_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
return v___x_634_;
}
}
}
else
{
v_item_586_ = v_item_577_;
v___y_587_ = v___y_578_;
v___y_588_ = v___y_579_;
v___y_589_ = v___y_580_;
v___y_590_ = v___y_581_;
v___y_591_ = v___y_582_;
v___y_592_ = v___y_583_;
goto v___jp_585_;
}
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_642_; 
lean_dec_ref(v_item_577_);
v_a_635_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_642_ == 0)
{
v___x_637_ = v___x_596_;
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_596_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_640_; 
if (v_isShared_638_ == 0)
{
v___x_640_ = v___x_637_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_641_; 
v_reuseFailAlloc_641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_641_, 0, v_a_635_);
v___x_640_ = v_reuseFailAlloc_641_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
return v___x_640_;
}
}
}
v___jp_585_:
{
lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_593_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___closed__0));
v___x_594_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_586_, v___x_593_, v___y_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_, v___y_592_);
return v___x_594_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_643_, lean_object* v_item_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_){
_start:
{
uint8_t v_config_3993__boxed_652_; lean_object* v_res_653_; 
v_config_3993__boxed_652_ = lean_unbox(v_config_643_);
v_res_653_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___lam__0(v_config_3993__boxed_652_, v_item_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_);
lean_dec(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0(lean_object* v_e_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_656_, v___y_660_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_e_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__0(v_e_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
lean_dec(v___y_671_);
lean_dec_ref(v___y_670_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2(lean_object* v_00_u03b1_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___redArg();
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object* v_00_u03b1_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__2(v_00_u03b1_683_, v___y_684_, v___y_685_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
lean_dec(v___y_689_);
lean_dec_ref(v___y_688_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
lean_dec(v___y_685_);
lean_dec_ref(v___y_684_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1(lean_object* v_00_u03b1_692_, lean_object* v_msg_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_693_, v___y_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_, v___y_699_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object* v_00_u03b1_702_, lean_object* v_msg_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1(v_00_u03b1_702_, v_msg_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
lean_dec(v___y_705_);
lean_dec_ref(v___y_704_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object* v_msgData_712_, lean_object* v_macroStack_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_){
_start:
{
lean_object* v___x_721_; 
v___x_721_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_712_, v_macroStack_713_, v___y_718_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object* v_msgData_722_, lean_object* v_macroStack_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2(v_msgData_722_, v_macroStack_723_, v___y_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
lean_dec(v___y_727_);
lean_dec_ref(v___y_726_);
lean_dec(v___y_725_);
lean_dec_ref(v___y_724_);
return v_res_731_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_732_ = lean_box(0);
v___x_733_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr___closed__2));
v___x_734_ = l_Lean_mkConst(v___x_733_, v___x_732_);
return v___x_734_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_735_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__0);
v___x_736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_736_, 0, v___x_735_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0(uint8_t v_cfg_737_, lean_object* v_cfgItem_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_){
_start:
{
lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_746_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___closed__1);
v___x_747_ = lean_box(v_cfg_737_);
v___x_748_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v___x_747_, v_cfgItem_738_, v___x_746_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0___boxed(lean_object* v_cfg_749_, lean_object* v_cfgItem_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_){
_start:
{
uint8_t v_cfg_boxed_758_; lean_object* v_res_759_; 
v_cfg_boxed_758_ = lean_unbox(v_cfg_749_);
v_res_759_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___lam__0(v_cfg_boxed_758_, v_cfgItem_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v_cfgItem_750_);
return v_res_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object* v_cfg_761_, uint8_t v_init_762_, uint8_t v_logExceptions_763_, lean_object* v_a_764_, lean_object* v_a_765_, lean_object* v_a_766_){
_start:
{
lean_object* v_onErr_768_; lean_object* v_eval_769_; 
v_onErr_768_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___closed__0));
v_eval_769_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem___closed__0));
if (v_logExceptions_763_ == 0)
{
lean_object* v___x_770_; lean_object* v___x_771_; 
v___x_770_ = lean_box(v_init_762_);
v___x_771_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_769_, v___x_770_, v_cfg_761_, v_onErr_768_, v_logExceptions_763_, v_a_765_, v_a_766_);
return v___x_771_;
}
else
{
uint8_t v_recover_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v_recover_772_ = lean_ctor_get_uint8(v_a_764_, sizeof(void*)*1);
v___x_773_ = lean_box(v_init_762_);
v___x_774_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_769_, v___x_773_, v_cfg_761_, v_onErr_768_, v_recover_772_, v_a_765_, v_a_766_);
return v___x_774_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg___boxed(lean_object* v_cfg_775_, lean_object* v_init_776_, lean_object* v_logExceptions_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_){
_start:
{
uint8_t v_init_boxed_782_; uint8_t v_logExceptions_boxed_783_; lean_object* v_res_784_; 
v_init_boxed_782_ = lean_unbox(v_init_776_);
v_logExceptions_boxed_783_ = lean_unbox(v_logExceptions_777_);
v_res_784_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v_cfg_775_, v_init_boxed_782_, v_logExceptions_boxed_783_, v_a_778_, v_a_779_, v_a_780_);
lean_dec(v_a_780_);
lean_dec_ref(v_a_779_);
lean_dec_ref(v_a_778_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig(lean_object* v_cfg_785_, uint8_t v_init_786_, uint8_t v_logExceptions_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_, lean_object* v_a_795_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v_cfg_785_, v_init_786_, v_logExceptions_787_, v_a_788_, v_a_794_, v_a_795_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___boxed(lean_object* v_cfg_798_, lean_object* v_init_799_, lean_object* v_logExceptions_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_){
_start:
{
uint8_t v_init_boxed_810_; uint8_t v_logExceptions_boxed_811_; lean_object* v_res_812_; 
v_init_boxed_810_ = lean_unbox(v_init_799_);
v_logExceptions_boxed_811_ = lean_unbox(v_logExceptions_800_);
v_res_812_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig(v_cfg_798_, v_init_boxed_810_, v_logExceptions_boxed_811_, v_a_801_, v_a_802_, v_a_803_, v_a_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_);
lean_dec(v_a_808_);
lean_dec_ref(v_a_807_);
lean_dec(v_a_806_);
lean_dec_ref(v_a_805_);
lean_dec(v_a_804_);
lean_dec_ref(v_a_803_);
lean_dec(v_a_802_);
lean_dec_ref(v_a_801_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_mkSimpStep(lean_object* v_e_813_, lean_object* v_pf_814_){
_start:
{
lean_object* v___x_815_; uint8_t v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_815_, 0, v_pf_814_);
v___x_816_ = 1;
v___x_817_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_817_, 0, v_e_813_);
lean_ctor_set(v___x_817_, 1, v___x_815_);
lean_ctor_set_uint8(v___x_817_, sizeof(void*)*2, v___x_816_);
v___x_818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
v___x_819_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg(lean_object* v_e_820_, lean_object* v___y_821_){
_start:
{
uint8_t v___x_823_; 
v___x_823_ = l_Lean_Expr_hasMVar(v_e_820_);
if (v___x_823_ == 0)
{
lean_object* v___x_824_; 
v___x_824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_824_, 0, v_e_820_);
return v___x_824_;
}
else
{
lean_object* v___x_825_; lean_object* v_mctx_826_; lean_object* v___x_827_; lean_object* v_fst_828_; lean_object* v_snd_829_; lean_object* v___x_830_; lean_object* v_cache_831_; lean_object* v_zetaDeltaFVarIds_832_; lean_object* v_postponed_833_; lean_object* v_diag_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_843_; 
v___x_825_ = lean_st_ref_get(v___y_821_);
v_mctx_826_ = lean_ctor_get(v___x_825_, 0);
lean_inc_ref(v_mctx_826_);
lean_dec(v___x_825_);
v___x_827_ = l_Lean_instantiateMVarsCore(v_mctx_826_, v_e_820_);
v_fst_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_fst_828_);
v_snd_829_ = lean_ctor_get(v___x_827_, 1);
lean_inc(v_snd_829_);
lean_dec_ref(v___x_827_);
v___x_830_ = lean_st_ref_take(v___y_821_);
v_cache_831_ = lean_ctor_get(v___x_830_, 1);
v_zetaDeltaFVarIds_832_ = lean_ctor_get(v___x_830_, 2);
v_postponed_833_ = lean_ctor_get(v___x_830_, 3);
v_diag_834_ = lean_ctor_get(v___x_830_, 4);
v_isSharedCheck_843_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_843_ == 0)
{
lean_object* v_unused_844_; 
v_unused_844_ = lean_ctor_get(v___x_830_, 0);
lean_dec(v_unused_844_);
v___x_836_ = v___x_830_;
v_isShared_837_ = v_isSharedCheck_843_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_diag_834_);
lean_inc(v_postponed_833_);
lean_inc(v_zetaDeltaFVarIds_832_);
lean_inc(v_cache_831_);
lean_dec(v___x_830_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_843_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
lean_object* v___x_839_; 
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 0, v_snd_829_);
v___x_839_ = v___x_836_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v_snd_829_);
lean_ctor_set(v_reuseFailAlloc_842_, 1, v_cache_831_);
lean_ctor_set(v_reuseFailAlloc_842_, 2, v_zetaDeltaFVarIds_832_);
lean_ctor_set(v_reuseFailAlloc_842_, 3, v_postponed_833_);
lean_ctor_set(v_reuseFailAlloc_842_, 4, v_diag_834_);
v___x_839_ = v_reuseFailAlloc_842_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_840_ = lean_st_ref_set(v___y_821_, v___x_839_);
v___x_841_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_841_, 0, v_fst_828_);
return v___x_841_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg___boxed(lean_object* v_e_845_, lean_object* v___y_846_, lean_object* v___y_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg(v_e_845_, v___y_846_);
lean_dec(v___y_846_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0(lean_object* v_e_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
lean_object* v___x_858_; 
v___x_858_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg(v_e_849_, v___y_854_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___boxed(lean_object* v_e_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0(v_e_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_, v___y_866_);
lean_dec(v___y_866_);
lean_dec_ref(v___y_865_);
lean_dec(v___y_864_);
lean_dec_ref(v___y_863_);
lean_dec(v___y_862_);
lean_dec_ref(v___y_861_);
lean_dec(v___y_860_);
return v_res_868_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6(void){
_start:
{
lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; 
v___x_881_ = lean_box(0);
v___x_882_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__5));
v___x_883_ = l_Lean_Expr_const___override(v___x_882_, v___x_881_);
return v___x_883_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9(void){
_start:
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_890_ = lean_box(0);
v___x_891_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__8));
v___x_892_ = l_Lean_Expr_const___override(v___x_891_, v___x_890_);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin(uint8_t v_cfg_902_, lean_object* v_e_903_, lean_object* v_a_904_, lean_object* v_a_905_, lean_object* v_a_906_, lean_object* v_a_907_, lean_object* v_a_908_, lean_object* v_a_909_, lean_object* v_a_910_){
_start:
{
lean_object* v___x_912_; lean_object* v_a_913_; lean_object* v___x_915_; uint8_t v_isShared_916_; uint8_t v_isSharedCheck_988_; 
v___x_912_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_spec__0___redArg(v_e_903_, v_a_908_);
v_a_913_ = lean_ctor_get(v___x_912_, 0);
v_isSharedCheck_988_ = !lean_is_exclusive(v___x_912_);
if (v_isSharedCheck_988_ == 0)
{
v___x_915_ = v___x_912_;
v_isShared_916_ = v_isSharedCheck_988_;
goto v_resetjp_914_;
}
else
{
lean_inc(v_a_913_);
lean_dec(v___x_912_);
v___x_915_ = lean_box(0);
v_isShared_916_ = v_isSharedCheck_988_;
goto v_resetjp_914_;
}
v_resetjp_914_:
{
lean_object* v___x_922_; 
v___x_922_ = l_Lean_Expr_cleanupAnnotations(v_a_913_);
switch(lean_obj_tag(v___x_922_))
{
case 5:
{
lean_object* v_fn_923_; 
v_fn_923_ = lean_ctor_get(v___x_922_, 0);
lean_inc_ref(v_fn_923_);
if (lean_obj_tag(v_fn_923_) == 5)
{
lean_object* v_fn_924_; 
v_fn_924_ = lean_ctor_get(v_fn_923_, 0);
if (lean_obj_tag(v_fn_924_) == 4)
{
lean_object* v_declName_925_; 
v_declName_925_ = lean_ctor_get(v_fn_924_, 0);
lean_inc(v_declName_925_);
if (lean_obj_tag(v_declName_925_) == 1)
{
lean_object* v_pre_926_; 
v_pre_926_ = lean_ctor_get(v_declName_925_, 0);
if (lean_obj_tag(v_pre_926_) == 0)
{
lean_object* v_arg_927_; lean_object* v_arg_928_; lean_object* v_str_929_; lean_object* v___x_930_; uint8_t v___x_931_; 
v_arg_927_ = lean_ctor_get(v___x_922_, 1);
lean_inc_ref(v_arg_927_);
lean_dec_ref_known(v___x_922_, 2);
v_arg_928_ = lean_ctor_get(v_fn_923_, 1);
lean_inc_ref(v_arg_928_);
lean_dec_ref_known(v_fn_923_, 2);
v_str_929_ = lean_ctor_get(v_declName_925_, 1);
lean_inc_ref(v_str_929_);
lean_dec_ref_known(v_declName_925_, 2);
v___x_930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__1));
v___x_931_ = lean_string_dec_eq(v_str_929_, v___x_930_);
lean_dec_ref(v_str_929_);
if (v___x_931_ == 0)
{
lean_dec_ref(v_arg_928_);
lean_dec_ref(v_arg_927_);
goto v___jp_917_;
}
else
{
lean_del_object(v___x_915_);
if (v_cfg_902_ == 0)
{
lean_object* v___x_932_; lean_object* v___x_933_; uint8_t v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; 
v___x_932_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__3));
lean_inc_ref(v_arg_927_);
v___x_933_ = l_Lean_mkNot(v_arg_927_);
v___x_934_ = 0;
lean_inc_ref(v_arg_928_);
v___x_935_ = l_Lean_Expr_forallE___override(v___x_932_, v_arg_928_, v___x_933_, v___x_934_);
v___x_936_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__6);
v___x_937_ = l_Lean_mkAppB(v___x_936_, v_arg_928_, v_arg_927_);
v___x_938_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_mkSimpStep(v___x_935_, v___x_937_);
v___x_939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_939_, 0, v___x_938_);
return v___x_939_;
}
else
{
lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; 
lean_inc_ref(v_arg_928_);
v___x_940_ = l_Lean_mkNot(v_arg_928_);
lean_inc_ref(v_arg_927_);
v___x_941_ = l_Lean_mkNot(v_arg_927_);
v___x_942_ = l_Lean_mkOr(v___x_940_, v___x_941_);
v___x_943_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9, &lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__9);
v___x_944_ = l_Lean_mkAppB(v___x_943_, v_arg_928_, v_arg_927_);
v___x_945_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_mkSimpStep(v___x_942_, v___x_944_);
v___x_946_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_946_, 0, v___x_945_);
return v___x_946_;
}
}
}
else
{
lean_dec_ref_known(v_declName_925_, 2);
lean_dec_ref_known(v_fn_923_, 2);
lean_dec_ref_known(v___x_922_, 2);
goto v___jp_917_;
}
}
else
{
lean_dec(v_declName_925_);
lean_dec_ref_known(v_fn_923_, 2);
lean_dec_ref_known(v___x_922_, 2);
goto v___jp_917_;
}
}
else
{
lean_dec_ref_known(v_fn_923_, 2);
lean_dec_ref_known(v___x_922_, 2);
goto v___jp_917_;
}
}
else
{
lean_dec_ref(v_fn_923_);
lean_dec_ref_known(v___x_922_, 2);
goto v___jp_917_;
}
}
case 7:
{
lean_object* v_binderName_947_; lean_object* v_binderType_948_; lean_object* v_body_949_; uint8_t v_binderInfo_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; 
lean_del_object(v___x_915_);
v_binderName_947_ = lean_ctor_get(v___x_922_, 0);
lean_inc_n(v_binderName_947_, 2);
v_binderType_948_ = lean_ctor_get(v___x_922_, 1);
lean_inc_ref_n(v_binderType_948_, 2);
v_body_949_ = lean_ctor_get(v___x_922_, 2);
lean_inc_ref_n(v_body_949_, 2);
v_binderInfo_950_ = lean_ctor_get_uint8(v___x_922_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v___x_922_, 3);
v___x_951_ = l_Lean_mkNot(v_body_949_);
v___x_952_ = l_Lean_Expr_lam___override(v_binderName_947_, v_binderType_948_, v___x_951_, v_binderInfo_950_);
v___x_953_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__11));
v___x_954_ = lean_unsigned_to_nat(1u);
v___x_955_ = lean_mk_empty_array_with_capacity(v___x_954_);
lean_inc_ref(v___x_955_);
v___x_956_ = lean_array_push(v___x_955_, v___x_952_);
v___x_957_ = l_Lean_Meta_mkAppM(v___x_953_, v___x_956_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
if (lean_obj_tag(v___x_957_) == 0)
{
lean_object* v_a_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; 
v_a_958_ = lean_ctor_get(v___x_957_, 0);
lean_inc(v_a_958_);
lean_dec_ref_known(v___x_957_, 1);
v___x_959_ = l_Lean_Expr_lam___override(v_binderName_947_, v_binderType_948_, v_body_949_, v_binderInfo_950_);
v___x_960_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__13));
v___x_961_ = lean_array_push(v___x_955_, v___x_959_);
v___x_962_ = l_Lean_Meta_mkAppM(v___x_960_, v___x_961_, v_a_907_, v_a_908_, v_a_909_, v_a_910_);
if (lean_obj_tag(v___x_962_) == 0)
{
lean_object* v_a_963_; lean_object* v___x_965_; uint8_t v_isShared_966_; uint8_t v_isSharedCheck_971_; 
v_a_963_ = lean_ctor_get(v___x_962_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_962_);
if (v_isSharedCheck_971_ == 0)
{
v___x_965_ = v___x_962_;
v_isShared_966_ = v_isSharedCheck_971_;
goto v_resetjp_964_;
}
else
{
lean_inc(v_a_963_);
lean_dec(v___x_962_);
v___x_965_ = lean_box(0);
v_isShared_966_ = v_isSharedCheck_971_;
goto v_resetjp_964_;
}
v_resetjp_964_:
{
lean_object* v___x_967_; lean_object* v___x_969_; 
v___x_967_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin_mkSimpStep(v_a_958_, v_a_963_);
if (v_isShared_966_ == 0)
{
lean_ctor_set(v___x_965_, 0, v___x_967_);
v___x_969_ = v___x_965_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v___x_967_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
}
else
{
lean_object* v_a_972_; lean_object* v___x_974_; uint8_t v_isShared_975_; uint8_t v_isSharedCheck_979_; 
lean_dec(v_a_958_);
v_a_972_ = lean_ctor_get(v___x_962_, 0);
v_isSharedCheck_979_ = !lean_is_exclusive(v___x_962_);
if (v_isSharedCheck_979_ == 0)
{
v___x_974_ = v___x_962_;
v_isShared_975_ = v_isSharedCheck_979_;
goto v_resetjp_973_;
}
else
{
lean_inc(v_a_972_);
lean_dec(v___x_962_);
v___x_974_ = lean_box(0);
v_isShared_975_ = v_isSharedCheck_979_;
goto v_resetjp_973_;
}
v_resetjp_973_:
{
lean_object* v___x_977_; 
if (v_isShared_975_ == 0)
{
v___x_977_ = v___x_974_;
goto v_reusejp_976_;
}
else
{
lean_object* v_reuseFailAlloc_978_; 
v_reuseFailAlloc_978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_978_, 0, v_a_972_);
v___x_977_ = v_reuseFailAlloc_978_;
goto v_reusejp_976_;
}
v_reusejp_976_:
{
return v___x_977_;
}
}
}
}
else
{
lean_object* v_a_980_; lean_object* v___x_982_; uint8_t v_isShared_983_; uint8_t v_isSharedCheck_987_; 
lean_dec_ref(v___x_955_);
lean_dec_ref(v_body_949_);
lean_dec_ref(v_binderType_948_);
lean_dec(v_binderName_947_);
v_a_980_ = lean_ctor_get(v___x_957_, 0);
v_isSharedCheck_987_ = !lean_is_exclusive(v___x_957_);
if (v_isSharedCheck_987_ == 0)
{
v___x_982_ = v___x_957_;
v_isShared_983_ = v_isSharedCheck_987_;
goto v_resetjp_981_;
}
else
{
lean_inc(v_a_980_);
lean_dec(v___x_957_);
v___x_982_ = lean_box(0);
v_isShared_983_ = v_isSharedCheck_987_;
goto v_resetjp_981_;
}
v_resetjp_981_:
{
lean_object* v___x_985_; 
if (v_isShared_983_ == 0)
{
v___x_985_ = v___x_982_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v_a_980_);
v___x_985_ = v_reuseFailAlloc_986_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
return v___x_985_;
}
}
}
}
default: 
{
lean_dec_ref(v___x_922_);
goto v___jp_917_;
}
}
v___jp_917_:
{
lean_object* v___x_918_; lean_object* v___x_920_; 
v___x_918_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
if (v_isShared_916_ == 0)
{
lean_ctor_set(v___x_915_, 0, v___x_918_);
v___x_920_ = v___x_915_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v___x_918_);
v___x_920_ = v_reuseFailAlloc_921_;
goto v_reusejp_919_;
}
v_reusejp_919_:
{
return v___x_920_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___boxed(lean_object* v_cfg_989_, lean_object* v_e_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_){
_start:
{
uint8_t v_cfg_boxed_999_; lean_object* v_res_1000_; 
v_cfg_boxed_999_ = lean_unbox(v_cfg_989_);
v_res_1000_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin(v_cfg_boxed_999_, v_e_990_, v_a_991_, v_a_992_, v_a_993_, v_a_994_, v_a_995_, v_a_996_, v_a_997_);
lean_dec(v_a_997_);
lean_dec_ref(v_a_996_);
lean_dec(v_a_995_);
lean_dec_ref(v_a_994_);
lean_dec(v_a_993_);
lean_dec_ref(v_a_992_);
lean_dec(v_a_991_);
return v_res_1000_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1009_; 
v___x_1009_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1009_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__0);
v___x_1011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1011_, 0, v___x_1010_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0(lean_object* v_00_u03b2_1012_){
_start:
{
lean_object* v___x_1013_; 
v___x_1013_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0___closed__1);
return v___x_1013_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0(void){
_start:
{
lean_object* v___x_1014_; 
v___x_1014_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_1014_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1(void){
_start:
{
lean_object* v___x_1015_; 
v___x_1015_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Push_pushStep_spec__0(lean_box(0));
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep(lean_object* v_head_1020_, uint8_t v_cfg_1021_, lean_object* v_e_1022_, lean_object* v_a_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_){
_start:
{
lean_object* v___x_1031_; 
lean_inc(v_a_1029_);
lean_inc_ref(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
lean_inc_ref(v_e_1022_);
v___x_1031_ = lean_whnf(v_e_1022_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
if (lean_obj_tag(v___x_1031_) == 0)
{
lean_object* v_a_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1092_; 
v_a_1032_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1092_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1034_ = v___x_1031_;
v_isShared_1035_ = v_isSharedCheck_1092_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_a_1032_);
lean_dec(v___x_1031_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1092_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v___x_1036_; 
v___x_1036_ = lp_mathlib_Mathlib_Tactic_Push_Head_ofExpr_x3f(v_a_1032_);
if (lean_obj_tag(v___x_1036_) == 1)
{
lean_object* v_val_1037_; uint8_t v___x_1038_; 
v_val_1037_ = lean_ctor_get(v___x_1036_, 0);
lean_inc(v_val_1037_);
lean_dec_ref_known(v___x_1036_, 1);
v___x_1038_ = lp_mathlib_Mathlib_Tactic_Push_instBEqHead_beq(v_val_1037_, v_head_1020_);
lean_dec(v_val_1037_);
if (v___x_1038_ == 0)
{
lean_object* v___x_1039_; lean_object* v___x_1041_; 
lean_dec(v_a_1032_);
lean_dec_ref(v_e_1022_);
v___x_1039_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1039_);
v___x_1041_ = v___x_1034_;
goto v_reusejp_1040_;
}
else
{
lean_object* v_reuseFailAlloc_1042_; 
v_reuseFailAlloc_1042_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1042_, 0, v___x_1039_);
v___x_1041_ = v_reuseFailAlloc_1042_;
goto v_reusejp_1040_;
}
v_reusejp_1040_:
{
return v___x_1041_;
}
}
else
{
lean_object* v___x_1043_; lean_object* v_env_1044_; lean_object* v___x_1045_; lean_object* v_ext_1046_; lean_object* v_toEnvExtension_1047_; lean_object* v_asyncMode_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; uint8_t v___x_1053_; lean_object* v___x_1054_; 
lean_del_object(v___x_1034_);
v___x_1043_ = lean_st_ref_get(v_a_1029_);
v_env_1044_ = lean_ctor_get(v___x_1043_, 0);
lean_inc_ref(v_env_1044_);
lean_dec(v___x_1043_);
v___x_1045_ = lp_mathlib_Mathlib_Tactic_Push_pushExt;
v_ext_1046_ = lean_ctor_get(v___x_1045_, 1);
v_toEnvExtension_1047_ = lean_ctor_get(v_ext_1046_, 0);
v_asyncMode_1048_ = lean_ctor_get(v_toEnvExtension_1047_, 2);
v___x_1049_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0, &lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0);
v___x_1050_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1049_, v___x_1045_, v_env_1044_, v_asyncMode_1048_);
v___x_1051_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1, &lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__1);
v___x_1052_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2));
v___x_1053_ = 0;
v___x_1054_ = l_Lean_Meta_Simp_rewrite_x3f(v_e_1022_, v___x_1050_, v___x_1051_, v___x_1052_, v___x_1053_, v_a_1023_, v_a_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
lean_dec(v___x_1050_);
if (lean_obj_tag(v___x_1054_) == 0)
{
lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1079_; 
v_a_1055_ = lean_ctor_get(v___x_1054_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1054_);
if (v_isSharedCheck_1079_ == 0)
{
v___x_1057_ = v___x_1054_;
v_isShared_1058_ = v_isSharedCheck_1079_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_1054_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1079_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
if (lean_obj_tag(v_a_1055_) == 1)
{
lean_object* v_val_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1069_; 
lean_dec(v_a_1032_);
v_val_1059_ = lean_ctor_get(v_a_1055_, 0);
v_isSharedCheck_1069_ = !lean_is_exclusive(v_a_1055_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1061_ = v_a_1055_;
v_isShared_1062_ = v_isSharedCheck_1069_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_val_1059_);
lean_dec(v_a_1055_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1069_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v_val_1059_);
v___x_1064_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
lean_object* v___x_1066_; 
if (v_isShared_1058_ == 0)
{
lean_ctor_set(v___x_1057_, 0, v___x_1064_);
v___x_1066_ = v___x_1057_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1067_; 
v_reuseFailAlloc_1067_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1067_, 0, v___x_1064_);
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
else
{
lean_object* v___x_1070_; lean_object* v___x_1071_; uint8_t v___x_1072_; 
lean_dec(v_a_1055_);
v___x_1070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4));
v___x_1071_ = lean_unsigned_to_nat(1u);
v___x_1072_ = l_Lean_Expr_isAppOfArity(v_a_1032_, v___x_1070_, v___x_1071_);
if (v___x_1072_ == 0)
{
lean_object* v___x_1073_; lean_object* v___x_1075_; 
lean_dec(v_a_1032_);
v___x_1073_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
if (v_isShared_1058_ == 0)
{
lean_ctor_set(v___x_1057_, 0, v___x_1073_);
v___x_1075_ = v___x_1057_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v___x_1073_);
v___x_1075_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
return v___x_1075_;
}
}
else
{
lean_object* v___x_1077_; lean_object* v___x_1078_; 
lean_del_object(v___x_1057_);
v___x_1077_ = l_Lean_Expr_appArg_x21(v_a_1032_);
lean_dec(v_a_1032_);
v___x_1078_ = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin(v_cfg_1021_, v___x_1077_, v_a_1023_, v_a_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
return v___x_1078_;
}
}
}
}
else
{
lean_object* v_a_1080_; lean_object* v___x_1082_; uint8_t v_isShared_1083_; uint8_t v_isSharedCheck_1087_; 
lean_dec(v_a_1032_);
v_a_1080_ = lean_ctor_get(v___x_1054_, 0);
v_isSharedCheck_1087_ = !lean_is_exclusive(v___x_1054_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1082_ = v___x_1054_;
v_isShared_1083_ = v_isSharedCheck_1087_;
goto v_resetjp_1081_;
}
else
{
lean_inc(v_a_1080_);
lean_dec(v___x_1054_);
v___x_1082_ = lean_box(0);
v_isShared_1083_ = v_isSharedCheck_1087_;
goto v_resetjp_1081_;
}
v_resetjp_1081_:
{
lean_object* v___x_1085_; 
if (v_isShared_1083_ == 0)
{
v___x_1085_ = v___x_1082_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v_a_1080_);
v___x_1085_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
return v___x_1085_;
}
}
}
}
}
else
{
lean_object* v___x_1088_; lean_object* v___x_1090_; 
lean_dec(v___x_1036_);
lean_dec(v_a_1032_);
lean_dec_ref(v_e_1022_);
v___x_1088_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1088_);
v___x_1090_ = v___x_1034_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v___x_1088_);
v___x_1090_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
return v___x_1090_;
}
}
}
}
else
{
lean_object* v_a_1093_; lean_object* v___x_1095_; uint8_t v_isShared_1096_; uint8_t v_isSharedCheck_1100_; 
lean_dec_ref(v_e_1022_);
v_a_1093_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1100_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1100_ == 0)
{
v___x_1095_ = v___x_1031_;
v_isShared_1096_ = v_isSharedCheck_1100_;
goto v_resetjp_1094_;
}
else
{
lean_inc(v_a_1093_);
lean_dec(v___x_1031_);
v___x_1095_ = lean_box(0);
v_isShared_1096_ = v_isSharedCheck_1100_;
goto v_resetjp_1094_;
}
v_resetjp_1094_:
{
lean_object* v___x_1098_; 
if (v_isShared_1096_ == 0)
{
v___x_1098_ = v___x_1095_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1099_; 
v_reuseFailAlloc_1099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1099_, 0, v_a_1093_);
v___x_1098_ = v_reuseFailAlloc_1099_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
return v___x_1098_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushStep___boxed(lean_object* v_head_1101_, lean_object* v_cfg_1102_, lean_object* v_e_1103_, lean_object* v_a_1104_, lean_object* v_a_1105_, lean_object* v_a_1106_, lean_object* v_a_1107_, lean_object* v_a_1108_, lean_object* v_a_1109_, lean_object* v_a_1110_, lean_object* v_a_1111_){
_start:
{
uint8_t v_cfg_boxed_1112_; lean_object* v_res_1113_; 
v_cfg_boxed_1112_ = lean_unbox(v_cfg_1102_);
v_res_1113_ = lp_mathlib_Mathlib_Tactic_Push_pushStep(v_head_1101_, v_cfg_boxed_1112_, v_e_1103_, v_a_1104_, v_a_1105_, v_a_1106_, v_a_1107_, v_a_1108_, v_a_1109_, v_a_1110_);
lean_dec(v_a_1110_);
lean_dec_ref(v_a_1109_);
lean_dec(v_a_1108_);
lean_dec_ref(v_a_1107_);
lean_dec(v_a_1106_);
lean_dec_ref(v_a_1105_);
lean_dec(v_a_1104_);
lean_dec(v_head_1101_);
return v_res_1113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0(lean_object* v_x_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; 
v___x_1123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
v___x_1124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1124_, 0, v___x_1123_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0___boxed(lean_object* v_x_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__0(v_x_1125_, v___y_1126_, v___y_1127_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_, v___y_1132_);
lean_dec(v___y_1132_);
lean_dec_ref(v___y_1131_);
lean_dec(v___y_1130_);
lean_dec_ref(v___y_1129_);
lean_dec(v___y_1128_);
lean_dec_ref(v___y_1127_);
lean_dec(v___y_1126_);
lean_dec_ref(v_x_1125_);
return v_res_1134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1(lean_object* v_x_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; 
v___x_1146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___closed__0));
v___x_1147_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1147_, 0, v___x_1146_);
return v___x_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1___boxed(lean_object* v_x_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_){
_start:
{
lean_object* v_res_1157_; 
v_res_1157_ = lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__1(v_x_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_);
lean_dec(v___y_1155_);
lean_dec_ref(v___y_1154_);
lean_dec(v___y_1153_);
lean_dec_ref(v___y_1152_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1149_);
lean_dec_ref(v_x_1148_);
return v_res_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2(lean_object* v_e_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1167_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1167_, 0, v_e_1158_);
v___x_1168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1167_);
return v___x_1168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2___boxed(lean_object* v_e_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_){
_start:
{
lean_object* v_res_1178_; 
v_res_1178_ = lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__2(v_e_1169_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
lean_dec(v___y_1176_);
lean_dec_ref(v___y_1175_);
lean_dec(v___y_1174_);
lean_dec_ref(v___y_1173_);
lean_dec(v___y_1172_);
lean_dec_ref(v___y_1171_);
lean_dec(v___y_1170_);
return v_res_1178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3(lean_object* v_x_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_){
_start:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; 
v___x_1188_ = lean_box(0);
v___x_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1188_);
return v___x_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3___boxed(lean_object* v_x_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_){
_start:
{
lean_object* v_res_1199_; 
v_res_1199_ = lp_mathlib_Mathlib_Tactic_Push_pushCore___lam__3(v_x_1190_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_, v___y_1195_, v___y_1196_, v___y_1197_);
lean_dec(v___y_1197_);
lean_dec_ref(v___y_1196_);
lean_dec(v___y_1195_);
lean_dec_ref(v___y_1194_);
lean_dec(v___y_1193_);
lean_dec_ref(v___y_1192_);
lean_dec(v___y_1191_);
lean_dec_ref(v_x_1190_);
return v_res_1199_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1(void){
_start:
{
lean_object* v___x_1202_; 
v___x_1202_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1202_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2(void){
_start:
{
lean_object* v___x_1203_; lean_object* v___x_1204_; 
v___x_1203_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__1);
v___x_1204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1203_);
return v___x_1204_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3(void){
_start:
{
lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
v___x_1205_ = lean_unsigned_to_nat(0u);
v___x_1206_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2);
v___x_1207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1207_, 0, v___x_1206_);
lean_ctor_set(v___x_1207_, 1, v___x_1205_);
return v___x_1207_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4(void){
_start:
{
lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; 
v___x_1208_ = lean_unsigned_to_nat(32u);
v___x_1209_ = lean_mk_empty_array_with_capacity(v___x_1208_);
v___x_1210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1210_, 0, v___x_1209_);
return v___x_1210_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5(void){
_start:
{
size_t v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; 
v___x_1211_ = ((size_t)5ULL);
v___x_1212_ = lean_unsigned_to_nat(0u);
v___x_1213_ = lean_unsigned_to_nat(32u);
v___x_1214_ = lean_mk_empty_array_with_capacity(v___x_1213_);
v___x_1215_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__4);
v___x_1216_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1216_, 0, v___x_1215_);
lean_ctor_set(v___x_1216_, 1, v___x_1214_);
lean_ctor_set(v___x_1216_, 2, v___x_1212_);
lean_ctor_set(v___x_1216_, 3, v___x_1212_);
lean_ctor_set_usize(v___x_1216_, 4, v___x_1211_);
return v___x_1216_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6(void){
_start:
{
lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; 
v___x_1217_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__5);
v___x_1218_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__2);
v___x_1219_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
lean_ctor_set(v___x_1219_, 1, v___x_1218_);
lean_ctor_set(v___x_1219_, 2, v___x_1218_);
lean_ctor_set(v___x_1219_, 3, v___x_1217_);
return v___x_1219_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7(void){
_start:
{
lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; 
v___x_1220_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__6);
v___x_1221_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__3);
v___x_1222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1221_);
lean_ctor_set(v___x_1222_, 1, v___x_1220_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore(lean_object* v_head_1227_, uint8_t v_cfg_1228_, lean_object* v_disch_x3f_1229_, lean_object* v_tgt_1230_, lean_object* v_a_1231_, lean_object* v_a_1232_, lean_object* v_a_1233_, lean_object* v_a_1234_){
_start:
{
lean_object* v___x_1236_; 
v___x_1236_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_1234_);
if (lean_obj_tag(v___x_1236_) == 0)
{
lean_object* v_a_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; 
v_a_1237_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1237_);
lean_dec_ref_known(v___x_1236_, 1);
v___x_1238_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig));
v___x_1239_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__0));
v___x_1240_ = l_Lean_Options_empty;
v___x_1241_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_1238_, v___x_1239_, v_a_1237_, v___x_1240_, v_a_1231_, v_a_1233_, v_a_1234_);
if (lean_obj_tag(v___x_1241_) == 0)
{
lean_object* v_a_1242_; lean_object* v___y_1244_; 
v_a_1242_ = lean_ctor_get(v___x_1241_, 0);
lean_inc(v_a_1242_);
lean_dec_ref_known(v___x_1241_, 1);
if (lean_obj_tag(v_disch_x3f_1229_) == 0)
{
lean_object* v___f_1264_; lean_object* v___f_1265_; lean_object* v___f_1266_; lean_object* v___f_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; uint8_t v___x_1270_; lean_object* v___x_1271_; 
v___f_1264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8));
v___f_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9));
v___f_1266_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10));
v___f_1267_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__11));
v___x_1268_ = lean_box(v_cfg_1228_);
v___x_1269_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___boxed), 11, 2);
lean_closure_set(v___x_1269_, 0, v_head_1227_);
lean_closure_set(v___x_1269_, 1, v___x_1268_);
v___x_1270_ = 1;
v___x_1271_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1271_, 0, v___x_1269_);
lean_ctor_set(v___x_1271_, 1, v___f_1264_);
lean_ctor_set(v___x_1271_, 2, v___f_1265_);
lean_ctor_set(v___x_1271_, 3, v___f_1266_);
lean_ctor_set(v___x_1271_, 4, v___f_1267_);
lean_ctor_set_uint8(v___x_1271_, sizeof(void*)*5, v___x_1270_);
v___y_1244_ = v___x_1271_;
goto v___jp_1243_;
}
else
{
lean_object* v_val_1272_; lean_object* v___f_1273_; lean_object* v___f_1274_; lean_object* v___f_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; uint8_t v___x_1278_; lean_object* v___x_1279_; 
v_val_1272_ = lean_ctor_get(v_disch_x3f_1229_, 0);
v___f_1273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8));
v___f_1274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9));
v___f_1275_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10));
v___x_1276_ = lean_box(v_cfg_1228_);
v___x_1277_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___boxed), 11, 2);
lean_closure_set(v___x_1277_, 0, v_head_1227_);
lean_closure_set(v___x_1277_, 1, v___x_1276_);
v___x_1278_ = 0;
lean_inc(v_val_1272_);
v___x_1279_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1279_, 0, v___x_1277_);
lean_ctor_set(v___x_1279_, 1, v___f_1273_);
lean_ctor_set(v___x_1279_, 2, v___f_1274_);
lean_ctor_set(v___x_1279_, 3, v___f_1275_);
lean_ctor_set(v___x_1279_, 4, v_val_1272_);
lean_ctor_set_uint8(v___x_1279_, sizeof(void*)*5, v___x_1278_);
v___y_1244_ = v___x_1279_;
goto v___jp_1243_;
}
v___jp_1243_:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1245_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7);
v___x_1246_ = l_Lean_Meta_Simp_main(v_tgt_1230_, v_a_1242_, v___x_1245_, v___y_1244_, v_a_1231_, v_a_1232_, v_a_1233_, v_a_1234_);
if (lean_obj_tag(v___x_1246_) == 0)
{
lean_object* v_a_1247_; lean_object* v___x_1249_; uint8_t v_isShared_1250_; uint8_t v_isSharedCheck_1255_; 
v_a_1247_ = lean_ctor_get(v___x_1246_, 0);
v_isSharedCheck_1255_ = !lean_is_exclusive(v___x_1246_);
if (v_isSharedCheck_1255_ == 0)
{
v___x_1249_ = v___x_1246_;
v_isShared_1250_ = v_isSharedCheck_1255_;
goto v_resetjp_1248_;
}
else
{
lean_inc(v_a_1247_);
lean_dec(v___x_1246_);
v___x_1249_ = lean_box(0);
v_isShared_1250_ = v_isSharedCheck_1255_;
goto v_resetjp_1248_;
}
v_resetjp_1248_:
{
lean_object* v_fst_1251_; lean_object* v___x_1253_; 
v_fst_1251_ = lean_ctor_get(v_a_1247_, 0);
lean_inc(v_fst_1251_);
lean_dec(v_a_1247_);
if (v_isShared_1250_ == 0)
{
lean_ctor_set(v___x_1249_, 0, v_fst_1251_);
v___x_1253_ = v___x_1249_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1254_; 
v_reuseFailAlloc_1254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1254_, 0, v_fst_1251_);
v___x_1253_ = v_reuseFailAlloc_1254_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
return v___x_1253_;
}
}
}
else
{
lean_object* v_a_1256_; lean_object* v___x_1258_; uint8_t v_isShared_1259_; uint8_t v_isSharedCheck_1263_; 
v_a_1256_ = lean_ctor_get(v___x_1246_, 0);
v_isSharedCheck_1263_ = !lean_is_exclusive(v___x_1246_);
if (v_isSharedCheck_1263_ == 0)
{
v___x_1258_ = v___x_1246_;
v_isShared_1259_ = v_isSharedCheck_1263_;
goto v_resetjp_1257_;
}
else
{
lean_inc(v_a_1256_);
lean_dec(v___x_1246_);
v___x_1258_ = lean_box(0);
v_isShared_1259_ = v_isSharedCheck_1263_;
goto v_resetjp_1257_;
}
v_resetjp_1257_:
{
lean_object* v___x_1261_; 
if (v_isShared_1259_ == 0)
{
v___x_1261_ = v___x_1258_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v_a_1256_);
v___x_1261_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1260_;
}
v_reusejp_1260_:
{
return v___x_1261_;
}
}
}
}
}
else
{
lean_object* v_a_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1287_; 
lean_dec_ref(v_tgt_1230_);
lean_dec(v_head_1227_);
v_a_1280_ = lean_ctor_get(v___x_1241_, 0);
v_isSharedCheck_1287_ = !lean_is_exclusive(v___x_1241_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1282_ = v___x_1241_;
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_a_1280_);
lean_dec(v___x_1241_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
lean_object* v___x_1285_; 
if (v_isShared_1283_ == 0)
{
v___x_1285_ = v___x_1282_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1286_; 
v_reuseFailAlloc_1286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1286_, 0, v_a_1280_);
v___x_1285_ = v_reuseFailAlloc_1286_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
return v___x_1285_;
}
}
}
}
else
{
lean_object* v_a_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1295_; 
lean_dec_ref(v_tgt_1230_);
lean_dec(v_head_1227_);
v_a_1288_ = lean_ctor_get(v___x_1236_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1236_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1290_ = v___x_1236_;
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_a_1288_);
lean_dec(v___x_1236_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v___x_1293_; 
if (v_isShared_1291_ == 0)
{
v___x_1293_ = v___x_1290_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v_a_1288_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pushCore___boxed(lean_object* v_head_1296_, lean_object* v_cfg_1297_, lean_object* v_disch_x3f_1298_, lean_object* v_tgt_1299_, lean_object* v_a_1300_, lean_object* v_a_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_){
_start:
{
uint8_t v_cfg_boxed_1305_; lean_object* v_res_1306_; 
v_cfg_boxed_1305_ = lean_unbox(v_cfg_1297_);
v_res_1306_ = lp_mathlib_Mathlib_Tactic_Push_pushCore(v_head_1296_, v_cfg_boxed_1305_, v_disch_x3f_1298_, v_tgt_1299_, v_a_1300_, v_a_1301_, v_a_1302_, v_a_1303_);
lean_dec(v_a_1303_);
lean_dec_ref(v_a_1302_);
lean_dec(v_a_1301_);
lean_dec_ref(v_a_1300_);
lean_dec(v_disch_x3f_1298_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0___redArg(lean_object* v_xs_1307_, lean_object* v_j_1308_){
_start:
{
lean_object* v_zero_1309_; uint8_t v_isZero_1310_; 
v_zero_1309_ = lean_unsigned_to_nat(0u);
v_isZero_1310_ = lean_nat_dec_eq(v_j_1308_, v_zero_1309_);
if (v_isZero_1310_ == 1)
{
lean_dec(v_j_1308_);
return v_xs_1307_;
}
else
{
lean_object* v_one_1311_; lean_object* v_n_1312_; lean_object* v___x_1313_; lean_object* v_fst_1314_; lean_object* v_fst_1315_; lean_object* v_priority_1316_; lean_object* v___x_1317_; lean_object* v_fst_1318_; lean_object* v_fst_1319_; lean_object* v_priority_1320_; uint8_t v___x_1321_; 
v_one_1311_ = lean_unsigned_to_nat(1u);
v_n_1312_ = lean_nat_sub(v_j_1308_, v_one_1311_);
v___x_1313_ = lean_array_fget_borrowed(v_xs_1307_, v_n_1312_);
v_fst_1314_ = lean_ctor_get(v___x_1313_, 0);
v_fst_1315_ = lean_ctor_get(v_fst_1314_, 0);
v_priority_1316_ = lean_ctor_get(v_fst_1315_, 3);
v___x_1317_ = lean_array_fget_borrowed(v_xs_1307_, v_j_1308_);
v_fst_1318_ = lean_ctor_get(v___x_1317_, 0);
v_fst_1319_ = lean_ctor_get(v_fst_1318_, 0);
v_priority_1320_ = lean_ctor_get(v_fst_1319_, 3);
v___x_1321_ = lean_nat_dec_lt(v_priority_1316_, v_priority_1320_);
if (v___x_1321_ == 0)
{
lean_dec(v_n_1312_);
lean_dec(v_j_1308_);
return v_xs_1307_;
}
else
{
lean_object* v___x_1322_; 
v___x_1322_ = lean_array_fswap(v_xs_1307_, v_j_1308_, v_n_1312_);
lean_dec(v_j_1308_);
v_xs_1307_ = v___x_1322_;
v_j_1308_ = v_n_1312_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0(lean_object* v_xs_1324_, lean_object* v_i_1325_, lean_object* v_fuel_1326_){
_start:
{
lean_object* v_zero_1327_; uint8_t v_isZero_1328_; 
v_zero_1327_ = lean_unsigned_to_nat(0u);
v_isZero_1328_ = lean_nat_dec_eq(v_fuel_1326_, v_zero_1327_);
if (v_isZero_1328_ == 1)
{
lean_dec(v_fuel_1326_);
lean_dec(v_i_1325_);
return v_xs_1324_;
}
else
{
lean_object* v___x_1329_; uint8_t v___x_1330_; 
v___x_1329_ = lean_array_get_size(v_xs_1324_);
v___x_1330_ = lean_nat_dec_lt(v_i_1325_, v___x_1329_);
if (v___x_1330_ == 0)
{
lean_dec(v_fuel_1326_);
lean_dec(v_i_1325_);
return v_xs_1324_;
}
else
{
lean_object* v_one_1331_; lean_object* v_n_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v_one_1331_ = lean_unsigned_to_nat(1u);
v_n_1332_ = lean_nat_sub(v_fuel_1326_, v_one_1331_);
lean_dec(v_fuel_1326_);
lean_inc(v_i_1325_);
v___x_1333_ = lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0___redArg(v_xs_1324_, v_i_1325_);
v___x_1334_ = lean_nat_add(v_i_1325_, v_one_1331_);
lean_dec(v_i_1325_);
v_xs_1324_ = v___x_1333_;
v_i_1325_ = v___x_1334_;
v_fuel_1326_ = v_n_1332_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1(lean_object* v_head_1339_, lean_object* v_e_1340_, lean_object* v_as_1341_, size_t v_sz_1342_, size_t v_i_1343_, lean_object* v_b_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v_a_1354_; uint8_t v___x_1358_; 
v___x_1358_ = lean_usize_dec_lt(v_i_1343_, v_sz_1342_);
if (v___x_1358_ == 0)
{
lean_object* v___x_1359_; 
lean_dec_ref(v_e_1340_);
v___x_1359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1359_, 0, v_b_1344_);
return v___x_1359_;
}
else
{
lean_object* v_a_1360_; lean_object* v_fst_1361_; lean_object* v_snd_1362_; lean_object* v_fst_1363_; lean_object* v_snd_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1393_; 
lean_dec_ref(v_b_1344_);
v_a_1360_ = lean_array_uget_borrowed(v_as_1341_, v_i_1343_);
v_fst_1361_ = lean_ctor_get(v_a_1360_, 0);
lean_inc(v_fst_1361_);
v_snd_1362_ = lean_ctor_get(v_a_1360_, 1);
v_fst_1363_ = lean_ctor_get(v_fst_1361_, 0);
v_snd_1364_ = lean_ctor_get(v_fst_1361_, 1);
v_isSharedCheck_1393_ = !lean_is_exclusive(v_fst_1361_);
if (v_isSharedCheck_1393_ == 0)
{
v___x_1366_ = v_fst_1361_;
v_isShared_1367_ = v_isSharedCheck_1393_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_snd_1364_);
lean_inc(v_fst_1363_);
lean_dec(v_fst_1361_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1393_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
lean_object* v___x_1368_; lean_object* v___x_1369_; uint8_t v___x_1370_; 
v___x_1368_ = lean_box(0);
v___x_1369_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___closed__0));
v___x_1370_ = lp_mathlib_Mathlib_Tactic_Push_instBEqHead_beq(v_snd_1364_, v_head_1339_);
lean_dec(v_snd_1364_);
if (v___x_1370_ == 0)
{
lean_del_object(v___x_1366_);
lean_dec(v_fst_1363_);
v_a_1354_ = v___x_1369_;
goto v___jp_1353_;
}
else
{
lean_object* v___x_1371_; 
lean_inc(v_snd_1362_);
lean_inc_ref(v_e_1340_);
v___x_1371_ = l_Lean_Meta_Simp_tryTheoremWithExtraArgs_x3f(v_e_1340_, v_fst_1363_, v_snd_1362_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_);
if (lean_obj_tag(v___x_1371_) == 0)
{
lean_object* v_a_1372_; lean_object* v___x_1374_; uint8_t v_isShared_1375_; uint8_t v_isSharedCheck_1384_; 
v_a_1372_ = lean_ctor_get(v___x_1371_, 0);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1371_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1374_ = v___x_1371_;
v_isShared_1375_ = v_isSharedCheck_1384_;
goto v_resetjp_1373_;
}
else
{
lean_inc(v_a_1372_);
lean_dec(v___x_1371_);
v___x_1374_ = lean_box(0);
v_isShared_1375_ = v_isSharedCheck_1384_;
goto v_resetjp_1373_;
}
v_resetjp_1373_:
{
if (lean_obj_tag(v_a_1372_) == 1)
{
lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1379_; 
lean_dec_ref(v_e_1340_);
v___x_1376_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1376_, 0, v_a_1372_);
v___x_1377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1377_, 0, v___x_1376_);
if (v_isShared_1367_ == 0)
{
lean_ctor_set(v___x_1366_, 1, v___x_1368_);
lean_ctor_set(v___x_1366_, 0, v___x_1377_);
v___x_1379_ = v___x_1366_;
goto v_reusejp_1378_;
}
else
{
lean_object* v_reuseFailAlloc_1383_; 
v_reuseFailAlloc_1383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1383_, 0, v___x_1377_);
lean_ctor_set(v_reuseFailAlloc_1383_, 1, v___x_1368_);
v___x_1379_ = v_reuseFailAlloc_1383_;
goto v_reusejp_1378_;
}
v_reusejp_1378_:
{
lean_object* v___x_1381_; 
if (v_isShared_1375_ == 0)
{
lean_ctor_set(v___x_1374_, 0, v___x_1379_);
v___x_1381_ = v___x_1374_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v___x_1379_);
v___x_1381_ = v_reuseFailAlloc_1382_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
return v___x_1381_;
}
}
}
else
{
lean_del_object(v___x_1374_);
lean_dec(v_a_1372_);
lean_del_object(v___x_1366_);
v_a_1354_ = v___x_1369_;
goto v___jp_1353_;
}
}
}
else
{
lean_object* v_a_1385_; lean_object* v___x_1387_; uint8_t v_isShared_1388_; uint8_t v_isSharedCheck_1392_; 
lean_del_object(v___x_1366_);
lean_dec_ref(v_e_1340_);
v_a_1385_ = lean_ctor_get(v___x_1371_, 0);
v_isSharedCheck_1392_ = !lean_is_exclusive(v___x_1371_);
if (v_isSharedCheck_1392_ == 0)
{
v___x_1387_ = v___x_1371_;
v_isShared_1388_ = v_isSharedCheck_1392_;
goto v_resetjp_1386_;
}
else
{
lean_inc(v_a_1385_);
lean_dec(v___x_1371_);
v___x_1387_ = lean_box(0);
v_isShared_1388_ = v_isSharedCheck_1392_;
goto v_resetjp_1386_;
}
v_resetjp_1386_:
{
lean_object* v___x_1390_; 
if (v_isShared_1388_ == 0)
{
v___x_1390_ = v___x_1387_;
goto v_reusejp_1389_;
}
else
{
lean_object* v_reuseFailAlloc_1391_; 
v_reuseFailAlloc_1391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1391_, 0, v_a_1385_);
v___x_1390_ = v_reuseFailAlloc_1391_;
goto v_reusejp_1389_;
}
v_reusejp_1389_:
{
return v___x_1390_;
}
}
}
}
}
}
v___jp_1353_:
{
size_t v___x_1355_; size_t v___x_1356_; 
v___x_1355_ = ((size_t)1ULL);
v___x_1356_ = lean_usize_add(v_i_1343_, v___x_1355_);
lean_inc_ref(v_a_1354_);
v_i_1343_ = v___x_1356_;
v_b_1344_ = v_a_1354_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___boxed(lean_object* v_head_1394_, lean_object* v_e_1395_, lean_object* v_as_1396_, lean_object* v_sz_1397_, lean_object* v_i_1398_, lean_object* v_b_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_){
_start:
{
size_t v_sz_boxed_1408_; size_t v_i_boxed_1409_; lean_object* v_res_1410_; 
v_sz_boxed_1408_ = lean_unbox_usize(v_sz_1397_);
lean_dec(v_sz_1397_);
v_i_boxed_1409_ = lean_unbox_usize(v_i_1398_);
lean_dec(v_i_1398_);
v_res_1410_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1(v_head_1394_, v_e_1395_, v_as_1396_, v_sz_boxed_1408_, v_i_boxed_1409_, v_b_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_);
lean_dec(v___y_1406_);
lean_dec_ref(v___y_1405_);
lean_dec(v___y_1404_);
lean_dec_ref(v___y_1403_);
lean_dec(v___y_1402_);
lean_dec_ref(v___y_1401_);
lean_dec(v___y_1400_);
lean_dec_ref(v_as_1396_);
lean_dec(v_head_1394_);
return v_res_1410_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1(void){
_start:
{
lean_object* v___x_1413_; 
v___x_1413_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_1413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep(lean_object* v_head_1414_, lean_object* v_e_1415_, lean_object* v_a_1416_, lean_object* v_a_1417_, lean_object* v_a_1418_, lean_object* v_a_1419_, lean_object* v_a_1420_, lean_object* v_a_1421_, lean_object* v_a_1422_){
_start:
{
lean_object* v_a_1425_; lean_object* v___x_1458_; lean_object* v_env_1459_; lean_object* v___x_1460_; lean_object* v_ext_1461_; lean_object* v_toEnvExtension_1462_; lean_object* v_indexConfig_1463_; lean_object* v_asyncMode_1464_; lean_object* v_config_1465_; uint8_t v_trackZetaDelta_1466_; lean_object* v_zetaDeltaSet_1467_; lean_object* v_lctx_1468_; lean_object* v_localInstances_1469_; lean_object* v_defEqCtx_x3f_1470_; lean_object* v_synthPendingDepth_1471_; lean_object* v_customCanUnfoldPredicate_x3f_1472_; uint8_t v_univApprox_1473_; uint8_t v_inTypeClassResolution_1474_; uint8_t v_cacheInferType_1475_; uint64_t v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1458_ = lean_st_ref_get(v_a_1422_);
v_env_1459_ = lean_ctor_get(v___x_1458_, 0);
lean_inc_ref(v_env_1459_);
lean_dec(v___x_1458_);
v___x_1460_ = lp_mathlib_Mathlib_Tactic_Push_pullExt;
v_ext_1461_ = lean_ctor_get(v___x_1460_, 1);
v_toEnvExtension_1462_ = lean_ctor_get(v_ext_1461_, 0);
v_indexConfig_1463_ = lean_ctor_get(v_a_1417_, 5);
v_asyncMode_1464_ = lean_ctor_get(v_toEnvExtension_1462_, 2);
v_config_1465_ = lean_ctor_get(v_indexConfig_1463_, 0);
v_trackZetaDelta_1466_ = lean_ctor_get_uint8(v_a_1419_, sizeof(void*)*7);
v_zetaDeltaSet_1467_ = lean_ctor_get(v_a_1419_, 1);
v_lctx_1468_ = lean_ctor_get(v_a_1419_, 2);
v_localInstances_1469_ = lean_ctor_get(v_a_1419_, 3);
v_defEqCtx_x3f_1470_ = lean_ctor_get(v_a_1419_, 4);
v_synthPendingDepth_1471_ = lean_ctor_get(v_a_1419_, 5);
v_customCanUnfoldPredicate_x3f_1472_ = lean_ctor_get(v_a_1419_, 6);
v_univApprox_1473_ = lean_ctor_get_uint8(v_a_1419_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1474_ = lean_ctor_get_uint8(v_a_1419_, sizeof(void*)*7 + 2);
v_cacheInferType_1475_ = lean_ctor_get_uint8(v_a_1419_, sizeof(void*)*7 + 3);
v___x_1476_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_config_1465_);
v___x_1477_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1, &lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__1);
v___x_1478_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1477_, v___x_1460_, v_env_1459_, v_asyncMode_1464_);
lean_inc_ref(v_config_1465_);
v___x_1479_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1479_, 0, v_config_1465_);
lean_ctor_set_uint64(v___x_1479_, sizeof(void*)*1, v___x_1476_);
lean_inc(v_customCanUnfoldPredicate_x3f_1472_);
lean_inc(v_synthPendingDepth_1471_);
lean_inc(v_defEqCtx_x3f_1470_);
lean_inc_ref(v_localInstances_1469_);
lean_inc_ref(v_lctx_1468_);
lean_inc(v_zetaDeltaSet_1467_);
v___x_1480_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1480_, 0, v___x_1479_);
lean_ctor_set(v___x_1480_, 1, v_zetaDeltaSet_1467_);
lean_ctor_set(v___x_1480_, 2, v_lctx_1468_);
lean_ctor_set(v___x_1480_, 3, v_localInstances_1469_);
lean_ctor_set(v___x_1480_, 4, v_defEqCtx_x3f_1470_);
lean_ctor_set(v___x_1480_, 5, v_synthPendingDepth_1471_);
lean_ctor_set(v___x_1480_, 6, v_customCanUnfoldPredicate_x3f_1472_);
lean_ctor_set_uint8(v___x_1480_, sizeof(void*)*7, v_trackZetaDelta_1466_);
lean_ctor_set_uint8(v___x_1480_, sizeof(void*)*7 + 1, v_univApprox_1473_);
lean_ctor_set_uint8(v___x_1480_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1474_);
lean_ctor_set_uint8(v___x_1480_, sizeof(void*)*7 + 3, v_cacheInferType_1475_);
lean_inc_ref(v_e_1415_);
v___x_1481_ = l_Lean_Meta_DiscrTree_getMatchWithExtra___redArg(v___x_1478_, v_e_1415_, v___x_1480_, v_a_1420_, v_a_1421_, v_a_1422_);
lean_dec_ref_known(v___x_1480_, 7);
lean_dec(v___x_1478_);
if (lean_obj_tag(v___x_1481_) == 0)
{
lean_object* v_a_1482_; 
v_a_1482_ = lean_ctor_get(v___x_1481_, 0);
lean_inc(v_a_1482_);
lean_dec_ref_known(v___x_1481_, 1);
v_a_1425_ = v_a_1482_;
goto v___jp_1424_;
}
else
{
if (lean_obj_tag(v___x_1481_) == 0)
{
lean_object* v_a_1483_; 
v_a_1483_ = lean_ctor_get(v___x_1481_, 0);
lean_inc(v_a_1483_);
lean_dec_ref_known(v___x_1481_, 1);
v_a_1425_ = v_a_1483_;
goto v___jp_1424_;
}
else
{
lean_object* v_a_1484_; lean_object* v___x_1486_; uint8_t v_isShared_1487_; uint8_t v_isSharedCheck_1491_; 
lean_dec_ref(v_e_1415_);
v_a_1484_ = lean_ctor_get(v___x_1481_, 0);
v_isSharedCheck_1491_ = !lean_is_exclusive(v___x_1481_);
if (v_isSharedCheck_1491_ == 0)
{
v___x_1486_ = v___x_1481_;
v_isShared_1487_ = v_isSharedCheck_1491_;
goto v_resetjp_1485_;
}
else
{
lean_inc(v_a_1484_);
lean_dec(v___x_1481_);
v___x_1486_ = lean_box(0);
v_isShared_1487_ = v_isSharedCheck_1491_;
goto v_resetjp_1485_;
}
v_resetjp_1485_:
{
lean_object* v___x_1489_; 
if (v_isShared_1487_ == 0)
{
v___x_1489_ = v___x_1486_;
goto v_reusejp_1488_;
}
else
{
lean_object* v_reuseFailAlloc_1490_; 
v_reuseFailAlloc_1490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1490_, 0, v_a_1484_);
v___x_1489_ = v_reuseFailAlloc_1490_;
goto v_reusejp_1488_;
}
v_reusejp_1488_:
{
return v___x_1489_;
}
}
}
}
v___jp_1424_:
{
lean_object* v___x_1426_; lean_object* v___x_1427_; uint8_t v___x_1428_; 
v___x_1426_ = lean_array_get_size(v_a_1425_);
v___x_1427_ = lean_unsigned_to_nat(0u);
v___x_1428_ = lean_nat_dec_eq(v___x_1426_, v___x_1427_);
if (v___x_1428_ == 0)
{
lean_object* v___x_1429_; lean_object* v___x_1430_; size_t v_sz_1431_; size_t v___x_1432_; lean_object* v___x_1433_; 
v___x_1429_ = lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0(v_a_1425_, v___x_1427_, v___x_1426_);
v___x_1430_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1___closed__0));
v_sz_1431_ = lean_array_size(v___x_1429_);
v___x_1432_ = ((size_t)0ULL);
v___x_1433_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Push_pullStep_spec__1(v_head_1414_, v_e_1415_, v___x_1429_, v_sz_1431_, v___x_1432_, v___x_1430_, v_a_1416_, v_a_1417_, v_a_1418_, v_a_1419_, v_a_1420_, v_a_1421_, v_a_1422_);
lean_dec_ref(v___x_1429_);
if (lean_obj_tag(v___x_1433_) == 0)
{
lean_object* v_a_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1447_; 
v_a_1434_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1447_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1447_ == 0)
{
v___x_1436_ = v___x_1433_;
v_isShared_1437_ = v_isSharedCheck_1447_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_a_1434_);
lean_dec(v___x_1433_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1447_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v_fst_1438_; 
v_fst_1438_ = lean_ctor_get(v_a_1434_, 0);
lean_inc(v_fst_1438_);
lean_dec(v_a_1434_);
if (lean_obj_tag(v_fst_1438_) == 0)
{
lean_object* v___x_1439_; lean_object* v___x_1441_; 
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pullStep___closed__0));
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 0, v___x_1439_);
v___x_1441_ = v___x_1436_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v___x_1439_);
v___x_1441_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
return v___x_1441_;
}
}
else
{
lean_object* v_val_1443_; lean_object* v___x_1445_; 
v_val_1443_ = lean_ctor_get(v_fst_1438_, 0);
lean_inc(v_val_1443_);
lean_dec_ref_known(v_fst_1438_, 1);
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 0, v_val_1443_);
v___x_1445_ = v___x_1436_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v_val_1443_);
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
else
{
lean_object* v_a_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1455_; 
v_a_1448_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1455_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1455_ == 0)
{
v___x_1450_ = v___x_1433_;
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_a_1448_);
lean_dec(v___x_1433_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
lean_object* v___x_1453_; 
if (v_isShared_1451_ == 0)
{
v___x_1453_ = v___x_1450_;
goto v_reusejp_1452_;
}
else
{
lean_object* v_reuseFailAlloc_1454_; 
v_reuseFailAlloc_1454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1454_, 0, v_a_1448_);
v___x_1453_ = v_reuseFailAlloc_1454_;
goto v_reusejp_1452_;
}
v_reusejp_1452_:
{
return v___x_1453_;
}
}
}
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1457_; 
lean_dec_ref(v_a_1425_);
lean_dec_ref(v_e_1415_);
v___x_1456_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_pushNegBuiltin___closed__0));
v___x_1457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1457_, 0, v___x_1456_);
return v___x_1457_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullStep___boxed(lean_object* v_head_1492_, lean_object* v_e_1493_, lean_object* v_a_1494_, lean_object* v_a_1495_, lean_object* v_a_1496_, lean_object* v_a_1497_, lean_object* v_a_1498_, lean_object* v_a_1499_, lean_object* v_a_1500_, lean_object* v_a_1501_){
_start:
{
lean_object* v_res_1502_; 
v_res_1502_ = lp_mathlib_Mathlib_Tactic_Push_pullStep(v_head_1492_, v_e_1493_, v_a_1494_, v_a_1495_, v_a_1496_, v_a_1497_, v_a_1498_, v_a_1499_, v_a_1500_);
lean_dec(v_a_1500_);
lean_dec_ref(v_a_1499_);
lean_dec(v_a_1498_);
lean_dec_ref(v_a_1497_);
lean_dec(v_a_1496_);
lean_dec_ref(v_a_1495_);
lean_dec(v_a_1494_);
lean_dec(v_head_1492_);
return v_res_1502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0(lean_object* v_xs_1503_, lean_object* v_j_1504_, lean_object* v_h_1505_){
_start:
{
lean_object* v___x_1506_; 
v___x_1506_ = lp_mathlib___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00Mathlib_Tactic_Push_pullStep_spec__0_spec__0___redArg(v_xs_1503_, v_j_1504_);
return v___x_1506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCore(lean_object* v_head_1507_, lean_object* v_tgt_1508_, lean_object* v_disch_x3f_1509_, lean_object* v_a_1510_, lean_object* v_a_1511_, lean_object* v_a_1512_, lean_object* v_a_1513_){
_start:
{
lean_object* v___x_1515_; 
v___x_1515_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_1513_);
if (lean_obj_tag(v___x_1515_) == 0)
{
lean_object* v_a_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; 
v_a_1516_ = lean_ctor_get(v___x_1515_, 0);
lean_inc(v_a_1516_);
lean_dec_ref_known(v___x_1515_, 1);
v___x_1517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushSimpConfig));
v___x_1518_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__0));
v___x_1519_ = l_Lean_Options_empty;
v___x_1520_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_1517_, v___x_1518_, v_a_1516_, v___x_1519_, v_a_1510_, v_a_1512_, v_a_1513_);
if (lean_obj_tag(v___x_1520_) == 0)
{
lean_object* v_a_1521_; lean_object* v___y_1523_; 
v_a_1521_ = lean_ctor_get(v___x_1520_, 0);
lean_inc(v_a_1521_);
lean_dec_ref_known(v___x_1520_, 1);
if (lean_obj_tag(v_disch_x3f_1509_) == 0)
{
lean_object* v___f_1545_; lean_object* v___f_1546_; lean_object* v___f_1547_; lean_object* v___f_1548_; lean_object* v___x_1549_; uint8_t v___x_1550_; lean_object* v___x_1551_; 
v___f_1545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8));
v___f_1546_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9));
v___f_1547_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10));
v___f_1548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__11));
v___x_1549_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_pullStep___boxed), 10, 1);
lean_closure_set(v___x_1549_, 0, v_head_1507_);
v___x_1550_ = 1;
v___x_1551_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1551_, 0, v___f_1545_);
lean_ctor_set(v___x_1551_, 1, v___x_1549_);
lean_ctor_set(v___x_1551_, 2, v___f_1546_);
lean_ctor_set(v___x_1551_, 3, v___f_1547_);
lean_ctor_set(v___x_1551_, 4, v___f_1548_);
lean_ctor_set_uint8(v___x_1551_, sizeof(void*)*5, v___x_1550_);
v___y_1523_ = v___x_1551_;
goto v___jp_1522_;
}
else
{
lean_object* v_val_1552_; lean_object* v___f_1553_; lean_object* v___f_1554_; lean_object* v___f_1555_; lean_object* v___x_1556_; uint8_t v___x_1557_; lean_object* v___x_1558_; 
v_val_1552_ = lean_ctor_get(v_disch_x3f_1509_, 0);
v___f_1553_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__8));
v___f_1554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__9));
v___f_1555_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__10));
v___x_1556_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_pullStep___boxed), 10, 1);
lean_closure_set(v___x_1556_, 0, v_head_1507_);
v___x_1557_ = 0;
lean_inc(v_val_1552_);
v___x_1558_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1558_, 0, v___f_1553_);
lean_ctor_set(v___x_1558_, 1, v___x_1556_);
lean_ctor_set(v___x_1558_, 2, v___f_1554_);
lean_ctor_set(v___x_1558_, 3, v___f_1555_);
lean_ctor_set(v___x_1558_, 4, v_val_1552_);
lean_ctor_set_uint8(v___x_1558_, sizeof(void*)*5, v___x_1557_);
v___y_1523_ = v___x_1558_;
goto v___jp_1522_;
}
v___jp_1522_:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v___x_1524_ = lean_unsigned_to_nat(32u);
v___x_1525_ = lean_mk_empty_array_with_capacity(v___x_1524_);
lean_dec_ref(v___x_1525_);
v___x_1526_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7, &lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCore___closed__7);
v___x_1527_ = l_Lean_Meta_Simp_main(v_tgt_1508_, v_a_1521_, v___x_1526_, v___y_1523_, v_a_1510_, v_a_1511_, v_a_1512_, v_a_1513_);
if (lean_obj_tag(v___x_1527_) == 0)
{
lean_object* v_a_1528_; lean_object* v___x_1530_; uint8_t v_isShared_1531_; uint8_t v_isSharedCheck_1536_; 
v_a_1528_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1536_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1536_ == 0)
{
v___x_1530_ = v___x_1527_;
v_isShared_1531_ = v_isSharedCheck_1536_;
goto v_resetjp_1529_;
}
else
{
lean_inc(v_a_1528_);
lean_dec(v___x_1527_);
v___x_1530_ = lean_box(0);
v_isShared_1531_ = v_isSharedCheck_1536_;
goto v_resetjp_1529_;
}
v_resetjp_1529_:
{
lean_object* v_fst_1532_; lean_object* v___x_1534_; 
v_fst_1532_ = lean_ctor_get(v_a_1528_, 0);
lean_inc(v_fst_1532_);
lean_dec(v_a_1528_);
if (v_isShared_1531_ == 0)
{
lean_ctor_set(v___x_1530_, 0, v_fst_1532_);
v___x_1534_ = v___x_1530_;
goto v_reusejp_1533_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v_fst_1532_);
v___x_1534_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1533_;
}
v_reusejp_1533_:
{
return v___x_1534_;
}
}
}
else
{
lean_object* v_a_1537_; lean_object* v___x_1539_; uint8_t v_isShared_1540_; uint8_t v_isSharedCheck_1544_; 
v_a_1537_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1544_ == 0)
{
v___x_1539_ = v___x_1527_;
v_isShared_1540_ = v_isSharedCheck_1544_;
goto v_resetjp_1538_;
}
else
{
lean_inc(v_a_1537_);
lean_dec(v___x_1527_);
v___x_1539_ = lean_box(0);
v_isShared_1540_ = v_isSharedCheck_1544_;
goto v_resetjp_1538_;
}
v_resetjp_1538_:
{
lean_object* v___x_1542_; 
if (v_isShared_1540_ == 0)
{
v___x_1542_ = v___x_1539_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v_a_1537_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
}
}
else
{
lean_object* v_a_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1566_; 
lean_dec_ref(v_tgt_1508_);
lean_dec(v_head_1507_);
v_a_1559_ = lean_ctor_get(v___x_1520_, 0);
v_isSharedCheck_1566_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1561_ = v___x_1520_;
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_a_1559_);
lean_dec(v___x_1520_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v___x_1564_; 
if (v_isShared_1562_ == 0)
{
v___x_1564_ = v___x_1561_;
goto v_reusejp_1563_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v_a_1559_);
v___x_1564_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1563_;
}
v_reusejp_1563_:
{
return v___x_1564_;
}
}
}
}
else
{
lean_object* v_a_1567_; lean_object* v___x_1569_; uint8_t v_isShared_1570_; uint8_t v_isSharedCheck_1574_; 
lean_dec_ref(v_tgt_1508_);
lean_dec(v_head_1507_);
v_a_1567_ = lean_ctor_get(v___x_1515_, 0);
v_isSharedCheck_1574_ = !lean_is_exclusive(v___x_1515_);
if (v_isSharedCheck_1574_ == 0)
{
v___x_1569_ = v___x_1515_;
v_isShared_1570_ = v_isSharedCheck_1574_;
goto v_resetjp_1568_;
}
else
{
lean_inc(v_a_1567_);
lean_dec(v___x_1515_);
v___x_1569_ = lean_box(0);
v_isShared_1570_ = v_isSharedCheck_1574_;
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
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v_a_1567_);
v___x_1572_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
return v___x_1572_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_pullCore___boxed(lean_object* v_head_1575_, lean_object* v_tgt_1576_, lean_object* v_disch_x3f_1577_, lean_object* v_a_1578_, lean_object* v_a_1579_, lean_object* v_a_1580_, lean_object* v_a_1581_, lean_object* v_a_1582_){
_start:
{
lean_object* v_res_1583_; 
v_res_1583_ = lp_mathlib_Mathlib_Tactic_Push_pullCore(v_head_1575_, v_tgt_1576_, v_disch_x3f_1577_, v_a_1578_, v_a_1579_, v_a_1580_, v_a_1581_);
lean_dec(v_a_1581_);
lean_dec_ref(v_a_1580_);
lean_dec(v_a_1579_);
lean_dec_ref(v_a_1578_);
lean_dec(v_disch_x3f_1577_);
return v_res_1583_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Push_isUnderscore(lean_object* v_x_1605_){
_start:
{
lean_object* v___x_1606_; uint8_t v___x_1607_; uint8_t v___x_1608_; 
v___x_1606_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
lean_inc(v_x_1605_);
v___x_1607_ = l_Lean_Syntax_isOfKind(v_x_1605_, v___x_1606_);
v___x_1608_ = 1;
if (v___x_1607_ == 0)
{
lean_object* v___x_1609_; uint8_t v___x_1610_; 
v___x_1609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6));
lean_inc(v_x_1605_);
v___x_1610_ = l_Lean_Syntax_isOfKind(v_x_1605_, v___x_1609_);
if (v___x_1610_ == 0)
{
lean_dec(v_x_1605_);
return v___x_1610_;
}
else
{
lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; uint8_t v___x_1614_; 
v___x_1611_ = lean_unsigned_to_nat(1u);
v___x_1612_ = l_Lean_Syntax_getArg(v_x_1605_, v___x_1611_);
lean_dec(v_x_1605_);
v___x_1613_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8));
lean_inc(v___x_1612_);
v___x_1614_ = l_Lean_Syntax_isOfKind(v___x_1612_, v___x_1613_);
if (v___x_1614_ == 0)
{
lean_dec(v___x_1612_);
return v___x_1614_;
}
else
{
lean_object* v___x_1615_; lean_object* v___x_1616_; uint8_t v___x_1617_; 
v___x_1615_ = lean_unsigned_to_nat(0u);
v___x_1616_ = l_Lean_Syntax_getArg(v___x_1612_, v___x_1615_);
v___x_1617_ = l_Lean_Syntax_matchesNull(v___x_1616_, v___x_1611_);
if (v___x_1617_ == 0)
{
lean_dec(v___x_1612_);
return v___x_1617_;
}
else
{
lean_object* v___x_1618_; uint8_t v___x_1619_; 
v___x_1618_ = l_Lean_Syntax_getArg(v___x_1612_, v___x_1611_);
v___x_1619_ = l_Lean_Syntax_matchesNull(v___x_1618_, v___x_1615_);
if (v___x_1619_ == 0)
{
lean_dec(v___x_1612_);
return v___x_1619_;
}
else
{
lean_object* v___x_1620_; lean_object* v___x_1621_; uint8_t v___x_1622_; 
v___x_1620_ = lean_unsigned_to_nat(3u);
v___x_1621_ = l_Lean_Syntax_getArg(v___x_1612_, v___x_1620_);
lean_dec(v___x_1612_);
v___x_1622_ = l_Lean_Syntax_isOfKind(v___x_1621_, v___x_1606_);
if (v___x_1622_ == 0)
{
return v___x_1622_;
}
else
{
return v___x_1608_;
}
}
}
}
}
}
else
{
lean_dec(v_x_1605_);
return v___x_1608_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_isUnderscore___boxed(lean_object* v_x_1623_){
_start:
{
uint8_t v_res_1624_; lean_object* v_r_1625_; 
v_res_1624_ = lp_mathlib_Mathlib_Tactic_Push_isUnderscore(v_x_1623_);
v_r_1625_ = lean_box(v_res_1624_);
return v_r_1625_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0(lean_object* v_k_1632_){
_start:
{
lean_object* v___x_1633_; uint8_t v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___closed__1));
v___x_1634_ = lean_name_eq(v_k_1632_, v___x_1633_);
if (v___x_1634_ == 0)
{
uint8_t v___x_1635_; 
v___x_1635_ = 1;
return v___x_1635_;
}
else
{
uint8_t v___x_1636_; 
v___x_1636_ = 0;
return v___x_1636_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0___boxed(lean_object* v_k_1637_){
_start:
{
uint8_t v_res_1638_; lean_object* v_r_1639_; 
v_res_1638_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___lam__0(v_k_1637_);
lean_dec(v_k_1637_);
v_r_1639_ = lean_box(v_res_1638_);
return v_r_1639_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_1645_; lean_object* v___x_1646_; 
v___x_1645_ = l_Lean_maxRecDepthErrorMessage;
v___x_1646_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1645_);
return v___x_1646_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4(void){
_start:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; 
v___x_1647_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__3);
v___x_1648_ = l_Lean_MessageData_ofFormat(v___x_1647_);
return v___x_1648_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; 
v___x_1649_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__4);
v___x_1650_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__2));
v___x_1651_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1651_, 0, v___x_1650_);
lean_ctor_set(v___x_1651_, 1, v___x_1649_);
return v___x_1651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg(lean_object* v_ref_1652_){
_start:
{
lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; 
v___x_1654_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___closed__5);
v___x_1655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1655_, 0, v_ref_1652_);
lean_ctor_set(v___x_1655_, 1, v___x_1654_);
v___x_1656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1656_, 0, v___x_1655_);
return v___x_1656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg___boxed(lean_object* v_ref_1657_, lean_object* v___y_1658_){
_start:
{
lean_object* v_res_1659_; 
v_res_1659_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg(v_ref_1657_);
return v_res_1659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2(lean_object* v_env_1660_, lean_object* v_currNamespace_1661_, lean_object* v_openDecls_1662_, lean_object* v_n_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_){
_start:
{
lean_object* v___x_1666_; lean_object* v___x_1667_; 
v___x_1666_ = l_Lean_ResolveName_resolveNamespace(v_env_1660_, v_currNamespace_1661_, v_openDecls_1662_, v_n_1663_);
v___x_1667_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1667_, 0, v___x_1666_);
lean_ctor_set(v___x_1667_, 1, v___y_1665_);
return v___x_1667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2___boxed(lean_object* v_env_1668_, lean_object* v_currNamespace_1669_, lean_object* v_openDecls_1670_, lean_object* v_n_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_){
_start:
{
lean_object* v_res_1674_; 
v_res_1674_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2(v_env_1668_, v_currNamespace_1669_, v_openDecls_1670_, v_n_1671_, v___y_1672_, v___y_1673_);
lean_dec_ref(v___y_1672_);
return v_res_1674_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; 
v___x_1675_ = lean_box(0);
v___x_1676_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1677_, 0, v___x_1676_);
lean_ctor_set(v___x_1677_, 1, v___x_1675_);
return v___x_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg(){
_start:
{
lean_object* v___x_1679_; lean_object* v___x_1680_; 
v___x_1679_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0);
v___x_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1680_, 0, v___x_1679_);
return v___x_1680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___boxed(lean_object* v___y_1681_){
_start:
{
lean_object* v_res_1682_; 
v_res_1682_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg();
return v_res_1682_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1683_; double v___x_1684_; 
v___x_1683_ = lean_unsigned_to_nat(0u);
v___x_1684_ = lean_float_of_nat(v___x_1683_);
return v___x_1684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(lean_object* v_cls_1688_, lean_object* v_msg_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_){
_start:
{
lean_object* v_ref_1695_; lean_object* v___x_1696_; lean_object* v_a_1697_; lean_object* v___x_1699_; uint8_t v_isShared_1700_; uint8_t v_isSharedCheck_1741_; 
v_ref_1695_ = lean_ctor_get(v___y_1692_, 5);
v___x_1696_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_1689_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_);
v_a_1697_ = lean_ctor_get(v___x_1696_, 0);
v_isSharedCheck_1741_ = !lean_is_exclusive(v___x_1696_);
if (v_isSharedCheck_1741_ == 0)
{
v___x_1699_ = v___x_1696_;
v_isShared_1700_ = v_isSharedCheck_1741_;
goto v_resetjp_1698_;
}
else
{
lean_inc(v_a_1697_);
lean_dec(v___x_1696_);
v___x_1699_ = lean_box(0);
v_isShared_1700_ = v_isSharedCheck_1741_;
goto v_resetjp_1698_;
}
v_resetjp_1698_:
{
lean_object* v___x_1701_; lean_object* v_traceState_1702_; lean_object* v_env_1703_; lean_object* v_nextMacroScope_1704_; lean_object* v_ngen_1705_; lean_object* v_auxDeclNGen_1706_; lean_object* v_cache_1707_; lean_object* v_messages_1708_; lean_object* v_infoState_1709_; lean_object* v_snapshotTasks_1710_; lean_object* v___x_1712_; uint8_t v_isShared_1713_; uint8_t v_isSharedCheck_1740_; 
v___x_1701_ = lean_st_ref_take(v___y_1693_);
v_traceState_1702_ = lean_ctor_get(v___x_1701_, 4);
v_env_1703_ = lean_ctor_get(v___x_1701_, 0);
v_nextMacroScope_1704_ = lean_ctor_get(v___x_1701_, 1);
v_ngen_1705_ = lean_ctor_get(v___x_1701_, 2);
v_auxDeclNGen_1706_ = lean_ctor_get(v___x_1701_, 3);
v_cache_1707_ = lean_ctor_get(v___x_1701_, 5);
v_messages_1708_ = lean_ctor_get(v___x_1701_, 6);
v_infoState_1709_ = lean_ctor_get(v___x_1701_, 7);
v_snapshotTasks_1710_ = lean_ctor_get(v___x_1701_, 8);
v_isSharedCheck_1740_ = !lean_is_exclusive(v___x_1701_);
if (v_isSharedCheck_1740_ == 0)
{
v___x_1712_ = v___x_1701_;
v_isShared_1713_ = v_isSharedCheck_1740_;
goto v_resetjp_1711_;
}
else
{
lean_inc(v_snapshotTasks_1710_);
lean_inc(v_infoState_1709_);
lean_inc(v_messages_1708_);
lean_inc(v_cache_1707_);
lean_inc(v_traceState_1702_);
lean_inc(v_auxDeclNGen_1706_);
lean_inc(v_ngen_1705_);
lean_inc(v_nextMacroScope_1704_);
lean_inc(v_env_1703_);
lean_dec(v___x_1701_);
v___x_1712_ = lean_box(0);
v_isShared_1713_ = v_isSharedCheck_1740_;
goto v_resetjp_1711_;
}
v_resetjp_1711_:
{
uint64_t v_tid_1714_; lean_object* v_traces_1715_; lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1739_; 
v_tid_1714_ = lean_ctor_get_uint64(v_traceState_1702_, sizeof(void*)*1);
v_traces_1715_ = lean_ctor_get(v_traceState_1702_, 0);
v_isSharedCheck_1739_ = !lean_is_exclusive(v_traceState_1702_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1717_ = v_traceState_1702_;
v_isShared_1718_ = v_isSharedCheck_1739_;
goto v_resetjp_1716_;
}
else
{
lean_inc(v_traces_1715_);
lean_dec(v_traceState_1702_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1739_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
lean_object* v___x_1719_; double v___x_1720_; uint8_t v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1729_; 
v___x_1719_ = lean_box(0);
v___x_1720_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__0);
v___x_1721_ = 0;
v___x_1722_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
v___x_1723_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1723_, 0, v_cls_1688_);
lean_ctor_set(v___x_1723_, 1, v___x_1719_);
lean_ctor_set(v___x_1723_, 2, v___x_1722_);
lean_ctor_set_float(v___x_1723_, sizeof(void*)*3, v___x_1720_);
lean_ctor_set_float(v___x_1723_, sizeof(void*)*3 + 8, v___x_1720_);
lean_ctor_set_uint8(v___x_1723_, sizeof(void*)*3 + 16, v___x_1721_);
v___x_1724_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__2));
v___x_1725_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1725_, 0, v___x_1723_);
lean_ctor_set(v___x_1725_, 1, v_a_1697_);
lean_ctor_set(v___x_1725_, 2, v___x_1724_);
lean_inc(v_ref_1695_);
v___x_1726_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1726_, 0, v_ref_1695_);
lean_ctor_set(v___x_1726_, 1, v___x_1725_);
v___x_1727_ = l_Lean_PersistentArray_push___redArg(v_traces_1715_, v___x_1726_);
if (v_isShared_1718_ == 0)
{
lean_ctor_set(v___x_1717_, 0, v___x_1727_);
v___x_1729_ = v___x_1717_;
goto v_reusejp_1728_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v___x_1727_);
lean_ctor_set_uint64(v_reuseFailAlloc_1738_, sizeof(void*)*1, v_tid_1714_);
v___x_1729_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1728_;
}
v_reusejp_1728_:
{
lean_object* v___x_1731_; 
if (v_isShared_1713_ == 0)
{
lean_ctor_set(v___x_1712_, 4, v___x_1729_);
v___x_1731_ = v___x_1712_;
goto v_reusejp_1730_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_env_1703_);
lean_ctor_set(v_reuseFailAlloc_1737_, 1, v_nextMacroScope_1704_);
lean_ctor_set(v_reuseFailAlloc_1737_, 2, v_ngen_1705_);
lean_ctor_set(v_reuseFailAlloc_1737_, 3, v_auxDeclNGen_1706_);
lean_ctor_set(v_reuseFailAlloc_1737_, 4, v___x_1729_);
lean_ctor_set(v_reuseFailAlloc_1737_, 5, v_cache_1707_);
lean_ctor_set(v_reuseFailAlloc_1737_, 6, v_messages_1708_);
lean_ctor_set(v_reuseFailAlloc_1737_, 7, v_infoState_1709_);
lean_ctor_set(v_reuseFailAlloc_1737_, 8, v_snapshotTasks_1710_);
v___x_1731_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1730_;
}
v_reusejp_1730_:
{
lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1735_; 
v___x_1732_ = lean_st_ref_set(v___y_1693_, v___x_1731_);
v___x_1733_ = lean_box(0);
if (v_isShared_1700_ == 0)
{
lean_ctor_set(v___x_1699_, 0, v___x_1733_);
v___x_1735_ = v___x_1699_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v___x_1733_);
v___x_1735_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
return v___x_1735_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_cls_1742_, lean_object* v_msg_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_){
_start:
{
lean_object* v_res_1749_; 
v_res_1749_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(v_cls_1742_, v_msg_1743_, v___y_1744_, v___y_1745_, v___y_1746_, v___y_1747_);
lean_dec(v___y_1747_);
lean_dec_ref(v___y_1746_);
lean_dec(v___y_1745_);
lean_dec_ref(v___y_1744_);
return v_res_1749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4(lean_object* v_as_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_){
_start:
{
if (lean_obj_tag(v_as_1753_) == 0)
{
lean_object* v___x_1761_; lean_object* v___x_1762_; 
v___x_1761_ = lean_box(0);
v___x_1762_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1762_, 0, v___x_1761_);
return v___x_1762_;
}
else
{
lean_object* v_options_1763_; uint8_t v_hasTrace_1764_; 
v_options_1763_ = lean_ctor_get(v___y_1758_, 2);
v_hasTrace_1764_ = lean_ctor_get_uint8(v_options_1763_, sizeof(void*)*1);
if (v_hasTrace_1764_ == 0)
{
lean_object* v_tail_1765_; 
v_tail_1765_ = lean_ctor_get(v_as_1753_, 1);
lean_inc(v_tail_1765_);
lean_dec_ref_known(v_as_1753_, 2);
v_as_1753_ = v_tail_1765_;
goto _start;
}
else
{
lean_object* v_head_1767_; lean_object* v_tail_1768_; lean_object* v_fst_1769_; lean_object* v_snd_1770_; lean_object* v_inheritedTraceOptions_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; uint8_t v___x_1774_; 
v_head_1767_ = lean_ctor_get(v_as_1753_, 0);
lean_inc(v_head_1767_);
v_tail_1768_ = lean_ctor_get(v_as_1753_, 1);
lean_inc(v_tail_1768_);
lean_dec_ref_known(v_as_1753_, 2);
v_fst_1769_ = lean_ctor_get(v_head_1767_, 0);
lean_inc_n(v_fst_1769_, 2);
v_snd_1770_ = lean_ctor_get(v_head_1767_, 1);
lean_inc(v_snd_1770_);
lean_dec(v_head_1767_);
v_inheritedTraceOptions_1771_ = lean_ctor_get(v___y_1758_, 13);
v___x_1772_ = ((lean_object*)(lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__1));
v___x_1773_ = l_Lean_Name_append(v___x_1772_, v_fst_1769_);
v___x_1774_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1771_, v_options_1763_, v___x_1773_);
lean_dec(v___x_1773_);
if (v___x_1774_ == 0)
{
lean_dec(v_snd_1770_);
lean_dec(v_fst_1769_);
v_as_1753_ = v_tail_1768_;
goto _start;
}
else
{
lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; 
v___x_1776_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1776_, 0, v_snd_1770_);
v___x_1777_ = l_Lean_MessageData_ofFormat(v___x_1776_);
v___x_1778_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(v_fst_1769_, v___x_1777_, v___y_1756_, v___y_1757_, v___y_1758_, v___y_1759_);
if (lean_obj_tag(v___x_1778_) == 0)
{
lean_dec_ref_known(v___x_1778_, 1);
v_as_1753_ = v_tail_1768_;
goto _start;
}
else
{
lean_dec(v_tail_1768_);
return v___x_1778_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___boxed(lean_object* v_as_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v_res_1788_; 
v_res_1788_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4(v_as_1780_, v___y_1781_, v___y_1782_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec(v___y_1784_);
lean_dec_ref(v___y_1783_);
lean_dec(v___y_1782_);
lean_dec_ref(v___y_1781_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4(lean_object* v_env_1789_, lean_object* v_options_1790_, lean_object* v_currNamespace_1791_, lean_object* v_openDecls_1792_, lean_object* v_n_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_){
_start:
{
lean_object* v___x_1796_; lean_object* v___x_1797_; 
v___x_1796_ = l_Lean_ResolveName_resolveGlobalName(v_env_1789_, v_options_1790_, v_currNamespace_1791_, v_openDecls_1792_, v_n_1793_);
v___x_1797_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1797_, 0, v___x_1796_);
lean_ctor_set(v___x_1797_, 1, v___y_1795_);
return v___x_1797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4___boxed(lean_object* v_env_1798_, lean_object* v_options_1799_, lean_object* v_currNamespace_1800_, lean_object* v_openDecls_1801_, lean_object* v_n_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_){
_start:
{
lean_object* v_res_1805_; 
v_res_1805_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4(v_env_1798_, v_options_1799_, v_currNamespace_1800_, v_openDecls_1801_, v_n_1802_, v___y_1803_, v___y_1804_);
lean_dec_ref(v___y_1803_);
lean_dec_ref(v_options_1799_);
return v_res_1805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg(lean_object* v_ref_1806_, lean_object* v_msg_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_){
_start:
{
lean_object* v_fileName_1815_; lean_object* v_fileMap_1816_; lean_object* v_options_1817_; lean_object* v_currRecDepth_1818_; lean_object* v_maxRecDepth_1819_; lean_object* v_ref_1820_; lean_object* v_currNamespace_1821_; lean_object* v_openDecls_1822_; lean_object* v_initHeartbeats_1823_; lean_object* v_maxHeartbeats_1824_; lean_object* v_quotContext_1825_; lean_object* v_currMacroScope_1826_; uint8_t v_diag_1827_; lean_object* v_cancelTk_x3f_1828_; uint8_t v_suppressElabErrors_1829_; lean_object* v_inheritedTraceOptions_1830_; lean_object* v_ref_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; 
v_fileName_1815_ = lean_ctor_get(v___y_1812_, 0);
v_fileMap_1816_ = lean_ctor_get(v___y_1812_, 1);
v_options_1817_ = lean_ctor_get(v___y_1812_, 2);
v_currRecDepth_1818_ = lean_ctor_get(v___y_1812_, 3);
v_maxRecDepth_1819_ = lean_ctor_get(v___y_1812_, 4);
v_ref_1820_ = lean_ctor_get(v___y_1812_, 5);
v_currNamespace_1821_ = lean_ctor_get(v___y_1812_, 6);
v_openDecls_1822_ = lean_ctor_get(v___y_1812_, 7);
v_initHeartbeats_1823_ = lean_ctor_get(v___y_1812_, 8);
v_maxHeartbeats_1824_ = lean_ctor_get(v___y_1812_, 9);
v_quotContext_1825_ = lean_ctor_get(v___y_1812_, 10);
v_currMacroScope_1826_ = lean_ctor_get(v___y_1812_, 11);
v_diag_1827_ = lean_ctor_get_uint8(v___y_1812_, sizeof(void*)*14);
v_cancelTk_x3f_1828_ = lean_ctor_get(v___y_1812_, 12);
v_suppressElabErrors_1829_ = lean_ctor_get_uint8(v___y_1812_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1830_ = lean_ctor_get(v___y_1812_, 13);
v_ref_1831_ = l_Lean_replaceRef(v_ref_1806_, v_ref_1820_);
lean_inc_ref(v_inheritedTraceOptions_1830_);
lean_inc(v_cancelTk_x3f_1828_);
lean_inc(v_currMacroScope_1826_);
lean_inc(v_quotContext_1825_);
lean_inc(v_maxHeartbeats_1824_);
lean_inc(v_initHeartbeats_1823_);
lean_inc(v_openDecls_1822_);
lean_inc(v_currNamespace_1821_);
lean_inc(v_maxRecDepth_1819_);
lean_inc(v_currRecDepth_1818_);
lean_inc_ref(v_options_1817_);
lean_inc_ref(v_fileMap_1816_);
lean_inc_ref(v_fileName_1815_);
v___x_1832_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1832_, 0, v_fileName_1815_);
lean_ctor_set(v___x_1832_, 1, v_fileMap_1816_);
lean_ctor_set(v___x_1832_, 2, v_options_1817_);
lean_ctor_set(v___x_1832_, 3, v_currRecDepth_1818_);
lean_ctor_set(v___x_1832_, 4, v_maxRecDepth_1819_);
lean_ctor_set(v___x_1832_, 5, v_ref_1831_);
lean_ctor_set(v___x_1832_, 6, v_currNamespace_1821_);
lean_ctor_set(v___x_1832_, 7, v_openDecls_1822_);
lean_ctor_set(v___x_1832_, 8, v_initHeartbeats_1823_);
lean_ctor_set(v___x_1832_, 9, v_maxHeartbeats_1824_);
lean_ctor_set(v___x_1832_, 10, v_quotContext_1825_);
lean_ctor_set(v___x_1832_, 11, v_currMacroScope_1826_);
lean_ctor_set(v___x_1832_, 12, v_cancelTk_x3f_1828_);
lean_ctor_set(v___x_1832_, 13, v_inheritedTraceOptions_1830_);
lean_ctor_set_uint8(v___x_1832_, sizeof(void*)*14, v_diag_1827_);
lean_ctor_set_uint8(v___x_1832_, sizeof(void*)*14 + 1, v_suppressElabErrors_1829_);
v___x_1833_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_1807_, v___y_1808_, v___y_1809_, v___y_1810_, v___y_1811_, v___x_1832_, v___y_1813_);
lean_dec_ref_known(v___x_1832_, 14);
return v___x_1833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg___boxed(lean_object* v_ref_1834_, lean_object* v_msg_1835_, lean_object* v___y_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_){
_start:
{
lean_object* v_res_1843_; 
v_res_1843_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg(v_ref_1834_, v_msg_1835_, v___y_1836_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_, v___y_1841_);
lean_dec(v___y_1841_);
lean_dec_ref(v___y_1840_);
lean_dec(v___y_1839_);
lean_dec_ref(v___y_1838_);
lean_dec(v___y_1837_);
lean_dec_ref(v___y_1836_);
lean_dec(v_ref_1834_);
return v_res_1843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1(lean_object* v_env_1844_, lean_object* v_declName_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_){
_start:
{
uint8_t v___x_1848_; lean_object* v_env_1849_; lean_object* v___x_1850_; uint8_t v___x_1851_; uint8_t v___x_1852_; 
v___x_1848_ = 0;
v_env_1849_ = l_Lean_Environment_setExporting(v_env_1844_, v___x_1848_);
lean_inc(v_declName_1845_);
v___x_1850_ = l_Lean_mkPrivateName(v_env_1849_, v_declName_1845_);
v___x_1851_ = 1;
lean_inc_ref(v_env_1849_);
v___x_1852_ = l_Lean_Environment_contains(v_env_1849_, v___x_1850_, v___x_1851_);
if (v___x_1852_ == 0)
{
lean_object* v___x_1853_; uint8_t v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; 
v___x_1853_ = l_Lean_privateToUserName(v_declName_1845_);
v___x_1854_ = l_Lean_Environment_contains(v_env_1849_, v___x_1853_, v___x_1851_);
v___x_1855_ = lean_box(v___x_1854_);
v___x_1856_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1856_, 0, v___x_1855_);
lean_ctor_set(v___x_1856_, 1, v___y_1847_);
return v___x_1856_;
}
else
{
lean_object* v___x_1857_; lean_object* v___x_1858_; 
lean_dec_ref(v_env_1849_);
lean_dec(v_declName_1845_);
v___x_1857_ = lean_box(v___x_1852_);
v___x_1858_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1858_, 0, v___x_1857_);
lean_ctor_set(v___x_1858_, 1, v___y_1847_);
return v___x_1858_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1___boxed(lean_object* v_env_1859_, lean_object* v_declName_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
lean_object* v_res_1863_; 
v_res_1863_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1(v_env_1859_, v_declName_1860_, v___y_1861_, v___y_1862_);
lean_dec_ref(v___y_1861_);
return v_res_1863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(lean_object* v_x_1864_, lean_object* v___y_1865_){
_start:
{
if (lean_obj_tag(v_x_1864_) == 0)
{
lean_object* v_a_1866_; lean_object* v___x_1867_; 
v_a_1866_ = lean_ctor_get(v_x_1864_, 0);
lean_inc(v_a_1866_);
v___x_1867_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1867_, 0, v_a_1866_);
lean_ctor_set(v___x_1867_, 1, v___y_1865_);
return v___x_1867_;
}
else
{
lean_object* v_a_1868_; lean_object* v___x_1869_; 
v_a_1868_ = lean_ctor_get(v_x_1864_, 0);
lean_inc(v_a_1868_);
v___x_1869_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1869_, 0, v_a_1868_);
lean_ctor_set(v___x_1869_, 1, v___y_1865_);
return v___x_1869_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg___boxed(lean_object* v_x_1870_, lean_object* v___y_1871_){
_start:
{
lean_object* v_res_1872_; 
v_res_1872_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(v_x_1870_, v___y_1871_);
lean_dec_ref(v_x_1870_);
return v_res_1872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0(lean_object* v_env_1873_, lean_object* v_stx_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_){
_start:
{
lean_object* v___x_1877_; 
v___x_1877_ = l_Lean_Elab_expandMacroImpl_x3f(v_env_1873_, v_stx_1874_, v___y_1875_, v___y_1876_);
if (lean_obj_tag(v___x_1877_) == 0)
{
lean_object* v_a_1878_; 
v_a_1878_ = lean_ctor_get(v___x_1877_, 0);
lean_inc(v_a_1878_);
if (lean_obj_tag(v_a_1878_) == 0)
{
lean_object* v_a_1879_; lean_object* v___x_1881_; uint8_t v_isShared_1882_; uint8_t v_isSharedCheck_1887_; 
v_a_1879_ = lean_ctor_get(v___x_1877_, 1);
v_isSharedCheck_1887_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1887_ == 0)
{
lean_object* v_unused_1888_; 
v_unused_1888_ = lean_ctor_get(v___x_1877_, 0);
lean_dec(v_unused_1888_);
v___x_1881_ = v___x_1877_;
v_isShared_1882_ = v_isSharedCheck_1887_;
goto v_resetjp_1880_;
}
else
{
lean_inc(v_a_1879_);
lean_dec(v___x_1877_);
v___x_1881_ = lean_box(0);
v_isShared_1882_ = v_isSharedCheck_1887_;
goto v_resetjp_1880_;
}
v_resetjp_1880_:
{
lean_object* v___x_1883_; lean_object* v___x_1885_; 
v___x_1883_ = lean_box(0);
if (v_isShared_1882_ == 0)
{
lean_ctor_set(v___x_1881_, 0, v___x_1883_);
v___x_1885_ = v___x_1881_;
goto v_reusejp_1884_;
}
else
{
lean_object* v_reuseFailAlloc_1886_; 
v_reuseFailAlloc_1886_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1886_, 0, v___x_1883_);
lean_ctor_set(v_reuseFailAlloc_1886_, 1, v_a_1879_);
v___x_1885_ = v_reuseFailAlloc_1886_;
goto v_reusejp_1884_;
}
v_reusejp_1884_:
{
return v___x_1885_;
}
}
}
else
{
lean_object* v_val_1889_; lean_object* v___x_1891_; uint8_t v_isShared_1892_; uint8_t v_isSharedCheck_1917_; 
v_val_1889_ = lean_ctor_get(v_a_1878_, 0);
v_isSharedCheck_1917_ = !lean_is_exclusive(v_a_1878_);
if (v_isSharedCheck_1917_ == 0)
{
v___x_1891_ = v_a_1878_;
v_isShared_1892_ = v_isSharedCheck_1917_;
goto v_resetjp_1890_;
}
else
{
lean_inc(v_val_1889_);
lean_dec(v_a_1878_);
v___x_1891_ = lean_box(0);
v_isShared_1892_ = v_isSharedCheck_1917_;
goto v_resetjp_1890_;
}
v_resetjp_1890_:
{
lean_object* v_snd_1893_; 
v_snd_1893_ = lean_ctor_get(v_val_1889_, 1);
lean_inc(v_snd_1893_);
lean_dec(v_val_1889_);
if (lean_obj_tag(v_snd_1893_) == 0)
{
lean_object* v_a_1894_; lean_object* v_a_1895_; lean_object* v___x_1897_; uint8_t v_isShared_1898_; uint8_t v_isSharedCheck_1903_; 
lean_del_object(v___x_1891_);
v_a_1894_ = lean_ctor_get(v___x_1877_, 1);
lean_inc(v_a_1894_);
lean_dec_ref_known(v___x_1877_, 2);
v_a_1895_ = lean_ctor_get(v_snd_1893_, 0);
v_isSharedCheck_1903_ = !lean_is_exclusive(v_snd_1893_);
if (v_isSharedCheck_1903_ == 0)
{
v___x_1897_ = v_snd_1893_;
v_isShared_1898_ = v_isSharedCheck_1903_;
goto v_resetjp_1896_;
}
else
{
lean_inc(v_a_1895_);
lean_dec(v_snd_1893_);
v___x_1897_ = lean_box(0);
v_isShared_1898_ = v_isSharedCheck_1903_;
goto v_resetjp_1896_;
}
v_resetjp_1896_:
{
lean_object* v___x_1900_; 
if (v_isShared_1898_ == 0)
{
v___x_1900_ = v___x_1897_;
goto v_reusejp_1899_;
}
else
{
lean_object* v_reuseFailAlloc_1902_; 
v_reuseFailAlloc_1902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1902_, 0, v_a_1895_);
v___x_1900_ = v_reuseFailAlloc_1902_;
goto v_reusejp_1899_;
}
v_reusejp_1899_:
{
lean_object* v___x_1901_; 
v___x_1901_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(v___x_1900_, v_a_1894_);
lean_dec_ref(v___x_1900_);
return v___x_1901_;
}
}
}
else
{
lean_object* v_a_1904_; lean_object* v_a_1905_; lean_object* v___x_1907_; uint8_t v_isShared_1908_; uint8_t v_isSharedCheck_1916_; 
v_a_1904_ = lean_ctor_get(v___x_1877_, 1);
lean_inc(v_a_1904_);
lean_dec_ref_known(v___x_1877_, 2);
v_a_1905_ = lean_ctor_get(v_snd_1893_, 0);
v_isSharedCheck_1916_ = !lean_is_exclusive(v_snd_1893_);
if (v_isSharedCheck_1916_ == 0)
{
v___x_1907_ = v_snd_1893_;
v_isShared_1908_ = v_isSharedCheck_1916_;
goto v_resetjp_1906_;
}
else
{
lean_inc(v_a_1905_);
lean_dec(v_snd_1893_);
v___x_1907_ = lean_box(0);
v_isShared_1908_ = v_isSharedCheck_1916_;
goto v_resetjp_1906_;
}
v_resetjp_1906_:
{
lean_object* v___x_1910_; 
if (v_isShared_1892_ == 0)
{
lean_ctor_set(v___x_1891_, 0, v_a_1905_);
v___x_1910_ = v___x_1891_;
goto v_reusejp_1909_;
}
else
{
lean_object* v_reuseFailAlloc_1915_; 
v_reuseFailAlloc_1915_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1915_, 0, v_a_1905_);
v___x_1910_ = v_reuseFailAlloc_1915_;
goto v_reusejp_1909_;
}
v_reusejp_1909_:
{
lean_object* v___x_1912_; 
if (v_isShared_1908_ == 0)
{
lean_ctor_set(v___x_1907_, 0, v___x_1910_);
v___x_1912_ = v___x_1907_;
goto v_reusejp_1911_;
}
else
{
lean_object* v_reuseFailAlloc_1914_; 
v_reuseFailAlloc_1914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1914_, 0, v___x_1910_);
v___x_1912_ = v_reuseFailAlloc_1914_;
goto v_reusejp_1911_;
}
v_reusejp_1911_:
{
lean_object* v___x_1913_; 
v___x_1913_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(v___x_1912_, v_a_1904_);
lean_dec_ref(v___x_1912_);
return v___x_1913_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1918_; lean_object* v_a_1919_; lean_object* v___x_1921_; uint8_t v_isShared_1922_; uint8_t v_isSharedCheck_1926_; 
v_a_1918_ = lean_ctor_get(v___x_1877_, 0);
v_a_1919_ = lean_ctor_get(v___x_1877_, 1);
v_isSharedCheck_1926_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1926_ == 0)
{
v___x_1921_ = v___x_1877_;
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
else
{
lean_inc(v_a_1919_);
lean_inc(v_a_1918_);
lean_dec(v___x_1877_);
v___x_1921_ = lean_box(0);
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
v_resetjp_1920_:
{
lean_object* v___x_1924_; 
if (v_isShared_1922_ == 0)
{
v___x_1924_ = v___x_1921_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_1925_; 
v_reuseFailAlloc_1925_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1925_, 0, v_a_1918_);
lean_ctor_set(v_reuseFailAlloc_1925_, 1, v_a_1919_);
v___x_1924_ = v_reuseFailAlloc_1925_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
return v___x_1924_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0___boxed(lean_object* v_env_1927_, lean_object* v_stx_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_){
_start:
{
lean_object* v_res_1931_; 
v_res_1931_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0(v_env_1927_, v_stx_1928_, v___y_1929_, v___y_1930_);
lean_dec_ref(v___y_1929_);
return v_res_1931_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg(lean_object* v_keys_1932_, lean_object* v_i_1933_, lean_object* v_k_1934_){
_start:
{
lean_object* v___x_1935_; uint8_t v___x_1936_; 
v___x_1935_ = lean_array_get_size(v_keys_1932_);
v___x_1936_ = lean_nat_dec_lt(v_i_1933_, v___x_1935_);
if (v___x_1936_ == 0)
{
lean_dec(v_i_1933_);
return v___x_1936_;
}
else
{
lean_object* v_k_x27_1937_; uint8_t v___x_1938_; 
v_k_x27_1937_ = lean_array_fget_borrowed(v_keys_1932_, v_i_1933_);
v___x_1938_ = l_Lean_instBEqExtraModUse_beq(v_k_1934_, v_k_x27_1937_);
if (v___x_1938_ == 0)
{
lean_object* v___x_1939_; lean_object* v___x_1940_; 
v___x_1939_ = lean_unsigned_to_nat(1u);
v___x_1940_ = lean_nat_add(v_i_1933_, v___x_1939_);
lean_dec(v_i_1933_);
v_i_1933_ = v___x_1940_;
goto _start;
}
else
{
lean_dec(v_i_1933_);
return v___x_1938_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg___boxed(lean_object* v_keys_1942_, lean_object* v_i_1943_, lean_object* v_k_1944_){
_start:
{
uint8_t v_res_1945_; lean_object* v_r_1946_; 
v_res_1945_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg(v_keys_1942_, v_i_1943_, v_k_1944_);
lean_dec_ref(v_k_1944_);
lean_dec_ref(v_keys_1942_);
v_r_1946_ = lean_box(v_res_1945_);
return v_r_1946_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg(lean_object* v_x_1947_, size_t v_x_1948_, lean_object* v_x_1949_){
_start:
{
if (lean_obj_tag(v_x_1947_) == 0)
{
lean_object* v_es_1950_; lean_object* v___x_1951_; size_t v___x_1952_; size_t v___x_1953_; lean_object* v_j_1954_; lean_object* v___x_1955_; 
v_es_1950_ = lean_ctor_get(v_x_1947_, 0);
v___x_1951_ = lean_box(2);
v___x_1952_ = ((size_t)31ULL);
v___x_1953_ = lean_usize_land(v_x_1948_, v___x_1952_);
v_j_1954_ = lean_usize_to_nat(v___x_1953_);
v___x_1955_ = lean_array_get_borrowed(v___x_1951_, v_es_1950_, v_j_1954_);
lean_dec(v_j_1954_);
switch(lean_obj_tag(v___x_1955_))
{
case 0:
{
lean_object* v_key_1956_; uint8_t v___x_1957_; 
v_key_1956_ = lean_ctor_get(v___x_1955_, 0);
v___x_1957_ = l_Lean_instBEqExtraModUse_beq(v_x_1949_, v_key_1956_);
return v___x_1957_;
}
case 1:
{
lean_object* v_node_1958_; size_t v___x_1959_; size_t v___x_1960_; 
v_node_1958_ = lean_ctor_get(v___x_1955_, 0);
v___x_1959_ = ((size_t)5ULL);
v___x_1960_ = lean_usize_shift_right(v_x_1948_, v___x_1959_);
v_x_1947_ = v_node_1958_;
v_x_1948_ = v___x_1960_;
goto _start;
}
default: 
{
uint8_t v___x_1962_; 
v___x_1962_ = 0;
return v___x_1962_;
}
}
}
else
{
lean_object* v_ks_1963_; lean_object* v___x_1964_; uint8_t v___x_1965_; 
v_ks_1963_ = lean_ctor_get(v_x_1947_, 0);
v___x_1964_ = lean_unsigned_to_nat(0u);
v___x_1965_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg(v_ks_1963_, v___x_1964_, v_x_1949_);
return v___x_1965_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg___boxed(lean_object* v_x_1966_, lean_object* v_x_1967_, lean_object* v_x_1968_){
_start:
{
size_t v_x_52034__boxed_1969_; uint8_t v_res_1970_; lean_object* v_r_1971_; 
v_x_52034__boxed_1969_ = lean_unbox_usize(v_x_1967_);
lean_dec(v_x_1967_);
v_res_1970_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg(v_x_1966_, v_x_52034__boxed_1969_, v_x_1968_);
lean_dec_ref(v_x_1968_);
lean_dec_ref(v_x_1966_);
v_r_1971_ = lean_box(v_res_1970_);
return v_r_1971_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg(lean_object* v_x_1972_, lean_object* v_x_1973_){
_start:
{
uint64_t v___x_1974_; size_t v___x_1975_; uint8_t v___x_1976_; 
v___x_1974_ = l_Lean_instHashableExtraModUse_hash(v_x_1973_);
v___x_1975_ = lean_uint64_to_usize(v___x_1974_);
v___x_1976_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg(v_x_1972_, v___x_1975_, v_x_1973_);
return v___x_1976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_x_1977_, lean_object* v_x_1978_){
_start:
{
uint8_t v_res_1979_; lean_object* v_r_1980_; 
v_res_1979_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg(v_x_1977_, v_x_1978_);
lean_dec_ref(v_x_1978_);
lean_dec_ref(v_x_1977_);
v_r_1980_ = lean_box(v_res_1979_);
return v_r_1980_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2(void){
_start:
{
lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
v___x_1983_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__1));
v___x_1984_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__0));
v___x_1985_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_1984_, v___x_1983_);
return v___x_1985_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1986_; 
v___x_1986_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1986_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4(void){
_start:
{
lean_object* v___x_1987_; lean_object* v___x_1988_; 
v___x_1987_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__3);
v___x_1988_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1988_, 0, v___x_1987_);
return v___x_1988_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5(void){
_start:
{
lean_object* v___x_1989_; lean_object* v___x_1990_; 
v___x_1989_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4);
v___x_1990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1990_, 0, v___x_1989_);
lean_ctor_set(v___x_1990_, 1, v___x_1989_);
return v___x_1990_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6(void){
_start:
{
lean_object* v___x_1991_; lean_object* v___x_1992_; 
v___x_1991_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__4);
v___x_1992_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1992_, 0, v___x_1991_);
lean_ctor_set(v___x_1992_, 1, v___x_1991_);
lean_ctor_set(v___x_1992_, 2, v___x_1991_);
lean_ctor_set(v___x_1992_, 3, v___x_1991_);
lean_ctor_set(v___x_1992_, 4, v___x_1991_);
lean_ctor_set(v___x_1992_, 5, v___x_1991_);
return v___x_1992_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10(void){
_start:
{
lean_object* v___x_1997_; lean_object* v___x_1998_; 
v___x_1997_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__9));
v___x_1998_ = l_Lean_stringToMessageData(v___x_1997_);
return v___x_1998_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12(void){
_start:
{
lean_object* v___x_2000_; lean_object* v___x_2001_; 
v___x_2000_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__11));
v___x_2001_ = l_Lean_stringToMessageData(v___x_2000_);
return v___x_2001_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13(void){
_start:
{
lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_2002_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
v___x_2003_ = l_Lean_stringToMessageData(v___x_2002_);
return v___x_2003_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14(void){
_start:
{
lean_object* v_cls_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; 
v_cls_2004_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__8));
v___x_2005_ = ((lean_object*)(lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__1));
v___x_2006_ = l_Lean_Name_append(v___x_2005_, v_cls_2004_);
return v___x_2006_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16(void){
_start:
{
lean_object* v___x_2008_; lean_object* v___x_2009_; 
v___x_2008_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__15));
v___x_2009_ = l_Lean_stringToMessageData(v___x_2008_);
return v___x_2009_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18(void){
_start:
{
lean_object* v___x_2011_; lean_object* v___x_2012_; 
v___x_2011_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__17));
v___x_2012_ = l_Lean_stringToMessageData(v___x_2011_);
return v___x_2012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3(lean_object* v_mod_2017_, uint8_t v_isMeta_2018_, lean_object* v_hint_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_){
_start:
{
lean_object* v___x_2027_; lean_object* v_env_2028_; uint8_t v_isExporting_2029_; lean_object* v___x_2030_; lean_object* v_env_2031_; lean_object* v___x_2032_; lean_object* v_entry_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___y_2038_; lean_object* v___y_2039_; lean_object* v___x_2079_; uint8_t v___x_2080_; 
v___x_2027_ = lean_st_ref_get(v___y_2025_);
v_env_2028_ = lean_ctor_get(v___x_2027_, 0);
lean_inc_ref(v_env_2028_);
lean_dec(v___x_2027_);
v_isExporting_2029_ = lean_ctor_get_uint8(v_env_2028_, sizeof(void*)*8);
lean_dec_ref(v_env_2028_);
v___x_2030_ = lean_st_ref_get(v___y_2025_);
v_env_2031_ = lean_ctor_get(v___x_2030_, 0);
lean_inc_ref(v_env_2031_);
lean_dec(v___x_2030_);
v___x_2032_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__2);
lean_inc(v_mod_2017_);
v_entry_2033_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_entry_2033_, 0, v_mod_2017_);
lean_ctor_set_uint8(v_entry_2033_, sizeof(void*)*1, v_isExporting_2029_);
lean_ctor_set_uint8(v_entry_2033_, sizeof(void*)*1 + 1, v_isMeta_2018_);
v___x_2034_ = l___private_Lean_ExtraModUses_0__Lean_extraModUses;
v___x_2035_ = lean_box(1);
v___x_2036_ = lean_box(0);
v___x_2079_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_2032_, v___x_2034_, v_env_2031_, v___x_2035_, v___x_2036_);
v___x_2080_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg(v___x_2079_, v_entry_2033_);
lean_dec(v___x_2079_);
if (v___x_2080_ == 0)
{
lean_object* v_options_2081_; uint8_t v_hasTrace_2082_; 
v_options_2081_ = lean_ctor_get(v___y_2024_, 2);
v_hasTrace_2082_ = lean_ctor_get_uint8(v_options_2081_, sizeof(void*)*1);
if (v_hasTrace_2082_ == 0)
{
lean_dec(v_hint_2019_);
lean_dec(v_mod_2017_);
v___y_2038_ = v___y_2023_;
v___y_2039_ = v___y_2025_;
goto v___jp_2037_;
}
else
{
lean_object* v_inheritedTraceOptions_2083_; lean_object* v_cls_2084_; lean_object* v___y_2086_; lean_object* v___y_2087_; lean_object* v___y_2091_; lean_object* v___y_2092_; lean_object* v___x_2104_; uint8_t v___x_2105_; 
v_inheritedTraceOptions_2083_ = lean_ctor_get(v___y_2024_, 13);
v_cls_2084_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__8));
v___x_2104_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__14);
v___x_2105_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2083_, v_options_2081_, v___x_2104_);
if (v___x_2105_ == 0)
{
lean_dec(v_hint_2019_);
lean_dec(v_mod_2017_);
v___y_2038_ = v___y_2023_;
v___y_2039_ = v___y_2025_;
goto v___jp_2037_;
}
else
{
lean_object* v___x_2106_; lean_object* v___y_2108_; 
v___x_2106_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__16);
if (v_isExporting_2029_ == 0)
{
lean_object* v___x_2115_; 
v___x_2115_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__21));
v___y_2108_ = v___x_2115_;
goto v___jp_2107_;
}
else
{
lean_object* v___x_2116_; 
v___x_2116_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__22));
v___y_2108_ = v___x_2116_;
goto v___jp_2107_;
}
v___jp_2107_:
{
lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; 
lean_inc_ref(v___y_2108_);
v___x_2109_ = l_Lean_stringToMessageData(v___y_2108_);
v___x_2110_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2110_, 0, v___x_2106_);
lean_ctor_set(v___x_2110_, 1, v___x_2109_);
v___x_2111_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__18);
v___x_2112_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2112_, 0, v___x_2110_);
lean_ctor_set(v___x_2112_, 1, v___x_2111_);
if (v_isMeta_2018_ == 0)
{
lean_object* v___x_2113_; 
v___x_2113_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__19));
v___y_2091_ = v___x_2112_;
v___y_2092_ = v___x_2113_;
goto v___jp_2090_;
}
else
{
lean_object* v___x_2114_; 
v___x_2114_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__20));
v___y_2091_ = v___x_2112_;
v___y_2092_ = v___x_2114_;
goto v___jp_2090_;
}
}
}
v___jp_2085_:
{
lean_object* v___x_2088_; lean_object* v___x_2089_; 
v___x_2088_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2088_, 0, v___y_2086_);
lean_ctor_set(v___x_2088_, 1, v___y_2087_);
v___x_2089_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(v_cls_2084_, v___x_2088_, v___y_2022_, v___y_2023_, v___y_2024_, v___y_2025_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_dec_ref_known(v___x_2089_, 1);
v___y_2038_ = v___y_2023_;
v___y_2039_ = v___y_2025_;
goto v___jp_2037_;
}
else
{
lean_dec_ref_known(v_entry_2033_, 1);
return v___x_2089_;
}
}
v___jp_2090_:
{
lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; uint8_t v___x_2099_; 
lean_inc_ref(v___y_2092_);
v___x_2093_ = l_Lean_stringToMessageData(v___y_2092_);
v___x_2094_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2094_, 0, v___y_2091_);
lean_ctor_set(v___x_2094_, 1, v___x_2093_);
v___x_2095_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__10);
v___x_2096_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2096_, 0, v___x_2094_);
lean_ctor_set(v___x_2096_, 1, v___x_2095_);
v___x_2097_ = l_Lean_MessageData_ofName(v_mod_2017_);
v___x_2098_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2098_, 0, v___x_2096_);
lean_ctor_set(v___x_2098_, 1, v___x_2097_);
v___x_2099_ = l_Lean_Name_isAnonymous(v_hint_2019_);
if (v___x_2099_ == 0)
{
lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; 
v___x_2100_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__12);
v___x_2101_ = l_Lean_MessageData_ofName(v_hint_2019_);
v___x_2102_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2102_, 0, v___x_2100_);
lean_ctor_set(v___x_2102_, 1, v___x_2101_);
v___y_2086_ = v___x_2098_;
v___y_2087_ = v___x_2102_;
goto v___jp_2085_;
}
else
{
lean_object* v___x_2103_; 
lean_dec(v_hint_2019_);
v___x_2103_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__13);
v___y_2086_ = v___x_2098_;
v___y_2087_ = v___x_2103_;
goto v___jp_2085_;
}
}
}
}
else
{
lean_object* v___x_2117_; lean_object* v___x_2118_; 
lean_dec_ref_known(v_entry_2033_, 1);
lean_dec(v_hint_2019_);
lean_dec(v_mod_2017_);
v___x_2117_ = lean_box(0);
v___x_2118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2118_, 0, v___x_2117_);
return v___x_2118_;
}
v___jp_2037_:
{
lean_object* v___x_2040_; lean_object* v_toEnvExtension_2041_; lean_object* v_env_2042_; lean_object* v_nextMacroScope_2043_; lean_object* v_ngen_2044_; lean_object* v_auxDeclNGen_2045_; lean_object* v_traceState_2046_; lean_object* v_messages_2047_; lean_object* v_infoState_2048_; lean_object* v_snapshotTasks_2049_; lean_object* v___x_2051_; uint8_t v_isShared_2052_; uint8_t v_isSharedCheck_2077_; 
v___x_2040_ = lean_st_ref_take(v___y_2039_);
v_toEnvExtension_2041_ = lean_ctor_get(v___x_2034_, 0);
v_env_2042_ = lean_ctor_get(v___x_2040_, 0);
v_nextMacroScope_2043_ = lean_ctor_get(v___x_2040_, 1);
v_ngen_2044_ = lean_ctor_get(v___x_2040_, 2);
v_auxDeclNGen_2045_ = lean_ctor_get(v___x_2040_, 3);
v_traceState_2046_ = lean_ctor_get(v___x_2040_, 4);
v_messages_2047_ = lean_ctor_get(v___x_2040_, 6);
v_infoState_2048_ = lean_ctor_get(v___x_2040_, 7);
v_snapshotTasks_2049_ = lean_ctor_get(v___x_2040_, 8);
v_isSharedCheck_2077_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2077_ == 0)
{
lean_object* v_unused_2078_; 
v_unused_2078_ = lean_ctor_get(v___x_2040_, 5);
lean_dec(v_unused_2078_);
v___x_2051_ = v___x_2040_;
v_isShared_2052_ = v_isSharedCheck_2077_;
goto v_resetjp_2050_;
}
else
{
lean_inc(v_snapshotTasks_2049_);
lean_inc(v_infoState_2048_);
lean_inc(v_messages_2047_);
lean_inc(v_traceState_2046_);
lean_inc(v_auxDeclNGen_2045_);
lean_inc(v_ngen_2044_);
lean_inc(v_nextMacroScope_2043_);
lean_inc(v_env_2042_);
lean_dec(v___x_2040_);
v___x_2051_ = lean_box(0);
v_isShared_2052_ = v_isSharedCheck_2077_;
goto v_resetjp_2050_;
}
v_resetjp_2050_:
{
lean_object* v_asyncMode_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2057_; 
v_asyncMode_2053_ = lean_ctor_get(v_toEnvExtension_2041_, 2);
v___x_2054_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_2034_, v_env_2042_, v_entry_2033_, v_asyncMode_2053_, v___x_2036_);
v___x_2055_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__5);
if (v_isShared_2052_ == 0)
{
lean_ctor_set(v___x_2051_, 5, v___x_2055_);
lean_ctor_set(v___x_2051_, 0, v___x_2054_);
v___x_2057_ = v___x_2051_;
goto v_reusejp_2056_;
}
else
{
lean_object* v_reuseFailAlloc_2076_; 
v_reuseFailAlloc_2076_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2076_, 0, v___x_2054_);
lean_ctor_set(v_reuseFailAlloc_2076_, 1, v_nextMacroScope_2043_);
lean_ctor_set(v_reuseFailAlloc_2076_, 2, v_ngen_2044_);
lean_ctor_set(v_reuseFailAlloc_2076_, 3, v_auxDeclNGen_2045_);
lean_ctor_set(v_reuseFailAlloc_2076_, 4, v_traceState_2046_);
lean_ctor_set(v_reuseFailAlloc_2076_, 5, v___x_2055_);
lean_ctor_set(v_reuseFailAlloc_2076_, 6, v_messages_2047_);
lean_ctor_set(v_reuseFailAlloc_2076_, 7, v_infoState_2048_);
lean_ctor_set(v_reuseFailAlloc_2076_, 8, v_snapshotTasks_2049_);
v___x_2057_ = v_reuseFailAlloc_2076_;
goto v_reusejp_2056_;
}
v_reusejp_2056_:
{
lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v_mctx_2060_; lean_object* v_zetaDeltaFVarIds_2061_; lean_object* v_postponed_2062_; lean_object* v_diag_2063_; lean_object* v___x_2065_; uint8_t v_isShared_2066_; uint8_t v_isSharedCheck_2074_; 
v___x_2058_ = lean_st_ref_set(v___y_2039_, v___x_2057_);
v___x_2059_ = lean_st_ref_take(v___y_2038_);
v_mctx_2060_ = lean_ctor_get(v___x_2059_, 0);
v_zetaDeltaFVarIds_2061_ = lean_ctor_get(v___x_2059_, 2);
v_postponed_2062_ = lean_ctor_get(v___x_2059_, 3);
v_diag_2063_ = lean_ctor_get(v___x_2059_, 4);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2059_);
if (v_isSharedCheck_2074_ == 0)
{
lean_object* v_unused_2075_; 
v_unused_2075_ = lean_ctor_get(v___x_2059_, 1);
lean_dec(v_unused_2075_);
v___x_2065_ = v___x_2059_;
v_isShared_2066_ = v_isSharedCheck_2074_;
goto v_resetjp_2064_;
}
else
{
lean_inc(v_diag_2063_);
lean_inc(v_postponed_2062_);
lean_inc(v_zetaDeltaFVarIds_2061_);
lean_inc(v_mctx_2060_);
lean_dec(v___x_2059_);
v___x_2065_ = lean_box(0);
v_isShared_2066_ = v_isSharedCheck_2074_;
goto v_resetjp_2064_;
}
v_resetjp_2064_:
{
lean_object* v___x_2067_; lean_object* v___x_2069_; 
v___x_2067_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___closed__6);
if (v_isShared_2066_ == 0)
{
lean_ctor_set(v___x_2065_, 1, v___x_2067_);
v___x_2069_ = v___x_2065_;
goto v_reusejp_2068_;
}
else
{
lean_object* v_reuseFailAlloc_2073_; 
v_reuseFailAlloc_2073_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2073_, 0, v_mctx_2060_);
lean_ctor_set(v_reuseFailAlloc_2073_, 1, v___x_2067_);
lean_ctor_set(v_reuseFailAlloc_2073_, 2, v_zetaDeltaFVarIds_2061_);
lean_ctor_set(v_reuseFailAlloc_2073_, 3, v_postponed_2062_);
lean_ctor_set(v_reuseFailAlloc_2073_, 4, v_diag_2063_);
v___x_2069_ = v_reuseFailAlloc_2073_;
goto v_reusejp_2068_;
}
v_reusejp_2068_:
{
lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; 
v___x_2070_ = lean_st_ref_set(v___y_2038_, v___x_2069_);
v___x_2071_ = lean_box(0);
v___x_2072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2072_, 0, v___x_2071_);
return v___x_2072_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3___boxed(lean_object* v_mod_2119_, lean_object* v_isMeta_2120_, lean_object* v_hint_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_){
_start:
{
uint8_t v_isMeta_boxed_2129_; lean_object* v_res_2130_; 
v_isMeta_boxed_2129_ = lean_unbox(v_isMeta_2120_);
v_res_2130_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3(v_mod_2119_, v_isMeta_boxed_2129_, v_hint_2121_, v___y_2122_, v___y_2123_, v___y_2124_, v___y_2125_, v___y_2126_, v___y_2127_);
lean_dec(v___y_2127_);
lean_dec_ref(v___y_2126_);
lean_dec(v___y_2125_);
lean_dec_ref(v___y_2124_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
return v_res_2130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4(lean_object* v___x_2131_, lean_object* v_declName_2132_, lean_object* v_as_2133_, size_t v_sz_2134_, size_t v_i_2135_, lean_object* v_b_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_, lean_object* v___y_2141_, lean_object* v___y_2142_){
_start:
{
uint8_t v___x_2144_; 
v___x_2144_ = lean_usize_dec_lt(v_i_2135_, v_sz_2134_);
if (v___x_2144_ == 0)
{
lean_object* v___x_2145_; 
lean_dec(v_declName_2132_);
v___x_2145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2145_, 0, v_b_2136_);
return v___x_2145_;
}
else
{
lean_object* v___x_2146_; lean_object* v_modules_2147_; lean_object* v___x_2148_; lean_object* v_a_2149_; lean_object* v___x_2150_; lean_object* v_toImport_2151_; lean_object* v_module_2152_; uint8_t v___x_2153_; lean_object* v___x_2154_; 
v___x_2146_ = l_Lean_Environment_header(v___x_2131_);
v_modules_2147_ = lean_ctor_get(v___x_2146_, 3);
lean_inc_ref(v_modules_2147_);
lean_dec_ref(v___x_2146_);
v___x_2148_ = l_Lean_instInhabitedEffectiveImport_default;
v_a_2149_ = lean_array_uget_borrowed(v_as_2133_, v_i_2135_);
v___x_2150_ = lean_array_get(v___x_2148_, v_modules_2147_, v_a_2149_);
lean_dec_ref(v_modules_2147_);
v_toImport_2151_ = lean_ctor_get(v___x_2150_, 0);
lean_inc_ref(v_toImport_2151_);
lean_dec(v___x_2150_);
v_module_2152_ = lean_ctor_get(v_toImport_2151_, 0);
lean_inc(v_module_2152_);
lean_dec_ref(v_toImport_2151_);
v___x_2153_ = 0;
lean_inc(v_declName_2132_);
v___x_2154_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3(v_module_2152_, v___x_2153_, v_declName_2132_, v___y_2137_, v___y_2138_, v___y_2139_, v___y_2140_, v___y_2141_, v___y_2142_);
if (lean_obj_tag(v___x_2154_) == 0)
{
lean_object* v___x_2155_; size_t v___x_2156_; size_t v___x_2157_; 
lean_dec_ref_known(v___x_2154_, 1);
v___x_2155_ = lean_box(0);
v___x_2156_ = ((size_t)1ULL);
v___x_2157_ = lean_usize_add(v_i_2135_, v___x_2156_);
v_i_2135_ = v___x_2157_;
v_b_2136_ = v___x_2155_;
goto _start;
}
else
{
lean_dec(v_declName_2132_);
return v___x_2154_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4___boxed(lean_object* v___x_2159_, lean_object* v_declName_2160_, lean_object* v_as_2161_, lean_object* v_sz_2162_, lean_object* v_i_2163_, lean_object* v_b_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_){
_start:
{
size_t v_sz_boxed_2172_; size_t v_i_boxed_2173_; lean_object* v_res_2174_; 
v_sz_boxed_2172_ = lean_unbox_usize(v_sz_2162_);
lean_dec(v_sz_2162_);
v_i_boxed_2173_ = lean_unbox_usize(v_i_2163_);
lean_dec(v_i_2163_);
v_res_2174_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4(v___x_2159_, v_declName_2160_, v_as_2161_, v_sz_boxed_2172_, v_i_boxed_2173_, v_b_2164_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec(v___y_2166_);
lean_dec_ref(v___y_2165_);
lean_dec_ref(v_as_2161_);
lean_dec_ref(v___x_2159_);
return v_res_2174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg(lean_object* v_a_2175_, lean_object* v_x_2176_){
_start:
{
if (lean_obj_tag(v_x_2176_) == 0)
{
lean_object* v___x_2177_; 
v___x_2177_ = lean_box(0);
return v___x_2177_;
}
else
{
lean_object* v_key_2178_; lean_object* v_value_2179_; lean_object* v_tail_2180_; uint8_t v___x_2181_; 
v_key_2178_ = lean_ctor_get(v_x_2176_, 0);
v_value_2179_ = lean_ctor_get(v_x_2176_, 1);
v_tail_2180_ = lean_ctor_get(v_x_2176_, 2);
v___x_2181_ = lean_name_eq(v_key_2178_, v_a_2175_);
if (v___x_2181_ == 0)
{
v_x_2176_ = v_tail_2180_;
goto _start;
}
else
{
lean_object* v___x_2183_; 
lean_inc(v_value_2179_);
v___x_2183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2183_, 0, v_value_2179_);
return v___x_2183_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg___boxed(lean_object* v_a_2184_, lean_object* v_x_2185_){
_start:
{
lean_object* v_res_2186_; 
v_res_2186_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg(v_a_2184_, v_x_2185_);
lean_dec(v_x_2185_);
lean_dec(v_a_2184_);
return v_res_2186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg(lean_object* v_m_2187_, lean_object* v_a_2188_){
_start:
{
lean_object* v_buckets_2189_; lean_object* v___x_2190_; uint64_t v___y_2192_; 
v_buckets_2189_ = lean_ctor_get(v_m_2187_, 1);
v___x_2190_ = lean_array_get_size(v_buckets_2189_);
if (lean_obj_tag(v_a_2188_) == 0)
{
uint64_t v___x_2206_; 
v___x_2206_ = 1723ULL;
v___y_2192_ = v___x_2206_;
goto v___jp_2191_;
}
else
{
uint64_t v_hash_2207_; 
v_hash_2207_ = lean_ctor_get_uint64(v_a_2188_, sizeof(void*)*2);
v___y_2192_ = v_hash_2207_;
goto v___jp_2191_;
}
v___jp_2191_:
{
uint64_t v___x_2193_; uint64_t v___x_2194_; uint64_t v_fold_2195_; uint64_t v___x_2196_; uint64_t v___x_2197_; uint64_t v___x_2198_; size_t v___x_2199_; size_t v___x_2200_; size_t v___x_2201_; size_t v___x_2202_; size_t v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; 
v___x_2193_ = 32ULL;
v___x_2194_ = lean_uint64_shift_right(v___y_2192_, v___x_2193_);
v_fold_2195_ = lean_uint64_xor(v___y_2192_, v___x_2194_);
v___x_2196_ = 16ULL;
v___x_2197_ = lean_uint64_shift_right(v_fold_2195_, v___x_2196_);
v___x_2198_ = lean_uint64_xor(v_fold_2195_, v___x_2197_);
v___x_2199_ = lean_uint64_to_usize(v___x_2198_);
v___x_2200_ = lean_usize_of_nat(v___x_2190_);
v___x_2201_ = ((size_t)1ULL);
v___x_2202_ = lean_usize_sub(v___x_2200_, v___x_2201_);
v___x_2203_ = lean_usize_land(v___x_2199_, v___x_2202_);
v___x_2204_ = lean_array_uget_borrowed(v_buckets_2189_, v___x_2203_);
v___x_2205_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg(v_a_2188_, v___x_2204_);
return v___x_2205_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg___boxed(lean_object* v_m_2208_, lean_object* v_a_2209_){
_start:
{
lean_object* v_res_2210_; 
v_res_2210_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg(v_m_2208_, v_a_2209_);
lean_dec(v_a_2209_);
lean_dec_ref(v_m_2208_);
return v_res_2210_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2(void){
_start:
{
lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; 
v___x_2213_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__1));
v___x_2214_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__0));
v___x_2215_ = l_Std_HashMap_instInhabited(lean_box(0), lean_box(0), v___x_2214_, v___x_2213_);
return v___x_2215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2(lean_object* v_declName_2218_, uint8_t v_isMeta_2219_, lean_object* v___y_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_){
_start:
{
lean_object* v___x_2227_; lean_object* v_env_2231_; lean_object* v___y_2233_; lean_object* v___x_2246_; 
v___x_2227_ = lean_st_ref_get(v___y_2225_);
v_env_2231_ = lean_ctor_get(v___x_2227_, 0);
lean_inc_ref(v_env_2231_);
lean_dec(v___x_2227_);
v___x_2246_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_2231_, v_declName_2218_);
if (lean_obj_tag(v___x_2246_) == 0)
{
lean_dec_ref(v_env_2231_);
lean_dec(v_declName_2218_);
goto v___jp_2228_;
}
else
{
lean_object* v_val_2247_; lean_object* v___x_2248_; lean_object* v_modules_2249_; lean_object* v___x_2250_; uint8_t v___x_2251_; 
v_val_2247_ = lean_ctor_get(v___x_2246_, 0);
lean_inc(v_val_2247_);
lean_dec_ref_known(v___x_2246_, 1);
v___x_2248_ = l_Lean_Environment_header(v_env_2231_);
v_modules_2249_ = lean_ctor_get(v___x_2248_, 3);
lean_inc_ref(v_modules_2249_);
lean_dec_ref(v___x_2248_);
v___x_2250_ = lean_array_get_size(v_modules_2249_);
v___x_2251_ = lean_nat_dec_lt(v_val_2247_, v___x_2250_);
if (v___x_2251_ == 0)
{
lean_dec_ref(v_modules_2249_);
lean_dec(v_val_2247_);
lean_dec_ref(v_env_2231_);
lean_dec(v_declName_2218_);
goto v___jp_2228_;
}
else
{
lean_object* v___x_2252_; lean_object* v_env_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; uint8_t v___y_2257_; 
v___x_2252_ = lean_st_ref_get(v___y_2225_);
v_env_2253_ = lean_ctor_get(v___x_2252_, 0);
lean_inc_ref(v_env_2253_);
lean_dec(v___x_2252_);
v___x_2254_ = lean_obj_once(&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2, &lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2_once, _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__2);
v___x_2255_ = lean_array_fget(v_modules_2249_, v_val_2247_);
lean_dec(v_val_2247_);
lean_dec_ref(v_modules_2249_);
if (v_isMeta_2219_ == 0)
{
lean_dec_ref(v_env_2253_);
v___y_2257_ = v_isMeta_2219_;
goto v___jp_2256_;
}
else
{
uint8_t v___x_2268_; 
lean_inc(v_declName_2218_);
v___x_2268_ = l_Lean_isMarkedMeta(v_env_2253_, v_declName_2218_);
if (v___x_2268_ == 0)
{
v___y_2257_ = v_isMeta_2219_;
goto v___jp_2256_;
}
else
{
uint8_t v___x_2269_; 
v___x_2269_ = 0;
v___y_2257_ = v___x_2269_;
goto v___jp_2256_;
}
}
v___jp_2256_:
{
lean_object* v_toImport_2258_; lean_object* v_module_2259_; lean_object* v___x_2260_; 
v_toImport_2258_ = lean_ctor_get(v___x_2255_, 0);
lean_inc_ref(v_toImport_2258_);
lean_dec(v___x_2255_);
v_module_2259_ = lean_ctor_get(v_toImport_2258_, 0);
lean_inc(v_module_2259_);
lean_dec_ref(v_toImport_2258_);
lean_inc(v_declName_2218_);
v___x_2260_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3(v_module_2259_, v___y_2257_, v_declName_2218_, v___y_2220_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_, v___y_2225_);
if (lean_obj_tag(v___x_2260_) == 0)
{
lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; 
lean_dec_ref_known(v___x_2260_, 1);
v___x_2261_ = l_Lean_indirectModUseExt;
v___x_2262_ = lean_box(1);
v___x_2263_ = lean_box(0);
lean_inc_ref(v_env_2231_);
v___x_2264_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_2254_, v___x_2261_, v_env_2231_, v___x_2262_, v___x_2263_);
v___x_2265_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg(v___x_2264_, v_declName_2218_);
lean_dec(v___x_2264_);
if (lean_obj_tag(v___x_2265_) == 0)
{
lean_object* v___x_2266_; 
v___x_2266_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___closed__3));
v___y_2233_ = v___x_2266_;
goto v___jp_2232_;
}
else
{
lean_object* v_val_2267_; 
v_val_2267_ = lean_ctor_get(v___x_2265_, 0);
lean_inc(v_val_2267_);
lean_dec_ref_known(v___x_2265_, 1);
v___y_2233_ = v_val_2267_;
goto v___jp_2232_;
}
}
else
{
lean_dec_ref(v_env_2231_);
lean_dec(v_declName_2218_);
return v___x_2260_;
}
}
}
}
v___jp_2228_:
{
lean_object* v___x_2229_; lean_object* v___x_2230_; 
v___x_2229_ = lean_box(0);
v___x_2230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2230_, 0, v___x_2229_);
return v___x_2230_;
}
v___jp_2232_:
{
lean_object* v___x_2234_; size_t v_sz_2235_; size_t v___x_2236_; lean_object* v___x_2237_; 
v___x_2234_ = lean_box(0);
v_sz_2235_ = lean_array_size(v___y_2233_);
v___x_2236_ = ((size_t)0ULL);
v___x_2237_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__4(v_env_2231_, v_declName_2218_, v___y_2233_, v_sz_2235_, v___x_2236_, v___x_2234_, v___y_2220_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_, v___y_2225_);
lean_dec_ref(v___y_2233_);
lean_dec_ref(v_env_2231_);
if (lean_obj_tag(v___x_2237_) == 0)
{
lean_object* v___x_2239_; uint8_t v_isShared_2240_; uint8_t v_isSharedCheck_2244_; 
v_isSharedCheck_2244_ = !lean_is_exclusive(v___x_2237_);
if (v_isSharedCheck_2244_ == 0)
{
lean_object* v_unused_2245_; 
v_unused_2245_ = lean_ctor_get(v___x_2237_, 0);
lean_dec(v_unused_2245_);
v___x_2239_ = v___x_2237_;
v_isShared_2240_ = v_isSharedCheck_2244_;
goto v_resetjp_2238_;
}
else
{
lean_dec(v___x_2237_);
v___x_2239_ = lean_box(0);
v_isShared_2240_ = v_isSharedCheck_2244_;
goto v_resetjp_2238_;
}
v_resetjp_2238_:
{
lean_object* v___x_2242_; 
if (v_isShared_2240_ == 0)
{
lean_ctor_set(v___x_2239_, 0, v___x_2234_);
v___x_2242_ = v___x_2239_;
goto v_reusejp_2241_;
}
else
{
lean_object* v_reuseFailAlloc_2243_; 
v_reuseFailAlloc_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2243_, 0, v___x_2234_);
v___x_2242_ = v_reuseFailAlloc_2243_;
goto v_reusejp_2241_;
}
v_reusejp_2241_:
{
return v___x_2242_;
}
}
}
else
{
return v___x_2237_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2___boxed(lean_object* v_declName_2270_, lean_object* v_isMeta_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_){
_start:
{
uint8_t v_isMeta_boxed_2279_; lean_object* v_res_2280_; 
v_isMeta_boxed_2279_ = lean_unbox(v_isMeta_2271_);
v_res_2280_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2(v_declName_2270_, v_isMeta_boxed_2279_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_);
lean_dec(v___y_2277_);
lean_dec_ref(v___y_2276_);
lean_dec(v___y_2275_);
lean_dec_ref(v___y_2274_);
lean_dec(v___y_2273_);
lean_dec_ref(v___y_2272_);
return v_res_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg(lean_object* v_as_x27_2281_, lean_object* v_b_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_){
_start:
{
if (lean_obj_tag(v_as_x27_2281_) == 0)
{
lean_object* v___x_2290_; 
v___x_2290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2290_, 0, v_b_2282_);
return v___x_2290_;
}
else
{
lean_object* v_head_2291_; lean_object* v_tail_2292_; uint8_t v___x_2293_; lean_object* v___x_2294_; 
v_head_2291_ = lean_ctor_get(v_as_x27_2281_, 0);
v_tail_2292_ = lean_ctor_get(v_as_x27_2281_, 1);
v___x_2293_ = 1;
lean_inc(v_head_2291_);
v___x_2294_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2(v_head_2291_, v___x_2293_, v___y_2283_, v___y_2284_, v___y_2285_, v___y_2286_, v___y_2287_, v___y_2288_);
if (lean_obj_tag(v___x_2294_) == 0)
{
lean_object* v___x_2295_; 
lean_dec_ref_known(v___x_2294_, 1);
v___x_2295_ = lean_box(0);
v_as_x27_2281_ = v_tail_2292_;
v_b_2282_ = v___x_2295_;
goto _start;
}
else
{
return v___x_2294_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg___boxed(lean_object* v_as_x27_2297_, lean_object* v_b_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_){
_start:
{
lean_object* v_res_2306_; 
v_res_2306_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg(v_as_x27_2297_, v_b_2298_, v___y_2299_, v___y_2300_, v___y_2301_, v___y_2302_, v___y_2303_, v___y_2304_);
lean_dec(v___y_2304_);
lean_dec_ref(v___y_2303_);
lean_dec(v___y_2302_);
lean_dec_ref(v___y_2301_);
lean_dec(v___y_2300_);
lean_dec_ref(v___y_2299_);
lean_dec(v_as_x27_2297_);
return v_res_2306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3(lean_object* v_currNamespace_2307_, lean_object* v___y_2308_, lean_object* v___y_2309_){
_start:
{
lean_object* v___x_2310_; 
v___x_2310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2310_, 0, v_currNamespace_2307_);
lean_ctor_set(v___x_2310_, 1, v___y_2309_);
return v___x_2310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3___boxed(lean_object* v_currNamespace_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_){
_start:
{
lean_object* v_res_2314_; 
v_res_2314_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3(v_currNamespace_2311_, v___y_2312_, v___y_2313_);
lean_dec_ref(v___y_2312_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg(lean_object* v_x_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_){
_start:
{
lean_object* v___x_2324_; lean_object* v_env_2325_; lean_object* v_options_2326_; lean_object* v_currRecDepth_2327_; lean_object* v_maxRecDepth_2328_; lean_object* v_ref_2329_; lean_object* v_currNamespace_2330_; lean_object* v_openDecls_2331_; lean_object* v_quotContext_2332_; lean_object* v_currMacroScope_2333_; lean_object* v___x_2334_; lean_object* v_nextMacroScope_2335_; lean_object* v___f_2336_; lean_object* v___f_2337_; lean_object* v___f_2338_; lean_object* v___f_2339_; lean_object* v___f_2340_; lean_object* v_methods_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; 
v___x_2324_ = lean_st_ref_get(v___y_2322_);
v_env_2325_ = lean_ctor_get(v___x_2324_, 0);
lean_inc_ref_n(v_env_2325_, 4);
lean_dec(v___x_2324_);
v_options_2326_ = lean_ctor_get(v___y_2321_, 2);
v_currRecDepth_2327_ = lean_ctor_get(v___y_2321_, 3);
v_maxRecDepth_2328_ = lean_ctor_get(v___y_2321_, 4);
v_ref_2329_ = lean_ctor_get(v___y_2321_, 5);
v_currNamespace_2330_ = lean_ctor_get(v___y_2321_, 6);
v_openDecls_2331_ = lean_ctor_get(v___y_2321_, 7);
v_quotContext_2332_ = lean_ctor_get(v___y_2321_, 10);
v_currMacroScope_2333_ = lean_ctor_get(v___y_2321_, 11);
v___x_2334_ = lean_st_ref_get(v___y_2322_);
v_nextMacroScope_2335_ = lean_ctor_get(v___x_2334_, 1);
lean_inc(v_nextMacroScope_2335_);
lean_dec(v___x_2334_);
v___f_2336_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_2336_, 0, v_env_2325_);
v___f_2337_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_2337_, 0, v_env_2325_);
lean_inc_n(v_openDecls_2331_, 2);
lean_inc_n(v_currNamespace_2330_, 3);
v___f_2338_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__2___boxed), 6, 3);
lean_closure_set(v___f_2338_, 0, v_env_2325_);
lean_closure_set(v___f_2338_, 1, v_currNamespace_2330_);
lean_closure_set(v___f_2338_, 2, v_openDecls_2331_);
v___f_2339_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_2339_, 0, v_currNamespace_2330_);
lean_inc_ref(v_options_2326_);
v___f_2340_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___lam__4___boxed), 7, 4);
lean_closure_set(v___f_2340_, 0, v_env_2325_);
lean_closure_set(v___f_2340_, 1, v_options_2326_);
lean_closure_set(v___f_2340_, 2, v_currNamespace_2330_);
lean_closure_set(v___f_2340_, 3, v_openDecls_2331_);
v_methods_2341_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_methods_2341_, 0, v___f_2336_);
lean_ctor_set(v_methods_2341_, 1, v___f_2339_);
lean_ctor_set(v_methods_2341_, 2, v___f_2337_);
lean_ctor_set(v_methods_2341_, 3, v___f_2338_);
lean_ctor_set(v_methods_2341_, 4, v___f_2340_);
lean_inc(v_ref_2329_);
lean_inc(v_maxRecDepth_2328_);
lean_inc(v_currRecDepth_2327_);
lean_inc(v_currMacroScope_2333_);
lean_inc(v_quotContext_2332_);
v___x_2342_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2342_, 0, v_methods_2341_);
lean_ctor_set(v___x_2342_, 1, v_quotContext_2332_);
lean_ctor_set(v___x_2342_, 2, v_currMacroScope_2333_);
lean_ctor_set(v___x_2342_, 3, v_currRecDepth_2327_);
lean_ctor_set(v___x_2342_, 4, v_maxRecDepth_2328_);
lean_ctor_set(v___x_2342_, 5, v_ref_2329_);
v___x_2343_ = lean_box(0);
v___x_2344_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2344_, 0, v_nextMacroScope_2335_);
lean_ctor_set(v___x_2344_, 1, v___x_2343_);
lean_ctor_set(v___x_2344_, 2, v___x_2343_);
v___x_2345_ = lean_apply_2(v_x_2316_, v___x_2342_, v___x_2344_);
if (lean_obj_tag(v___x_2345_) == 0)
{
lean_object* v_a_2346_; lean_object* v_a_2347_; lean_object* v_macroScope_2348_; lean_object* v_traceMsgs_2349_; lean_object* v_expandedMacroDecls_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; 
v_a_2346_ = lean_ctor_get(v___x_2345_, 1);
lean_inc(v_a_2346_);
v_a_2347_ = lean_ctor_get(v___x_2345_, 0);
lean_inc(v_a_2347_);
lean_dec_ref_known(v___x_2345_, 2);
v_macroScope_2348_ = lean_ctor_get(v_a_2346_, 0);
lean_inc(v_macroScope_2348_);
v_traceMsgs_2349_ = lean_ctor_get(v_a_2346_, 1);
lean_inc(v_traceMsgs_2349_);
v_expandedMacroDecls_2350_ = lean_ctor_get(v_a_2346_, 2);
lean_inc(v_expandedMacroDecls_2350_);
lean_dec(v_a_2346_);
v___x_2351_ = lean_box(0);
v___x_2352_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg(v_expandedMacroDecls_2350_, v___x_2351_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_);
lean_dec(v_expandedMacroDecls_2350_);
if (lean_obj_tag(v___x_2352_) == 0)
{
lean_object* v___x_2353_; lean_object* v_env_2354_; lean_object* v_ngen_2355_; lean_object* v_auxDeclNGen_2356_; lean_object* v_traceState_2357_; lean_object* v_cache_2358_; lean_object* v_messages_2359_; lean_object* v_infoState_2360_; lean_object* v_snapshotTasks_2361_; lean_object* v___x_2363_; uint8_t v_isShared_2364_; uint8_t v_isSharedCheck_2387_; 
lean_dec_ref_known(v___x_2352_, 1);
v___x_2353_ = lean_st_ref_take(v___y_2322_);
v_env_2354_ = lean_ctor_get(v___x_2353_, 0);
v_ngen_2355_ = lean_ctor_get(v___x_2353_, 2);
v_auxDeclNGen_2356_ = lean_ctor_get(v___x_2353_, 3);
v_traceState_2357_ = lean_ctor_get(v___x_2353_, 4);
v_cache_2358_ = lean_ctor_get(v___x_2353_, 5);
v_messages_2359_ = lean_ctor_get(v___x_2353_, 6);
v_infoState_2360_ = lean_ctor_get(v___x_2353_, 7);
v_snapshotTasks_2361_ = lean_ctor_get(v___x_2353_, 8);
v_isSharedCheck_2387_ = !lean_is_exclusive(v___x_2353_);
if (v_isSharedCheck_2387_ == 0)
{
lean_object* v_unused_2388_; 
v_unused_2388_ = lean_ctor_get(v___x_2353_, 1);
lean_dec(v_unused_2388_);
v___x_2363_ = v___x_2353_;
v_isShared_2364_ = v_isSharedCheck_2387_;
goto v_resetjp_2362_;
}
else
{
lean_inc(v_snapshotTasks_2361_);
lean_inc(v_infoState_2360_);
lean_inc(v_messages_2359_);
lean_inc(v_cache_2358_);
lean_inc(v_traceState_2357_);
lean_inc(v_auxDeclNGen_2356_);
lean_inc(v_ngen_2355_);
lean_inc(v_env_2354_);
lean_dec(v___x_2353_);
v___x_2363_ = lean_box(0);
v_isShared_2364_ = v_isSharedCheck_2387_;
goto v_resetjp_2362_;
}
v_resetjp_2362_:
{
lean_object* v___x_2366_; 
if (v_isShared_2364_ == 0)
{
lean_ctor_set(v___x_2363_, 1, v_macroScope_2348_);
v___x_2366_ = v___x_2363_;
goto v_reusejp_2365_;
}
else
{
lean_object* v_reuseFailAlloc_2386_; 
v_reuseFailAlloc_2386_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2386_, 0, v_env_2354_);
lean_ctor_set(v_reuseFailAlloc_2386_, 1, v_macroScope_2348_);
lean_ctor_set(v_reuseFailAlloc_2386_, 2, v_ngen_2355_);
lean_ctor_set(v_reuseFailAlloc_2386_, 3, v_auxDeclNGen_2356_);
lean_ctor_set(v_reuseFailAlloc_2386_, 4, v_traceState_2357_);
lean_ctor_set(v_reuseFailAlloc_2386_, 5, v_cache_2358_);
lean_ctor_set(v_reuseFailAlloc_2386_, 6, v_messages_2359_);
lean_ctor_set(v_reuseFailAlloc_2386_, 7, v_infoState_2360_);
lean_ctor_set(v_reuseFailAlloc_2386_, 8, v_snapshotTasks_2361_);
v___x_2366_ = v_reuseFailAlloc_2386_;
goto v_reusejp_2365_;
}
v_reusejp_2365_:
{
lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; 
v___x_2367_ = lean_st_ref_set(v___y_2322_, v___x_2366_);
v___x_2368_ = l_List_reverse___redArg(v_traceMsgs_2349_);
v___x_2369_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4(v___x_2368_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_);
if (lean_obj_tag(v___x_2369_) == 0)
{
lean_object* v___x_2371_; uint8_t v_isShared_2372_; uint8_t v_isSharedCheck_2376_; 
v_isSharedCheck_2376_ = !lean_is_exclusive(v___x_2369_);
if (v_isSharedCheck_2376_ == 0)
{
lean_object* v_unused_2377_; 
v_unused_2377_ = lean_ctor_get(v___x_2369_, 0);
lean_dec(v_unused_2377_);
v___x_2371_ = v___x_2369_;
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
else
{
lean_dec(v___x_2369_);
v___x_2371_ = lean_box(0);
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
v_resetjp_2370_:
{
lean_object* v___x_2374_; 
if (v_isShared_2372_ == 0)
{
lean_ctor_set(v___x_2371_, 0, v_a_2347_);
v___x_2374_ = v___x_2371_;
goto v_reusejp_2373_;
}
else
{
lean_object* v_reuseFailAlloc_2375_; 
v_reuseFailAlloc_2375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2375_, 0, v_a_2347_);
v___x_2374_ = v_reuseFailAlloc_2375_;
goto v_reusejp_2373_;
}
v_reusejp_2373_:
{
return v___x_2374_;
}
}
}
else
{
lean_object* v_a_2378_; lean_object* v___x_2380_; uint8_t v_isShared_2381_; uint8_t v_isSharedCheck_2385_; 
lean_dec(v_a_2347_);
v_a_2378_ = lean_ctor_get(v___x_2369_, 0);
v_isSharedCheck_2385_ = !lean_is_exclusive(v___x_2369_);
if (v_isSharedCheck_2385_ == 0)
{
v___x_2380_ = v___x_2369_;
v_isShared_2381_ = v_isSharedCheck_2385_;
goto v_resetjp_2379_;
}
else
{
lean_inc(v_a_2378_);
lean_dec(v___x_2369_);
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
}
}
else
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2396_; 
lean_dec(v_traceMsgs_2349_);
lean_dec(v_macroScope_2348_);
lean_dec(v_a_2347_);
v_a_2389_ = lean_ctor_get(v___x_2352_, 0);
v_isSharedCheck_2396_ = !lean_is_exclusive(v___x_2352_);
if (v_isSharedCheck_2396_ == 0)
{
v___x_2391_ = v___x_2352_;
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___x_2352_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
lean_object* v___x_2394_; 
if (v_isShared_2392_ == 0)
{
v___x_2394_ = v___x_2391_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_a_2389_);
v___x_2394_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
return v___x_2394_;
}
}
}
}
else
{
lean_object* v_a_2397_; 
v_a_2397_ = lean_ctor_get(v___x_2345_, 0);
lean_inc(v_a_2397_);
lean_dec_ref_known(v___x_2345_, 2);
if (lean_obj_tag(v_a_2397_) == 0)
{
lean_object* v_a_2398_; lean_object* v_a_2399_; lean_object* v___x_2400_; uint8_t v___x_2401_; 
v_a_2398_ = lean_ctor_get(v_a_2397_, 0);
lean_inc(v_a_2398_);
v_a_2399_ = lean_ctor_get(v_a_2397_, 1);
lean_inc_ref(v_a_2399_);
lean_dec_ref_known(v_a_2397_, 2);
v___x_2400_ = ((lean_object*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___closed__0));
v___x_2401_ = lean_string_dec_eq(v_a_2399_, v___x_2400_);
if (v___x_2401_ == 0)
{
lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; 
v___x_2402_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2402_, 0, v_a_2399_);
v___x_2403_ = l_Lean_MessageData_ofFormat(v___x_2402_);
v___x_2404_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg(v_a_2398_, v___x_2403_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_);
lean_dec(v_a_2398_);
return v___x_2404_;
}
else
{
lean_object* v___x_2405_; 
lean_dec_ref(v_a_2399_);
v___x_2405_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg(v_a_2398_);
return v___x_2405_;
}
}
else
{
lean_object* v___x_2406_; 
v___x_2406_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg();
return v___x_2406_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg___boxed(lean_object* v_x_2407_, lean_object* v___y_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_){
_start:
{
lean_object* v_res_2415_; 
v_res_2415_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg(v_x_2407_, v___y_2408_, v___y_2409_, v___y_2410_, v___y_2411_, v___y_2412_, v___y_2413_);
lean_dec(v___y_2413_);
lean_dec_ref(v___y_2412_);
lean_dec(v___y_2411_);
lean_dec_ref(v___y_2410_);
lean_dec(v___y_2409_);
lean_dec_ref(v___y_2408_);
return v_res_2415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(lean_object* v_stx_2466_, lean_object* v_a_2467_, lean_object* v_a_2468_, lean_object* v_a_2469_, lean_object* v_a_2470_, lean_object* v_a_2471_, lean_object* v_a_2472_){
_start:
{
lean_object* v___y_2475_; uint8_t v___y_2476_; lean_object* v___f_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; 
v___f_2479_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__0));
v___x_2480_ = lean_alloc_closure((void*)(l_Lean_expandMacros), 4, 2);
lean_closure_set(v___x_2480_, 0, v_stx_2466_);
lean_closure_set(v___x_2480_, 1, v___f_2479_);
v___x_2481_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg(v___x_2480_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2481_) == 0)
{
lean_object* v_a_2482_; lean_object* v___x_2484_; uint8_t v_isShared_2485_; uint8_t v_isSharedCheck_2920_; 
v_a_2482_ = lean_ctor_get(v___x_2481_, 0);
v_isSharedCheck_2920_ = !lean_is_exclusive(v___x_2481_);
if (v_isSharedCheck_2920_ == 0)
{
v___x_2484_ = v___x_2481_;
v_isShared_2485_ = v_isSharedCheck_2920_;
goto v_resetjp_2483_;
}
else
{
lean_inc(v_a_2482_);
lean_dec(v___x_2481_);
v___x_2484_ = lean_box(0);
v_isShared_2485_ = v_isSharedCheck_2920_;
goto v_resetjp_2483_;
}
v_resetjp_2483_:
{
lean_object* v___x_2486_; uint8_t v___x_2487_; 
v___x_2486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__2));
lean_inc(v_a_2482_);
v___x_2487_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2486_);
if (v___x_2487_ == 0)
{
lean_object* v___x_2488_; uint8_t v___x_2489_; 
lean_del_object(v___x_2484_);
v___x_2488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__4));
lean_inc(v_a_2482_);
v___x_2489_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2488_);
if (v___x_2489_ == 0)
{
lean_object* v___x_2490_; uint8_t v___x_2491_; 
v___x_2490_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__6));
lean_inc(v_a_2482_);
v___x_2491_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2490_);
if (v___x_2491_ == 0)
{
lean_object* v___x_2492_; uint8_t v___x_2493_; 
v___x_2492_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__8));
lean_inc(v_a_2482_);
v___x_2493_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2492_);
if (v___x_2493_ == 0)
{
lean_object* v___x_2494_; uint8_t v___x_2495_; 
v___x_2494_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__10));
lean_inc(v_a_2482_);
v___x_2495_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2494_);
if (v___x_2495_ == 0)
{
lean_object* v___x_2496_; uint8_t v___x_2497_; 
v___x_2496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__12));
lean_inc(v_a_2482_);
v___x_2497_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2496_);
if (v___x_2497_ == 0)
{
lean_object* v___x_2498_; uint8_t v___x_2499_; 
v___x_2498_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__14));
lean_inc(v_a_2482_);
v___x_2499_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2498_);
if (v___x_2499_ == 0)
{
lean_object* v___x_2500_; uint8_t v___x_2501_; 
v___x_2500_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__16));
lean_inc(v_a_2482_);
v___x_2501_ = l_Lean_Syntax_isOfKind(v_a_2482_, v___x_2500_);
if (v___x_2501_ == 0)
{
lean_object* v___x_2502_; lean_object* v___x_2503_; 
v___x_2502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2503_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2502_, v___x_2501_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2503_) == 0)
{
return v___x_2503_;
}
else
{
lean_object* v_a_2504_; uint8_t v___y_2506_; uint8_t v___x_2516_; 
v_a_2504_ = lean_ctor_get(v___x_2503_, 0);
lean_inc(v_a_2504_);
v___x_2516_ = l_Lean_Exception_isInterrupt(v_a_2504_);
if (v___x_2516_ == 0)
{
uint8_t v___x_2517_; 
v___x_2517_ = l_Lean_Exception_isRuntime(v_a_2504_);
v___y_2506_ = v___x_2517_;
goto v___jp_2505_;
}
else
{
lean_dec(v_a_2504_);
v___y_2506_ = v___x_2516_;
goto v___jp_2505_;
}
v___jp_2505_:
{
if (v___y_2506_ == 0)
{
lean_object* v___x_2508_; uint8_t v_isShared_2509_; uint8_t v_isSharedCheck_2514_; 
v_isSharedCheck_2514_ = !lean_is_exclusive(v___x_2503_);
if (v_isSharedCheck_2514_ == 0)
{
lean_object* v_unused_2515_; 
v_unused_2515_ = lean_ctor_get(v___x_2503_, 0);
lean_dec(v_unused_2515_);
v___x_2508_ = v___x_2503_;
v_isShared_2509_ = v_isSharedCheck_2514_;
goto v_resetjp_2507_;
}
else
{
lean_dec(v___x_2503_);
v___x_2508_ = lean_box(0);
v_isShared_2509_ = v_isSharedCheck_2514_;
goto v_resetjp_2507_;
}
v_resetjp_2507_:
{
lean_object* v___x_2510_; lean_object* v___x_2512_; 
v___x_2510_ = lean_box(0);
if (v_isShared_2509_ == 0)
{
lean_ctor_set_tag(v___x_2508_, 0);
lean_ctor_set(v___x_2508_, 0, v___x_2510_);
v___x_2512_ = v___x_2508_;
goto v_reusejp_2511_;
}
else
{
lean_object* v_reuseFailAlloc_2513_; 
v_reuseFailAlloc_2513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2513_, 0, v___x_2510_);
v___x_2512_ = v_reuseFailAlloc_2513_;
goto v_reusejp_2511_;
}
v_reusejp_2511_:
{
return v___x_2512_;
}
}
}
else
{
return v___x_2503_;
}
}
}
}
else
{
lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; uint8_t v___x_2521_; 
v___x_2518_ = lean_unsigned_to_nat(2u);
v___x_2519_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2518_);
v___x_2520_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2521_ = l_Lean_Syntax_isOfKind(v___x_2519_, v___x_2520_);
if (v___x_2521_ == 0)
{
lean_object* v___x_2522_; lean_object* v___x_2523_; 
v___x_2522_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2523_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2522_, v___x_2521_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2523_) == 0)
{
return v___x_2523_;
}
else
{
lean_object* v_a_2524_; uint8_t v___y_2526_; uint8_t v___x_2536_; 
v_a_2524_ = lean_ctor_get(v___x_2523_, 0);
lean_inc(v_a_2524_);
v___x_2536_ = l_Lean_Exception_isInterrupt(v_a_2524_);
if (v___x_2536_ == 0)
{
uint8_t v___x_2537_; 
v___x_2537_ = l_Lean_Exception_isRuntime(v_a_2524_);
v___y_2526_ = v___x_2537_;
goto v___jp_2525_;
}
else
{
lean_dec(v_a_2524_);
v___y_2526_ = v___x_2536_;
goto v___jp_2525_;
}
v___jp_2525_:
{
if (v___y_2526_ == 0)
{
lean_object* v___x_2528_; uint8_t v_isShared_2529_; uint8_t v_isSharedCheck_2534_; 
v_isSharedCheck_2534_ = !lean_is_exclusive(v___x_2523_);
if (v_isSharedCheck_2534_ == 0)
{
lean_object* v_unused_2535_; 
v_unused_2535_ = lean_ctor_get(v___x_2523_, 0);
lean_dec(v_unused_2535_);
v___x_2528_ = v___x_2523_;
v_isShared_2529_ = v_isSharedCheck_2534_;
goto v_resetjp_2527_;
}
else
{
lean_dec(v___x_2523_);
v___x_2528_ = lean_box(0);
v_isShared_2529_ = v_isSharedCheck_2534_;
goto v_resetjp_2527_;
}
v_resetjp_2527_:
{
lean_object* v___x_2530_; lean_object* v___x_2532_; 
v___x_2530_ = lean_box(0);
if (v_isShared_2529_ == 0)
{
lean_ctor_set_tag(v___x_2528_, 0);
lean_ctor_set(v___x_2528_, 0, v___x_2530_);
v___x_2532_ = v___x_2528_;
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
return v___x_2523_;
}
}
}
}
else
{
lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; 
v___x_2538_ = lean_unsigned_to_nat(1u);
v___x_2539_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2538_);
lean_dec(v_a_2482_);
v___x_2540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2541_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2539_, v___x_2540_, v___x_2499_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2541_) == 0)
{
return v___x_2541_;
}
else
{
lean_object* v_a_2542_; uint8_t v___y_2544_; uint8_t v___x_2554_; 
v_a_2542_ = lean_ctor_get(v___x_2541_, 0);
lean_inc(v_a_2542_);
v___x_2554_ = l_Lean_Exception_isInterrupt(v_a_2542_);
if (v___x_2554_ == 0)
{
uint8_t v___x_2555_; 
v___x_2555_ = l_Lean_Exception_isRuntime(v_a_2542_);
v___y_2544_ = v___x_2555_;
goto v___jp_2543_;
}
else
{
lean_dec(v_a_2542_);
v___y_2544_ = v___x_2554_;
goto v___jp_2543_;
}
v___jp_2543_:
{
if (v___y_2544_ == 0)
{
lean_object* v___x_2546_; uint8_t v_isShared_2547_; uint8_t v_isSharedCheck_2552_; 
v_isSharedCheck_2552_ = !lean_is_exclusive(v___x_2541_);
if (v_isSharedCheck_2552_ == 0)
{
lean_object* v_unused_2553_; 
v_unused_2553_ = lean_ctor_get(v___x_2541_, 0);
lean_dec(v_unused_2553_);
v___x_2546_ = v___x_2541_;
v_isShared_2547_ = v_isSharedCheck_2552_;
goto v_resetjp_2545_;
}
else
{
lean_dec(v___x_2541_);
v___x_2546_ = lean_box(0);
v_isShared_2547_ = v_isSharedCheck_2552_;
goto v_resetjp_2545_;
}
v_resetjp_2545_:
{
lean_object* v___x_2548_; lean_object* v___x_2550_; 
v___x_2548_ = lean_box(0);
if (v_isShared_2547_ == 0)
{
lean_ctor_set_tag(v___x_2546_, 0);
lean_ctor_set(v___x_2546_, 0, v___x_2548_);
v___x_2550_ = v___x_2546_;
goto v_reusejp_2549_;
}
else
{
lean_object* v_reuseFailAlloc_2551_; 
v_reuseFailAlloc_2551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2551_, 0, v___x_2548_);
v___x_2550_ = v_reuseFailAlloc_2551_;
goto v_reusejp_2549_;
}
v_reusejp_2549_:
{
return v___x_2550_;
}
}
}
else
{
return v___x_2541_;
}
}
}
}
}
}
else
{
lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; uint8_t v___x_2559_; 
v___x_2556_ = lean_unsigned_to_nat(2u);
v___x_2557_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2556_);
v___x_2558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2559_ = l_Lean_Syntax_isOfKind(v___x_2557_, v___x_2558_);
if (v___x_2559_ == 0)
{
lean_object* v___x_2560_; lean_object* v___x_2561_; 
v___x_2560_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2561_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2560_, v___x_2559_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2561_) == 0)
{
return v___x_2561_;
}
else
{
lean_object* v_a_2562_; uint8_t v___y_2564_; uint8_t v___x_2574_; 
v_a_2562_ = lean_ctor_get(v___x_2561_, 0);
lean_inc(v_a_2562_);
v___x_2574_ = l_Lean_Exception_isInterrupt(v_a_2562_);
if (v___x_2574_ == 0)
{
uint8_t v___x_2575_; 
v___x_2575_ = l_Lean_Exception_isRuntime(v_a_2562_);
v___y_2564_ = v___x_2575_;
goto v___jp_2563_;
}
else
{
lean_dec(v_a_2562_);
v___y_2564_ = v___x_2574_;
goto v___jp_2563_;
}
v___jp_2563_:
{
if (v___y_2564_ == 0)
{
lean_object* v___x_2566_; uint8_t v_isShared_2567_; uint8_t v_isSharedCheck_2572_; 
v_isSharedCheck_2572_ = !lean_is_exclusive(v___x_2561_);
if (v_isSharedCheck_2572_ == 0)
{
lean_object* v_unused_2573_; 
v_unused_2573_ = lean_ctor_get(v___x_2561_, 0);
lean_dec(v_unused_2573_);
v___x_2566_ = v___x_2561_;
v_isShared_2567_ = v_isSharedCheck_2572_;
goto v_resetjp_2565_;
}
else
{
lean_dec(v___x_2561_);
v___x_2566_ = lean_box(0);
v_isShared_2567_ = v_isSharedCheck_2572_;
goto v_resetjp_2565_;
}
v_resetjp_2565_:
{
lean_object* v___x_2568_; lean_object* v___x_2570_; 
v___x_2568_ = lean_box(0);
if (v_isShared_2567_ == 0)
{
lean_ctor_set_tag(v___x_2566_, 0);
lean_ctor_set(v___x_2566_, 0, v___x_2568_);
v___x_2570_ = v___x_2566_;
goto v_reusejp_2569_;
}
else
{
lean_object* v_reuseFailAlloc_2571_; 
v_reuseFailAlloc_2571_ = lean_alloc_ctor(0, 1, 0);
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
else
{
return v___x_2561_;
}
}
}
}
else
{
lean_object* v___x_2576_; lean_object* v___x_2577_; uint8_t v___x_2578_; 
v___x_2576_ = lean_unsigned_to_nat(3u);
v___x_2577_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2576_);
v___x_2578_ = l_Lean_Syntax_isOfKind(v___x_2577_, v___x_2558_);
if (v___x_2578_ == 0)
{
lean_object* v___x_2579_; lean_object* v___x_2580_; 
v___x_2579_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2580_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2579_, v___x_2578_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2580_) == 0)
{
return v___x_2580_;
}
else
{
lean_object* v_a_2581_; uint8_t v___y_2583_; uint8_t v___x_2593_; 
v_a_2581_ = lean_ctor_get(v___x_2580_, 0);
lean_inc(v_a_2581_);
v___x_2593_ = l_Lean_Exception_isInterrupt(v_a_2581_);
if (v___x_2593_ == 0)
{
uint8_t v___x_2594_; 
v___x_2594_ = l_Lean_Exception_isRuntime(v_a_2581_);
v___y_2583_ = v___x_2594_;
goto v___jp_2582_;
}
else
{
lean_dec(v_a_2581_);
v___y_2583_ = v___x_2593_;
goto v___jp_2582_;
}
v___jp_2582_:
{
if (v___y_2583_ == 0)
{
lean_object* v___x_2585_; uint8_t v_isShared_2586_; uint8_t v_isSharedCheck_2591_; 
v_isSharedCheck_2591_ = !lean_is_exclusive(v___x_2580_);
if (v_isSharedCheck_2591_ == 0)
{
lean_object* v_unused_2592_; 
v_unused_2592_ = lean_ctor_get(v___x_2580_, 0);
lean_dec(v_unused_2592_);
v___x_2585_ = v___x_2580_;
v_isShared_2586_ = v_isSharedCheck_2591_;
goto v_resetjp_2584_;
}
else
{
lean_dec(v___x_2580_);
v___x_2585_ = lean_box(0);
v_isShared_2586_ = v_isSharedCheck_2591_;
goto v_resetjp_2584_;
}
v_resetjp_2584_:
{
lean_object* v___x_2587_; lean_object* v___x_2589_; 
v___x_2587_ = lean_box(0);
if (v_isShared_2586_ == 0)
{
lean_ctor_set_tag(v___x_2585_, 0);
lean_ctor_set(v___x_2585_, 0, v___x_2587_);
v___x_2589_ = v___x_2585_;
goto v_reusejp_2588_;
}
else
{
lean_object* v_reuseFailAlloc_2590_; 
v_reuseFailAlloc_2590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2590_, 0, v___x_2587_);
v___x_2589_ = v_reuseFailAlloc_2590_;
goto v_reusejp_2588_;
}
v_reusejp_2588_:
{
return v___x_2589_;
}
}
}
else
{
return v___x_2580_;
}
}
}
}
else
{
lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; 
v___x_2595_ = lean_unsigned_to_nat(1u);
v___x_2596_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2595_);
lean_dec(v_a_2482_);
v___x_2597_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2598_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2596_, v___x_2597_, v___x_2497_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2598_) == 0)
{
return v___x_2598_;
}
else
{
lean_object* v_a_2599_; uint8_t v___y_2601_; uint8_t v___x_2611_; 
v_a_2599_ = lean_ctor_get(v___x_2598_, 0);
lean_inc(v_a_2599_);
v___x_2611_ = l_Lean_Exception_isInterrupt(v_a_2599_);
if (v___x_2611_ == 0)
{
uint8_t v___x_2612_; 
v___x_2612_ = l_Lean_Exception_isRuntime(v_a_2599_);
v___y_2601_ = v___x_2612_;
goto v___jp_2600_;
}
else
{
lean_dec(v_a_2599_);
v___y_2601_ = v___x_2611_;
goto v___jp_2600_;
}
v___jp_2600_:
{
if (v___y_2601_ == 0)
{
lean_object* v___x_2603_; uint8_t v_isShared_2604_; uint8_t v_isSharedCheck_2609_; 
v_isSharedCheck_2609_ = !lean_is_exclusive(v___x_2598_);
if (v_isSharedCheck_2609_ == 0)
{
lean_object* v_unused_2610_; 
v_unused_2610_ = lean_ctor_get(v___x_2598_, 0);
lean_dec(v_unused_2610_);
v___x_2603_ = v___x_2598_;
v_isShared_2604_ = v_isSharedCheck_2609_;
goto v_resetjp_2602_;
}
else
{
lean_dec(v___x_2598_);
v___x_2603_ = lean_box(0);
v_isShared_2604_ = v_isSharedCheck_2609_;
goto v_resetjp_2602_;
}
v_resetjp_2602_:
{
lean_object* v___x_2605_; lean_object* v___x_2607_; 
v___x_2605_ = lean_box(0);
if (v_isShared_2604_ == 0)
{
lean_ctor_set_tag(v___x_2603_, 0);
lean_ctor_set(v___x_2603_, 0, v___x_2605_);
v___x_2607_ = v___x_2603_;
goto v_reusejp_2606_;
}
else
{
lean_object* v_reuseFailAlloc_2608_; 
v_reuseFailAlloc_2608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2608_, 0, v___x_2605_);
v___x_2607_ = v_reuseFailAlloc_2608_;
goto v_reusejp_2606_;
}
v_reusejp_2606_:
{
return v___x_2607_;
}
}
}
else
{
return v___x_2598_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; uint8_t v___x_2616_; 
v___x_2613_ = lean_unsigned_to_nat(2u);
v___x_2614_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2613_);
v___x_2615_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2616_ = l_Lean_Syntax_isOfKind(v___x_2614_, v___x_2615_);
if (v___x_2616_ == 0)
{
lean_object* v___x_2617_; lean_object* v___x_2618_; 
v___x_2617_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2618_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2617_, v___x_2616_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2618_) == 0)
{
return v___x_2618_;
}
else
{
lean_object* v_a_2619_; uint8_t v___y_2621_; uint8_t v___x_2631_; 
v_a_2619_ = lean_ctor_get(v___x_2618_, 0);
lean_inc(v_a_2619_);
v___x_2631_ = l_Lean_Exception_isInterrupt(v_a_2619_);
if (v___x_2631_ == 0)
{
uint8_t v___x_2632_; 
v___x_2632_ = l_Lean_Exception_isRuntime(v_a_2619_);
v___y_2621_ = v___x_2632_;
goto v___jp_2620_;
}
else
{
lean_dec(v_a_2619_);
v___y_2621_ = v___x_2631_;
goto v___jp_2620_;
}
v___jp_2620_:
{
if (v___y_2621_ == 0)
{
lean_object* v___x_2623_; uint8_t v_isShared_2624_; uint8_t v_isSharedCheck_2629_; 
v_isSharedCheck_2629_ = !lean_is_exclusive(v___x_2618_);
if (v_isSharedCheck_2629_ == 0)
{
lean_object* v_unused_2630_; 
v_unused_2630_ = lean_ctor_get(v___x_2618_, 0);
lean_dec(v_unused_2630_);
v___x_2623_ = v___x_2618_;
v_isShared_2624_ = v_isSharedCheck_2629_;
goto v_resetjp_2622_;
}
else
{
lean_dec(v___x_2618_);
v___x_2623_ = lean_box(0);
v_isShared_2624_ = v_isSharedCheck_2629_;
goto v_resetjp_2622_;
}
v_resetjp_2622_:
{
lean_object* v___x_2625_; lean_object* v___x_2627_; 
v___x_2625_ = lean_box(0);
if (v_isShared_2624_ == 0)
{
lean_ctor_set_tag(v___x_2623_, 0);
lean_ctor_set(v___x_2623_, 0, v___x_2625_);
v___x_2627_ = v___x_2623_;
goto v_reusejp_2626_;
}
else
{
lean_object* v_reuseFailAlloc_2628_; 
v_reuseFailAlloc_2628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2628_, 0, v___x_2625_);
v___x_2627_ = v_reuseFailAlloc_2628_;
goto v_reusejp_2626_;
}
v_reusejp_2626_:
{
return v___x_2627_;
}
}
}
else
{
return v___x_2618_;
}
}
}
}
else
{
lean_object* v___x_2633_; lean_object* v___x_2634_; uint8_t v___x_2635_; 
v___x_2633_ = lean_unsigned_to_nat(3u);
v___x_2634_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2633_);
v___x_2635_ = l_Lean_Syntax_isOfKind(v___x_2634_, v___x_2615_);
if (v___x_2635_ == 0)
{
lean_object* v___x_2636_; lean_object* v___x_2637_; 
v___x_2636_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2637_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2636_, v___x_2635_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2637_) == 0)
{
return v___x_2637_;
}
else
{
lean_object* v_a_2638_; uint8_t v___y_2640_; uint8_t v___x_2650_; 
v_a_2638_ = lean_ctor_get(v___x_2637_, 0);
lean_inc(v_a_2638_);
v___x_2650_ = l_Lean_Exception_isInterrupt(v_a_2638_);
if (v___x_2650_ == 0)
{
uint8_t v___x_2651_; 
v___x_2651_ = l_Lean_Exception_isRuntime(v_a_2638_);
v___y_2640_ = v___x_2651_;
goto v___jp_2639_;
}
else
{
lean_dec(v_a_2638_);
v___y_2640_ = v___x_2650_;
goto v___jp_2639_;
}
v___jp_2639_:
{
if (v___y_2640_ == 0)
{
lean_object* v___x_2642_; uint8_t v_isShared_2643_; uint8_t v_isSharedCheck_2648_; 
v_isSharedCheck_2648_ = !lean_is_exclusive(v___x_2637_);
if (v_isSharedCheck_2648_ == 0)
{
lean_object* v_unused_2649_; 
v_unused_2649_ = lean_ctor_get(v___x_2637_, 0);
lean_dec(v_unused_2649_);
v___x_2642_ = v___x_2637_;
v_isShared_2643_ = v_isSharedCheck_2648_;
goto v_resetjp_2641_;
}
else
{
lean_dec(v___x_2637_);
v___x_2642_ = lean_box(0);
v_isShared_2643_ = v_isSharedCheck_2648_;
goto v_resetjp_2641_;
}
v_resetjp_2641_:
{
lean_object* v___x_2644_; lean_object* v___x_2646_; 
v___x_2644_ = lean_box(0);
if (v_isShared_2643_ == 0)
{
lean_ctor_set_tag(v___x_2642_, 0);
lean_ctor_set(v___x_2642_, 0, v___x_2644_);
v___x_2646_ = v___x_2642_;
goto v_reusejp_2645_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2647_, 0, v___x_2644_);
v___x_2646_ = v_reuseFailAlloc_2647_;
goto v_reusejp_2645_;
}
v_reusejp_2645_:
{
return v___x_2646_;
}
}
}
else
{
return v___x_2637_;
}
}
}
}
else
{
lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; 
v___x_2652_ = lean_unsigned_to_nat(1u);
v___x_2653_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2652_);
lean_dec(v_a_2482_);
v___x_2654_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2655_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2653_, v___x_2654_, v___x_2495_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2655_) == 0)
{
return v___x_2655_;
}
else
{
lean_object* v_a_2656_; uint8_t v___y_2658_; uint8_t v___x_2668_; 
v_a_2656_ = lean_ctor_get(v___x_2655_, 0);
lean_inc(v_a_2656_);
v___x_2668_ = l_Lean_Exception_isInterrupt(v_a_2656_);
if (v___x_2668_ == 0)
{
uint8_t v___x_2669_; 
v___x_2669_ = l_Lean_Exception_isRuntime(v_a_2656_);
v___y_2658_ = v___x_2669_;
goto v___jp_2657_;
}
else
{
lean_dec(v_a_2656_);
v___y_2658_ = v___x_2668_;
goto v___jp_2657_;
}
v___jp_2657_:
{
if (v___y_2658_ == 0)
{
lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2666_; 
v_isSharedCheck_2666_ = !lean_is_exclusive(v___x_2655_);
if (v_isSharedCheck_2666_ == 0)
{
lean_object* v_unused_2667_; 
v_unused_2667_ = lean_ctor_get(v___x_2655_, 0);
lean_dec(v_unused_2667_);
v___x_2660_ = v___x_2655_;
v_isShared_2661_ = v_isSharedCheck_2666_;
goto v_resetjp_2659_;
}
else
{
lean_dec(v___x_2655_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2666_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v___x_2662_; lean_object* v___x_2664_; 
v___x_2662_ = lean_box(0);
if (v_isShared_2661_ == 0)
{
lean_ctor_set_tag(v___x_2660_, 0);
lean_ctor_set(v___x_2660_, 0, v___x_2662_);
v___x_2664_ = v___x_2660_;
goto v_reusejp_2663_;
}
else
{
lean_object* v_reuseFailAlloc_2665_; 
v_reuseFailAlloc_2665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2665_, 0, v___x_2662_);
v___x_2664_ = v_reuseFailAlloc_2665_;
goto v_reusejp_2663_;
}
v_reusejp_2663_:
{
return v___x_2664_;
}
}
}
else
{
return v___x_2655_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; uint8_t v___x_2673_; 
v___x_2670_ = lean_unsigned_to_nat(2u);
v___x_2671_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2670_);
v___x_2672_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2673_ = l_Lean_Syntax_isOfKind(v___x_2671_, v___x_2672_);
if (v___x_2673_ == 0)
{
lean_object* v___x_2674_; lean_object* v___x_2675_; 
v___x_2674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2675_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2674_, v___x_2673_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2675_) == 0)
{
return v___x_2675_;
}
else
{
lean_object* v_a_2676_; uint8_t v___y_2678_; uint8_t v___x_2688_; 
v_a_2676_ = lean_ctor_get(v___x_2675_, 0);
lean_inc(v_a_2676_);
v___x_2688_ = l_Lean_Exception_isInterrupt(v_a_2676_);
if (v___x_2688_ == 0)
{
uint8_t v___x_2689_; 
v___x_2689_ = l_Lean_Exception_isRuntime(v_a_2676_);
v___y_2678_ = v___x_2689_;
goto v___jp_2677_;
}
else
{
lean_dec(v_a_2676_);
v___y_2678_ = v___x_2688_;
goto v___jp_2677_;
}
v___jp_2677_:
{
if (v___y_2678_ == 0)
{
lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2686_; 
v_isSharedCheck_2686_ = !lean_is_exclusive(v___x_2675_);
if (v_isSharedCheck_2686_ == 0)
{
lean_object* v_unused_2687_; 
v_unused_2687_ = lean_ctor_get(v___x_2675_, 0);
lean_dec(v_unused_2687_);
v___x_2680_ = v___x_2675_;
v_isShared_2681_ = v_isSharedCheck_2686_;
goto v_resetjp_2679_;
}
else
{
lean_dec(v___x_2675_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2686_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___x_2682_; lean_object* v___x_2684_; 
v___x_2682_ = lean_box(0);
if (v_isShared_2681_ == 0)
{
lean_ctor_set_tag(v___x_2680_, 0);
lean_ctor_set(v___x_2680_, 0, v___x_2682_);
v___x_2684_ = v___x_2680_;
goto v_reusejp_2683_;
}
else
{
lean_object* v_reuseFailAlloc_2685_; 
v_reuseFailAlloc_2685_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2685_, 0, v___x_2682_);
v___x_2684_ = v_reuseFailAlloc_2685_;
goto v_reusejp_2683_;
}
v_reusejp_2683_:
{
return v___x_2684_;
}
}
}
else
{
return v___x_2675_;
}
}
}
}
else
{
lean_object* v___x_2690_; lean_object* v___x_2691_; uint8_t v___x_2692_; 
v___x_2690_ = lean_unsigned_to_nat(3u);
v___x_2691_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2690_);
v___x_2692_ = l_Lean_Syntax_isOfKind(v___x_2691_, v___x_2672_);
if (v___x_2692_ == 0)
{
lean_object* v___x_2693_; lean_object* v___x_2694_; 
v___x_2693_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2694_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2693_, v___x_2692_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2694_) == 0)
{
return v___x_2694_;
}
else
{
lean_object* v_a_2695_; uint8_t v___y_2697_; uint8_t v___x_2707_; 
v_a_2695_ = lean_ctor_get(v___x_2694_, 0);
lean_inc(v_a_2695_);
v___x_2707_ = l_Lean_Exception_isInterrupt(v_a_2695_);
if (v___x_2707_ == 0)
{
uint8_t v___x_2708_; 
v___x_2708_ = l_Lean_Exception_isRuntime(v_a_2695_);
v___y_2697_ = v___x_2708_;
goto v___jp_2696_;
}
else
{
lean_dec(v_a_2695_);
v___y_2697_ = v___x_2707_;
goto v___jp_2696_;
}
v___jp_2696_:
{
if (v___y_2697_ == 0)
{
lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2705_; 
v_isSharedCheck_2705_ = !lean_is_exclusive(v___x_2694_);
if (v_isSharedCheck_2705_ == 0)
{
lean_object* v_unused_2706_; 
v_unused_2706_ = lean_ctor_get(v___x_2694_, 0);
lean_dec(v_unused_2706_);
v___x_2699_ = v___x_2694_;
v_isShared_2700_ = v_isSharedCheck_2705_;
goto v_resetjp_2698_;
}
else
{
lean_dec(v___x_2694_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2705_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
lean_object* v___x_2701_; lean_object* v___x_2703_; 
v___x_2701_ = lean_box(0);
if (v_isShared_2700_ == 0)
{
lean_ctor_set_tag(v___x_2699_, 0);
lean_ctor_set(v___x_2699_, 0, v___x_2701_);
v___x_2703_ = v___x_2699_;
goto v_reusejp_2702_;
}
else
{
lean_object* v_reuseFailAlloc_2704_; 
v_reuseFailAlloc_2704_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2704_, 0, v___x_2701_);
v___x_2703_ = v_reuseFailAlloc_2704_;
goto v_reusejp_2702_;
}
v_reusejp_2702_:
{
return v___x_2703_;
}
}
}
else
{
return v___x_2694_;
}
}
}
}
else
{
lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; 
v___x_2709_ = lean_unsigned_to_nat(1u);
v___x_2710_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2709_);
lean_dec(v_a_2482_);
v___x_2711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2712_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2710_, v___x_2711_, v___x_2493_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2712_) == 0)
{
return v___x_2712_;
}
else
{
lean_object* v_a_2713_; uint8_t v___y_2715_; uint8_t v___x_2725_; 
v_a_2713_ = lean_ctor_get(v___x_2712_, 0);
lean_inc(v_a_2713_);
v___x_2725_ = l_Lean_Exception_isInterrupt(v_a_2713_);
if (v___x_2725_ == 0)
{
uint8_t v___x_2726_; 
v___x_2726_ = l_Lean_Exception_isRuntime(v_a_2713_);
v___y_2715_ = v___x_2726_;
goto v___jp_2714_;
}
else
{
lean_dec(v_a_2713_);
v___y_2715_ = v___x_2725_;
goto v___jp_2714_;
}
v___jp_2714_:
{
if (v___y_2715_ == 0)
{
lean_object* v___x_2717_; uint8_t v_isShared_2718_; uint8_t v_isSharedCheck_2723_; 
v_isSharedCheck_2723_ = !lean_is_exclusive(v___x_2712_);
if (v_isSharedCheck_2723_ == 0)
{
lean_object* v_unused_2724_; 
v_unused_2724_ = lean_ctor_get(v___x_2712_, 0);
lean_dec(v_unused_2724_);
v___x_2717_ = v___x_2712_;
v_isShared_2718_ = v_isSharedCheck_2723_;
goto v_resetjp_2716_;
}
else
{
lean_dec(v___x_2712_);
v___x_2717_ = lean_box(0);
v_isShared_2718_ = v_isSharedCheck_2723_;
goto v_resetjp_2716_;
}
v_resetjp_2716_:
{
lean_object* v___x_2719_; lean_object* v___x_2721_; 
v___x_2719_ = lean_box(0);
if (v_isShared_2718_ == 0)
{
lean_ctor_set_tag(v___x_2717_, 0);
lean_ctor_set(v___x_2717_, 0, v___x_2719_);
v___x_2721_ = v___x_2717_;
goto v_reusejp_2720_;
}
else
{
lean_object* v_reuseFailAlloc_2722_; 
v_reuseFailAlloc_2722_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2722_, 0, v___x_2719_);
v___x_2721_ = v_reuseFailAlloc_2722_;
goto v_reusejp_2720_;
}
v_reusejp_2720_:
{
return v___x_2721_;
}
}
}
else
{
return v___x_2712_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2727_; lean_object* v___x_2728_; lean_object* v___x_2729_; uint8_t v___x_2730_; 
v___x_2727_ = lean_unsigned_to_nat(2u);
v___x_2728_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2727_);
v___x_2729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2730_ = l_Lean_Syntax_isOfKind(v___x_2728_, v___x_2729_);
if (v___x_2730_ == 0)
{
lean_object* v___x_2731_; lean_object* v___x_2732_; 
v___x_2731_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2732_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2731_, v___x_2730_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2732_) == 0)
{
return v___x_2732_;
}
else
{
lean_object* v_a_2733_; uint8_t v___y_2735_; uint8_t v___x_2745_; 
v_a_2733_ = lean_ctor_get(v___x_2732_, 0);
lean_inc(v_a_2733_);
v___x_2745_ = l_Lean_Exception_isInterrupt(v_a_2733_);
if (v___x_2745_ == 0)
{
uint8_t v___x_2746_; 
v___x_2746_ = l_Lean_Exception_isRuntime(v_a_2733_);
v___y_2735_ = v___x_2746_;
goto v___jp_2734_;
}
else
{
lean_dec(v_a_2733_);
v___y_2735_ = v___x_2745_;
goto v___jp_2734_;
}
v___jp_2734_:
{
if (v___y_2735_ == 0)
{
lean_object* v___x_2737_; uint8_t v_isShared_2738_; uint8_t v_isSharedCheck_2743_; 
v_isSharedCheck_2743_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2743_ == 0)
{
lean_object* v_unused_2744_; 
v_unused_2744_ = lean_ctor_get(v___x_2732_, 0);
lean_dec(v_unused_2744_);
v___x_2737_ = v___x_2732_;
v_isShared_2738_ = v_isSharedCheck_2743_;
goto v_resetjp_2736_;
}
else
{
lean_dec(v___x_2732_);
v___x_2737_ = lean_box(0);
v_isShared_2738_ = v_isSharedCheck_2743_;
goto v_resetjp_2736_;
}
v_resetjp_2736_:
{
lean_object* v___x_2739_; lean_object* v___x_2741_; 
v___x_2739_ = lean_box(0);
if (v_isShared_2738_ == 0)
{
lean_ctor_set_tag(v___x_2737_, 0);
lean_ctor_set(v___x_2737_, 0, v___x_2739_);
v___x_2741_ = v___x_2737_;
goto v_reusejp_2740_;
}
else
{
lean_object* v_reuseFailAlloc_2742_; 
v_reuseFailAlloc_2742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2742_, 0, v___x_2739_);
v___x_2741_ = v_reuseFailAlloc_2742_;
goto v_reusejp_2740_;
}
v_reusejp_2740_:
{
return v___x_2741_;
}
}
}
else
{
return v___x_2732_;
}
}
}
}
else
{
lean_object* v___x_2747_; lean_object* v___x_2748_; uint8_t v___x_2749_; 
v___x_2747_ = lean_unsigned_to_nat(3u);
v___x_2748_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2747_);
v___x_2749_ = l_Lean_Syntax_isOfKind(v___x_2748_, v___x_2729_);
if (v___x_2749_ == 0)
{
lean_object* v___x_2750_; lean_object* v___x_2751_; 
v___x_2750_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2751_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2750_, v___x_2749_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2751_) == 0)
{
return v___x_2751_;
}
else
{
lean_object* v_a_2752_; uint8_t v___y_2754_; uint8_t v___x_2764_; 
v_a_2752_ = lean_ctor_get(v___x_2751_, 0);
lean_inc(v_a_2752_);
v___x_2764_ = l_Lean_Exception_isInterrupt(v_a_2752_);
if (v___x_2764_ == 0)
{
uint8_t v___x_2765_; 
v___x_2765_ = l_Lean_Exception_isRuntime(v_a_2752_);
v___y_2754_ = v___x_2765_;
goto v___jp_2753_;
}
else
{
lean_dec(v_a_2752_);
v___y_2754_ = v___x_2764_;
goto v___jp_2753_;
}
v___jp_2753_:
{
if (v___y_2754_ == 0)
{
lean_object* v___x_2756_; uint8_t v_isShared_2757_; uint8_t v_isSharedCheck_2762_; 
v_isSharedCheck_2762_ = !lean_is_exclusive(v___x_2751_);
if (v_isSharedCheck_2762_ == 0)
{
lean_object* v_unused_2763_; 
v_unused_2763_ = lean_ctor_get(v___x_2751_, 0);
lean_dec(v_unused_2763_);
v___x_2756_ = v___x_2751_;
v_isShared_2757_ = v_isSharedCheck_2762_;
goto v_resetjp_2755_;
}
else
{
lean_dec(v___x_2751_);
v___x_2756_ = lean_box(0);
v_isShared_2757_ = v_isSharedCheck_2762_;
goto v_resetjp_2755_;
}
v_resetjp_2755_:
{
lean_object* v___x_2758_; lean_object* v___x_2760_; 
v___x_2758_ = lean_box(0);
if (v_isShared_2757_ == 0)
{
lean_ctor_set_tag(v___x_2756_, 0);
lean_ctor_set(v___x_2756_, 0, v___x_2758_);
v___x_2760_ = v___x_2756_;
goto v_reusejp_2759_;
}
else
{
lean_object* v_reuseFailAlloc_2761_; 
v_reuseFailAlloc_2761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2761_, 0, v___x_2758_);
v___x_2760_ = v_reuseFailAlloc_2761_;
goto v_reusejp_2759_;
}
v_reusejp_2759_:
{
return v___x_2760_;
}
}
}
else
{
return v___x_2751_;
}
}
}
}
else
{
lean_object* v___x_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v___x_2769_; 
v___x_2766_ = lean_unsigned_to_nat(1u);
v___x_2767_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2766_);
lean_dec(v_a_2482_);
v___x_2768_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2769_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2767_, v___x_2768_, v___x_2491_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2769_) == 0)
{
return v___x_2769_;
}
else
{
lean_object* v_a_2770_; uint8_t v___y_2772_; uint8_t v___x_2782_; 
v_a_2770_ = lean_ctor_get(v___x_2769_, 0);
lean_inc(v_a_2770_);
v___x_2782_ = l_Lean_Exception_isInterrupt(v_a_2770_);
if (v___x_2782_ == 0)
{
uint8_t v___x_2783_; 
v___x_2783_ = l_Lean_Exception_isRuntime(v_a_2770_);
v___y_2772_ = v___x_2783_;
goto v___jp_2771_;
}
else
{
lean_dec(v_a_2770_);
v___y_2772_ = v___x_2782_;
goto v___jp_2771_;
}
v___jp_2771_:
{
if (v___y_2772_ == 0)
{
lean_object* v___x_2774_; uint8_t v_isShared_2775_; uint8_t v_isSharedCheck_2780_; 
v_isSharedCheck_2780_ = !lean_is_exclusive(v___x_2769_);
if (v_isSharedCheck_2780_ == 0)
{
lean_object* v_unused_2781_; 
v_unused_2781_ = lean_ctor_get(v___x_2769_, 0);
lean_dec(v_unused_2781_);
v___x_2774_ = v___x_2769_;
v_isShared_2775_ = v_isSharedCheck_2780_;
goto v_resetjp_2773_;
}
else
{
lean_dec(v___x_2769_);
v___x_2774_ = lean_box(0);
v_isShared_2775_ = v_isSharedCheck_2780_;
goto v_resetjp_2773_;
}
v_resetjp_2773_:
{
lean_object* v___x_2776_; lean_object* v___x_2778_; 
v___x_2776_ = lean_box(0);
if (v_isShared_2775_ == 0)
{
lean_ctor_set_tag(v___x_2774_, 0);
lean_ctor_set(v___x_2774_, 0, v___x_2776_);
v___x_2778_ = v___x_2774_;
goto v_reusejp_2777_;
}
else
{
lean_object* v_reuseFailAlloc_2779_; 
v_reuseFailAlloc_2779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2779_, 0, v___x_2776_);
v___x_2778_ = v_reuseFailAlloc_2779_;
goto v_reusejp_2777_;
}
v_reusejp_2777_:
{
return v___x_2778_;
}
}
}
else
{
return v___x_2769_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; uint8_t v___x_2787_; 
v___x_2784_ = lean_unsigned_to_nat(2u);
v___x_2785_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2784_);
v___x_2786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2787_ = l_Lean_Syntax_isOfKind(v___x_2785_, v___x_2786_);
if (v___x_2787_ == 0)
{
lean_object* v___x_2788_; lean_object* v___x_2789_; 
v___x_2788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2789_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2788_, v___x_2787_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2789_) == 0)
{
return v___x_2789_;
}
else
{
lean_object* v_a_2790_; uint8_t v___y_2792_; uint8_t v___x_2802_; 
v_a_2790_ = lean_ctor_get(v___x_2789_, 0);
lean_inc(v_a_2790_);
v___x_2802_ = l_Lean_Exception_isInterrupt(v_a_2790_);
if (v___x_2802_ == 0)
{
uint8_t v___x_2803_; 
v___x_2803_ = l_Lean_Exception_isRuntime(v_a_2790_);
v___y_2792_ = v___x_2803_;
goto v___jp_2791_;
}
else
{
lean_dec(v_a_2790_);
v___y_2792_ = v___x_2802_;
goto v___jp_2791_;
}
v___jp_2791_:
{
if (v___y_2792_ == 0)
{
lean_object* v___x_2794_; uint8_t v_isShared_2795_; uint8_t v_isSharedCheck_2800_; 
v_isSharedCheck_2800_ = !lean_is_exclusive(v___x_2789_);
if (v_isSharedCheck_2800_ == 0)
{
lean_object* v_unused_2801_; 
v_unused_2801_ = lean_ctor_get(v___x_2789_, 0);
lean_dec(v_unused_2801_);
v___x_2794_ = v___x_2789_;
v_isShared_2795_ = v_isSharedCheck_2800_;
goto v_resetjp_2793_;
}
else
{
lean_dec(v___x_2789_);
v___x_2794_ = lean_box(0);
v_isShared_2795_ = v_isSharedCheck_2800_;
goto v_resetjp_2793_;
}
v_resetjp_2793_:
{
lean_object* v___x_2796_; lean_object* v___x_2798_; 
v___x_2796_ = lean_box(0);
if (v_isShared_2795_ == 0)
{
lean_ctor_set_tag(v___x_2794_, 0);
lean_ctor_set(v___x_2794_, 0, v___x_2796_);
v___x_2798_ = v___x_2794_;
goto v_reusejp_2797_;
}
else
{
lean_object* v_reuseFailAlloc_2799_; 
v_reuseFailAlloc_2799_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2799_, 0, v___x_2796_);
v___x_2798_ = v_reuseFailAlloc_2799_;
goto v_reusejp_2797_;
}
v_reusejp_2797_:
{
return v___x_2798_;
}
}
}
else
{
return v___x_2789_;
}
}
}
}
else
{
lean_object* v___x_2804_; lean_object* v___x_2805_; uint8_t v___x_2806_; 
v___x_2804_ = lean_unsigned_to_nat(3u);
v___x_2805_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2804_);
v___x_2806_ = l_Lean_Syntax_isOfKind(v___x_2805_, v___x_2786_);
if (v___x_2806_ == 0)
{
lean_object* v___x_2807_; lean_object* v___x_2808_; 
v___x_2807_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2808_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2807_, v___x_2806_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2808_) == 0)
{
return v___x_2808_;
}
else
{
lean_object* v_a_2809_; uint8_t v___y_2811_; uint8_t v___x_2821_; 
v_a_2809_ = lean_ctor_get(v___x_2808_, 0);
lean_inc(v_a_2809_);
v___x_2821_ = l_Lean_Exception_isInterrupt(v_a_2809_);
if (v___x_2821_ == 0)
{
uint8_t v___x_2822_; 
v___x_2822_ = l_Lean_Exception_isRuntime(v_a_2809_);
v___y_2811_ = v___x_2822_;
goto v___jp_2810_;
}
else
{
lean_dec(v_a_2809_);
v___y_2811_ = v___x_2821_;
goto v___jp_2810_;
}
v___jp_2810_:
{
if (v___y_2811_ == 0)
{
lean_object* v___x_2813_; uint8_t v_isShared_2814_; uint8_t v_isSharedCheck_2819_; 
v_isSharedCheck_2819_ = !lean_is_exclusive(v___x_2808_);
if (v_isSharedCheck_2819_ == 0)
{
lean_object* v_unused_2820_; 
v_unused_2820_ = lean_ctor_get(v___x_2808_, 0);
lean_dec(v_unused_2820_);
v___x_2813_ = v___x_2808_;
v_isShared_2814_ = v_isSharedCheck_2819_;
goto v_resetjp_2812_;
}
else
{
lean_dec(v___x_2808_);
v___x_2813_ = lean_box(0);
v_isShared_2814_ = v_isSharedCheck_2819_;
goto v_resetjp_2812_;
}
v_resetjp_2812_:
{
lean_object* v___x_2815_; lean_object* v___x_2817_; 
v___x_2815_ = lean_box(0);
if (v_isShared_2814_ == 0)
{
lean_ctor_set_tag(v___x_2813_, 0);
lean_ctor_set(v___x_2813_, 0, v___x_2815_);
v___x_2817_ = v___x_2813_;
goto v_reusejp_2816_;
}
else
{
lean_object* v_reuseFailAlloc_2818_; 
v_reuseFailAlloc_2818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2818_, 0, v___x_2815_);
v___x_2817_ = v_reuseFailAlloc_2818_;
goto v_reusejp_2816_;
}
v_reusejp_2816_:
{
return v___x_2817_;
}
}
}
else
{
return v___x_2808_;
}
}
}
}
else
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; 
v___x_2823_ = lean_unsigned_to_nat(1u);
v___x_2824_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2823_);
lean_dec(v_a_2482_);
v___x_2825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2826_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2824_, v___x_2825_, v___x_2489_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2826_) == 0)
{
return v___x_2826_;
}
else
{
lean_object* v_a_2827_; uint8_t v___y_2829_; uint8_t v___x_2839_; 
v_a_2827_ = lean_ctor_get(v___x_2826_, 0);
lean_inc(v_a_2827_);
v___x_2839_ = l_Lean_Exception_isInterrupt(v_a_2827_);
if (v___x_2839_ == 0)
{
uint8_t v___x_2840_; 
v___x_2840_ = l_Lean_Exception_isRuntime(v_a_2827_);
v___y_2829_ = v___x_2840_;
goto v___jp_2828_;
}
else
{
lean_dec(v_a_2827_);
v___y_2829_ = v___x_2839_;
goto v___jp_2828_;
}
v___jp_2828_:
{
if (v___y_2829_ == 0)
{
lean_object* v___x_2831_; uint8_t v_isShared_2832_; uint8_t v_isSharedCheck_2837_; 
v_isSharedCheck_2837_ = !lean_is_exclusive(v___x_2826_);
if (v_isSharedCheck_2837_ == 0)
{
lean_object* v_unused_2838_; 
v_unused_2838_ = lean_ctor_get(v___x_2826_, 0);
lean_dec(v_unused_2838_);
v___x_2831_ = v___x_2826_;
v_isShared_2832_ = v_isSharedCheck_2837_;
goto v_resetjp_2830_;
}
else
{
lean_dec(v___x_2826_);
v___x_2831_ = lean_box(0);
v_isShared_2832_ = v_isSharedCheck_2837_;
goto v_resetjp_2830_;
}
v_resetjp_2830_:
{
lean_object* v___x_2833_; lean_object* v___x_2835_; 
v___x_2833_ = lean_box(0);
if (v_isShared_2832_ == 0)
{
lean_ctor_set_tag(v___x_2831_, 0);
lean_ctor_set(v___x_2831_, 0, v___x_2833_);
v___x_2835_ = v___x_2831_;
goto v_reusejp_2834_;
}
else
{
lean_object* v_reuseFailAlloc_2836_; 
v_reuseFailAlloc_2836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2836_, 0, v___x_2833_);
v___x_2835_ = v_reuseFailAlloc_2836_;
goto v_reusejp_2834_;
}
v_reusejp_2834_:
{
return v___x_2835_;
}
}
}
else
{
return v___x_2826_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; uint8_t v___x_2844_; 
v___x_2841_ = lean_unsigned_to_nat(2u);
v___x_2842_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2841_);
v___x_2843_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_2844_ = l_Lean_Syntax_isOfKind(v___x_2842_, v___x_2843_);
if (v___x_2844_ == 0)
{
lean_object* v___x_2845_; lean_object* v___x_2846_; 
v___x_2845_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2846_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2845_, v___x_2844_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2846_) == 0)
{
return v___x_2846_;
}
else
{
lean_object* v_a_2847_; uint8_t v___y_2849_; uint8_t v___x_2859_; 
v_a_2847_ = lean_ctor_get(v___x_2846_, 0);
lean_inc(v_a_2847_);
v___x_2859_ = l_Lean_Exception_isInterrupt(v_a_2847_);
if (v___x_2859_ == 0)
{
uint8_t v___x_2860_; 
v___x_2860_ = l_Lean_Exception_isRuntime(v_a_2847_);
v___y_2849_ = v___x_2860_;
goto v___jp_2848_;
}
else
{
lean_dec(v_a_2847_);
v___y_2849_ = v___x_2859_;
goto v___jp_2848_;
}
v___jp_2848_:
{
if (v___y_2849_ == 0)
{
lean_object* v___x_2851_; uint8_t v_isShared_2852_; uint8_t v_isSharedCheck_2857_; 
v_isSharedCheck_2857_ = !lean_is_exclusive(v___x_2846_);
if (v_isSharedCheck_2857_ == 0)
{
lean_object* v_unused_2858_; 
v_unused_2858_ = lean_ctor_get(v___x_2846_, 0);
lean_dec(v_unused_2858_);
v___x_2851_ = v___x_2846_;
v_isShared_2852_ = v_isSharedCheck_2857_;
goto v_resetjp_2850_;
}
else
{
lean_dec(v___x_2846_);
v___x_2851_ = lean_box(0);
v_isShared_2852_ = v_isSharedCheck_2857_;
goto v_resetjp_2850_;
}
v_resetjp_2850_:
{
lean_object* v___x_2853_; lean_object* v___x_2855_; 
v___x_2853_ = lean_box(0);
if (v_isShared_2852_ == 0)
{
lean_ctor_set_tag(v___x_2851_, 0);
lean_ctor_set(v___x_2851_, 0, v___x_2853_);
v___x_2855_ = v___x_2851_;
goto v_reusejp_2854_;
}
else
{
lean_object* v_reuseFailAlloc_2856_; 
v_reuseFailAlloc_2856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2856_, 0, v___x_2853_);
v___x_2855_ = v_reuseFailAlloc_2856_;
goto v_reusejp_2854_;
}
v_reusejp_2854_:
{
return v___x_2855_;
}
}
}
else
{
return v___x_2846_;
}
}
}
}
else
{
lean_object* v___x_2861_; lean_object* v___x_2862_; uint8_t v___x_2863_; 
v___x_2861_ = lean_unsigned_to_nat(3u);
v___x_2862_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2861_);
v___x_2863_ = l_Lean_Syntax_isOfKind(v___x_2862_, v___x_2843_);
if (v___x_2863_ == 0)
{
lean_object* v___x_2864_; lean_object* v___x_2865_; 
v___x_2864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2865_ = l_Lean_Elab_Term_resolveId_x3f(v_a_2482_, v___x_2864_, v___x_2863_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2865_) == 0)
{
return v___x_2865_;
}
else
{
lean_object* v_a_2866_; uint8_t v___y_2868_; uint8_t v___x_2878_; 
v_a_2866_ = lean_ctor_get(v___x_2865_, 0);
lean_inc(v_a_2866_);
v___x_2878_ = l_Lean_Exception_isInterrupt(v_a_2866_);
if (v___x_2878_ == 0)
{
uint8_t v___x_2879_; 
v___x_2879_ = l_Lean_Exception_isRuntime(v_a_2866_);
v___y_2868_ = v___x_2879_;
goto v___jp_2867_;
}
else
{
lean_dec(v_a_2866_);
v___y_2868_ = v___x_2878_;
goto v___jp_2867_;
}
v___jp_2867_:
{
if (v___y_2868_ == 0)
{
lean_object* v___x_2870_; uint8_t v_isShared_2871_; uint8_t v_isSharedCheck_2876_; 
v_isSharedCheck_2876_ = !lean_is_exclusive(v___x_2865_);
if (v_isSharedCheck_2876_ == 0)
{
lean_object* v_unused_2877_; 
v_unused_2877_ = lean_ctor_get(v___x_2865_, 0);
lean_dec(v_unused_2877_);
v___x_2870_ = v___x_2865_;
v_isShared_2871_ = v_isSharedCheck_2876_;
goto v_resetjp_2869_;
}
else
{
lean_dec(v___x_2865_);
v___x_2870_ = lean_box(0);
v_isShared_2871_ = v_isSharedCheck_2876_;
goto v_resetjp_2869_;
}
v_resetjp_2869_:
{
lean_object* v___x_2872_; lean_object* v___x_2874_; 
v___x_2872_ = lean_box(0);
if (v_isShared_2871_ == 0)
{
lean_ctor_set_tag(v___x_2870_, 0);
lean_ctor_set(v___x_2870_, 0, v___x_2872_);
v___x_2874_ = v___x_2870_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2875_; 
v_reuseFailAlloc_2875_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2875_, 0, v___x_2872_);
v___x_2874_ = v_reuseFailAlloc_2875_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
return v___x_2874_;
}
}
}
else
{
return v___x_2865_;
}
}
}
}
else
{
lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; 
v___x_2880_ = lean_unsigned_to_nat(1u);
v___x_2881_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2880_);
lean_dec(v_a_2482_);
v___x_2882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2883_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2881_, v___x_2882_, v___x_2487_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2883_) == 0)
{
return v___x_2883_;
}
else
{
lean_object* v_a_2884_; uint8_t v___y_2886_; uint8_t v___x_2896_; 
v_a_2884_ = lean_ctor_get(v___x_2883_, 0);
lean_inc(v_a_2884_);
v___x_2896_ = l_Lean_Exception_isInterrupt(v_a_2884_);
if (v___x_2896_ == 0)
{
uint8_t v___x_2897_; 
v___x_2897_ = l_Lean_Exception_isRuntime(v_a_2884_);
v___y_2886_ = v___x_2897_;
goto v___jp_2885_;
}
else
{
lean_dec(v_a_2884_);
v___y_2886_ = v___x_2896_;
goto v___jp_2885_;
}
v___jp_2885_:
{
if (v___y_2886_ == 0)
{
lean_object* v___x_2888_; uint8_t v_isShared_2889_; uint8_t v_isSharedCheck_2894_; 
v_isSharedCheck_2894_ = !lean_is_exclusive(v___x_2883_);
if (v_isSharedCheck_2894_ == 0)
{
lean_object* v_unused_2895_; 
v_unused_2895_ = lean_ctor_get(v___x_2883_, 0);
lean_dec(v_unused_2895_);
v___x_2888_ = v___x_2883_;
v_isShared_2889_ = v_isSharedCheck_2894_;
goto v_resetjp_2887_;
}
else
{
lean_dec(v___x_2883_);
v___x_2888_ = lean_box(0);
v_isShared_2889_ = v_isSharedCheck_2894_;
goto v_resetjp_2887_;
}
v_resetjp_2887_:
{
lean_object* v___x_2890_; lean_object* v___x_2892_; 
v___x_2890_ = lean_box(0);
if (v_isShared_2889_ == 0)
{
lean_ctor_set_tag(v___x_2888_, 0);
lean_ctor_set(v___x_2888_, 0, v___x_2890_);
v___x_2892_ = v___x_2888_;
goto v_reusejp_2891_;
}
else
{
lean_object* v_reuseFailAlloc_2893_; 
v_reuseFailAlloc_2893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2893_, 0, v___x_2890_);
v___x_2892_ = v_reuseFailAlloc_2893_;
goto v_reusejp_2891_;
}
v_reusejp_2891_:
{
return v___x_2892_;
}
}
}
else
{
return v___x_2883_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2902_; uint8_t v___y_2904_; lean_object* v___x_2914_; lean_object* v___x_2915_; lean_object* v___x_2916_; uint8_t v___x_2917_; 
v___x_2898_ = lean_unsigned_to_nat(0u);
v___x_2899_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2898_);
v___x_2900_ = lean_unsigned_to_nat(1u);
v___x_2901_ = l_Lean_Syntax_getArg(v_a_2482_, v___x_2900_);
lean_dec(v_a_2482_);
v___x_2902_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___closed__17));
v___x_2914_ = l_Lean_Syntax_getArgs(v___x_2901_);
lean_dec(v___x_2901_);
v___x_2915_ = lean_array_get_size(v___x_2914_);
v___x_2916_ = lean_nat_sub(v___x_2915_, v___x_2900_);
v___x_2917_ = lean_nat_dec_lt(v___x_2916_, v___x_2915_);
if (v___x_2917_ == 0)
{
lean_dec(v___x_2916_);
lean_dec_ref(v___x_2914_);
v___y_2904_ = v___x_2487_;
goto v___jp_2903_;
}
else
{
lean_object* v___x_2918_; uint8_t v___x_2919_; 
v___x_2918_ = lean_array_fget(v___x_2914_, v___x_2916_);
lean_dec(v___x_2916_);
lean_dec_ref(v___x_2914_);
v___x_2919_ = lp_mathlib_Mathlib_Tactic_Push_isUnderscore(v___x_2918_);
v___y_2904_ = v___x_2919_;
goto v___jp_2903_;
}
v___jp_2903_:
{
if (v___y_2904_ == 0)
{
lean_object* v___x_2905_; lean_object* v___x_2907_; 
lean_dec(v___x_2899_);
v___x_2905_ = lean_box(0);
if (v_isShared_2485_ == 0)
{
lean_ctor_set(v___x_2484_, 0, v___x_2905_);
v___x_2907_ = v___x_2484_;
goto v_reusejp_2906_;
}
else
{
lean_object* v_reuseFailAlloc_2908_; 
v_reuseFailAlloc_2908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2908_, 0, v___x_2905_);
v___x_2907_ = v_reuseFailAlloc_2908_;
goto v_reusejp_2906_;
}
v_reusejp_2906_:
{
return v___x_2907_;
}
}
else
{
uint8_t v___x_2909_; lean_object* v___x_2910_; 
lean_del_object(v___x_2484_);
v___x_2909_ = 0;
v___x_2910_ = l_Lean_Elab_Term_resolveId_x3f(v___x_2899_, v___x_2902_, v___x_2909_, v_a_2467_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_, v_a_2472_);
if (lean_obj_tag(v___x_2910_) == 0)
{
return v___x_2910_;
}
else
{
lean_object* v_a_2911_; uint8_t v___x_2912_; 
v_a_2911_ = lean_ctor_get(v___x_2910_, 0);
lean_inc(v_a_2911_);
v___x_2912_ = l_Lean_Exception_isInterrupt(v_a_2911_);
if (v___x_2912_ == 0)
{
uint8_t v___x_2913_; 
v___x_2913_ = l_Lean_Exception_isRuntime(v_a_2911_);
v___y_2475_ = v___x_2910_;
v___y_2476_ = v___x_2913_;
goto v___jp_2474_;
}
else
{
lean_dec(v_a_2911_);
v___y_2475_ = v___x_2910_;
v___y_2476_ = v___x_2912_;
goto v___jp_2474_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2921_; lean_object* v___x_2923_; uint8_t v_isShared_2924_; uint8_t v_isSharedCheck_2928_; 
v_a_2921_ = lean_ctor_get(v___x_2481_, 0);
v_isSharedCheck_2928_ = !lean_is_exclusive(v___x_2481_);
if (v_isSharedCheck_2928_ == 0)
{
v___x_2923_ = v___x_2481_;
v_isShared_2924_ = v_isSharedCheck_2928_;
goto v_resetjp_2922_;
}
else
{
lean_inc(v_a_2921_);
lean_dec(v___x_2481_);
v___x_2923_ = lean_box(0);
v_isShared_2924_ = v_isSharedCheck_2928_;
goto v_resetjp_2922_;
}
v_resetjp_2922_:
{
lean_object* v___x_2926_; 
if (v_isShared_2924_ == 0)
{
v___x_2926_ = v___x_2923_;
goto v_reusejp_2925_;
}
else
{
lean_object* v_reuseFailAlloc_2927_; 
v_reuseFailAlloc_2927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2927_, 0, v_a_2921_);
v___x_2926_ = v_reuseFailAlloc_2927_;
goto v_reusejp_2925_;
}
v_reusejp_2925_:
{
return v___x_2926_;
}
}
}
v___jp_2474_:
{
if (v___y_2476_ == 0)
{
lean_object* v___x_2477_; lean_object* v___x_2478_; 
lean_dec_ref(v___y_2475_);
v___x_2477_ = lean_box(0);
v___x_2478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2478_, 0, v___x_2477_);
return v___x_2478_;
}
else
{
return v___y_2475_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f___boxed(lean_object* v_stx_2929_, lean_object* v_a_2930_, lean_object* v_a_2931_, lean_object* v_a_2932_, lean_object* v_a_2933_, lean_object* v_a_2934_, lean_object* v_a_2935_, lean_object* v_a_2936_){
_start:
{
lean_object* v_res_2937_; 
v_res_2937_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_2929_, v_a_2930_, v_a_2931_, v_a_2932_, v_a_2933_, v_a_2934_, v_a_2935_);
lean_dec(v_a_2935_);
lean_dec_ref(v_a_2934_);
lean_dec(v_a_2933_);
lean_dec_ref(v_a_2932_);
lean_dec(v_a_2931_);
lean_dec_ref(v_a_2930_);
return v_res_2937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1(lean_object* v_00_u03b1_2938_, lean_object* v_x_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_){
_start:
{
lean_object* v___x_2942_; 
v___x_2942_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___redArg(v_x_2939_, v___y_2941_);
return v___x_2942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1___boxed(lean_object* v_00_u03b1_2943_, lean_object* v_x_2944_, lean_object* v___y_2945_, lean_object* v___y_2946_){
_start:
{
lean_object* v_res_2947_; 
v_res_2947_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__1(v_00_u03b1_2943_, v_x_2944_, v___y_2945_, v___y_2946_);
lean_dec_ref(v___y_2945_);
lean_dec_ref(v_x_2944_);
return v_res_2947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6(lean_object* v_00_u03b1_2948_, lean_object* v_ref_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_, lean_object* v___y_2955_){
_start:
{
lean_object* v___x_2957_; 
v___x_2957_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___redArg(v_ref_2949_);
return v___x_2957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6___boxed(lean_object* v_00_u03b1_2958_, lean_object* v_ref_2959_, lean_object* v___y_2960_, lean_object* v___y_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_){
_start:
{
lean_object* v_res_2967_; 
v_res_2967_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__6(v_00_u03b1_2958_, v_ref_2959_, v___y_2960_, v___y_2961_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_);
lean_dec(v___y_2965_);
lean_dec_ref(v___y_2964_);
lean_dec(v___y_2963_);
lean_dec_ref(v___y_2962_);
lean_dec(v___y_2961_);
lean_dec_ref(v___y_2960_);
return v_res_2967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7(lean_object* v_00_u03b1_2968_, lean_object* v___y_2969_, lean_object* v___y_2970_, lean_object* v___y_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_){
_start:
{
lean_object* v___x_2976_; 
v___x_2976_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg();
return v___x_2976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___boxed(lean_object* v_00_u03b1_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_){
_start:
{
lean_object* v_res_2985_; 
v_res_2985_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7(v_00_u03b1_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_, v___y_2983_);
lean_dec(v___y_2983_);
lean_dec_ref(v___y_2982_);
lean_dec(v___y_2981_);
lean_dec_ref(v___y_2980_);
lean_dec(v___y_2979_);
lean_dec_ref(v___y_2978_);
return v_res_2985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0(lean_object* v_00_u03b1_2986_, lean_object* v_x_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_){
_start:
{
lean_object* v___x_2995_; 
v___x_2995_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___redArg(v_x_2987_, v___y_2988_, v___y_2989_, v___y_2990_, v___y_2991_, v___y_2992_, v___y_2993_);
return v___x_2995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0___boxed(lean_object* v_00_u03b1_2996_, lean_object* v_x_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_){
_start:
{
lean_object* v_res_3005_; 
v_res_3005_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0(v_00_u03b1_2996_, v_x_2997_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_, v___y_3002_, v___y_3003_);
lean_dec(v___y_3003_);
lean_dec_ref(v___y_3002_);
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
return v_res_3005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0(lean_object* v_cls_3006_, lean_object* v_msg_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_){
_start:
{
lean_object* v___x_3015_; 
v___x_3015_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg(v_cls_3006_, v_msg_3007_, v___y_3010_, v___y_3011_, v___y_3012_, v___y_3013_);
return v___x_3015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___boxed(lean_object* v_cls_3016_, lean_object* v_msg_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_){
_start:
{
lean_object* v_res_3025_; 
v_res_3025_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0(v_cls_3016_, v_msg_3017_, v___y_3018_, v___y_3019_, v___y_3020_, v___y_3021_, v___y_3022_, v___y_3023_);
lean_dec(v___y_3023_);
lean_dec_ref(v___y_3022_);
lean_dec(v___y_3021_);
lean_dec_ref(v___y_3020_);
lean_dec(v___y_3019_);
lean_dec_ref(v___y_3018_);
return v_res_3025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3(lean_object* v_as_3026_, lean_object* v_as_x27_3027_, lean_object* v_b_3028_, lean_object* v_a_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_){
_start:
{
lean_object* v___x_3037_; 
v___x_3037_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___redArg(v_as_x27_3027_, v_b_3028_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
return v___x_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3___boxed(lean_object* v_as_3038_, lean_object* v_as_x27_3039_, lean_object* v_b_3040_, lean_object* v_a_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_, lean_object* v___y_3045_, lean_object* v___y_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_){
_start:
{
lean_object* v_res_3049_; 
v_res_3049_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__3(v_as_3038_, v_as_x27_3039_, v_b_3040_, v_a_3041_, v___y_3042_, v___y_3043_, v___y_3044_, v___y_3045_, v___y_3046_, v___y_3047_);
lean_dec(v___y_3047_);
lean_dec_ref(v___y_3046_);
lean_dec(v___y_3045_);
lean_dec_ref(v___y_3044_);
lean_dec(v___y_3043_);
lean_dec_ref(v___y_3042_);
lean_dec(v_as_x27_3039_);
lean_dec(v_as_3038_);
return v_res_3049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5(lean_object* v_00_u03b1_3050_, lean_object* v_ref_3051_, lean_object* v_msg_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_, lean_object* v___y_3058_){
_start:
{
lean_object* v___x_3060_; 
v___x_3060_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___redArg(v_ref_3051_, v_msg_3052_, v___y_3053_, v___y_3054_, v___y_3055_, v___y_3056_, v___y_3057_, v___y_3058_);
return v___x_3060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5___boxed(lean_object* v_00_u03b1_3061_, lean_object* v_ref_3062_, lean_object* v_msg_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_){
_start:
{
lean_object* v_res_3071_; 
v_res_3071_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__5(v_00_u03b1_3061_, v_ref_3062_, v_msg_3063_, v___y_3064_, v___y_3065_, v___y_3066_, v___y_3067_, v___y_3068_, v___y_3069_);
lean_dec(v___y_3069_);
lean_dec_ref(v___y_3068_);
lean_dec(v___y_3067_);
lean_dec_ref(v___y_3066_);
lean_dec(v___y_3065_);
lean_dec_ref(v___y_3064_);
lean_dec(v_ref_3062_);
return v_res_3071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5(lean_object* v_00_u03b2_3072_, lean_object* v_m_3073_, lean_object* v_a_3074_){
_start:
{
lean_object* v___x_3075_; 
v___x_3075_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___redArg(v_m_3073_, v_a_3074_);
return v___x_3075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5___boxed(lean_object* v_00_u03b2_3076_, lean_object* v_m_3077_, lean_object* v_a_3078_){
_start:
{
lean_object* v_res_3079_; 
v_res_3079_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5(v_00_u03b2_3076_, v_m_3077_, v_a_3078_);
lean_dec(v_a_3078_);
lean_dec_ref(v_m_3077_);
return v_res_3079_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6(lean_object* v_00_u03b2_3080_, lean_object* v_x_3081_, lean_object* v_x_3082_){
_start:
{
uint8_t v___x_3083_; 
v___x_3083_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___redArg(v_x_3081_, v_x_3082_);
return v___x_3083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6___boxed(lean_object* v_00_u03b2_3084_, lean_object* v_x_3085_, lean_object* v_x_3086_){
_start:
{
uint8_t v_res_3087_; lean_object* v_r_3088_; 
v_res_3087_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6(v_00_u03b2_3084_, v_x_3085_, v_x_3086_);
lean_dec_ref(v_x_3086_);
lean_dec_ref(v_x_3085_);
v_r_3088_ = lean_box(v_res_3087_);
return v_r_3088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9(lean_object* v_00_u03b2_3089_, lean_object* v_a_3090_, lean_object* v_x_3091_){
_start:
{
lean_object* v___x_3092_; 
v___x_3092_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___redArg(v_a_3090_, v_x_3091_);
return v___x_3092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9___boxed(lean_object* v_00_u03b2_3093_, lean_object* v_a_3094_, lean_object* v_x_3095_){
_start:
{
lean_object* v_res_3096_; 
v_res_3096_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__5_spec__9(v_00_u03b2_3093_, v_a_3094_, v_x_3095_);
lean_dec(v_x_3095_);
lean_dec(v_a_3094_);
return v_res_3096_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10(lean_object* v_00_u03b2_3097_, lean_object* v_x_3098_, size_t v_x_3099_, lean_object* v_x_3100_){
_start:
{
uint8_t v___x_3101_; 
v___x_3101_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___redArg(v_x_3098_, v_x_3099_, v_x_3100_);
return v___x_3101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10___boxed(lean_object* v_00_u03b2_3102_, lean_object* v_x_3103_, lean_object* v_x_3104_, lean_object* v_x_3105_){
_start:
{
size_t v_x_54107__boxed_3106_; uint8_t v_res_3107_; lean_object* v_r_3108_; 
v_x_54107__boxed_3106_ = lean_unbox_usize(v_x_3104_);
lean_dec(v_x_3104_);
v_res_3107_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10(v_00_u03b2_3102_, v_x_3103_, v_x_54107__boxed_3106_, v_x_3105_);
lean_dec_ref(v_x_3105_);
lean_dec_ref(v_x_3103_);
v_r_3108_ = lean_box(v_res_3107_);
return v_r_3108_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13(lean_object* v_00_u03b2_3109_, lean_object* v_keys_3110_, lean_object* v_vals_3111_, lean_object* v_heq_3112_, lean_object* v_i_3113_, lean_object* v_k_3114_){
_start:
{
uint8_t v___x_3115_; 
v___x_3115_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___redArg(v_keys_3110_, v_i_3113_, v_k_3114_);
return v___x_3115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13___boxed(lean_object* v_00_u03b2_3116_, lean_object* v_keys_3117_, lean_object* v_vals_3118_, lean_object* v_heq_3119_, lean_object* v_i_3120_, lean_object* v_k_3121_){
_start:
{
uint8_t v_res_3122_; lean_object* v_r_3123_; 
v_res_3122_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__2_spec__3_spec__6_spec__10_spec__13(v_00_u03b2_3116_, v_keys_3117_, v_vals_3118_, v_heq_3119_, v_i_3120_, v_k_3121_);
lean_dec_ref(v_k_3121_);
lean_dec_ref(v_vals_3118_);
lean_dec_ref(v_keys_3117_);
v_r_3123_ = lean_box(v_res_3122_);
return v_r_3123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___redArg(lean_object* v_a_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_, lean_object* v___y_3129_, lean_object* v___y_3130_){
_start:
{
lean_object* v___x_3132_; 
v___x_3132_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_3124_, v___y_3125_, v___y_3126_, v___y_3127_, v___y_3128_, v___y_3129_, v___y_3130_);
return v___x_3132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___redArg___boxed(lean_object* v_a_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_, lean_object* v___y_3138_, lean_object* v___y_3139_, lean_object* v___y_3140_){
_start:
{
lean_object* v_res_3141_; 
v_res_3141_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___redArg(v_a_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_, v___y_3138_, v___y_3139_);
lean_dec(v___y_3139_);
lean_dec_ref(v___y_3138_);
lean_dec(v___y_3137_);
lean_dec_ref(v___y_3136_);
lean_dec(v___y_3135_);
lean_dec_ref(v___y_3134_);
return v_res_3141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0(lean_object* v_00_u03b1_3142_, lean_object* v_a_3143_, lean_object* v___y_3144_, lean_object* v___y_3145_, lean_object* v___y_3146_, lean_object* v___y_3147_, lean_object* v___y_3148_, lean_object* v___y_3149_){
_start:
{
lean_object* v___x_3151_; 
v___x_3151_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_3143_, v___y_3144_, v___y_3145_, v___y_3146_, v___y_3147_, v___y_3148_, v___y_3149_);
return v___x_3151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___boxed(lean_object* v_00_u03b1_3152_, lean_object* v_a_3153_, lean_object* v___y_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_){
_start:
{
lean_object* v_res_3161_; 
v_res_3161_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0(v_00_u03b1_3152_, v_a_3153_, v___y_3154_, v___y_3155_, v___y_3156_, v___y_3157_, v___y_3158_, v___y_3159_);
lean_dec(v___y_3159_);
lean_dec_ref(v___y_3158_);
lean_dec(v___y_3157_);
lean_dec_ref(v___y_3156_);
lean_dec(v___y_3155_);
lean_dec_ref(v___y_3154_);
return v_res_3161_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3(void){
_start:
{
lean_object* v___x_3169_; lean_object* v___x_3170_; 
v___x_3169_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__2));
v___x_3170_ = l_Lean_stringToMessageData(v___x_3169_);
return v___x_3170_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5(void){
_start:
{
lean_object* v___x_3172_; lean_object* v___x_3173_; 
v___x_3172_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__4));
v___x_3173_ = l_Lean_stringToMessageData(v___x_3172_);
return v___x_3173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead(lean_object* v_stx_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_, lean_object* v_a_3178_, lean_object* v_a_3179_, lean_object* v_a_3180_){
_start:
{
lean_object* v___x_3182_; uint8_t v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v_fileName_3187_; lean_object* v_fileMap_3188_; lean_object* v_options_3189_; lean_object* v_currRecDepth_3190_; lean_object* v_maxRecDepth_3191_; lean_object* v_ref_3192_; lean_object* v_currNamespace_3193_; lean_object* v_openDecls_3194_; lean_object* v_initHeartbeats_3195_; lean_object* v_maxHeartbeats_3196_; lean_object* v_quotContext_3197_; lean_object* v_currMacroScope_3198_; uint8_t v_diag_3199_; lean_object* v_cancelTk_x3f_3200_; uint8_t v_suppressElabErrors_3201_; lean_object* v_inheritedTraceOptions_3202_; lean_object* v_declName_x3f_3203_; lean_object* v_macroStack_3204_; uint8_t v_mayPostpone_3205_; uint8_t v_errToSorry_3206_; lean_object* v_autoBoundImplicitContext_3207_; lean_object* v_autoBoundImplicitForbidden_3208_; lean_object* v_sectionVars_3209_; lean_object* v_sectionFVars_3210_; uint8_t v_implicitLambda_3211_; uint8_t v_heedElabAsElim_3212_; uint8_t v_isNoncomputableSection_3213_; uint8_t v_isMetaSection_3214_; uint8_t v_inPattern_3215_; lean_object* v_tacSnap_x3f_3216_; uint8_t v_saveRecAppSyntax_3217_; uint8_t v_holesAsSyntheticOpaque_3218_; uint8_t v_checkDeprecated_3219_; lean_object* v_fixedTermElabs_3220_; lean_object* v___x_3221_; lean_object* v_ref_3222_; lean_object* v___x_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; 
v___x_3182_ = lean_box(0);
v___x_3183_ = 1;
v___x_3184_ = lean_box(v___x_3183_);
v___x_3185_ = lean_box(v___x_3183_);
lean_inc(v_stx_3174_);
v___x_3186_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_3186_, 0, v_stx_3174_);
lean_closure_set(v___x_3186_, 1, v___x_3182_);
lean_closure_set(v___x_3186_, 2, v___x_3184_);
lean_closure_set(v___x_3186_, 3, v___x_3185_);
v_fileName_3187_ = lean_ctor_get(v_a_3179_, 0);
v_fileMap_3188_ = lean_ctor_get(v_a_3179_, 1);
v_options_3189_ = lean_ctor_get(v_a_3179_, 2);
v_currRecDepth_3190_ = lean_ctor_get(v_a_3179_, 3);
v_maxRecDepth_3191_ = lean_ctor_get(v_a_3179_, 4);
v_ref_3192_ = lean_ctor_get(v_a_3179_, 5);
v_currNamespace_3193_ = lean_ctor_get(v_a_3179_, 6);
v_openDecls_3194_ = lean_ctor_get(v_a_3179_, 7);
v_initHeartbeats_3195_ = lean_ctor_get(v_a_3179_, 8);
v_maxHeartbeats_3196_ = lean_ctor_get(v_a_3179_, 9);
v_quotContext_3197_ = lean_ctor_get(v_a_3179_, 10);
v_currMacroScope_3198_ = lean_ctor_get(v_a_3179_, 11);
v_diag_3199_ = lean_ctor_get_uint8(v_a_3179_, sizeof(void*)*14);
v_cancelTk_x3f_3200_ = lean_ctor_get(v_a_3179_, 12);
v_suppressElabErrors_3201_ = lean_ctor_get_uint8(v_a_3179_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3202_ = lean_ctor_get(v_a_3179_, 13);
v_declName_x3f_3203_ = lean_ctor_get(v_a_3175_, 0);
v_macroStack_3204_ = lean_ctor_get(v_a_3175_, 1);
v_mayPostpone_3205_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8);
v_errToSorry_3206_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 1);
v_autoBoundImplicitContext_3207_ = lean_ctor_get(v_a_3175_, 2);
v_autoBoundImplicitForbidden_3208_ = lean_ctor_get(v_a_3175_, 3);
v_sectionVars_3209_ = lean_ctor_get(v_a_3175_, 4);
v_sectionFVars_3210_ = lean_ctor_get(v_a_3175_, 5);
v_implicitLambda_3211_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 2);
v_heedElabAsElim_3212_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_3213_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 4);
v_isMetaSection_3214_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 5);
v_inPattern_3215_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_3216_ = lean_ctor_get(v_a_3175_, 6);
v_saveRecAppSyntax_3217_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_3218_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 9);
v_checkDeprecated_3219_ = lean_ctor_get_uint8(v_a_3175_, sizeof(void*)*8 + 10);
v_fixedTermElabs_3220_ = lean_ctor_get(v_a_3175_, 7);
v___x_3221_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Push_elabHead_spec__0___boxed), 9, 2);
lean_closure_set(v___x_3221_, 0, lean_box(0));
lean_closure_set(v___x_3221_, 1, v___x_3186_);
v_ref_3222_ = l_Lean_replaceRef(v_stx_3174_, v_ref_3192_);
lean_inc_ref(v_inheritedTraceOptions_3202_);
lean_inc(v_cancelTk_x3f_3200_);
lean_inc(v_currMacroScope_3198_);
lean_inc(v_quotContext_3197_);
lean_inc(v_maxHeartbeats_3196_);
lean_inc(v_initHeartbeats_3195_);
lean_inc(v_openDecls_3194_);
lean_inc(v_currNamespace_3193_);
lean_inc(v_maxRecDepth_3191_);
lean_inc(v_currRecDepth_3190_);
lean_inc_ref(v_options_3189_);
lean_inc_ref(v_fileMap_3188_);
lean_inc_ref(v_fileName_3187_);
v___x_3223_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3223_, 0, v_fileName_3187_);
lean_ctor_set(v___x_3223_, 1, v_fileMap_3188_);
lean_ctor_set(v___x_3223_, 2, v_options_3189_);
lean_ctor_set(v___x_3223_, 3, v_currRecDepth_3190_);
lean_ctor_set(v___x_3223_, 4, v_maxRecDepth_3191_);
lean_ctor_set(v___x_3223_, 5, v_ref_3222_);
lean_ctor_set(v___x_3223_, 6, v_currNamespace_3193_);
lean_ctor_set(v___x_3223_, 7, v_openDecls_3194_);
lean_ctor_set(v___x_3223_, 8, v_initHeartbeats_3195_);
lean_ctor_set(v___x_3223_, 9, v_maxHeartbeats_3196_);
lean_ctor_set(v___x_3223_, 10, v_quotContext_3197_);
lean_ctor_set(v___x_3223_, 11, v_currMacroScope_3198_);
lean_ctor_set(v___x_3223_, 12, v_cancelTk_x3f_3200_);
lean_ctor_set(v___x_3223_, 13, v_inheritedTraceOptions_3202_);
lean_ctor_set_uint8(v___x_3223_, sizeof(void*)*14, v_diag_3199_);
lean_ctor_set_uint8(v___x_3223_, sizeof(void*)*14 + 1, v_suppressElabErrors_3201_);
lean_inc_ref(v_fixedTermElabs_3220_);
lean_inc(v_tacSnap_x3f_3216_);
lean_inc(v_sectionFVars_3210_);
lean_inc(v_sectionVars_3209_);
lean_inc_ref(v_autoBoundImplicitForbidden_3208_);
lean_inc(v_autoBoundImplicitContext_3207_);
lean_inc(v_macroStack_3204_);
lean_inc(v_declName_x3f_3203_);
v___x_3224_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_3224_, 0, v_declName_x3f_3203_);
lean_ctor_set(v___x_3224_, 1, v_macroStack_3204_);
lean_ctor_set(v___x_3224_, 2, v_autoBoundImplicitContext_3207_);
lean_ctor_set(v___x_3224_, 3, v_autoBoundImplicitForbidden_3208_);
lean_ctor_set(v___x_3224_, 4, v_sectionVars_3209_);
lean_ctor_set(v___x_3224_, 5, v_sectionFVars_3210_);
lean_ctor_set(v___x_3224_, 6, v_tacSnap_x3f_3216_);
lean_ctor_set(v___x_3224_, 7, v_fixedTermElabs_3220_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8, v_mayPostpone_3205_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 1, v_errToSorry_3206_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 2, v_implicitLambda_3211_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 3, v_heedElabAsElim_3212_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 4, v_isNoncomputableSection_3213_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 5, v_isMetaSection_3214_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 6, v___x_3183_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 7, v_inPattern_3215_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_3217_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_3218_);
lean_ctor_set_uint8(v___x_3224_, sizeof(void*)*8 + 10, v_checkDeprecated_3219_);
v___x_3225_ = l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___redArg(v___x_3221_, v___x_3224_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
lean_dec_ref_known(v___x_3224_, 8);
if (lean_obj_tag(v___x_3225_) == 0)
{
lean_object* v___x_3227_; uint8_t v_isShared_3228_; uint8_t v_isSharedCheck_3577_; 
v_isSharedCheck_3577_ = !lean_is_exclusive(v___x_3225_);
if (v_isSharedCheck_3577_ == 0)
{
lean_object* v_unused_3578_; 
v_unused_3578_ = lean_ctor_get(v___x_3225_, 0);
lean_dec(v_unused_3578_);
v___x_3227_ = v___x_3225_;
v_isShared_3228_ = v_isSharedCheck_3577_;
goto v_resetjp_3226_;
}
else
{
lean_dec(v___x_3225_);
v___x_3227_ = lean_box(0);
v_isShared_3228_ = v_isSharedCheck_3577_;
goto v_resetjp_3226_;
}
v_resetjp_3226_:
{
lean_object* v___x_3229_; uint8_t v___x_3230_; 
v___x_3229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__6));
lean_inc(v_stx_3174_);
v___x_3230_ = l_Lean_Syntax_isOfKind(v_stx_3174_, v___x_3229_);
if (v___x_3230_ == 0)
{
lean_object* v___x_3231_; uint8_t v___x_3232_; 
v___x_3231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__1));
lean_inc(v_stx_3174_);
v___x_3232_ = l_Lean_Syntax_isOfKind(v_stx_3174_, v___x_3231_);
if (v___x_3232_ == 0)
{
lean_object* v___x_3233_; 
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3233_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3233_) == 0)
{
lean_object* v_a_3234_; lean_object* v___x_3236_; uint8_t v_isShared_3237_; uint8_t v_isSharedCheck_3263_; 
v_a_3234_ = lean_ctor_get(v___x_3233_, 0);
v_isSharedCheck_3263_ = !lean_is_exclusive(v___x_3233_);
if (v_isSharedCheck_3263_ == 0)
{
v___x_3236_ = v___x_3233_;
v_isShared_3237_ = v_isSharedCheck_3263_;
goto v_resetjp_3235_;
}
else
{
lean_inc(v_a_3234_);
lean_dec(v___x_3233_);
v___x_3236_ = lean_box(0);
v_isShared_3237_ = v_isSharedCheck_3263_;
goto v_resetjp_3235_;
}
v_resetjp_3235_:
{
lean_object* v___y_3239_; lean_object* v___y_3240_; lean_object* v___y_3241_; lean_object* v___y_3242_; lean_object* v___y_3243_; lean_object* v___y_3244_; 
if (lean_obj_tag(v_a_3234_) == 1)
{
lean_object* v_val_3251_; lean_object* v___x_3253_; uint8_t v_isShared_3254_; uint8_t v_isSharedCheck_3262_; 
v_val_3251_ = lean_ctor_get(v_a_3234_, 0);
v_isSharedCheck_3262_ = !lean_is_exclusive(v_a_3234_);
if (v_isSharedCheck_3262_ == 0)
{
v___x_3253_ = v_a_3234_;
v_isShared_3254_ = v_isSharedCheck_3262_;
goto v_resetjp_3252_;
}
else
{
lean_inc(v_val_3251_);
lean_dec(v_a_3234_);
v___x_3253_ = lean_box(0);
v_isShared_3254_ = v_isSharedCheck_3262_;
goto v_resetjp_3252_;
}
v_resetjp_3252_:
{
if (lean_obj_tag(v_val_3251_) == 4)
{
lean_object* v_declName_3255_; lean_object* v___x_3257_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3255_ = lean_ctor_get(v_val_3251_, 0);
lean_inc(v_declName_3255_);
lean_dec_ref_known(v_val_3251_, 2);
if (v_isShared_3254_ == 0)
{
lean_ctor_set_tag(v___x_3253_, 0);
lean_ctor_set(v___x_3253_, 0, v_declName_3255_);
v___x_3257_ = v___x_3253_;
goto v_reusejp_3256_;
}
else
{
lean_object* v_reuseFailAlloc_3261_; 
v_reuseFailAlloc_3261_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3261_, 0, v_declName_3255_);
v___x_3257_ = v_reuseFailAlloc_3261_;
goto v_reusejp_3256_;
}
v_reusejp_3256_:
{
lean_object* v___x_3259_; 
if (v_isShared_3237_ == 0)
{
lean_ctor_set(v___x_3236_, 0, v___x_3257_);
v___x_3259_ = v___x_3236_;
goto v_reusejp_3258_;
}
else
{
lean_object* v_reuseFailAlloc_3260_; 
v_reuseFailAlloc_3260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3260_, 0, v___x_3257_);
v___x_3259_ = v_reuseFailAlloc_3260_;
goto v_reusejp_3258_;
}
v_reusejp_3258_:
{
return v___x_3259_;
}
}
}
else
{
lean_del_object(v___x_3253_);
lean_dec(v_val_3251_);
lean_del_object(v___x_3236_);
v___y_3239_ = v_a_3175_;
v___y_3240_ = v_a_3176_;
v___y_3241_ = v_a_3177_;
v___y_3242_ = v_a_3178_;
v___y_3243_ = v___x_3223_;
v___y_3244_ = v_a_3180_;
goto v___jp_3238_;
}
}
}
else
{
lean_del_object(v___x_3236_);
lean_dec(v_a_3234_);
v___y_3239_ = v_a_3175_;
v___y_3240_ = v_a_3176_;
v___y_3241_ = v_a_3177_;
v___y_3242_ = v_a_3178_;
v___y_3243_ = v___x_3223_;
v___y_3244_ = v_a_3180_;
goto v___jp_3238_;
}
v___jp_3238_:
{
lean_object* v___x_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; lean_object* v___x_3248_; lean_object* v___x_3249_; lean_object* v___x_3250_; 
v___x_3245_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3246_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3247_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3247_, 0, v___x_3245_);
lean_ctor_set(v___x_3247_, 1, v___x_3246_);
v___x_3248_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3249_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3249_, 0, v___x_3247_);
lean_ctor_set(v___x_3249_, 1, v___x_3248_);
v___x_3250_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3249_, v___y_3239_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_, v___y_3244_);
lean_dec_ref(v___y_3243_);
return v___x_3250_;
}
}
}
else
{
lean_object* v_a_3264_; lean_object* v___x_3266_; uint8_t v_isShared_3267_; uint8_t v_isSharedCheck_3271_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3264_ = lean_ctor_get(v___x_3233_, 0);
v_isSharedCheck_3271_ = !lean_is_exclusive(v___x_3233_);
if (v_isSharedCheck_3271_ == 0)
{
v___x_3266_ = v___x_3233_;
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
else
{
lean_inc(v_a_3264_);
lean_dec(v___x_3233_);
v___x_3266_ = lean_box(0);
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
v_resetjp_3265_:
{
lean_object* v___x_3269_; 
if (v_isShared_3267_ == 0)
{
v___x_3269_ = v___x_3266_;
goto v_reusejp_3268_;
}
else
{
lean_object* v_reuseFailAlloc_3270_; 
v_reuseFailAlloc_3270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3270_, 0, v_a_3264_);
v___x_3269_ = v_reuseFailAlloc_3270_;
goto v_reusejp_3268_;
}
v_reusejp_3268_:
{
return v___x_3269_;
}
}
}
}
else
{
lean_object* v___x_3272_; lean_object* v___x_3273_; uint8_t v___x_3274_; 
v___x_3272_ = lean_unsigned_to_nat(1u);
v___x_3273_ = l_Lean_Syntax_getArg(v_stx_3174_, v___x_3272_);
v___x_3274_ = l_Lean_Syntax_matchesNull(v___x_3273_, v___x_3272_);
if (v___x_3274_ == 0)
{
lean_object* v___x_3275_; 
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3275_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3275_) == 0)
{
lean_object* v_a_3276_; lean_object* v___x_3278_; uint8_t v_isShared_3279_; uint8_t v_isSharedCheck_3305_; 
v_a_3276_ = lean_ctor_get(v___x_3275_, 0);
v_isSharedCheck_3305_ = !lean_is_exclusive(v___x_3275_);
if (v_isSharedCheck_3305_ == 0)
{
v___x_3278_ = v___x_3275_;
v_isShared_3279_ = v_isSharedCheck_3305_;
goto v_resetjp_3277_;
}
else
{
lean_inc(v_a_3276_);
lean_dec(v___x_3275_);
v___x_3278_ = lean_box(0);
v_isShared_3279_ = v_isSharedCheck_3305_;
goto v_resetjp_3277_;
}
v_resetjp_3277_:
{
lean_object* v___y_3281_; lean_object* v___y_3282_; lean_object* v___y_3283_; lean_object* v___y_3284_; lean_object* v___y_3285_; lean_object* v___y_3286_; 
if (lean_obj_tag(v_a_3276_) == 1)
{
lean_object* v_val_3293_; lean_object* v___x_3295_; uint8_t v_isShared_3296_; uint8_t v_isSharedCheck_3304_; 
v_val_3293_ = lean_ctor_get(v_a_3276_, 0);
v_isSharedCheck_3304_ = !lean_is_exclusive(v_a_3276_);
if (v_isSharedCheck_3304_ == 0)
{
v___x_3295_ = v_a_3276_;
v_isShared_3296_ = v_isSharedCheck_3304_;
goto v_resetjp_3294_;
}
else
{
lean_inc(v_val_3293_);
lean_dec(v_a_3276_);
v___x_3295_ = lean_box(0);
v_isShared_3296_ = v_isSharedCheck_3304_;
goto v_resetjp_3294_;
}
v_resetjp_3294_:
{
if (lean_obj_tag(v_val_3293_) == 4)
{
lean_object* v_declName_3297_; lean_object* v___x_3299_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3297_ = lean_ctor_get(v_val_3293_, 0);
lean_inc(v_declName_3297_);
lean_dec_ref_known(v_val_3293_, 2);
if (v_isShared_3296_ == 0)
{
lean_ctor_set_tag(v___x_3295_, 0);
lean_ctor_set(v___x_3295_, 0, v_declName_3297_);
v___x_3299_ = v___x_3295_;
goto v_reusejp_3298_;
}
else
{
lean_object* v_reuseFailAlloc_3303_; 
v_reuseFailAlloc_3303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3303_, 0, v_declName_3297_);
v___x_3299_ = v_reuseFailAlloc_3303_;
goto v_reusejp_3298_;
}
v_reusejp_3298_:
{
lean_object* v___x_3301_; 
if (v_isShared_3279_ == 0)
{
lean_ctor_set(v___x_3278_, 0, v___x_3299_);
v___x_3301_ = v___x_3278_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3302_; 
v_reuseFailAlloc_3302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3302_, 0, v___x_3299_);
v___x_3301_ = v_reuseFailAlloc_3302_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
return v___x_3301_;
}
}
}
else
{
lean_del_object(v___x_3295_);
lean_dec(v_val_3293_);
lean_del_object(v___x_3278_);
v___y_3281_ = v_a_3175_;
v___y_3282_ = v_a_3176_;
v___y_3283_ = v_a_3177_;
v___y_3284_ = v_a_3178_;
v___y_3285_ = v___x_3223_;
v___y_3286_ = v_a_3180_;
goto v___jp_3280_;
}
}
}
else
{
lean_del_object(v___x_3278_);
lean_dec(v_a_3276_);
v___y_3281_ = v_a_3175_;
v___y_3282_ = v_a_3176_;
v___y_3283_ = v_a_3177_;
v___y_3284_ = v_a_3178_;
v___y_3285_ = v___x_3223_;
v___y_3286_ = v_a_3180_;
goto v___jp_3280_;
}
v___jp_3280_:
{
lean_object* v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; lean_object* v___x_3290_; lean_object* v___x_3291_; lean_object* v___x_3292_; 
v___x_3287_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3288_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3289_, 0, v___x_3287_);
lean_ctor_set(v___x_3289_, 1, v___x_3288_);
v___x_3290_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3291_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3291_, 0, v___x_3289_);
lean_ctor_set(v___x_3291_, 1, v___x_3290_);
v___x_3292_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3291_, v___y_3281_, v___y_3282_, v___y_3283_, v___y_3284_, v___y_3285_, v___y_3286_);
lean_dec_ref(v___y_3285_);
return v___x_3292_;
}
}
}
else
{
lean_object* v_a_3306_; lean_object* v___x_3308_; uint8_t v_isShared_3309_; uint8_t v_isSharedCheck_3313_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3306_ = lean_ctor_get(v___x_3275_, 0);
v_isSharedCheck_3313_ = !lean_is_exclusive(v___x_3275_);
if (v_isSharedCheck_3313_ == 0)
{
v___x_3308_ = v___x_3275_;
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
else
{
lean_inc(v_a_3306_);
lean_dec(v___x_3275_);
v___x_3308_ = lean_box(0);
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
v_resetjp_3307_:
{
lean_object* v___x_3311_; 
if (v_isShared_3309_ == 0)
{
v___x_3311_ = v___x_3308_;
goto v_reusejp_3310_;
}
else
{
lean_object* v_reuseFailAlloc_3312_; 
v_reuseFailAlloc_3312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3312_, 0, v_a_3306_);
v___x_3311_ = v_reuseFailAlloc_3312_;
goto v_reusejp_3310_;
}
v_reusejp_3310_:
{
return v___x_3311_;
}
}
}
}
else
{
lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; uint8_t v___x_3317_; 
v___x_3314_ = lean_unsigned_to_nat(0u);
v___x_3315_ = lean_unsigned_to_nat(2u);
v___x_3316_ = l_Lean_Syntax_getArg(v_stx_3174_, v___x_3315_);
v___x_3317_ = l_Lean_Syntax_matchesNull(v___x_3316_, v___x_3314_);
if (v___x_3317_ == 0)
{
lean_object* v___x_3318_; 
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3318_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3318_) == 0)
{
lean_object* v_a_3319_; lean_object* v___x_3321_; uint8_t v_isShared_3322_; uint8_t v_isSharedCheck_3348_; 
v_a_3319_ = lean_ctor_get(v___x_3318_, 0);
v_isSharedCheck_3348_ = !lean_is_exclusive(v___x_3318_);
if (v_isSharedCheck_3348_ == 0)
{
v___x_3321_ = v___x_3318_;
v_isShared_3322_ = v_isSharedCheck_3348_;
goto v_resetjp_3320_;
}
else
{
lean_inc(v_a_3319_);
lean_dec(v___x_3318_);
v___x_3321_ = lean_box(0);
v_isShared_3322_ = v_isSharedCheck_3348_;
goto v_resetjp_3320_;
}
v_resetjp_3320_:
{
lean_object* v___y_3324_; lean_object* v___y_3325_; lean_object* v___y_3326_; lean_object* v___y_3327_; lean_object* v___y_3328_; lean_object* v___y_3329_; 
if (lean_obj_tag(v_a_3319_) == 1)
{
lean_object* v_val_3336_; lean_object* v___x_3338_; uint8_t v_isShared_3339_; uint8_t v_isSharedCheck_3347_; 
v_val_3336_ = lean_ctor_get(v_a_3319_, 0);
v_isSharedCheck_3347_ = !lean_is_exclusive(v_a_3319_);
if (v_isSharedCheck_3347_ == 0)
{
v___x_3338_ = v_a_3319_;
v_isShared_3339_ = v_isSharedCheck_3347_;
goto v_resetjp_3337_;
}
else
{
lean_inc(v_val_3336_);
lean_dec(v_a_3319_);
v___x_3338_ = lean_box(0);
v_isShared_3339_ = v_isSharedCheck_3347_;
goto v_resetjp_3337_;
}
v_resetjp_3337_:
{
if (lean_obj_tag(v_val_3336_) == 4)
{
lean_object* v_declName_3340_; lean_object* v___x_3342_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3340_ = lean_ctor_get(v_val_3336_, 0);
lean_inc(v_declName_3340_);
lean_dec_ref_known(v_val_3336_, 2);
if (v_isShared_3339_ == 0)
{
lean_ctor_set_tag(v___x_3338_, 0);
lean_ctor_set(v___x_3338_, 0, v_declName_3340_);
v___x_3342_ = v___x_3338_;
goto v_reusejp_3341_;
}
else
{
lean_object* v_reuseFailAlloc_3346_; 
v_reuseFailAlloc_3346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3346_, 0, v_declName_3340_);
v___x_3342_ = v_reuseFailAlloc_3346_;
goto v_reusejp_3341_;
}
v_reusejp_3341_:
{
lean_object* v___x_3344_; 
if (v_isShared_3322_ == 0)
{
lean_ctor_set(v___x_3321_, 0, v___x_3342_);
v___x_3344_ = v___x_3321_;
goto v_reusejp_3343_;
}
else
{
lean_object* v_reuseFailAlloc_3345_; 
v_reuseFailAlloc_3345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3345_, 0, v___x_3342_);
v___x_3344_ = v_reuseFailAlloc_3345_;
goto v_reusejp_3343_;
}
v_reusejp_3343_:
{
return v___x_3344_;
}
}
}
else
{
lean_del_object(v___x_3338_);
lean_dec(v_val_3336_);
lean_del_object(v___x_3321_);
v___y_3324_ = v_a_3175_;
v___y_3325_ = v_a_3176_;
v___y_3326_ = v_a_3177_;
v___y_3327_ = v_a_3178_;
v___y_3328_ = v___x_3223_;
v___y_3329_ = v_a_3180_;
goto v___jp_3323_;
}
}
}
else
{
lean_del_object(v___x_3321_);
lean_dec(v_a_3319_);
v___y_3324_ = v_a_3175_;
v___y_3325_ = v_a_3176_;
v___y_3326_ = v_a_3177_;
v___y_3327_ = v_a_3178_;
v___y_3328_ = v___x_3223_;
v___y_3329_ = v_a_3180_;
goto v___jp_3323_;
}
v___jp_3323_:
{
lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; 
v___x_3330_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3331_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3332_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3332_, 0, v___x_3330_);
lean_ctor_set(v___x_3332_, 1, v___x_3331_);
v___x_3333_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3334_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3334_, 0, v___x_3332_);
lean_ctor_set(v___x_3334_, 1, v___x_3333_);
v___x_3335_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3334_, v___y_3324_, v___y_3325_, v___y_3326_, v___y_3327_, v___y_3328_, v___y_3329_);
lean_dec_ref(v___y_3328_);
return v___x_3335_;
}
}
}
else
{
lean_object* v_a_3349_; lean_object* v___x_3351_; uint8_t v_isShared_3352_; uint8_t v_isSharedCheck_3356_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3349_ = lean_ctor_get(v___x_3318_, 0);
v_isSharedCheck_3356_ = !lean_is_exclusive(v___x_3318_);
if (v_isSharedCheck_3356_ == 0)
{
v___x_3351_ = v___x_3318_;
v_isShared_3352_ = v_isSharedCheck_3356_;
goto v_resetjp_3350_;
}
else
{
lean_inc(v_a_3349_);
lean_dec(v___x_3318_);
v___x_3351_ = lean_box(0);
v_isShared_3352_ = v_isSharedCheck_3356_;
goto v_resetjp_3350_;
}
v_resetjp_3350_:
{
lean_object* v___x_3354_; 
if (v_isShared_3352_ == 0)
{
v___x_3354_ = v___x_3351_;
goto v_reusejp_3353_;
}
else
{
lean_object* v_reuseFailAlloc_3355_; 
v_reuseFailAlloc_3355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3355_, 0, v_a_3349_);
v___x_3354_ = v_reuseFailAlloc_3355_;
goto v_reusejp_3353_;
}
v_reusejp_3353_:
{
return v___x_3354_;
}
}
}
}
else
{
lean_object* v___x_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; uint8_t v___x_3360_; 
v___x_3357_ = lean_unsigned_to_nat(4u);
v___x_3358_ = l_Lean_Syntax_getArg(v_stx_3174_, v___x_3357_);
v___x_3359_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_3360_ = l_Lean_Syntax_isOfKind(v___x_3358_, v___x_3359_);
if (v___x_3360_ == 0)
{
lean_object* v___x_3361_; 
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3361_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3361_) == 0)
{
lean_object* v_a_3362_; lean_object* v___x_3364_; uint8_t v_isShared_3365_; uint8_t v_isSharedCheck_3391_; 
v_a_3362_ = lean_ctor_get(v___x_3361_, 0);
v_isSharedCheck_3391_ = !lean_is_exclusive(v___x_3361_);
if (v_isSharedCheck_3391_ == 0)
{
v___x_3364_ = v___x_3361_;
v_isShared_3365_ = v_isSharedCheck_3391_;
goto v_resetjp_3363_;
}
else
{
lean_inc(v_a_3362_);
lean_dec(v___x_3361_);
v___x_3364_ = lean_box(0);
v_isShared_3365_ = v_isSharedCheck_3391_;
goto v_resetjp_3363_;
}
v_resetjp_3363_:
{
lean_object* v___y_3367_; lean_object* v___y_3368_; lean_object* v___y_3369_; lean_object* v___y_3370_; lean_object* v___y_3371_; lean_object* v___y_3372_; 
if (lean_obj_tag(v_a_3362_) == 1)
{
lean_object* v_val_3379_; lean_object* v___x_3381_; uint8_t v_isShared_3382_; uint8_t v_isSharedCheck_3390_; 
v_val_3379_ = lean_ctor_get(v_a_3362_, 0);
v_isSharedCheck_3390_ = !lean_is_exclusive(v_a_3362_);
if (v_isSharedCheck_3390_ == 0)
{
v___x_3381_ = v_a_3362_;
v_isShared_3382_ = v_isSharedCheck_3390_;
goto v_resetjp_3380_;
}
else
{
lean_inc(v_val_3379_);
lean_dec(v_a_3362_);
v___x_3381_ = lean_box(0);
v_isShared_3382_ = v_isSharedCheck_3390_;
goto v_resetjp_3380_;
}
v_resetjp_3380_:
{
if (lean_obj_tag(v_val_3379_) == 4)
{
lean_object* v_declName_3383_; lean_object* v___x_3385_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3383_ = lean_ctor_get(v_val_3379_, 0);
lean_inc(v_declName_3383_);
lean_dec_ref_known(v_val_3379_, 2);
if (v_isShared_3382_ == 0)
{
lean_ctor_set_tag(v___x_3381_, 0);
lean_ctor_set(v___x_3381_, 0, v_declName_3383_);
v___x_3385_ = v___x_3381_;
goto v_reusejp_3384_;
}
else
{
lean_object* v_reuseFailAlloc_3389_; 
v_reuseFailAlloc_3389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3389_, 0, v_declName_3383_);
v___x_3385_ = v_reuseFailAlloc_3389_;
goto v_reusejp_3384_;
}
v_reusejp_3384_:
{
lean_object* v___x_3387_; 
if (v_isShared_3365_ == 0)
{
lean_ctor_set(v___x_3364_, 0, v___x_3385_);
v___x_3387_ = v___x_3364_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v___x_3385_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
else
{
lean_del_object(v___x_3381_);
lean_dec(v_val_3379_);
lean_del_object(v___x_3364_);
v___y_3367_ = v_a_3175_;
v___y_3368_ = v_a_3176_;
v___y_3369_ = v_a_3177_;
v___y_3370_ = v_a_3178_;
v___y_3371_ = v___x_3223_;
v___y_3372_ = v_a_3180_;
goto v___jp_3366_;
}
}
}
else
{
lean_del_object(v___x_3364_);
lean_dec(v_a_3362_);
v___y_3367_ = v_a_3175_;
v___y_3368_ = v_a_3176_;
v___y_3369_ = v_a_3177_;
v___y_3370_ = v_a_3178_;
v___y_3371_ = v___x_3223_;
v___y_3372_ = v_a_3180_;
goto v___jp_3366_;
}
v___jp_3366_:
{
lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; lean_object* v___x_3377_; lean_object* v___x_3378_; 
v___x_3373_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3374_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3375_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3375_, 0, v___x_3373_);
lean_ctor_set(v___x_3375_, 1, v___x_3374_);
v___x_3376_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3377_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3377_, 0, v___x_3375_);
lean_ctor_set(v___x_3377_, 1, v___x_3376_);
v___x_3378_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3377_, v___y_3367_, v___y_3368_, v___y_3369_, v___y_3370_, v___y_3371_, v___y_3372_);
lean_dec_ref(v___y_3371_);
return v___x_3378_;
}
}
}
else
{
lean_object* v_a_3392_; lean_object* v___x_3394_; uint8_t v_isShared_3395_; uint8_t v_isSharedCheck_3399_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3392_ = lean_ctor_get(v___x_3361_, 0);
v_isSharedCheck_3399_ = !lean_is_exclusive(v___x_3361_);
if (v_isSharedCheck_3399_ == 0)
{
v___x_3394_ = v___x_3361_;
v_isShared_3395_ = v_isSharedCheck_3399_;
goto v_resetjp_3393_;
}
else
{
lean_inc(v_a_3392_);
lean_dec(v___x_3361_);
v___x_3394_ = lean_box(0);
v_isShared_3395_ = v_isSharedCheck_3399_;
goto v_resetjp_3393_;
}
v_resetjp_3393_:
{
lean_object* v___x_3397_; 
if (v_isShared_3395_ == 0)
{
v___x_3397_ = v___x_3394_;
goto v_reusejp_3396_;
}
else
{
lean_object* v_reuseFailAlloc_3398_; 
v_reuseFailAlloc_3398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3398_, 0, v_a_3392_);
v___x_3397_ = v_reuseFailAlloc_3398_;
goto v_reusejp_3396_;
}
v_reusejp_3396_:
{
return v___x_3397_;
}
}
}
}
else
{
lean_object* v___x_3400_; lean_object* v___x_3402_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v___x_3400_ = lean_box(2);
if (v_isShared_3228_ == 0)
{
lean_ctor_set(v___x_3227_, 0, v___x_3400_);
v___x_3402_ = v___x_3227_;
goto v_reusejp_3401_;
}
else
{
lean_object* v_reuseFailAlloc_3403_; 
v_reuseFailAlloc_3403_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3403_, 0, v___x_3400_);
v___x_3402_ = v_reuseFailAlloc_3403_;
goto v_reusejp_3401_;
}
v_reusejp_3401_:
{
return v___x_3402_;
}
}
}
}
}
}
else
{
lean_object* v___x_3404_; lean_object* v___x_3405_; lean_object* v___x_3406_; uint8_t v___x_3407_; 
v___x_3404_ = lean_unsigned_to_nat(1u);
v___x_3405_ = l_Lean_Syntax_getArg(v_stx_3174_, v___x_3404_);
v___x_3406_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__8));
lean_inc(v___x_3405_);
v___x_3407_ = l_Lean_Syntax_isOfKind(v___x_3405_, v___x_3406_);
if (v___x_3407_ == 0)
{
lean_object* v___x_3408_; 
lean_dec(v___x_3405_);
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3408_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3408_) == 0)
{
lean_object* v_a_3409_; lean_object* v___x_3411_; uint8_t v_isShared_3412_; uint8_t v_isSharedCheck_3438_; 
v_a_3409_ = lean_ctor_get(v___x_3408_, 0);
v_isSharedCheck_3438_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3438_ == 0)
{
v___x_3411_ = v___x_3408_;
v_isShared_3412_ = v_isSharedCheck_3438_;
goto v_resetjp_3410_;
}
else
{
lean_inc(v_a_3409_);
lean_dec(v___x_3408_);
v___x_3411_ = lean_box(0);
v_isShared_3412_ = v_isSharedCheck_3438_;
goto v_resetjp_3410_;
}
v_resetjp_3410_:
{
lean_object* v___y_3414_; lean_object* v___y_3415_; lean_object* v___y_3416_; lean_object* v___y_3417_; lean_object* v___y_3418_; lean_object* v___y_3419_; 
if (lean_obj_tag(v_a_3409_) == 1)
{
lean_object* v_val_3426_; lean_object* v___x_3428_; uint8_t v_isShared_3429_; uint8_t v_isSharedCheck_3437_; 
v_val_3426_ = lean_ctor_get(v_a_3409_, 0);
v_isSharedCheck_3437_ = !lean_is_exclusive(v_a_3409_);
if (v_isSharedCheck_3437_ == 0)
{
v___x_3428_ = v_a_3409_;
v_isShared_3429_ = v_isSharedCheck_3437_;
goto v_resetjp_3427_;
}
else
{
lean_inc(v_val_3426_);
lean_dec(v_a_3409_);
v___x_3428_ = lean_box(0);
v_isShared_3429_ = v_isSharedCheck_3437_;
goto v_resetjp_3427_;
}
v_resetjp_3427_:
{
if (lean_obj_tag(v_val_3426_) == 4)
{
lean_object* v_declName_3430_; lean_object* v___x_3432_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3430_ = lean_ctor_get(v_val_3426_, 0);
lean_inc(v_declName_3430_);
lean_dec_ref_known(v_val_3426_, 2);
if (v_isShared_3429_ == 0)
{
lean_ctor_set_tag(v___x_3428_, 0);
lean_ctor_set(v___x_3428_, 0, v_declName_3430_);
v___x_3432_ = v___x_3428_;
goto v_reusejp_3431_;
}
else
{
lean_object* v_reuseFailAlloc_3436_; 
v_reuseFailAlloc_3436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3436_, 0, v_declName_3430_);
v___x_3432_ = v_reuseFailAlloc_3436_;
goto v_reusejp_3431_;
}
v_reusejp_3431_:
{
lean_object* v___x_3434_; 
if (v_isShared_3412_ == 0)
{
lean_ctor_set(v___x_3411_, 0, v___x_3432_);
v___x_3434_ = v___x_3411_;
goto v_reusejp_3433_;
}
else
{
lean_object* v_reuseFailAlloc_3435_; 
v_reuseFailAlloc_3435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3435_, 0, v___x_3432_);
v___x_3434_ = v_reuseFailAlloc_3435_;
goto v_reusejp_3433_;
}
v_reusejp_3433_:
{
return v___x_3434_;
}
}
}
else
{
lean_del_object(v___x_3428_);
lean_dec(v_val_3426_);
lean_del_object(v___x_3411_);
v___y_3414_ = v_a_3175_;
v___y_3415_ = v_a_3176_;
v___y_3416_ = v_a_3177_;
v___y_3417_ = v_a_3178_;
v___y_3418_ = v___x_3223_;
v___y_3419_ = v_a_3180_;
goto v___jp_3413_;
}
}
}
else
{
lean_del_object(v___x_3411_);
lean_dec(v_a_3409_);
v___y_3414_ = v_a_3175_;
v___y_3415_ = v_a_3176_;
v___y_3416_ = v_a_3177_;
v___y_3417_ = v_a_3178_;
v___y_3418_ = v___x_3223_;
v___y_3419_ = v_a_3180_;
goto v___jp_3413_;
}
v___jp_3413_:
{
lean_object* v___x_3420_; lean_object* v___x_3421_; lean_object* v___x_3422_; lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v___x_3425_; 
v___x_3420_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3421_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3422_, 0, v___x_3420_);
lean_ctor_set(v___x_3422_, 1, v___x_3421_);
v___x_3423_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3424_, 0, v___x_3422_);
lean_ctor_set(v___x_3424_, 1, v___x_3423_);
v___x_3425_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3424_, v___y_3414_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_);
lean_dec_ref(v___y_3418_);
return v___x_3425_;
}
}
}
else
{
lean_object* v_a_3439_; lean_object* v___x_3441_; uint8_t v_isShared_3442_; uint8_t v_isSharedCheck_3446_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3439_ = lean_ctor_get(v___x_3408_, 0);
v_isSharedCheck_3446_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3446_ == 0)
{
v___x_3441_ = v___x_3408_;
v_isShared_3442_ = v_isSharedCheck_3446_;
goto v_resetjp_3440_;
}
else
{
lean_inc(v_a_3439_);
lean_dec(v___x_3408_);
v___x_3441_ = lean_box(0);
v_isShared_3442_ = v_isSharedCheck_3446_;
goto v_resetjp_3440_;
}
v_resetjp_3440_:
{
lean_object* v___x_3444_; 
if (v_isShared_3442_ == 0)
{
v___x_3444_ = v___x_3441_;
goto v_reusejp_3443_;
}
else
{
lean_object* v_reuseFailAlloc_3445_; 
v_reuseFailAlloc_3445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3445_, 0, v_a_3439_);
v___x_3444_ = v_reuseFailAlloc_3445_;
goto v_reusejp_3443_;
}
v_reusejp_3443_:
{
return v___x_3444_;
}
}
}
}
else
{
lean_object* v___x_3447_; lean_object* v___x_3448_; uint8_t v___x_3449_; 
v___x_3447_ = lean_unsigned_to_nat(0u);
v___x_3448_ = l_Lean_Syntax_getArg(v___x_3405_, v___x_3447_);
v___x_3449_ = l_Lean_Syntax_matchesNull(v___x_3448_, v___x_3404_);
if (v___x_3449_ == 0)
{
lean_object* v___x_3450_; 
lean_dec(v___x_3405_);
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3450_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3450_) == 0)
{
lean_object* v_a_3451_; lean_object* v___x_3453_; uint8_t v_isShared_3454_; uint8_t v_isSharedCheck_3480_; 
v_a_3451_ = lean_ctor_get(v___x_3450_, 0);
v_isSharedCheck_3480_ = !lean_is_exclusive(v___x_3450_);
if (v_isSharedCheck_3480_ == 0)
{
v___x_3453_ = v___x_3450_;
v_isShared_3454_ = v_isSharedCheck_3480_;
goto v_resetjp_3452_;
}
else
{
lean_inc(v_a_3451_);
lean_dec(v___x_3450_);
v___x_3453_ = lean_box(0);
v_isShared_3454_ = v_isSharedCheck_3480_;
goto v_resetjp_3452_;
}
v_resetjp_3452_:
{
lean_object* v___y_3456_; lean_object* v___y_3457_; lean_object* v___y_3458_; lean_object* v___y_3459_; lean_object* v___y_3460_; lean_object* v___y_3461_; 
if (lean_obj_tag(v_a_3451_) == 1)
{
lean_object* v_val_3468_; lean_object* v___x_3470_; uint8_t v_isShared_3471_; uint8_t v_isSharedCheck_3479_; 
v_val_3468_ = lean_ctor_get(v_a_3451_, 0);
v_isSharedCheck_3479_ = !lean_is_exclusive(v_a_3451_);
if (v_isSharedCheck_3479_ == 0)
{
v___x_3470_ = v_a_3451_;
v_isShared_3471_ = v_isSharedCheck_3479_;
goto v_resetjp_3469_;
}
else
{
lean_inc(v_val_3468_);
lean_dec(v_a_3451_);
v___x_3470_ = lean_box(0);
v_isShared_3471_ = v_isSharedCheck_3479_;
goto v_resetjp_3469_;
}
v_resetjp_3469_:
{
if (lean_obj_tag(v_val_3468_) == 4)
{
lean_object* v_declName_3472_; lean_object* v___x_3474_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3472_ = lean_ctor_get(v_val_3468_, 0);
lean_inc(v_declName_3472_);
lean_dec_ref_known(v_val_3468_, 2);
if (v_isShared_3471_ == 0)
{
lean_ctor_set_tag(v___x_3470_, 0);
lean_ctor_set(v___x_3470_, 0, v_declName_3472_);
v___x_3474_ = v___x_3470_;
goto v_reusejp_3473_;
}
else
{
lean_object* v_reuseFailAlloc_3478_; 
v_reuseFailAlloc_3478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3478_, 0, v_declName_3472_);
v___x_3474_ = v_reuseFailAlloc_3478_;
goto v_reusejp_3473_;
}
v_reusejp_3473_:
{
lean_object* v___x_3476_; 
if (v_isShared_3454_ == 0)
{
lean_ctor_set(v___x_3453_, 0, v___x_3474_);
v___x_3476_ = v___x_3453_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3477_; 
v_reuseFailAlloc_3477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3477_, 0, v___x_3474_);
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
lean_del_object(v___x_3470_);
lean_dec(v_val_3468_);
lean_del_object(v___x_3453_);
v___y_3456_ = v_a_3175_;
v___y_3457_ = v_a_3176_;
v___y_3458_ = v_a_3177_;
v___y_3459_ = v_a_3178_;
v___y_3460_ = v___x_3223_;
v___y_3461_ = v_a_3180_;
goto v___jp_3455_;
}
}
}
else
{
lean_del_object(v___x_3453_);
lean_dec(v_a_3451_);
v___y_3456_ = v_a_3175_;
v___y_3457_ = v_a_3176_;
v___y_3458_ = v_a_3177_;
v___y_3459_ = v_a_3178_;
v___y_3460_ = v___x_3223_;
v___y_3461_ = v_a_3180_;
goto v___jp_3455_;
}
v___jp_3455_:
{
lean_object* v___x_3462_; lean_object* v___x_3463_; lean_object* v___x_3464_; lean_object* v___x_3465_; lean_object* v___x_3466_; lean_object* v___x_3467_; 
v___x_3462_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3463_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3464_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3464_, 0, v___x_3462_);
lean_ctor_set(v___x_3464_, 1, v___x_3463_);
v___x_3465_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3466_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3466_, 0, v___x_3464_);
lean_ctor_set(v___x_3466_, 1, v___x_3465_);
v___x_3467_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3466_, v___y_3456_, v___y_3457_, v___y_3458_, v___y_3459_, v___y_3460_, v___y_3461_);
lean_dec_ref(v___y_3460_);
return v___x_3467_;
}
}
}
else
{
lean_object* v_a_3481_; lean_object* v___x_3483_; uint8_t v_isShared_3484_; uint8_t v_isSharedCheck_3488_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3481_ = lean_ctor_get(v___x_3450_, 0);
v_isSharedCheck_3488_ = !lean_is_exclusive(v___x_3450_);
if (v_isSharedCheck_3488_ == 0)
{
v___x_3483_ = v___x_3450_;
v_isShared_3484_ = v_isSharedCheck_3488_;
goto v_resetjp_3482_;
}
else
{
lean_inc(v_a_3481_);
lean_dec(v___x_3450_);
v___x_3483_ = lean_box(0);
v_isShared_3484_ = v_isSharedCheck_3488_;
goto v_resetjp_3482_;
}
v_resetjp_3482_:
{
lean_object* v___x_3486_; 
if (v_isShared_3484_ == 0)
{
v___x_3486_ = v___x_3483_;
goto v_reusejp_3485_;
}
else
{
lean_object* v_reuseFailAlloc_3487_; 
v_reuseFailAlloc_3487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3487_, 0, v_a_3481_);
v___x_3486_ = v_reuseFailAlloc_3487_;
goto v_reusejp_3485_;
}
v_reusejp_3485_:
{
return v___x_3486_;
}
}
}
}
else
{
lean_object* v___x_3489_; uint8_t v___x_3490_; 
v___x_3489_ = l_Lean_Syntax_getArg(v___x_3405_, v___x_3404_);
v___x_3490_ = l_Lean_Syntax_matchesNull(v___x_3489_, v___x_3447_);
if (v___x_3490_ == 0)
{
lean_object* v___x_3491_; 
lean_dec(v___x_3405_);
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3491_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3491_) == 0)
{
lean_object* v_a_3492_; lean_object* v___x_3494_; uint8_t v_isShared_3495_; uint8_t v_isSharedCheck_3521_; 
v_a_3492_ = lean_ctor_get(v___x_3491_, 0);
v_isSharedCheck_3521_ = !lean_is_exclusive(v___x_3491_);
if (v_isSharedCheck_3521_ == 0)
{
v___x_3494_ = v___x_3491_;
v_isShared_3495_ = v_isSharedCheck_3521_;
goto v_resetjp_3493_;
}
else
{
lean_inc(v_a_3492_);
lean_dec(v___x_3491_);
v___x_3494_ = lean_box(0);
v_isShared_3495_ = v_isSharedCheck_3521_;
goto v_resetjp_3493_;
}
v_resetjp_3493_:
{
lean_object* v___y_3497_; lean_object* v___y_3498_; lean_object* v___y_3499_; lean_object* v___y_3500_; lean_object* v___y_3501_; lean_object* v___y_3502_; 
if (lean_obj_tag(v_a_3492_) == 1)
{
lean_object* v_val_3509_; lean_object* v___x_3511_; uint8_t v_isShared_3512_; uint8_t v_isSharedCheck_3520_; 
v_val_3509_ = lean_ctor_get(v_a_3492_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v_a_3492_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3511_ = v_a_3492_;
v_isShared_3512_ = v_isSharedCheck_3520_;
goto v_resetjp_3510_;
}
else
{
lean_inc(v_val_3509_);
lean_dec(v_a_3492_);
v___x_3511_ = lean_box(0);
v_isShared_3512_ = v_isSharedCheck_3520_;
goto v_resetjp_3510_;
}
v_resetjp_3510_:
{
if (lean_obj_tag(v_val_3509_) == 4)
{
lean_object* v_declName_3513_; lean_object* v___x_3515_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3513_ = lean_ctor_get(v_val_3509_, 0);
lean_inc(v_declName_3513_);
lean_dec_ref_known(v_val_3509_, 2);
if (v_isShared_3512_ == 0)
{
lean_ctor_set_tag(v___x_3511_, 0);
lean_ctor_set(v___x_3511_, 0, v_declName_3513_);
v___x_3515_ = v___x_3511_;
goto v_reusejp_3514_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v_declName_3513_);
v___x_3515_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3514_;
}
v_reusejp_3514_:
{
lean_object* v___x_3517_; 
if (v_isShared_3495_ == 0)
{
lean_ctor_set(v___x_3494_, 0, v___x_3515_);
v___x_3517_ = v___x_3494_;
goto v_reusejp_3516_;
}
else
{
lean_object* v_reuseFailAlloc_3518_; 
v_reuseFailAlloc_3518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3518_, 0, v___x_3515_);
v___x_3517_ = v_reuseFailAlloc_3518_;
goto v_reusejp_3516_;
}
v_reusejp_3516_:
{
return v___x_3517_;
}
}
}
else
{
lean_del_object(v___x_3511_);
lean_dec(v_val_3509_);
lean_del_object(v___x_3494_);
v___y_3497_ = v_a_3175_;
v___y_3498_ = v_a_3176_;
v___y_3499_ = v_a_3177_;
v___y_3500_ = v_a_3178_;
v___y_3501_ = v___x_3223_;
v___y_3502_ = v_a_3180_;
goto v___jp_3496_;
}
}
}
else
{
lean_del_object(v___x_3494_);
lean_dec(v_a_3492_);
v___y_3497_ = v_a_3175_;
v___y_3498_ = v_a_3176_;
v___y_3499_ = v_a_3177_;
v___y_3500_ = v_a_3178_;
v___y_3501_ = v___x_3223_;
v___y_3502_ = v_a_3180_;
goto v___jp_3496_;
}
v___jp_3496_:
{
lean_object* v___x_3503_; lean_object* v___x_3504_; lean_object* v___x_3505_; lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3508_; 
v___x_3503_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3504_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3505_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3505_, 0, v___x_3503_);
lean_ctor_set(v___x_3505_, 1, v___x_3504_);
v___x_3506_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3507_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3507_, 0, v___x_3505_);
lean_ctor_set(v___x_3507_, 1, v___x_3506_);
v___x_3508_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3507_, v___y_3497_, v___y_3498_, v___y_3499_, v___y_3500_, v___y_3501_, v___y_3502_);
lean_dec_ref(v___y_3501_);
return v___x_3508_;
}
}
}
else
{
lean_object* v_a_3522_; lean_object* v___x_3524_; uint8_t v_isShared_3525_; uint8_t v_isSharedCheck_3529_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3522_ = lean_ctor_get(v___x_3491_, 0);
v_isSharedCheck_3529_ = !lean_is_exclusive(v___x_3491_);
if (v_isSharedCheck_3529_ == 0)
{
v___x_3524_ = v___x_3491_;
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
else
{
lean_inc(v_a_3522_);
lean_dec(v___x_3491_);
v___x_3524_ = lean_box(0);
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
v_resetjp_3523_:
{
lean_object* v___x_3527_; 
if (v_isShared_3525_ == 0)
{
v___x_3527_ = v___x_3524_;
goto v_reusejp_3526_;
}
else
{
lean_object* v_reuseFailAlloc_3528_; 
v_reuseFailAlloc_3528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3528_, 0, v_a_3522_);
v___x_3527_ = v_reuseFailAlloc_3528_;
goto v_reusejp_3526_;
}
v_reusejp_3526_:
{
return v___x_3527_;
}
}
}
}
else
{
lean_object* v___x_3530_; lean_object* v___x_3531_; lean_object* v___x_3532_; uint8_t v___x_3533_; 
v___x_3530_ = lean_unsigned_to_nat(3u);
v___x_3531_ = l_Lean_Syntax_getArg(v___x_3405_, v___x_3530_);
lean_dec(v___x_3405_);
v___x_3532_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_isUnderscore___closed__4));
v___x_3533_ = l_Lean_Syntax_isOfKind(v___x_3531_, v___x_3532_);
if (v___x_3533_ == 0)
{
lean_object* v___x_3534_; 
lean_del_object(v___x_3227_);
lean_inc(v_stx_3174_);
v___x_3534_ = lp_mathlib_Mathlib_Tactic_Push_resolvePushId_x3f(v_stx_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v___x_3223_, v_a_3180_);
if (lean_obj_tag(v___x_3534_) == 0)
{
lean_object* v_a_3535_; lean_object* v___x_3537_; uint8_t v_isShared_3538_; uint8_t v_isSharedCheck_3564_; 
v_a_3535_ = lean_ctor_get(v___x_3534_, 0);
v_isSharedCheck_3564_ = !lean_is_exclusive(v___x_3534_);
if (v_isSharedCheck_3564_ == 0)
{
v___x_3537_ = v___x_3534_;
v_isShared_3538_ = v_isSharedCheck_3564_;
goto v_resetjp_3536_;
}
else
{
lean_inc(v_a_3535_);
lean_dec(v___x_3534_);
v___x_3537_ = lean_box(0);
v_isShared_3538_ = v_isSharedCheck_3564_;
goto v_resetjp_3536_;
}
v_resetjp_3536_:
{
lean_object* v___y_3540_; lean_object* v___y_3541_; lean_object* v___y_3542_; lean_object* v___y_3543_; lean_object* v___y_3544_; lean_object* v___y_3545_; 
if (lean_obj_tag(v_a_3535_) == 1)
{
lean_object* v_val_3552_; lean_object* v___x_3554_; uint8_t v_isShared_3555_; uint8_t v_isSharedCheck_3563_; 
v_val_3552_ = lean_ctor_get(v_a_3535_, 0);
v_isSharedCheck_3563_ = !lean_is_exclusive(v_a_3535_);
if (v_isSharedCheck_3563_ == 0)
{
v___x_3554_ = v_a_3535_;
v_isShared_3555_ = v_isSharedCheck_3563_;
goto v_resetjp_3553_;
}
else
{
lean_inc(v_val_3552_);
lean_dec(v_a_3535_);
v___x_3554_ = lean_box(0);
v_isShared_3555_ = v_isSharedCheck_3563_;
goto v_resetjp_3553_;
}
v_resetjp_3553_:
{
if (lean_obj_tag(v_val_3552_) == 4)
{
lean_object* v_declName_3556_; lean_object* v___x_3558_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_declName_3556_ = lean_ctor_get(v_val_3552_, 0);
lean_inc(v_declName_3556_);
lean_dec_ref_known(v_val_3552_, 2);
if (v_isShared_3555_ == 0)
{
lean_ctor_set_tag(v___x_3554_, 0);
lean_ctor_set(v___x_3554_, 0, v_declName_3556_);
v___x_3558_ = v___x_3554_;
goto v_reusejp_3557_;
}
else
{
lean_object* v_reuseFailAlloc_3562_; 
v_reuseFailAlloc_3562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3562_, 0, v_declName_3556_);
v___x_3558_ = v_reuseFailAlloc_3562_;
goto v_reusejp_3557_;
}
v_reusejp_3557_:
{
lean_object* v___x_3560_; 
if (v_isShared_3538_ == 0)
{
lean_ctor_set(v___x_3537_, 0, v___x_3558_);
v___x_3560_ = v___x_3537_;
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
else
{
lean_del_object(v___x_3554_);
lean_dec(v_val_3552_);
lean_del_object(v___x_3537_);
v___y_3540_ = v_a_3175_;
v___y_3541_ = v_a_3176_;
v___y_3542_ = v_a_3177_;
v___y_3543_ = v_a_3178_;
v___y_3544_ = v___x_3223_;
v___y_3545_ = v_a_3180_;
goto v___jp_3539_;
}
}
}
else
{
lean_del_object(v___x_3537_);
lean_dec(v_a_3535_);
v___y_3540_ = v_a_3175_;
v___y_3541_ = v_a_3176_;
v___y_3542_ = v_a_3177_;
v___y_3543_ = v_a_3178_;
v___y_3544_ = v___x_3223_;
v___y_3545_ = v_a_3180_;
goto v___jp_3539_;
}
v___jp_3539_:
{
lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; lean_object* v___x_3550_; lean_object* v___x_3551_; 
v___x_3546_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__3);
v___x_3547_ = l_Lean_MessageData_ofSyntax(v_stx_3174_);
v___x_3548_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3548_, 0, v___x_3546_);
lean_ctor_set(v___x_3548_, 1, v___x_3547_);
v___x_3549_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5, &lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabHead___closed__5);
v___x_3550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3550_, 0, v___x_3548_);
lean_ctor_set(v___x_3550_, 1, v___x_3549_);
v___x_3551_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3550_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_, v___y_3544_, v___y_3545_);
lean_dec_ref(v___y_3544_);
return v___x_3551_;
}
}
}
else
{
lean_object* v_a_3565_; lean_object* v___x_3567_; uint8_t v_isShared_3568_; uint8_t v_isSharedCheck_3572_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3565_ = lean_ctor_get(v___x_3534_, 0);
v_isSharedCheck_3572_ = !lean_is_exclusive(v___x_3534_);
if (v_isSharedCheck_3572_ == 0)
{
v___x_3567_ = v___x_3534_;
v_isShared_3568_ = v_isSharedCheck_3572_;
goto v_resetjp_3566_;
}
else
{
lean_inc(v_a_3565_);
lean_dec(v___x_3534_);
v___x_3567_ = lean_box(0);
v_isShared_3568_ = v_isSharedCheck_3572_;
goto v_resetjp_3566_;
}
v_resetjp_3566_:
{
lean_object* v___x_3570_; 
if (v_isShared_3568_ == 0)
{
v___x_3570_ = v___x_3567_;
goto v_reusejp_3569_;
}
else
{
lean_object* v_reuseFailAlloc_3571_; 
v_reuseFailAlloc_3571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3571_, 0, v_a_3565_);
v___x_3570_ = v_reuseFailAlloc_3571_;
goto v_reusejp_3569_;
}
v_reusejp_3569_:
{
return v___x_3570_;
}
}
}
}
else
{
lean_object* v___x_3573_; lean_object* v___x_3575_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v___x_3573_ = lean_box(1);
if (v_isShared_3228_ == 0)
{
lean_ctor_set(v___x_3227_, 0, v___x_3573_);
v___x_3575_ = v___x_3227_;
goto v_reusejp_3574_;
}
else
{
lean_object* v_reuseFailAlloc_3576_; 
v_reuseFailAlloc_3576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3576_, 0, v___x_3573_);
v___x_3575_ = v_reuseFailAlloc_3576_;
goto v_reusejp_3574_;
}
v_reusejp_3574_:
{
return v___x_3575_;
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
lean_object* v_a_3579_; lean_object* v___x_3581_; uint8_t v_isShared_3582_; uint8_t v_isSharedCheck_3586_; 
lean_dec_ref_known(v___x_3223_, 14);
lean_dec(v_stx_3174_);
v_a_3579_ = lean_ctor_get(v___x_3225_, 0);
v_isSharedCheck_3586_ = !lean_is_exclusive(v___x_3225_);
if (v_isSharedCheck_3586_ == 0)
{
v___x_3581_ = v___x_3225_;
v_isShared_3582_ = v_isSharedCheck_3586_;
goto v_resetjp_3580_;
}
else
{
lean_inc(v_a_3579_);
lean_dec(v___x_3225_);
v___x_3581_ = lean_box(0);
v_isShared_3582_ = v_isSharedCheck_3586_;
goto v_resetjp_3580_;
}
v_resetjp_3580_:
{
lean_object* v___x_3584_; 
if (v_isShared_3582_ == 0)
{
v___x_3584_ = v___x_3581_;
goto v_reusejp_3583_;
}
else
{
lean_object* v_reuseFailAlloc_3585_; 
v_reuseFailAlloc_3585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3585_, 0, v_a_3579_);
v___x_3584_ = v_reuseFailAlloc_3585_;
goto v_reusejp_3583_;
}
v_reusejp_3583_:
{
return v___x_3584_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabHead___boxed(lean_object* v_stx_3587_, lean_object* v_a_3588_, lean_object* v_a_3589_, lean_object* v_a_3590_, lean_object* v_a_3591_, lean_object* v_a_3592_, lean_object* v_a_3593_, lean_object* v_a_3594_){
_start:
{
lean_object* v_res_3595_; 
v_res_3595_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v_stx_3587_, v_a_3588_, v_a_3589_, v_a_3590_, v_a_3591_, v_a_3592_, v_a_3593_);
lean_dec(v_a_3593_);
lean_dec_ref(v_a_3592_);
lean_dec(v_a_3591_);
lean_dec_ref(v_a_3590_);
lean_dec(v_a_3589_);
lean_dec_ref(v_a_3588_);
return v_res_3595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0(lean_object* v_a_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_, lean_object* v___y_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_, lean_object* v___y_3604_){
_start:
{
lean_object* v_snd_3606_; lean_object* v___x_3607_; 
v_snd_3606_ = lean_ctor_get(v_a_3596_, 1);
lean_inc(v_snd_3606_);
lean_dec_ref(v_a_3596_);
v___x_3607_ = lean_apply_9(v_snd_3606_, v___y_3597_, v___y_3598_, v___y_3599_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_, v___y_3604_, lean_box(0));
return v___x_3607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0___boxed(lean_object* v_a_3608_, lean_object* v___y_3609_, lean_object* v___y_3610_, lean_object* v___y_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_){
_start:
{
lean_object* v_res_3618_; 
v_res_3618_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0(v_a_3608_, v___y_3609_, v___y_3610_, v___y_3611_, v___y_3612_, v___y_3613_, v___y_3614_, v___y_3615_, v___y_3616_);
return v_res_3618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(lean_object* v_stx_3619_, lean_object* v_a_3620_, lean_object* v_a_3621_, lean_object* v_a_3622_){
_start:
{
lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3626_; 
v___x_3624_ = lean_unsigned_to_nat(3u);
v___x_3625_ = l_Lean_Syntax_getArg(v_stx_3619_, v___x_3624_);
v___x_3626_ = l_Lean_Elab_Tactic_tacticToDischarge___redArg(v___x_3625_, v_a_3620_, v_a_3621_, v_a_3622_);
if (lean_obj_tag(v___x_3626_) == 0)
{
lean_object* v_a_3627_; lean_object* v___x_3629_; uint8_t v_isShared_3630_; uint8_t v_isSharedCheck_3635_; 
v_a_3627_ = lean_ctor_get(v___x_3626_, 0);
v_isSharedCheck_3635_ = !lean_is_exclusive(v___x_3626_);
if (v_isSharedCheck_3635_ == 0)
{
v___x_3629_ = v___x_3626_;
v_isShared_3630_ = v_isSharedCheck_3635_;
goto v_resetjp_3628_;
}
else
{
lean_inc(v_a_3627_);
lean_dec(v___x_3626_);
v___x_3629_ = lean_box(0);
v_isShared_3630_ = v_isSharedCheck_3635_;
goto v_resetjp_3628_;
}
v_resetjp_3628_:
{
lean_object* v___f_3631_; lean_object* v___x_3633_; 
v___f_3631_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_3631_, 0, v_a_3627_);
if (v_isShared_3630_ == 0)
{
lean_ctor_set(v___x_3629_, 0, v___f_3631_);
v___x_3633_ = v___x_3629_;
goto v_reusejp_3632_;
}
else
{
lean_object* v_reuseFailAlloc_3634_; 
v_reuseFailAlloc_3634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3634_, 0, v___f_3631_);
v___x_3633_ = v_reuseFailAlloc_3634_;
goto v_reusejp_3632_;
}
v_reusejp_3632_:
{
return v___x_3633_;
}
}
}
else
{
lean_object* v_a_3636_; lean_object* v___x_3638_; uint8_t v_isShared_3639_; uint8_t v_isSharedCheck_3643_; 
v_a_3636_ = lean_ctor_get(v___x_3626_, 0);
v_isSharedCheck_3643_ = !lean_is_exclusive(v___x_3626_);
if (v_isSharedCheck_3643_ == 0)
{
v___x_3638_ = v___x_3626_;
v_isShared_3639_ = v_isSharedCheck_3643_;
goto v_resetjp_3637_;
}
else
{
lean_inc(v_a_3636_);
lean_dec(v___x_3626_);
v___x_3638_ = lean_box(0);
v_isShared_3639_ = v_isSharedCheck_3643_;
goto v_resetjp_3637_;
}
v_resetjp_3637_:
{
lean_object* v___x_3641_; 
if (v_isShared_3639_ == 0)
{
v___x_3641_ = v___x_3638_;
goto v_reusejp_3640_;
}
else
{
lean_object* v_reuseFailAlloc_3642_; 
v_reuseFailAlloc_3642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3642_, 0, v_a_3636_);
v___x_3641_ = v_reuseFailAlloc_3642_;
goto v_reusejp_3640_;
}
v_reusejp_3640_:
{
return v___x_3641_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg___boxed(lean_object* v_stx_3644_, lean_object* v_a_3645_, lean_object* v_a_3646_, lean_object* v_a_3647_, lean_object* v_a_3648_){
_start:
{
lean_object* v_res_3649_; 
v_res_3649_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_stx_3644_, v_a_3645_, v_a_3646_, v_a_3647_);
lean_dec_ref(v_a_3647_);
lean_dec(v_a_3646_);
lean_dec_ref(v_a_3645_);
lean_dec(v_stx_3644_);
return v_res_3649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger(lean_object* v_stx_3650_, lean_object* v_a_3651_, lean_object* v_a_3652_, lean_object* v_a_3653_, lean_object* v_a_3654_, lean_object* v_a_3655_, lean_object* v_a_3656_, lean_object* v_a_3657_, lean_object* v_a_3658_){
_start:
{
lean_object* v___x_3660_; 
v___x_3660_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_stx_3650_, v_a_3653_, v_a_3654_, v_a_3657_);
return v___x_3660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabDischarger___boxed(lean_object* v_stx_3661_, lean_object* v_a_3662_, lean_object* v_a_3663_, lean_object* v_a_3664_, lean_object* v_a_3665_, lean_object* v_a_3666_, lean_object* v_a_3667_, lean_object* v_a_3668_, lean_object* v_a_3669_, lean_object* v_a_3670_){
_start:
{
lean_object* v_res_3671_; 
v_res_3671_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger(v_stx_3661_, v_a_3662_, v_a_3663_, v_a_3664_, v_a_3665_, v_a_3666_, v_a_3667_, v_a_3668_, v_a_3669_);
lean_dec(v_a_3669_);
lean_dec_ref(v_a_3668_);
lean_dec(v_a_3667_);
lean_dec_ref(v_a_3666_);
lean_dec(v_a_3665_);
lean_dec_ref(v_a_3664_);
lean_dec(v_a_3663_);
lean_dec_ref(v_a_3662_);
lean_dec(v_stx_3661_);
return v_res_3671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(lean_object* v_k_3672_, uint8_t v_defValue_3673_, lean_object* v___y_3674_){
_start:
{
lean_object* v_options_3676_; lean_object* v_map_3677_; lean_object* v___x_3678_; 
v_options_3676_ = lean_ctor_get(v___y_3674_, 2);
v_map_3677_ = lean_ctor_get(v_options_3676_, 0);
v___x_3678_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_3677_, v_k_3672_);
if (lean_obj_tag(v___x_3678_) == 0)
{
lean_object* v___x_3679_; lean_object* v___x_3680_; 
v___x_3679_ = lean_box(v_defValue_3673_);
v___x_3680_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3680_, 0, v___x_3679_);
return v___x_3680_;
}
else
{
lean_object* v_val_3681_; lean_object* v___x_3683_; uint8_t v_isShared_3684_; uint8_t v_isSharedCheck_3694_; 
v_val_3681_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3694_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3694_ == 0)
{
v___x_3683_ = v___x_3678_;
v_isShared_3684_ = v_isSharedCheck_3694_;
goto v_resetjp_3682_;
}
else
{
lean_inc(v_val_3681_);
lean_dec(v___x_3678_);
v___x_3683_ = lean_box(0);
v_isShared_3684_ = v_isSharedCheck_3694_;
goto v_resetjp_3682_;
}
v_resetjp_3682_:
{
if (lean_obj_tag(v_val_3681_) == 1)
{
uint8_t v_v_3685_; lean_object* v___x_3686_; lean_object* v___x_3688_; 
v_v_3685_ = lean_ctor_get_uint8(v_val_3681_, 0);
lean_dec_ref_known(v_val_3681_, 0);
v___x_3686_ = lean_box(v_v_3685_);
if (v_isShared_3684_ == 0)
{
lean_ctor_set_tag(v___x_3683_, 0);
lean_ctor_set(v___x_3683_, 0, v___x_3686_);
v___x_3688_ = v___x_3683_;
goto v_reusejp_3687_;
}
else
{
lean_object* v_reuseFailAlloc_3689_; 
v_reuseFailAlloc_3689_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3689_, 0, v___x_3686_);
v___x_3688_ = v_reuseFailAlloc_3689_;
goto v_reusejp_3687_;
}
v_reusejp_3687_:
{
return v___x_3688_;
}
}
else
{
lean_object* v___x_3690_; lean_object* v___x_3692_; 
lean_dec(v_val_3681_);
v___x_3690_ = lean_box(v_defValue_3673_);
if (v_isShared_3684_ == 0)
{
lean_ctor_set_tag(v___x_3683_, 0);
lean_ctor_set(v___x_3683_, 0, v___x_3690_);
v___x_3692_ = v___x_3683_;
goto v_reusejp_3691_;
}
else
{
lean_object* v_reuseFailAlloc_3693_; 
v_reuseFailAlloc_3693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3693_, 0, v___x_3690_);
v___x_3692_ = v_reuseFailAlloc_3693_;
goto v_reusejp_3691_;
}
v_reusejp_3691_:
{
return v___x_3692_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg___boxed(lean_object* v_k_3695_, lean_object* v_defValue_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
uint8_t v_defValue_boxed_3699_; lean_object* v_res_3700_; 
v_defValue_boxed_3699_ = lean_unbox(v_defValue_3696_);
v_res_3700_ = lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(v_k_3695_, v_defValue_boxed_3699_, v___y_3697_);
lean_dec_ref(v___y_3697_);
lean_dec(v_k_3695_);
return v_res_3700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0(lean_object* v_k_3701_, uint8_t v_defValue_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_, lean_object* v___y_3706_, lean_object* v___y_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_){
_start:
{
lean_object* v___x_3712_; 
v___x_3712_ = lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(v_k_3701_, v_defValue_3702_, v___y_3709_);
return v___x_3712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___boxed(lean_object* v_k_3713_, lean_object* v_defValue_3714_, lean_object* v___y_3715_, lean_object* v___y_3716_, lean_object* v___y_3717_, lean_object* v___y_3718_, lean_object* v___y_3719_, lean_object* v___y_3720_, lean_object* v___y_3721_, lean_object* v___y_3722_, lean_object* v___y_3723_){
_start:
{
uint8_t v_defValue_boxed_3724_; lean_object* v_res_3725_; 
v_defValue_boxed_3724_ = lean_unbox(v_defValue_3714_);
v_res_3725_ = lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0(v_k_3713_, v_defValue_boxed_3724_, v___y_3715_, v___y_3716_, v___y_3717_, v___y_3718_, v___y_3719_, v___y_3720_, v___y_3721_, v___y_3722_);
lean_dec(v___y_3722_);
lean_dec_ref(v___y_3721_);
lean_dec(v___y_3720_);
lean_dec_ref(v___y_3719_);
lean_dec(v___y_3718_);
lean_dec_ref(v___y_3717_);
lean_dec(v___y_3716_);
lean_dec_ref(v___y_3715_);
lean_dec(v_k_3713_);
return v_res_3725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___lam__0(lean_object* v_head_3726_, uint8_t v___y_3727_, lean_object* v_disch_x3f_3728_, lean_object* v_x_3729_, lean_object* v___y_3730_, lean_object* v___y_3731_, lean_object* v___y_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_){
_start:
{
lean_object* v___x_3736_; 
v___x_3736_ = lp_mathlib_Mathlib_Tactic_Push_pushCore(v_head_3726_, v___y_3727_, v_disch_x3f_3728_, v_x_3729_, v___y_3731_, v___y_3732_, v___y_3733_, v___y_3734_);
return v___x_3736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___lam__0___boxed(lean_object* v_head_3737_, lean_object* v___y_3738_, lean_object* v_disch_x3f_3739_, lean_object* v_x_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_, lean_object* v___y_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_, lean_object* v___y_3746_){
_start:
{
uint8_t v___y_609__boxed_3747_; lean_object* v_res_3748_; 
v___y_609__boxed_3747_ = lean_unbox(v___y_3738_);
v_res_3748_ = lp_mathlib_Mathlib_Tactic_Push_push___lam__0(v_head_3737_, v___y_609__boxed_3747_, v_disch_x3f_3739_, v_x_3740_, v___y_3741_, v___y_3742_, v___y_3743_, v___y_3744_, v___y_3745_);
lean_dec(v___y_3745_);
lean_dec_ref(v___y_3744_);
lean_dec(v___y_3743_);
lean_dec_ref(v___y_3742_);
lean_dec_ref(v___y_3741_);
lean_dec(v_disch_x3f_3739_);
return v_res_3748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t v_cfg_3750_, lean_object* v_disch_x3f_3751_, lean_object* v_head_3752_, lean_object* v_loc_3753_, uint8_t v_ifUnchanged_3754_, lean_object* v_a_3755_, lean_object* v_a_3756_, lean_object* v_a_3757_, lean_object* v_a_3758_, lean_object* v_a_3759_, lean_object* v_a_3760_, lean_object* v_a_3761_, lean_object* v_a_3762_){
_start:
{
lean_object* v___x_3764_; uint8_t v___x_3765_; lean_object* v___x_3766_; lean_object* v_a_3767_; uint8_t v___y_3769_; 
v___x_3764_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_3765_ = 0;
v___x_3766_ = lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(v___x_3764_, v___x_3765_, v_a_3761_);
v_a_3767_ = lean_ctor_get(v___x_3766_, 0);
lean_inc(v_a_3767_);
lean_dec_ref(v___x_3766_);
if (v_cfg_3750_ == 0)
{
uint8_t v___x_3777_; 
v___x_3777_ = lean_unbox(v_a_3767_);
lean_dec(v_a_3767_);
v___y_3769_ = v___x_3777_;
goto v___jp_3768_;
}
else
{
lean_dec(v_a_3767_);
v___y_3769_ = v_cfg_3750_;
goto v___jp_3768_;
}
v___jp_3768_:
{
lean_object* v___x_3770_; lean_object* v___f_3771_; lean_object* v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_3776_; 
v___x_3770_ = lean_box(v___y_3769_);
lean_inc(v_head_3752_);
v___f_3771_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_push___lam__0___boxed), 10, 3);
lean_closure_set(v___f_3771_, 0, v_head_3752_);
lean_closure_set(v___f_3771_, 1, v___x_3770_);
lean_closure_set(v___f_3771_, 2, v_disch_x3f_3751_);
v___x_3772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_push___closed__0));
v___x_3773_ = lp_mathlib_Mathlib_Tactic_Push_Head_toString(v_head_3752_);
v___x_3774_ = lean_string_append(v___x_3772_, v___x_3773_);
lean_dec_ref(v___x_3773_);
v___x_3775_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3776_ = lp_mathlib_Mathlib_Tactic_transformAtLocation(v___f_3771_, v___x_3774_, v_loc_3753_, v_ifUnchanged_3754_, v___x_3765_, v___x_3775_, v_a_3755_, v_a_3756_, v_a_3757_, v_a_3758_, v_a_3759_, v_a_3760_, v_a_3761_, v_a_3762_);
return v___x_3776_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_push___boxed(lean_object* v_cfg_3778_, lean_object* v_disch_x3f_3779_, lean_object* v_head_3780_, lean_object* v_loc_3781_, lean_object* v_ifUnchanged_3782_, lean_object* v_a_3783_, lean_object* v_a_3784_, lean_object* v_a_3785_, lean_object* v_a_3786_, lean_object* v_a_3787_, lean_object* v_a_3788_, lean_object* v_a_3789_, lean_object* v_a_3790_, lean_object* v_a_3791_){
_start:
{
uint8_t v_cfg_boxed_3792_; uint8_t v_ifUnchanged_boxed_3793_; lean_object* v_res_3794_; 
v_cfg_boxed_3792_ = lean_unbox(v_cfg_3778_);
v_ifUnchanged_boxed_3793_ = lean_unbox(v_ifUnchanged_3782_);
v_res_3794_ = lp_mathlib_Mathlib_Tactic_Push_push(v_cfg_boxed_3792_, v_disch_x3f_3779_, v_head_3780_, v_loc_3781_, v_ifUnchanged_boxed_3793_, v_a_3783_, v_a_3784_, v_a_3785_, v_a_3786_, v_a_3787_, v_a_3788_, v_a_3789_, v_a_3790_);
lean_dec(v_a_3790_);
lean_dec_ref(v_a_3789_);
lean_dec(v_a_3788_);
lean_dec_ref(v_a_3787_);
lean_dec(v_a_3786_);
lean_dec_ref(v_a_3785_);
lean_dec(v_a_3784_);
lean_dec_ref(v_a_3783_);
lean_dec(v_loc_3781_);
return v_res_3794_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5(void){
_start:
{
lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; 
v___x_3807_ = l_Lean_Parser_Tactic_optConfig;
v___x_3808_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__4));
v___x_3809_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_3810_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3810_, 0, v___x_3809_);
lean_ctor_set(v___x_3810_, 1, v___x_3808_);
lean_ctor_set(v___x_3810_, 2, v___x_3807_);
return v___x_3810_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8(void){
_start:
{
lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; 
v___x_3814_ = l_Lean_Parser_Tactic_discharger;
v___x_3815_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__7));
v___x_3816_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3816_, 0, v___x_3815_);
lean_ctor_set(v___x_3816_, 1, v___x_3814_);
return v___x_3816_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9(void){
_start:
{
lean_object* v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3819_; lean_object* v___x_3820_; 
v___x_3817_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8);
v___x_3818_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__5);
v___x_3819_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_3820_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3820_, 0, v___x_3819_);
lean_ctor_set(v___x_3820_, 1, v___x_3818_);
lean_ctor_set(v___x_3820_, 2, v___x_3817_);
return v___x_3820_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20(void){
_start:
{
lean_object* v___x_3844_; lean_object* v___x_3845_; lean_object* v___x_3846_; lean_object* v___x_3847_; 
v___x_3844_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__19));
v___x_3845_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__9);
v___x_3846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_3847_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3847_, 0, v___x_3846_);
lean_ctor_set(v___x_3847_, 1, v___x_3845_);
lean_ctor_set(v___x_3847_, 2, v___x_3844_);
return v___x_3847_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21(void){
_start:
{
lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; 
v___x_3848_ = l_Lean_Parser_Tactic_location;
v___x_3849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__7));
v___x_3850_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3850_, 0, v___x_3849_);
lean_ctor_set(v___x_3850_, 1, v___x_3848_);
return v___x_3850_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22(void){
_start:
{
lean_object* v___x_3851_; lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; 
v___x_3851_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21);
v___x_3852_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20);
v___x_3853_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_3854_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3854_, 0, v___x_3853_);
lean_ctor_set(v___x_3854_, 1, v___x_3852_);
lean_ctor_set(v___x_3854_, 2, v___x_3851_);
return v___x_3854_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23(void){
_start:
{
lean_object* v___x_3855_; lean_object* v___x_3856_; lean_object* v___x_3857_; lean_object* v___x_3858_; 
v___x_3855_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__22);
v___x_3856_ = lean_unsigned_to_nat(1022u);
v___x_3857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1));
v___x_3858_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3858_, 0, v___x_3857_);
lean_ctor_set(v___x_3858_, 1, v___x_3856_);
lean_ctor_set(v___x_3858_, 2, v___x_3855_);
return v___x_3858_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushStx(void){
_start:
{
lean_object* v___x_3859_; 
v___x_3859_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__23);
return v___x_3859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3861_; lean_object* v___x_3862_; 
v___x_3861_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__7___redArg___closed__0);
v___x_3862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3862_, 0, v___x_3861_);
return v___x_3862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg___boxed(lean_object* v___y_3863_){
_start:
{
lean_object* v_res_3864_; 
v_res_3864_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v_res_3864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0(lean_object* v_00_u03b1_3865_, lean_object* v___y_3866_, lean_object* v___y_3867_, lean_object* v___y_3868_, lean_object* v___y_3869_, lean_object* v___y_3870_, lean_object* v___y_3871_, lean_object* v___y_3872_, lean_object* v___y_3873_){
_start:
{
lean_object* v___x_3875_; 
v___x_3875_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_3875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___boxed(lean_object* v_00_u03b1_3876_, lean_object* v___y_3877_, lean_object* v___y_3878_, lean_object* v___y_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_, lean_object* v___y_3885_){
_start:
{
lean_object* v_res_3886_; 
v_res_3886_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0(v_00_u03b1_3876_, v___y_3877_, v___y_3878_, v___y_3879_, v___y_3880_, v___y_3881_, v___y_3882_, v___y_3883_, v___y_3884_);
lean_dec(v___y_3884_);
lean_dec_ref(v___y_3883_);
lean_dec(v___y_3882_);
lean_dec_ref(v___y_3881_);
lean_dec(v___y_3880_);
lean_dec_ref(v___y_3879_);
lean_dec(v___y_3878_);
lean_dec_ref(v___y_3877_);
return v_res_3886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1(lean_object* v_x_3889_, lean_object* v_a_3890_, lean_object* v_a_3891_, lean_object* v_a_3892_, lean_object* v_a_3893_, lean_object* v_a_3894_, lean_object* v_a_3895_, lean_object* v_a_3896_, lean_object* v_a_3897_){
_start:
{
lean_object* v___x_3899_; uint8_t v___x_3900_; 
v___x_3899_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__1));
lean_inc(v_x_3889_);
v___x_3900_ = l_Lean_Syntax_isOfKind(v_x_3889_, v___x_3899_);
if (v___x_3900_ == 0)
{
lean_object* v___x_3901_; 
lean_dec(v_x_3889_);
v___x_3901_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_3901_;
}
else
{
lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v_head_3907_; lean_object* v___y_3909_; lean_object* v___y_3910_; lean_object* v___y_3936_; lean_object* v_a_3937_; lean_object* v___y_3943_; lean_object* v___x_3964_; lean_object* v___x_3965_; lean_object* v___x_3966_; 
v___x_3902_ = lean_unsigned_to_nat(1u);
v___x_3903_ = l_Lean_Syntax_getArg(v_x_3889_, v___x_3902_);
v___x_3904_ = lean_unsigned_to_nat(2u);
v___x_3905_ = l_Lean_Syntax_getArg(v_x_3889_, v___x_3904_);
v___x_3906_ = lean_unsigned_to_nat(3u);
v_head_3907_ = l_Lean_Syntax_getArg(v_x_3889_, v___x_3906_);
v___x_3964_ = lean_unsigned_to_nat(4u);
v___x_3965_ = l_Lean_Syntax_getArg(v_x_3889_, v___x_3964_);
lean_dec(v_x_3889_);
v___x_3966_ = l_Lean_Syntax_getOptional_x3f(v___x_3965_);
lean_dec(v___x_3965_);
if (lean_obj_tag(v___x_3966_) == 0)
{
lean_object* v___x_3967_; 
v___x_3967_ = lean_box(0);
v___y_3943_ = v___x_3967_;
goto v___jp_3942_;
}
else
{
lean_object* v_val_3968_; lean_object* v___x_3970_; uint8_t v_isShared_3971_; uint8_t v_isSharedCheck_3975_; 
v_val_3968_ = lean_ctor_get(v___x_3966_, 0);
v_isSharedCheck_3975_ = !lean_is_exclusive(v___x_3966_);
if (v_isSharedCheck_3975_ == 0)
{
v___x_3970_ = v___x_3966_;
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
else
{
lean_inc(v_val_3968_);
lean_dec(v___x_3966_);
v___x_3970_ = lean_box(0);
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
v_resetjp_3969_:
{
lean_object* v___x_3973_; 
if (v_isShared_3971_ == 0)
{
v___x_3973_ = v___x_3970_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_3974_; 
v_reuseFailAlloc_3974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3974_, 0, v_val_3968_);
v___x_3973_ = v_reuseFailAlloc_3974_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
v___y_3943_ = v___x_3973_;
goto v___jp_3942_;
}
}
}
v___jp_3908_:
{
uint8_t v___x_3911_; lean_object* v___x_3912_; 
v___x_3911_ = 0;
v___x_3912_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_3903_, v___x_3911_, v___x_3900_, v_a_3890_, v_a_3896_, v_a_3897_);
if (lean_obj_tag(v___x_3912_) == 0)
{
lean_object* v_a_3913_; lean_object* v___x_3914_; 
v_a_3913_ = lean_ctor_get(v___x_3912_, 0);
lean_inc(v_a_3913_);
lean_dec_ref_known(v___x_3912_, 1);
v___x_3914_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v_head_3907_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, v_a_3897_);
if (lean_obj_tag(v___x_3914_) == 0)
{
lean_object* v_a_3915_; uint8_t v___x_3916_; uint8_t v___x_3917_; lean_object* v___x_3918_; 
v_a_3915_ = lean_ctor_get(v___x_3914_, 0);
lean_inc(v_a_3915_);
lean_dec_ref_known(v___x_3914_, 1);
v___x_3916_ = 2;
v___x_3917_ = lean_unbox(v_a_3913_);
lean_dec(v_a_3913_);
v___x_3918_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_3917_, v___y_3909_, v_a_3915_, v___y_3910_, v___x_3916_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, v_a_3897_);
lean_dec(v___y_3910_);
return v___x_3918_;
}
else
{
lean_object* v_a_3919_; lean_object* v___x_3921_; uint8_t v_isShared_3922_; uint8_t v_isSharedCheck_3926_; 
lean_dec(v_a_3913_);
lean_dec(v___y_3910_);
lean_dec(v___y_3909_);
v_a_3919_ = lean_ctor_get(v___x_3914_, 0);
v_isSharedCheck_3926_ = !lean_is_exclusive(v___x_3914_);
if (v_isSharedCheck_3926_ == 0)
{
v___x_3921_ = v___x_3914_;
v_isShared_3922_ = v_isSharedCheck_3926_;
goto v_resetjp_3920_;
}
else
{
lean_inc(v_a_3919_);
lean_dec(v___x_3914_);
v___x_3921_ = lean_box(0);
v_isShared_3922_ = v_isSharedCheck_3926_;
goto v_resetjp_3920_;
}
v_resetjp_3920_:
{
lean_object* v___x_3924_; 
if (v_isShared_3922_ == 0)
{
v___x_3924_ = v___x_3921_;
goto v_reusejp_3923_;
}
else
{
lean_object* v_reuseFailAlloc_3925_; 
v_reuseFailAlloc_3925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3925_, 0, v_a_3919_);
v___x_3924_ = v_reuseFailAlloc_3925_;
goto v_reusejp_3923_;
}
v_reusejp_3923_:
{
return v___x_3924_;
}
}
}
}
else
{
lean_object* v_a_3927_; lean_object* v___x_3929_; uint8_t v_isShared_3930_; uint8_t v_isSharedCheck_3934_; 
lean_dec(v___y_3910_);
lean_dec(v___y_3909_);
lean_dec(v_head_3907_);
v_a_3927_ = lean_ctor_get(v___x_3912_, 0);
v_isSharedCheck_3934_ = !lean_is_exclusive(v___x_3912_);
if (v_isSharedCheck_3934_ == 0)
{
v___x_3929_ = v___x_3912_;
v_isShared_3930_ = v_isSharedCheck_3934_;
goto v_resetjp_3928_;
}
else
{
lean_inc(v_a_3927_);
lean_dec(v___x_3912_);
v___x_3929_ = lean_box(0);
v_isShared_3930_ = v_isSharedCheck_3934_;
goto v_resetjp_3928_;
}
v_resetjp_3928_:
{
lean_object* v___x_3932_; 
if (v_isShared_3930_ == 0)
{
v___x_3932_ = v___x_3929_;
goto v_reusejp_3931_;
}
else
{
lean_object* v_reuseFailAlloc_3933_; 
v_reuseFailAlloc_3933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3933_, 0, v_a_3927_);
v___x_3932_ = v_reuseFailAlloc_3933_;
goto v_reusejp_3931_;
}
v_reusejp_3931_:
{
return v___x_3932_;
}
}
}
}
v___jp_3935_:
{
if (lean_obj_tag(v___y_3936_) == 0)
{
lean_object* v___x_3938_; lean_object* v___x_3939_; 
v___x_3938_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0));
v___x_3939_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_3939_, 0, v___x_3938_);
lean_ctor_set_uint8(v___x_3939_, sizeof(void*)*1, v___x_3900_);
v___y_3909_ = v_a_3937_;
v___y_3910_ = v___x_3939_;
goto v___jp_3908_;
}
else
{
lean_object* v_val_3940_; lean_object* v___x_3941_; 
v_val_3940_ = lean_ctor_get(v___y_3936_, 0);
lean_inc(v_val_3940_);
lean_dec_ref_known(v___y_3936_, 1);
v___x_3941_ = l_Lean_Elab_Tactic_expandLocation(v_val_3940_);
lean_dec(v_val_3940_);
v___y_3909_ = v_a_3937_;
v___y_3910_ = v___x_3941_;
goto v___jp_3908_;
}
}
v___jp_3942_:
{
lean_object* v___x_3944_; 
v___x_3944_ = l_Lean_Syntax_getOptional_x3f(v___x_3905_);
lean_dec(v___x_3905_);
if (lean_obj_tag(v___x_3944_) == 0)
{
lean_object* v___x_3945_; 
v___x_3945_ = lean_box(0);
v___y_3936_ = v___y_3943_;
v_a_3937_ = v___x_3945_;
goto v___jp_3935_;
}
else
{
lean_object* v_val_3946_; lean_object* v___x_3948_; uint8_t v_isShared_3949_; uint8_t v_isSharedCheck_3963_; 
v_val_3946_ = lean_ctor_get(v___x_3944_, 0);
v_isSharedCheck_3963_ = !lean_is_exclusive(v___x_3944_);
if (v_isSharedCheck_3963_ == 0)
{
v___x_3948_ = v___x_3944_;
v_isShared_3949_ = v_isSharedCheck_3963_;
goto v_resetjp_3947_;
}
else
{
lean_inc(v_val_3946_);
lean_dec(v___x_3944_);
v___x_3948_ = lean_box(0);
v_isShared_3949_ = v_isSharedCheck_3963_;
goto v_resetjp_3947_;
}
v_resetjp_3947_:
{
lean_object* v___x_3950_; 
v___x_3950_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_val_3946_, v_a_3892_, v_a_3893_, v_a_3896_);
lean_dec(v_val_3946_);
if (lean_obj_tag(v___x_3950_) == 0)
{
lean_object* v_a_3951_; lean_object* v___x_3953_; 
v_a_3951_ = lean_ctor_get(v___x_3950_, 0);
lean_inc(v_a_3951_);
lean_dec_ref_known(v___x_3950_, 1);
if (v_isShared_3949_ == 0)
{
lean_ctor_set(v___x_3948_, 0, v_a_3951_);
v___x_3953_ = v___x_3948_;
goto v_reusejp_3952_;
}
else
{
lean_object* v_reuseFailAlloc_3954_; 
v_reuseFailAlloc_3954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3954_, 0, v_a_3951_);
v___x_3953_ = v_reuseFailAlloc_3954_;
goto v_reusejp_3952_;
}
v_reusejp_3952_:
{
v___y_3936_ = v___y_3943_;
v_a_3937_ = v___x_3953_;
goto v___jp_3935_;
}
}
else
{
lean_object* v_a_3955_; lean_object* v___x_3957_; uint8_t v_isShared_3958_; uint8_t v_isSharedCheck_3962_; 
lean_del_object(v___x_3948_);
lean_dec(v___y_3943_);
lean_dec(v_head_3907_);
lean_dec(v___x_3903_);
v_a_3955_ = lean_ctor_get(v___x_3950_, 0);
v_isSharedCheck_3962_ = !lean_is_exclusive(v___x_3950_);
if (v_isSharedCheck_3962_ == 0)
{
v___x_3957_ = v___x_3950_;
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
else
{
lean_inc(v_a_3955_);
lean_dec(v___x_3950_);
v___x_3957_ = lean_box(0);
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
v_resetjp_3956_:
{
lean_object* v___x_3960_; 
if (v_isShared_3958_ == 0)
{
v___x_3960_ = v___x_3957_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3961_; 
v_reuseFailAlloc_3961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3961_, 0, v_a_3955_);
v___x_3960_ = v_reuseFailAlloc_3961_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
return v___x_3960_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___boxed(lean_object* v_x_3976_, lean_object* v_a_3977_, lean_object* v_a_3978_, lean_object* v_a_3979_, lean_object* v_a_3980_, lean_object* v_a_3981_, lean_object* v_a_3982_, lean_object* v_a_3983_, lean_object* v_a_3984_, lean_object* v_a_3985_){
_start:
{
lean_object* v_res_3986_; 
v_res_3986_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1(v_x_3976_, v_a_3977_, v_a_3978_, v_a_3979_, v_a_3980_, v_a_3981_, v_a_3982_, v_a_3983_, v_a_3984_);
lean_dec(v_a_3984_);
lean_dec_ref(v_a_3983_);
lean_dec(v_a_3982_);
lean_dec_ref(v_a_3981_);
lean_dec(v_a_3980_);
lean_dec_ref(v_a_3979_);
lean_dec(v_a_3978_);
lean_dec_ref(v_a_3977_);
return v_res_3986_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2(void){
_start:
{
lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; 
v___x_3995_ = l_Lean_Parser_Tactic_optConfig;
v___x_3996_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__1));
v___x_3997_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_3998_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3998_, 0, v___x_3997_);
lean_ctor_set(v___x_3998_, 1, v___x_3996_);
lean_ctor_set(v___x_3998_, 2, v___x_3995_);
return v___x_3998_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3(void){
_start:
{
lean_object* v___x_3999_; lean_object* v___x_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; 
v___x_3999_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21);
v___x_4000_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2, &lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2);
v___x_4001_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4002_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4002_, 0, v___x_4001_);
lean_ctor_set(v___x_4002_, 1, v___x_4000_);
lean_ctor_set(v___x_4002_, 2, v___x_3999_);
return v___x_4002_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4(void){
_start:
{
lean_object* v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; 
v___x_4003_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3, &lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__3);
v___x_4004_ = lean_unsigned_to_nat(1022u);
v___x_4005_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0));
v___x_4006_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4006_, 0, v___x_4005_);
lean_ctor_set(v___x_4006_, 1, v___x_4004_);
lean_ctor_set(v___x_4006_, 2, v___x_4003_);
return v___x_4006_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_push__neg(void){
_start:
{
lean_object* v___x_4007_; 
v___x_4007_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4, &lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__4);
return v___x_4007_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0(uint8_t v___y_4014_, uint8_t v_suppressElabErrors_4015_, lean_object* v_x_4016_){
_start:
{
if (lean_obj_tag(v_x_4016_) == 1)
{
lean_object* v_pre_4017_; 
v_pre_4017_ = lean_ctor_get(v_x_4016_, 0);
switch(lean_obj_tag(v_pre_4017_))
{
case 1:
{
lean_object* v_pre_4018_; 
v_pre_4018_ = lean_ctor_get(v_pre_4017_, 0);
switch(lean_obj_tag(v_pre_4018_))
{
case 0:
{
lean_object* v_str_4019_; lean_object* v_str_4020_; lean_object* v___x_4021_; uint8_t v___x_4022_; 
v_str_4019_ = lean_ctor_get(v_x_4016_, 1);
v_str_4020_ = lean_ctor_get(v_pre_4017_, 1);
v___x_4021_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__0));
v___x_4022_ = lean_string_dec_eq(v_str_4020_, v___x_4021_);
if (v___x_4022_ == 0)
{
lean_object* v___x_4023_; uint8_t v___x_4024_; 
v___x_4023_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__6_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_4024_ = lean_string_dec_eq(v_str_4020_, v___x_4023_);
if (v___x_4024_ == 0)
{
return v___y_4014_;
}
else
{
lean_object* v___x_4025_; uint8_t v___x_4026_; 
v___x_4025_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__1));
v___x_4026_ = lean_string_dec_eq(v_str_4019_, v___x_4025_);
if (v___x_4026_ == 0)
{
return v___y_4014_;
}
else
{
return v_suppressElabErrors_4015_;
}
}
}
else
{
lean_object* v___x_4027_; uint8_t v___x_4028_; 
v___x_4027_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__2));
v___x_4028_ = lean_string_dec_eq(v_str_4019_, v___x_4027_);
if (v___x_4028_ == 0)
{
return v___y_4014_;
}
else
{
return v_suppressElabErrors_4015_;
}
}
}
case 1:
{
lean_object* v_pre_4029_; 
v_pre_4029_ = lean_ctor_get(v_pre_4018_, 0);
if (lean_obj_tag(v_pre_4029_) == 0)
{
lean_object* v_str_4030_; lean_object* v_str_4031_; lean_object* v_str_4032_; lean_object* v___x_4033_; uint8_t v___x_4034_; 
v_str_4030_ = lean_ctor_get(v_x_4016_, 1);
v_str_4031_ = lean_ctor_get(v_pre_4017_, 1);
v_str_4032_ = lean_ctor_get(v_pre_4018_, 1);
v___x_4033_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__3));
v___x_4034_ = lean_string_dec_eq(v_str_4032_, v___x_4033_);
if (v___x_4034_ == 0)
{
return v___y_4014_;
}
else
{
lean_object* v___x_4035_; uint8_t v___x_4036_; 
v___x_4035_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__4));
v___x_4036_ = lean_string_dec_eq(v_str_4031_, v___x_4035_);
if (v___x_4036_ == 0)
{
return v___y_4014_;
}
else
{
lean_object* v___x_4037_; uint8_t v___x_4038_; 
v___x_4037_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___closed__5));
v___x_4038_ = lean_string_dec_eq(v_str_4030_, v___x_4037_);
if (v___x_4038_ == 0)
{
return v___y_4014_;
}
else
{
return v_suppressElabErrors_4015_;
}
}
}
}
else
{
return v___y_4014_;
}
}
default: 
{
return v___y_4014_;
}
}
}
case 0:
{
lean_object* v_str_4039_; lean_object* v___x_4040_; uint8_t v___x_4041_; 
v_str_4039_ = lean_ctor_get(v_x_4016_, 1);
v___x_4040_ = ((lean_object*)(lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__4___closed__0));
v___x_4041_ = lean_string_dec_eq(v_str_4039_, v___x_4040_);
if (v___x_4041_ == 0)
{
return v___y_4014_;
}
else
{
return v_suppressElabErrors_4015_;
}
}
default: 
{
return v___y_4014_;
}
}
}
else
{
return v___y_4014_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object* v___y_4042_, lean_object* v_suppressElabErrors_4043_, lean_object* v_x_4044_){
_start:
{
uint8_t v___y_4514__boxed_4045_; uint8_t v_suppressElabErrors_boxed_4046_; uint8_t v_res_4047_; lean_object* v_r_4048_; 
v___y_4514__boxed_4045_ = lean_unbox(v___y_4042_);
v_suppressElabErrors_boxed_4046_ = lean_unbox(v_suppressElabErrors_4043_);
v_res_4047_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0(v___y_4514__boxed_4045_, v_suppressElabErrors_boxed_4046_, v_x_4044_);
lean_dec(v_x_4044_);
v_r_4048_ = lean_box(v_res_4047_);
return v_r_4048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_4049_, lean_object* v_msgData_4050_, uint8_t v_severity_4051_, uint8_t v_isSilent_4052_, lean_object* v___y_4053_, lean_object* v___y_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_){
_start:
{
uint8_t v___y_4059_; lean_object* v___y_4060_; lean_object* v___y_4061_; lean_object* v___y_4062_; lean_object* v___y_4063_; lean_object* v___y_4064_; uint8_t v___y_4065_; lean_object* v___y_4066_; lean_object* v___y_4067_; lean_object* v___y_4095_; uint8_t v___y_4096_; lean_object* v___y_4097_; lean_object* v___y_4098_; uint8_t v___y_4099_; uint8_t v___y_4100_; lean_object* v___y_4101_; lean_object* v___y_4102_; lean_object* v___y_4120_; uint8_t v___y_4121_; lean_object* v___y_4122_; lean_object* v___y_4123_; uint8_t v___y_4124_; uint8_t v___y_4125_; lean_object* v___y_4126_; lean_object* v___y_4127_; lean_object* v___y_4131_; lean_object* v___y_4132_; lean_object* v___y_4133_; lean_object* v___y_4134_; uint8_t v___y_4135_; uint8_t v___y_4136_; uint8_t v___y_4137_; uint8_t v___x_4142_; lean_object* v___y_4144_; lean_object* v___y_4145_; lean_object* v___y_4146_; uint8_t v___y_4147_; lean_object* v___y_4148_; uint8_t v___y_4149_; uint8_t v___y_4150_; uint8_t v___y_4152_; uint8_t v___x_4167_; 
v___x_4142_ = 2;
v___x_4167_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4051_, v___x_4142_);
if (v___x_4167_ == 0)
{
v___y_4152_ = v___x_4167_;
goto v___jp_4151_;
}
else
{
uint8_t v___x_4168_; 
lean_inc_ref(v_msgData_4050_);
v___x_4168_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_4050_);
v___y_4152_ = v___x_4168_;
goto v___jp_4151_;
}
v___jp_4058_:
{
lean_object* v___x_4068_; lean_object* v_currNamespace_4069_; lean_object* v_openDecls_4070_; lean_object* v_env_4071_; lean_object* v_nextMacroScope_4072_; lean_object* v_ngen_4073_; lean_object* v_auxDeclNGen_4074_; lean_object* v_traceState_4075_; lean_object* v_cache_4076_; lean_object* v_messages_4077_; lean_object* v_infoState_4078_; lean_object* v_snapshotTasks_4079_; lean_object* v___x_4081_; uint8_t v_isShared_4082_; uint8_t v_isSharedCheck_4093_; 
v___x_4068_ = lean_st_ref_take(v___y_4067_);
v_currNamespace_4069_ = lean_ctor_get(v___y_4066_, 6);
v_openDecls_4070_ = lean_ctor_get(v___y_4066_, 7);
v_env_4071_ = lean_ctor_get(v___x_4068_, 0);
v_nextMacroScope_4072_ = lean_ctor_get(v___x_4068_, 1);
v_ngen_4073_ = lean_ctor_get(v___x_4068_, 2);
v_auxDeclNGen_4074_ = lean_ctor_get(v___x_4068_, 3);
v_traceState_4075_ = lean_ctor_get(v___x_4068_, 4);
v_cache_4076_ = lean_ctor_get(v___x_4068_, 5);
v_messages_4077_ = lean_ctor_get(v___x_4068_, 6);
v_infoState_4078_ = lean_ctor_get(v___x_4068_, 7);
v_snapshotTasks_4079_ = lean_ctor_get(v___x_4068_, 8);
v_isSharedCheck_4093_ = !lean_is_exclusive(v___x_4068_);
if (v_isSharedCheck_4093_ == 0)
{
v___x_4081_ = v___x_4068_;
v_isShared_4082_ = v_isSharedCheck_4093_;
goto v_resetjp_4080_;
}
else
{
lean_inc(v_snapshotTasks_4079_);
lean_inc(v_infoState_4078_);
lean_inc(v_messages_4077_);
lean_inc(v_cache_4076_);
lean_inc(v_traceState_4075_);
lean_inc(v_auxDeclNGen_4074_);
lean_inc(v_ngen_4073_);
lean_inc(v_nextMacroScope_4072_);
lean_inc(v_env_4071_);
lean_dec(v___x_4068_);
v___x_4081_ = lean_box(0);
v_isShared_4082_ = v_isSharedCheck_4093_;
goto v_resetjp_4080_;
}
v_resetjp_4080_:
{
lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; lean_object* v___x_4088_; 
lean_inc(v_openDecls_4070_);
lean_inc(v_currNamespace_4069_);
v___x_4083_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4083_, 0, v_currNamespace_4069_);
lean_ctor_set(v___x_4083_, 1, v_openDecls_4070_);
v___x_4084_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_4084_, 0, v___x_4083_);
lean_ctor_set(v___x_4084_, 1, v___y_4063_);
lean_inc_ref(v___y_4060_);
lean_inc_ref(v___y_4062_);
v___x_4085_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_4085_, 0, v___y_4062_);
lean_ctor_set(v___x_4085_, 1, v___y_4061_);
lean_ctor_set(v___x_4085_, 2, v___y_4064_);
lean_ctor_set(v___x_4085_, 3, v___y_4060_);
lean_ctor_set(v___x_4085_, 4, v___x_4084_);
lean_ctor_set_uint8(v___x_4085_, sizeof(void*)*5, v___y_4065_);
lean_ctor_set_uint8(v___x_4085_, sizeof(void*)*5 + 1, v___y_4059_);
lean_ctor_set_uint8(v___x_4085_, sizeof(void*)*5 + 2, v_isSilent_4052_);
v___x_4086_ = l_Lean_MessageLog_add(v___x_4085_, v_messages_4077_);
if (v_isShared_4082_ == 0)
{
lean_ctor_set(v___x_4081_, 6, v___x_4086_);
v___x_4088_ = v___x_4081_;
goto v_reusejp_4087_;
}
else
{
lean_object* v_reuseFailAlloc_4092_; 
v_reuseFailAlloc_4092_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4092_, 0, v_env_4071_);
lean_ctor_set(v_reuseFailAlloc_4092_, 1, v_nextMacroScope_4072_);
lean_ctor_set(v_reuseFailAlloc_4092_, 2, v_ngen_4073_);
lean_ctor_set(v_reuseFailAlloc_4092_, 3, v_auxDeclNGen_4074_);
lean_ctor_set(v_reuseFailAlloc_4092_, 4, v_traceState_4075_);
lean_ctor_set(v_reuseFailAlloc_4092_, 5, v_cache_4076_);
lean_ctor_set(v_reuseFailAlloc_4092_, 6, v___x_4086_);
lean_ctor_set(v_reuseFailAlloc_4092_, 7, v_infoState_4078_);
lean_ctor_set(v_reuseFailAlloc_4092_, 8, v_snapshotTasks_4079_);
v___x_4088_ = v_reuseFailAlloc_4092_;
goto v_reusejp_4087_;
}
v_reusejp_4087_:
{
lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v___x_4091_; 
v___x_4089_ = lean_st_ref_set(v___y_4067_, v___x_4088_);
v___x_4090_ = lean_box(0);
v___x_4091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4091_, 0, v___x_4090_);
return v___x_4091_;
}
}
}
v___jp_4094_:
{
lean_object* v___x_4103_; lean_object* v___x_4104_; lean_object* v_a_4105_; lean_object* v___x_4107_; uint8_t v_isShared_4108_; uint8_t v_isSharedCheck_4118_; 
v___x_4103_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_4050_);
v___x_4104_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v___x_4103_, v___y_4053_, v___y_4054_, v___y_4055_, v___y_4056_);
v_a_4105_ = lean_ctor_get(v___x_4104_, 0);
v_isSharedCheck_4118_ = !lean_is_exclusive(v___x_4104_);
if (v_isSharedCheck_4118_ == 0)
{
v___x_4107_ = v___x_4104_;
v_isShared_4108_ = v_isSharedCheck_4118_;
goto v_resetjp_4106_;
}
else
{
lean_inc(v_a_4105_);
lean_dec(v___x_4104_);
v___x_4107_ = lean_box(0);
v_isShared_4108_ = v_isSharedCheck_4118_;
goto v_resetjp_4106_;
}
v_resetjp_4106_:
{
lean_object* v___x_4109_; lean_object* v___x_4110_; lean_object* v___x_4111_; lean_object* v___x_4112_; 
lean_inc_ref_n(v___y_4097_, 2);
v___x_4109_ = l_Lean_FileMap_toPosition(v___y_4097_, v___y_4101_);
lean_dec(v___y_4101_);
v___x_4110_ = l_Lean_FileMap_toPosition(v___y_4097_, v___y_4102_);
lean_dec(v___y_4102_);
v___x_4111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4111_, 0, v___x_4110_);
v___x_4112_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
if (v___y_4099_ == 0)
{
lean_del_object(v___x_4107_);
lean_dec_ref(v___y_4095_);
v___y_4059_ = v___y_4096_;
v___y_4060_ = v___x_4112_;
v___y_4061_ = v___x_4109_;
v___y_4062_ = v___y_4098_;
v___y_4063_ = v_a_4105_;
v___y_4064_ = v___x_4111_;
v___y_4065_ = v___y_4100_;
v___y_4066_ = v___y_4055_;
v___y_4067_ = v___y_4056_;
goto v___jp_4058_;
}
else
{
uint8_t v___x_4113_; 
lean_inc(v_a_4105_);
v___x_4113_ = l_Lean_MessageData_hasTag(v___y_4095_, v_a_4105_);
if (v___x_4113_ == 0)
{
lean_object* v___x_4114_; lean_object* v___x_4116_; 
lean_dec_ref_known(v___x_4111_, 1);
lean_dec_ref(v___x_4109_);
lean_dec(v_a_4105_);
v___x_4114_ = lean_box(0);
if (v_isShared_4108_ == 0)
{
lean_ctor_set(v___x_4107_, 0, v___x_4114_);
v___x_4116_ = v___x_4107_;
goto v_reusejp_4115_;
}
else
{
lean_object* v_reuseFailAlloc_4117_; 
v_reuseFailAlloc_4117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4117_, 0, v___x_4114_);
v___x_4116_ = v_reuseFailAlloc_4117_;
goto v_reusejp_4115_;
}
v_reusejp_4115_:
{
return v___x_4116_;
}
}
else
{
lean_del_object(v___x_4107_);
v___y_4059_ = v___y_4096_;
v___y_4060_ = v___x_4112_;
v___y_4061_ = v___x_4109_;
v___y_4062_ = v___y_4098_;
v___y_4063_ = v_a_4105_;
v___y_4064_ = v___x_4111_;
v___y_4065_ = v___y_4100_;
v___y_4066_ = v___y_4055_;
v___y_4067_ = v___y_4056_;
goto v___jp_4058_;
}
}
}
}
v___jp_4119_:
{
lean_object* v___x_4128_; 
v___x_4128_ = l_Lean_Syntax_getTailPos_x3f(v___y_4126_, v___y_4124_);
lean_dec(v___y_4126_);
if (lean_obj_tag(v___x_4128_) == 0)
{
lean_inc(v___y_4127_);
v___y_4095_ = v___y_4120_;
v___y_4096_ = v___y_4121_;
v___y_4097_ = v___y_4122_;
v___y_4098_ = v___y_4123_;
v___y_4099_ = v___y_4125_;
v___y_4100_ = v___y_4124_;
v___y_4101_ = v___y_4127_;
v___y_4102_ = v___y_4127_;
goto v___jp_4094_;
}
else
{
lean_object* v_val_4129_; 
v_val_4129_ = lean_ctor_get(v___x_4128_, 0);
lean_inc(v_val_4129_);
lean_dec_ref_known(v___x_4128_, 1);
v___y_4095_ = v___y_4120_;
v___y_4096_ = v___y_4121_;
v___y_4097_ = v___y_4122_;
v___y_4098_ = v___y_4123_;
v___y_4099_ = v___y_4125_;
v___y_4100_ = v___y_4124_;
v___y_4101_ = v___y_4127_;
v___y_4102_ = v_val_4129_;
goto v___jp_4094_;
}
}
v___jp_4130_:
{
lean_object* v_ref_4138_; lean_object* v___x_4139_; 
v_ref_4138_ = l_Lean_replaceRef(v_ref_4049_, v___y_4134_);
v___x_4139_ = l_Lean_Syntax_getPos_x3f(v_ref_4138_, v___y_4136_);
if (lean_obj_tag(v___x_4139_) == 0)
{
lean_object* v___x_4140_; 
v___x_4140_ = lean_unsigned_to_nat(0u);
v___y_4120_ = v___y_4131_;
v___y_4121_ = v___y_4137_;
v___y_4122_ = v___y_4132_;
v___y_4123_ = v___y_4133_;
v___y_4124_ = v___y_4136_;
v___y_4125_ = v___y_4135_;
v___y_4126_ = v_ref_4138_;
v___y_4127_ = v___x_4140_;
goto v___jp_4119_;
}
else
{
lean_object* v_val_4141_; 
v_val_4141_ = lean_ctor_get(v___x_4139_, 0);
lean_inc(v_val_4141_);
lean_dec_ref_known(v___x_4139_, 1);
v___y_4120_ = v___y_4131_;
v___y_4121_ = v___y_4137_;
v___y_4122_ = v___y_4132_;
v___y_4123_ = v___y_4133_;
v___y_4124_ = v___y_4136_;
v___y_4125_ = v___y_4135_;
v___y_4126_ = v_ref_4138_;
v___y_4127_ = v_val_4141_;
goto v___jp_4119_;
}
}
v___jp_4143_:
{
if (v___y_4150_ == 0)
{
v___y_4131_ = v___y_4148_;
v___y_4132_ = v___y_4144_;
v___y_4133_ = v___y_4145_;
v___y_4134_ = v___y_4146_;
v___y_4135_ = v___y_4147_;
v___y_4136_ = v___y_4149_;
v___y_4137_ = v_severity_4051_;
goto v___jp_4130_;
}
else
{
v___y_4131_ = v___y_4148_;
v___y_4132_ = v___y_4144_;
v___y_4133_ = v___y_4145_;
v___y_4134_ = v___y_4146_;
v___y_4135_ = v___y_4147_;
v___y_4136_ = v___y_4149_;
v___y_4137_ = v___x_4142_;
goto v___jp_4130_;
}
}
v___jp_4151_:
{
if (v___y_4152_ == 0)
{
lean_object* v_fileName_4153_; lean_object* v_fileMap_4154_; lean_object* v_options_4155_; lean_object* v_ref_4156_; uint8_t v_suppressElabErrors_4157_; lean_object* v___x_4158_; lean_object* v___x_4159_; lean_object* v___f_4160_; uint8_t v___x_4161_; uint8_t v___x_4162_; 
v_fileName_4153_ = lean_ctor_get(v___y_4055_, 0);
v_fileMap_4154_ = lean_ctor_get(v___y_4055_, 1);
v_options_4155_ = lean_ctor_get(v___y_4055_, 2);
v_ref_4156_ = lean_ctor_get(v___y_4055_, 5);
v_suppressElabErrors_4157_ = lean_ctor_get_uint8(v___y_4055_, sizeof(void*)*14 + 1);
v___x_4158_ = lean_box(v___y_4152_);
v___x_4159_ = lean_box(v_suppressElabErrors_4157_);
v___f_4160_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_4160_, 0, v___x_4158_);
lean_closure_set(v___f_4160_, 1, v___x_4159_);
v___x_4161_ = 1;
v___x_4162_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4051_, v___x_4161_);
if (v___x_4162_ == 0)
{
v___y_4144_ = v_fileMap_4154_;
v___y_4145_ = v_fileName_4153_;
v___y_4146_ = v_ref_4156_;
v___y_4147_ = v_suppressElabErrors_4157_;
v___y_4148_ = v___f_4160_;
v___y_4149_ = v___y_4152_;
v___y_4150_ = v___x_4162_;
goto v___jp_4143_;
}
else
{
lean_object* v___x_4163_; uint8_t v___x_4164_; 
v___x_4163_ = l_Lean_warningAsError;
v___x_4164_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_options_4155_, v___x_4163_);
v___y_4144_ = v_fileMap_4154_;
v___y_4145_ = v_fileName_4153_;
v___y_4146_ = v_ref_4156_;
v___y_4147_ = v_suppressElabErrors_4157_;
v___y_4148_ = v___f_4160_;
v___y_4149_ = v___y_4152_;
v___y_4150_ = v___x_4164_;
goto v___jp_4143_;
}
}
else
{
lean_object* v___x_4165_; lean_object* v___x_4166_; 
lean_dec_ref(v_msgData_4050_);
v___x_4165_ = lean_box(0);
v___x_4166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4166_, 0, v___x_4165_);
return v___x_4166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_4169_, lean_object* v_msgData_4170_, lean_object* v_severity_4171_, lean_object* v_isSilent_4172_, lean_object* v___y_4173_, lean_object* v___y_4174_, lean_object* v___y_4175_, lean_object* v___y_4176_, lean_object* v___y_4177_){
_start:
{
uint8_t v_severity_boxed_4178_; uint8_t v_isSilent_boxed_4179_; lean_object* v_res_4180_; 
v_severity_boxed_4178_ = lean_unbox(v_severity_4171_);
v_isSilent_boxed_4179_ = lean_unbox(v_isSilent_4172_);
v_res_4180_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg(v_ref_4169_, v_msgData_4170_, v_severity_boxed_4178_, v_isSilent_boxed_4179_, v___y_4173_, v___y_4174_, v___y_4175_, v___y_4176_);
lean_dec(v___y_4176_);
lean_dec_ref(v___y_4175_);
lean_dec(v___y_4174_);
lean_dec_ref(v___y_4173_);
lean_dec(v_ref_4169_);
return v_res_4180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0(lean_object* v_msgData_4181_, uint8_t v_severity_4182_, uint8_t v_isSilent_4183_, lean_object* v___y_4184_, lean_object* v___y_4185_, lean_object* v___y_4186_, lean_object* v___y_4187_, lean_object* v___y_4188_, lean_object* v___y_4189_, lean_object* v___y_4190_, lean_object* v___y_4191_){
_start:
{
lean_object* v_ref_4193_; lean_object* v___x_4194_; 
v_ref_4193_ = lean_ctor_get(v___y_4190_, 5);
v___x_4194_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg(v_ref_4193_, v_msgData_4181_, v_severity_4182_, v_isSilent_4183_, v___y_4188_, v___y_4189_, v___y_4190_, v___y_4191_);
return v___x_4194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0___boxed(lean_object* v_msgData_4195_, lean_object* v_severity_4196_, lean_object* v_isSilent_4197_, lean_object* v___y_4198_, lean_object* v___y_4199_, lean_object* v___y_4200_, lean_object* v___y_4201_, lean_object* v___y_4202_, lean_object* v___y_4203_, lean_object* v___y_4204_, lean_object* v___y_4205_, lean_object* v___y_4206_){
_start:
{
uint8_t v_severity_boxed_4207_; uint8_t v_isSilent_boxed_4208_; lean_object* v_res_4209_; 
v_severity_boxed_4207_ = lean_unbox(v_severity_4196_);
v_isSilent_boxed_4208_ = lean_unbox(v_isSilent_4197_);
v_res_4209_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0(v_msgData_4195_, v_severity_boxed_4207_, v_isSilent_boxed_4208_, v___y_4198_, v___y_4199_, v___y_4200_, v___y_4201_, v___y_4202_, v___y_4203_, v___y_4204_, v___y_4205_);
lean_dec(v___y_4205_);
lean_dec_ref(v___y_4204_);
lean_dec(v___y_4203_);
lean_dec_ref(v___y_4202_);
lean_dec(v___y_4201_);
lean_dec_ref(v___y_4200_);
lean_dec(v___y_4199_);
lean_dec_ref(v___y_4198_);
return v_res_4209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0(lean_object* v_msgData_4210_, lean_object* v___y_4211_, lean_object* v___y_4212_, lean_object* v___y_4213_, lean_object* v___y_4214_, lean_object* v___y_4215_, lean_object* v___y_4216_, lean_object* v___y_4217_, lean_object* v___y_4218_){
_start:
{
uint8_t v___x_4220_; uint8_t v___x_4221_; lean_object* v___x_4222_; 
v___x_4220_ = 1;
v___x_4221_ = 0;
v___x_4222_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0(v_msgData_4210_, v___x_4220_, v___x_4221_, v___y_4211_, v___y_4212_, v___y_4213_, v___y_4214_, v___y_4215_, v___y_4216_, v___y_4217_, v___y_4218_);
return v___x_4222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0___boxed(lean_object* v_msgData_4223_, lean_object* v___y_4224_, lean_object* v___y_4225_, lean_object* v___y_4226_, lean_object* v___y_4227_, lean_object* v___y_4228_, lean_object* v___y_4229_, lean_object* v___y_4230_, lean_object* v___y_4231_, lean_object* v___y_4232_){
_start:
{
lean_object* v_res_4233_; 
v_res_4233_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0(v_msgData_4223_, v___y_4224_, v___y_4225_, v___y_4226_, v___y_4227_, v___y_4228_, v___y_4229_, v___y_4230_, v___y_4231_);
lean_dec(v___y_4231_);
lean_dec_ref(v___y_4230_);
lean_dec(v___y_4229_);
lean_dec_ref(v___y_4228_);
lean_dec(v___y_4227_);
lean_dec_ref(v___y_4226_);
lean_dec(v___y_4225_);
lean_dec_ref(v___y_4224_);
return v_res_4233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3(void){
_start:
{
lean_object* v___x_4239_; lean_object* v___x_4240_; 
v___x_4239_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__2));
v___x_4240_ = l_Lean_MessageData_ofFormat(v___x_4239_);
return v___x_4240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1(lean_object* v_x_4241_, lean_object* v_a_4242_, lean_object* v_a_4243_, lean_object* v_a_4244_, lean_object* v_a_4245_, lean_object* v_a_4246_, lean_object* v_a_4247_, lean_object* v_a_4248_, lean_object* v_a_4249_){
_start:
{
lean_object* v___x_4251_; uint8_t v___x_4252_; 
v___x_4251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__0));
lean_inc(v_x_4241_);
v___x_4252_ = l_Lean_Syntax_isOfKind(v_x_4241_, v___x_4251_);
if (v___x_4252_ == 0)
{
lean_object* v___x_4253_; 
lean_dec(v_x_4241_);
v___x_4253_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_4253_;
}
else
{
lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___y_4257_; lean_object* v___y_4275_; lean_object* v___x_4282_; lean_object* v___x_4283_; lean_object* v___x_4284_; 
v___x_4254_ = lean_unsigned_to_nat(1u);
v___x_4255_ = l_Lean_Syntax_getArg(v_x_4241_, v___x_4254_);
v___x_4282_ = lean_unsigned_to_nat(2u);
v___x_4283_ = l_Lean_Syntax_getArg(v_x_4241_, v___x_4282_);
lean_dec(v_x_4241_);
v___x_4284_ = l_Lean_Syntax_getOptional_x3f(v___x_4283_);
lean_dec(v___x_4283_);
if (lean_obj_tag(v___x_4284_) == 0)
{
lean_object* v___x_4285_; 
v___x_4285_ = lean_box(0);
v___y_4275_ = v___x_4285_;
goto v___jp_4274_;
}
else
{
lean_object* v_val_4286_; lean_object* v___x_4288_; uint8_t v_isShared_4289_; uint8_t v_isSharedCheck_4293_; 
v_val_4286_ = lean_ctor_get(v___x_4284_, 0);
v_isSharedCheck_4293_ = !lean_is_exclusive(v___x_4284_);
if (v_isSharedCheck_4293_ == 0)
{
v___x_4288_ = v___x_4284_;
v_isShared_4289_ = v_isSharedCheck_4293_;
goto v_resetjp_4287_;
}
else
{
lean_inc(v_val_4286_);
lean_dec(v___x_4284_);
v___x_4288_ = lean_box(0);
v_isShared_4289_ = v_isSharedCheck_4293_;
goto v_resetjp_4287_;
}
v_resetjp_4287_:
{
lean_object* v___x_4291_; 
if (v_isShared_4289_ == 0)
{
v___x_4291_ = v___x_4288_;
goto v_reusejp_4290_;
}
else
{
lean_object* v_reuseFailAlloc_4292_; 
v_reuseFailAlloc_4292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4292_, 0, v_val_4286_);
v___x_4291_ = v_reuseFailAlloc_4292_;
goto v_reusejp_4290_;
}
v_reusejp_4290_:
{
v___y_4275_ = v___x_4291_;
goto v___jp_4274_;
}
}
}
v___jp_4256_:
{
uint8_t v___x_4258_; lean_object* v___x_4259_; 
v___x_4258_ = 0;
v___x_4259_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_4255_, v___x_4258_, v___x_4252_, v_a_4242_, v_a_4248_, v_a_4249_);
if (lean_obj_tag(v___x_4259_) == 0)
{
lean_object* v_a_4260_; lean_object* v___x_4261_; lean_object* v___x_4262_; uint8_t v___x_4263_; uint8_t v___x_4264_; lean_object* v___x_4265_; 
v_a_4260_ = lean_ctor_get(v___x_4259_, 0);
lean_inc(v_a_4260_);
lean_dec_ref_known(v___x_4259_, 1);
v___x_4261_ = lean_box(0);
v___x_4262_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__0));
v___x_4263_ = 2;
v___x_4264_ = lean_unbox(v_a_4260_);
lean_dec(v_a_4260_);
v___x_4265_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_4264_, v___x_4261_, v___x_4262_, v___y_4257_, v___x_4263_, v_a_4242_, v_a_4243_, v_a_4244_, v_a_4245_, v_a_4246_, v_a_4247_, v_a_4248_, v_a_4249_);
lean_dec(v___y_4257_);
return v___x_4265_;
}
else
{
lean_object* v_a_4266_; lean_object* v___x_4268_; uint8_t v_isShared_4269_; uint8_t v_isSharedCheck_4273_; 
lean_dec(v___y_4257_);
v_a_4266_ = lean_ctor_get(v___x_4259_, 0);
v_isSharedCheck_4273_ = !lean_is_exclusive(v___x_4259_);
if (v_isSharedCheck_4273_ == 0)
{
v___x_4268_ = v___x_4259_;
v_isShared_4269_ = v_isSharedCheck_4273_;
goto v_resetjp_4267_;
}
else
{
lean_inc(v_a_4266_);
lean_dec(v___x_4259_);
v___x_4268_ = lean_box(0);
v_isShared_4269_ = v_isSharedCheck_4273_;
goto v_resetjp_4267_;
}
v_resetjp_4267_:
{
lean_object* v___x_4271_; 
if (v_isShared_4269_ == 0)
{
v___x_4271_ = v___x_4268_;
goto v_reusejp_4270_;
}
else
{
lean_object* v_reuseFailAlloc_4272_; 
v_reuseFailAlloc_4272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4272_, 0, v_a_4266_);
v___x_4271_ = v_reuseFailAlloc_4272_;
goto v_reusejp_4270_;
}
v_reusejp_4270_:
{
return v___x_4271_;
}
}
}
}
v___jp_4274_:
{
lean_object* v___x_4276_; lean_object* v___x_4277_; 
v___x_4276_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___closed__3);
v___x_4277_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0(v___x_4276_, v_a_4242_, v_a_4243_, v_a_4244_, v_a_4245_, v_a_4246_, v_a_4247_, v_a_4248_, v_a_4249_);
if (lean_obj_tag(v___x_4277_) == 0)
{
lean_dec_ref_known(v___x_4277_, 1);
if (lean_obj_tag(v___y_4275_) == 0)
{
lean_object* v___x_4278_; lean_object* v___x_4279_; 
v___x_4278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0));
v___x_4279_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_4279_, 0, v___x_4278_);
lean_ctor_set_uint8(v___x_4279_, sizeof(void*)*1, v___x_4252_);
v___y_4257_ = v___x_4279_;
goto v___jp_4256_;
}
else
{
lean_object* v_val_4280_; lean_object* v___x_4281_; 
v_val_4280_ = lean_ctor_get(v___y_4275_, 0);
lean_inc(v_val_4280_);
lean_dec_ref_known(v___y_4275_, 1);
v___x_4281_ = l_Lean_Elab_Tactic_expandLocation(v_val_4280_);
lean_dec(v_val_4280_);
v___y_4257_ = v___x_4281_;
goto v___jp_4256_;
}
}
else
{
lean_dec(v___y_4275_);
lean_dec(v___x_4255_);
return v___x_4277_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1___boxed(lean_object* v_x_4294_, lean_object* v_a_4295_, lean_object* v_a_4296_, lean_object* v_a_4297_, lean_object* v_a_4298_, lean_object* v_a_4299_, lean_object* v_a_4300_, lean_object* v_a_4301_, lean_object* v_a_4302_, lean_object* v_a_4303_){
_start:
{
lean_object* v_res_4304_; 
v_res_4304_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1(v_x_4294_, v_a_4295_, v_a_4296_, v_a_4297_, v_a_4298_, v_a_4299_, v_a_4300_, v_a_4301_, v_a_4302_);
lean_dec(v_a_4302_);
lean_dec_ref(v_a_4301_);
lean_dec(v_a_4300_);
lean_dec_ref(v_a_4299_);
lean_dec(v_a_4298_);
lean_dec_ref(v_a_4297_);
lean_dec(v_a_4296_);
lean_dec_ref(v_a_4295_);
return v_res_4304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1(lean_object* v_ref_4305_, lean_object* v_msgData_4306_, uint8_t v_severity_4307_, uint8_t v_isSilent_4308_, lean_object* v___y_4309_, lean_object* v___y_4310_, lean_object* v___y_4311_, lean_object* v___y_4312_, lean_object* v___y_4313_, lean_object* v___y_4314_, lean_object* v___y_4315_, lean_object* v___y_4316_){
_start:
{
lean_object* v___x_4318_; 
v___x_4318_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg(v_ref_4305_, v_msgData_4306_, v_severity_4307_, v_isSilent_4308_, v___y_4313_, v___y_4314_, v___y_4315_, v___y_4316_);
return v___x_4318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___boxed(lean_object* v_ref_4319_, lean_object* v_msgData_4320_, lean_object* v_severity_4321_, lean_object* v_isSilent_4322_, lean_object* v___y_4323_, lean_object* v___y_4324_, lean_object* v___y_4325_, lean_object* v___y_4326_, lean_object* v___y_4327_, lean_object* v___y_4328_, lean_object* v___y_4329_, lean_object* v___y_4330_, lean_object* v___y_4331_){
_start:
{
uint8_t v_severity_boxed_4332_; uint8_t v_isSilent_boxed_4333_; lean_object* v_res_4334_; 
v_severity_boxed_4332_ = lean_unbox(v_severity_4321_);
v_isSilent_boxed_4333_ = lean_unbox(v_isSilent_4322_);
v_res_4334_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1(v_ref_4319_, v_msgData_4320_, v_severity_boxed_4332_, v_isSilent_boxed_4333_, v___y_4323_, v___y_4324_, v___y_4325_, v___y_4326_, v___y_4327_, v___y_4328_, v___y_4329_, v___y_4330_);
lean_dec(v___y_4330_);
lean_dec_ref(v___y_4329_);
lean_dec(v___y_4328_);
lean_dec_ref(v___y_4327_);
lean_dec(v___y_4326_);
lean_dec_ref(v___y_4325_);
lean_dec(v___y_4324_);
lean_dec_ref(v___y_4323_);
lean_dec(v_ref_4319_);
return v_res_4334_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__3(void){
_start:
{
lean_object* v___x_4344_; lean_object* v___x_4345_; lean_object* v___x_4346_; lean_object* v___x_4347_; 
v___x_4344_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8);
v___x_4345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pull___closed__2));
v___x_4346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4347_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4347_, 0, v___x_4346_);
lean_ctor_set(v___x_4347_, 1, v___x_4345_);
lean_ctor_set(v___x_4347_, 2, v___x_4344_);
return v___x_4347_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__4(void){
_start:
{
lean_object* v___x_4348_; lean_object* v___x_4349_; lean_object* v___x_4350_; lean_object* v___x_4351_; 
v___x_4348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__19));
v___x_4349_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pull___closed__3, &lp_mathlib_Mathlib_Tactic_Push_pull___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__3);
v___x_4350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4351_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4351_, 0, v___x_4350_);
lean_ctor_set(v___x_4351_, 1, v___x_4349_);
lean_ctor_set(v___x_4351_, 2, v___x_4348_);
return v___x_4351_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__5(void){
_start:
{
lean_object* v___x_4352_; lean_object* v___x_4353_; lean_object* v___x_4354_; lean_object* v___x_4355_; 
v___x_4352_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__21);
v___x_4353_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pull___closed__4, &lp_mathlib_Mathlib_Tactic_Push_pull___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__4);
v___x_4354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4355_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4355_, 0, v___x_4354_);
lean_ctor_set(v___x_4355_, 1, v___x_4353_);
lean_ctor_set(v___x_4355_, 2, v___x_4352_);
return v___x_4355_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__6(void){
_start:
{
lean_object* v___x_4356_; lean_object* v___x_4357_; lean_object* v___x_4358_; lean_object* v___x_4359_; 
v___x_4356_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pull___closed__5, &lp_mathlib_Mathlib_Tactic_Push_pull___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__5);
v___x_4357_ = lean_unsigned_to_nat(1022u);
v___x_4358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pull___closed__1));
v___x_4359_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4359_, 0, v___x_4358_);
lean_ctor_set(v___x_4359_, 1, v___x_4357_);
lean_ctor_set(v___x_4359_, 2, v___x_4356_);
return v___x_4359_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pull(void){
_start:
{
lean_object* v___x_4360_; 
v___x_4360_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pull___closed__6, &lp_mathlib_Mathlib_Tactic_Push_pull___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__6);
return v___x_4360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0(lean_object* v_a_4361_, lean_object* v_a_4362_, lean_object* v_x_4363_, lean_object* v___y_4364_, lean_object* v___y_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_, lean_object* v___y_4368_){
_start:
{
lean_object* v___x_4370_; 
v___x_4370_ = lp_mathlib_Mathlib_Tactic_Push_pullCore(v_a_4361_, v_x_4363_, v_a_4362_, v___y_4365_, v___y_4366_, v___y_4367_, v___y_4368_);
return v___x_4370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0___boxed(lean_object* v_a_4371_, lean_object* v_a_4372_, lean_object* v_x_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_){
_start:
{
lean_object* v_res_4380_; 
v_res_4380_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0(v_a_4371_, v_a_4372_, v_x_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
lean_dec(v___y_4378_);
lean_dec_ref(v___y_4377_);
lean_dec(v___y_4376_);
lean_dec_ref(v___y_4375_);
lean_dec_ref(v___y_4374_);
lean_dec(v_a_4372_);
return v_res_4380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1(lean_object* v_x_4381_, lean_object* v_a_4382_, lean_object* v_a_4383_, lean_object* v_a_4384_, lean_object* v_a_4385_, lean_object* v_a_4386_, lean_object* v_a_4387_, lean_object* v_a_4388_, lean_object* v_a_4389_){
_start:
{
lean_object* v___x_4391_; lean_object* v___y_4393_; lean_object* v___y_4394_; lean_object* v___x_4399_; uint8_t v___x_4400_; 
v___x_4391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pull___closed__0));
v___x_4399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pull___closed__1));
lean_inc(v_x_4381_);
v___x_4400_ = l_Lean_Syntax_isOfKind(v_x_4381_, v___x_4399_);
if (v___x_4400_ == 0)
{
lean_object* v___x_4401_; 
lean_dec(v_x_4381_);
v___x_4401_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_4401_;
}
else
{
lean_object* v___x_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; lean_object* v_head_4405_; lean_object* v___y_4407_; lean_object* v_a_4408_; lean_object* v___y_4425_; lean_object* v___x_4446_; lean_object* v___x_4447_; lean_object* v___x_4448_; 
v___x_4402_ = lean_unsigned_to_nat(1u);
v___x_4403_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4402_);
v___x_4404_ = lean_unsigned_to_nat(2u);
v_head_4405_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4404_);
v___x_4446_ = lean_unsigned_to_nat(3u);
v___x_4447_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4446_);
lean_dec(v_x_4381_);
v___x_4448_ = l_Lean_Syntax_getOptional_x3f(v___x_4447_);
lean_dec(v___x_4447_);
if (lean_obj_tag(v___x_4448_) == 0)
{
lean_object* v___x_4449_; 
v___x_4449_ = lean_box(0);
v___y_4425_ = v___x_4449_;
goto v___jp_4424_;
}
else
{
lean_object* v_val_4450_; lean_object* v___x_4452_; uint8_t v_isShared_4453_; uint8_t v_isSharedCheck_4457_; 
v_val_4450_ = lean_ctor_get(v___x_4448_, 0);
v_isSharedCheck_4457_ = !lean_is_exclusive(v___x_4448_);
if (v_isSharedCheck_4457_ == 0)
{
v___x_4452_ = v___x_4448_;
v_isShared_4453_ = v_isSharedCheck_4457_;
goto v_resetjp_4451_;
}
else
{
lean_inc(v_val_4450_);
lean_dec(v___x_4448_);
v___x_4452_ = lean_box(0);
v_isShared_4453_ = v_isSharedCheck_4457_;
goto v_resetjp_4451_;
}
v_resetjp_4451_:
{
lean_object* v___x_4455_; 
if (v_isShared_4453_ == 0)
{
v___x_4455_ = v___x_4452_;
goto v_reusejp_4454_;
}
else
{
lean_object* v_reuseFailAlloc_4456_; 
v_reuseFailAlloc_4456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4456_, 0, v_val_4450_);
v___x_4455_ = v_reuseFailAlloc_4456_;
goto v_reusejp_4454_;
}
v_reusejp_4454_:
{
v___y_4425_ = v___x_4455_;
goto v___jp_4424_;
}
}
}
v___jp_4406_:
{
lean_object* v___x_4409_; 
v___x_4409_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v_head_4405_, v_a_4384_, v_a_4385_, v_a_4386_, v_a_4387_, v_a_4388_, v_a_4389_);
if (lean_obj_tag(v___x_4409_) == 0)
{
lean_object* v_a_4410_; lean_object* v___f_4411_; 
v_a_4410_ = lean_ctor_get(v___x_4409_, 0);
lean_inc(v_a_4410_);
lean_dec_ref_known(v___x_4409_, 1);
v___f_4411_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___lam__0___boxed), 9, 2);
lean_closure_set(v___f_4411_, 0, v_a_4410_);
lean_closure_set(v___f_4411_, 1, v_a_4408_);
if (lean_obj_tag(v___y_4407_) == 0)
{
lean_object* v___x_4412_; lean_object* v___x_4413_; 
v___x_4412_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1___closed__0));
v___x_4413_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_4413_, 0, v___x_4412_);
lean_ctor_set_uint8(v___x_4413_, sizeof(void*)*1, v___x_4400_);
v___y_4393_ = v___f_4411_;
v___y_4394_ = v___x_4413_;
goto v___jp_4392_;
}
else
{
lean_object* v_val_4414_; lean_object* v___x_4415_; 
v_val_4414_ = lean_ctor_get(v___y_4407_, 0);
lean_inc(v_val_4414_);
lean_dec_ref_known(v___y_4407_, 1);
v___x_4415_ = l_Lean_Elab_Tactic_expandLocation(v_val_4414_);
lean_dec(v_val_4414_);
v___y_4393_ = v___f_4411_;
v___y_4394_ = v___x_4415_;
goto v___jp_4392_;
}
}
else
{
lean_object* v_a_4416_; lean_object* v___x_4418_; uint8_t v_isShared_4419_; uint8_t v_isSharedCheck_4423_; 
lean_dec(v_a_4408_);
lean_dec(v___y_4407_);
v_a_4416_ = lean_ctor_get(v___x_4409_, 0);
v_isSharedCheck_4423_ = !lean_is_exclusive(v___x_4409_);
if (v_isSharedCheck_4423_ == 0)
{
v___x_4418_ = v___x_4409_;
v_isShared_4419_ = v_isSharedCheck_4423_;
goto v_resetjp_4417_;
}
else
{
lean_inc(v_a_4416_);
lean_dec(v___x_4409_);
v___x_4418_ = lean_box(0);
v_isShared_4419_ = v_isSharedCheck_4423_;
goto v_resetjp_4417_;
}
v_resetjp_4417_:
{
lean_object* v___x_4421_; 
if (v_isShared_4419_ == 0)
{
v___x_4421_ = v___x_4418_;
goto v_reusejp_4420_;
}
else
{
lean_object* v_reuseFailAlloc_4422_; 
v_reuseFailAlloc_4422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4422_, 0, v_a_4416_);
v___x_4421_ = v_reuseFailAlloc_4422_;
goto v_reusejp_4420_;
}
v_reusejp_4420_:
{
return v___x_4421_;
}
}
}
}
v___jp_4424_:
{
lean_object* v___x_4426_; 
v___x_4426_ = l_Lean_Syntax_getOptional_x3f(v___x_4403_);
lean_dec(v___x_4403_);
if (lean_obj_tag(v___x_4426_) == 0)
{
lean_object* v___x_4427_; 
v___x_4427_ = lean_box(0);
v___y_4407_ = v___y_4425_;
v_a_4408_ = v___x_4427_;
goto v___jp_4406_;
}
else
{
lean_object* v_val_4428_; lean_object* v___x_4430_; uint8_t v_isShared_4431_; uint8_t v_isSharedCheck_4445_; 
v_val_4428_ = lean_ctor_get(v___x_4426_, 0);
v_isSharedCheck_4445_ = !lean_is_exclusive(v___x_4426_);
if (v_isSharedCheck_4445_ == 0)
{
v___x_4430_ = v___x_4426_;
v_isShared_4431_ = v_isSharedCheck_4445_;
goto v_resetjp_4429_;
}
else
{
lean_inc(v_val_4428_);
lean_dec(v___x_4426_);
v___x_4430_ = lean_box(0);
v_isShared_4431_ = v_isSharedCheck_4445_;
goto v_resetjp_4429_;
}
v_resetjp_4429_:
{
lean_object* v___x_4432_; 
v___x_4432_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_val_4428_, v_a_4384_, v_a_4385_, v_a_4388_);
lean_dec(v_val_4428_);
if (lean_obj_tag(v___x_4432_) == 0)
{
lean_object* v_a_4433_; lean_object* v___x_4435_; 
v_a_4433_ = lean_ctor_get(v___x_4432_, 0);
lean_inc(v_a_4433_);
lean_dec_ref_known(v___x_4432_, 1);
if (v_isShared_4431_ == 0)
{
lean_ctor_set(v___x_4430_, 0, v_a_4433_);
v___x_4435_ = v___x_4430_;
goto v_reusejp_4434_;
}
else
{
lean_object* v_reuseFailAlloc_4436_; 
v_reuseFailAlloc_4436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4436_, 0, v_a_4433_);
v___x_4435_ = v_reuseFailAlloc_4436_;
goto v_reusejp_4434_;
}
v_reusejp_4434_:
{
v___y_4407_ = v___y_4425_;
v_a_4408_ = v___x_4435_;
goto v___jp_4406_;
}
}
else
{
lean_object* v_a_4437_; lean_object* v___x_4439_; uint8_t v_isShared_4440_; uint8_t v_isSharedCheck_4444_; 
lean_del_object(v___x_4430_);
lean_dec(v___y_4425_);
lean_dec(v_head_4405_);
v_a_4437_ = lean_ctor_get(v___x_4432_, 0);
v_isSharedCheck_4444_ = !lean_is_exclusive(v___x_4432_);
if (v_isSharedCheck_4444_ == 0)
{
v___x_4439_ = v___x_4432_;
v_isShared_4440_ = v_isSharedCheck_4444_;
goto v_resetjp_4438_;
}
else
{
lean_inc(v_a_4437_);
lean_dec(v___x_4432_);
v___x_4439_ = lean_box(0);
v_isShared_4440_ = v_isSharedCheck_4444_;
goto v_resetjp_4438_;
}
v_resetjp_4438_:
{
lean_object* v___x_4442_; 
if (v_isShared_4440_ == 0)
{
v___x_4442_ = v___x_4439_;
goto v_reusejp_4441_;
}
else
{
lean_object* v_reuseFailAlloc_4443_; 
v_reuseFailAlloc_4443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4443_, 0, v_a_4437_);
v___x_4442_ = v_reuseFailAlloc_4443_;
goto v_reusejp_4441_;
}
v_reusejp_4441_:
{
return v___x_4442_;
}
}
}
}
}
}
}
v___jp_4392_:
{
uint8_t v___x_4395_; uint8_t v___x_4396_; lean_object* v___x_4397_; lean_object* v___x_4398_; 
v___x_4395_ = 2;
v___x_4396_ = 0;
v___x_4397_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_4398_ = lp_mathlib_Mathlib_Tactic_transformAtLocation(v___y_4393_, v___x_4391_, v___y_4394_, v___x_4395_, v___x_4396_, v___x_4397_, v_a_4382_, v_a_4383_, v_a_4384_, v_a_4385_, v_a_4386_, v_a_4387_, v_a_4388_, v_a_4389_);
lean_dec(v___y_4394_);
return v___x_4398_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1___boxed(lean_object* v_x_4458_, lean_object* v_a_4459_, lean_object* v_a_4460_, lean_object* v_a_4461_, lean_object* v_a_4462_, lean_object* v_a_4463_, lean_object* v_a_4464_, lean_object* v_a_4465_, lean_object* v_a_4466_, lean_object* v_a_4467_){
_start:
{
lean_object* v_res_4468_; 
v_res_4468_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pull__1(v_x_4458_, v_a_4459_, v_a_4460_, v_a_4461_, v_a_4462_, v_a_4463_, v_a_4464_, v_a_4465_, v_a_4466_);
lean_dec(v_a_4466_);
lean_dec_ref(v_a_4465_);
lean_dec(v_a_4464_);
lean_dec_ref(v_a_4463_);
lean_dec(v_a_4462_);
lean_dec_ref(v_a_4461_);
lean_dec(v_a_4460_);
lean_dec_ref(v_a_4459_);
return v_res_4468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pushFun(lean_object* v_a_4469_, lean_object* v_a_4470_, lean_object* v_a_4471_, lean_object* v_a_4472_, lean_object* v_a_4473_, lean_object* v_a_4474_, lean_object* v_a_4475_, lean_object* v_a_4476_){
_start:
{
lean_object* v___x_4478_; uint8_t v___x_4479_; lean_object* v___x_4480_; 
v___x_4478_ = lean_box(1);
v___x_4479_ = 0;
v___x_4480_ = lp_mathlib_Mathlib_Tactic_Push_pushStep(v___x_4478_, v___x_4479_, v_a_4469_, v_a_4470_, v_a_4471_, v_a_4472_, v_a_4473_, v_a_4474_, v_a_4475_, v_a_4476_);
return v___x_4480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pushFun___boxed(lean_object* v_a_4481_, lean_object* v_a_4482_, lean_object* v_a_4483_, lean_object* v_a_4484_, lean_object* v_a_4485_, lean_object* v_a_4486_, lean_object* v_a_4487_, lean_object* v_a_4488_, lean_object* v_a_4489_){
_start:
{
lean_object* v_res_4490_; 
v_res_4490_ = lp_mathlib_pushFun(v_a_4481_, v_a_4482_, v_a_4483_, v_a_4484_, v_a_4485_, v_a_4486_, v_a_4487_, v_a_4488_);
lean_dec(v_a_4488_);
lean_dec_ref(v_a_4487_);
lean_dec(v_a_4486_);
lean_dec_ref(v_a_4485_);
lean_dec(v_a_4484_);
lean_dec_ref(v_a_4483_);
lean_dec(v_a_4482_);
return v_res_4490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pullFun(lean_object* v_a_4491_, lean_object* v_a_4492_, lean_object* v_a_4493_, lean_object* v_a_4494_, lean_object* v_a_4495_, lean_object* v_a_4496_, lean_object* v_a_4497_, lean_object* v_a_4498_){
_start:
{
lean_object* v___x_4500_; lean_object* v___x_4501_; 
v___x_4500_ = lean_box(1);
v___x_4501_ = lp_mathlib_Mathlib_Tactic_Push_pullStep(v___x_4500_, v_a_4491_, v_a_4492_, v_a_4493_, v_a_4494_, v_a_4495_, v_a_4496_, v_a_4497_, v_a_4498_);
return v___x_4501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_pullFun___boxed(lean_object* v_a_4502_, lean_object* v_a_4503_, lean_object* v_a_4504_, lean_object* v_a_4505_, lean_object* v_a_4506_, lean_object* v_a_4507_, lean_object* v_a_4508_, lean_object* v_a_4509_, lean_object* v_a_4510_){
_start:
{
lean_object* v_res_4511_; 
v_res_4511_ = lp_mathlib_pullFun(v_a_4502_, v_a_4503_, v_a_4504_, v_a_4505_, v_a_4506_, v_a_4507_, v_a_4508_, v_a_4509_);
lean_dec(v_a_4509_);
lean_dec_ref(v_a_4508_);
lean_dec(v_a_4507_);
lean_dec_ref(v_a_4506_);
lean_dec(v_a_4505_);
lean_dec_ref(v_a_4504_);
lean_dec(v_a_4503_);
return v_res_4511_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2(void){
_start:
{
lean_object* v___x_4518_; lean_object* v___x_4519_; lean_object* v___x_4520_; lean_object* v___x_4521_; 
v___x_4518_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__20);
v___x_4519_ = lean_unsigned_to_nat(1022u);
v___x_4520_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1));
v___x_4521_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4521_, 0, v___x_4520_);
lean_ctor_set(v___x_4521_, 1, v___x_4519_);
lean_ctor_set(v___x_4521_, 2, v___x_4518_);
return v___x_4521_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPush__________(void){
_start:
{
lean_object* v___x_4522_; 
v___x_4522_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2, &lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__2);
return v___x_4522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(lean_object* v_e_4523_, lean_object* v___y_4524_){
_start:
{
uint8_t v___x_4526_; 
v___x_4526_ = l_Lean_Expr_hasMVar(v_e_4523_);
if (v___x_4526_ == 0)
{
lean_object* v___x_4527_; 
v___x_4527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4527_, 0, v_e_4523_);
return v___x_4527_;
}
else
{
lean_object* v___x_4528_; lean_object* v_mctx_4529_; lean_object* v___x_4530_; lean_object* v_fst_4531_; lean_object* v_snd_4532_; lean_object* v___x_4533_; lean_object* v_cache_4534_; lean_object* v_zetaDeltaFVarIds_4535_; lean_object* v_postponed_4536_; lean_object* v_diag_4537_; lean_object* v___x_4539_; uint8_t v_isShared_4540_; uint8_t v_isSharedCheck_4546_; 
v___x_4528_ = lean_st_ref_get(v___y_4524_);
v_mctx_4529_ = lean_ctor_get(v___x_4528_, 0);
lean_inc_ref(v_mctx_4529_);
lean_dec(v___x_4528_);
v___x_4530_ = l_Lean_instantiateMVarsCore(v_mctx_4529_, v_e_4523_);
v_fst_4531_ = lean_ctor_get(v___x_4530_, 0);
lean_inc(v_fst_4531_);
v_snd_4532_ = lean_ctor_get(v___x_4530_, 1);
lean_inc(v_snd_4532_);
lean_dec_ref(v___x_4530_);
v___x_4533_ = lean_st_ref_take(v___y_4524_);
v_cache_4534_ = lean_ctor_get(v___x_4533_, 1);
v_zetaDeltaFVarIds_4535_ = lean_ctor_get(v___x_4533_, 2);
v_postponed_4536_ = lean_ctor_get(v___x_4533_, 3);
v_diag_4537_ = lean_ctor_get(v___x_4533_, 4);
v_isSharedCheck_4546_ = !lean_is_exclusive(v___x_4533_);
if (v_isSharedCheck_4546_ == 0)
{
lean_object* v_unused_4547_; 
v_unused_4547_ = lean_ctor_get(v___x_4533_, 0);
lean_dec(v_unused_4547_);
v___x_4539_ = v___x_4533_;
v_isShared_4540_ = v_isSharedCheck_4546_;
goto v_resetjp_4538_;
}
else
{
lean_inc(v_diag_4537_);
lean_inc(v_postponed_4536_);
lean_inc(v_zetaDeltaFVarIds_4535_);
lean_inc(v_cache_4534_);
lean_dec(v___x_4533_);
v___x_4539_ = lean_box(0);
v_isShared_4540_ = v_isSharedCheck_4546_;
goto v_resetjp_4538_;
}
v_resetjp_4538_:
{
lean_object* v___x_4542_; 
if (v_isShared_4540_ == 0)
{
lean_ctor_set(v___x_4539_, 0, v_snd_4532_);
v___x_4542_ = v___x_4539_;
goto v_reusejp_4541_;
}
else
{
lean_object* v_reuseFailAlloc_4545_; 
v_reuseFailAlloc_4545_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4545_, 0, v_snd_4532_);
lean_ctor_set(v_reuseFailAlloc_4545_, 1, v_cache_4534_);
lean_ctor_set(v_reuseFailAlloc_4545_, 2, v_zetaDeltaFVarIds_4535_);
lean_ctor_set(v_reuseFailAlloc_4545_, 3, v_postponed_4536_);
lean_ctor_set(v_reuseFailAlloc_4545_, 4, v_diag_4537_);
v___x_4542_ = v_reuseFailAlloc_4545_;
goto v_reusejp_4541_;
}
v_reusejp_4541_:
{
lean_object* v___x_4543_; lean_object* v___x_4544_; 
v___x_4543_ = lean_st_ref_set(v___y_4524_, v___x_4542_);
v___x_4544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4544_, 0, v_fst_4531_);
return v___x_4544_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg___boxed(lean_object* v_e_4548_, lean_object* v___y_4549_, lean_object* v___y_4550_){
_start:
{
lean_object* v_res_4551_; 
v_res_4551_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(v_e_4548_, v___y_4549_);
lean_dec(v___y_4549_);
return v_res_4551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0(lean_object* v_e_4552_, lean_object* v___y_4553_, lean_object* v___y_4554_, lean_object* v___y_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_, lean_object* v___y_4560_){
_start:
{
lean_object* v___x_4562_; 
v___x_4562_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(v_e_4552_, v___y_4558_);
return v___x_4562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___boxed(lean_object* v_e_4563_, lean_object* v___y_4564_, lean_object* v___y_4565_, lean_object* v___y_4566_, lean_object* v___y_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_, lean_object* v___y_4571_, lean_object* v___y_4572_){
_start:
{
lean_object* v_res_4573_; 
v_res_4573_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0(v_e_4563_, v___y_4564_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_, v___y_4569_, v___y_4570_, v___y_4571_);
lean_dec(v___y_4571_);
lean_dec_ref(v___y_4570_);
lean_dec(v___y_4569_);
lean_dec_ref(v___y_4568_);
lean_dec(v___y_4567_);
lean_dec_ref(v___y_4566_);
lean_dec(v___y_4565_);
lean_dec_ref(v___y_4564_);
return v_res_4573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0(lean_object* v___x_4574_, uint8_t v___x_4575_, uint8_t v___x_4576_, lean_object* v_head_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_, lean_object* v___y_4582_, lean_object* v___y_4583_, lean_object* v___y_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_){
_start:
{
uint8_t v___y_4589_; lean_object* v_a_4590_; lean_object* v___x_4624_; 
v___x_4624_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_4574_, v___x_4575_, v___x_4576_, v___y_4579_, v___y_4585_, v___y_4586_);
if (lean_obj_tag(v___x_4624_) == 0)
{
lean_object* v_a_4625_; lean_object* v___x_4626_; lean_object* v___x_4627_; lean_object* v_a_4628_; uint8_t v___y_4630_; uint8_t v___x_4650_; 
v_a_4625_ = lean_ctor_get(v___x_4624_, 0);
lean_inc(v_a_4625_);
lean_dec_ref_known(v___x_4624_, 1);
v___x_4626_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn___closed__2_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_));
v___x_4627_ = lp_mathlib_Lean_getBoolOption___at___00Mathlib_Tactic_Push_push_spec__0___redArg(v___x_4626_, v___x_4575_, v___y_4585_);
v_a_4628_ = lean_ctor_get(v___x_4627_, 0);
lean_inc(v_a_4628_);
lean_dec_ref(v___x_4627_);
v___x_4650_ = lean_unbox(v_a_4625_);
lean_dec(v_a_4625_);
if (v___x_4650_ == 0)
{
uint8_t v___x_4651_; 
v___x_4651_ = lean_unbox(v_a_4628_);
lean_dec(v_a_4628_);
v___y_4630_ = v___x_4651_;
goto v___jp_4629_;
}
else
{
lean_dec(v_a_4628_);
v___y_4630_ = v___x_4576_;
goto v___jp_4629_;
}
v___jp_4629_:
{
if (lean_obj_tag(v___y_4578_) == 0)
{
lean_object* v___x_4631_; 
v___x_4631_ = lean_box(0);
v___y_4589_ = v___y_4630_;
v_a_4590_ = v___x_4631_;
goto v___jp_4588_;
}
else
{
lean_object* v_val_4632_; lean_object* v___x_4634_; uint8_t v_isShared_4635_; uint8_t v_isSharedCheck_4649_; 
v_val_4632_ = lean_ctor_get(v___y_4578_, 0);
v_isSharedCheck_4649_ = !lean_is_exclusive(v___y_4578_);
if (v_isSharedCheck_4649_ == 0)
{
v___x_4634_ = v___y_4578_;
v_isShared_4635_ = v_isSharedCheck_4649_;
goto v_resetjp_4633_;
}
else
{
lean_inc(v_val_4632_);
lean_dec(v___y_4578_);
v___x_4634_ = lean_box(0);
v_isShared_4635_ = v_isSharedCheck_4649_;
goto v_resetjp_4633_;
}
v_resetjp_4633_:
{
lean_object* v___x_4636_; 
v___x_4636_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_val_4632_, v___y_4581_, v___y_4582_, v___y_4585_);
lean_dec(v_val_4632_);
if (lean_obj_tag(v___x_4636_) == 0)
{
lean_object* v_a_4637_; lean_object* v___x_4639_; 
v_a_4637_ = lean_ctor_get(v___x_4636_, 0);
lean_inc(v_a_4637_);
lean_dec_ref_known(v___x_4636_, 1);
if (v_isShared_4635_ == 0)
{
lean_ctor_set(v___x_4634_, 0, v_a_4637_);
v___x_4639_ = v___x_4634_;
goto v_reusejp_4638_;
}
else
{
lean_object* v_reuseFailAlloc_4640_; 
v_reuseFailAlloc_4640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4640_, 0, v_a_4637_);
v___x_4639_ = v_reuseFailAlloc_4640_;
goto v_reusejp_4638_;
}
v_reusejp_4638_:
{
v___y_4589_ = v___y_4630_;
v_a_4590_ = v___x_4639_;
goto v___jp_4588_;
}
}
else
{
lean_object* v_a_4641_; lean_object* v___x_4643_; uint8_t v_isShared_4644_; uint8_t v_isSharedCheck_4648_; 
lean_del_object(v___x_4634_);
lean_dec(v_head_4577_);
v_a_4641_ = lean_ctor_get(v___x_4636_, 0);
v_isSharedCheck_4648_ = !lean_is_exclusive(v___x_4636_);
if (v_isSharedCheck_4648_ == 0)
{
v___x_4643_ = v___x_4636_;
v_isShared_4644_ = v_isSharedCheck_4648_;
goto v_resetjp_4642_;
}
else
{
lean_inc(v_a_4641_);
lean_dec(v___x_4636_);
v___x_4643_ = lean_box(0);
v_isShared_4644_ = v_isSharedCheck_4648_;
goto v_resetjp_4642_;
}
v_resetjp_4642_:
{
lean_object* v___x_4646_; 
if (v_isShared_4644_ == 0)
{
v___x_4646_ = v___x_4643_;
goto v_reusejp_4645_;
}
else
{
lean_object* v_reuseFailAlloc_4647_; 
v_reuseFailAlloc_4647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4647_, 0, v_a_4641_);
v___x_4646_ = v_reuseFailAlloc_4647_;
goto v_reusejp_4645_;
}
v_reusejp_4645_:
{
return v___x_4646_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4652_; lean_object* v___x_4654_; uint8_t v_isShared_4655_; uint8_t v_isSharedCheck_4659_; 
lean_dec(v___y_4578_);
lean_dec(v_head_4577_);
v_a_4652_ = lean_ctor_get(v___x_4624_, 0);
v_isSharedCheck_4659_ = !lean_is_exclusive(v___x_4624_);
if (v_isSharedCheck_4659_ == 0)
{
v___x_4654_ = v___x_4624_;
v_isShared_4655_ = v_isSharedCheck_4659_;
goto v_resetjp_4653_;
}
else
{
lean_inc(v_a_4652_);
lean_dec(v___x_4624_);
v___x_4654_ = lean_box(0);
v_isShared_4655_ = v_isSharedCheck_4659_;
goto v_resetjp_4653_;
}
v_resetjp_4653_:
{
lean_object* v___x_4657_; 
if (v_isShared_4655_ == 0)
{
v___x_4657_ = v___x_4654_;
goto v_reusejp_4656_;
}
else
{
lean_object* v_reuseFailAlloc_4658_; 
v_reuseFailAlloc_4658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4658_, 0, v_a_4652_);
v___x_4657_ = v_reuseFailAlloc_4658_;
goto v_reusejp_4656_;
}
v_reusejp_4656_:
{
return v___x_4657_;
}
}
}
v___jp_4588_:
{
lean_object* v___x_4591_; 
v___x_4591_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v_head_4577_, v___y_4581_, v___y_4582_, v___y_4583_, v___y_4584_, v___y_4585_, v___y_4586_);
if (lean_obj_tag(v___x_4591_) == 0)
{
lean_object* v_a_4592_; lean_object* v___x_4593_; 
v_a_4592_ = lean_ctor_get(v___x_4591_, 0);
lean_inc(v_a_4592_);
lean_dec_ref_known(v___x_4591_, 1);
v___x_4593_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_4580_, v___y_4583_, v___y_4584_, v___y_4585_, v___y_4586_);
if (lean_obj_tag(v___x_4593_) == 0)
{
lean_object* v_a_4594_; lean_object* v___x_4595_; lean_object* v_a_4596_; lean_object* v___x_4597_; 
v_a_4594_ = lean_ctor_get(v___x_4593_, 0);
lean_inc(v_a_4594_);
lean_dec_ref_known(v___x_4593_, 1);
v___x_4595_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(v_a_4594_, v___y_4584_);
v_a_4596_ = lean_ctor_get(v___x_4595_, 0);
lean_inc(v_a_4596_);
lean_dec_ref(v___x_4595_);
v___x_4597_ = lp_mathlib_Mathlib_Tactic_Push_pushCore(v_a_4592_, v___y_4589_, v_a_4590_, v_a_4596_, v___y_4583_, v___y_4584_, v___y_4585_, v___y_4586_);
lean_dec(v_a_4590_);
if (lean_obj_tag(v___x_4597_) == 0)
{
lean_object* v_a_4598_; lean_object* v___x_4599_; 
v_a_4598_ = lean_ctor_get(v___x_4597_, 0);
lean_inc(v_a_4598_);
lean_dec_ref_known(v___x_4597_, 1);
v___x_4599_ = l_Lean_Elab_Tactic_Conv_applySimpResult(v_a_4598_, v___y_4579_, v___y_4580_, v___y_4581_, v___y_4582_, v___y_4583_, v___y_4584_, v___y_4585_, v___y_4586_);
return v___x_4599_;
}
else
{
lean_object* v_a_4600_; lean_object* v___x_4602_; uint8_t v_isShared_4603_; uint8_t v_isSharedCheck_4607_; 
v_a_4600_ = lean_ctor_get(v___x_4597_, 0);
v_isSharedCheck_4607_ = !lean_is_exclusive(v___x_4597_);
if (v_isSharedCheck_4607_ == 0)
{
v___x_4602_ = v___x_4597_;
v_isShared_4603_ = v_isSharedCheck_4607_;
goto v_resetjp_4601_;
}
else
{
lean_inc(v_a_4600_);
lean_dec(v___x_4597_);
v___x_4602_ = lean_box(0);
v_isShared_4603_ = v_isSharedCheck_4607_;
goto v_resetjp_4601_;
}
v_resetjp_4601_:
{
lean_object* v___x_4605_; 
if (v_isShared_4603_ == 0)
{
v___x_4605_ = v___x_4602_;
goto v_reusejp_4604_;
}
else
{
lean_object* v_reuseFailAlloc_4606_; 
v_reuseFailAlloc_4606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4606_, 0, v_a_4600_);
v___x_4605_ = v_reuseFailAlloc_4606_;
goto v_reusejp_4604_;
}
v_reusejp_4604_:
{
return v___x_4605_;
}
}
}
}
else
{
lean_object* v_a_4608_; lean_object* v___x_4610_; uint8_t v_isShared_4611_; uint8_t v_isSharedCheck_4615_; 
lean_dec(v_a_4592_);
lean_dec(v_a_4590_);
v_a_4608_ = lean_ctor_get(v___x_4593_, 0);
v_isSharedCheck_4615_ = !lean_is_exclusive(v___x_4593_);
if (v_isSharedCheck_4615_ == 0)
{
v___x_4610_ = v___x_4593_;
v_isShared_4611_ = v_isSharedCheck_4615_;
goto v_resetjp_4609_;
}
else
{
lean_inc(v_a_4608_);
lean_dec(v___x_4593_);
v___x_4610_ = lean_box(0);
v_isShared_4611_ = v_isSharedCheck_4615_;
goto v_resetjp_4609_;
}
v_resetjp_4609_:
{
lean_object* v___x_4613_; 
if (v_isShared_4611_ == 0)
{
v___x_4613_ = v___x_4610_;
goto v_reusejp_4612_;
}
else
{
lean_object* v_reuseFailAlloc_4614_; 
v_reuseFailAlloc_4614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4614_, 0, v_a_4608_);
v___x_4613_ = v_reuseFailAlloc_4614_;
goto v_reusejp_4612_;
}
v_reusejp_4612_:
{
return v___x_4613_;
}
}
}
}
else
{
lean_object* v_a_4616_; lean_object* v___x_4618_; uint8_t v_isShared_4619_; uint8_t v_isSharedCheck_4623_; 
lean_dec(v_a_4590_);
v_a_4616_ = lean_ctor_get(v___x_4591_, 0);
v_isSharedCheck_4623_ = !lean_is_exclusive(v___x_4591_);
if (v_isSharedCheck_4623_ == 0)
{
v___x_4618_ = v___x_4591_;
v_isShared_4619_ = v_isSharedCheck_4623_;
goto v_resetjp_4617_;
}
else
{
lean_inc(v_a_4616_);
lean_dec(v___x_4591_);
v___x_4618_ = lean_box(0);
v_isShared_4619_ = v_isSharedCheck_4623_;
goto v_resetjp_4617_;
}
v_resetjp_4617_:
{
lean_object* v___x_4621_; 
if (v_isShared_4619_ == 0)
{
v___x_4621_ = v___x_4618_;
goto v_reusejp_4620_;
}
else
{
lean_object* v_reuseFailAlloc_4622_; 
v_reuseFailAlloc_4622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4622_, 0, v_a_4616_);
v___x_4621_ = v_reuseFailAlloc_4622_;
goto v_reusejp_4620_;
}
v_reusejp_4620_:
{
return v___x_4621_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0___boxed(lean_object* v___x_4660_, lean_object* v___x_4661_, lean_object* v___x_4662_, lean_object* v_head_4663_, lean_object* v___y_4664_, lean_object* v___y_4665_, lean_object* v___y_4666_, lean_object* v___y_4667_, lean_object* v___y_4668_, lean_object* v___y_4669_, lean_object* v___y_4670_, lean_object* v___y_4671_, lean_object* v___y_4672_, lean_object* v___y_4673_){
_start:
{
uint8_t v___x_2987__boxed_4674_; uint8_t v___x_2988__boxed_4675_; lean_object* v_res_4676_; 
v___x_2987__boxed_4674_ = lean_unbox(v___x_4661_);
v___x_2988__boxed_4675_ = lean_unbox(v___x_4662_);
v_res_4676_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0(v___x_4660_, v___x_2987__boxed_4674_, v___x_2988__boxed_4675_, v_head_4663_, v___y_4664_, v___y_4665_, v___y_4666_, v___y_4667_, v___y_4668_, v___y_4669_, v___y_4670_, v___y_4671_, v___y_4672_);
lean_dec(v___y_4672_);
lean_dec_ref(v___y_4671_);
lean_dec(v___y_4670_);
lean_dec_ref(v___y_4669_);
lean_dec(v___y_4668_);
lean_dec_ref(v___y_4667_);
lean_dec(v___y_4666_);
lean_dec_ref(v___y_4665_);
return v_res_4676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1(lean_object* v_x_4677_, lean_object* v_a_4678_, lean_object* v_a_4679_, lean_object* v_a_4680_, lean_object* v_a_4681_, lean_object* v_a_4682_, lean_object* v_a_4683_, lean_object* v_a_4684_, lean_object* v_a_4685_){
_start:
{
lean_object* v___x_4687_; uint8_t v___x_4688_; 
v___x_4687_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1));
lean_inc(v_x_4677_);
v___x_4688_ = l_Lean_Syntax_isOfKind(v_x_4677_, v___x_4687_);
if (v___x_4688_ == 0)
{
lean_object* v___x_4689_; 
lean_dec(v_x_4677_);
v___x_4689_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_4689_;
}
else
{
lean_object* v___x_4690_; lean_object* v___x_4691_; lean_object* v___x_4692_; lean_object* v___x_4693_; lean_object* v___x_4694_; lean_object* v_head_4695_; lean_object* v___y_4697_; lean_object* v___x_4703_; 
v___x_4690_ = lean_unsigned_to_nat(1u);
v___x_4691_ = l_Lean_Syntax_getArg(v_x_4677_, v___x_4690_);
v___x_4692_ = lean_unsigned_to_nat(2u);
v___x_4693_ = l_Lean_Syntax_getArg(v_x_4677_, v___x_4692_);
v___x_4694_ = lean_unsigned_to_nat(3u);
v_head_4695_ = l_Lean_Syntax_getArg(v_x_4677_, v___x_4694_);
lean_dec(v_x_4677_);
v___x_4703_ = l_Lean_Syntax_getOptional_x3f(v___x_4693_);
lean_dec(v___x_4693_);
if (lean_obj_tag(v___x_4703_) == 0)
{
lean_object* v___x_4704_; 
v___x_4704_ = lean_box(0);
v___y_4697_ = v___x_4704_;
goto v___jp_4696_;
}
else
{
lean_object* v_val_4705_; lean_object* v___x_4707_; uint8_t v_isShared_4708_; uint8_t v_isSharedCheck_4712_; 
v_val_4705_ = lean_ctor_get(v___x_4703_, 0);
v_isSharedCheck_4712_ = !lean_is_exclusive(v___x_4703_);
if (v_isSharedCheck_4712_ == 0)
{
v___x_4707_ = v___x_4703_;
v_isShared_4708_ = v_isSharedCheck_4712_;
goto v_resetjp_4706_;
}
else
{
lean_inc(v_val_4705_);
lean_dec(v___x_4703_);
v___x_4707_ = lean_box(0);
v_isShared_4708_ = v_isSharedCheck_4712_;
goto v_resetjp_4706_;
}
v_resetjp_4706_:
{
lean_object* v___x_4710_; 
if (v_isShared_4708_ == 0)
{
v___x_4710_ = v___x_4707_;
goto v_reusejp_4709_;
}
else
{
lean_object* v_reuseFailAlloc_4711_; 
v_reuseFailAlloc_4711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4711_, 0, v_val_4705_);
v___x_4710_ = v_reuseFailAlloc_4711_;
goto v_reusejp_4709_;
}
v_reusejp_4709_:
{
v___y_4697_ = v___x_4710_;
goto v___jp_4696_;
}
}
}
v___jp_4696_:
{
uint8_t v___x_4698_; lean_object* v___x_4699_; lean_object* v___x_4700_; lean_object* v___f_4701_; lean_object* v___x_4702_; 
v___x_4698_ = 0;
v___x_4699_ = lean_box(v___x_4698_);
v___x_4700_ = lean_box(v___x_4688_);
v___f_4701_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___lam__0___boxed), 14, 5);
lean_closure_set(v___f_4701_, 0, v___x_4691_);
lean_closure_set(v___f_4701_, 1, v___x_4699_);
lean_closure_set(v___f_4701_, 2, v___x_4700_);
lean_closure_set(v___f_4701_, 3, v_head_4695_);
lean_closure_set(v___f_4701_, 4, v___y_4697_);
v___x_4702_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4701_, v_a_4678_, v_a_4679_, v_a_4680_, v_a_4681_, v_a_4682_, v_a_4683_, v_a_4684_, v_a_4685_);
return v___x_4702_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1___boxed(lean_object* v_x_4713_, lean_object* v_a_4714_, lean_object* v_a_4715_, lean_object* v_a_4716_, lean_object* v_a_4717_, lean_object* v_a_4718_, lean_object* v_a_4719_, lean_object* v_a_4720_, lean_object* v_a_4721_, lean_object* v_a_4722_){
_start:
{
lean_object* v_res_4723_; 
v_res_4723_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1(v_x_4713_, v_a_4714_, v_a_4715_, v_a_4716_, v_a_4717_, v_a_4718_, v_a_4719_, v_a_4720_, v_a_4721_);
lean_dec(v_a_4721_);
lean_dec_ref(v_a_4720_);
lean_dec(v_a_4719_);
lean_dec_ref(v_a_4718_);
lean_dec(v_a_4717_);
lean_dec_ref(v_a_4716_);
lean_dec(v_a_4715_);
lean_dec_ref(v_a_4714_);
return v_res_4723_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2(void){
_start:
{
lean_object* v___x_4730_; lean_object* v___x_4731_; lean_object* v___x_4732_; lean_object* v___x_4733_; 
v___x_4730_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2, &lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_push__neg___closed__2);
v___x_4731_ = lean_unsigned_to_nat(1022u);
v___x_4732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1));
v___x_4733_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4733_, 0, v___x_4732_);
lean_ctor_set(v___x_4733_, 1, v___x_4731_);
lean_ctor_set(v___x_4733_, 2, v___x_4730_);
return v___x_4733_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPush__neg__(void){
_start:
{
lean_object* v___x_4734_; 
v___x_4734_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2, &lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__2);
return v___x_4734_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2(void){
_start:
{
lean_object* v___x_4738_; lean_object* v___x_4739_; 
v___x_4738_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__1));
v___x_4739_ = l_Lean_MessageData_ofFormat(v___x_4738_);
return v___x_4739_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5(void){
_start:
{
lean_object* v___x_4743_; 
v___x_4743_ = l_Array_mkArray0(lean_box(0));
return v___x_4743_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6(void){
_start:
{
lean_object* v___x_4744_; lean_object* v___x_4745_; 
v___x_4744_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__3));
v___x_4745_ = l_String_toRawSubstring_x27(v___x_4744_);
return v___x_4745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1(lean_object* v_x_4757_, lean_object* v_a_4758_, lean_object* v_a_4759_, lean_object* v_a_4760_, lean_object* v_a_4761_, lean_object* v_a_4762_, lean_object* v_a_4763_, lean_object* v_a_4764_, lean_object* v_a_4765_){
_start:
{
lean_object* v___x_4767_; uint8_t v___x_4768_; 
v___x_4767_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush__neg___00__closed__1));
lean_inc(v_x_4757_);
v___x_4768_ = l_Lean_Syntax_isOfKind(v_x_4757_, v___x_4767_);
if (v___x_4768_ == 0)
{
lean_object* v___x_4769_; 
lean_dec(v_x_4757_);
v___x_4769_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_4769_;
}
else
{
lean_object* v___x_4770_; lean_object* v___x_4771_; 
v___x_4770_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__2);
v___x_4771_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0(v___x_4770_, v_a_4758_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, v_a_4763_, v_a_4764_, v_a_4765_);
if (lean_obj_tag(v___x_4771_) == 0)
{
lean_object* v_ref_4772_; lean_object* v_quotContext_4773_; lean_object* v_currMacroScope_4774_; lean_object* v___x_4775_; lean_object* v___x_4776_; uint8_t v___x_4777_; lean_object* v___x_4778_; lean_object* v___x_4779_; lean_object* v___x_4780_; lean_object* v___x_4781_; lean_object* v___x_4782_; lean_object* v___x_4783_; lean_object* v___x_4784_; lean_object* v___x_4785_; lean_object* v___x_4786_; lean_object* v___x_4787_; lean_object* v___x_4788_; lean_object* v___x_4789_; lean_object* v___x_4790_; lean_object* v___x_4791_; 
lean_dec_ref_known(v___x_4771_, 1);
v_ref_4772_ = lean_ctor_get(v_a_4764_, 5);
v_quotContext_4773_ = lean_ctor_get(v_a_4764_, 10);
v_currMacroScope_4774_ = lean_ctor_get(v_a_4764_, 11);
v___x_4775_ = lean_unsigned_to_nat(1u);
v___x_4776_ = l_Lean_Syntax_getArg(v_x_4757_, v___x_4775_);
lean_dec(v_x_4757_);
v___x_4777_ = 0;
v___x_4778_ = l_Lean_SourceInfo_fromRef(v_ref_4772_, v___x_4777_);
v___x_4779_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1));
v___x_4780_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2));
lean_inc_n(v___x_4778_, 3);
v___x_4781_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4781_, 0, v___x_4778_);
lean_ctor_set(v___x_4781_, 1, v___x_4780_);
v___x_4782_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4));
v___x_4783_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5);
v___x_4784_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4784_, 0, v___x_4778_);
lean_ctor_set(v___x_4784_, 1, v___x_4782_);
lean_ctor_set(v___x_4784_, 2, v___x_4783_);
v___x_4785_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__6);
v___x_4786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__4));
lean_inc(v_currMacroScope_4774_);
lean_inc(v_quotContext_4773_);
v___x_4787_ = l_Lean_addMacroScope(v_quotContext_4773_, v___x_4786_, v_currMacroScope_4774_);
v___x_4788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__10));
v___x_4789_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4789_, 0, v___x_4778_);
lean_ctor_set(v___x_4789_, 1, v___x_4785_);
lean_ctor_set(v___x_4789_, 2, v___x_4787_);
lean_ctor_set(v___x_4789_, 3, v___x_4788_);
v___x_4790_ = l_Lean_Syntax_node4(v___x_4778_, v___x_4779_, v___x_4781_, v___x_4776_, v___x_4784_, v___x_4789_);
v___x_4791_ = l_Lean_Elab_Tactic_evalTactic(v___x_4790_, v_a_4758_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, v_a_4763_, v_a_4764_, v_a_4765_);
return v___x_4791_;
}
else
{
lean_dec(v_x_4757_);
return v___x_4771_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___boxed(lean_object* v_x_4792_, lean_object* v_a_4793_, lean_object* v_a_4794_, lean_object* v_a_4795_, lean_object* v_a_4796_, lean_object* v_a_4797_, lean_object* v_a_4798_, lean_object* v_a_4799_, lean_object* v_a_4800_, lean_object* v_a_4801_){
_start:
{
lean_object* v_res_4802_; 
v_res_4802_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1(v_x_4792_, v_a_4793_, v_a_4794_, v_a_4795_, v_a_4796_, v_a_4797_, v_a_4798_, v_a_4799_, v_a_4800_);
lean_dec(v_a_4800_);
lean_dec_ref(v_a_4799_);
lean_dec(v_a_4798_);
lean_dec_ref(v_a_4797_);
lean_dec(v_a_4796_);
lean_dec_ref(v_a_4795_);
lean_dec(v_a_4794_);
lean_dec_ref(v_a_4793_);
return v_res_4802_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4(void){
_start:
{
lean_object* v___x_4812_; lean_object* v___x_4813_; lean_object* v___x_4814_; lean_object* v___x_4815_; 
v___x_4812_ = l_Lean_Parser_Tactic_optConfig;
v___x_4813_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__3));
v___x_4814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4815_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4815_, 0, v___x_4814_);
lean_ctor_set(v___x_4815_, 1, v___x_4813_);
lean_ctor_set(v___x_4815_, 2, v___x_4812_);
return v___x_4815_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5(void){
_start:
{
lean_object* v___x_4816_; lean_object* v___x_4817_; lean_object* v___x_4818_; lean_object* v___x_4819_; 
v___x_4816_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8);
v___x_4817_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__4);
v___x_4818_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4819_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4819_, 0, v___x_4818_);
lean_ctor_set(v___x_4819_, 1, v___x_4817_);
lean_ctor_set(v___x_4819_, 2, v___x_4816_);
return v___x_4819_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9(void){
_start:
{
lean_object* v___x_4826_; lean_object* v___x_4827_; lean_object* v___x_4828_; lean_object* v___x_4829_; 
v___x_4826_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__8));
v___x_4827_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__5);
v___x_4828_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4829_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4829_, 0, v___x_4828_);
lean_ctor_set(v___x_4829_, 1, v___x_4827_);
lean_ctor_set(v___x_4829_, 2, v___x_4826_);
return v___x_4829_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10(void){
_start:
{
lean_object* v___x_4830_; lean_object* v___x_4831_; lean_object* v___x_4832_; lean_object* v___x_4833_; 
v___x_4830_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18));
v___x_4831_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__9);
v___x_4832_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4833_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4833_, 0, v___x_4832_);
lean_ctor_set(v___x_4833_, 1, v___x_4831_);
lean_ctor_set(v___x_4833_, 2, v___x_4830_);
return v___x_4833_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13(void){
_start:
{
lean_object* v___x_4837_; lean_object* v___x_4838_; lean_object* v___x_4839_; lean_object* v___x_4840_; 
v___x_4837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__12));
v___x_4838_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__10);
v___x_4839_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4840_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4840_, 0, v___x_4839_);
lean_ctor_set(v___x_4840_, 1, v___x_4838_);
lean_ctor_set(v___x_4840_, 2, v___x_4837_);
return v___x_4840_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14(void){
_start:
{
lean_object* v___x_4841_; lean_object* v___x_4842_; lean_object* v___x_4843_; lean_object* v___x_4844_; 
v___x_4841_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18));
v___x_4842_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__13);
v___x_4843_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_4844_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4844_, 0, v___x_4843_);
lean_ctor_set(v___x_4844_, 1, v___x_4842_);
lean_ctor_set(v___x_4844_, 2, v___x_4841_);
return v___x_4844_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15(void){
_start:
{
lean_object* v___x_4845_; lean_object* v___x_4846_; lean_object* v___x_4847_; lean_object* v___x_4848_; 
v___x_4845_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__14);
v___x_4846_ = lean_unsigned_to_nat(1022u);
v___x_4847_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1));
v___x_4848_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4848_, 0, v___x_4847_);
lean_ctor_set(v___x_4848_, 1, v___x_4846_);
lean_ctor_set(v___x_4848_, 2, v___x_4845_);
return v___x_4848_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand(void){
_start:
{
lean_object* v___x_4849_; 
v___x_4849_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15, &lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__15);
return v___x_4849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1(lean_object* v_x_4861_, lean_object* v_a_4862_, lean_object* v_a_4863_){
_start:
{
lean_object* v___x_4864_; uint8_t v___x_4865_; 
v___x_4864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__1));
lean_inc(v_x_4861_);
v___x_4865_ = l_Lean_Syntax_isOfKind(v_x_4861_, v___x_4864_);
if (v___x_4865_ == 0)
{
lean_object* v___x_4866_; lean_object* v___x_4867_; 
lean_dec(v_x_4861_);
v___x_4866_ = lean_box(1);
v___x_4867_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4867_, 0, v___x_4866_);
lean_ctor_set(v___x_4867_, 1, v_a_4863_);
return v___x_4867_;
}
else
{
lean_object* v___x_4868_; lean_object* v_tk_4869_; lean_object* v___x_4870_; lean_object* v___x_4871_; lean_object* v___x_4872_; lean_object* v___x_4873_; lean_object* v___x_4874_; lean_object* v___x_4875_; lean_object* v___x_4876_; lean_object* v___x_4877_; lean_object* v___y_4879_; lean_object* v___y_4880_; lean_object* v___y_4881_; lean_object* v___y_4882_; lean_object* v___y_4883_; lean_object* v___y_4884_; lean_object* v___y_4885_; lean_object* v___y_4886_; lean_object* v___y_4895_; lean_object* v___x_4911_; 
v___x_4868_ = lean_unsigned_to_nat(0u);
v_tk_4869_ = l_Lean_Syntax_getArg(v_x_4861_, v___x_4868_);
v___x_4870_ = lean_unsigned_to_nat(1u);
v___x_4871_ = l_Lean_Syntax_getArg(v_x_4861_, v___x_4870_);
v___x_4872_ = lean_unsigned_to_nat(2u);
v___x_4873_ = l_Lean_Syntax_getArg(v_x_4861_, v___x_4872_);
v___x_4874_ = lean_unsigned_to_nat(4u);
v___x_4875_ = l_Lean_Syntax_getArg(v_x_4861_, v___x_4874_);
v___x_4876_ = lean_unsigned_to_nat(6u);
v___x_4877_ = l_Lean_Syntax_getArg(v_x_4861_, v___x_4876_);
lean_dec(v_x_4861_);
v___x_4911_ = l_Lean_Syntax_getOptional_x3f(v___x_4873_);
lean_dec(v___x_4873_);
if (lean_obj_tag(v___x_4911_) == 0)
{
lean_object* v___x_4912_; 
v___x_4912_ = lean_box(0);
v___y_4895_ = v___x_4912_;
goto v___jp_4894_;
}
else
{
lean_object* v_val_4913_; lean_object* v___x_4915_; uint8_t v_isShared_4916_; uint8_t v_isSharedCheck_4920_; 
v_val_4913_ = lean_ctor_get(v___x_4911_, 0);
v_isSharedCheck_4920_ = !lean_is_exclusive(v___x_4911_);
if (v_isSharedCheck_4920_ == 0)
{
v___x_4915_ = v___x_4911_;
v_isShared_4916_ = v_isSharedCheck_4920_;
goto v_resetjp_4914_;
}
else
{
lean_inc(v_val_4913_);
lean_dec(v___x_4911_);
v___x_4915_ = lean_box(0);
v_isShared_4916_ = v_isSharedCheck_4920_;
goto v_resetjp_4914_;
}
v_resetjp_4914_:
{
lean_object* v___x_4918_; 
if (v_isShared_4916_ == 0)
{
v___x_4918_ = v___x_4915_;
goto v_reusejp_4917_;
}
else
{
lean_object* v_reuseFailAlloc_4919_; 
v_reuseFailAlloc_4919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4919_, 0, v_val_4913_);
v___x_4918_ = v_reuseFailAlloc_4919_;
goto v_reusejp_4917_;
}
v_reusejp_4917_:
{
v___y_4895_ = v___x_4918_;
goto v___jp_4894_;
}
}
}
v___jp_4878_:
{
lean_object* v___x_4887_; lean_object* v___x_4888_; lean_object* v___x_4889_; lean_object* v___x_4890_; lean_object* v___x_4891_; lean_object* v___x_4892_; lean_object* v___x_4893_; 
lean_inc_ref(v___y_4881_);
v___x_4887_ = l_Array_append___redArg(v___y_4881_, v___y_4886_);
lean_dec_ref(v___y_4886_);
lean_inc(v___y_4883_);
lean_inc_n(v___y_4882_, 3);
v___x_4888_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4888_, 0, v___y_4882_);
lean_ctor_set(v___x_4888_, 1, v___y_4883_);
lean_ctor_set(v___x_4888_, 2, v___x_4887_);
lean_inc(v___y_4879_);
v___x_4889_ = l_Lean_Syntax_node4(v___y_4882_, v___y_4879_, v___y_4884_, v___x_4871_, v___x_4888_, v___x_4875_);
v___x_4890_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__0));
v___x_4891_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4891_, 0, v___y_4882_);
lean_ctor_set(v___x_4891_, 1, v___x_4890_);
lean_inc(v___y_4885_);
v___x_4892_ = l_Lean_Syntax_node4(v___y_4882_, v___y_4885_, v___y_4880_, v___x_4889_, v___x_4891_, v___x_4877_);
v___x_4893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4893_, 0, v___x_4892_);
lean_ctor_set(v___x_4893_, 1, v_a_4863_);
return v___x_4893_;
}
v___jp_4894_:
{
lean_object* v_ref_4896_; uint8_t v___x_4897_; lean_object* v___x_4898_; lean_object* v___x_4899_; lean_object* v___x_4900_; lean_object* v___x_4901_; lean_object* v___x_4902_; lean_object* v___x_4903_; lean_object* v___x_4904_; lean_object* v___x_4905_; lean_object* v___x_4906_; lean_object* v___x_4907_; 
v_ref_4896_ = lean_ctor_get(v_a_4862_, 5);
v___x_4897_ = 0;
v___x_4898_ = l_Lean_SourceInfo_fromRef(v_ref_4896_, v___x_4897_);
v___x_4899_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3));
v___x_4900_ = l_Lean_SourceInfo_fromRef(v_tk_4869_, v___x_4865_);
lean_dec(v_tk_4869_);
v___x_4901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__4));
v___x_4902_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4902_, 0, v___x_4900_);
lean_ctor_set(v___x_4902_, 1, v___x_4901_);
v___x_4903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPush___________00__closed__1));
v___x_4904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__2));
lean_inc(v___x_4898_);
v___x_4905_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4905_, 0, v___x_4898_);
lean_ctor_set(v___x_4905_, 1, v___x_4904_);
v___x_4906_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4));
v___x_4907_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5);
if (lean_obj_tag(v___y_4895_) == 1)
{
lean_object* v_val_4908_; lean_object* v___x_4909_; 
v_val_4908_ = lean_ctor_get(v___y_4895_, 0);
lean_inc(v_val_4908_);
lean_dec_ref_known(v___y_4895_, 1);
v___x_4909_ = l_Array_mkArray1___redArg(v_val_4908_);
v___y_4879_ = v___x_4903_;
v___y_4880_ = v___x_4902_;
v___y_4881_ = v___x_4907_;
v___y_4882_ = v___x_4898_;
v___y_4883_ = v___x_4906_;
v___y_4884_ = v___x_4905_;
v___y_4885_ = v___x_4899_;
v___y_4886_ = v___x_4909_;
goto v___jp_4878_;
}
else
{
lean_object* v___x_4910_; 
lean_dec(v___y_4895_);
v___x_4910_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__5));
v___y_4879_ = v___x_4903_;
v___y_4880_ = v___x_4902_;
v___y_4881_ = v___x_4907_;
v___y_4882_ = v___x_4898_;
v___y_4883_ = v___x_4906_;
v___y_4884_ = v___x_4905_;
v___y_4885_ = v___x_4899_;
v___y_4886_ = v___x_4910_;
goto v___jp_4878_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___boxed(lean_object* v_x_4921_, lean_object* v_a_4922_, lean_object* v_a_4923_){
_start:
{
lean_object* v_res_4924_; 
v_res_4924_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1(v_x_4921_, v_a_4922_, v_a_4923_);
lean_dec_ref(v_a_4922_);
return v_res_4924_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2(void){
_start:
{
lean_object* v___x_4931_; lean_object* v___x_4932_; lean_object* v___x_4933_; lean_object* v___x_4934_; 
v___x_4931_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pull___closed__4, &lp_mathlib_Mathlib_Tactic_Push_pull___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_pull___closed__4);
v___x_4932_ = lean_unsigned_to_nat(1022u);
v___x_4933_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1));
v___x_4934_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4934_, 0, v___x_4933_);
lean_ctor_set(v___x_4934_, 1, v___x_4932_);
lean_ctor_set(v___x_4934_, 2, v___x_4931_);
return v___x_4934_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_convPull________(void){
_start:
{
lean_object* v___x_4935_; 
v___x_4935_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2, &lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__2);
return v___x_4935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0(lean_object* v_head_4936_, lean_object* v___y_4937_, lean_object* v___y_4938_, lean_object* v___y_4939_, lean_object* v___y_4940_, lean_object* v___y_4941_, lean_object* v___y_4942_, lean_object* v___y_4943_, lean_object* v___y_4944_, lean_object* v___y_4945_){
_start:
{
lean_object* v_a_4948_; 
if (lean_obj_tag(v___y_4937_) == 0)
{
lean_object* v___x_4982_; 
v___x_4982_ = lean_box(0);
v_a_4948_ = v___x_4982_;
goto v___jp_4947_;
}
else
{
lean_object* v_val_4983_; lean_object* v___x_4985_; uint8_t v_isShared_4986_; uint8_t v_isSharedCheck_5000_; 
v_val_4983_ = lean_ctor_get(v___y_4937_, 0);
v_isSharedCheck_5000_ = !lean_is_exclusive(v___y_4937_);
if (v_isSharedCheck_5000_ == 0)
{
v___x_4985_ = v___y_4937_;
v_isShared_4986_ = v_isSharedCheck_5000_;
goto v_resetjp_4984_;
}
else
{
lean_inc(v_val_4983_);
lean_dec(v___y_4937_);
v___x_4985_ = lean_box(0);
v_isShared_4986_ = v_isSharedCheck_5000_;
goto v_resetjp_4984_;
}
v_resetjp_4984_:
{
lean_object* v___x_4987_; 
v___x_4987_ = lp_mathlib_Mathlib_Tactic_Push_elabDischarger___redArg(v_val_4983_, v___y_4940_, v___y_4941_, v___y_4944_);
lean_dec(v_val_4983_);
if (lean_obj_tag(v___x_4987_) == 0)
{
lean_object* v_a_4988_; lean_object* v___x_4990_; 
v_a_4988_ = lean_ctor_get(v___x_4987_, 0);
lean_inc(v_a_4988_);
lean_dec_ref_known(v___x_4987_, 1);
if (v_isShared_4986_ == 0)
{
lean_ctor_set(v___x_4985_, 0, v_a_4988_);
v___x_4990_ = v___x_4985_;
goto v_reusejp_4989_;
}
else
{
lean_object* v_reuseFailAlloc_4991_; 
v_reuseFailAlloc_4991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4991_, 0, v_a_4988_);
v___x_4990_ = v_reuseFailAlloc_4991_;
goto v_reusejp_4989_;
}
v_reusejp_4989_:
{
v_a_4948_ = v___x_4990_;
goto v___jp_4947_;
}
}
else
{
lean_object* v_a_4992_; lean_object* v___x_4994_; uint8_t v_isShared_4995_; uint8_t v_isSharedCheck_4999_; 
lean_del_object(v___x_4985_);
lean_dec(v_head_4936_);
v_a_4992_ = lean_ctor_get(v___x_4987_, 0);
v_isSharedCheck_4999_ = !lean_is_exclusive(v___x_4987_);
if (v_isSharedCheck_4999_ == 0)
{
v___x_4994_ = v___x_4987_;
v_isShared_4995_ = v_isSharedCheck_4999_;
goto v_resetjp_4993_;
}
else
{
lean_inc(v_a_4992_);
lean_dec(v___x_4987_);
v___x_4994_ = lean_box(0);
v_isShared_4995_ = v_isSharedCheck_4999_;
goto v_resetjp_4993_;
}
v_resetjp_4993_:
{
lean_object* v___x_4997_; 
if (v_isShared_4995_ == 0)
{
v___x_4997_ = v___x_4994_;
goto v_reusejp_4996_;
}
else
{
lean_object* v_reuseFailAlloc_4998_; 
v_reuseFailAlloc_4998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4998_, 0, v_a_4992_);
v___x_4997_ = v_reuseFailAlloc_4998_;
goto v_reusejp_4996_;
}
v_reusejp_4996_:
{
return v___x_4997_;
}
}
}
}
}
v___jp_4947_:
{
lean_object* v___x_4949_; 
v___x_4949_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v_head_4936_, v___y_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_, v___y_4945_);
if (lean_obj_tag(v___x_4949_) == 0)
{
lean_object* v_a_4950_; lean_object* v___x_4951_; 
v_a_4950_ = lean_ctor_get(v___x_4949_, 0);
lean_inc(v_a_4950_);
lean_dec_ref_known(v___x_4949_, 1);
v___x_4951_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_4939_, v___y_4942_, v___y_4943_, v___y_4944_, v___y_4945_);
if (lean_obj_tag(v___x_4951_) == 0)
{
lean_object* v_a_4952_; lean_object* v___x_4953_; lean_object* v_a_4954_; lean_object* v___x_4955_; 
v_a_4952_ = lean_ctor_get(v___x_4951_, 0);
lean_inc(v_a_4952_);
lean_dec_ref_known(v___x_4951_, 1);
v___x_4953_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush____________1_spec__0___redArg(v_a_4952_, v___y_4943_);
v_a_4954_ = lean_ctor_get(v___x_4953_, 0);
lean_inc(v_a_4954_);
lean_dec_ref(v___x_4953_);
v___x_4955_ = lp_mathlib_Mathlib_Tactic_Push_pullCore(v_a_4950_, v_a_4954_, v_a_4948_, v___y_4942_, v___y_4943_, v___y_4944_, v___y_4945_);
lean_dec(v_a_4948_);
if (lean_obj_tag(v___x_4955_) == 0)
{
lean_object* v_a_4956_; lean_object* v___x_4957_; 
v_a_4956_ = lean_ctor_get(v___x_4955_, 0);
lean_inc(v_a_4956_);
lean_dec_ref_known(v___x_4955_, 1);
v___x_4957_ = l_Lean_Elab_Tactic_Conv_applySimpResult(v_a_4956_, v___y_4938_, v___y_4939_, v___y_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_, v___y_4945_);
return v___x_4957_;
}
else
{
lean_object* v_a_4958_; lean_object* v___x_4960_; uint8_t v_isShared_4961_; uint8_t v_isSharedCheck_4965_; 
v_a_4958_ = lean_ctor_get(v___x_4955_, 0);
v_isSharedCheck_4965_ = !lean_is_exclusive(v___x_4955_);
if (v_isSharedCheck_4965_ == 0)
{
v___x_4960_ = v___x_4955_;
v_isShared_4961_ = v_isSharedCheck_4965_;
goto v_resetjp_4959_;
}
else
{
lean_inc(v_a_4958_);
lean_dec(v___x_4955_);
v___x_4960_ = lean_box(0);
v_isShared_4961_ = v_isSharedCheck_4965_;
goto v_resetjp_4959_;
}
v_resetjp_4959_:
{
lean_object* v___x_4963_; 
if (v_isShared_4961_ == 0)
{
v___x_4963_ = v___x_4960_;
goto v_reusejp_4962_;
}
else
{
lean_object* v_reuseFailAlloc_4964_; 
v_reuseFailAlloc_4964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4964_, 0, v_a_4958_);
v___x_4963_ = v_reuseFailAlloc_4964_;
goto v_reusejp_4962_;
}
v_reusejp_4962_:
{
return v___x_4963_;
}
}
}
}
else
{
lean_object* v_a_4966_; lean_object* v___x_4968_; uint8_t v_isShared_4969_; uint8_t v_isSharedCheck_4973_; 
lean_dec(v_a_4950_);
lean_dec(v_a_4948_);
v_a_4966_ = lean_ctor_get(v___x_4951_, 0);
v_isSharedCheck_4973_ = !lean_is_exclusive(v___x_4951_);
if (v_isSharedCheck_4973_ == 0)
{
v___x_4968_ = v___x_4951_;
v_isShared_4969_ = v_isSharedCheck_4973_;
goto v_resetjp_4967_;
}
else
{
lean_inc(v_a_4966_);
lean_dec(v___x_4951_);
v___x_4968_ = lean_box(0);
v_isShared_4969_ = v_isSharedCheck_4973_;
goto v_resetjp_4967_;
}
v_resetjp_4967_:
{
lean_object* v___x_4971_; 
if (v_isShared_4969_ == 0)
{
v___x_4971_ = v___x_4968_;
goto v_reusejp_4970_;
}
else
{
lean_object* v_reuseFailAlloc_4972_; 
v_reuseFailAlloc_4972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4972_, 0, v_a_4966_);
v___x_4971_ = v_reuseFailAlloc_4972_;
goto v_reusejp_4970_;
}
v_reusejp_4970_:
{
return v___x_4971_;
}
}
}
}
else
{
lean_object* v_a_4974_; lean_object* v___x_4976_; uint8_t v_isShared_4977_; uint8_t v_isSharedCheck_4981_; 
lean_dec(v_a_4948_);
v_a_4974_ = lean_ctor_get(v___x_4949_, 0);
v_isSharedCheck_4981_ = !lean_is_exclusive(v___x_4949_);
if (v_isSharedCheck_4981_ == 0)
{
v___x_4976_ = v___x_4949_;
v_isShared_4977_ = v_isSharedCheck_4981_;
goto v_resetjp_4975_;
}
else
{
lean_inc(v_a_4974_);
lean_dec(v___x_4949_);
v___x_4976_ = lean_box(0);
v_isShared_4977_ = v_isSharedCheck_4981_;
goto v_resetjp_4975_;
}
v_resetjp_4975_:
{
lean_object* v___x_4979_; 
if (v_isShared_4977_ == 0)
{
v___x_4979_ = v___x_4976_;
goto v_reusejp_4978_;
}
else
{
lean_object* v_reuseFailAlloc_4980_; 
v_reuseFailAlloc_4980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4980_, 0, v_a_4974_);
v___x_4979_ = v_reuseFailAlloc_4980_;
goto v_reusejp_4978_;
}
v_reusejp_4978_:
{
return v___x_4979_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0___boxed(lean_object* v_head_5001_, lean_object* v___y_5002_, lean_object* v___y_5003_, lean_object* v___y_5004_, lean_object* v___y_5005_, lean_object* v___y_5006_, lean_object* v___y_5007_, lean_object* v___y_5008_, lean_object* v___y_5009_, lean_object* v___y_5010_, lean_object* v___y_5011_){
_start:
{
lean_object* v_res_5012_; 
v_res_5012_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0(v_head_5001_, v___y_5002_, v___y_5003_, v___y_5004_, v___y_5005_, v___y_5006_, v___y_5007_, v___y_5008_, v___y_5009_, v___y_5010_);
lean_dec(v___y_5010_);
lean_dec_ref(v___y_5009_);
lean_dec(v___y_5008_);
lean_dec_ref(v___y_5007_);
lean_dec(v___y_5006_);
lean_dec_ref(v___y_5005_);
lean_dec(v___y_5004_);
lean_dec_ref(v___y_5003_);
return v_res_5012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1(lean_object* v_x_5013_, lean_object* v_a_5014_, lean_object* v_a_5015_, lean_object* v_a_5016_, lean_object* v_a_5017_, lean_object* v_a_5018_, lean_object* v_a_5019_, lean_object* v_a_5020_, lean_object* v_a_5021_){
_start:
{
lean_object* v___x_5023_; uint8_t v___x_5024_; 
v___x_5023_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1));
lean_inc(v_x_5013_);
v___x_5024_ = l_Lean_Syntax_isOfKind(v_x_5013_, v___x_5023_);
if (v___x_5024_ == 0)
{
lean_object* v___x_5025_; 
lean_dec(v_x_5013_);
v___x_5025_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__pushStx__1_spec__0___redArg();
return v___x_5025_;
}
else
{
lean_object* v___x_5026_; lean_object* v___x_5027_; lean_object* v___x_5028_; lean_object* v_head_5029_; lean_object* v___y_5031_; lean_object* v___x_5034_; 
v___x_5026_ = lean_unsigned_to_nat(1u);
v___x_5027_ = l_Lean_Syntax_getArg(v_x_5013_, v___x_5026_);
v___x_5028_ = lean_unsigned_to_nat(2u);
v_head_5029_ = l_Lean_Syntax_getArg(v_x_5013_, v___x_5028_);
lean_dec(v_x_5013_);
v___x_5034_ = l_Lean_Syntax_getOptional_x3f(v___x_5027_);
lean_dec(v___x_5027_);
if (lean_obj_tag(v___x_5034_) == 0)
{
lean_object* v___x_5035_; 
v___x_5035_ = lean_box(0);
v___y_5031_ = v___x_5035_;
goto v___jp_5030_;
}
else
{
lean_object* v_val_5036_; lean_object* v___x_5038_; uint8_t v_isShared_5039_; uint8_t v_isSharedCheck_5043_; 
v_val_5036_ = lean_ctor_get(v___x_5034_, 0);
v_isSharedCheck_5043_ = !lean_is_exclusive(v___x_5034_);
if (v_isSharedCheck_5043_ == 0)
{
v___x_5038_ = v___x_5034_;
v_isShared_5039_ = v_isSharedCheck_5043_;
goto v_resetjp_5037_;
}
else
{
lean_inc(v_val_5036_);
lean_dec(v___x_5034_);
v___x_5038_ = lean_box(0);
v_isShared_5039_ = v_isSharedCheck_5043_;
goto v_resetjp_5037_;
}
v_resetjp_5037_:
{
lean_object* v___x_5041_; 
if (v_isShared_5039_ == 0)
{
v___x_5041_ = v___x_5038_;
goto v_reusejp_5040_;
}
else
{
lean_object* v_reuseFailAlloc_5042_; 
v_reuseFailAlloc_5042_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5042_, 0, v_val_5036_);
v___x_5041_ = v_reuseFailAlloc_5042_;
goto v_reusejp_5040_;
}
v_reusejp_5040_:
{
v___y_5031_ = v___x_5041_;
goto v___jp_5030_;
}
}
}
v___jp_5030_:
{
lean_object* v___f_5032_; lean_object* v___x_5033_; 
v___f_5032_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_5032_, 0, v_head_5029_);
lean_closure_set(v___f_5032_, 1, v___y_5031_);
v___x_5033_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5032_, v_a_5014_, v_a_5015_, v_a_5016_, v_a_5017_, v_a_5018_, v_a_5019_, v_a_5020_, v_a_5021_);
return v___x_5033_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1___boxed(lean_object* v_x_5044_, lean_object* v_a_5045_, lean_object* v_a_5046_, lean_object* v_a_5047_, lean_object* v_a_5048_, lean_object* v_a_5049_, lean_object* v_a_5050_, lean_object* v_a_5051_, lean_object* v_a_5052_, lean_object* v_a_5053_){
_start:
{
lean_object* v_res_5054_; 
v_res_5054_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPull__________1(v_x_5044_, v_a_5045_, v_a_5046_, v_a_5047_, v_a_5048_, v_a_5049_, v_a_5050_, v_a_5051_, v_a_5052_);
lean_dec(v_a_5052_);
lean_dec_ref(v_a_5051_);
lean_dec(v_a_5050_);
lean_dec_ref(v_a_5049_);
lean_dec(v_a_5048_);
lean_dec_ref(v_a_5047_);
lean_dec(v_a_5046_);
lean_dec_ref(v_a_5045_);
return v_res_5054_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4(void){
_start:
{
lean_object* v___x_5064_; lean_object* v___x_5065_; lean_object* v___x_5066_; lean_object* v___x_5067_; 
v___x_5064_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8, &lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__8);
v___x_5065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__3));
v___x_5066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_5067_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5067_, 0, v___x_5066_);
lean_ctor_set(v___x_5067_, 1, v___x_5065_);
lean_ctor_set(v___x_5067_, 2, v___x_5064_);
return v___x_5067_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5(void){
_start:
{
lean_object* v___x_5068_; lean_object* v___x_5069_; lean_object* v___x_5070_; lean_object* v___x_5071_; 
v___x_5068_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__8));
v___x_5069_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__4);
v___x_5070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_5071_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5071_, 0, v___x_5070_);
lean_ctor_set(v___x_5071_, 1, v___x_5069_);
lean_ctor_set(v___x_5071_, 2, v___x_5068_);
return v___x_5071_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6(void){
_start:
{
lean_object* v___x_5072_; lean_object* v___x_5073_; lean_object* v___x_5074_; lean_object* v___x_5075_; 
v___x_5072_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18));
v___x_5073_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__5);
v___x_5074_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_5075_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5075_, 0, v___x_5074_);
lean_ctor_set(v___x_5075_, 1, v___x_5073_);
lean_ctor_set(v___x_5075_, 2, v___x_5072_);
return v___x_5075_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7(void){
_start:
{
lean_object* v___x_5076_; lean_object* v___x_5077_; lean_object* v___x_5078_; lean_object* v___x_5079_; 
v___x_5076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushCommand___closed__12));
v___x_5077_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__6);
v___x_5078_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_5079_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5079_, 0, v___x_5078_);
lean_ctor_set(v___x_5079_, 1, v___x_5077_);
lean_ctor_set(v___x_5079_, 2, v___x_5076_);
return v___x_5079_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8(void){
_start:
{
lean_object* v___x_5080_; lean_object* v___x_5081_; lean_object* v___x_5082_; lean_object* v___x_5083_; 
v___x_5080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__18));
v___x_5081_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__7);
v___x_5082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pushStx___closed__3));
v___x_5083_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5083_, 0, v___x_5082_);
lean_ctor_set(v___x_5083_, 1, v___x_5081_);
lean_ctor_set(v___x_5083_, 2, v___x_5080_);
return v___x_5083_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9(void){
_start:
{
lean_object* v___x_5084_; lean_object* v___x_5085_; lean_object* v___x_5086_; lean_object* v___x_5087_; 
v___x_5084_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__8);
v___x_5085_ = lean_unsigned_to_nat(1022u);
v___x_5086_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1));
v___x_5087_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_5087_, 0, v___x_5086_);
lean_ctor_set(v___x_5087_, 1, v___x_5085_);
lean_ctor_set(v___x_5087_, 2, v___x_5084_);
return v___x_5087_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand(void){
_start:
{
lean_object* v___x_5088_; 
v___x_5088_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9, &lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__9);
return v___x_5088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pullCommand__1(lean_object* v_x_5089_, lean_object* v_a_5090_, lean_object* v_a_5091_){
_start:
{
lean_object* v___x_5092_; uint8_t v___x_5093_; 
v___x_5092_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pullCommand___closed__1));
lean_inc(v_x_5089_);
v___x_5093_ = l_Lean_Syntax_isOfKind(v_x_5089_, v___x_5092_);
if (v___x_5093_ == 0)
{
lean_object* v___x_5094_; lean_object* v___x_5095_; 
lean_dec(v_x_5089_);
v___x_5094_ = lean_box(1);
v___x_5095_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5095_, 0, v___x_5094_);
lean_ctor_set(v___x_5095_, 1, v_a_5091_);
return v___x_5095_;
}
else
{
lean_object* v___x_5096_; lean_object* v_tk_5097_; lean_object* v___x_5098_; lean_object* v___x_5099_; lean_object* v___x_5100_; lean_object* v___x_5101_; lean_object* v___x_5102_; lean_object* v___x_5103_; lean_object* v___y_5105_; lean_object* v___y_5106_; lean_object* v___y_5107_; lean_object* v___y_5108_; lean_object* v___y_5109_; lean_object* v___y_5110_; lean_object* v___y_5111_; lean_object* v___y_5112_; lean_object* v___y_5121_; lean_object* v___x_5137_; 
v___x_5096_ = lean_unsigned_to_nat(0u);
v_tk_5097_ = l_Lean_Syntax_getArg(v_x_5089_, v___x_5096_);
v___x_5098_ = lean_unsigned_to_nat(1u);
v___x_5099_ = l_Lean_Syntax_getArg(v_x_5089_, v___x_5098_);
v___x_5100_ = lean_unsigned_to_nat(3u);
v___x_5101_ = l_Lean_Syntax_getArg(v_x_5089_, v___x_5100_);
v___x_5102_ = lean_unsigned_to_nat(5u);
v___x_5103_ = l_Lean_Syntax_getArg(v_x_5089_, v___x_5102_);
lean_dec(v_x_5089_);
v___x_5137_ = l_Lean_Syntax_getOptional_x3f(v___x_5099_);
lean_dec(v___x_5099_);
if (lean_obj_tag(v___x_5137_) == 0)
{
lean_object* v___x_5138_; 
v___x_5138_ = lean_box(0);
v___y_5121_ = v___x_5138_;
goto v___jp_5120_;
}
else
{
lean_object* v_val_5139_; lean_object* v___x_5141_; uint8_t v_isShared_5142_; uint8_t v_isSharedCheck_5146_; 
v_val_5139_ = lean_ctor_get(v___x_5137_, 0);
v_isSharedCheck_5146_ = !lean_is_exclusive(v___x_5137_);
if (v_isSharedCheck_5146_ == 0)
{
v___x_5141_ = v___x_5137_;
v_isShared_5142_ = v_isSharedCheck_5146_;
goto v_resetjp_5140_;
}
else
{
lean_inc(v_val_5139_);
lean_dec(v___x_5137_);
v___x_5141_ = lean_box(0);
v_isShared_5142_ = v_isSharedCheck_5146_;
goto v_resetjp_5140_;
}
v_resetjp_5140_:
{
lean_object* v___x_5144_; 
if (v_isShared_5142_ == 0)
{
v___x_5144_ = v___x_5141_;
goto v_reusejp_5143_;
}
else
{
lean_object* v_reuseFailAlloc_5145_; 
v_reuseFailAlloc_5145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5145_, 0, v_val_5139_);
v___x_5144_ = v_reuseFailAlloc_5145_;
goto v_reusejp_5143_;
}
v_reusejp_5143_:
{
v___y_5121_ = v___x_5144_;
goto v___jp_5120_;
}
}
}
v___jp_5104_:
{
lean_object* v___x_5113_; lean_object* v___x_5114_; lean_object* v___x_5115_; lean_object* v___x_5116_; lean_object* v___x_5117_; lean_object* v___x_5118_; lean_object* v___x_5119_; 
lean_inc_ref(v___y_5107_);
v___x_5113_ = l_Array_append___redArg(v___y_5107_, v___y_5112_);
lean_dec_ref(v___y_5112_);
lean_inc(v___y_5106_);
lean_inc_n(v___y_5105_, 3);
v___x_5114_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5114_, 0, v___y_5105_);
lean_ctor_set(v___x_5114_, 1, v___y_5106_);
lean_ctor_set(v___x_5114_, 2, v___x_5113_);
lean_inc(v___y_5110_);
v___x_5115_ = l_Lean_Syntax_node3(v___y_5105_, v___y_5110_, v___y_5109_, v___x_5114_, v___x_5101_);
v___x_5116_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__0));
v___x_5117_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5117_, 0, v___y_5105_);
lean_ctor_set(v___x_5117_, 1, v___x_5116_);
lean_inc(v___y_5108_);
v___x_5118_ = l_Lean_Syntax_node4(v___y_5105_, v___y_5108_, v___y_5111_, v___x_5115_, v___x_5117_, v___x_5103_);
v___x_5119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5119_, 0, v___x_5118_);
lean_ctor_set(v___x_5119_, 1, v_a_5091_);
return v___x_5119_;
}
v___jp_5120_:
{
lean_object* v_ref_5122_; uint8_t v___x_5123_; lean_object* v___x_5124_; lean_object* v___x_5125_; lean_object* v___x_5126_; lean_object* v___x_5127_; lean_object* v___x_5128_; lean_object* v___x_5129_; lean_object* v___x_5130_; lean_object* v___x_5131_; lean_object* v___x_5132_; lean_object* v___x_5133_; 
v_ref_5122_ = lean_ctor_get(v_a_5090_, 5);
v___x_5123_ = 0;
v___x_5124_ = l_Lean_SourceInfo_fromRef(v_ref_5122_, v___x_5123_);
v___x_5125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__3));
v___x_5126_ = l_Lean_SourceInfo_fromRef(v_tk_5097_, v___x_5093_);
lean_dec(v_tk_5097_);
v___x_5127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__4));
v___x_5128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5128_, 0, v___x_5126_);
lean_ctor_set(v___x_5128_, 1, v___x_5127_);
v___x_5129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_convPull_________00__closed__1));
v___x_5130_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_pull___closed__0));
lean_inc(v___x_5124_);
v___x_5131_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5131_, 0, v___x_5124_);
lean_ctor_set(v___x_5131_, 1, v___x_5130_);
v___x_5132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__4));
v___x_5133_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5, &lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__convPush__neg____1___closed__5);
if (lean_obj_tag(v___y_5121_) == 1)
{
lean_object* v_val_5134_; lean_object* v___x_5135_; 
v_val_5134_ = lean_ctor_get(v___y_5121_, 0);
lean_inc(v_val_5134_);
lean_dec_ref_known(v___y_5121_, 1);
v___x_5135_ = l_Array_mkArray1___redArg(v_val_5134_);
v___y_5105_ = v___x_5124_;
v___y_5106_ = v___x_5132_;
v___y_5107_ = v___x_5133_;
v___y_5108_ = v___x_5125_;
v___y_5109_ = v___x_5131_;
v___y_5110_ = v___x_5129_;
v___y_5111_ = v___x_5128_;
v___y_5112_ = v___x_5135_;
goto v___jp_5104_;
}
else
{
lean_object* v___x_5136_; 
lean_dec(v___y_5121_);
v___x_5136_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pushCommand__1___closed__5));
v___y_5105_ = v___x_5124_;
v___y_5106_ = v___x_5132_;
v___y_5107_ = v___x_5133_;
v___y_5108_ = v___x_5125_;
v___y_5109_ = v___x_5131_;
v___y_5110_ = v___x_5129_;
v___y_5111_ = v___x_5128_;
v___y_5112_ = v___x_5136_;
goto v___jp_5104_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pullCommand__1___boxed(lean_object* v_x_5147_, lean_object* v_a_5148_, lean_object* v_a_5149_){
_start:
{
lean_object* v_res_5150_; 
v_res_5150_ = lp_mathlib_Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______macroRules__Mathlib__Tactic__Push__pullCommand__1(v_x_5147_, v_a_5148_, v_a_5149_);
lean_dec_ref(v_a_5148_);
return v_res_5150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6(lean_object* v_x_5177_, lean_object* v_x_5178_, lean_object* v_x_5179_){
_start:
{
if (lean_obj_tag(v_x_5179_) == 0)
{
lean_dec(v_x_5177_);
return v_x_5178_;
}
else
{
lean_object* v_head_5180_; lean_object* v_tail_5181_; lean_object* v___x_5183_; uint8_t v_isShared_5184_; uint8_t v_isSharedCheck_5208_; 
v_head_5180_ = lean_ctor_get(v_x_5179_, 0);
v_tail_5181_ = lean_ctor_get(v_x_5179_, 1);
v_isSharedCheck_5208_ = !lean_is_exclusive(v_x_5179_);
if (v_isSharedCheck_5208_ == 0)
{
v___x_5183_ = v_x_5179_;
v_isShared_5184_ = v_isSharedCheck_5208_;
goto v_resetjp_5182_;
}
else
{
lean_inc(v_tail_5181_);
lean_inc(v_head_5180_);
lean_dec(v_x_5179_);
v___x_5183_ = lean_box(0);
v_isShared_5184_ = v_isSharedCheck_5208_;
goto v_resetjp_5182_;
}
v_resetjp_5182_:
{
lean_object* v_priority_5185_; uint8_t v_perm_5186_; lean_object* v_origin_5187_; lean_object* v___x_5189_; 
v_priority_5185_ = lean_ctor_get(v_head_5180_, 3);
lean_inc(v_priority_5185_);
v_perm_5186_ = lean_ctor_get_uint8(v_head_5180_, sizeof(void*)*5 + 1);
v_origin_5187_ = lean_ctor_get(v_head_5180_, 4);
lean_inc_ref(v_origin_5187_);
lean_dec(v_head_5180_);
lean_inc(v_x_5177_);
if (v_isShared_5184_ == 0)
{
lean_ctor_set_tag(v___x_5183_, 5);
lean_ctor_set(v___x_5183_, 1, v_x_5177_);
lean_ctor_set(v___x_5183_, 0, v_x_5178_);
v___x_5189_ = v___x_5183_;
goto v_reusejp_5188_;
}
else
{
lean_object* v_reuseFailAlloc_5207_; 
v_reuseFailAlloc_5207_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5207_, 0, v_x_5178_);
lean_ctor_set(v_reuseFailAlloc_5207_, 1, v_x_5177_);
v___x_5189_ = v_reuseFailAlloc_5207_;
goto v_reusejp_5188_;
}
v_reusejp_5188_:
{
lean_object* v___y_5191_; 
if (v_perm_5186_ == 0)
{
lean_object* v___x_5205_; 
v___x_5205_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
v___y_5191_ = v___x_5205_;
goto v___jp_5190_;
}
else
{
lean_object* v___x_5206_; 
v___x_5206_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2));
v___y_5191_ = v___x_5206_;
goto v___jp_5190_;
}
v___jp_5190_:
{
lean_object* v___x_5192_; uint8_t v___x_5193_; lean_object* v___x_5194_; lean_object* v_name_5195_; lean_object* v___x_5196_; lean_object* v___x_5197_; lean_object* v___x_5198_; lean_object* v_prio_5199_; lean_object* v___x_5200_; lean_object* v___x_5201_; lean_object* v___x_5202_; lean_object* v___x_5203_; 
v___x_5192_ = l_Lean_Meta_Origin_key(v_origin_5187_);
lean_dec_ref(v_origin_5187_);
v___x_5193_ = 1;
v___x_5194_ = l_Lean_Name_toString(v___x_5192_, v___x_5193_);
v_name_5195_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_name_5195_, 0, v___x_5194_);
v___x_5196_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1));
v___x_5197_ = l_Nat_reprFast(v_priority_5185_);
v___x_5198_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5198_, 0, v___x_5197_);
v_prio_5199_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_prio_5199_, 0, v___x_5196_);
lean_ctor_set(v_prio_5199_, 1, v___x_5198_);
v___x_5200_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5200_, 0, v_name_5195_);
lean_ctor_set(v___x_5200_, 1, v_prio_5199_);
lean_inc_ref(v___y_5191_);
v___x_5201_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5201_, 0, v___y_5191_);
v___x_5202_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5202_, 0, v___x_5200_);
lean_ctor_set(v___x_5202_, 1, v___x_5201_);
v___x_5203_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5203_, 0, v___x_5189_);
lean_ctor_set(v___x_5203_, 1, v___x_5202_);
v_x_5178_ = v___x_5203_;
v_x_5179_ = v_tail_5181_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3(lean_object* v_x_5209_, lean_object* v_x_5210_){
_start:
{
if (lean_obj_tag(v_x_5209_) == 0)
{
lean_object* v___x_5211_; 
lean_dec(v_x_5210_);
v___x_5211_ = lean_box(0);
return v___x_5211_;
}
else
{
lean_object* v_tail_5212_; 
v_tail_5212_ = lean_ctor_get(v_x_5209_, 1);
if (lean_obj_tag(v_tail_5212_) == 0)
{
lean_object* v_head_5213_; lean_object* v___x_5215_; uint8_t v_isShared_5216_; uint8_t v_isSharedCheck_5237_; 
lean_dec(v_x_5210_);
v_head_5213_ = lean_ctor_get(v_x_5209_, 0);
v_isSharedCheck_5237_ = !lean_is_exclusive(v_x_5209_);
if (v_isSharedCheck_5237_ == 0)
{
lean_object* v_unused_5238_; 
v_unused_5238_ = lean_ctor_get(v_x_5209_, 1);
lean_dec(v_unused_5238_);
v___x_5215_ = v_x_5209_;
v_isShared_5216_ = v_isSharedCheck_5237_;
goto v_resetjp_5214_;
}
else
{
lean_inc(v_head_5213_);
lean_dec(v_x_5209_);
v___x_5215_ = lean_box(0);
v_isShared_5216_ = v_isSharedCheck_5237_;
goto v_resetjp_5214_;
}
v_resetjp_5214_:
{
lean_object* v_priority_5217_; uint8_t v_perm_5218_; lean_object* v_origin_5219_; lean_object* v___y_5221_; 
v_priority_5217_ = lean_ctor_get(v_head_5213_, 3);
lean_inc(v_priority_5217_);
v_perm_5218_ = lean_ctor_get_uint8(v_head_5213_, sizeof(void*)*5 + 1);
v_origin_5219_ = lean_ctor_get(v_head_5213_, 4);
lean_inc_ref(v_origin_5219_);
lean_dec(v_head_5213_);
if (v_perm_5218_ == 0)
{
lean_object* v___x_5235_; 
v___x_5235_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
v___y_5221_ = v___x_5235_;
goto v___jp_5220_;
}
else
{
lean_object* v___x_5236_; 
v___x_5236_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2));
v___y_5221_ = v___x_5236_;
goto v___jp_5220_;
}
v___jp_5220_:
{
lean_object* v___x_5222_; uint8_t v___x_5223_; lean_object* v___x_5224_; lean_object* v_name_5225_; lean_object* v___x_5226_; lean_object* v___x_5227_; lean_object* v___x_5228_; lean_object* v_prio_5230_; 
v___x_5222_ = l_Lean_Meta_Origin_key(v_origin_5219_);
lean_dec_ref(v_origin_5219_);
v___x_5223_ = 1;
v___x_5224_ = l_Lean_Name_toString(v___x_5222_, v___x_5223_);
v_name_5225_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_name_5225_, 0, v___x_5224_);
v___x_5226_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1));
v___x_5227_ = l_Nat_reprFast(v_priority_5217_);
v___x_5228_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5228_, 0, v___x_5227_);
if (v_isShared_5216_ == 0)
{
lean_ctor_set_tag(v___x_5215_, 5);
lean_ctor_set(v___x_5215_, 1, v___x_5228_);
lean_ctor_set(v___x_5215_, 0, v___x_5226_);
v_prio_5230_ = v___x_5215_;
goto v_reusejp_5229_;
}
else
{
lean_object* v_reuseFailAlloc_5234_; 
v_reuseFailAlloc_5234_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5234_, 0, v___x_5226_);
lean_ctor_set(v_reuseFailAlloc_5234_, 1, v___x_5228_);
v_prio_5230_ = v_reuseFailAlloc_5234_;
goto v_reusejp_5229_;
}
v_reusejp_5229_:
{
lean_object* v___x_5231_; lean_object* v___x_5232_; lean_object* v___x_5233_; 
v___x_5231_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5231_, 0, v_name_5225_);
lean_ctor_set(v___x_5231_, 1, v_prio_5230_);
lean_inc_ref(v___y_5221_);
v___x_5232_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5232_, 0, v___y_5221_);
v___x_5233_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5233_, 0, v___x_5231_);
lean_ctor_set(v___x_5233_, 1, v___x_5232_);
return v___x_5233_;
}
}
}
}
else
{
lean_object* v_head_5239_; lean_object* v___x_5241_; uint8_t v_isShared_5242_; uint8_t v_isSharedCheck_5264_; 
lean_inc(v_tail_5212_);
v_head_5239_ = lean_ctor_get(v_x_5209_, 0);
v_isSharedCheck_5264_ = !lean_is_exclusive(v_x_5209_);
if (v_isSharedCheck_5264_ == 0)
{
lean_object* v_unused_5265_; 
v_unused_5265_ = lean_ctor_get(v_x_5209_, 1);
lean_dec(v_unused_5265_);
v___x_5241_ = v_x_5209_;
v_isShared_5242_ = v_isSharedCheck_5264_;
goto v_resetjp_5240_;
}
else
{
lean_inc(v_head_5239_);
lean_dec(v_x_5209_);
v___x_5241_ = lean_box(0);
v_isShared_5242_ = v_isSharedCheck_5264_;
goto v_resetjp_5240_;
}
v_resetjp_5240_:
{
lean_object* v_priority_5243_; uint8_t v_perm_5244_; lean_object* v_origin_5245_; lean_object* v___y_5247_; 
v_priority_5243_ = lean_ctor_get(v_head_5239_, 3);
lean_inc(v_priority_5243_);
v_perm_5244_ = lean_ctor_get_uint8(v_head_5239_, sizeof(void*)*5 + 1);
v_origin_5245_ = lean_ctor_get(v_head_5239_, 4);
lean_inc_ref(v_origin_5245_);
lean_dec(v_head_5239_);
if (v_perm_5244_ == 0)
{
lean_object* v___x_5262_; 
v___x_5262_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
v___y_5247_ = v___x_5262_;
goto v___jp_5246_;
}
else
{
lean_object* v___x_5263_; 
v___x_5263_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__2));
v___y_5247_ = v___x_5263_;
goto v___jp_5246_;
}
v___jp_5246_:
{
lean_object* v___x_5248_; uint8_t v___x_5249_; lean_object* v___x_5250_; lean_object* v_name_5251_; lean_object* v___x_5252_; lean_object* v___x_5253_; lean_object* v___x_5254_; lean_object* v_prio_5256_; 
v___x_5248_ = l_Lean_Meta_Origin_key(v_origin_5245_);
lean_dec_ref(v_origin_5245_);
v___x_5249_ = 1;
v___x_5250_ = l_Lean_Name_toString(v___x_5248_, v___x_5249_);
v_name_5251_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_name_5251_, 0, v___x_5250_);
v___x_5252_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__1));
v___x_5253_ = l_Nat_reprFast(v_priority_5243_);
v___x_5254_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5254_, 0, v___x_5253_);
if (v_isShared_5242_ == 0)
{
lean_ctor_set_tag(v___x_5241_, 5);
lean_ctor_set(v___x_5241_, 1, v___x_5254_);
lean_ctor_set(v___x_5241_, 0, v___x_5252_);
v_prio_5256_ = v___x_5241_;
goto v_reusejp_5255_;
}
else
{
lean_object* v_reuseFailAlloc_5261_; 
v_reuseFailAlloc_5261_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5261_, 0, v___x_5252_);
lean_ctor_set(v_reuseFailAlloc_5261_, 1, v___x_5254_);
v_prio_5256_ = v_reuseFailAlloc_5261_;
goto v_reusejp_5255_;
}
v_reusejp_5255_:
{
lean_object* v___x_5257_; lean_object* v___x_5258_; lean_object* v___x_5259_; lean_object* v___x_5260_; 
v___x_5257_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5257_, 0, v_name_5251_);
lean_ctor_set(v___x_5257_, 1, v_prio_5256_);
lean_inc_ref(v___y_5247_);
v___x_5258_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5258_, 0, v___y_5247_);
v___x_5259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5259_, 0, v___x_5257_);
lean_ctor_set(v___x_5259_, 1, v___x_5258_);
v___x_5260_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6(v_x_5210_, v___x_5259_, v_tail_5212_);
return v___x_5260_;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7(void){
_start:
{
lean_object* v___x_5277_; lean_object* v___x_5278_; 
v___x_5277_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__5));
v___x_5278_ = lean_string_length(v___x_5277_);
return v___x_5278_;
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8(void){
_start:
{
lean_object* v___x_5279_; lean_object* v___x_5280_; 
v___x_5279_ = lean_obj_once(&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7, &lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7_once, _init_lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__7);
v___x_5280_ = lean_nat_to_int(v___x_5279_);
return v___x_5280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2(lean_object* v_x_5285_){
_start:
{
if (lean_obj_tag(v_x_5285_) == 0)
{
lean_object* v___x_5286_; 
v___x_5286_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__1));
return v___x_5286_;
}
else
{
lean_object* v___x_5287_; lean_object* v___x_5288_; lean_object* v___x_5289_; lean_object* v___x_5290_; lean_object* v___x_5291_; lean_object* v___x_5292_; lean_object* v___x_5293_; lean_object* v___x_5294_; uint8_t v___x_5295_; lean_object* v___x_5296_; 
v___x_5287_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__4));
v___x_5288_ = lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3(v_x_5285_, v___x_5287_);
v___x_5289_ = lean_obj_once(&lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8, &lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8_once, _init_lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__8);
v___x_5290_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__9));
v___x_5291_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5291_, 0, v___x_5290_);
lean_ctor_set(v___x_5291_, 1, v___x_5288_);
v___x_5292_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2___closed__10));
v___x_5293_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5293_, 0, v___x_5291_);
lean_ctor_set(v___x_5293_, 1, v___x_5292_);
v___x_5294_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_5294_, 0, v___x_5289_);
lean_ctor_set(v___x_5294_, 1, v___x_5293_);
v___x_5295_ = 0;
v___x_5296_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_5296_, 0, v___x_5294_);
lean_ctor_set_uint8(v___x_5296_, sizeof(void*)*1, v___x_5295_);
return v___x_5296_;
}
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_5301_; lean_object* v___x_5302_; 
v___x_5301_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__1));
v___x_5302_ = lean_string_length(v___x_5301_);
return v___x_5302_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4(void){
_start:
{
lean_object* v___x_5303_; lean_object* v___x_5304_; 
v___x_5303_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3, &lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3_once, _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__3);
v___x_5304_ = lean_nat_to_int(v___x_5303_);
return v___x_5304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0(lean_object* v_x_5317_){
_start:
{
lean_object* v_vs_5318_; lean_object* v_children_5319_; lean_object* v___x_5321_; uint8_t v_isShared_5322_; uint8_t v_isSharedCheck_5353_; 
v_vs_5318_ = lean_ctor_get(v_x_5317_, 0);
v_children_5319_ = lean_ctor_get(v_x_5317_, 1);
v_isSharedCheck_5353_ = !lean_is_exclusive(v_x_5317_);
if (v_isSharedCheck_5353_ == 0)
{
v___x_5321_ = v_x_5317_;
v_isShared_5322_ = v_isSharedCheck_5353_;
goto v_resetjp_5320_;
}
else
{
lean_inc(v_children_5319_);
lean_inc(v_vs_5318_);
lean_dec(v_x_5317_);
v___x_5321_ = lean_box(0);
v_isShared_5322_ = v_isSharedCheck_5353_;
goto v_resetjp_5320_;
}
v_resetjp_5320_:
{
lean_object* v___x_5323_; lean_object* v___y_5325_; lean_object* v___x_5343_; lean_object* v___x_5344_; uint8_t v___x_5345_; 
v___x_5323_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__1));
v___x_5343_ = lean_array_get_size(v_vs_5318_);
v___x_5344_ = lean_unsigned_to_nat(0u);
v___x_5345_ = lean_nat_dec_eq(v___x_5343_, v___x_5344_);
if (v___x_5345_ == 0)
{
lean_object* v___x_5346_; lean_object* v___x_5347_; lean_object* v___x_5348_; lean_object* v___x_5349_; lean_object* v___x_5350_; lean_object* v___x_5351_; 
v___x_5346_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__2));
v___x_5347_ = ((lean_object*)(lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0___closed__4));
v___x_5348_ = lean_array_to_list(v_vs_5318_);
v___x_5349_ = lp_mathlib_List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2(v___x_5348_);
v___x_5350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5350_, 0, v___x_5347_);
lean_ctor_set(v___x_5350_, 1, v___x_5349_);
v___x_5351_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5351_, 0, v___x_5346_);
lean_ctor_set(v___x_5351_, 1, v___x_5350_);
v___y_5325_ = v___x_5351_;
goto v___jp_5324_;
}
else
{
lean_object* v___x_5352_; 
lean_dec_ref(v_vs_5318_);
v___x_5352_ = lean_box(0);
v___y_5325_ = v___x_5352_;
goto v___jp_5324_;
}
v___jp_5324_:
{
lean_object* v___x_5327_; 
if (v_isShared_5322_ == 0)
{
lean_ctor_set_tag(v___x_5321_, 5);
lean_ctor_set(v___x_5321_, 1, v___y_5325_);
lean_ctor_set(v___x_5321_, 0, v___x_5323_);
v___x_5327_ = v___x_5321_;
goto v_reusejp_5326_;
}
else
{
lean_object* v_reuseFailAlloc_5342_; 
v_reuseFailAlloc_5342_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5342_, 0, v___x_5323_);
lean_ctor_set(v_reuseFailAlloc_5342_, 1, v___y_5325_);
v___x_5327_ = v_reuseFailAlloc_5342_;
goto v_reusejp_5326_;
}
v_reusejp_5326_:
{
lean_object* v___x_5328_; lean_object* v___x_5329_; lean_object* v___x_5330_; lean_object* v___x_5331_; lean_object* v___x_5332_; lean_object* v___x_5333_; lean_object* v___x_5334_; lean_object* v___x_5335_; lean_object* v___x_5336_; lean_object* v___x_5337_; lean_object* v___x_5338_; uint8_t v___x_5339_; lean_object* v___x_5340_; lean_object* v___x_5341_; 
v___x_5328_ = lean_array_to_list(v_children_5319_);
v___x_5329_ = lean_box(0);
v___x_5330_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1(v___x_5328_, v___x_5329_);
v___x_5331_ = l_Std_Format_join(v___x_5330_);
v___x_5332_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5332_, 0, v___x_5327_);
lean_ctor_set(v___x_5332_, 1, v___x_5331_);
v___x_5333_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4, &lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4_once, _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4);
v___x_5334_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__5));
v___x_5335_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5335_, 0, v___x_5334_);
lean_ctor_set(v___x_5335_, 1, v___x_5332_);
v___x_5336_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__6));
v___x_5337_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5337_, 0, v___x_5335_);
lean_ctor_set(v___x_5337_, 1, v___x_5336_);
v___x_5338_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_5338_, 0, v___x_5333_);
lean_ctor_set(v___x_5338_, 1, v___x_5337_);
v___x_5339_ = 0;
v___x_5340_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_5340_, 0, v___x_5338_);
lean_ctor_set_uint8(v___x_5340_, sizeof(void*)*1, v___x_5339_);
v___x_5341_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_5341_, 0, v___x_5340_);
lean_ctor_set_uint8(v___x_5341_, sizeof(void*)*1, v___x_5339_);
return v___x_5341_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1(lean_object* v_a_5354_, lean_object* v_a_5355_){
_start:
{
if (lean_obj_tag(v_a_5354_) == 0)
{
lean_object* v___x_5356_; 
v___x_5356_ = l_List_reverse___redArg(v_a_5355_);
return v___x_5356_;
}
else
{
lean_object* v_head_5357_; lean_object* v_tail_5358_; lean_object* v___x_5360_; uint8_t v_isShared_5361_; uint8_t v_isSharedCheck_5389_; 
v_head_5357_ = lean_ctor_get(v_a_5354_, 0);
v_tail_5358_ = lean_ctor_get(v_a_5354_, 1);
v_isSharedCheck_5389_ = !lean_is_exclusive(v_a_5354_);
if (v_isSharedCheck_5389_ == 0)
{
v___x_5360_ = v_a_5354_;
v_isShared_5361_ = v_isSharedCheck_5389_;
goto v_resetjp_5359_;
}
else
{
lean_inc(v_tail_5358_);
lean_inc(v_head_5357_);
lean_dec(v_a_5354_);
v___x_5360_ = lean_box(0);
v_isShared_5361_ = v_isSharedCheck_5389_;
goto v_resetjp_5359_;
}
v_resetjp_5359_:
{
lean_object* v_fst_5362_; lean_object* v_snd_5363_; lean_object* v___x_5365_; uint8_t v_isShared_5366_; uint8_t v_isSharedCheck_5388_; 
v_fst_5362_ = lean_ctor_get(v_head_5357_, 0);
v_snd_5363_ = lean_ctor_get(v_head_5357_, 1);
v_isSharedCheck_5388_ = !lean_is_exclusive(v_head_5357_);
if (v_isSharedCheck_5388_ == 0)
{
v___x_5365_ = v_head_5357_;
v_isShared_5366_ = v_isSharedCheck_5388_;
goto v_resetjp_5364_;
}
else
{
lean_inc(v_snd_5363_);
lean_inc(v_fst_5362_);
lean_dec(v_head_5357_);
v___x_5365_ = lean_box(0);
v_isShared_5366_ = v_isSharedCheck_5388_;
goto v_resetjp_5364_;
}
v_resetjp_5364_:
{
lean_object* v___x_5367_; lean_object* v___x_5368_; lean_object* v___x_5369_; lean_object* v___x_5371_; 
v___x_5367_ = lean_box(1);
v___x_5368_ = l_Lean_Meta_DiscrTree_Key_format(v_fst_5362_);
v___x_5369_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__0));
if (v_isShared_5366_ == 0)
{
lean_ctor_set_tag(v___x_5365_, 5);
lean_ctor_set(v___x_5365_, 1, v___x_5369_);
lean_ctor_set(v___x_5365_, 0, v___x_5368_);
v___x_5371_ = v___x_5365_;
goto v_reusejp_5370_;
}
else
{
lean_object* v_reuseFailAlloc_5387_; 
v_reuseFailAlloc_5387_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5387_, 0, v___x_5368_);
lean_ctor_set(v_reuseFailAlloc_5387_, 1, v___x_5369_);
v___x_5371_ = v_reuseFailAlloc_5387_;
goto v_reusejp_5370_;
}
v_reusejp_5370_:
{
lean_object* v___x_5372_; lean_object* v___x_5373_; lean_object* v___x_5374_; lean_object* v___x_5375_; lean_object* v___x_5376_; lean_object* v___x_5377_; lean_object* v___x_5378_; lean_object* v___x_5379_; uint8_t v___x_5380_; lean_object* v___x_5381_; lean_object* v___x_5382_; lean_object* v___x_5384_; 
v___x_5372_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0(v_snd_5363_);
v___x_5373_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5373_, 0, v___x_5371_);
lean_ctor_set(v___x_5373_, 1, v___x_5372_);
v___x_5374_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4, &lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4_once, _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__4);
v___x_5375_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__5));
v___x_5376_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5376_, 0, v___x_5375_);
lean_ctor_set(v___x_5376_, 1, v___x_5373_);
v___x_5377_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__1___closed__6));
v___x_5378_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5378_, 0, v___x_5376_);
lean_ctor_set(v___x_5378_, 1, v___x_5377_);
v___x_5379_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_5379_, 0, v___x_5374_);
lean_ctor_set(v___x_5379_, 1, v___x_5378_);
v___x_5380_ = 0;
v___x_5381_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_5381_, 0, v___x_5379_);
lean_ctor_set_uint8(v___x_5381_, sizeof(void*)*1, v___x_5380_);
v___x_5382_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_5382_, 0, v___x_5367_);
lean_ctor_set(v___x_5382_, 1, v___x_5381_);
if (v_isShared_5361_ == 0)
{
lean_ctor_set(v___x_5360_, 1, v_a_5355_);
lean_ctor_set(v___x_5360_, 0, v___x_5382_);
v___x_5384_ = v___x_5360_;
goto v_reusejp_5383_;
}
else
{
lean_object* v_reuseFailAlloc_5386_; 
v_reuseFailAlloc_5386_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5386_, 0, v___x_5382_);
lean_ctor_set(v_reuseFailAlloc_5386_, 1, v_a_5355_);
v___x_5384_ = v_reuseFailAlloc_5386_;
goto v_reusejp_5383_;
}
v_reusejp_5383_:
{
v_a_5354_ = v_tail_5358_;
v_a_5355_ = v___x_5384_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg(lean_object* v_ref_5390_, lean_object* v_msgData_5391_, uint8_t v_severity_5392_, uint8_t v_isSilent_5393_, lean_object* v___y_5394_, lean_object* v___y_5395_, lean_object* v___y_5396_, lean_object* v___y_5397_){
_start:
{
lean_object* v___y_5400_; lean_object* v___y_5401_; lean_object* v___y_5402_; uint8_t v___y_5403_; lean_object* v___y_5404_; uint8_t v___y_5405_; lean_object* v___y_5406_; lean_object* v___y_5407_; lean_object* v___y_5408_; lean_object* v___y_5436_; lean_object* v___y_5437_; lean_object* v___y_5438_; uint8_t v___y_5439_; lean_object* v___y_5440_; uint8_t v___y_5441_; uint8_t v___y_5442_; lean_object* v___y_5443_; lean_object* v___y_5461_; lean_object* v___y_5462_; uint8_t v___y_5463_; lean_object* v___y_5464_; lean_object* v___y_5465_; uint8_t v___y_5466_; uint8_t v___y_5467_; lean_object* v___y_5468_; lean_object* v___y_5472_; lean_object* v___y_5473_; lean_object* v___y_5474_; uint8_t v___y_5475_; lean_object* v___y_5476_; uint8_t v___y_5477_; uint8_t v___y_5478_; uint8_t v___x_5483_; lean_object* v___y_5485_; lean_object* v___y_5486_; lean_object* v___y_5487_; lean_object* v___y_5488_; uint8_t v___y_5489_; uint8_t v___y_5490_; uint8_t v___y_5491_; uint8_t v___y_5493_; uint8_t v___x_5508_; 
v___x_5483_ = 2;
v___x_5508_ = l_Lean_instBEqMessageSeverity_beq(v_severity_5392_, v___x_5483_);
if (v___x_5508_ == 0)
{
v___y_5493_ = v___x_5508_;
goto v___jp_5492_;
}
else
{
uint8_t v___x_5509_; 
lean_inc_ref(v_msgData_5391_);
v___x_5509_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_5391_);
v___y_5493_ = v___x_5509_;
goto v___jp_5492_;
}
v___jp_5399_:
{
lean_object* v___x_5409_; lean_object* v_currNamespace_5410_; lean_object* v_openDecls_5411_; lean_object* v_env_5412_; lean_object* v_nextMacroScope_5413_; lean_object* v_ngen_5414_; lean_object* v_auxDeclNGen_5415_; lean_object* v_traceState_5416_; lean_object* v_cache_5417_; lean_object* v_messages_5418_; lean_object* v_infoState_5419_; lean_object* v_snapshotTasks_5420_; lean_object* v___x_5422_; uint8_t v_isShared_5423_; uint8_t v_isSharedCheck_5434_; 
v___x_5409_ = lean_st_ref_take(v___y_5408_);
v_currNamespace_5410_ = lean_ctor_get(v___y_5407_, 6);
v_openDecls_5411_ = lean_ctor_get(v___y_5407_, 7);
v_env_5412_ = lean_ctor_get(v___x_5409_, 0);
v_nextMacroScope_5413_ = lean_ctor_get(v___x_5409_, 1);
v_ngen_5414_ = lean_ctor_get(v___x_5409_, 2);
v_auxDeclNGen_5415_ = lean_ctor_get(v___x_5409_, 3);
v_traceState_5416_ = lean_ctor_get(v___x_5409_, 4);
v_cache_5417_ = lean_ctor_get(v___x_5409_, 5);
v_messages_5418_ = lean_ctor_get(v___x_5409_, 6);
v_infoState_5419_ = lean_ctor_get(v___x_5409_, 7);
v_snapshotTasks_5420_ = lean_ctor_get(v___x_5409_, 8);
v_isSharedCheck_5434_ = !lean_is_exclusive(v___x_5409_);
if (v_isSharedCheck_5434_ == 0)
{
v___x_5422_ = v___x_5409_;
v_isShared_5423_ = v_isSharedCheck_5434_;
goto v_resetjp_5421_;
}
else
{
lean_inc(v_snapshotTasks_5420_);
lean_inc(v_infoState_5419_);
lean_inc(v_messages_5418_);
lean_inc(v_cache_5417_);
lean_inc(v_traceState_5416_);
lean_inc(v_auxDeclNGen_5415_);
lean_inc(v_ngen_5414_);
lean_inc(v_nextMacroScope_5413_);
lean_inc(v_env_5412_);
lean_dec(v___x_5409_);
v___x_5422_ = lean_box(0);
v_isShared_5423_ = v_isSharedCheck_5434_;
goto v_resetjp_5421_;
}
v_resetjp_5421_:
{
lean_object* v___x_5424_; lean_object* v___x_5425_; lean_object* v___x_5426_; lean_object* v___x_5427_; lean_object* v___x_5429_; 
lean_inc(v_openDecls_5411_);
lean_inc(v_currNamespace_5410_);
v___x_5424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5424_, 0, v_currNamespace_5410_);
lean_ctor_set(v___x_5424_, 1, v_openDecls_5411_);
v___x_5425_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_5425_, 0, v___x_5424_);
lean_ctor_set(v___x_5425_, 1, v___y_5401_);
lean_inc_ref(v___y_5402_);
lean_inc_ref(v___y_5404_);
v___x_5426_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_5426_, 0, v___y_5404_);
lean_ctor_set(v___x_5426_, 1, v___y_5406_);
lean_ctor_set(v___x_5426_, 2, v___y_5400_);
lean_ctor_set(v___x_5426_, 3, v___y_5402_);
lean_ctor_set(v___x_5426_, 4, v___x_5425_);
lean_ctor_set_uint8(v___x_5426_, sizeof(void*)*5, v___y_5403_);
lean_ctor_set_uint8(v___x_5426_, sizeof(void*)*5 + 1, v___y_5405_);
lean_ctor_set_uint8(v___x_5426_, sizeof(void*)*5 + 2, v_isSilent_5393_);
v___x_5427_ = l_Lean_MessageLog_add(v___x_5426_, v_messages_5418_);
if (v_isShared_5423_ == 0)
{
lean_ctor_set(v___x_5422_, 6, v___x_5427_);
v___x_5429_ = v___x_5422_;
goto v_reusejp_5428_;
}
else
{
lean_object* v_reuseFailAlloc_5433_; 
v_reuseFailAlloc_5433_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_5433_, 0, v_env_5412_);
lean_ctor_set(v_reuseFailAlloc_5433_, 1, v_nextMacroScope_5413_);
lean_ctor_set(v_reuseFailAlloc_5433_, 2, v_ngen_5414_);
lean_ctor_set(v_reuseFailAlloc_5433_, 3, v_auxDeclNGen_5415_);
lean_ctor_set(v_reuseFailAlloc_5433_, 4, v_traceState_5416_);
lean_ctor_set(v_reuseFailAlloc_5433_, 5, v_cache_5417_);
lean_ctor_set(v_reuseFailAlloc_5433_, 6, v___x_5427_);
lean_ctor_set(v_reuseFailAlloc_5433_, 7, v_infoState_5419_);
lean_ctor_set(v_reuseFailAlloc_5433_, 8, v_snapshotTasks_5420_);
v___x_5429_ = v_reuseFailAlloc_5433_;
goto v_reusejp_5428_;
}
v_reusejp_5428_:
{
lean_object* v___x_5430_; lean_object* v___x_5431_; lean_object* v___x_5432_; 
v___x_5430_ = lean_st_ref_set(v___y_5408_, v___x_5429_);
v___x_5431_ = lean_box(0);
v___x_5432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5432_, 0, v___x_5431_);
return v___x_5432_;
}
}
}
v___jp_5435_:
{
lean_object* v___x_5444_; lean_object* v___x_5445_; lean_object* v_a_5446_; lean_object* v___x_5448_; uint8_t v_isShared_5449_; uint8_t v_isSharedCheck_5459_; 
v___x_5444_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_5391_);
v___x_5445_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig_evalExpr_spec__1_spec__1(v___x_5444_, v___y_5394_, v___y_5395_, v___y_5396_, v___y_5397_);
v_a_5446_ = lean_ctor_get(v___x_5445_, 0);
v_isSharedCheck_5459_ = !lean_is_exclusive(v___x_5445_);
if (v_isSharedCheck_5459_ == 0)
{
v___x_5448_ = v___x_5445_;
v_isShared_5449_ = v_isSharedCheck_5459_;
goto v_resetjp_5447_;
}
else
{
lean_inc(v_a_5446_);
lean_dec(v___x_5445_);
v___x_5448_ = lean_box(0);
v_isShared_5449_ = v_isSharedCheck_5459_;
goto v_resetjp_5447_;
}
v_resetjp_5447_:
{
lean_object* v___x_5450_; lean_object* v___x_5451_; lean_object* v___x_5452_; lean_object* v___x_5453_; 
lean_inc_ref_n(v___y_5440_, 2);
v___x_5450_ = l_Lean_FileMap_toPosition(v___y_5440_, v___y_5437_);
lean_dec(v___y_5437_);
v___x_5451_ = l_Lean_FileMap_toPosition(v___y_5440_, v___y_5443_);
lean_dec(v___y_5443_);
v___x_5452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5452_, 0, v___x_5451_);
v___x_5453_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Mathlib_Tactic_Push_resolvePushId_x3f_spec__0_spec__0___redArg___closed__1));
if (v___y_5442_ == 0)
{
lean_del_object(v___x_5448_);
lean_dec_ref(v___y_5436_);
v___y_5400_ = v___x_5452_;
v___y_5401_ = v_a_5446_;
v___y_5402_ = v___x_5453_;
v___y_5403_ = v___y_5439_;
v___y_5404_ = v___y_5438_;
v___y_5405_ = v___y_5441_;
v___y_5406_ = v___x_5450_;
v___y_5407_ = v___y_5396_;
v___y_5408_ = v___y_5397_;
goto v___jp_5399_;
}
else
{
uint8_t v___x_5454_; 
lean_inc(v_a_5446_);
v___x_5454_ = l_Lean_MessageData_hasTag(v___y_5436_, v_a_5446_);
if (v___x_5454_ == 0)
{
lean_object* v___x_5455_; lean_object* v___x_5457_; 
lean_dec_ref_known(v___x_5452_, 1);
lean_dec_ref(v___x_5450_);
lean_dec(v_a_5446_);
v___x_5455_ = lean_box(0);
if (v_isShared_5449_ == 0)
{
lean_ctor_set(v___x_5448_, 0, v___x_5455_);
v___x_5457_ = v___x_5448_;
goto v_reusejp_5456_;
}
else
{
lean_object* v_reuseFailAlloc_5458_; 
v_reuseFailAlloc_5458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5458_, 0, v___x_5455_);
v___x_5457_ = v_reuseFailAlloc_5458_;
goto v_reusejp_5456_;
}
v_reusejp_5456_:
{
return v___x_5457_;
}
}
else
{
lean_del_object(v___x_5448_);
v___y_5400_ = v___x_5452_;
v___y_5401_ = v_a_5446_;
v___y_5402_ = v___x_5453_;
v___y_5403_ = v___y_5439_;
v___y_5404_ = v___y_5438_;
v___y_5405_ = v___y_5441_;
v___y_5406_ = v___x_5450_;
v___y_5407_ = v___y_5396_;
v___y_5408_ = v___y_5397_;
goto v___jp_5399_;
}
}
}
}
v___jp_5460_:
{
lean_object* v___x_5469_; 
v___x_5469_ = l_Lean_Syntax_getTailPos_x3f(v___y_5462_, v___y_5463_);
lean_dec(v___y_5462_);
if (lean_obj_tag(v___x_5469_) == 0)
{
lean_inc(v___y_5468_);
v___y_5436_ = v___y_5461_;
v___y_5437_ = v___y_5468_;
v___y_5438_ = v___y_5464_;
v___y_5439_ = v___y_5463_;
v___y_5440_ = v___y_5465_;
v___y_5441_ = v___y_5466_;
v___y_5442_ = v___y_5467_;
v___y_5443_ = v___y_5468_;
goto v___jp_5435_;
}
else
{
lean_object* v_val_5470_; 
v_val_5470_ = lean_ctor_get(v___x_5469_, 0);
lean_inc(v_val_5470_);
lean_dec_ref_known(v___x_5469_, 1);
v___y_5436_ = v___y_5461_;
v___y_5437_ = v___y_5468_;
v___y_5438_ = v___y_5464_;
v___y_5439_ = v___y_5463_;
v___y_5440_ = v___y_5465_;
v___y_5441_ = v___y_5466_;
v___y_5442_ = v___y_5467_;
v___y_5443_ = v_val_5470_;
goto v___jp_5435_;
}
}
v___jp_5471_:
{
lean_object* v_ref_5479_; lean_object* v___x_5480_; 
v_ref_5479_ = l_Lean_replaceRef(v_ref_5390_, v___y_5473_);
v___x_5480_ = l_Lean_Syntax_getPos_x3f(v_ref_5479_, v___y_5475_);
if (lean_obj_tag(v___x_5480_) == 0)
{
lean_object* v___x_5481_; 
v___x_5481_ = lean_unsigned_to_nat(0u);
v___y_5461_ = v___y_5472_;
v___y_5462_ = v_ref_5479_;
v___y_5463_ = v___y_5475_;
v___y_5464_ = v___y_5474_;
v___y_5465_ = v___y_5476_;
v___y_5466_ = v___y_5478_;
v___y_5467_ = v___y_5477_;
v___y_5468_ = v___x_5481_;
goto v___jp_5460_;
}
else
{
lean_object* v_val_5482_; 
v_val_5482_ = lean_ctor_get(v___x_5480_, 0);
lean_inc(v_val_5482_);
lean_dec_ref_known(v___x_5480_, 1);
v___y_5461_ = v___y_5472_;
v___y_5462_ = v_ref_5479_;
v___y_5463_ = v___y_5475_;
v___y_5464_ = v___y_5474_;
v___y_5465_ = v___y_5476_;
v___y_5466_ = v___y_5478_;
v___y_5467_ = v___y_5477_;
v___y_5468_ = v_val_5482_;
goto v___jp_5460_;
}
}
v___jp_5484_:
{
if (v___y_5491_ == 0)
{
v___y_5472_ = v___y_5486_;
v___y_5473_ = v___y_5485_;
v___y_5474_ = v___y_5487_;
v___y_5475_ = v___y_5490_;
v___y_5476_ = v___y_5488_;
v___y_5477_ = v___y_5489_;
v___y_5478_ = v_severity_5392_;
goto v___jp_5471_;
}
else
{
v___y_5472_ = v___y_5486_;
v___y_5473_ = v___y_5485_;
v___y_5474_ = v___y_5487_;
v___y_5475_ = v___y_5490_;
v___y_5476_ = v___y_5488_;
v___y_5477_ = v___y_5489_;
v___y_5478_ = v___x_5483_;
goto v___jp_5471_;
}
}
v___jp_5492_:
{
if (v___y_5493_ == 0)
{
lean_object* v_fileName_5494_; lean_object* v_fileMap_5495_; lean_object* v_options_5496_; lean_object* v_ref_5497_; uint8_t v_suppressElabErrors_5498_; lean_object* v___x_5499_; lean_object* v___x_5500_; lean_object* v___f_5501_; uint8_t v___x_5502_; uint8_t v___x_5503_; 
v_fileName_5494_ = lean_ctor_get(v___y_5396_, 0);
v_fileMap_5495_ = lean_ctor_get(v___y_5396_, 1);
v_options_5496_ = lean_ctor_get(v___y_5396_, 2);
v_ref_5497_ = lean_ctor_get(v___y_5396_, 5);
v_suppressElabErrors_5498_ = lean_ctor_get_uint8(v___y_5396_, sizeof(void*)*14 + 1);
v___x_5499_ = lean_box(v___y_5493_);
v___x_5500_ = lean_box(v_suppressElabErrors_5498_);
v___f_5501_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Push___aux__Mathlib__Tactic__Push______elabRules__Mathlib__Tactic__Push__push__neg__1_spec__0_spec__0_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_5501_, 0, v___x_5499_);
lean_closure_set(v___f_5501_, 1, v___x_5500_);
v___x_5502_ = 1;
v___x_5503_ = l_Lean_instBEqMessageSeverity_beq(v_severity_5392_, v___x_5502_);
if (v___x_5503_ == 0)
{
v___y_5485_ = v_ref_5497_;
v___y_5486_ = v___f_5501_;
v___y_5487_ = v_fileName_5494_;
v___y_5488_ = v_fileMap_5495_;
v___y_5489_ = v_suppressElabErrors_5498_;
v___y_5490_ = v___y_5493_;
v___y_5491_ = v___x_5503_;
goto v___jp_5484_;
}
else
{
lean_object* v___x_5504_; uint8_t v___x_5505_; 
v___x_5504_ = l_Lean_warningAsError;
v___x_5505_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_options_5496_, v___x_5504_);
v___y_5485_ = v_ref_5497_;
v___y_5486_ = v___f_5501_;
v___y_5487_ = v_fileName_5494_;
v___y_5488_ = v_fileMap_5495_;
v___y_5489_ = v_suppressElabErrors_5498_;
v___y_5490_ = v___y_5493_;
v___y_5491_ = v___x_5505_;
goto v___jp_5484_;
}
}
else
{
lean_object* v___x_5506_; lean_object* v___x_5507_; 
lean_dec_ref(v_msgData_5391_);
v___x_5506_ = lean_box(0);
v___x_5507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5507_, 0, v___x_5506_);
return v___x_5507_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg___boxed(lean_object* v_ref_5510_, lean_object* v_msgData_5511_, lean_object* v_severity_5512_, lean_object* v_isSilent_5513_, lean_object* v___y_5514_, lean_object* v___y_5515_, lean_object* v___y_5516_, lean_object* v___y_5517_, lean_object* v___y_5518_){
_start:
{
uint8_t v_severity_boxed_5519_; uint8_t v_isSilent_boxed_5520_; lean_object* v_res_5521_; 
v_severity_boxed_5519_ = lean_unbox(v_severity_5512_);
v_isSilent_boxed_5520_ = lean_unbox(v_isSilent_5513_);
v_res_5521_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg(v_ref_5510_, v_msgData_5511_, v_severity_boxed_5519_, v_isSilent_boxed_5520_, v___y_5514_, v___y_5515_, v___y_5516_, v___y_5517_);
lean_dec(v___y_5517_);
lean_dec_ref(v___y_5516_);
lean_dec(v___y_5515_);
lean_dec_ref(v___y_5514_);
lean_dec(v_ref_5510_);
return v_res_5521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4(lean_object* v_msgData_5522_, uint8_t v_severity_5523_, uint8_t v_isSilent_5524_, lean_object* v___y_5525_, lean_object* v___y_5526_, lean_object* v___y_5527_, lean_object* v___y_5528_, lean_object* v___y_5529_, lean_object* v___y_5530_){
_start:
{
lean_object* v_ref_5532_; lean_object* v___x_5533_; 
v_ref_5532_ = lean_ctor_get(v___y_5529_, 5);
v___x_5533_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg(v_ref_5532_, v_msgData_5522_, v_severity_5523_, v_isSilent_5524_, v___y_5527_, v___y_5528_, v___y_5529_, v___y_5530_);
return v___x_5533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4___boxed(lean_object* v_msgData_5534_, lean_object* v_severity_5535_, lean_object* v_isSilent_5536_, lean_object* v___y_5537_, lean_object* v___y_5538_, lean_object* v___y_5539_, lean_object* v___y_5540_, lean_object* v___y_5541_, lean_object* v___y_5542_, lean_object* v___y_5543_){
_start:
{
uint8_t v_severity_boxed_5544_; uint8_t v_isSilent_boxed_5545_; lean_object* v_res_5546_; 
v_severity_boxed_5544_ = lean_unbox(v_severity_5535_);
v_isSilent_boxed_5545_ = lean_unbox(v_isSilent_5536_);
v_res_5546_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4(v_msgData_5534_, v_severity_boxed_5544_, v_isSilent_boxed_5545_, v___y_5537_, v___y_5538_, v___y_5539_, v___y_5540_, v___y_5541_, v___y_5542_);
lean_dec(v___y_5542_);
lean_dec_ref(v___y_5541_);
lean_dec(v___y_5540_);
lean_dec_ref(v___y_5539_);
lean_dec(v___y_5538_);
lean_dec_ref(v___y_5537_);
return v_res_5546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1(lean_object* v_msgData_5547_, lean_object* v___y_5548_, lean_object* v___y_5549_, lean_object* v___y_5550_, lean_object* v___y_5551_, lean_object* v___y_5552_, lean_object* v___y_5553_){
_start:
{
uint8_t v___x_5555_; uint8_t v___x_5556_; lean_object* v___x_5557_; 
v___x_5555_ = 0;
v___x_5556_ = 0;
v___x_5557_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4(v_msgData_5547_, v___x_5555_, v___x_5556_, v___y_5548_, v___y_5549_, v___y_5550_, v___y_5551_, v___y_5552_, v___y_5553_);
return v___x_5557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1___boxed(lean_object* v_msgData_5558_, lean_object* v___y_5559_, lean_object* v___y_5560_, lean_object* v___y_5561_, lean_object* v___y_5562_, lean_object* v___y_5563_, lean_object* v___y_5564_, lean_object* v___y_5565_){
_start:
{
lean_object* v_res_5566_; 
v_res_5566_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1(v_msgData_5558_, v___y_5559_, v___y_5560_, v___y_5561_, v___y_5562_, v___y_5563_, v___y_5564_);
lean_dec(v___y_5564_);
lean_dec_ref(v___y_5563_);
lean_dec(v___y_5562_);
lean_dec_ref(v___y_5561_);
lean_dec(v___y_5560_);
lean_dec_ref(v___y_5559_);
return v_res_5566_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1(void){
_start:
{
lean_object* v___x_5568_; lean_object* v___x_5569_; 
v___x_5568_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__0));
v___x_5569_ = l_Lean_stringToMessageData(v___x_5568_);
return v___x_5569_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2(void){
_start:
{
lean_object* v___x_5570_; lean_object* v___x_5571_; 
v___x_5570_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__2_spec__3_spec__6___closed__0));
v___x_5571_ = l_Lean_stringToMessageData(v___x_5570_);
return v___x_5571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0(lean_object* v_a_5575_, lean_object* v_x_5576_, uint8_t v_____s_5577_, lean_object* v___y_5578_, lean_object* v___y_5579_, lean_object* v___y_5580_, lean_object* v___y_5581_, lean_object* v___y_5582_, lean_object* v___y_5583_){
_start:
{
lean_object* v_fst_5589_; lean_object* v_snd_5590_; lean_object* v___x_5592_; uint8_t v_isShared_5593_; uint8_t v_isSharedCheck_5628_; 
v_fst_5589_ = lean_ctor_get(v_x_5576_, 0);
v_snd_5590_ = lean_ctor_get(v_x_5576_, 1);
v_isSharedCheck_5628_ = !lean_is_exclusive(v_x_5576_);
if (v_isSharedCheck_5628_ == 0)
{
v___x_5592_ = v_x_5576_;
v_isShared_5593_ = v_isSharedCheck_5628_;
goto v_resetjp_5591_;
}
else
{
lean_inc(v_snd_5590_);
lean_inc(v_fst_5589_);
lean_dec(v_x_5576_);
v___x_5592_ = lean_box(0);
v_isShared_5593_ = v_isSharedCheck_5628_;
goto v_resetjp_5591_;
}
v___jp_5585_:
{
lean_object* v___x_5586_; lean_object* v___x_5587_; lean_object* v___x_5588_; 
v___x_5586_ = lean_box(v_____s_5577_);
v___x_5587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5587_, 0, v___x_5586_);
v___x_5588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5588_, 0, v___x_5587_);
return v___x_5588_;
}
v_resetjp_5591_:
{
switch(lean_obj_tag(v_fst_5589_))
{
case 4:
{
if (lean_obj_tag(v_a_5575_) == 0)
{
lean_object* v_a_5625_; lean_object* v_c_5626_; uint8_t v___x_5627_; 
v_a_5625_ = lean_ctor_get(v_fst_5589_, 0);
v_c_5626_ = lean_ctor_get(v_a_5575_, 0);
v___x_5627_ = lean_name_eq(v_a_5625_, v_c_5626_);
if (v___x_5627_ == 0)
{
lean_dec_ref_known(v_fst_5589_, 2);
lean_del_object(v___x_5592_);
lean_dec(v_snd_5590_);
goto v___jp_5585_;
}
else
{
goto v___jp_5594_;
}
}
else
{
lean_dec_ref_known(v_fst_5589_, 2);
lean_del_object(v___x_5592_);
lean_dec(v_snd_5590_);
goto v___jp_5585_;
}
}
case 1:
{
if (lean_obj_tag(v_a_5575_) == 1)
{
goto v___jp_5594_;
}
else
{
lean_del_object(v___x_5592_);
lean_dec(v_snd_5590_);
goto v___jp_5585_;
}
}
case 5:
{
if (lean_obj_tag(v_a_5575_) == 2)
{
goto v___jp_5594_;
}
else
{
lean_del_object(v___x_5592_);
lean_dec(v_snd_5590_);
goto v___jp_5585_;
}
}
default: 
{
lean_del_object(v___x_5592_);
lean_dec(v_snd_5590_);
lean_dec(v_fst_5589_);
goto v___jp_5585_;
}
}
v___jp_5594_:
{
lean_object* v___x_5595_; lean_object* v___x_5596_; lean_object* v___x_5597_; lean_object* v___x_5599_; 
v___x_5595_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__1);
v___x_5596_ = l_Lean_Meta_DiscrTree_Key_format(v_fst_5589_);
v___x_5597_ = l_Lean_MessageData_ofFormat(v___x_5596_);
if (v_isShared_5593_ == 0)
{
lean_ctor_set_tag(v___x_5592_, 7);
lean_ctor_set(v___x_5592_, 1, v___x_5597_);
lean_ctor_set(v___x_5592_, 0, v___x_5595_);
v___x_5599_ = v___x_5592_;
goto v_reusejp_5598_;
}
else
{
lean_object* v_reuseFailAlloc_5624_; 
v_reuseFailAlloc_5624_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5624_, 0, v___x_5595_);
lean_ctor_set(v_reuseFailAlloc_5624_, 1, v___x_5597_);
v___x_5599_ = v_reuseFailAlloc_5624_;
goto v_reusejp_5598_;
}
v_reusejp_5598_:
{
lean_object* v___x_5600_; lean_object* v___x_5601_; lean_object* v___x_5602_; lean_object* v___x_5603_; lean_object* v___x_5604_; lean_object* v___x_5605_; lean_object* v___x_5606_; 
v___x_5600_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__2);
v___x_5601_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5601_, 0, v___x_5599_);
lean_ctor_set(v___x_5601_, 1, v___x_5600_);
v___x_5602_ = lp_mathlib_Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0(v_snd_5590_);
v___x_5603_ = l_Lean_MessageData_ofFormat(v___x_5602_);
v___x_5604_ = l_Lean_indentD(v___x_5603_);
v___x_5605_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5605_, 0, v___x_5601_);
lean_ctor_set(v___x_5605_, 1, v___x_5604_);
v___x_5606_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1(v___x_5605_, v___y_5578_, v___y_5579_, v___y_5580_, v___y_5581_, v___y_5582_, v___y_5583_);
if (lean_obj_tag(v___x_5606_) == 0)
{
lean_object* v___x_5608_; uint8_t v_isShared_5609_; uint8_t v_isSharedCheck_5614_; 
v_isSharedCheck_5614_ = !lean_is_exclusive(v___x_5606_);
if (v_isSharedCheck_5614_ == 0)
{
lean_object* v_unused_5615_; 
v_unused_5615_ = lean_ctor_get(v___x_5606_, 0);
lean_dec(v_unused_5615_);
v___x_5608_ = v___x_5606_;
v_isShared_5609_ = v_isSharedCheck_5614_;
goto v_resetjp_5607_;
}
else
{
lean_dec(v___x_5606_);
v___x_5608_ = lean_box(0);
v_isShared_5609_ = v_isSharedCheck_5614_;
goto v_resetjp_5607_;
}
v_resetjp_5607_:
{
lean_object* v___x_5610_; lean_object* v___x_5612_; 
v___x_5610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___closed__3));
if (v_isShared_5609_ == 0)
{
lean_ctor_set(v___x_5608_, 0, v___x_5610_);
v___x_5612_ = v___x_5608_;
goto v_reusejp_5611_;
}
else
{
lean_object* v_reuseFailAlloc_5613_; 
v_reuseFailAlloc_5613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5613_, 0, v___x_5610_);
v___x_5612_ = v_reuseFailAlloc_5613_;
goto v_reusejp_5611_;
}
v_reusejp_5611_:
{
return v___x_5612_;
}
}
}
else
{
lean_object* v_a_5616_; lean_object* v___x_5618_; uint8_t v_isShared_5619_; uint8_t v_isSharedCheck_5623_; 
v_a_5616_ = lean_ctor_get(v___x_5606_, 0);
v_isSharedCheck_5623_ = !lean_is_exclusive(v___x_5606_);
if (v_isSharedCheck_5623_ == 0)
{
v___x_5618_ = v___x_5606_;
v_isShared_5619_ = v_isSharedCheck_5623_;
goto v_resetjp_5617_;
}
else
{
lean_inc(v_a_5616_);
lean_dec(v___x_5606_);
v___x_5618_ = lean_box(0);
v_isShared_5619_ = v_isSharedCheck_5623_;
goto v_resetjp_5617_;
}
v_resetjp_5617_:
{
lean_object* v___x_5621_; 
if (v_isShared_5619_ == 0)
{
v___x_5621_ = v___x_5618_;
goto v_reusejp_5620_;
}
else
{
lean_object* v_reuseFailAlloc_5622_; 
v_reuseFailAlloc_5622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5622_, 0, v_a_5616_);
v___x_5621_ = v_reuseFailAlloc_5622_;
goto v_reusejp_5620_;
}
v_reusejp_5620_:
{
return v___x_5621_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___boxed(lean_object* v_a_5629_, lean_object* v_x_5630_, lean_object* v_____s_5631_, lean_object* v___y_5632_, lean_object* v___y_5633_, lean_object* v___y_5634_, lean_object* v___y_5635_, lean_object* v___y_5636_, lean_object* v___y_5637_, lean_object* v___y_5638_){
_start:
{
uint8_t v_____s_8274__boxed_5639_; lean_object* v_res_5640_; 
v_____s_8274__boxed_5639_ = lean_unbox(v_____s_5631_);
v_res_5640_ = lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0(v_a_5629_, v_x_5630_, v_____s_8274__boxed_5639_, v___y_5632_, v___y_5633_, v___y_5634_, v___y_5635_, v___y_5636_, v___y_5637_);
lean_dec(v___y_5637_);
lean_dec_ref(v___y_5636_);
lean_dec(v___y_5635_);
lean_dec_ref(v___y_5634_);
lean_dec(v___y_5633_);
lean_dec_ref(v___y_5632_);
lean_dec(v_a_5629_);
return v_res_5640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg(lean_object* v_f_5641_, lean_object* v_keys_5642_, lean_object* v_vals_5643_, lean_object* v_i_5644_, lean_object* v_acc_5645_, lean_object* v___y_5646_, lean_object* v___y_5647_, lean_object* v___y_5648_, lean_object* v___y_5649_, lean_object* v___y_5650_, lean_object* v___y_5651_){
_start:
{
lean_object* v___x_5653_; uint8_t v___x_5654_; 
v___x_5653_ = lean_array_get_size(v_keys_5642_);
v___x_5654_ = lean_nat_dec_lt(v_i_5644_, v___x_5653_);
if (v___x_5654_ == 0)
{
lean_object* v___x_5655_; lean_object* v___x_5656_; 
lean_dec(v_i_5644_);
lean_dec_ref(v_f_5641_);
v___x_5655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5655_, 0, v_acc_5645_);
v___x_5656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5656_, 0, v___x_5655_);
return v___x_5656_;
}
else
{
lean_object* v_k_5657_; lean_object* v_v_5658_; lean_object* v___x_5659_; 
v_k_5657_ = lean_array_fget_borrowed(v_keys_5642_, v_i_5644_);
v_v_5658_ = lean_array_fget_borrowed(v_vals_5643_, v_i_5644_);
lean_inc_ref(v_f_5641_);
lean_inc(v___y_5651_);
lean_inc_ref(v___y_5650_);
lean_inc(v___y_5649_);
lean_inc_ref(v___y_5648_);
lean_inc(v___y_5647_);
lean_inc_ref(v___y_5646_);
lean_inc(v_v_5658_);
lean_inc(v_k_5657_);
v___x_5659_ = lean_apply_10(v_f_5641_, v_acc_5645_, v_k_5657_, v_v_5658_, v___y_5646_, v___y_5647_, v___y_5648_, v___y_5649_, v___y_5650_, v___y_5651_, lean_box(0));
if (lean_obj_tag(v___x_5659_) == 0)
{
lean_object* v_a_5660_; 
v_a_5660_ = lean_ctor_get(v___x_5659_, 0);
lean_inc(v_a_5660_);
if (lean_obj_tag(v_a_5660_) == 0)
{
lean_dec_ref_known(v_a_5660_, 1);
lean_dec(v_i_5644_);
lean_dec_ref(v_f_5641_);
return v___x_5659_;
}
else
{
lean_object* v_a_5661_; lean_object* v___x_5662_; lean_object* v___x_5663_; 
lean_dec_ref_known(v___x_5659_, 1);
v_a_5661_ = lean_ctor_get(v_a_5660_, 0);
lean_inc(v_a_5661_);
lean_dec_ref_known(v_a_5660_, 1);
v___x_5662_ = lean_unsigned_to_nat(1u);
v___x_5663_ = lean_nat_add(v_i_5644_, v___x_5662_);
lean_dec(v_i_5644_);
v_i_5644_ = v___x_5663_;
v_acc_5645_ = v_a_5661_;
goto _start;
}
}
else
{
lean_dec(v_i_5644_);
lean_dec_ref(v_f_5641_);
return v___x_5659_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg___boxed(lean_object* v_f_5665_, lean_object* v_keys_5666_, lean_object* v_vals_5667_, lean_object* v_i_5668_, lean_object* v_acc_5669_, lean_object* v___y_5670_, lean_object* v___y_5671_, lean_object* v___y_5672_, lean_object* v___y_5673_, lean_object* v___y_5674_, lean_object* v___y_5675_, lean_object* v___y_5676_){
_start:
{
lean_object* v_res_5677_; 
v_res_5677_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg(v_f_5665_, v_keys_5666_, v_vals_5667_, v_i_5668_, v_acc_5669_, v___y_5670_, v___y_5671_, v___y_5672_, v___y_5673_, v___y_5674_, v___y_5675_);
lean_dec(v___y_5675_);
lean_dec_ref(v___y_5674_);
lean_dec(v___y_5673_);
lean_dec_ref(v___y_5672_);
lean_dec(v___y_5671_);
lean_dec_ref(v___y_5670_);
lean_dec_ref(v_vals_5667_);
lean_dec_ref(v_keys_5666_);
return v_res_5677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(lean_object* v_f_5678_, lean_object* v_x_5679_, lean_object* v_x_5680_, lean_object* v___y_5681_, lean_object* v___y_5682_, lean_object* v___y_5683_, lean_object* v___y_5684_, lean_object* v___y_5685_, lean_object* v___y_5686_){
_start:
{
if (lean_obj_tag(v_x_5679_) == 0)
{
lean_object* v_es_5688_; lean_object* v___x_5690_; uint8_t v_isShared_5691_; uint8_t v_isSharedCheck_5710_; 
v_es_5688_ = lean_ctor_get(v_x_5679_, 0);
v_isSharedCheck_5710_ = !lean_is_exclusive(v_x_5679_);
if (v_isSharedCheck_5710_ == 0)
{
v___x_5690_ = v_x_5679_;
v_isShared_5691_ = v_isSharedCheck_5710_;
goto v_resetjp_5689_;
}
else
{
lean_inc(v_es_5688_);
lean_dec(v_x_5679_);
v___x_5690_ = lean_box(0);
v_isShared_5691_ = v_isSharedCheck_5710_;
goto v_resetjp_5689_;
}
v_resetjp_5689_:
{
lean_object* v___x_5692_; lean_object* v___x_5693_; uint8_t v___x_5694_; 
v___x_5692_ = lean_unsigned_to_nat(0u);
v___x_5693_ = lean_array_get_size(v_es_5688_);
v___x_5694_ = lean_nat_dec_lt(v___x_5692_, v___x_5693_);
if (v___x_5694_ == 0)
{
lean_object* v___x_5696_; 
lean_dec_ref(v_es_5688_);
lean_dec_ref(v_f_5678_);
if (v_isShared_5691_ == 0)
{
lean_ctor_set_tag(v___x_5690_, 1);
lean_ctor_set(v___x_5690_, 0, v_x_5680_);
v___x_5696_ = v___x_5690_;
goto v_reusejp_5695_;
}
else
{
lean_object* v_reuseFailAlloc_5698_; 
v_reuseFailAlloc_5698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5698_, 0, v_x_5680_);
v___x_5696_ = v_reuseFailAlloc_5698_;
goto v_reusejp_5695_;
}
v_reusejp_5695_:
{
lean_object* v___x_5697_; 
v___x_5697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5697_, 0, v___x_5696_);
return v___x_5697_;
}
}
else
{
uint8_t v___x_5699_; 
v___x_5699_ = lean_nat_dec_le(v___x_5693_, v___x_5693_);
if (v___x_5699_ == 0)
{
if (v___x_5694_ == 0)
{
lean_object* v___x_5701_; 
lean_dec_ref(v_es_5688_);
lean_dec_ref(v_f_5678_);
if (v_isShared_5691_ == 0)
{
lean_ctor_set_tag(v___x_5690_, 1);
lean_ctor_set(v___x_5690_, 0, v_x_5680_);
v___x_5701_ = v___x_5690_;
goto v_reusejp_5700_;
}
else
{
lean_object* v_reuseFailAlloc_5703_; 
v_reuseFailAlloc_5703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5703_, 0, v_x_5680_);
v___x_5701_ = v_reuseFailAlloc_5703_;
goto v_reusejp_5700_;
}
v_reusejp_5700_:
{
lean_object* v___x_5702_; 
v___x_5702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5702_, 0, v___x_5701_);
return v___x_5702_;
}
}
else
{
size_t v___x_5704_; size_t v___x_5705_; lean_object* v___x_5706_; 
lean_del_object(v___x_5690_);
v___x_5704_ = ((size_t)0ULL);
v___x_5705_ = lean_usize_of_nat(v___x_5693_);
v___x_5706_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(v_f_5678_, v_es_5688_, v___x_5704_, v___x_5705_, v_x_5680_, v___y_5681_, v___y_5682_, v___y_5683_, v___y_5684_, v___y_5685_, v___y_5686_);
lean_dec_ref(v_es_5688_);
return v___x_5706_;
}
}
else
{
size_t v___x_5707_; size_t v___x_5708_; lean_object* v___x_5709_; 
lean_del_object(v___x_5690_);
v___x_5707_ = ((size_t)0ULL);
v___x_5708_ = lean_usize_of_nat(v___x_5693_);
v___x_5709_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(v_f_5678_, v_es_5688_, v___x_5707_, v___x_5708_, v_x_5680_, v___y_5681_, v___y_5682_, v___y_5683_, v___y_5684_, v___y_5685_, v___y_5686_);
lean_dec_ref(v_es_5688_);
return v___x_5709_;
}
}
}
}
else
{
lean_object* v_ks_5711_; lean_object* v_vs_5712_; lean_object* v___x_5713_; lean_object* v___x_5714_; 
v_ks_5711_ = lean_ctor_get(v_x_5679_, 0);
lean_inc_ref(v_ks_5711_);
v_vs_5712_ = lean_ctor_get(v_x_5679_, 1);
lean_inc_ref(v_vs_5712_);
lean_dec_ref_known(v_x_5679_, 2);
v___x_5713_ = lean_unsigned_to_nat(0u);
v___x_5714_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg(v_f_5678_, v_ks_5711_, v_vs_5712_, v___x_5713_, v_x_5680_, v___y_5681_, v___y_5682_, v___y_5683_, v___y_5684_, v___y_5685_, v___y_5686_);
lean_dec_ref(v_vs_5712_);
lean_dec_ref(v_ks_5711_);
return v___x_5714_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(lean_object* v_f_5715_, lean_object* v_as_5716_, size_t v_i_5717_, size_t v_stop_5718_, lean_object* v_b_5719_, lean_object* v___y_5720_, lean_object* v___y_5721_, lean_object* v___y_5722_, lean_object* v___y_5723_, lean_object* v___y_5724_, lean_object* v___y_5725_){
_start:
{
lean_object* v_a_5728_; lean_object* v___y_5733_; uint8_t v___x_5736_; 
v___x_5736_ = lean_usize_dec_eq(v_i_5717_, v_stop_5718_);
if (v___x_5736_ == 0)
{
lean_object* v___x_5737_; 
v___x_5737_ = lean_array_uget_borrowed(v_as_5716_, v_i_5717_);
switch(lean_obj_tag(v___x_5737_))
{
case 0:
{
lean_object* v_key_5738_; lean_object* v_val_5739_; lean_object* v___x_5740_; 
v_key_5738_ = lean_ctor_get(v___x_5737_, 0);
v_val_5739_ = lean_ctor_get(v___x_5737_, 1);
lean_inc_ref(v_f_5715_);
lean_inc(v___y_5725_);
lean_inc_ref(v___y_5724_);
lean_inc(v___y_5723_);
lean_inc_ref(v___y_5722_);
lean_inc(v___y_5721_);
lean_inc_ref(v___y_5720_);
lean_inc(v_val_5739_);
lean_inc(v_key_5738_);
v___x_5740_ = lean_apply_10(v_f_5715_, v_b_5719_, v_key_5738_, v_val_5739_, v___y_5720_, v___y_5721_, v___y_5722_, v___y_5723_, v___y_5724_, v___y_5725_, lean_box(0));
v___y_5733_ = v___x_5740_;
goto v___jp_5732_;
}
case 1:
{
lean_object* v_node_5741_; lean_object* v___x_5742_; 
v_node_5741_ = lean_ctor_get(v___x_5737_, 0);
lean_inc(v_node_5741_);
lean_inc_ref(v_f_5715_);
v___x_5742_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v_f_5715_, v_node_5741_, v_b_5719_, v___y_5720_, v___y_5721_, v___y_5722_, v___y_5723_, v___y_5724_, v___y_5725_);
v___y_5733_ = v___x_5742_;
goto v___jp_5732_;
}
default: 
{
v_a_5728_ = v_b_5719_;
goto v___jp_5727_;
}
}
}
else
{
lean_object* v___x_5743_; lean_object* v___x_5744_; 
lean_dec_ref(v_f_5715_);
v___x_5743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5743_, 0, v_b_5719_);
v___x_5744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5744_, 0, v___x_5743_);
return v___x_5744_;
}
v___jp_5727_:
{
size_t v___x_5729_; size_t v___x_5730_; 
v___x_5729_ = ((size_t)1ULL);
v___x_5730_ = lean_usize_add(v_i_5717_, v___x_5729_);
v_i_5717_ = v___x_5730_;
v_b_5719_ = v_a_5728_;
goto _start;
}
v___jp_5732_:
{
if (lean_obj_tag(v___y_5733_) == 0)
{
lean_object* v_a_5734_; 
v_a_5734_ = lean_ctor_get(v___y_5733_, 0);
if (lean_obj_tag(v_a_5734_) == 0)
{
lean_dec_ref(v_f_5715_);
return v___y_5733_;
}
else
{
lean_object* v_a_5735_; 
lean_inc_ref(v_a_5734_);
lean_dec_ref_known(v___y_5733_, 1);
v_a_5735_ = lean_ctor_get(v_a_5734_, 0);
lean_inc(v_a_5735_);
lean_dec_ref_known(v_a_5734_, 1);
v_a_5728_ = v_a_5735_;
goto v___jp_5727_;
}
}
else
{
lean_dec_ref(v_f_5715_);
return v___y_5733_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg___boxed(lean_object* v_f_5745_, lean_object* v_as_5746_, lean_object* v_i_5747_, lean_object* v_stop_5748_, lean_object* v_b_5749_, lean_object* v___y_5750_, lean_object* v___y_5751_, lean_object* v___y_5752_, lean_object* v___y_5753_, lean_object* v___y_5754_, lean_object* v___y_5755_, lean_object* v___y_5756_){
_start:
{
size_t v_i_boxed_5757_; size_t v_stop_boxed_5758_; lean_object* v_res_5759_; 
v_i_boxed_5757_ = lean_unbox_usize(v_i_5747_);
lean_dec(v_i_5747_);
v_stop_boxed_5758_ = lean_unbox_usize(v_stop_5748_);
lean_dec(v_stop_5748_);
v_res_5759_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(v_f_5745_, v_as_5746_, v_i_boxed_5757_, v_stop_boxed_5758_, v_b_5749_, v___y_5750_, v___y_5751_, v___y_5752_, v___y_5753_, v___y_5754_, v___y_5755_);
lean_dec(v___y_5755_);
lean_dec_ref(v___y_5754_);
lean_dec(v___y_5753_);
lean_dec_ref(v___y_5752_);
lean_dec(v___y_5751_);
lean_dec_ref(v___y_5750_);
lean_dec_ref(v_as_5746_);
return v_res_5759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg___boxed(lean_object* v_f_5760_, lean_object* v_x_5761_, lean_object* v_x_5762_, lean_object* v___y_5763_, lean_object* v___y_5764_, lean_object* v___y_5765_, lean_object* v___y_5766_, lean_object* v___y_5767_, lean_object* v___y_5768_, lean_object* v___y_5769_){
_start:
{
lean_object* v_res_5770_; 
v_res_5770_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v_f_5760_, v_x_5761_, v_x_5762_, v___y_5763_, v___y_5764_, v___y_5765_, v___y_5766_, v___y_5767_, v___y_5768_);
lean_dec(v___y_5768_);
lean_dec_ref(v___y_5767_);
lean_dec(v___y_5766_);
lean_dec_ref(v___y_5765_);
lean_dec(v___y_5764_);
lean_dec_ref(v___y_5763_);
return v_res_5770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0(lean_object* v_f_5771_, lean_object* v_s_5772_, lean_object* v_a_5773_, lean_object* v_b_5774_, lean_object* v___y_5775_, lean_object* v___y_5776_, lean_object* v___y_5777_, lean_object* v___y_5778_, lean_object* v___y_5779_, lean_object* v___y_5780_){
_start:
{
lean_object* v___x_5782_; lean_object* v___x_5783_; 
v___x_5782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5782_, 0, v_a_5773_);
lean_ctor_set(v___x_5782_, 1, v_b_5774_);
lean_inc(v___y_5780_);
lean_inc_ref(v___y_5779_);
lean_inc(v___y_5778_);
lean_inc_ref(v___y_5777_);
lean_inc(v___y_5776_);
lean_inc_ref(v___y_5775_);
v___x_5783_ = lean_apply_9(v_f_5771_, v___x_5782_, v_s_5772_, v___y_5775_, v___y_5776_, v___y_5777_, v___y_5778_, v___y_5779_, v___y_5780_, lean_box(0));
if (lean_obj_tag(v___x_5783_) == 0)
{
lean_object* v_a_5784_; lean_object* v___x_5786_; uint8_t v_isShared_5787_; uint8_t v_isSharedCheck_5810_; 
v_a_5784_ = lean_ctor_get(v___x_5783_, 0);
v_isSharedCheck_5810_ = !lean_is_exclusive(v___x_5783_);
if (v_isSharedCheck_5810_ == 0)
{
v___x_5786_ = v___x_5783_;
v_isShared_5787_ = v_isSharedCheck_5810_;
goto v_resetjp_5785_;
}
else
{
lean_inc(v_a_5784_);
lean_dec(v___x_5783_);
v___x_5786_ = lean_box(0);
v_isShared_5787_ = v_isSharedCheck_5810_;
goto v_resetjp_5785_;
}
v_resetjp_5785_:
{
if (lean_obj_tag(v_a_5784_) == 0)
{
lean_object* v_a_5788_; lean_object* v___x_5790_; uint8_t v_isShared_5791_; uint8_t v_isSharedCheck_5798_; 
v_a_5788_ = lean_ctor_get(v_a_5784_, 0);
v_isSharedCheck_5798_ = !lean_is_exclusive(v_a_5784_);
if (v_isSharedCheck_5798_ == 0)
{
v___x_5790_ = v_a_5784_;
v_isShared_5791_ = v_isSharedCheck_5798_;
goto v_resetjp_5789_;
}
else
{
lean_inc(v_a_5788_);
lean_dec(v_a_5784_);
v___x_5790_ = lean_box(0);
v_isShared_5791_ = v_isSharedCheck_5798_;
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
lean_object* v_reuseFailAlloc_5797_; 
v_reuseFailAlloc_5797_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5797_, 0, v_a_5788_);
v___x_5793_ = v_reuseFailAlloc_5797_;
goto v_reusejp_5792_;
}
v_reusejp_5792_:
{
lean_object* v___x_5795_; 
if (v_isShared_5787_ == 0)
{
lean_ctor_set(v___x_5786_, 0, v___x_5793_);
v___x_5795_ = v___x_5786_;
goto v_reusejp_5794_;
}
else
{
lean_object* v_reuseFailAlloc_5796_; 
v_reuseFailAlloc_5796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5796_, 0, v___x_5793_);
v___x_5795_ = v_reuseFailAlloc_5796_;
goto v_reusejp_5794_;
}
v_reusejp_5794_:
{
return v___x_5795_;
}
}
}
}
else
{
lean_object* v_a_5799_; lean_object* v___x_5801_; uint8_t v_isShared_5802_; uint8_t v_isSharedCheck_5809_; 
v_a_5799_ = lean_ctor_get(v_a_5784_, 0);
v_isSharedCheck_5809_ = !lean_is_exclusive(v_a_5784_);
if (v_isSharedCheck_5809_ == 0)
{
v___x_5801_ = v_a_5784_;
v_isShared_5802_ = v_isSharedCheck_5809_;
goto v_resetjp_5800_;
}
else
{
lean_inc(v_a_5799_);
lean_dec(v_a_5784_);
v___x_5801_ = lean_box(0);
v_isShared_5802_ = v_isSharedCheck_5809_;
goto v_resetjp_5800_;
}
v_resetjp_5800_:
{
lean_object* v___x_5804_; 
if (v_isShared_5802_ == 0)
{
v___x_5804_ = v___x_5801_;
goto v_reusejp_5803_;
}
else
{
lean_object* v_reuseFailAlloc_5808_; 
v_reuseFailAlloc_5808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5808_, 0, v_a_5799_);
v___x_5804_ = v_reuseFailAlloc_5808_;
goto v_reusejp_5803_;
}
v_reusejp_5803_:
{
lean_object* v___x_5806_; 
if (v_isShared_5787_ == 0)
{
lean_ctor_set(v___x_5786_, 0, v___x_5804_);
v___x_5806_ = v___x_5786_;
goto v_reusejp_5805_;
}
else
{
lean_object* v_reuseFailAlloc_5807_; 
v_reuseFailAlloc_5807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5807_, 0, v___x_5804_);
v___x_5806_ = v_reuseFailAlloc_5807_;
goto v_reusejp_5805_;
}
v_reusejp_5805_:
{
return v___x_5806_;
}
}
}
}
}
}
else
{
lean_object* v_a_5811_; lean_object* v___x_5813_; uint8_t v_isShared_5814_; uint8_t v_isSharedCheck_5818_; 
v_a_5811_ = lean_ctor_get(v___x_5783_, 0);
v_isSharedCheck_5818_ = !lean_is_exclusive(v___x_5783_);
if (v_isSharedCheck_5818_ == 0)
{
v___x_5813_ = v___x_5783_;
v_isShared_5814_ = v_isSharedCheck_5818_;
goto v_resetjp_5812_;
}
else
{
lean_inc(v_a_5811_);
lean_dec(v___x_5783_);
v___x_5813_ = lean_box(0);
v_isShared_5814_ = v_isSharedCheck_5818_;
goto v_resetjp_5812_;
}
v_resetjp_5812_:
{
lean_object* v___x_5816_; 
if (v_isShared_5814_ == 0)
{
v___x_5816_ = v___x_5813_;
goto v_reusejp_5815_;
}
else
{
lean_object* v_reuseFailAlloc_5817_; 
v_reuseFailAlloc_5817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5817_, 0, v_a_5811_);
v___x_5816_ = v_reuseFailAlloc_5817_;
goto v_reusejp_5815_;
}
v_reusejp_5815_:
{
return v___x_5816_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0___boxed(lean_object* v_f_5819_, lean_object* v_s_5820_, lean_object* v_a_5821_, lean_object* v_b_5822_, lean_object* v___y_5823_, lean_object* v___y_5824_, lean_object* v___y_5825_, lean_object* v___y_5826_, lean_object* v___y_5827_, lean_object* v___y_5828_, lean_object* v___y_5829_){
_start:
{
lean_object* v_res_5830_; 
v_res_5830_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0(v_f_5819_, v_s_5820_, v_a_5821_, v_b_5822_, v___y_5823_, v___y_5824_, v___y_5825_, v___y_5826_, v___y_5827_, v___y_5828_);
lean_dec(v___y_5828_);
lean_dec_ref(v___y_5827_);
lean_dec(v___y_5826_);
lean_dec_ref(v___y_5825_);
lean_dec(v___y_5824_);
lean_dec_ref(v___y_5823_);
return v_res_5830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg(lean_object* v_map_5831_, lean_object* v_init_5832_, lean_object* v_f_5833_, lean_object* v___y_5834_, lean_object* v___y_5835_, lean_object* v___y_5836_, lean_object* v___y_5837_, lean_object* v___y_5838_, lean_object* v___y_5839_){
_start:
{
lean_object* v___f_5841_; lean_object* v___x_5842_; 
v___f_5841_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_5841_, 0, v_f_5833_);
lean_inc_ref(v_map_5831_);
v___x_5842_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v___f_5841_, v_map_5831_, v_init_5832_, v___y_5834_, v___y_5835_, v___y_5836_, v___y_5837_, v___y_5838_, v___y_5839_);
if (lean_obj_tag(v___x_5842_) == 0)
{
lean_object* v_a_5843_; lean_object* v___x_5845_; uint8_t v_isShared_5846_; uint8_t v_isSharedCheck_5851_; 
v_a_5843_ = lean_ctor_get(v___x_5842_, 0);
v_isSharedCheck_5851_ = !lean_is_exclusive(v___x_5842_);
if (v_isSharedCheck_5851_ == 0)
{
v___x_5845_ = v___x_5842_;
v_isShared_5846_ = v_isSharedCheck_5851_;
goto v_resetjp_5844_;
}
else
{
lean_inc(v_a_5843_);
lean_dec(v___x_5842_);
v___x_5845_ = lean_box(0);
v_isShared_5846_ = v_isSharedCheck_5851_;
goto v_resetjp_5844_;
}
v_resetjp_5844_:
{
lean_object* v_a_5847_; lean_object* v___x_5849_; 
v_a_5847_ = lean_ctor_get(v_a_5843_, 0);
lean_inc(v_a_5847_);
lean_dec(v_a_5843_);
if (v_isShared_5846_ == 0)
{
lean_ctor_set(v___x_5845_, 0, v_a_5847_);
v___x_5849_ = v___x_5845_;
goto v_reusejp_5848_;
}
else
{
lean_object* v_reuseFailAlloc_5850_; 
v_reuseFailAlloc_5850_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5850_, 0, v_a_5847_);
v___x_5849_ = v_reuseFailAlloc_5850_;
goto v_reusejp_5848_;
}
v_reusejp_5848_:
{
return v___x_5849_;
}
}
}
else
{
lean_object* v_a_5852_; lean_object* v___x_5854_; uint8_t v_isShared_5855_; uint8_t v_isSharedCheck_5859_; 
v_a_5852_ = lean_ctor_get(v___x_5842_, 0);
v_isSharedCheck_5859_ = !lean_is_exclusive(v___x_5842_);
if (v_isSharedCheck_5859_ == 0)
{
v___x_5854_ = v___x_5842_;
v_isShared_5855_ = v_isSharedCheck_5859_;
goto v_resetjp_5853_;
}
else
{
lean_inc(v_a_5852_);
lean_dec(v___x_5842_);
v___x_5854_ = lean_box(0);
v_isShared_5855_ = v_isSharedCheck_5859_;
goto v_resetjp_5853_;
}
v_resetjp_5853_:
{
lean_object* v___x_5857_; 
if (v_isShared_5855_ == 0)
{
v___x_5857_ = v___x_5854_;
goto v_reusejp_5856_;
}
else
{
lean_object* v_reuseFailAlloc_5858_; 
v_reuseFailAlloc_5858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5858_, 0, v_a_5852_);
v___x_5857_ = v_reuseFailAlloc_5858_;
goto v_reusejp_5856_;
}
v_reusejp_5856_:
{
return v___x_5857_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg___boxed(lean_object* v_map_5860_, lean_object* v_init_5861_, lean_object* v_f_5862_, lean_object* v___y_5863_, lean_object* v___y_5864_, lean_object* v___y_5865_, lean_object* v___y_5866_, lean_object* v___y_5867_, lean_object* v___y_5868_, lean_object* v___y_5869_){
_start:
{
lean_object* v_res_5870_; 
v_res_5870_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg(v_map_5860_, v_init_5861_, v_f_5862_, v___y_5863_, v___y_5864_, v___y_5865_, v___y_5866_, v___y_5867_, v___y_5868_);
lean_dec(v___y_5868_);
lean_dec_ref(v___y_5867_);
lean_dec(v___y_5866_);
lean_dec_ref(v___y_5865_);
lean_dec(v___y_5864_);
lean_dec_ref(v___y_5863_);
lean_dec_ref(v_map_5860_);
return v_res_5870_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1(void){
_start:
{
lean_object* v___x_5872_; lean_object* v___x_5873_; 
v___x_5872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__0));
v___x_5873_ = l_Lean_stringToMessageData(v___x_5872_);
return v___x_5873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1(lean_object* v_stx_5874_, lean_object* v___x_5875_, lean_object* v_x_5876_, lean_object* v___y_5877_, lean_object* v___y_5878_, lean_object* v___y_5879_, lean_object* v___y_5880_, lean_object* v___y_5881_, lean_object* v___y_5882_){
_start:
{
lean_object* v___x_5884_; lean_object* v___x_5885_; lean_object* v___x_5886_; 
v___x_5884_ = lean_unsigned_to_nat(1u);
v___x_5885_ = l_Lean_Syntax_getArg(v_stx_5874_, v___x_5884_);
v___x_5886_ = lp_mathlib_Mathlib_Tactic_Push_elabHead(v___x_5885_, v___y_5877_, v___y_5878_, v___y_5879_, v___y_5880_, v___y_5881_, v___y_5882_);
if (lean_obj_tag(v___x_5886_) == 0)
{
lean_object* v_a_5887_; lean_object* v___x_5888_; lean_object* v_env_5889_; lean_object* v___x_5890_; lean_object* v_ext_5891_; lean_object* v_toEnvExtension_5892_; lean_object* v_asyncMode_5893_; lean_object* v___f_5894_; lean_object* v___x_5895_; uint8_t v___x_5896_; lean_object* v___x_5897_; lean_object* v___x_5898_; 
v_a_5887_ = lean_ctor_get(v___x_5886_, 0);
lean_inc_n(v_a_5887_, 2);
lean_dec_ref_known(v___x_5886_, 1);
v___x_5888_ = lean_st_ref_get(v___y_5882_);
v_env_5889_ = lean_ctor_get(v___x_5888_, 0);
lean_inc_ref(v_env_5889_);
lean_dec(v___x_5888_);
v___x_5890_ = lp_mathlib_Mathlib_Tactic_Push_pushExt;
v_ext_5891_ = lean_ctor_get(v___x_5890_, 1);
v_toEnvExtension_5892_ = lean_ctor_get(v_ext_5891_, 0);
v_asyncMode_5893_ = lean_ctor_get(v_toEnvExtension_5892_, 2);
v___f_5894_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__0___boxed), 10, 1);
lean_closure_set(v___f_5894_, 0, v_a_5887_);
v___x_5895_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_5875_, v___x_5890_, v_env_5889_, v_asyncMode_5893_);
v___x_5896_ = 0;
v___x_5897_ = lean_box(v___x_5896_);
v___x_5898_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg(v___x_5895_, v___x_5897_, v___f_5894_, v___y_5877_, v___y_5878_, v___y_5879_, v___y_5880_, v___y_5881_, v___y_5882_);
lean_dec(v___x_5895_);
if (lean_obj_tag(v___x_5898_) == 0)
{
lean_object* v_a_5899_; lean_object* v___x_5901_; uint8_t v_isShared_5902_; uint8_t v_isSharedCheck_5915_; 
v_a_5899_ = lean_ctor_get(v___x_5898_, 0);
v_isSharedCheck_5915_ = !lean_is_exclusive(v___x_5898_);
if (v_isSharedCheck_5915_ == 0)
{
v___x_5901_ = v___x_5898_;
v_isShared_5902_ = v_isSharedCheck_5915_;
goto v_resetjp_5900_;
}
else
{
lean_inc(v_a_5899_);
lean_dec(v___x_5898_);
v___x_5901_ = lean_box(0);
v_isShared_5902_ = v_isSharedCheck_5915_;
goto v_resetjp_5900_;
}
v_resetjp_5900_:
{
uint8_t v___x_5903_; 
v___x_5903_ = lean_unbox(v_a_5899_);
lean_dec(v_a_5899_);
if (v___x_5903_ == 0)
{
lean_object* v___x_5904_; lean_object* v___x_5905_; lean_object* v___x_5906_; lean_object* v___x_5907_; lean_object* v___x_5908_; lean_object* v___x_5909_; lean_object* v___x_5910_; 
lean_del_object(v___x_5901_);
v___x_5904_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___closed__1);
v___x_5905_ = lp_mathlib_Mathlib_Tactic_Push_Head_toString(v_a_5887_);
v___x_5906_ = l_Lean_stringToMessageData(v___x_5905_);
v___x_5907_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5907_, 0, v___x_5904_);
lean_ctor_set(v___x_5907_, 1, v___x_5906_);
v___x_5908_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_elabPushConfig_evalConfigItem_spec__0___closed__5);
v___x_5909_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5909_, 0, v___x_5907_);
lean_ctor_set(v___x_5909_, 1, v___x_5908_);
v___x_5910_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1(v___x_5909_, v___y_5877_, v___y_5878_, v___y_5879_, v___y_5880_, v___y_5881_, v___y_5882_);
return v___x_5910_;
}
else
{
lean_object* v___x_5911_; lean_object* v___x_5913_; 
lean_dec(v_a_5887_);
v___x_5911_ = lean_box(0);
if (v_isShared_5902_ == 0)
{
lean_ctor_set(v___x_5901_, 0, v___x_5911_);
v___x_5913_ = v___x_5901_;
goto v_reusejp_5912_;
}
else
{
lean_object* v_reuseFailAlloc_5914_; 
v_reuseFailAlloc_5914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5914_, 0, v___x_5911_);
v___x_5913_ = v_reuseFailAlloc_5914_;
goto v_reusejp_5912_;
}
v_reusejp_5912_:
{
return v___x_5913_;
}
}
}
}
else
{
lean_object* v_a_5916_; lean_object* v___x_5918_; uint8_t v_isShared_5919_; uint8_t v_isSharedCheck_5923_; 
lean_dec(v_a_5887_);
v_a_5916_ = lean_ctor_get(v___x_5898_, 0);
v_isSharedCheck_5923_ = !lean_is_exclusive(v___x_5898_);
if (v_isSharedCheck_5923_ == 0)
{
v___x_5918_ = v___x_5898_;
v_isShared_5919_ = v_isSharedCheck_5923_;
goto v_resetjp_5917_;
}
else
{
lean_inc(v_a_5916_);
lean_dec(v___x_5898_);
v___x_5918_ = lean_box(0);
v_isShared_5919_ = v_isSharedCheck_5923_;
goto v_resetjp_5917_;
}
v_resetjp_5917_:
{
lean_object* v___x_5921_; 
if (v_isShared_5919_ == 0)
{
v___x_5921_ = v___x_5918_;
goto v_reusejp_5920_;
}
else
{
lean_object* v_reuseFailAlloc_5922_; 
v_reuseFailAlloc_5922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5922_, 0, v_a_5916_);
v___x_5921_ = v_reuseFailAlloc_5922_;
goto v_reusejp_5920_;
}
v_reusejp_5920_:
{
return v___x_5921_;
}
}
}
}
else
{
lean_object* v_a_5924_; lean_object* v___x_5926_; uint8_t v_isShared_5927_; uint8_t v_isSharedCheck_5931_; 
v_a_5924_ = lean_ctor_get(v___x_5886_, 0);
v_isSharedCheck_5931_ = !lean_is_exclusive(v___x_5886_);
if (v_isSharedCheck_5931_ == 0)
{
v___x_5926_ = v___x_5886_;
v_isShared_5927_ = v_isSharedCheck_5931_;
goto v_resetjp_5925_;
}
else
{
lean_inc(v_a_5924_);
lean_dec(v___x_5886_);
v___x_5926_ = lean_box(0);
v_isShared_5927_ = v_isSharedCheck_5931_;
goto v_resetjp_5925_;
}
v_resetjp_5925_:
{
lean_object* v___x_5929_; 
if (v_isShared_5927_ == 0)
{
v___x_5929_ = v___x_5926_;
goto v_reusejp_5928_;
}
else
{
lean_object* v_reuseFailAlloc_5930_; 
v_reuseFailAlloc_5930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5930_, 0, v_a_5924_);
v___x_5929_ = v_reuseFailAlloc_5930_;
goto v_reusejp_5928_;
}
v_reusejp_5928_:
{
return v___x_5929_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___boxed(lean_object* v_stx_5932_, lean_object* v___x_5933_, lean_object* v_x_5934_, lean_object* v___y_5935_, lean_object* v___y_5936_, lean_object* v___y_5937_, lean_object* v___y_5938_, lean_object* v___y_5939_, lean_object* v___y_5940_, lean_object* v___y_5941_){
_start:
{
lean_object* v_res_5942_; 
v_res_5942_ = lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1(v_stx_5932_, v___x_5933_, v_x_5934_, v___y_5935_, v___y_5936_, v___y_5937_, v___y_5938_, v___y_5939_, v___y_5940_);
lean_dec(v___y_5940_);
lean_dec_ref(v___y_5939_);
lean_dec(v___y_5938_);
lean_dec_ref(v___y_5937_);
lean_dec(v___y_5936_);
lean_dec_ref(v___y_5935_);
lean_dec_ref(v_x_5934_);
lean_dec_ref(v___x_5933_);
lean_dec(v_stx_5932_);
return v_res_5942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree(lean_object* v_stx_5943_, lean_object* v_a_5944_, lean_object* v_a_5945_){
_start:
{
lean_object* v___x_5947_; lean_object* v___f_5948_; lean_object* v___x_5949_; 
v___x_5947_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0, &lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Push_pushStep___closed__0);
v___f_5948_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Push_elabPushTree___lam__1___boxed), 10, 2);
lean_closure_set(v___f_5948_, 0, v_stx_5943_);
lean_closure_set(v___f_5948_, 1, v___x_5947_);
v___x_5949_ = l_Lean_Elab_Command_runTermElabM___redArg(v___f_5948_, v_a_5944_, v_a_5945_);
return v___x_5949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushTree___boxed(lean_object* v_stx_5950_, lean_object* v_a_5951_, lean_object* v_a_5952_, lean_object* v_a_5953_){
_start:
{
lean_object* v_res_5954_; 
v_res_5954_ = lp_mathlib_Mathlib_Tactic_Push_elabPushTree(v_stx_5950_, v_a_5951_, v_a_5952_);
lean_dec(v_a_5952_);
lean_dec_ref(v_a_5951_);
return v_res_5954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Lean_Meta_DiscrTree_Trie_format___at___00Mathlib_Tactic_Push_elabPushTree_spec__0_spec__0(lean_object* v_a_5955_){
_start:
{
lean_object* v___x_5956_; 
v___x_5956_ = lean_nat_to_int(v_a_5955_);
return v___x_5956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2(lean_object* v_00_u03c3_5957_, lean_object* v_00_u03b2_5958_, lean_object* v_map_5959_, lean_object* v_init_5960_, lean_object* v_f_5961_, lean_object* v___y_5962_, lean_object* v___y_5963_, lean_object* v___y_5964_, lean_object* v___y_5965_, lean_object* v___y_5966_, lean_object* v___y_5967_){
_start:
{
lean_object* v___x_5969_; 
v___x_5969_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___redArg(v_map_5959_, v_init_5960_, v_f_5961_, v___y_5962_, v___y_5963_, v___y_5964_, v___y_5965_, v___y_5966_, v___y_5967_);
return v___x_5969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2___boxed(lean_object* v_00_u03c3_5970_, lean_object* v_00_u03b2_5971_, lean_object* v_map_5972_, lean_object* v_init_5973_, lean_object* v_f_5974_, lean_object* v___y_5975_, lean_object* v___y_5976_, lean_object* v___y_5977_, lean_object* v___y_5978_, lean_object* v___y_5979_, lean_object* v___y_5980_, lean_object* v___y_5981_){
_start:
{
lean_object* v_res_5982_; 
v_res_5982_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2(v_00_u03c3_5970_, v_00_u03b2_5971_, v_map_5972_, v_init_5973_, v_f_5974_, v___y_5975_, v___y_5976_, v___y_5977_, v___y_5978_, v___y_5979_, v___y_5980_);
lean_dec(v___y_5980_);
lean_dec_ref(v___y_5979_);
lean_dec(v___y_5978_);
lean_dec_ref(v___y_5977_);
lean_dec(v___y_5976_);
lean_dec_ref(v___y_5975_);
lean_dec_ref(v_map_5972_);
return v_res_5982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___redArg(lean_object* v_map_5983_, lean_object* v_f_5984_, lean_object* v_init_5985_, lean_object* v___y_5986_, lean_object* v___y_5987_, lean_object* v___y_5988_, lean_object* v___y_5989_, lean_object* v___y_5990_, lean_object* v___y_5991_){
_start:
{
lean_object* v___x_5993_; 
v___x_5993_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v_f_5984_, v_map_5983_, v_init_5985_, v___y_5986_, v___y_5987_, v___y_5988_, v___y_5989_, v___y_5990_, v___y_5991_);
return v___x_5993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___redArg___boxed(lean_object* v_map_5994_, lean_object* v_f_5995_, lean_object* v_init_5996_, lean_object* v___y_5997_, lean_object* v___y_5998_, lean_object* v___y_5999_, lean_object* v___y_6000_, lean_object* v___y_6001_, lean_object* v___y_6002_, lean_object* v___y_6003_){
_start:
{
lean_object* v_res_6004_; 
v_res_6004_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___redArg(v_map_5994_, v_f_5995_, v_init_5996_, v___y_5997_, v___y_5998_, v___y_5999_, v___y_6000_, v___y_6001_, v___y_6002_);
lean_dec(v___y_6002_);
lean_dec_ref(v___y_6001_);
lean_dec(v___y_6000_);
lean_dec_ref(v___y_5999_);
lean_dec(v___y_5998_);
lean_dec_ref(v___y_5997_);
return v_res_6004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6(lean_object* v_00_u03c3_6005_, lean_object* v_00_u03c3_6006_, lean_object* v_00_u03b2_6007_, lean_object* v_map_6008_, lean_object* v_f_6009_, lean_object* v_init_6010_, lean_object* v___y_6011_, lean_object* v___y_6012_, lean_object* v___y_6013_, lean_object* v___y_6014_, lean_object* v___y_6015_, lean_object* v___y_6016_){
_start:
{
lean_object* v___x_6018_; 
v___x_6018_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v_f_6009_, v_map_6008_, v_init_6010_, v___y_6011_, v___y_6012_, v___y_6013_, v___y_6014_, v___y_6015_, v___y_6016_);
return v___x_6018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6___boxed(lean_object* v_00_u03c3_6019_, lean_object* v_00_u03c3_6020_, lean_object* v_00_u03b2_6021_, lean_object* v_map_6022_, lean_object* v_f_6023_, lean_object* v_init_6024_, lean_object* v___y_6025_, lean_object* v___y_6026_, lean_object* v___y_6027_, lean_object* v___y_6028_, lean_object* v___y_6029_, lean_object* v___y_6030_, lean_object* v___y_6031_){
_start:
{
lean_object* v_res_6032_; 
v_res_6032_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6(v_00_u03c3_6019_, v_00_u03c3_6020_, v_00_u03b2_6021_, v_map_6022_, v_f_6023_, v_init_6024_, v___y_6025_, v___y_6026_, v___y_6027_, v___y_6028_, v___y_6029_, v___y_6030_);
lean_dec(v___y_6030_);
lean_dec_ref(v___y_6029_);
lean_dec(v___y_6028_);
lean_dec_ref(v___y_6027_);
lean_dec(v___y_6026_);
lean_dec_ref(v___y_6025_);
return v_res_6032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6(lean_object* v_ref_6033_, lean_object* v_msgData_6034_, uint8_t v_severity_6035_, uint8_t v_isSilent_6036_, lean_object* v___y_6037_, lean_object* v___y_6038_, lean_object* v___y_6039_, lean_object* v___y_6040_, lean_object* v___y_6041_, lean_object* v___y_6042_){
_start:
{
lean_object* v___x_6044_; 
v___x_6044_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___redArg(v_ref_6033_, v_msgData_6034_, v_severity_6035_, v_isSilent_6036_, v___y_6039_, v___y_6040_, v___y_6041_, v___y_6042_);
return v___x_6044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6___boxed(lean_object* v_ref_6045_, lean_object* v_msgData_6046_, lean_object* v_severity_6047_, lean_object* v_isSilent_6048_, lean_object* v___y_6049_, lean_object* v___y_6050_, lean_object* v___y_6051_, lean_object* v___y_6052_, lean_object* v___y_6053_, lean_object* v___y_6054_, lean_object* v___y_6055_){
_start:
{
uint8_t v_severity_boxed_6056_; uint8_t v_isSilent_boxed_6057_; lean_object* v_res_6058_; 
v_severity_boxed_6056_ = lean_unbox(v_severity_6047_);
v_isSilent_boxed_6057_ = lean_unbox(v_isSilent_6048_);
v_res_6058_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_Push_elabPushTree_spec__1_spec__4_spec__6(v_ref_6045_, v_msgData_6046_, v_severity_boxed_6056_, v_isSilent_boxed_6057_, v___y_6049_, v___y_6050_, v___y_6051_, v___y_6052_, v___y_6053_, v___y_6054_);
lean_dec(v___y_6054_);
lean_dec_ref(v___y_6053_);
lean_dec(v___y_6052_);
lean_dec_ref(v___y_6051_);
lean_dec(v___y_6050_);
lean_dec_ref(v___y_6049_);
lean_dec(v_ref_6045_);
return v_res_6058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9(lean_object* v_00_u03c3_6059_, lean_object* v_00_u03c3_6060_, lean_object* v_00_u03b1_6061_, lean_object* v_00_u03b2_6062_, lean_object* v_f_6063_, lean_object* v_x_6064_, lean_object* v_x_6065_, lean_object* v___y_6066_, lean_object* v___y_6067_, lean_object* v___y_6068_, lean_object* v___y_6069_, lean_object* v___y_6070_, lean_object* v___y_6071_){
_start:
{
lean_object* v___x_6073_; 
v___x_6073_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___redArg(v_f_6063_, v_x_6064_, v_x_6065_, v___y_6066_, v___y_6067_, v___y_6068_, v___y_6069_, v___y_6070_, v___y_6071_);
return v___x_6073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9___boxed(lean_object* v_00_u03c3_6074_, lean_object* v_00_u03c3_6075_, lean_object* v_00_u03b1_6076_, lean_object* v_00_u03b2_6077_, lean_object* v_f_6078_, lean_object* v_x_6079_, lean_object* v_x_6080_, lean_object* v___y_6081_, lean_object* v___y_6082_, lean_object* v___y_6083_, lean_object* v___y_6084_, lean_object* v___y_6085_, lean_object* v___y_6086_, lean_object* v___y_6087_){
_start:
{
lean_object* v_res_6088_; 
v_res_6088_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9(v_00_u03c3_6074_, v_00_u03c3_6075_, v_00_u03b1_6076_, v_00_u03b2_6077_, v_f_6078_, v_x_6079_, v_x_6080_, v___y_6081_, v___y_6082_, v___y_6083_, v___y_6084_, v___y_6085_, v___y_6086_);
lean_dec(v___y_6086_);
lean_dec_ref(v___y_6085_);
lean_dec(v___y_6084_);
lean_dec_ref(v___y_6083_);
lean_dec(v___y_6082_);
lean_dec_ref(v___y_6081_);
return v_res_6088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11(lean_object* v_00_u03b1_6089_, lean_object* v_00_u03b2_6090_, lean_object* v_00_u03c3_6091_, lean_object* v_00_u03c3_6092_, lean_object* v_f_6093_, lean_object* v_as_6094_, size_t v_i_6095_, size_t v_stop_6096_, lean_object* v_b_6097_, lean_object* v___y_6098_, lean_object* v___y_6099_, lean_object* v___y_6100_, lean_object* v___y_6101_, lean_object* v___y_6102_, lean_object* v___y_6103_){
_start:
{
lean_object* v___x_6105_; 
v___x_6105_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___redArg(v_f_6093_, v_as_6094_, v_i_6095_, v_stop_6096_, v_b_6097_, v___y_6098_, v___y_6099_, v___y_6100_, v___y_6101_, v___y_6102_, v___y_6103_);
return v___x_6105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11___boxed(lean_object* v_00_u03b1_6106_, lean_object* v_00_u03b2_6107_, lean_object* v_00_u03c3_6108_, lean_object* v_00_u03c3_6109_, lean_object* v_f_6110_, lean_object* v_as_6111_, lean_object* v_i_6112_, lean_object* v_stop_6113_, lean_object* v_b_6114_, lean_object* v___y_6115_, lean_object* v___y_6116_, lean_object* v___y_6117_, lean_object* v___y_6118_, lean_object* v___y_6119_, lean_object* v___y_6120_, lean_object* v___y_6121_){
_start:
{
size_t v_i_boxed_6122_; size_t v_stop_boxed_6123_; lean_object* v_res_6124_; 
v_i_boxed_6122_ = lean_unbox_usize(v_i_6112_);
lean_dec(v_i_6112_);
v_stop_boxed_6123_ = lean_unbox_usize(v_stop_6113_);
lean_dec(v_stop_6113_);
v_res_6124_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__11(v_00_u03b1_6106_, v_00_u03b2_6107_, v_00_u03c3_6108_, v_00_u03c3_6109_, v_f_6110_, v_as_6111_, v_i_boxed_6122_, v_stop_boxed_6123_, v_b_6114_, v___y_6115_, v___y_6116_, v___y_6117_, v___y_6118_, v___y_6119_, v___y_6120_);
lean_dec(v___y_6120_);
lean_dec_ref(v___y_6119_);
lean_dec(v___y_6118_);
lean_dec_ref(v___y_6117_);
lean_dec(v___y_6116_);
lean_dec_ref(v___y_6115_);
lean_dec_ref(v_as_6111_);
return v_res_6124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12(lean_object* v_00_u03c3_6125_, lean_object* v_00_u03c3_6126_, lean_object* v_00_u03b1_6127_, lean_object* v_00_u03b2_6128_, lean_object* v_f_6129_, lean_object* v_keys_6130_, lean_object* v_vals_6131_, lean_object* v_heq_6132_, lean_object* v_i_6133_, lean_object* v_acc_6134_, lean_object* v___y_6135_, lean_object* v___y_6136_, lean_object* v___y_6137_, lean_object* v___y_6138_, lean_object* v___y_6139_, lean_object* v___y_6140_){
_start:
{
lean_object* v___x_6142_; 
v___x_6142_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___redArg(v_f_6129_, v_keys_6130_, v_vals_6131_, v_i_6133_, v_acc_6134_, v___y_6135_, v___y_6136_, v___y_6137_, v___y_6138_, v___y_6139_, v___y_6140_);
return v___x_6142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12___boxed(lean_object** _args){
lean_object* v_00_u03c3_6143_ = _args[0];
lean_object* v_00_u03c3_6144_ = _args[1];
lean_object* v_00_u03b1_6145_ = _args[2];
lean_object* v_00_u03b2_6146_ = _args[3];
lean_object* v_f_6147_ = _args[4];
lean_object* v_keys_6148_ = _args[5];
lean_object* v_vals_6149_ = _args[6];
lean_object* v_heq_6150_ = _args[7];
lean_object* v_i_6151_ = _args[8];
lean_object* v_acc_6152_ = _args[9];
lean_object* v___y_6153_ = _args[10];
lean_object* v___y_6154_ = _args[11];
lean_object* v___y_6155_ = _args[12];
lean_object* v___y_6156_ = _args[13];
lean_object* v___y_6157_ = _args[14];
lean_object* v___y_6158_ = _args[15];
lean_object* v___y_6159_ = _args[16];
_start:
{
lean_object* v_res_6160_; 
v_res_6160_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Mathlib_Tactic_Push_elabPushTree_spec__2_spec__6_spec__9_spec__12(v_00_u03c3_6143_, v_00_u03c3_6144_, v_00_u03b1_6145_, v_00_u03b2_6146_, v_f_6147_, v_keys_6148_, v_vals_6149_, v_heq_6150_, v_i_6151_, v_acc_6152_, v___y_6153_, v___y_6154_, v___y_6155_, v___y_6156_, v___y_6157_, v___y_6158_);
lean_dec(v___y_6158_);
lean_dec_ref(v___y_6157_);
lean_dec(v___y_6156_);
lean_dec_ref(v___y_6155_);
lean_dec(v___y_6154_);
lean_dec_ref(v___y_6153_);
lean_dec_ref(v_vals_6149_);
lean_dec_ref(v_keys_6148_);
return v_res_6160_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Conv_Simp(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Conv_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_initFn_00___x40_Mathlib_Tactic_Push_609137422____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Push_push__neg_use__distrib = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_push__neg_use__distrib);
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Push_0__Mathlib_Tactic_Push_instEvalExprConfig);
lp_mathlib_Mathlib_Tactic_Push_pushStx = _init_lp_mathlib_Mathlib_Tactic_Push_pushStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_pushStx);
lp_mathlib_Mathlib_Tactic_Push_push__neg = _init_lp_mathlib_Mathlib_Tactic_Push_push__neg();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_push__neg);
lp_mathlib_Mathlib_Tactic_Push_pull = _init_lp_mathlib_Mathlib_Tactic_Push_pull();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_pull);
lp_mathlib_Mathlib_Tactic_Push_convPush__________ = _init_lp_mathlib_Mathlib_Tactic_Push_convPush__________();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_convPush__________);
lp_mathlib_Mathlib_Tactic_Push_convPush__neg__ = _init_lp_mathlib_Mathlib_Tactic_Push_convPush__neg__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_convPush__neg__);
lp_mathlib_Mathlib_Tactic_Push_pushCommand = _init_lp_mathlib_Mathlib_Tactic_Push_pushCommand();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_pushCommand);
lp_mathlib_Mathlib_Tactic_Push_convPull________ = _init_lp_mathlib_Mathlib_Tactic_Push_convPull________();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_convPull________);
lp_mathlib_Mathlib_Tactic_Push_pullCommand = _init_lp_mathlib_Mathlib_Tactic_Push_pullCommand();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Push_pullCommand);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Conv_Simp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Conv_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Push(builtin);
}
#ifdef __cplusplus
}
#endif
