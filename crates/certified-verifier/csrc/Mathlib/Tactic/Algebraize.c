// Lean compiler output
// Module: Mathlib.Tactic.Algebraize
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Tower public meta import Mathlib.Tactic.Attr.Core public meta import Mathlib.Tactic.ToAdditive
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfUntil(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshBinderNameForTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_refl(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_define(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_registerParametricAttribute___redArg(lean_object*);
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_note(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
uint8_t l_Lean_ConstantInfo_isInductive(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEqGuarded(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "theorem name must be of the form `RingHom.Property` if no argument is provided"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__0 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__4 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__5 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__4_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__5_value),LEAN_SCALAR_PTR_LITERAL(107, 67, 254, 234, 65, 174, 209, 53)}};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "unexpected algebraize argument"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__7 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "algebraize"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__9 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__9_value),LEAN_SCALAR_PTR_LITERAL(232, 143, 7, 213, 23, 237, 135, 104)}};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__10 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "RingHom"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__13 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__14 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Attr_algebraizeGetParam___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___closed__15 = (const lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "algebraizeAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__4_value),LEAN_SCALAR_PTR_LITERAL(210, 101, 113, 140, 170, 221, 55, 102)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(202, 34, 85, 25, 85, 234, 169, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__3_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 108, .m_capacity = 108, .m_length = 107, .m_data = "Tag that lets the `algebraize` tactic know which `Algebra` property corresponds to this `RingHom`\nproperty."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__3_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__3_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__4_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__2_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__3_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__4_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__4_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__5_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Attr_algebraizeGetParam___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__5_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__5_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__6_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__6_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__6_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__7_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__4_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__5_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__6_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__7_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__7_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Attr_algebraizeAttr;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "algInst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 150, 23, 235, 127, 169, 82, 193)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toAlgebra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(160, 54, 50, 56, 192, 87, 45, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "scalarTowerInst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 172, 173, 63, 251, 227, 58, 60)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(138, 189, 49, 252, 33, 48, 247, 28)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(198, 110, 17, 209, 153, 148, 146, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "of_algebraMap_eq'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IsScalarTower"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 105, 70, 18, 181, 76, 90, 42)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toModule"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__12_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(228, 0, 138, 108, 112, 125, 203, 77)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(170, 85, 122, 254, 221, 148, 79, 195)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "algebraizeInst"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 212, 153, 81, 137, 39, 52, 125)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Hypothesis "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " has type"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = ".\nIts head symbol "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = " is (effectively) tagged with `@[algebraize "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]`, but no constant"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "\nhas been found.\nCheck for missing imports, missing namespaces or typos."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Algebraize"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(97, 221, 196, 45, 173, 251, 17, 179)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(58, 216, 114, 233, 50, 252, 173, 31)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "properties"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(97, 221, 196, 45, 173, 251, 17, 179)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(58, 216, 114, 233, 50, 252, 173, 31)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(250, 125, 65, 27, 13, 71, 240, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "algebraizeTermSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(111, 82, 222, 118, 37, 219, 179, 202)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "withoutPosition"};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__6_value),LEAN_SCALAR_PTR_LITERAL(69, 6, 27, 142, 141, 165, 41, 16)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_algebraizeTermSeq = (const lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticAlgebraize__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 86, 230, 98, 24, 132, 148, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "algebraize "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize____;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = " is not of type `RingHom`"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg(uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "`algebraize []` without arguments has no effect!"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "`algebraize` expects a list of arguments: `algebraize [f]`"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4(uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tacticAlgebraize_only__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(173, 132, 240, 0, 18, 191, 11, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "algebraize_only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "negConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(196, 29, 29, 161, 247, 206, 181, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(15, 175, 253, 162, 13, 144, 103, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__0);
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
lean_ctor_set(v___x_6_, 2, v___x_5_);
lean_ctor_set(v___x_6_, 3, v___x_5_);
lean_ctor_set(v___x_6_, 4, v___x_4_);
lean_ctor_set(v___x_6_, 5, v___x_4_);
lean_ctor_set(v___x_6_, 6, v___x_4_);
lean_ctor_set(v___x_6_, 7, v___x_4_);
lean_ctor_set(v___x_6_, 8, v___x_4_);
lean_ctor_set(v___x_6_, 9, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_7_ = lean_unsigned_to_nat(32u);
v___x_8_ = lean_mk_empty_array_with_capacity(v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4(void){
_start:
{
size_t v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_10_ = ((size_t)5ULL);
v___x_11_ = lean_unsigned_to_nat(0u);
v___x_12_ = lean_unsigned_to_nat(32u);
v___x_13_ = lean_mk_empty_array_with_capacity(v___x_12_);
v___x_14_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__3);
v___x_15_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_15_, 0, v___x_14_);
lean_ctor_set(v___x_15_, 1, v___x_13_);
lean_ctor_set(v___x_15_, 2, v___x_11_);
lean_ctor_set(v___x_15_, 3, v___x_11_);
lean_ctor_set_usize(v___x_15_, 4, v___x_10_);
return v___x_15_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_16_ = lean_box(1);
v___x_17_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__4);
v___x_18_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__1);
v___x_19_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
lean_ctor_set(v___x_19_, 2, v___x_16_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0(lean_object* v_msgData_20_, lean_object* v___y_21_, lean_object* v___y_22_){
_start:
{
lean_object* v___x_24_; lean_object* v_env_25_; lean_object* v_options_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_24_ = lean_st_ref_get(v___y_22_);
v_env_25_ = lean_ctor_get(v___x_24_, 0);
lean_inc_ref(v_env_25_);
lean_dec(v___x_24_);
v_options_26_ = lean_ctor_get(v___y_21_, 2);
v___x_27_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2);
v___x_28_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5);
lean_inc_ref(v_options_26_);
v___x_29_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_29_, 0, v_env_25_);
lean_ctor_set(v___x_29_, 1, v___x_27_);
lean_ctor_set(v___x_29_, 2, v___x_28_);
lean_ctor_set(v___x_29_, 3, v_options_26_);
v___x_30_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v_msgData_20_);
v___x_31_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_31_, 0, v___x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___boxed(lean_object* v_msgData_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0(v_msgData_32_, v___y_33_, v___y_34_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(lean_object* v_msg_37_, lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
lean_object* v_ref_41_; lean_object* v___x_42_; lean_object* v_a_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_51_; 
v_ref_41_ = lean_ctor_get(v___y_38_, 5);
v___x_42_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0(v_msg_37_, v___y_38_, v___y_39_);
v_a_43_ = lean_ctor_get(v___x_42_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_42_);
if (v_isSharedCheck_51_ == 0)
{
v___x_45_ = v___x_42_;
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_a_43_);
lean_dec(v___x_42_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_47_; lean_object* v___x_49_; 
lean_inc(v_ref_41_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v_ref_41_);
lean_ctor_set(v___x_47_, 1, v_a_43_);
if (v_isShared_46_ == 0)
{
lean_ctor_set_tag(v___x_45_, 1);
lean_ctor_set(v___x_45_, 0, v___x_47_);
v___x_49_ = v___x_45_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v___x_47_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg___boxed(lean_object* v_msg_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v_msg_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
return v_res_56_;
}
}
static lean_object* _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__0));
v___x_59_ = l_Lean_stringToMessageData(v___x_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__7));
v___x_71_ = l_Lean_stringToMessageData(v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam(lean_object* v_thm_82_, lean_object* v_stx_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___y_88_; lean_object* v___y_89_; lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__6));
lean_inc(v_stx_83_);
v___x_93_ = l_Lean_Syntax_isOfKind(v_stx_83_, v___x_92_);
if (v___x_93_ == 0)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
lean_dec(v_stx_83_);
lean_dec(v_thm_82_);
v___x_94_ = lean_obj_once(&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8, &lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8_once, _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8);
v___x_95_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v___x_94_, v_a_84_, v_a_85_);
return v___x_95_;
}
else
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_96_ = lean_unsigned_to_nat(0u);
v___x_97_ = l_Lean_Syntax_getArg(v_stx_83_, v___x_96_);
v___x_98_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__10));
v___x_99_ = l_Lean_Syntax_matchesIdent(v___x_97_, v___x_98_);
lean_dec(v___x_97_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v_stx_83_);
lean_dec(v_thm_82_);
v___x_100_ = lean_obj_once(&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8, &lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8_once, _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8);
v___x_101_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v___x_100_, v_a_84_, v_a_85_);
return v___x_101_;
}
else
{
lean_object* v___x_102_; lean_object* v___x_103_; uint8_t v___x_104_; 
v___x_102_ = lean_unsigned_to_nat(1u);
v___x_103_ = l_Lean_Syntax_getArg(v_stx_83_, v___x_102_);
lean_dec(v_stx_83_);
lean_inc(v___x_103_);
v___x_104_ = l_Lean_Syntax_matchesNull(v___x_103_, v___x_102_);
if (v___x_104_ == 0)
{
uint8_t v___x_105_; 
v___x_105_ = l_Lean_Syntax_matchesNull(v___x_103_, v___x_96_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v_thm_82_);
v___x_106_ = lean_obj_once(&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8, &lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8_once, _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8);
v___x_107_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v___x_106_, v_a_84_, v_a_85_);
return v___x_107_;
}
else
{
if (lean_obj_tag(v_thm_82_) == 1)
{
lean_object* v_pre_108_; 
v_pre_108_ = lean_ctor_get(v_thm_82_, 0);
lean_inc(v_pre_108_);
if (lean_obj_tag(v_pre_108_) == 1)
{
lean_object* v_pre_109_; 
v_pre_109_ = lean_ctor_get(v_pre_108_, 0);
if (lean_obj_tag(v_pre_109_) == 0)
{
lean_object* v_str_110_; lean_object* v_str_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v_str_110_ = lean_ctor_get(v_thm_82_, 1);
lean_inc_ref(v_str_110_);
lean_dec_ref_known(v_thm_82_, 2);
v_str_111_ = lean_ctor_get(v_pre_108_, 1);
lean_inc_ref(v_str_111_);
lean_dec_ref_known(v_pre_108_, 2);
v___x_112_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11));
v___x_113_ = lean_string_dec_eq(v_str_111_, v___x_112_);
lean_dec_ref(v_str_111_);
if (v___x_113_ == 0)
{
lean_dec_ref(v_str_110_);
v___y_88_ = v_a_84_;
v___y_89_ = v_a_85_;
goto v___jp_87_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__13));
v___x_115_ = l_Lean_Name_str___override(v___x_114_, v_str_110_);
v___x_116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
return v___x_116_;
}
}
else
{
lean_dec_ref_known(v_pre_108_, 2);
lean_dec_ref_known(v_thm_82_, 2);
v___y_88_ = v_a_84_;
v___y_89_ = v_a_85_;
goto v___jp_87_;
}
}
else
{
lean_dec_ref_known(v_thm_82_, 2);
lean_dec(v_pre_108_);
v___y_88_ = v_a_84_;
v___y_89_ = v_a_85_;
goto v___jp_87_;
}
}
else
{
lean_dec(v_thm_82_);
v___y_88_ = v_a_84_;
v___y_89_ = v_a_85_;
goto v___jp_87_;
}
}
}
else
{
lean_object* v_name_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
lean_dec(v_thm_82_);
v_name_117_ = l_Lean_Syntax_getArg(v___x_103_, v___x_96_);
lean_dec(v___x_103_);
v___x_118_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__15));
lean_inc(v_name_117_);
v___x_119_ = l_Lean_Syntax_isOfKind(v_name_117_, v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; 
lean_dec(v_name_117_);
v___x_120_ = lean_obj_once(&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8, &lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8_once, _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__8);
v___x_121_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v___x_120_, v_a_84_, v_a_85_);
return v___x_121_;
}
else
{
lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_122_ = l_Lean_TSyntax_getId(v_name_117_);
lean_dec(v_name_117_);
v___x_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
return v___x_123_;
}
}
}
}
v___jp_87_:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = lean_obj_once(&lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1, &lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1_once, _init_lp_mathlib_Lean_Attr_algebraizeGetParam___closed__1);
v___x_91_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v___x_90_, v___y_88_, v___y_89_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Attr_algebraizeGetParam___boxed(lean_object* v_thm_124_, lean_object* v_stx_125_, lean_object* v_a_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_Lean_Attr_algebraizeGetParam(v_thm_124_, v_stx_125_, v_a_126_, v_a_127_);
lean_dec(v_a_127_);
lean_dec_ref(v_a_126_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0(lean_object* v_00_u03b1_130_, lean_object* v_msg_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___redArg(v_msg_131_, v___y_132_, v___y_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0___boxed(lean_object* v_00_u03b1_136_, lean_object* v_msg_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0(v_00_u03b1_136_, v_msg_137_, v___y_138_, v___y_139_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(lean_object* v_x_142_, lean_object* v_x_143_, lean_object* v_x_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lean_box(0);
v___x_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object* v_x_149_, lean_object* v_x_150_, lean_object* v_x_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__0_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(v_x_149_, v_x_150_, v_x_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v_x_151_);
lean_dec(v_x_150_);
lean_dec(v_x_149_);
return v_res_154_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(uint8_t v___x_155_, lean_object* v_env_156_, lean_object* v_n_157_, lean_object* v_x_158_){
_start:
{
uint8_t v___x_159_; 
v___x_159_ = l_Lean_Environment_contains(v_env_156_, v_n_157_, v___x_155_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object* v___x_160_, lean_object* v_env_161_, lean_object* v_n_162_, lean_object* v_x_163_){
_start:
{
uint8_t v___x_99__boxed_164_; uint8_t v_res_165_; lean_object* v_r_166_; 
v___x_99__boxed_164_ = lean_unbox(v___x_160_);
v_res_165_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___lam__1_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(v___x_99__boxed_164_, v_env_161_, v_n_162_, v_x_163_);
lean_dec(v_x_163_);
v_r_166_ = lean_box(v_res_165_);
return v_r_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn___closed__7_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_));
v___x_191_ = l_Lean_registerParametricAttribute___redArg(v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2____boxed(lean_object* v_a_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_();
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0(lean_object* v___x_201_, lean_object* v_f_202_, lean_object* v_a_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_205_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_213_) == 0)
{
lean_object* v_a_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v_a_214_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_a_214_);
lean_dec_ref_known(v___x_213_, 1);
v___x_215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__1));
v___x_216_ = l_Lean_Meta_mkFreshBinderNameForTactic___redArg(v___x_215_, v___y_208_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_216_) == 0)
{
lean_object* v_a_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v_a_217_ = lean_ctor_get(v___x_216_, 0);
lean_inc(v_a_217_);
lean_dec_ref_known(v___x_216_, 1);
v___x_218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3));
v___x_219_ = lean_mk_empty_array_with_capacity(v___x_201_);
v___x_220_ = lean_array_push(v___x_219_, v_f_202_);
v___x_221_ = l_Lean_Meta_mkAppM(v___x_218_, v___x_220_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_221_) == 0)
{
lean_object* v_a_222_; lean_object* v___x_223_; 
v_a_222_ = lean_ctor_get(v___x_221_, 0);
lean_inc(v_a_222_);
lean_dec_ref_known(v___x_221_, 1);
v___x_223_ = l_Lean_MVarId_define(v_a_214_, v_a_217_, v_a_203_, v_a_222_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_object* v_a_224_; uint8_t v___x_225_; lean_object* v___x_226_; 
v_a_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc(v_a_224_);
lean_dec_ref_known(v___x_223_, 1);
v___x_225_ = 1;
v___x_226_ = l_Lean_Meta_intro1Core(v_a_224_, v___x_225_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_226_) == 0)
{
lean_object* v_a_227_; lean_object* v_snd_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_246_; 
v_a_227_ = lean_ctor_get(v___x_226_, 0);
lean_inc(v_a_227_);
lean_dec_ref_known(v___x_226_, 1);
v_snd_228_ = lean_ctor_get(v_a_227_, 1);
v_isSharedCheck_246_ = !lean_is_exclusive(v_a_227_);
if (v_isSharedCheck_246_ == 0)
{
lean_object* v_unused_247_; 
v_unused_247_ = lean_ctor_get(v_a_227_, 0);
lean_dec(v_unused_247_);
v___x_230_ = v_a_227_;
v_isShared_231_ = v_isSharedCheck_246_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_snd_228_);
lean_dec(v_a_227_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_246_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_232_; lean_object* v___x_234_; 
v___x_232_ = lean_box(0);
if (v_isShared_231_ == 0)
{
lean_ctor_set_tag(v___x_230_, 1);
lean_ctor_set(v___x_230_, 1, v___x_232_);
lean_ctor_set(v___x_230_, 0, v_snd_228_);
v___x_234_ = v___x_230_;
goto v_reusejp_233_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_snd_228_);
lean_ctor_set(v_reuseFailAlloc_245_, 1, v___x_232_);
v___x_234_ = v_reuseFailAlloc_245_;
goto v_reusejp_233_;
}
v_reusejp_233_:
{
lean_object* v___x_235_; 
v___x_235_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_234_, v___y_205_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
if (lean_obj_tag(v___x_235_) == 0)
{
lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_243_; 
v_isSharedCheck_243_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_243_ == 0)
{
lean_object* v_unused_244_; 
v_unused_244_ = lean_ctor_get(v___x_235_, 0);
lean_dec(v_unused_244_);
v___x_237_ = v___x_235_;
v_isShared_238_ = v_isSharedCheck_243_;
goto v_resetjp_236_;
}
else
{
lean_dec(v___x_235_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_243_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_239_; lean_object* v___x_241_; 
v___x_239_ = lean_box(0);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_239_);
v___x_241_ = v___x_237_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
else
{
return v___x_235_;
}
}
}
}
else
{
lean_object* v_a_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
v_a_248_ = lean_ctor_get(v___x_226_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_255_ == 0)
{
v___x_250_ = v___x_226_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_a_248_);
lean_dec(v___x_226_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_263_; 
v_a_256_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_263_ == 0)
{
v___x_258_ = v___x_223_;
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_223_);
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
}
else
{
lean_object* v_a_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_271_; 
lean_dec(v_a_217_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_203_);
v_a_264_ = lean_ctor_get(v___x_221_, 0);
v_isSharedCheck_271_ = !lean_is_exclusive(v___x_221_);
if (v_isSharedCheck_271_ == 0)
{
v___x_266_ = v___x_221_;
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_a_264_);
lean_dec(v___x_221_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_269_; 
if (v_isShared_267_ == 0)
{
v___x_269_ = v___x_266_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_a_264_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
}
else
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
lean_dec(v_a_214_);
lean_dec_ref(v_a_203_);
lean_dec_ref(v_f_202_);
v_a_272_ = lean_ctor_get(v___x_216_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_216_);
if (v_isSharedCheck_279_ == 0)
{
v___x_274_ = v___x_216_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_216_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_a_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
else
{
lean_object* v_a_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_287_; 
lean_dec_ref(v_a_203_);
lean_dec_ref(v_f_202_);
v_a_280_ = lean_ctor_get(v___x_213_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_213_);
if (v_isSharedCheck_287_ == 0)
{
v___x_282_ = v___x_213_;
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_a_280_);
lean_dec(v___x_213_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_285_; 
if (v_isShared_283_ == 0)
{
v___x_285_ = v___x_282_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v_a_280_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___boxed(lean_object* v___x_288_, lean_object* v_f_289_, lean_object* v_a_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0(v___x_288_, v_f_289_, v_a_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
lean_dec(v___y_294_);
lean_dec_ref(v___y_293_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___x_288_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1(lean_object* v___x_301_, lean_object* v___x_302_, lean_object* v_f_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_snd_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; 
v_snd_313_ = lean_ctor_get(v___x_301_, 1);
v___x_314_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__13));
v___x_315_ = lean_unsigned_to_nat(0u);
v___x_316_ = lean_array_get_borrowed(v___x_302_, v_snd_313_, v___x_315_);
lean_inc(v___x_316_);
v___x_317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
v___x_318_ = lean_unsigned_to_nat(1u);
v___x_319_ = lean_array_get_borrowed(v___x_302_, v_snd_313_, v___x_318_);
lean_inc(v___x_319_);
v___x_320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
v___x_321_ = lean_box(0);
v___x_322_ = lean_unsigned_to_nat(4u);
v___x_323_ = lean_mk_empty_array_with_capacity(v___x_322_);
v___x_324_ = lean_array_push(v___x_323_, v___x_317_);
v___x_325_ = lean_array_push(v___x_324_, v___x_320_);
v___x_326_ = lean_array_push(v___x_325_, v___x_321_);
v___x_327_ = lean_array_push(v___x_326_, v___x_321_);
v___x_328_ = l_Lean_Meta_mkAppOptM(v___x_314_, v___x_327_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
if (lean_obj_tag(v___x_328_) == 0)
{
lean_object* v_a_329_; lean_object* v___x_330_; 
v_a_329_ = lean_ctor_get(v___x_328_, 0);
lean_inc_n(v_a_329_, 2);
lean_dec_ref_known(v___x_328_, 1);
v___x_330_ = l_Lean_Meta_synthInstance_x3f(v_a_329_, v___x_321_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v_a_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_341_; 
v_a_331_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_341_ == 0)
{
v___x_333_ = v___x_330_;
v_isShared_334_ = v_isSharedCheck_341_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_a_331_);
lean_dec(v___x_330_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_341_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
if (lean_obj_tag(v_a_331_) == 0)
{
lean_object* v___f_335_; lean_object* v___x_336_; 
lean_del_object(v___x_333_);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___boxed), 12, 3);
lean_closure_set(v___f_335_, 0, v___x_318_);
lean_closure_set(v___f_335_, 1, v_f_303_);
lean_closure_set(v___f_335_, 2, v_a_329_);
v___x_336_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_335_, v___y_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
return v___x_336_;
}
else
{
lean_object* v___x_337_; lean_object* v___x_339_; 
lean_dec_ref_known(v_a_331_, 1);
lean_dec(v_a_329_);
lean_dec_ref(v_f_303_);
v___x_337_ = lean_box(0);
if (v_isShared_334_ == 0)
{
lean_ctor_set(v___x_333_, 0, v___x_337_);
v___x_339_ = v___x_333_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_337_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
else
{
lean_object* v_a_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_349_; 
lean_dec(v_a_329_);
lean_dec_ref(v_f_303_);
v_a_342_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_349_ == 0)
{
v___x_344_ = v___x_330_;
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_a_342_);
lean_dec(v___x_330_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v___x_347_; 
if (v_isShared_345_ == 0)
{
v___x_347_ = v___x_344_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v_a_342_);
v___x_347_ = v_reuseFailAlloc_348_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
return v___x_347_;
}
}
}
}
else
{
lean_object* v_a_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_357_; 
lean_dec_ref(v_f_303_);
v_a_350_ = lean_ctor_get(v___x_328_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_328_);
if (v_isSharedCheck_357_ == 0)
{
v___x_352_ = v___x_328_;
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_a_350_);
lean_dec(v___x_328_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v___x_355_; 
if (v_isShared_353_ == 0)
{
v___x_355_ = v___x_352_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v_a_350_);
v___x_355_ = v_reuseFailAlloc_356_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
return v___x_355_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1___boxed(lean_object* v___x_358_, lean_object* v___x_359_, lean_object* v_f_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1(v___x_358_, v___x_359_, v_f_360_, v___y_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_, v___y_368_);
lean_dec(v___y_368_);
lean_dec_ref(v___y_367_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v___y_364_);
lean_dec_ref(v___y_363_);
lean_dec(v___y_362_);
lean_dec_ref(v___y_361_);
lean_dec_ref(v___x_359_);
lean_dec_ref(v___x_358_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom(lean_object* v_f_371_, lean_object* v_ft_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_){
_start:
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___y_384_; lean_object* v___x_385_; 
v___x_382_ = l_Lean_instInhabitedExpr;
v___x_383_ = l_Lean_Expr_getAppFnArgs(v_ft_372_);
v___y_384_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__1___boxed), 12, 3);
lean_closure_set(v___y_384_, 0, v___x_383_);
lean_closure_set(v___y_384_, 1, v___x_382_);
lean_closure_set(v___y_384_, 2, v_f_371_);
v___x_385_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_384_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_, v_a_378_, v_a_379_, v_a_380_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___boxed(lean_object* v_f_386_, lean_object* v_ft_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom(v_f_386_, v_ft_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
lean_dec(v_a_395_);
lean_dec_ref(v_a_394_);
lean_dec(v_a_393_);
lean_dec_ref(v_a_392_);
lean_dec(v_a_391_);
lean_dec_ref(v_a_390_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0(lean_object* v___x_413_, lean_object* v___x_414_, lean_object* v___x_415_, lean_object* v___x_416_, lean_object* v___x_417_, uint8_t v___x_418_, lean_object* v___x_419_, lean_object* v_a_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_422_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v_a_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v_a_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_a_431_);
lean_dec_ref_known(v___x_430_, 1);
v___x_432_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__1));
v___x_433_ = l_Lean_Meta_mkFreshBinderNameForTactic___redArg(v___x_432_, v___y_425_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__3));
v___x_436_ = lean_unsigned_to_nat(5u);
v___x_437_ = lean_mk_empty_array_with_capacity(v___x_436_);
lean_inc(v___x_413_);
lean_inc_ref(v___x_437_);
v___x_438_ = lean_array_push(v___x_437_, v___x_413_);
lean_inc(v___x_414_);
lean_inc_ref(v___x_438_);
v___x_439_ = lean_array_push(v___x_438_, v___x_414_);
lean_inc_n(v___x_415_, 3);
v___x_440_ = lean_array_push(v___x_439_, v___x_415_);
v___x_441_ = lean_array_push(v___x_440_, v___x_415_);
v___x_442_ = lean_array_push(v___x_441_, v___x_415_);
v___x_443_ = l_Lean_Meta_mkAppOptM(v___x_435_, v___x_442_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_443_) == 0)
{
lean_object* v_a_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; 
v_a_444_ = lean_ctor_get(v___x_443_, 0);
lean_inc(v_a_444_);
lean_dec_ref_known(v___x_443_, 1);
lean_inc(v___x_416_);
v___x_445_ = lean_array_push(v___x_437_, v___x_416_);
lean_inc(v___x_414_);
v___x_446_ = lean_array_push(v___x_445_, v___x_414_);
lean_inc_n(v___x_415_, 3);
v___x_447_ = lean_array_push(v___x_446_, v___x_415_);
v___x_448_ = lean_array_push(v___x_447_, v___x_415_);
v___x_449_ = lean_array_push(v___x_448_, v___x_415_);
v___x_450_ = l_Lean_Meta_mkAppOptM(v___x_435_, v___x_449_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_450_) == 0)
{
lean_object* v_a_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v_a_451_ = lean_ctor_get(v___x_450_, 0);
lean_inc(v_a_451_);
lean_dec_ref_known(v___x_450_, 1);
lean_inc(v___x_416_);
v___x_452_ = lean_array_push(v___x_438_, v___x_416_);
lean_inc_n(v___x_415_, 3);
v___x_453_ = lean_array_push(v___x_452_, v___x_415_);
v___x_454_ = lean_array_push(v___x_453_, v___x_415_);
v___x_455_ = lean_array_push(v___x_454_, v___x_415_);
v___x_456_ = l_Lean_Meta_mkAppOptM(v___x_435_, v___x_455_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_object* v_a_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v_a_457_ = lean_ctor_get(v___x_456_, 0);
lean_inc(v_a_457_);
lean_dec_ref_known(v___x_456_, 1);
v___x_458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__5));
v___x_459_ = lean_mk_empty_array_with_capacity(v___x_417_);
lean_inc_ref(v___x_459_);
v___x_460_ = lean_array_push(v___x_459_, v_a_451_);
v___x_461_ = lean_array_push(v___x_460_, v_a_457_);
v___x_462_ = l_Lean_Meta_mkAppM(v___x_458_, v___x_461_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_462_) == 0)
{
lean_object* v_a_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v_a_463_ = lean_ctor_get(v___x_462_, 0);
lean_inc(v_a_463_);
lean_dec_ref_known(v___x_462_, 1);
v___x_464_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__7));
v___x_465_ = lean_array_push(v___x_459_, v_a_444_);
v___x_466_ = lean_array_push(v___x_465_, v_a_463_);
v___x_467_ = l_Lean_Meta_mkAppM(v___x_464_, v___x_466_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_467_) == 0)
{
lean_object* v_a_468_; lean_object* v___x_469_; uint8_t v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v_a_468_ = lean_ctor_get(v___x_467_, 0);
lean_inc(v_a_468_);
lean_dec_ref_known(v___x_467_, 1);
v___x_469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_469_, 0, v_a_468_);
v___x_470_ = 0;
v___x_471_ = lean_box(0);
v___x_472_ = l_Lean_Meta_mkFreshExprMVar(v___x_469_, v___x_470_, v___x_471_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_472_) == 0)
{
lean_object* v_a_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v_a_473_ = lean_ctor_get(v___x_472_, 0);
lean_inc(v_a_473_);
lean_dec_ref_known(v___x_472_, 1);
v___x_474_ = l_Lean_Expr_mvarId_x21(v_a_473_);
v___x_475_ = l_Lean_MVarId_refl(v___x_474_, v___x_418_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_475_) == 0)
{
lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_546_; 
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_475_);
if (v_isSharedCheck_546_ == 0)
{
lean_object* v_unused_547_; 
v_unused_547_ = lean_ctor_get(v___x_475_, 0);
lean_dec(v_unused_547_);
v___x_477_ = v___x_475_;
v_isShared_478_ = v_isSharedCheck_546_;
goto v_resetjp_476_;
}
else
{
lean_dec(v___x_475_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_546_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_482_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__8));
v___x_480_ = l_Lean_Name_mkStr2(v___x_419_, v___x_479_);
if (v_isShared_478_ == 0)
{
lean_ctor_set_tag(v___x_477_, 1);
lean_ctor_set(v___x_477_, 0, v_a_473_);
v___x_482_ = v___x_477_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_a_473_);
v___x_482_ = v_reuseFailAlloc_545_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; 
v___x_483_ = lean_unsigned_to_nat(10u);
v___x_484_ = lean_mk_empty_array_with_capacity(v___x_483_);
v___x_485_ = lean_array_push(v___x_484_, v___x_413_);
v___x_486_ = lean_array_push(v___x_485_, v___x_416_);
v___x_487_ = lean_array_push(v___x_486_, v___x_414_);
lean_inc_n(v___x_415_, 5);
v___x_488_ = lean_array_push(v___x_487_, v___x_415_);
v___x_489_ = lean_array_push(v___x_488_, v___x_415_);
v___x_490_ = lean_array_push(v___x_489_, v___x_415_);
v___x_491_ = lean_array_push(v___x_490_, v___x_415_);
v___x_492_ = lean_array_push(v___x_491_, v___x_415_);
v___x_493_ = lean_array_push(v___x_492_, v___x_415_);
v___x_494_ = lean_array_push(v___x_493_, v___x_482_);
v___x_495_ = l_Lean_Meta_mkAppOptM(v___x_480_, v___x_494_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_495_) == 0)
{
lean_object* v_a_496_; lean_object* v___x_497_; 
v_a_496_ = lean_ctor_get(v___x_495_, 0);
lean_inc(v_a_496_);
lean_dec_ref_known(v___x_495_, 1);
v___x_497_ = l_Lean_MVarId_define(v_a_431_, v_a_434_, v_a_420_, v_a_496_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_497_) == 0)
{
lean_object* v_a_498_; lean_object* v___x_499_; 
v_a_498_ = lean_ctor_get(v___x_497_, 0);
lean_inc(v_a_498_);
lean_dec_ref_known(v___x_497_, 1);
v___x_499_ = l_Lean_Meta_intro1Core(v_a_498_, v___x_418_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_object* v_a_500_; lean_object* v_snd_501_; lean_object* v___x_503_; uint8_t v_isShared_504_; uint8_t v_isSharedCheck_519_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_a_500_);
lean_dec_ref_known(v___x_499_, 1);
v_snd_501_ = lean_ctor_get(v_a_500_, 1);
v_isSharedCheck_519_ = !lean_is_exclusive(v_a_500_);
if (v_isSharedCheck_519_ == 0)
{
lean_object* v_unused_520_; 
v_unused_520_ = lean_ctor_get(v_a_500_, 0);
lean_dec(v_unused_520_);
v___x_503_ = v_a_500_;
v_isShared_504_ = v_isSharedCheck_519_;
goto v_resetjp_502_;
}
else
{
lean_inc(v_snd_501_);
lean_dec(v_a_500_);
v___x_503_ = lean_box(0);
v_isShared_504_ = v_isSharedCheck_519_;
goto v_resetjp_502_;
}
v_resetjp_502_:
{
lean_object* v___x_505_; lean_object* v___x_507_; 
v___x_505_ = lean_box(0);
if (v_isShared_504_ == 0)
{
lean_ctor_set_tag(v___x_503_, 1);
lean_ctor_set(v___x_503_, 1, v___x_505_);
lean_ctor_set(v___x_503_, 0, v_snd_501_);
v___x_507_ = v___x_503_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_snd_501_);
lean_ctor_set(v_reuseFailAlloc_518_, 1, v___x_505_);
v___x_507_ = v_reuseFailAlloc_518_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
lean_object* v___x_508_; 
v___x_508_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_507_, v___y_422_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_516_; 
v_isSharedCheck_516_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_516_ == 0)
{
lean_object* v_unused_517_; 
v_unused_517_ = lean_ctor_get(v___x_508_, 0);
lean_dec(v_unused_517_);
v___x_510_ = v___x_508_;
v_isShared_511_ = v_isSharedCheck_516_;
goto v_resetjp_509_;
}
else
{
lean_dec(v___x_508_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_516_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_512_; lean_object* v___x_514_; 
v___x_512_ = lean_box(0);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 0, v___x_512_);
v___x_514_ = v___x_510_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_512_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
else
{
return v___x_508_;
}
}
}
}
else
{
lean_object* v_a_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_528_; 
v_a_521_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_528_ == 0)
{
v___x_523_ = v___x_499_;
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_a_521_);
lean_dec(v___x_499_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_524_ == 0)
{
v___x_526_ = v___x_523_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_a_521_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
else
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
v_a_529_ = lean_ctor_get(v___x_497_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_497_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_497_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_497_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
else
{
lean_object* v_a_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
v_a_537_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_544_ == 0)
{
v___x_539_ = v___x_495_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_a_537_);
lean_dec(v___x_495_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v_a_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
}
}
else
{
lean_dec(v_a_473_);
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
return v___x_475_;
}
}
else
{
lean_object* v_a_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_555_; 
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_548_ = lean_ctor_get(v___x_472_, 0);
v_isSharedCheck_555_ = !lean_is_exclusive(v___x_472_);
if (v_isSharedCheck_555_ == 0)
{
v___x_550_ = v___x_472_;
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_a_548_);
lean_dec(v___x_472_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_553_; 
if (v_isShared_551_ == 0)
{
v___x_553_ = v___x_550_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v_a_548_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
else
{
lean_object* v_a_556_; lean_object* v___x_558_; uint8_t v_isShared_559_; uint8_t v_isSharedCheck_563_; 
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_556_ = lean_ctor_get(v___x_467_, 0);
v_isSharedCheck_563_ = !lean_is_exclusive(v___x_467_);
if (v_isSharedCheck_563_ == 0)
{
v___x_558_ = v___x_467_;
v_isShared_559_ = v_isSharedCheck_563_;
goto v_resetjp_557_;
}
else
{
lean_inc(v_a_556_);
lean_dec(v___x_467_);
v___x_558_ = lean_box(0);
v_isShared_559_ = v_isSharedCheck_563_;
goto v_resetjp_557_;
}
v_resetjp_557_:
{
lean_object* v___x_561_; 
if (v_isShared_559_ == 0)
{
v___x_561_ = v___x_558_;
goto v_reusejp_560_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v_a_556_);
v___x_561_ = v_reuseFailAlloc_562_;
goto v_reusejp_560_;
}
v_reusejp_560_:
{
return v___x_561_;
}
}
}
}
else
{
lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_571_; 
lean_dec_ref(v___x_459_);
lean_dec(v_a_444_);
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_564_ = lean_ctor_get(v___x_462_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_571_ == 0)
{
v___x_566_ = v___x_462_;
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_462_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_569_; 
if (v_isShared_567_ == 0)
{
v___x_569_ = v___x_566_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_a_564_);
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
else
{
lean_object* v_a_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_579_; 
lean_dec(v_a_451_);
lean_dec(v_a_444_);
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_572_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_579_ == 0)
{
v___x_574_ = v___x_456_;
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_a_572_);
lean_dec(v___x_456_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_577_; 
if (v_isShared_575_ == 0)
{
v___x_577_ = v___x_574_;
goto v_reusejp_576_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_a_572_);
v___x_577_ = v_reuseFailAlloc_578_;
goto v_reusejp_576_;
}
v_reusejp_576_:
{
return v___x_577_;
}
}
}
}
else
{
lean_object* v_a_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_587_; 
lean_dec(v_a_444_);
lean_dec_ref(v___x_438_);
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_580_ = lean_ctor_get(v___x_450_, 0);
v_isSharedCheck_587_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_587_ == 0)
{
v___x_582_ = v___x_450_;
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_a_580_);
lean_dec(v___x_450_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v___x_585_; 
if (v_isShared_583_ == 0)
{
v___x_585_ = v___x_582_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_586_; 
v_reuseFailAlloc_586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_586_, 0, v_a_580_);
v___x_585_ = v_reuseFailAlloc_586_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
return v___x_585_;
}
}
}
}
else
{
lean_object* v_a_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_595_; 
lean_dec_ref(v___x_438_);
lean_dec_ref(v___x_437_);
lean_dec(v_a_434_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_588_ = lean_ctor_get(v___x_443_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v___x_443_);
if (v_isSharedCheck_595_ == 0)
{
v___x_590_ = v___x_443_;
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_a_588_);
lean_dec(v___x_443_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_593_; 
if (v_isShared_591_ == 0)
{
v___x_593_ = v___x_590_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_a_588_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
}
}
else
{
lean_object* v_a_596_; lean_object* v___x_598_; uint8_t v_isShared_599_; uint8_t v_isSharedCheck_603_; 
lean_dec(v_a_431_);
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_596_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_603_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_603_ == 0)
{
v___x_598_ = v___x_433_;
v_isShared_599_ = v_isSharedCheck_603_;
goto v_resetjp_597_;
}
else
{
lean_inc(v_a_596_);
lean_dec(v___x_433_);
v___x_598_ = lean_box(0);
v_isShared_599_ = v_isSharedCheck_603_;
goto v_resetjp_597_;
}
v_resetjp_597_:
{
lean_object* v___x_601_; 
if (v_isShared_599_ == 0)
{
v___x_601_ = v___x_598_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_602_; 
v_reuseFailAlloc_602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_602_, 0, v_a_596_);
v___x_601_ = v_reuseFailAlloc_602_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
return v___x_601_;
}
}
}
}
else
{
lean_object* v_a_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_611_; 
lean_dec_ref(v_a_420_);
lean_dec_ref(v___x_419_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec(v___x_414_);
lean_dec(v___x_413_);
v_a_604_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_611_ == 0)
{
v___x_606_ = v___x_430_;
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_a_604_);
lean_dec(v___x_430_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
lean_object* v___x_609_; 
if (v_isShared_607_ == 0)
{
v___x_609_ = v___x_606_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v_a_604_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___boxed(lean_object** _args){
lean_object* v___x_612_ = _args[0];
lean_object* v___x_613_ = _args[1];
lean_object* v___x_614_ = _args[2];
lean_object* v___x_615_ = _args[3];
lean_object* v___x_616_ = _args[4];
lean_object* v___x_617_ = _args[5];
lean_object* v___x_618_ = _args[6];
lean_object* v_a_619_ = _args[7];
lean_object* v___y_620_ = _args[8];
lean_object* v___y_621_ = _args[9];
lean_object* v___y_622_ = _args[10];
lean_object* v___y_623_ = _args[11];
lean_object* v___y_624_ = _args[12];
lean_object* v___y_625_ = _args[13];
lean_object* v___y_626_ = _args[14];
lean_object* v___y_627_ = _args[15];
lean_object* v___y_628_ = _args[16];
_start:
{
uint8_t v___x_4607__boxed_629_; lean_object* v_res_630_; 
v___x_4607__boxed_629_ = lean_unbox(v___x_617_);
v_res_630_ = lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0(v___x_612_, v___x_613_, v___x_614_, v___x_615_, v___x_616_, v___x_4607__boxed_629_, v___x_618_, v_a_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_, v___y_627_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec(v___y_625_);
lean_dec_ref(v___y_624_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
lean_dec(v___y_621_);
lean_dec_ref(v___y_620_);
lean_dec(v___x_616_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1(lean_object* v___x_634_, lean_object* v___x_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_snd_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v_snd_645_ = lean_ctor_get(v___x_634_, 1);
v___x_646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__0));
v___x_647_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___closed__1));
v___x_648_ = lean_unsigned_to_nat(0u);
v___x_649_ = lean_array_get_borrowed(v___x_635_, v_snd_645_, v___x_648_);
lean_inc(v___x_649_);
v___x_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_650_, 0, v___x_649_);
v___x_651_ = lean_unsigned_to_nat(1u);
v___x_652_ = lean_array_get_borrowed(v___x_635_, v_snd_645_, v___x_651_);
lean_inc(v___x_652_);
v___x_653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_653_, 0, v___x_652_);
v___x_654_ = lean_unsigned_to_nat(2u);
v___x_655_ = lean_array_get_borrowed(v___x_635_, v_snd_645_, v___x_654_);
lean_inc(v___x_655_);
v___x_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_656_, 0, v___x_655_);
v___x_657_ = lean_box(0);
v___x_658_ = lean_unsigned_to_nat(6u);
v___x_659_ = lean_mk_empty_array_with_capacity(v___x_658_);
lean_inc_ref(v___x_650_);
v___x_660_ = lean_array_push(v___x_659_, v___x_650_);
lean_inc_ref(v___x_653_);
v___x_661_ = lean_array_push(v___x_660_, v___x_653_);
lean_inc_ref(v___x_656_);
v___x_662_ = lean_array_push(v___x_661_, v___x_656_);
v___x_663_ = lean_array_push(v___x_662_, v___x_657_);
v___x_664_ = lean_array_push(v___x_663_, v___x_657_);
v___x_665_ = lean_array_push(v___x_664_, v___x_657_);
v___x_666_ = l_Lean_Meta_mkAppOptM(v___x_647_, v___x_665_, v___y_640_, v___y_641_, v___y_642_, v___y_643_);
if (lean_obj_tag(v___x_666_) == 0)
{
lean_object* v_a_667_; lean_object* v___x_668_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
lean_inc_n(v_a_667_, 2);
lean_dec_ref_known(v___x_666_, 1);
v___x_668_ = l_Lean_Meta_synthInstance_x3f(v_a_667_, v___x_657_, v___y_640_, v___y_641_, v___y_642_, v___y_643_);
if (lean_obj_tag(v___x_668_) == 0)
{
lean_object* v_a_669_; lean_object* v___x_671_; uint8_t v_isShared_672_; uint8_t v_isSharedCheck_681_; 
v_a_669_ = lean_ctor_get(v___x_668_, 0);
v_isSharedCheck_681_ = !lean_is_exclusive(v___x_668_);
if (v_isSharedCheck_681_ == 0)
{
v___x_671_ = v___x_668_;
v_isShared_672_ = v_isSharedCheck_681_;
goto v_resetjp_670_;
}
else
{
lean_inc(v_a_669_);
lean_dec(v___x_668_);
v___x_671_ = lean_box(0);
v_isShared_672_ = v_isSharedCheck_681_;
goto v_resetjp_670_;
}
v_resetjp_670_:
{
if (lean_obj_tag(v_a_669_) == 0)
{
uint8_t v___x_673_; lean_object* v___x_674_; lean_object* v___f_675_; lean_object* v___x_676_; 
lean_del_object(v___x_671_);
v___x_673_ = 1;
v___x_674_ = lean_box(v___x_673_);
v___f_675_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___boxed), 17, 8);
lean_closure_set(v___f_675_, 0, v___x_650_);
lean_closure_set(v___f_675_, 1, v___x_656_);
lean_closure_set(v___f_675_, 2, v___x_657_);
lean_closure_set(v___f_675_, 3, v___x_653_);
lean_closure_set(v___f_675_, 4, v___x_654_);
lean_closure_set(v___f_675_, 5, v___x_674_);
lean_closure_set(v___f_675_, 6, v___x_646_);
lean_closure_set(v___f_675_, 7, v_a_667_);
v___x_676_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_675_, v___y_636_, v___y_637_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_);
return v___x_676_;
}
else
{
lean_object* v___x_677_; lean_object* v___x_679_; 
lean_dec_ref_known(v_a_669_, 1);
lean_dec(v_a_667_);
lean_dec_ref_known(v___x_656_, 1);
lean_dec_ref_known(v___x_653_, 1);
lean_dec_ref_known(v___x_650_, 1);
v___x_677_ = lean_box(0);
if (v_isShared_672_ == 0)
{
lean_ctor_set(v___x_671_, 0, v___x_677_);
v___x_679_ = v___x_671_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_680_; 
v_reuseFailAlloc_680_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_680_, 0, v___x_677_);
v___x_679_ = v_reuseFailAlloc_680_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
return v___x_679_;
}
}
}
}
else
{
lean_object* v_a_682_; lean_object* v___x_684_; uint8_t v_isShared_685_; uint8_t v_isSharedCheck_689_; 
lean_dec(v_a_667_);
lean_dec_ref_known(v___x_656_, 1);
lean_dec_ref_known(v___x_653_, 1);
lean_dec_ref_known(v___x_650_, 1);
v_a_682_ = lean_ctor_get(v___x_668_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_668_);
if (v_isSharedCheck_689_ == 0)
{
v___x_684_ = v___x_668_;
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
else
{
lean_inc(v_a_682_);
lean_dec(v___x_668_);
v___x_684_ = lean_box(0);
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
v_resetjp_683_:
{
lean_object* v___x_687_; 
if (v_isShared_685_ == 0)
{
v___x_687_ = v___x_684_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v_a_682_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
}
}
else
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec_ref_known(v___x_656_, 1);
lean_dec_ref_known(v___x_653_, 1);
lean_dec_ref_known(v___x_650_, 1);
v_a_690_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_666_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_666_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___boxed(lean_object* v___x_698_, lean_object* v___x_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1(v___x_698_, v___x_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
lean_dec(v___y_705_);
lean_dec_ref(v___y_704_);
lean_dec(v___y_703_);
lean_dec_ref(v___y_702_);
lean_dec(v___y_701_);
lean_dec_ref(v___y_700_);
lean_dec_ref(v___x_699_);
lean_dec_ref(v___x_698_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp(lean_object* v_fn_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_, lean_object* v_a_716_, lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___y_722_; lean_object* v___x_723_; 
v___x_720_ = l_Lean_instInhabitedExpr;
v___x_721_ = l_Lean_Expr_getAppFnArgs(v_fn_710_);
v___y_722_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__1___boxed), 11, 2);
lean_closure_set(v___y_722_, 0, v___x_721_);
lean_closure_set(v___y_722_, 1, v___x_720_);
v___x_723_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_722_, v_a_711_, v_a_712_, v_a_713_, v_a_714_, v_a_715_, v_a_716_, v_a_717_, v_a_718_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___boxed(lean_object* v_fn_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_){
_start:
{
lean_object* v_res_734_; 
v_res_734_ = lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp(v_fn_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_, v_a_729_, v_a_730_, v_a_731_, v_a_732_);
lean_dec(v_a_732_);
lean_dec_ref(v_a_731_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
lean_dec(v_a_728_);
lean_dec_ref(v_a_727_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
return v_res_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0(lean_object* v___y_735_, lean_object* v_mctx_736_, lean_object* v_cache_737_, lean_object* v_a_x3f_738_){
_start:
{
lean_object* v___x_740_; lean_object* v_zetaDeltaFVarIds_741_; lean_object* v_postponed_742_; lean_object* v_diag_743_; lean_object* v___x_745_; uint8_t v_isShared_746_; uint8_t v_isSharedCheck_753_; 
v___x_740_ = lean_st_ref_take(v___y_735_);
v_zetaDeltaFVarIds_741_ = lean_ctor_get(v___x_740_, 2);
v_postponed_742_ = lean_ctor_get(v___x_740_, 3);
v_diag_743_ = lean_ctor_get(v___x_740_, 4);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_740_);
if (v_isSharedCheck_753_ == 0)
{
lean_object* v_unused_754_; lean_object* v_unused_755_; 
v_unused_754_ = lean_ctor_get(v___x_740_, 1);
lean_dec(v_unused_754_);
v_unused_755_ = lean_ctor_get(v___x_740_, 0);
lean_dec(v_unused_755_);
v___x_745_ = v___x_740_;
v_isShared_746_ = v_isSharedCheck_753_;
goto v_resetjp_744_;
}
else
{
lean_inc(v_diag_743_);
lean_inc(v_postponed_742_);
lean_inc(v_zetaDeltaFVarIds_741_);
lean_dec(v___x_740_);
v___x_745_ = lean_box(0);
v_isShared_746_ = v_isSharedCheck_753_;
goto v_resetjp_744_;
}
v_resetjp_744_:
{
lean_object* v___x_748_; 
if (v_isShared_746_ == 0)
{
lean_ctor_set(v___x_745_, 1, v_cache_737_);
lean_ctor_set(v___x_745_, 0, v_mctx_736_);
v___x_748_ = v___x_745_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_mctx_736_);
lean_ctor_set(v_reuseFailAlloc_752_, 1, v_cache_737_);
lean_ctor_set(v_reuseFailAlloc_752_, 2, v_zetaDeltaFVarIds_741_);
lean_ctor_set(v_reuseFailAlloc_752_, 3, v_postponed_742_);
lean_ctor_set(v_reuseFailAlloc_752_, 4, v_diag_743_);
v___x_748_ = v_reuseFailAlloc_752_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; 
v___x_749_ = lean_st_ref_set(v___y_735_, v___x_748_);
v___x_750_ = lean_box(0);
v___x_751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_751_, 0, v___x_750_);
return v___x_751_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0___boxed(lean_object* v___y_756_, lean_object* v_mctx_757_, lean_object* v_cache_758_, lean_object* v_a_x3f_759_, lean_object* v___y_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0(v___y_756_, v_mctx_757_, v_cache_758_, v_a_x3f_759_);
lean_dec(v_a_x3f_759_);
lean_dec(v___y_756_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg(lean_object* v_x_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_){
_start:
{
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v_mctx_774_; lean_object* v_cache_775_; lean_object* v___x_776_; 
v___x_772_ = lean_st_ref_get(v___y_768_);
v___x_773_ = lean_st_ref_get(v___y_768_);
v_mctx_774_ = lean_ctor_get(v___x_772_, 0);
lean_inc_ref(v_mctx_774_);
lean_dec(v___x_772_);
v_cache_775_ = lean_ctor_get(v___x_773_, 1);
lean_inc_ref(v_cache_775_);
lean_dec(v___x_773_);
lean_inc(v___y_770_);
lean_inc_ref(v___y_769_);
lean_inc(v___y_768_);
lean_inc_ref(v___y_767_);
lean_inc(v___y_766_);
lean_inc_ref(v___y_765_);
lean_inc(v___y_764_);
lean_inc_ref(v___y_763_);
v___x_776_ = lean_apply_9(v_x_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_, v___y_769_, v___y_770_, lean_box(0));
if (lean_obj_tag(v___x_776_) == 0)
{
lean_object* v_a_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_793_; 
v_a_777_ = lean_ctor_get(v___x_776_, 0);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_776_);
if (v_isSharedCheck_793_ == 0)
{
v___x_779_ = v___x_776_;
v_isShared_780_ = v_isSharedCheck_793_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_a_777_);
lean_dec(v___x_776_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_793_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_782_; 
lean_inc(v_a_777_);
if (v_isShared_780_ == 0)
{
lean_ctor_set_tag(v___x_779_, 1);
v___x_782_ = v___x_779_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v_a_777_);
v___x_782_ = v_reuseFailAlloc_792_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
lean_object* v___x_783_; lean_object* v___x_785_; uint8_t v_isShared_786_; uint8_t v_isSharedCheck_790_; 
v___x_783_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0(v___y_768_, v_mctx_774_, v_cache_775_, v___x_782_);
lean_dec_ref(v___x_782_);
v_isSharedCheck_790_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_790_ == 0)
{
lean_object* v_unused_791_; 
v_unused_791_ = lean_ctor_get(v___x_783_, 0);
lean_dec(v_unused_791_);
v___x_785_ = v___x_783_;
v_isShared_786_ = v_isSharedCheck_790_;
goto v_resetjp_784_;
}
else
{
lean_dec(v___x_783_);
v___x_785_ = lean_box(0);
v_isShared_786_ = v_isSharedCheck_790_;
goto v_resetjp_784_;
}
v_resetjp_784_:
{
lean_object* v___x_788_; 
if (v_isShared_786_ == 0)
{
lean_ctor_set(v___x_785_, 0, v_a_777_);
v___x_788_ = v___x_785_;
goto v_reusejp_787_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v_a_777_);
v___x_788_ = v_reuseFailAlloc_789_;
goto v_reusejp_787_;
}
v_reusejp_787_:
{
return v___x_788_;
}
}
}
}
}
else
{
lean_object* v_a_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_803_; 
v_a_794_ = lean_ctor_get(v___x_776_, 0);
lean_inc(v_a_794_);
lean_dec_ref_known(v___x_776_, 1);
v___x_795_ = lean_box(0);
v___x_796_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___lam__0(v___y_768_, v_mctx_774_, v_cache_775_, v___x_795_);
v_isSharedCheck_803_ = !lean_is_exclusive(v___x_796_);
if (v_isSharedCheck_803_ == 0)
{
lean_object* v_unused_804_; 
v_unused_804_ = lean_ctor_get(v___x_796_, 0);
lean_dec(v_unused_804_);
v___x_798_ = v___x_796_;
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
else
{
lean_dec(v___x_796_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
lean_object* v___x_801_; 
if (v_isShared_799_ == 0)
{
lean_ctor_set_tag(v___x_798_, 1);
lean_ctor_set(v___x_798_, 0, v_a_794_);
v___x_801_ = v___x_798_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v_a_794_);
v___x_801_ = v_reuseFailAlloc_802_;
goto v_reusejp_800_;
}
v_reusejp_800_:
{
return v___x_801_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg___boxed(lean_object* v_x_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg(v_x_805_, v___y_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_, v___y_813_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
lean_dec(v___y_811_);
lean_dec_ref(v___y_810_);
lean_dec(v___y_809_);
lean_dec_ref(v___y_808_);
lean_dec(v___y_807_);
lean_dec_ref(v___y_806_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0(lean_object* v_00_u03b1_816_, lean_object* v_x_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_827_; 
v___x_827_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg(v_x_817_, v___y_818_, v___y_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_);
return v___x_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___boxed(lean_object* v_00_u03b1_828_, lean_object* v_x_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0(v_00_u03b1_828_, v_x_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_);
lean_dec(v___y_837_);
lean_dec_ref(v___y_836_);
lean_dec(v___y_835_);
lean_dec_ref(v___y_834_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(lean_object* v_e_840_, lean_object* v___y_841_){
_start:
{
uint8_t v___x_843_; 
v___x_843_ = l_Lean_Expr_hasMVar(v_e_840_);
if (v___x_843_ == 0)
{
lean_object* v___x_844_; 
v___x_844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_844_, 0, v_e_840_);
return v___x_844_;
}
else
{
lean_object* v___x_845_; lean_object* v_mctx_846_; lean_object* v___x_847_; lean_object* v_fst_848_; lean_object* v_snd_849_; lean_object* v___x_850_; lean_object* v_cache_851_; lean_object* v_zetaDeltaFVarIds_852_; lean_object* v_postponed_853_; lean_object* v_diag_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_863_; 
v___x_845_ = lean_st_ref_get(v___y_841_);
v_mctx_846_ = lean_ctor_get(v___x_845_, 0);
lean_inc_ref(v_mctx_846_);
lean_dec(v___x_845_);
v___x_847_ = l_Lean_instantiateMVarsCore(v_mctx_846_, v_e_840_);
v_fst_848_ = lean_ctor_get(v___x_847_, 0);
lean_inc(v_fst_848_);
v_snd_849_ = lean_ctor_get(v___x_847_, 1);
lean_inc(v_snd_849_);
lean_dec_ref(v___x_847_);
v___x_850_ = lean_st_ref_take(v___y_841_);
v_cache_851_ = lean_ctor_get(v___x_850_, 1);
v_zetaDeltaFVarIds_852_ = lean_ctor_get(v___x_850_, 2);
v_postponed_853_ = lean_ctor_get(v___x_850_, 3);
v_diag_854_ = lean_ctor_get(v___x_850_, 4);
v_isSharedCheck_863_ = !lean_is_exclusive(v___x_850_);
if (v_isSharedCheck_863_ == 0)
{
lean_object* v_unused_864_; 
v_unused_864_ = lean_ctor_get(v___x_850_, 0);
lean_dec(v_unused_864_);
v___x_856_ = v___x_850_;
v_isShared_857_ = v_isSharedCheck_863_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_diag_854_);
lean_inc(v_postponed_853_);
lean_inc(v_zetaDeltaFVarIds_852_);
lean_inc(v_cache_851_);
lean_dec(v___x_850_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_863_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___x_859_; 
if (v_isShared_857_ == 0)
{
lean_ctor_set(v___x_856_, 0, v_snd_849_);
v___x_859_ = v___x_856_;
goto v_reusejp_858_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v_snd_849_);
lean_ctor_set(v_reuseFailAlloc_862_, 1, v_cache_851_);
lean_ctor_set(v_reuseFailAlloc_862_, 2, v_zetaDeltaFVarIds_852_);
lean_ctor_set(v_reuseFailAlloc_862_, 3, v_postponed_853_);
lean_ctor_set(v_reuseFailAlloc_862_, 4, v_diag_854_);
v___x_859_ = v_reuseFailAlloc_862_;
goto v_reusejp_858_;
}
v_reusejp_858_:
{
lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_860_ = lean_st_ref_set(v___y_841_, v___x_859_);
v___x_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_861_, 0, v_fst_848_);
return v___x_861_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg___boxed(lean_object* v_e_865_, lean_object* v___y_866_, lean_object* v___y_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(v_e_865_, v___y_866_);
lean_dec(v___y_866_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4(lean_object* v_e_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(v_e_869_, v___y_875_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___boxed(lean_object* v_e_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_){
_start:
{
lean_object* v_res_890_; 
v_res_890_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4(v_e_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_, v___y_888_);
lean_dec(v___y_888_);
lean_dec_ref(v___y_887_);
lean_dec(v___y_886_);
lean_dec_ref(v___y_885_);
lean_dec(v___y_884_);
lean_dec_ref(v___y_883_);
lean_dec(v___y_882_);
lean_dec_ref(v___y_881_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(lean_object* v_e_891_, lean_object* v___y_892_){
_start:
{
uint8_t v___x_894_; 
v___x_894_ = l_Lean_Expr_hasMVar(v_e_891_);
if (v___x_894_ == 0)
{
lean_object* v___x_895_; 
v___x_895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_895_, 0, v_e_891_);
return v___x_895_;
}
else
{
lean_object* v___x_896_; lean_object* v_mctx_897_; lean_object* v___x_898_; lean_object* v_fst_899_; lean_object* v_snd_900_; lean_object* v___x_901_; lean_object* v_cache_902_; lean_object* v_zetaDeltaFVarIds_903_; lean_object* v_postponed_904_; lean_object* v_diag_905_; lean_object* v___x_907_; uint8_t v_isShared_908_; uint8_t v_isSharedCheck_914_; 
v___x_896_ = lean_st_ref_get(v___y_892_);
v_mctx_897_ = lean_ctor_get(v___x_896_, 0);
lean_inc_ref(v_mctx_897_);
lean_dec(v___x_896_);
v___x_898_ = l_Lean_instantiateMVarsCore(v_mctx_897_, v_e_891_);
v_fst_899_ = lean_ctor_get(v___x_898_, 0);
lean_inc(v_fst_899_);
v_snd_900_ = lean_ctor_get(v___x_898_, 1);
lean_inc(v_snd_900_);
lean_dec_ref(v___x_898_);
v___x_901_ = lean_st_ref_take(v___y_892_);
v_cache_902_ = lean_ctor_get(v___x_901_, 1);
v_zetaDeltaFVarIds_903_ = lean_ctor_get(v___x_901_, 2);
v_postponed_904_ = lean_ctor_get(v___x_901_, 3);
v_diag_905_ = lean_ctor_get(v___x_901_, 4);
v_isSharedCheck_914_ = !lean_is_exclusive(v___x_901_);
if (v_isSharedCheck_914_ == 0)
{
lean_object* v_unused_915_; 
v_unused_915_ = lean_ctor_get(v___x_901_, 0);
lean_dec(v_unused_915_);
v___x_907_ = v___x_901_;
v_isShared_908_ = v_isSharedCheck_914_;
goto v_resetjp_906_;
}
else
{
lean_inc(v_diag_905_);
lean_inc(v_postponed_904_);
lean_inc(v_zetaDeltaFVarIds_903_);
lean_inc(v_cache_902_);
lean_dec(v___x_901_);
v___x_907_ = lean_box(0);
v_isShared_908_ = v_isSharedCheck_914_;
goto v_resetjp_906_;
}
v_resetjp_906_:
{
lean_object* v___x_910_; 
if (v_isShared_908_ == 0)
{
lean_ctor_set(v___x_907_, 0, v_snd_900_);
v___x_910_ = v___x_907_;
goto v_reusejp_909_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_snd_900_);
lean_ctor_set(v_reuseFailAlloc_913_, 1, v_cache_902_);
lean_ctor_set(v_reuseFailAlloc_913_, 2, v_zetaDeltaFVarIds_903_);
lean_ctor_set(v_reuseFailAlloc_913_, 3, v_postponed_904_);
lean_ctor_set(v_reuseFailAlloc_913_, 4, v_diag_905_);
v___x_910_ = v_reuseFailAlloc_913_;
goto v_reusejp_909_;
}
v_reusejp_909_:
{
lean_object* v___x_911_; lean_object* v___x_912_; 
v___x_911_ = lean_st_ref_set(v___y_892_, v___x_910_);
v___x_912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_912_, 0, v_fst_899_);
return v___x_912_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg___boxed(lean_object* v_e_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v_res_919_; 
v_res_919_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v_e_916_, v___y_917_);
lean_dec(v___y_917_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6(lean_object* v_e_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
lean_object* v___x_926_; 
v___x_926_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v_e_920_, v___y_922_);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___boxed(lean_object* v_e_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6(v_e_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_);
lean_dec(v___y_931_);
lean_dec_ref(v___y_930_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
return v_res_933_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_934_; lean_object* v_dummy_935_; 
v___x_934_ = lean_box(0);
v_dummy_935_ = l_Lean_Expr_sort___override(v___x_934_);
return v_dummy_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(lean_object* v_as_943_, size_t v_i_944_, size_t v_stop_945_, lean_object* v_b_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_){
_start:
{
lean_object* v_a_953_; lean_object* v_val_958_; lean_object* v_val_961_; lean_object* v___y_973_; uint8_t v___x_984_; 
v___x_984_ = lean_usize_dec_eq(v_i_944_, v_stop_945_);
if (v___x_984_ == 0)
{
lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_985_ = lean_array_uget_borrowed(v_as_943_, v_i_944_);
v___x_986_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__2));
lean_inc(v___x_985_);
v___x_987_ = l_Lean_Meta_whnfUntil(v___x_985_, v___x_986_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; lean_object* v___y_990_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
lean_inc(v_a_988_);
lean_dec_ref_known(v___x_987_, 1);
if (lean_obj_tag(v_a_988_) == 0)
{
lean_inc(v___x_985_);
v___y_990_ = v___x_985_;
goto v___jp_989_;
}
else
{
lean_object* v_val_999_; lean_object* v_dummy_1000_; lean_object* v_nargs_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; uint8_t v___x_1008_; 
v_val_999_ = lean_ctor_get(v_a_988_, 0);
lean_inc(v_val_999_);
lean_dec_ref_known(v_a_988_, 1);
v_dummy_1000_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0);
v_nargs_1001_ = l_Lean_Expr_getAppNumArgs(v_val_999_);
lean_inc(v_nargs_1001_);
v___x_1002_ = lean_mk_array(v_nargs_1001_, v_dummy_1000_);
v___x_1003_ = lean_unsigned_to_nat(1u);
v___x_1004_ = lean_nat_sub(v_nargs_1001_, v___x_1003_);
lean_dec(v_nargs_1001_);
v___x_1005_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_val_999_, v___x_1002_, v___x_1004_);
v___x_1006_ = lean_array_get_size(v___x_1005_);
v___x_1007_ = lean_nat_sub(v___x_1006_, v___x_1003_);
v___x_1008_ = lean_nat_dec_lt(v___x_1007_, v___x_1006_);
if (v___x_1008_ == 0)
{
lean_dec(v___x_1007_);
lean_dec_ref(v___x_1005_);
lean_inc(v___x_985_);
v___y_990_ = v___x_985_;
goto v___jp_989_;
}
else
{
lean_object* v___x_1009_; 
v___x_1009_ = lean_array_fget(v___x_1005_, v___x_1007_);
lean_dec(v___x_1007_);
lean_dec_ref(v___x_1005_);
v___y_990_ = v___x_1009_;
goto v___jp_989_;
}
}
v___jp_989_:
{
lean_object* v___x_991_; lean_object* v___x_992_; 
v___x_991_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom___lam__0___closed__3));
v___x_992_ = l_Lean_Meta_whnfUntil(v___y_990_, v___x_991_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
if (lean_obj_tag(v___x_992_) == 0)
{
lean_object* v_a_993_; lean_object* v___x_994_; lean_object* v___x_995_; 
v_a_993_ = lean_ctor_get(v___x_992_, 0);
lean_inc(v_a_993_);
lean_dec_ref_known(v___x_992_, 1);
v___x_994_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__3));
lean_inc(v___x_985_);
v___x_995_ = l_Lean_Meta_whnfUntil(v___x_985_, v___x_994_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
if (lean_obj_tag(v___x_995_) == 0)
{
if (lean_obj_tag(v_a_993_) == 0)
{
lean_object* v_a_996_; 
v_a_996_ = lean_ctor_get(v___x_995_, 0);
lean_inc(v_a_996_);
if (lean_obj_tag(v_a_996_) == 0)
{
v___y_973_ = v___x_995_;
goto v___jp_972_;
}
else
{
lean_object* v_val_997_; 
lean_dec_ref_known(v___x_995_, 1);
v_val_997_ = lean_ctor_get(v_a_996_, 0);
lean_inc(v_val_997_);
lean_dec_ref_known(v_a_996_, 1);
v_val_961_ = v_val_997_;
goto v___jp_960_;
}
}
else
{
lean_object* v_val_998_; 
lean_dec_ref_known(v___x_995_, 1);
v_val_998_ = lean_ctor_get(v_a_993_, 0);
lean_inc(v_val_998_);
lean_dec_ref_known(v_a_993_, 1);
v_val_961_ = v_val_998_;
goto v___jp_960_;
}
}
else
{
lean_dec(v_a_993_);
v___y_973_ = v___x_995_;
goto v___jp_972_;
}
}
else
{
v___y_973_ = v___x_992_;
goto v___jp_972_;
}
}
}
else
{
v___y_973_ = v___x_987_;
goto v___jp_972_;
}
}
else
{
lean_object* v___x_1010_; 
v___x_1010_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1010_, 0, v_b_946_);
return v___x_1010_;
}
v___jp_952_:
{
size_t v___x_954_; size_t v___x_955_; 
v___x_954_ = ((size_t)1ULL);
v___x_955_ = lean_usize_add(v_i_944_, v___x_954_);
v_i_944_ = v___x_955_;
v_b_946_ = v_a_953_;
goto _start;
}
v___jp_957_:
{
lean_object* v___x_959_; 
v___x_959_ = lean_array_push(v_b_946_, v_val_958_);
v_a_953_ = v___x_959_;
goto v___jp_952_;
}
v___jp_960_:
{
lean_object* v_dummy_962_; lean_object* v_nargs_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; uint8_t v___x_970_; 
v_dummy_962_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0);
v_nargs_963_ = l_Lean_Expr_getAppNumArgs(v_val_961_);
lean_inc(v_nargs_963_);
v___x_964_ = lean_mk_array(v_nargs_963_, v_dummy_962_);
v___x_965_ = lean_unsigned_to_nat(1u);
v___x_966_ = lean_nat_sub(v_nargs_963_, v___x_965_);
lean_dec(v_nargs_963_);
v___x_967_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_val_961_, v___x_964_, v___x_966_);
v___x_968_ = lean_array_get_size(v___x_967_);
v___x_969_ = lean_nat_sub(v___x_968_, v___x_965_);
v___x_970_ = lean_nat_dec_lt(v___x_969_, v___x_968_);
if (v___x_970_ == 0)
{
lean_dec(v___x_969_);
lean_dec_ref(v___x_967_);
v_a_953_ = v_b_946_;
goto v___jp_952_;
}
else
{
lean_object* v___x_971_; 
v___x_971_ = lean_array_fget(v___x_967_, v___x_969_);
lean_dec(v___x_969_);
lean_dec_ref(v___x_967_);
v_val_958_ = v___x_971_;
goto v___jp_957_;
}
}
v___jp_972_:
{
if (lean_obj_tag(v___y_973_) == 0)
{
lean_object* v_a_974_; 
v_a_974_ = lean_ctor_get(v___y_973_, 0);
lean_inc(v_a_974_);
lean_dec_ref_known(v___y_973_, 1);
if (lean_obj_tag(v_a_974_) == 0)
{
v_a_953_ = v_b_946_;
goto v___jp_952_;
}
else
{
lean_object* v_val_975_; 
v_val_975_ = lean_ctor_get(v_a_974_, 0);
lean_inc(v_val_975_);
lean_dec_ref_known(v_a_974_, 1);
v_val_958_ = v_val_975_;
goto v___jp_957_;
}
}
else
{
lean_object* v_a_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_983_; 
lean_dec_ref(v_b_946_);
v_a_976_ = lean_ctor_get(v___y_973_, 0);
v_isSharedCheck_983_ = !lean_is_exclusive(v___y_973_);
if (v_isSharedCheck_983_ == 0)
{
v___x_978_ = v___y_973_;
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_a_976_);
lean_dec(v___y_973_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_981_; 
if (v_isShared_979_ == 0)
{
v___x_981_ = v___x_978_;
goto v_reusejp_980_;
}
else
{
lean_object* v_reuseFailAlloc_982_; 
v_reuseFailAlloc_982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_982_, 0, v_a_976_);
v___x_981_ = v_reuseFailAlloc_982_;
goto v_reusejp_980_;
}
v_reusejp_980_:
{
return v___x_981_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___boxed(lean_object* v_as_1011_, lean_object* v_i_1012_, lean_object* v_stop_1013_, lean_object* v_b_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
size_t v_i_boxed_1020_; size_t v_stop_boxed_1021_; lean_object* v_res_1022_; 
v_i_boxed_1020_ = lean_unbox_usize(v_i_1012_);
lean_dec(v_i_1012_);
v_stop_boxed_1021_ = lean_unbox_usize(v_stop_1013_);
lean_dec(v_stop_1013_);
v_res_1022_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(v_as_1011_, v_i_boxed_1020_, v_stop_boxed_1021_, v_b_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_);
lean_dec(v___y_1018_);
lean_dec_ref(v___y_1017_);
lean_dec(v___y_1016_);
lean_dec_ref(v___y_1015_);
lean_dec_ref(v_as_1011_);
return v_res_1022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2(lean_object* v_as_1025_, lean_object* v_start_1026_, lean_object* v_stop_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v___x_1037_; uint8_t v___x_1038_; 
v___x_1037_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___closed__0));
v___x_1038_ = lean_nat_dec_lt(v_start_1026_, v_stop_1027_);
if (v___x_1038_ == 0)
{
lean_object* v___x_1039_; 
v___x_1039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1039_, 0, v___x_1037_);
return v___x_1039_;
}
else
{
lean_object* v___x_1040_; uint8_t v___x_1041_; 
v___x_1040_ = lean_array_get_size(v_as_1025_);
v___x_1041_ = lean_nat_dec_le(v_stop_1027_, v___x_1040_);
if (v___x_1041_ == 0)
{
uint8_t v___x_1042_; 
v___x_1042_ = lean_nat_dec_lt(v_start_1026_, v___x_1040_);
if (v___x_1042_ == 0)
{
lean_object* v___x_1043_; 
v___x_1043_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1043_, 0, v___x_1037_);
return v___x_1043_;
}
else
{
size_t v___x_1044_; size_t v___x_1045_; lean_object* v___x_1046_; 
v___x_1044_ = lean_usize_of_nat(v_start_1026_);
v___x_1045_ = lean_usize_of_nat(v___x_1040_);
v___x_1046_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(v_as_1025_, v___x_1044_, v___x_1045_, v___x_1037_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_);
return v___x_1046_;
}
}
else
{
size_t v___x_1047_; size_t v___x_1048_; lean_object* v___x_1049_; 
v___x_1047_ = lean_usize_of_nat(v_start_1026_);
v___x_1048_ = lean_usize_of_nat(v_stop_1027_);
v___x_1049_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(v_as_1025_, v___x_1047_, v___x_1048_, v___x_1037_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_);
return v___x_1049_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2___boxed(lean_object* v_as_1050_, lean_object* v_start_1051_, lean_object* v_stop_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2(v_as_1050_, v_start_1051_, v_stop_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
lean_dec(v___y_1060_);
lean_dec_ref(v___y_1059_);
lean_dec(v___y_1058_);
lean_dec_ref(v___y_1057_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
lean_dec(v___y_1054_);
lean_dec_ref(v___y_1053_);
lean_dec(v_stop_1052_);
lean_dec(v_start_1051_);
lean_dec_ref(v_as_1050_);
return v_res_1062_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1(void){
_start:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1064_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__0));
v___x_1065_ = l_Lean_stringToMessageData(v___x_1064_);
return v___x_1065_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3(void){
_start:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1067_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__2));
v___x_1068_ = l_Lean_stringToMessageData(v___x_1067_);
return v___x_1068_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5(void){
_start:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___x_1070_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__4));
v___x_1071_ = l_Lean_stringToMessageData(v___x_1070_);
return v___x_1071_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7(void){
_start:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___x_1073_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__6));
v___x_1074_ = l_Lean_stringToMessageData(v___x_1073_);
return v___x_1074_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9(void){
_start:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1076_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__8));
v___x_1077_ = l_Lean_stringToMessageData(v___x_1076_);
return v___x_1077_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11(void){
_start:
{
lean_object* v___x_1079_; lean_object* v___x_1080_; 
v___x_1079_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__10));
v___x_1080_ = l_Lean_stringToMessageData(v___x_1079_);
return v___x_1080_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13(void){
_start:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; 
v___x_1082_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__12));
v___x_1083_ = l_Lean_stringToMessageData(v___x_1082_);
return v___x_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg(lean_object* v_msg_1084_, lean_object* v_declHint_1085_, lean_object* v___y_1086_){
_start:
{
lean_object* v___x_1088_; lean_object* v_env_1089_; uint8_t v___x_1090_; 
v___x_1088_ = lean_st_ref_get(v___y_1086_);
v_env_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc_ref(v_env_1089_);
lean_dec(v___x_1088_);
v___x_1090_ = l_Lean_Name_isAnonymous(v_declHint_1085_);
if (v___x_1090_ == 0)
{
uint8_t v_isExporting_1091_; 
v_isExporting_1091_ = lean_ctor_get_uint8(v_env_1089_, sizeof(void*)*8);
if (v_isExporting_1091_ == 0)
{
lean_object* v___x_1092_; 
lean_dec_ref(v_env_1089_);
lean_dec(v_declHint_1085_);
v___x_1092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1092_, 0, v_msg_1084_);
return v___x_1092_;
}
else
{
lean_object* v___x_1093_; uint8_t v___x_1094_; 
lean_inc_ref(v_env_1089_);
v___x_1093_ = l_Lean_Environment_setExporting(v_env_1089_, v___x_1090_);
lean_inc(v_declHint_1085_);
lean_inc_ref(v___x_1093_);
v___x_1094_ = l_Lean_Environment_contains(v___x_1093_, v_declHint_1085_, v_isExporting_1091_);
if (v___x_1094_ == 0)
{
lean_object* v___x_1095_; 
lean_dec_ref(v___x_1093_);
lean_dec_ref(v_env_1089_);
lean_dec(v_declHint_1085_);
v___x_1095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1095_, 0, v_msg_1084_);
return v___x_1095_;
}
else
{
lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v_c_1101_; lean_object* v___x_1102_; 
v___x_1096_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__2);
v___x_1097_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Attr_algebraizeGetParam_spec__0_spec__0___closed__5);
v___x_1098_ = l_Lean_Options_empty;
v___x_1099_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1093_);
lean_ctor_set(v___x_1099_, 1, v___x_1096_);
lean_ctor_set(v___x_1099_, 2, v___x_1097_);
lean_ctor_set(v___x_1099_, 3, v___x_1098_);
lean_inc(v_declHint_1085_);
v___x_1100_ = l_Lean_MessageData_ofConstName(v_declHint_1085_, v___x_1090_);
v_c_1101_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_1101_, 0, v___x_1099_);
lean_ctor_set(v_c_1101_, 1, v___x_1100_);
v___x_1102_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1089_, v_declHint_1085_);
if (lean_obj_tag(v___x_1102_) == 0)
{
lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
lean_dec_ref(v_env_1089_);
lean_dec(v_declHint_1085_);
v___x_1103_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1);
v___x_1104_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1104_, 0, v___x_1103_);
lean_ctor_set(v___x_1104_, 1, v_c_1101_);
v___x_1105_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__3);
v___x_1106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1106_, 0, v___x_1104_);
lean_ctor_set(v___x_1106_, 1, v___x_1105_);
v___x_1107_ = l_Lean_MessageData_note(v___x_1106_);
v___x_1108_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1108_, 0, v_msg_1084_);
lean_ctor_set(v___x_1108_, 1, v___x_1107_);
v___x_1109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1109_, 0, v___x_1108_);
return v___x_1109_;
}
else
{
lean_object* v_val_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1145_; 
v_val_1110_ = lean_ctor_get(v___x_1102_, 0);
v_isSharedCheck_1145_ = !lean_is_exclusive(v___x_1102_);
if (v_isSharedCheck_1145_ == 0)
{
v___x_1112_ = v___x_1102_;
v_isShared_1113_ = v_isSharedCheck_1145_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_val_1110_);
lean_dec(v___x_1102_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1145_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v_mod_1117_; uint8_t v___x_1118_; 
v___x_1114_ = lean_box(0);
v___x_1115_ = l_Lean_Environment_header(v_env_1089_);
lean_dec_ref(v_env_1089_);
v___x_1116_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1115_);
v_mod_1117_ = lean_array_get(v___x_1114_, v___x_1116_, v_val_1110_);
lean_dec(v_val_1110_);
lean_dec_ref(v___x_1116_);
v___x_1118_ = l_Lean_isPrivateName(v_declHint_1085_);
lean_dec(v_declHint_1085_);
if (v___x_1118_ == 0)
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1130_; 
v___x_1119_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__5);
v___x_1120_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1120_, 0, v___x_1119_);
lean_ctor_set(v___x_1120_, 1, v_c_1101_);
v___x_1121_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__7);
v___x_1122_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1120_);
lean_ctor_set(v___x_1122_, 1, v___x_1121_);
v___x_1123_ = l_Lean_MessageData_ofName(v_mod_1117_);
v___x_1124_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1124_, 0, v___x_1122_);
lean_ctor_set(v___x_1124_, 1, v___x_1123_);
v___x_1125_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__9);
v___x_1126_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___x_1124_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = l_Lean_MessageData_note(v___x_1126_);
v___x_1128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1128_, 0, v_msg_1084_);
lean_ctor_set(v___x_1128_, 1, v___x_1127_);
if (v_isShared_1113_ == 0)
{
lean_ctor_set_tag(v___x_1112_, 0);
lean_ctor_set(v___x_1112_, 0, v___x_1128_);
v___x_1130_ = v___x_1112_;
goto v_reusejp_1129_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v___x_1128_);
v___x_1130_ = v_reuseFailAlloc_1131_;
goto v_reusejp_1129_;
}
v_reusejp_1129_:
{
return v___x_1130_;
}
}
else
{
lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1143_; 
v___x_1132_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__1);
v___x_1133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1132_);
lean_ctor_set(v___x_1133_, 1, v_c_1101_);
v___x_1134_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__11);
v___x_1135_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1135_, 0, v___x_1133_);
lean_ctor_set(v___x_1135_, 1, v___x_1134_);
v___x_1136_ = l_Lean_MessageData_ofName(v_mod_1117_);
v___x_1137_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1137_, 0, v___x_1135_);
lean_ctor_set(v___x_1137_, 1, v___x_1136_);
v___x_1138_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___closed__13);
v___x_1139_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1139_, 0, v___x_1137_);
lean_ctor_set(v___x_1139_, 1, v___x_1138_);
v___x_1140_ = l_Lean_MessageData_note(v___x_1139_);
v___x_1141_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1141_, 0, v_msg_1084_);
lean_ctor_set(v___x_1141_, 1, v___x_1140_);
if (v_isShared_1113_ == 0)
{
lean_ctor_set_tag(v___x_1112_, 0);
lean_ctor_set(v___x_1112_, 0, v___x_1141_);
v___x_1143_ = v___x_1112_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v___x_1141_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1146_; 
lean_dec_ref(v_env_1089_);
lean_dec(v_declHint_1085_);
v___x_1146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1146_, 0, v_msg_1084_);
return v___x_1146_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg___boxed(lean_object* v_msg_1147_, lean_object* v_declHint_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_){
_start:
{
lean_object* v_res_1151_; 
v_res_1151_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg(v_msg_1147_, v_declHint_1148_, v___y_1149_);
lean_dec(v___y_1149_);
return v_res_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14(lean_object* v_msg_1152_, lean_object* v_declHint_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_){
_start:
{
lean_object* v___x_1163_; lean_object* v_a_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1173_; 
v___x_1163_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg(v_msg_1152_, v_declHint_1153_, v___y_1161_);
v_a_1164_ = lean_ctor_get(v___x_1163_, 0);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1163_);
if (v_isSharedCheck_1173_ == 0)
{
v___x_1166_ = v___x_1163_;
v_isShared_1167_ = v_isSharedCheck_1173_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_a_1164_);
lean_dec(v___x_1163_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1173_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1171_; 
v___x_1168_ = l_Lean_unknownIdentifierMessageTag;
v___x_1169_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1169_, 0, v___x_1168_);
lean_ctor_set(v___x_1169_, 1, v_a_1164_);
if (v_isShared_1167_ == 0)
{
lean_ctor_set(v___x_1166_, 0, v___x_1169_);
v___x_1171_ = v___x_1166_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v___x_1169_);
v___x_1171_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
return v___x_1171_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14___boxed(lean_object* v_msg_1174_, lean_object* v_declHint_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_){
_start:
{
lean_object* v_res_1185_; 
v_res_1185_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14(v_msg_1174_, v_declHint_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
lean_dec(v___y_1183_);
lean_dec_ref(v___y_1182_);
lean_dec(v___y_1181_);
lean_dec_ref(v___y_1180_);
lean_dec(v___y_1179_);
lean_dec_ref(v___y_1178_);
lean_dec(v___y_1177_);
lean_dec_ref(v___y_1176_);
return v_res_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(lean_object* v_msgData_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_){
_start:
{
lean_object* v___x_1192_; lean_object* v_env_1193_; lean_object* v___x_1194_; lean_object* v_mctx_1195_; lean_object* v_lctx_1196_; lean_object* v_options_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; 
v___x_1192_ = lean_st_ref_get(v___y_1190_);
v_env_1193_ = lean_ctor_get(v___x_1192_, 0);
lean_inc_ref(v_env_1193_);
lean_dec(v___x_1192_);
v___x_1194_ = lean_st_ref_get(v___y_1188_);
v_mctx_1195_ = lean_ctor_get(v___x_1194_, 0);
lean_inc_ref(v_mctx_1195_);
lean_dec(v___x_1194_);
v_lctx_1196_ = lean_ctor_get(v___y_1187_, 2);
v_options_1197_ = lean_ctor_get(v___y_1189_, 2);
lean_inc_ref(v_options_1197_);
lean_inc_ref(v_lctx_1196_);
v___x_1198_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1198_, 0, v_env_1193_);
lean_ctor_set(v___x_1198_, 1, v_mctx_1195_);
lean_ctor_set(v___x_1198_, 2, v_lctx_1196_);
lean_ctor_set(v___x_1198_, 3, v_options_1197_);
v___x_1199_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1198_);
lean_ctor_set(v___x_1199_, 1, v_msgData_1186_);
v___x_1200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1200_, 0, v___x_1199_);
return v___x_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14___boxed(lean_object* v_msgData_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(v_msgData_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(lean_object* v_msg_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_){
_start:
{
lean_object* v_ref_1214_; lean_object* v___x_1215_; lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1224_; 
v_ref_1214_ = lean_ctor_get(v___y_1211_, 5);
v___x_1215_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(v_msg_1208_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
v_a_1216_ = lean_ctor_get(v___x_1215_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1215_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1218_ = v___x_1215_;
v_isShared_1219_ = v_isSharedCheck_1224_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1215_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1224_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1220_; lean_object* v___x_1222_; 
lean_inc(v_ref_1214_);
v___x_1220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1220_, 0, v_ref_1214_);
lean_ctor_set(v___x_1220_, 1, v_a_1216_);
if (v_isShared_1219_ == 0)
{
lean_ctor_set_tag(v___x_1218_, 1);
lean_ctor_set(v___x_1218_, 0, v___x_1220_);
v___x_1222_ = v___x_1218_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v___x_1220_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg___boxed(lean_object* v_msg_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_){
_start:
{
lean_object* v_res_1231_; 
v_res_1231_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v_msg_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
lean_dec(v___y_1227_);
lean_dec_ref(v___y_1226_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg(lean_object* v_ref_1232_, lean_object* v_msg_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v_fileName_1243_; lean_object* v_fileMap_1244_; lean_object* v_options_1245_; lean_object* v_currRecDepth_1246_; lean_object* v_maxRecDepth_1247_; lean_object* v_ref_1248_; lean_object* v_currNamespace_1249_; lean_object* v_openDecls_1250_; lean_object* v_initHeartbeats_1251_; lean_object* v_maxHeartbeats_1252_; lean_object* v_quotContext_1253_; lean_object* v_currMacroScope_1254_; uint8_t v_diag_1255_; lean_object* v_cancelTk_x3f_1256_; uint8_t v_suppressElabErrors_1257_; lean_object* v_inheritedTraceOptions_1258_; lean_object* v_ref_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; 
v_fileName_1243_ = lean_ctor_get(v___y_1240_, 0);
v_fileMap_1244_ = lean_ctor_get(v___y_1240_, 1);
v_options_1245_ = lean_ctor_get(v___y_1240_, 2);
v_currRecDepth_1246_ = lean_ctor_get(v___y_1240_, 3);
v_maxRecDepth_1247_ = lean_ctor_get(v___y_1240_, 4);
v_ref_1248_ = lean_ctor_get(v___y_1240_, 5);
v_currNamespace_1249_ = lean_ctor_get(v___y_1240_, 6);
v_openDecls_1250_ = lean_ctor_get(v___y_1240_, 7);
v_initHeartbeats_1251_ = lean_ctor_get(v___y_1240_, 8);
v_maxHeartbeats_1252_ = lean_ctor_get(v___y_1240_, 9);
v_quotContext_1253_ = lean_ctor_get(v___y_1240_, 10);
v_currMacroScope_1254_ = lean_ctor_get(v___y_1240_, 11);
v_diag_1255_ = lean_ctor_get_uint8(v___y_1240_, sizeof(void*)*14);
v_cancelTk_x3f_1256_ = lean_ctor_get(v___y_1240_, 12);
v_suppressElabErrors_1257_ = lean_ctor_get_uint8(v___y_1240_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1258_ = lean_ctor_get(v___y_1240_, 13);
v_ref_1259_ = l_Lean_replaceRef(v_ref_1232_, v_ref_1248_);
lean_inc_ref(v_inheritedTraceOptions_1258_);
lean_inc(v_cancelTk_x3f_1256_);
lean_inc(v_currMacroScope_1254_);
lean_inc(v_quotContext_1253_);
lean_inc(v_maxHeartbeats_1252_);
lean_inc(v_initHeartbeats_1251_);
lean_inc(v_openDecls_1250_);
lean_inc(v_currNamespace_1249_);
lean_inc(v_maxRecDepth_1247_);
lean_inc(v_currRecDepth_1246_);
lean_inc_ref(v_options_1245_);
lean_inc_ref(v_fileMap_1244_);
lean_inc_ref(v_fileName_1243_);
v___x_1260_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1260_, 0, v_fileName_1243_);
lean_ctor_set(v___x_1260_, 1, v_fileMap_1244_);
lean_ctor_set(v___x_1260_, 2, v_options_1245_);
lean_ctor_set(v___x_1260_, 3, v_currRecDepth_1246_);
lean_ctor_set(v___x_1260_, 4, v_maxRecDepth_1247_);
lean_ctor_set(v___x_1260_, 5, v_ref_1259_);
lean_ctor_set(v___x_1260_, 6, v_currNamespace_1249_);
lean_ctor_set(v___x_1260_, 7, v_openDecls_1250_);
lean_ctor_set(v___x_1260_, 8, v_initHeartbeats_1251_);
lean_ctor_set(v___x_1260_, 9, v_maxHeartbeats_1252_);
lean_ctor_set(v___x_1260_, 10, v_quotContext_1253_);
lean_ctor_set(v___x_1260_, 11, v_currMacroScope_1254_);
lean_ctor_set(v___x_1260_, 12, v_cancelTk_x3f_1256_);
lean_ctor_set(v___x_1260_, 13, v_inheritedTraceOptions_1258_);
lean_ctor_set_uint8(v___x_1260_, sizeof(void*)*14, v_diag_1255_);
lean_ctor_set_uint8(v___x_1260_, sizeof(void*)*14 + 1, v_suppressElabErrors_1257_);
v___x_1261_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v_msg_1233_, v___y_1238_, v___y_1239_, v___x_1260_, v___y_1241_);
lean_dec_ref_known(v___x_1260_, 14);
return v___x_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg___boxed(lean_object* v_ref_1262_, lean_object* v_msg_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_){
_start:
{
lean_object* v_res_1273_; 
v_res_1273_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg(v_ref_1262_, v_msg_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_);
lean_dec(v___y_1271_);
lean_dec_ref(v___y_1270_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v_ref_1262_);
return v_res_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg(lean_object* v_ref_1274_, lean_object* v_msg_1275_, lean_object* v_declHint_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v___x_1286_; lean_object* v_a_1287_; lean_object* v___x_1288_; 
v___x_1286_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14(v_msg_1275_, v_declHint_1276_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
v_a_1287_ = lean_ctor_get(v___x_1286_, 0);
lean_inc(v_a_1287_);
lean_dec_ref(v___x_1286_);
v___x_1288_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg(v_ref_1274_, v_a_1287_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
return v___x_1288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg___boxed(lean_object* v_ref_1289_, lean_object* v_msg_1290_, lean_object* v_declHint_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v_res_1301_; 
v_res_1301_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg(v_ref_1289_, v_msg_1290_, v_declHint_1291_, v___y_1292_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_, v___y_1299_);
lean_dec(v___y_1299_);
lean_dec_ref(v___y_1298_);
lean_dec(v___y_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v___y_1295_);
lean_dec_ref(v___y_1294_);
lean_dec(v___y_1293_);
lean_dec_ref(v___y_1292_);
lean_dec(v_ref_1289_);
return v_res_1301_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1(void){
_start:
{
lean_object* v___x_1303_; lean_object* v___x_1304_; 
v___x_1303_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__0));
v___x_1304_ = l_Lean_stringToMessageData(v___x_1303_);
return v___x_1304_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3(void){
_start:
{
lean_object* v___x_1306_; lean_object* v___x_1307_; 
v___x_1306_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__2));
v___x_1307_ = l_Lean_stringToMessageData(v___x_1306_);
return v___x_1307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg(lean_object* v_ref_1308_, lean_object* v_constName_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_){
_start:
{
lean_object* v___x_1319_; uint8_t v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; 
v___x_1319_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__1);
v___x_1320_ = 0;
lean_inc(v_constName_1309_);
v___x_1321_ = l_Lean_MessageData_ofConstName(v_constName_1309_, v___x_1320_);
v___x_1322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1319_);
lean_ctor_set(v___x_1322_, 1, v___x_1321_);
v___x_1323_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3);
v___x_1324_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1322_);
lean_ctor_set(v___x_1324_, 1, v___x_1323_);
v___x_1325_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg(v_ref_1308_, v___x_1324_, v_constName_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_);
return v___x_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___boxed(lean_object* v_ref_1326_, lean_object* v_constName_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg(v_ref_1326_, v_constName_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec(v___y_1335_);
lean_dec_ref(v___y_1334_);
lean_dec(v___y_1333_);
lean_dec_ref(v___y_1332_);
lean_dec(v___y_1331_);
lean_dec_ref(v___y_1330_);
lean_dec(v___y_1329_);
lean_dec_ref(v___y_1328_);
lean_dec(v_ref_1326_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg(lean_object* v_constName_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_){
_start:
{
lean_object* v_ref_1348_; lean_object* v___x_1349_; 
v_ref_1348_ = lean_ctor_get(v___y_1345_, 5);
v___x_1349_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg(v_ref_1348_, v_constName_1338_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_, v___y_1344_, v___y_1345_, v___y_1346_);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg___boxed(lean_object* v_constName_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_){
_start:
{
lean_object* v_res_1360_; 
v_res_1360_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg(v_constName_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_, v___y_1356_, v___y_1357_, v___y_1358_);
lean_dec(v___y_1358_);
lean_dec_ref(v___y_1357_);
lean_dec(v___y_1356_);
lean_dec_ref(v___y_1355_);
lean_dec(v___y_1354_);
lean_dec_ref(v___y_1353_);
lean_dec(v___y_1352_);
lean_dec_ref(v___y_1351_);
return v_res_1360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5(lean_object* v_constName_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_){
_start:
{
lean_object* v___x_1371_; lean_object* v_env_1372_; uint8_t v___x_1373_; lean_object* v___x_1374_; 
v___x_1371_ = lean_st_ref_get(v___y_1369_);
v_env_1372_ = lean_ctor_get(v___x_1371_, 0);
lean_inc_ref(v_env_1372_);
lean_dec(v___x_1371_);
v___x_1373_ = 0;
lean_inc(v_constName_1361_);
v___x_1374_ = l_Lean_Environment_find_x3f(v_env_1372_, v_constName_1361_, v___x_1373_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v___x_1375_; 
v___x_1375_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg(v_constName_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_);
return v___x_1375_;
}
else
{
lean_object* v_val_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1383_; 
lean_dec(v_constName_1361_);
v_val_1376_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1383_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1378_ = v___x_1374_;
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_val_1376_);
lean_dec(v___x_1374_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1381_; 
if (v_isShared_1379_ == 0)
{
lean_ctor_set_tag(v___x_1378_, 0);
v___x_1381_ = v___x_1378_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v_val_1376_);
v___x_1381_ = v_reuseFailAlloc_1382_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
return v___x_1381_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5___boxed(lean_object* v_constName_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_){
_start:
{
lean_object* v_res_1394_; 
v_res_1394_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5(v_constName_1384_, v___y_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v___y_1390_);
lean_dec_ref(v___y_1389_);
lean_dec(v___y_1388_);
lean_dec_ref(v___y_1387_);
lean_dec(v___y_1386_);
lean_dec_ref(v___y_1385_);
return v_res_1394_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0(uint8_t v___y_1403_, uint8_t v_suppressElabErrors_1404_, lean_object* v_x_1405_){
_start:
{
if (lean_obj_tag(v_x_1405_) == 1)
{
lean_object* v_pre_1406_; 
v_pre_1406_ = lean_ctor_get(v_x_1405_, 0);
switch(lean_obj_tag(v_pre_1406_))
{
case 1:
{
lean_object* v_pre_1407_; 
v_pre_1407_ = lean_ctor_get(v_pre_1406_, 0);
switch(lean_obj_tag(v_pre_1407_))
{
case 0:
{
lean_object* v_str_1408_; lean_object* v_str_1409_; lean_object* v___x_1410_; uint8_t v___x_1411_; 
v_str_1408_ = lean_ctor_get(v_x_1405_, 1);
v_str_1409_ = lean_ctor_get(v_pre_1406_, 1);
v___x_1410_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__0));
v___x_1411_ = lean_string_dec_eq(v_str_1409_, v___x_1410_);
if (v___x_1411_ == 0)
{
lean_object* v___x_1412_; uint8_t v___x_1413_; 
v___x_1412_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__1));
v___x_1413_ = lean_string_dec_eq(v_str_1409_, v___x_1412_);
if (v___x_1413_ == 0)
{
return v___y_1403_;
}
else
{
lean_object* v___x_1414_; uint8_t v___x_1415_; 
v___x_1414_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__2));
v___x_1415_ = lean_string_dec_eq(v_str_1408_, v___x_1414_);
if (v___x_1415_ == 0)
{
return v___y_1403_;
}
else
{
return v_suppressElabErrors_1404_;
}
}
}
else
{
lean_object* v___x_1416_; uint8_t v___x_1417_; 
v___x_1416_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__3));
v___x_1417_ = lean_string_dec_eq(v_str_1408_, v___x_1416_);
if (v___x_1417_ == 0)
{
return v___y_1403_;
}
else
{
return v_suppressElabErrors_1404_;
}
}
}
case 1:
{
lean_object* v_pre_1418_; 
v_pre_1418_ = lean_ctor_get(v_pre_1407_, 0);
if (lean_obj_tag(v_pre_1418_) == 0)
{
lean_object* v_str_1419_; lean_object* v_str_1420_; lean_object* v_str_1421_; lean_object* v___x_1422_; uint8_t v___x_1423_; 
v_str_1419_ = lean_ctor_get(v_x_1405_, 1);
v_str_1420_ = lean_ctor_get(v_pre_1406_, 1);
v_str_1421_ = lean_ctor_get(v_pre_1407_, 1);
v___x_1422_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__4));
v___x_1423_ = lean_string_dec_eq(v_str_1421_, v___x_1422_);
if (v___x_1423_ == 0)
{
return v___y_1403_;
}
else
{
lean_object* v___x_1424_; uint8_t v___x_1425_; 
v___x_1424_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__5));
v___x_1425_ = lean_string_dec_eq(v_str_1420_, v___x_1424_);
if (v___x_1425_ == 0)
{
return v___y_1403_;
}
else
{
lean_object* v___x_1426_; uint8_t v___x_1427_; 
v___x_1426_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__6));
v___x_1427_ = lean_string_dec_eq(v_str_1419_, v___x_1426_);
if (v___x_1427_ == 0)
{
return v___y_1403_;
}
else
{
return v_suppressElabErrors_1404_;
}
}
}
}
else
{
return v___y_1403_;
}
}
default: 
{
return v___y_1403_;
}
}
}
case 0:
{
lean_object* v_str_1428_; lean_object* v___x_1429_; uint8_t v___x_1430_; 
v_str_1428_ = lean_ctor_get(v_x_1405_, 1);
v___x_1429_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___closed__7));
v___x_1430_ = lean_string_dec_eq(v_str_1428_, v___x_1429_);
if (v___x_1430_ == 0)
{
return v___y_1403_;
}
else
{
return v_suppressElabErrors_1404_;
}
}
default: 
{
return v___y_1403_;
}
}
}
else
{
return v___y_1403_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___boxed(lean_object* v___y_1431_, lean_object* v_suppressElabErrors_1432_, lean_object* v_x_1433_){
_start:
{
uint8_t v___y_44774__boxed_1434_; uint8_t v_suppressElabErrors_boxed_1435_; uint8_t v_res_1436_; lean_object* v_r_1437_; 
v___y_44774__boxed_1434_ = lean_unbox(v___y_1431_);
v_suppressElabErrors_boxed_1435_ = lean_unbox(v_suppressElabErrors_1432_);
v_res_1436_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0(v___y_44774__boxed_1434_, v_suppressElabErrors_boxed_1435_, v_x_1433_);
lean_dec(v_x_1433_);
v_r_1437_ = lean_box(v_res_1436_);
return v_r_1437_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15(lean_object* v_opts_1438_, lean_object* v_opt_1439_){
_start:
{
lean_object* v_name_1440_; lean_object* v_defValue_1441_; lean_object* v_map_1442_; lean_object* v___x_1443_; 
v_name_1440_ = lean_ctor_get(v_opt_1439_, 0);
v_defValue_1441_ = lean_ctor_get(v_opt_1439_, 1);
v_map_1442_ = lean_ctor_get(v_opts_1438_, 0);
v___x_1443_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1442_, v_name_1440_);
if (lean_obj_tag(v___x_1443_) == 0)
{
uint8_t v___x_1444_; 
v___x_1444_ = lean_unbox(v_defValue_1441_);
return v___x_1444_;
}
else
{
lean_object* v_val_1445_; 
v_val_1445_ = lean_ctor_get(v___x_1443_, 0);
lean_inc(v_val_1445_);
lean_dec_ref_known(v___x_1443_, 1);
if (lean_obj_tag(v_val_1445_) == 1)
{
uint8_t v_v_1446_; 
v_v_1446_ = lean_ctor_get_uint8(v_val_1445_, 0);
lean_dec_ref_known(v_val_1445_, 0);
return v_v_1446_;
}
else
{
uint8_t v___x_1447_; 
lean_dec(v_val_1445_);
v___x_1447_ = lean_unbox(v_defValue_1441_);
return v___x_1447_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15___boxed(lean_object* v_opts_1448_, lean_object* v_opt_1449_){
_start:
{
uint8_t v_res_1450_; lean_object* v_r_1451_; 
v_res_1450_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15(v_opts_1448_, v_opt_1449_);
lean_dec_ref(v_opt_1449_);
lean_dec_ref(v_opts_1448_);
v_r_1451_ = lean_box(v_res_1450_);
return v_r_1451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(lean_object* v_ref_1453_, lean_object* v_msgData_1454_, uint8_t v_severity_1455_, uint8_t v_isSilent_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_){
_start:
{
lean_object* v___y_1463_; lean_object* v___y_1464_; lean_object* v___y_1465_; lean_object* v___y_1466_; uint8_t v___y_1467_; lean_object* v___y_1468_; uint8_t v___y_1469_; lean_object* v___y_1470_; lean_object* v___y_1471_; lean_object* v___y_1499_; lean_object* v___y_1500_; uint8_t v___y_1501_; lean_object* v___y_1502_; lean_object* v___y_1503_; uint8_t v___y_1504_; uint8_t v___y_1505_; lean_object* v___y_1506_; lean_object* v___y_1524_; lean_object* v___y_1525_; uint8_t v___y_1526_; lean_object* v___y_1527_; uint8_t v___y_1528_; lean_object* v___y_1529_; uint8_t v___y_1530_; lean_object* v___y_1531_; lean_object* v___y_1535_; lean_object* v___y_1536_; uint8_t v___y_1537_; lean_object* v___y_1538_; lean_object* v___y_1539_; uint8_t v___y_1540_; uint8_t v___y_1541_; uint8_t v___x_1546_; lean_object* v___y_1548_; uint8_t v___y_1549_; lean_object* v___y_1550_; lean_object* v___y_1551_; lean_object* v___y_1552_; uint8_t v___y_1553_; uint8_t v___y_1554_; uint8_t v___y_1556_; uint8_t v___x_1571_; 
v___x_1546_ = 2;
v___x_1571_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1455_, v___x_1546_);
if (v___x_1571_ == 0)
{
v___y_1556_ = v___x_1571_;
goto v___jp_1555_;
}
else
{
uint8_t v___x_1572_; 
lean_inc_ref(v_msgData_1454_);
v___x_1572_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1454_);
v___y_1556_ = v___x_1572_;
goto v___jp_1555_;
}
v___jp_1462_:
{
lean_object* v___x_1472_; lean_object* v_currNamespace_1473_; lean_object* v_openDecls_1474_; lean_object* v_env_1475_; lean_object* v_nextMacroScope_1476_; lean_object* v_ngen_1477_; lean_object* v_auxDeclNGen_1478_; lean_object* v_traceState_1479_; lean_object* v_cache_1480_; lean_object* v_messages_1481_; lean_object* v_infoState_1482_; lean_object* v_snapshotTasks_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1497_; 
v___x_1472_ = lean_st_ref_take(v___y_1471_);
v_currNamespace_1473_ = lean_ctor_get(v___y_1470_, 6);
v_openDecls_1474_ = lean_ctor_get(v___y_1470_, 7);
v_env_1475_ = lean_ctor_get(v___x_1472_, 0);
v_nextMacroScope_1476_ = lean_ctor_get(v___x_1472_, 1);
v_ngen_1477_ = lean_ctor_get(v___x_1472_, 2);
v_auxDeclNGen_1478_ = lean_ctor_get(v___x_1472_, 3);
v_traceState_1479_ = lean_ctor_get(v___x_1472_, 4);
v_cache_1480_ = lean_ctor_get(v___x_1472_, 5);
v_messages_1481_ = lean_ctor_get(v___x_1472_, 6);
v_infoState_1482_ = lean_ctor_get(v___x_1472_, 7);
v_snapshotTasks_1483_ = lean_ctor_get(v___x_1472_, 8);
v_isSharedCheck_1497_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1485_ = v___x_1472_;
v_isShared_1486_ = v_isSharedCheck_1497_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_snapshotTasks_1483_);
lean_inc(v_infoState_1482_);
lean_inc(v_messages_1481_);
lean_inc(v_cache_1480_);
lean_inc(v_traceState_1479_);
lean_inc(v_auxDeclNGen_1478_);
lean_inc(v_ngen_1477_);
lean_inc(v_nextMacroScope_1476_);
lean_inc(v_env_1475_);
lean_dec(v___x_1472_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1497_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1492_; 
lean_inc(v_openDecls_1474_);
lean_inc(v_currNamespace_1473_);
v___x_1487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1487_, 0, v_currNamespace_1473_);
lean_ctor_set(v___x_1487_, 1, v_openDecls_1474_);
v___x_1488_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1488_, 0, v___x_1487_);
lean_ctor_set(v___x_1488_, 1, v___y_1465_);
lean_inc_ref(v___y_1464_);
lean_inc_ref(v___y_1468_);
v___x_1489_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1489_, 0, v___y_1468_);
lean_ctor_set(v___x_1489_, 1, v___y_1463_);
lean_ctor_set(v___x_1489_, 2, v___y_1466_);
lean_ctor_set(v___x_1489_, 3, v___y_1464_);
lean_ctor_set(v___x_1489_, 4, v___x_1488_);
lean_ctor_set_uint8(v___x_1489_, sizeof(void*)*5, v___y_1469_);
lean_ctor_set_uint8(v___x_1489_, sizeof(void*)*5 + 1, v___y_1467_);
lean_ctor_set_uint8(v___x_1489_, sizeof(void*)*5 + 2, v_isSilent_1456_);
v___x_1490_ = l_Lean_MessageLog_add(v___x_1489_, v_messages_1481_);
if (v_isShared_1486_ == 0)
{
lean_ctor_set(v___x_1485_, 6, v___x_1490_);
v___x_1492_ = v___x_1485_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_env_1475_);
lean_ctor_set(v_reuseFailAlloc_1496_, 1, v_nextMacroScope_1476_);
lean_ctor_set(v_reuseFailAlloc_1496_, 2, v_ngen_1477_);
lean_ctor_set(v_reuseFailAlloc_1496_, 3, v_auxDeclNGen_1478_);
lean_ctor_set(v_reuseFailAlloc_1496_, 4, v_traceState_1479_);
lean_ctor_set(v_reuseFailAlloc_1496_, 5, v_cache_1480_);
lean_ctor_set(v_reuseFailAlloc_1496_, 6, v___x_1490_);
lean_ctor_set(v_reuseFailAlloc_1496_, 7, v_infoState_1482_);
lean_ctor_set(v_reuseFailAlloc_1496_, 8, v_snapshotTasks_1483_);
v___x_1492_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1493_ = lean_st_ref_set(v___y_1471_, v___x_1492_);
v___x_1494_ = lean_box(0);
v___x_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1495_, 0, v___x_1494_);
return v___x_1495_;
}
}
}
v___jp_1498_:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v_a_1509_; lean_object* v___x_1511_; uint8_t v_isShared_1512_; uint8_t v_isSharedCheck_1522_; 
v___x_1507_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1454_);
v___x_1508_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(v___x_1507_, v___y_1457_, v___y_1458_, v___y_1459_, v___y_1460_);
v_a_1509_ = lean_ctor_get(v___x_1508_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1508_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1511_ = v___x_1508_;
v_isShared_1512_ = v_isSharedCheck_1522_;
goto v_resetjp_1510_;
}
else
{
lean_inc(v_a_1509_);
lean_dec(v___x_1508_);
v___x_1511_ = lean_box(0);
v_isShared_1512_ = v_isSharedCheck_1522_;
goto v_resetjp_1510_;
}
v_resetjp_1510_:
{
lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; 
lean_inc_ref_n(v___y_1500_, 2);
v___x_1513_ = l_Lean_FileMap_toPosition(v___y_1500_, v___y_1502_);
lean_dec(v___y_1502_);
v___x_1514_ = l_Lean_FileMap_toPosition(v___y_1500_, v___y_1506_);
lean_dec(v___y_1506_);
v___x_1515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1515_, 0, v___x_1514_);
v___x_1516_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___closed__0));
if (v___y_1501_ == 0)
{
lean_del_object(v___x_1511_);
lean_dec_ref(v___y_1499_);
v___y_1463_ = v___x_1513_;
v___y_1464_ = v___x_1516_;
v___y_1465_ = v_a_1509_;
v___y_1466_ = v___x_1515_;
v___y_1467_ = v___y_1504_;
v___y_1468_ = v___y_1503_;
v___y_1469_ = v___y_1505_;
v___y_1470_ = v___y_1459_;
v___y_1471_ = v___y_1460_;
goto v___jp_1462_;
}
else
{
uint8_t v___x_1517_; 
lean_inc(v_a_1509_);
v___x_1517_ = l_Lean_MessageData_hasTag(v___y_1499_, v_a_1509_);
if (v___x_1517_ == 0)
{
lean_object* v___x_1518_; lean_object* v___x_1520_; 
lean_dec_ref_known(v___x_1515_, 1);
lean_dec_ref(v___x_1513_);
lean_dec(v_a_1509_);
v___x_1518_ = lean_box(0);
if (v_isShared_1512_ == 0)
{
lean_ctor_set(v___x_1511_, 0, v___x_1518_);
v___x_1520_ = v___x_1511_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v___x_1518_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
else
{
lean_del_object(v___x_1511_);
v___y_1463_ = v___x_1513_;
v___y_1464_ = v___x_1516_;
v___y_1465_ = v_a_1509_;
v___y_1466_ = v___x_1515_;
v___y_1467_ = v___y_1504_;
v___y_1468_ = v___y_1503_;
v___y_1469_ = v___y_1505_;
v___y_1470_ = v___y_1459_;
v___y_1471_ = v___y_1460_;
goto v___jp_1462_;
}
}
}
}
v___jp_1523_:
{
lean_object* v___x_1532_; 
v___x_1532_ = l_Lean_Syntax_getTailPos_x3f(v___y_1527_, v___y_1530_);
lean_dec(v___y_1527_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_inc(v___y_1531_);
v___y_1499_ = v___y_1524_;
v___y_1500_ = v___y_1525_;
v___y_1501_ = v___y_1526_;
v___y_1502_ = v___y_1531_;
v___y_1503_ = v___y_1529_;
v___y_1504_ = v___y_1528_;
v___y_1505_ = v___y_1530_;
v___y_1506_ = v___y_1531_;
goto v___jp_1498_;
}
else
{
lean_object* v_val_1533_; 
v_val_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_val_1533_);
lean_dec_ref_known(v___x_1532_, 1);
v___y_1499_ = v___y_1524_;
v___y_1500_ = v___y_1525_;
v___y_1501_ = v___y_1526_;
v___y_1502_ = v___y_1531_;
v___y_1503_ = v___y_1529_;
v___y_1504_ = v___y_1528_;
v___y_1505_ = v___y_1530_;
v___y_1506_ = v_val_1533_;
goto v___jp_1498_;
}
}
v___jp_1534_:
{
lean_object* v_ref_1542_; lean_object* v___x_1543_; 
v_ref_1542_ = l_Lean_replaceRef(v_ref_1453_, v___y_1539_);
v___x_1543_ = l_Lean_Syntax_getPos_x3f(v_ref_1542_, v___y_1540_);
if (lean_obj_tag(v___x_1543_) == 0)
{
lean_object* v___x_1544_; 
v___x_1544_ = lean_unsigned_to_nat(0u);
v___y_1524_ = v___y_1535_;
v___y_1525_ = v___y_1536_;
v___y_1526_ = v___y_1537_;
v___y_1527_ = v_ref_1542_;
v___y_1528_ = v___y_1541_;
v___y_1529_ = v___y_1538_;
v___y_1530_ = v___y_1540_;
v___y_1531_ = v___x_1544_;
goto v___jp_1523_;
}
else
{
lean_object* v_val_1545_; 
v_val_1545_ = lean_ctor_get(v___x_1543_, 0);
lean_inc(v_val_1545_);
lean_dec_ref_known(v___x_1543_, 1);
v___y_1524_ = v___y_1535_;
v___y_1525_ = v___y_1536_;
v___y_1526_ = v___y_1537_;
v___y_1527_ = v_ref_1542_;
v___y_1528_ = v___y_1541_;
v___y_1529_ = v___y_1538_;
v___y_1530_ = v___y_1540_;
v___y_1531_ = v_val_1545_;
goto v___jp_1523_;
}
}
v___jp_1547_:
{
if (v___y_1554_ == 0)
{
v___y_1535_ = v___y_1550_;
v___y_1536_ = v___y_1548_;
v___y_1537_ = v___y_1549_;
v___y_1538_ = v___y_1552_;
v___y_1539_ = v___y_1551_;
v___y_1540_ = v___y_1553_;
v___y_1541_ = v_severity_1455_;
goto v___jp_1534_;
}
else
{
v___y_1535_ = v___y_1550_;
v___y_1536_ = v___y_1548_;
v___y_1537_ = v___y_1549_;
v___y_1538_ = v___y_1552_;
v___y_1539_ = v___y_1551_;
v___y_1540_ = v___y_1553_;
v___y_1541_ = v___x_1546_;
goto v___jp_1534_;
}
}
v___jp_1555_:
{
if (v___y_1556_ == 0)
{
lean_object* v_fileName_1557_; lean_object* v_fileMap_1558_; lean_object* v_options_1559_; lean_object* v_ref_1560_; uint8_t v_suppressElabErrors_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___f_1564_; uint8_t v___x_1565_; uint8_t v___x_1566_; 
v_fileName_1557_ = lean_ctor_get(v___y_1459_, 0);
v_fileMap_1558_ = lean_ctor_get(v___y_1459_, 1);
v_options_1559_ = lean_ctor_get(v___y_1459_, 2);
v_ref_1560_ = lean_ctor_get(v___y_1459_, 5);
v_suppressElabErrors_1561_ = lean_ctor_get_uint8(v___y_1459_, sizeof(void*)*14 + 1);
v___x_1562_ = lean_box(v___y_1556_);
v___x_1563_ = lean_box(v_suppressElabErrors_1561_);
v___f_1564_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1564_, 0, v___x_1562_);
lean_closure_set(v___f_1564_, 1, v___x_1563_);
v___x_1565_ = 1;
v___x_1566_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1455_, v___x_1565_);
if (v___x_1566_ == 0)
{
v___y_1548_ = v_fileMap_1558_;
v___y_1549_ = v_suppressElabErrors_1561_;
v___y_1550_ = v___f_1564_;
v___y_1551_ = v_ref_1560_;
v___y_1552_ = v_fileName_1557_;
v___y_1553_ = v___y_1556_;
v___y_1554_ = v___x_1566_;
goto v___jp_1547_;
}
else
{
lean_object* v___x_1567_; uint8_t v___x_1568_; 
v___x_1567_ = l_Lean_warningAsError;
v___x_1568_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15(v_options_1559_, v___x_1567_);
v___y_1548_ = v_fileMap_1558_;
v___y_1549_ = v_suppressElabErrors_1561_;
v___y_1550_ = v___f_1564_;
v___y_1551_ = v_ref_1560_;
v___y_1552_ = v_fileName_1557_;
v___y_1553_ = v___y_1556_;
v___y_1554_ = v___x_1568_;
goto v___jp_1547_;
}
}
else
{
lean_object* v___x_1569_; lean_object* v___x_1570_; 
lean_dec_ref(v_msgData_1454_);
v___x_1569_ = lean_box(0);
v___x_1570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1570_, 0, v___x_1569_);
return v___x_1570_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___boxed(lean_object* v_ref_1573_, lean_object* v_msgData_1574_, lean_object* v_severity_1575_, lean_object* v_isSilent_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_){
_start:
{
uint8_t v_severity_boxed_1582_; uint8_t v_isSilent_boxed_1583_; lean_object* v_res_1584_; 
v_severity_boxed_1582_ = lean_unbox(v_severity_1575_);
v_isSilent_boxed_1583_ = lean_unbox(v_isSilent_1576_);
v_res_1584_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(v_ref_1573_, v_msgData_1574_, v_severity_boxed_1582_, v_isSilent_boxed_1583_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_);
lean_dec(v___y_1580_);
lean_dec_ref(v___y_1579_);
lean_dec(v___y_1578_);
lean_dec_ref(v___y_1577_);
lean_dec(v_ref_1573_);
return v_res_1584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9(lean_object* v_msgData_1585_, uint8_t v_severity_1586_, uint8_t v_isSilent_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_, lean_object* v___y_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_){
_start:
{
lean_object* v_ref_1597_; lean_object* v___x_1598_; 
v_ref_1597_ = lean_ctor_get(v___y_1594_, 5);
v___x_1598_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(v_ref_1597_, v_msgData_1585_, v_severity_1586_, v_isSilent_1587_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9___boxed(lean_object* v_msgData_1599_, lean_object* v_severity_1600_, lean_object* v_isSilent_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_){
_start:
{
uint8_t v_severity_boxed_1611_; uint8_t v_isSilent_boxed_1612_; lean_object* v_res_1613_; 
v_severity_boxed_1611_ = lean_unbox(v_severity_1600_);
v_isSilent_boxed_1612_ = lean_unbox(v_isSilent_1601_);
v_res_1613_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9(v_msgData_1599_, v_severity_boxed_1611_, v_isSilent_boxed_1612_, v___y_1602_, v___y_1603_, v___y_1604_, v___y_1605_, v___y_1606_, v___y_1607_, v___y_1608_, v___y_1609_);
lean_dec(v___y_1609_);
lean_dec_ref(v___y_1608_);
lean_dec(v___y_1607_);
lean_dec_ref(v___y_1606_);
lean_dec(v___y_1605_);
lean_dec_ref(v___y_1604_);
lean_dec(v___y_1603_);
lean_dec_ref(v___y_1602_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7(lean_object* v_msgData_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_){
_start:
{
uint8_t v___x_1624_; uint8_t v___x_1625_; lean_object* v___x_1626_; 
v___x_1624_ = 1;
v___x_1625_ = 0;
v___x_1626_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9(v_msgData_1614_, v___x_1624_, v___x_1625_, v___y_1615_, v___y_1616_, v___y_1617_, v___y_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_);
return v___x_1626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7___boxed(lean_object* v_msgData_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_){
_start:
{
lean_object* v_res_1637_; 
v_res_1637_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7(v_msgData_1627_, v___y_1628_, v___y_1629_, v___y_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
lean_dec(v___y_1635_);
lean_dec_ref(v___y_1634_);
lean_dec(v___y_1633_);
lean_dec_ref(v___y_1632_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1630_);
lean_dec(v___y_1629_);
lean_dec_ref(v___y_1628_);
return v_res_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0(lean_object* v_v_1638_, lean_object* v___x_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_){
_start:
{
lean_object* v___x_1649_; 
v___x_1649_ = l_Lean_Meta_isExprDefEq(v_v_1638_, v___x_1639_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
return v___x_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0___boxed(lean_object* v_v_1650_, lean_object* v___x_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_){
_start:
{
lean_object* v_res_1661_; 
v_res_1661_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0(v_v_1650_, v___x_1651_, v___y_1652_, v___y_1653_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_, v___y_1659_);
lean_dec(v___y_1659_);
lean_dec_ref(v___y_1658_);
lean_dec(v___y_1657_);
lean_dec_ref(v___y_1656_);
lean_dec(v___y_1655_);
lean_dec_ref(v___y_1654_);
lean_dec(v___y_1653_);
lean_dec_ref(v___y_1652_);
return v_res_1661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1(lean_object* v_v_1662_, lean_object* v_as_1663_, size_t v_i_1664_, size_t v_stop_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_){
_start:
{
uint8_t v___x_1675_; 
v___x_1675_ = lean_usize_dec_eq(v_i_1664_, v_stop_1665_);
if (v___x_1675_ == 0)
{
lean_object* v___x_1676_; lean_object* v___f_1677_; lean_object* v___x_1678_; 
v___x_1676_ = lean_array_uget_borrowed(v_as_1663_, v_i_1664_);
lean_inc(v___x_1676_);
lean_inc_ref(v_v_1662_);
v___f_1677_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_1677_, 0, v_v_1662_);
lean_closure_set(v___f_1677_, 1, v___x_1676_);
v___x_1678_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_Algebraize_addProperties_spec__0___redArg(v___f_1677_, v___y_1666_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_);
if (lean_obj_tag(v___x_1678_) == 0)
{
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1690_; 
v_a_1679_ = lean_ctor_get(v___x_1678_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v___x_1678_);
if (v_isSharedCheck_1690_ == 0)
{
v___x_1681_ = v___x_1678_;
v_isShared_1682_ = v_isSharedCheck_1690_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1678_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1690_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
uint8_t v___x_1683_; 
v___x_1683_ = lean_unbox(v_a_1679_);
if (v___x_1683_ == 0)
{
size_t v___x_1684_; size_t v___x_1685_; 
lean_del_object(v___x_1681_);
lean_dec(v_a_1679_);
v___x_1684_ = ((size_t)1ULL);
v___x_1685_ = lean_usize_add(v_i_1664_, v___x_1684_);
v_i_1664_ = v___x_1685_;
goto _start;
}
else
{
lean_object* v___x_1688_; 
lean_dec_ref(v_v_1662_);
if (v_isShared_1682_ == 0)
{
v___x_1688_ = v___x_1681_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_a_1679_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
else
{
lean_dec_ref(v_v_1662_);
return v___x_1678_;
}
}
else
{
uint8_t v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; 
lean_dec_ref(v_v_1662_);
v___x_1691_ = 0;
v___x_1692_ = lean_box(v___x_1691_);
v___x_1693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1693_, 0, v___x_1692_);
return v___x_1693_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1___boxed(lean_object* v_v_1694_, lean_object* v_as_1695_, lean_object* v_i_1696_, lean_object* v_stop_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
size_t v_i_boxed_1707_; size_t v_stop_boxed_1708_; lean_object* v_res_1709_; 
v_i_boxed_1707_ = lean_unbox_usize(v_i_1696_);
lean_dec(v_i_1696_);
v_stop_boxed_1708_ = lean_unbox_usize(v_stop_1697_);
lean_dec(v_stop_1697_);
v_res_1709_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1(v_v_1694_, v_as_1695_, v_i_boxed_1707_, v_stop_boxed_1708_, v___y_1698_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
lean_dec(v___y_1699_);
lean_dec_ref(v___y_1698_);
lean_dec_ref(v_as_1695_);
return v_res_1709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3(lean_object* v_t_1710_, uint8_t v___x_1711_, lean_object* v_as_1712_, size_t v_i_1713_, size_t v_stop_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_){
_start:
{
uint8_t v___x_1724_; 
v___x_1724_ = lean_usize_dec_eq(v_i_1713_, v_stop_1714_);
if (v___x_1724_ == 0)
{
lean_object* v___x_1725_; uint8_t v___x_1726_; uint8_t v_a_1728_; lean_object* v___x_1734_; uint8_t v___x_1735_; 
v___x_1725_ = lean_unsigned_to_nat(0u);
v___x_1726_ = 1;
v___x_1734_ = lean_array_get_size(v_t_1710_);
v___x_1735_ = lean_nat_dec_lt(v___x_1725_, v___x_1734_);
if (v___x_1735_ == 0)
{
lean_object* v___x_1736_; lean_object* v___x_1737_; 
v___x_1736_ = lean_box(v___x_1726_);
v___x_1737_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1737_, 0, v___x_1736_);
return v___x_1737_;
}
else
{
if (v___x_1735_ == 0)
{
lean_object* v___x_1738_; lean_object* v___x_1739_; 
v___x_1738_ = lean_box(v___x_1726_);
v___x_1739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1739_, 0, v___x_1738_);
return v___x_1739_;
}
else
{
lean_object* v___x_1740_; size_t v___x_1741_; size_t v___x_1742_; lean_object* v___x_1743_; 
v___x_1740_ = lean_array_uget_borrowed(v_as_1712_, v_i_1713_);
v___x_1741_ = ((size_t)0ULL);
v___x_1742_ = lean_usize_of_nat(v___x_1734_);
lean_inc(v___x_1740_);
v___x_1743_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__1(v___x_1740_, v_t_1710_, v___x_1741_, v___x_1742_, v___y_1715_, v___y_1716_, v___y_1717_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_, v___y_1722_);
if (lean_obj_tag(v___x_1743_) == 0)
{
lean_object* v_a_1744_; lean_object* v___x_1746_; uint8_t v_isShared_1747_; uint8_t v_isSharedCheck_1753_; 
v_a_1744_ = lean_ctor_get(v___x_1743_, 0);
v_isSharedCheck_1753_ = !lean_is_exclusive(v___x_1743_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1746_ = v___x_1743_;
v_isShared_1747_ = v_isSharedCheck_1753_;
goto v_resetjp_1745_;
}
else
{
lean_inc(v_a_1744_);
lean_dec(v___x_1743_);
v___x_1746_ = lean_box(0);
v_isShared_1747_ = v_isSharedCheck_1753_;
goto v_resetjp_1745_;
}
v_resetjp_1745_:
{
uint8_t v___x_1748_; 
v___x_1748_ = lean_unbox(v_a_1744_);
lean_dec(v_a_1744_);
if (v___x_1748_ == 0)
{
lean_object* v___x_1749_; lean_object* v___x_1751_; 
v___x_1749_ = lean_box(v___x_1726_);
if (v_isShared_1747_ == 0)
{
lean_ctor_set(v___x_1746_, 0, v___x_1749_);
v___x_1751_ = v___x_1746_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1752_; 
v_reuseFailAlloc_1752_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1752_, 0, v___x_1749_);
v___x_1751_ = v_reuseFailAlloc_1752_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
return v___x_1751_;
}
}
else
{
lean_del_object(v___x_1746_);
v_a_1728_ = v___x_1711_;
goto v___jp_1727_;
}
}
}
else
{
if (lean_obj_tag(v___x_1743_) == 0)
{
lean_object* v_a_1754_; uint8_t v___x_1755_; 
v_a_1754_ = lean_ctor_get(v___x_1743_, 0);
lean_inc(v_a_1754_);
lean_dec_ref_known(v___x_1743_, 1);
v___x_1755_ = lean_unbox(v_a_1754_);
lean_dec(v_a_1754_);
v_a_1728_ = v___x_1755_;
goto v___jp_1727_;
}
else
{
return v___x_1743_;
}
}
}
}
v___jp_1727_:
{
if (v_a_1728_ == 0)
{
size_t v___x_1729_; size_t v___x_1730_; 
v___x_1729_ = ((size_t)1ULL);
v___x_1730_ = lean_usize_add(v_i_1713_, v___x_1729_);
v_i_1713_ = v___x_1730_;
goto _start;
}
else
{
lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1732_ = lean_box(v___x_1726_);
v___x_1733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1733_, 0, v___x_1732_);
return v___x_1733_;
}
}
}
else
{
uint8_t v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; 
v___x_1756_ = 0;
v___x_1757_ = lean_box(v___x_1756_);
v___x_1758_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1758_, 0, v___x_1757_);
return v___x_1758_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3___boxed(lean_object* v_t_1759_, lean_object* v___x_1760_, lean_object* v_as_1761_, lean_object* v_i_1762_, lean_object* v_stop_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_){
_start:
{
uint8_t v___x_45223__boxed_1773_; size_t v_i_boxed_1774_; size_t v_stop_boxed_1775_; lean_object* v_res_1776_; 
v___x_45223__boxed_1773_ = lean_unbox(v___x_1760_);
v_i_boxed_1774_ = lean_unbox_usize(v_i_1762_);
lean_dec(v_i_1762_);
v_stop_boxed_1775_ = lean_unbox_usize(v_stop_1763_);
lean_dec(v_stop_1763_);
v_res_1776_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3(v_t_1759_, v___x_45223__boxed_1773_, v_as_1761_, v_i_boxed_1774_, v_stop_boxed_1775_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v___y_1769_);
lean_dec_ref(v___y_1768_);
lean_dec(v___y_1767_);
lean_dec_ref(v___y_1766_);
lean_dec(v___y_1765_);
lean_dec_ref(v___y_1764_);
lean_dec_ref(v_as_1761_);
lean_dec_ref(v_t_1759_);
return v_res_1776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(uint8_t v___x_1777_, uint8_t v_____do__lift_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_){
_start:
{
if (v_____do__lift_1778_ == 0)
{
uint8_t v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; 
v___x_1788_ = 1;
v___x_1789_ = lean_box(v___x_1788_);
v___x_1790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1790_, 0, v___x_1789_);
return v___x_1790_;
}
else
{
lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1791_ = lean_box(v___x_1777_);
v___x_1792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1792_, 0, v___x_1791_);
return v___x_1792_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0___boxed(lean_object* v___x_1793_, lean_object* v_____do__lift_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_){
_start:
{
uint8_t v___x_45324__boxed_1804_; uint8_t v_____do__lift_45325__boxed_1805_; lean_object* v_res_1806_; 
v___x_45324__boxed_1804_ = lean_unbox(v___x_1793_);
v_____do__lift_45325__boxed_1805_ = lean_unbox(v_____do__lift_1794_);
v_res_1806_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(v___x_45324__boxed_1804_, v_____do__lift_45325__boxed_1805_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_);
lean_dec(v___y_1802_);
lean_dec_ref(v___y_1801_);
lean_dec(v___y_1800_);
lean_dec_ref(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec_ref(v___y_1797_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
return v_res_1806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1(lean_object* v_snd_1810_, lean_object* v_fst_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_){
_start:
{
lean_object* v___x_1821_; 
v___x_1821_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1813_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
if (lean_obj_tag(v___x_1821_) == 0)
{
lean_object* v_a_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; 
v_a_1822_ = lean_ctor_get(v___x_1821_, 0);
lean_inc(v_a_1822_);
lean_dec_ref_known(v___x_1821_, 1);
v___x_1823_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___closed__1));
v___x_1824_ = l_Lean_Meta_mkFreshBinderNameForTactic___redArg(v___x_1823_, v___y_1816_, v___y_1818_, v___y_1819_);
if (lean_obj_tag(v___x_1824_) == 0)
{
lean_object* v_a_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; 
v_a_1825_ = lean_ctor_get(v___x_1824_, 0);
lean_inc(v_a_1825_);
lean_dec_ref_known(v___x_1824_, 1);
v___x_1826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1826_, 0, v_snd_1810_);
v___x_1827_ = l_Lean_MVarId_note(v_a_1822_, v_a_1825_, v_fst_1811_, v___x_1826_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
if (lean_obj_tag(v___x_1827_) == 0)
{
lean_object* v_a_1828_; lean_object* v_snd_1829_; lean_object* v___x_1831_; uint8_t v_isShared_1832_; uint8_t v_isSharedCheck_1847_; 
v_a_1828_ = lean_ctor_get(v___x_1827_, 0);
lean_inc(v_a_1828_);
lean_dec_ref_known(v___x_1827_, 1);
v_snd_1829_ = lean_ctor_get(v_a_1828_, 1);
v_isSharedCheck_1847_ = !lean_is_exclusive(v_a_1828_);
if (v_isSharedCheck_1847_ == 0)
{
lean_object* v_unused_1848_; 
v_unused_1848_ = lean_ctor_get(v_a_1828_, 0);
lean_dec(v_unused_1848_);
v___x_1831_ = v_a_1828_;
v_isShared_1832_ = v_isSharedCheck_1847_;
goto v_resetjp_1830_;
}
else
{
lean_inc(v_snd_1829_);
lean_dec(v_a_1828_);
v___x_1831_ = lean_box(0);
v_isShared_1832_ = v_isSharedCheck_1847_;
goto v_resetjp_1830_;
}
v_resetjp_1830_:
{
lean_object* v___x_1833_; lean_object* v___x_1835_; 
v___x_1833_ = lean_box(0);
if (v_isShared_1832_ == 0)
{
lean_ctor_set_tag(v___x_1831_, 1);
lean_ctor_set(v___x_1831_, 1, v___x_1833_);
lean_ctor_set(v___x_1831_, 0, v_snd_1829_);
v___x_1835_ = v___x_1831_;
goto v_reusejp_1834_;
}
else
{
lean_object* v_reuseFailAlloc_1846_; 
v_reuseFailAlloc_1846_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1846_, 0, v_snd_1829_);
lean_ctor_set(v_reuseFailAlloc_1846_, 1, v___x_1833_);
v___x_1835_ = v_reuseFailAlloc_1846_;
goto v_reusejp_1834_;
}
v_reusejp_1834_:
{
lean_object* v___x_1836_; 
v___x_1836_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1835_, v___y_1813_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
if (lean_obj_tag(v___x_1836_) == 0)
{
lean_object* v___x_1838_; uint8_t v_isShared_1839_; uint8_t v_isSharedCheck_1844_; 
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1836_);
if (v_isSharedCheck_1844_ == 0)
{
lean_object* v_unused_1845_; 
v_unused_1845_ = lean_ctor_get(v___x_1836_, 0);
lean_dec(v_unused_1845_);
v___x_1838_ = v___x_1836_;
v_isShared_1839_ = v_isSharedCheck_1844_;
goto v_resetjp_1837_;
}
else
{
lean_dec(v___x_1836_);
v___x_1838_ = lean_box(0);
v_isShared_1839_ = v_isSharedCheck_1844_;
goto v_resetjp_1837_;
}
v_resetjp_1837_:
{
lean_object* v___x_1840_; lean_object* v___x_1842_; 
v___x_1840_ = lean_box(0);
if (v_isShared_1839_ == 0)
{
lean_ctor_set(v___x_1838_, 0, v___x_1840_);
v___x_1842_ = v___x_1838_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v___x_1840_);
v___x_1842_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
return v___x_1842_;
}
}
}
else
{
return v___x_1836_;
}
}
}
}
else
{
lean_object* v_a_1849_; lean_object* v___x_1851_; uint8_t v_isShared_1852_; uint8_t v_isSharedCheck_1856_; 
v_a_1849_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1856_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1856_ == 0)
{
v___x_1851_ = v___x_1827_;
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
else
{
lean_inc(v_a_1849_);
lean_dec(v___x_1827_);
v___x_1851_ = lean_box(0);
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
v_resetjp_1850_:
{
lean_object* v___x_1854_; 
if (v_isShared_1852_ == 0)
{
v___x_1854_ = v___x_1851_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v_a_1849_);
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
else
{
lean_object* v_a_1857_; lean_object* v___x_1859_; uint8_t v_isShared_1860_; uint8_t v_isSharedCheck_1864_; 
lean_dec(v_a_1822_);
lean_dec_ref(v_fst_1811_);
lean_dec_ref(v_snd_1810_);
v_a_1857_ = lean_ctor_get(v___x_1824_, 0);
v_isSharedCheck_1864_ = !lean_is_exclusive(v___x_1824_);
if (v_isSharedCheck_1864_ == 0)
{
v___x_1859_ = v___x_1824_;
v_isShared_1860_ = v_isSharedCheck_1864_;
goto v_resetjp_1858_;
}
else
{
lean_inc(v_a_1857_);
lean_dec(v___x_1824_);
v___x_1859_ = lean_box(0);
v_isShared_1860_ = v_isSharedCheck_1864_;
goto v_resetjp_1858_;
}
v_resetjp_1858_:
{
lean_object* v___x_1862_; 
if (v_isShared_1860_ == 0)
{
v___x_1862_ = v___x_1859_;
goto v_reusejp_1861_;
}
else
{
lean_object* v_reuseFailAlloc_1863_; 
v_reuseFailAlloc_1863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1863_, 0, v_a_1857_);
v___x_1862_ = v_reuseFailAlloc_1863_;
goto v_reusejp_1861_;
}
v_reusejp_1861_:
{
return v___x_1862_;
}
}
}
}
else
{
lean_object* v_a_1865_; lean_object* v___x_1867_; uint8_t v_isShared_1868_; uint8_t v_isSharedCheck_1872_; 
lean_dec_ref(v_fst_1811_);
lean_dec_ref(v_snd_1810_);
v_a_1865_ = lean_ctor_get(v___x_1821_, 0);
v_isSharedCheck_1872_ = !lean_is_exclusive(v___x_1821_);
if (v_isSharedCheck_1872_ == 0)
{
v___x_1867_ = v___x_1821_;
v_isShared_1868_ = v_isSharedCheck_1872_;
goto v_resetjp_1866_;
}
else
{
lean_inc(v_a_1865_);
lean_dec(v___x_1821_);
v___x_1867_ = lean_box(0);
v_isShared_1868_ = v_isSharedCheck_1872_;
goto v_resetjp_1866_;
}
v_resetjp_1866_:
{
lean_object* v___x_1870_; 
if (v_isShared_1868_ == 0)
{
v___x_1870_ = v___x_1867_;
goto v_reusejp_1869_;
}
else
{
lean_object* v_reuseFailAlloc_1871_; 
v_reuseFailAlloc_1871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1871_, 0, v_a_1865_);
v___x_1870_ = v_reuseFailAlloc_1871_;
goto v_reusejp_1869_;
}
v_reusejp_1869_:
{
return v___x_1870_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___boxed(lean_object* v_snd_1873_, lean_object* v_fst_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_, lean_object* v___y_1882_, lean_object* v___y_1883_){
_start:
{
lean_object* v_res_1884_; 
v_res_1884_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1(v_snd_1873_, v_fst_1874_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_, v___y_1880_, v___y_1881_, v___y_1882_);
lean_dec(v___y_1882_);
lean_dec_ref(v___y_1881_);
lean_dec(v___y_1880_);
lean_dec_ref(v___y_1879_);
lean_dec(v___y_1878_);
lean_dec_ref(v___y_1877_);
lean_dec(v___y_1876_);
lean_dec_ref(v___y_1875_);
return v_res_1884_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1(void){
_start:
{
lean_object* v___x_1886_; lean_object* v___x_1887_; 
v___x_1886_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__0));
v___x_1887_ = l_Lean_stringToMessageData(v___x_1886_);
return v___x_1887_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3(void){
_start:
{
lean_object* v___x_1889_; lean_object* v___x_1890_; 
v___x_1889_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__2));
v___x_1890_ = l_Lean_stringToMessageData(v___x_1889_);
return v___x_1890_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5(void){
_start:
{
lean_object* v___x_1892_; lean_object* v___x_1893_; 
v___x_1892_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__4));
v___x_1893_ = l_Lean_stringToMessageData(v___x_1892_);
return v___x_1893_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7(void){
_start:
{
lean_object* v___x_1895_; lean_object* v___x_1896_; 
v___x_1895_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__6));
v___x_1896_ = l_Lean_stringToMessageData(v___x_1895_);
return v___x_1896_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9(void){
_start:
{
lean_object* v___x_1898_; lean_object* v___x_1899_; 
v___x_1898_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__8));
v___x_1899_ = l_Lean_stringToMessageData(v___x_1898_);
return v___x_1899_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11(void){
_start:
{
lean_object* v___x_1901_; lean_object* v___x_1902_; 
v___x_1901_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__10));
v___x_1902_ = l_Lean_stringToMessageData(v___x_1901_);
return v___x_1902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21(lean_object* v_t_1903_, lean_object* v_as_1904_, size_t v_i_1905_, size_t v_stop_1906_, lean_object* v_b_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_){
_start:
{
lean_object* v_a_1918_; lean_object* v___y_1927_; uint8_t v___x_1929_; 
v___x_1929_ = lean_usize_dec_eq(v_i_1905_, v_stop_1906_);
if (v___x_1929_ == 0)
{
lean_object* v___x_1930_; 
v___x_1930_ = lean_array_uget_borrowed(v_as_1904_, v_i_1905_);
if (lean_obj_tag(v___x_1930_) == 0)
{
lean_object* v___x_1931_; 
v___x_1931_ = lean_box(0);
v_a_1918_ = v___x_1931_;
goto v___jp_1917_;
}
else
{
lean_object* v_val_1932_; uint8_t v___x_1933_; lean_object* v___y_1935_; lean_object* v___y_1936_; uint8_t v_a_1937_; lean_object* v___y_1940_; lean_object* v___y_1941_; lean_object* v___y_1942_; 
v_val_1932_ = lean_ctor_get(v___x_1930_, 0);
v___x_1933_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1932_);
if (v___x_1933_ == 0)
{
lean_object* v___x_1953_; lean_object* v___x_1954_; 
v___x_1953_ = l_Lean_LocalDecl_type(v_val_1932_);
lean_inc_ref(v___x_1953_);
v___x_1954_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(v___x_1953_, v___y_1913_);
if (lean_obj_tag(v___x_1954_) == 0)
{
lean_object* v_a_1955_; lean_object* v___x_1956_; lean_object* v_fst_1957_; lean_object* v_snd_1958_; lean_object* v___x_1960_; uint8_t v_isShared_1961_; uint8_t v_isSharedCheck_2163_; 
v_a_1955_ = lean_ctor_get(v___x_1954_, 0);
lean_inc(v_a_1955_);
lean_dec_ref_known(v___x_1954_, 1);
v___x_1956_ = l_Lean_Expr_getAppFnArgs(v_a_1955_);
v_fst_1957_ = lean_ctor_get(v___x_1956_, 0);
v_snd_1958_ = lean_ctor_get(v___x_1956_, 1);
v_isSharedCheck_2163_ = !lean_is_exclusive(v___x_1956_);
if (v_isSharedCheck_2163_ == 0)
{
v___x_1960_ = v___x_1956_;
v_isShared_1961_ = v_isSharedCheck_2163_;
goto v_resetjp_1959_;
}
else
{
lean_inc(v_snd_1958_);
lean_inc(v_fst_1957_);
lean_dec(v___x_1956_);
v___x_1960_ = lean_box(0);
v_isShared_1961_ = v_isSharedCheck_2163_;
goto v_resetjp_1959_;
}
v_resetjp_1959_:
{
lean_object* v___x_1962_; lean_object* v_env_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; 
v___x_1962_ = lean_st_ref_get(v___y_1915_);
v_env_1963_ = lean_ctor_get(v___x_1962_, 0);
lean_inc_ref(v_env_1963_);
lean_dec(v___x_1962_);
v___x_1964_ = lean_box(0);
v___x_1965_ = lp_mathlib_Lean_Attr_algebraizeAttr;
lean_inc(v_fst_1957_);
v___x_1966_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_1964_, v___x_1965_, v_env_1963_, v_fst_1957_);
if (lean_obj_tag(v___x_1966_) == 0)
{
lean_object* v___x_1967_; 
lean_del_object(v___x_1960_);
lean_dec(v_snd_1958_);
lean_dec(v_fst_1957_);
lean_dec_ref(v___x_1953_);
v___x_1967_ = lean_box(0);
v_a_1918_ = v___x_1967_;
goto v___jp_1917_;
}
else
{
lean_object* v_val_1968_; lean_object* v___x_1969_; 
v_val_1968_ = lean_ctor_get(v___x_1966_, 0);
lean_inc(v_val_1968_);
lean_dec_ref_known(v___x_1966_, 1);
v___x_1969_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1909_, v___y_1911_, v___y_1913_, v___y_1915_);
if (lean_obj_tag(v___x_1969_) == 0)
{
lean_object* v_a_1970_; lean_object* v___x_1971_; 
v_a_1970_ = lean_ctor_get(v___x_1969_, 0);
lean_inc(v_a_1970_);
lean_dec_ref_known(v___x_1969_, 1);
lean_inc(v_val_1968_);
v___x_1971_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5(v_val_1968_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1971_) == 0)
{
lean_object* v_a_1972_; lean_object* v___x_1973_; 
lean_dec(v_a_1970_);
lean_del_object(v___x_1960_);
lean_dec(v_fst_1957_);
v_a_1972_ = lean_ctor_get(v___x_1971_, 0);
lean_inc(v_a_1972_);
lean_dec_ref_known(v___x_1971_, 1);
v___x_1973_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_val_1968_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1973_) == 0)
{
lean_object* v_a_1974_; lean_object* v___x_1975_; 
v_a_1974_ = lean_ctor_get(v___x_1973_, 0);
lean_inc_n(v_a_1974_, 2);
lean_dec_ref_known(v___x_1973_, 1);
lean_inc(v___y_1915_);
lean_inc_ref(v___y_1914_);
lean_inc(v___y_1913_);
lean_inc_ref(v___y_1912_);
v___x_1975_ = lean_infer_type(v_a_1974_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1975_) == 0)
{
lean_object* v_a_1976_; uint8_t v___x_1977_; lean_object* v___x_1978_; 
v_a_1976_ = lean_ctor_get(v___x_1975_, 0);
lean_inc(v_a_1976_);
lean_dec_ref_known(v___x_1975_, 1);
v___x_1977_ = 0;
v___x_1978_ = l_Lean_Meta_forallMetaTelescope(v_a_1976_, v___x_1977_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1978_) == 0)
{
lean_object* v_a_1979_; lean_object* v_fst_1980_; lean_object* v_fst_1982_; lean_object* v_snd_1983_; lean_object* v___x_2023_; uint8_t v___x_2024_; 
v_a_1979_ = lean_ctor_get(v___x_1978_, 0);
lean_inc(v_a_1979_);
lean_dec_ref_known(v___x_1978_, 1);
v_fst_1980_ = lean_ctor_get(v_a_1979_, 0);
lean_inc(v_fst_1980_);
lean_dec(v_a_1979_);
v___x_2023_ = l_Lean_mkAppN(v_a_1974_, v_fst_1980_);
v___x_2024_ = l_Lean_ConstantInfo_isInductive(v_a_1972_);
lean_dec(v_a_1972_);
if (v___x_2024_ == 0)
{
lean_object* v___x_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; 
lean_dec(v_snd_1958_);
lean_dec_ref(v___x_1953_);
v___x_2025_ = l_Lean_instInhabitedExpr;
v___x_2026_ = lean_array_get_size(v_fst_1980_);
v___x_2027_ = lean_unsigned_to_nat(1u);
v___x_2028_ = lean_nat_sub(v___x_2026_, v___x_2027_);
v___x_2029_ = lean_array_get(v___x_2025_, v_fst_1980_, v___x_2028_);
lean_dec(v___x_2028_);
lean_dec(v_fst_1980_);
v___x_2030_ = l_Lean_Expr_mvarId_x21(v___x_2029_);
lean_dec(v___x_2029_);
lean_inc(v_val_1932_);
v___x_2031_ = l_Lean_LocalDecl_toExpr(v_val_1932_);
v___x_2032_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2030_, v___x_2031_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2032_) == 0)
{
lean_object* v___x_2033_; 
lean_dec_ref_known(v___x_2032_, 1);
v___x_2033_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v___x_2023_, v___y_1913_);
if (lean_obj_tag(v___x_2033_) == 0)
{
lean_object* v_a_2034_; lean_object* v___x_2035_; 
v_a_2034_ = lean_ctor_get(v___x_2033_, 0);
lean_inc_n(v_a_2034_, 2);
lean_dec_ref_known(v___x_2033_, 1);
lean_inc(v___y_1915_);
lean_inc_ref(v___y_1914_);
lean_inc(v___y_1913_);
lean_inc_ref(v___y_1912_);
v___x_2035_ = lean_infer_type(v_a_2034_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2035_) == 0)
{
lean_object* v_a_2036_; 
v_a_2036_ = lean_ctor_get(v___x_2035_, 0);
lean_inc(v_a_2036_);
lean_dec_ref_known(v___x_2035_, 1);
v_fst_1982_ = v_a_2034_;
v_snd_1983_ = v_a_2036_;
goto v___jp_1981_;
}
else
{
lean_object* v_a_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2044_; 
lean_dec(v_a_2034_);
v_a_2037_ = lean_ctor_get(v___x_2035_, 0);
v_isSharedCheck_2044_ = !lean_is_exclusive(v___x_2035_);
if (v_isSharedCheck_2044_ == 0)
{
v___x_2039_ = v___x_2035_;
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_a_2037_);
lean_dec(v___x_2035_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2042_; 
if (v_isShared_2040_ == 0)
{
v___x_2042_ = v___x_2039_;
goto v_reusejp_2041_;
}
else
{
lean_object* v_reuseFailAlloc_2043_; 
v_reuseFailAlloc_2043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2043_, 0, v_a_2037_);
v___x_2042_ = v_reuseFailAlloc_2043_;
goto v_reusejp_2041_;
}
v_reusejp_2041_:
{
return v___x_2042_;
}
}
}
}
else
{
lean_object* v_a_2045_; lean_object* v___x_2047_; uint8_t v_isShared_2048_; uint8_t v_isSharedCheck_2052_; 
v_a_2045_ = lean_ctor_get(v___x_2033_, 0);
v_isSharedCheck_2052_ = !lean_is_exclusive(v___x_2033_);
if (v_isSharedCheck_2052_ == 0)
{
v___x_2047_ = v___x_2033_;
v_isShared_2048_ = v_isSharedCheck_2052_;
goto v_resetjp_2046_;
}
else
{
lean_inc(v_a_2045_);
lean_dec(v___x_2033_);
v___x_2047_ = lean_box(0);
v_isShared_2048_ = v_isSharedCheck_2052_;
goto v_resetjp_2046_;
}
v_resetjp_2046_:
{
lean_object* v___x_2050_; 
if (v_isShared_2048_ == 0)
{
v___x_2050_ = v___x_2047_;
goto v_reusejp_2049_;
}
else
{
lean_object* v_reuseFailAlloc_2051_; 
v_reuseFailAlloc_2051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2051_, 0, v_a_2045_);
v___x_2050_ = v_reuseFailAlloc_2051_;
goto v_reusejp_2049_;
}
v_reusejp_2049_:
{
return v___x_2050_;
}
}
}
}
else
{
lean_object* v_a_2053_; uint8_t v___y_2055_; uint8_t v___x_2056_; 
lean_dec_ref(v___x_2023_);
v_a_2053_ = lean_ctor_get(v___x_2032_, 0);
lean_inc(v_a_2053_);
v___x_2056_ = l_Lean_Exception_isInterrupt(v_a_2053_);
if (v___x_2056_ == 0)
{
uint8_t v___x_2057_; 
v___x_2057_ = l_Lean_Exception_isRuntime(v_a_2053_);
v___y_2055_ = v___x_2057_;
goto v___jp_2054_;
}
else
{
lean_dec(v_a_2053_);
v___y_2055_ = v___x_2056_;
goto v___jp_2054_;
}
v___jp_2054_:
{
if (v___y_2055_ == 0)
{
lean_dec_ref_known(v___x_2032_, 1);
goto v___jp_1924_;
}
else
{
v___y_1927_ = v___x_2032_;
goto v___jp_1926_;
}
}
}
}
else
{
lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; 
v___x_2058_ = l_Lean_instInhabitedExpr;
v___x_2059_ = lean_unsigned_to_nat(0u);
v___x_2060_ = lean_array_get_borrowed(v___x_2058_, v_fst_1980_, v___x_2059_);
v___x_2061_ = l_Lean_Expr_mvarId_x21(v___x_2060_);
v___x_2062_ = lean_array_get_borrowed(v___x_2058_, v_snd_1958_, v___x_2059_);
lean_inc(v___x_2062_);
v___x_2063_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2061_, v___x_2062_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2063_) == 0)
{
lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; 
lean_dec_ref_known(v___x_2063_, 1);
v___x_2064_ = lean_unsigned_to_nat(1u);
v___x_2065_ = lean_array_get(v___x_2058_, v_fst_1980_, v___x_2064_);
lean_dec(v_fst_1980_);
v___x_2066_ = l_Lean_Expr_mvarId_x21(v___x_2065_);
lean_dec(v___x_2065_);
v___x_2067_ = lean_array_get(v___x_2058_, v_snd_1958_, v___x_2064_);
lean_dec(v_snd_1958_);
v___x_2068_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2066_, v___x_2067_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2068_) == 0)
{
lean_object* v___x_2069_; 
lean_dec_ref_known(v___x_2068_, 1);
v___x_2069_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v___x_2023_, v___y_1913_);
if (lean_obj_tag(v___x_2069_) == 0)
{
lean_object* v_a_2070_; lean_object* v___x_2071_; 
v_a_2070_ = lean_ctor_get(v___x_2069_, 0);
lean_inc_n(v_a_2070_, 2);
lean_dec_ref_known(v___x_2069_, 1);
v___x_2071_ = l_Lean_Meta_isExprDefEqGuarded(v___x_1953_, v_a_2070_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2071_) == 0)
{
lean_object* v_a_2072_; uint8_t v___x_2073_; 
v_a_2072_ = lean_ctor_get(v___x_2071_, 0);
lean_inc(v_a_2072_);
lean_dec_ref_known(v___x_2071_, 1);
v___x_2073_ = lean_unbox(v_a_2072_);
lean_dec(v_a_2072_);
if (v___x_2073_ == 0)
{
lean_dec(v_a_2070_);
goto v___jp_1924_;
}
else
{
lean_object* v___x_2074_; 
lean_inc(v_val_1932_);
v___x_2074_ = l_Lean_LocalDecl_toExpr(v_val_1932_);
v_fst_1982_ = v___x_2074_;
v_snd_1983_ = v_a_2070_;
goto v___jp_1981_;
}
}
else
{
lean_object* v_a_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2082_; 
lean_dec(v_a_2070_);
v_a_2075_ = lean_ctor_get(v___x_2071_, 0);
v_isSharedCheck_2082_ = !lean_is_exclusive(v___x_2071_);
if (v_isSharedCheck_2082_ == 0)
{
v___x_2077_ = v___x_2071_;
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_a_2075_);
lean_dec(v___x_2071_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2080_; 
if (v_isShared_2078_ == 0)
{
v___x_2080_ = v___x_2077_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2081_; 
v_reuseFailAlloc_2081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2081_, 0, v_a_2075_);
v___x_2080_ = v_reuseFailAlloc_2081_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
return v___x_2080_;
}
}
}
}
else
{
lean_object* v_a_2083_; lean_object* v___x_2085_; uint8_t v_isShared_2086_; uint8_t v_isSharedCheck_2090_; 
lean_dec_ref(v___x_1953_);
v_a_2083_ = lean_ctor_get(v___x_2069_, 0);
v_isSharedCheck_2090_ = !lean_is_exclusive(v___x_2069_);
if (v_isSharedCheck_2090_ == 0)
{
v___x_2085_ = v___x_2069_;
v_isShared_2086_ = v_isSharedCheck_2090_;
goto v_resetjp_2084_;
}
else
{
lean_inc(v_a_2083_);
lean_dec(v___x_2069_);
v___x_2085_ = lean_box(0);
v_isShared_2086_ = v_isSharedCheck_2090_;
goto v_resetjp_2084_;
}
v_resetjp_2084_:
{
lean_object* v___x_2088_; 
if (v_isShared_2086_ == 0)
{
v___x_2088_ = v___x_2085_;
goto v_reusejp_2087_;
}
else
{
lean_object* v_reuseFailAlloc_2089_; 
v_reuseFailAlloc_2089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2089_, 0, v_a_2083_);
v___x_2088_ = v_reuseFailAlloc_2089_;
goto v_reusejp_2087_;
}
v_reusejp_2087_:
{
return v___x_2088_;
}
}
}
}
else
{
lean_dec_ref(v___x_2023_);
lean_dec_ref(v___x_1953_);
v___y_1927_ = v___x_2068_;
goto v___jp_1926_;
}
}
else
{
lean_dec_ref(v___x_2023_);
lean_dec(v_fst_1980_);
lean_dec(v_snd_1958_);
lean_dec_ref(v___x_1953_);
v___y_1927_ = v___x_2063_;
goto v___jp_1926_;
}
}
v___jp_1981_:
{
lean_object* v_dummy_1984_; lean_object* v_nargs_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; 
v_dummy_1984_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0);
v_nargs_1985_ = l_Lean_Expr_getAppNumArgs(v_snd_1983_);
lean_inc(v_nargs_1985_);
v___x_1986_ = lean_mk_array(v_nargs_1985_, v_dummy_1984_);
v___x_1987_ = lean_unsigned_to_nat(1u);
v___x_1988_ = lean_nat_sub(v_nargs_1985_, v___x_1987_);
lean_dec(v_nargs_1985_);
lean_inc_ref(v_snd_1983_);
v___x_1989_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_snd_1983_, v___x_1986_, v___x_1988_);
v___x_1990_ = lean_unsigned_to_nat(0u);
v___x_1991_ = lean_array_get_size(v___x_1989_);
v___x_1992_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2(v___x_1989_, v___x_1990_, v___x_1991_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec_ref(v___x_1989_);
if (lean_obj_tag(v___x_1992_) == 0)
{
lean_object* v_a_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; 
v_a_1993_ = lean_ctor_get(v___x_1992_, 0);
lean_inc(v_a_1993_);
lean_dec_ref_known(v___x_1992_, 1);
v___x_1994_ = lean_box(0);
lean_inc_ref(v_snd_1983_);
v___x_1995_ = l_Lean_Meta_synthInstance_x3f(v_snd_1983_, v___x_1994_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1995_) == 0)
{
lean_object* v_a_1996_; lean_object* v___f_1997_; lean_object* v___x_1998_; uint8_t v___x_1999_; 
v_a_1996_ = lean_ctor_get(v___x_1995_, 0);
lean_inc(v_a_1996_);
lean_dec_ref_known(v___x_1995_, 1);
v___f_1997_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___boxed), 11, 2);
lean_closure_set(v___f_1997_, 0, v_snd_1983_);
lean_closure_set(v___f_1997_, 1, v_fst_1982_);
v___x_1998_ = lean_array_get_size(v_a_1993_);
v___x_1999_ = lean_nat_dec_lt(v___x_1990_, v___x_1998_);
if (v___x_1999_ == 0)
{
lean_object* v___x_2000_; 
lean_dec(v_a_1993_);
v___x_2000_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(v___x_1933_, v___x_1933_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
v___y_1940_ = v_a_1996_;
v___y_1941_ = v___f_1997_;
v___y_1942_ = v___x_2000_;
goto v___jp_1939_;
}
else
{
if (v___x_1999_ == 0)
{
lean_dec(v_a_1993_);
v___y_1935_ = v_a_1996_;
v___y_1936_ = v___f_1997_;
v_a_1937_ = v___x_1999_;
goto v___jp_1934_;
}
else
{
size_t v___x_2001_; size_t v___x_2002_; lean_object* v___x_2003_; 
v___x_2001_ = ((size_t)0ULL);
v___x_2002_ = lean_usize_of_nat(v___x_1998_);
v___x_2003_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3(v_t_1903_, v___x_1933_, v_a_1993_, v___x_2001_, v___x_2002_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v_a_1993_);
if (lean_obj_tag(v___x_2003_) == 0)
{
lean_object* v_a_2004_; uint8_t v___x_2005_; lean_object* v___x_2006_; 
v_a_2004_ = lean_ctor_get(v___x_2003_, 0);
lean_inc(v_a_2004_);
lean_dec_ref_known(v___x_2003_, 1);
v___x_2005_ = lean_unbox(v_a_2004_);
lean_dec(v_a_2004_);
v___x_2006_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(v___x_1933_, v___x_2005_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
v___y_1940_ = v_a_1996_;
v___y_1941_ = v___f_1997_;
v___y_1942_ = v___x_2006_;
goto v___jp_1939_;
}
else
{
v___y_1940_ = v_a_1996_;
v___y_1941_ = v___f_1997_;
v___y_1942_ = v___x_2003_;
goto v___jp_1939_;
}
}
}
}
else
{
lean_object* v_a_2007_; lean_object* v___x_2009_; uint8_t v_isShared_2010_; uint8_t v_isSharedCheck_2014_; 
lean_dec(v_a_1993_);
lean_dec_ref(v_snd_1983_);
lean_dec_ref(v_fst_1982_);
v_a_2007_ = lean_ctor_get(v___x_1995_, 0);
v_isSharedCheck_2014_ = !lean_is_exclusive(v___x_1995_);
if (v_isSharedCheck_2014_ == 0)
{
v___x_2009_ = v___x_1995_;
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
else
{
lean_inc(v_a_2007_);
lean_dec(v___x_1995_);
v___x_2009_ = lean_box(0);
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
v_resetjp_2008_:
{
lean_object* v___x_2012_; 
if (v_isShared_2010_ == 0)
{
v___x_2012_ = v___x_2009_;
goto v_reusejp_2011_;
}
else
{
lean_object* v_reuseFailAlloc_2013_; 
v_reuseFailAlloc_2013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2013_, 0, v_a_2007_);
v___x_2012_ = v_reuseFailAlloc_2013_;
goto v_reusejp_2011_;
}
v_reusejp_2011_:
{
return v___x_2012_;
}
}
}
}
else
{
lean_object* v_a_2015_; lean_object* v___x_2017_; uint8_t v_isShared_2018_; uint8_t v_isSharedCheck_2022_; 
lean_dec_ref(v_snd_1983_);
lean_dec_ref(v_fst_1982_);
v_a_2015_ = lean_ctor_get(v___x_1992_, 0);
v_isSharedCheck_2022_ = !lean_is_exclusive(v___x_1992_);
if (v_isSharedCheck_2022_ == 0)
{
v___x_2017_ = v___x_1992_;
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
else
{
lean_inc(v_a_2015_);
lean_dec(v___x_1992_);
v___x_2017_ = lean_box(0);
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
v_resetjp_2016_:
{
lean_object* v___x_2020_; 
if (v_isShared_2018_ == 0)
{
v___x_2020_ = v___x_2017_;
goto v_reusejp_2019_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v_a_2015_);
v___x_2020_ = v_reuseFailAlloc_2021_;
goto v_reusejp_2019_;
}
v_reusejp_2019_:
{
return v___x_2020_;
}
}
}
}
}
else
{
lean_object* v_a_2091_; lean_object* v___x_2093_; uint8_t v_isShared_2094_; uint8_t v_isSharedCheck_2098_; 
lean_dec(v_a_1974_);
lean_dec(v_a_1972_);
lean_dec(v_snd_1958_);
lean_dec_ref(v___x_1953_);
v_a_2091_ = lean_ctor_get(v___x_1978_, 0);
v_isSharedCheck_2098_ = !lean_is_exclusive(v___x_1978_);
if (v_isSharedCheck_2098_ == 0)
{
v___x_2093_ = v___x_1978_;
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
else
{
lean_inc(v_a_2091_);
lean_dec(v___x_1978_);
v___x_2093_ = lean_box(0);
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
v_resetjp_2092_:
{
lean_object* v___x_2096_; 
if (v_isShared_2094_ == 0)
{
v___x_2096_ = v___x_2093_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2097_; 
v_reuseFailAlloc_2097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2097_, 0, v_a_2091_);
v___x_2096_ = v_reuseFailAlloc_2097_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
return v___x_2096_;
}
}
}
}
else
{
lean_object* v_a_2099_; lean_object* v___x_2101_; uint8_t v_isShared_2102_; uint8_t v_isSharedCheck_2106_; 
lean_dec(v_a_1974_);
lean_dec(v_a_1972_);
lean_dec(v_snd_1958_);
lean_dec_ref(v___x_1953_);
v_a_2099_ = lean_ctor_get(v___x_1975_, 0);
v_isSharedCheck_2106_ = !lean_is_exclusive(v___x_1975_);
if (v_isSharedCheck_2106_ == 0)
{
v___x_2101_ = v___x_1975_;
v_isShared_2102_ = v_isSharedCheck_2106_;
goto v_resetjp_2100_;
}
else
{
lean_inc(v_a_2099_);
lean_dec(v___x_1975_);
v___x_2101_ = lean_box(0);
v_isShared_2102_ = v_isSharedCheck_2106_;
goto v_resetjp_2100_;
}
v_resetjp_2100_:
{
lean_object* v___x_2104_; 
if (v_isShared_2102_ == 0)
{
v___x_2104_ = v___x_2101_;
goto v_reusejp_2103_;
}
else
{
lean_object* v_reuseFailAlloc_2105_; 
v_reuseFailAlloc_2105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2105_, 0, v_a_2099_);
v___x_2104_ = v_reuseFailAlloc_2105_;
goto v_reusejp_2103_;
}
v_reusejp_2103_:
{
return v___x_2104_;
}
}
}
}
else
{
lean_object* v_a_2107_; lean_object* v___x_2109_; uint8_t v_isShared_2110_; uint8_t v_isSharedCheck_2114_; 
lean_dec(v_a_1972_);
lean_dec(v_snd_1958_);
lean_dec_ref(v___x_1953_);
v_a_2107_ = lean_ctor_get(v___x_1973_, 0);
v_isSharedCheck_2114_ = !lean_is_exclusive(v___x_1973_);
if (v_isSharedCheck_2114_ == 0)
{
v___x_2109_ = v___x_1973_;
v_isShared_2110_ = v_isSharedCheck_2114_;
goto v_resetjp_2108_;
}
else
{
lean_inc(v_a_2107_);
lean_dec(v___x_1973_);
v___x_2109_ = lean_box(0);
v_isShared_2110_ = v_isSharedCheck_2114_;
goto v_resetjp_2108_;
}
v_resetjp_2108_:
{
lean_object* v___x_2112_; 
if (v_isShared_2110_ == 0)
{
v___x_2112_ = v___x_2109_;
goto v_reusejp_2111_;
}
else
{
lean_object* v_reuseFailAlloc_2113_; 
v_reuseFailAlloc_2113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2113_, 0, v_a_2107_);
v___x_2112_ = v_reuseFailAlloc_2113_;
goto v_reusejp_2111_;
}
v_reusejp_2111_:
{
return v___x_2112_;
}
}
}
}
else
{
lean_object* v_a_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2154_; 
lean_dec(v_snd_1958_);
v_a_2115_ = lean_ctor_get(v___x_1971_, 0);
v_isSharedCheck_2154_ = !lean_is_exclusive(v___x_1971_);
if (v_isSharedCheck_2154_ == 0)
{
v___x_2117_ = v___x_1971_;
v_isShared_2118_ = v_isSharedCheck_2154_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_a_2115_);
lean_dec(v___x_1971_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2154_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
uint8_t v___y_2120_; uint8_t v___x_2152_; 
v___x_2152_ = l_Lean_Exception_isInterrupt(v_a_2115_);
if (v___x_2152_ == 0)
{
uint8_t v___x_2153_; 
lean_inc(v_a_2115_);
v___x_2153_ = l_Lean_Exception_isRuntime(v_a_2115_);
v___y_2120_ = v___x_2153_;
goto v___jp_2119_;
}
else
{
v___y_2120_ = v___x_2152_;
goto v___jp_2119_;
}
v___jp_2119_:
{
if (v___y_2120_ == 0)
{
lean_object* v___x_2121_; 
lean_del_object(v___x_2117_);
lean_dec(v_a_2115_);
v___x_2121_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1970_, v___y_2120_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2121_) == 0)
{
lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2126_; 
lean_dec_ref_known(v___x_2121_, 1);
v___x_2122_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1);
lean_inc(v_val_1932_);
v___x_2123_ = l_Lean_LocalDecl_toExpr(v_val_1932_);
v___x_2124_ = l_Lean_MessageData_ofExpr(v___x_2123_);
if (v_isShared_1961_ == 0)
{
lean_ctor_set_tag(v___x_1960_, 7);
lean_ctor_set(v___x_1960_, 1, v___x_2124_);
lean_ctor_set(v___x_1960_, 0, v___x_2122_);
v___x_2126_ = v___x_1960_;
goto v_reusejp_2125_;
}
else
{
lean_object* v_reuseFailAlloc_2148_; 
v_reuseFailAlloc_2148_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2148_, 0, v___x_2122_);
lean_ctor_set(v_reuseFailAlloc_2148_, 1, v___x_2124_);
v___x_2126_ = v_reuseFailAlloc_2148_;
goto v_reusejp_2125_;
}
v_reusejp_2125_:
{
lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; 
v___x_2127_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3);
v___x_2128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2128_, 0, v___x_2126_);
lean_ctor_set(v___x_2128_, 1, v___x_2127_);
v___x_2129_ = l_Lean_MessageData_ofExpr(v___x_1953_);
v___x_2130_ = l_Lean_indentD(v___x_2129_);
v___x_2131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2131_, 0, v___x_2128_);
lean_ctor_set(v___x_2131_, 1, v___x_2130_);
v___x_2132_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5);
v___x_2133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2133_, 0, v___x_2131_);
lean_ctor_set(v___x_2133_, 1, v___x_2132_);
v___x_2134_ = l_Lean_MessageData_ofConstName(v_fst_1957_, v___y_2120_);
v___x_2135_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2135_, 0, v___x_2133_);
lean_ctor_set(v___x_2135_, 1, v___x_2134_);
v___x_2136_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7);
v___x_2137_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2135_);
lean_ctor_set(v___x_2137_, 1, v___x_2136_);
v___x_2138_ = l_Lean_MessageData_ofName(v_val_1968_);
lean_inc_ref(v___x_2138_);
v___x_2139_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2139_, 0, v___x_2137_);
lean_ctor_set(v___x_2139_, 1, v___x_2138_);
v___x_2140_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9);
v___x_2141_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2141_, 0, v___x_2139_);
lean_ctor_set(v___x_2141_, 1, v___x_2140_);
v___x_2142_ = l_Lean_indentD(v___x_2138_);
v___x_2143_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2143_, 0, v___x_2141_);
lean_ctor_set(v___x_2143_, 1, v___x_2142_);
v___x_2144_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11);
v___x_2145_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2145_, 0, v___x_2143_);
lean_ctor_set(v___x_2145_, 1, v___x_2144_);
v___x_2146_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7(v___x_2145_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_2146_) == 0)
{
lean_object* v___x_2147_; 
lean_dec_ref_known(v___x_2146_, 1);
v___x_2147_ = lean_box(0);
v_a_1918_ = v___x_2147_;
goto v___jp_1917_;
}
else
{
v___y_1927_ = v___x_2146_;
goto v___jp_1926_;
}
}
}
else
{
lean_dec(v_val_1968_);
lean_del_object(v___x_1960_);
lean_dec(v_fst_1957_);
lean_dec_ref(v___x_1953_);
v___y_1927_ = v___x_2121_;
goto v___jp_1926_;
}
}
else
{
lean_object* v___x_2150_; 
lean_dec(v_a_1970_);
lean_dec(v_val_1968_);
lean_del_object(v___x_1960_);
lean_dec(v_fst_1957_);
lean_dec_ref(v___x_1953_);
if (v_isShared_2118_ == 0)
{
v___x_2150_ = v___x_2117_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2151_; 
v_reuseFailAlloc_2151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2151_, 0, v_a_2115_);
v___x_2150_ = v_reuseFailAlloc_2151_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
return v___x_2150_;
}
}
}
}
}
}
else
{
lean_object* v_a_2155_; lean_object* v___x_2157_; uint8_t v_isShared_2158_; uint8_t v_isSharedCheck_2162_; 
lean_dec(v_val_1968_);
lean_del_object(v___x_1960_);
lean_dec(v_snd_1958_);
lean_dec(v_fst_1957_);
lean_dec_ref(v___x_1953_);
v_a_2155_ = lean_ctor_get(v___x_1969_, 0);
v_isSharedCheck_2162_ = !lean_is_exclusive(v___x_1969_);
if (v_isSharedCheck_2162_ == 0)
{
v___x_2157_ = v___x_1969_;
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
else
{
lean_inc(v_a_2155_);
lean_dec(v___x_1969_);
v___x_2157_ = lean_box(0);
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
v_resetjp_2156_:
{
lean_object* v___x_2160_; 
if (v_isShared_2158_ == 0)
{
v___x_2160_ = v___x_2157_;
goto v_reusejp_2159_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v_a_2155_);
v___x_2160_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2159_;
}
v_reusejp_2159_:
{
return v___x_2160_;
}
}
}
}
}
}
else
{
lean_object* v_a_2164_; lean_object* v___x_2166_; uint8_t v_isShared_2167_; uint8_t v_isSharedCheck_2171_; 
lean_dec_ref(v___x_1953_);
v_a_2164_ = lean_ctor_get(v___x_1954_, 0);
v_isSharedCheck_2171_ = !lean_is_exclusive(v___x_1954_);
if (v_isSharedCheck_2171_ == 0)
{
v___x_2166_ = v___x_1954_;
v_isShared_2167_ = v_isSharedCheck_2171_;
goto v_resetjp_2165_;
}
else
{
lean_inc(v_a_2164_);
lean_dec(v___x_1954_);
v___x_2166_ = lean_box(0);
v_isShared_2167_ = v_isSharedCheck_2171_;
goto v_resetjp_2165_;
}
v_resetjp_2165_:
{
lean_object* v___x_2169_; 
if (v_isShared_2167_ == 0)
{
v___x_2169_ = v___x_2166_;
goto v_reusejp_2168_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v_a_2164_);
v___x_2169_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2168_;
}
v_reusejp_2168_:
{
return v___x_2169_;
}
}
}
}
else
{
lean_object* v___x_2172_; 
v___x_2172_ = lean_box(0);
v_a_1918_ = v___x_2172_;
goto v___jp_1917_;
}
v___jp_1934_:
{
if (lean_obj_tag(v___y_1935_) == 0)
{
if (v___x_1933_ == 0)
{
if (v_a_1937_ == 0)
{
lean_dec_ref(v___y_1936_);
goto v___jp_1922_;
}
else
{
lean_object* v___x_1938_; 
v___x_1938_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_1936_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
v___y_1927_ = v___x_1938_;
goto v___jp_1926_;
}
}
else
{
lean_dec_ref(v___y_1936_);
goto v___jp_1922_;
}
}
else
{
lean_dec_ref_known(v___y_1935_, 1);
lean_dec_ref(v___y_1936_);
goto v___jp_1922_;
}
}
v___jp_1939_:
{
if (lean_obj_tag(v___y_1942_) == 0)
{
lean_object* v_a_1943_; uint8_t v___x_1944_; 
v_a_1943_ = lean_ctor_get(v___y_1942_, 0);
lean_inc(v_a_1943_);
lean_dec_ref_known(v___y_1942_, 1);
v___x_1944_ = lean_unbox(v_a_1943_);
lean_dec(v_a_1943_);
v___y_1935_ = v___y_1940_;
v___y_1936_ = v___y_1941_;
v_a_1937_ = v___x_1944_;
goto v___jp_1934_;
}
else
{
lean_object* v_a_1945_; lean_object* v___x_1947_; uint8_t v_isShared_1948_; uint8_t v_isSharedCheck_1952_; 
lean_dec_ref(v___y_1941_);
lean_dec(v___y_1940_);
v_a_1945_ = lean_ctor_get(v___y_1942_, 0);
v_isSharedCheck_1952_ = !lean_is_exclusive(v___y_1942_);
if (v_isSharedCheck_1952_ == 0)
{
v___x_1947_ = v___y_1942_;
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
else
{
lean_inc(v_a_1945_);
lean_dec(v___y_1942_);
v___x_1947_ = lean_box(0);
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
v_resetjp_1946_:
{
lean_object* v___x_1950_; 
if (v_isShared_1948_ == 0)
{
v___x_1950_ = v___x_1947_;
goto v_reusejp_1949_;
}
else
{
lean_object* v_reuseFailAlloc_1951_; 
v_reuseFailAlloc_1951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1951_, 0, v_a_1945_);
v___x_1950_ = v_reuseFailAlloc_1951_;
goto v_reusejp_1949_;
}
v_reusejp_1949_:
{
return v___x_1950_;
}
}
}
}
}
}
else
{
lean_object* v___x_2173_; 
v___x_2173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2173_, 0, v_b_1907_);
return v___x_2173_;
}
v___jp_1917_:
{
size_t v___x_1919_; size_t v___x_1920_; 
v___x_1919_ = ((size_t)1ULL);
v___x_1920_ = lean_usize_add(v_i_1905_, v___x_1919_);
v_i_1905_ = v___x_1920_;
v_b_1907_ = v_a_1918_;
goto _start;
}
v___jp_1922_:
{
lean_object* v___x_1923_; 
v___x_1923_ = lean_box(0);
v_a_1918_ = v___x_1923_;
goto v___jp_1917_;
}
v___jp_1924_:
{
lean_object* v___x_1925_; 
v___x_1925_ = lean_box(0);
v_a_1918_ = v___x_1925_;
goto v___jp_1917_;
}
v___jp_1926_:
{
if (lean_obj_tag(v___y_1927_) == 0)
{
lean_object* v_a_1928_; 
v_a_1928_ = lean_ctor_get(v___y_1927_, 0);
lean_inc(v_a_1928_);
lean_dec_ref_known(v___y_1927_, 1);
v_a_1918_ = v_a_1928_;
goto v___jp_1917_;
}
else
{
return v___y_1927_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___boxed(lean_object* v_t_2174_, lean_object* v_as_2175_, lean_object* v_i_2176_, lean_object* v_stop_2177_, lean_object* v_b_2178_, lean_object* v___y_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_){
_start:
{
size_t v_i_boxed_2188_; size_t v_stop_boxed_2189_; lean_object* v_res_2190_; 
v_i_boxed_2188_ = lean_unbox_usize(v_i_2176_);
lean_dec(v_i_2176_);
v_stop_boxed_2189_ = lean_unbox_usize(v_stop_2177_);
lean_dec(v_stop_2177_);
v_res_2190_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21(v_t_2174_, v_as_2175_, v_i_boxed_2188_, v_stop_boxed_2189_, v_b_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_, v___y_2186_);
lean_dec(v___y_2186_);
lean_dec_ref(v___y_2185_);
lean_dec(v___y_2184_);
lean_dec_ref(v___y_2183_);
lean_dec(v___y_2182_);
lean_dec_ref(v___y_2181_);
lean_dec(v___y_2180_);
lean_dec_ref(v___y_2179_);
lean_dec_ref(v_as_2175_);
lean_dec_ref(v_t_2174_);
return v_res_2190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(lean_object* v_t_2191_, lean_object* v_as_2192_, size_t v_i_2193_, size_t v_stop_2194_, lean_object* v_b_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_){
_start:
{
lean_object* v_a_2206_; lean_object* v___y_2215_; uint8_t v___x_2217_; 
v___x_2217_ = lean_usize_dec_eq(v_i_2193_, v_stop_2194_);
if (v___x_2217_ == 0)
{
lean_object* v___x_2218_; 
v___x_2218_ = lean_array_uget_borrowed(v_as_2192_, v_i_2193_);
if (lean_obj_tag(v___x_2218_) == 0)
{
lean_object* v___x_2219_; 
v___x_2219_ = lean_box(0);
v_a_2206_ = v___x_2219_;
goto v___jp_2205_;
}
else
{
lean_object* v_val_2220_; uint8_t v___x_2221_; lean_object* v___y_2223_; lean_object* v___y_2224_; uint8_t v_a_2225_; lean_object* v___y_2228_; lean_object* v___y_2229_; lean_object* v___y_2230_; 
v_val_2220_ = lean_ctor_get(v___x_2218_, 0);
v___x_2221_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2220_);
if (v___x_2221_ == 0)
{
lean_object* v___x_2241_; lean_object* v___x_2242_; 
v___x_2241_ = l_Lean_LocalDecl_type(v_val_2220_);
lean_inc_ref(v___x_2241_);
v___x_2242_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__4___redArg(v___x_2241_, v___y_2201_);
if (lean_obj_tag(v___x_2242_) == 0)
{
lean_object* v_a_2243_; lean_object* v___x_2244_; lean_object* v_fst_2245_; lean_object* v_snd_2246_; lean_object* v___x_2248_; uint8_t v_isShared_2249_; uint8_t v_isSharedCheck_2451_; 
v_a_2243_ = lean_ctor_get(v___x_2242_, 0);
lean_inc(v_a_2243_);
lean_dec_ref_known(v___x_2242_, 1);
v___x_2244_ = l_Lean_Expr_getAppFnArgs(v_a_2243_);
v_fst_2245_ = lean_ctor_get(v___x_2244_, 0);
v_snd_2246_ = lean_ctor_get(v___x_2244_, 1);
v_isSharedCheck_2451_ = !lean_is_exclusive(v___x_2244_);
if (v_isSharedCheck_2451_ == 0)
{
v___x_2248_ = v___x_2244_;
v_isShared_2249_ = v_isSharedCheck_2451_;
goto v_resetjp_2247_;
}
else
{
lean_inc(v_snd_2246_);
lean_inc(v_fst_2245_);
lean_dec(v___x_2244_);
v___x_2248_ = lean_box(0);
v_isShared_2249_ = v_isSharedCheck_2451_;
goto v_resetjp_2247_;
}
v_resetjp_2247_:
{
lean_object* v___x_2250_; lean_object* v_env_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; 
v___x_2250_ = lean_st_ref_get(v___y_2203_);
v_env_2251_ = lean_ctor_get(v___x_2250_, 0);
lean_inc_ref(v_env_2251_);
lean_dec(v___x_2250_);
v___x_2252_ = lean_box(0);
v___x_2253_ = lp_mathlib_Lean_Attr_algebraizeAttr;
lean_inc(v_fst_2245_);
v___x_2254_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_2252_, v___x_2253_, v_env_2251_, v_fst_2245_);
if (lean_obj_tag(v___x_2254_) == 0)
{
lean_object* v___x_2255_; 
lean_del_object(v___x_2248_);
lean_dec(v_snd_2246_);
lean_dec(v_fst_2245_);
lean_dec_ref(v___x_2241_);
v___x_2255_ = lean_box(0);
v_a_2206_ = v___x_2255_;
goto v___jp_2205_;
}
else
{
lean_object* v_val_2256_; lean_object* v___x_2257_; 
v_val_2256_ = lean_ctor_get(v___x_2254_, 0);
lean_inc(v_val_2256_);
lean_dec_ref_known(v___x_2254_, 1);
v___x_2257_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_2197_, v___y_2199_, v___y_2201_, v___y_2203_);
if (lean_obj_tag(v___x_2257_) == 0)
{
lean_object* v_a_2258_; lean_object* v___x_2259_; 
v_a_2258_ = lean_ctor_get(v___x_2257_, 0);
lean_inc(v_a_2258_);
lean_dec_ref_known(v___x_2257_, 1);
lean_inc(v_val_2256_);
v___x_2259_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5(v_val_2256_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2259_) == 0)
{
lean_object* v_a_2260_; lean_object* v___x_2261_; 
lean_dec(v_a_2258_);
lean_del_object(v___x_2248_);
lean_dec(v_fst_2245_);
v_a_2260_ = lean_ctor_get(v___x_2259_, 0);
lean_inc(v_a_2260_);
lean_dec_ref_known(v___x_2259_, 1);
v___x_2261_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_val_2256_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2261_) == 0)
{
lean_object* v_a_2262_; lean_object* v___x_2263_; 
v_a_2262_ = lean_ctor_get(v___x_2261_, 0);
lean_inc_n(v_a_2262_, 2);
lean_dec_ref_known(v___x_2261_, 1);
lean_inc(v___y_2203_);
lean_inc_ref(v___y_2202_);
lean_inc(v___y_2201_);
lean_inc_ref(v___y_2200_);
v___x_2263_ = lean_infer_type(v_a_2262_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2263_) == 0)
{
lean_object* v_a_2264_; uint8_t v___x_2265_; lean_object* v___x_2266_; 
v_a_2264_ = lean_ctor_get(v___x_2263_, 0);
lean_inc(v_a_2264_);
lean_dec_ref_known(v___x_2263_, 1);
v___x_2265_ = 0;
v___x_2266_ = l_Lean_Meta_forallMetaTelescope(v_a_2264_, v___x_2265_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2266_) == 0)
{
lean_object* v_a_2267_; lean_object* v_fst_2268_; lean_object* v_fst_2270_; lean_object* v_snd_2271_; lean_object* v___x_2311_; uint8_t v___x_2312_; 
v_a_2267_ = lean_ctor_get(v___x_2266_, 0);
lean_inc(v_a_2267_);
lean_dec_ref_known(v___x_2266_, 1);
v_fst_2268_ = lean_ctor_get(v_a_2267_, 0);
lean_inc(v_fst_2268_);
lean_dec(v_a_2267_);
v___x_2311_ = l_Lean_mkAppN(v_a_2262_, v_fst_2268_);
v___x_2312_ = l_Lean_ConstantInfo_isInductive(v_a_2260_);
lean_dec(v_a_2260_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; 
lean_dec(v_snd_2246_);
lean_dec_ref(v___x_2241_);
v___x_2313_ = l_Lean_instInhabitedExpr;
v___x_2314_ = lean_array_get_size(v_fst_2268_);
v___x_2315_ = lean_unsigned_to_nat(1u);
v___x_2316_ = lean_nat_sub(v___x_2314_, v___x_2315_);
v___x_2317_ = lean_array_get(v___x_2313_, v_fst_2268_, v___x_2316_);
lean_dec(v___x_2316_);
lean_dec(v_fst_2268_);
v___x_2318_ = l_Lean_Expr_mvarId_x21(v___x_2317_);
lean_dec(v___x_2317_);
lean_inc(v_val_2220_);
v___x_2319_ = l_Lean_LocalDecl_toExpr(v_val_2220_);
v___x_2320_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2318_, v___x_2319_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2320_) == 0)
{
lean_object* v___x_2321_; 
lean_dec_ref_known(v___x_2320_, 1);
v___x_2321_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v___x_2311_, v___y_2201_);
if (lean_obj_tag(v___x_2321_) == 0)
{
lean_object* v_a_2322_; lean_object* v___x_2323_; 
v_a_2322_ = lean_ctor_get(v___x_2321_, 0);
lean_inc_n(v_a_2322_, 2);
lean_dec_ref_known(v___x_2321_, 1);
lean_inc(v___y_2203_);
lean_inc_ref(v___y_2202_);
lean_inc(v___y_2201_);
lean_inc_ref(v___y_2200_);
v___x_2323_ = lean_infer_type(v_a_2322_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2323_) == 0)
{
lean_object* v_a_2324_; 
v_a_2324_ = lean_ctor_get(v___x_2323_, 0);
lean_inc(v_a_2324_);
lean_dec_ref_known(v___x_2323_, 1);
v_fst_2270_ = v_a_2322_;
v_snd_2271_ = v_a_2324_;
goto v___jp_2269_;
}
else
{
lean_object* v_a_2325_; lean_object* v___x_2327_; uint8_t v_isShared_2328_; uint8_t v_isSharedCheck_2332_; 
lean_dec(v_a_2322_);
v_a_2325_ = lean_ctor_get(v___x_2323_, 0);
v_isSharedCheck_2332_ = !lean_is_exclusive(v___x_2323_);
if (v_isSharedCheck_2332_ == 0)
{
v___x_2327_ = v___x_2323_;
v_isShared_2328_ = v_isSharedCheck_2332_;
goto v_resetjp_2326_;
}
else
{
lean_inc(v_a_2325_);
lean_dec(v___x_2323_);
v___x_2327_ = lean_box(0);
v_isShared_2328_ = v_isSharedCheck_2332_;
goto v_resetjp_2326_;
}
v_resetjp_2326_:
{
lean_object* v___x_2330_; 
if (v_isShared_2328_ == 0)
{
v___x_2330_ = v___x_2327_;
goto v_reusejp_2329_;
}
else
{
lean_object* v_reuseFailAlloc_2331_; 
v_reuseFailAlloc_2331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2331_, 0, v_a_2325_);
v___x_2330_ = v_reuseFailAlloc_2331_;
goto v_reusejp_2329_;
}
v_reusejp_2329_:
{
return v___x_2330_;
}
}
}
}
else
{
lean_object* v_a_2333_; lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2340_; 
v_a_2333_ = lean_ctor_get(v___x_2321_, 0);
v_isSharedCheck_2340_ = !lean_is_exclusive(v___x_2321_);
if (v_isSharedCheck_2340_ == 0)
{
v___x_2335_ = v___x_2321_;
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
else
{
lean_inc(v_a_2333_);
lean_dec(v___x_2321_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
lean_object* v___x_2338_; 
if (v_isShared_2336_ == 0)
{
v___x_2338_ = v___x_2335_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2339_; 
v_reuseFailAlloc_2339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2339_, 0, v_a_2333_);
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
else
{
lean_object* v_a_2341_; uint8_t v___y_2343_; uint8_t v___x_2344_; 
lean_dec_ref(v___x_2311_);
v_a_2341_ = lean_ctor_get(v___x_2320_, 0);
lean_inc(v_a_2341_);
v___x_2344_ = l_Lean_Exception_isInterrupt(v_a_2341_);
if (v___x_2344_ == 0)
{
uint8_t v___x_2345_; 
v___x_2345_ = l_Lean_Exception_isRuntime(v_a_2341_);
v___y_2343_ = v___x_2345_;
goto v___jp_2342_;
}
else
{
lean_dec(v_a_2341_);
v___y_2343_ = v___x_2344_;
goto v___jp_2342_;
}
v___jp_2342_:
{
if (v___y_2343_ == 0)
{
lean_dec_ref_known(v___x_2320_, 1);
goto v___jp_2212_;
}
else
{
v___y_2215_ = v___x_2320_;
goto v___jp_2214_;
}
}
}
}
else
{
lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; 
v___x_2346_ = l_Lean_instInhabitedExpr;
v___x_2347_ = lean_unsigned_to_nat(0u);
v___x_2348_ = lean_array_get_borrowed(v___x_2346_, v_fst_2268_, v___x_2347_);
v___x_2349_ = l_Lean_Expr_mvarId_x21(v___x_2348_);
v___x_2350_ = lean_array_get_borrowed(v___x_2346_, v_snd_2246_, v___x_2347_);
lean_inc(v___x_2350_);
v___x_2351_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2349_, v___x_2350_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2351_) == 0)
{
lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; 
lean_dec_ref_known(v___x_2351_, 1);
v___x_2352_ = lean_unsigned_to_nat(1u);
v___x_2353_ = lean_array_get(v___x_2346_, v_fst_2268_, v___x_2352_);
lean_dec(v_fst_2268_);
v___x_2354_ = l_Lean_Expr_mvarId_x21(v___x_2353_);
lean_dec(v___x_2353_);
v___x_2355_ = lean_array_get(v___x_2346_, v_snd_2246_, v___x_2352_);
lean_dec(v_snd_2246_);
v___x_2356_ = lp_batteries_Lean_MVarId_assignIfDefEq(v___x_2354_, v___x_2355_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2356_) == 0)
{
lean_object* v___x_2357_; 
lean_dec_ref_known(v___x_2356_, 1);
v___x_2357_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebraize_addProperties_spec__6___redArg(v___x_2311_, v___y_2201_);
if (lean_obj_tag(v___x_2357_) == 0)
{
lean_object* v_a_2358_; lean_object* v___x_2359_; 
v_a_2358_ = lean_ctor_get(v___x_2357_, 0);
lean_inc_n(v_a_2358_, 2);
lean_dec_ref_known(v___x_2357_, 1);
v___x_2359_ = l_Lean_Meta_isExprDefEqGuarded(v___x_2241_, v_a_2358_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2359_) == 0)
{
lean_object* v_a_2360_; uint8_t v___x_2361_; 
v_a_2360_ = lean_ctor_get(v___x_2359_, 0);
lean_inc(v_a_2360_);
lean_dec_ref_known(v___x_2359_, 1);
v___x_2361_ = lean_unbox(v_a_2360_);
lean_dec(v_a_2360_);
if (v___x_2361_ == 0)
{
lean_dec(v_a_2358_);
goto v___jp_2212_;
}
else
{
lean_object* v___x_2362_; 
lean_inc(v_val_2220_);
v___x_2362_ = l_Lean_LocalDecl_toExpr(v_val_2220_);
v_fst_2270_ = v___x_2362_;
v_snd_2271_ = v_a_2358_;
goto v___jp_2269_;
}
}
else
{
lean_object* v_a_2363_; lean_object* v___x_2365_; uint8_t v_isShared_2366_; uint8_t v_isSharedCheck_2370_; 
lean_dec(v_a_2358_);
v_a_2363_ = lean_ctor_get(v___x_2359_, 0);
v_isSharedCheck_2370_ = !lean_is_exclusive(v___x_2359_);
if (v_isSharedCheck_2370_ == 0)
{
v___x_2365_ = v___x_2359_;
v_isShared_2366_ = v_isSharedCheck_2370_;
goto v_resetjp_2364_;
}
else
{
lean_inc(v_a_2363_);
lean_dec(v___x_2359_);
v___x_2365_ = lean_box(0);
v_isShared_2366_ = v_isSharedCheck_2370_;
goto v_resetjp_2364_;
}
v_resetjp_2364_:
{
lean_object* v___x_2368_; 
if (v_isShared_2366_ == 0)
{
v___x_2368_ = v___x_2365_;
goto v_reusejp_2367_;
}
else
{
lean_object* v_reuseFailAlloc_2369_; 
v_reuseFailAlloc_2369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2369_, 0, v_a_2363_);
v___x_2368_ = v_reuseFailAlloc_2369_;
goto v_reusejp_2367_;
}
v_reusejp_2367_:
{
return v___x_2368_;
}
}
}
}
else
{
lean_object* v_a_2371_; lean_object* v___x_2373_; uint8_t v_isShared_2374_; uint8_t v_isSharedCheck_2378_; 
lean_dec_ref(v___x_2241_);
v_a_2371_ = lean_ctor_get(v___x_2357_, 0);
v_isSharedCheck_2378_ = !lean_is_exclusive(v___x_2357_);
if (v_isSharedCheck_2378_ == 0)
{
v___x_2373_ = v___x_2357_;
v_isShared_2374_ = v_isSharedCheck_2378_;
goto v_resetjp_2372_;
}
else
{
lean_inc(v_a_2371_);
lean_dec(v___x_2357_);
v___x_2373_ = lean_box(0);
v_isShared_2374_ = v_isSharedCheck_2378_;
goto v_resetjp_2372_;
}
v_resetjp_2372_:
{
lean_object* v___x_2376_; 
if (v_isShared_2374_ == 0)
{
v___x_2376_ = v___x_2373_;
goto v_reusejp_2375_;
}
else
{
lean_object* v_reuseFailAlloc_2377_; 
v_reuseFailAlloc_2377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2377_, 0, v_a_2371_);
v___x_2376_ = v_reuseFailAlloc_2377_;
goto v_reusejp_2375_;
}
v_reusejp_2375_:
{
return v___x_2376_;
}
}
}
}
else
{
lean_dec_ref(v___x_2311_);
lean_dec_ref(v___x_2241_);
v___y_2215_ = v___x_2356_;
goto v___jp_2214_;
}
}
else
{
lean_dec_ref(v___x_2311_);
lean_dec(v_fst_2268_);
lean_dec(v_snd_2246_);
lean_dec_ref(v___x_2241_);
v___y_2215_ = v___x_2351_;
goto v___jp_2214_;
}
}
v___jp_2269_:
{
lean_object* v_dummy_2272_; lean_object* v_nargs_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; 
v_dummy_2272_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg___closed__0);
v_nargs_2273_ = l_Lean_Expr_getAppNumArgs(v_snd_2271_);
lean_inc(v_nargs_2273_);
v___x_2274_ = lean_mk_array(v_nargs_2273_, v_dummy_2272_);
v___x_2275_ = lean_unsigned_to_nat(1u);
v___x_2276_ = lean_nat_sub(v_nargs_2273_, v___x_2275_);
lean_dec(v_nargs_2273_);
lean_inc_ref(v_snd_2271_);
v___x_2277_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_snd_2271_, v___x_2274_, v___x_2276_);
v___x_2278_ = lean_unsigned_to_nat(0u);
v___x_2279_ = lean_array_get_size(v___x_2277_);
v___x_2280_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2(v___x_2277_, v___x_2278_, v___x_2279_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
lean_dec_ref(v___x_2277_);
if (lean_obj_tag(v___x_2280_) == 0)
{
lean_object* v_a_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; 
v_a_2281_ = lean_ctor_get(v___x_2280_, 0);
lean_inc(v_a_2281_);
lean_dec_ref_known(v___x_2280_, 1);
v___x_2282_ = lean_box(0);
lean_inc_ref(v_snd_2271_);
v___x_2283_ = l_Lean_Meta_synthInstance_x3f(v_snd_2271_, v___x_2282_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2283_) == 0)
{
lean_object* v_a_2284_; lean_object* v___f_2285_; lean_object* v___x_2286_; uint8_t v___x_2287_; 
v_a_2284_ = lean_ctor_get(v___x_2283_, 0);
lean_inc(v_a_2284_);
lean_dec_ref_known(v___x_2283_, 1);
v___f_2285_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__1___boxed), 11, 2);
lean_closure_set(v___f_2285_, 0, v_snd_2271_);
lean_closure_set(v___f_2285_, 1, v_fst_2270_);
v___x_2286_ = lean_array_get_size(v_a_2281_);
v___x_2287_ = lean_nat_dec_lt(v___x_2278_, v___x_2286_);
if (v___x_2287_ == 0)
{
lean_object* v___x_2288_; 
lean_dec(v_a_2281_);
v___x_2288_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(v___x_2221_, v___x_2221_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
v___y_2228_ = v_a_2284_;
v___y_2229_ = v___f_2285_;
v___y_2230_ = v___x_2288_;
goto v___jp_2227_;
}
else
{
if (v___x_2287_ == 0)
{
lean_dec(v_a_2281_);
v___y_2223_ = v_a_2284_;
v___y_2224_ = v___f_2285_;
v_a_2225_ = v___x_2287_;
goto v___jp_2222_;
}
else
{
size_t v___x_2289_; size_t v___x_2290_; lean_object* v___x_2291_; 
v___x_2289_ = ((size_t)0ULL);
v___x_2290_ = lean_usize_of_nat(v___x_2286_);
v___x_2291_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_Algebraize_addProperties_spec__3(v_t_2191_, v___x_2221_, v_a_2281_, v___x_2289_, v___x_2290_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
lean_dec(v_a_2281_);
if (lean_obj_tag(v___x_2291_) == 0)
{
lean_object* v_a_2292_; uint8_t v___x_2293_; lean_object* v___x_2294_; 
v_a_2292_ = lean_ctor_get(v___x_2291_, 0);
lean_inc(v_a_2292_);
lean_dec_ref_known(v___x_2291_, 1);
v___x_2293_ = lean_unbox(v_a_2292_);
lean_dec(v_a_2292_);
v___x_2294_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___lam__0(v___x_2221_, v___x_2293_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
v___y_2228_ = v_a_2284_;
v___y_2229_ = v___f_2285_;
v___y_2230_ = v___x_2294_;
goto v___jp_2227_;
}
else
{
v___y_2228_ = v_a_2284_;
v___y_2229_ = v___f_2285_;
v___y_2230_ = v___x_2291_;
goto v___jp_2227_;
}
}
}
}
else
{
lean_object* v_a_2295_; lean_object* v___x_2297_; uint8_t v_isShared_2298_; uint8_t v_isSharedCheck_2302_; 
lean_dec(v_a_2281_);
lean_dec_ref(v_snd_2271_);
lean_dec_ref(v_fst_2270_);
v_a_2295_ = lean_ctor_get(v___x_2283_, 0);
v_isSharedCheck_2302_ = !lean_is_exclusive(v___x_2283_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2297_ = v___x_2283_;
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
else
{
lean_inc(v_a_2295_);
lean_dec(v___x_2283_);
v___x_2297_ = lean_box(0);
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
v_resetjp_2296_:
{
lean_object* v___x_2300_; 
if (v_isShared_2298_ == 0)
{
v___x_2300_ = v___x_2297_;
goto v_reusejp_2299_;
}
else
{
lean_object* v_reuseFailAlloc_2301_; 
v_reuseFailAlloc_2301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2301_, 0, v_a_2295_);
v___x_2300_ = v_reuseFailAlloc_2301_;
goto v_reusejp_2299_;
}
v_reusejp_2299_:
{
return v___x_2300_;
}
}
}
}
else
{
lean_object* v_a_2303_; lean_object* v___x_2305_; uint8_t v_isShared_2306_; uint8_t v_isSharedCheck_2310_; 
lean_dec_ref(v_snd_2271_);
lean_dec_ref(v_fst_2270_);
v_a_2303_ = lean_ctor_get(v___x_2280_, 0);
v_isSharedCheck_2310_ = !lean_is_exclusive(v___x_2280_);
if (v_isSharedCheck_2310_ == 0)
{
v___x_2305_ = v___x_2280_;
v_isShared_2306_ = v_isSharedCheck_2310_;
goto v_resetjp_2304_;
}
else
{
lean_inc(v_a_2303_);
lean_dec(v___x_2280_);
v___x_2305_ = lean_box(0);
v_isShared_2306_ = v_isSharedCheck_2310_;
goto v_resetjp_2304_;
}
v_resetjp_2304_:
{
lean_object* v___x_2308_; 
if (v_isShared_2306_ == 0)
{
v___x_2308_ = v___x_2305_;
goto v_reusejp_2307_;
}
else
{
lean_object* v_reuseFailAlloc_2309_; 
v_reuseFailAlloc_2309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2309_, 0, v_a_2303_);
v___x_2308_ = v_reuseFailAlloc_2309_;
goto v_reusejp_2307_;
}
v_reusejp_2307_:
{
return v___x_2308_;
}
}
}
}
}
else
{
lean_object* v_a_2379_; lean_object* v___x_2381_; uint8_t v_isShared_2382_; uint8_t v_isSharedCheck_2386_; 
lean_dec(v_a_2262_);
lean_dec(v_a_2260_);
lean_dec(v_snd_2246_);
lean_dec_ref(v___x_2241_);
v_a_2379_ = lean_ctor_get(v___x_2266_, 0);
v_isSharedCheck_2386_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2386_ == 0)
{
v___x_2381_ = v___x_2266_;
v_isShared_2382_ = v_isSharedCheck_2386_;
goto v_resetjp_2380_;
}
else
{
lean_inc(v_a_2379_);
lean_dec(v___x_2266_);
v___x_2381_ = lean_box(0);
v_isShared_2382_ = v_isSharedCheck_2386_;
goto v_resetjp_2380_;
}
v_resetjp_2380_:
{
lean_object* v___x_2384_; 
if (v_isShared_2382_ == 0)
{
v___x_2384_ = v___x_2381_;
goto v_reusejp_2383_;
}
else
{
lean_object* v_reuseFailAlloc_2385_; 
v_reuseFailAlloc_2385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2385_, 0, v_a_2379_);
v___x_2384_ = v_reuseFailAlloc_2385_;
goto v_reusejp_2383_;
}
v_reusejp_2383_:
{
return v___x_2384_;
}
}
}
}
else
{
lean_object* v_a_2387_; lean_object* v___x_2389_; uint8_t v_isShared_2390_; uint8_t v_isSharedCheck_2394_; 
lean_dec(v_a_2262_);
lean_dec(v_a_2260_);
lean_dec(v_snd_2246_);
lean_dec_ref(v___x_2241_);
v_a_2387_ = lean_ctor_get(v___x_2263_, 0);
v_isSharedCheck_2394_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2394_ == 0)
{
v___x_2389_ = v___x_2263_;
v_isShared_2390_ = v_isSharedCheck_2394_;
goto v_resetjp_2388_;
}
else
{
lean_inc(v_a_2387_);
lean_dec(v___x_2263_);
v___x_2389_ = lean_box(0);
v_isShared_2390_ = v_isSharedCheck_2394_;
goto v_resetjp_2388_;
}
v_resetjp_2388_:
{
lean_object* v___x_2392_; 
if (v_isShared_2390_ == 0)
{
v___x_2392_ = v___x_2389_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2393_; 
v_reuseFailAlloc_2393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2393_, 0, v_a_2387_);
v___x_2392_ = v_reuseFailAlloc_2393_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
return v___x_2392_;
}
}
}
}
else
{
lean_object* v_a_2395_; lean_object* v___x_2397_; uint8_t v_isShared_2398_; uint8_t v_isSharedCheck_2402_; 
lean_dec(v_a_2260_);
lean_dec(v_snd_2246_);
lean_dec_ref(v___x_2241_);
v_a_2395_ = lean_ctor_get(v___x_2261_, 0);
v_isSharedCheck_2402_ = !lean_is_exclusive(v___x_2261_);
if (v_isSharedCheck_2402_ == 0)
{
v___x_2397_ = v___x_2261_;
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
else
{
lean_inc(v_a_2395_);
lean_dec(v___x_2261_);
v___x_2397_ = lean_box(0);
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
v_resetjp_2396_:
{
lean_object* v___x_2400_; 
if (v_isShared_2398_ == 0)
{
v___x_2400_ = v___x_2397_;
goto v_reusejp_2399_;
}
else
{
lean_object* v_reuseFailAlloc_2401_; 
v_reuseFailAlloc_2401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2401_, 0, v_a_2395_);
v___x_2400_ = v_reuseFailAlloc_2401_;
goto v_reusejp_2399_;
}
v_reusejp_2399_:
{
return v___x_2400_;
}
}
}
}
else
{
lean_object* v_a_2403_; lean_object* v___x_2405_; uint8_t v_isShared_2406_; uint8_t v_isSharedCheck_2442_; 
lean_dec(v_snd_2246_);
v_a_2403_ = lean_ctor_get(v___x_2259_, 0);
v_isSharedCheck_2442_ = !lean_is_exclusive(v___x_2259_);
if (v_isSharedCheck_2442_ == 0)
{
v___x_2405_ = v___x_2259_;
v_isShared_2406_ = v_isSharedCheck_2442_;
goto v_resetjp_2404_;
}
else
{
lean_inc(v_a_2403_);
lean_dec(v___x_2259_);
v___x_2405_ = lean_box(0);
v_isShared_2406_ = v_isSharedCheck_2442_;
goto v_resetjp_2404_;
}
v_resetjp_2404_:
{
uint8_t v___y_2408_; uint8_t v___x_2440_; 
v___x_2440_ = l_Lean_Exception_isInterrupt(v_a_2403_);
if (v___x_2440_ == 0)
{
uint8_t v___x_2441_; 
lean_inc(v_a_2403_);
v___x_2441_ = l_Lean_Exception_isRuntime(v_a_2403_);
v___y_2408_ = v___x_2441_;
goto v___jp_2407_;
}
else
{
v___y_2408_ = v___x_2440_;
goto v___jp_2407_;
}
v___jp_2407_:
{
if (v___y_2408_ == 0)
{
lean_object* v___x_2409_; 
lean_del_object(v___x_2405_);
lean_dec(v_a_2403_);
v___x_2409_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2258_, v___y_2408_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2409_) == 0)
{
lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2414_; 
lean_dec_ref_known(v___x_2409_, 1);
v___x_2410_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__1);
lean_inc(v_val_2220_);
v___x_2411_ = l_Lean_LocalDecl_toExpr(v_val_2220_);
v___x_2412_ = l_Lean_MessageData_ofExpr(v___x_2411_);
if (v_isShared_2249_ == 0)
{
lean_ctor_set_tag(v___x_2248_, 7);
lean_ctor_set(v___x_2248_, 1, v___x_2412_);
lean_ctor_set(v___x_2248_, 0, v___x_2410_);
v___x_2414_ = v___x_2248_;
goto v_reusejp_2413_;
}
else
{
lean_object* v_reuseFailAlloc_2436_; 
v_reuseFailAlloc_2436_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2436_, 0, v___x_2410_);
lean_ctor_set(v_reuseFailAlloc_2436_, 1, v___x_2412_);
v___x_2414_ = v_reuseFailAlloc_2436_;
goto v_reusejp_2413_;
}
v_reusejp_2413_:
{
lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; 
v___x_2415_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__3);
v___x_2416_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2416_, 0, v___x_2414_);
lean_ctor_set(v___x_2416_, 1, v___x_2415_);
v___x_2417_ = l_Lean_MessageData_ofExpr(v___x_2241_);
v___x_2418_ = l_Lean_indentD(v___x_2417_);
v___x_2419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___x_2416_);
lean_ctor_set(v___x_2419_, 1, v___x_2418_);
v___x_2420_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__5);
v___x_2421_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2421_, 0, v___x_2419_);
lean_ctor_set(v___x_2421_, 1, v___x_2420_);
v___x_2422_ = l_Lean_MessageData_ofConstName(v_fst_2245_, v___y_2408_);
v___x_2423_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2423_, 0, v___x_2421_);
lean_ctor_set(v___x_2423_, 1, v___x_2422_);
v___x_2424_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__7);
v___x_2425_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2425_, 0, v___x_2423_);
lean_ctor_set(v___x_2425_, 1, v___x_2424_);
v___x_2426_ = l_Lean_MessageData_ofName(v_val_2256_);
lean_inc_ref(v___x_2426_);
v___x_2427_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2427_, 0, v___x_2425_);
lean_ctor_set(v___x_2427_, 1, v___x_2426_);
v___x_2428_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__9);
v___x_2429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2429_, 0, v___x_2427_);
lean_ctor_set(v___x_2429_, 1, v___x_2428_);
v___x_2430_ = l_Lean_indentD(v___x_2426_);
v___x_2431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2429_);
lean_ctor_set(v___x_2431_, 1, v___x_2430_);
v___x_2432_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21___closed__11);
v___x_2433_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2433_, 0, v___x_2431_);
lean_ctor_set(v___x_2433_, 1, v___x_2432_);
v___x_2434_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7(v___x_2433_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
if (lean_obj_tag(v___x_2434_) == 0)
{
lean_object* v___x_2435_; 
lean_dec_ref_known(v___x_2434_, 1);
v___x_2435_ = lean_box(0);
v_a_2206_ = v___x_2435_;
goto v___jp_2205_;
}
else
{
v___y_2215_ = v___x_2434_;
goto v___jp_2214_;
}
}
}
else
{
lean_dec(v_val_2256_);
lean_del_object(v___x_2248_);
lean_dec(v_fst_2245_);
lean_dec_ref(v___x_2241_);
v___y_2215_ = v___x_2409_;
goto v___jp_2214_;
}
}
else
{
lean_object* v___x_2438_; 
lean_dec(v_a_2258_);
lean_dec(v_val_2256_);
lean_del_object(v___x_2248_);
lean_dec(v_fst_2245_);
lean_dec_ref(v___x_2241_);
if (v_isShared_2406_ == 0)
{
v___x_2438_ = v___x_2405_;
goto v_reusejp_2437_;
}
else
{
lean_object* v_reuseFailAlloc_2439_; 
v_reuseFailAlloc_2439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2439_, 0, v_a_2403_);
v___x_2438_ = v_reuseFailAlloc_2439_;
goto v_reusejp_2437_;
}
v_reusejp_2437_:
{
return v___x_2438_;
}
}
}
}
}
}
else
{
lean_object* v_a_2443_; lean_object* v___x_2445_; uint8_t v_isShared_2446_; uint8_t v_isSharedCheck_2450_; 
lean_dec(v_val_2256_);
lean_del_object(v___x_2248_);
lean_dec(v_snd_2246_);
lean_dec(v_fst_2245_);
lean_dec_ref(v___x_2241_);
v_a_2443_ = lean_ctor_get(v___x_2257_, 0);
v_isSharedCheck_2450_ = !lean_is_exclusive(v___x_2257_);
if (v_isSharedCheck_2450_ == 0)
{
v___x_2445_ = v___x_2257_;
v_isShared_2446_ = v_isSharedCheck_2450_;
goto v_resetjp_2444_;
}
else
{
lean_inc(v_a_2443_);
lean_dec(v___x_2257_);
v___x_2445_ = lean_box(0);
v_isShared_2446_ = v_isSharedCheck_2450_;
goto v_resetjp_2444_;
}
v_resetjp_2444_:
{
lean_object* v___x_2448_; 
if (v_isShared_2446_ == 0)
{
v___x_2448_ = v___x_2445_;
goto v_reusejp_2447_;
}
else
{
lean_object* v_reuseFailAlloc_2449_; 
v_reuseFailAlloc_2449_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2449_, 0, v_a_2443_);
v___x_2448_ = v_reuseFailAlloc_2449_;
goto v_reusejp_2447_;
}
v_reusejp_2447_:
{
return v___x_2448_;
}
}
}
}
}
}
else
{
lean_object* v_a_2452_; lean_object* v___x_2454_; uint8_t v_isShared_2455_; uint8_t v_isSharedCheck_2459_; 
lean_dec_ref(v___x_2241_);
v_a_2452_ = lean_ctor_get(v___x_2242_, 0);
v_isSharedCheck_2459_ = !lean_is_exclusive(v___x_2242_);
if (v_isSharedCheck_2459_ == 0)
{
v___x_2454_ = v___x_2242_;
v_isShared_2455_ = v_isSharedCheck_2459_;
goto v_resetjp_2453_;
}
else
{
lean_inc(v_a_2452_);
lean_dec(v___x_2242_);
v___x_2454_ = lean_box(0);
v_isShared_2455_ = v_isSharedCheck_2459_;
goto v_resetjp_2453_;
}
v_resetjp_2453_:
{
lean_object* v___x_2457_; 
if (v_isShared_2455_ == 0)
{
v___x_2457_ = v___x_2454_;
goto v_reusejp_2456_;
}
else
{
lean_object* v_reuseFailAlloc_2458_; 
v_reuseFailAlloc_2458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2458_, 0, v_a_2452_);
v___x_2457_ = v_reuseFailAlloc_2458_;
goto v_reusejp_2456_;
}
v_reusejp_2456_:
{
return v___x_2457_;
}
}
}
}
else
{
lean_object* v___x_2460_; 
v___x_2460_ = lean_box(0);
v_a_2206_ = v___x_2460_;
goto v___jp_2205_;
}
v___jp_2222_:
{
if (lean_obj_tag(v___y_2223_) == 0)
{
if (v___x_2221_ == 0)
{
if (v_a_2225_ == 0)
{
lean_dec_ref(v___y_2224_);
goto v___jp_2210_;
}
else
{
lean_object* v___x_2226_; 
v___x_2226_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_2224_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
v___y_2215_ = v___x_2226_;
goto v___jp_2214_;
}
}
else
{
lean_dec_ref(v___y_2224_);
goto v___jp_2210_;
}
}
else
{
lean_dec_ref_known(v___y_2223_, 1);
lean_dec_ref(v___y_2224_);
goto v___jp_2210_;
}
}
v___jp_2227_:
{
if (lean_obj_tag(v___y_2230_) == 0)
{
lean_object* v_a_2231_; uint8_t v___x_2232_; 
v_a_2231_ = lean_ctor_get(v___y_2230_, 0);
lean_inc(v_a_2231_);
lean_dec_ref_known(v___y_2230_, 1);
v___x_2232_ = lean_unbox(v_a_2231_);
lean_dec(v_a_2231_);
v___y_2223_ = v___y_2228_;
v___y_2224_ = v___y_2229_;
v_a_2225_ = v___x_2232_;
goto v___jp_2222_;
}
else
{
lean_object* v_a_2233_; lean_object* v___x_2235_; uint8_t v_isShared_2236_; uint8_t v_isSharedCheck_2240_; 
lean_dec_ref(v___y_2229_);
lean_dec(v___y_2228_);
v_a_2233_ = lean_ctor_get(v___y_2230_, 0);
v_isSharedCheck_2240_ = !lean_is_exclusive(v___y_2230_);
if (v_isSharedCheck_2240_ == 0)
{
v___x_2235_ = v___y_2230_;
v_isShared_2236_ = v_isSharedCheck_2240_;
goto v_resetjp_2234_;
}
else
{
lean_inc(v_a_2233_);
lean_dec(v___y_2230_);
v___x_2235_ = lean_box(0);
v_isShared_2236_ = v_isSharedCheck_2240_;
goto v_resetjp_2234_;
}
v_resetjp_2234_:
{
lean_object* v___x_2238_; 
if (v_isShared_2236_ == 0)
{
v___x_2238_ = v___x_2235_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v_a_2233_);
v___x_2238_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
return v___x_2238_;
}
}
}
}
}
}
else
{
lean_object* v___x_2461_; 
v___x_2461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2461_, 0, v_b_2195_);
return v___x_2461_;
}
v___jp_2205_:
{
size_t v___x_2207_; size_t v___x_2208_; lean_object* v___x_2209_; 
v___x_2207_ = ((size_t)1ULL);
v___x_2208_ = lean_usize_add(v_i_2193_, v___x_2207_);
v___x_2209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15_spec__21(v_t_2191_, v_as_2192_, v___x_2208_, v_stop_2194_, v_a_2206_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_);
return v___x_2209_;
}
v___jp_2210_:
{
lean_object* v___x_2211_; 
v___x_2211_ = lean_box(0);
v_a_2206_ = v___x_2211_;
goto v___jp_2205_;
}
v___jp_2212_:
{
lean_object* v___x_2213_; 
v___x_2213_ = lean_box(0);
v_a_2206_ = v___x_2213_;
goto v___jp_2205_;
}
v___jp_2214_:
{
if (lean_obj_tag(v___y_2215_) == 0)
{
lean_object* v_a_2216_; 
v_a_2216_ = lean_ctor_get(v___y_2215_, 0);
lean_inc(v_a_2216_);
lean_dec_ref_known(v___y_2215_, 1);
v_a_2206_ = v_a_2216_;
goto v___jp_2205_;
}
else
{
return v___y_2215_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15___boxed(lean_object* v_t_2462_, lean_object* v_as_2463_, lean_object* v_i_2464_, lean_object* v_stop_2465_, lean_object* v_b_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_, lean_object* v___y_2470_, lean_object* v___y_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_){
_start:
{
size_t v_i_boxed_2476_; size_t v_stop_boxed_2477_; lean_object* v_res_2478_; 
v_i_boxed_2476_ = lean_unbox_usize(v_i_2464_);
lean_dec(v_i_2464_);
v_stop_boxed_2477_ = lean_unbox_usize(v_stop_2465_);
lean_dec(v_stop_2465_);
v_res_2478_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2462_, v_as_2463_, v_i_boxed_2476_, v_stop_boxed_2477_, v_b_2466_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_, v___y_2471_, v___y_2472_, v___y_2473_, v___y_2474_);
lean_dec(v___y_2474_);
lean_dec_ref(v___y_2473_);
lean_dec(v___y_2472_);
lean_dec_ref(v___y_2471_);
lean_dec(v___y_2470_);
lean_dec_ref(v___y_2469_);
lean_dec(v___y_2468_);
lean_dec_ref(v___y_2467_);
lean_dec_ref(v_as_2463_);
lean_dec_ref(v_t_2462_);
return v_res_2478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18(lean_object* v_t_2479_, lean_object* v_x_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_){
_start:
{
if (lean_obj_tag(v_x_2480_) == 0)
{
lean_object* v_cs_2490_; lean_object* v___x_2492_; uint8_t v_isShared_2493_; uint8_t v_isSharedCheck_2511_; 
v_cs_2490_ = lean_ctor_get(v_x_2480_, 0);
v_isSharedCheck_2511_ = !lean_is_exclusive(v_x_2480_);
if (v_isSharedCheck_2511_ == 0)
{
v___x_2492_ = v_x_2480_;
v_isShared_2493_ = v_isSharedCheck_2511_;
goto v_resetjp_2491_;
}
else
{
lean_inc(v_cs_2490_);
lean_dec(v_x_2480_);
v___x_2492_ = lean_box(0);
v_isShared_2493_ = v_isSharedCheck_2511_;
goto v_resetjp_2491_;
}
v_resetjp_2491_:
{
lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; uint8_t v___x_2497_; 
v___x_2494_ = lean_unsigned_to_nat(0u);
v___x_2495_ = lean_array_get_size(v_cs_2490_);
v___x_2496_ = lean_box(0);
v___x_2497_ = lean_nat_dec_lt(v___x_2494_, v___x_2495_);
if (v___x_2497_ == 0)
{
lean_object* v___x_2499_; 
lean_dec_ref(v_cs_2490_);
if (v_isShared_2493_ == 0)
{
lean_ctor_set(v___x_2492_, 0, v___x_2496_);
v___x_2499_ = v___x_2492_;
goto v_reusejp_2498_;
}
else
{
lean_object* v_reuseFailAlloc_2500_; 
v_reuseFailAlloc_2500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2500_, 0, v___x_2496_);
v___x_2499_ = v_reuseFailAlloc_2500_;
goto v_reusejp_2498_;
}
v_reusejp_2498_:
{
return v___x_2499_;
}
}
else
{
uint8_t v___x_2501_; 
v___x_2501_ = lean_nat_dec_le(v___x_2495_, v___x_2495_);
if (v___x_2501_ == 0)
{
if (v___x_2497_ == 0)
{
lean_object* v___x_2503_; 
lean_dec_ref(v_cs_2490_);
if (v_isShared_2493_ == 0)
{
lean_ctor_set(v___x_2492_, 0, v___x_2496_);
v___x_2503_ = v___x_2492_;
goto v_reusejp_2502_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v___x_2496_);
v___x_2503_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2502_;
}
v_reusejp_2502_:
{
return v___x_2503_;
}
}
else
{
size_t v___x_2505_; size_t v___x_2506_; lean_object* v___x_2507_; 
lean_del_object(v___x_2492_);
v___x_2505_ = ((size_t)0ULL);
v___x_2506_ = lean_usize_of_nat(v___x_2495_);
v___x_2507_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(v_t_2479_, v_cs_2490_, v___x_2505_, v___x_2506_, v___x_2496_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_);
lean_dec_ref(v_cs_2490_);
return v___x_2507_;
}
}
else
{
size_t v___x_2508_; size_t v___x_2509_; lean_object* v___x_2510_; 
lean_del_object(v___x_2492_);
v___x_2508_ = ((size_t)0ULL);
v___x_2509_ = lean_usize_of_nat(v___x_2495_);
v___x_2510_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(v_t_2479_, v_cs_2490_, v___x_2508_, v___x_2509_, v___x_2496_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_);
lean_dec_ref(v_cs_2490_);
return v___x_2510_;
}
}
}
}
else
{
lean_object* v_vs_2512_; lean_object* v___x_2514_; uint8_t v_isShared_2515_; uint8_t v_isSharedCheck_2533_; 
v_vs_2512_ = lean_ctor_get(v_x_2480_, 0);
v_isSharedCheck_2533_ = !lean_is_exclusive(v_x_2480_);
if (v_isSharedCheck_2533_ == 0)
{
v___x_2514_ = v_x_2480_;
v_isShared_2515_ = v_isSharedCheck_2533_;
goto v_resetjp_2513_;
}
else
{
lean_inc(v_vs_2512_);
lean_dec(v_x_2480_);
v___x_2514_ = lean_box(0);
v_isShared_2515_ = v_isSharedCheck_2533_;
goto v_resetjp_2513_;
}
v_resetjp_2513_:
{
lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2518_; uint8_t v___x_2519_; 
v___x_2516_ = lean_unsigned_to_nat(0u);
v___x_2517_ = lean_array_get_size(v_vs_2512_);
v___x_2518_ = lean_box(0);
v___x_2519_ = lean_nat_dec_lt(v___x_2516_, v___x_2517_);
if (v___x_2519_ == 0)
{
lean_object* v___x_2521_; 
lean_dec_ref(v_vs_2512_);
if (v_isShared_2515_ == 0)
{
lean_ctor_set_tag(v___x_2514_, 0);
lean_ctor_set(v___x_2514_, 0, v___x_2518_);
v___x_2521_ = v___x_2514_;
goto v_reusejp_2520_;
}
else
{
lean_object* v_reuseFailAlloc_2522_; 
v_reuseFailAlloc_2522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2522_, 0, v___x_2518_);
v___x_2521_ = v_reuseFailAlloc_2522_;
goto v_reusejp_2520_;
}
v_reusejp_2520_:
{
return v___x_2521_;
}
}
else
{
uint8_t v___x_2523_; 
v___x_2523_ = lean_nat_dec_le(v___x_2517_, v___x_2517_);
if (v___x_2523_ == 0)
{
if (v___x_2519_ == 0)
{
lean_object* v___x_2525_; 
lean_dec_ref(v_vs_2512_);
if (v_isShared_2515_ == 0)
{
lean_ctor_set_tag(v___x_2514_, 0);
lean_ctor_set(v___x_2514_, 0, v___x_2518_);
v___x_2525_ = v___x_2514_;
goto v_reusejp_2524_;
}
else
{
lean_object* v_reuseFailAlloc_2526_; 
v_reuseFailAlloc_2526_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2526_, 0, v___x_2518_);
v___x_2525_ = v_reuseFailAlloc_2526_;
goto v_reusejp_2524_;
}
v_reusejp_2524_:
{
return v___x_2525_;
}
}
else
{
size_t v___x_2527_; size_t v___x_2528_; lean_object* v___x_2529_; 
lean_del_object(v___x_2514_);
v___x_2527_ = ((size_t)0ULL);
v___x_2528_ = lean_usize_of_nat(v___x_2517_);
v___x_2529_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2479_, v_vs_2512_, v___x_2527_, v___x_2528_, v___x_2518_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_);
lean_dec_ref(v_vs_2512_);
return v___x_2529_;
}
}
else
{
size_t v___x_2530_; size_t v___x_2531_; lean_object* v___x_2532_; 
lean_del_object(v___x_2514_);
v___x_2530_ = ((size_t)0ULL);
v___x_2531_ = lean_usize_of_nat(v___x_2517_);
v___x_2532_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2479_, v_vs_2512_, v___x_2530_, v___x_2531_, v___x_2518_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_);
lean_dec_ref(v_vs_2512_);
return v___x_2532_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(lean_object* v_t_2534_, lean_object* v_as_2535_, size_t v_i_2536_, size_t v_stop_2537_, lean_object* v_b_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_){
_start:
{
uint8_t v___x_2548_; 
v___x_2548_ = lean_usize_dec_eq(v_i_2536_, v_stop_2537_);
if (v___x_2548_ == 0)
{
lean_object* v___x_2549_; lean_object* v___x_2550_; 
v___x_2549_ = lean_array_uget_borrowed(v_as_2535_, v_i_2536_);
lean_inc(v___x_2549_);
v___x_2550_ = lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18(v_t_2534_, v___x_2549_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_, v___y_2543_, v___y_2544_, v___y_2545_, v___y_2546_);
if (lean_obj_tag(v___x_2550_) == 0)
{
lean_object* v_a_2551_; size_t v___x_2552_; size_t v___x_2553_; 
v_a_2551_ = lean_ctor_get(v___x_2550_, 0);
lean_inc(v_a_2551_);
lean_dec_ref_known(v___x_2550_, 1);
v___x_2552_ = ((size_t)1ULL);
v___x_2553_ = lean_usize_add(v_i_2536_, v___x_2552_);
v_i_2536_ = v___x_2553_;
v_b_2538_ = v_a_2551_;
goto _start;
}
else
{
return v___x_2550_;
}
}
else
{
lean_object* v___x_2555_; 
v___x_2555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2555_, 0, v_b_2538_);
return v___x_2555_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19___boxed(lean_object* v_t_2556_, lean_object* v_as_2557_, lean_object* v_i_2558_, lean_object* v_stop_2559_, lean_object* v_b_2560_, lean_object* v___y_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_, lean_object* v___y_2569_){
_start:
{
size_t v_i_boxed_2570_; size_t v_stop_boxed_2571_; lean_object* v_res_2572_; 
v_i_boxed_2570_ = lean_unbox_usize(v_i_2558_);
lean_dec(v_i_2558_);
v_stop_boxed_2571_ = lean_unbox_usize(v_stop_2559_);
lean_dec(v_stop_2559_);
v_res_2572_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(v_t_2556_, v_as_2557_, v_i_boxed_2570_, v_stop_boxed_2571_, v_b_2560_, v___y_2561_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_);
lean_dec(v___y_2568_);
lean_dec_ref(v___y_2567_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
lean_dec(v___y_2564_);
lean_dec_ref(v___y_2563_);
lean_dec(v___y_2562_);
lean_dec_ref(v___y_2561_);
lean_dec_ref(v_as_2557_);
lean_dec_ref(v_t_2556_);
return v_res_2572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18___boxed(lean_object* v_t_2573_, lean_object* v_x_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_){
_start:
{
lean_object* v_res_2584_; 
v_res_2584_ = lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18(v_t_2573_, v_x_2574_, v___y_2575_, v___y_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_, v___y_2582_);
lean_dec(v___y_2582_);
lean_dec_ref(v___y_2581_);
lean_dec(v___y_2580_);
lean_dec_ref(v___y_2579_);
lean_dec(v___y_2578_);
lean_dec_ref(v___y_2577_);
lean_dec(v___y_2576_);
lean_dec_ref(v___y_2575_);
lean_dec_ref(v_t_2573_);
return v_res_2584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16(lean_object* v_t_2585_, lean_object* v_t_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_, lean_object* v___y_2590_, lean_object* v___y_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_){
_start:
{
lean_object* v_root_2596_; lean_object* v_tail_2597_; lean_object* v___x_2598_; 
v_root_2596_ = lean_ctor_get(v_t_2586_, 0);
lean_inc_ref(v_root_2596_);
v_tail_2597_ = lean_ctor_get(v_t_2586_, 1);
lean_inc_ref(v_tail_2597_);
lean_dec_ref(v_t_2586_);
v___x_2598_ = lp_mathlib_Lean_PersistentArray_forMAux___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__18(v_t_2585_, v_root_2596_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_, v___y_2594_);
if (lean_obj_tag(v___x_2598_) == 0)
{
lean_object* v___x_2600_; uint8_t v_isShared_2601_; uint8_t v_isSharedCheck_2619_; 
v_isSharedCheck_2619_ = !lean_is_exclusive(v___x_2598_);
if (v_isSharedCheck_2619_ == 0)
{
lean_object* v_unused_2620_; 
v_unused_2620_ = lean_ctor_get(v___x_2598_, 0);
lean_dec(v_unused_2620_);
v___x_2600_ = v___x_2598_;
v_isShared_2601_ = v_isSharedCheck_2619_;
goto v_resetjp_2599_;
}
else
{
lean_dec(v___x_2598_);
v___x_2600_ = lean_box(0);
v_isShared_2601_ = v_isSharedCheck_2619_;
goto v_resetjp_2599_;
}
v_resetjp_2599_:
{
lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; uint8_t v___x_2605_; 
v___x_2602_ = lean_unsigned_to_nat(0u);
v___x_2603_ = lean_array_get_size(v_tail_2597_);
v___x_2604_ = lean_box(0);
v___x_2605_ = lean_nat_dec_lt(v___x_2602_, v___x_2603_);
if (v___x_2605_ == 0)
{
lean_object* v___x_2607_; 
lean_dec_ref(v_tail_2597_);
if (v_isShared_2601_ == 0)
{
lean_ctor_set(v___x_2600_, 0, v___x_2604_);
v___x_2607_ = v___x_2600_;
goto v_reusejp_2606_;
}
else
{
lean_object* v_reuseFailAlloc_2608_; 
v_reuseFailAlloc_2608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2608_, 0, v___x_2604_);
v___x_2607_ = v_reuseFailAlloc_2608_;
goto v_reusejp_2606_;
}
v_reusejp_2606_:
{
return v___x_2607_;
}
}
else
{
uint8_t v___x_2609_; 
v___x_2609_ = lean_nat_dec_le(v___x_2603_, v___x_2603_);
if (v___x_2609_ == 0)
{
if (v___x_2605_ == 0)
{
lean_object* v___x_2611_; 
lean_dec_ref(v_tail_2597_);
if (v_isShared_2601_ == 0)
{
lean_ctor_set(v___x_2600_, 0, v___x_2604_);
v___x_2611_ = v___x_2600_;
goto v_reusejp_2610_;
}
else
{
lean_object* v_reuseFailAlloc_2612_; 
v_reuseFailAlloc_2612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2612_, 0, v___x_2604_);
v___x_2611_ = v_reuseFailAlloc_2612_;
goto v_reusejp_2610_;
}
v_reusejp_2610_:
{
return v___x_2611_;
}
}
else
{
size_t v___x_2613_; size_t v___x_2614_; lean_object* v___x_2615_; 
lean_del_object(v___x_2600_);
v___x_2613_ = ((size_t)0ULL);
v___x_2614_ = lean_usize_of_nat(v___x_2603_);
v___x_2615_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2585_, v_tail_2597_, v___x_2613_, v___x_2614_, v___x_2604_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_, v___y_2594_);
lean_dec_ref(v_tail_2597_);
return v___x_2615_;
}
}
else
{
size_t v___x_2616_; size_t v___x_2617_; lean_object* v___x_2618_; 
lean_del_object(v___x_2600_);
v___x_2616_ = ((size_t)0ULL);
v___x_2617_ = lean_usize_of_nat(v___x_2603_);
v___x_2618_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2585_, v_tail_2597_, v___x_2616_, v___x_2617_, v___x_2604_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_, v___y_2594_);
lean_dec_ref(v_tail_2597_);
return v___x_2618_;
}
}
}
}
else
{
lean_dec_ref(v_tail_2597_);
return v___x_2598_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16___boxed(lean_object* v_t_2621_, lean_object* v_t_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
lean_object* v_res_2632_; 
v_res_2632_ = lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16(v_t_2621_, v_t_2622_, v___y_2623_, v___y_2624_, v___y_2625_, v___y_2626_, v___y_2627_, v___y_2628_, v___y_2629_, v___y_2630_);
lean_dec(v___y_2630_);
lean_dec_ref(v___y_2629_);
lean_dec(v___y_2628_);
lean_dec_ref(v___y_2627_);
lean_dec(v___y_2626_);
lean_dec_ref(v___y_2625_);
lean_dec(v___y_2624_);
lean_dec_ref(v___y_2623_);
lean_dec_ref(v_t_2621_);
return v_res_2632_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0(void){
_start:
{
lean_object* v___x_2633_; 
v___x_2633_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_2633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14(lean_object* v_t_2634_, lean_object* v_x_2635_, size_t v_x_2636_, size_t v_x_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_){
_start:
{
if (lean_obj_tag(v_x_2635_) == 0)
{
lean_object* v_cs_2647_; lean_object* v___x_2648_; size_t v___x_2649_; lean_object* v_j_2650_; lean_object* v___x_2651_; size_t v___x_2652_; size_t v___x_2653_; size_t v___x_2654_; size_t v___x_2655_; size_t v___x_2656_; size_t v___x_2657_; lean_object* v___x_2658_; 
v_cs_2647_ = lean_ctor_get(v_x_2635_, 0);
lean_inc_ref(v_cs_2647_);
lean_dec_ref_known(v_x_2635_, 1);
v___x_2648_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___closed__0);
v___x_2649_ = lean_usize_shift_right(v_x_2636_, v_x_2637_);
v_j_2650_ = lean_usize_to_nat(v___x_2649_);
v___x_2651_ = lean_array_get_borrowed(v___x_2648_, v_cs_2647_, v_j_2650_);
v___x_2652_ = ((size_t)1ULL);
v___x_2653_ = lean_usize_shift_left(v___x_2652_, v_x_2637_);
v___x_2654_ = lean_usize_sub(v___x_2653_, v___x_2652_);
v___x_2655_ = lean_usize_land(v_x_2636_, v___x_2654_);
v___x_2656_ = ((size_t)5ULL);
v___x_2657_ = lean_usize_sub(v_x_2637_, v___x_2656_);
lean_inc(v___x_2651_);
v___x_2658_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14(v_t_2634_, v___x_2651_, v___x_2655_, v___x_2657_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
if (lean_obj_tag(v___x_2658_) == 0)
{
lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2680_; 
v_isSharedCheck_2680_ = !lean_is_exclusive(v___x_2658_);
if (v_isSharedCheck_2680_ == 0)
{
lean_object* v_unused_2681_; 
v_unused_2681_ = lean_ctor_get(v___x_2658_, 0);
lean_dec(v_unused_2681_);
v___x_2660_ = v___x_2658_;
v_isShared_2661_ = v_isSharedCheck_2680_;
goto v_resetjp_2659_;
}
else
{
lean_dec(v___x_2658_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2680_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; uint8_t v___x_2666_; 
v___x_2662_ = lean_unsigned_to_nat(1u);
v___x_2663_ = lean_nat_add(v_j_2650_, v___x_2662_);
lean_dec(v_j_2650_);
v___x_2664_ = lean_array_get_size(v_cs_2647_);
v___x_2665_ = lean_box(0);
v___x_2666_ = lean_nat_dec_lt(v___x_2663_, v___x_2664_);
if (v___x_2666_ == 0)
{
lean_object* v___x_2668_; 
lean_dec(v___x_2663_);
lean_dec_ref(v_cs_2647_);
if (v_isShared_2661_ == 0)
{
lean_ctor_set(v___x_2660_, 0, v___x_2665_);
v___x_2668_ = v___x_2660_;
goto v_reusejp_2667_;
}
else
{
lean_object* v_reuseFailAlloc_2669_; 
v_reuseFailAlloc_2669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2669_, 0, v___x_2665_);
v___x_2668_ = v_reuseFailAlloc_2669_;
goto v_reusejp_2667_;
}
v_reusejp_2667_:
{
return v___x_2668_;
}
}
else
{
uint8_t v___x_2670_; 
v___x_2670_ = lean_nat_dec_le(v___x_2664_, v___x_2664_);
if (v___x_2670_ == 0)
{
if (v___x_2666_ == 0)
{
lean_object* v___x_2672_; 
lean_dec(v___x_2663_);
lean_dec_ref(v_cs_2647_);
if (v_isShared_2661_ == 0)
{
lean_ctor_set(v___x_2660_, 0, v___x_2665_);
v___x_2672_ = v___x_2660_;
goto v_reusejp_2671_;
}
else
{
lean_object* v_reuseFailAlloc_2673_; 
v_reuseFailAlloc_2673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2673_, 0, v___x_2665_);
v___x_2672_ = v_reuseFailAlloc_2673_;
goto v_reusejp_2671_;
}
v_reusejp_2671_:
{
return v___x_2672_;
}
}
else
{
size_t v___x_2674_; size_t v___x_2675_; lean_object* v___x_2676_; 
lean_del_object(v___x_2660_);
v___x_2674_ = lean_usize_of_nat(v___x_2663_);
lean_dec(v___x_2663_);
v___x_2675_ = lean_usize_of_nat(v___x_2664_);
v___x_2676_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(v_t_2634_, v_cs_2647_, v___x_2674_, v___x_2675_, v___x_2665_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
lean_dec_ref(v_cs_2647_);
return v___x_2676_;
}
}
else
{
size_t v___x_2677_; size_t v___x_2678_; lean_object* v___x_2679_; 
lean_del_object(v___x_2660_);
v___x_2677_ = lean_usize_of_nat(v___x_2663_);
lean_dec(v___x_2663_);
v___x_2678_ = lean_usize_of_nat(v___x_2664_);
v___x_2679_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14_spec__19(v_t_2634_, v_cs_2647_, v___x_2677_, v___x_2678_, v___x_2665_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
lean_dec_ref(v_cs_2647_);
return v___x_2679_;
}
}
}
}
else
{
lean_dec(v_j_2650_);
lean_dec_ref(v_cs_2647_);
return v___x_2658_;
}
}
else
{
lean_object* v_vs_2682_; lean_object* v___x_2684_; uint8_t v_isShared_2685_; uint8_t v_isSharedCheck_2703_; 
v_vs_2682_ = lean_ctor_get(v_x_2635_, 0);
v_isSharedCheck_2703_ = !lean_is_exclusive(v_x_2635_);
if (v_isSharedCheck_2703_ == 0)
{
v___x_2684_ = v_x_2635_;
v_isShared_2685_ = v_isSharedCheck_2703_;
goto v_resetjp_2683_;
}
else
{
lean_inc(v_vs_2682_);
lean_dec(v_x_2635_);
v___x_2684_ = lean_box(0);
v_isShared_2685_ = v_isSharedCheck_2703_;
goto v_resetjp_2683_;
}
v_resetjp_2683_:
{
lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; uint8_t v___x_2689_; 
v___x_2686_ = lean_usize_to_nat(v_x_2636_);
v___x_2687_ = lean_array_get_size(v_vs_2682_);
v___x_2688_ = lean_box(0);
v___x_2689_ = lean_nat_dec_lt(v___x_2686_, v___x_2687_);
if (v___x_2689_ == 0)
{
lean_object* v___x_2691_; 
lean_dec(v___x_2686_);
lean_dec_ref(v_vs_2682_);
if (v_isShared_2685_ == 0)
{
lean_ctor_set_tag(v___x_2684_, 0);
lean_ctor_set(v___x_2684_, 0, v___x_2688_);
v___x_2691_ = v___x_2684_;
goto v_reusejp_2690_;
}
else
{
lean_object* v_reuseFailAlloc_2692_; 
v_reuseFailAlloc_2692_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2692_, 0, v___x_2688_);
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
uint8_t v___x_2693_; 
v___x_2693_ = lean_nat_dec_le(v___x_2687_, v___x_2687_);
if (v___x_2693_ == 0)
{
if (v___x_2689_ == 0)
{
lean_object* v___x_2695_; 
lean_dec(v___x_2686_);
lean_dec_ref(v_vs_2682_);
if (v_isShared_2685_ == 0)
{
lean_ctor_set_tag(v___x_2684_, 0);
lean_ctor_set(v___x_2684_, 0, v___x_2688_);
v___x_2695_ = v___x_2684_;
goto v_reusejp_2694_;
}
else
{
lean_object* v_reuseFailAlloc_2696_; 
v_reuseFailAlloc_2696_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2696_, 0, v___x_2688_);
v___x_2695_ = v_reuseFailAlloc_2696_;
goto v_reusejp_2694_;
}
v_reusejp_2694_:
{
return v___x_2695_;
}
}
else
{
size_t v___x_2697_; size_t v___x_2698_; lean_object* v___x_2699_; 
lean_del_object(v___x_2684_);
v___x_2697_ = lean_usize_of_nat(v___x_2686_);
lean_dec(v___x_2686_);
v___x_2698_ = lean_usize_of_nat(v___x_2687_);
v___x_2699_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2634_, v_vs_2682_, v___x_2697_, v___x_2698_, v___x_2688_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
lean_dec_ref(v_vs_2682_);
return v___x_2699_;
}
}
else
{
size_t v___x_2700_; size_t v___x_2701_; lean_object* v___x_2702_; 
lean_del_object(v___x_2684_);
v___x_2700_ = lean_usize_of_nat(v___x_2686_);
lean_dec(v___x_2686_);
v___x_2701_ = lean_usize_of_nat(v___x_2687_);
v___x_2702_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2634_, v_vs_2682_, v___x_2700_, v___x_2701_, v___x_2688_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_);
lean_dec_ref(v_vs_2682_);
return v___x_2702_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14___boxed(lean_object* v_t_2704_, lean_object* v_x_2705_, lean_object* v_x_2706_, lean_object* v_x_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_, lean_object* v___y_2714_, lean_object* v___y_2715_, lean_object* v___y_2716_){
_start:
{
size_t v_x_46933__boxed_2717_; size_t v_x_46934__boxed_2718_; lean_object* v_res_2719_; 
v_x_46933__boxed_2717_ = lean_unbox_usize(v_x_2706_);
lean_dec(v_x_2706_);
v_x_46934__boxed_2718_ = lean_unbox_usize(v_x_2707_);
lean_dec(v_x_2707_);
v_res_2719_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14(v_t_2704_, v_x_2705_, v_x_46933__boxed_2717_, v_x_46934__boxed_2718_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_, v___y_2714_, v___y_2715_);
lean_dec(v___y_2715_);
lean_dec_ref(v___y_2714_);
lean_dec(v___y_2713_);
lean_dec_ref(v___y_2712_);
lean_dec(v___y_2711_);
lean_dec_ref(v___y_2710_);
lean_dec(v___y_2709_);
lean_dec_ref(v___y_2708_);
lean_dec_ref(v_t_2704_);
return v_res_2719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11(lean_object* v_t_2720_, lean_object* v_t_2721_, lean_object* v_start_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_){
_start:
{
lean_object* v___x_2732_; uint8_t v___x_2733_; 
v___x_2732_ = lean_unsigned_to_nat(0u);
v___x_2733_ = lean_nat_dec_eq(v_start_2722_, v___x_2732_);
if (v___x_2733_ == 0)
{
lean_object* v_root_2734_; lean_object* v_tail_2735_; size_t v_shift_2736_; lean_object* v_tailOff_2737_; uint8_t v___x_2738_; 
v_root_2734_ = lean_ctor_get(v_t_2721_, 0);
lean_inc_ref(v_root_2734_);
v_tail_2735_ = lean_ctor_get(v_t_2721_, 1);
lean_inc_ref(v_tail_2735_);
v_shift_2736_ = lean_ctor_get_usize(v_t_2721_, 4);
v_tailOff_2737_ = lean_ctor_get(v_t_2721_, 3);
lean_inc(v_tailOff_2737_);
lean_dec_ref(v_t_2721_);
v___x_2738_ = lean_nat_dec_le(v_tailOff_2737_, v_start_2722_);
if (v___x_2738_ == 0)
{
size_t v___x_2739_; lean_object* v___x_2740_; 
lean_dec(v_tailOff_2737_);
v___x_2739_ = lean_usize_of_nat(v_start_2722_);
v___x_2740_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_forFromMAux___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__14(v_t_2720_, v_root_2734_, v___x_2739_, v_shift_2736_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
if (lean_obj_tag(v___x_2740_) == 0)
{
lean_object* v___x_2742_; uint8_t v_isShared_2743_; uint8_t v_isSharedCheck_2760_; 
v_isSharedCheck_2760_ = !lean_is_exclusive(v___x_2740_);
if (v_isSharedCheck_2760_ == 0)
{
lean_object* v_unused_2761_; 
v_unused_2761_ = lean_ctor_get(v___x_2740_, 0);
lean_dec(v_unused_2761_);
v___x_2742_ = v___x_2740_;
v_isShared_2743_ = v_isSharedCheck_2760_;
goto v_resetjp_2741_;
}
else
{
lean_dec(v___x_2740_);
v___x_2742_ = lean_box(0);
v_isShared_2743_ = v_isSharedCheck_2760_;
goto v_resetjp_2741_;
}
v_resetjp_2741_:
{
lean_object* v___x_2744_; lean_object* v___x_2745_; uint8_t v___x_2746_; 
v___x_2744_ = lean_array_get_size(v_tail_2735_);
v___x_2745_ = lean_box(0);
v___x_2746_ = lean_nat_dec_lt(v___x_2732_, v___x_2744_);
if (v___x_2746_ == 0)
{
lean_object* v___x_2748_; 
lean_dec_ref(v_tail_2735_);
if (v_isShared_2743_ == 0)
{
lean_ctor_set(v___x_2742_, 0, v___x_2745_);
v___x_2748_ = v___x_2742_;
goto v_reusejp_2747_;
}
else
{
lean_object* v_reuseFailAlloc_2749_; 
v_reuseFailAlloc_2749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2749_, 0, v___x_2745_);
v___x_2748_ = v_reuseFailAlloc_2749_;
goto v_reusejp_2747_;
}
v_reusejp_2747_:
{
return v___x_2748_;
}
}
else
{
uint8_t v___x_2750_; 
v___x_2750_ = lean_nat_dec_le(v___x_2744_, v___x_2744_);
if (v___x_2750_ == 0)
{
if (v___x_2746_ == 0)
{
lean_object* v___x_2752_; 
lean_dec_ref(v_tail_2735_);
if (v_isShared_2743_ == 0)
{
lean_ctor_set(v___x_2742_, 0, v___x_2745_);
v___x_2752_ = v___x_2742_;
goto v_reusejp_2751_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v___x_2745_);
v___x_2752_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2751_;
}
v_reusejp_2751_:
{
return v___x_2752_;
}
}
else
{
size_t v___x_2754_; size_t v___x_2755_; lean_object* v___x_2756_; 
lean_del_object(v___x_2742_);
v___x_2754_ = ((size_t)0ULL);
v___x_2755_ = lean_usize_of_nat(v___x_2744_);
v___x_2756_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2720_, v_tail_2735_, v___x_2754_, v___x_2755_, v___x_2745_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
lean_dec_ref(v_tail_2735_);
return v___x_2756_;
}
}
else
{
size_t v___x_2757_; size_t v___x_2758_; lean_object* v___x_2759_; 
lean_del_object(v___x_2742_);
v___x_2757_ = ((size_t)0ULL);
v___x_2758_ = lean_usize_of_nat(v___x_2744_);
v___x_2759_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2720_, v_tail_2735_, v___x_2757_, v___x_2758_, v___x_2745_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
lean_dec_ref(v_tail_2735_);
return v___x_2759_;
}
}
}
}
else
{
lean_dec_ref(v_tail_2735_);
return v___x_2740_;
}
}
else
{
lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; uint8_t v___x_2765_; 
lean_dec_ref(v_root_2734_);
v___x_2762_ = lean_nat_sub(v_start_2722_, v_tailOff_2737_);
lean_dec(v_tailOff_2737_);
v___x_2763_ = lean_array_get_size(v_tail_2735_);
v___x_2764_ = lean_box(0);
v___x_2765_ = lean_nat_dec_lt(v___x_2762_, v___x_2763_);
if (v___x_2765_ == 0)
{
lean_object* v___x_2766_; 
lean_dec(v___x_2762_);
lean_dec_ref(v_tail_2735_);
v___x_2766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2766_, 0, v___x_2764_);
return v___x_2766_;
}
else
{
uint8_t v___x_2767_; 
v___x_2767_ = lean_nat_dec_le(v___x_2763_, v___x_2763_);
if (v___x_2767_ == 0)
{
if (v___x_2765_ == 0)
{
lean_object* v___x_2768_; 
lean_dec(v___x_2762_);
lean_dec_ref(v_tail_2735_);
v___x_2768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2768_, 0, v___x_2764_);
return v___x_2768_;
}
else
{
size_t v___x_2769_; size_t v___x_2770_; lean_object* v___x_2771_; 
v___x_2769_ = lean_usize_of_nat(v___x_2762_);
lean_dec(v___x_2762_);
v___x_2770_ = lean_usize_of_nat(v___x_2763_);
v___x_2771_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2720_, v_tail_2735_, v___x_2769_, v___x_2770_, v___x_2764_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
lean_dec_ref(v_tail_2735_);
return v___x_2771_;
}
}
else
{
size_t v___x_2772_; size_t v___x_2773_; lean_object* v___x_2774_; 
v___x_2772_ = lean_usize_of_nat(v___x_2762_);
lean_dec(v___x_2762_);
v___x_2773_ = lean_usize_of_nat(v___x_2763_);
v___x_2774_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__15(v_t_2720_, v_tail_2735_, v___x_2772_, v___x_2773_, v___x_2764_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
lean_dec_ref(v_tail_2735_);
return v___x_2774_;
}
}
}
}
else
{
lean_object* v___x_2775_; 
v___x_2775_ = lp_mathlib_Lean_PersistentArray_forMFrom0___at___00Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11_spec__16(v_t_2720_, v_t_2721_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
return v___x_2775_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11___boxed(lean_object* v_t_2776_, lean_object* v_t_2777_, lean_object* v_start_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_){
_start:
{
lean_object* v_res_2788_; 
v_res_2788_ = lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11(v_t_2776_, v_t_2777_, v_start_2778_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_, v___y_2785_, v___y_2786_);
lean_dec(v___y_2786_);
lean_dec_ref(v___y_2785_);
lean_dec(v___y_2784_);
lean_dec_ref(v___y_2783_);
lean_dec(v___y_2782_);
lean_dec_ref(v___y_2781_);
lean_dec(v___y_2780_);
lean_dec_ref(v___y_2779_);
lean_dec(v_start_2778_);
lean_dec_ref(v_t_2776_);
return v_res_2788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8(lean_object* v_t_2789_, lean_object* v_lctx_2790_, lean_object* v_start_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_){
_start:
{
lean_object* v_decls_2801_; lean_object* v___x_2802_; 
v_decls_2801_ = lean_ctor_get(v_lctx_2790_, 1);
lean_inc_ref(v_decls_2801_);
lean_dec_ref(v_lctx_2790_);
v___x_2802_ = lp_mathlib_Lean_PersistentArray_forM___at___00Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8_spec__11(v_t_2789_, v_decls_2801_, v_start_2791_, v___y_2792_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_, v___y_2797_, v___y_2798_, v___y_2799_);
return v___x_2802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8___boxed(lean_object* v_t_2803_, lean_object* v_lctx_2804_, lean_object* v_start_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_){
_start:
{
lean_object* v_res_2815_; 
v_res_2815_ = lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8(v_t_2803_, v_lctx_2804_, v_start_2805_, v___y_2806_, v___y_2807_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
lean_dec(v___y_2811_);
lean_dec_ref(v___y_2810_);
lean_dec(v___y_2809_);
lean_dec_ref(v___y_2808_);
lean_dec(v___y_2807_);
lean_dec_ref(v___y_2806_);
lean_dec(v_start_2805_);
lean_dec_ref(v_t_2803_);
return v_res_2815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0(lean_object* v_t_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_){
_start:
{
lean_object* v_lctx_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; 
v_lctx_2826_ = lean_ctor_get(v___y_2821_, 2);
lean_inc_ref(v_lctx_2826_);
v___x_2827_ = lean_unsigned_to_nat(0u);
v___x_2828_ = lp_mathlib_Lean_LocalContext_forM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__8(v_t_2816_, v_lctx_2826_, v___x_2827_, v___y_2817_, v___y_2818_, v___y_2819_, v___y_2820_, v___y_2821_, v___y_2822_, v___y_2823_, v___y_2824_);
lean_dec_ref(v___y_2821_);
return v___x_2828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0___boxed(lean_object* v_t_2829_, lean_object* v___y_2830_, lean_object* v___y_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_){
_start:
{
lean_object* v_res_2839_; 
v_res_2839_ = lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0(v_t_2829_, v___y_2830_, v___y_2831_, v___y_2832_, v___y_2833_, v___y_2834_, v___y_2835_, v___y_2836_, v___y_2837_);
lean_dec(v___y_2837_);
lean_dec_ref(v___y_2836_);
lean_dec(v___y_2835_);
lean_dec(v___y_2833_);
lean_dec_ref(v___y_2832_);
lean_dec(v___y_2831_);
lean_dec_ref(v___y_2830_);
lean_dec_ref(v_t_2829_);
return v_res_2839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties(lean_object* v_t_2840_, lean_object* v_a_2841_, lean_object* v_a_2842_, lean_object* v_a_2843_, lean_object* v_a_2844_, lean_object* v_a_2845_, lean_object* v_a_2846_, lean_object* v_a_2847_, lean_object* v_a_2848_){
_start:
{
lean_object* v___f_2850_; lean_object* v___x_2851_; 
v___f_2850_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___lam__0___boxed), 10, 1);
lean_closure_set(v___f_2850_, 0, v_t_2840_);
v___x_2851_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2850_, v_a_2841_, v_a_2842_, v_a_2843_, v_a_2844_, v_a_2845_, v_a_2846_, v_a_2847_, v_a_2848_);
return v___x_2851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_addProperties___boxed(lean_object* v_t_2852_, lean_object* v_a_2853_, lean_object* v_a_2854_, lean_object* v_a_2855_, lean_object* v_a_2856_, lean_object* v_a_2857_, lean_object* v_a_2858_, lean_object* v_a_2859_, lean_object* v_a_2860_, lean_object* v_a_2861_){
_start:
{
lean_object* v_res_2862_; 
v_res_2862_ = lp_mathlib_Mathlib_Tactic_Algebraize_addProperties(v_t_2852_, v_a_2853_, v_a_2854_, v_a_2855_, v_a_2856_, v_a_2857_, v_a_2858_, v_a_2859_, v_a_2860_);
lean_dec(v_a_2860_);
lean_dec_ref(v_a_2859_);
lean_dec(v_a_2858_);
lean_dec_ref(v_a_2857_);
lean_dec(v_a_2856_);
lean_dec_ref(v_a_2855_);
lean_dec(v_a_2854_);
lean_dec_ref(v_a_2853_);
return v_res_2862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2(lean_object* v_as_2863_, size_t v_i_2864_, size_t v_stop_2865_, lean_object* v_b_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_, lean_object* v___y_2870_, lean_object* v___y_2871_, lean_object* v___y_2872_, lean_object* v___y_2873_, lean_object* v___y_2874_){
_start:
{
lean_object* v___x_2876_; 
v___x_2876_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___redArg(v_as_2863_, v_i_2864_, v_stop_2865_, v_b_2866_, v___y_2871_, v___y_2872_, v___y_2873_, v___y_2874_);
return v___x_2876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2___boxed(lean_object* v_as_2877_, lean_object* v_i_2878_, lean_object* v_stop_2879_, lean_object* v_b_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_, lean_object* v___y_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_){
_start:
{
size_t v_i_boxed_2890_; size_t v_stop_boxed_2891_; lean_object* v_res_2892_; 
v_i_boxed_2890_ = lean_unbox_usize(v_i_2878_);
lean_dec(v_i_2878_);
v_stop_boxed_2891_ = lean_unbox_usize(v_stop_2879_);
lean_dec(v_stop_2879_);
v_res_2892_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Algebraize_addProperties_spec__2_spec__2(v_as_2877_, v_i_boxed_2890_, v_stop_boxed_2891_, v_b_2880_, v___y_2881_, v___y_2882_, v___y_2883_, v___y_2884_, v___y_2885_, v___y_2886_, v___y_2887_, v___y_2888_);
lean_dec(v___y_2888_);
lean_dec_ref(v___y_2887_);
lean_dec(v___y_2886_);
lean_dec_ref(v___y_2885_);
lean_dec(v___y_2884_);
lean_dec_ref(v___y_2883_);
lean_dec(v___y_2882_);
lean_dec_ref(v___y_2881_);
lean_dec_ref(v_as_2877_);
return v_res_2892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6(lean_object* v_00_u03b1_2893_, lean_object* v_constName_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_, lean_object* v___y_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_){
_start:
{
lean_object* v___x_2904_; 
v___x_2904_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___redArg(v_constName_2894_, v___y_2895_, v___y_2896_, v___y_2897_, v___y_2898_, v___y_2899_, v___y_2900_, v___y_2901_, v___y_2902_);
return v___x_2904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6___boxed(lean_object* v_00_u03b1_2905_, lean_object* v_constName_2906_, lean_object* v___y_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_, lean_object* v___y_2914_, lean_object* v___y_2915_){
_start:
{
lean_object* v_res_2916_; 
v_res_2916_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6(v_00_u03b1_2905_, v_constName_2906_, v___y_2907_, v___y_2908_, v___y_2909_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_, v___y_2914_);
lean_dec(v___y_2914_);
lean_dec_ref(v___y_2913_);
lean_dec(v___y_2912_);
lean_dec_ref(v___y_2911_);
lean_dec(v___y_2910_);
lean_dec_ref(v___y_2909_);
lean_dec(v___y_2908_);
lean_dec_ref(v___y_2907_);
return v_res_2916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8(lean_object* v_00_u03b1_2917_, lean_object* v_ref_2918_, lean_object* v_constName_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_){
_start:
{
lean_object* v___x_2929_; 
v___x_2929_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg(v_ref_2918_, v_constName_2919_, v___y_2920_, v___y_2921_, v___y_2922_, v___y_2923_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_);
return v___x_2929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___boxed(lean_object* v_00_u03b1_2930_, lean_object* v_ref_2931_, lean_object* v_constName_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_, lean_object* v___y_2935_, lean_object* v___y_2936_, lean_object* v___y_2937_, lean_object* v___y_2938_, lean_object* v___y_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_){
_start:
{
lean_object* v_res_2942_; 
v_res_2942_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8(v_00_u03b1_2930_, v_ref_2931_, v_constName_2932_, v___y_2933_, v___y_2934_, v___y_2935_, v___y_2936_, v___y_2937_, v___y_2938_, v___y_2939_, v___y_2940_);
lean_dec(v___y_2940_);
lean_dec_ref(v___y_2939_);
lean_dec(v___y_2938_);
lean_dec_ref(v___y_2937_);
lean_dec(v___y_2936_);
lean_dec_ref(v___y_2935_);
lean_dec(v___y_2934_);
lean_dec_ref(v___y_2933_);
lean_dec(v_ref_2931_);
return v_res_2942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11(lean_object* v_ref_2943_, lean_object* v_msgData_2944_, uint8_t v_severity_2945_, uint8_t v_isSilent_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_){
_start:
{
lean_object* v___x_2956_; 
v___x_2956_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(v_ref_2943_, v_msgData_2944_, v_severity_2945_, v_isSilent_2946_, v___y_2951_, v___y_2952_, v___y_2953_, v___y_2954_);
return v___x_2956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___boxed(lean_object* v_ref_2957_, lean_object* v_msgData_2958_, lean_object* v_severity_2959_, lean_object* v_isSilent_2960_, lean_object* v___y_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_, lean_object* v___y_2967_, lean_object* v___y_2968_, lean_object* v___y_2969_){
_start:
{
uint8_t v_severity_boxed_2970_; uint8_t v_isSilent_boxed_2971_; lean_object* v_res_2972_; 
v_severity_boxed_2970_ = lean_unbox(v_severity_2959_);
v_isSilent_boxed_2971_ = lean_unbox(v_isSilent_2960_);
v_res_2972_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11(v_ref_2957_, v_msgData_2958_, v_severity_boxed_2970_, v_isSilent_boxed_2971_, v___y_2961_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_, v___y_2966_, v___y_2967_, v___y_2968_);
lean_dec(v___y_2968_);
lean_dec_ref(v___y_2967_);
lean_dec(v___y_2966_);
lean_dec_ref(v___y_2965_);
lean_dec(v___y_2964_);
lean_dec_ref(v___y_2963_);
lean_dec(v___y_2962_);
lean_dec_ref(v___y_2961_);
lean_dec(v_ref_2957_);
return v_res_2972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11(lean_object* v_00_u03b1_2973_, lean_object* v_ref_2974_, lean_object* v_msg_2975_, lean_object* v_declHint_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_){
_start:
{
lean_object* v___x_2986_; 
v___x_2986_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___redArg(v_ref_2974_, v_msg_2975_, v_declHint_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_, v___y_2983_, v___y_2984_);
return v___x_2986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11___boxed(lean_object* v_00_u03b1_2987_, lean_object* v_ref_2988_, lean_object* v_msg_2989_, lean_object* v_declHint_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_){
_start:
{
lean_object* v_res_3000_; 
v_res_3000_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11(v_00_u03b1_2987_, v_ref_2988_, v_msg_2989_, v_declHint_2990_, v___y_2991_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_, v___y_2998_);
lean_dec(v___y_2998_);
lean_dec_ref(v___y_2997_);
lean_dec(v___y_2996_);
lean_dec_ref(v___y_2995_);
lean_dec(v___y_2994_);
lean_dec_ref(v___y_2993_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
lean_dec(v_ref_2988_);
return v_res_3000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21(lean_object* v_msg_3001_, lean_object* v_declHint_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_){
_start:
{
lean_object* v___x_3012_; 
v___x_3012_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___redArg(v_msg_3001_, v_declHint_3002_, v___y_3010_);
return v___x_3012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21___boxed(lean_object* v_msg_3013_, lean_object* v_declHint_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_){
_start:
{
lean_object* v_res_3024_; 
v_res_3024_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__14_spec__21(v_msg_3013_, v_declHint_3014_, v___y_3015_, v___y_3016_, v___y_3017_, v___y_3018_, v___y_3019_, v___y_3020_, v___y_3021_, v___y_3022_);
lean_dec(v___y_3022_);
lean_dec_ref(v___y_3021_);
lean_dec(v___y_3020_);
lean_dec_ref(v___y_3019_);
lean_dec(v___y_3018_);
lean_dec_ref(v___y_3017_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
return v_res_3024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15(lean_object* v_00_u03b1_3025_, lean_object* v_ref_3026_, lean_object* v_msg_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_){
_start:
{
lean_object* v___x_3037_; 
v___x_3037_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___redArg(v_ref_3026_, v_msg_3027_, v___y_3028_, v___y_3029_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
return v___x_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15___boxed(lean_object* v_00_u03b1_3038_, lean_object* v_ref_3039_, lean_object* v_msg_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_, lean_object* v___y_3045_, lean_object* v___y_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_, lean_object* v___y_3049_){
_start:
{
lean_object* v_res_3050_; 
v_res_3050_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15(v_00_u03b1_3038_, v_ref_3039_, v_msg_3040_, v___y_3041_, v___y_3042_, v___y_3043_, v___y_3044_, v___y_3045_, v___y_3046_, v___y_3047_, v___y_3048_);
lean_dec(v___y_3048_);
lean_dec_ref(v___y_3047_);
lean_dec(v___y_3046_);
lean_dec_ref(v___y_3045_);
lean_dec(v___y_3044_);
lean_dec_ref(v___y_3043_);
lean_dec(v___y_3042_);
lean_dec_ref(v___y_3041_);
lean_dec(v_ref_3039_);
return v_res_3050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23(lean_object* v_00_u03b1_3051_, lean_object* v_msg_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_, lean_object* v___y_3058_, lean_object* v___y_3059_, lean_object* v___y_3060_){
_start:
{
lean_object* v___x_3062_; 
v___x_3062_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v_msg_3052_, v___y_3057_, v___y_3058_, v___y_3059_, v___y_3060_);
return v___x_3062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___boxed(lean_object* v_00_u03b1_3063_, lean_object* v_msg_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_){
_start:
{
lean_object* v_res_3074_; 
v_res_3074_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23(v_00_u03b1_3063_, v_msg_3064_, v___y_3065_, v___y_3066_, v___y_3067_, v___y_3068_, v___y_3069_, v___y_3070_, v___y_3071_, v___y_3072_);
lean_dec(v___y_3072_);
lean_dec_ref(v___y_3071_);
lean_dec(v___y_3070_);
lean_dec_ref(v___y_3069_);
lean_dec(v___y_3068_);
lean_dec_ref(v___y_3067_);
lean_dec(v___y_3066_);
lean_dec_ref(v___y_3065_);
return v_res_3074_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig_default(void){
_start:
{
uint8_t v___x_3075_; 
v___x_3075_ = 1;
return v___x_3075_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig(void){
_start:
{
uint8_t v___x_3076_; 
v___x_3076_ = 1;
return v___x_3076_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v___x_3079_; 
v___x_3077_ = lean_box(0);
v___x_3078_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_3079_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3079_, 0, v___x_3078_);
lean_ctor_set(v___x_3079_, 1, v___x_3077_);
return v___x_3079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_3081_; lean_object* v___x_3082_; 
v___x_3081_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_3082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3082_, 0, v___x_3081_);
return v___x_3082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_3083_){
_start:
{
lean_object* v_res_3084_; 
v_res_3084_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_3084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_){
_start:
{
lean_object* v___x_3091_; 
v___x_3091_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_3091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_){
_start:
{
lean_object* v_res_3098_; 
v_res_3098_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_3092_, v___y_3093_, v___y_3094_, v___y_3095_, v___y_3096_);
lean_dec(v___y_3096_);
lean_dec_ref(v___y_3095_);
lean_dec(v___y_3094_);
lean_dec_ref(v___y_3093_);
return v_res_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_){
_start:
{
lean_object* v_ref_3105_; lean_object* v___x_3106_; lean_object* v_a_3107_; lean_object* v___x_3109_; uint8_t v_isShared_3110_; uint8_t v_isSharedCheck_3115_; 
v_ref_3105_ = lean_ctor_get(v___y_3102_, 5);
v___x_3106_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(v_msg_3099_, v___y_3100_, v___y_3101_, v___y_3102_, v___y_3103_);
v_a_3107_ = lean_ctor_get(v___x_3106_, 0);
v_isSharedCheck_3115_ = !lean_is_exclusive(v___x_3106_);
if (v_isSharedCheck_3115_ == 0)
{
v___x_3109_ = v___x_3106_;
v_isShared_3110_ = v_isSharedCheck_3115_;
goto v_resetjp_3108_;
}
else
{
lean_inc(v_a_3107_);
lean_dec(v___x_3106_);
v___x_3109_ = lean_box(0);
v_isShared_3110_ = v_isSharedCheck_3115_;
goto v_resetjp_3108_;
}
v_resetjp_3108_:
{
lean_object* v___x_3111_; lean_object* v___x_3113_; 
lean_inc(v_ref_3105_);
v___x_3111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3111_, 0, v_ref_3105_);
lean_ctor_set(v___x_3111_, 1, v_a_3107_);
if (v_isShared_3110_ == 0)
{
lean_ctor_set_tag(v___x_3109_, 1);
lean_ctor_set(v___x_3109_, 0, v___x_3111_);
v___x_3113_ = v___x_3109_;
goto v_reusejp_3112_;
}
else
{
lean_object* v_reuseFailAlloc_3114_; 
v_reuseFailAlloc_3114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3114_, 0, v___x_3111_);
v___x_3113_ = v_reuseFailAlloc_3114_;
goto v_reusejp_3112_;
}
v_reusejp_3112_:
{
return v___x_3113_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_){
_start:
{
lean_object* v_res_3122_; 
v_res_3122_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_3116_, v___y_3117_, v___y_3118_, v___y_3119_, v___y_3120_);
lean_dec(v___y_3120_);
lean_dec_ref(v___y_3119_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
return v_res_3122_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_3125_; lean_object* v___x_3126_; 
v___x_3125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_3126_ = l_Lean_stringToMessageData(v___x_3125_);
return v___x_3126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_3127_, lean_object* v_args_3128_, lean_object* v___y_3129_, lean_object* v___y_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_){
_start:
{
lean_object* v___x_3155_; uint8_t v___x_3156_; 
v___x_3155_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_3156_ = lean_string_dec_eq(v_ctor_3127_, v___x_3155_);
if (v___x_3156_ == 0)
{
lean_object* v___x_3157_; 
v___x_3157_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_3157_;
}
else
{
lean_object* v___x_3158_; lean_object* v___x_3159_; uint8_t v___x_3160_; 
v___x_3158_ = lean_array_get_size(v_args_3128_);
v___x_3159_ = lean_unsigned_to_nat(1u);
v___x_3160_ = lean_nat_dec_eq(v___x_3158_, v___x_3159_);
if (v___x_3160_ == 0)
{
lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v_a_3163_; lean_object* v___x_3165_; uint8_t v_isShared_3166_; uint8_t v_isSharedCheck_3170_; 
v___x_3161_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_3162_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_3161_, v___y_3129_, v___y_3130_, v___y_3131_, v___y_3132_);
v_a_3163_ = lean_ctor_get(v___x_3162_, 0);
v_isSharedCheck_3170_ = !lean_is_exclusive(v___x_3162_);
if (v_isSharedCheck_3170_ == 0)
{
v___x_3165_ = v___x_3162_;
v_isShared_3166_ = v_isSharedCheck_3170_;
goto v_resetjp_3164_;
}
else
{
lean_inc(v_a_3163_);
lean_dec(v___x_3162_);
v___x_3165_ = lean_box(0);
v_isShared_3166_ = v_isSharedCheck_3170_;
goto v_resetjp_3164_;
}
v_resetjp_3164_:
{
lean_object* v___x_3168_; 
if (v_isShared_3166_ == 0)
{
v___x_3168_ = v___x_3165_;
goto v_reusejp_3167_;
}
else
{
lean_object* v_reuseFailAlloc_3169_; 
v_reuseFailAlloc_3169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3169_, 0, v_a_3163_);
v___x_3168_ = v_reuseFailAlloc_3169_;
goto v_reusejp_3167_;
}
v_reusejp_3167_:
{
return v___x_3168_;
}
}
}
else
{
goto v___jp_3134_;
}
}
v___jp_3134_:
{
lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; lean_object* v___x_3138_; 
v___x_3135_ = l_Lean_instInhabitedExpr;
v___x_3136_ = lean_unsigned_to_nat(0u);
v___x_3137_ = lean_array_get_borrowed(v___x_3135_, v_args_3128_, v___x_3136_);
lean_inc(v___x_3137_);
v___x_3138_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_3137_, v___y_3129_, v___y_3130_, v___y_3131_, v___y_3132_);
if (lean_obj_tag(v___x_3138_) == 0)
{
lean_object* v_a_3139_; lean_object* v___x_3141_; uint8_t v_isShared_3142_; uint8_t v_isSharedCheck_3146_; 
v_a_3139_ = lean_ctor_get(v___x_3138_, 0);
v_isSharedCheck_3146_ = !lean_is_exclusive(v___x_3138_);
if (v_isSharedCheck_3146_ == 0)
{
v___x_3141_ = v___x_3138_;
v_isShared_3142_ = v_isSharedCheck_3146_;
goto v_resetjp_3140_;
}
else
{
lean_inc(v_a_3139_);
lean_dec(v___x_3138_);
v___x_3141_ = lean_box(0);
v_isShared_3142_ = v_isSharedCheck_3146_;
goto v_resetjp_3140_;
}
v_resetjp_3140_:
{
lean_object* v___x_3144_; 
if (v_isShared_3142_ == 0)
{
v___x_3144_ = v___x_3141_;
goto v_reusejp_3143_;
}
else
{
lean_object* v_reuseFailAlloc_3145_; 
v_reuseFailAlloc_3145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3145_, 0, v_a_3139_);
v___x_3144_ = v_reuseFailAlloc_3145_;
goto v_reusejp_3143_;
}
v_reusejp_3143_:
{
return v___x_3144_;
}
}
}
else
{
lean_object* v_a_3147_; lean_object* v___x_3149_; uint8_t v_isShared_3150_; uint8_t v_isSharedCheck_3154_; 
v_a_3147_ = lean_ctor_get(v___x_3138_, 0);
v_isSharedCheck_3154_ = !lean_is_exclusive(v___x_3138_);
if (v_isSharedCheck_3154_ == 0)
{
v___x_3149_ = v___x_3138_;
v_isShared_3150_ = v_isSharedCheck_3154_;
goto v_resetjp_3148_;
}
else
{
lean_inc(v_a_3147_);
lean_dec(v___x_3138_);
v___x_3149_ = lean_box(0);
v_isShared_3150_ = v_isSharedCheck_3154_;
goto v_resetjp_3148_;
}
v_resetjp_3148_:
{
lean_object* v___x_3152_; 
if (v_isShared_3150_ == 0)
{
v___x_3152_ = v___x_3149_;
goto v_reusejp_3151_;
}
else
{
lean_object* v_reuseFailAlloc_3153_; 
v_reuseFailAlloc_3153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3153_, 0, v_a_3147_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_3171_, lean_object* v_args_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_, lean_object* v___y_3176_, lean_object* v___y_3177_){
_start:
{
lean_object* v_res_3178_; 
v_res_3178_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___lam__0(v_ctor_3171_, v_args_3172_, v___y_3173_, v___y_3174_, v___y_3175_, v___y_3176_);
lean_dec(v___y_3176_);
lean_dec_ref(v___y_3175_);
lean_dec(v___y_3174_);
lean_dec_ref(v___y_3173_);
lean_dec_ref(v_args_3172_);
lean_dec_ref(v_ctor_3171_);
return v_res_3178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr(lean_object* v_a_3188_, lean_object* v_a_3189_, lean_object* v_a_3190_, lean_object* v_a_3191_, lean_object* v_a_3192_){
_start:
{
lean_object* v___f_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; 
v___f_3194_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__0));
v___x_3195_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4));
v___x_3196_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_3195_, v___f_3194_, v_a_3188_, v_a_3189_, v_a_3190_, v_a_3191_, v_a_3192_);
return v___x_3196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_3197_, lean_object* v_a_3198_, lean_object* v_a_3199_, lean_object* v_a_3200_, lean_object* v_a_3201_, lean_object* v_a_3202_){
_start:
{
lean_object* v_res_3203_; 
v_res_3203_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr(v_a_3197_, v_a_3198_, v_a_3199_, v_a_3200_, v_a_3201_);
lean_dec(v_a_3201_);
lean_dec_ref(v_a_3200_);
lean_dec(v_a_3199_);
lean_dec_ref(v_a_3198_);
return v_res_3203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_3204_, lean_object* v_msg_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_){
_start:
{
lean_object* v___x_3211_; 
v___x_3211_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_3205_, v___y_3206_, v___y_3207_, v___y_3208_, v___y_3209_);
return v___x_3211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_3212_, lean_object* v_msg_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_){
_start:
{
lean_object* v_res_3219_; 
v_res_3219_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_3212_, v_msg_3213_, v___y_3214_, v___y_3215_, v___y_3216_, v___y_3217_);
lean_dec(v___y_3217_);
lean_dec_ref(v___y_3216_);
lean_dec(v___y_3215_);
lean_dec_ref(v___y_3214_);
return v_res_3219_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_3221_; lean_object* v___x_3222_; lean_object* v___x_3223_; 
v___x_3221_ = lean_box(0);
v___x_3222_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4));
v___x_3223_ = l_Lean_Expr_const___override(v___x_3222_, v___x_3221_);
return v___x_3223_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_3224_; lean_object* v___x_3225_; 
v___x_3224_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1);
v___x_3225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3225_, 0, v___x_3224_);
return v___x_3225_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_3226_; lean_object* v___x_3227_; lean_object* v___x_3228_; 
v___x_3226_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2);
v___x_3227_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__0));
v___x_3228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3228_, 0, v___x_3227_);
lean_ctor_set(v___x_3228_, 1, v___x_3226_);
return v___x_3228_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig(void){
_start:
{
lean_object* v___x_3229_; 
v___x_3229_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__3);
return v___x_3229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object* v_e_3230_, lean_object* v___y_3231_){
_start:
{
uint8_t v___x_3233_; 
v___x_3233_ = l_Lean_Expr_hasMVar(v_e_3230_);
if (v___x_3233_ == 0)
{
lean_object* v___x_3234_; 
v___x_3234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3234_, 0, v_e_3230_);
return v___x_3234_;
}
else
{
lean_object* v___x_3235_; lean_object* v_mctx_3236_; lean_object* v___x_3237_; lean_object* v_fst_3238_; lean_object* v_snd_3239_; lean_object* v___x_3240_; lean_object* v_cache_3241_; lean_object* v_zetaDeltaFVarIds_3242_; lean_object* v_postponed_3243_; lean_object* v_diag_3244_; lean_object* v___x_3246_; uint8_t v_isShared_3247_; uint8_t v_isSharedCheck_3253_; 
v___x_3235_ = lean_st_ref_get(v___y_3231_);
v_mctx_3236_ = lean_ctor_get(v___x_3235_, 0);
lean_inc_ref(v_mctx_3236_);
lean_dec(v___x_3235_);
v___x_3237_ = l_Lean_instantiateMVarsCore(v_mctx_3236_, v_e_3230_);
v_fst_3238_ = lean_ctor_get(v___x_3237_, 0);
lean_inc(v_fst_3238_);
v_snd_3239_ = lean_ctor_get(v___x_3237_, 1);
lean_inc(v_snd_3239_);
lean_dec_ref(v___x_3237_);
v___x_3240_ = lean_st_ref_take(v___y_3231_);
v_cache_3241_ = lean_ctor_get(v___x_3240_, 1);
v_zetaDeltaFVarIds_3242_ = lean_ctor_get(v___x_3240_, 2);
v_postponed_3243_ = lean_ctor_get(v___x_3240_, 3);
v_diag_3244_ = lean_ctor_get(v___x_3240_, 4);
v_isSharedCheck_3253_ = !lean_is_exclusive(v___x_3240_);
if (v_isSharedCheck_3253_ == 0)
{
lean_object* v_unused_3254_; 
v_unused_3254_ = lean_ctor_get(v___x_3240_, 0);
lean_dec(v_unused_3254_);
v___x_3246_ = v___x_3240_;
v_isShared_3247_ = v_isSharedCheck_3253_;
goto v_resetjp_3245_;
}
else
{
lean_inc(v_diag_3244_);
lean_inc(v_postponed_3243_);
lean_inc(v_zetaDeltaFVarIds_3242_);
lean_inc(v_cache_3241_);
lean_dec(v___x_3240_);
v___x_3246_ = lean_box(0);
v_isShared_3247_ = v_isSharedCheck_3253_;
goto v_resetjp_3245_;
}
v_resetjp_3245_:
{
lean_object* v___x_3249_; 
if (v_isShared_3247_ == 0)
{
lean_ctor_set(v___x_3246_, 0, v_snd_3239_);
v___x_3249_ = v___x_3246_;
goto v_reusejp_3248_;
}
else
{
lean_object* v_reuseFailAlloc_3252_; 
v_reuseFailAlloc_3252_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3252_, 0, v_snd_3239_);
lean_ctor_set(v_reuseFailAlloc_3252_, 1, v_cache_3241_);
lean_ctor_set(v_reuseFailAlloc_3252_, 2, v_zetaDeltaFVarIds_3242_);
lean_ctor_set(v_reuseFailAlloc_3252_, 3, v_postponed_3243_);
lean_ctor_set(v_reuseFailAlloc_3252_, 4, v_diag_3244_);
v___x_3249_ = v_reuseFailAlloc_3252_;
goto v_reusejp_3248_;
}
v_reusejp_3248_:
{
lean_object* v___x_3250_; lean_object* v___x_3251_; 
v___x_3250_ = lean_st_ref_set(v___y_3231_, v___x_3249_);
v___x_3251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3251_, 0, v_fst_3238_);
return v___x_3251_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object* v_e_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_){
_start:
{
lean_object* v_res_3258_; 
v_res_3258_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_3255_, v___y_3256_);
lean_dec(v___y_3256_);
return v_res_3258_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_3259_; lean_object* v___x_3260_; 
v___x_3259_ = lean_box(1);
v___x_3260_ = l_Lean_MessageData_ofFormat(v___x_3259_);
return v___x_3260_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3(void){
_start:
{
lean_object* v___x_3264_; lean_object* v___x_3265_; 
v___x_3264_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__2));
v___x_3265_ = l_Lean_MessageData_ofFormat(v___x_3264_);
return v___x_3265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object* v_x_3266_, lean_object* v_x_3267_){
_start:
{
if (lean_obj_tag(v_x_3267_) == 0)
{
return v_x_3266_;
}
else
{
lean_object* v_head_3268_; lean_object* v_tail_3269_; lean_object* v___x_3271_; uint8_t v_isShared_3272_; uint8_t v_isSharedCheck_3291_; 
v_head_3268_ = lean_ctor_get(v_x_3267_, 0);
v_tail_3269_ = lean_ctor_get(v_x_3267_, 1);
v_isSharedCheck_3291_ = !lean_is_exclusive(v_x_3267_);
if (v_isSharedCheck_3291_ == 0)
{
v___x_3271_ = v_x_3267_;
v_isShared_3272_ = v_isSharedCheck_3291_;
goto v_resetjp_3270_;
}
else
{
lean_inc(v_tail_3269_);
lean_inc(v_head_3268_);
lean_dec(v_x_3267_);
v___x_3271_ = lean_box(0);
v_isShared_3272_ = v_isSharedCheck_3291_;
goto v_resetjp_3270_;
}
v_resetjp_3270_:
{
lean_object* v_before_3273_; lean_object* v___x_3275_; uint8_t v_isShared_3276_; uint8_t v_isSharedCheck_3289_; 
v_before_3273_ = lean_ctor_get(v_head_3268_, 0);
v_isSharedCheck_3289_ = !lean_is_exclusive(v_head_3268_);
if (v_isSharedCheck_3289_ == 0)
{
lean_object* v_unused_3290_; 
v_unused_3290_ = lean_ctor_get(v_head_3268_, 1);
lean_dec(v_unused_3290_);
v___x_3275_ = v_head_3268_;
v_isShared_3276_ = v_isSharedCheck_3289_;
goto v_resetjp_3274_;
}
else
{
lean_inc(v_before_3273_);
lean_dec(v_head_3268_);
v___x_3275_ = lean_box(0);
v_isShared_3276_ = v_isSharedCheck_3289_;
goto v_resetjp_3274_;
}
v_resetjp_3274_:
{
lean_object* v___x_3277_; lean_object* v___x_3279_; 
v___x_3277_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0);
if (v_isShared_3276_ == 0)
{
lean_ctor_set_tag(v___x_3275_, 7);
lean_ctor_set(v___x_3275_, 1, v___x_3277_);
lean_ctor_set(v___x_3275_, 0, v_x_3266_);
v___x_3279_ = v___x_3275_;
goto v_reusejp_3278_;
}
else
{
lean_object* v_reuseFailAlloc_3288_; 
v_reuseFailAlloc_3288_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3288_, 0, v_x_3266_);
lean_ctor_set(v_reuseFailAlloc_3288_, 1, v___x_3277_);
v___x_3279_ = v_reuseFailAlloc_3288_;
goto v_reusejp_3278_;
}
v_reusejp_3278_:
{
lean_object* v___x_3280_; lean_object* v___x_3282_; 
v___x_3280_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__3);
if (v_isShared_3272_ == 0)
{
lean_ctor_set_tag(v___x_3271_, 7);
lean_ctor_set(v___x_3271_, 1, v___x_3280_);
lean_ctor_set(v___x_3271_, 0, v___x_3279_);
v___x_3282_ = v___x_3271_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3287_; 
v_reuseFailAlloc_3287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3287_, 0, v___x_3279_);
lean_ctor_set(v_reuseFailAlloc_3287_, 1, v___x_3280_);
v___x_3282_ = v_reuseFailAlloc_3287_;
goto v_reusejp_3281_;
}
v_reusejp_3281_:
{
lean_object* v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3285_; 
v___x_3283_ = l_Lean_MessageData_ofSyntax(v_before_3273_);
v___x_3284_ = l_Lean_indentD(v___x_3283_);
v___x_3285_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3285_, 0, v___x_3282_);
lean_ctor_set(v___x_3285_, 1, v___x_3284_);
v_x_3266_ = v___x_3285_;
v_x_3267_ = v_tail_3269_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_3295_; lean_object* v___x_3296_; 
v___x_3295_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1));
v___x_3296_ = l_Lean_MessageData_ofFormat(v___x_3295_);
return v___x_3296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object* v_msgData_3297_, lean_object* v_macroStack_3298_, lean_object* v___y_3299_){
_start:
{
lean_object* v_options_3301_; lean_object* v___x_3302_; uint8_t v___x_3303_; 
v_options_3301_ = lean_ctor_get(v___y_3299_, 2);
v___x_3302_ = l_Lean_Elab_pp_macroStack;
v___x_3303_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__15(v_options_3301_, v___x_3302_);
if (v___x_3303_ == 0)
{
lean_object* v___x_3304_; 
lean_dec(v_macroStack_3298_);
v___x_3304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3304_, 0, v_msgData_3297_);
return v___x_3304_;
}
else
{
if (lean_obj_tag(v_macroStack_3298_) == 0)
{
lean_object* v___x_3305_; 
v___x_3305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3305_, 0, v_msgData_3297_);
return v___x_3305_;
}
else
{
lean_object* v_head_3306_; lean_object* v_after_3307_; lean_object* v___x_3309_; uint8_t v_isShared_3310_; uint8_t v_isSharedCheck_3322_; 
v_head_3306_ = lean_ctor_get(v_macroStack_3298_, 0);
lean_inc(v_head_3306_);
v_after_3307_ = lean_ctor_get(v_head_3306_, 1);
v_isSharedCheck_3322_ = !lean_is_exclusive(v_head_3306_);
if (v_isSharedCheck_3322_ == 0)
{
lean_object* v_unused_3323_; 
v_unused_3323_ = lean_ctor_get(v_head_3306_, 0);
lean_dec(v_unused_3323_);
v___x_3309_ = v_head_3306_;
v_isShared_3310_ = v_isSharedCheck_3322_;
goto v_resetjp_3308_;
}
else
{
lean_inc(v_after_3307_);
lean_dec(v_head_3306_);
v___x_3309_ = lean_box(0);
v_isShared_3310_ = v_isSharedCheck_3322_;
goto v_resetjp_3308_;
}
v_resetjp_3308_:
{
lean_object* v___x_3311_; lean_object* v___x_3313_; 
v___x_3311_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___closed__0);
if (v_isShared_3310_ == 0)
{
lean_ctor_set_tag(v___x_3309_, 7);
lean_ctor_set(v___x_3309_, 1, v___x_3311_);
lean_ctor_set(v___x_3309_, 0, v_msgData_3297_);
v___x_3313_ = v___x_3309_;
goto v_reusejp_3312_;
}
else
{
lean_object* v_reuseFailAlloc_3321_; 
v_reuseFailAlloc_3321_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3321_, 0, v_msgData_3297_);
lean_ctor_set(v_reuseFailAlloc_3321_, 1, v___x_3311_);
v___x_3313_ = v_reuseFailAlloc_3321_;
goto v_reusejp_3312_;
}
v_reusejp_3312_:
{
lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v___x_3317_; lean_object* v_msgData_3318_; lean_object* v___x_3319_; lean_object* v___x_3320_; 
v___x_3314_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2);
v___x_3315_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3315_, 0, v___x_3313_);
lean_ctor_set(v___x_3315_, 1, v___x_3314_);
v___x_3316_ = l_Lean_MessageData_ofSyntax(v_after_3307_);
v___x_3317_ = l_Lean_indentD(v___x_3316_);
v_msgData_3318_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_3318_, 0, v___x_3315_);
lean_ctor_set(v_msgData_3318_, 1, v___x_3317_);
v___x_3319_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_msgData_3318_, v_macroStack_3298_);
v___x_3320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3320_, 0, v___x_3319_);
return v___x_3320_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_3324_, lean_object* v_macroStack_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_){
_start:
{
lean_object* v_res_3328_; 
v_res_3328_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_3324_, v_macroStack_3325_, v___y_3326_);
lean_dec_ref(v___y_3326_);
return v_res_3328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object* v_msg_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_){
_start:
{
lean_object* v_ref_3337_; lean_object* v___x_3338_; lean_object* v_a_3339_; lean_object* v_macroStack_3340_; lean_object* v___x_3341_; lean_object* v___x_3342_; lean_object* v_a_3343_; lean_object* v___x_3345_; uint8_t v_isShared_3346_; uint8_t v_isSharedCheck_3351_; 
v_ref_3337_ = lean_ctor_get(v___y_3334_, 5);
v___x_3338_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11_spec__14(v_msg_3329_, v___y_3332_, v___y_3333_, v___y_3334_, v___y_3335_);
v_a_3339_ = lean_ctor_get(v___x_3338_, 0);
lean_inc(v_a_3339_);
lean_dec_ref(v___x_3338_);
v_macroStack_3340_ = lean_ctor_get(v___y_3330_, 1);
v___x_3341_ = l_Lean_Elab_getBetterRef(v_ref_3337_, v_macroStack_3340_);
lean_inc(v_macroStack_3340_);
v___x_3342_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_a_3339_, v_macroStack_3340_, v___y_3334_);
v_a_3343_ = lean_ctor_get(v___x_3342_, 0);
v_isSharedCheck_3351_ = !lean_is_exclusive(v___x_3342_);
if (v_isSharedCheck_3351_ == 0)
{
v___x_3345_ = v___x_3342_;
v_isShared_3346_ = v_isSharedCheck_3351_;
goto v_resetjp_3344_;
}
else
{
lean_inc(v_a_3343_);
lean_dec(v___x_3342_);
v___x_3345_ = lean_box(0);
v_isShared_3346_ = v_isSharedCheck_3351_;
goto v_resetjp_3344_;
}
v_resetjp_3344_:
{
lean_object* v___x_3347_; lean_object* v___x_3349_; 
v___x_3347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3347_, 0, v___x_3341_);
lean_ctor_set(v___x_3347_, 1, v_a_3343_);
if (v_isShared_3346_ == 0)
{
lean_ctor_set_tag(v___x_3345_, 1);
lean_ctor_set(v___x_3345_, 0, v___x_3347_);
v___x_3349_ = v___x_3345_;
goto v_reusejp_3348_;
}
else
{
lean_object* v_reuseFailAlloc_3350_; 
v_reuseFailAlloc_3350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3350_, 0, v___x_3347_);
v___x_3349_ = v_reuseFailAlloc_3350_;
goto v_reusejp_3348_;
}
v_reusejp_3348_:
{
return v___x_3349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object* v_msg_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_, lean_object* v___y_3355_, lean_object* v___y_3356_, lean_object* v___y_3357_, lean_object* v___y_3358_, lean_object* v___y_3359_){
_start:
{
lean_object* v_res_3360_; 
v_res_3360_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_3352_, v___y_3353_, v___y_3354_, v___y_3355_, v___y_3356_, v___y_3357_, v___y_3358_);
lean_dec(v___y_3358_);
lean_dec_ref(v___y_3357_);
lean_dec(v___y_3356_);
lean_dec_ref(v___y_3355_);
lean_dec(v___y_3354_);
lean_dec_ref(v___y_3353_);
return v_res_3360_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_3361_; lean_object* v___x_3362_; lean_object* v___x_3363_; 
v___x_3361_ = lean_box(0);
v___x_3362_ = l_Lean_Elab_abortTermExceptionId;
v___x_3363_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3363_, 0, v___x_3362_);
lean_ctor_set(v___x_3363_, 1, v___x_3361_);
return v___x_3363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg(){
_start:
{
lean_object* v___x_3365_; lean_object* v___x_3366_; 
v___x_3365_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0);
v___x_3366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3366_, 0, v___x_3365_);
return v___x_3366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object* v___y_3367_){
_start:
{
lean_object* v_res_3368_; 
v_res_3368_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg();
return v_res_3368_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1(void){
_start:
{
lean_object* v___x_3370_; lean_object* v___x_3371_; 
v___x_3370_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__0));
v___x_3371_ = l_Lean_stringToMessageData(v___x_3370_);
return v___x_3371_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2(void){
_start:
{
lean_object* v___x_3372_; lean_object* v___x_3373_; 
v___x_3372_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__1);
v___x_3373_ = l_Lean_MessageData_ofExpr(v___x_3372_);
return v___x_3373_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3(void){
_start:
{
lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; 
v___x_3374_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__2);
v___x_3375_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__1);
v___x_3376_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3376_, 0, v___x_3375_);
lean_ctor_set(v___x_3376_, 1, v___x_3374_);
return v___x_3376_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4(void){
_start:
{
lean_object* v___x_3377_; lean_object* v___x_3378_; lean_object* v___x_3379_; 
v___x_3377_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8___redArg___closed__3);
v___x_3378_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__3);
v___x_3379_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3379_, 0, v___x_3378_);
lean_ctor_set(v___x_3379_, 1, v___x_3377_);
return v___x_3379_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6(void){
_start:
{
lean_object* v___x_3381_; lean_object* v___x_3382_; 
v___x_3381_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__5));
v___x_3382_ = l_Lean_stringToMessageData(v___x_3381_);
return v___x_3382_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8(void){
_start:
{
lean_object* v___x_3384_; lean_object* v___x_3385_; 
v___x_3384_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__7));
v___x_3385_ = l_Lean_stringToMessageData(v___x_3384_);
return v___x_3385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0(lean_object* v_stx_3386_, lean_object* v_a_3387_, lean_object* v_a_3388_, lean_object* v_a_3389_, lean_object* v_a_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_){
_start:
{
lean_object* v_ty_x3f_3394_; uint8_t v___x_3395_; lean_object* v___x_3396_; lean_object* v___x_3397_; lean_object* v___x_3398_; lean_object* v___x_3399_; lean_object* v_fileName_3400_; lean_object* v_fileMap_3401_; lean_object* v_options_3402_; lean_object* v_currRecDepth_3403_; lean_object* v_maxRecDepth_3404_; lean_object* v_ref_3405_; lean_object* v_currNamespace_3406_; lean_object* v_openDecls_3407_; lean_object* v_initHeartbeats_3408_; lean_object* v_maxHeartbeats_3409_; lean_object* v_quotContext_3410_; lean_object* v_currMacroScope_3411_; uint8_t v_diag_3412_; lean_object* v_cancelTk_x3f_3413_; uint8_t v_suppressElabErrors_3414_; lean_object* v_inheritedTraceOptions_3415_; uint8_t v___x_3416_; lean_object* v_ref_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; 
v_ty_x3f_3394_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig___closed__2);
v___x_3395_ = 1;
v___x_3396_ = lean_box(0);
v___x_3397_ = lean_box(v___x_3395_);
v___x_3398_ = lean_box(v___x_3395_);
lean_inc(v_stx_3386_);
v___x_3399_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_3399_, 0, v_stx_3386_);
lean_closure_set(v___x_3399_, 1, v_ty_x3f_3394_);
lean_closure_set(v___x_3399_, 2, v___x_3397_);
lean_closure_set(v___x_3399_, 3, v___x_3398_);
lean_closure_set(v___x_3399_, 4, v___x_3396_);
v_fileName_3400_ = lean_ctor_get(v_a_3391_, 0);
v_fileMap_3401_ = lean_ctor_get(v_a_3391_, 1);
v_options_3402_ = lean_ctor_get(v_a_3391_, 2);
v_currRecDepth_3403_ = lean_ctor_get(v_a_3391_, 3);
v_maxRecDepth_3404_ = lean_ctor_get(v_a_3391_, 4);
v_ref_3405_ = lean_ctor_get(v_a_3391_, 5);
v_currNamespace_3406_ = lean_ctor_get(v_a_3391_, 6);
v_openDecls_3407_ = lean_ctor_get(v_a_3391_, 7);
v_initHeartbeats_3408_ = lean_ctor_get(v_a_3391_, 8);
v_maxHeartbeats_3409_ = lean_ctor_get(v_a_3391_, 9);
v_quotContext_3410_ = lean_ctor_get(v_a_3391_, 10);
v_currMacroScope_3411_ = lean_ctor_get(v_a_3391_, 11);
v_diag_3412_ = lean_ctor_get_uint8(v_a_3391_, sizeof(void*)*14);
v_cancelTk_x3f_3413_ = lean_ctor_get(v_a_3391_, 12);
v_suppressElabErrors_3414_ = lean_ctor_get_uint8(v_a_3391_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3415_ = lean_ctor_get(v_a_3391_, 13);
v___x_3416_ = 1;
v_ref_3417_ = l_Lean_replaceRef(v_stx_3386_, v_ref_3405_);
lean_dec(v_stx_3386_);
lean_inc_ref(v_inheritedTraceOptions_3415_);
lean_inc(v_cancelTk_x3f_3413_);
lean_inc(v_currMacroScope_3411_);
lean_inc(v_quotContext_3410_);
lean_inc(v_maxHeartbeats_3409_);
lean_inc(v_initHeartbeats_3408_);
lean_inc(v_openDecls_3407_);
lean_inc(v_currNamespace_3406_);
lean_inc(v_maxRecDepth_3404_);
lean_inc(v_currRecDepth_3403_);
lean_inc_ref(v_options_3402_);
lean_inc_ref(v_fileMap_3401_);
lean_inc_ref(v_fileName_3400_);
v___x_3418_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3418_, 0, v_fileName_3400_);
lean_ctor_set(v___x_3418_, 1, v_fileMap_3401_);
lean_ctor_set(v___x_3418_, 2, v_options_3402_);
lean_ctor_set(v___x_3418_, 3, v_currRecDepth_3403_);
lean_ctor_set(v___x_3418_, 4, v_maxRecDepth_3404_);
lean_ctor_set(v___x_3418_, 5, v_ref_3417_);
lean_ctor_set(v___x_3418_, 6, v_currNamespace_3406_);
lean_ctor_set(v___x_3418_, 7, v_openDecls_3407_);
lean_ctor_set(v___x_3418_, 8, v_initHeartbeats_3408_);
lean_ctor_set(v___x_3418_, 9, v_maxHeartbeats_3409_);
lean_ctor_set(v___x_3418_, 10, v_quotContext_3410_);
lean_ctor_set(v___x_3418_, 11, v_currMacroScope_3411_);
lean_ctor_set(v___x_3418_, 12, v_cancelTk_x3f_3413_);
lean_ctor_set(v___x_3418_, 13, v_inheritedTraceOptions_3415_);
lean_ctor_set_uint8(v___x_3418_, sizeof(void*)*14, v_diag_3412_);
lean_ctor_set_uint8(v___x_3418_, sizeof(void*)*14 + 1, v_suppressElabErrors_3414_);
v___x_3419_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_3399_, v___x_3416_, v_a_3387_, v_a_3388_, v_a_3389_, v_a_3390_, v___x_3418_, v_a_3392_);
if (lean_obj_tag(v___x_3419_) == 0)
{
lean_object* v_a_3420_; lean_object* v___x_3421_; lean_object* v_a_3422_; lean_object* v___y_3424_; lean_object* v___y_3425_; lean_object* v___y_3426_; lean_object* v___y_3427_; lean_object* v___y_3428_; lean_object* v___y_3429_; lean_object* v___y_3430_; lean_object* v___y_3431_; lean_object* v___y_3432_; uint8_t v___y_3433_; lean_object* v___y_3450_; lean_object* v___y_3451_; lean_object* v___y_3452_; lean_object* v___y_3453_; lean_object* v___y_3454_; lean_object* v___y_3455_; lean_object* v___y_3462_; lean_object* v___y_3463_; lean_object* v___y_3464_; lean_object* v___y_3465_; lean_object* v___y_3466_; lean_object* v___y_3467_; lean_object* v___y_3499_; lean_object* v___y_3500_; lean_object* v___y_3501_; lean_object* v___y_3502_; lean_object* v___y_3503_; lean_object* v___y_3504_; uint8_t v___x_3517_; 
v_a_3420_ = lean_ctor_get(v___x_3419_, 0);
lean_inc(v_a_3420_);
lean_dec_ref_known(v___x_3419_, 1);
v___x_3421_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg(v_a_3420_, v_a_3390_);
v_a_3422_ = lean_ctor_get(v___x_3421_, 0);
lean_inc(v_a_3422_);
lean_dec_ref(v___x_3421_);
v___x_3517_ = l_Lean_Expr_hasSorry(v_a_3422_);
if (v___x_3517_ == 0)
{
v___y_3462_ = v_a_3387_;
v___y_3463_ = v_a_3388_;
v___y_3464_ = v_a_3389_;
v___y_3465_ = v_a_3390_;
v___y_3466_ = v___x_3418_;
v___y_3467_ = v_a_3392_;
goto v___jp_3461_;
}
else
{
uint8_t v___x_3518_; 
v___x_3518_ = l_Lean_Expr_hasSyntheticSorry(v_a_3422_);
if (v___x_3518_ == 0)
{
v___y_3499_ = v_a_3387_;
v___y_3500_ = v_a_3388_;
v___y_3501_ = v_a_3389_;
v___y_3502_ = v_a_3390_;
v___y_3503_ = v___x_3418_;
v___y_3504_ = v_a_3392_;
goto v___jp_3498_;
}
else
{
lean_object* v___x_3519_; lean_object* v_a_3520_; lean_object* v___x_3522_; uint8_t v_isShared_3523_; uint8_t v_isSharedCheck_3527_; 
lean_dec(v_a_3422_);
lean_dec_ref_known(v___x_3418_, 14);
v___x_3519_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_3520_ = lean_ctor_get(v___x_3519_, 0);
v_isSharedCheck_3527_ = !lean_is_exclusive(v___x_3519_);
if (v_isSharedCheck_3527_ == 0)
{
v___x_3522_ = v___x_3519_;
v_isShared_3523_ = v_isSharedCheck_3527_;
goto v_resetjp_3521_;
}
else
{
lean_inc(v_a_3520_);
lean_dec(v___x_3519_);
v___x_3522_ = lean_box(0);
v_isShared_3523_ = v_isSharedCheck_3527_;
goto v_resetjp_3521_;
}
v_resetjp_3521_:
{
lean_object* v___x_3525_; 
if (v_isShared_3523_ == 0)
{
v___x_3525_ = v___x_3522_;
goto v_reusejp_3524_;
}
else
{
lean_object* v_reuseFailAlloc_3526_; 
v_reuseFailAlloc_3526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3526_, 0, v_a_3520_);
v___x_3525_ = v_reuseFailAlloc_3526_;
goto v_reusejp_3524_;
}
v_reusejp_3524_:
{
return v___x_3525_;
}
}
}
}
v___jp_3423_:
{
if (v___y_3433_ == 0)
{
if (lean_obj_tag(v___y_3424_) == 0)
{
lean_dec_ref_known(v___y_3424_, 2);
lean_dec_ref(v___y_3431_);
lean_dec(v_a_3422_);
return v___y_3430_;
}
else
{
lean_object* v_id_3434_; lean_object* v___x_3436_; uint8_t v_isShared_3437_; uint8_t v_isSharedCheck_3447_; 
v_id_3434_ = lean_ctor_get(v___y_3424_, 0);
v_isSharedCheck_3447_ = !lean_is_exclusive(v___y_3424_);
if (v_isSharedCheck_3447_ == 0)
{
lean_object* v_unused_3448_; 
v_unused_3448_ = lean_ctor_get(v___y_3424_, 1);
lean_dec(v_unused_3448_);
v___x_3436_ = v___y_3424_;
v_isShared_3437_ = v_isSharedCheck_3447_;
goto v_resetjp_3435_;
}
else
{
lean_inc(v_id_3434_);
lean_dec(v___y_3424_);
v___x_3436_ = lean_box(0);
v_isShared_3437_ = v_isSharedCheck_3447_;
goto v_resetjp_3435_;
}
v_resetjp_3435_:
{
uint8_t v___x_3438_; 
v___x_3438_ = l_Lean_instBEqInternalExceptionId_beq(v___y_3432_, v_id_3434_);
lean_dec(v_id_3434_);
if (v___x_3438_ == 0)
{
lean_del_object(v___x_3436_);
lean_dec_ref(v___y_3431_);
lean_dec(v_a_3422_);
return v___y_3430_;
}
else
{
lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___x_3441_; lean_object* v___x_3443_; 
lean_dec_ref(v___y_3430_);
v___x_3439_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__4);
v___x_3440_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__6);
v___x_3441_ = l_Lean_indentExpr(v_a_3422_);
if (v_isShared_3437_ == 0)
{
lean_ctor_set_tag(v___x_3436_, 7);
lean_ctor_set(v___x_3436_, 1, v___x_3441_);
lean_ctor_set(v___x_3436_, 0, v___x_3440_);
v___x_3443_ = v___x_3436_;
goto v_reusejp_3442_;
}
else
{
lean_object* v_reuseFailAlloc_3446_; 
v_reuseFailAlloc_3446_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3446_, 0, v___x_3440_);
lean_ctor_set(v_reuseFailAlloc_3446_, 1, v___x_3441_);
v___x_3443_ = v_reuseFailAlloc_3446_;
goto v_reusejp_3442_;
}
v_reusejp_3442_:
{
lean_object* v___x_3444_; lean_object* v___x_3445_; 
v___x_3444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3444_, 0, v___x_3443_);
lean_ctor_set(v___x_3444_, 1, v___x_3439_);
v___x_3445_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3444_, v___y_3427_, v___y_3428_, v___y_3429_, v___y_3425_, v___y_3431_, v___y_3426_);
lean_dec_ref(v___y_3431_);
return v___x_3445_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_3431_);
lean_dec_ref(v___y_3424_);
lean_dec(v_a_3422_);
return v___y_3430_;
}
}
v___jp_3449_:
{
lean_object* v___x_3456_; 
lean_inc(v_a_3422_);
v___x_3456_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr(v_a_3422_, v___y_3452_, v___y_3453_, v___y_3454_, v___y_3455_);
if (lean_obj_tag(v___x_3456_) == 0)
{
lean_dec_ref(v___y_3454_);
lean_dec(v_a_3422_);
return v___x_3456_;
}
else
{
lean_object* v_a_3457_; lean_object* v___x_3458_; uint8_t v___x_3459_; 
v_a_3457_ = lean_ctor_get(v___x_3456_, 0);
lean_inc(v_a_3457_);
v___x_3458_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3459_ = l_Lean_Exception_isInterrupt(v_a_3457_);
if (v___x_3459_ == 0)
{
uint8_t v___x_3460_; 
lean_inc(v_a_3457_);
v___x_3460_ = l_Lean_Exception_isRuntime(v_a_3457_);
v___y_3424_ = v_a_3457_;
v___y_3425_ = v___y_3453_;
v___y_3426_ = v___y_3455_;
v___y_3427_ = v___y_3450_;
v___y_3428_ = v___y_3451_;
v___y_3429_ = v___y_3452_;
v___y_3430_ = v___x_3456_;
v___y_3431_ = v___y_3454_;
v___y_3432_ = v___x_3458_;
v___y_3433_ = v___x_3460_;
goto v___jp_3423_;
}
else
{
v___y_3424_ = v_a_3457_;
v___y_3425_ = v___y_3453_;
v___y_3426_ = v___y_3455_;
v___y_3427_ = v___y_3450_;
v___y_3428_ = v___y_3451_;
v___y_3429_ = v___y_3452_;
v___y_3430_ = v___x_3456_;
v___y_3431_ = v___y_3454_;
v___y_3432_ = v___x_3458_;
v___y_3433_ = v___x_3459_;
goto v___jp_3423_;
}
}
}
v___jp_3461_:
{
lean_object* v___x_3468_; 
lean_inc(v_a_3422_);
v___x_3468_ = l_Lean_Meta_getMVars(v_a_3422_, v___y_3464_, v___y_3465_, v___y_3466_, v___y_3467_);
if (lean_obj_tag(v___x_3468_) == 0)
{
lean_object* v_a_3469_; lean_object* v___x_3470_; 
v_a_3469_ = lean_ctor_get(v___x_3468_, 0);
lean_inc(v_a_3469_);
lean_dec_ref_known(v___x_3468_, 1);
v___x_3470_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_3469_, v___x_3396_, v___y_3462_, v___y_3463_, v___y_3464_, v___y_3465_, v___y_3466_, v___y_3467_);
lean_dec(v_a_3469_);
if (lean_obj_tag(v___x_3470_) == 0)
{
lean_object* v_a_3471_; uint8_t v___x_3472_; 
v_a_3471_ = lean_ctor_get(v___x_3470_, 0);
lean_inc(v_a_3471_);
lean_dec_ref_known(v___x_3470_, 1);
v___x_3472_ = lean_unbox(v_a_3471_);
lean_dec(v_a_3471_);
if (v___x_3472_ == 0)
{
v___y_3450_ = v___y_3462_;
v___y_3451_ = v___y_3463_;
v___y_3452_ = v___y_3464_;
v___y_3453_ = v___y_3465_;
v___y_3454_ = v___y_3466_;
v___y_3455_ = v___y_3467_;
goto v___jp_3449_;
}
else
{
lean_object* v___x_3473_; lean_object* v_a_3474_; lean_object* v___x_3476_; uint8_t v_isShared_3477_; uint8_t v_isSharedCheck_3481_; 
lean_dec_ref(v___y_3466_);
lean_dec(v_a_3422_);
v___x_3473_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_3474_ = lean_ctor_get(v___x_3473_, 0);
v_isSharedCheck_3481_ = !lean_is_exclusive(v___x_3473_);
if (v_isSharedCheck_3481_ == 0)
{
v___x_3476_ = v___x_3473_;
v_isShared_3477_ = v_isSharedCheck_3481_;
goto v_resetjp_3475_;
}
else
{
lean_inc(v_a_3474_);
lean_dec(v___x_3473_);
v___x_3476_ = lean_box(0);
v_isShared_3477_ = v_isSharedCheck_3481_;
goto v_resetjp_3475_;
}
v_resetjp_3475_:
{
lean_object* v___x_3479_; 
if (v_isShared_3477_ == 0)
{
v___x_3479_ = v___x_3476_;
goto v_reusejp_3478_;
}
else
{
lean_object* v_reuseFailAlloc_3480_; 
v_reuseFailAlloc_3480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3480_, 0, v_a_3474_);
v___x_3479_ = v_reuseFailAlloc_3480_;
goto v_reusejp_3478_;
}
v_reusejp_3478_:
{
return v___x_3479_;
}
}
}
}
else
{
lean_object* v_a_3482_; lean_object* v___x_3484_; uint8_t v_isShared_3485_; uint8_t v_isSharedCheck_3489_; 
lean_dec_ref(v___y_3466_);
lean_dec(v_a_3422_);
v_a_3482_ = lean_ctor_get(v___x_3470_, 0);
v_isSharedCheck_3489_ = !lean_is_exclusive(v___x_3470_);
if (v_isSharedCheck_3489_ == 0)
{
v___x_3484_ = v___x_3470_;
v_isShared_3485_ = v_isSharedCheck_3489_;
goto v_resetjp_3483_;
}
else
{
lean_inc(v_a_3482_);
lean_dec(v___x_3470_);
v___x_3484_ = lean_box(0);
v_isShared_3485_ = v_isSharedCheck_3489_;
goto v_resetjp_3483_;
}
v_resetjp_3483_:
{
lean_object* v___x_3487_; 
if (v_isShared_3485_ == 0)
{
v___x_3487_ = v___x_3484_;
goto v_reusejp_3486_;
}
else
{
lean_object* v_reuseFailAlloc_3488_; 
v_reuseFailAlloc_3488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3488_, 0, v_a_3482_);
v___x_3487_ = v_reuseFailAlloc_3488_;
goto v_reusejp_3486_;
}
v_reusejp_3486_:
{
return v___x_3487_;
}
}
}
}
else
{
lean_object* v_a_3490_; lean_object* v___x_3492_; uint8_t v_isShared_3493_; uint8_t v_isSharedCheck_3497_; 
lean_dec_ref(v___y_3466_);
lean_dec(v_a_3422_);
v_a_3490_ = lean_ctor_get(v___x_3468_, 0);
v_isSharedCheck_3497_ = !lean_is_exclusive(v___x_3468_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3492_ = v___x_3468_;
v_isShared_3493_ = v_isSharedCheck_3497_;
goto v_resetjp_3491_;
}
else
{
lean_inc(v_a_3490_);
lean_dec(v___x_3468_);
v___x_3492_ = lean_box(0);
v_isShared_3493_ = v_isSharedCheck_3497_;
goto v_resetjp_3491_;
}
v_resetjp_3491_:
{
lean_object* v___x_3495_; 
if (v_isShared_3493_ == 0)
{
v___x_3495_ = v___x_3492_;
goto v_reusejp_3494_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v_a_3490_);
v___x_3495_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3494_;
}
v_reusejp_3494_:
{
return v___x_3495_;
}
}
}
}
v___jp_3498_:
{
lean_object* v___x_3505_; lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3508_; lean_object* v_a_3509_; lean_object* v___x_3511_; uint8_t v_isShared_3512_; uint8_t v_isSharedCheck_3516_; 
v___x_3505_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___closed__8);
v___x_3506_ = l_Lean_indentExpr(v_a_3422_);
v___x_3507_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3507_, 0, v___x_3505_);
lean_ctor_set(v___x_3507_, 1, v___x_3506_);
v___x_3508_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3507_, v___y_3499_, v___y_3500_, v___y_3501_, v___y_3502_, v___y_3503_, v___y_3504_);
lean_dec_ref(v___y_3503_);
v_a_3509_ = lean_ctor_get(v___x_3508_, 0);
v_isSharedCheck_3516_ = !lean_is_exclusive(v___x_3508_);
if (v_isSharedCheck_3516_ == 0)
{
v___x_3511_ = v___x_3508_;
v_isShared_3512_ = v_isSharedCheck_3516_;
goto v_resetjp_3510_;
}
else
{
lean_inc(v_a_3509_);
lean_dec(v___x_3508_);
v___x_3511_ = lean_box(0);
v_isShared_3512_ = v_isSharedCheck_3516_;
goto v_resetjp_3510_;
}
v_resetjp_3510_:
{
lean_object* v___x_3514_; 
if (v_isShared_3512_ == 0)
{
v___x_3514_ = v___x_3511_;
goto v_reusejp_3513_;
}
else
{
lean_object* v_reuseFailAlloc_3515_; 
v_reuseFailAlloc_3515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3515_, 0, v_a_3509_);
v___x_3514_ = v_reuseFailAlloc_3515_;
goto v_reusejp_3513_;
}
v_reusejp_3513_:
{
return v___x_3514_;
}
}
}
}
else
{
lean_object* v_a_3528_; lean_object* v___x_3530_; uint8_t v_isShared_3531_; uint8_t v_isSharedCheck_3535_; 
lean_dec_ref_known(v___x_3418_, 14);
v_a_3528_ = lean_ctor_get(v___x_3419_, 0);
v_isSharedCheck_3535_ = !lean_is_exclusive(v___x_3419_);
if (v_isSharedCheck_3535_ == 0)
{
v___x_3530_ = v___x_3419_;
v_isShared_3531_ = v_isSharedCheck_3535_;
goto v_resetjp_3529_;
}
else
{
lean_inc(v_a_3528_);
lean_dec(v___x_3419_);
v___x_3530_ = lean_box(0);
v_isShared_3531_ = v_isSharedCheck_3535_;
goto v_resetjp_3529_;
}
v_resetjp_3529_:
{
lean_object* v___x_3533_; 
if (v_isShared_3531_ == 0)
{
v___x_3533_ = v___x_3530_;
goto v_reusejp_3532_;
}
else
{
lean_object* v_reuseFailAlloc_3534_; 
v_reuseFailAlloc_3534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3534_, 0, v_a_3528_);
v___x_3533_ = v_reuseFailAlloc_3534_;
goto v_reusejp_3532_;
}
v_reusejp_3532_:
{
return v___x_3533_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_3536_, lean_object* v_a_3537_, lean_object* v_a_3538_, lean_object* v_a_3539_, lean_object* v_a_3540_, lean_object* v_a_3541_, lean_object* v_a_3542_, lean_object* v_a_3543_){
_start:
{
lean_object* v_res_3544_; 
v_res_3544_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0(v_stx_3536_, v_a_3537_, v_a_3538_, v_a_3539_, v_a_3540_, v_a_3541_, v_a_3542_);
lean_dec(v_a_3542_);
lean_dec_ref(v_a_3541_);
lean_dec(v_a_3540_);
lean_dec_ref(v_a_3539_);
lean_dec(v_a_3538_);
lean_dec_ref(v_a_3537_);
return v_res_3544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0(uint8_t v_config_3555_, lean_object* v_item_3556_, lean_object* v___y_3557_, lean_object* v___y_3558_, lean_object* v___y_3559_, lean_object* v___y_3560_, lean_object* v___y_3561_, lean_object* v___y_3562_){
_start:
{
lean_object* v_item_3565_; lean_object* v___y_3566_; lean_object* v___y_3567_; lean_object* v___y_3568_; lean_object* v___y_3569_; lean_object* v___y_3570_; lean_object* v___y_3571_; lean_object* v___x_3574_; lean_object* v___x_3575_; 
v___x_3574_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4));
v___x_3575_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_3556_, v___x_3574_, v___y_3557_, v___y_3558_, v___y_3559_, v___y_3560_, v___y_3561_, v___y_3562_);
if (lean_obj_tag(v___x_3575_) == 0)
{
uint8_t v___x_3576_; 
lean_dec_ref_known(v___x_3575_, 1);
v___x_3576_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_3556_);
if (v___x_3576_ == 0)
{
lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; uint8_t v___x_3580_; 
v___x_3577_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_3556_);
lean_inc_ref(v_item_3556_);
v___x_3578_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_3556_);
v___x_3579_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__1));
v___x_3580_ = lean_string_dec_eq(v___x_3577_, v___x_3579_);
if (v___x_3580_ == 0)
{
lean_object* v___x_3581_; uint8_t v___x_3582_; 
v___x_3581_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2));
v___x_3582_ = lean_string_dec_eq(v___x_3577_, v___x_3581_);
lean_dec_ref(v___x_3577_);
if (v___x_3582_ == 0)
{
lean_dec_ref(v_item_3556_);
v_item_3565_ = v___x_3578_;
v___y_3566_ = v___y_3557_;
v___y_3567_ = v___y_3558_;
v___y_3568_ = v___y_3559_;
v___y_3569_ = v___y_3560_;
v___y_3570_ = v___y_3561_;
v___y_3571_ = v___y_3562_;
goto v___jp_3564_;
}
else
{
lean_object* v___x_3583_; lean_object* v___x_3584_; 
v___x_3583_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__3));
v___x_3584_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_3556_, v___x_3583_, v___y_3557_, v___y_3558_, v___y_3559_, v___y_3560_, v___y_3561_, v___y_3562_);
if (lean_obj_tag(v___x_3584_) == 0)
{
uint8_t v___x_3585_; 
lean_dec_ref_known(v___x_3584_, 1);
v___x_3585_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_3578_);
if (v___x_3585_ == 0)
{
lean_dec_ref(v_item_3556_);
v_item_3565_ = v___x_3578_;
v___y_3566_ = v___y_3557_;
v___y_3567_ = v___y_3558_;
v___y_3568_ = v___y_3559_;
v___y_3569_ = v___y_3560_;
v___y_3570_ = v___y_3561_;
v___y_3571_ = v___y_3562_;
goto v___jp_3564_;
}
else
{
lean_object* v___x_3586_; 
lean_dec_ref(v___x_3578_);
v___x_3586_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_3556_, v___y_3557_, v___y_3558_, v___y_3559_, v___y_3560_, v___y_3561_, v___y_3562_);
if (lean_obj_tag(v___x_3586_) == 0)
{
lean_object* v_a_3587_; lean_object* v___x_3589_; uint8_t v_isShared_3590_; uint8_t v_isSharedCheck_3594_; 
v_a_3587_ = lean_ctor_get(v___x_3586_, 0);
v_isSharedCheck_3594_ = !lean_is_exclusive(v___x_3586_);
if (v_isSharedCheck_3594_ == 0)
{
v___x_3589_ = v___x_3586_;
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
else
{
lean_inc(v_a_3587_);
lean_dec(v___x_3586_);
v___x_3589_ = lean_box(0);
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
v_resetjp_3588_:
{
lean_object* v___x_3592_; 
if (v_isShared_3590_ == 0)
{
v___x_3592_ = v___x_3589_;
goto v_reusejp_3591_;
}
else
{
lean_object* v_reuseFailAlloc_3593_; 
v_reuseFailAlloc_3593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3593_, 0, v_a_3587_);
v___x_3592_ = v_reuseFailAlloc_3593_;
goto v_reusejp_3591_;
}
v_reusejp_3591_:
{
return v___x_3592_;
}
}
}
else
{
lean_object* v_a_3595_; lean_object* v___x_3597_; uint8_t v_isShared_3598_; uint8_t v_isSharedCheck_3602_; 
v_a_3595_ = lean_ctor_get(v___x_3586_, 0);
v_isSharedCheck_3602_ = !lean_is_exclusive(v___x_3586_);
if (v_isSharedCheck_3602_ == 0)
{
v___x_3597_ = v___x_3586_;
v_isShared_3598_ = v_isSharedCheck_3602_;
goto v_resetjp_3596_;
}
else
{
lean_inc(v_a_3595_);
lean_dec(v___x_3586_);
v___x_3597_ = lean_box(0);
v_isShared_3598_ = v_isSharedCheck_3602_;
goto v_resetjp_3596_;
}
v_resetjp_3596_:
{
lean_object* v___x_3600_; 
if (v_isShared_3598_ == 0)
{
v___x_3600_ = v___x_3597_;
goto v_reusejp_3599_;
}
else
{
lean_object* v_reuseFailAlloc_3601_; 
v_reuseFailAlloc_3601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3601_, 0, v_a_3595_);
v___x_3600_ = v_reuseFailAlloc_3601_;
goto v_reusejp_3599_;
}
v_reusejp_3599_:
{
return v___x_3600_;
}
}
}
}
}
else
{
lean_object* v_a_3603_; lean_object* v___x_3605_; uint8_t v_isShared_3606_; uint8_t v_isSharedCheck_3610_; 
lean_dec_ref(v___x_3578_);
lean_dec_ref(v_item_3556_);
v_a_3603_ = lean_ctor_get(v___x_3584_, 0);
v_isSharedCheck_3610_ = !lean_is_exclusive(v___x_3584_);
if (v_isSharedCheck_3610_ == 0)
{
v___x_3605_ = v___x_3584_;
v_isShared_3606_ = v_isSharedCheck_3610_;
goto v_resetjp_3604_;
}
else
{
lean_inc(v_a_3603_);
lean_dec(v___x_3584_);
v___x_3605_ = lean_box(0);
v_isShared_3606_ = v_isSharedCheck_3610_;
goto v_resetjp_3604_;
}
v_resetjp_3604_:
{
lean_object* v___x_3608_; 
if (v_isShared_3606_ == 0)
{
v___x_3608_ = v___x_3605_;
goto v_reusejp_3607_;
}
else
{
lean_object* v_reuseFailAlloc_3609_; 
v_reuseFailAlloc_3609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3609_, 0, v_a_3603_);
v___x_3608_ = v_reuseFailAlloc_3609_;
goto v_reusejp_3607_;
}
v_reusejp_3607_:
{
return v___x_3608_;
}
}
}
}
}
else
{
uint8_t v___x_3611_; 
lean_dec_ref(v___x_3577_);
v___x_3611_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_3578_);
if (v___x_3611_ == 0)
{
lean_dec_ref(v_item_3556_);
v_item_3565_ = v___x_3578_;
v___y_3566_ = v___y_3557_;
v___y_3567_ = v___y_3558_;
v___y_3568_ = v___y_3559_;
v___y_3569_ = v___y_3560_;
v___y_3570_ = v___y_3561_;
v___y_3571_ = v___y_3562_;
goto v___jp_3564_;
}
else
{
lean_object* v_value_3612_; lean_object* v___x_3613_; 
lean_dec_ref(v___x_3578_);
v_value_3612_ = lean_ctor_get(v_item_3556_, 2);
lean_inc(v_value_3612_);
lean_dec_ref(v_item_3556_);
v___x_3613_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0(v_value_3612_, v___y_3557_, v___y_3558_, v___y_3559_, v___y_3560_, v___y_3561_, v___y_3562_);
return v___x_3613_;
}
}
}
else
{
v_item_3565_ = v_item_3556_;
v___y_3566_ = v___y_3557_;
v___y_3567_ = v___y_3558_;
v___y_3568_ = v___y_3559_;
v___y_3569_ = v___y_3560_;
v___y_3570_ = v___y_3561_;
v___y_3571_ = v___y_3562_;
goto v___jp_3564_;
}
}
else
{
lean_object* v_a_3614_; lean_object* v___x_3616_; uint8_t v_isShared_3617_; uint8_t v_isSharedCheck_3621_; 
lean_dec_ref(v_item_3556_);
v_a_3614_ = lean_ctor_get(v___x_3575_, 0);
v_isSharedCheck_3621_ = !lean_is_exclusive(v___x_3575_);
if (v_isSharedCheck_3621_ == 0)
{
v___x_3616_ = v___x_3575_;
v_isShared_3617_ = v_isSharedCheck_3621_;
goto v_resetjp_3615_;
}
else
{
lean_inc(v_a_3614_);
lean_dec(v___x_3575_);
v___x_3616_ = lean_box(0);
v_isShared_3617_ = v_isSharedCheck_3621_;
goto v_resetjp_3615_;
}
v_resetjp_3615_:
{
lean_object* v___x_3619_; 
if (v_isShared_3617_ == 0)
{
v___x_3619_ = v___x_3616_;
goto v_reusejp_3618_;
}
else
{
lean_object* v_reuseFailAlloc_3620_; 
v_reuseFailAlloc_3620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3620_, 0, v_a_3614_);
v___x_3619_ = v_reuseFailAlloc_3620_;
goto v_reusejp_3618_;
}
v_reusejp_3618_:
{
return v___x_3619_;
}
}
}
v___jp_3564_:
{
lean_object* v___x_3572_; lean_object* v___x_3573_; 
v___x_3572_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__0));
v___x_3573_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_3565_, v___x_3572_, v___y_3566_, v___y_3567_, v___y_3568_, v___y_3569_, v___y_3570_, v___y_3571_);
return v___x_3573_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_3622_, lean_object* v_item_3623_, lean_object* v___y_3624_, lean_object* v___y_3625_, lean_object* v___y_3626_, lean_object* v___y_3627_, lean_object* v___y_3628_, lean_object* v___y_3629_, lean_object* v___y_3630_){
_start:
{
uint8_t v_config_3942__boxed_3631_; lean_object* v_res_3632_; 
v_config_3942__boxed_3631_ = lean_unbox(v_config_3622_);
v_res_3632_ = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0(v_config_3942__boxed_3631_, v_item_3623_, v___y_3624_, v___y_3625_, v___y_3626_, v___y_3627_, v___y_3628_, v___y_3629_);
lean_dec(v___y_3629_);
lean_dec_ref(v___y_3628_);
lean_dec(v___y_3627_);
lean_dec_ref(v___y_3626_);
lean_dec(v___y_3625_);
lean_dec_ref(v___y_3624_);
return v_res_3632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0(lean_object* v_e_3635_, lean_object* v___y_3636_, lean_object* v___y_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_){
_start:
{
lean_object* v___x_3643_; 
v___x_3643_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_3635_, v___y_3639_);
return v___x_3643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_e_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_, lean_object* v___y_3651_){
_start:
{
lean_object* v_res_3652_; 
v_res_3652_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__0(v_e_3644_, v___y_3645_, v___y_3646_, v___y_3647_, v___y_3648_, v___y_3649_, v___y_3650_);
lean_dec(v___y_3650_);
lean_dec_ref(v___y_3649_);
lean_dec(v___y_3648_);
lean_dec_ref(v___y_3647_);
lean_dec(v___y_3646_);
lean_dec_ref(v___y_3645_);
return v_res_3652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2(lean_object* v_00_u03b1_3653_, lean_object* v___y_3654_, lean_object* v___y_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_){
_start:
{
lean_object* v___x_3661_; 
v___x_3661_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___redArg();
return v___x_3661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object* v_00_u03b1_3662_, lean_object* v___y_3663_, lean_object* v___y_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_){
_start:
{
lean_object* v_res_3670_; 
v_res_3670_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__2(v_00_u03b1_3662_, v___y_3663_, v___y_3664_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_);
lean_dec(v___y_3668_);
lean_dec_ref(v___y_3667_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
lean_dec(v___y_3664_);
lean_dec_ref(v___y_3663_);
return v_res_3670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1(lean_object* v_00_u03b1_3671_, lean_object* v_msg_3672_, lean_object* v___y_3673_, lean_object* v___y_3674_, lean_object* v___y_3675_, lean_object* v___y_3676_, lean_object* v___y_3677_, lean_object* v___y_3678_){
_start:
{
lean_object* v___x_3680_; 
v___x_3680_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_3672_, v___y_3673_, v___y_3674_, v___y_3675_, v___y_3676_, v___y_3677_, v___y_3678_);
return v___x_3680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object* v_00_u03b1_3681_, lean_object* v_msg_3682_, lean_object* v___y_3683_, lean_object* v___y_3684_, lean_object* v___y_3685_, lean_object* v___y_3686_, lean_object* v___y_3687_, lean_object* v___y_3688_, lean_object* v___y_3689_){
_start:
{
lean_object* v_res_3690_; 
v_res_3690_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1(v_00_u03b1_3681_, v_msg_3682_, v___y_3683_, v___y_3684_, v___y_3685_, v___y_3686_, v___y_3687_, v___y_3688_);
lean_dec(v___y_3688_);
lean_dec_ref(v___y_3687_);
lean_dec(v___y_3686_);
lean_dec_ref(v___y_3685_);
lean_dec(v___y_3684_);
lean_dec_ref(v___y_3683_);
return v_res_3690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object* v_msgData_3691_, lean_object* v_macroStack_3692_, lean_object* v___y_3693_, lean_object* v___y_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
lean_object* v___x_3700_; 
v___x_3700_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_3691_, v_macroStack_3692_, v___y_3697_);
return v___x_3700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object* v_msgData_3701_, lean_object* v_macroStack_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_, lean_object* v___y_3706_, lean_object* v___y_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_){
_start:
{
lean_object* v_res_3710_; 
v_res_3710_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem_spec__0_spec__1_spec__2(v_msgData_3701_, v_macroStack_3702_, v___y_3703_, v___y_3704_, v___y_3705_, v___y_3706_, v___y_3707_, v___y_3708_);
lean_dec(v___y_3708_);
lean_dec_ref(v___y_3707_);
lean_dec(v___y_3706_);
lean_dec_ref(v___y_3705_);
lean_dec(v___y_3704_);
lean_dec_ref(v___y_3703_);
return v_res_3710_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_3711_; lean_object* v___x_3712_; lean_object* v___x_3713_; 
v___x_3711_ = lean_box(0);
v___x_3712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig_evalExpr___closed__4));
v___x_3713_ = l_Lean_mkConst(v___x_3712_, v___x_3711_);
return v___x_3713_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3714_; lean_object* v___x_3715_; 
v___x_3714_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__0);
v___x_3715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3715_, 0, v___x_3714_);
return v___x_3715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0(uint8_t v_cfg_3716_, lean_object* v_cfgItem_3717_, lean_object* v___y_3718_, lean_object* v___y_3719_, lean_object* v___y_3720_, lean_object* v___y_3721_, lean_object* v___y_3722_, lean_object* v___y_3723_){
_start:
{
lean_object* v___x_3725_; lean_object* v___x_3726_; lean_object* v___x_3727_; 
v___x_3725_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___closed__1);
v___x_3726_ = lean_box(v_cfg_3716_);
v___x_3727_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v___x_3726_, v_cfgItem_3717_, v___x_3725_, v___y_3718_, v___y_3719_, v___y_3720_, v___y_3721_, v___y_3722_, v___y_3723_);
return v___x_3727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0___boxed(lean_object* v_cfg_3728_, lean_object* v_cfgItem_3729_, lean_object* v___y_3730_, lean_object* v___y_3731_, lean_object* v___y_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_){
_start:
{
uint8_t v_cfg_boxed_3737_; lean_object* v_res_3738_; 
v_cfg_boxed_3737_ = lean_unbox(v_cfg_3728_);
v_res_3738_ = lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___lam__0(v_cfg_boxed_3737_, v_cfgItem_3729_, v___y_3730_, v___y_3731_, v___y_3732_, v___y_3733_, v___y_3734_, v___y_3735_);
lean_dec(v___y_3735_);
lean_dec_ref(v___y_3734_);
lean_dec(v___y_3733_);
lean_dec_ref(v___y_3732_);
lean_dec(v___y_3731_);
lean_dec_ref(v___y_3730_);
lean_dec(v_cfgItem_3729_);
return v_res_3738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg(lean_object* v_cfg_3740_, uint8_t v_init_3741_, uint8_t v_logExceptions_3742_, lean_object* v_a_3743_, lean_object* v_a_3744_, lean_object* v_a_3745_){
_start:
{
lean_object* v_onErr_3747_; lean_object* v_eval_3748_; 
v_onErr_3747_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___closed__0));
v_eval_3748_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___closed__0));
if (v_logExceptions_3742_ == 0)
{
lean_object* v___x_3749_; lean_object* v___x_3750_; 
v___x_3749_ = lean_box(v_init_3741_);
v___x_3750_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3748_, v___x_3749_, v_cfg_3740_, v_onErr_3747_, v_logExceptions_3742_, v_a_3744_, v_a_3745_);
return v___x_3750_;
}
else
{
uint8_t v_recover_3751_; lean_object* v___x_3752_; lean_object* v___x_3753_; 
v_recover_3751_ = lean_ctor_get_uint8(v_a_3743_, sizeof(void*)*1);
v___x_3752_ = lean_box(v_init_3741_);
v___x_3753_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3748_, v___x_3752_, v_cfg_3740_, v_onErr_3747_, v_recover_3751_, v_a_3744_, v_a_3745_);
return v___x_3753_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg___boxed(lean_object* v_cfg_3754_, lean_object* v_init_3755_, lean_object* v_logExceptions_3756_, lean_object* v_a_3757_, lean_object* v_a_3758_, lean_object* v_a_3759_, lean_object* v_a_3760_){
_start:
{
uint8_t v_init_boxed_3761_; uint8_t v_logExceptions_boxed_3762_; lean_object* v_res_3763_; 
v_init_boxed_3761_ = lean_unbox(v_init_3755_);
v_logExceptions_boxed_3762_ = lean_unbox(v_logExceptions_3756_);
v_res_3763_ = lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg(v_cfg_3754_, v_init_boxed_3761_, v_logExceptions_boxed_3762_, v_a_3757_, v_a_3758_, v_a_3759_);
lean_dec(v_a_3759_);
lean_dec_ref(v_a_3758_);
lean_dec_ref(v_a_3757_);
return v_res_3763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig(lean_object* v_cfg_3764_, uint8_t v_init_3765_, uint8_t v_logExceptions_3766_, lean_object* v_a_3767_, lean_object* v_a_3768_, lean_object* v_a_3769_, lean_object* v_a_3770_, lean_object* v_a_3771_, lean_object* v_a_3772_, lean_object* v_a_3773_, lean_object* v_a_3774_){
_start:
{
lean_object* v___x_3776_; 
v___x_3776_ = lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg(v_cfg_3764_, v_init_3765_, v_logExceptions_3766_, v_a_3767_, v_a_3773_, v_a_3774_);
return v___x_3776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___boxed(lean_object* v_cfg_3777_, lean_object* v_init_3778_, lean_object* v_logExceptions_3779_, lean_object* v_a_3780_, lean_object* v_a_3781_, lean_object* v_a_3782_, lean_object* v_a_3783_, lean_object* v_a_3784_, lean_object* v_a_3785_, lean_object* v_a_3786_, lean_object* v_a_3787_, lean_object* v_a_3788_){
_start:
{
uint8_t v_init_boxed_3789_; uint8_t v_logExceptions_boxed_3790_; lean_object* v_res_3791_; 
v_init_boxed_3789_ = lean_unbox(v_init_3778_);
v_logExceptions_boxed_3790_ = lean_unbox(v_logExceptions_3779_);
v_res_3791_ = lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig(v_cfg_3777_, v_init_boxed_3789_, v_logExceptions_boxed_3790_, v_a_3780_, v_a_3781_, v_a_3782_, v_a_3783_, v_a_3784_, v_a_3785_, v_a_3786_, v_a_3787_);
lean_dec(v_a_3787_);
lean_dec_ref(v_a_3786_);
lean_dec(v_a_3785_);
lean_dec_ref(v_a_3784_);
lean_dec(v_a_3783_);
lean_dec_ref(v_a_3782_);
lean_dec(v_a_3781_);
lean_dec_ref(v_a_3780_);
return v_res_3791_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4(void){
_start:
{
lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_3852_; 
v___x_3849_ = l_Lean_Parser_Tactic_optConfig;
v___x_3850_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__3));
v___x_3851_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3));
v___x_3852_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3852_, 0, v___x_3851_);
lean_ctor_set(v___x_3852_, 1, v___x_3850_);
lean_ctor_set(v___x_3852_, 2, v___x_3849_);
return v___x_3852_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8(void){
_start:
{
lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3861_; lean_object* v___x_3862_; 
v___x_3859_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__7));
v___x_3860_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4, &lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__4);
v___x_3861_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__3));
v___x_3862_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3862_, 0, v___x_3861_);
lean_ctor_set(v___x_3862_, 1, v___x_3860_);
lean_ctor_set(v___x_3862_, 2, v___x_3859_);
return v___x_3862_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9(void){
_start:
{
lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v___x_3865_; lean_object* v___x_3866_; 
v___x_3863_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8, &lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__8);
v___x_3864_ = lean_unsigned_to_nat(1022u);
v___x_3865_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1));
v___x_3866_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3866_, 0, v___x_3865_);
lean_ctor_set(v___x_3866_, 1, v___x_3864_);
lean_ctor_set(v___x_3866_, 2, v___x_3863_);
return v___x_3866_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize____(void){
_start:
{
lean_object* v___x_3867_; 
v___x_3867_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9, &lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9_once, _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__9);
return v___x_3867_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; 
v___x_3868_ = lean_box(0);
v___x_3869_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3870_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3870_, 0, v___x_3869_);
lean_ctor_set(v___x_3870_, 1, v___x_3868_);
return v___x_3870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg(){
_start:
{
lean_object* v___x_3872_; lean_object* v___x_3873_; 
v___x_3872_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___closed__0);
v___x_3873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3873_, 0, v___x_3872_);
return v___x_3873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg___boxed(lean_object* v___y_3874_){
_start:
{
lean_object* v_res_3875_; 
v_res_3875_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v_res_3875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0(lean_object* v_00_u03b1_3876_, lean_object* v___y_3877_, lean_object* v___y_3878_, lean_object* v___y_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_){
_start:
{
lean_object* v___x_3886_; 
v___x_3886_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_3886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___boxed(lean_object* v_00_u03b1_3887_, lean_object* v___y_3888_, lean_object* v___y_3889_, lean_object* v___y_3890_, lean_object* v___y_3891_, lean_object* v___y_3892_, lean_object* v___y_3893_, lean_object* v___y_3894_, lean_object* v___y_3895_, lean_object* v___y_3896_){
_start:
{
lean_object* v_res_3897_; 
v_res_3897_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0(v_00_u03b1_3887_, v___y_3888_, v___y_3889_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_, v___y_3894_, v___y_3895_);
lean_dec(v___y_3895_);
lean_dec_ref(v___y_3894_);
lean_dec(v___y_3893_);
lean_dec_ref(v___y_3892_);
lean_dec(v___y_3891_);
lean_dec_ref(v___y_3890_);
lean_dec(v___y_3889_);
lean_dec_ref(v___y_3888_);
return v_res_3897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg(lean_object* v_ref_3898_, lean_object* v_msgData_3899_, lean_object* v___y_3900_, lean_object* v___y_3901_, lean_object* v___y_3902_, lean_object* v___y_3903_){
_start:
{
uint8_t v___x_3905_; uint8_t v___x_3906_; lean_object* v___x_3907_; 
v___x_3905_ = 1;
v___x_3906_ = 0;
v___x_3907_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg(v_ref_3898_, v_msgData_3899_, v___x_3905_, v___x_3906_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
return v___x_3907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg___boxed(lean_object* v_ref_3908_, lean_object* v_msgData_3909_, lean_object* v___y_3910_, lean_object* v___y_3911_, lean_object* v___y_3912_, lean_object* v___y_3913_, lean_object* v___y_3914_){
_start:
{
lean_object* v_res_3915_; 
v_res_3915_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg(v_ref_3908_, v_msgData_3909_, v___y_3910_, v___y_3911_, v___y_3912_, v___y_3913_);
lean_dec(v___y_3913_);
lean_dec_ref(v___y_3912_);
lean_dec(v___y_3911_);
lean_dec_ref(v___y_3910_);
lean_dec(v_ref_3908_);
return v_res_3915_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1(void){
_start:
{
lean_object* v___x_3917_; lean_object* v___x_3918_; 
v___x_3917_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__0));
v___x_3918_ = l_Lean_stringToMessageData(v___x_3917_);
return v___x_3918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1(lean_object* v_as_3919_, size_t v_sz_3920_, size_t v_i_3921_, lean_object* v_b_3922_, lean_object* v___y_3923_, lean_object* v___y_3924_, lean_object* v___y_3925_, lean_object* v___y_3926_, lean_object* v___y_3927_, lean_object* v___y_3928_, lean_object* v___y_3929_, lean_object* v___y_3930_){
_start:
{
lean_object* v_a_3933_; uint8_t v___x_3937_; 
v___x_3937_ = lean_usize_dec_lt(v_i_3921_, v_sz_3920_);
if (v___x_3937_ == 0)
{
lean_object* v___x_3938_; 
v___x_3938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3938_, 0, v_b_3922_);
return v___x_3938_;
}
else
{
lean_object* v_a_3939_; lean_object* v___x_3940_; 
v_a_3939_ = lean_array_uget_borrowed(v_as_3919_, v_i_3921_);
lean_inc(v___y_3930_);
lean_inc_ref(v___y_3929_);
lean_inc(v___y_3928_);
lean_inc_ref(v___y_3927_);
lean_inc(v_a_3939_);
v___x_3940_ = lean_infer_type(v_a_3939_, v___y_3927_, v___y_3928_, v___y_3929_, v___y_3930_);
if (lean_obj_tag(v___x_3940_) == 0)
{
lean_object* v_a_3941_; lean_object* v___x_3942_; lean_object* v___y_3944_; lean_object* v___y_3945_; lean_object* v___y_3946_; lean_object* v___y_3947_; lean_object* v___x_3952_; 
v_a_3941_ = lean_ctor_get(v___x_3940_, 0);
lean_inc(v_a_3941_);
lean_dec_ref_known(v___x_3940_, 1);
v___x_3942_ = lean_box(0);
v___x_3952_ = l_Lean_Expr_getAppFn(v_a_3941_);
if (lean_obj_tag(v___x_3952_) == 4)
{
lean_object* v_declName_3953_; 
v_declName_3953_ = lean_ctor_get(v___x_3952_, 0);
lean_inc(v_declName_3953_);
lean_dec_ref_known(v___x_3952_, 2);
if (lean_obj_tag(v_declName_3953_) == 1)
{
lean_object* v_pre_3954_; 
v_pre_3954_ = lean_ctor_get(v_declName_3953_, 0);
if (lean_obj_tag(v_pre_3954_) == 0)
{
lean_object* v_str_3955_; lean_object* v___x_3956_; uint8_t v___x_3957_; 
v_str_3955_ = lean_ctor_get(v_declName_3953_, 1);
lean_inc_ref(v_str_3955_);
lean_dec_ref_known(v_declName_3953_, 2);
v___x_3956_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11));
v___x_3957_ = lean_string_dec_eq(v_str_3955_, v___x_3956_);
lean_dec_ref(v_str_3955_);
if (v___x_3957_ == 0)
{
lean_dec(v_a_3941_);
v___y_3944_ = v___y_3927_;
v___y_3945_ = v___y_3928_;
v___y_3946_ = v___y_3929_;
v___y_3947_ = v___y_3930_;
goto v___jp_3943_;
}
else
{
lean_object* v___x_3958_; 
lean_inc(v_a_3939_);
v___x_3958_ = lp_mathlib_Mathlib_Tactic_Algebraize_addAlgebraInstanceFromRingHom(v_a_3939_, v_a_3941_, v___y_3923_, v___y_3924_, v___y_3925_, v___y_3926_, v___y_3927_, v___y_3928_, v___y_3929_, v___y_3930_);
if (lean_obj_tag(v___x_3958_) == 0)
{
lean_dec_ref_known(v___x_3958_, 1);
v_a_3933_ = v___x_3942_;
goto v___jp_3932_;
}
else
{
return v___x_3958_;
}
}
}
else
{
lean_dec_ref_known(v_declName_3953_, 2);
lean_dec(v_a_3941_);
v___y_3944_ = v___y_3927_;
v___y_3945_ = v___y_3928_;
v___y_3946_ = v___y_3929_;
v___y_3947_ = v___y_3930_;
goto v___jp_3943_;
}
}
else
{
lean_dec(v_declName_3953_);
lean_dec(v_a_3941_);
v___y_3944_ = v___y_3927_;
v___y_3945_ = v___y_3928_;
v___y_3946_ = v___y_3929_;
v___y_3947_ = v___y_3930_;
goto v___jp_3943_;
}
}
else
{
lean_dec_ref(v___x_3952_);
lean_dec(v_a_3941_);
v___y_3944_ = v___y_3927_;
v___y_3945_ = v___y_3928_;
v___y_3946_ = v___y_3929_;
v___y_3947_ = v___y_3930_;
goto v___jp_3943_;
}
v___jp_3943_:
{
lean_object* v___x_3948_; lean_object* v___x_3949_; lean_object* v___x_3950_; lean_object* v___x_3951_; 
lean_inc(v_a_3939_);
v___x_3948_ = l_Lean_MessageData_ofExpr(v_a_3939_);
v___x_3949_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___closed__1);
v___x_3950_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3950_, 0, v___x_3948_);
lean_ctor_set(v___x_3950_, 1, v___x_3949_);
v___x_3951_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v___x_3950_, v___y_3944_, v___y_3945_, v___y_3946_, v___y_3947_);
if (lean_obj_tag(v___x_3951_) == 0)
{
lean_dec_ref_known(v___x_3951_, 1);
v_a_3933_ = v___x_3942_;
goto v___jp_3932_;
}
else
{
return v___x_3951_;
}
}
}
else
{
lean_object* v_a_3959_; lean_object* v___x_3961_; uint8_t v_isShared_3962_; uint8_t v_isSharedCheck_3966_; 
v_a_3959_ = lean_ctor_get(v___x_3940_, 0);
v_isSharedCheck_3966_ = !lean_is_exclusive(v___x_3940_);
if (v_isSharedCheck_3966_ == 0)
{
v___x_3961_ = v___x_3940_;
v_isShared_3962_ = v_isSharedCheck_3966_;
goto v_resetjp_3960_;
}
else
{
lean_inc(v_a_3959_);
lean_dec(v___x_3940_);
v___x_3961_ = lean_box(0);
v_isShared_3962_ = v_isSharedCheck_3966_;
goto v_resetjp_3960_;
}
v_resetjp_3960_:
{
lean_object* v___x_3964_; 
if (v_isShared_3962_ == 0)
{
v___x_3964_ = v___x_3961_;
goto v_reusejp_3963_;
}
else
{
lean_object* v_reuseFailAlloc_3965_; 
v_reuseFailAlloc_3965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3965_, 0, v_a_3959_);
v___x_3964_ = v_reuseFailAlloc_3965_;
goto v_reusejp_3963_;
}
v_reusejp_3963_:
{
return v___x_3964_;
}
}
}
}
v___jp_3932_:
{
size_t v___x_3934_; size_t v___x_3935_; 
v___x_3934_ = ((size_t)1ULL);
v___x_3935_ = lean_usize_add(v_i_3921_, v___x_3934_);
v_i_3921_ = v___x_3935_;
v_b_3922_ = v_a_3933_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1___boxed(lean_object* v_as_3967_, lean_object* v_sz_3968_, lean_object* v_i_3969_, lean_object* v_b_3970_, lean_object* v___y_3971_, lean_object* v___y_3972_, lean_object* v___y_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_, lean_object* v___y_3978_, lean_object* v___y_3979_){
_start:
{
size_t v_sz_boxed_3980_; size_t v_i_boxed_3981_; lean_object* v_res_3982_; 
v_sz_boxed_3980_ = lean_unbox_usize(v_sz_3968_);
lean_dec(v_sz_3968_);
v_i_boxed_3981_ = lean_unbox_usize(v_i_3969_);
lean_dec(v_i_3969_);
v_res_3982_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1(v_as_3967_, v_sz_boxed_3980_, v_i_boxed_3981_, v_b_3970_, v___y_3971_, v___y_3972_, v___y_3973_, v___y_3974_, v___y_3975_, v___y_3976_, v___y_3977_, v___y_3978_);
lean_dec(v___y_3978_);
lean_dec_ref(v___y_3977_);
lean_dec(v___y_3976_);
lean_dec_ref(v___y_3975_);
lean_dec(v___y_3974_);
lean_dec_ref(v___y_3973_);
lean_dec(v___y_3972_);
lean_dec_ref(v___y_3971_);
lean_dec_ref(v_as_3967_);
return v_res_3982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2(lean_object* v_as_3983_, size_t v_sz_3984_, size_t v_i_3985_, lean_object* v_b_3986_, lean_object* v___y_3987_, lean_object* v___y_3988_, lean_object* v___y_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_, lean_object* v___y_3994_){
_start:
{
lean_object* v_a_3997_; uint8_t v___x_4001_; 
v___x_4001_ = lean_usize_dec_lt(v_i_3985_, v_sz_3984_);
if (v___x_4001_ == 0)
{
lean_object* v___x_4002_; 
v___x_4002_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4002_, 0, v_b_3986_);
return v___x_4002_;
}
else
{
lean_object* v___x_4003_; lean_object* v_a_4004_; lean_object* v___x_4005_; 
v___x_4003_ = lean_box(0);
v_a_4004_ = lean_array_uget_borrowed(v_as_3983_, v_i_3985_);
v___x_4005_ = l_Lean_Expr_getAppFn(v_a_4004_);
if (lean_obj_tag(v___x_4005_) == 4)
{
lean_object* v_declName_4006_; 
v_declName_4006_ = lean_ctor_get(v___x_4005_, 0);
lean_inc(v_declName_4006_);
lean_dec_ref_known(v___x_4005_, 2);
if (lean_obj_tag(v_declName_4006_) == 1)
{
lean_object* v_pre_4007_; 
v_pre_4007_ = lean_ctor_get(v_declName_4006_, 0);
lean_inc(v_pre_4007_);
if (lean_obj_tag(v_pre_4007_) == 1)
{
lean_object* v_pre_4008_; 
v_pre_4008_ = lean_ctor_get(v_pre_4007_, 0);
if (lean_obj_tag(v_pre_4008_) == 0)
{
lean_object* v_str_4009_; lean_object* v_str_4010_; lean_object* v___x_4011_; uint8_t v___x_4012_; 
v_str_4009_ = lean_ctor_get(v_declName_4006_, 1);
lean_inc_ref(v_str_4009_);
lean_dec_ref_known(v_declName_4006_, 2);
v_str_4010_ = lean_ctor_get(v_pre_4007_, 1);
lean_inc_ref(v_str_4010_);
lean_dec_ref_known(v_pre_4007_, 2);
v___x_4011_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__11));
v___x_4012_ = lean_string_dec_eq(v_str_4010_, v___x_4011_);
lean_dec_ref(v_str_4010_);
if (v___x_4012_ == 0)
{
lean_dec_ref(v_str_4009_);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
else
{
lean_object* v___x_4013_; uint8_t v___x_4014_; 
v___x_4013_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp___lam__0___closed__4));
v___x_4014_ = lean_string_dec_eq(v_str_4009_, v___x_4013_);
lean_dec_ref(v_str_4009_);
if (v___x_4014_ == 0)
{
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
else
{
lean_object* v___x_4015_; 
v___x_4015_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3988_, v___y_3990_, v___y_3992_, v___y_3994_);
if (lean_obj_tag(v___x_4015_) == 0)
{
lean_object* v_a_4016_; lean_object* v___x_4017_; 
v_a_4016_ = lean_ctor_get(v___x_4015_, 0);
lean_inc(v_a_4016_);
lean_dec_ref_known(v___x_4015_, 1);
lean_inc(v_a_4004_);
v___x_4017_ = lp_mathlib_Mathlib_Tactic_Algebraize_addIsScalarTowerInstanceFromRingHomComp(v_a_4004_, v___y_3987_, v___y_3988_, v___y_3989_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_, v___y_3994_);
if (lean_obj_tag(v___x_4017_) == 0)
{
lean_dec_ref_known(v___x_4017_, 1);
lean_dec(v_a_4016_);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
else
{
lean_object* v_a_4018_; uint8_t v___y_4020_; uint8_t v___x_4022_; 
v_a_4018_ = lean_ctor_get(v___x_4017_, 0);
lean_inc(v_a_4018_);
v___x_4022_ = l_Lean_Exception_isInterrupt(v_a_4018_);
if (v___x_4022_ == 0)
{
uint8_t v___x_4023_; 
v___x_4023_ = l_Lean_Exception_isRuntime(v_a_4018_);
v___y_4020_ = v___x_4023_;
goto v___jp_4019_;
}
else
{
lean_dec(v_a_4018_);
v___y_4020_ = v___x_4022_;
goto v___jp_4019_;
}
v___jp_4019_:
{
if (v___y_4020_ == 0)
{
lean_object* v___x_4021_; 
lean_dec_ref_known(v___x_4017_, 1);
v___x_4021_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_4016_, v___y_4020_, v___y_3988_, v___y_3989_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_, v___y_3994_);
if (lean_obj_tag(v___x_4021_) == 0)
{
lean_dec_ref_known(v___x_4021_, 1);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
else
{
return v___x_4021_;
}
}
else
{
lean_dec(v_a_4016_);
return v___x_4017_;
}
}
}
}
else
{
lean_object* v_a_4024_; lean_object* v___x_4026_; uint8_t v_isShared_4027_; uint8_t v_isSharedCheck_4031_; 
v_a_4024_ = lean_ctor_get(v___x_4015_, 0);
v_isSharedCheck_4031_ = !lean_is_exclusive(v___x_4015_);
if (v_isSharedCheck_4031_ == 0)
{
v___x_4026_ = v___x_4015_;
v_isShared_4027_ = v_isSharedCheck_4031_;
goto v_resetjp_4025_;
}
else
{
lean_inc(v_a_4024_);
lean_dec(v___x_4015_);
v___x_4026_ = lean_box(0);
v_isShared_4027_ = v_isSharedCheck_4031_;
goto v_resetjp_4025_;
}
v_resetjp_4025_:
{
lean_object* v___x_4029_; 
if (v_isShared_4027_ == 0)
{
v___x_4029_ = v___x_4026_;
goto v_reusejp_4028_;
}
else
{
lean_object* v_reuseFailAlloc_4030_; 
v_reuseFailAlloc_4030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4030_, 0, v_a_4024_);
v___x_4029_ = v_reuseFailAlloc_4030_;
goto v_reusejp_4028_;
}
v_reusejp_4028_:
{
return v___x_4029_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_4007_, 2);
lean_dec_ref_known(v_declName_4006_, 2);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
}
else
{
lean_dec(v_pre_4007_);
lean_dec_ref_known(v_declName_4006_, 2);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
}
else
{
lean_dec(v_declName_4006_);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
}
else
{
lean_dec_ref(v___x_4005_);
v_a_3997_ = v___x_4003_;
goto v___jp_3996_;
}
}
v___jp_3996_:
{
size_t v___x_3998_; size_t v___x_3999_; 
v___x_3998_ = ((size_t)1ULL);
v___x_3999_ = lean_usize_add(v_i_3985_, v___x_3998_);
v_i_3985_ = v___x_3999_;
v_b_3986_ = v_a_3997_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2___boxed(lean_object* v_as_4032_, lean_object* v_sz_4033_, lean_object* v_i_4034_, lean_object* v_b_4035_, lean_object* v___y_4036_, lean_object* v___y_4037_, lean_object* v___y_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_, lean_object* v___y_4044_){
_start:
{
size_t v_sz_boxed_4045_; size_t v_i_boxed_4046_; lean_object* v_res_4047_; 
v_sz_boxed_4045_ = lean_unbox_usize(v_sz_4033_);
lean_dec(v_sz_4033_);
v_i_boxed_4046_ = lean_unbox_usize(v_i_4034_);
lean_dec(v_i_4034_);
v_res_4047_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2(v_as_4032_, v_sz_boxed_4045_, v_i_boxed_4046_, v_b_4035_, v___y_4036_, v___y_4037_, v___y_4038_, v___y_4039_, v___y_4040_, v___y_4041_, v___y_4042_, v___y_4043_);
lean_dec(v___y_4043_);
lean_dec_ref(v___y_4042_);
lean_dec(v___y_4041_);
lean_dec_ref(v___y_4040_);
lean_dec(v___y_4039_);
lean_dec_ref(v___y_4038_);
lean_dec(v___y_4037_);
lean_dec_ref(v___y_4036_);
lean_dec_ref(v_as_4032_);
return v_res_4047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg(uint8_t v___x_4048_, size_t v_sz_4049_, size_t v_i_4050_, lean_object* v_bs_4051_, lean_object* v___y_4052_, lean_object* v___y_4053_, lean_object* v___y_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_){
_start:
{
uint8_t v___x_4059_; 
v___x_4059_ = lean_usize_dec_lt(v_i_4050_, v_sz_4049_);
if (v___x_4059_ == 0)
{
lean_object* v___x_4060_; 
v___x_4060_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4060_, 0, v_bs_4051_);
return v___x_4060_;
}
else
{
lean_object* v_v_4061_; lean_object* v___x_4062_; lean_object* v___x_4063_; 
v_v_4061_ = lean_array_uget_borrowed(v_bs_4051_, v_i_4050_);
v___x_4062_ = lean_box(0);
lean_inc(v_v_4061_);
v___x_4063_ = l_Lean_Elab_Term_elabTerm(v_v_4061_, v___x_4062_, v___x_4048_, v___x_4048_, v___y_4052_, v___y_4053_, v___y_4054_, v___y_4055_, v___y_4056_, v___y_4057_);
if (lean_obj_tag(v___x_4063_) == 0)
{
lean_object* v_a_4064_; lean_object* v___x_4065_; lean_object* v_bs_x27_4066_; size_t v___x_4067_; size_t v___x_4068_; lean_object* v___x_4069_; 
v_a_4064_ = lean_ctor_get(v___x_4063_, 0);
lean_inc(v_a_4064_);
lean_dec_ref_known(v___x_4063_, 1);
v___x_4065_ = lean_unsigned_to_nat(0u);
v_bs_x27_4066_ = lean_array_uset(v_bs_4051_, v_i_4050_, v___x_4065_);
v___x_4067_ = ((size_t)1ULL);
v___x_4068_ = lean_usize_add(v_i_4050_, v___x_4067_);
v___x_4069_ = lean_array_uset(v_bs_x27_4066_, v_i_4050_, v_a_4064_);
v_i_4050_ = v___x_4068_;
v_bs_4051_ = v___x_4069_;
goto _start;
}
else
{
lean_object* v_a_4071_; lean_object* v___x_4073_; uint8_t v_isShared_4074_; uint8_t v_isSharedCheck_4078_; 
lean_dec_ref(v_bs_4051_);
v_a_4071_ = lean_ctor_get(v___x_4063_, 0);
v_isSharedCheck_4078_ = !lean_is_exclusive(v___x_4063_);
if (v_isSharedCheck_4078_ == 0)
{
v___x_4073_ = v___x_4063_;
v_isShared_4074_ = v_isSharedCheck_4078_;
goto v_resetjp_4072_;
}
else
{
lean_inc(v_a_4071_);
lean_dec(v___x_4063_);
v___x_4073_ = lean_box(0);
v_isShared_4074_ = v_isSharedCheck_4078_;
goto v_resetjp_4072_;
}
v_resetjp_4072_:
{
lean_object* v___x_4076_; 
if (v_isShared_4074_ == 0)
{
v___x_4076_ = v___x_4073_;
goto v_reusejp_4075_;
}
else
{
lean_object* v_reuseFailAlloc_4077_; 
v_reuseFailAlloc_4077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4077_, 0, v_a_4071_);
v___x_4076_ = v_reuseFailAlloc_4077_;
goto v_reusejp_4075_;
}
v_reusejp_4075_:
{
return v___x_4076_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg___boxed(lean_object* v___x_4079_, lean_object* v_sz_4080_, lean_object* v_i_4081_, lean_object* v_bs_4082_, lean_object* v___y_4083_, lean_object* v___y_4084_, lean_object* v___y_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_){
_start:
{
uint8_t v___x_9287__boxed_4090_; size_t v_sz_boxed_4091_; size_t v_i_boxed_4092_; lean_object* v_res_4093_; 
v___x_9287__boxed_4090_ = lean_unbox(v___x_4079_);
v_sz_boxed_4091_ = lean_unbox_usize(v_sz_4080_);
lean_dec(v_sz_4080_);
v_i_boxed_4092_ = lean_unbox_usize(v_i_4081_);
lean_dec(v_i_4081_);
v_res_4093_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg(v___x_9287__boxed_4090_, v_sz_boxed_4091_, v_i_boxed_4092_, v_bs_4082_, v___y_4083_, v___y_4084_, v___y_4085_, v___y_4086_, v___y_4087_, v___y_4088_);
lean_dec(v___y_4088_);
lean_dec_ref(v___y_4087_);
lean_dec(v___y_4086_);
lean_dec_ref(v___y_4085_);
lean_dec(v___y_4084_);
lean_dec_ref(v___y_4083_);
return v_res_4093_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2(void){
_start:
{
lean_object* v___x_4097_; lean_object* v___x_4098_; 
v___x_4097_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__1));
v___x_4098_ = l_Lean_MessageData_ofFormat(v___x_4097_);
return v___x_4098_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_4099_; lean_object* v___x_4100_; 
v___x_4099_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_Algebraize_addProperties_spec__7_spec__9_spec__11___redArg___closed__0));
v___x_4100_ = l_Lean_stringToMessageData(v___x_4099_);
return v___x_4100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0(lean_object* v___x_4101_, uint8_t v___x_4102_, lean_object* v___x_4103_, lean_object* v___x_4104_, lean_object* v___x_4105_, lean_object* v___x_4106_, lean_object* v___y_4107_, lean_object* v___y_4108_, lean_object* v___y_4109_, lean_object* v___y_4110_, lean_object* v___y_4111_, lean_object* v___y_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_){
_start:
{
lean_object* v___x_4116_; 
v___x_4116_ = lp_mathlib_Mathlib_Tactic_Algebraize_elabAlgebraizeConfig___redArg(v___x_4101_, v___x_4102_, v___x_4102_, v___y_4107_, v___y_4113_, v___y_4114_);
if (lean_obj_tag(v___x_4116_) == 0)
{
lean_object* v_a_4117_; lean_object* v___y_4119_; lean_object* v___y_4120_; lean_object* v___y_4121_; lean_object* v___y_4122_; lean_object* v___y_4123_; lean_object* v___y_4124_; lean_object* v___y_4125_; lean_object* v___y_4126_; lean_object* v___y_4127_; lean_object* v_t_4144_; lean_object* v___y_4145_; lean_object* v___y_4146_; lean_object* v___y_4147_; lean_object* v___y_4148_; lean_object* v___y_4149_; lean_object* v___y_4150_; lean_object* v___y_4151_; lean_object* v___y_4152_; uint8_t v___x_4157_; 
v_a_4117_ = lean_ctor_get(v___x_4116_, 0);
lean_inc(v_a_4117_);
lean_dec_ref_known(v___x_4116_, 1);
lean_inc(v___x_4104_);
v___x_4157_ = l_Lean_Syntax_isOfKind(v___x_4104_, v___x_4105_);
if (v___x_4157_ == 0)
{
lean_object* v___x_4158_; lean_object* v___x_4159_; lean_object* v_a_4160_; lean_object* v___x_4162_; uint8_t v_isShared_4163_; uint8_t v_isSharedCheck_4167_; 
lean_dec(v_a_4117_);
lean_dec(v___x_4104_);
v___x_4158_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__3);
v___x_4159_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v___x_4158_, v___y_4111_, v___y_4112_, v___y_4113_, v___y_4114_);
v_a_4160_ = lean_ctor_get(v___x_4159_, 0);
v_isSharedCheck_4167_ = !lean_is_exclusive(v___x_4159_);
if (v_isSharedCheck_4167_ == 0)
{
v___x_4162_ = v___x_4159_;
v_isShared_4163_ = v_isSharedCheck_4167_;
goto v_resetjp_4161_;
}
else
{
lean_inc(v_a_4160_);
lean_dec(v___x_4159_);
v___x_4162_ = lean_box(0);
v_isShared_4163_ = v_isSharedCheck_4167_;
goto v_resetjp_4161_;
}
v_resetjp_4161_:
{
lean_object* v___x_4165_; 
if (v_isShared_4163_ == 0)
{
v___x_4165_ = v___x_4162_;
goto v_reusejp_4164_;
}
else
{
lean_object* v_reuseFailAlloc_4166_; 
v_reuseFailAlloc_4166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4166_, 0, v_a_4160_);
v___x_4165_ = v_reuseFailAlloc_4166_;
goto v_reusejp_4164_;
}
v_reusejp_4164_:
{
return v___x_4165_;
}
}
}
else
{
lean_object* v___x_4168_; lean_object* v___x_4169_; lean_object* v___x_4170_; size_t v_sz_4171_; size_t v___x_4172_; lean_object* v___x_4173_; 
v___x_4168_ = l_Lean_Syntax_getArg(v___x_4104_, v___x_4106_);
v___x_4169_ = l_Lean_Syntax_getArgs(v___x_4168_);
lean_dec(v___x_4168_);
v___x_4170_ = l_Lean_Syntax_TSepArray_getElems___redArg(v___x_4169_);
lean_dec_ref(v___x_4169_);
v_sz_4171_ = lean_array_size(v___x_4170_);
v___x_4172_ = ((size_t)0ULL);
v___x_4173_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg(v___x_4102_, v_sz_4171_, v___x_4172_, v___x_4170_, v___y_4109_, v___y_4110_, v___y_4111_, v___y_4112_, v___y_4113_, v___y_4114_);
if (lean_obj_tag(v___x_4173_) == 0)
{
lean_object* v_a_4174_; 
v_a_4174_ = lean_ctor_get(v___x_4173_, 0);
lean_inc(v_a_4174_);
lean_dec_ref_known(v___x_4173_, 1);
v_t_4144_ = v_a_4174_;
v___y_4145_ = v___y_4107_;
v___y_4146_ = v___y_4108_;
v___y_4147_ = v___y_4109_;
v___y_4148_ = v___y_4110_;
v___y_4149_ = v___y_4111_;
v___y_4150_ = v___y_4112_;
v___y_4151_ = v___y_4113_;
v___y_4152_ = v___y_4114_;
goto v___jp_4143_;
}
else
{
lean_object* v_a_4175_; lean_object* v___x_4177_; uint8_t v_isShared_4178_; uint8_t v_isSharedCheck_4182_; 
lean_dec(v_a_4117_);
lean_dec(v___x_4104_);
v_a_4175_ = lean_ctor_get(v___x_4173_, 0);
v_isSharedCheck_4182_ = !lean_is_exclusive(v___x_4173_);
if (v_isSharedCheck_4182_ == 0)
{
v___x_4177_ = v___x_4173_;
v_isShared_4178_ = v_isSharedCheck_4182_;
goto v_resetjp_4176_;
}
else
{
lean_inc(v_a_4175_);
lean_dec(v___x_4173_);
v___x_4177_ = lean_box(0);
v_isShared_4178_ = v_isSharedCheck_4182_;
goto v_resetjp_4176_;
}
v_resetjp_4176_:
{
lean_object* v___x_4180_; 
if (v_isShared_4178_ == 0)
{
v___x_4180_ = v___x_4177_;
goto v_reusejp_4179_;
}
else
{
lean_object* v_reuseFailAlloc_4181_; 
v_reuseFailAlloc_4181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4181_, 0, v_a_4175_);
v___x_4180_ = v_reuseFailAlloc_4181_;
goto v_reusejp_4179_;
}
v_reusejp_4179_:
{
return v___x_4180_;
}
}
}
}
v___jp_4118_:
{
lean_object* v___x_4128_; size_t v_sz_4129_; size_t v___x_4130_; lean_object* v___x_4131_; 
v___x_4128_ = lean_box(0);
v_sz_4129_ = lean_array_size(v___y_4119_);
v___x_4130_ = ((size_t)0ULL);
v___x_4131_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__1(v___y_4119_, v_sz_4129_, v___x_4130_, v___x_4128_, v___y_4120_, v___y_4121_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_);
if (lean_obj_tag(v___x_4131_) == 0)
{
lean_object* v___x_4132_; 
lean_dec_ref_known(v___x_4131_, 1);
v___x_4132_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__2(v___y_4119_, v_sz_4129_, v___x_4130_, v___x_4128_, v___y_4120_, v___y_4121_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_);
if (lean_obj_tag(v___x_4132_) == 0)
{
lean_object* v___x_4134_; uint8_t v_isShared_4135_; uint8_t v_isSharedCheck_4141_; 
v_isSharedCheck_4141_ = !lean_is_exclusive(v___x_4132_);
if (v_isSharedCheck_4141_ == 0)
{
lean_object* v_unused_4142_; 
v_unused_4142_ = lean_ctor_get(v___x_4132_, 0);
lean_dec(v_unused_4142_);
v___x_4134_ = v___x_4132_;
v_isShared_4135_ = v_isSharedCheck_4141_;
goto v_resetjp_4133_;
}
else
{
lean_dec(v___x_4132_);
v___x_4134_ = lean_box(0);
v_isShared_4135_ = v_isSharedCheck_4141_;
goto v_resetjp_4133_;
}
v_resetjp_4133_:
{
uint8_t v___x_4136_; 
v___x_4136_ = lean_unbox(v_a_4117_);
lean_dec(v_a_4117_);
if (v___x_4136_ == 0)
{
lean_object* v___x_4138_; 
lean_dec_ref(v___y_4119_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 0, v___x_4128_);
v___x_4138_ = v___x_4134_;
goto v_reusejp_4137_;
}
else
{
lean_object* v_reuseFailAlloc_4139_; 
v_reuseFailAlloc_4139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4139_, 0, v___x_4128_);
v___x_4138_ = v_reuseFailAlloc_4139_;
goto v_reusejp_4137_;
}
v_reusejp_4137_:
{
return v___x_4138_;
}
}
else
{
lean_object* v___x_4140_; 
lean_del_object(v___x_4134_);
v___x_4140_ = lp_mathlib_Mathlib_Tactic_Algebraize_addProperties(v___y_4119_, v___y_4120_, v___y_4121_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_);
return v___x_4140_;
}
}
}
else
{
lean_dec_ref(v___y_4119_);
lean_dec(v_a_4117_);
return v___x_4132_;
}
}
else
{
lean_dec_ref(v___y_4119_);
lean_dec(v_a_4117_);
return v___x_4131_;
}
}
v___jp_4143_:
{
lean_object* v___x_4153_; uint8_t v___x_4154_; 
v___x_4153_ = lean_array_get_size(v_t_4144_);
v___x_4154_ = lean_nat_dec_eq(v___x_4153_, v___x_4103_);
if (v___x_4154_ == 0)
{
lean_dec(v___x_4104_);
v___y_4119_ = v_t_4144_;
v___y_4120_ = v___y_4145_;
v___y_4121_ = v___y_4146_;
v___y_4122_ = v___y_4147_;
v___y_4123_ = v___y_4148_;
v___y_4124_ = v___y_4149_;
v___y_4125_ = v___y_4150_;
v___y_4126_ = v___y_4151_;
v___y_4127_ = v___y_4152_;
goto v___jp_4118_;
}
else
{
lean_object* v___x_4155_; lean_object* v___x_4156_; 
v___x_4155_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___closed__2);
v___x_4156_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg(v___x_4104_, v___x_4155_, v___y_4149_, v___y_4150_, v___y_4151_, v___y_4152_);
lean_dec(v___x_4104_);
if (lean_obj_tag(v___x_4156_) == 0)
{
lean_dec_ref_known(v___x_4156_, 1);
v___y_4119_ = v_t_4144_;
v___y_4120_ = v___y_4145_;
v___y_4121_ = v___y_4146_;
v___y_4122_ = v___y_4147_;
v___y_4123_ = v___y_4148_;
v___y_4124_ = v___y_4149_;
v___y_4125_ = v___y_4150_;
v___y_4126_ = v___y_4151_;
v___y_4127_ = v___y_4152_;
goto v___jp_4118_;
}
else
{
lean_dec_ref(v_t_4144_);
lean_dec(v_a_4117_);
return v___x_4156_;
}
}
}
}
else
{
lean_object* v_a_4183_; lean_object* v___x_4185_; uint8_t v_isShared_4186_; uint8_t v_isSharedCheck_4190_; 
lean_dec(v___x_4104_);
v_a_4183_ = lean_ctor_get(v___x_4116_, 0);
v_isSharedCheck_4190_ = !lean_is_exclusive(v___x_4116_);
if (v_isSharedCheck_4190_ == 0)
{
v___x_4185_ = v___x_4116_;
v_isShared_4186_ = v_isSharedCheck_4190_;
goto v_resetjp_4184_;
}
else
{
lean_inc(v_a_4183_);
lean_dec(v___x_4116_);
v___x_4185_ = lean_box(0);
v_isShared_4186_ = v_isSharedCheck_4190_;
goto v_resetjp_4184_;
}
v_resetjp_4184_:
{
lean_object* v___x_4188_; 
if (v_isShared_4186_ == 0)
{
v___x_4188_ = v___x_4185_;
goto v_reusejp_4187_;
}
else
{
lean_object* v_reuseFailAlloc_4189_; 
v_reuseFailAlloc_4189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4189_, 0, v_a_4183_);
v___x_4188_ = v_reuseFailAlloc_4189_;
goto v_reusejp_4187_;
}
v_reusejp_4187_:
{
return v___x_4188_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___boxed(lean_object* v___x_4191_, lean_object* v___x_4192_, lean_object* v___x_4193_, lean_object* v___x_4194_, lean_object* v___x_4195_, lean_object* v___x_4196_, lean_object* v___y_4197_, lean_object* v___y_4198_, lean_object* v___y_4199_, lean_object* v___y_4200_, lean_object* v___y_4201_, lean_object* v___y_4202_, lean_object* v___y_4203_, lean_object* v___y_4204_, lean_object* v___y_4205_){
_start:
{
uint8_t v___x_9363__boxed_4206_; lean_object* v_res_4207_; 
v___x_9363__boxed_4206_ = lean_unbox(v___x_4192_);
v_res_4207_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0(v___x_4191_, v___x_9363__boxed_4206_, v___x_4193_, v___x_4194_, v___x_4195_, v___x_4196_, v___y_4197_, v___y_4198_, v___y_4199_, v___y_4200_, v___y_4201_, v___y_4202_, v___y_4203_, v___y_4204_);
lean_dec(v___y_4204_);
lean_dec_ref(v___y_4203_);
lean_dec(v___y_4202_);
lean_dec_ref(v___y_4201_);
lean_dec(v___y_4200_);
lean_dec_ref(v___y_4199_);
lean_dec(v___y_4198_);
lean_dec_ref(v___y_4197_);
lean_dec(v___x_4196_);
lean_dec(v___x_4195_);
lean_dec(v___x_4193_);
return v_res_4207_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1(void){
_start:
{
lean_object* v___x_4209_; lean_object* v___x_4210_; 
v___x_4209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__0));
v___x_4210_ = l_Lean_stringToMessageData(v___x_4209_);
return v___x_4210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1(lean_object* v_x_4217_, lean_object* v_a_4218_, lean_object* v_a_4219_, lean_object* v_a_4220_, lean_object* v_a_4221_, lean_object* v_a_4222_, lean_object* v_a_4223_, lean_object* v_a_4224_, lean_object* v_a_4225_){
_start:
{
lean_object* v___y_4228_; lean_object* v___y_4229_; lean_object* v___y_4230_; lean_object* v___y_4231_; lean_object* v___x_4234_; uint8_t v___x_4235_; 
v___x_4234_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1));
lean_inc(v_x_4217_);
v___x_4235_ = l_Lean_Syntax_isOfKind(v_x_4217_, v___x_4234_);
if (v___x_4235_ == 0)
{
lean_object* v___x_4236_; 
lean_dec(v_x_4217_);
v___x_4236_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_4236_;
}
else
{
lean_object* v___x_4237_; lean_object* v___x_4238_; lean_object* v___x_4239_; lean_object* v___x_4240_; uint8_t v___x_4241_; 
v___x_4237_ = lean_unsigned_to_nat(0u);
v___x_4238_ = lean_unsigned_to_nat(1u);
v___x_4239_ = l_Lean_Syntax_getArg(v_x_4217_, v___x_4238_);
v___x_4240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3));
lean_inc(v___x_4239_);
v___x_4241_ = l_Lean_Syntax_isOfKind(v___x_4239_, v___x_4240_);
if (v___x_4241_ == 0)
{
if (v___x_4241_ == 0)
{
lean_object* v___x_4242_; 
lean_dec(v___x_4239_);
lean_dec(v_x_4217_);
v___x_4242_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_4242_;
}
else
{
lean_object* v___x_4243_; uint8_t v___x_4244_; 
v___x_4243_ = l_Lean_Syntax_getArg(v___x_4239_, v___x_4237_);
lean_dec(v___x_4239_);
v___x_4244_ = l_Lean_Syntax_matchesNull(v___x_4243_, v___x_4237_);
if (v___x_4244_ == 0)
{
lean_object* v___x_4245_; 
lean_dec(v_x_4217_);
v___x_4245_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_4245_;
}
else
{
lean_object* v___x_4246_; lean_object* v___x_4247_; uint8_t v___x_4248_; 
v___x_4246_ = lean_unsigned_to_nat(2u);
v___x_4247_ = l_Lean_Syntax_getArg(v_x_4217_, v___x_4246_);
lean_dec(v_x_4217_);
v___x_4248_ = l_Lean_Syntax_isNone(v___x_4247_);
if (v___x_4248_ == 0)
{
uint8_t v___x_4249_; 
v___x_4249_ = l_Lean_Syntax_matchesNull(v___x_4247_, v___x_4238_);
if (v___x_4249_ == 0)
{
lean_object* v___x_4250_; 
v___x_4250_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_4250_;
}
else
{
v___y_4228_ = v_a_4222_;
v___y_4229_ = v_a_4223_;
v___y_4230_ = v_a_4224_;
v___y_4231_ = v_a_4225_;
goto v___jp_4227_;
}
}
else
{
lean_dec(v___x_4247_);
v___y_4228_ = v_a_4222_;
v___y_4229_ = v_a_4223_;
v___y_4230_ = v_a_4224_;
v___y_4231_ = v_a_4225_;
goto v___jp_4227_;
}
}
}
}
else
{
lean_object* v___x_4251_; lean_object* v___x_4252_; uint8_t v___x_4253_; 
v___x_4251_ = lean_unsigned_to_nat(2u);
v___x_4252_ = l_Lean_Syntax_getArg(v_x_4217_, v___x_4251_);
lean_dec(v_x_4217_);
lean_inc(v___x_4252_);
v___x_4253_ = l_Lean_Syntax_matchesNull(v___x_4252_, v___x_4238_);
if (v___x_4253_ == 0)
{
lean_object* v___x_4254_; 
lean_dec(v___x_4252_);
lean_dec(v___x_4239_);
v___x_4254_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__0___redArg();
return v___x_4254_;
}
else
{
lean_object* v___x_4255_; lean_object* v___x_4256_; lean_object* v___x_4257_; lean_object* v___f_4258_; lean_object* v___x_4259_; 
v___x_4255_ = l_Lean_Syntax_getArg(v___x_4252_, v___x_4237_);
lean_dec(v___x_4252_);
v___x_4256_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_algebraizeTermSeq___closed__1));
v___x_4257_ = lean_box(v___x_4253_);
v___f_4258_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___lam__0___boxed), 15, 6);
lean_closure_set(v___f_4258_, 0, v___x_4239_);
lean_closure_set(v___f_4258_, 1, v___x_4257_);
lean_closure_set(v___f_4258_, 2, v___x_4237_);
lean_closure_set(v___f_4258_, 3, v___x_4255_);
lean_closure_set(v___f_4258_, 4, v___x_4256_);
lean_closure_set(v___f_4258_, 5, v___x_4238_);
v___x_4259_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4258_, v_a_4218_, v_a_4219_, v_a_4220_, v_a_4221_, v_a_4222_, v_a_4223_, v_a_4224_, v_a_4225_);
return v___x_4259_;
}
}
}
v___jp_4227_:
{
lean_object* v___x_4232_; lean_object* v___x_4233_; 
v___x_4232_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__1);
v___x_4233_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_Algebraize_addProperties_spec__5_spec__6_spec__8_spec__11_spec__15_spec__23___redArg(v___x_4232_, v___y_4228_, v___y_4229_, v___y_4230_, v___y_4231_);
return v___x_4233_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___boxed(lean_object* v_x_4260_, lean_object* v_a_4261_, lean_object* v_a_4262_, lean_object* v_a_4263_, lean_object* v_a_4264_, lean_object* v_a_4265_, lean_object* v_a_4266_, lean_object* v_a_4267_, lean_object* v_a_4268_, lean_object* v_a_4269_){
_start:
{
lean_object* v_res_4270_; 
v_res_4270_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1(v_x_4260_, v_a_4261_, v_a_4262_, v_a_4263_, v_a_4264_, v_a_4265_, v_a_4266_, v_a_4267_, v_a_4268_);
lean_dec(v_a_4268_);
lean_dec_ref(v_a_4267_);
lean_dec(v_a_4266_);
lean_dec_ref(v_a_4265_);
lean_dec(v_a_4264_);
lean_dec_ref(v_a_4263_);
lean_dec(v_a_4262_);
lean_dec_ref(v_a_4261_);
return v_res_4270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3(lean_object* v_ref_4271_, lean_object* v_msgData_4272_, lean_object* v___y_4273_, lean_object* v___y_4274_, lean_object* v___y_4275_, lean_object* v___y_4276_, lean_object* v___y_4277_, lean_object* v___y_4278_, lean_object* v___y_4279_, lean_object* v___y_4280_){
_start:
{
lean_object* v___x_4282_; 
v___x_4282_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___redArg(v_ref_4271_, v_msgData_4272_, v___y_4277_, v___y_4278_, v___y_4279_, v___y_4280_);
return v___x_4282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3___boxed(lean_object* v_ref_4283_, lean_object* v_msgData_4284_, lean_object* v___y_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_, lean_object* v___y_4289_, lean_object* v___y_4290_, lean_object* v___y_4291_, lean_object* v___y_4292_, lean_object* v___y_4293_){
_start:
{
lean_object* v_res_4294_; 
v_res_4294_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__3(v_ref_4283_, v_msgData_4284_, v___y_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_, v___y_4291_, v___y_4292_);
lean_dec(v___y_4292_);
lean_dec_ref(v___y_4291_);
lean_dec(v___y_4290_);
lean_dec_ref(v___y_4289_);
lean_dec(v___y_4288_);
lean_dec_ref(v___y_4287_);
lean_dec(v___y_4286_);
lean_dec_ref(v___y_4285_);
lean_dec(v_ref_4283_);
return v_res_4294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4(uint8_t v___x_4295_, size_t v_sz_4296_, size_t v_i_4297_, lean_object* v_bs_4298_, lean_object* v___y_4299_, lean_object* v___y_4300_, lean_object* v___y_4301_, lean_object* v___y_4302_, lean_object* v___y_4303_, lean_object* v___y_4304_, lean_object* v___y_4305_, lean_object* v___y_4306_){
_start:
{
lean_object* v___x_4308_; 
v___x_4308_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___redArg(v___x_4295_, v_sz_4296_, v_i_4297_, v_bs_4298_, v___y_4301_, v___y_4302_, v___y_4303_, v___y_4304_, v___y_4305_, v___y_4306_);
return v___x_4308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4___boxed(lean_object* v___x_4309_, lean_object* v_sz_4310_, lean_object* v_i_4311_, lean_object* v_bs_4312_, lean_object* v___y_4313_, lean_object* v___y_4314_, lean_object* v___y_4315_, lean_object* v___y_4316_, lean_object* v___y_4317_, lean_object* v___y_4318_, lean_object* v___y_4319_, lean_object* v___y_4320_, lean_object* v___y_4321_){
_start:
{
uint8_t v___x_9706__boxed_4322_; size_t v_sz_boxed_4323_; size_t v_i_boxed_4324_; lean_object* v_res_4325_; 
v___x_9706__boxed_4322_ = lean_unbox(v___x_4309_);
v_sz_boxed_4323_ = lean_unbox_usize(v_sz_4310_);
lean_dec(v_sz_4310_);
v_i_boxed_4324_ = lean_unbox_usize(v_i_4311_);
lean_dec(v_i_4311_);
v_res_4325_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1_spec__4(v___x_9706__boxed_4322_, v_sz_boxed_4323_, v_i_boxed_4324_, v_bs_4312_, v___y_4313_, v___y_4314_, v___y_4315_, v___y_4316_, v___y_4317_, v___y_4318_, v___y_4319_, v___y_4320_);
lean_dec(v___y_4320_);
lean_dec_ref(v___y_4319_);
lean_dec(v___y_4318_);
lean_dec_ref(v___y_4317_);
lean_dec(v___y_4316_);
lean_dec_ref(v___y_4315_);
lean_dec(v___y_4314_);
lean_dec_ref(v___y_4313_);
return v_res_4325_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7(void){
_start:
{
lean_object* v___x_4372_; lean_object* v___x_4373_; 
v___x_4372_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_elabAlgebraizeConfig_evalConfigItem___lam__0___closed__2));
v___x_4373_ = l_String_toRawSubstring_x27(v___x_4372_);
return v___x_4373_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9(void){
_start:
{
lean_object* v___x_4376_; 
v___x_4376_ = l_Array_mkArray0(lean_box(0));
return v___x_4376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1(lean_object* v_x_4379_, lean_object* v_a_4380_, lean_object* v_a_4381_){
_start:
{
lean_object* v___y_4383_; lean_object* v___y_4384_; lean_object* v___y_4385_; lean_object* v___y_4386_; lean_object* v___y_4387_; lean_object* v___y_4388_; lean_object* v___y_4389_; lean_object* v___y_4390_; lean_object* v_args_4396_; lean_object* v___y_4397_; lean_object* v___y_4398_; lean_object* v___x_4426_; uint8_t v___x_4427_; 
v___x_4426_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize__only_____00__closed__1));
lean_inc(v_x_4379_);
v___x_4427_ = l_Lean_Syntax_isOfKind(v_x_4379_, v___x_4426_);
if (v___x_4427_ == 0)
{
lean_object* v___x_4428_; lean_object* v___x_4429_; 
lean_dec(v_x_4379_);
v___x_4428_ = lean_box(1);
v___x_4429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4429_, 0, v___x_4428_);
lean_ctor_set(v___x_4429_, 1, v_a_4381_);
return v___x_4429_;
}
else
{
lean_object* v___x_4430_; lean_object* v___x_4431_; uint8_t v___x_4432_; 
v___x_4430_ = lean_unsigned_to_nat(1u);
v___x_4431_ = l_Lean_Syntax_getArg(v_x_4379_, v___x_4430_);
lean_dec(v_x_4379_);
v___x_4432_ = l_Lean_Syntax_isNone(v___x_4431_);
if (v___x_4432_ == 0)
{
uint8_t v___x_4433_; 
lean_inc(v___x_4431_);
v___x_4433_ = l_Lean_Syntax_matchesNull(v___x_4431_, v___x_4430_);
if (v___x_4433_ == 0)
{
lean_object* v___x_4434_; lean_object* v___x_4435_; 
lean_dec(v___x_4431_);
v___x_4434_ = lean_box(1);
v___x_4435_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4435_, 0, v___x_4434_);
lean_ctor_set(v___x_4435_, 1, v_a_4381_);
return v___x_4435_;
}
else
{
lean_object* v___x_4436_; lean_object* v_args_4437_; lean_object* v___x_4438_; 
v___x_4436_ = lean_unsigned_to_nat(0u);
v_args_4437_ = l_Lean_Syntax_getArg(v___x_4431_, v___x_4436_);
lean_dec(v___x_4431_);
v___x_4438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4438_, 0, v_args_4437_);
v_args_4396_ = v___x_4438_;
v___y_4397_ = v_a_4380_;
v___y_4398_ = v_a_4381_;
goto v___jp_4395_;
}
}
else
{
lean_object* v___x_4439_; 
lean_dec(v___x_4431_);
v___x_4439_ = lean_box(0);
v_args_4396_ = v___x_4439_;
v___y_4397_ = v_a_4380_;
v___y_4398_ = v_a_4381_;
goto v___jp_4395_;
}
}
v___jp_4382_:
{
lean_object* v___x_4391_; lean_object* v___x_4392_; lean_object* v___x_4393_; lean_object* v___x_4394_; 
lean_inc_ref(v___y_4389_);
v___x_4391_ = l_Array_append___redArg(v___y_4389_, v___y_4390_);
lean_dec_ref(v___y_4390_);
lean_inc(v___y_4386_);
v___x_4392_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4392_, 0, v___y_4386_);
lean_ctor_set(v___x_4392_, 1, v___y_4385_);
lean_ctor_set(v___x_4392_, 2, v___x_4391_);
lean_inc(v___y_4388_);
v___x_4393_ = l_Lean_Syntax_node3(v___y_4386_, v___y_4388_, v___y_4384_, v___y_4387_, v___x_4392_);
v___x_4394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4394_, 0, v___x_4393_);
lean_ctor_set(v___x_4394_, 1, v___y_4383_);
return v___x_4394_;
}
v___jp_4395_:
{
lean_object* v_quotContext_4399_; lean_object* v_currMacroScope_4400_; lean_object* v_ref_4401_; uint8_t v___x_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; lean_object* v___x_4405_; lean_object* v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4409_; lean_object* v___x_4410_; lean_object* v___x_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4416_; lean_object* v___x_4417_; lean_object* v___x_4418_; lean_object* v___x_4419_; lean_object* v___x_4420_; lean_object* v___x_4421_; lean_object* v___x_4422_; 
v_quotContext_4399_ = lean_ctor_get(v___y_4397_, 1);
v_currMacroScope_4400_ = lean_ctor_get(v___y_4397_, 2);
v_ref_4401_ = lean_ctor_get(v___y_4397_, 5);
v___x_4402_ = 0;
v___x_4403_ = l_Lean_SourceInfo_fromRef(v_ref_4401_, v___x_4402_);
v___x_4404_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAlgebraize_____00__closed__1));
v___x_4405_ = ((lean_object*)(lp_mathlib_Lean_Attr_algebraizeGetParam___closed__9));
lean_inc_n(v___x_4403_, 7);
v___x_4406_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4406_, 0, v___x_4403_);
lean_ctor_set(v___x_4406_, 1, v___x_4405_);
v___x_4407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______elabRules__Mathlib__Tactic__tacticAlgebraize______1___closed__3));
v___x_4408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__1));
v___x_4409_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__3));
v___x_4410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__5));
v___x_4411_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__6));
v___x_4412_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4412_, 0, v___x_4403_);
lean_ctor_set(v___x_4412_, 1, v___x_4411_);
v___x_4413_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__7);
v___x_4414_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__8));
lean_inc(v_currMacroScope_4400_);
lean_inc(v_quotContext_4399_);
v___x_4415_ = l_Lean_addMacroScope(v_quotContext_4399_, v___x_4414_, v_currMacroScope_4400_);
v___x_4416_ = lean_box(0);
v___x_4417_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4417_, 0, v___x_4403_);
lean_ctor_set(v___x_4417_, 1, v___x_4413_);
lean_ctor_set(v___x_4417_, 2, v___x_4415_);
lean_ctor_set(v___x_4417_, 3, v___x_4416_);
v___x_4418_ = l_Lean_Syntax_node2(v___x_4403_, v___x_4410_, v___x_4412_, v___x_4417_);
v___x_4419_ = l_Lean_Syntax_node1(v___x_4403_, v___x_4409_, v___x_4418_);
v___x_4420_ = l_Lean_Syntax_node1(v___x_4403_, v___x_4408_, v___x_4419_);
v___x_4421_ = l_Lean_Syntax_node1(v___x_4403_, v___x_4407_, v___x_4420_);
v___x_4422_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__9);
if (lean_obj_tag(v_args_4396_) == 1)
{
lean_object* v_val_4423_; lean_object* v___x_4424_; 
v_val_4423_ = lean_ctor_get(v_args_4396_, 0);
lean_inc(v_val_4423_);
lean_dec_ref_known(v_args_4396_, 1);
v___x_4424_ = l_Array_mkArray1___redArg(v_val_4423_);
v___y_4383_ = v___y_4398_;
v___y_4384_ = v___x_4406_;
v___y_4385_ = v___x_4408_;
v___y_4386_ = v___x_4403_;
v___y_4387_ = v___x_4421_;
v___y_4388_ = v___x_4404_;
v___y_4389_ = v___x_4422_;
v___y_4390_ = v___x_4424_;
goto v___jp_4382_;
}
else
{
lean_object* v___x_4425_; 
lean_dec(v_args_4396_);
v___x_4425_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___closed__10));
v___y_4383_ = v___y_4398_;
v___y_4384_ = v___x_4406_;
v___y_4385_ = v___x_4408_;
v___y_4386_ = v___x_4403_;
v___y_4387_ = v___x_4421_;
v___y_4388_ = v___x_4404_;
v___y_4389_ = v___x_4422_;
v___y_4390_ = v___x_4425_;
goto v___jp_4382_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1___boxed(lean_object* v_x_4440_, lean_object* v_a_4441_, lean_object* v_a_4442_){
_start:
{
lean_object* v_res_4443_; 
v_res_4443_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Algebraize______macroRules__Mathlib__Tactic__tacticAlgebraize__only______1(v_x_4440_, v_a_4441_, v_a_4442_);
lean_dec_ref(v_a_4441_);
return v_res_4443_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Algebraize(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Algebraize(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Lean_Attr_initFn_00___x40_Mathlib_Tactic_Algebraize_3251003831____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Lean_Attr_algebraizeAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Lean_Attr_algebraizeAttr);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig_default = _init_lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig_default();
lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig = _init_lp_mathlib_Mathlib_Tactic_Algebraize_instInhabitedConfig();
lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Algebraize_0__Mathlib_Tactic_Algebraize_instEvalExprConfig);
lp_mathlib_Mathlib_Tactic_tacticAlgebraize____ = _init_lp_mathlib_Mathlib_Tactic_tacticAlgebraize____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticAlgebraize____);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Algebraize(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Algebraize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Algebraize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Algebraize(builtin);
}
#ifdef __cplusplus
}
#endif
