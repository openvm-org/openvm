// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.Unfold
// Imports: public import Init public meta import Init public import Mathlib.Tactic.NthRewrite public import ProofWidgets.Component.Basic public import Mathlib.Tactic.ClickSuggestions.Util
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
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Name_isInternalDetail(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_Meta_ParamInfo_isExplicit(lean_object*);
lean_object* l_Lean_Meta_whnfCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* l_Lean_Expr_etaExpandedStrict_x3f(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Meta_reduceNat_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_reduceNative_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Environment_getProjectionFnInfo_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Meta_defaultInstanceExtension;
extern lean_object* l_Lean_Meta_instInhabitedDefaultInstances_default;
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_unfoldDefinition_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_project_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_levelMVarToParam___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_runTermElabM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(78, 102, 233, 39, 129, 161, 235, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_=_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(167, 251, 107, 62, 223, 239, 203, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "from"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fromTerm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(64, 243, 96, 22, 30, 196, 76, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__6_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " unfold ("};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ") "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "ClickSuggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "command#unfold\?_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(231, 102, 210, 246, 204, 42, 207, 225)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(165, 254, 144, 57, 195, 128, 128, 129)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "#unfold\? "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 2, .m_data = "· "};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Unfolds for "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "No unfolds found for "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg(lean_object* v_declName_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_4_; lean_object* v_env_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_4_ = lean_st_ref_get(v___y_2_);
v_env_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc_ref(v_env_5_);
lean_dec(v___x_4_);
v___x_6_ = l_Lean_Environment_getProjectionFnInfo_x3f(v_env_5_, v_declName_1_);
v___x_7_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_7_, 0, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg___boxed(lean_object* v_declName_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg(v_declName_8_, v___y_9_);
lean_dec(v___y_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0(lean_object* v_declName_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg(v_declName_12_, v___y_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___boxed(lean_object* v_declName_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0(v_declName_19_, v___y_20_, v___y_21_, v___y_22_, v___y_23_);
lean_dec(v___y_23_);
lean_dec_ref(v___y_22_);
lean_dec(v___y_21_);
lean_dec_ref(v___y_20_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1(lean_object* v___x_26_, uint8_t v___x_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
if (lean_obj_tag(v_a_28_) == 0)
{
lean_object* v___x_30_; 
lean_dec_ref(v___x_26_);
v___x_30_ = l_List_reverse___redArg(v_a_29_);
return v___x_30_;
}
else
{
lean_object* v_head_31_; lean_object* v_tail_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_43_; 
v_head_31_ = lean_ctor_get(v_a_28_, 0);
v_tail_32_ = lean_ctor_get(v_a_28_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v_a_28_);
if (v_isSharedCheck_43_ == 0)
{
v___x_34_ = v_a_28_;
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_tail_32_);
lean_inc(v_head_31_);
lean_dec(v_a_28_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v_fst_36_; uint8_t v___x_37_; 
v_fst_36_ = lean_ctor_get(v_head_31_, 0);
lean_inc(v_fst_36_);
lean_inc_ref(v___x_26_);
v___x_37_ = l_Lean_Environment_contains(v___x_26_, v_fst_36_, v___x_27_);
if (v___x_37_ == 0)
{
lean_del_object(v___x_34_);
lean_dec(v_head_31_);
v_a_28_ = v_tail_32_;
goto _start;
}
else
{
lean_object* v___x_40_; 
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 1, v_a_29_);
v___x_40_ = v___x_34_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_head_31_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_a_29_);
v___x_40_ = v_reuseFailAlloc_42_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
v_a_28_ = v_tail_32_;
v_a_29_ = v___x_40_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1___boxed(lean_object* v___x_44_, lean_object* v___x_45_, lean_object* v_a_46_, lean_object* v_a_47_){
_start:
{
uint8_t v___x_5089__boxed_48_; lean_object* v_res_49_; 
v___x_5089__boxed_48_ = lean_unbox(v___x_45_);
v_res_49_ = lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1(v___x_44_, v___x_5089__boxed_48_, v_a_46_, v_a_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg(lean_object* v_className_50_, lean_object* v___y_51_){
_start:
{
lean_object* v___x_53_; lean_object* v_env_54_; lean_object* v___y_56_; lean_object* v___x_62_; lean_object* v_toEnvExtension_63_; lean_object* v_asyncMode_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v_defaultInstances_68_; lean_object* v___x_69_; 
v___x_53_ = lean_st_ref_get(v___y_51_);
v_env_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc_ref_n(v_env_54_, 2);
lean_dec(v___x_53_);
v___x_62_ = l_Lean_Meta_defaultInstanceExtension;
v_toEnvExtension_63_ = lean_ctor_get(v___x_62_, 0);
v_asyncMode_64_ = lean_ctor_get(v_toEnvExtension_63_, 2);
v___x_65_ = l_Lean_Meta_instInhabitedDefaultInstances_default;
v___x_66_ = lean_box(0);
v___x_67_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_65_, v___x_62_, v_env_54_, v_asyncMode_64_, v___x_66_);
v_defaultInstances_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc(v_defaultInstances_68_);
lean_dec(v___x_67_);
v___x_69_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_defaultInstances_68_, v_className_50_);
lean_dec(v_defaultInstances_68_);
if (lean_obj_tag(v___x_69_) == 0)
{
lean_object* v___x_70_; 
v___x_70_ = lean_box(0);
v___y_56_ = v___x_70_;
goto v___jp_55_;
}
else
{
lean_object* v_val_71_; 
v_val_71_ = lean_ctor_get(v___x_69_, 0);
lean_inc(v_val_71_);
lean_dec_ref_known(v___x_69_, 1);
v___y_56_ = v_val_71_;
goto v___jp_55_;
}
v___jp_55_:
{
uint8_t v_isExporting_57_; 
v_isExporting_57_ = lean_ctor_get_uint8(v_env_54_, sizeof(void*)*8);
if (v_isExporting_57_ == 0)
{
lean_object* v___x_58_; 
lean_dec_ref(v_env_54_);
v___x_58_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_58_, 0, v___y_56_);
return v___x_58_;
}
else
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = lean_box(0);
v___x_60_ = lp_mathlib_List_filterTR_loop___at___00Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1_spec__1(v_env_54_, v_isExporting_57_, v___y_56_, v___x_59_);
v___x_61_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
return v___x_61_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg___boxed(lean_object* v_className_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg(v_className_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec(v_className_72_);
return v_res_75_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2(lean_object* v_declName_76_, lean_object* v_x_77_){
_start:
{
if (lean_obj_tag(v_x_77_) == 0)
{
uint8_t v___x_78_; 
v___x_78_ = 0;
return v___x_78_;
}
else
{
lean_object* v_head_79_; lean_object* v_tail_80_; lean_object* v_fst_81_; uint8_t v___x_82_; 
v_head_79_ = lean_ctor_get(v_x_77_, 0);
v_tail_80_ = lean_ctor_get(v_x_77_, 1);
v_fst_81_ = lean_ctor_get(v_head_79_, 0);
v___x_82_ = lean_name_eq(v_fst_81_, v_declName_76_);
if (v___x_82_ == 0)
{
v_x_77_ = v_tail_80_;
goto _start;
}
else
{
return v___x_82_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2___boxed(lean_object* v_declName_84_, lean_object* v_x_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2(v_declName_84_, v_x_85_);
lean_dec(v_x_85_);
lean_dec(v_declName_84_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0(void){
_start:
{
lean_object* v___x_88_; lean_object* v_dummy_89_; 
v___x_88_ = lean_box(0);
v_dummy_89_ = l_Lean_Expr_sort___override(v___x_88_);
return v_dummy_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f(lean_object* v_e_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = l_Lean_Expr_getAppFn(v_e_90_);
if (lean_obj_tag(v___x_99_) == 4)
{
lean_object* v_declName_100_; lean_object* v___x_101_; lean_object* v_a_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_207_; 
v_declName_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_declName_100_);
lean_dec_ref_known(v___x_99_, 2);
v___x_101_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__0___redArg(v_declName_100_, v_a_94_);
v_a_102_ = lean_ctor_get(v___x_101_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_207_ == 0)
{
v___x_104_ = v___x_101_;
v_isShared_105_ = v_isSharedCheck_207_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_a_102_);
lean_dec(v___x_101_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_207_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
if (lean_obj_tag(v_a_102_) == 1)
{
lean_object* v_val_111_; uint8_t v_fromClass_112_; 
v_val_111_ = lean_ctor_get(v_a_102_, 0);
lean_inc(v_val_111_);
lean_dec_ref_known(v_a_102_, 1);
v_fromClass_112_ = lean_ctor_get_uint8(v_val_111_, sizeof(void*)*3);
if (v_fromClass_112_ == 1)
{
lean_object* v_ctorName_113_; lean_object* v___x_114_; lean_object* v_env_115_; uint8_t v___x_116_; lean_object* v___x_117_; 
lean_del_object(v___x_104_);
v_ctorName_113_ = lean_ctor_get(v_val_111_, 0);
lean_inc(v_ctorName_113_);
lean_dec(v_val_111_);
v___x_114_ = lean_st_ref_get(v_a_94_);
v_env_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc_ref(v_env_115_);
lean_dec(v___x_114_);
v___x_116_ = 0;
v___x_117_ = l_Lean_Environment_find_x3f(v_env_115_, v_ctorName_113_, v___x_116_);
if (lean_obj_tag(v___x_117_) == 1)
{
lean_object* v_val_118_; 
v_val_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc(v_val_118_);
lean_dec_ref_known(v___x_117_, 1);
if (lean_obj_tag(v_val_118_) == 6)
{
lean_object* v_val_119_; lean_object* v_induct_120_; lean_object* v___x_121_; lean_object* v_a_122_; lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_206_; 
v_val_119_ = lean_ctor_get(v_val_118_, 0);
lean_inc_ref(v_val_119_);
lean_dec_ref_known(v_val_118_, 1);
v_induct_120_ = lean_ctor_get(v_val_119_, 1);
lean_inc(v_induct_120_);
lean_dec_ref(v_val_119_);
v___x_121_ = lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg(v_induct_120_, v_a_94_);
lean_dec(v_induct_120_);
v_a_122_ = lean_ctor_get(v___x_121_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_121_);
if (v_isSharedCheck_206_ == 0)
{
v___x_124_ = v___x_121_;
v_isShared_125_ = v_isSharedCheck_206_;
goto v_resetjp_123_;
}
else
{
lean_inc(v_a_122_);
lean_dec(v___x_121_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_206_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
uint8_t v___x_126_; 
v___x_126_ = l_List_isEmpty___redArg(v_a_122_);
if (v___x_126_ == 0)
{
lean_object* v_keyedConfig_127_; uint8_t v_trackZetaDelta_128_; lean_object* v_zetaDeltaSet_129_; lean_object* v_lctx_130_; lean_object* v_localInstances_131_; lean_object* v_defEqCtx_x3f_132_; lean_object* v_synthPendingDepth_133_; lean_object* v_customCanUnfoldPredicate_x3f_134_; uint8_t v_univApprox_135_; uint8_t v_inTypeClassResolution_136_; uint8_t v_cacheInferType_137_; uint8_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
lean_del_object(v___x_124_);
v_keyedConfig_127_ = lean_ctor_get(v_a_91_, 0);
v_trackZetaDelta_128_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7);
v_zetaDeltaSet_129_ = lean_ctor_get(v_a_91_, 1);
v_lctx_130_ = lean_ctor_get(v_a_91_, 2);
v_localInstances_131_ = lean_ctor_get(v_a_91_, 3);
v_defEqCtx_x3f_132_ = lean_ctor_get(v_a_91_, 4);
v_synthPendingDepth_133_ = lean_ctor_get(v_a_91_, 5);
v_customCanUnfoldPredicate_x3f_134_ = lean_ctor_get(v_a_91_, 6);
v_univApprox_135_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_136_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 2);
v_cacheInferType_137_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 3);
v___x_138_ = 1;
lean_inc_ref(v_keyedConfig_127_);
v___x_139_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_138_, v_keyedConfig_127_);
lean_inc(v_customCanUnfoldPredicate_x3f_134_);
lean_inc(v_synthPendingDepth_133_);
lean_inc(v_defEqCtx_x3f_132_);
lean_inc_ref(v_localInstances_131_);
lean_inc_ref(v_lctx_130_);
lean_inc(v_zetaDeltaSet_129_);
v___x_140_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_zetaDeltaSet_129_);
lean_ctor_set(v___x_140_, 2, v_lctx_130_);
lean_ctor_set(v___x_140_, 3, v_localInstances_131_);
lean_ctor_set(v___x_140_, 4, v_defEqCtx_x3f_132_);
lean_ctor_set(v___x_140_, 5, v_synthPendingDepth_133_);
lean_ctor_set(v___x_140_, 6, v_customCanUnfoldPredicate_x3f_134_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*7, v_trackZetaDelta_128_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*7 + 1, v_univApprox_135_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*7 + 2, v_inTypeClassResolution_136_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*7 + 3, v_cacheInferType_137_);
v___x_141_ = l_Lean_Meta_unfoldDefinition_x3f(v_e_90_, v___x_126_, v___x_140_, v_a_92_, v_a_93_, v_a_94_);
lean_dec_ref_known(v___x_140_, 7);
if (lean_obj_tag(v___x_141_) == 0)
{
lean_object* v_a_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_201_; 
v_a_142_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_201_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_201_ == 0)
{
v___x_144_ = v___x_141_;
v_isShared_145_ = v_isSharedCheck_201_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_a_142_);
lean_dec(v___x_141_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_201_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
if (lean_obj_tag(v_a_142_) == 1)
{
lean_object* v_val_146_; lean_object* v___x_147_; 
v_val_146_ = lean_ctor_get(v_a_142_, 0);
lean_inc(v_val_146_);
lean_dec_ref_known(v_a_142_, 1);
v___x_147_ = l_Lean_Expr_getAppFn(v_val_146_);
if (lean_obj_tag(v___x_147_) == 11)
{
lean_object* v_idx_148_; lean_object* v_struct_149_; lean_object* v___x_150_; 
v_idx_148_ = lean_ctor_get(v___x_147_, 1);
lean_inc(v_idx_148_);
v_struct_149_ = lean_ctor_get(v___x_147_, 2);
lean_inc_ref(v_struct_149_);
lean_dec_ref_known(v___x_147_, 3);
v___x_150_ = l_Lean_Expr_getAppFn(v_struct_149_);
if (lean_obj_tag(v___x_150_) == 4)
{
lean_object* v_declName_151_; uint8_t v___x_152_; 
v_declName_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc(v_declName_151_);
lean_dec_ref_known(v___x_150_, 2);
v___x_152_ = lp_mathlib_List_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__2(v_declName_151_, v_a_122_);
lean_dec(v_a_122_);
lean_dec(v_declName_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_155_; 
lean_dec_ref(v_struct_149_);
lean_dec(v_idx_148_);
lean_dec(v_val_146_);
v___x_153_ = lean_box(0);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_153_);
v___x_155_ = v___x_144_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___x_153_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
else
{
uint8_t v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
lean_del_object(v___x_144_);
v___x_157_ = 3;
lean_inc_ref(v_keyedConfig_127_);
v___x_158_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_157_, v_keyedConfig_127_);
lean_inc(v_customCanUnfoldPredicate_x3f_134_);
lean_inc(v_synthPendingDepth_133_);
lean_inc(v_defEqCtx_x3f_132_);
lean_inc_ref(v_localInstances_131_);
lean_inc_ref(v_lctx_130_);
lean_inc(v_zetaDeltaSet_129_);
v___x_159_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v_zetaDeltaSet_129_);
lean_ctor_set(v___x_159_, 2, v_lctx_130_);
lean_ctor_set(v___x_159_, 3, v_localInstances_131_);
lean_ctor_set(v___x_159_, 4, v_defEqCtx_x3f_132_);
lean_ctor_set(v___x_159_, 5, v_synthPendingDepth_133_);
lean_ctor_set(v___x_159_, 6, v_customCanUnfoldPredicate_x3f_134_);
lean_ctor_set_uint8(v___x_159_, sizeof(void*)*7, v_trackZetaDelta_128_);
lean_ctor_set_uint8(v___x_159_, sizeof(void*)*7 + 1, v_univApprox_135_);
lean_ctor_set_uint8(v___x_159_, sizeof(void*)*7 + 2, v_inTypeClassResolution_136_);
lean_ctor_set_uint8(v___x_159_, sizeof(void*)*7 + 3, v_cacheInferType_137_);
v___x_160_ = l_Lean_Meta_project_x3f(v_struct_149_, v_idx_148_, v___x_159_, v_a_92_, v_a_93_, v_a_94_);
lean_dec_ref_known(v___x_159_, 7);
lean_dec(v_idx_148_);
if (lean_obj_tag(v___x_160_) == 0)
{
lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_188_; 
v_a_161_ = lean_ctor_get(v___x_160_, 0);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_188_ == 0)
{
v___x_163_ = v___x_160_;
v_isShared_164_ = v_isSharedCheck_188_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_160_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_188_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
if (lean_obj_tag(v_a_161_) == 1)
{
lean_object* v_val_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_183_; 
v_val_165_ = lean_ctor_get(v_a_161_, 0);
v_isSharedCheck_183_ = !lean_is_exclusive(v_a_161_);
if (v_isSharedCheck_183_ == 0)
{
v___x_167_ = v_a_161_;
v_isShared_168_ = v_isSharedCheck_183_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_val_165_);
lean_dec(v_a_161_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_183_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v_dummy_169_; lean_object* v_nargs_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_178_; 
v_dummy_169_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0);
v_nargs_170_ = l_Lean_Expr_getAppNumArgs(v_val_146_);
lean_inc(v_nargs_170_);
v___x_171_ = lean_mk_array(v_nargs_170_, v_dummy_169_);
v___x_172_ = lean_unsigned_to_nat(1u);
v___x_173_ = lean_nat_sub(v_nargs_170_, v___x_172_);
lean_dec(v_nargs_170_);
v___x_174_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_val_146_, v___x_171_, v___x_173_);
v___x_175_ = l_Lean_mkAppN(v_val_165_, v___x_174_);
lean_dec_ref(v___x_174_);
v___x_176_ = l_Lean_Expr_headBeta(v___x_175_);
if (v_isShared_168_ == 0)
{
lean_ctor_set(v___x_167_, 0, v___x_176_);
v___x_178_ = v___x_167_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_182_; 
v_reuseFailAlloc_182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_182_, 0, v___x_176_);
v___x_178_ = v_reuseFailAlloc_182_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
lean_object* v___x_180_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v___x_178_);
v___x_180_ = v___x_163_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v___x_178_);
v___x_180_ = v_reuseFailAlloc_181_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
return v___x_180_;
}
}
}
}
else
{
lean_object* v___x_184_; lean_object* v___x_186_; 
lean_dec(v_a_161_);
lean_dec(v_val_146_);
v___x_184_ = lean_box(0);
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v___x_184_);
v___x_186_ = v___x_163_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_184_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
else
{
lean_dec(v_val_146_);
return v___x_160_;
}
}
}
else
{
lean_object* v___x_189_; lean_object* v___x_191_; 
lean_dec_ref(v___x_150_);
lean_dec_ref(v_struct_149_);
lean_dec(v_idx_148_);
lean_dec(v_val_146_);
lean_dec(v_a_122_);
v___x_189_ = lean_box(0);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_189_);
v___x_191_ = v___x_144_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v___x_189_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
else
{
lean_object* v___x_193_; lean_object* v___x_195_; 
lean_dec_ref(v___x_147_);
lean_dec(v_val_146_);
lean_dec(v_a_122_);
v___x_193_ = lean_box(0);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_193_);
v___x_195_ = v___x_144_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_193_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
else
{
lean_object* v___x_197_; lean_object* v___x_199_; 
lean_dec(v_a_142_);
lean_dec(v_a_122_);
v___x_197_ = lean_box(0);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_197_);
v___x_199_ = v___x_144_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v___x_197_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
else
{
lean_dec(v_a_122_);
return v___x_141_;
}
}
else
{
lean_object* v___x_202_; lean_object* v___x_204_; 
lean_dec(v_a_122_);
lean_dec_ref(v_e_90_);
v___x_202_ = lean_box(0);
if (v_isShared_125_ == 0)
{
lean_ctor_set(v___x_124_, 0, v___x_202_);
v___x_204_ = v___x_124_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v___x_202_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
}
else
{
lean_dec(v_val_118_);
lean_dec_ref(v_e_90_);
goto v___jp_96_;
}
}
else
{
lean_dec(v___x_117_);
lean_dec_ref(v_e_90_);
goto v___jp_96_;
}
}
else
{
lean_dec(v_val_111_);
lean_dec_ref(v_e_90_);
goto v___jp_106_;
}
}
else
{
lean_dec(v_a_102_);
lean_dec_ref(v_e_90_);
goto v___jp_106_;
}
v___jp_106_:
{
lean_object* v___x_107_; lean_object* v___x_109_; 
v___x_107_ = lean_box(0);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 0, v___x_107_);
v___x_109_ = v___x_104_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_107_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
else
{
lean_object* v___x_208_; lean_object* v___x_209_; 
lean_dec_ref(v___x_99_);
lean_dec_ref(v_e_90_);
v___x_208_ = lean_box(0);
v___x_209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
return v___x_209_;
}
v___jp_96_:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
return v___x_98_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___boxed(lean_object* v_e_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f(v_e_210_, v_a_211_, v_a_212_, v_a_213_, v_a_214_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
lean_dec(v_a_212_);
lean_dec_ref(v_a_211_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1(lean_object* v_className_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___redArg(v_className_217_, v___y_221_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1___boxed(lean_object* v_className_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib_Lean_Meta_getDefaultInstances___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f_spec__1(v_className_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_);
lean_dec(v___y_228_);
lean_dec_ref(v___y_227_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec(v_className_224_);
return v_res_230_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = l_Lean_maxRecDepthErrorMessage;
v___x_237_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
return v___x_237_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__3);
v___x_239_ = l_Lean_MessageData_ofFormat(v___x_238_);
return v___x_239_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_240_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__4);
v___x_241_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__2));
v___x_242_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v___x_240_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg(lean_object* v_ref_243_){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_245_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___closed__5);
v___x_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_246_, 0, v_ref_243_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg___boxed(lean_object* v_ref_248_, lean_object* v___y_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg(v_ref_248_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0(lean_object* v_00_u03b1_251_, lean_object* v_ref_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg(v_ref_252_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___boxed(lean_object* v_00_u03b1_259_, lean_object* v_ref_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0(v_00_u03b1_259_, v_ref_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1(lean_object* v_acc_267_, lean_object* v_e_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_274_ = lean_array_push(v_acc_267_, v_e_268_);
v___x_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_275_, 0, v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1___boxed(lean_object* v_acc_276_, lean_object* v_e_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1(v_acc_276_, v_e_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v___y_278_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(lean_object* v_e_284_, lean_object* v_acc_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_){
_start:
{
lean_object* v___y_292_; lean_object* v_a_293_; lean_object* v___y_297_; lean_object* v_fileName_299_; lean_object* v_fileMap_300_; lean_object* v_options_301_; lean_object* v_currRecDepth_302_; lean_object* v_maxRecDepth_303_; lean_object* v_ref_304_; lean_object* v_currNamespace_305_; lean_object* v_openDecls_306_; lean_object* v_initHeartbeats_307_; lean_object* v_maxHeartbeats_308_; lean_object* v_quotContext_309_; lean_object* v_currMacroScope_310_; uint8_t v_diag_311_; lean_object* v_cancelTk_x3f_312_; uint8_t v_suppressElabErrors_313_; lean_object* v_inheritedTraceOptions_314_; lean_object* v___x_315_; lean_object* v___x_388_; uint8_t v___x_389_; 
v_fileName_299_ = lean_ctor_get(v_a_288_, 0);
v_fileMap_300_ = lean_ctor_get(v_a_288_, 1);
v_options_301_ = lean_ctor_get(v_a_288_, 2);
v_currRecDepth_302_ = lean_ctor_get(v_a_288_, 3);
v_maxRecDepth_303_ = lean_ctor_get(v_a_288_, 4);
v_ref_304_ = lean_ctor_get(v_a_288_, 5);
v_currNamespace_305_ = lean_ctor_get(v_a_288_, 6);
v_openDecls_306_ = lean_ctor_get(v_a_288_, 7);
v_initHeartbeats_307_ = lean_ctor_get(v_a_288_, 8);
v_maxHeartbeats_308_ = lean_ctor_get(v_a_288_, 9);
v_quotContext_309_ = lean_ctor_get(v_a_288_, 10);
v_currMacroScope_310_ = lean_ctor_get(v_a_288_, 11);
v_diag_311_ = lean_ctor_get_uint8(v_a_288_, sizeof(void*)*14);
v_cancelTk_x3f_312_ = lean_ctor_get(v_a_288_, 12);
v_suppressElabErrors_313_ = lean_ctor_get_uint8(v_a_288_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_314_ = lean_ctor_get(v_a_288_, 13);
lean_inc_ref(v_e_284_);
v___x_315_ = l_Lean_Expr_etaExpandedStrict_x3f(v_e_284_);
v___x_388_ = lean_unsigned_to_nat(0u);
v___x_389_ = lean_nat_dec_eq(v_maxRecDepth_303_, v___x_388_);
if (v___x_389_ == 0)
{
uint8_t v___x_390_; 
v___x_390_ = lean_nat_dec_eq(v_currRecDepth_302_, v_maxRecDepth_303_);
if (v___x_390_ == 0)
{
goto v___jp_316_;
}
else
{
lean_object* v___x_391_; 
lean_dec(v___x_315_);
lean_dec_ref(v_e_284_);
lean_inc(v_ref_304_);
v___x_391_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go_spec__0___redArg(v_ref_304_);
v___y_297_ = v___x_391_;
goto v___jp_296_;
}
}
else
{
goto v___jp_316_;
}
v___jp_291_:
{
uint8_t v___x_294_; 
v___x_294_ = l_Lean_Exception_isInterrupt(v_a_293_);
lean_dec_ref(v_a_293_);
if (v___x_294_ == 0)
{
lean_object* v___x_295_; 
lean_dec_ref(v___y_292_);
v___x_295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_295_, 0, v_acc_285_);
return v___x_295_;
}
else
{
lean_dec_ref(v_acc_285_);
return v___y_292_;
}
}
v___jp_296_:
{
if (lean_obj_tag(v___y_297_) == 0)
{
lean_dec_ref(v_acc_285_);
return v___y_297_;
}
else
{
lean_object* v_a_298_; 
v_a_298_ = lean_ctor_get(v___y_297_, 0);
lean_inc(v_a_298_);
v___y_292_ = v___y_297_;
v_a_293_ = v_a_298_;
goto v___jp_291_;
}
}
v___jp_316_:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_317_ = lean_unsigned_to_nat(1u);
v___x_318_ = lean_nat_add(v_currRecDepth_302_, v___x_317_);
lean_inc_ref(v_inheritedTraceOptions_314_);
lean_inc(v_cancelTk_x3f_312_);
lean_inc(v_currMacroScope_310_);
lean_inc(v_quotContext_309_);
lean_inc(v_maxHeartbeats_308_);
lean_inc(v_initHeartbeats_307_);
lean_inc(v_openDecls_306_);
lean_inc(v_currNamespace_305_);
lean_inc(v_ref_304_);
lean_inc(v_maxRecDepth_303_);
lean_inc_ref(v_options_301_);
lean_inc_ref(v_fileMap_300_);
lean_inc_ref(v_fileName_299_);
v___x_319_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_319_, 0, v_fileName_299_);
lean_ctor_set(v___x_319_, 1, v_fileMap_300_);
lean_ctor_set(v___x_319_, 2, v_options_301_);
lean_ctor_set(v___x_319_, 3, v___x_318_);
lean_ctor_set(v___x_319_, 4, v_maxRecDepth_303_);
lean_ctor_set(v___x_319_, 5, v_ref_304_);
lean_ctor_set(v___x_319_, 6, v_currNamespace_305_);
lean_ctor_set(v___x_319_, 7, v_openDecls_306_);
lean_ctor_set(v___x_319_, 8, v_initHeartbeats_307_);
lean_ctor_set(v___x_319_, 9, v_maxHeartbeats_308_);
lean_ctor_set(v___x_319_, 10, v_quotContext_309_);
lean_ctor_set(v___x_319_, 11, v_currMacroScope_310_);
lean_ctor_set(v___x_319_, 12, v_cancelTk_x3f_312_);
lean_ctor_set(v___x_319_, 13, v_inheritedTraceOptions_314_);
lean_ctor_set_uint8(v___x_319_, sizeof(void*)*14, v_diag_311_);
lean_ctor_set_uint8(v___x_319_, sizeof(void*)*14 + 1, v_suppressElabErrors_313_);
if (lean_obj_tag(v___x_315_) == 1)
{
lean_object* v_val_320_; lean_object* v___x_321_; 
lean_dec_ref(v_e_284_);
v_val_320_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_val_320_);
lean_dec_ref_known(v___x_315_, 1);
lean_inc_ref(v_acc_285_);
v___x_321_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0(v_acc_285_, v_val_320_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
lean_dec_ref_known(v___x_319_, 14);
v___y_297_ = v___x_321_;
goto v___jp_296_;
}
else
{
lean_object* v___x_322_; 
lean_dec(v___x_315_);
lean_inc_ref(v_e_284_);
v___x_322_ = l_Lean_Meta_reduceNat_x3f(v_e_284_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
if (lean_obj_tag(v___x_322_) == 0)
{
lean_object* v_a_323_; 
v_a_323_ = lean_ctor_get(v___x_322_, 0);
lean_inc(v_a_323_);
lean_dec_ref_known(v___x_322_, 1);
if (lean_obj_tag(v_a_323_) == 1)
{
lean_object* v_val_324_; lean_object* v___x_325_; 
lean_dec_ref(v_e_284_);
v_val_324_ = lean_ctor_get(v_a_323_, 0);
lean_inc(v_val_324_);
lean_dec_ref_known(v_a_323_, 1);
lean_inc_ref(v_acc_285_);
v___x_325_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1(v_acc_285_, v_val_324_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
lean_dec_ref_known(v___x_319_, 14);
v___y_297_ = v___x_325_;
goto v___jp_296_;
}
else
{
lean_object* v___x_326_; 
lean_dec(v_a_323_);
lean_inc_ref(v_e_284_);
v___x_326_ = l_Lean_Meta_reduceNative_x3f(v_e_284_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
if (lean_obj_tag(v___x_326_) == 0)
{
lean_object* v_a_327_; 
v_a_327_ = lean_ctor_get(v___x_326_, 0);
lean_inc(v_a_327_);
lean_dec_ref_known(v___x_326_, 1);
if (lean_obj_tag(v_a_327_) == 1)
{
lean_object* v_val_328_; lean_object* v___x_329_; 
lean_dec_ref(v_e_284_);
v_val_328_ = lean_ctor_get(v_a_327_, 0);
lean_inc(v_val_328_);
lean_dec_ref_known(v_a_327_, 1);
lean_inc_ref(v_acc_285_);
v___x_329_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__1(v_acc_285_, v_val_328_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
lean_dec_ref_known(v___x_319_, 14);
v___y_297_ = v___x_329_;
goto v___jp_296_;
}
else
{
lean_object* v___x_330_; 
lean_dec(v_a_327_);
lean_inc_ref(v_e_284_);
v___x_330_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f(v_e_284_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v_a_331_; 
v_a_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_a_331_);
lean_dec_ref_known(v___x_330_, 1);
if (lean_obj_tag(v_a_331_) == 1)
{
lean_object* v_val_332_; lean_object* v___x_333_; 
lean_dec_ref(v_e_284_);
v_val_332_ = lean_ctor_get(v_a_331_, 0);
lean_inc(v_val_332_);
lean_dec_ref_known(v_a_331_, 1);
v___x_333_ = l_Lean_Meta_whnfCore(v_val_332_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
if (lean_obj_tag(v___x_333_) == 0)
{
lean_object* v_a_334_; lean_object* v___x_335_; 
v_a_334_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_a_334_);
lean_dec_ref_known(v___x_333_, 1);
lean_inc_ref(v_acc_285_);
v___x_335_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(v_a_334_, v_acc_285_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
lean_dec_ref_known(v___x_319_, 14);
v___y_297_ = v___x_335_;
goto v___jp_296_;
}
else
{
lean_object* v_a_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_343_; 
lean_dec_ref_known(v___x_319_, 14);
v_a_336_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_343_ == 0)
{
v___x_338_ = v___x_333_;
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_a_336_);
lean_dec(v___x_333_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v___x_341_; 
lean_inc(v_a_336_);
if (v_isShared_339_ == 0)
{
v___x_341_ = v___x_338_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v_a_336_);
v___x_341_ = v_reuseFailAlloc_342_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
v___y_292_ = v___x_341_;
v_a_293_ = v_a_336_;
goto v___jp_291_;
}
}
}
}
else
{
uint8_t v___x_344_; lean_object* v___x_345_; 
lean_dec(v_a_331_);
v___x_344_ = 0;
v___x_345_ = l_Lean_Meta_unfoldDefinition_x3f(v_e_284_, v___x_344_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
if (lean_obj_tag(v___x_345_) == 0)
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_355_; 
v_a_346_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_355_ == 0)
{
v___x_348_ = v___x_345_;
v_isShared_349_ = v_isSharedCheck_355_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_345_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_355_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
if (lean_obj_tag(v_a_346_) == 1)
{
lean_object* v_val_350_; lean_object* v___x_351_; 
lean_del_object(v___x_348_);
v_val_350_ = lean_ctor_get(v_a_346_, 0);
lean_inc(v_val_350_);
lean_dec_ref_known(v_a_346_, 1);
lean_inc_ref(v_acc_285_);
v___x_351_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0(v_acc_285_, v_val_350_, v_a_286_, v_a_287_, v___x_319_, v_a_289_);
lean_dec_ref_known(v___x_319_, 14);
v___y_297_ = v___x_351_;
goto v___jp_296_;
}
else
{
lean_object* v___x_353_; 
lean_dec(v_a_346_);
lean_dec_ref_known(v___x_319_, 14);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 0, v_acc_285_);
v___x_353_ = v___x_348_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_acc_285_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
}
else
{
lean_object* v_a_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_363_; 
lean_dec_ref_known(v___x_319_, 14);
v_a_356_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_363_ == 0)
{
v___x_358_ = v___x_345_;
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_a_356_);
lean_dec(v___x_345_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_361_; 
lean_inc(v_a_356_);
if (v_isShared_359_ == 0)
{
v___x_361_ = v___x_358_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_356_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
v___y_292_ = v___x_361_;
v_a_293_ = v_a_356_;
goto v___jp_291_;
}
}
}
}
}
else
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_371_; 
lean_dec_ref_known(v___x_319_, 14);
lean_dec_ref(v_e_284_);
v_a_364_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_371_ == 0)
{
v___x_366_ = v___x_330_;
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_330_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v___x_369_; 
lean_inc(v_a_364_);
if (v_isShared_367_ == 0)
{
v___x_369_ = v___x_366_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_a_364_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
v___y_292_ = v___x_369_;
v_a_293_ = v_a_364_;
goto v___jp_291_;
}
}
}
}
}
else
{
lean_object* v_a_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_379_; 
lean_dec_ref_known(v___x_319_, 14);
lean_dec_ref(v_e_284_);
v_a_372_ = lean_ctor_get(v___x_326_, 0);
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_379_ == 0)
{
v___x_374_ = v___x_326_;
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_a_372_);
lean_dec(v___x_326_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v___x_377_; 
lean_inc(v_a_372_);
if (v_isShared_375_ == 0)
{
v___x_377_ = v___x_374_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_a_372_);
v___x_377_ = v_reuseFailAlloc_378_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
v___y_292_ = v___x_377_;
v_a_293_ = v_a_372_;
goto v___jp_291_;
}
}
}
}
}
else
{
lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_387_; 
lean_dec_ref_known(v___x_319_, 14);
lean_dec_ref(v_e_284_);
v_a_380_ = lean_ctor_get(v___x_322_, 0);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_322_);
if (v_isSharedCheck_387_ == 0)
{
v___x_382_ = v___x_322_;
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_322_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
lean_inc(v_a_380_);
if (v_isShared_383_ == 0)
{
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_380_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
v___y_292_ = v___x_385_;
v_a_293_ = v_a_380_;
goto v___jp_291_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0(lean_object* v_acc_392_, lean_object* v_e_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = l_Lean_Meta_whnfCore(v_e_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
lean_inc_n(v_a_400_, 2);
lean_dec_ref_known(v___x_399_, 1);
v___x_401_ = lean_array_push(v_acc_392_, v_a_400_);
v___x_402_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(v_a_400_, v___x_401_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
return v___x_402_;
}
else
{
lean_object* v_a_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_410_; 
lean_dec_ref(v_acc_392_);
v_a_403_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_410_ == 0)
{
v___x_405_ = v___x_399_;
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_a_403_);
lean_dec(v___x_399_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0___boxed(lean_object* v_acc_411_, lean_object* v_e_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___lam__0(v_acc_411_, v_e_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go___boxed(lean_object* v_e_419_, lean_object* v_acc_420_, lean_object* v_a_421_, lean_object* v_a_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(v_e_419_, v_acc_420_, v_a_421_, v_a_422_, v_a_423_, v_a_424_);
lean_dec(v_a_424_);
lean_dec_ref(v_a_423_);
lean_dec(v_a_422_);
lean_dec_ref(v_a_421_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds(lean_object* v_e_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_){
_start:
{
lean_object* v___x_435_; 
lean_inc_ref(v_e_429_);
v___x_435_ = l_Lean_Meta_whnfCore(v_e_429_, v_a_430_, v_a_431_, v_a_432_, v_a_433_);
if (lean_obj_tag(v___x_435_) == 0)
{
lean_object* v_a_436_; uint8_t v___x_437_; 
v_a_436_ = lean_ctor_get(v___x_435_, 0);
lean_inc(v_a_436_);
lean_dec_ref_known(v___x_435_, 1);
v___x_437_ = lean_expr_eqv(v_e_429_, v_a_436_);
lean_dec_ref(v_e_429_);
if (v___x_437_ == 0)
{
lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_438_ = lean_unsigned_to_nat(1u);
v___x_439_ = lean_mk_empty_array_with_capacity(v___x_438_);
lean_inc(v_a_436_);
v___x_440_ = lean_array_push(v___x_439_, v_a_436_);
v___x_441_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(v_a_436_, v___x_440_, v_a_430_, v_a_431_, v_a_432_, v_a_433_);
return v___x_441_;
}
else
{
lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_442_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___closed__0));
v___x_443_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds_go(v_a_436_, v___x_442_, v_a_430_, v_a_431_, v_a_432_, v_a_433_);
return v___x_443_;
}
}
else
{
lean_object* v_a_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_451_; 
lean_dec_ref(v_e_429_);
v_a_444_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_451_ == 0)
{
v___x_446_ = v___x_435_;
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_a_444_);
lean_dec(v___x_435_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_449_; 
if (v_isShared_447_ == 0)
{
v___x_449_ = v___x_446_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_a_444_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___boxed(lean_object* v_e_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds(v_e_452_, v_a_453_, v_a_454_, v_a_455_, v_a_456_);
lean_dec(v_a_456_);
lean_dec_ref(v_a_455_);
lean_dec(v_a_454_);
lean_dec_ref(v_a_453_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(lean_object* v_e_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_){
_start:
{
switch(lean_obj_tag(v_e_459_))
{
case 4:
{
lean_object* v_declName_465_; uint8_t v___x_466_; 
v_declName_465_ = lean_ctor_get(v_e_459_, 0);
lean_inc(v_declName_465_);
lean_dec_ref_known(v_e_459_, 2);
v___x_466_ = l_Lean_Name_isInternalDetail(v_declName_465_);
if (v___x_466_ == 0)
{
uint8_t v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_467_ = 1;
v___x_468_ = lean_box(v___x_467_);
v___x_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
return v___x_469_;
}
else
{
uint8_t v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_470_ = 0;
v___x_471_ = lean_box(v___x_470_);
v___x_472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
return v___x_472_;
}
}
case 11:
{
uint8_t v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
lean_dec_ref_known(v_e_459_, 3);
v___x_473_ = 0;
v___x_474_ = lean_box(v___x_473_);
v___x_475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
return v___x_475_;
}
case 5:
{
lean_object* v_dummy_476_; lean_object* v_nargs_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v_dummy_476_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfoldProjDefaultInst_x3f___closed__0);
v_nargs_477_ = l_Lean_Expr_getAppNumArgs(v_e_459_);
lean_inc(v_nargs_477_);
v___x_478_ = lean_mk_array(v_nargs_477_, v_dummy_476_);
v___x_479_ = lean_unsigned_to_nat(1u);
v___x_480_ = lean_nat_sub(v_nargs_477_, v___x_479_);
lean_dec(v_nargs_477_);
lean_inc_ref(v_e_459_);
v___x_481_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1(v_e_459_, v_e_459_, v___x_478_, v___x_480_, v_a_460_, v_a_461_, v_a_462_, v_a_463_);
lean_dec_ref_known(v_e_459_, 2);
return v___x_481_;
}
default: 
{
uint8_t v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
lean_dec_ref(v_e_459_);
v___x_482_ = 1;
v___x_483_ = lean_box(v___x_482_);
v___x_484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_484_, 0, v___x_483_);
return v___x_484_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg(lean_object* v_args_485_, lean_object* v_a_486_, uint8_t v_a_487_, lean_object* v_n_488_, lean_object* v_i_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_){
_start:
{
lean_object* v_zero_495_; uint8_t v_isZero_496_; 
v_zero_495_ = lean_unsigned_to_nat(0u);
v_isZero_496_ = lean_nat_dec_eq(v_i_489_, v_zero_495_);
if (v_isZero_496_ == 1)
{
lean_object* v___x_497_; lean_object* v___x_498_; 
lean_dec(v_i_489_);
v___x_497_ = lean_box(v_isZero_496_);
v___x_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_498_, 0, v___x_497_);
return v___x_498_;
}
else
{
lean_object* v_paramInfo_499_; lean_object* v_one_500_; lean_object* v_n_501_; lean_object* v___y_503_; uint8_t v_a_504_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_514_; uint8_t v___x_515_; 
v_paramInfo_499_ = lean_ctor_get(v_a_486_, 0);
v_one_500_ = lean_unsigned_to_nat(1u);
v_n_501_ = lean_nat_sub(v_i_489_, v_one_500_);
lean_dec(v_i_489_);
v___x_506_ = lean_nat_sub(v_n_488_, v_n_501_);
v___x_507_ = lean_nat_sub(v___x_506_, v_one_500_);
lean_dec(v___x_506_);
v___x_514_ = lean_array_get_size(v_paramInfo_499_);
v___x_515_ = lean_nat_dec_lt(v___x_507_, v___x_514_);
if (v___x_515_ == 0)
{
goto v___jp_508_;
}
else
{
lean_object* v___x_516_; uint8_t v___x_517_; 
v___x_516_ = lean_array_fget_borrowed(v_paramInfo_499_, v___x_507_);
v___x_517_ = l_Lean_Meta_ParamInfo_isExplicit(v___x_516_);
if (v___x_517_ == 0)
{
lean_object* v___x_518_; lean_object* v___x_519_; 
lean_dec(v___x_507_);
v___x_518_ = lean_box(v_a_487_);
v___x_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_519_, 0, v___x_518_);
v___y_503_ = v___x_519_;
v_a_504_ = v_a_487_;
goto v___jp_502_;
}
else
{
goto v___jp_508_;
}
}
v___jp_502_:
{
if (v_a_504_ == 0)
{
lean_dec(v_n_501_);
return v___y_503_;
}
else
{
lean_dec_ref(v___y_503_);
v_i_489_ = v_n_501_;
goto _start;
}
}
v___jp_508_:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_509_ = l_Lean_instInhabitedExpr;
v___x_510_ = lean_array_get_borrowed(v___x_509_, v_args_485_, v___x_507_);
lean_dec(v___x_507_);
lean_inc(v___x_510_);
v___x_511_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(v___x_510_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; uint8_t v___x_513_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
v___x_513_ = lean_unbox(v_a_512_);
lean_dec(v_a_512_);
v___y_503_ = v___x_511_;
v_a_504_ = v___x_513_;
goto v___jp_502_;
}
else
{
lean_dec(v_n_501_);
return v___x_511_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1(lean_object* v_e_520_, lean_object* v_x_521_, lean_object* v_x_522_, lean_object* v_x_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
if (lean_obj_tag(v_x_521_) == 5)
{
lean_object* v_fn_529_; lean_object* v_arg_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; 
v_fn_529_ = lean_ctor_get(v_x_521_, 0);
lean_inc_ref(v_fn_529_);
v_arg_530_ = lean_ctor_get(v_x_521_, 1);
lean_inc_ref(v_arg_530_);
lean_dec_ref_known(v_x_521_, 2);
v___x_531_ = lean_array_set(v_x_522_, v_x_523_, v_arg_530_);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_nat_sub(v_x_523_, v___x_532_);
lean_dec(v_x_523_);
v_x_521_ = v_fn_529_;
v_x_522_ = v___x_531_;
v_x_523_ = v___x_533_;
goto _start;
}
else
{
lean_object* v___x_535_; 
lean_dec(v_x_523_);
lean_inc_ref(v_x_521_);
v___x_535_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(v_x_521_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
if (lean_obj_tag(v___x_535_) == 0)
{
lean_object* v_a_536_; uint8_t v___x_537_; 
v_a_536_ = lean_ctor_get(v___x_535_, 0);
lean_inc(v_a_536_);
v___x_537_ = lean_unbox(v_a_536_);
if (v___x_537_ == 0)
{
lean_dec(v_a_536_);
lean_dec_ref(v_x_522_);
lean_dec_ref(v_x_521_);
return v___x_535_;
}
else
{
lean_object* v___x_538_; lean_object* v___x_539_; 
lean_dec_ref_known(v___x_535_, 1);
v___x_538_ = l_Lean_Expr_getAppNumArgs(v_e_520_);
lean_inc(v___x_538_);
v___x_539_ = l_Lean_Meta_getFunInfoNArgs(v_x_521_, v___x_538_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
if (lean_obj_tag(v___x_539_) == 0)
{
lean_object* v_a_540_; uint8_t v___x_541_; lean_object* v___x_542_; 
v_a_540_ = lean_ctor_get(v___x_539_, 0);
lean_inc(v_a_540_);
lean_dec_ref_known(v___x_539_, 1);
v___x_541_ = lean_unbox(v_a_536_);
lean_dec(v_a_536_);
lean_inc(v___x_538_);
v___x_542_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg(v_x_522_, v_a_540_, v___x_541_, v___x_538_, v___x_538_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
lean_dec(v___x_538_);
lean_dec(v_a_540_);
lean_dec_ref(v_x_522_);
return v___x_542_;
}
else
{
lean_object* v_a_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_550_; 
lean_dec(v___x_538_);
lean_dec(v_a_536_);
lean_dec_ref(v_x_522_);
v_a_543_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_550_ == 0)
{
v___x_545_ = v___x_539_;
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_a_543_);
lean_dec(v___x_539_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
lean_object* v___x_548_; 
if (v_isShared_546_ == 0)
{
v___x_548_ = v___x_545_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v_a_543_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
return v___x_548_;
}
}
}
}
}
else
{
lean_dec_ref(v_x_522_);
lean_dec_ref(v_x_521_);
return v___x_535_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1___boxed(lean_object* v_e_551_, lean_object* v_x_552_, lean_object* v_x_553_, lean_object* v_x_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__1(v_e_551_, v_x_552_, v_x_553_, v_x_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_);
lean_dec(v___y_558_);
lean_dec_ref(v___y_557_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec_ref(v_e_551_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly___boxed(lean_object* v_e_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(v_e_561_, v_a_562_, v_a_563_, v_a_564_, v_a_565_);
lean_dec(v_a_565_);
lean_dec_ref(v_a_564_);
lean_dec(v_a_563_);
lean_dec_ref(v_a_562_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg___boxed(lean_object* v_args_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_n_571_, lean_object* v_i_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
uint8_t v_a_1639__boxed_578_; lean_object* v_res_579_; 
v_a_1639__boxed_578_ = lean_unbox(v_a_570_);
v_res_579_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg(v_args_568_, v_a_569_, v_a_1639__boxed_578_, v_n_571_, v_i_572_, v___y_573_, v___y_574_, v___y_575_, v___y_576_);
lean_dec(v___y_576_);
lean_dec_ref(v___y_575_);
lean_dec(v___y_574_);
lean_dec_ref(v___y_573_);
lean_dec(v_n_571_);
lean_dec_ref(v_a_569_);
lean_dec_ref(v_args_568_);
return v_res_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0(lean_object* v_args_580_, lean_object* v_a_581_, uint8_t v_a_582_, lean_object* v_n_583_, lean_object* v_i_584_, lean_object* v_a_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v___x_591_; 
v___x_591_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___redArg(v_args_580_, v_a_581_, v_a_582_, v_n_583_, v_i_584_, v___y_586_, v___y_587_, v___y_588_, v___y_589_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0___boxed(lean_object* v_args_592_, lean_object* v_a_593_, lean_object* v_a_594_, lean_object* v_n_595_, lean_object* v_i_596_, lean_object* v_a_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
uint8_t v_a_1773__boxed_603_; lean_object* v_res_604_; 
v_a_1773__boxed_603_ = lean_unbox(v_a_594_);
v_res_604_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly_spec__0(v_args_592_, v_a_593_, v_a_1773__boxed_603_, v_n_595_, v_i_596_, v_a_597_, v___y_598_, v___y_599_, v___y_600_, v___y_601_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec(v___y_599_);
lean_dec_ref(v___y_598_);
lean_dec(v_n_595_);
lean_dec_ref(v_a_593_);
lean_dec_ref(v_args_592_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0(lean_object* v_as_605_, size_t v_i_606_, size_t v_stop_607_, lean_object* v_b_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
uint8_t v___x_614_; 
v___x_614_ = lean_usize_dec_eq(v_i_606_, v_stop_607_);
if (v___x_614_ == 0)
{
lean_object* v___x_615_; lean_object* v___x_616_; 
v___x_615_ = lean_array_uget_borrowed(v_as_605_, v_i_606_);
lean_inc(v___x_615_);
v___x_616_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_isUserFriendly(v___x_615_, v___y_609_, v___y_610_, v___y_611_, v___y_612_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v_a_617_; lean_object* v_a_619_; uint8_t v___x_623_; 
v_a_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_a_617_);
lean_dec_ref_known(v___x_616_, 1);
v___x_623_ = lean_unbox(v_a_617_);
lean_dec(v_a_617_);
if (v___x_623_ == 0)
{
v_a_619_ = v_b_608_;
goto v___jp_618_;
}
else
{
lean_object* v___x_624_; 
lean_inc(v___x_615_);
v___x_624_ = lean_array_push(v_b_608_, v___x_615_);
v_a_619_ = v___x_624_;
goto v___jp_618_;
}
v___jp_618_:
{
size_t v___x_620_; size_t v___x_621_; 
v___x_620_ = ((size_t)1ULL);
v___x_621_ = lean_usize_add(v_i_606_, v___x_620_);
v_i_606_ = v___x_621_;
v_b_608_ = v_a_619_;
goto _start;
}
}
else
{
lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_632_; 
lean_dec_ref(v_b_608_);
v_a_625_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_632_ == 0)
{
v___x_627_ = v___x_616_;
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_616_);
v___x_627_ = lean_box(0);
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
v_resetjp_626_:
{
lean_object* v___x_630_; 
if (v_isShared_628_ == 0)
{
v___x_630_ = v___x_627_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_a_625_);
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
else
{
lean_object* v___x_633_; 
v___x_633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_633_, 0, v_b_608_);
return v___x_633_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0___boxed(lean_object* v_as_634_, lean_object* v_i_635_, lean_object* v_stop_636_, lean_object* v_b_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
size_t v_i_boxed_643_; size_t v_stop_boxed_644_; lean_object* v_res_645_; 
v_i_boxed_643_ = lean_unbox_usize(v_i_635_);
lean_dec(v_i_635_);
v_stop_boxed_644_ = lean_unbox_usize(v_stop_636_);
lean_dec(v_stop_636_);
v_res_645_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0(v_as_634_, v_i_boxed_643_, v_stop_boxed_644_, v_b_637_, v___y_638_, v___y_639_, v___y_640_, v___y_641_);
lean_dec(v___y_641_);
lean_dec_ref(v___y_640_);
lean_dec(v___y_639_);
lean_dec_ref(v___y_638_);
lean_dec_ref(v_as_634_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds(lean_object* v_e_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
lean_object* v_keyedConfig_652_; uint8_t v_trackZetaDelta_653_; lean_object* v_zetaDeltaSet_654_; lean_object* v_lctx_655_; lean_object* v_localInstances_656_; lean_object* v_defEqCtx_x3f_657_; lean_object* v_synthPendingDepth_658_; lean_object* v_customCanUnfoldPredicate_x3f_659_; uint8_t v_univApprox_660_; uint8_t v_inTypeClassResolution_661_; uint8_t v_cacheInferType_662_; uint8_t v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v_keyedConfig_652_ = lean_ctor_get(v_a_647_, 0);
v_trackZetaDelta_653_ = lean_ctor_get_uint8(v_a_647_, sizeof(void*)*7);
v_zetaDeltaSet_654_ = lean_ctor_get(v_a_647_, 1);
v_lctx_655_ = lean_ctor_get(v_a_647_, 2);
v_localInstances_656_ = lean_ctor_get(v_a_647_, 3);
v_defEqCtx_x3f_657_ = lean_ctor_get(v_a_647_, 4);
v_synthPendingDepth_658_ = lean_ctor_get(v_a_647_, 5);
v_customCanUnfoldPredicate_x3f_659_ = lean_ctor_get(v_a_647_, 6);
v_univApprox_660_ = lean_ctor_get_uint8(v_a_647_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_661_ = lean_ctor_get_uint8(v_a_647_, sizeof(void*)*7 + 2);
v_cacheInferType_662_ = lean_ctor_get_uint8(v_a_647_, sizeof(void*)*7 + 3);
v___x_663_ = 1;
lean_inc_ref(v_keyedConfig_652_);
v___x_664_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_663_, v_keyedConfig_652_);
lean_inc(v_customCanUnfoldPredicate_x3f_659_);
lean_inc(v_synthPendingDepth_658_);
lean_inc(v_defEqCtx_x3f_657_);
lean_inc_ref(v_localInstances_656_);
lean_inc_ref(v_lctx_655_);
lean_inc(v_zetaDeltaSet_654_);
v___x_665_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_665_, 0, v___x_664_);
lean_ctor_set(v___x_665_, 1, v_zetaDeltaSet_654_);
lean_ctor_set(v___x_665_, 2, v_lctx_655_);
lean_ctor_set(v___x_665_, 3, v_localInstances_656_);
lean_ctor_set(v___x_665_, 4, v_defEqCtx_x3f_657_);
lean_ctor_set(v___x_665_, 5, v_synthPendingDepth_658_);
lean_ctor_set(v___x_665_, 6, v_customCanUnfoldPredicate_x3f_659_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*7, v_trackZetaDelta_653_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*7 + 1, v_univApprox_660_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*7 + 2, v_inTypeClassResolution_661_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*7 + 3, v_cacheInferType_662_);
v___x_666_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds(v_e_646_, v___x_665_, v_a_648_, v_a_649_, v_a_650_);
if (lean_obj_tag(v___x_666_) == 0)
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_688_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_688_ == 0)
{
v___x_669_ = v___x_666_;
v_isShared_670_ = v_isSharedCheck_688_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___x_666_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_688_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; uint8_t v___x_674_; 
v___x_671_ = lean_unsigned_to_nat(0u);
v___x_672_ = lean_array_get_size(v_a_667_);
v___x_673_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_unfolds___closed__0));
v___x_674_ = lean_nat_dec_lt(v___x_671_, v___x_672_);
if (v___x_674_ == 0)
{
lean_object* v___x_676_; 
lean_dec(v_a_667_);
lean_dec_ref_known(v___x_665_, 7);
if (v_isShared_670_ == 0)
{
lean_ctor_set(v___x_669_, 0, v___x_673_);
v___x_676_ = v___x_669_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v___x_673_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
else
{
uint8_t v___x_678_; 
v___x_678_ = lean_nat_dec_le(v___x_672_, v___x_672_);
if (v___x_678_ == 0)
{
if (v___x_674_ == 0)
{
lean_object* v___x_680_; 
lean_dec(v_a_667_);
lean_dec_ref_known(v___x_665_, 7);
if (v_isShared_670_ == 0)
{
lean_ctor_set(v___x_669_, 0, v___x_673_);
v___x_680_ = v___x_669_;
goto v_reusejp_679_;
}
else
{
lean_object* v_reuseFailAlloc_681_; 
v_reuseFailAlloc_681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_681_, 0, v___x_673_);
v___x_680_ = v_reuseFailAlloc_681_;
goto v_reusejp_679_;
}
v_reusejp_679_:
{
return v___x_680_;
}
}
else
{
size_t v___x_682_; size_t v___x_683_; lean_object* v___x_684_; 
lean_del_object(v___x_669_);
v___x_682_ = ((size_t)0ULL);
v___x_683_ = lean_usize_of_nat(v___x_672_);
v___x_684_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0(v_a_667_, v___x_682_, v___x_683_, v___x_673_, v___x_665_, v_a_648_, v_a_649_, v_a_650_);
lean_dec_ref_known(v___x_665_, 7);
lean_dec(v_a_667_);
return v___x_684_;
}
}
else
{
size_t v___x_685_; size_t v___x_686_; lean_object* v___x_687_; 
lean_del_object(v___x_669_);
v___x_685_ = ((size_t)0ULL);
v___x_686_ = lean_usize_of_nat(v___x_672_);
v___x_687_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds_spec__0(v_a_667_, v___x_685_, v___x_686_, v___x_673_, v___x_665_, v_a_648_, v_a_649_, v_a_650_);
lean_dec_ref_known(v___x_665_, 7);
lean_dec(v_a_667_);
return v___x_687_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_665_, 7);
return v___x_666_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds___boxed(lean_object* v_e_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds(v_e_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_);
lean_dec(v_a_693_);
lean_dec_ref(v_a_692_);
lean_dec(v_a_691_);
lean_dec_ref(v_a_690_);
return v_res_695_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13(void){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_719_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__12));
v___x_720_ = l_Lean_mkIdent(v___x_719_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(lean_object* v_e_721_, lean_object* v_eNew_722_, lean_object* v_rwKind_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_){
_start:
{
lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_730_ = lean_box(1);
v___x_731_ = l_Lean_PrettyPrinter_delab(v_e_721_, v___x_730_, v_a_725_, v_a_726_, v_a_727_, v_a_728_);
if (lean_obj_tag(v___x_731_) == 0)
{
lean_object* v_a_732_; lean_object* v___x_733_; 
v_a_732_ = lean_ctor_get(v___x_731_, 0);
lean_inc(v_a_732_);
lean_dec_ref_known(v___x_731_, 1);
v___x_733_ = l_Lean_PrettyPrinter_delab(v_eNew_722_, v___x_730_, v_a_725_, v_a_726_, v_a_727_, v_a_728_);
if (lean_obj_tag(v___x_733_) == 0)
{
lean_object* v_a_734_; lean_object* v_ref_735_; uint8_t v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v_a_734_ = lean_ctor_get(v___x_733_, 0);
lean_inc(v_a_734_);
lean_dec_ref_known(v___x_733_, 1);
v_ref_735_ = lean_ctor_get(v_a_727_, 5);
v___x_736_ = 0;
v___x_737_ = l_Lean_SourceInfo_fromRef(v_ref_735_, v___x_736_);
v___x_738_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__3));
v___x_739_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__4));
lean_inc_n(v___x_737_, 3);
v___x_740_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_737_);
lean_ctor_set(v___x_740_, 1, v___x_738_);
v___x_741_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__6));
v___x_742_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__7));
v___x_743_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_737_);
lean_ctor_set(v___x_743_, 1, v___x_742_);
v___x_744_ = l_Lean_Syntax_node3(v___x_737_, v___x_741_, v_a_732_, v___x_743_, v_a_734_);
v___x_745_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(v_a_724_, v_a_725_, v_a_727_, v_a_728_);
if (lean_obj_tag(v___x_745_) == 0)
{
lean_object* v_a_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v_a_746_ = lean_ctor_get(v___x_745_, 0);
lean_inc(v_a_746_);
lean_dec_ref_known(v___x_745_, 1);
v___x_747_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__8));
lean_inc_n(v___x_737_, 2);
v___x_748_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_748_, 0, v___x_737_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
v___x_749_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__10));
v___x_750_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__13);
v___x_751_ = l_Lean_Syntax_node2(v___x_737_, v___x_749_, v___x_748_, v___x_750_);
v___x_752_ = l_Lean_Syntax_node3(v___x_737_, v___x_739_, v___x_740_, v___x_744_, v___x_751_);
v___x_753_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(v_rwKind_723_, v___x_736_, v___x_752_, v_a_746_, v___x_736_, v_a_727_);
return v___x_753_;
}
else
{
lean_object* v_a_754_; lean_object* v___x_756_; uint8_t v_isShared_757_; uint8_t v_isSharedCheck_761_; 
lean_dec(v___x_744_);
lean_dec_ref_known(v___x_740_, 2);
lean_dec(v___x_737_);
lean_dec(v_rwKind_723_);
v_a_754_ = lean_ctor_get(v___x_745_, 0);
v_isSharedCheck_761_ = !lean_is_exclusive(v___x_745_);
if (v_isSharedCheck_761_ == 0)
{
v___x_756_ = v___x_745_;
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
else
{
lean_inc(v_a_754_);
lean_dec(v___x_745_);
v___x_756_ = lean_box(0);
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
v_resetjp_755_:
{
lean_object* v___x_759_; 
if (v_isShared_757_ == 0)
{
v___x_759_ = v___x_756_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_a_754_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
return v___x_759_;
}
}
}
}
else
{
lean_dec(v_a_732_);
lean_dec(v_rwKind_723_);
return v___x_733_;
}
}
else
{
lean_dec(v_rwKind_723_);
lean_dec_ref(v_eNew_722_);
return v___x_731_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___boxed(lean_object* v_e_762_, lean_object* v_eNew_763_, lean_object* v_rwKind_764_, lean_object* v_a_765_, lean_object* v_a_766_, lean_object* v_a_767_, lean_object* v_a_768_, lean_object* v_a_769_, lean_object* v_a_770_){
_start:
{
lean_object* v_res_771_; 
v_res_771_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_e_762_, v_eNew_763_, v_rwKind_764_, v_a_765_, v_a_766_, v_a_767_, v_a_768_, v_a_769_);
lean_dec(v_a_769_);
lean_dec_ref(v_a_768_);
lean_dec(v_a_767_);
lean_dec_ref(v_a_766_);
lean_dec_ref(v_a_765_);
return v_res_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object* v_e_772_, lean_object* v_eNew_773_, lean_object* v_rwKind_774_, lean_object* v_a_775_, lean_object* v_a_776_, lean_object* v_a_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_){
_start:
{
lean_object* v___x_782_; 
v___x_782_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_e_772_, v_eNew_773_, v_rwKind_774_, v_a_775_, v_a_777_, v_a_778_, v_a_779_, v_a_780_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object* v_e_783_, lean_object* v_eNew_784_, lean_object* v_rwKind_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_e_783_, v_eNew_784_, v_rwKind_785_, v_a_786_, v_a_787_, v_a_788_, v_a_789_, v_a_790_, v_a_791_);
lean_dec(v_a_791_);
lean_dec_ref(v_a_790_);
lean_dec(v_a_789_);
lean_dec_ref(v_a_788_);
lean_dec(v_a_787_);
lean_dec_ref(v_a_786_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0(lean_object* v_e_794_, lean_object* v_rwKind_795_, lean_object* v___x_796_, size_t v_sz_797_, size_t v_i_798_, lean_object* v_bs_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
uint8_t v___x_807_; 
v___x_807_ = lean_usize_dec_lt(v_i_798_, v_sz_797_);
if (v___x_807_ == 0)
{
lean_object* v___x_808_; 
lean_dec(v_rwKind_795_);
lean_dec_ref(v_e_794_);
v___x_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_808_, 0, v_bs_799_);
return v___x_808_;
}
else
{
lean_object* v_v_809_; lean_object* v___x_810_; 
v_v_809_ = lean_array_uget(v_bs_799_, v_i_798_);
lean_inc(v_rwKind_795_);
lean_inc(v_v_809_);
lean_inc_ref(v_e_794_);
v___x_810_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_e_794_, v_v_809_, v_rwKind_795_, v___y_800_, v___y_802_, v___y_803_, v___y_804_, v___y_805_);
if (lean_obj_tag(v___x_810_) == 0)
{
lean_object* v_a_811_; lean_object* v___x_812_; lean_object* v_bs_x27_813_; lean_object* v___y_815_; lean_object* v___x_829_; 
v_a_811_ = lean_ctor_get(v___x_810_, 0);
lean_inc(v_a_811_);
lean_dec_ref_known(v___x_810_, 1);
v___x_812_ = lean_unsigned_to_nat(0u);
v_bs_x27_813_ = lean_array_uset(v_bs_799_, v_i_798_, v___x_812_);
v___x_829_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_v_809_, v___y_802_, v___y_803_, v___y_804_, v___y_805_);
if (lean_obj_tag(v___x_829_) == 0)
{
lean_object* v_a_830_; uint8_t v___x_831_; lean_object* v___x_832_; 
v_a_830_ = lean_ctor_get(v___x_829_, 0);
lean_inc(v_a_830_);
lean_dec_ref_known(v___x_829_, 1);
v___x_831_ = lean_nat_dec_eq(v___x_796_, v___x_812_);
v___x_832_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v_a_811_, v_a_830_, v___x_831_, v___y_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_, v___y_805_);
v___y_815_ = v___x_832_;
goto v___jp_814_;
}
else
{
lean_dec(v_a_811_);
v___y_815_ = v___x_829_;
goto v___jp_814_;
}
v___jp_814_:
{
if (lean_obj_tag(v___y_815_) == 0)
{
lean_object* v_a_816_; size_t v___x_817_; size_t v___x_818_; lean_object* v___x_819_; 
v_a_816_ = lean_ctor_get(v___y_815_, 0);
lean_inc(v_a_816_);
lean_dec_ref_known(v___y_815_, 1);
v___x_817_ = ((size_t)1ULL);
v___x_818_ = lean_usize_add(v_i_798_, v___x_817_);
v___x_819_ = lean_array_uset(v_bs_x27_813_, v_i_798_, v_a_816_);
v_i_798_ = v___x_818_;
v_bs_799_ = v___x_819_;
goto _start;
}
else
{
lean_object* v_a_821_; lean_object* v___x_823_; uint8_t v_isShared_824_; uint8_t v_isSharedCheck_828_; 
lean_dec_ref(v_bs_x27_813_);
lean_dec(v_rwKind_795_);
lean_dec_ref(v_e_794_);
v_a_821_ = lean_ctor_get(v___y_815_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v___y_815_);
if (v_isSharedCheck_828_ == 0)
{
v___x_823_ = v___y_815_;
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
else
{
lean_inc(v_a_821_);
lean_dec(v___y_815_);
v___x_823_ = lean_box(0);
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
v_resetjp_822_:
{
lean_object* v___x_826_; 
if (v_isShared_824_ == 0)
{
v___x_826_ = v___x_823_;
goto v_reusejp_825_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v_a_821_);
v___x_826_ = v_reuseFailAlloc_827_;
goto v_reusejp_825_;
}
v_reusejp_825_:
{
return v___x_826_;
}
}
}
}
}
else
{
lean_object* v_a_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_840_; 
lean_dec(v_v_809_);
lean_dec_ref(v_bs_799_);
lean_dec(v_rwKind_795_);
lean_dec_ref(v_e_794_);
v_a_833_ = lean_ctor_get(v___x_810_, 0);
v_isSharedCheck_840_ = !lean_is_exclusive(v___x_810_);
if (v_isSharedCheck_840_ == 0)
{
v___x_835_ = v___x_810_;
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_a_833_);
lean_dec(v___x_810_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v___x_838_; 
if (v_isShared_836_ == 0)
{
v___x_838_ = v___x_835_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_a_833_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0___boxed(lean_object* v_e_841_, lean_object* v_rwKind_842_, lean_object* v___x_843_, lean_object* v_sz_844_, lean_object* v_i_845_, lean_object* v_bs_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
size_t v_sz_boxed_854_; size_t v_i_boxed_855_; lean_object* v_res_856_; 
v_sz_boxed_854_ = lean_unbox_usize(v_sz_844_);
lean_dec(v_sz_844_);
v_i_boxed_855_ = lean_unbox_usize(v_i_845_);
lean_dec(v_i_845_);
v_res_856_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0(v_e_841_, v_rwKind_842_, v___x_843_, v_sz_boxed_854_, v_i_boxed_855_, v_bs_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___x_843_);
return v_res_856_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12(void){
_start:
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__9));
v___x_879_ = lean_unsigned_to_nat(3u);
v___x_880_ = lean_mk_empty_array_with_capacity(v___x_879_);
v___x_881_ = lean_array_push(v___x_880_, v___x_878_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold(lean_object* v_e_883_, lean_object* v_rwKind_884_, lean_object* v_a_885_, lean_object* v_a_886_, lean_object* v_a_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_){
_start:
{
lean_object* v___x_892_; 
lean_inc_ref(v_e_883_);
v___x_892_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds(v_e_883_, v_a_887_, v_a_888_, v_a_889_, v_a_890_);
if (lean_obj_tag(v___x_892_) == 0)
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_950_; 
v_a_893_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_950_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_950_ == 0)
{
v___x_895_ = v___x_892_;
v_isShared_896_ = v_isSharedCheck_950_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_892_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_950_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_897_; lean_object* v___x_898_; uint8_t v___x_899_; 
v___x_897_ = lean_array_get_size(v_a_893_);
v___x_898_ = lean_unsigned_to_nat(0u);
v___x_899_ = lean_nat_dec_eq(v___x_897_, v___x_898_);
if (v___x_899_ == 0)
{
size_t v_sz_900_; size_t v___x_901_; lean_object* v___x_902_; 
lean_del_object(v___x_895_);
v_sz_900_ = lean_array_size(v_a_893_);
v___x_901_ = ((size_t)0ULL);
lean_inc_ref(v_e_883_);
v___x_902_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_suggestUnfold_spec__0(v_e_883_, v_rwKind_884_, v___x_897_, v_sz_900_, v___x_901_, v_a_893_, v_a_885_, v_a_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_);
if (lean_obj_tag(v___x_902_) == 0)
{
lean_object* v_a_903_; lean_object* v___x_904_; 
v_a_903_ = lean_ctor_get(v___x_902_, 0);
lean_inc(v_a_903_);
lean_dec_ref_known(v___x_902_, 1);
v___x_904_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_e_883_, v_a_887_, v_a_888_, v_a_889_, v_a_890_);
if (lean_obj_tag(v___x_904_) == 0)
{
lean_object* v_a_905_; lean_object* v___x_907_; uint8_t v_isShared_908_; uint8_t v_isSharedCheck_929_; 
v_a_905_ = lean_ctor_get(v___x_904_, 0);
v_isSharedCheck_929_ = !lean_is_exclusive(v___x_904_);
if (v_isSharedCheck_929_ == 0)
{
v___x_907_ = v___x_904_;
v_isShared_908_ = v_isSharedCheck_929_;
goto v_resetjp_906_;
}
else
{
lean_inc(v_a_905_);
lean_dec(v___x_904_);
v___x_907_ = lean_box(0);
v_isShared_908_ = v_isSharedCheck_929_;
goto v_resetjp_906_;
}
v_resetjp_906_:
{
lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_927_; 
v___x_909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__0));
v___x_910_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__1));
v___x_911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__2));
v___x_912_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__7));
v___x_913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__11));
v___x_914_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__12);
v___x_915_ = lean_array_push(v___x_914_, v_a_905_);
v___x_916_ = lean_array_push(v___x_915_, v___x_913_);
v___x_917_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_917_, 0, v___x_911_);
lean_ctor_set(v___x_917_, 1, v___x_912_);
lean_ctor_set(v___x_917_, 2, v___x_916_);
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___closed__13));
v___x_919_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_919_, 0, v___x_918_);
lean_ctor_set(v___x_919_, 1, v___x_910_);
lean_ctor_set(v___x_919_, 2, v_a_903_);
v___x_920_ = lean_unsigned_to_nat(2u);
v___x_921_ = lean_mk_empty_array_with_capacity(v___x_920_);
v___x_922_ = lean_array_push(v___x_921_, v___x_917_);
v___x_923_ = lean_array_push(v___x_922_, v___x_919_);
v___x_924_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_924_, 0, v___x_909_);
lean_ctor_set(v___x_924_, 1, v___x_910_);
lean_ctor_set(v___x_924_, 2, v___x_923_);
v___x_925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
if (v_isShared_908_ == 0)
{
lean_ctor_set(v___x_907_, 0, v___x_925_);
v___x_927_ = v___x_907_;
goto v_reusejp_926_;
}
else
{
lean_object* v_reuseFailAlloc_928_; 
v_reuseFailAlloc_928_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_928_, 0, v___x_925_);
v___x_927_ = v_reuseFailAlloc_928_;
goto v_reusejp_926_;
}
v_reusejp_926_:
{
return v___x_927_;
}
}
}
else
{
lean_object* v_a_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
lean_dec(v_a_903_);
v_a_930_ = lean_ctor_get(v___x_904_, 0);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_904_);
if (v_isSharedCheck_937_ == 0)
{
v___x_932_ = v___x_904_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_a_930_);
lean_dec(v___x_904_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_a_930_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
else
{
lean_object* v_a_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_945_; 
lean_dec_ref(v_e_883_);
v_a_938_ = lean_ctor_get(v___x_902_, 0);
v_isSharedCheck_945_ = !lean_is_exclusive(v___x_902_);
if (v_isSharedCheck_945_ == 0)
{
v___x_940_ = v___x_902_;
v_isShared_941_ = v_isSharedCheck_945_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_a_938_);
lean_dec(v___x_902_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_945_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v___x_943_; 
if (v_isShared_941_ == 0)
{
v___x_943_ = v___x_940_;
goto v_reusejp_942_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v_a_938_);
v___x_943_ = v_reuseFailAlloc_944_;
goto v_reusejp_942_;
}
v_reusejp_942_:
{
return v___x_943_;
}
}
}
}
else
{
lean_object* v___x_946_; lean_object* v___x_948_; 
lean_dec(v_a_893_);
lean_dec(v_rwKind_884_);
lean_dec_ref(v_e_883_);
v___x_946_ = lean_box(0);
if (v_isShared_896_ == 0)
{
lean_ctor_set(v___x_895_, 0, v___x_946_);
v___x_948_ = v___x_895_;
goto v_reusejp_947_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_949_, 0, v___x_946_);
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
else
{
lean_object* v_a_951_; lean_object* v___x_953_; uint8_t v_isShared_954_; uint8_t v_isSharedCheck_958_; 
lean_dec(v_rwKind_884_);
lean_dec_ref(v_e_883_);
v_a_951_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_958_ == 0)
{
v___x_953_ = v___x_892_;
v_isShared_954_ = v_isSharedCheck_958_;
goto v_resetjp_952_;
}
else
{
lean_inc(v_a_951_);
lean_dec(v___x_892_);
v___x_953_ = lean_box(0);
v_isShared_954_ = v_isSharedCheck_958_;
goto v_resetjp_952_;
}
v_resetjp_952_:
{
lean_object* v___x_956_; 
if (v_isShared_954_ == 0)
{
v___x_956_ = v___x_953_;
goto v_reusejp_955_;
}
else
{
lean_object* v_reuseFailAlloc_957_; 
v_reuseFailAlloc_957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_957_, 0, v_a_951_);
v___x_956_ = v_reuseFailAlloc_957_;
goto v_reusejp_955_;
}
v_reusejp_955_:
{
return v___x_956_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold___boxed(lean_object* v_e_959_, lean_object* v_rwKind_960_, lean_object* v_a_961_, lean_object* v_a_962_, lean_object* v_a_963_, lean_object* v_a_964_, lean_object* v_a_965_, lean_object* v_a_966_, lean_object* v_a_967_){
_start:
{
lean_object* v_res_968_; 
v_res_968_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold(v_e_959_, v_rwKind_960_, v_a_961_, v_a_962_, v_a_963_, v_a_964_, v_a_965_, v_a_966_);
lean_dec(v_a_966_);
lean_dec_ref(v_a_965_);
lean_dec(v_a_964_);
lean_dec_ref(v_a_963_);
lean_dec(v_a_962_);
lean_dec_ref(v_a_961_);
return v_res_968_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; 
v___x_999_ = lean_box(0);
v___x_1000_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1001_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1001_, 0, v___x_1000_);
lean_ctor_set(v___x_1001_, 1, v___x_999_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1003_; lean_object* v___x_1004_; 
v___x_1003_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___closed__0);
v___x_1004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1004_, 0, v___x_1003_);
return v___x_1004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg___boxed(lean_object* v___y_1005_){
_start:
{
lean_object* v_res_1006_; 
v_res_1006_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg();
return v_res_1006_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0(lean_object* v_00_u03b1_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_){
_start:
{
lean_object* v___x_1011_; 
v___x_1011_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg();
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___boxed(lean_object* v_00_u03b1_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_){
_start:
{
lean_object* v_res_1016_; 
v_res_1016_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0(v_00_u03b1_1012_, v___y_1013_, v___y_1014_);
lean_dec(v___y_1014_);
lean_dec_ref(v___y_1013_);
return v_res_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg(lean_object* v_e_1017_, lean_object* v___y_1018_){
_start:
{
uint8_t v___x_1020_; 
v___x_1020_ = l_Lean_Expr_hasMVar(v_e_1017_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; 
v___x_1021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1021_, 0, v_e_1017_);
return v___x_1021_;
}
else
{
lean_object* v___x_1022_; lean_object* v_mctx_1023_; lean_object* v___x_1024_; lean_object* v_fst_1025_; lean_object* v_snd_1026_; lean_object* v___x_1027_; lean_object* v_cache_1028_; lean_object* v_zetaDeltaFVarIds_1029_; lean_object* v_postponed_1030_; lean_object* v_diag_1031_; lean_object* v___x_1033_; uint8_t v_isShared_1034_; uint8_t v_isSharedCheck_1040_; 
v___x_1022_ = lean_st_ref_get(v___y_1018_);
v_mctx_1023_ = lean_ctor_get(v___x_1022_, 0);
lean_inc_ref(v_mctx_1023_);
lean_dec(v___x_1022_);
v___x_1024_ = l_Lean_instantiateMVarsCore(v_mctx_1023_, v_e_1017_);
v_fst_1025_ = lean_ctor_get(v___x_1024_, 0);
lean_inc(v_fst_1025_);
v_snd_1026_ = lean_ctor_get(v___x_1024_, 1);
lean_inc(v_snd_1026_);
lean_dec_ref(v___x_1024_);
v___x_1027_ = lean_st_ref_take(v___y_1018_);
v_cache_1028_ = lean_ctor_get(v___x_1027_, 1);
v_zetaDeltaFVarIds_1029_ = lean_ctor_get(v___x_1027_, 2);
v_postponed_1030_ = lean_ctor_get(v___x_1027_, 3);
v_diag_1031_ = lean_ctor_get(v___x_1027_, 4);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_1027_);
if (v_isSharedCheck_1040_ == 0)
{
lean_object* v_unused_1041_; 
v_unused_1041_ = lean_ctor_get(v___x_1027_, 0);
lean_dec(v_unused_1041_);
v___x_1033_ = v___x_1027_;
v_isShared_1034_ = v_isSharedCheck_1040_;
goto v_resetjp_1032_;
}
else
{
lean_inc(v_diag_1031_);
lean_inc(v_postponed_1030_);
lean_inc(v_zetaDeltaFVarIds_1029_);
lean_inc(v_cache_1028_);
lean_dec(v___x_1027_);
v___x_1033_ = lean_box(0);
v_isShared_1034_ = v_isSharedCheck_1040_;
goto v_resetjp_1032_;
}
v_resetjp_1032_:
{
lean_object* v___x_1036_; 
if (v_isShared_1034_ == 0)
{
lean_ctor_set(v___x_1033_, 0, v_snd_1026_);
v___x_1036_ = v___x_1033_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1039_; 
v_reuseFailAlloc_1039_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1039_, 0, v_snd_1026_);
lean_ctor_set(v_reuseFailAlloc_1039_, 1, v_cache_1028_);
lean_ctor_set(v_reuseFailAlloc_1039_, 2, v_zetaDeltaFVarIds_1029_);
lean_ctor_set(v_reuseFailAlloc_1039_, 3, v_postponed_1030_);
lean_ctor_set(v_reuseFailAlloc_1039_, 4, v_diag_1031_);
v___x_1036_ = v_reuseFailAlloc_1039_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1037_ = lean_st_ref_set(v___y_1018_, v___x_1036_);
v___x_1038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1038_, 0, v_fst_1025_);
return v___x_1038_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg___boxed(lean_object* v_e_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_){
_start:
{
lean_object* v_res_1045_; 
v_res_1045_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg(v_e_1042_, v___y_1043_);
lean_dec(v___y_1043_);
return v_res_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1(lean_object* v_e_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_){
_start:
{
lean_object* v___x_1054_; 
v___x_1054_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg(v_e_1046_, v___y_1050_);
return v___x_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___boxed(lean_object* v_e_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1(v_e_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
return v_res_1063_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0(uint8_t v___x_1064_, lean_object* v_x_1065_){
_start:
{
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0___boxed(lean_object* v___x_1066_, lean_object* v_x_1067_){
_start:
{
uint8_t v___x_6234__boxed_1068_; uint8_t v_res_1069_; lean_object* v_r_1070_; 
v___x_6234__boxed_1068_ = lean_unbox(v___x_1066_);
v_res_1069_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__0(v___x_6234__boxed_1068_, v_x_1067_);
lean_dec(v_x_1067_);
v_r_1070_ = lean_box(v_res_1069_);
return v_r_1070_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1072_; lean_object* v___x_1073_; 
v___x_1072_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__0));
v___x_1073_ = l_Lean_stringToMessageData(v___x_1072_);
return v___x_1073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2(lean_object* v_a_1074_, lean_object* v_a_1075_){
_start:
{
if (lean_obj_tag(v_a_1074_) == 0)
{
lean_object* v___x_1076_; 
v___x_1076_ = l_List_reverse___redArg(v_a_1075_);
return v___x_1076_;
}
else
{
lean_object* v_head_1077_; lean_object* v_tail_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1089_; 
v_head_1077_ = lean_ctor_get(v_a_1074_, 0);
v_tail_1078_ = lean_ctor_get(v_a_1074_, 1);
v_isSharedCheck_1089_ = !lean_is_exclusive(v_a_1074_);
if (v_isSharedCheck_1089_ == 0)
{
v___x_1080_ = v_a_1074_;
v_isShared_1081_ = v_isSharedCheck_1089_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_tail_1078_);
lean_inc(v_head_1077_);
lean_dec(v_a_1074_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1089_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1086_; 
v___x_1082_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2___closed__1);
v___x_1083_ = l_Lean_MessageData_ofExpr(v_head_1077_);
v___x_1084_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1082_);
lean_ctor_set(v___x_1084_, 1, v___x_1083_);
if (v_isShared_1081_ == 0)
{
lean_ctor_set(v___x_1080_, 1, v_a_1075_);
lean_ctor_set(v___x_1080_, 0, v___x_1084_);
v___x_1086_ = v___x_1080_;
goto v_reusejp_1085_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v___x_1084_);
lean_ctor_set(v_reuseFailAlloc_1088_, 1, v_a_1075_);
v___x_1086_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1085_;
}
v_reusejp_1085_:
{
v_a_1074_ = v_tail_1078_;
v_a_1075_ = v___x_1086_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6(lean_object* v_opts_1090_, lean_object* v_opt_1091_){
_start:
{
lean_object* v_name_1092_; lean_object* v_defValue_1093_; lean_object* v_map_1094_; lean_object* v___x_1095_; 
v_name_1092_ = lean_ctor_get(v_opt_1091_, 0);
v_defValue_1093_ = lean_ctor_get(v_opt_1091_, 1);
v_map_1094_ = lean_ctor_get(v_opts_1090_, 0);
v___x_1095_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1094_, v_name_1092_);
if (lean_obj_tag(v___x_1095_) == 0)
{
uint8_t v___x_1096_; 
v___x_1096_ = lean_unbox(v_defValue_1093_);
return v___x_1096_;
}
else
{
lean_object* v_val_1097_; 
v_val_1097_ = lean_ctor_get(v___x_1095_, 0);
lean_inc(v_val_1097_);
lean_dec_ref_known(v___x_1095_, 1);
if (lean_obj_tag(v_val_1097_) == 1)
{
uint8_t v_v_1098_; 
v_v_1098_ = lean_ctor_get_uint8(v_val_1097_, 0);
lean_dec_ref_known(v_val_1097_, 0);
return v_v_1098_;
}
else
{
uint8_t v___x_1099_; 
lean_dec(v_val_1097_);
v___x_1099_ = lean_unbox(v_defValue_1093_);
return v___x_1099_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6___boxed(lean_object* v_opts_1100_, lean_object* v_opt_1101_){
_start:
{
uint8_t v_res_1102_; lean_object* v_r_1103_; 
v_res_1102_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6(v_opts_1100_, v_opt_1101_);
lean_dec_ref(v_opt_1101_);
lean_dec_ref(v_opts_1100_);
v_r_1103_ = lean_box(v_res_1102_);
return v_r_1103_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0(uint8_t v___y_1111_, uint8_t v_suppressElabErrors_1112_, lean_object* v_x_1113_){
_start:
{
if (lean_obj_tag(v_x_1113_) == 1)
{
lean_object* v_pre_1114_; 
v_pre_1114_ = lean_ctor_get(v_x_1113_, 0);
switch(lean_obj_tag(v_pre_1114_))
{
case 1:
{
lean_object* v_pre_1115_; 
v_pre_1115_ = lean_ctor_get(v_pre_1114_, 0);
switch(lean_obj_tag(v_pre_1115_))
{
case 0:
{
lean_object* v_str_1116_; lean_object* v_str_1117_; lean_object* v___x_1118_; uint8_t v___x_1119_; 
v_str_1116_ = lean_ctor_get(v_x_1113_, 1);
v_str_1117_ = lean_ctor_get(v_pre_1114_, 1);
v___x_1118_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0));
v___x_1119_ = lean_string_dec_eq(v_str_1117_, v___x_1118_);
if (v___x_1119_ == 0)
{
lean_object* v___x_1120_; uint8_t v___x_1121_; 
v___x_1120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__1));
v___x_1121_ = lean_string_dec_eq(v_str_1117_, v___x_1120_);
if (v___x_1121_ == 0)
{
return v___y_1111_;
}
else
{
lean_object* v___x_1122_; uint8_t v___x_1123_; 
v___x_1122_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1));
v___x_1123_ = lean_string_dec_eq(v_str_1116_, v___x_1122_);
if (v___x_1123_ == 0)
{
return v___y_1111_;
}
else
{
return v_suppressElabErrors_1112_;
}
}
}
else
{
lean_object* v___x_1124_; uint8_t v___x_1125_; 
v___x_1124_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2));
v___x_1125_ = lean_string_dec_eq(v_str_1116_, v___x_1124_);
if (v___x_1125_ == 0)
{
return v___y_1111_;
}
else
{
return v_suppressElabErrors_1112_;
}
}
}
case 1:
{
lean_object* v_pre_1126_; 
v_pre_1126_ = lean_ctor_get(v_pre_1115_, 0);
if (lean_obj_tag(v_pre_1126_) == 0)
{
lean_object* v_str_1127_; lean_object* v_str_1128_; lean_object* v_str_1129_; lean_object* v___x_1130_; uint8_t v___x_1131_; 
v_str_1127_ = lean_ctor_get(v_x_1113_, 1);
v_str_1128_ = lean_ctor_get(v_pre_1114_, 1);
v_str_1129_ = lean_ctor_get(v_pre_1115_, 1);
v___x_1130_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3));
v___x_1131_ = lean_string_dec_eq(v_str_1129_, v___x_1130_);
if (v___x_1131_ == 0)
{
return v___y_1111_;
}
else
{
lean_object* v___x_1132_; uint8_t v___x_1133_; 
v___x_1132_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4));
v___x_1133_ = lean_string_dec_eq(v_str_1128_, v___x_1132_);
if (v___x_1133_ == 0)
{
return v___y_1111_;
}
else
{
lean_object* v___x_1134_; uint8_t v___x_1135_; 
v___x_1134_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5));
v___x_1135_ = lean_string_dec_eq(v_str_1127_, v___x_1134_);
if (v___x_1135_ == 0)
{
return v___y_1111_;
}
else
{
return v_suppressElabErrors_1112_;
}
}
}
}
else
{
return v___y_1111_;
}
}
default: 
{
return v___y_1111_;
}
}
}
case 0:
{
lean_object* v_str_1136_; lean_object* v___x_1137_; uint8_t v___x_1138_; 
v_str_1136_ = lean_ctor_get(v_x_1113_, 1);
v___x_1137_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___closed__6));
v___x_1138_ = lean_string_dec_eq(v_str_1136_, v___x_1137_);
if (v___x_1138_ == 0)
{
return v___y_1111_;
}
else
{
return v_suppressElabErrors_1112_;
}
}
default: 
{
return v___y_1111_;
}
}
}
else
{
return v___y_1111_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v___y_1139_, lean_object* v_suppressElabErrors_1140_, lean_object* v_x_1141_){
_start:
{
uint8_t v___y_6306__boxed_1142_; uint8_t v_suppressElabErrors_boxed_1143_; uint8_t v_res_1144_; lean_object* v_r_1145_; 
v___y_6306__boxed_1142_ = lean_unbox(v___y_1139_);
v_suppressElabErrors_boxed_1143_ = lean_unbox(v_suppressElabErrors_1140_);
v_res_1144_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0(v___y_6306__boxed_1142_, v_suppressElabErrors_boxed_1143_, v_x_1141_);
lean_dec(v_x_1141_);
v_r_1145_ = lean_box(v_res_1144_);
return v_r_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5(lean_object* v_msgData_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_){
_start:
{
lean_object* v___x_1152_; lean_object* v_env_1153_; lean_object* v___x_1154_; lean_object* v_mctx_1155_; lean_object* v_lctx_1156_; lean_object* v_options_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_1152_ = lean_st_ref_get(v___y_1150_);
v_env_1153_ = lean_ctor_get(v___x_1152_, 0);
lean_inc_ref(v_env_1153_);
lean_dec(v___x_1152_);
v___x_1154_ = lean_st_ref_get(v___y_1148_);
v_mctx_1155_ = lean_ctor_get(v___x_1154_, 0);
lean_inc_ref(v_mctx_1155_);
lean_dec(v___x_1154_);
v_lctx_1156_ = lean_ctor_get(v___y_1147_, 2);
v_options_1157_ = lean_ctor_get(v___y_1149_, 2);
lean_inc_ref(v_options_1157_);
lean_inc_ref(v_lctx_1156_);
v___x_1158_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1158_, 0, v_env_1153_);
lean_ctor_set(v___x_1158_, 1, v_mctx_1155_);
lean_ctor_set(v___x_1158_, 2, v_lctx_1156_);
lean_ctor_set(v___x_1158_, 3, v_options_1157_);
v___x_1159_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1158_);
lean_ctor_set(v___x_1159_, 1, v_msgData_1146_);
v___x_1160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1160_, 0, v___x_1159_);
return v___x_1160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5___boxed(lean_object* v_msgData_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5(v_msgData_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg(lean_object* v_ref_1169_, lean_object* v_msgData_1170_, uint8_t v_severity_1171_, uint8_t v_isSilent_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; lean_object* v___y_1182_; uint8_t v___y_1183_; lean_object* v___y_1184_; uint8_t v___y_1185_; lean_object* v___y_1186_; lean_object* v___y_1187_; lean_object* v___y_1215_; lean_object* v___y_1216_; uint8_t v___y_1217_; uint8_t v___y_1218_; uint8_t v___y_1219_; lean_object* v___y_1220_; lean_object* v___y_1221_; lean_object* v___y_1222_; lean_object* v___y_1240_; lean_object* v___y_1241_; uint8_t v___y_1242_; uint8_t v___y_1243_; uint8_t v___y_1244_; lean_object* v___y_1245_; lean_object* v___y_1246_; lean_object* v___y_1247_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___y_1253_; uint8_t v___y_1254_; uint8_t v___y_1255_; lean_object* v___y_1256_; uint8_t v___y_1257_; uint8_t v___x_1262_; lean_object* v___y_1264_; lean_object* v___y_1265_; lean_object* v___y_1266_; uint8_t v___y_1267_; lean_object* v___y_1268_; uint8_t v___y_1269_; uint8_t v___y_1270_; uint8_t v___y_1272_; uint8_t v___x_1287_; 
v___x_1262_ = 2;
v___x_1287_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1171_, v___x_1262_);
if (v___x_1287_ == 0)
{
v___y_1272_ = v___x_1287_;
goto v___jp_1271_;
}
else
{
uint8_t v___x_1288_; 
lean_inc_ref(v_msgData_1170_);
v___x_1288_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1170_);
v___y_1272_ = v___x_1288_;
goto v___jp_1271_;
}
v___jp_1178_:
{
lean_object* v___x_1188_; lean_object* v_currNamespace_1189_; lean_object* v_openDecls_1190_; lean_object* v_env_1191_; lean_object* v_nextMacroScope_1192_; lean_object* v_ngen_1193_; lean_object* v_auxDeclNGen_1194_; lean_object* v_traceState_1195_; lean_object* v_cache_1196_; lean_object* v_messages_1197_; lean_object* v_infoState_1198_; lean_object* v_snapshotTasks_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1213_; 
v___x_1188_ = lean_st_ref_take(v___y_1187_);
v_currNamespace_1189_ = lean_ctor_get(v___y_1186_, 6);
v_openDecls_1190_ = lean_ctor_get(v___y_1186_, 7);
v_env_1191_ = lean_ctor_get(v___x_1188_, 0);
v_nextMacroScope_1192_ = lean_ctor_get(v___x_1188_, 1);
v_ngen_1193_ = lean_ctor_get(v___x_1188_, 2);
v_auxDeclNGen_1194_ = lean_ctor_get(v___x_1188_, 3);
v_traceState_1195_ = lean_ctor_get(v___x_1188_, 4);
v_cache_1196_ = lean_ctor_get(v___x_1188_, 5);
v_messages_1197_ = lean_ctor_get(v___x_1188_, 6);
v_infoState_1198_ = lean_ctor_get(v___x_1188_, 7);
v_snapshotTasks_1199_ = lean_ctor_get(v___x_1188_, 8);
v_isSharedCheck_1213_ = !lean_is_exclusive(v___x_1188_);
if (v_isSharedCheck_1213_ == 0)
{
v___x_1201_ = v___x_1188_;
v_isShared_1202_ = v_isSharedCheck_1213_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_snapshotTasks_1199_);
lean_inc(v_infoState_1198_);
lean_inc(v_messages_1197_);
lean_inc(v_cache_1196_);
lean_inc(v_traceState_1195_);
lean_inc(v_auxDeclNGen_1194_);
lean_inc(v_ngen_1193_);
lean_inc(v_nextMacroScope_1192_);
lean_inc(v_env_1191_);
lean_dec(v___x_1188_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1213_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1208_; 
lean_inc(v_openDecls_1190_);
lean_inc(v_currNamespace_1189_);
v___x_1203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1203_, 0, v_currNamespace_1189_);
lean_ctor_set(v___x_1203_, 1, v_openDecls_1190_);
v___x_1204_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1203_);
lean_ctor_set(v___x_1204_, 1, v___y_1179_);
lean_inc_ref(v___y_1181_);
lean_inc_ref(v___y_1182_);
v___x_1205_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1205_, 0, v___y_1182_);
lean_ctor_set(v___x_1205_, 1, v___y_1184_);
lean_ctor_set(v___x_1205_, 2, v___y_1180_);
lean_ctor_set(v___x_1205_, 3, v___y_1181_);
lean_ctor_set(v___x_1205_, 4, v___x_1204_);
lean_ctor_set_uint8(v___x_1205_, sizeof(void*)*5, v___y_1183_);
lean_ctor_set_uint8(v___x_1205_, sizeof(void*)*5 + 1, v___y_1185_);
lean_ctor_set_uint8(v___x_1205_, sizeof(void*)*5 + 2, v_isSilent_1172_);
v___x_1206_ = l_Lean_MessageLog_add(v___x_1205_, v_messages_1197_);
if (v_isShared_1202_ == 0)
{
lean_ctor_set(v___x_1201_, 6, v___x_1206_);
v___x_1208_ = v___x_1201_;
goto v_reusejp_1207_;
}
else
{
lean_object* v_reuseFailAlloc_1212_; 
v_reuseFailAlloc_1212_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1212_, 0, v_env_1191_);
lean_ctor_set(v_reuseFailAlloc_1212_, 1, v_nextMacroScope_1192_);
lean_ctor_set(v_reuseFailAlloc_1212_, 2, v_ngen_1193_);
lean_ctor_set(v_reuseFailAlloc_1212_, 3, v_auxDeclNGen_1194_);
lean_ctor_set(v_reuseFailAlloc_1212_, 4, v_traceState_1195_);
lean_ctor_set(v_reuseFailAlloc_1212_, 5, v_cache_1196_);
lean_ctor_set(v_reuseFailAlloc_1212_, 6, v___x_1206_);
lean_ctor_set(v_reuseFailAlloc_1212_, 7, v_infoState_1198_);
lean_ctor_set(v_reuseFailAlloc_1212_, 8, v_snapshotTasks_1199_);
v___x_1208_ = v_reuseFailAlloc_1212_;
goto v_reusejp_1207_;
}
v_reusejp_1207_:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1209_ = lean_st_ref_set(v___y_1187_, v___x_1208_);
v___x_1210_ = lean_box(0);
v___x_1211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
return v___x_1211_;
}
}
}
v___jp_1214_:
{
lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1238_; 
v___x_1223_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1170_);
v___x_1224_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__5(v___x_1223_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
v_a_1225_ = lean_ctor_get(v___x_1224_, 0);
v_isSharedCheck_1238_ = !lean_is_exclusive(v___x_1224_);
if (v_isSharedCheck_1238_ == 0)
{
v___x_1227_ = v___x_1224_;
v_isShared_1228_ = v_isSharedCheck_1238_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1224_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1238_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; 
lean_inc_ref_n(v___y_1220_, 2);
v___x_1229_ = l_Lean_FileMap_toPosition(v___y_1220_, v___y_1221_);
lean_dec(v___y_1221_);
v___x_1230_ = l_Lean_FileMap_toPosition(v___y_1220_, v___y_1222_);
lean_dec(v___y_1222_);
v___x_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
v___x_1232_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___closed__0));
if (v___y_1219_ == 0)
{
lean_del_object(v___x_1227_);
lean_dec_ref(v___y_1215_);
v___y_1179_ = v_a_1225_;
v___y_1180_ = v___x_1231_;
v___y_1181_ = v___x_1232_;
v___y_1182_ = v___y_1216_;
v___y_1183_ = v___y_1217_;
v___y_1184_ = v___x_1229_;
v___y_1185_ = v___y_1218_;
v___y_1186_ = v___y_1175_;
v___y_1187_ = v___y_1176_;
goto v___jp_1178_;
}
else
{
uint8_t v___x_1233_; 
lean_inc(v_a_1225_);
v___x_1233_ = l_Lean_MessageData_hasTag(v___y_1215_, v_a_1225_);
if (v___x_1233_ == 0)
{
lean_object* v___x_1234_; lean_object* v___x_1236_; 
lean_dec_ref_known(v___x_1231_, 1);
lean_dec_ref(v___x_1229_);
lean_dec(v_a_1225_);
v___x_1234_ = lean_box(0);
if (v_isShared_1228_ == 0)
{
lean_ctor_set(v___x_1227_, 0, v___x_1234_);
v___x_1236_ = v___x_1227_;
goto v_reusejp_1235_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v___x_1234_);
v___x_1236_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1235_;
}
v_reusejp_1235_:
{
return v___x_1236_;
}
}
else
{
lean_del_object(v___x_1227_);
v___y_1179_ = v_a_1225_;
v___y_1180_ = v___x_1231_;
v___y_1181_ = v___x_1232_;
v___y_1182_ = v___y_1216_;
v___y_1183_ = v___y_1217_;
v___y_1184_ = v___x_1229_;
v___y_1185_ = v___y_1218_;
v___y_1186_ = v___y_1175_;
v___y_1187_ = v___y_1176_;
goto v___jp_1178_;
}
}
}
}
v___jp_1239_:
{
lean_object* v___x_1248_; 
v___x_1248_ = l_Lean_Syntax_getTailPos_x3f(v___y_1245_, v___y_1242_);
lean_dec(v___y_1245_);
if (lean_obj_tag(v___x_1248_) == 0)
{
lean_inc(v___y_1247_);
v___y_1215_ = v___y_1240_;
v___y_1216_ = v___y_1241_;
v___y_1217_ = v___y_1242_;
v___y_1218_ = v___y_1243_;
v___y_1219_ = v___y_1244_;
v___y_1220_ = v___y_1246_;
v___y_1221_ = v___y_1247_;
v___y_1222_ = v___y_1247_;
goto v___jp_1214_;
}
else
{
lean_object* v_val_1249_; 
v_val_1249_ = lean_ctor_get(v___x_1248_, 0);
lean_inc(v_val_1249_);
lean_dec_ref_known(v___x_1248_, 1);
v___y_1215_ = v___y_1240_;
v___y_1216_ = v___y_1241_;
v___y_1217_ = v___y_1242_;
v___y_1218_ = v___y_1243_;
v___y_1219_ = v___y_1244_;
v___y_1220_ = v___y_1246_;
v___y_1221_ = v___y_1247_;
v___y_1222_ = v_val_1249_;
goto v___jp_1214_;
}
}
v___jp_1250_:
{
lean_object* v_ref_1258_; lean_object* v___x_1259_; 
v_ref_1258_ = l_Lean_replaceRef(v_ref_1169_, v___y_1252_);
v___x_1259_ = l_Lean_Syntax_getPos_x3f(v_ref_1258_, v___y_1254_);
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_object* v___x_1260_; 
v___x_1260_ = lean_unsigned_to_nat(0u);
v___y_1240_ = v___y_1251_;
v___y_1241_ = v___y_1253_;
v___y_1242_ = v___y_1254_;
v___y_1243_ = v___y_1257_;
v___y_1244_ = v___y_1255_;
v___y_1245_ = v_ref_1258_;
v___y_1246_ = v___y_1256_;
v___y_1247_ = v___x_1260_;
goto v___jp_1239_;
}
else
{
lean_object* v_val_1261_; 
v_val_1261_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_val_1261_);
lean_dec_ref_known(v___x_1259_, 1);
v___y_1240_ = v___y_1251_;
v___y_1241_ = v___y_1253_;
v___y_1242_ = v___y_1254_;
v___y_1243_ = v___y_1257_;
v___y_1244_ = v___y_1255_;
v___y_1245_ = v_ref_1258_;
v___y_1246_ = v___y_1256_;
v___y_1247_ = v_val_1261_;
goto v___jp_1239_;
}
}
v___jp_1263_:
{
if (v___y_1270_ == 0)
{
v___y_1251_ = v___y_1264_;
v___y_1252_ = v___y_1265_;
v___y_1253_ = v___y_1266_;
v___y_1254_ = v___y_1269_;
v___y_1255_ = v___y_1267_;
v___y_1256_ = v___y_1268_;
v___y_1257_ = v_severity_1171_;
goto v___jp_1250_;
}
else
{
v___y_1251_ = v___y_1264_;
v___y_1252_ = v___y_1265_;
v___y_1253_ = v___y_1266_;
v___y_1254_ = v___y_1269_;
v___y_1255_ = v___y_1267_;
v___y_1256_ = v___y_1268_;
v___y_1257_ = v___x_1262_;
goto v___jp_1250_;
}
}
v___jp_1271_:
{
if (v___y_1272_ == 0)
{
lean_object* v_fileName_1273_; lean_object* v_fileMap_1274_; lean_object* v_options_1275_; lean_object* v_ref_1276_; uint8_t v_suppressElabErrors_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___f_1280_; uint8_t v___x_1281_; uint8_t v___x_1282_; 
v_fileName_1273_ = lean_ctor_get(v___y_1175_, 0);
v_fileMap_1274_ = lean_ctor_get(v___y_1175_, 1);
v_options_1275_ = lean_ctor_get(v___y_1175_, 2);
v_ref_1276_ = lean_ctor_get(v___y_1175_, 5);
v_suppressElabErrors_1277_ = lean_ctor_get_uint8(v___y_1175_, sizeof(void*)*14 + 1);
v___x_1278_ = lean_box(v___y_1272_);
v___x_1279_ = lean_box(v_suppressElabErrors_1277_);
v___f_1280_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1280_, 0, v___x_1278_);
lean_closure_set(v___f_1280_, 1, v___x_1279_);
v___x_1281_ = 1;
v___x_1282_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1171_, v___x_1281_);
if (v___x_1282_ == 0)
{
v___y_1264_ = v___f_1280_;
v___y_1265_ = v_ref_1276_;
v___y_1266_ = v_fileName_1273_;
v___y_1267_ = v_suppressElabErrors_1277_;
v___y_1268_ = v_fileMap_1274_;
v___y_1269_ = v___y_1272_;
v___y_1270_ = v___x_1282_;
goto v___jp_1263_;
}
else
{
lean_object* v___x_1283_; uint8_t v___x_1284_; 
v___x_1283_ = l_Lean_warningAsError;
v___x_1284_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4_spec__6(v_options_1275_, v___x_1283_);
v___y_1264_ = v___f_1280_;
v___y_1265_ = v_ref_1276_;
v___y_1266_ = v_fileName_1273_;
v___y_1267_ = v_suppressElabErrors_1277_;
v___y_1268_ = v_fileMap_1274_;
v___y_1269_ = v___y_1272_;
v___y_1270_ = v___x_1284_;
goto v___jp_1263_;
}
}
else
{
lean_object* v___x_1285_; lean_object* v___x_1286_; 
lean_dec_ref(v_msgData_1170_);
v___x_1285_ = lean_box(0);
v___x_1286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1286_, 0, v___x_1285_);
return v___x_1286_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg___boxed(lean_object* v_ref_1289_, lean_object* v_msgData_1290_, lean_object* v_severity_1291_, lean_object* v_isSilent_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_){
_start:
{
uint8_t v_severity_boxed_1298_; uint8_t v_isSilent_boxed_1299_; lean_object* v_res_1300_; 
v_severity_boxed_1298_ = lean_unbox(v_severity_1291_);
v_isSilent_boxed_1299_ = lean_unbox(v_isSilent_1292_);
v_res_1300_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg(v_ref_1289_, v_msgData_1290_, v_severity_boxed_1298_, v_isSilent_boxed_1299_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_);
lean_dec(v___y_1296_);
lean_dec_ref(v___y_1295_);
lean_dec(v___y_1294_);
lean_dec_ref(v___y_1293_);
lean_dec(v_ref_1289_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3(lean_object* v_msgData_1301_, uint8_t v_severity_1302_, uint8_t v_isSilent_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_){
_start:
{
lean_object* v_ref_1311_; lean_object* v___x_1312_; 
v_ref_1311_ = lean_ctor_get(v___y_1308_, 5);
v___x_1312_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg(v_ref_1311_, v_msgData_1301_, v_severity_1302_, v_isSilent_1303_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_);
return v___x_1312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3___boxed(lean_object* v_msgData_1313_, lean_object* v_severity_1314_, lean_object* v_isSilent_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
uint8_t v_severity_boxed_1323_; uint8_t v_isSilent_boxed_1324_; lean_object* v_res_1325_; 
v_severity_boxed_1323_ = lean_unbox(v_severity_1314_);
v_isSilent_boxed_1324_ = lean_unbox(v_isSilent_1315_);
v_res_1325_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3(v_msgData_1313_, v_severity_boxed_1323_, v_isSilent_boxed_1324_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
lean_dec(v___y_1319_);
lean_dec_ref(v___y_1318_);
lean_dec(v___y_1317_);
lean_dec_ref(v___y_1316_);
return v_res_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3(lean_object* v_msgData_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_){
_start:
{
uint8_t v___x_1334_; uint8_t v___x_1335_; lean_object* v___x_1336_; 
v___x_1334_ = 0;
v___x_1335_ = 0;
v___x_1336_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3(v_msgData_1326_, v___x_1334_, v___x_1335_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_);
return v___x_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3___boxed(lean_object* v_msgData_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_){
_start:
{
lean_object* v_res_1345_; 
v_res_1345_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3(v_msgData_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
return v_res_1345_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1350_; lean_object* v___x_1351_; 
v___x_1350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__1));
v___x_1351_ = l_Lean_stringToMessageData(v___x_1350_);
return v___x_1351_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1353_; lean_object* v___x_1354_; 
v___x_1353_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__3));
v___x_1354_ = l_Lean_stringToMessageData(v___x_1353_);
return v___x_1354_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1358_; lean_object* v___x_1359_; 
v___x_1358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__6));
v___x_1359_ = l_Lean_MessageData_ofFormat(v___x_1358_);
return v___x_1359_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1361_; lean_object* v___x_1362_; 
v___x_1361_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__8));
v___x_1362_ = l_Lean_stringToMessageData(v___x_1361_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1(lean_object* v___x_1363_, uint8_t v___x_1364_, lean_object* v_x_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1373_ = lean_box(0);
v___x_1374_ = l_Lean_Elab_Term_elabTerm(v___x_1363_, v___x_1373_, v___x_1364_, v___x_1364_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; uint8_t v___x_1376_; lean_object* v___x_1377_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1375_);
lean_dec_ref_known(v___x_1374_, 1);
v___x_1376_ = 0;
v___x_1377_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_1376_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
if (lean_obj_tag(v___x_1377_) == 0)
{
lean_object* v___x_1378_; lean_object* v_a_1379_; lean_object* v___f_1380_; lean_object* v___x_1381_; 
lean_dec_ref_known(v___x_1377_, 1);
v___x_1378_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__1___redArg(v_a_1375_, v___y_1369_);
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc(v_a_1379_);
lean_dec_ref(v___x_1378_);
v___f_1380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__0));
v___x_1381_ = l_Lean_Elab_Term_levelMVarToParam___redArg(v_a_1379_, v___f_1380_, v___y_1367_, v___y_1369_);
if (lean_obj_tag(v___x_1381_) == 0)
{
lean_object* v_a_1382_; lean_object* v___x_1383_; 
v_a_1382_ = lean_ctor_get(v___x_1381_, 0);
lean_inc_n(v_a_1382_, 2);
lean_dec_ref_known(v___x_1381_, 1);
v___x_1383_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Unfold_0__Mathlib_Tactic_ClickSuggestions_filteredUnfolds(v_a_1382_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
if (lean_obj_tag(v___x_1383_) == 0)
{
lean_object* v_a_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; uint8_t v___x_1387_; 
v_a_1384_ = lean_ctor_get(v___x_1383_, 0);
lean_inc(v_a_1384_);
lean_dec_ref_known(v___x_1383_, 1);
v___x_1385_ = lean_array_get_size(v_a_1384_);
v___x_1386_ = lean_unsigned_to_nat(0u);
v___x_1387_ = lean_nat_dec_eq(v___x_1385_, v___x_1386_);
if (v___x_1387_ == 0)
{
lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; 
v___x_1388_ = lean_array_to_list(v_a_1384_);
v___x_1389_ = lean_box(0);
v___x_1390_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__2(v___x_1388_, v___x_1389_);
v___x_1391_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__2);
v___x_1392_ = l_Lean_MessageData_ofExpr(v_a_1382_);
v___x_1393_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1391_);
lean_ctor_set(v___x_1393_, 1, v___x_1392_);
v___x_1394_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__4);
v___x_1395_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1395_, 0, v___x_1393_);
lean_ctor_set(v___x_1395_, 1, v___x_1394_);
v___x_1396_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__7);
v___x_1397_ = l_Lean_MessageData_joinSep(v___x_1390_, v___x_1396_);
v___x_1398_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1398_, 0, v___x_1395_);
lean_ctor_set(v___x_1398_, 1, v___x_1397_);
v___x_1399_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3(v___x_1398_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
return v___x_1399_;
}
else
{
lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; 
lean_dec(v_a_1384_);
v___x_1400_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___closed__9);
v___x_1401_ = l_Lean_MessageData_ofExpr(v_a_1382_);
v___x_1402_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1402_, 0, v___x_1400_);
lean_ctor_set(v___x_1402_, 1, v___x_1401_);
v___x_1403_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3(v___x_1402_, v___y_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_);
return v___x_1403_;
}
}
else
{
lean_object* v_a_1404_; lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1411_; 
lean_dec(v_a_1382_);
v_a_1404_ = lean_ctor_get(v___x_1383_, 0);
v_isSharedCheck_1411_ = !lean_is_exclusive(v___x_1383_);
if (v_isSharedCheck_1411_ == 0)
{
v___x_1406_ = v___x_1383_;
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
else
{
lean_inc(v_a_1404_);
lean_dec(v___x_1383_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v___x_1409_; 
if (v_isShared_1407_ == 0)
{
v___x_1409_ = v___x_1406_;
goto v_reusejp_1408_;
}
else
{
lean_object* v_reuseFailAlloc_1410_; 
v_reuseFailAlloc_1410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1410_, 0, v_a_1404_);
v___x_1409_ = v_reuseFailAlloc_1410_;
goto v_reusejp_1408_;
}
v_reusejp_1408_:
{
return v___x_1409_;
}
}
}
}
else
{
lean_object* v_a_1412_; lean_object* v___x_1414_; uint8_t v_isShared_1415_; uint8_t v_isSharedCheck_1419_; 
v_a_1412_ = lean_ctor_get(v___x_1381_, 0);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1381_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1414_ = v___x_1381_;
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
else
{
lean_inc(v_a_1412_);
lean_dec(v___x_1381_);
v___x_1414_ = lean_box(0);
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
v_resetjp_1413_:
{
lean_object* v___x_1417_; 
if (v_isShared_1415_ == 0)
{
v___x_1417_ = v___x_1414_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_a_1412_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
}
}
else
{
lean_dec(v_a_1375_);
return v___x_1377_;
}
}
else
{
lean_object* v_a_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1427_; 
v_a_1420_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1427_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1427_ == 0)
{
v___x_1422_ = v___x_1374_;
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_a_1420_);
lean_dec(v___x_1374_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1425_; 
if (v_isShared_1423_ == 0)
{
v___x_1425_ = v___x_1422_;
goto v_reusejp_1424_;
}
else
{
lean_object* v_reuseFailAlloc_1426_; 
v_reuseFailAlloc_1426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1426_, 0, v_a_1420_);
v___x_1425_ = v_reuseFailAlloc_1426_;
goto v_reusejp_1424_;
}
v_reusejp_1424_:
{
return v___x_1425_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___boxed(lean_object* v___x_1428_, lean_object* v___x_1429_, lean_object* v_x_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_){
_start:
{
uint8_t v___x_6692__boxed_1438_; lean_object* v_res_1439_; 
v___x_6692__boxed_1438_ = lean_unbox(v___x_1429_);
v_res_1439_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1(v___x_1428_, v___x_6692__boxed_1438_, v_x_1430_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec(v___y_1434_);
lean_dec_ref(v___y_1433_);
lean_dec(v___y_1432_);
lean_dec_ref(v___y_1431_);
lean_dec_ref(v_x_1430_);
return v_res_1439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1(lean_object* v_x_1440_, lean_object* v_a_1441_, lean_object* v_a_1442_){
_start:
{
lean_object* v___x_1444_; uint8_t v___x_1445_; 
v___x_1444_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23unfold_x3f___00__closed__4));
lean_inc(v_x_1440_);
v___x_1445_ = l_Lean_Syntax_isOfKind(v_x_1440_, v___x_1444_);
if (v___x_1445_ == 0)
{
lean_object* v___x_1446_; 
lean_dec(v_x_1440_);
v___x_1446_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__0___redArg();
return v___x_1446_;
}
else
{
lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___f_1450_; lean_object* v___x_1451_; 
v___x_1447_ = lean_unsigned_to_nat(1u);
v___x_1448_ = l_Lean_Syntax_getArg(v_x_1440_, v___x_1447_);
lean_dec(v_x_1440_);
v___x_1449_ = lean_box(v___x_1445_);
v___f_1450_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___lam__1___boxed), 10, 2);
lean_closure_set(v___f_1450_, 0, v___x_1448_);
lean_closure_set(v___f_1450_, 1, v___x_1449_);
v___x_1451_ = l_Lean_Elab_Command_runTermElabM___redArg(v___f_1450_, v_a_1441_, v_a_1442_);
return v___x_1451_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1___boxed(lean_object* v_x_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_, lean_object* v_a_1455_){
_start:
{
lean_object* v_res_1456_; 
v_res_1456_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1(v_x_1452_, v_a_1453_, v_a_1454_);
lean_dec(v_a_1454_);
lean_dec_ref(v_a_1453_);
return v_res_1456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4(lean_object* v_ref_1457_, lean_object* v_msgData_1458_, uint8_t v_severity_1459_, uint8_t v_isSilent_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
lean_object* v___x_1468_; 
v___x_1468_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___redArg(v_ref_1457_, v_msgData_1458_, v_severity_1459_, v_isSilent_1460_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_);
return v___x_1468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4___boxed(lean_object* v_ref_1469_, lean_object* v_msgData_1470_, lean_object* v_severity_1471_, lean_object* v_isSilent_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_){
_start:
{
uint8_t v_severity_boxed_1480_; uint8_t v_isSilent_boxed_1481_; lean_object* v_res_1482_; 
v_severity_boxed_1480_ = lean_unbox(v_severity_1471_);
v_isSilent_boxed_1481_ = lean_unbox(v_isSilent_1472_);
v_res_1482_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions__Unfold______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23unfold_x3f____1_spec__3_spec__3_spec__4(v_ref_1469_, v_msgData_1470_, v_severity_boxed_1480_, v_isSilent_boxed_1481_, v___y_1473_, v___y_1474_, v___y_1475_, v___y_1476_, v___y_1477_, v___y_1478_);
lean_dec(v___y_1478_);
lean_dec_ref(v___y_1477_);
lean_dec(v___y_1476_);
lean_dec_ref(v___y_1475_);
lean_dec(v___y_1474_);
lean_dec_ref(v___y_1473_);
lean_dec(v_ref_1469_);
return v_res_1482_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(builtin);
}
#ifdef __cplusplus
}
#endif
