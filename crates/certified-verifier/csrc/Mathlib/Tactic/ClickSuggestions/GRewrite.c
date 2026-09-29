// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.GRewrite
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.SectionState public meta import Lean.Meta.ExprLens public meta import Mathlib.Tactic.ClickSuggestions.Util
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_applySymm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_collectFVars(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l_Lean_LocalContext_contains(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_kabstractFindsPositions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*);
extern lean_object* l_Lean_pp_mvars;
extern lean_object* l_Lean_diagnostics;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_Expr_toHeadIndex(lean_object*);
uint8_t l_Lean_instBEqHeadIndex_beq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headNumArgs(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_mkHoleAnnotation(lean_object*);
size_t lean_ptr_addr(lean_object*);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_mathlib_Lean_MVarId_gcongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Meta_mkFreshTypeMVar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getDecLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_gcongrDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_GCongrM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_lt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(26, 46, 54, 245, 81, 108, 136, 63)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "AntisymmRel"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(236, 176, 179, 137, 228, 101, 211, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "b"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(47, 22, 244, 233, 226, 169, 241, 142)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__4_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "invalid relation "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "` is not a relation"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "dummyError"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " doesn't have "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " on either side"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = " is not a generalized relation"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = " is not a relation"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Lean.Expr"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "_private.Lean.Expr.0.Lean.Expr.updateMData!Impl"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mdata expected"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Invalid coordinate "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " for "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__6 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Lensing on types is not supported"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__8 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_GCongr_gcongrDischarger___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_a"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 106, 112, 29, 6, 211, 214, 169)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwKey_isDuplicate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwKey_isDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strong"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "goal-vdash"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__4_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__3_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__5_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__6_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__6_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⊢ "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__9_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__9_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__2_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__7_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__10_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Expected relation, not "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "click_suggestions"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(15, 73, 51, 51, 21, 209, 204, 170)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__3_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " and "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = " do not match according to the head-constant indexing"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " does not unify with "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "no suitable `grw` relation was found"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(v_e_30_, v___y_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___boxed(lean_object* v_e_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0(v_e_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg(lean_object* v_k_44_, uint8_t v_allowLevelAssignments_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_45_, v_k_44_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
v_a_52_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_51_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_51_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
else
{
lean_object* v_a_60_; lean_object* v___x_62_; uint8_t v_isShared_63_; uint8_t v_isSharedCheck_67_; 
v_a_60_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_67_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_67_ == 0)
{
v___x_62_ = v___x_51_;
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
else
{
lean_inc(v_a_60_);
lean_dec(v___x_51_);
v___x_62_ = lean_box(0);
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
v_resetjp_61_:
{
lean_object* v___x_65_; 
if (v_isShared_63_ == 0)
{
v___x_65_ = v___x_62_;
goto v_reusejp_64_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v_a_60_);
v___x_65_ = v_reuseFailAlloc_66_;
goto v_reusejp_64_;
}
v_reusejp_64_:
{
return v___x_65_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg___boxed(lean_object* v_k_68_, lean_object* v_allowLevelAssignments_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_75_; lean_object* v_res_76_; 
v_allowLevelAssignments_boxed_75_ = lean_unbox(v_allowLevelAssignments_69_);
v_res_76_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg(v_k_68_, v_allowLevelAssignments_boxed_75_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1(lean_object* v_00_u03b1_77_, lean_object* v_k_78_, uint8_t v_allowLevelAssignments_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg(v_k_78_, v_allowLevelAssignments_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___boxed(lean_object* v_00_u03b1_86_, lean_object* v_k_87_, lean_object* v_allowLevelAssignments_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_94_; lean_object* v_res_95_; 
v_allowLevelAssignments_boxed_94_ = lean_unbox(v_allowLevelAssignments_88_);
v_res_95_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1(v_00_u03b1_86_, v_k_87_, v_allowLevelAssignments_boxed_94_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0(uint8_t v_symm_96_, lean_object* v_x_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_box(v_symm_96_);
v___x_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
v___x_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0___boxed(lean_object* v_symm_106_, lean_object* v_x_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
uint8_t v_symm_boxed_113_; lean_object* v_res_114_; 
v_symm_boxed_113_ = lean_unbox(v_symm_106_);
v_res_114_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0(v_symm_boxed_113_, v_x_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
lean_dec_ref(v_x_107_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1(lean_object* v_relName_131_, lean_object* v_relation_132_, uint8_t v___x_133_, uint8_t v_symm_134_, lean_object* v_a_135_, lean_object* v_b_136_, lean_object* v___x_137_, lean_object* v___f_138_, lean_object* v___x_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___y_148_; lean_object* v___y_149_; lean_object* v___y_150_; lean_object* v___y_151_; lean_object* v___y_152_; lean_object* v_a_153_; lean_object* v___y_249_; lean_object* v___y_250_; lean_object* v___y_251_; lean_object* v___y_252_; lean_object* v___y_253_; lean_object* v___y_254_; uint8_t v___y_255_; lean_object* v___y_260_; lean_object* v___y_261_; lean_object* v___y_262_; lean_object* v___y_263_; lean_object* v___y_264_; lean_object* v_a_265_; lean_object* v___y_269_; lean_object* v___y_270_; lean_object* v___y_271_; lean_object* v___y_272_; lean_object* v___y_273_; lean_object* v___y_274_; lean_object* v___y_278_; lean_object* v___y_279_; lean_object* v___y_280_; lean_object* v___y_281_; lean_object* v___y_282_; lean_object* v___y_286_; lean_object* v___y_287_; lean_object* v___y_288_; lean_object* v___y_289_; lean_object* v___y_290_; lean_object* v___y_291_; uint8_t v___y_292_; lean_object* v_result_296_; lean_object* v___y_297_; lean_object* v___y_298_; lean_object* v___y_299_; lean_object* v___y_300_; lean_object* v___x_320_; lean_object* v_env_321_; lean_object* v___x_322_; uint8_t v___x_323_; uint8_t v___x_324_; 
v___x_320_ = lean_st_ref_get(v___y_145_);
v_env_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc_ref(v_env_321_);
lean_dec(v___x_320_);
v___x_322_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__9));
v___x_323_ = 1;
v___x_324_ = l_Lean_Environment_contains(v_env_321_, v___x_322_, v___x_323_);
if (v___x_324_ == 0)
{
lean_dec_ref(v_a_141_);
lean_dec(v_a_140_);
v_result_296_ = v___x_139_;
v___y_297_ = v___y_142_;
v___y_298_ = v___y_143_;
v___y_299_ = v___y_144_;
v___y_300_ = v___y_145_;
goto v___jp_295_;
}
else
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_325_ = lean_box(0);
v___x_326_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_326_, 0, v_a_140_);
lean_ctor_set(v___x_326_, 1, v___x_325_);
v___x_327_ = l_Lean_Expr_const___override(v___x_322_, v___x_326_);
lean_inc_ref(v_relation_132_);
v___x_328_ = l_Lean_mkAppB(v___x_327_, v_a_141_, v_relation_132_);
v___x_329_ = lean_box(0);
v___x_330_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_330_, 0, v___x_322_);
lean_ctor_set(v___x_330_, 1, v___x_328_);
lean_ctor_set(v___x_330_, 2, v___x_329_);
v___x_331_ = lean_array_push(v___x_139_, v___x_330_);
v_result_296_ = v___x_331_;
v___y_297_ = v___y_142_;
v___y_298_ = v___y_143_;
v___y_299_ = v___y_144_;
v___y_300_ = v___y_145_;
goto v___jp_295_;
}
v___jp_147_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
lean_inc_ref(v_relation_132_);
lean_inc(v_relName_131_);
v___x_154_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_154_, 0, v_relName_131_);
lean_ctor_set(v___x_154_, 1, v_relation_132_);
lean_ctor_set(v___x_154_, 2, v_a_153_);
v___x_155_ = lean_array_push(v___y_152_, v___x_154_);
v___x_156_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__2));
v___x_157_ = lean_name_eq(v_relName_131_, v___x_156_);
lean_dec(v_relName_131_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; 
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_relation_132_);
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_155_);
return v___x_158_;
}
else
{
lean_object* v___x_159_; lean_object* v_env_160_; lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_159_ = lean_st_ref_get(v___y_151_);
v_env_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc_ref(v_env_160_);
lean_dec(v___x_159_);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__4));
v___x_162_ = l_Lean_Environment_contains(v_env_160_, v___x_161_, v___x_157_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_relation_132_);
v___x_163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_163_, 0, v___x_155_);
return v___x_163_;
}
else
{
lean_object* v___x_164_; 
v___x_164_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v___x_161_, v___y_149_, v___y_150_, v___y_148_, v___y_151_);
if (lean_obj_tag(v___x_164_) == 0)
{
lean_object* v_a_165_; lean_object* v___x_166_; 
v_a_165_ = lean_ctor_get(v___x_164_, 0);
lean_inc(v_a_165_);
lean_dec_ref_known(v___x_164_, 1);
lean_inc(v___y_151_);
lean_inc_ref(v___y_148_);
lean_inc(v___y_150_);
lean_inc_ref(v___y_149_);
v___x_166_ = lean_infer_type(v_a_165_, v___y_149_, v___y_150_, v___y_148_, v___y_151_);
if (lean_obj_tag(v___x_166_) == 0)
{
lean_object* v_a_167_; lean_object* v___x_168_; 
v_a_167_ = lean_ctor_get(v___x_166_, 0);
lean_inc(v_a_167_);
lean_dec_ref_known(v___x_166_, 1);
v___x_168_ = l_Lean_Meta_forallMetaTelescope(v_a_167_, v___x_133_, v___y_149_, v___y_150_, v___y_148_, v___y_151_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; lean_object* v_snd_170_; lean_object* v_fst_171_; lean_object* v_snd_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_169_);
lean_dec_ref_known(v___x_168_, 1);
v_snd_170_ = lean_ctor_get(v_a_169_, 1);
lean_inc(v_snd_170_);
v_fst_171_ = lean_ctor_get(v_a_169_, 0);
lean_inc(v_fst_171_);
lean_dec(v_a_169_);
v_snd_172_ = lean_ctor_get(v_snd_170_, 1);
lean_inc(v_snd_172_);
lean_dec(v_snd_170_);
v___x_173_ = l_Lean_Expr_appFn_x21(v_snd_172_);
lean_dec(v_snd_172_);
v___x_174_ = l_Lean_Expr_appFn_x21(v___x_173_);
lean_dec_ref(v___x_173_);
v___x_175_ = l_Lean_Meta_isExprDefEq(v___x_174_, v_relation_132_, v___y_149_, v___y_150_, v___y_148_, v___y_151_);
if (lean_obj_tag(v___x_175_) == 0)
{
lean_object* v_a_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_215_; 
v_a_176_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_215_ == 0)
{
v___x_178_ = v___x_175_;
v_isShared_179_ = v_isSharedCheck_215_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_a_176_);
lean_dec(v___x_175_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_215_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
uint8_t v___x_180_; 
v___x_180_ = lean_unbox(v_a_176_);
lean_dec(v_a_176_);
if (v___x_180_ == 0)
{
lean_object* v___x_182_; 
lean_dec(v_fst_171_);
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 0, v___x_155_);
v___x_182_ = v___x_178_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v___x_155_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
else
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
lean_del_object(v___x_178_);
v___x_184_ = l_Lean_instInhabitedExpr;
v___x_185_ = lean_array_get_size(v_fst_171_);
v___x_186_ = lean_unsigned_to_nat(1u);
v___x_187_ = lean_nat_sub(v___x_185_, v___x_186_);
v___x_188_ = lean_array_get(v___x_184_, v_fst_171_, v___x_187_);
lean_dec(v___x_187_);
lean_dec(v_fst_171_);
lean_inc(v___y_150_);
v___x_189_ = lean_infer_type(v___x_188_, v___y_149_, v___y_150_, v___y_148_, v___y_151_);
if (lean_obj_tag(v___x_189_) == 0)
{
lean_object* v_a_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_206_; 
v_a_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_a_190_);
lean_dec_ref_known(v___x_189_, 1);
v___x_191_ = l_Lean_Expr_appFn_x21(v_a_190_);
lean_dec(v_a_190_);
v___x_192_ = l_Lean_Expr_appFn_x21(v___x_191_);
lean_dec_ref(v___x_191_);
v___x_193_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(v___x_192_, v___y_150_);
lean_dec(v___y_150_);
v_a_194_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_206_ == 0)
{
v___x_196_ = v___x_193_;
v_isShared_197_ = v_isSharedCheck_206_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_dec(v___x_193_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_206_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_204_; 
v___x_198_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___closed__7));
v___x_199_ = lean_box(v_symm_134_);
v___x_200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
v___x_201_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_201_, 0, v___x_198_);
lean_ctor_set(v___x_201_, 1, v_a_194_);
lean_ctor_set(v___x_201_, 2, v___x_200_);
v___x_202_ = lean_array_push(v___x_155_, v___x_201_);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 0, v___x_202_);
v___x_204_ = v___x_196_;
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
else
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_214_; 
lean_dec_ref(v___x_155_);
lean_dec(v___y_150_);
v_a_207_ = lean_ctor_get(v___x_189_, 0);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_189_);
if (v_isSharedCheck_214_ == 0)
{
v___x_209_ = v___x_189_;
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_189_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_212_; 
if (v_isShared_210_ == 0)
{
v___x_212_ = v___x_209_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_a_207_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
}
}
}
}
else
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_223_; 
lean_dec(v_fst_171_);
lean_dec_ref(v___x_155_);
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
v_a_216_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_223_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_223_ == 0)
{
v___x_218_ = v___x_175_;
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_175_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_221_; 
if (v_isShared_219_ == 0)
{
v___x_221_ = v___x_218_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v_a_216_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
else
{
lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_231_; 
lean_dec_ref(v___x_155_);
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_relation_132_);
v_a_224_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_231_ == 0)
{
v___x_226_ = v___x_168_;
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_168_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_229_; 
if (v_isShared_227_ == 0)
{
v___x_229_ = v___x_226_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_a_224_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
}
else
{
lean_object* v_a_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_239_; 
lean_dec_ref(v___x_155_);
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_relation_132_);
v_a_232_ = lean_ctor_get(v___x_166_, 0);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_166_);
if (v_isSharedCheck_239_ == 0)
{
v___x_234_ = v___x_166_;
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_a_232_);
lean_dec(v___x_166_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v_a_232_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
}
else
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
lean_dec_ref(v___x_155_);
lean_dec(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_relation_132_);
v_a_240_ = lean_ctor_get(v___x_164_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_164_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___x_164_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___x_164_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
}
v___jp_248_:
{
if (v___y_255_ == 0)
{
lean_object* v___x_256_; lean_object* v___x_257_; 
lean_dec_ref(v___y_249_);
v___x_256_ = lean_box(v_symm_134_);
v___x_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
v___y_148_ = v___y_250_;
v___y_149_ = v___y_251_;
v___y_150_ = v___y_253_;
v___y_151_ = v___y_252_;
v___y_152_ = v___y_254_;
v_a_153_ = v___x_257_;
goto v___jp_147_;
}
else
{
lean_object* v___x_258_; 
lean_dec_ref(v___y_254_);
lean_dec(v___y_253_);
lean_dec(v___y_252_);
lean_dec_ref(v___y_251_);
lean_dec_ref(v___y_250_);
lean_dec_ref(v_relation_132_);
lean_dec(v_relName_131_);
v___x_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_258_, 0, v___y_249_);
return v___x_258_;
}
}
v___jp_259_:
{
uint8_t v___x_266_; 
v___x_266_ = l_Lean_Exception_isInterrupt(v_a_265_);
if (v___x_266_ == 0)
{
uint8_t v___x_267_; 
lean_inc_ref(v_a_265_);
v___x_267_ = l_Lean_Exception_isRuntime(v_a_265_);
v___y_249_ = v_a_265_;
v___y_250_ = v___y_260_;
v___y_251_ = v___y_261_;
v___y_252_ = v___y_263_;
v___y_253_ = v___y_262_;
v___y_254_ = v___y_264_;
v___y_255_ = v___x_267_;
goto v___jp_248_;
}
else
{
v___y_249_ = v_a_265_;
v___y_250_ = v___y_260_;
v___y_251_ = v___y_261_;
v___y_252_ = v___y_263_;
v___y_253_ = v___y_262_;
v___y_254_ = v___y_264_;
v___y_255_ = v___x_266_;
goto v___jp_248_;
}
}
v___jp_268_:
{
if (lean_obj_tag(v___y_274_) == 0)
{
lean_object* v_a_275_; 
v_a_275_ = lean_ctor_get(v___y_274_, 0);
lean_inc(v_a_275_);
lean_dec_ref_known(v___y_274_, 1);
v___y_148_ = v___y_269_;
v___y_149_ = v___y_270_;
v___y_150_ = v___y_272_;
v___y_151_ = v___y_271_;
v___y_152_ = v___y_273_;
v_a_153_ = v_a_275_;
goto v___jp_147_;
}
else
{
lean_object* v_a_276_; 
v_a_276_ = lean_ctor_get(v___y_274_, 0);
lean_inc(v_a_276_);
lean_dec_ref_known(v___y_274_, 1);
v___y_260_ = v___y_269_;
v___y_261_ = v___y_270_;
v___y_262_ = v___y_272_;
v___y_263_ = v___y_271_;
v___y_264_ = v___y_273_;
v_a_265_ = v_a_276_;
goto v___jp_259_;
}
}
v___jp_277_:
{
lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_283_ = lean_box(v_symm_134_);
v___x_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
v___y_148_ = v___y_278_;
v___y_149_ = v___y_279_;
v___y_150_ = v___y_281_;
v___y_151_ = v___y_280_;
v___y_152_ = v___y_282_;
v_a_153_ = v___x_284_;
goto v___jp_147_;
}
v___jp_285_:
{
if (v___y_292_ == 0)
{
lean_dec_ref(v___y_290_);
lean_dec_ref(v_a_135_);
v___y_278_ = v___y_286_;
v___y_279_ = v___y_287_;
v___y_280_ = v___y_289_;
v___y_281_ = v___y_288_;
v___y_282_ = v___y_291_;
goto v___jp_277_;
}
else
{
uint8_t v___x_293_; 
v___x_293_ = lean_expr_eqv(v_a_135_, v___y_290_);
lean_dec_ref(v___y_290_);
lean_dec_ref(v_a_135_);
if (v___x_293_ == 0)
{
v___y_278_ = v___y_286_;
v___y_279_ = v___y_287_;
v___y_280_ = v___y_289_;
v___y_281_ = v___y_288_;
v___y_282_ = v___y_291_;
goto v___jp_277_;
}
else
{
lean_object* v___x_294_; 
v___x_294_ = lean_box(0);
v___y_148_ = v___y_286_;
v___y_149_ = v___y_287_;
v___y_150_ = v___y_288_;
v___y_151_ = v___y_289_;
v___y_152_ = v___y_291_;
v_a_153_ = v___x_294_;
goto v___jp_147_;
}
}
}
v___jp_295_:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
lean_inc_ref(v_b_136_);
lean_inc_ref(v_a_135_);
lean_inc_ref(v_relation_132_);
v___x_301_ = l_Lean_mkAppB(v_relation_132_, v_a_135_, v_b_136_);
v___x_302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
v___x_303_ = l_Lean_Meta_mkFreshExprMVar(v___x_302_, v___x_133_, v___x_137_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
if (lean_obj_tag(v___x_303_) == 0)
{
lean_object* v_a_304_; lean_object* v___x_305_; 
v_a_304_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_a_304_);
lean_dec_ref_known(v___x_303_, 1);
v___x_305_ = l_Lean_Expr_applySymm(v_a_304_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v_a_306_; lean_object* v___x_307_; 
v_a_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v___x_305_, 1);
lean_inc(v___y_300_);
lean_inc_ref(v___y_299_);
lean_inc(v___y_298_);
lean_inc_ref(v___y_297_);
v___x_307_ = lean_infer_type(v_a_306_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v_a_308_; 
v_a_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc(v_a_308_);
lean_dec_ref_known(v___x_307_, 1);
if (lean_obj_tag(v_a_308_) == 5)
{
lean_object* v_fn_309_; 
v_fn_309_ = lean_ctor_get(v_a_308_, 0);
if (lean_obj_tag(v_fn_309_) == 5)
{
lean_object* v_arg_310_; lean_object* v_fn_311_; lean_object* v_arg_312_; uint8_t v___x_313_; 
lean_inc_ref(v_fn_309_);
lean_dec_ref(v___f_138_);
v_arg_310_ = lean_ctor_get(v_a_308_, 1);
lean_inc_ref(v_arg_310_);
lean_dec_ref_known(v_a_308_, 2);
v_fn_311_ = lean_ctor_get(v_fn_309_, 0);
lean_inc_ref(v_fn_311_);
v_arg_312_ = lean_ctor_get(v_fn_309_, 1);
lean_inc_ref(v_arg_312_);
lean_dec_ref_known(v_fn_309_, 2);
v___x_313_ = lean_expr_eqv(v_fn_311_, v_relation_132_);
lean_dec_ref(v_fn_311_);
if (v___x_313_ == 0)
{
lean_dec_ref(v_arg_312_);
lean_dec_ref(v_b_136_);
v___y_286_ = v___y_299_;
v___y_287_ = v___y_297_;
v___y_288_ = v___y_298_;
v___y_289_ = v___y_300_;
v___y_290_ = v_arg_310_;
v___y_291_ = v_result_296_;
v___y_292_ = v___x_313_;
goto v___jp_285_;
}
else
{
uint8_t v___x_314_; 
v___x_314_ = lean_expr_eqv(v_b_136_, v_arg_312_);
lean_dec_ref(v_arg_312_);
lean_dec_ref(v_b_136_);
v___y_286_ = v___y_299_;
v___y_287_ = v___y_297_;
v___y_288_ = v___y_298_;
v___y_289_ = v___y_300_;
v___y_290_ = v_arg_310_;
v___y_291_ = v_result_296_;
v___y_292_ = v___x_314_;
goto v___jp_285_;
}
}
else
{
lean_object* v___x_315_; 
lean_dec_ref(v_b_136_);
lean_dec_ref(v_a_135_);
lean_inc(v___y_300_);
lean_inc_ref(v___y_299_);
lean_inc(v___y_298_);
lean_inc_ref(v___y_297_);
v___x_315_ = lean_apply_6(v___f_138_, v_a_308_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, lean_box(0));
v___y_269_ = v___y_299_;
v___y_270_ = v___y_297_;
v___y_271_ = v___y_300_;
v___y_272_ = v___y_298_;
v___y_273_ = v_result_296_;
v___y_274_ = v___x_315_;
goto v___jp_268_;
}
}
else
{
lean_object* v___x_316_; 
lean_dec_ref(v_b_136_);
lean_dec_ref(v_a_135_);
lean_inc(v___y_300_);
lean_inc_ref(v___y_299_);
lean_inc(v___y_298_);
lean_inc_ref(v___y_297_);
v___x_316_ = lean_apply_6(v___f_138_, v_a_308_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, lean_box(0));
v___y_269_ = v___y_299_;
v___y_270_ = v___y_297_;
v___y_271_ = v___y_300_;
v___y_272_ = v___y_298_;
v___y_273_ = v_result_296_;
v___y_274_ = v___x_316_;
goto v___jp_268_;
}
}
else
{
lean_object* v_a_317_; 
lean_dec_ref(v___f_138_);
lean_dec_ref(v_b_136_);
lean_dec_ref(v_a_135_);
v_a_317_ = lean_ctor_get(v___x_307_, 0);
lean_inc(v_a_317_);
lean_dec_ref_known(v___x_307_, 1);
v___y_260_ = v___y_299_;
v___y_261_ = v___y_297_;
v___y_262_ = v___y_298_;
v___y_263_ = v___y_300_;
v___y_264_ = v_result_296_;
v_a_265_ = v_a_317_;
goto v___jp_259_;
}
}
else
{
lean_object* v_a_318_; 
lean_dec_ref(v___f_138_);
lean_dec_ref(v_b_136_);
lean_dec_ref(v_a_135_);
v_a_318_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_318_);
lean_dec_ref_known(v___x_305_, 1);
v___y_260_ = v___y_299_;
v___y_261_ = v___y_297_;
v___y_262_ = v___y_298_;
v___y_263_ = v___y_300_;
v___y_264_ = v_result_296_;
v_a_265_ = v_a_318_;
goto v___jp_259_;
}
}
else
{
lean_object* v_a_319_; 
lean_dec_ref(v___f_138_);
lean_dec_ref(v_b_136_);
lean_dec_ref(v_a_135_);
v_a_319_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_a_319_);
lean_dec_ref_known(v___x_303_, 1);
v___y_260_ = v___y_299_;
v___y_261_ = v___y_297_;
v___y_262_ = v___y_298_;
v___y_263_ = v___y_300_;
v___y_264_ = v_result_296_;
v_a_265_ = v_a_319_;
goto v___jp_259_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___boxed(lean_object* v_relName_332_, lean_object* v_relation_333_, lean_object* v___x_334_, lean_object* v_symm_335_, lean_object* v_a_336_, lean_object* v_b_337_, lean_object* v___x_338_, lean_object* v___f_339_, lean_object* v___x_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
uint8_t v___x_8862__boxed_348_; uint8_t v_symm_boxed_349_; lean_object* v_res_350_; 
v___x_8862__boxed_348_ = lean_unbox(v___x_334_);
v_symm_boxed_349_ = lean_unbox(v_symm_335_);
v_res_350_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1(v_relName_332_, v_relation_333_, v___x_8862__boxed_348_, v_symm_boxed_349_, v_a_336_, v_b_337_, v___x_338_, v___f_339_, v___x_340_, v_a_341_, v_a_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2(lean_object* v_relName_353_, lean_object* v_relation_354_, uint8_t v___x_355_, uint8_t v_symm_356_, lean_object* v_a_357_, lean_object* v___x_358_, lean_object* v___f_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_b_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___f_371_; uint8_t v___x_372_; lean_object* v___x_373_; 
v___x_368_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0));
v___x_369_ = lean_box(v___x_355_);
v___x_370_ = lean_box(v_symm_356_);
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__1___boxed), 16, 11);
lean_closure_set(v___f_371_, 0, v_relName_353_);
lean_closure_set(v___f_371_, 1, v_relation_354_);
lean_closure_set(v___f_371_, 2, v___x_369_);
lean_closure_set(v___f_371_, 3, v___x_370_);
lean_closure_set(v___f_371_, 4, v_a_357_);
lean_closure_set(v___f_371_, 5, v_b_362_);
lean_closure_set(v___f_371_, 6, v___x_358_);
lean_closure_set(v___f_371_, 7, v___f_359_);
lean_closure_set(v___f_371_, 8, v___x_368_);
lean_closure_set(v___f_371_, 9, v_a_360_);
lean_closure_set(v___f_371_, 10, v_a_361_);
v___x_372_ = 0;
v___x_373_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__1___redArg(v___f_371_, v___x_372_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___boxed(lean_object* v_relName_374_, lean_object* v_relation_375_, lean_object* v___x_376_, lean_object* v_symm_377_, lean_object* v_a_378_, lean_object* v___x_379_, lean_object* v___f_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_b_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
uint8_t v___x_9281__boxed_389_; uint8_t v_symm_boxed_390_; lean_object* v_res_391_; 
v___x_9281__boxed_389_ = lean_unbox(v___x_376_);
v_symm_boxed_390_ = lean_unbox(v_symm_377_);
v_res_391_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2(v_relName_374_, v_relation_375_, v___x_9281__boxed_389_, v_symm_boxed_390_, v_a_378_, v___x_379_, v___f_380_, v_a_381_, v_a_382_, v_b_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0(lean_object* v_k_392_, lean_object* v_b_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_){
_start:
{
lean_object* v___x_399_; 
lean_inc(v___y_397_);
lean_inc_ref(v___y_396_);
lean_inc(v___y_395_);
lean_inc_ref(v___y_394_);
v___x_399_ = lean_apply_6(v_k_392_, v_b_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, lean_box(0));
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0___boxed(lean_object* v_k_400_, lean_object* v_b_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0(v_k_400_, v_b_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg(lean_object* v_name_408_, uint8_t v_bi_409_, lean_object* v_type_410_, lean_object* v_k_411_, uint8_t v_kind_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v___f_418_; lean_object* v___x_419_; 
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_418_, 0, v_k_411_);
v___x_419_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_408_, v_bi_409_, v_type_410_, v___f_418_, v_kind_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
if (lean_obj_tag(v___x_419_) == 0)
{
lean_object* v_a_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_427_; 
v_a_420_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_427_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_427_ == 0)
{
v___x_422_ = v___x_419_;
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_a_420_);
lean_dec(v___x_419_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___x_425_; 
if (v_isShared_423_ == 0)
{
v___x_425_ = v___x_422_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_a_420_);
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
lean_object* v_a_428_; lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_435_; 
v_a_428_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_435_ == 0)
{
v___x_430_ = v___x_419_;
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
else
{
lean_inc(v_a_428_);
lean_dec(v___x_419_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v___x_433_; 
if (v_isShared_431_ == 0)
{
v___x_433_ = v___x_430_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_a_428_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___boxed(lean_object* v_name_436_, lean_object* v_bi_437_, lean_object* v_type_438_, lean_object* v_k_439_, lean_object* v_kind_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
uint8_t v_bi_boxed_446_; uint8_t v_kind_boxed_447_; lean_object* v_res_448_; 
v_bi_boxed_446_ = lean_unbox(v_bi_437_);
v_kind_boxed_447_ = lean_unbox(v_kind_440_);
v_res_448_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg(v_name_436_, v_bi_boxed_446_, v_type_438_, v_k_439_, v_kind_boxed_447_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(lean_object* v_name_449_, lean_object* v_type_450_, lean_object* v_k_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_){
_start:
{
uint8_t v___x_457_; uint8_t v___x_458_; lean_object* v___x_459_; 
v___x_457_ = 0;
v___x_458_ = 0;
v___x_459_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg(v_name_449_, v___x_457_, v_type_450_, v_k_451_, v___x_458_, v___y_452_, v___y_453_, v___y_454_, v___y_455_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg___boxed(lean_object* v_name_460_, lean_object* v_type_461_, lean_object* v_k_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(v_name_460_, v_type_461_, v_k_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3(lean_object* v_relName_472_, lean_object* v_relation_473_, uint8_t v___x_474_, uint8_t v_symm_475_, lean_object* v___x_476_, lean_object* v___f_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___f_488_; lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_486_ = lean_box(v___x_474_);
v___x_487_ = lean_box(v_symm_475_);
lean_inc_ref(v_a_479_);
v___f_488_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___boxed), 15, 9);
lean_closure_set(v___f_488_, 0, v_relName_472_);
lean_closure_set(v___f_488_, 1, v_relation_473_);
lean_closure_set(v___f_488_, 2, v___x_486_);
lean_closure_set(v___f_488_, 3, v___x_487_);
lean_closure_set(v___f_488_, 4, v_a_480_);
lean_closure_set(v___f_488_, 5, v___x_476_);
lean_closure_set(v___f_488_, 6, v___f_477_);
lean_closure_set(v___f_488_, 7, v_a_478_);
lean_closure_set(v___f_488_, 8, v_a_479_);
v___x_489_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___closed__1));
v___x_490_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(v___x_489_, v_a_479_, v___f_488_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___boxed(lean_object* v_relName_491_, lean_object* v_relation_492_, lean_object* v___x_493_, lean_object* v_symm_494_, lean_object* v___x_495_, lean_object* v___f_496_, lean_object* v_a_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
uint8_t v___x_9420__boxed_505_; uint8_t v_symm_boxed_506_; lean_object* v_res_507_; 
v___x_9420__boxed_505_ = lean_unbox(v___x_493_);
v_symm_boxed_506_ = lean_unbox(v_symm_494_);
v_res_507_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3(v_relName_491_, v_relation_492_, v___x_9420__boxed_505_, v_symm_boxed_506_, v___x_495_, v___f_496_, v_a_497_, v_a_498_, v_a_499_, v___y_500_, v___y_501_, v___y_502_, v___y_503_);
lean_dec(v___y_503_);
lean_dec_ref(v___y_502_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4(lean_object* v_msgData_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
lean_object* v___x_514_; lean_object* v_env_515_; lean_object* v___x_516_; lean_object* v_mctx_517_; lean_object* v_lctx_518_; lean_object* v_options_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v___x_514_ = lean_st_ref_get(v___y_512_);
v_env_515_ = lean_ctor_get(v___x_514_, 0);
lean_inc_ref(v_env_515_);
lean_dec(v___x_514_);
v___x_516_ = lean_st_ref_get(v___y_510_);
v_mctx_517_ = lean_ctor_get(v___x_516_, 0);
lean_inc_ref(v_mctx_517_);
lean_dec(v___x_516_);
v_lctx_518_ = lean_ctor_get(v___y_509_, 2);
v_options_519_ = lean_ctor_get(v___y_511_, 2);
lean_inc_ref(v_options_519_);
lean_inc_ref(v_lctx_518_);
v___x_520_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_520_, 0, v_env_515_);
lean_ctor_set(v___x_520_, 1, v_mctx_517_);
lean_ctor_set(v___x_520_, 2, v_lctx_518_);
lean_ctor_set(v___x_520_, 3, v_options_519_);
v___x_521_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v_msgData_508_);
v___x_522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_522_, 0, v___x_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4___boxed(lean_object* v_msgData_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4(v_msgData_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(lean_object* v_msg_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
lean_object* v_ref_536_; lean_object* v___x_537_; lean_object* v_a_538_; lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_546_; 
v_ref_536_ = lean_ctor_get(v___y_533_, 5);
v___x_537_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4(v_msg_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_);
v_a_538_ = lean_ctor_get(v___x_537_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_546_ == 0)
{
v___x_540_ = v___x_537_;
v_isShared_541_ = v_isSharedCheck_546_;
goto v_resetjp_539_;
}
else
{
lean_inc(v_a_538_);
lean_dec(v___x_537_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_546_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
lean_object* v___x_542_; lean_object* v___x_544_; 
lean_inc(v_ref_536_);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v_ref_536_);
lean_ctor_set(v___x_542_, 1, v_a_538_);
if (v_isShared_541_ == 0)
{
lean_ctor_set_tag(v___x_540_, 1);
lean_ctor_set(v___x_540_, 0, v___x_542_);
v___x_544_ = v___x_540_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg___boxed(lean_object* v_msg_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
lean_object* v_res_553_; 
v_res_553_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v_msg_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
return v_res_553_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2(void){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_557_ = lean_unsigned_to_nat(0u);
v___x_558_ = l_Lean_Level_ofNat(v___x_557_);
return v___x_558_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3(void){
_start:
{
lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_559_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__2);
v___x_560_ = l_Lean_Expr_sort___override(v___x_559_);
return v___x_560_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7(void){
_start:
{
lean_object* v___x_565_; lean_object* v___x_566_; 
v___x_565_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__6));
v___x_566_ = l_Lean_stringToMessageData(v___x_565_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward(lean_object* v_relName_567_, lean_object* v_relation_568_, uint8_t v_symm_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
lean_object* v___x_575_; 
lean_inc(v_a_573_);
lean_inc_ref(v_a_572_);
lean_inc(v_a_571_);
lean_inc_ref(v_a_570_);
lean_inc_ref(v_relation_568_);
v___x_575_ = lean_infer_type(v_relation_568_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
if (lean_obj_tag(v___x_575_) == 0)
{
lean_object* v_a_576_; uint8_t v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v_a_576_ = lean_ctor_get(v___x_575_, 0);
lean_inc(v_a_576_);
lean_dec_ref_known(v___x_575_, 1);
v___x_577_ = 0;
v___x_578_ = lean_box(0);
v___x_579_ = l_Lean_Meta_mkFreshTypeMVar(v___x_577_, v___x_578_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
if (lean_obj_tag(v___x_579_) == 0)
{
lean_object* v_a_580_; lean_object* v___x_581_; lean_object* v___x_582_; uint8_t v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v_a_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc_n(v_a_580_, 3);
lean_dec_ref_known(v___x_579_, 1);
v___x_581_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__1));
v___x_582_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__3);
v___x_583_ = 0;
v___x_584_ = l_Lean_Expr_forallE___override(v___x_581_, v_a_580_, v___x_582_, v___x_583_);
v___x_585_ = l_Lean_Expr_forallE___override(v___x_581_, v_a_580_, v___x_584_, v___x_583_);
v___x_586_ = l_Lean_Meta_isExprDefEq(v_a_576_, v___x_585_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v_a_587_; lean_object* v___x_588_; lean_object* v___f_589_; lean_object* v___y_591_; lean_object* v___y_592_; lean_object* v___y_593_; lean_object* v___y_594_; uint8_t v___x_612_; 
v_a_587_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_a_587_);
lean_dec_ref_known(v___x_586_, 1);
v___x_588_ = lean_box(v_symm_569_);
v___f_589_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__0___boxed), 7, 1);
lean_closure_set(v___f_589_, 0, v___x_588_);
v___x_612_ = lean_unbox(v_a_587_);
lean_dec(v_a_587_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_624_; 
lean_dec_ref(v___f_589_);
lean_dec(v_a_580_);
lean_dec(v_relName_567_);
v___x_613_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__7);
v___x_614_ = l_Lean_MessageData_ofExpr(v_relation_568_);
v___x_615_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_613_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
v___x_616_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_615_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
v_a_617_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_624_ == 0)
{
v___x_619_ = v___x_616_;
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_616_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___x_622_; 
if (v_isShared_620_ == 0)
{
v___x_622_ = v___x_619_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_a_617_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
else
{
v___y_591_ = v_a_570_;
v___y_592_ = v_a_571_;
v___y_593_ = v_a_572_;
v___y_594_ = v_a_573_;
goto v___jp_590_;
}
v___jp_590_:
{
lean_object* v___x_595_; lean_object* v_a_596_; lean_object* v___x_597_; 
v___x_595_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(v_a_580_, v___y_592_);
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc_n(v_a_596_, 2);
lean_dec_ref(v___x_595_);
v___x_597_ = l_Lean_Meta_getDecLevel(v_a_596_, v___y_591_, v___y_592_, v___y_593_, v___y_594_);
if (lean_obj_tag(v___x_597_) == 0)
{
lean_object* v_a_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___f_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v_a_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_a_598_);
lean_dec_ref_known(v___x_597_, 1);
v___x_599_ = lean_box(v___x_577_);
v___x_600_ = lean_box(v_symm_569_);
lean_inc(v_a_596_);
v___f_601_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__3___boxed), 14, 8);
lean_closure_set(v___f_601_, 0, v_relName_567_);
lean_closure_set(v___f_601_, 1, v_relation_568_);
lean_closure_set(v___f_601_, 2, v___x_599_);
lean_closure_set(v___f_601_, 3, v___x_600_);
lean_closure_set(v___f_601_, 4, v___x_578_);
lean_closure_set(v___f_601_, 5, v___f_589_);
lean_closure_set(v___f_601_, 6, v_a_598_);
lean_closure_set(v___f_601_, 7, v_a_596_);
v___x_602_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___closed__5));
v___x_603_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(v___x_602_, v_a_596_, v___f_601_, v___y_591_, v___y_592_, v___y_593_, v___y_594_);
return v___x_603_;
}
else
{
lean_object* v_a_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_611_; 
lean_dec(v_a_596_);
lean_dec_ref(v___f_589_);
lean_dec_ref(v_relation_568_);
lean_dec(v_relName_567_);
v_a_604_ = lean_ctor_get(v___x_597_, 0);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_597_);
if (v_isSharedCheck_611_ == 0)
{
v___x_606_ = v___x_597_;
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_a_604_);
lean_dec(v___x_597_);
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
else
{
lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_632_; 
lean_dec(v_a_580_);
lean_dec_ref(v_relation_568_);
lean_dec(v_relName_567_);
v_a_625_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_632_ == 0)
{
v___x_627_ = v___x_586_;
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_586_);
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
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
lean_dec(v_a_576_);
lean_dec_ref(v_relation_568_);
lean_dec(v_relName_567_);
v_a_633_ = lean_ctor_get(v___x_579_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_579_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_579_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_a_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
else
{
lean_object* v_a_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_648_; 
lean_dec_ref(v_relation_568_);
lean_dec(v_relName_567_);
v_a_641_ = lean_ctor_get(v___x_575_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v___x_575_);
if (v_isSharedCheck_648_ == 0)
{
v___x_643_ = v___x_575_;
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_a_641_);
lean_dec(v___x_575_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_646_; 
if (v_isShared_644_ == 0)
{
v___x_646_ = v___x_643_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v_a_641_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___boxed(lean_object* v_relName_649_, lean_object* v_relation_650_, lean_object* v_symm_651_, lean_object* v_a_652_, lean_object* v_a_653_, lean_object* v_a_654_, lean_object* v_a_655_, lean_object* v_a_656_){
_start:
{
uint8_t v_symm_boxed_657_; lean_object* v_res_658_; 
v_symm_boxed_657_ = lean_unbox(v_symm_651_);
v_res_658_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward(v_relName_649_, v_relation_650_, v_symm_boxed_657_, v_a_652_, v_a_653_, v_a_654_, v_a_655_);
lean_dec(v_a_655_);
lean_dec_ref(v_a_654_);
lean_dec(v_a_653_);
lean_dec_ref(v_a_652_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2(lean_object* v_00_u03b1_659_, lean_object* v_name_660_, uint8_t v_bi_661_, lean_object* v_type_662_, lean_object* v_k_663_, uint8_t v_kind_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_){
_start:
{
lean_object* v___x_670_; 
v___x_670_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg(v_name_660_, v_bi_661_, v_type_662_, v_k_663_, v_kind_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___boxed(lean_object* v_00_u03b1_671_, lean_object* v_name_672_, lean_object* v_bi_673_, lean_object* v_type_674_, lean_object* v_k_675_, lean_object* v_kind_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_){
_start:
{
uint8_t v_bi_boxed_682_; uint8_t v_kind_boxed_683_; lean_object* v_res_684_; 
v_bi_boxed_682_ = lean_unbox(v_bi_673_);
v_kind_boxed_683_ = lean_unbox(v_kind_676_);
v_res_684_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2(v_00_u03b1_671_, v_name_672_, v_bi_boxed_682_, v_type_674_, v_k_675_, v_kind_boxed_683_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2(lean_object* v_00_u03b1_685_, lean_object* v_name_686_, lean_object* v_type_687_, lean_object* v_k_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(v_name_686_, v_type_687_, v_k_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___boxed(lean_object* v_00_u03b1_695_, lean_object* v_name_696_, lean_object* v_type_697_, lean_object* v_k_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2(v_00_u03b1_695_, v_name_696_, v_type_697_, v_k_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
return v_res_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3(lean_object* v_00_u03b1_705_, lean_object* v_msg_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v_msg_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___boxed(lean_object* v_00_u03b1_713_, lean_object* v_msg_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_){
_start:
{
lean_object* v_res_720_; 
v_res_720_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3(v_00_u03b1_713_, v_msg_714_, v___y_715_, v___y_716_, v___y_717_, v___y_718_);
lean_dec(v___y_718_);
lean_dec_ref(v___y_717_);
lean_dec(v___y_716_);
lean_dec_ref(v___y_715_);
return v_res_720_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1(void){
_start:
{
lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_722_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__0));
v___x_723_ = l_Lean_stringToMessageData(v___x_722_);
return v___x_723_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3(void){
_start:
{
lean_object* v___x_725_; lean_object* v___x_726_; 
v___x_725_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__2));
v___x_726_ = l_Lean_stringToMessageData(v___x_725_);
return v___x_726_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6(void){
_start:
{
lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_730_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__5));
v___x_731_ = l_Lean_MessageData_ofFormat(v___x_730_);
return v___x_731_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7(void){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_732_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__6);
v___x_733_ = lean_box(0);
v___x_734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
lean_ctor_set(v___x_734_, 1, v___x_732_);
return v___x_734_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9(void){
_start:
{
lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_736_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__8));
v___x_737_ = l_Lean_stringToMessageData(v___x_736_);
return v___x_737_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11(void){
_start:
{
lean_object* v___x_739_; lean_object* v___x_740_; 
v___x_739_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__10));
v___x_740_ = l_Lean_stringToMessageData(v___x_739_);
return v___x_740_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13(void){
_start:
{
lean_object* v___x_742_; lean_object* v___x_743_; 
v___x_742_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__12));
v___x_743_ = l_Lean_stringToMessageData(v___x_742_);
return v___x_743_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_747_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__16));
v___x_748_ = l_Lean_stringToMessageData(v___x_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger(lean_object* v_ref_749_, uint8_t v_hyp_x3f_750_, lean_object* v_fvar_751_, lean_object* v_goal_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_){
_start:
{
lean_object* v___x_758_; 
v___x_758_ = l_Lean_MVarId_getType(v_goal_752_, v_a_753_, v_a_754_, v_a_755_, v_a_756_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_760_; lean_object* v_a_761_; lean_object* v___y_763_; lean_object* v___y_764_; lean_object* v___y_765_; lean_object* v___y_766_; 
v_a_759_ = lean_ctor_get(v___x_758_, 0);
lean_inc(v_a_759_);
lean_dec_ref_known(v___x_758_, 1);
v___x_760_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__0___redArg(v_a_759_, v_a_754_);
v_a_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_a_761_);
lean_dec_ref(v___x_760_);
if (lean_obj_tag(v_a_761_) == 5)
{
lean_object* v_fn_773_; 
v_fn_773_ = lean_ctor_get(v_a_761_, 0);
if (lean_obj_tag(v_fn_773_) == 5)
{
lean_object* v_arg_774_; lean_object* v_fn_775_; lean_object* v_arg_776_; lean_object* v___x_777_; 
v_arg_774_ = lean_ctor_get(v_a_761_, 1);
v_fn_775_ = lean_ctor_get(v_fn_773_, 0);
v_arg_776_ = lean_ctor_get(v_fn_773_, 1);
v___x_777_ = l_Lean_Expr_getAppFn(v_fn_775_);
if (lean_obj_tag(v___x_777_) == 4)
{
lean_object* v_declName_778_; uint8_t v_symm_780_; lean_object* v___y_781_; lean_object* v___y_782_; lean_object* v___y_783_; lean_object* v___y_784_; lean_object* v___y_805_; lean_object* v___y_806_; lean_object* v___y_807_; lean_object* v___y_808_; 
v_declName_778_ = lean_ctor_get(v___x_777_, 0);
lean_inc(v_declName_778_);
lean_dec_ref_known(v___x_777_, 2);
if (lean_obj_tag(v_declName_778_) == 1)
{
lean_object* v_pre_834_; 
v_pre_834_ = lean_ctor_get(v_declName_778_, 0);
if (lean_obj_tag(v_pre_834_) == 0)
{
lean_object* v_str_835_; lean_object* v___x_836_; uint8_t v___x_837_; 
v_str_835_ = lean_ctor_get(v_declName_778_, 1);
v___x_836_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__14));
v___x_837_ = lean_string_dec_eq(v_str_835_, v___x_836_);
if (v___x_837_ == 0)
{
lean_object* v___x_838_; uint8_t v___x_839_; 
v___x_838_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__15));
v___x_839_ = lean_string_dec_eq(v_str_835_, v___x_838_);
if (v___x_839_ == 0)
{
v___y_805_ = v_a_753_;
v___y_806_ = v_a_754_;
v___y_807_ = v_a_755_;
v___y_808_ = v_a_756_;
goto v___jp_804_;
}
else
{
lean_dec_ref_known(v_declName_778_, 2);
lean_dec_ref(v_fvar_751_);
goto v___jp_821_;
}
}
else
{
lean_dec_ref_known(v_declName_778_, 2);
lean_dec_ref(v_fvar_751_);
goto v___jp_821_;
}
}
else
{
v___y_805_ = v_a_753_;
v___y_806_ = v_a_754_;
v___y_807_ = v_a_755_;
v___y_808_ = v_a_756_;
goto v___jp_804_;
}
}
else
{
v___y_805_ = v_a_753_;
v___y_806_ = v_a_754_;
v___y_807_ = v_a_755_;
v___y_808_ = v_a_756_;
goto v___jp_804_;
}
v___jp_779_:
{
lean_object* v___x_785_; 
v___x_785_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward(v_declName_778_, v_fn_775_, v_symm_780_, v___y_781_, v___y_782_, v___y_783_, v___y_784_);
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v_a_786_; lean_object* v___x_788_; uint8_t v_isShared_789_; uint8_t v_isSharedCheck_795_; 
v_a_786_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_795_ == 0)
{
v___x_788_ = v___x_785_;
v_isShared_789_ = v_isSharedCheck_795_;
goto v_resetjp_787_;
}
else
{
lean_inc(v_a_786_);
lean_dec(v___x_785_);
v___x_788_ = lean_box(0);
v_isShared_789_ = v_isSharedCheck_795_;
goto v_resetjp_787_;
}
v_resetjp_787_:
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_793_; 
v___x_790_ = lean_st_ref_set(v_ref_749_, v_a_786_);
v___x_791_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__7);
if (v_isShared_789_ == 0)
{
lean_ctor_set_tag(v___x_788_, 1);
lean_ctor_set(v___x_788_, 0, v___x_791_);
v___x_793_ = v___x_788_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v___x_791_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
else
{
lean_object* v_a_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_803_; 
v_a_796_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_803_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_803_ == 0)
{
v___x_798_ = v___x_785_;
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
else
{
lean_inc(v_a_796_);
lean_dec(v___x_785_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
lean_object* v___x_801_; 
if (v_isShared_799_ == 0)
{
v___x_801_ = v___x_798_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v_a_796_);
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
v___jp_804_:
{
lean_object* v___x_809_; uint8_t v___x_810_; 
lean_inc_ref(v_arg_776_);
v___x_809_ = l_Lean_Expr_cleanupAnnotations(v_arg_776_);
v___x_810_ = lean_expr_eqv(v___x_809_, v_fvar_751_);
lean_dec_ref(v___x_809_);
if (v___x_810_ == 0)
{
lean_object* v___x_811_; uint8_t v___x_812_; 
lean_inc_ref(v_arg_774_);
v___x_811_ = l_Lean_Expr_cleanupAnnotations(v_arg_774_);
v___x_812_ = lean_expr_eqv(v___x_811_, v_fvar_751_);
lean_dec_ref(v___x_811_);
if (v___x_812_ == 0)
{
lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; 
lean_dec(v_declName_778_);
v___x_813_ = l_Lean_MessageData_ofExpr(v_a_761_);
v___x_814_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__9);
v___x_815_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_815_, 0, v___x_813_);
lean_ctor_set(v___x_815_, 1, v___x_814_);
v___x_816_ = l_Lean_MessageData_ofExpr(v_fvar_751_);
v___x_817_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_817_, 0, v___x_815_);
lean_ctor_set(v___x_817_, 1, v___x_816_);
v___x_818_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__11);
v___x_819_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_817_);
lean_ctor_set(v___x_819_, 1, v___x_818_);
v___x_820_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_819_, v___y_805_, v___y_806_, v___y_807_, v___y_808_);
return v___x_820_;
}
else
{
lean_inc_ref(v_fn_775_);
lean_dec_ref_known(v_a_761_, 2);
lean_dec_ref(v_fvar_751_);
if (v_hyp_x3f_750_ == 0)
{
v_symm_780_ = v___x_812_;
v___y_781_ = v___y_805_;
v___y_782_ = v___y_806_;
v___y_783_ = v___y_807_;
v___y_784_ = v___y_808_;
goto v___jp_779_;
}
else
{
v_symm_780_ = v___x_810_;
v___y_781_ = v___y_805_;
v___y_782_ = v___y_806_;
v___y_783_ = v___y_807_;
v___y_784_ = v___y_808_;
goto v___jp_779_;
}
}
}
else
{
lean_inc_ref(v_fn_775_);
lean_dec_ref_known(v_a_761_, 2);
lean_dec_ref(v_fvar_751_);
v_symm_780_ = v_hyp_x3f_750_;
v___y_781_ = v___y_805_;
v___y_782_ = v___y_806_;
v___y_783_ = v___y_807_;
v___y_784_ = v___y_808_;
goto v___jp_779_;
}
}
v___jp_821_:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v_a_826_; lean_object* v___x_828_; uint8_t v_isShared_829_; uint8_t v_isSharedCheck_833_; 
v___x_822_ = l_Lean_MessageData_ofExpr(v_a_761_);
v___x_823_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__13);
v___x_824_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_824_, 0, v___x_822_);
lean_ctor_set(v___x_824_, 1, v___x_823_);
v___x_825_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_824_, v_a_753_, v_a_754_, v_a_755_, v_a_756_);
v_a_826_ = lean_ctor_get(v___x_825_, 0);
v_isSharedCheck_833_ = !lean_is_exclusive(v___x_825_);
if (v_isSharedCheck_833_ == 0)
{
v___x_828_ = v___x_825_;
v_isShared_829_ = v_isSharedCheck_833_;
goto v_resetjp_827_;
}
else
{
lean_inc(v_a_826_);
lean_dec(v___x_825_);
v___x_828_ = lean_box(0);
v_isShared_829_ = v_isSharedCheck_833_;
goto v_resetjp_827_;
}
v_resetjp_827_:
{
lean_object* v___x_831_; 
if (v_isShared_829_ == 0)
{
v___x_831_ = v___x_828_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v_a_826_);
v___x_831_ = v_reuseFailAlloc_832_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
return v___x_831_;
}
}
}
}
else
{
lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
lean_dec_ref(v___x_777_);
lean_dec_ref(v_fvar_751_);
v___x_840_ = l_Lean_MessageData_ofExpr(v_a_761_);
v___x_841_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__17);
v___x_842_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_842_, 0, v___x_840_);
lean_ctor_set(v___x_842_, 1, v___x_841_);
v___x_843_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_842_, v_a_753_, v_a_754_, v_a_755_, v_a_756_);
return v___x_843_;
}
}
else
{
lean_dec_ref(v_fvar_751_);
v___y_763_ = v_a_753_;
v___y_764_ = v_a_754_;
v___y_765_ = v_a_755_;
v___y_766_ = v_a_756_;
goto v___jp_762_;
}
}
else
{
lean_dec_ref(v_fvar_751_);
v___y_763_ = v_a_753_;
v___y_764_ = v_a_754_;
v___y_765_ = v_a_755_;
v___y_766_ = v_a_756_;
goto v___jp_762_;
}
v___jp_762_:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_767_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__1);
v___x_768_ = l_Lean_MessageData_ofExpr(v_a_761_);
v___x_769_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_767_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
v___x_770_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__3);
v___x_771_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_771_, 0, v___x_769_);
lean_ctor_set(v___x_771_, 1, v___x_770_);
v___x_772_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_771_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
return v___x_772_;
}
}
else
{
lean_object* v_a_844_; lean_object* v___x_846_; uint8_t v_isShared_847_; uint8_t v_isSharedCheck_851_; 
lean_dec_ref(v_fvar_751_);
v_a_844_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_851_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_851_ == 0)
{
v___x_846_ = v___x_758_;
v_isShared_847_ = v_isSharedCheck_851_;
goto v_resetjp_845_;
}
else
{
lean_inc(v_a_844_);
lean_dec(v___x_758_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___boxed(lean_object* v_ref_852_, lean_object* v_hyp_x3f_853_, lean_object* v_fvar_854_, lean_object* v_goal_855_, lean_object* v_a_856_, lean_object* v_a_857_, lean_object* v_a_858_, lean_object* v_a_859_, lean_object* v_a_860_){
_start:
{
uint8_t v_hyp_x3f_boxed_861_; lean_object* v_res_862_; 
v_hyp_x3f_boxed_861_ = lean_unbox(v_hyp_x3f_853_);
v_res_862_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger(v_ref_852_, v_hyp_x3f_boxed_861_, v_fvar_854_, v_goal_855_, v_a_856_, v_a_857_, v_a_858_, v_a_859_);
lean_dec(v_a_859_);
lean_dec_ref(v_a_858_);
lean_dec(v_a_857_);
lean_dec_ref(v_a_856_);
lean_dec(v_ref_852_);
return v_res_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0(lean_object* v_fvar_863_, lean_object* v_x_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_){
_start:
{
lean_object* v___x_870_; lean_object* v___x_871_; 
v___x_870_ = lp_mathlib_Mathlib_Tactic_GCongr_mkHoleAnnotation(v_fvar_863_);
v___x_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
return v___x_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0___boxed(lean_object* v_fvar_872_, lean_object* v_x_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0(v_fvar_872_, v_x_873_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec_ref(v_x_873_);
return v_res_879_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1(lean_object* v___x_880_, lean_object* v_val_881_, uint8_t v___y_882_, lean_object* v_as_883_, size_t v_i_884_, size_t v_stop_885_){
_start:
{
uint8_t v___x_886_; 
v___x_886_ = lean_usize_dec_eq(v_i_884_, v_stop_885_);
if (v___x_886_ == 0)
{
uint8_t v___x_887_; uint8_t v___y_889_; lean_object* v___x_893_; uint8_t v___x_894_; 
v___x_887_ = 1;
v___x_893_ = lean_array_uget_borrowed(v_as_883_, v_i_884_);
v___x_894_ = l_Lean_LocalContext_contains(v___x_880_, v___x_893_);
if (v___x_894_ == 0)
{
lean_object* v___x_895_; uint8_t v___x_896_; 
v___x_895_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4));
v___x_896_ = lean_string_dec_eq(v_val_881_, v___x_895_);
v___y_889_ = v___x_896_;
goto v___jp_888_;
}
else
{
v___y_889_ = v___y_882_;
goto v___jp_888_;
}
v___jp_888_:
{
if (v___y_889_ == 0)
{
size_t v___x_890_; size_t v___x_891_; 
v___x_890_ = ((size_t)1ULL);
v___x_891_ = lean_usize_add(v_i_884_, v___x_890_);
v_i_884_ = v___x_891_;
goto _start;
}
else
{
return v___x_887_;
}
}
}
else
{
uint8_t v___x_897_; 
v___x_897_ = 0;
return v___x_897_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1___boxed(lean_object* v___x_898_, lean_object* v_val_899_, lean_object* v___y_900_, lean_object* v_as_901_, lean_object* v_i_902_, lean_object* v_stop_903_){
_start:
{
uint8_t v___y_8457__boxed_904_; size_t v_i_boxed_905_; size_t v_stop_boxed_906_; uint8_t v_res_907_; lean_object* v_r_908_; 
v___y_8457__boxed_904_ = lean_unbox(v___y_900_);
v_i_boxed_905_ = lean_unbox_usize(v_i_902_);
lean_dec(v_i_902_);
v_stop_boxed_906_ = lean_unbox_usize(v_stop_903_);
lean_dec(v_stop_903_);
v_res_907_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1(v___x_898_, v_val_899_, v___y_8457__boxed_904_, v_as_901_, v_i_boxed_905_, v_stop_boxed_906_);
lean_dec_ref(v_as_901_);
lean_dec_ref(v_val_899_);
lean_dec_ref(v___x_898_);
v_r_908_ = lean_box(v_res_907_);
return v_r_908_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; 
v___x_909_ = lean_box(0);
v___x_910_ = lean_unsigned_to_nat(16u);
v___x_911_ = lean_mk_array(v___x_910_, v___x_909_);
return v___x_911_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; 
v___x_912_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__0);
v___x_913_ = lean_unsigned_to_nat(0u);
v___x_914_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_913_);
lean_ctor_set(v___x_914_, 1, v___x_912_);
return v___x_914_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; 
v___x_915_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0));
v___x_916_ = lean_box(1);
v___x_917_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__1);
v___x_918_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_918_, 0, v___x_917_);
lean_ctor_set(v___x_918_, 1, v___x_916_);
lean_ctor_set(v___x_918_, 2, v___x_915_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg(lean_object* v_val_927_, uint8_t v___y_928_, lean_object* v_as_929_, size_t v_sz_930_, size_t v_i_931_, lean_object* v_b_932_, lean_object* v___y_933_){
_start:
{
lean_object* v_a_936_; uint8_t v___x_940_; 
v___x_940_ = lean_usize_dec_lt(v_i_931_, v_sz_930_);
if (v___x_940_ == 0)
{
lean_object* v___x_941_; 
v___x_941_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_941_, 0, v_b_932_);
return v___x_941_;
}
else
{
lean_object* v_a_942_; lean_object* v_relation_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v_fvarIds_947_; lean_object* v___x_948_; lean_object* v___x_949_; uint8_t v___x_950_; 
lean_dec_ref(v_b_932_);
v_a_942_ = lean_array_uget_borrowed(v_as_929_, v_i_931_);
v_relation_943_ = lean_ctor_get(v_a_942_, 1);
v___x_944_ = lean_unsigned_to_nat(0u);
v___x_945_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__2);
lean_inc_ref(v_relation_943_);
v___x_946_ = l_Lean_collectFVars(v___x_945_, v_relation_943_);
v_fvarIds_947_ = lean_ctor_get(v___x_946_, 2);
lean_inc_ref(v_fvarIds_947_);
lean_dec_ref(v___x_946_);
v___x_948_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__3));
v___x_949_ = lean_array_get_size(v_fvarIds_947_);
v___x_950_ = lean_nat_dec_lt(v___x_944_, v___x_949_);
if (v___x_950_ == 0)
{
lean_dec_ref(v_fvarIds_947_);
v_a_936_ = v___x_948_;
goto v___jp_935_;
}
else
{
if (v___x_950_ == 0)
{
lean_dec_ref(v_fvarIds_947_);
v_a_936_ = v___x_948_;
goto v___jp_935_;
}
else
{
lean_object* v_lctx_951_; size_t v___x_952_; size_t v___x_953_; uint8_t v___x_954_; 
v_lctx_951_ = lean_ctor_get(v___y_933_, 2);
v___x_952_ = ((size_t)0ULL);
v___x_953_ = lean_usize_of_nat(v___x_949_);
v___x_954_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__1(v_lctx_951_, v_val_927_, v___y_928_, v_fvarIds_947_, v___x_952_, v___x_953_);
lean_dec_ref(v_fvarIds_947_);
if (v___x_954_ == 0)
{
v_a_936_ = v___x_948_;
goto v___jp_935_;
}
else
{
lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_955_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__5));
v___x_956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_956_, 0, v___x_955_);
return v___x_956_;
}
}
}
}
v___jp_935_:
{
size_t v___x_937_; size_t v___x_938_; 
v___x_937_ = ((size_t)1ULL);
v___x_938_ = lean_usize_add(v_i_931_, v___x_937_);
lean_inc_ref(v_a_936_);
v_i_931_ = v___x_938_;
v_b_932_ = v_a_936_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___boxed(lean_object* v_val_957_, lean_object* v___y_958_, lean_object* v_as_959_, lean_object* v_sz_960_, lean_object* v_i_961_, lean_object* v_b_962_, lean_object* v___y_963_, lean_object* v___y_964_){
_start:
{
uint8_t v___y_8527__boxed_965_; size_t v_sz_boxed_966_; size_t v_i_boxed_967_; lean_object* v_res_968_; 
v___y_8527__boxed_965_ = lean_unbox(v___y_958_);
v_sz_boxed_966_ = lean_unbox_usize(v_sz_960_);
lean_dec(v_sz_960_);
v_i_boxed_967_ = lean_unbox_usize(v_i_961_);
lean_dec(v_i_961_);
v_res_968_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg(v_val_957_, v___y_8527__boxed_965_, v_as_959_, v_sz_boxed_966_, v_i_boxed_967_, v_b_962_, v___y_963_);
lean_dec_ref(v___y_963_);
lean_dec_ref(v_as_959_);
lean_dec_ref(v_val_957_);
return v_res_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__4(lean_object* v_msg_969_){
_start:
{
lean_object* v___x_970_; lean_object* v___x_971_; 
v___x_970_ = l_Lean_instInhabitedExpr;
v___x_971_ = lean_panic_fn_borrowed(v___x_970_, v_msg_969_);
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0(lean_object* v___x_972_, lean_object* v_body_973_, lean_object* v_g_974_, lean_object* v_b_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_){
_start:
{
lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_981_ = lean_mk_empty_array_with_capacity(v___x_972_);
v___x_982_ = lean_array_push(v___x_981_, v_b_975_);
v___x_983_ = lean_expr_instantiate_rev(v_body_973_, v___x_982_);
lean_inc(v___y_979_);
lean_inc_ref(v___y_978_);
lean_inc(v___y_977_);
lean_inc_ref(v___y_976_);
v___x_984_ = lean_apply_6(v_g_974_, v___x_983_, v___y_976_, v___y_977_, v___y_978_, v___y_979_, lean_box(0));
if (lean_obj_tag(v___x_984_) == 0)
{
lean_object* v_a_985_; uint8_t v___x_986_; uint8_t v___x_987_; uint8_t v___x_988_; lean_object* v___x_989_; 
v_a_985_ = lean_ctor_get(v___x_984_, 0);
lean_inc(v_a_985_);
lean_dec_ref_known(v___x_984_, 1);
v___x_986_ = 0;
v___x_987_ = 1;
v___x_988_ = 1;
v___x_989_ = l_Lean_Meta_mkLambdaFVars(v___x_982_, v_a_985_, v___x_986_, v___x_987_, v___x_986_, v___x_987_, v___x_988_, v___y_976_, v___y_977_, v___y_978_, v___y_979_);
lean_dec_ref(v___x_982_);
return v___x_989_;
}
else
{
lean_dec_ref(v___x_982_);
return v___x_984_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0___boxed(lean_object* v___x_990_, lean_object* v_body_991_, lean_object* v_g_992_, lean_object* v_b_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_){
_start:
{
lean_object* v_res_999_; 
v_res_999_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0(v___x_990_, v_body_991_, v_g_992_, v_b_993_, v___y_994_, v___y_995_, v___y_996_, v___y_997_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
lean_dec_ref(v_body_991_);
lean_dec(v___x_990_);
return v_res_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8(lean_object* v_body_1000_, lean_object* v_g_1001_, lean_object* v_name_1002_, uint8_t v_bi_1003_, lean_object* v_type_1004_, uint8_t v_kind_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_){
_start:
{
lean_object* v___x_1011_; lean_object* v___f_1012_; lean_object* v___x_1013_; 
v___x_1011_ = lean_unsigned_to_nat(1u);
v___f_1012_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___lam__0___boxed), 9, 3);
lean_closure_set(v___f_1012_, 0, v___x_1011_);
lean_closure_set(v___f_1012_, 1, v_body_1000_);
lean_closure_set(v___f_1012_, 2, v_g_1001_);
v___x_1013_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1002_, v_bi_1003_, v_type_1004_, v___f_1012_, v_kind_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
if (lean_obj_tag(v___x_1013_) == 0)
{
lean_object* v_a_1014_; lean_object* v___x_1016_; uint8_t v_isShared_1017_; uint8_t v_isSharedCheck_1021_; 
v_a_1014_ = lean_ctor_get(v___x_1013_, 0);
v_isSharedCheck_1021_ = !lean_is_exclusive(v___x_1013_);
if (v_isSharedCheck_1021_ == 0)
{
v___x_1016_ = v___x_1013_;
v_isShared_1017_ = v_isSharedCheck_1021_;
goto v_resetjp_1015_;
}
else
{
lean_inc(v_a_1014_);
lean_dec(v___x_1013_);
v___x_1016_ = lean_box(0);
v_isShared_1017_ = v_isSharedCheck_1021_;
goto v_resetjp_1015_;
}
v_resetjp_1015_:
{
lean_object* v___x_1019_; 
if (v_isShared_1017_ == 0)
{
v___x_1019_ = v___x_1016_;
goto v_reusejp_1018_;
}
else
{
lean_object* v_reuseFailAlloc_1020_; 
v_reuseFailAlloc_1020_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1020_, 0, v_a_1014_);
v___x_1019_ = v_reuseFailAlloc_1020_;
goto v_reusejp_1018_;
}
v_reusejp_1018_:
{
return v___x_1019_;
}
}
}
else
{
lean_object* v_a_1022_; lean_object* v___x_1024_; uint8_t v_isShared_1025_; uint8_t v_isSharedCheck_1029_; 
v_a_1022_ = lean_ctor_get(v___x_1013_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1013_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1024_ = v___x_1013_;
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
else
{
lean_inc(v_a_1022_);
lean_dec(v___x_1013_);
v___x_1024_ = lean_box(0);
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
v_resetjp_1023_:
{
lean_object* v___x_1027_; 
if (v_isShared_1025_ == 0)
{
v___x_1027_ = v___x_1024_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v_a_1022_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8___boxed(lean_object* v_body_1030_, lean_object* v_g_1031_, lean_object* v_name_1032_, lean_object* v_bi_1033_, lean_object* v_type_1034_, lean_object* v_kind_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
uint8_t v_bi_boxed_1041_; uint8_t v_kind_boxed_1042_; lean_object* v_res_1043_; 
v_bi_boxed_1041_ = lean_unbox(v_bi_1033_);
v_kind_boxed_1042_ = lean_unbox(v_kind_1035_);
v_res_1043_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8(v_body_1030_, v_g_1031_, v_name_1032_, v_bi_boxed_1041_, v_type_1034_, v_kind_boxed_1042_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0(lean_object* v_body_1044_, lean_object* v_g_1045_, lean_object* v_x_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1052_ = lean_expr_instantiate1(v_body_1044_, v_x_1046_);
lean_inc(v___y_1050_);
lean_inc_ref(v___y_1049_);
lean_inc(v___y_1048_);
lean_inc_ref(v___y_1047_);
v___x_1053_ = lean_apply_6(v_g_1045_, v___x_1052_, v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_, lean_box(0));
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v_body_1054_, lean_object* v_g_1055_, lean_object* v_x_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0(v_body_1054_, v_g_1055_, v_x_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
lean_dec(v___y_1060_);
lean_dec_ref(v___y_1059_);
lean_dec(v___y_1058_);
lean_dec_ref(v___y_1057_);
lean_dec_ref(v_x_1056_);
lean_dec_ref(v_body_1054_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(lean_object* v_name_1063_, lean_object* v_type_1064_, lean_object* v_val_1065_, lean_object* v_k_1066_, uint8_t v_nondep_1067_, uint8_t v_kind_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v___f_1074_; lean_object* v___x_1075_; 
v___f_1074_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1074_, 0, v_k_1066_);
v___x_1075_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_1063_, v_type_1064_, v_val_1065_, v___f_1074_, v_nondep_1067_, v_kind_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_);
if (lean_obj_tag(v___x_1075_) == 0)
{
lean_object* v_a_1076_; lean_object* v___x_1078_; uint8_t v_isShared_1079_; uint8_t v_isSharedCheck_1083_; 
v_a_1076_ = lean_ctor_get(v___x_1075_, 0);
v_isSharedCheck_1083_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1083_ == 0)
{
v___x_1078_ = v___x_1075_;
v_isShared_1079_ = v_isSharedCheck_1083_;
goto v_resetjp_1077_;
}
else
{
lean_inc(v_a_1076_);
lean_dec(v___x_1075_);
v___x_1078_ = lean_box(0);
v_isShared_1079_ = v_isSharedCheck_1083_;
goto v_resetjp_1077_;
}
v_resetjp_1077_:
{
lean_object* v___x_1081_; 
if (v_isShared_1079_ == 0)
{
v___x_1081_ = v___x_1078_;
goto v_reusejp_1080_;
}
else
{
lean_object* v_reuseFailAlloc_1082_; 
v_reuseFailAlloc_1082_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1082_, 0, v_a_1076_);
v___x_1081_ = v_reuseFailAlloc_1082_;
goto v_reusejp_1080_;
}
v_reusejp_1080_:
{
return v___x_1081_;
}
}
}
else
{
lean_object* v_a_1084_; lean_object* v___x_1086_; uint8_t v_isShared_1087_; uint8_t v_isSharedCheck_1091_; 
v_a_1084_ = lean_ctor_get(v___x_1075_, 0);
v_isSharedCheck_1091_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1091_ == 0)
{
v___x_1086_ = v___x_1075_;
v_isShared_1087_ = v_isSharedCheck_1091_;
goto v_resetjp_1085_;
}
else
{
lean_inc(v_a_1084_);
lean_dec(v___x_1075_);
v___x_1086_ = lean_box(0);
v_isShared_1087_ = v_isSharedCheck_1091_;
goto v_resetjp_1085_;
}
v_resetjp_1085_:
{
lean_object* v___x_1089_; 
if (v_isShared_1087_ == 0)
{
v___x_1089_ = v___x_1086_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v_a_1084_);
v___x_1089_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
return v___x_1089_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg___boxed(lean_object* v_name_1092_, lean_object* v_type_1093_, lean_object* v_val_1094_, lean_object* v_k_1095_, lean_object* v_nondep_1096_, lean_object* v_kind_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
uint8_t v_nondep_boxed_1103_; uint8_t v_kind_boxed_1104_; lean_object* v_res_1105_; 
v_nondep_boxed_1103_ = lean_unbox(v_nondep_1096_);
v_kind_boxed_1104_ = lean_unbox(v_kind_1097_);
v_res_1105_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_1092_, v_type_1093_, v_val_1094_, v_k_1095_, v_nondep_boxed_1103_, v_kind_boxed_1104_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
return v_res_1105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0(lean_object* v_k_1106_, uint8_t v_usedLetOnly_1107_, lean_object* v_x_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
lean_object* v___x_1114_; 
lean_inc(v___y_1112_);
lean_inc_ref(v___y_1111_);
lean_inc(v___y_1110_);
lean_inc_ref(v___y_1109_);
lean_inc_ref(v_x_1108_);
v___x_1114_ = lean_apply_6(v_k_1106_, v_x_1108_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_, lean_box(0));
if (lean_obj_tag(v___x_1114_) == 0)
{
lean_object* v_a_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; uint8_t v___x_1119_; uint8_t v___x_1120_; lean_object* v___x_1121_; 
v_a_1115_ = lean_ctor_get(v___x_1114_, 0);
lean_inc(v_a_1115_);
lean_dec_ref_known(v___x_1114_, 1);
v___x_1116_ = lean_unsigned_to_nat(1u);
v___x_1117_ = lean_mk_empty_array_with_capacity(v___x_1116_);
v___x_1118_ = lean_array_push(v___x_1117_, v_x_1108_);
v___x_1119_ = 0;
v___x_1120_ = 1;
v___x_1121_ = l_Lean_Meta_mkLetFVars(v___x_1118_, v_a_1115_, v_usedLetOnly_1107_, v___x_1119_, v___x_1120_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
lean_dec_ref(v___x_1118_);
return v___x_1121_;
}
else
{
lean_dec_ref(v_x_1108_);
return v___x_1114_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0___boxed(lean_object* v_k_1122_, lean_object* v_usedLetOnly_1123_, lean_object* v_x_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
uint8_t v_usedLetOnly_boxed_1130_; lean_object* v_res_1131_; 
v_usedLetOnly_boxed_1130_ = lean_unbox(v_usedLetOnly_1123_);
v_res_1131_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0(v_k_1122_, v_usedLetOnly_boxed_1130_, v_x_1124_, v___y_1125_, v___y_1126_, v___y_1127_, v___y_1128_);
lean_dec(v___y_1128_);
lean_dec_ref(v___y_1127_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5(lean_object* v_name_1132_, lean_object* v_type_1133_, lean_object* v_val_1134_, lean_object* v_k_1135_, uint8_t v_nondep_1136_, uint8_t v_kind_1137_, uint8_t v_usedLetOnly_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v___x_1144_; lean_object* v___f_1145_; lean_object* v___x_1146_; 
v___x_1144_ = lean_box(v_usedLetOnly_1138_);
v___f_1145_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1145_, 0, v_k_1135_);
lean_closure_set(v___f_1145_, 1, v___x_1144_);
v___x_1146_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_1132_, v_type_1133_, v_val_1134_, v___f_1145_, v_nondep_1136_, v_kind_1137_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
return v___x_1146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_name_1147_, lean_object* v_type_1148_, lean_object* v_val_1149_, lean_object* v_k_1150_, lean_object* v_nondep_1151_, lean_object* v_kind_1152_, lean_object* v_usedLetOnly_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
uint8_t v_nondep_boxed_1159_; uint8_t v_kind_boxed_1160_; uint8_t v_usedLetOnly_boxed_1161_; lean_object* v_res_1162_; 
v_nondep_boxed_1159_ = lean_unbox(v_nondep_1151_);
v_kind_boxed_1160_ = lean_unbox(v_kind_1152_);
v_usedLetOnly_boxed_1161_ = lean_unbox(v_usedLetOnly_1153_);
v_res_1162_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5(v_name_1147_, v_type_1148_, v_val_1149_, v_k_1150_, v_nondep_boxed_1159_, v_kind_boxed_1160_, v_usedLetOnly_boxed_1161_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_);
lean_dec(v___y_1157_);
lean_dec_ref(v___y_1156_);
lean_dec(v___y_1155_);
lean_dec_ref(v___y_1154_);
return v_res_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0(lean_object* v___x_1163_, lean_object* v_body_1164_, lean_object* v_g_1165_, lean_object* v_b_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_){
_start:
{
lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; 
v___x_1172_ = lean_mk_empty_array_with_capacity(v___x_1163_);
v___x_1173_ = lean_array_push(v___x_1172_, v_b_1166_);
v___x_1174_ = lean_expr_instantiate_rev(v_body_1164_, v___x_1173_);
lean_inc(v___y_1170_);
lean_inc_ref(v___y_1169_);
lean_inc(v___y_1168_);
lean_inc_ref(v___y_1167_);
v___x_1175_ = lean_apply_6(v_g_1165_, v___x_1174_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_, lean_box(0));
if (lean_obj_tag(v___x_1175_) == 0)
{
lean_object* v_a_1176_; uint8_t v___x_1177_; uint8_t v___x_1178_; uint8_t v___x_1179_; lean_object* v___x_1180_; 
v_a_1176_ = lean_ctor_get(v___x_1175_, 0);
lean_inc(v_a_1176_);
lean_dec_ref_known(v___x_1175_, 1);
v___x_1177_ = 0;
v___x_1178_ = 1;
v___x_1179_ = 1;
v___x_1180_ = l_Lean_Meta_mkForallFVars(v___x_1173_, v_a_1176_, v___x_1177_, v___x_1178_, v___x_1178_, v___x_1179_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
lean_dec_ref(v___x_1173_);
return v___x_1180_;
}
else
{
lean_dec_ref(v___x_1173_);
return v___x_1175_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0___boxed(lean_object* v___x_1181_, lean_object* v_body_1182_, lean_object* v_g_1183_, lean_object* v_b_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0(v___x_1181_, v_body_1182_, v_g_1183_, v_b_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
lean_dec(v___y_1186_);
lean_dec_ref(v___y_1185_);
lean_dec_ref(v_body_1182_);
lean_dec(v___x_1181_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9(lean_object* v_body_1191_, lean_object* v_g_1192_, lean_object* v_name_1193_, uint8_t v_bi_1194_, lean_object* v_type_1195_, uint8_t v_kind_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_){
_start:
{
lean_object* v___x_1202_; lean_object* v___f_1203_; lean_object* v___x_1204_; 
v___x_1202_ = lean_unsigned_to_nat(1u);
v___f_1203_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___lam__0___boxed), 9, 3);
lean_closure_set(v___f_1203_, 0, v___x_1202_);
lean_closure_set(v___f_1203_, 1, v_body_1191_);
lean_closure_set(v___f_1203_, 2, v_g_1192_);
v___x_1204_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1193_, v_bi_1194_, v_type_1195_, v___f_1203_, v_kind_1196_, v___y_1197_, v___y_1198_, v___y_1199_, v___y_1200_);
if (lean_obj_tag(v___x_1204_) == 0)
{
lean_object* v_a_1205_; lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1212_; 
v_a_1205_ = lean_ctor_get(v___x_1204_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_1204_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1207_ = v___x_1204_;
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
else
{
lean_inc(v_a_1205_);
lean_dec(v___x_1204_);
v___x_1207_ = lean_box(0);
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
v_resetjp_1206_:
{
lean_object* v___x_1210_; 
if (v_isShared_1208_ == 0)
{
v___x_1210_ = v___x_1207_;
goto v_reusejp_1209_;
}
else
{
lean_object* v_reuseFailAlloc_1211_; 
v_reuseFailAlloc_1211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1211_, 0, v_a_1205_);
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
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1220_; 
v_a_1213_ = lean_ctor_get(v___x_1204_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v___x_1204_);
if (v_isSharedCheck_1220_ == 0)
{
v___x_1215_ = v___x_1204_;
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1204_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v___x_1218_; 
if (v_isShared_1216_ == 0)
{
v___x_1218_ = v___x_1215_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v_a_1213_);
v___x_1218_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
return v___x_1218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9___boxed(lean_object* v_body_1221_, lean_object* v_g_1222_, lean_object* v_name_1223_, lean_object* v_bi_1224_, lean_object* v_type_1225_, lean_object* v_kind_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_){
_start:
{
uint8_t v_bi_boxed_1232_; uint8_t v_kind_boxed_1233_; lean_object* v_res_1234_; 
v_bi_boxed_1232_ = lean_unbox(v_bi_1224_);
v_kind_boxed_1233_ = lean_unbox(v_kind_1226_);
v_res_1234_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9(v_body_1221_, v_g_1222_, v_name_1223_, v_bi_boxed_1232_, v_type_1225_, v_kind_boxed_1233_, v___y_1227_, v___y_1228_, v___y_1229_, v___y_1230_);
lean_dec(v___y_1230_);
lean_dec_ref(v___y_1229_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
return v_res_1234_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; 
v___x_1238_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__2));
v___x_1239_ = lean_unsigned_to_nat(17u);
v___x_1240_ = lean_unsigned_to_nat(1885u);
v___x_1241_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__1));
v___x_1242_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__0));
v___x_1243_ = l_mkPanicMessageWithDecl(v___x_1242_, v___x_1241_, v___x_1240_, v___x_1239_, v___x_1238_);
return v___x_1243_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5(void){
_start:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1245_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__4));
v___x_1246_ = l_Lean_stringToMessageData(v___x_1245_);
return v___x_1246_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7(void){
_start:
{
lean_object* v___x_1248_; lean_object* v___x_1249_; 
v___x_1248_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__6));
v___x_1249_ = l_Lean_stringToMessageData(v___x_1248_);
return v___x_1249_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9(void){
_start:
{
lean_object* v___x_1251_; lean_object* v___x_1252_; 
v___x_1251_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__8));
v___x_1252_ = l_Lean_stringToMessageData(v___x_1251_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1(lean_object* v_g_1253_, lean_object* v_n_1254_, lean_object* v_e_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_){
_start:
{
lean_object* v_n_1262_; lean_object* v_a_1263_; lean_object* v_c_1293_; lean_object* v_e_1294_; lean_object* v___x_1305_; uint8_t v___x_1306_; 
v___x_1305_ = lean_unsigned_to_nat(0u);
v___x_1306_ = lean_nat_dec_eq(v_n_1254_, v___x_1305_);
if (v___x_1306_ == 0)
{
lean_object* v___x_1307_; uint8_t v___x_1308_; 
v___x_1307_ = lean_unsigned_to_nat(1u);
v___x_1308_ = lean_nat_dec_eq(v_n_1254_, v___x_1307_);
if (v___x_1308_ == 0)
{
lean_object* v___x_1309_; uint8_t v___x_1310_; 
v___x_1309_ = lean_unsigned_to_nat(2u);
v___x_1310_ = lean_nat_dec_eq(v_n_1254_, v___x_1309_);
if (v___x_1310_ == 0)
{
lean_object* v___x_1311_; uint8_t v___x_1312_; 
v___x_1311_ = lean_unsigned_to_nat(3u);
v___x_1312_ = lean_nat_dec_eq(v_n_1254_, v___x_1311_);
if (v___x_1312_ == 0)
{
if (lean_obj_tag(v_e_1255_) == 10)
{
lean_object* v_expr_1313_; 
v_expr_1313_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_expr_1313_);
v_n_1262_ = v_n_1254_;
v_a_1263_ = v_expr_1313_;
goto v___jp_1261_;
}
else
{
lean_dec_ref(v_g_1253_);
v_c_1293_ = v_n_1254_;
v_e_1294_ = v_e_1255_;
goto v___jp_1292_;
}
}
else
{
lean_dec(v_n_1254_);
if (lean_obj_tag(v_e_1255_) == 10)
{
lean_object* v_expr_1314_; 
v_expr_1314_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_expr_1314_);
v_n_1262_ = v___x_1311_;
v_a_1263_ = v_expr_1314_;
goto v___jp_1261_;
}
else
{
lean_object* v___x_1315_; lean_object* v___x_1316_; 
lean_dec_ref(v_e_1255_);
lean_dec_ref(v_g_1253_);
v___x_1315_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__9);
v___x_1316_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_1315_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
return v___x_1316_;
}
}
}
else
{
lean_dec(v_n_1254_);
switch(lean_obj_tag(v_e_1255_))
{
case 8:
{
lean_object* v_declName_1317_; lean_object* v_type_1318_; lean_object* v_value_1319_; lean_object* v_body_1320_; uint8_t v_nondep_1321_; lean_object* v___f_1322_; uint8_t v___x_1323_; lean_object* v___x_1324_; 
v_declName_1317_ = lean_ctor_get(v_e_1255_, 0);
lean_inc(v_declName_1317_);
v_type_1318_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_type_1318_);
v_value_1319_ = lean_ctor_get(v_e_1255_, 2);
lean_inc_ref(v_value_1319_);
v_body_1320_ = lean_ctor_get(v_e_1255_, 3);
lean_inc_ref(v_body_1320_);
v_nondep_1321_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_1255_, 4);
v___f_1322_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1322_, 0, v_body_1320_);
lean_closure_set(v___f_1322_, 1, v_g_1253_);
v___x_1323_ = 0;
v___x_1324_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5(v_declName_1317_, v_type_1318_, v_value_1319_, v___f_1322_, v_nondep_1321_, v___x_1323_, v___x_1308_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
return v___x_1324_;
}
case 10:
{
lean_object* v_expr_1325_; 
v_expr_1325_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_expr_1325_);
v_n_1262_ = v___x_1309_;
v_a_1263_ = v_expr_1325_;
goto v___jp_1261_;
}
default: 
{
lean_dec_ref(v_g_1253_);
v_c_1293_ = v___x_1309_;
v_e_1294_ = v_e_1255_;
goto v___jp_1292_;
}
}
}
}
else
{
lean_dec(v_n_1254_);
switch(lean_obj_tag(v_e_1255_))
{
case 5:
{
lean_object* v_fn_1326_; lean_object* v_arg_1327_; lean_object* v___x_1328_; 
v_fn_1326_ = lean_ctor_get(v_e_1255_, 0);
v_arg_1327_ = lean_ctor_get(v_e_1255_, 1);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_arg_1327_);
v___x_1328_ = lean_apply_6(v_g_1253_, v_arg_1327_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1328_) == 0)
{
lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1347_; 
v_a_1329_ = lean_ctor_get(v___x_1328_, 0);
v_isSharedCheck_1347_ = !lean_is_exclusive(v___x_1328_);
if (v_isSharedCheck_1347_ == 0)
{
v___x_1331_ = v___x_1328_;
v_isShared_1332_ = v_isSharedCheck_1347_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1328_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1347_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
uint8_t v___y_1334_; size_t v___x_1342_; uint8_t v___x_1343_; 
v___x_1342_ = lean_ptr_addr(v_fn_1326_);
v___x_1343_ = lean_usize_dec_eq(v___x_1342_, v___x_1342_);
if (v___x_1343_ == 0)
{
v___y_1334_ = v___x_1343_;
goto v___jp_1333_;
}
else
{
size_t v___x_1344_; size_t v___x_1345_; uint8_t v___x_1346_; 
v___x_1344_ = lean_ptr_addr(v_arg_1327_);
v___x_1345_ = lean_ptr_addr(v_a_1329_);
v___x_1346_ = lean_usize_dec_eq(v___x_1344_, v___x_1345_);
v___y_1334_ = v___x_1346_;
goto v___jp_1333_;
}
v___jp_1333_:
{
if (v___y_1334_ == 0)
{
lean_object* v___x_1335_; lean_object* v___x_1337_; 
lean_inc_ref(v_fn_1326_);
lean_dec_ref_known(v_e_1255_, 2);
v___x_1335_ = l_Lean_Expr_app___override(v_fn_1326_, v_a_1329_);
if (v_isShared_1332_ == 0)
{
lean_ctor_set(v___x_1331_, 0, v___x_1335_);
v___x_1337_ = v___x_1331_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1338_; 
v_reuseFailAlloc_1338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1338_, 0, v___x_1335_);
v___x_1337_ = v_reuseFailAlloc_1338_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
return v___x_1337_;
}
}
else
{
lean_object* v___x_1340_; 
lean_dec(v_a_1329_);
if (v_isShared_1332_ == 0)
{
lean_ctor_set(v___x_1331_, 0, v_e_1255_);
v___x_1340_ = v___x_1331_;
goto v_reusejp_1339_;
}
else
{
lean_object* v_reuseFailAlloc_1341_; 
v_reuseFailAlloc_1341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1341_, 0, v_e_1255_);
v___x_1340_ = v_reuseFailAlloc_1341_;
goto v_reusejp_1339_;
}
v_reusejp_1339_:
{
return v___x_1340_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_1255_, 2);
return v___x_1328_;
}
}
case 6:
{
lean_object* v_binderName_1348_; lean_object* v_binderType_1349_; lean_object* v_body_1350_; uint8_t v_binderInfo_1351_; uint8_t v___x_1352_; lean_object* v___x_1353_; 
v_binderName_1348_ = lean_ctor_get(v_e_1255_, 0);
lean_inc(v_binderName_1348_);
v_binderType_1349_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_binderType_1349_);
v_body_1350_ = lean_ctor_get(v_e_1255_, 2);
lean_inc_ref(v_body_1350_);
v_binderInfo_1351_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1352_ = 0;
v___x_1353_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__8(v_body_1350_, v_g_1253_, v_binderName_1348_, v_binderInfo_1351_, v_binderType_1349_, v___x_1352_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
return v___x_1353_;
}
case 7:
{
lean_object* v_binderName_1354_; lean_object* v_binderType_1355_; lean_object* v_body_1356_; uint8_t v_binderInfo_1357_; uint8_t v___x_1358_; lean_object* v___x_1359_; 
v_binderName_1354_ = lean_ctor_get(v_e_1255_, 0);
lean_inc(v_binderName_1354_);
v_binderType_1355_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_binderType_1355_);
v_body_1356_ = lean_ctor_get(v_e_1255_, 2);
lean_inc_ref(v_body_1356_);
v_binderInfo_1357_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1358_ = 0;
v___x_1359_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__9(v_body_1356_, v_g_1253_, v_binderName_1354_, v_binderInfo_1357_, v_binderType_1355_, v___x_1358_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
return v___x_1359_;
}
case 8:
{
lean_object* v_declName_1360_; lean_object* v_type_1361_; lean_object* v_value_1362_; lean_object* v_body_1363_; uint8_t v_nondep_1364_; lean_object* v___x_1365_; 
v_declName_1360_ = lean_ctor_get(v_e_1255_, 0);
v_type_1361_ = lean_ctor_get(v_e_1255_, 1);
v_value_1362_ = lean_ctor_get(v_e_1255_, 2);
v_body_1363_ = lean_ctor_get(v_e_1255_, 3);
v_nondep_1364_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*4 + 8);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_value_1362_);
v___x_1365_ = lean_apply_6(v_g_1253_, v_value_1362_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v_a_1366_; lean_object* v___x_1368_; uint8_t v_isShared_1369_; uint8_t v_isSharedCheck_1390_; 
v_a_1366_ = lean_ctor_get(v___x_1365_, 0);
v_isSharedCheck_1390_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1368_ = v___x_1365_;
v_isShared_1369_ = v_isSharedCheck_1390_;
goto v_resetjp_1367_;
}
else
{
lean_inc(v_a_1366_);
lean_dec(v___x_1365_);
v___x_1368_ = lean_box(0);
v_isShared_1369_ = v_isSharedCheck_1390_;
goto v_resetjp_1367_;
}
v_resetjp_1367_:
{
uint8_t v___y_1371_; size_t v___x_1385_; uint8_t v___x_1386_; 
v___x_1385_ = lean_ptr_addr(v_type_1361_);
v___x_1386_ = lean_usize_dec_eq(v___x_1385_, v___x_1385_);
if (v___x_1386_ == 0)
{
v___y_1371_ = v___x_1386_;
goto v___jp_1370_;
}
else
{
size_t v___x_1387_; size_t v___x_1388_; uint8_t v___x_1389_; 
v___x_1387_ = lean_ptr_addr(v_value_1362_);
v___x_1388_ = lean_ptr_addr(v_a_1366_);
v___x_1389_ = lean_usize_dec_eq(v___x_1387_, v___x_1388_);
v___y_1371_ = v___x_1389_;
goto v___jp_1370_;
}
v___jp_1370_:
{
if (v___y_1371_ == 0)
{
lean_object* v___x_1372_; lean_object* v___x_1374_; 
lean_inc_ref(v_body_1363_);
lean_inc_ref(v_type_1361_);
lean_inc(v_declName_1360_);
lean_dec_ref_known(v_e_1255_, 4);
v___x_1372_ = l_Lean_Expr_letE___override(v_declName_1360_, v_type_1361_, v_a_1366_, v_body_1363_, v_nondep_1364_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 0, v___x_1372_);
v___x_1374_ = v___x_1368_;
goto v_reusejp_1373_;
}
else
{
lean_object* v_reuseFailAlloc_1375_; 
v_reuseFailAlloc_1375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1375_, 0, v___x_1372_);
v___x_1374_ = v_reuseFailAlloc_1375_;
goto v_reusejp_1373_;
}
v_reusejp_1373_:
{
return v___x_1374_;
}
}
else
{
size_t v___x_1376_; uint8_t v___x_1377_; 
v___x_1376_ = lean_ptr_addr(v_body_1363_);
v___x_1377_ = lean_usize_dec_eq(v___x_1376_, v___x_1376_);
if (v___x_1377_ == 0)
{
lean_object* v___x_1378_; lean_object* v___x_1380_; 
lean_inc_ref(v_body_1363_);
lean_inc_ref(v_type_1361_);
lean_inc(v_declName_1360_);
lean_dec_ref_known(v_e_1255_, 4);
v___x_1378_ = l_Lean_Expr_letE___override(v_declName_1360_, v_type_1361_, v_a_1366_, v_body_1363_, v_nondep_1364_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 0, v___x_1378_);
v___x_1380_ = v___x_1368_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v___x_1378_);
v___x_1380_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
return v___x_1380_;
}
}
else
{
lean_object* v___x_1383_; 
lean_dec(v_a_1366_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 0, v_e_1255_);
v___x_1383_ = v___x_1368_;
goto v_reusejp_1382_;
}
else
{
lean_object* v_reuseFailAlloc_1384_; 
v_reuseFailAlloc_1384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1384_, 0, v_e_1255_);
v___x_1383_ = v_reuseFailAlloc_1384_;
goto v_reusejp_1382_;
}
v_reusejp_1382_:
{
return v___x_1383_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_1255_, 4);
return v___x_1365_;
}
}
case 10:
{
lean_object* v_expr_1391_; 
v_expr_1391_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_expr_1391_);
v_n_1262_ = v___x_1307_;
v_a_1263_ = v_expr_1391_;
goto v___jp_1261_;
}
default: 
{
lean_dec_ref(v_g_1253_);
v_c_1293_ = v___x_1307_;
v_e_1294_ = v_e_1255_;
goto v___jp_1292_;
}
}
}
}
else
{
lean_dec(v_n_1254_);
switch(lean_obj_tag(v_e_1255_))
{
case 5:
{
lean_object* v_fn_1392_; lean_object* v_arg_1393_; lean_object* v___x_1394_; 
v_fn_1392_ = lean_ctor_get(v_e_1255_, 0);
v_arg_1393_ = lean_ctor_get(v_e_1255_, 1);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_fn_1392_);
v___x_1394_ = lean_apply_6(v_g_1253_, v_fn_1392_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1394_) == 0)
{
lean_object* v_a_1395_; lean_object* v___x_1397_; uint8_t v_isShared_1398_; uint8_t v_isSharedCheck_1413_; 
v_a_1395_ = lean_ctor_get(v___x_1394_, 0);
v_isSharedCheck_1413_ = !lean_is_exclusive(v___x_1394_);
if (v_isSharedCheck_1413_ == 0)
{
v___x_1397_ = v___x_1394_;
v_isShared_1398_ = v_isSharedCheck_1413_;
goto v_resetjp_1396_;
}
else
{
lean_inc(v_a_1395_);
lean_dec(v___x_1394_);
v___x_1397_ = lean_box(0);
v_isShared_1398_ = v_isSharedCheck_1413_;
goto v_resetjp_1396_;
}
v_resetjp_1396_:
{
uint8_t v___y_1400_; size_t v___x_1408_; size_t v___x_1409_; uint8_t v___x_1410_; 
v___x_1408_ = lean_ptr_addr(v_fn_1392_);
v___x_1409_ = lean_ptr_addr(v_a_1395_);
v___x_1410_ = lean_usize_dec_eq(v___x_1408_, v___x_1409_);
if (v___x_1410_ == 0)
{
v___y_1400_ = v___x_1410_;
goto v___jp_1399_;
}
else
{
size_t v___x_1411_; uint8_t v___x_1412_; 
v___x_1411_ = lean_ptr_addr(v_arg_1393_);
v___x_1412_ = lean_usize_dec_eq(v___x_1411_, v___x_1411_);
v___y_1400_ = v___x_1412_;
goto v___jp_1399_;
}
v___jp_1399_:
{
if (v___y_1400_ == 0)
{
lean_object* v___x_1401_; lean_object* v___x_1403_; 
lean_inc_ref(v_arg_1393_);
lean_dec_ref_known(v_e_1255_, 2);
v___x_1401_ = l_Lean_Expr_app___override(v_a_1395_, v_arg_1393_);
if (v_isShared_1398_ == 0)
{
lean_ctor_set(v___x_1397_, 0, v___x_1401_);
v___x_1403_ = v___x_1397_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v___x_1401_);
v___x_1403_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
return v___x_1403_;
}
}
else
{
lean_object* v___x_1406_; 
lean_dec(v_a_1395_);
if (v_isShared_1398_ == 0)
{
lean_ctor_set(v___x_1397_, 0, v_e_1255_);
v___x_1406_ = v___x_1397_;
goto v_reusejp_1405_;
}
else
{
lean_object* v_reuseFailAlloc_1407_; 
v_reuseFailAlloc_1407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1407_, 0, v_e_1255_);
v___x_1406_ = v_reuseFailAlloc_1407_;
goto v_reusejp_1405_;
}
v_reusejp_1405_:
{
return v___x_1406_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_1255_, 2);
return v___x_1394_;
}
}
case 6:
{
lean_object* v_binderName_1414_; lean_object* v_binderType_1415_; lean_object* v_body_1416_; uint8_t v_binderInfo_1417_; lean_object* v___x_1418_; 
v_binderName_1414_ = lean_ctor_get(v_e_1255_, 0);
v_binderType_1415_ = lean_ctor_get(v_e_1255_, 1);
v_body_1416_ = lean_ctor_get(v_e_1255_, 2);
v_binderInfo_1417_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*3 + 8);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_binderType_1415_);
v___x_1418_ = lean_apply_6(v_g_1253_, v_binderType_1415_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_object* v_a_1419_; lean_object* v___x_1421_; uint8_t v_isShared_1422_; uint8_t v_isSharedCheck_1442_; 
v_a_1419_ = lean_ctor_get(v___x_1418_, 0);
v_isSharedCheck_1442_ = !lean_is_exclusive(v___x_1418_);
if (v_isSharedCheck_1442_ == 0)
{
v___x_1421_ = v___x_1418_;
v_isShared_1422_ = v_isSharedCheck_1442_;
goto v_resetjp_1420_;
}
else
{
lean_inc(v_a_1419_);
lean_dec(v___x_1418_);
v___x_1421_ = lean_box(0);
v_isShared_1422_ = v_isSharedCheck_1442_;
goto v_resetjp_1420_;
}
v_resetjp_1420_:
{
uint8_t v___y_1424_; size_t v___x_1437_; size_t v___x_1438_; uint8_t v___x_1439_; 
v___x_1437_ = lean_ptr_addr(v_binderType_1415_);
v___x_1438_ = lean_ptr_addr(v_a_1419_);
v___x_1439_ = lean_usize_dec_eq(v___x_1437_, v___x_1438_);
if (v___x_1439_ == 0)
{
v___y_1424_ = v___x_1439_;
goto v___jp_1423_;
}
else
{
size_t v___x_1440_; uint8_t v___x_1441_; 
v___x_1440_ = lean_ptr_addr(v_body_1416_);
v___x_1441_ = lean_usize_dec_eq(v___x_1440_, v___x_1440_);
v___y_1424_ = v___x_1441_;
goto v___jp_1423_;
}
v___jp_1423_:
{
if (v___y_1424_ == 0)
{
lean_object* v___x_1425_; lean_object* v___x_1427_; 
lean_inc_ref(v_body_1416_);
lean_inc(v_binderName_1414_);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1425_ = l_Lean_Expr_lam___override(v_binderName_1414_, v_a_1419_, v_body_1416_, v_binderInfo_1417_);
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 0, v___x_1425_);
v___x_1427_ = v___x_1421_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v___x_1425_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
else
{
uint8_t v___x_1429_; 
v___x_1429_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_1417_, v_binderInfo_1417_);
if (v___x_1429_ == 0)
{
lean_object* v___x_1430_; lean_object* v___x_1432_; 
lean_inc_ref(v_body_1416_);
lean_inc(v_binderName_1414_);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1430_ = l_Lean_Expr_lam___override(v_binderName_1414_, v_a_1419_, v_body_1416_, v_binderInfo_1417_);
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 0, v___x_1430_);
v___x_1432_ = v___x_1421_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1433_; 
v_reuseFailAlloc_1433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1433_, 0, v___x_1430_);
v___x_1432_ = v_reuseFailAlloc_1433_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
return v___x_1432_;
}
}
else
{
lean_object* v___x_1435_; 
lean_dec(v_a_1419_);
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 0, v_e_1255_);
v___x_1435_ = v___x_1421_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v_e_1255_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_1255_, 3);
return v___x_1418_;
}
}
case 7:
{
lean_object* v_binderName_1443_; lean_object* v_binderType_1444_; lean_object* v_body_1445_; uint8_t v_binderInfo_1446_; lean_object* v___x_1447_; 
v_binderName_1443_ = lean_ctor_get(v_e_1255_, 0);
v_binderType_1444_ = lean_ctor_get(v_e_1255_, 1);
v_body_1445_ = lean_ctor_get(v_e_1255_, 2);
v_binderInfo_1446_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*3 + 8);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_binderType_1444_);
v___x_1447_ = lean_apply_6(v_g_1253_, v_binderType_1444_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1447_) == 0)
{
lean_object* v_a_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1471_; 
v_a_1448_ = lean_ctor_get(v___x_1447_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1447_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1450_ = v___x_1447_;
v_isShared_1451_ = v_isSharedCheck_1471_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_a_1448_);
lean_dec(v___x_1447_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1471_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
uint8_t v___y_1453_; size_t v___x_1466_; size_t v___x_1467_; uint8_t v___x_1468_; 
v___x_1466_ = lean_ptr_addr(v_binderType_1444_);
v___x_1467_ = lean_ptr_addr(v_a_1448_);
v___x_1468_ = lean_usize_dec_eq(v___x_1466_, v___x_1467_);
if (v___x_1468_ == 0)
{
v___y_1453_ = v___x_1468_;
goto v___jp_1452_;
}
else
{
size_t v___x_1469_; uint8_t v___x_1470_; 
v___x_1469_ = lean_ptr_addr(v_body_1445_);
v___x_1470_ = lean_usize_dec_eq(v___x_1469_, v___x_1469_);
v___y_1453_ = v___x_1470_;
goto v___jp_1452_;
}
v___jp_1452_:
{
if (v___y_1453_ == 0)
{
lean_object* v___x_1454_; lean_object* v___x_1456_; 
lean_inc_ref(v_body_1445_);
lean_inc(v_binderName_1443_);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1454_ = l_Lean_Expr_forallE___override(v_binderName_1443_, v_a_1448_, v_body_1445_, v_binderInfo_1446_);
if (v_isShared_1451_ == 0)
{
lean_ctor_set(v___x_1450_, 0, v___x_1454_);
v___x_1456_ = v___x_1450_;
goto v_reusejp_1455_;
}
else
{
lean_object* v_reuseFailAlloc_1457_; 
v_reuseFailAlloc_1457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1457_, 0, v___x_1454_);
v___x_1456_ = v_reuseFailAlloc_1457_;
goto v_reusejp_1455_;
}
v_reusejp_1455_:
{
return v___x_1456_;
}
}
else
{
uint8_t v___x_1458_; 
v___x_1458_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_1446_, v_binderInfo_1446_);
if (v___x_1458_ == 0)
{
lean_object* v___x_1459_; lean_object* v___x_1461_; 
lean_inc_ref(v_body_1445_);
lean_inc(v_binderName_1443_);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1459_ = l_Lean_Expr_forallE___override(v_binderName_1443_, v_a_1448_, v_body_1445_, v_binderInfo_1446_);
if (v_isShared_1451_ == 0)
{
lean_ctor_set(v___x_1450_, 0, v___x_1459_);
v___x_1461_ = v___x_1450_;
goto v_reusejp_1460_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v___x_1459_);
v___x_1461_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1460_;
}
v_reusejp_1460_:
{
return v___x_1461_;
}
}
else
{
lean_object* v___x_1464_; 
lean_dec(v_a_1448_);
if (v_isShared_1451_ == 0)
{
lean_ctor_set(v___x_1450_, 0, v_e_1255_);
v___x_1464_ = v___x_1450_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v_e_1255_);
v___x_1464_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
return v___x_1464_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_1255_, 3);
return v___x_1447_;
}
}
case 8:
{
lean_object* v_declName_1472_; lean_object* v_type_1473_; lean_object* v_value_1474_; lean_object* v_body_1475_; uint8_t v_nondep_1476_; lean_object* v___x_1477_; 
v_declName_1472_ = lean_ctor_get(v_e_1255_, 0);
v_type_1473_ = lean_ctor_get(v_e_1255_, 1);
v_value_1474_ = lean_ctor_get(v_e_1255_, 2);
v_body_1475_ = lean_ctor_get(v_e_1255_, 3);
v_nondep_1476_ = lean_ctor_get_uint8(v_e_1255_, sizeof(void*)*4 + 8);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_type_1473_);
v___x_1477_ = lean_apply_6(v_g_1253_, v_type_1473_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1477_) == 0)
{
lean_object* v_a_1478_; lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1502_; 
v_a_1478_ = lean_ctor_get(v___x_1477_, 0);
v_isSharedCheck_1502_ = !lean_is_exclusive(v___x_1477_);
if (v_isSharedCheck_1502_ == 0)
{
v___x_1480_ = v___x_1477_;
v_isShared_1481_ = v_isSharedCheck_1502_;
goto v_resetjp_1479_;
}
else
{
lean_inc(v_a_1478_);
lean_dec(v___x_1477_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1502_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
uint8_t v___y_1483_; size_t v___x_1497_; size_t v___x_1498_; uint8_t v___x_1499_; 
v___x_1497_ = lean_ptr_addr(v_type_1473_);
v___x_1498_ = lean_ptr_addr(v_a_1478_);
v___x_1499_ = lean_usize_dec_eq(v___x_1497_, v___x_1498_);
if (v___x_1499_ == 0)
{
v___y_1483_ = v___x_1499_;
goto v___jp_1482_;
}
else
{
size_t v___x_1500_; uint8_t v___x_1501_; 
v___x_1500_ = lean_ptr_addr(v_value_1474_);
v___x_1501_ = lean_usize_dec_eq(v___x_1500_, v___x_1500_);
v___y_1483_ = v___x_1501_;
goto v___jp_1482_;
}
v___jp_1482_:
{
if (v___y_1483_ == 0)
{
lean_object* v___x_1484_; lean_object* v___x_1486_; 
lean_inc_ref(v_body_1475_);
lean_inc_ref(v_value_1474_);
lean_inc(v_declName_1472_);
lean_dec_ref_known(v_e_1255_, 4);
v___x_1484_ = l_Lean_Expr_letE___override(v_declName_1472_, v_a_1478_, v_value_1474_, v_body_1475_, v_nondep_1476_);
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 0, v___x_1484_);
v___x_1486_ = v___x_1480_;
goto v_reusejp_1485_;
}
else
{
lean_object* v_reuseFailAlloc_1487_; 
v_reuseFailAlloc_1487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1487_, 0, v___x_1484_);
v___x_1486_ = v_reuseFailAlloc_1487_;
goto v_reusejp_1485_;
}
v_reusejp_1485_:
{
return v___x_1486_;
}
}
else
{
size_t v___x_1488_; uint8_t v___x_1489_; 
v___x_1488_ = lean_ptr_addr(v_body_1475_);
v___x_1489_ = lean_usize_dec_eq(v___x_1488_, v___x_1488_);
if (v___x_1489_ == 0)
{
lean_object* v___x_1490_; lean_object* v___x_1492_; 
lean_inc_ref(v_body_1475_);
lean_inc_ref(v_value_1474_);
lean_inc(v_declName_1472_);
lean_dec_ref_known(v_e_1255_, 4);
v___x_1490_ = l_Lean_Expr_letE___override(v_declName_1472_, v_a_1478_, v_value_1474_, v_body_1475_, v_nondep_1476_);
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 0, v___x_1490_);
v___x_1492_ = v___x_1480_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1493_; 
v_reuseFailAlloc_1493_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1493_, 0, v___x_1490_);
v___x_1492_ = v_reuseFailAlloc_1493_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
return v___x_1492_;
}
}
else
{
lean_object* v___x_1495_; 
lean_dec(v_a_1478_);
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 0, v_e_1255_);
v___x_1495_ = v___x_1480_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_e_1255_);
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
}
}
else
{
lean_dec_ref_known(v_e_1255_, 4);
return v___x_1477_;
}
}
case 11:
{
lean_object* v_typeName_1503_; lean_object* v_idx_1504_; lean_object* v_struct_1505_; lean_object* v___x_1506_; 
v_typeName_1503_ = lean_ctor_get(v_e_1255_, 0);
v_idx_1504_ = lean_ctor_get(v_e_1255_, 1);
v_struct_1505_ = lean_ctor_get(v_e_1255_, 2);
lean_inc(v___y_1259_);
lean_inc_ref(v___y_1258_);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc_ref(v_struct_1505_);
v___x_1506_ = lean_apply_6(v_g_1253_, v_struct_1505_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, lean_box(0));
if (lean_obj_tag(v___x_1506_) == 0)
{
lean_object* v_a_1507_; lean_object* v___x_1509_; uint8_t v_isShared_1510_; uint8_t v_isSharedCheck_1521_; 
v_a_1507_ = lean_ctor_get(v___x_1506_, 0);
v_isSharedCheck_1521_ = !lean_is_exclusive(v___x_1506_);
if (v_isSharedCheck_1521_ == 0)
{
v___x_1509_ = v___x_1506_;
v_isShared_1510_ = v_isSharedCheck_1521_;
goto v_resetjp_1508_;
}
else
{
lean_inc(v_a_1507_);
lean_dec(v___x_1506_);
v___x_1509_ = lean_box(0);
v_isShared_1510_ = v_isSharedCheck_1521_;
goto v_resetjp_1508_;
}
v_resetjp_1508_:
{
size_t v___x_1511_; size_t v___x_1512_; uint8_t v___x_1513_; 
v___x_1511_ = lean_ptr_addr(v_struct_1505_);
v___x_1512_ = lean_ptr_addr(v_a_1507_);
v___x_1513_ = lean_usize_dec_eq(v___x_1511_, v___x_1512_);
if (v___x_1513_ == 0)
{
lean_object* v___x_1514_; lean_object* v___x_1516_; 
lean_inc(v_idx_1504_);
lean_inc(v_typeName_1503_);
lean_dec_ref_known(v_e_1255_, 3);
v___x_1514_ = l_Lean_Expr_proj___override(v_typeName_1503_, v_idx_1504_, v_a_1507_);
if (v_isShared_1510_ == 0)
{
lean_ctor_set(v___x_1509_, 0, v___x_1514_);
v___x_1516_ = v___x_1509_;
goto v_reusejp_1515_;
}
else
{
lean_object* v_reuseFailAlloc_1517_; 
v_reuseFailAlloc_1517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1517_, 0, v___x_1514_);
v___x_1516_ = v_reuseFailAlloc_1517_;
goto v_reusejp_1515_;
}
v_reusejp_1515_:
{
return v___x_1516_;
}
}
else
{
lean_object* v___x_1519_; 
lean_dec(v_a_1507_);
if (v_isShared_1510_ == 0)
{
lean_ctor_set(v___x_1509_, 0, v_e_1255_);
v___x_1519_ = v___x_1509_;
goto v_reusejp_1518_;
}
else
{
lean_object* v_reuseFailAlloc_1520_; 
v_reuseFailAlloc_1520_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1520_, 0, v_e_1255_);
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
else
{
lean_dec_ref_known(v_e_1255_, 3);
return v___x_1506_;
}
}
case 10:
{
lean_object* v_expr_1522_; 
v_expr_1522_ = lean_ctor_get(v_e_1255_, 1);
lean_inc_ref(v_expr_1522_);
v_n_1262_ = v___x_1305_;
v_a_1263_ = v_expr_1522_;
goto v___jp_1261_;
}
default: 
{
lean_dec_ref(v_g_1253_);
v_c_1293_ = v___x_1305_;
v_e_1294_ = v_e_1255_;
goto v___jp_1292_;
}
}
}
v___jp_1261_:
{
lean_object* v___x_1264_; 
v___x_1264_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1(v_g_1253_, v_n_1262_, v_a_1263_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
if (lean_obj_tag(v___x_1264_) == 0)
{
if (lean_obj_tag(v_e_1255_) == 10)
{
lean_object* v_a_1265_; lean_object* v___x_1267_; uint8_t v_isShared_1268_; uint8_t v_isSharedCheck_1281_; 
v_a_1265_ = lean_ctor_get(v___x_1264_, 0);
v_isSharedCheck_1281_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1281_ == 0)
{
v___x_1267_ = v___x_1264_;
v_isShared_1268_ = v_isSharedCheck_1281_;
goto v_resetjp_1266_;
}
else
{
lean_inc(v_a_1265_);
lean_dec(v___x_1264_);
v___x_1267_ = lean_box(0);
v_isShared_1268_ = v_isSharedCheck_1281_;
goto v_resetjp_1266_;
}
v_resetjp_1266_:
{
lean_object* v_data_1269_; lean_object* v_expr_1270_; size_t v___x_1271_; size_t v___x_1272_; uint8_t v___x_1273_; 
v_data_1269_ = lean_ctor_get(v_e_1255_, 0);
v_expr_1270_ = lean_ctor_get(v_e_1255_, 1);
v___x_1271_ = lean_ptr_addr(v_expr_1270_);
v___x_1272_ = lean_ptr_addr(v_a_1265_);
v___x_1273_ = lean_usize_dec_eq(v___x_1271_, v___x_1272_);
if (v___x_1273_ == 0)
{
lean_object* v___x_1274_; lean_object* v___x_1276_; 
lean_inc(v_data_1269_);
lean_dec_ref_known(v_e_1255_, 2);
v___x_1274_ = l_Lean_Expr_mdata___override(v_data_1269_, v_a_1265_);
if (v_isShared_1268_ == 0)
{
lean_ctor_set(v___x_1267_, 0, v___x_1274_);
v___x_1276_ = v___x_1267_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1277_; 
v_reuseFailAlloc_1277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1277_, 0, v___x_1274_);
v___x_1276_ = v_reuseFailAlloc_1277_;
goto v_reusejp_1275_;
}
v_reusejp_1275_:
{
return v___x_1276_;
}
}
else
{
lean_object* v___x_1279_; 
lean_dec(v_a_1265_);
if (v_isShared_1268_ == 0)
{
lean_ctor_set(v___x_1267_, 0, v_e_1255_);
v___x_1279_ = v___x_1267_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v_e_1255_);
v___x_1279_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
return v___x_1279_;
}
}
}
}
else
{
lean_object* v___x_1283_; uint8_t v_isShared_1284_; uint8_t v_isSharedCheck_1290_; 
lean_dec_ref(v_e_1255_);
v_isSharedCheck_1290_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1290_ == 0)
{
lean_object* v_unused_1291_; 
v_unused_1291_ = lean_ctor_get(v___x_1264_, 0);
lean_dec(v_unused_1291_);
v___x_1283_ = v___x_1264_;
v_isShared_1284_ = v_isSharedCheck_1290_;
goto v_resetjp_1282_;
}
else
{
lean_dec(v___x_1264_);
v___x_1283_ = lean_box(0);
v_isShared_1284_ = v_isSharedCheck_1290_;
goto v_resetjp_1282_;
}
v_resetjp_1282_:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1288_; 
v___x_1285_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__3);
v___x_1286_ = lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__4(v___x_1285_);
if (v_isShared_1284_ == 0)
{
lean_ctor_set(v___x_1283_, 0, v___x_1286_);
v___x_1288_ = v___x_1283_;
goto v_reusejp_1287_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v___x_1286_);
v___x_1288_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1287_;
}
v_reusejp_1287_:
{
return v___x_1288_;
}
}
}
}
else
{
lean_dec_ref(v_e_1255_);
return v___x_1264_;
}
}
v___jp_1292_:
{
lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; 
v___x_1295_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__5);
v___x_1296_ = l_Nat_reprFast(v_c_1293_);
v___x_1297_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1297_, 0, v___x_1296_);
v___x_1298_ = l_Lean_MessageData_ofFormat(v___x_1297_);
v___x_1299_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1299_, 0, v___x_1295_);
lean_ctor_set(v___x_1299_, 1, v___x_1298_);
v___x_1300_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___closed__7);
v___x_1301_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1301_, 0, v___x_1299_);
lean_ctor_set(v___x_1301_, 1, v___x_1300_);
v___x_1302_ = l_Lean_MessageData_ofExpr(v_e_1294_);
v___x_1303_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1303_, 0, v___x_1301_);
lean_ctor_set(v___x_1303_, 1, v___x_1302_);
v___x_1304_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3___redArg(v___x_1303_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
return v___x_1304_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1___boxed(lean_object* v_g_1523_, lean_object* v_n_1524_, lean_object* v_e_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_){
_start:
{
lean_object* v_res_1531_; 
v_res_1531_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1(v_g_1523_, v_n_1524_, v_e_1525_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
lean_dec(v___y_1529_);
lean_dec_ref(v___y_1528_);
lean_dec(v___y_1527_);
lean_dec_ref(v___y_1526_);
return v_res_1531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0___boxed(lean_object* v_g_1532_, lean_object* v_x_1533_, lean_object* v_x_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_){
_start:
{
lean_object* v_res_1540_; 
v_res_1540_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0(v_g_1532_, v_x_1533_, v_x_1534_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_);
lean_dec(v___y_1538_);
lean_dec_ref(v___y_1537_);
lean_dec(v___y_1536_);
lean_dec_ref(v___y_1535_);
return v_res_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0(lean_object* v_g_1541_, lean_object* v_x_1542_, lean_object* v_x_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
if (lean_obj_tag(v_x_1542_) == 0)
{
lean_object* v___x_1549_; 
lean_inc(v___y_1547_);
lean_inc_ref(v___y_1546_);
lean_inc(v___y_1545_);
lean_inc_ref(v___y_1544_);
v___x_1549_ = lean_apply_6(v_g_1541_, v_x_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, lean_box(0));
return v___x_1549_;
}
else
{
lean_object* v_head_1550_; lean_object* v_tail_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; 
v_head_1550_ = lean_ctor_get(v_x_1542_, 0);
lean_inc(v_head_1550_);
v_tail_1551_ = lean_ctor_get(v_x_1542_, 1);
lean_inc(v_tail_1551_);
lean_dec_ref_known(v_x_1542_, 2);
v___x_1552_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0___boxed), 8, 2);
lean_closure_set(v___x_1552_, 0, v_g_1541_);
lean_closure_set(v___x_1552_, 1, v_tail_1551_);
v___x_1553_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1(v___x_1552_, v_head_1550_, v_x_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
return v___x_1553_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0(lean_object* v_replace_1554_, lean_object* v_p_1555_, lean_object* v_root_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_){
_start:
{
lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; 
v___x_1562_ = l_Lean_SubExpr_Pos_toArray(v_p_1555_);
v___x_1563_ = lean_array_to_list(v___x_1562_);
v___x_1564_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0(v_replace_1554_, v___x_1563_, v_root_1556_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_);
return v___x_1564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0___boxed(lean_object* v_replace_1565_, lean_object* v_p_1566_, lean_object* v_root_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
lean_object* v_res_1573_; 
v_res_1573_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0(v_replace_1565_, v_p_1566_, v_root_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
lean_dec(v___y_1571_);
lean_dec_ref(v___y_1570_);
lean_dec(v___y_1569_);
lean_dec_ref(v___y_1568_);
lean_dec(v_p_1566_);
return v_res_1573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1(lean_object* v_pos_1578_, lean_object* v_rootExpr_1579_, lean_object* v___x_1580_, uint8_t v_hyp_x3f_1581_, lean_object* v_fvar_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
lean_object* v___f_1588_; lean_object* v___x_1589_; 
lean_inc_ref(v_fvar_1582_);
v___f_1588_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1588_, 0, v_fvar_1582_);
lean_inc_ref(v_rootExpr_1579_);
v___x_1589_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0(v___f_1588_, v_pos_1578_, v_rootExpr_1579_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_);
if (lean_obj_tag(v___x_1589_) == 0)
{
lean_object* v_a_1590_; uint8_t v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; uint8_t v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; 
v_a_1590_ = lean_ctor_get(v___x_1589_, 0);
lean_inc(v_a_1590_);
lean_dec_ref_known(v___x_1589_, 1);
v___x_1591_ = 0;
v___x_1592_ = l_Lean_Expr_forallE___override(v___x_1580_, v_rootExpr_1579_, v_a_1590_, v___x_1591_);
v___x_1593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1593_, 0, v___x_1592_);
v___x_1594_ = 0;
v___x_1595_ = lean_box(0);
v___x_1596_ = l_Lean_Meta_mkFreshExprMVar(v___x_1593_, v___x_1594_, v___x_1595_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_);
if (lean_obj_tag(v___x_1596_) == 0)
{
lean_object* v_a_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; 
v_a_1597_ = lean_ctor_get(v___x_1596_, 0);
lean_inc(v_a_1597_);
lean_dec_ref_known(v___x_1596_, 1);
v___x_1598_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward___lam__2___closed__0));
v___x_1599_ = lean_st_mk_ref(v___x_1598_);
v___x_1600_ = l_Lean_Expr_mvarId_x21(v_a_1597_);
lean_dec(v_a_1597_);
v___x_1601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__0));
v___x_1602_ = lean_unsigned_to_nat(1000000u);
v___x_1603_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_gcongr___boxed), 10, 3);
lean_closure_set(v___x_1603_, 0, v___x_1600_);
lean_closure_set(v___x_1603_, 1, v___x_1601_);
lean_closure_set(v___x_1603_, 2, v___x_1602_);
v___x_1604_ = lean_box(0);
v___x_1605_ = lean_box(v_hyp_x3f_1581_);
lean_inc(v___x_1599_);
v___x_1606_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___boxed), 9, 3);
lean_closure_set(v___x_1606_, 0, v___x_1599_);
lean_closure_set(v___x_1606_, 1, v___x_1605_);
lean_closure_set(v___x_1606_, 2, v_fvar_1582_);
v___x_1607_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___closed__1));
v___x_1608_ = lp_mathlib_Mathlib_Tactic_GCongr_GCongrM_run___redArg(v___x_1603_, v___x_1604_, v___x_1606_, v___x_1607_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v___x_1610_; uint8_t v_isShared_1611_; uint8_t v_isSharedCheck_1615_; 
lean_dec(v___x_1599_);
v_isSharedCheck_1615_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1615_ == 0)
{
lean_object* v_unused_1616_; 
v_unused_1616_ = lean_ctor_get(v___x_1608_, 0);
lean_dec(v_unused_1616_);
v___x_1610_ = v___x_1608_;
v_isShared_1611_ = v_isSharedCheck_1615_;
goto v_resetjp_1609_;
}
else
{
lean_dec(v___x_1608_);
v___x_1610_ = lean_box(0);
v_isShared_1611_ = v_isSharedCheck_1615_;
goto v_resetjp_1609_;
}
v_resetjp_1609_:
{
lean_object* v___x_1613_; 
if (v_isShared_1611_ == 0)
{
lean_ctor_set(v___x_1610_, 0, v___x_1598_);
v___x_1613_ = v___x_1610_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1614_; 
v_reuseFailAlloc_1614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1614_, 0, v___x_1598_);
v___x_1613_ = v_reuseFailAlloc_1614_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
return v___x_1613_;
}
}
}
else
{
lean_object* v_a_1617_; lean_object* v___x_1619_; uint8_t v_isShared_1620_; uint8_t v_isSharedCheck_1661_; 
v_a_1617_ = lean_ctor_get(v___x_1608_, 0);
v_isSharedCheck_1661_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1661_ == 0)
{
v___x_1619_ = v___x_1608_;
v_isShared_1620_ = v_isSharedCheck_1661_;
goto v_resetjp_1618_;
}
else
{
lean_inc(v_a_1617_);
lean_dec(v___x_1608_);
v___x_1619_ = lean_box(0);
v_isShared_1620_ = v_isSharedCheck_1661_;
goto v_resetjp_1618_;
}
v_resetjp_1618_:
{
uint8_t v___y_1622_; uint8_t v___x_1659_; 
v___x_1659_ = l_Lean_Exception_isInterrupt(v_a_1617_);
if (v___x_1659_ == 0)
{
uint8_t v___x_1660_; 
lean_inc(v_a_1617_);
v___x_1660_ = l_Lean_Exception_isRuntime(v_a_1617_);
v___y_1622_ = v___x_1660_;
goto v___jp_1621_;
}
else
{
v___y_1622_ = v___x_1659_;
goto v___jp_1621_;
}
v___jp_1621_:
{
if (v___y_1622_ == 0)
{
lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; uint8_t v___x_1626_; 
v___x_1623_ = l_Lean_Exception_toMessageData(v_a_1617_);
v___x_1624_ = l_Lean_MessageData_toString(v___x_1623_);
v___x_1625_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_dummyDischarger___closed__4));
v___x_1626_ = lean_string_dec_eq(v___x_1624_, v___x_1625_);
if (v___x_1626_ == 0)
{
lean_object* v___x_1628_; 
lean_dec_ref(v___x_1624_);
lean_dec(v___x_1599_);
if (v_isShared_1620_ == 0)
{
lean_ctor_set_tag(v___x_1619_, 0);
lean_ctor_set(v___x_1619_, 0, v___x_1598_);
v___x_1628_ = v___x_1619_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v___x_1598_);
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
lean_object* v___x_1630_; lean_object* v___x_1631_; size_t v_sz_1632_; size_t v___x_1633_; lean_object* v___x_1634_; 
lean_del_object(v___x_1619_);
v___x_1630_ = lean_st_ref_get(v___x_1599_);
lean_dec(v___x_1599_);
v___x_1631_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg___closed__3));
v_sz_1632_ = lean_array_size(v___x_1630_);
v___x_1633_ = ((size_t)0ULL);
v___x_1634_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg(v___x_1624_, v___y_1622_, v___x_1630_, v_sz_1632_, v___x_1633_, v___x_1631_, v___y_1583_);
lean_dec_ref(v___x_1624_);
if (lean_obj_tag(v___x_1634_) == 0)
{
lean_object* v_a_1635_; lean_object* v___x_1637_; uint8_t v_isShared_1638_; uint8_t v_isSharedCheck_1647_; 
v_a_1635_ = lean_ctor_get(v___x_1634_, 0);
v_isSharedCheck_1647_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1647_ == 0)
{
v___x_1637_ = v___x_1634_;
v_isShared_1638_ = v_isSharedCheck_1647_;
goto v_resetjp_1636_;
}
else
{
lean_inc(v_a_1635_);
lean_dec(v___x_1634_);
v___x_1637_ = lean_box(0);
v_isShared_1638_ = v_isSharedCheck_1647_;
goto v_resetjp_1636_;
}
v_resetjp_1636_:
{
lean_object* v_fst_1639_; 
v_fst_1639_ = lean_ctor_get(v_a_1635_, 0);
lean_inc(v_fst_1639_);
lean_dec(v_a_1635_);
if (lean_obj_tag(v_fst_1639_) == 0)
{
lean_object* v___x_1641_; 
if (v_isShared_1638_ == 0)
{
lean_ctor_set(v___x_1637_, 0, v___x_1630_);
v___x_1641_ = v___x_1637_;
goto v_reusejp_1640_;
}
else
{
lean_object* v_reuseFailAlloc_1642_; 
v_reuseFailAlloc_1642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1642_, 0, v___x_1630_);
v___x_1641_ = v_reuseFailAlloc_1642_;
goto v_reusejp_1640_;
}
v_reusejp_1640_:
{
return v___x_1641_;
}
}
else
{
lean_object* v_val_1643_; lean_object* v___x_1645_; 
lean_dec(v___x_1630_);
v_val_1643_ = lean_ctor_get(v_fst_1639_, 0);
lean_inc(v_val_1643_);
lean_dec_ref_known(v_fst_1639_, 1);
if (v_isShared_1638_ == 0)
{
lean_ctor_set(v___x_1637_, 0, v_val_1643_);
v___x_1645_ = v___x_1637_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1646_; 
v_reuseFailAlloc_1646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1646_, 0, v_val_1643_);
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
lean_dec(v___x_1630_);
v_a_1648_ = lean_ctor_get(v___x_1634_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1634_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1634_);
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
lean_object* v___x_1657_; 
lean_dec(v___x_1599_);
if (v_isShared_1620_ == 0)
{
v___x_1657_ = v___x_1619_;
goto v_reusejp_1656_;
}
else
{
lean_object* v_reuseFailAlloc_1658_; 
v_reuseFailAlloc_1658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1658_, 0, v_a_1617_);
v___x_1657_ = v_reuseFailAlloc_1658_;
goto v_reusejp_1656_;
}
v_reusejp_1656_:
{
return v___x_1657_;
}
}
}
}
}
}
else
{
lean_object* v_a_1662_; lean_object* v___x_1664_; uint8_t v_isShared_1665_; uint8_t v_isSharedCheck_1669_; 
lean_dec_ref(v_fvar_1582_);
v_a_1662_ = lean_ctor_get(v___x_1596_, 0);
v_isSharedCheck_1669_ = !lean_is_exclusive(v___x_1596_);
if (v_isSharedCheck_1669_ == 0)
{
v___x_1664_ = v___x_1596_;
v_isShared_1665_ = v_isSharedCheck_1669_;
goto v_resetjp_1663_;
}
else
{
lean_inc(v_a_1662_);
lean_dec(v___x_1596_);
v___x_1664_ = lean_box(0);
v_isShared_1665_ = v_isSharedCheck_1669_;
goto v_resetjp_1663_;
}
v_resetjp_1663_:
{
lean_object* v___x_1667_; 
if (v_isShared_1665_ == 0)
{
v___x_1667_ = v___x_1664_;
goto v_reusejp_1666_;
}
else
{
lean_object* v_reuseFailAlloc_1668_; 
v_reuseFailAlloc_1668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1668_, 0, v_a_1662_);
v___x_1667_ = v_reuseFailAlloc_1668_;
goto v_reusejp_1666_;
}
v_reusejp_1666_:
{
return v___x_1667_;
}
}
}
}
else
{
lean_object* v_a_1670_; lean_object* v___x_1672_; uint8_t v_isShared_1673_; uint8_t v_isSharedCheck_1677_; 
lean_dec_ref(v_fvar_1582_);
lean_dec(v___x_1580_);
lean_dec_ref(v_rootExpr_1579_);
v_a_1670_ = lean_ctor_get(v___x_1589_, 0);
v_isSharedCheck_1677_ = !lean_is_exclusive(v___x_1589_);
if (v_isSharedCheck_1677_ == 0)
{
v___x_1672_ = v___x_1589_;
v_isShared_1673_ = v_isSharedCheck_1677_;
goto v_resetjp_1671_;
}
else
{
lean_inc(v_a_1670_);
lean_dec(v___x_1589_);
v___x_1672_ = lean_box(0);
v_isShared_1673_ = v_isSharedCheck_1677_;
goto v_resetjp_1671_;
}
v_resetjp_1671_:
{
lean_object* v___x_1675_; 
if (v_isShared_1673_ == 0)
{
v___x_1675_ = v___x_1672_;
goto v_reusejp_1674_;
}
else
{
lean_object* v_reuseFailAlloc_1676_; 
v_reuseFailAlloc_1676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1676_, 0, v_a_1670_);
v___x_1675_ = v_reuseFailAlloc_1676_;
goto v_reusejp_1674_;
}
v_reusejp_1674_:
{
return v___x_1675_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___boxed(lean_object* v_pos_1678_, lean_object* v_rootExpr_1679_, lean_object* v___x_1680_, lean_object* v_hyp_x3f_1681_, lean_object* v_fvar_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_){
_start:
{
uint8_t v_hyp_x3f_boxed_1688_; lean_object* v_res_1689_; 
v_hyp_x3f_boxed_1688_ = lean_unbox(v_hyp_x3f_1681_);
v_res_1689_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1(v_pos_1678_, v_rootExpr_1679_, v___x_1680_, v_hyp_x3f_boxed_1688_, v_fvar_1682_, v___y_1683_, v___y_1684_, v___y_1685_, v___y_1686_);
lean_dec(v___y_1686_);
lean_dec_ref(v___y_1685_);
lean_dec(v___y_1684_);
lean_dec_ref(v___y_1683_);
lean_dec(v_pos_1678_);
return v_res_1689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f(lean_object* v_rootExpr_1693_, lean_object* v_subExpr_1694_, lean_object* v_pos_1695_, uint8_t v_hyp_x3f_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_, lean_object* v_a_1700_){
_start:
{
lean_object* v___x_1702_; 
lean_inc(v_a_1700_);
lean_inc_ref(v_a_1699_);
lean_inc(v_a_1698_);
lean_inc_ref(v_a_1697_);
v___x_1702_ = lean_infer_type(v_subExpr_1694_, v_a_1697_, v_a_1698_, v_a_1699_, v_a_1700_);
if (lean_obj_tag(v___x_1702_) == 0)
{
lean_object* v_a_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___f_1706_; lean_object* v___x_1707_; 
v_a_1703_ = lean_ctor_get(v___x_1702_, 0);
lean_inc(v_a_1703_);
lean_dec_ref_known(v___x_1702_, 1);
v___x_1704_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___closed__1));
v___x_1705_ = lean_box(v_hyp_x3f_1696_);
v___f_1706_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___lam__1___boxed), 10, 4);
lean_closure_set(v___f_1706_, 0, v_pos_1695_);
lean_closure_set(v___f_1706_, 1, v_rootExpr_1693_);
lean_closure_set(v___f_1706_, 2, v___x_1704_);
lean_closure_set(v___f_1706_, 3, v___x_1705_);
v___x_1707_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__2___redArg(v___x_1704_, v_a_1703_, v___f_1706_, v_a_1697_, v_a_1698_, v_a_1699_, v_a_1700_);
return v___x_1707_;
}
else
{
lean_object* v_a_1708_; lean_object* v___x_1710_; uint8_t v_isShared_1711_; uint8_t v_isSharedCheck_1715_; 
lean_dec(v_pos_1695_);
lean_dec_ref(v_rootExpr_1693_);
v_a_1708_ = lean_ctor_get(v___x_1702_, 0);
v_isSharedCheck_1715_ = !lean_is_exclusive(v___x_1702_);
if (v_isSharedCheck_1715_ == 0)
{
v___x_1710_ = v___x_1702_;
v_isShared_1711_ = v_isSharedCheck_1715_;
goto v_resetjp_1709_;
}
else
{
lean_inc(v_a_1708_);
lean_dec(v___x_1702_);
v___x_1710_ = lean_box(0);
v_isShared_1711_ = v_isSharedCheck_1715_;
goto v_resetjp_1709_;
}
v_resetjp_1709_:
{
lean_object* v___x_1713_; 
if (v_isShared_1711_ == 0)
{
v___x_1713_ = v___x_1710_;
goto v_reusejp_1712_;
}
else
{
lean_object* v_reuseFailAlloc_1714_; 
v_reuseFailAlloc_1714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1714_, 0, v_a_1708_);
v___x_1713_ = v_reuseFailAlloc_1714_;
goto v_reusejp_1712_;
}
v_reusejp_1712_:
{
return v___x_1713_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f___boxed(lean_object* v_rootExpr_1716_, lean_object* v_subExpr_1717_, lean_object* v_pos_1718_, lean_object* v_hyp_x3f_1719_, lean_object* v_a_1720_, lean_object* v_a_1721_, lean_object* v_a_1722_, lean_object* v_a_1723_, lean_object* v_a_1724_){
_start:
{
uint8_t v_hyp_x3f_boxed_1725_; lean_object* v_res_1726_; 
v_hyp_x3f_boxed_1725_ = lean_unbox(v_hyp_x3f_1719_);
v_res_1726_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f(v_rootExpr_1716_, v_subExpr_1717_, v_pos_1718_, v_hyp_x3f_boxed_1725_, v_a_1720_, v_a_1721_, v_a_1722_, v_a_1723_);
lean_dec(v_a_1723_);
lean_dec_ref(v_a_1722_);
lean_dec(v_a_1721_);
lean_dec_ref(v_a_1720_);
return v_res_1726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2(lean_object* v_val_1727_, uint8_t v___y_1728_, lean_object* v_as_1729_, size_t v_sz_1730_, size_t v_i_1731_, lean_object* v_b_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_){
_start:
{
lean_object* v___x_1738_; 
v___x_1738_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___redArg(v_val_1727_, v___y_1728_, v_as_1729_, v_sz_1730_, v_i_1731_, v_b_1732_, v___y_1733_);
return v___x_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2___boxed(lean_object* v_val_1739_, lean_object* v___y_1740_, lean_object* v_as_1741_, lean_object* v_sz_1742_, lean_object* v_i_1743_, lean_object* v_b_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_){
_start:
{
uint8_t v___y_9821__boxed_1750_; size_t v_sz_boxed_1751_; size_t v_i_boxed_1752_; lean_object* v_res_1753_; 
v___y_9821__boxed_1750_ = lean_unbox(v___y_1740_);
v_sz_boxed_1751_ = lean_unbox_usize(v_sz_1742_);
lean_dec(v_sz_1742_);
v_i_boxed_1752_ = lean_unbox_usize(v_i_1743_);
lean_dec(v_i_1743_);
v_res_1753_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__2(v_val_1739_, v___y_9821__boxed_1750_, v_as_1741_, v_sz_boxed_1751_, v_i_boxed_1752_, v_b_1744_, v___y_1745_, v___y_1746_, v___y_1747_, v___y_1748_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
lean_dec(v___y_1746_);
lean_dec_ref(v___y_1745_);
lean_dec_ref(v_as_1741_);
lean_dec_ref(v_val_1739_);
return v_res_1753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6(lean_object* v_00_u03b1_1754_, lean_object* v_name_1755_, lean_object* v_type_1756_, lean_object* v_val_1757_, lean_object* v_k_1758_, uint8_t v_nondep_1759_, uint8_t v_kind_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_){
_start:
{
lean_object* v___x_1766_; 
v___x_1766_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_1755_, v_type_1756_, v_val_1757_, v_k_1758_, v_nondep_1759_, v_kind_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
return v___x_1766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6___boxed(lean_object* v_00_u03b1_1767_, lean_object* v_name_1768_, lean_object* v_type_1769_, lean_object* v_val_1770_, lean_object* v_k_1771_, lean_object* v_nondep_1772_, lean_object* v_kind_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_){
_start:
{
uint8_t v_nondep_boxed_1779_; uint8_t v_kind_boxed_1780_; lean_object* v_res_1781_; 
v_nondep_boxed_1779_ = lean_unbox(v_nondep_1772_);
v_kind_boxed_1780_ = lean_unbox(v_kind_1773_);
v_res_1781_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Mathlib_Tactic_ClickSuggestions_getGrwPos_x3f_spec__0_spec__0_spec__1_spec__5_spec__6(v_00_u03b1_1767_, v_name_1768_, v_type_1769_, v_val_1770_, v_k_1771_, v_nondep_boxed_1779_, v_kind_boxed_1780_, v___y_1774_, v___y_1775_, v___y_1776_, v___y_1777_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
lean_dec(v___y_1775_);
lean_dec_ref(v___y_1774_);
return v_res_1781_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1(void){
_start:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; 
v___x_1783_ = l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
v___x_1784_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__0));
v___x_1785_ = lean_unsigned_to_nat(0u);
v___x_1786_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1785_);
lean_ctor_set(v___x_1786_, 1, v___x_1785_);
lean_ctor_set(v___x_1786_, 2, v___x_1785_);
lean_ctor_set(v___x_1786_, 3, v___x_1784_);
lean_ctor_set(v___x_1786_, 4, v___x_1783_);
return v___x_1786_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default(void){
_start:
{
lean_object* v___x_1787_; 
v___x_1787_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default___closed__1);
return v___x_1787_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey(void){
_start:
{
lean_object* v___x_1788_; 
v___x_1788_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default;
return v___x_1788_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0(lean_object* v_a_1789_, lean_object* v_b_1790_){
_start:
{
lean_object* v_numGoals_1791_; lean_object* v_nameLength_1792_; lean_object* v_replacementSize_1793_; lean_object* v_name_1794_; lean_object* v_numGoals_1795_; lean_object* v_nameLength_1796_; lean_object* v_replacementSize_1797_; lean_object* v_name_1798_; uint8_t v___x_1799_; 
v_numGoals_1791_ = lean_ctor_get(v_a_1789_, 0);
v_nameLength_1792_ = lean_ctor_get(v_a_1789_, 1);
v_replacementSize_1793_ = lean_ctor_get(v_a_1789_, 2);
v_name_1794_ = lean_ctor_get(v_a_1789_, 3);
v_numGoals_1795_ = lean_ctor_get(v_b_1790_, 0);
v_nameLength_1796_ = lean_ctor_get(v_b_1790_, 1);
v_replacementSize_1797_ = lean_ctor_get(v_b_1790_, 2);
v_name_1798_ = lean_ctor_get(v_b_1790_, 3);
v___x_1799_ = lean_nat_dec_lt(v_numGoals_1791_, v_numGoals_1795_);
if (v___x_1799_ == 0)
{
uint8_t v___x_1800_; 
v___x_1800_ = lean_nat_dec_eq(v_numGoals_1791_, v_numGoals_1795_);
if (v___x_1800_ == 0)
{
uint8_t v___x_1801_; 
v___x_1801_ = 2;
return v___x_1801_;
}
else
{
uint8_t v___x_1802_; 
v___x_1802_ = lean_nat_dec_lt(v_nameLength_1792_, v_nameLength_1796_);
if (v___x_1802_ == 0)
{
uint8_t v___x_1803_; 
v___x_1803_ = lean_nat_dec_eq(v_nameLength_1792_, v_nameLength_1796_);
if (v___x_1803_ == 0)
{
uint8_t v___x_1804_; 
v___x_1804_ = 2;
return v___x_1804_;
}
else
{
uint8_t v___x_1805_; 
v___x_1805_ = lean_nat_dec_lt(v_replacementSize_1793_, v_replacementSize_1797_);
if (v___x_1805_ == 0)
{
uint8_t v___x_1806_; 
v___x_1806_ = lean_nat_dec_eq(v_replacementSize_1793_, v_replacementSize_1797_);
if (v___x_1806_ == 0)
{
uint8_t v___x_1807_; 
v___x_1807_ = 2;
return v___x_1807_;
}
else
{
uint8_t v___x_1808_; 
v___x_1808_ = lean_string_compare(v_name_1794_, v_name_1798_);
return v___x_1808_;
}
}
else
{
uint8_t v___x_1809_; 
v___x_1809_ = 0;
return v___x_1809_;
}
}
}
else
{
uint8_t v___x_1810_; 
v___x_1810_ = 0;
return v___x_1810_;
}
}
}
else
{
uint8_t v___x_1811_; 
v___x_1811_ = 0;
return v___x_1811_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0___boxed(lean_object* v_a_1812_, lean_object* v_b_1813_){
_start:
{
uint8_t v_res_1814_; lean_object* v_r_1815_; 
v_res_1814_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdGrwKey___lam__0(v_a_1812_, v_b_1813_);
lean_dec_ref(v_b_1813_);
lean_dec_ref(v_a_1812_);
v_r_1815_ = lean_box(v_res_1814_);
return v_r_1815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwKey_isDuplicate(lean_object* v_a_1818_, lean_object* v_b_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_, lean_object* v_a_1822_, lean_object* v_a_1823_){
_start:
{
lean_object* v_replacement_1825_; lean_object* v_replacement_1826_; lean_object* v_mvars_1827_; lean_object* v_expr_1828_; lean_object* v_mvars_1829_; lean_object* v_expr_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; uint8_t v___x_1833_; 
v_replacement_1825_ = lean_ctor_get(v_a_1818_, 4);
lean_inc_ref(v_replacement_1825_);
lean_dec_ref(v_a_1818_);
v_replacement_1826_ = lean_ctor_get(v_b_1819_, 4);
lean_inc_ref(v_replacement_1826_);
lean_dec_ref(v_b_1819_);
v_mvars_1827_ = lean_ctor_get(v_replacement_1825_, 1);
lean_inc_ref(v_mvars_1827_);
v_expr_1828_ = lean_ctor_get(v_replacement_1825_, 2);
lean_inc_ref(v_expr_1828_);
lean_dec_ref(v_replacement_1825_);
v_mvars_1829_ = lean_ctor_get(v_replacement_1826_, 1);
lean_inc_ref(v_mvars_1829_);
v_expr_1830_ = lean_ctor_get(v_replacement_1826_, 2);
lean_inc_ref(v_expr_1830_);
lean_dec_ref(v_replacement_1826_);
v___x_1831_ = lean_array_get_size(v_mvars_1827_);
lean_dec_ref(v_mvars_1827_);
v___x_1832_ = lean_array_get_size(v_mvars_1829_);
lean_dec_ref(v_mvars_1829_);
v___x_1833_ = lean_nat_dec_eq(v___x_1831_, v___x_1832_);
if (v___x_1833_ == 0)
{
lean_object* v___x_1834_; lean_object* v___x_1835_; 
lean_dec_ref(v_expr_1830_);
lean_dec_ref(v_expr_1828_);
v___x_1834_ = lean_box(v___x_1833_);
v___x_1835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1834_);
return v___x_1835_;
}
else
{
lean_object* v___x_1836_; 
v___x_1836_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(v_expr_1828_, v_expr_1830_, v_a_1820_, v_a_1821_, v_a_1822_, v_a_1823_);
return v___x_1836_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwKey_isDuplicate___boxed(lean_object* v_a_1837_, lean_object* v_b_1838_, lean_object* v_a_1839_, lean_object* v_a_1840_, lean_object* v_a_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_){
_start:
{
lean_object* v_res_1844_; 
v_res_1844_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwKey_isDuplicate(v_a_1837_, v_b_1838_, v_a_1839_, v_a_1840_, v_a_1841_, v_a_1842_);
lean_dec(v_a_1842_);
lean_dec_ref(v_a_1841_);
lean_dec(v_a_1840_);
lean_dec_ref(v_a_1839_);
return v_res_1844_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object* v_opts_1845_, lean_object* v_opt_1846_){
_start:
{
lean_object* v_name_1847_; lean_object* v_defValue_1848_; lean_object* v_map_1849_; lean_object* v___x_1850_; 
v_name_1847_ = lean_ctor_get(v_opt_1846_, 0);
v_defValue_1848_ = lean_ctor_get(v_opt_1846_, 1);
v_map_1849_ = lean_ctor_get(v_opts_1845_, 0);
v___x_1850_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1849_, v_name_1847_);
if (lean_obj_tag(v___x_1850_) == 0)
{
uint8_t v___x_1851_; 
v___x_1851_ = lean_unbox(v_defValue_1848_);
return v___x_1851_;
}
else
{
lean_object* v_val_1852_; 
v_val_1852_ = lean_ctor_get(v___x_1850_, 0);
lean_inc(v_val_1852_);
lean_dec_ref_known(v___x_1850_, 1);
if (lean_obj_tag(v_val_1852_) == 1)
{
uint8_t v_v_1853_; 
v_v_1853_ = lean_ctor_get_uint8(v_val_1852_, 0);
lean_dec_ref_known(v_val_1852_, 0);
return v_v_1853_;
}
else
{
uint8_t v___x_1854_; 
lean_dec(v_val_1852_);
v___x_1854_ = lean_unbox(v_defValue_1848_);
return v___x_1854_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object* v_opts_1855_, lean_object* v_opt_1856_){
_start:
{
uint8_t v_res_1857_; lean_object* v_r_1858_; 
v_res_1857_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v_opts_1855_, v_opt_1856_);
lean_dec_ref(v_opt_1856_);
lean_dec_ref(v_opts_1855_);
v_r_1858_ = lean_box(v_res_1857_);
return v_r_1858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object* v_opts_1859_, lean_object* v_opt_1860_){
_start:
{
lean_object* v_name_1861_; lean_object* v_defValue_1862_; lean_object* v_map_1863_; lean_object* v___x_1864_; 
v_name_1861_ = lean_ctor_get(v_opt_1860_, 0);
v_defValue_1862_ = lean_ctor_get(v_opt_1860_, 1);
v_map_1863_ = lean_ctor_get(v_opts_1859_, 0);
v___x_1864_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1863_, v_name_1861_);
if (lean_obj_tag(v___x_1864_) == 0)
{
lean_inc(v_defValue_1862_);
return v_defValue_1862_;
}
else
{
lean_object* v_val_1865_; 
v_val_1865_ = lean_ctor_get(v___x_1864_, 0);
lean_inc(v_val_1865_);
lean_dec_ref_known(v___x_1864_, 1);
if (lean_obj_tag(v_val_1865_) == 3)
{
lean_object* v_v_1866_; 
v_v_1866_ = lean_ctor_get(v_val_1865_, 0);
lean_inc(v_v_1866_);
lean_dec_ref_known(v_val_1865_, 1);
return v_v_1866_;
}
else
{
lean_dec(v_val_1865_);
lean_inc(v_defValue_1862_);
return v_defValue_1862_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object* v_opts_1867_, lean_object* v_opt_1868_){
_start:
{
lean_object* v_res_1869_; 
v_res_1869_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v_opts_1867_, v_opt_1868_);
lean_dec_ref(v_opt_1868_);
lean_dec_ref(v_opts_1867_);
return v_res_1869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object* v_o_1873_, lean_object* v_k_1874_, uint8_t v_v_1875_){
_start:
{
lean_object* v_map_1876_; uint8_t v_hasTrace_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1891_; 
v_map_1876_ = lean_ctor_get(v_o_1873_, 0);
v_hasTrace_1877_ = lean_ctor_get_uint8(v_o_1873_, sizeof(void*)*1);
v_isSharedCheck_1891_ = !lean_is_exclusive(v_o_1873_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1879_ = v_o_1873_;
v_isShared_1880_ = v_isSharedCheck_1891_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_map_1876_);
lean_dec(v_o_1873_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1891_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1881_; lean_object* v___x_1882_; 
v___x_1881_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_1881_, 0, v_v_1875_);
lean_inc(v_k_1874_);
v___x_1882_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_1874_, v___x_1881_, v_map_1876_);
if (v_hasTrace_1877_ == 0)
{
lean_object* v___x_1883_; uint8_t v___x_1884_; lean_object* v___x_1886_; 
v___x_1883_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1));
v___x_1884_ = l_Lean_Name_isPrefixOf(v___x_1883_, v_k_1874_);
lean_dec(v_k_1874_);
if (v_isShared_1880_ == 0)
{
lean_ctor_set(v___x_1879_, 0, v___x_1882_);
v___x_1886_ = v___x_1879_;
goto v_reusejp_1885_;
}
else
{
lean_object* v_reuseFailAlloc_1887_; 
v_reuseFailAlloc_1887_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_1887_, 0, v___x_1882_);
v___x_1886_ = v_reuseFailAlloc_1887_;
goto v_reusejp_1885_;
}
v_reusejp_1885_:
{
lean_ctor_set_uint8(v___x_1886_, sizeof(void*)*1, v___x_1884_);
return v___x_1886_;
}
}
else
{
lean_object* v___x_1889_; 
lean_dec(v_k_1874_);
if (v_isShared_1880_ == 0)
{
lean_ctor_set(v___x_1879_, 0, v___x_1882_);
v___x_1889_ = v___x_1879_;
goto v_reusejp_1888_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_1890_, 0, v___x_1882_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, sizeof(void*)*1, v_hasTrace_1877_);
v___x_1889_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1888_;
}
v_reusejp_1888_:
{
return v___x_1889_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object* v_o_1892_, lean_object* v_k_1893_, lean_object* v_v_1894_){
_start:
{
uint8_t v_v_boxed_1895_; lean_object* v_res_1896_; 
v_v_boxed_1895_ = lean_unbox(v_v_1894_);
v_res_1896_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_o_1892_, v_k_1893_, v_v_boxed_1895_);
return v_res_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object* v_opts_1897_, lean_object* v_opt_1898_, uint8_t v_val_1899_){
_start:
{
lean_object* v_name_1900_; lean_object* v___x_1901_; 
v_name_1900_ = lean_ctor_get(v_opt_1898_, 0);
lean_inc(v_name_1900_);
lean_dec_ref(v_opt_1898_);
v___x_1901_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_opts_1897_, v_name_1900_, v_val_1899_);
return v___x_1901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object* v_opts_1902_, lean_object* v_opt_1903_, lean_object* v_val_1904_){
_start:
{
uint8_t v_val_boxed_1905_; lean_object* v_res_1906_; 
v_val_boxed_1905_ = lean_unbox(v_val_1904_);
v_res_1906_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_opts_1902_, v_opt_1903_, v_val_boxed_1905_);
return v_res_1906_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0(void){
_start:
{
lean_object* v___x_1907_; 
v___x_1907_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1907_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1(void){
_start:
{
lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___x_1908_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__0);
v___x_1909_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1909_, 0, v___x_1908_);
return v___x_1909_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2(void){
_start:
{
lean_object* v___x_1910_; lean_object* v___x_1911_; 
v___x_1910_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__1);
v___x_1911_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1911_, 0, v___x_1910_);
lean_ctor_set(v___x_1911_, 1, v___x_1910_);
return v___x_1911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(lean_object* v_lem_1912_, lean_object* v_i_1913_, lean_object* v_proof_1914_, uint8_t v_justLemmaName_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_){
_start:
{
lean_object* v_proof_1923_; lean_object* v___y_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; 
if (v_justLemmaName_1915_ == 0)
{
lean_object* v___x_1942_; lean_object* v_fileName_1943_; lean_object* v_fileMap_1944_; lean_object* v_options_1945_; lean_object* v_currRecDepth_1946_; lean_object* v_ref_1947_; lean_object* v_currNamespace_1948_; lean_object* v_openDecls_1949_; lean_object* v_initHeartbeats_1950_; lean_object* v_maxHeartbeats_1951_; lean_object* v_quotContext_1952_; lean_object* v_currMacroScope_1953_; lean_object* v_cancelTk_x3f_1954_; uint8_t v_suppressElabErrors_1955_; lean_object* v_inheritedTraceOptions_1956_; lean_object* v_env_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; uint8_t v___x_1962_; lean_object* v_fileName_1964_; lean_object* v_fileMap_1965_; lean_object* v_currRecDepth_1966_; lean_object* v_ref_1967_; lean_object* v_currNamespace_1968_; lean_object* v_openDecls_1969_; lean_object* v_initHeartbeats_1970_; lean_object* v_maxHeartbeats_1971_; lean_object* v_quotContext_1972_; lean_object* v_currMacroScope_1973_; lean_object* v_cancelTk_x3f_1974_; uint8_t v_suppressElabErrors_1975_; lean_object* v_inheritedTraceOptions_1976_; lean_object* v___y_1977_; uint8_t v___y_1984_; uint8_t v___x_2005_; 
v___x_1942_ = lean_st_ref_get(v_a_1920_);
v_fileName_1943_ = lean_ctor_get(v_a_1919_, 0);
v_fileMap_1944_ = lean_ctor_get(v_a_1919_, 1);
v_options_1945_ = lean_ctor_get(v_a_1919_, 2);
v_currRecDepth_1946_ = lean_ctor_get(v_a_1919_, 3);
v_ref_1947_ = lean_ctor_get(v_a_1919_, 5);
v_currNamespace_1948_ = lean_ctor_get(v_a_1919_, 6);
v_openDecls_1949_ = lean_ctor_get(v_a_1919_, 7);
v_initHeartbeats_1950_ = lean_ctor_get(v_a_1919_, 8);
v_maxHeartbeats_1951_ = lean_ctor_get(v_a_1919_, 9);
v_quotContext_1952_ = lean_ctor_get(v_a_1919_, 10);
v_currMacroScope_1953_ = lean_ctor_get(v_a_1919_, 11);
v_cancelTk_x3f_1954_ = lean_ctor_get(v_a_1919_, 12);
v_suppressElabErrors_1955_ = lean_ctor_get_uint8(v_a_1919_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1956_ = lean_ctor_get(v_a_1919_, 13);
v_env_1957_ = lean_ctor_get(v___x_1942_, 0);
lean_inc_ref(v_env_1957_);
lean_dec(v___x_1942_);
v___x_1958_ = lean_box(1);
v___x_1959_ = l_Lean_pp_mvars;
lean_inc_ref(v_options_1945_);
v___x_1960_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_options_1945_, v___x_1959_, v_justLemmaName_1915_);
v___x_1961_ = l_Lean_diagnostics;
v___x_1962_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v___x_1960_, v___x_1961_);
v___x_2005_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_1957_);
lean_dec_ref(v_env_1957_);
if (v___x_2005_ == 0)
{
if (v___x_1962_ == 0)
{
v_fileName_1964_ = v_fileName_1943_;
v_fileMap_1965_ = v_fileMap_1944_;
v_currRecDepth_1966_ = v_currRecDepth_1946_;
v_ref_1967_ = v_ref_1947_;
v_currNamespace_1968_ = v_currNamespace_1948_;
v_openDecls_1969_ = v_openDecls_1949_;
v_initHeartbeats_1970_ = v_initHeartbeats_1950_;
v_maxHeartbeats_1971_ = v_maxHeartbeats_1951_;
v_quotContext_1972_ = v_quotContext_1952_;
v_currMacroScope_1973_ = v_currMacroScope_1953_;
v_cancelTk_x3f_1974_ = v_cancelTk_x3f_1954_;
v_suppressElabErrors_1975_ = v_suppressElabErrors_1955_;
v_inheritedTraceOptions_1976_ = v_inheritedTraceOptions_1956_;
v___y_1977_ = v_a_1920_;
goto v___jp_1963_;
}
else
{
v___y_1984_ = v___x_2005_;
goto v___jp_1983_;
}
}
else
{
v___y_1984_ = v___x_1962_;
goto v___jp_1983_;
}
v___jp_1963_:
{
lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; 
v___x_1978_ = l_Lean_maxRecDepth;
v___x_1979_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v___x_1960_, v___x_1978_);
lean_inc_ref(v_inheritedTraceOptions_1976_);
lean_inc(v_cancelTk_x3f_1974_);
lean_inc(v_currMacroScope_1973_);
lean_inc(v_quotContext_1972_);
lean_inc(v_maxHeartbeats_1971_);
lean_inc(v_initHeartbeats_1970_);
lean_inc(v_openDecls_1969_);
lean_inc(v_currNamespace_1968_);
lean_inc(v_ref_1967_);
lean_inc(v_currRecDepth_1966_);
lean_inc_ref(v_fileMap_1965_);
lean_inc_ref(v_fileName_1964_);
v___x_1980_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1980_, 0, v_fileName_1964_);
lean_ctor_set(v___x_1980_, 1, v_fileMap_1965_);
lean_ctor_set(v___x_1980_, 2, v___x_1960_);
lean_ctor_set(v___x_1980_, 3, v_currRecDepth_1966_);
lean_ctor_set(v___x_1980_, 4, v___x_1979_);
lean_ctor_set(v___x_1980_, 5, v_ref_1967_);
lean_ctor_set(v___x_1980_, 6, v_currNamespace_1968_);
lean_ctor_set(v___x_1980_, 7, v_openDecls_1969_);
lean_ctor_set(v___x_1980_, 8, v_initHeartbeats_1970_);
lean_ctor_set(v___x_1980_, 9, v_maxHeartbeats_1971_);
lean_ctor_set(v___x_1980_, 10, v_quotContext_1972_);
lean_ctor_set(v___x_1980_, 11, v_currMacroScope_1973_);
lean_ctor_set(v___x_1980_, 12, v_cancelTk_x3f_1974_);
lean_ctor_set(v___x_1980_, 13, v_inheritedTraceOptions_1976_);
lean_ctor_set_uint8(v___x_1980_, sizeof(void*)*14, v___x_1962_);
lean_ctor_set_uint8(v___x_1980_, sizeof(void*)*14 + 1, v_suppressElabErrors_1975_);
v___x_1981_ = l_Lean_PrettyPrinter_delab(v_proof_1914_, v___x_1958_, v_a_1917_, v_a_1918_, v___x_1980_, v___y_1977_);
lean_dec_ref_known(v___x_1980_, 14);
if (lean_obj_tag(v___x_1981_) == 0)
{
lean_object* v_a_1982_; 
v_a_1982_ = lean_ctor_get(v___x_1981_, 0);
lean_inc(v_a_1982_);
lean_dec_ref_known(v___x_1981_, 1);
v_proof_1923_ = v_a_1982_;
v___y_1924_ = v_a_1916_;
v___y_1925_ = v_a_1917_;
v___y_1926_ = v_a_1919_;
v___y_1927_ = v_a_1920_;
goto v___jp_1922_;
}
else
{
lean_dec_ref(v_i_1913_);
lean_dec_ref(v_lem_1912_);
return v___x_1981_;
}
}
v___jp_1983_:
{
if (v___y_1984_ == 0)
{
lean_object* v___x_1985_; lean_object* v_env_1986_; lean_object* v_nextMacroScope_1987_; lean_object* v_ngen_1988_; lean_object* v_auxDeclNGen_1989_; lean_object* v_traceState_1990_; lean_object* v_messages_1991_; lean_object* v_infoState_1992_; lean_object* v_snapshotTasks_1993_; lean_object* v___x_1995_; uint8_t v_isShared_1996_; uint8_t v_isSharedCheck_2003_; 
v___x_1985_ = lean_st_ref_take(v_a_1920_);
v_env_1986_ = lean_ctor_get(v___x_1985_, 0);
v_nextMacroScope_1987_ = lean_ctor_get(v___x_1985_, 1);
v_ngen_1988_ = lean_ctor_get(v___x_1985_, 2);
v_auxDeclNGen_1989_ = lean_ctor_get(v___x_1985_, 3);
v_traceState_1990_ = lean_ctor_get(v___x_1985_, 4);
v_messages_1991_ = lean_ctor_get(v___x_1985_, 6);
v_infoState_1992_ = lean_ctor_get(v___x_1985_, 7);
v_snapshotTasks_1993_ = lean_ctor_get(v___x_1985_, 8);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1985_);
if (v_isSharedCheck_2003_ == 0)
{
lean_object* v_unused_2004_; 
v_unused_2004_ = lean_ctor_get(v___x_1985_, 5);
lean_dec(v_unused_2004_);
v___x_1995_ = v___x_1985_;
v_isShared_1996_ = v_isSharedCheck_2003_;
goto v_resetjp_1994_;
}
else
{
lean_inc(v_snapshotTasks_1993_);
lean_inc(v_infoState_1992_);
lean_inc(v_messages_1991_);
lean_inc(v_traceState_1990_);
lean_inc(v_auxDeclNGen_1989_);
lean_inc(v_ngen_1988_);
lean_inc(v_nextMacroScope_1987_);
lean_inc(v_env_1986_);
lean_dec(v___x_1985_);
v___x_1995_ = lean_box(0);
v_isShared_1996_ = v_isSharedCheck_2003_;
goto v_resetjp_1994_;
}
v_resetjp_1994_:
{
lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_2000_; 
v___x_1997_ = l_Lean_Kernel_enableDiag(v_env_1986_, v___x_1962_);
v___x_1998_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___closed__2);
if (v_isShared_1996_ == 0)
{
lean_ctor_set(v___x_1995_, 5, v___x_1998_);
lean_ctor_set(v___x_1995_, 0, v___x_1997_);
v___x_2000_ = v___x_1995_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v___x_1997_);
lean_ctor_set(v_reuseFailAlloc_2002_, 1, v_nextMacroScope_1987_);
lean_ctor_set(v_reuseFailAlloc_2002_, 2, v_ngen_1988_);
lean_ctor_set(v_reuseFailAlloc_2002_, 3, v_auxDeclNGen_1989_);
lean_ctor_set(v_reuseFailAlloc_2002_, 4, v_traceState_1990_);
lean_ctor_set(v_reuseFailAlloc_2002_, 5, v___x_1998_);
lean_ctor_set(v_reuseFailAlloc_2002_, 6, v_messages_1991_);
lean_ctor_set(v_reuseFailAlloc_2002_, 7, v_infoState_1992_);
lean_ctor_set(v_reuseFailAlloc_2002_, 8, v_snapshotTasks_1993_);
v___x_2000_ = v_reuseFailAlloc_2002_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
lean_object* v___x_2001_; 
v___x_2001_ = lean_st_ref_set(v_a_1920_, v___x_2000_);
v_fileName_1964_ = v_fileName_1943_;
v_fileMap_1965_ = v_fileMap_1944_;
v_currRecDepth_1966_ = v_currRecDepth_1946_;
v_ref_1967_ = v_ref_1947_;
v_currNamespace_1968_ = v_currNamespace_1948_;
v_openDecls_1969_ = v_openDecls_1949_;
v_initHeartbeats_1970_ = v_initHeartbeats_1950_;
v_maxHeartbeats_1971_ = v_maxHeartbeats_1951_;
v_quotContext_1972_ = v_quotContext_1952_;
v_currMacroScope_1973_ = v_currMacroScope_1953_;
v_cancelTk_x3f_1974_ = v_cancelTk_x3f_1954_;
v_suppressElabErrors_1975_ = v_suppressElabErrors_1955_;
v_inheritedTraceOptions_1976_ = v_inheritedTraceOptions_1956_;
v___y_1977_ = v_a_1920_;
goto v___jp_1963_;
}
}
}
else
{
v_fileName_1964_ = v_fileName_1943_;
v_fileMap_1965_ = v_fileMap_1944_;
v_currRecDepth_1966_ = v_currRecDepth_1946_;
v_ref_1967_ = v_ref_1947_;
v_currNamespace_1968_ = v_currNamespace_1948_;
v_openDecls_1969_ = v_openDecls_1949_;
v_initHeartbeats_1970_ = v_initHeartbeats_1950_;
v_maxHeartbeats_1971_ = v_maxHeartbeats_1951_;
v_quotContext_1972_ = v_quotContext_1952_;
v_currMacroScope_1973_ = v_currMacroScope_1953_;
v_cancelTk_x3f_1974_ = v_cancelTk_x3f_1954_;
v_suppressElabErrors_1975_ = v_suppressElabErrors_1955_;
v_inheritedTraceOptions_1976_ = v_inheritedTraceOptions_1956_;
v___y_1977_ = v_a_1920_;
goto v___jp_1963_;
}
}
}
else
{
lean_object* v_name_2006_; lean_object* v___x_2007_; 
lean_dec_ref(v_proof_1914_);
v_name_2006_ = lean_ctor_get(v_lem_1912_, 0);
lean_inc_ref(v_name_2006_);
v___x_2007_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(v_name_2006_, v_a_1917_, v_a_1918_, v_a_1919_, v_a_1920_);
if (lean_obj_tag(v___x_2007_) == 0)
{
lean_object* v_a_2008_; lean_object* v___x_2009_; 
v_a_2008_ = lean_ctor_get(v___x_2007_, 0);
lean_inc(v_a_2008_);
lean_dec_ref_known(v___x_2007_, 1);
v___x_2009_ = l_Lean_mkIdent(v_a_2008_);
v_proof_1923_ = v___x_2009_;
v___y_1924_ = v_a_1916_;
v___y_1925_ = v_a_1917_;
v___y_1926_ = v_a_1919_;
v___y_1927_ = v_a_1920_;
goto v___jp_1922_;
}
else
{
lean_object* v_a_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2017_; 
lean_dec_ref(v_i_1913_);
lean_dec_ref(v_lem_1912_);
v_a_2010_ = lean_ctor_get(v___x_2007_, 0);
v_isSharedCheck_2017_ = !lean_is_exclusive(v___x_2007_);
if (v_isSharedCheck_2017_ == 0)
{
v___x_2012_ = v___x_2007_;
v_isShared_2013_ = v_isSharedCheck_2017_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_a_2010_);
lean_dec(v___x_2007_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2017_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v___x_2015_; 
if (v_isShared_2013_ == 0)
{
v___x_2015_ = v___x_2012_;
goto v_reusejp_2014_;
}
else
{
lean_object* v_reuseFailAlloc_2016_; 
v_reuseFailAlloc_2016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2016_, 0, v_a_2010_);
v___x_2015_ = v_reuseFailAlloc_2016_;
goto v_reusejp_2014_;
}
v_reusejp_2014_:
{
return v___x_2015_;
}
}
}
}
v___jp_1922_:
{
lean_object* v___x_1928_; 
v___x_1928_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_);
if (lean_obj_tag(v___x_1928_) == 0)
{
lean_object* v_a_1929_; lean_object* v_rwKind_1930_; uint8_t v_symm_1931_; uint8_t v___x_1932_; lean_object* v___x_1933_; 
v_a_1929_ = lean_ctor_get(v___x_1928_, 0);
lean_inc(v_a_1929_);
lean_dec_ref_known(v___x_1928_, 1);
v_rwKind_1930_ = lean_ctor_get(v_i_1913_, 4);
lean_inc(v_rwKind_1930_);
lean_dec_ref(v_i_1913_);
v_symm_1931_ = lean_ctor_get_uint8(v_lem_1912_, sizeof(void*)*2);
lean_dec_ref(v_lem_1912_);
v___x_1932_ = 1;
v___x_1933_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(v_rwKind_1930_, v_symm_1931_, v_proof_1923_, v_a_1929_, v___x_1932_, v___y_1926_);
return v___x_1933_;
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_1941_; 
lean_dec(v_proof_1923_);
lean_dec_ref(v_i_1913_);
lean_dec_ref(v_lem_1912_);
v_a_1934_ = lean_ctor_get(v___x_1928_, 0);
v_isSharedCheck_1941_ = !lean_is_exclusive(v___x_1928_);
if (v_isSharedCheck_1941_ == 0)
{
v___x_1936_ = v___x_1928_;
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1928_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___x_1939_; 
if (v_isShared_1937_ == 0)
{
v___x_1939_ = v___x_1936_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v_a_1934_);
v___x_1939_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
return v___x_1939_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg___boxed(lean_object* v_lem_2018_, lean_object* v_i_2019_, lean_object* v_proof_2020_, lean_object* v_justLemmaName_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_, lean_object* v_a_2026_, lean_object* v_a_2027_){
_start:
{
uint8_t v_justLemmaName_boxed_2028_; lean_object* v_res_2029_; 
v_justLemmaName_boxed_2028_ = lean_unbox(v_justLemmaName_2021_);
v_res_2029_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_lem_2018_, v_i_2019_, v_proof_2020_, v_justLemmaName_boxed_2028_, v_a_2022_, v_a_2023_, v_a_2024_, v_a_2025_, v_a_2026_);
lean_dec(v_a_2026_);
lean_dec_ref(v_a_2025_);
lean_dec(v_a_2024_);
lean_dec_ref(v_a_2023_);
lean_dec_ref(v_a_2022_);
return v_res_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object* v_lem_2030_, lean_object* v_i_2031_, lean_object* v_proof_2032_, uint8_t v_justLemmaName_2033_, lean_object* v_a_2034_, lean_object* v_a_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_, lean_object* v_a_2039_){
_start:
{
lean_object* v___x_2041_; 
v___x_2041_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_lem_2030_, v_i_2031_, v_proof_2032_, v_justLemmaName_2033_, v_a_2034_, v_a_2036_, v_a_2037_, v_a_2038_, v_a_2039_);
return v___x_2041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object* v_lem_2042_, lean_object* v_i_2043_, lean_object* v_proof_2044_, lean_object* v_justLemmaName_2045_, lean_object* v_a_2046_, lean_object* v_a_2047_, lean_object* v_a_2048_, lean_object* v_a_2049_, lean_object* v_a_2050_, lean_object* v_a_2051_, lean_object* v_a_2052_){
_start:
{
uint8_t v_justLemmaName_boxed_2053_; lean_object* v_res_2054_; 
v_justLemmaName_boxed_2053_ = lean_unbox(v_justLemmaName_2045_);
v_res_2054_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_2042_, v_i_2043_, v_proof_2044_, v_justLemmaName_boxed_2053_, v_a_2046_, v_a_2047_, v_a_2048_, v_a_2049_, v_a_2050_, v_a_2051_);
lean_dec(v_a_2051_);
lean_dec_ref(v_a_2050_);
lean_dec(v_a_2049_);
lean_dec_ref(v_a_2048_);
lean_dec(v_a_2047_);
lean_dec_ref(v_a_2046_);
return v_res_2054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(lean_object* v_e_2055_, lean_object* v___y_2056_){
_start:
{
uint8_t v___x_2058_; 
v___x_2058_ = l_Lean_Expr_hasMVar(v_e_2055_);
if (v___x_2058_ == 0)
{
lean_object* v___x_2059_; 
v___x_2059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2059_, 0, v_e_2055_);
return v___x_2059_;
}
else
{
lean_object* v___x_2060_; lean_object* v_mctx_2061_; lean_object* v___x_2062_; lean_object* v_fst_2063_; lean_object* v_snd_2064_; lean_object* v___x_2065_; lean_object* v_cache_2066_; lean_object* v_zetaDeltaFVarIds_2067_; lean_object* v_postponed_2068_; lean_object* v_diag_2069_; lean_object* v___x_2071_; uint8_t v_isShared_2072_; uint8_t v_isSharedCheck_2078_; 
v___x_2060_ = lean_st_ref_get(v___y_2056_);
v_mctx_2061_ = lean_ctor_get(v___x_2060_, 0);
lean_inc_ref(v_mctx_2061_);
lean_dec(v___x_2060_);
v___x_2062_ = l_Lean_instantiateMVarsCore(v_mctx_2061_, v_e_2055_);
v_fst_2063_ = lean_ctor_get(v___x_2062_, 0);
lean_inc(v_fst_2063_);
v_snd_2064_ = lean_ctor_get(v___x_2062_, 1);
lean_inc(v_snd_2064_);
lean_dec_ref(v___x_2062_);
v___x_2065_ = lean_st_ref_take(v___y_2056_);
v_cache_2066_ = lean_ctor_get(v___x_2065_, 1);
v_zetaDeltaFVarIds_2067_ = lean_ctor_get(v___x_2065_, 2);
v_postponed_2068_ = lean_ctor_get(v___x_2065_, 3);
v_diag_2069_ = lean_ctor_get(v___x_2065_, 4);
v_isSharedCheck_2078_ = !lean_is_exclusive(v___x_2065_);
if (v_isSharedCheck_2078_ == 0)
{
lean_object* v_unused_2079_; 
v_unused_2079_ = lean_ctor_get(v___x_2065_, 0);
lean_dec(v_unused_2079_);
v___x_2071_ = v___x_2065_;
v_isShared_2072_ = v_isSharedCheck_2078_;
goto v_resetjp_2070_;
}
else
{
lean_inc(v_diag_2069_);
lean_inc(v_postponed_2068_);
lean_inc(v_zetaDeltaFVarIds_2067_);
lean_inc(v_cache_2066_);
lean_dec(v___x_2065_);
v___x_2071_ = lean_box(0);
v_isShared_2072_ = v_isSharedCheck_2078_;
goto v_resetjp_2070_;
}
v_resetjp_2070_:
{
lean_object* v___x_2074_; 
if (v_isShared_2072_ == 0)
{
lean_ctor_set(v___x_2071_, 0, v_snd_2064_);
v___x_2074_ = v___x_2071_;
goto v_reusejp_2073_;
}
else
{
lean_object* v_reuseFailAlloc_2077_; 
v_reuseFailAlloc_2077_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2077_, 0, v_snd_2064_);
lean_ctor_set(v_reuseFailAlloc_2077_, 1, v_cache_2066_);
lean_ctor_set(v_reuseFailAlloc_2077_, 2, v_zetaDeltaFVarIds_2067_);
lean_ctor_set(v_reuseFailAlloc_2077_, 3, v_postponed_2068_);
lean_ctor_set(v_reuseFailAlloc_2077_, 4, v_diag_2069_);
v___x_2074_ = v_reuseFailAlloc_2077_;
goto v_reusejp_2073_;
}
v_reusejp_2073_:
{
lean_object* v___x_2075_; lean_object* v___x_2076_; 
v___x_2075_ = lean_st_ref_set(v___y_2056_, v___x_2074_);
v___x_2076_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2076_, 0, v_fst_2063_);
return v___x_2076_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg___boxed(lean_object* v_e_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_){
_start:
{
lean_object* v_res_2083_; 
v_res_2083_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_e_2080_, v___y_2081_);
lean_dec(v___y_2081_);
return v_res_2083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1(lean_object* v_e_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_){
_start:
{
lean_object* v___x_2092_; 
v___x_2092_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_e_2084_, v___y_2088_);
return v___x_2092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___boxed(lean_object* v_e_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_){
_start:
{
lean_object* v_res_2101_; 
v_res_2101_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1(v_e_2093_, v___y_2094_, v___y_2095_, v___y_2096_, v___y_2097_, v___y_2098_, v___y_2099_);
lean_dec(v___y_2099_);
lean_dec_ref(v___y_2098_);
lean_dec(v___y_2097_);
lean_dec_ref(v___y_2096_);
lean_dec(v___y_2095_);
lean_dec_ref(v___y_2094_);
return v_res_2101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0(lean_object* v___y_2102_, lean_object* v_mctx_2103_, lean_object* v_cache_2104_, lean_object* v_a_x3f_2105_){
_start:
{
lean_object* v___x_2107_; lean_object* v_zetaDeltaFVarIds_2108_; lean_object* v_postponed_2109_; lean_object* v_diag_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2120_; 
v___x_2107_ = lean_st_ref_take(v___y_2102_);
v_zetaDeltaFVarIds_2108_ = lean_ctor_get(v___x_2107_, 2);
v_postponed_2109_ = lean_ctor_get(v___x_2107_, 3);
v_diag_2110_ = lean_ctor_get(v___x_2107_, 4);
v_isSharedCheck_2120_ = !lean_is_exclusive(v___x_2107_);
if (v_isSharedCheck_2120_ == 0)
{
lean_object* v_unused_2121_; lean_object* v_unused_2122_; 
v_unused_2121_ = lean_ctor_get(v___x_2107_, 1);
lean_dec(v_unused_2121_);
v_unused_2122_ = lean_ctor_get(v___x_2107_, 0);
lean_dec(v_unused_2122_);
v___x_2112_ = v___x_2107_;
v_isShared_2113_ = v_isSharedCheck_2120_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_diag_2110_);
lean_inc(v_postponed_2109_);
lean_inc(v_zetaDeltaFVarIds_2108_);
lean_dec(v___x_2107_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2120_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2115_; 
if (v_isShared_2113_ == 0)
{
lean_ctor_set(v___x_2112_, 1, v_cache_2104_);
lean_ctor_set(v___x_2112_, 0, v_mctx_2103_);
v___x_2115_ = v___x_2112_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2119_; 
v_reuseFailAlloc_2119_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2119_, 0, v_mctx_2103_);
lean_ctor_set(v_reuseFailAlloc_2119_, 1, v_cache_2104_);
lean_ctor_set(v_reuseFailAlloc_2119_, 2, v_zetaDeltaFVarIds_2108_);
lean_ctor_set(v_reuseFailAlloc_2119_, 3, v_postponed_2109_);
lean_ctor_set(v_reuseFailAlloc_2119_, 4, v_diag_2110_);
v___x_2115_ = v_reuseFailAlloc_2119_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; 
v___x_2116_ = lean_st_ref_set(v___y_2102_, v___x_2115_);
v___x_2117_ = lean_box(0);
v___x_2118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2118_, 0, v___x_2117_);
return v___x_2118_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0___boxed(lean_object* v___y_2123_, lean_object* v_mctx_2124_, lean_object* v_cache_2125_, lean_object* v_a_x3f_2126_, lean_object* v___y_2127_){
_start:
{
lean_object* v_res_2128_; 
v_res_2128_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0(v___y_2123_, v_mctx_2124_, v_cache_2125_, v_a_x3f_2126_);
lean_dec(v_a_x3f_2126_);
lean_dec(v___y_2123_);
return v_res_2128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg(lean_object* v_x_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_){
_start:
{
lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v_mctx_2139_; lean_object* v_cache_2140_; lean_object* v___x_2141_; 
v___x_2137_ = lean_st_ref_get(v___y_2133_);
v___x_2138_ = lean_st_ref_get(v___y_2133_);
v_mctx_2139_ = lean_ctor_get(v___x_2137_, 0);
lean_inc_ref(v_mctx_2139_);
lean_dec(v___x_2137_);
v_cache_2140_ = lean_ctor_get(v___x_2138_, 1);
lean_inc_ref(v_cache_2140_);
lean_dec(v___x_2138_);
lean_inc(v___y_2135_);
lean_inc_ref(v___y_2134_);
lean_inc(v___y_2133_);
lean_inc_ref(v___y_2132_);
lean_inc(v___y_2131_);
lean_inc_ref(v___y_2130_);
v___x_2141_ = lean_apply_7(v_x_2129_, v___y_2130_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, lean_box(0));
if (lean_obj_tag(v___x_2141_) == 0)
{
lean_object* v_a_2142_; lean_object* v___x_2144_; uint8_t v_isShared_2145_; uint8_t v_isSharedCheck_2158_; 
v_a_2142_ = lean_ctor_get(v___x_2141_, 0);
v_isSharedCheck_2158_ = !lean_is_exclusive(v___x_2141_);
if (v_isSharedCheck_2158_ == 0)
{
v___x_2144_ = v___x_2141_;
v_isShared_2145_ = v_isSharedCheck_2158_;
goto v_resetjp_2143_;
}
else
{
lean_inc(v_a_2142_);
lean_dec(v___x_2141_);
v___x_2144_ = lean_box(0);
v_isShared_2145_ = v_isSharedCheck_2158_;
goto v_resetjp_2143_;
}
v_resetjp_2143_:
{
lean_object* v___x_2147_; 
lean_inc(v_a_2142_);
if (v_isShared_2145_ == 0)
{
lean_ctor_set_tag(v___x_2144_, 1);
v___x_2147_ = v___x_2144_;
goto v_reusejp_2146_;
}
else
{
lean_object* v_reuseFailAlloc_2157_; 
v_reuseFailAlloc_2157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2157_, 0, v_a_2142_);
v___x_2147_ = v_reuseFailAlloc_2157_;
goto v_reusejp_2146_;
}
v_reusejp_2146_:
{
lean_object* v___x_2148_; lean_object* v___x_2150_; uint8_t v_isShared_2151_; uint8_t v_isSharedCheck_2155_; 
v___x_2148_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0(v___y_2133_, v_mctx_2139_, v_cache_2140_, v___x_2147_);
lean_dec_ref(v___x_2147_);
v_isSharedCheck_2155_ = !lean_is_exclusive(v___x_2148_);
if (v_isSharedCheck_2155_ == 0)
{
lean_object* v_unused_2156_; 
v_unused_2156_ = lean_ctor_get(v___x_2148_, 0);
lean_dec(v_unused_2156_);
v___x_2150_ = v___x_2148_;
v_isShared_2151_ = v_isSharedCheck_2155_;
goto v_resetjp_2149_;
}
else
{
lean_dec(v___x_2148_);
v___x_2150_ = lean_box(0);
v_isShared_2151_ = v_isSharedCheck_2155_;
goto v_resetjp_2149_;
}
v_resetjp_2149_:
{
lean_object* v___x_2153_; 
if (v_isShared_2151_ == 0)
{
lean_ctor_set(v___x_2150_, 0, v_a_2142_);
v___x_2153_ = v___x_2150_;
goto v_reusejp_2152_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v_a_2142_);
v___x_2153_ = v_reuseFailAlloc_2154_;
goto v_reusejp_2152_;
}
v_reusejp_2152_:
{
return v___x_2153_;
}
}
}
}
}
else
{
lean_object* v_a_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2163_; uint8_t v_isShared_2164_; uint8_t v_isSharedCheck_2168_; 
v_a_2159_ = lean_ctor_get(v___x_2141_, 0);
lean_inc(v_a_2159_);
lean_dec_ref_known(v___x_2141_, 1);
v___x_2160_ = lean_box(0);
v___x_2161_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___lam__0(v___y_2133_, v_mctx_2139_, v_cache_2140_, v___x_2160_);
v_isSharedCheck_2168_ = !lean_is_exclusive(v___x_2161_);
if (v_isSharedCheck_2168_ == 0)
{
lean_object* v_unused_2169_; 
v_unused_2169_ = lean_ctor_get(v___x_2161_, 0);
lean_dec(v_unused_2169_);
v___x_2163_ = v___x_2161_;
v_isShared_2164_ = v_isSharedCheck_2168_;
goto v_resetjp_2162_;
}
else
{
lean_dec(v___x_2161_);
v___x_2163_ = lean_box(0);
v_isShared_2164_ = v_isSharedCheck_2168_;
goto v_resetjp_2162_;
}
v_resetjp_2162_:
{
lean_object* v___x_2166_; 
if (v_isShared_2164_ == 0)
{
lean_ctor_set_tag(v___x_2163_, 1);
lean_ctor_set(v___x_2163_, 0, v_a_2159_);
v___x_2166_ = v___x_2163_;
goto v_reusejp_2165_;
}
else
{
lean_object* v_reuseFailAlloc_2167_; 
v_reuseFailAlloc_2167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2167_, 0, v_a_2159_);
v___x_2166_ = v_reuseFailAlloc_2167_;
goto v_reusejp_2165_;
}
v_reusejp_2165_:
{
return v___x_2166_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg___boxed(lean_object* v_x_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
lean_object* v_res_2178_; 
v_res_2178_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg(v_x_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
lean_dec(v___y_2176_);
lean_dec_ref(v___y_2175_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2171_);
return v_res_2178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6(lean_object* v_00_u03b1_2179_, lean_object* v_x_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_){
_start:
{
lean_object* v___x_2188_; 
v___x_2188_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg(v_x_2180_, v___y_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_, v___y_2186_);
return v___x_2188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___boxed(lean_object* v_00_u03b1_2189_, lean_object* v_x_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_){
_start:
{
lean_object* v_res_2198_; 
v_res_2198_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6(v_00_u03b1_2189_, v_x_2190_, v___y_2191_, v___y_2192_, v___y_2193_, v___y_2194_, v___y_2195_, v___y_2196_);
lean_dec(v___y_2196_);
lean_dec_ref(v___y_2195_);
lean_dec(v___y_2194_);
lean_dec_ref(v___y_2193_);
lean_dec(v___y_2192_);
lean_dec_ref(v___y_2191_);
return v_res_2198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0(lean_object* v_x_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_){
_start:
{
lean_object* v___x_2207_; 
lean_inc(v___y_2201_);
lean_inc_ref(v___y_2200_);
v___x_2207_ = lean_apply_7(v_x_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_, v___y_2205_, lean_box(0));
return v___x_2207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0___boxed(lean_object* v_x_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_){
_start:
{
lean_object* v_res_2216_; 
v_res_2216_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0(v_x_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_, v___y_2213_, v___y_2214_);
lean_dec(v___y_2210_);
lean_dec_ref(v___y_2209_);
return v_res_2216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg(lean_object* v_mctx_2217_, lean_object* v_x_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_){
_start:
{
lean_object* v___f_2226_; lean_object* v___x_2227_; 
lean_inc(v___y_2220_);
lean_inc_ref(v___y_2219_);
v___f_2226_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2226_, 0, v_x_2218_);
lean_closure_set(v___f_2226_, 1, v___y_2219_);
lean_closure_set(v___f_2226_, 2, v___y_2220_);
v___x_2227_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), v_mctx_2217_, v___f_2226_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_);
if (lean_obj_tag(v___x_2227_) == 0)
{
return v___x_2227_;
}
else
{
lean_object* v_a_2228_; lean_object* v___x_2230_; uint8_t v_isShared_2231_; uint8_t v_isSharedCheck_2235_; 
v_a_2228_ = lean_ctor_get(v___x_2227_, 0);
v_isSharedCheck_2235_ = !lean_is_exclusive(v___x_2227_);
if (v_isSharedCheck_2235_ == 0)
{
v___x_2230_ = v___x_2227_;
v_isShared_2231_ = v_isSharedCheck_2235_;
goto v_resetjp_2229_;
}
else
{
lean_inc(v_a_2228_);
lean_dec(v___x_2227_);
v___x_2230_ = lean_box(0);
v_isShared_2231_ = v_isSharedCheck_2235_;
goto v_resetjp_2229_;
}
v_resetjp_2229_:
{
lean_object* v___x_2233_; 
if (v_isShared_2231_ == 0)
{
v___x_2233_ = v___x_2230_;
goto v_reusejp_2232_;
}
else
{
lean_object* v_reuseFailAlloc_2234_; 
v_reuseFailAlloc_2234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2234_, 0, v_a_2228_);
v___x_2233_ = v_reuseFailAlloc_2234_;
goto v_reusejp_2232_;
}
v_reusejp_2232_:
{
return v___x_2233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg___boxed(lean_object* v_mctx_2236_, lean_object* v_x_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_){
_start:
{
lean_object* v_res_2245_; 
v_res_2245_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg(v_mctx_2236_, v_x_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_);
lean_dec(v___y_2243_);
lean_dec_ref(v___y_2242_);
lean_dec(v___y_2241_);
lean_dec_ref(v___y_2240_);
lean_dec(v___y_2239_);
lean_dec_ref(v___y_2238_);
return v_res_2245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7(lean_object* v_00_u03b1_2246_, lean_object* v_mctx_2247_, lean_object* v_x_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_){
_start:
{
lean_object* v___x_2256_; 
v___x_2256_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg(v_mctx_2247_, v_x_2248_, v___y_2249_, v___y_2250_, v___y_2251_, v___y_2252_, v___y_2253_, v___y_2254_);
return v___x_2256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___boxed(lean_object* v_00_u03b1_2257_, lean_object* v_mctx_2258_, lean_object* v_x_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_){
_start:
{
lean_object* v_res_2267_; 
v_res_2267_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7(v_00_u03b1_2257_, v_mctx_2258_, v_x_2259_, v___y_2260_, v___y_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v___y_2263_);
lean_dec_ref(v___y_2262_);
lean_dec(v___y_2261_);
lean_dec_ref(v___y_2260_);
return v_res_2267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0(lean_object* v_x_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_){
_start:
{
lean_object* v___x_2276_; lean_object* v___x_2277_; 
v___x_2276_ = lean_box(0);
v___x_2277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2277_, 0, v___x_2276_);
return v___x_2277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0___boxed(lean_object* v_x_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_){
_start:
{
lean_object* v_res_2286_; 
v_res_2286_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0(v_x_2278_, v___y_2279_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_, v___y_2284_);
lean_dec(v___y_2284_);
lean_dec_ref(v___y_2283_);
lean_dec(v___y_2282_);
lean_dec_ref(v___y_2281_);
lean_dec(v___y_2280_);
lean_dec_ref(v___y_2279_);
lean_dec_ref(v_x_2278_);
return v_res_2286_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12(void){
_start:
{
lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; 
v___x_2313_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__11));
v___x_2314_ = lean_unsigned_to_nat(2u);
v___x_2315_ = lean_mk_empty_array_with_capacity(v___x_2314_);
v___x_2316_ = lean_array_push(v___x_2315_, v___x_2313_);
return v___x_2316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg(lean_object* v_as_2317_, size_t v_sz_2318_, size_t v_i_2319_, lean_object* v_b_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_){
_start:
{
uint8_t v___x_2326_; 
v___x_2326_ = lean_usize_dec_lt(v_i_2319_, v_sz_2318_);
if (v___x_2326_ == 0)
{
lean_object* v___x_2327_; 
v___x_2327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2327_, 0, v_b_2320_);
return v___x_2327_;
}
else
{
lean_object* v_a_2328_; lean_object* v___x_2329_; 
v_a_2328_ = lean_array_uget_borrowed(v_as_2317_, v_i_2319_);
lean_inc(v_a_2328_);
v___x_2329_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_a_2328_, v___y_2321_, v___y_2322_, v___y_2323_, v___y_2324_);
if (lean_obj_tag(v___x_2329_) == 0)
{
lean_object* v_a_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; size_t v___x_2337_; size_t v___x_2338_; 
v_a_2330_ = lean_ctor_get(v___x_2329_, 0);
lean_inc(v_a_2330_);
lean_dec_ref_known(v___x_2329_, 1);
v___x_2331_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0));
v___x_2332_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__1));
v___x_2333_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__12);
v___x_2334_ = lean_array_push(v___x_2333_, v_a_2330_);
v___x_2335_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2335_, 0, v___x_2331_);
lean_ctor_set(v___x_2335_, 1, v___x_2332_);
lean_ctor_set(v___x_2335_, 2, v___x_2334_);
v___x_2336_ = lean_array_push(v_b_2320_, v___x_2335_);
v___x_2337_ = ((size_t)1ULL);
v___x_2338_ = lean_usize_add(v_i_2319_, v___x_2337_);
v_i_2319_ = v___x_2338_;
v_b_2320_ = v___x_2336_;
goto _start;
}
else
{
lean_object* v_a_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2347_; 
lean_dec_ref(v_b_2320_);
v_a_2340_ = lean_ctor_get(v___x_2329_, 0);
v_isSharedCheck_2347_ = !lean_is_exclusive(v___x_2329_);
if (v_isSharedCheck_2347_ == 0)
{
v___x_2342_ = v___x_2329_;
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___x_2329_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2345_; 
if (v_isShared_2343_ == 0)
{
v___x_2345_ = v___x_2342_;
goto v_reusejp_2344_;
}
else
{
lean_object* v_reuseFailAlloc_2346_; 
v_reuseFailAlloc_2346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2346_, 0, v_a_2340_);
v___x_2345_ = v_reuseFailAlloc_2346_;
goto v_reusejp_2344_;
}
v_reusejp_2344_:
{
return v___x_2345_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___boxed(lean_object* v_as_2348_, lean_object* v_sz_2349_, lean_object* v_i_2350_, lean_object* v_b_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
size_t v_sz_boxed_2357_; size_t v_i_boxed_2358_; lean_object* v_res_2359_; 
v_sz_boxed_2357_ = lean_unbox_usize(v_sz_2349_);
lean_dec(v_sz_2349_);
v_i_boxed_2358_ = lean_unbox_usize(v_i_2350_);
lean_dec(v_i_2350_);
v_res_2359_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg(v_as_2348_, v_sz_boxed_2357_, v_i_boxed_2358_, v_b_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
lean_dec(v___y_2355_);
lean_dec_ref(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
lean_dec_ref(v_as_2348_);
return v_res_2359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2(lean_object* v_a_2360_, lean_object* v_val_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_){
_start:
{
lean_object* v___x_2369_; 
v___x_2369_ = l_Lean_Meta_isExprDefEq(v_a_2360_, v_val_2361_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_);
return v___x_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2___boxed(lean_object* v_a_2370_, lean_object* v_val_2371_, lean_object* v___y_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_){
_start:
{
lean_object* v_res_2379_; 
v_res_2379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2(v_a_2370_, v_val_2371_, v___y_2372_, v___y_2373_, v___y_2374_, v___y_2375_, v___y_2376_, v___y_2377_);
lean_dec(v___y_2377_);
lean_dec_ref(v___y_2376_);
lean_dec(v___y_2375_);
lean_dec_ref(v___y_2374_);
lean_dec(v___y_2373_);
lean_dec_ref(v___y_2372_);
return v_res_2379_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg(lean_object* v_keys_2380_, lean_object* v_i_2381_, lean_object* v_k_2382_){
_start:
{
lean_object* v___x_2383_; uint8_t v___x_2384_; 
v___x_2383_ = lean_array_get_size(v_keys_2380_);
v___x_2384_ = lean_nat_dec_lt(v_i_2381_, v___x_2383_);
if (v___x_2384_ == 0)
{
lean_dec(v_i_2381_);
return v___x_2384_;
}
else
{
lean_object* v_k_x27_2385_; uint8_t v___x_2386_; 
v_k_x27_2385_ = lean_array_fget_borrowed(v_keys_2380_, v_i_2381_);
v___x_2386_ = l_Lean_instBEqMVarId_beq(v_k_2382_, v_k_x27_2385_);
if (v___x_2386_ == 0)
{
lean_object* v___x_2387_; lean_object* v___x_2388_; 
v___x_2387_ = lean_unsigned_to_nat(1u);
v___x_2388_ = lean_nat_add(v_i_2381_, v___x_2387_);
lean_dec(v_i_2381_);
v_i_2381_ = v___x_2388_;
goto _start;
}
else
{
lean_dec(v_i_2381_);
return v___x_2386_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg___boxed(lean_object* v_keys_2390_, lean_object* v_i_2391_, lean_object* v_k_2392_){
_start:
{
uint8_t v_res_2393_; lean_object* v_r_2394_; 
v_res_2393_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg(v_keys_2390_, v_i_2391_, v_k_2392_);
lean_dec(v_k_2392_);
lean_dec_ref(v_keys_2390_);
v_r_2394_ = lean_box(v_res_2393_);
return v_r_2394_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg(lean_object* v_x_2395_, size_t v_x_2396_, lean_object* v_x_2397_){
_start:
{
if (lean_obj_tag(v_x_2395_) == 0)
{
lean_object* v_es_2398_; lean_object* v___x_2399_; size_t v___x_2400_; size_t v___x_2401_; lean_object* v_j_2402_; lean_object* v___x_2403_; 
v_es_2398_ = lean_ctor_get(v_x_2395_, 0);
v___x_2399_ = lean_box(2);
v___x_2400_ = ((size_t)31ULL);
v___x_2401_ = lean_usize_land(v_x_2396_, v___x_2400_);
v_j_2402_ = lean_usize_to_nat(v___x_2401_);
v___x_2403_ = lean_array_get_borrowed(v___x_2399_, v_es_2398_, v_j_2402_);
lean_dec(v_j_2402_);
switch(lean_obj_tag(v___x_2403_))
{
case 0:
{
lean_object* v_key_2404_; uint8_t v___x_2405_; 
v_key_2404_ = lean_ctor_get(v___x_2403_, 0);
v___x_2405_ = l_Lean_instBEqMVarId_beq(v_x_2397_, v_key_2404_);
return v___x_2405_;
}
case 1:
{
lean_object* v_node_2406_; size_t v___x_2407_; size_t v___x_2408_; 
v_node_2406_ = lean_ctor_get(v___x_2403_, 0);
v___x_2407_ = ((size_t)5ULL);
v___x_2408_ = lean_usize_shift_right(v_x_2396_, v___x_2407_);
v_x_2395_ = v_node_2406_;
v_x_2396_ = v___x_2408_;
goto _start;
}
default: 
{
uint8_t v___x_2410_; 
v___x_2410_ = 0;
return v___x_2410_;
}
}
}
else
{
lean_object* v_ks_2411_; lean_object* v___x_2412_; uint8_t v___x_2413_; 
v_ks_2411_ = lean_ctor_get(v_x_2395_, 0);
v___x_2412_ = lean_unsigned_to_nat(0u);
v___x_2413_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg(v_ks_2411_, v___x_2412_, v_x_2397_);
return v___x_2413_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_x_2414_, lean_object* v_x_2415_, lean_object* v_x_2416_){
_start:
{
size_t v_x_96584__boxed_2417_; uint8_t v_res_2418_; lean_object* v_r_2419_; 
v_x_96584__boxed_2417_ = lean_unbox_usize(v_x_2415_);
lean_dec(v_x_2415_);
v_res_2418_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg(v_x_2414_, v_x_96584__boxed_2417_, v_x_2416_);
lean_dec(v_x_2416_);
lean_dec_ref(v_x_2414_);
v_r_2419_ = lean_box(v_res_2418_);
return v_r_2419_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg(lean_object* v_x_2420_, lean_object* v_x_2421_){
_start:
{
uint64_t v___x_2422_; size_t v___x_2423_; uint8_t v___x_2424_; 
v___x_2422_ = l_Lean_instHashableMVarId_hash(v_x_2421_);
v___x_2423_ = lean_uint64_to_usize(v___x_2422_);
v___x_2424_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg(v_x_2420_, v___x_2423_, v_x_2421_);
return v___x_2424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg___boxed(lean_object* v_x_2425_, lean_object* v_x_2426_){
_start:
{
uint8_t v_res_2427_; lean_object* v_r_2428_; 
v_res_2427_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg(v_x_2425_, v_x_2426_);
lean_dec(v_x_2426_);
lean_dec_ref(v_x_2425_);
v_r_2428_ = lean_box(v_res_2427_);
return v_r_2428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg(lean_object* v_mvarId_2429_, lean_object* v___y_2430_){
_start:
{
lean_object* v___x_2432_; lean_object* v_mctx_2433_; lean_object* v_eAssignment_2434_; uint8_t v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; 
v___x_2432_ = lean_st_ref_get(v___y_2430_);
v_mctx_2433_ = lean_ctor_get(v___x_2432_, 0);
lean_inc_ref(v_mctx_2433_);
lean_dec(v___x_2432_);
v_eAssignment_2434_ = lean_ctor_get(v_mctx_2433_, 8);
lean_inc_ref(v_eAssignment_2434_);
lean_dec_ref(v_mctx_2433_);
v___x_2435_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg(v_eAssignment_2434_, v_mvarId_2429_);
lean_dec_ref(v_eAssignment_2434_);
v___x_2436_ = lean_box(v___x_2435_);
v___x_2437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2437_, 0, v___x_2436_);
return v___x_2437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg___boxed(lean_object* v_mvarId_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_){
_start:
{
lean_object* v_res_2441_; 
v_res_2441_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg(v_mvarId_2438_, v___y_2439_);
lean_dec(v___y_2439_);
lean_dec(v_mvarId_2438_);
return v_res_2441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8(lean_object* v_as_2442_, size_t v_i_2443_, size_t v_stop_2444_, lean_object* v_b_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_){
_start:
{
lean_object* v_a_2454_; uint8_t v___x_2458_; 
v___x_2458_ = lean_usize_dec_eq(v_i_2443_, v_stop_2444_);
if (v___x_2458_ == 0)
{
lean_object* v___x_2459_; lean_object* v___x_2462_; 
v___x_2459_ = lean_array_uget_borrowed(v_as_2442_, v_i_2443_);
v___x_2462_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg(v___x_2459_, v___y_2449_);
if (lean_obj_tag(v___x_2462_) == 0)
{
lean_object* v_a_2463_; uint8_t v___x_2464_; 
v_a_2463_ = lean_ctor_get(v___x_2462_, 0);
lean_inc(v_a_2463_);
lean_dec_ref_known(v___x_2462_, 1);
v___x_2464_ = lean_unbox(v_a_2463_);
lean_dec(v_a_2463_);
if (v___x_2464_ == 0)
{
goto v___jp_2460_;
}
else
{
v_a_2454_ = v_b_2445_;
goto v___jp_2453_;
}
}
else
{
if (lean_obj_tag(v___x_2462_) == 0)
{
lean_object* v_a_2465_; uint8_t v___x_2466_; 
v_a_2465_ = lean_ctor_get(v___x_2462_, 0);
lean_inc(v_a_2465_);
lean_dec_ref_known(v___x_2462_, 1);
v___x_2466_ = lean_unbox(v_a_2465_);
lean_dec(v_a_2465_);
if (v___x_2466_ == 0)
{
v_a_2454_ = v_b_2445_;
goto v___jp_2453_;
}
else
{
goto v___jp_2460_;
}
}
else
{
lean_object* v_a_2467_; lean_object* v___x_2469_; uint8_t v_isShared_2470_; uint8_t v_isSharedCheck_2474_; 
lean_dec_ref(v_b_2445_);
v_a_2467_ = lean_ctor_get(v___x_2462_, 0);
v_isSharedCheck_2474_ = !lean_is_exclusive(v___x_2462_);
if (v_isSharedCheck_2474_ == 0)
{
v___x_2469_ = v___x_2462_;
v_isShared_2470_ = v_isSharedCheck_2474_;
goto v_resetjp_2468_;
}
else
{
lean_inc(v_a_2467_);
lean_dec(v___x_2462_);
v___x_2469_ = lean_box(0);
v_isShared_2470_ = v_isSharedCheck_2474_;
goto v_resetjp_2468_;
}
v_resetjp_2468_:
{
lean_object* v___x_2472_; 
if (v_isShared_2470_ == 0)
{
v___x_2472_ = v___x_2469_;
goto v_reusejp_2471_;
}
else
{
lean_object* v_reuseFailAlloc_2473_; 
v_reuseFailAlloc_2473_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2473_, 0, v_a_2467_);
v___x_2472_ = v_reuseFailAlloc_2473_;
goto v_reusejp_2471_;
}
v_reusejp_2471_:
{
return v___x_2472_;
}
}
}
}
v___jp_2460_:
{
lean_object* v___x_2461_; 
lean_inc(v___x_2459_);
v___x_2461_ = lean_array_push(v_b_2445_, v___x_2459_);
v_a_2454_ = v___x_2461_;
goto v___jp_2453_;
}
}
else
{
lean_object* v___x_2475_; 
v___x_2475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2475_, 0, v_b_2445_);
return v___x_2475_;
}
v___jp_2453_:
{
size_t v___x_2455_; size_t v___x_2456_; 
v___x_2455_ = ((size_t)1ULL);
v___x_2456_ = lean_usize_add(v_i_2443_, v___x_2455_);
v_i_2443_ = v___x_2456_;
v_b_2445_ = v_a_2454_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8___boxed(lean_object* v_as_2476_, lean_object* v_i_2477_, lean_object* v_stop_2478_, lean_object* v_b_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_){
_start:
{
size_t v_i_boxed_2487_; size_t v_stop_boxed_2488_; lean_object* v_res_2489_; 
v_i_boxed_2487_ = lean_unbox_usize(v_i_2477_);
lean_dec(v_i_2477_);
v_stop_boxed_2488_ = lean_unbox_usize(v_stop_2478_);
lean_dec(v_stop_2478_);
v_res_2489_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8(v_as_2476_, v_i_boxed_2487_, v_stop_boxed_2488_, v_b_2479_, v___y_2480_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_);
lean_dec(v___y_2485_);
lean_dec_ref(v___y_2484_);
lean_dec(v___y_2483_);
lean_dec_ref(v___y_2482_);
lean_dec(v___y_2481_);
lean_dec_ref(v___y_2480_);
lean_dec_ref(v_as_2476_);
return v_res_2489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3(size_t v_sz_2490_, size_t v_i_2491_, lean_object* v_bs_2492_){
_start:
{
uint8_t v___x_2493_; 
v___x_2493_ = lean_usize_dec_lt(v_i_2491_, v_sz_2490_);
if (v___x_2493_ == 0)
{
return v_bs_2492_;
}
else
{
lean_object* v_v_2494_; lean_object* v___x_2495_; lean_object* v_bs_x27_2496_; lean_object* v___x_2497_; size_t v___x_2498_; size_t v___x_2499_; lean_object* v___x_2500_; 
v_v_2494_ = lean_array_uget(v_bs_2492_, v_i_2491_);
v___x_2495_ = lean_unsigned_to_nat(0u);
v_bs_x27_2496_ = lean_array_uset(v_bs_2492_, v_i_2491_, v___x_2495_);
v___x_2497_ = l_Lean_Expr_mvarId_x21(v_v_2494_);
lean_dec(v_v_2494_);
v___x_2498_ = ((size_t)1ULL);
v___x_2499_ = lean_usize_add(v_i_2491_, v___x_2498_);
v___x_2500_ = lean_array_uset(v_bs_x27_2496_, v_i_2491_, v___x_2497_);
v_i_2491_ = v___x_2499_;
v_bs_2492_ = v___x_2500_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3___boxed(lean_object* v_sz_2502_, lean_object* v_i_2503_, lean_object* v_bs_2504_){
_start:
{
size_t v_sz_boxed_2505_; size_t v_i_boxed_2506_; lean_object* v_res_2507_; 
v_sz_boxed_2505_ = lean_unbox_usize(v_sz_2502_);
lean_dec(v_sz_2502_);
v_i_boxed_2506_ = lean_unbox_usize(v_i_2503_);
lean_dec(v_i_2503_);
v_res_2507_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3(v_sz_boxed_2505_, v_i_boxed_2506_, v_bs_2504_);
return v_res_2507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4(size_t v_sz_2508_, size_t v_i_2509_, lean_object* v_bs_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_){
_start:
{
uint8_t v___x_2518_; 
v___x_2518_ = lean_usize_dec_lt(v_i_2509_, v_sz_2508_);
if (v___x_2518_ == 0)
{
lean_object* v___x_2519_; 
v___x_2519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2519_, 0, v_bs_2510_);
return v___x_2519_;
}
else
{
lean_object* v_v_2520_; lean_object* v___x_2521_; lean_object* v_bs_x27_2522_; lean_object* v___y_2524_; lean_object* v___x_2538_; 
v_v_2520_ = lean_array_uget(v_bs_2510_, v_i_2509_);
v___x_2521_ = lean_unsigned_to_nat(0u);
v_bs_x27_2522_ = lean_array_uset(v_bs_2510_, v_i_2509_, v___x_2521_);
v___x_2538_ = l_Lean_MVarId_getType(v_v_2520_, v___y_2513_, v___y_2514_, v___y_2515_, v___y_2516_);
if (lean_obj_tag(v___x_2538_) == 0)
{
lean_object* v_a_2539_; lean_object* v___x_2540_; 
v_a_2539_ = lean_ctor_get(v___x_2538_, 0);
lean_inc(v_a_2539_);
lean_dec_ref_known(v___x_2538_, 1);
v___x_2540_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_a_2539_, v___y_2514_);
v___y_2524_ = v___x_2540_;
goto v___jp_2523_;
}
else
{
v___y_2524_ = v___x_2538_;
goto v___jp_2523_;
}
v___jp_2523_:
{
if (lean_obj_tag(v___y_2524_) == 0)
{
lean_object* v_a_2525_; size_t v___x_2526_; size_t v___x_2527_; lean_object* v___x_2528_; 
v_a_2525_ = lean_ctor_get(v___y_2524_, 0);
lean_inc(v_a_2525_);
lean_dec_ref_known(v___y_2524_, 1);
v___x_2526_ = ((size_t)1ULL);
v___x_2527_ = lean_usize_add(v_i_2509_, v___x_2526_);
v___x_2528_ = lean_array_uset(v_bs_x27_2522_, v_i_2509_, v_a_2525_);
v_i_2509_ = v___x_2527_;
v_bs_2510_ = v___x_2528_;
goto _start;
}
else
{
lean_object* v_a_2530_; lean_object* v___x_2532_; uint8_t v_isShared_2533_; uint8_t v_isSharedCheck_2537_; 
lean_dec_ref(v_bs_x27_2522_);
v_a_2530_ = lean_ctor_get(v___y_2524_, 0);
v_isSharedCheck_2537_ = !lean_is_exclusive(v___y_2524_);
if (v_isSharedCheck_2537_ == 0)
{
v___x_2532_ = v___y_2524_;
v_isShared_2533_ = v_isSharedCheck_2537_;
goto v_resetjp_2531_;
}
else
{
lean_inc(v_a_2530_);
lean_dec(v___y_2524_);
v___x_2532_ = lean_box(0);
v_isShared_2533_ = v_isSharedCheck_2537_;
goto v_resetjp_2531_;
}
v_resetjp_2531_:
{
lean_object* v___x_2535_; 
if (v_isShared_2533_ == 0)
{
v___x_2535_ = v___x_2532_;
goto v_reusejp_2534_;
}
else
{
lean_object* v_reuseFailAlloc_2536_; 
v_reuseFailAlloc_2536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2536_, 0, v_a_2530_);
v___x_2535_ = v_reuseFailAlloc_2536_;
goto v_reusejp_2534_;
}
v_reusejp_2534_:
{
return v___x_2535_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4___boxed(lean_object* v_sz_2541_, lean_object* v_i_2542_, lean_object* v_bs_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_){
_start:
{
size_t v_sz_boxed_2551_; size_t v_i_boxed_2552_; lean_object* v_res_2553_; 
v_sz_boxed_2551_ = lean_unbox_usize(v_sz_2541_);
lean_dec(v_sz_2541_);
v_i_boxed_2552_ = lean_unbox_usize(v_i_2542_);
lean_dec(v_i_2542_);
v_res_2553_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4(v_sz_boxed_2551_, v_i_boxed_2552_, v_bs_2543_, v___y_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_, v___y_2549_);
lean_dec(v___y_2549_);
lean_dec_ref(v___y_2548_);
lean_dec(v___y_2547_);
lean_dec_ref(v___y_2546_);
lean_dec(v___y_2545_);
lean_dec_ref(v___y_2544_);
return v_res_2553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(lean_object* v_msg_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_){
_start:
{
lean_object* v_ref_2560_; lean_object* v___x_2561_; lean_object* v_a_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2570_; 
v_ref_2560_ = lean_ctor_get(v___y_2557_, 5);
v___x_2561_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_gcongrBackward_spec__3_spec__4(v_msg_2554_, v___y_2555_, v___y_2556_, v___y_2557_, v___y_2558_);
v_a_2562_ = lean_ctor_get(v___x_2561_, 0);
v_isSharedCheck_2570_ = !lean_is_exclusive(v___x_2561_);
if (v_isSharedCheck_2570_ == 0)
{
v___x_2564_ = v___x_2561_;
v_isShared_2565_ = v_isSharedCheck_2570_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_a_2562_);
lean_dec(v___x_2561_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2570_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2566_; lean_object* v___x_2568_; 
lean_inc(v_ref_2560_);
v___x_2566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2566_, 0, v_ref_2560_);
lean_ctor_set(v___x_2566_, 1, v_a_2562_);
if (v_isShared_2565_ == 0)
{
lean_ctor_set_tag(v___x_2564_, 1);
lean_ctor_set(v___x_2564_, 0, v___x_2566_);
v___x_2568_ = v___x_2564_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2569_; 
v_reuseFailAlloc_2569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2569_, 0, v___x_2566_);
v___x_2568_ = v_reuseFailAlloc_2569_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
return v___x_2568_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg___boxed(lean_object* v_msg_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_){
_start:
{
lean_object* v_res_2577_; 
v_res_2577_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v_msg_2571_, v___y_2572_, v___y_2573_, v___y_2574_, v___y_2575_);
lean_dec(v___y_2575_);
lean_dec_ref(v___y_2574_);
lean_dec(v___y_2573_);
lean_dec_ref(v___y_2572_);
return v_res_2577_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1(void){
_start:
{
lean_object* v___x_2579_; lean_object* v___x_2580_; 
v___x_2579_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__0));
v___x_2580_ = l_Lean_stringToMessageData(v___x_2579_);
return v___x_2580_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6(void){
_start:
{
lean_object* v___x_2587_; lean_object* v___x_2588_; 
v___x_2587_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__5));
v___x_2588_ = l_Lean_stringToMessageData(v___x_2587_);
return v___x_2588_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8(void){
_start:
{
lean_object* v___x_2590_; lean_object* v___x_2591_; 
v___x_2590_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__7));
v___x_2591_ = l_Lean_stringToMessageData(v___x_2590_);
return v___x_2591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3(lean_object* v_fst_2592_, lean_object* v_name_2593_, uint8_t v_symm_2594_, lean_object* v_snd_2595_, lean_object* v_assignableMVars_2596_, lean_object* v_fst_2597_, lean_object* v_subExpr_2598_, uint8_t v_a_2599_, lean_object* v_lem_2600_, lean_object* v_i_2601_, lean_object* v_rflTarget_x3f_2602_, lean_object* v_rwKind_2603_, lean_object* v_mctx_2604_, lean_object* v___f_2605_, lean_object* v_fst_2606_, lean_object* v_fst_2607_, lean_object* v_____r_2608_, lean_object* v___y_2609_, lean_object* v___y_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_, lean_object* v___y_2613_, lean_object* v___y_2614_){
_start:
{
lean_object* v___y_2617_; lean_object* v___y_2618_; lean_object* v___y_2619_; lean_object* v_pattern_2620_; lean_object* v___y_2624_; lean_object* v___y_2625_; lean_object* v___y_2626_; lean_object* v___y_2627_; lean_object* v___y_2628_; lean_object* v___y_2629_; lean_object* v___y_2630_; lean_object* v___y_2631_; lean_object* v___y_2632_; lean_object* v___y_2633_; lean_object* v___y_2648_; lean_object* v___y_2649_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2652_; lean_object* v___y_2653_; lean_object* v___y_2654_; lean_object* v___y_2655_; lean_object* v___y_2667_; uint8_t v___y_2668_; lean_object* v___y_2669_; lean_object* v___y_2670_; lean_object* v___y_2671_; lean_object* v___y_2672_; lean_object* v_filtered_2673_; lean_object* v___y_2674_; lean_object* v___y_2675_; lean_object* v___y_2676_; lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v___y_2679_; uint8_t v___y_2745_; lean_object* v___y_2746_; lean_object* v___y_2747_; lean_object* v___y_2748_; lean_object* v___y_2749_; lean_object* v___y_2750_; lean_object* v___y_2751_; lean_object* v___y_2752_; lean_object* v___y_2753_; lean_object* v___y_2754_; lean_object* v___y_2755_; lean_object* v___y_2756_; lean_object* v___y_2759_; uint8_t v___y_2760_; uint8_t v___y_2761_; lean_object* v___y_2762_; lean_object* v___y_2763_; uint8_t v___y_2764_; size_t v___y_2765_; lean_object* v___y_2766_; lean_object* v___y_2767_; lean_object* v___y_2768_; lean_object* v___y_2769_; lean_object* v___y_2770_; lean_object* v___y_2771_; lean_object* v___y_2772_; lean_object* v___y_2773_; lean_object* v___y_2815_; lean_object* v___y_2816_; lean_object* v___y_2817_; lean_object* v___y_2818_; uint8_t v___y_2819_; size_t v___y_2820_; lean_object* v___y_2821_; lean_object* v___y_2822_; lean_object* v___y_2823_; uint8_t v___y_2824_; lean_object* v___y_2825_; lean_object* v___y_2826_; lean_object* v___y_2827_; lean_object* v___y_2828_; uint8_t v_a_2829_; uint8_t v___y_2840_; lean_object* v___y_2841_; lean_object* v___y_2842_; uint8_t v___y_2843_; uint8_t v___y_2844_; size_t v___y_2845_; lean_object* v___y_2846_; lean_object* v___y_2847_; lean_object* v___y_2848_; uint8_t v_justLemmaName_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; lean_object* v___y_2852_; lean_object* v___y_2853_; lean_object* v___y_2854_; lean_object* v___y_2855_; lean_object* v___y_2908_; lean_object* v___y_2909_; lean_object* v___y_2910_; lean_object* v___y_2911_; lean_object* v___y_2912_; uint8_t v___y_2913_; lean_object* v___y_2914_; lean_object* v___y_2915_; size_t v___y_2916_; lean_object* v_a_2917_; lean_object* v___y_2987_; lean_object* v___y_2988_; lean_object* v___y_2989_; lean_object* v___y_2990_; lean_object* v___y_2991_; uint8_t v___y_2992_; lean_object* v___y_2993_; size_t v___y_2994_; lean_object* v___y_2995_; lean_object* v___y_2996_; lean_object* v___y_3007_; lean_object* v___y_3008_; lean_object* v___y_3009_; lean_object* v___y_3010_; lean_object* v___y_3011_; lean_object* v___y_3012_; lean_object* v___x_3037_; 
v___x_3037_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_fst_2592_, v___y_2612_);
if (lean_obj_tag(v___x_3037_) == 0)
{
lean_object* v_a_3038_; lean_object* v___x_3056_; lean_object* v___x_3057_; uint8_t v___x_3058_; 
v_a_3038_ = lean_ctor_get(v___x_3037_, 0);
lean_inc_n(v_a_3038_, 2);
lean_dec_ref_known(v___x_3037_, 1);
v___x_3056_ = l_Lean_Expr_toHeadIndex(v_a_3038_);
lean_inc_ref(v_subExpr_2598_);
v___x_3057_ = l_Lean_Expr_toHeadIndex(v_subExpr_2598_);
v___x_3058_ = l_Lean_instBEqHeadIndex_beq(v___x_3056_, v___x_3057_);
lean_dec(v___x_3057_);
lean_dec(v___x_3056_);
if (v___x_3058_ == 0)
{
goto v___jp_3039_;
}
else
{
lean_object* v___x_3059_; lean_object* v___x_3060_; uint8_t v___x_3061_; 
v___x_3059_ = l_Lean_Expr_headNumArgs(v_a_3038_);
v___x_3060_ = l_Lean_Expr_headNumArgs(v_subExpr_2598_);
v___x_3061_ = lean_nat_dec_eq(v___x_3059_, v___x_3060_);
lean_dec(v___x_3060_);
lean_dec(v___x_3059_);
if (v___x_3061_ == 0)
{
goto v___jp_3039_;
}
else
{
lean_dec(v_a_3038_);
v___y_3007_ = v___y_2609_;
v___y_3008_ = v___y_2610_;
v___y_3009_ = v___y_2611_;
v___y_3010_ = v___y_2612_;
v___y_3011_ = v___y_2613_;
v___y_3012_ = v___y_2614_;
goto v___jp_3006_;
}
}
v___jp_3039_:
{
lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; 
v___x_3040_ = l_Lean_MessageData_ofExpr(v_a_3038_);
v___x_3041_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__6);
v___x_3042_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3042_, 0, v___x_3040_);
lean_ctor_set(v___x_3042_, 1, v___x_3041_);
lean_inc_ref(v_subExpr_2598_);
v___x_3043_ = l_Lean_MessageData_ofExpr(v_subExpr_2598_);
v___x_3044_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3044_, 0, v___x_3042_);
lean_ctor_set(v___x_3044_, 1, v___x_3043_);
v___x_3045_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__8);
v___x_3046_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3046_, 0, v___x_3044_);
lean_ctor_set(v___x_3046_, 1, v___x_3045_);
v___x_3047_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v___x_3046_, v___y_2611_, v___y_2612_, v___y_2613_, v___y_2614_);
if (lean_obj_tag(v___x_3047_) == 0)
{
lean_dec_ref_known(v___x_3047_, 1);
v___y_3007_ = v___y_2609_;
v___y_3008_ = v___y_2610_;
v___y_3009_ = v___y_2611_;
v___y_3010_ = v___y_2612_;
v___y_3011_ = v___y_2613_;
v___y_3012_ = v___y_2614_;
goto v___jp_3006_;
}
else
{
lean_object* v_a_3048_; lean_object* v___x_3050_; uint8_t v_isShared_3051_; uint8_t v_isSharedCheck_3055_; 
lean_dec_ref(v_fst_2607_);
lean_dec_ref(v_fst_2606_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_snd_2595_);
lean_dec_ref(v_name_2593_);
v_a_3048_ = lean_ctor_get(v___x_3047_, 0);
v_isSharedCheck_3055_ = !lean_is_exclusive(v___x_3047_);
if (v_isSharedCheck_3055_ == 0)
{
v___x_3050_ = v___x_3047_;
v_isShared_3051_ = v_isSharedCheck_3055_;
goto v_resetjp_3049_;
}
else
{
lean_inc(v_a_3048_);
lean_dec(v___x_3047_);
v___x_3050_ = lean_box(0);
v_isShared_3051_ = v_isSharedCheck_3055_;
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
lean_object* v_reuseFailAlloc_3054_; 
v_reuseFailAlloc_3054_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3054_, 0, v_a_3048_);
v___x_3053_ = v_reuseFailAlloc_3054_;
goto v_reusejp_3052_;
}
v_reusejp_3052_:
{
return v___x_3053_;
}
}
}
}
}
else
{
lean_object* v_a_3062_; lean_object* v___x_3064_; uint8_t v_isShared_3065_; uint8_t v_isSharedCheck_3069_; 
lean_dec_ref(v_fst_2607_);
lean_dec_ref(v_fst_2606_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_snd_2595_);
lean_dec_ref(v_name_2593_);
v_a_3062_ = lean_ctor_get(v___x_3037_, 0);
v_isSharedCheck_3069_ = !lean_is_exclusive(v___x_3037_);
if (v_isSharedCheck_3069_ == 0)
{
v___x_3064_ = v___x_3037_;
v_isShared_3065_ = v_isSharedCheck_3069_;
goto v_resetjp_3063_;
}
else
{
lean_inc(v_a_3062_);
lean_dec(v___x_3037_);
v___x_3064_ = lean_box(0);
v_isShared_3065_ = v_isSharedCheck_3069_;
goto v_resetjp_3063_;
}
v_resetjp_3063_:
{
lean_object* v___x_3067_; 
if (v_isShared_3065_ == 0)
{
v___x_3067_ = v___x_3064_;
goto v_reusejp_3066_;
}
else
{
lean_object* v_reuseFailAlloc_3068_; 
v_reuseFailAlloc_3068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3068_, 0, v_a_3062_);
v___x_3067_ = v_reuseFailAlloc_3068_;
goto v_reusejp_3066_;
}
v_reusejp_3066_:
{
return v___x_3067_;
}
}
}
v___jp_2616_:
{
lean_object* v___x_2621_; lean_object* v___x_2622_; 
v___x_2621_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2621_, 0, v___y_2619_);
lean_ctor_set(v___x_2621_, 1, v___y_2617_);
lean_ctor_set(v___x_2621_, 2, v___y_2618_);
lean_ctor_set(v___x_2621_, 3, v_pattern_2620_);
v___x_2622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2622_, 0, v___x_2621_);
return v___x_2622_;
}
v___jp_2623_:
{
lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; 
v___x_2634_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__1);
v___x_2635_ = l_Lean_indentExpr(v___y_2626_);
v___x_2636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2636_, 0, v___x_2634_);
lean_ctor_set(v___x_2636_, 1, v___x_2635_);
v___x_2637_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v___x_2636_, v___y_2630_, v___y_2631_, v___y_2632_, v___y_2633_);
if (lean_obj_tag(v___x_2637_) == 0)
{
lean_object* v_a_2638_; 
v_a_2638_ = lean_ctor_get(v___x_2637_, 0);
lean_inc(v_a_2638_);
lean_dec_ref_known(v___x_2637_, 1);
v___y_2617_ = v___y_2624_;
v___y_2618_ = v___y_2625_;
v___y_2619_ = v___y_2627_;
v_pattern_2620_ = v_a_2638_;
goto v___jp_2616_;
}
else
{
lean_object* v_a_2639_; lean_object* v___x_2641_; uint8_t v_isShared_2642_; uint8_t v_isSharedCheck_2646_; 
lean_dec(v___y_2627_);
lean_dec_ref(v___y_2625_);
lean_dec_ref(v___y_2624_);
v_a_2639_ = lean_ctor_get(v___x_2637_, 0);
v_isSharedCheck_2646_ = !lean_is_exclusive(v___x_2637_);
if (v_isSharedCheck_2646_ == 0)
{
v___x_2641_ = v___x_2637_;
v_isShared_2642_ = v_isSharedCheck_2646_;
goto v_resetjp_2640_;
}
else
{
lean_inc(v_a_2639_);
lean_dec(v___x_2637_);
v___x_2641_ = lean_box(0);
v_isShared_2642_ = v_isSharedCheck_2646_;
goto v_resetjp_2640_;
}
v_resetjp_2640_:
{
lean_object* v___x_2644_; 
if (v_isShared_2642_ == 0)
{
v___x_2644_ = v___x_2641_;
goto v_reusejp_2643_;
}
else
{
lean_object* v_reuseFailAlloc_2645_; 
v_reuseFailAlloc_2645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2645_, 0, v_a_2639_);
v___x_2644_ = v_reuseFailAlloc_2645_;
goto v_reusejp_2643_;
}
v_reusejp_2643_:
{
return v___x_2644_;
}
}
}
}
v___jp_2647_:
{
lean_object* v___x_2656_; 
v___x_2656_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___y_2655_, v___y_2650_, v___y_2654_, v___y_2651_, v___y_2653_);
if (lean_obj_tag(v___x_2656_) == 0)
{
lean_object* v_a_2657_; 
v_a_2657_ = lean_ctor_get(v___x_2656_, 0);
lean_inc(v_a_2657_);
lean_dec_ref_known(v___x_2656_, 1);
v___y_2617_ = v___y_2648_;
v___y_2618_ = v___y_2649_;
v___y_2619_ = v___y_2652_;
v_pattern_2620_ = v_a_2657_;
goto v___jp_2616_;
}
else
{
lean_object* v_a_2658_; lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2665_; 
lean_dec(v___y_2652_);
lean_dec_ref(v___y_2649_);
lean_dec_ref(v___y_2648_);
v_a_2658_ = lean_ctor_get(v___x_2656_, 0);
v_isSharedCheck_2665_ = !lean_is_exclusive(v___x_2656_);
if (v_isSharedCheck_2665_ == 0)
{
v___x_2660_ = v___x_2656_;
v_isShared_2661_ = v_isSharedCheck_2665_;
goto v_resetjp_2659_;
}
else
{
lean_inc(v_a_2658_);
lean_dec(v___x_2656_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2665_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v___x_2663_; 
if (v_isShared_2661_ == 0)
{
v___x_2663_ = v___x_2660_;
goto v_reusejp_2662_;
}
else
{
lean_object* v_reuseFailAlloc_2664_; 
v_reuseFailAlloc_2664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2664_, 0, v_a_2658_);
v___x_2663_ = v_reuseFailAlloc_2664_;
goto v_reusejp_2662_;
}
v_reusejp_2662_:
{
return v___x_2663_;
}
}
}
}
v___jp_2666_:
{
lean_object* v___x_2680_; 
lean_inc_ref(v_name_2593_);
v___x_2680_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(v_name_2593_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2680_) == 0)
{
lean_object* v_a_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; 
v_a_2681_ = lean_ctor_get(v___x_2680_, 0);
lean_inc(v_a_2681_);
lean_dec_ref_known(v___x_2680_, 1);
v___x_2682_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0));
v___x_2683_ = lean_mk_empty_array_with_capacity(v___y_2667_);
lean_dec(v___y_2667_);
v___x_2684_ = lean_array_push(v___y_2672_, v_a_2681_);
lean_inc_ref(v___x_2683_);
v___x_2685_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2685_, 0, v___x_2682_);
lean_ctor_set(v___x_2685_, 1, v___x_2683_);
lean_ctor_set(v___x_2685_, 2, v___x_2684_);
v___x_2686_ = lean_array_push(v___y_2669_, v___x_2685_);
v___x_2687_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2682_);
lean_ctor_set(v___x_2687_, 1, v___x_2683_);
lean_ctor_set(v___x_2687_, 2, v___x_2686_);
v___x_2688_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_2671_, v___x_2687_, v___y_2668_, v___y_2674_, v___y_2675_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2688_) == 0)
{
lean_object* v_a_2689_; lean_object* v___x_2690_; 
v_a_2689_ = lean_ctor_get(v___x_2688_, 0);
lean_inc(v_a_2689_);
lean_dec_ref_known(v___x_2688_, 1);
v___x_2690_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(v_name_2593_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2690_) == 0)
{
lean_object* v_a_2691_; lean_object* v___x_2692_; uint8_t v___x_2693_; lean_object* v___x_2694_; 
v_a_2691_ = lean_ctor_get(v___x_2690_, 0);
lean_inc(v_a_2691_);
lean_dec_ref_known(v___x_2690_, 1);
v___x_2692_ = lean_box(0);
v___x_2693_ = 0;
v___x_2694_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_2691_, v___x_2692_, v___x_2693_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2694_) == 0)
{
lean_object* v_a_2695_; lean_object* v_snd_2696_; lean_object* v_snd_2697_; lean_object* v___x_2698_; 
v_a_2695_ = lean_ctor_get(v___x_2694_, 0);
lean_inc(v_a_2695_);
lean_dec_ref_known(v___x_2694_, 1);
v_snd_2696_ = lean_ctor_get(v_a_2695_, 1);
lean_inc(v_snd_2696_);
lean_dec(v_a_2695_);
v_snd_2697_ = lean_ctor_get(v_snd_2696_, 1);
lean_inc_n(v_snd_2697_, 2);
lean_dec(v_snd_2696_);
v___x_2698_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_snd_2697_, v___y_2677_);
if (lean_obj_tag(v___x_2698_) == 0)
{
lean_object* v_a_2699_; lean_object* v___x_2700_; 
v_a_2699_ = lean_ctor_get(v___x_2698_, 0);
lean_inc(v_a_2699_);
lean_dec_ref_known(v___x_2698_, 1);
v___x_2700_ = l_Lean_Expr_cleanupAnnotations(v_a_2699_);
if (lean_obj_tag(v___x_2700_) == 5)
{
lean_object* v_fn_2701_; 
v_fn_2701_ = lean_ctor_get(v___x_2700_, 0);
lean_inc_ref(v_fn_2701_);
if (lean_obj_tag(v_fn_2701_) == 5)
{
lean_dec(v_snd_2697_);
if (v_symm_2594_ == 0)
{
lean_object* v_arg_2702_; 
lean_dec_ref_known(v___x_2700_, 2);
v_arg_2702_ = lean_ctor_get(v_fn_2701_, 1);
lean_inc_ref(v_arg_2702_);
lean_dec_ref_known(v_fn_2701_, 2);
v___y_2648_ = v_a_2689_;
v___y_2649_ = v___y_2670_;
v___y_2650_ = v___y_2676_;
v___y_2651_ = v___y_2678_;
v___y_2652_ = v_filtered_2673_;
v___y_2653_ = v___y_2679_;
v___y_2654_ = v___y_2677_;
v___y_2655_ = v_arg_2702_;
goto v___jp_2647_;
}
else
{
lean_object* v_arg_2703_; 
lean_dec_ref_known(v_fn_2701_, 2);
v_arg_2703_ = lean_ctor_get(v___x_2700_, 1);
lean_inc_ref(v_arg_2703_);
lean_dec_ref_known(v___x_2700_, 2);
v___y_2648_ = v_a_2689_;
v___y_2649_ = v___y_2670_;
v___y_2650_ = v___y_2676_;
v___y_2651_ = v___y_2678_;
v___y_2652_ = v_filtered_2673_;
v___y_2653_ = v___y_2679_;
v___y_2654_ = v___y_2677_;
v___y_2655_ = v_arg_2703_;
goto v___jp_2647_;
}
}
else
{
lean_dec_ref_known(v___x_2700_, 2);
lean_dec_ref(v_fn_2701_);
v___y_2624_ = v_a_2689_;
v___y_2625_ = v___y_2670_;
v___y_2626_ = v_snd_2697_;
v___y_2627_ = v_filtered_2673_;
v___y_2628_ = v___y_2674_;
v___y_2629_ = v___y_2675_;
v___y_2630_ = v___y_2676_;
v___y_2631_ = v___y_2677_;
v___y_2632_ = v___y_2678_;
v___y_2633_ = v___y_2679_;
goto v___jp_2623_;
}
}
else
{
lean_dec_ref(v___x_2700_);
v___y_2624_ = v_a_2689_;
v___y_2625_ = v___y_2670_;
v___y_2626_ = v_snd_2697_;
v___y_2627_ = v_filtered_2673_;
v___y_2628_ = v___y_2674_;
v___y_2629_ = v___y_2675_;
v___y_2630_ = v___y_2676_;
v___y_2631_ = v___y_2677_;
v___y_2632_ = v___y_2678_;
v___y_2633_ = v___y_2679_;
goto v___jp_2623_;
}
}
else
{
lean_object* v_a_2704_; lean_object* v___x_2706_; uint8_t v_isShared_2707_; uint8_t v_isSharedCheck_2711_; 
lean_dec(v_snd_2697_);
lean_dec(v_a_2689_);
lean_dec(v_filtered_2673_);
lean_dec_ref(v___y_2670_);
v_a_2704_ = lean_ctor_get(v___x_2698_, 0);
v_isSharedCheck_2711_ = !lean_is_exclusive(v___x_2698_);
if (v_isSharedCheck_2711_ == 0)
{
v___x_2706_ = v___x_2698_;
v_isShared_2707_ = v_isSharedCheck_2711_;
goto v_resetjp_2705_;
}
else
{
lean_inc(v_a_2704_);
lean_dec(v___x_2698_);
v___x_2706_ = lean_box(0);
v_isShared_2707_ = v_isSharedCheck_2711_;
goto v_resetjp_2705_;
}
v_resetjp_2705_:
{
lean_object* v___x_2709_; 
if (v_isShared_2707_ == 0)
{
v___x_2709_ = v___x_2706_;
goto v_reusejp_2708_;
}
else
{
lean_object* v_reuseFailAlloc_2710_; 
v_reuseFailAlloc_2710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2710_, 0, v_a_2704_);
v___x_2709_ = v_reuseFailAlloc_2710_;
goto v_reusejp_2708_;
}
v_reusejp_2708_:
{
return v___x_2709_;
}
}
}
}
else
{
lean_object* v_a_2712_; lean_object* v___x_2714_; uint8_t v_isShared_2715_; uint8_t v_isSharedCheck_2719_; 
lean_dec(v_a_2689_);
lean_dec(v_filtered_2673_);
lean_dec_ref(v___y_2670_);
v_a_2712_ = lean_ctor_get(v___x_2694_, 0);
v_isSharedCheck_2719_ = !lean_is_exclusive(v___x_2694_);
if (v_isSharedCheck_2719_ == 0)
{
v___x_2714_ = v___x_2694_;
v_isShared_2715_ = v_isSharedCheck_2719_;
goto v_resetjp_2713_;
}
else
{
lean_inc(v_a_2712_);
lean_dec(v___x_2694_);
v___x_2714_ = lean_box(0);
v_isShared_2715_ = v_isSharedCheck_2719_;
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
lean_object* v_reuseFailAlloc_2718_; 
v_reuseFailAlloc_2718_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2718_, 0, v_a_2712_);
v___x_2717_ = v_reuseFailAlloc_2718_;
goto v_reusejp_2716_;
}
v_reusejp_2716_:
{
return v___x_2717_;
}
}
}
}
else
{
lean_object* v_a_2720_; lean_object* v___x_2722_; uint8_t v_isShared_2723_; uint8_t v_isSharedCheck_2727_; 
lean_dec(v_a_2689_);
lean_dec(v_filtered_2673_);
lean_dec_ref(v___y_2670_);
v_a_2720_ = lean_ctor_get(v___x_2690_, 0);
v_isSharedCheck_2727_ = !lean_is_exclusive(v___x_2690_);
if (v_isSharedCheck_2727_ == 0)
{
v___x_2722_ = v___x_2690_;
v_isShared_2723_ = v_isSharedCheck_2727_;
goto v_resetjp_2721_;
}
else
{
lean_inc(v_a_2720_);
lean_dec(v___x_2690_);
v___x_2722_ = lean_box(0);
v_isShared_2723_ = v_isSharedCheck_2727_;
goto v_resetjp_2721_;
}
v_resetjp_2721_:
{
lean_object* v___x_2725_; 
if (v_isShared_2723_ == 0)
{
v___x_2725_ = v___x_2722_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2726_; 
v_reuseFailAlloc_2726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2726_, 0, v_a_2720_);
v___x_2725_ = v_reuseFailAlloc_2726_;
goto v_reusejp_2724_;
}
v_reusejp_2724_:
{
return v___x_2725_;
}
}
}
}
else
{
lean_object* v_a_2728_; lean_object* v___x_2730_; uint8_t v_isShared_2731_; uint8_t v_isSharedCheck_2735_; 
lean_dec(v_filtered_2673_);
lean_dec_ref(v___y_2670_);
lean_dec_ref(v_name_2593_);
v_a_2728_ = lean_ctor_get(v___x_2688_, 0);
v_isSharedCheck_2735_ = !lean_is_exclusive(v___x_2688_);
if (v_isSharedCheck_2735_ == 0)
{
v___x_2730_ = v___x_2688_;
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
else
{
lean_inc(v_a_2728_);
lean_dec(v___x_2688_);
v___x_2730_ = lean_box(0);
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
v_resetjp_2729_:
{
lean_object* v___x_2733_; 
if (v_isShared_2731_ == 0)
{
v___x_2733_ = v___x_2730_;
goto v_reusejp_2732_;
}
else
{
lean_object* v_reuseFailAlloc_2734_; 
v_reuseFailAlloc_2734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2734_, 0, v_a_2728_);
v___x_2733_ = v_reuseFailAlloc_2734_;
goto v_reusejp_2732_;
}
v_reusejp_2732_:
{
return v___x_2733_;
}
}
}
}
else
{
lean_object* v_a_2736_; lean_object* v___x_2738_; uint8_t v_isShared_2739_; uint8_t v_isSharedCheck_2743_; 
lean_dec(v_filtered_2673_);
lean_dec_ref(v___y_2672_);
lean_dec(v___y_2671_);
lean_dec_ref(v___y_2670_);
lean_dec_ref(v___y_2669_);
lean_dec(v___y_2667_);
lean_dec_ref(v_name_2593_);
v_a_2736_ = lean_ctor_get(v___x_2680_, 0);
v_isSharedCheck_2743_ = !lean_is_exclusive(v___x_2680_);
if (v_isSharedCheck_2743_ == 0)
{
v___x_2738_ = v___x_2680_;
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
else
{
lean_inc(v_a_2736_);
lean_dec(v___x_2680_);
v___x_2738_ = lean_box(0);
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
v_resetjp_2737_:
{
lean_object* v___x_2741_; 
if (v_isShared_2739_ == 0)
{
v___x_2741_ = v___x_2738_;
goto v_reusejp_2740_;
}
else
{
lean_object* v_reuseFailAlloc_2742_; 
v_reuseFailAlloc_2742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2742_, 0, v_a_2736_);
v___x_2741_ = v_reuseFailAlloc_2742_;
goto v_reusejp_2740_;
}
v_reusejp_2740_:
{
return v___x_2741_;
}
}
}
}
v___jp_2744_:
{
lean_object* v___x_2757_; 
v___x_2757_ = lean_box(0);
v___y_2667_ = v___y_2753_;
v___y_2668_ = v___y_2745_;
v___y_2669_ = v___y_2746_;
v___y_2670_ = v___y_2747_;
v___y_2671_ = v___y_2750_;
v___y_2672_ = v___y_2756_;
v_filtered_2673_ = v___x_2757_;
v___y_2674_ = v___y_2752_;
v___y_2675_ = v___y_2754_;
v___y_2676_ = v___y_2751_;
v___y_2677_ = v___y_2755_;
v___y_2678_ = v___y_2748_;
v___y_2679_ = v___y_2749_;
goto v___jp_2666_;
}
v___jp_2758_:
{
lean_object* v___x_2774_; 
v___x_2774_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___y_2766_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_);
if (lean_obj_tag(v___x_2774_) == 0)
{
lean_object* v_a_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; size_t v_sz_2779_; lean_object* v___x_2780_; 
v_a_2775_ = lean_ctor_get(v___x_2774_, 0);
lean_inc(v_a_2775_);
lean_dec_ref_known(v___x_2774_, 1);
v___x_2776_ = lean_unsigned_to_nat(1u);
v___x_2777_ = lean_mk_empty_array_with_capacity(v___x_2776_);
lean_inc_ref(v___x_2777_);
v___x_2778_ = lean_array_push(v___x_2777_, v_a_2775_);
v_sz_2779_ = lean_array_size(v___y_2767_);
v___x_2780_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg(v___y_2767_, v_sz_2779_, v___y_2765_, v___x_2778_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_);
lean_dec_ref(v___y_2767_);
if (lean_obj_tag(v___x_2780_) == 0)
{
if (v___y_2764_ == 0)
{
if (v___y_2760_ == 0)
{
lean_object* v_a_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; 
v_a_2781_ = lean_ctor_get(v___x_2780_, 0);
lean_inc_n(v_a_2781_, 2);
lean_dec_ref_known(v___x_2780_, 1);
v___x_2782_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg___closed__0));
v___x_2783_ = lean_mk_empty_array_with_capacity(v___y_2759_);
v___x_2784_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2784_, 0, v___x_2782_);
lean_ctor_set(v___x_2784_, 1, v___x_2783_);
lean_ctor_set(v___x_2784_, 2, v_a_2781_);
lean_inc(v___y_2763_);
v___x_2785_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_2763_, v___x_2784_, v___y_2761_, v___y_2768_, v___y_2769_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_);
if (lean_obj_tag(v___x_2785_) == 0)
{
lean_object* v_a_2786_; lean_object* v___x_2787_; 
v_a_2786_ = lean_ctor_get(v___x_2785_, 0);
lean_inc(v_a_2786_);
lean_dec_ref_known(v___x_2785_, 1);
v___x_2787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2787_, 0, v_a_2786_);
v___y_2667_ = v___y_2759_;
v___y_2668_ = v___y_2761_;
v___y_2669_ = v_a_2781_;
v___y_2670_ = v___y_2762_;
v___y_2671_ = v___y_2763_;
v___y_2672_ = v___x_2777_;
v_filtered_2673_ = v___x_2787_;
v___y_2674_ = v___y_2768_;
v___y_2675_ = v___y_2769_;
v___y_2676_ = v___y_2770_;
v___y_2677_ = v___y_2771_;
v___y_2678_ = v___y_2772_;
v___y_2679_ = v___y_2773_;
goto v___jp_2666_;
}
else
{
lean_object* v_a_2788_; lean_object* v___x_2790_; uint8_t v_isShared_2791_; uint8_t v_isSharedCheck_2795_; 
lean_dec(v_a_2781_);
lean_dec_ref(v___x_2777_);
lean_dec(v___y_2763_);
lean_dec_ref(v___y_2762_);
lean_dec(v___y_2759_);
lean_dec_ref(v_name_2593_);
v_a_2788_ = lean_ctor_get(v___x_2785_, 0);
v_isSharedCheck_2795_ = !lean_is_exclusive(v___x_2785_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2790_ = v___x_2785_;
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
else
{
lean_inc(v_a_2788_);
lean_dec(v___x_2785_);
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
lean_object* v_a_2796_; 
v_a_2796_ = lean_ctor_get(v___x_2780_, 0);
lean_inc(v_a_2796_);
lean_dec_ref_known(v___x_2780_, 1);
v___y_2745_ = v___y_2761_;
v___y_2746_ = v_a_2796_;
v___y_2747_ = v___y_2762_;
v___y_2748_ = v___y_2772_;
v___y_2749_ = v___y_2773_;
v___y_2750_ = v___y_2763_;
v___y_2751_ = v___y_2770_;
v___y_2752_ = v___y_2768_;
v___y_2753_ = v___y_2759_;
v___y_2754_ = v___y_2769_;
v___y_2755_ = v___y_2771_;
v___y_2756_ = v___x_2777_;
goto v___jp_2744_;
}
}
else
{
lean_object* v_a_2797_; 
v_a_2797_ = lean_ctor_get(v___x_2780_, 0);
lean_inc(v_a_2797_);
lean_dec_ref_known(v___x_2780_, 1);
v___y_2745_ = v___y_2761_;
v___y_2746_ = v_a_2797_;
v___y_2747_ = v___y_2762_;
v___y_2748_ = v___y_2772_;
v___y_2749_ = v___y_2773_;
v___y_2750_ = v___y_2763_;
v___y_2751_ = v___y_2770_;
v___y_2752_ = v___y_2768_;
v___y_2753_ = v___y_2759_;
v___y_2754_ = v___y_2769_;
v___y_2755_ = v___y_2771_;
v___y_2756_ = v___x_2777_;
goto v___jp_2744_;
}
}
else
{
lean_object* v_a_2798_; lean_object* v___x_2800_; uint8_t v_isShared_2801_; uint8_t v_isSharedCheck_2805_; 
lean_dec_ref(v___x_2777_);
lean_dec(v___y_2763_);
lean_dec_ref(v___y_2762_);
lean_dec(v___y_2759_);
lean_dec_ref(v_name_2593_);
v_a_2798_ = lean_ctor_get(v___x_2780_, 0);
v_isSharedCheck_2805_ = !lean_is_exclusive(v___x_2780_);
if (v_isSharedCheck_2805_ == 0)
{
v___x_2800_ = v___x_2780_;
v_isShared_2801_ = v_isSharedCheck_2805_;
goto v_resetjp_2799_;
}
else
{
lean_inc(v_a_2798_);
lean_dec(v___x_2780_);
v___x_2800_ = lean_box(0);
v_isShared_2801_ = v_isSharedCheck_2805_;
goto v_resetjp_2799_;
}
v_resetjp_2799_:
{
lean_object* v___x_2803_; 
if (v_isShared_2801_ == 0)
{
v___x_2803_ = v___x_2800_;
goto v_reusejp_2802_;
}
else
{
lean_object* v_reuseFailAlloc_2804_; 
v_reuseFailAlloc_2804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2804_, 0, v_a_2798_);
v___x_2803_ = v_reuseFailAlloc_2804_;
goto v_reusejp_2802_;
}
v_reusejp_2802_:
{
return v___x_2803_;
}
}
}
}
else
{
lean_object* v_a_2806_; lean_object* v___x_2808_; uint8_t v_isShared_2809_; uint8_t v_isSharedCheck_2813_; 
lean_dec_ref(v___y_2767_);
lean_dec(v___y_2763_);
lean_dec_ref(v___y_2762_);
lean_dec(v___y_2759_);
lean_dec_ref(v_name_2593_);
v_a_2806_ = lean_ctor_get(v___x_2774_, 0);
v_isSharedCheck_2813_ = !lean_is_exclusive(v___x_2774_);
if (v_isSharedCheck_2813_ == 0)
{
v___x_2808_ = v___x_2774_;
v_isShared_2809_ = v_isSharedCheck_2813_;
goto v_resetjp_2807_;
}
else
{
lean_inc(v_a_2806_);
lean_dec(v___x_2774_);
v___x_2808_ = lean_box(0);
v_isShared_2809_ = v_isSharedCheck_2813_;
goto v_resetjp_2807_;
}
v_resetjp_2807_:
{
lean_object* v___x_2811_; 
if (v_isShared_2809_ == 0)
{
v___x_2811_ = v___x_2808_;
goto v_reusejp_2810_;
}
else
{
lean_object* v_reuseFailAlloc_2812_; 
v_reuseFailAlloc_2812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2812_, 0, v_a_2806_);
v___x_2811_ = v_reuseFailAlloc_2812_;
goto v_reusejp_2810_;
}
v_reusejp_2810_:
{
return v___x_2811_;
}
}
}
}
v___jp_2814_:
{
if (v_a_2829_ == 0)
{
v___y_2759_ = v___y_2823_;
v___y_2760_ = v___y_2824_;
v___y_2761_ = v_a_2829_;
v___y_2762_ = v___y_2815_;
v___y_2763_ = v___y_2816_;
v___y_2764_ = v___y_2819_;
v___y_2765_ = v___y_2820_;
v___y_2766_ = v___y_2827_;
v___y_2767_ = v___y_2828_;
v___y_2768_ = v___y_2822_;
v___y_2769_ = v___y_2826_;
v___y_2770_ = v___y_2817_;
v___y_2771_ = v___y_2821_;
v___y_2772_ = v___y_2818_;
v___y_2773_ = v___y_2825_;
goto v___jp_2758_;
}
else
{
lean_object* v___x_2830_; 
lean_inc(v___y_2816_);
v___x_2830_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(v___y_2816_, v___y_2822_, v___y_2826_, v___y_2817_, v___y_2821_, v___y_2818_, v___y_2825_);
if (lean_obj_tag(v___x_2830_) == 0)
{
lean_dec_ref_known(v___x_2830_, 1);
v___y_2759_ = v___y_2823_;
v___y_2760_ = v___y_2824_;
v___y_2761_ = v_a_2829_;
v___y_2762_ = v___y_2815_;
v___y_2763_ = v___y_2816_;
v___y_2764_ = v___y_2819_;
v___y_2765_ = v___y_2820_;
v___y_2766_ = v___y_2827_;
v___y_2767_ = v___y_2828_;
v___y_2768_ = v___y_2822_;
v___y_2769_ = v___y_2826_;
v___y_2770_ = v___y_2817_;
v___y_2771_ = v___y_2821_;
v___y_2772_ = v___y_2818_;
v___y_2773_ = v___y_2825_;
goto v___jp_2758_;
}
else
{
lean_object* v_a_2831_; lean_object* v___x_2833_; uint8_t v_isShared_2834_; uint8_t v_isSharedCheck_2838_; 
lean_dec_ref(v___y_2828_);
lean_dec_ref(v___y_2827_);
lean_dec(v___y_2823_);
lean_dec(v___y_2816_);
lean_dec_ref(v___y_2815_);
lean_dec_ref(v_name_2593_);
v_a_2831_ = lean_ctor_get(v___x_2830_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2830_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2833_ = v___x_2830_;
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
else
{
lean_inc(v_a_2831_);
lean_dec(v___x_2830_);
v___x_2833_ = lean_box(0);
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
v_resetjp_2832_:
{
lean_object* v___x_2836_; 
if (v_isShared_2834_ == 0)
{
v___x_2836_ = v___x_2833_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_a_2831_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
}
}
v___jp_2839_:
{
lean_object* v___x_2856_; 
lean_inc_ref(v___y_2847_);
v___x_2856_ = l_Lean_Meta_ppExpr(v___y_2847_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2856_) == 0)
{
lean_object* v_a_2857_; lean_object* v___x_2858_; 
v_a_2857_ = lean_ctor_get(v___x_2856_, 0);
lean_inc(v_a_2857_);
lean_dec_ref_known(v___x_2856_, 1);
lean_inc_ref(v___y_2847_);
v___x_2858_ = l_Lean_Meta_abstractMVars(v___y_2847_, v_a_2599_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2858_) == 0)
{
lean_object* v_a_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; 
v_a_2859_ = lean_ctor_get(v___x_2858_, 0);
lean_inc(v_a_2859_);
lean_dec_ref_known(v___x_2858_, 1);
v___x_2860_ = l_Std_Format_defWidth;
lean_inc_n(v___y_2842_, 2);
v___x_2861_ = l_Std_Format_pretty(v_a_2857_, v___x_2860_, v___y_2842_, v___y_2842_);
v___x_2862_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_GRewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___redArg(v_lem_2600_, v_i_2601_, v___y_2846_, v_justLemmaName_2849_, v___y_2850_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2862_) == 0)
{
lean_object* v_a_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; lean_object* v___x_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; uint8_t v___x_2869_; 
v_a_2863_ = lean_ctor_get(v___x_2862_, 0);
lean_inc(v_a_2863_);
lean_dec_ref_known(v___x_2862_, 1);
lean_inc_ref_n(v_name_2593_, 2);
v___x_2864_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(v_name_2593_);
v___x_2865_ = lean_array_get_size(v___y_2848_);
v___x_2866_ = lean_string_length(v___x_2861_);
lean_dec_ref(v___x_2861_);
v___x_2867_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(v_name_2593_);
v___x_2868_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2868_, 0, v___x_2865_);
lean_ctor_set(v___x_2868_, 1, v___x_2864_);
lean_ctor_set(v___x_2868_, 2, v___x_2866_);
lean_ctor_set(v___x_2868_, 3, v___x_2867_);
lean_ctor_set(v___x_2868_, 4, v_a_2859_);
v___x_2869_ = lean_nat_dec_eq(v___x_2865_, v___y_2842_);
if (v___x_2869_ == 0)
{
lean_dec_ref(v___y_2841_);
lean_dec(v_rflTarget_x3f_2602_);
v___y_2815_ = v___x_2868_;
v___y_2816_ = v_a_2863_;
v___y_2817_ = v___y_2852_;
v___y_2818_ = v___y_2854_;
v___y_2819_ = v___y_2840_;
v___y_2820_ = v___y_2845_;
v___y_2821_ = v___y_2853_;
v___y_2822_ = v___y_2850_;
v___y_2823_ = v___y_2842_;
v___y_2824_ = v___y_2843_;
v___y_2825_ = v___y_2855_;
v___y_2826_ = v___y_2851_;
v___y_2827_ = v___y_2847_;
v___y_2828_ = v___y_2848_;
v_a_2829_ = v___y_2844_;
goto v___jp_2814_;
}
else
{
if (lean_obj_tag(v_rflTarget_x3f_2602_) == 1)
{
lean_object* v_val_2870_; lean_object* v___f_2871_; lean_object* v___x_2872_; 
v_val_2870_ = lean_ctor_get(v_rflTarget_x3f_2602_, 0);
lean_inc(v_val_2870_);
lean_dec_ref_known(v_rflTarget_x3f_2602_, 1);
v___f_2871_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__2___boxed), 9, 2);
lean_closure_set(v___f_2871_, 0, v___y_2841_);
lean_closure_set(v___f_2871_, 1, v_val_2870_);
v___x_2872_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__6___redArg(v___f_2871_, v___y_2850_, v___y_2851_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2872_) == 0)
{
lean_object* v_a_2873_; uint8_t v___x_2874_; 
v_a_2873_ = lean_ctor_get(v___x_2872_, 0);
lean_inc(v_a_2873_);
lean_dec_ref_known(v___x_2872_, 1);
v___x_2874_ = lean_unbox(v_a_2873_);
lean_dec(v_a_2873_);
v___y_2815_ = v___x_2868_;
v___y_2816_ = v_a_2863_;
v___y_2817_ = v___y_2852_;
v___y_2818_ = v___y_2854_;
v___y_2819_ = v___y_2840_;
v___y_2820_ = v___y_2845_;
v___y_2821_ = v___y_2853_;
v___y_2822_ = v___y_2850_;
v___y_2823_ = v___y_2842_;
v___y_2824_ = v___y_2843_;
v___y_2825_ = v___y_2855_;
v___y_2826_ = v___y_2851_;
v___y_2827_ = v___y_2847_;
v___y_2828_ = v___y_2848_;
v_a_2829_ = v___x_2874_;
goto v___jp_2814_;
}
else
{
lean_object* v_a_2875_; lean_object* v___x_2877_; uint8_t v_isShared_2878_; uint8_t v_isSharedCheck_2882_; 
lean_dec_ref_known(v___x_2868_, 5);
lean_dec(v_a_2863_);
lean_dec_ref(v___y_2848_);
lean_dec_ref(v___y_2847_);
lean_dec(v___y_2842_);
lean_dec_ref(v_name_2593_);
v_a_2875_ = lean_ctor_get(v___x_2872_, 0);
v_isSharedCheck_2882_ = !lean_is_exclusive(v___x_2872_);
if (v_isSharedCheck_2882_ == 0)
{
v___x_2877_ = v___x_2872_;
v_isShared_2878_ = v_isSharedCheck_2882_;
goto v_resetjp_2876_;
}
else
{
lean_inc(v_a_2875_);
lean_dec(v___x_2872_);
v___x_2877_ = lean_box(0);
v_isShared_2878_ = v_isSharedCheck_2882_;
goto v_resetjp_2876_;
}
v_resetjp_2876_:
{
lean_object* v___x_2880_; 
if (v_isShared_2878_ == 0)
{
v___x_2880_ = v___x_2877_;
goto v_reusejp_2879_;
}
else
{
lean_object* v_reuseFailAlloc_2881_; 
v_reuseFailAlloc_2881_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2881_, 0, v_a_2875_);
v___x_2880_ = v_reuseFailAlloc_2881_;
goto v_reusejp_2879_;
}
v_reusejp_2879_:
{
return v___x_2880_;
}
}
}
}
else
{
lean_dec_ref(v___y_2841_);
lean_dec(v_rflTarget_x3f_2602_);
v___y_2815_ = v___x_2868_;
v___y_2816_ = v_a_2863_;
v___y_2817_ = v___y_2852_;
v___y_2818_ = v___y_2854_;
v___y_2819_ = v___y_2840_;
v___y_2820_ = v___y_2845_;
v___y_2821_ = v___y_2853_;
v___y_2822_ = v___y_2850_;
v___y_2823_ = v___y_2842_;
v___y_2824_ = v___y_2843_;
v___y_2825_ = v___y_2855_;
v___y_2826_ = v___y_2851_;
v___y_2827_ = v___y_2847_;
v___y_2828_ = v___y_2848_;
v_a_2829_ = v___y_2844_;
goto v___jp_2814_;
}
}
}
else
{
lean_object* v_a_2883_; lean_object* v___x_2885_; uint8_t v_isShared_2886_; uint8_t v_isSharedCheck_2890_; 
lean_dec_ref(v___x_2861_);
lean_dec(v_a_2859_);
lean_dec_ref(v___y_2848_);
lean_dec_ref(v___y_2847_);
lean_dec(v___y_2842_);
lean_dec_ref(v___y_2841_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_name_2593_);
v_a_2883_ = lean_ctor_get(v___x_2862_, 0);
v_isSharedCheck_2890_ = !lean_is_exclusive(v___x_2862_);
if (v_isSharedCheck_2890_ == 0)
{
v___x_2885_ = v___x_2862_;
v_isShared_2886_ = v_isSharedCheck_2890_;
goto v_resetjp_2884_;
}
else
{
lean_inc(v_a_2883_);
lean_dec(v___x_2862_);
v___x_2885_ = lean_box(0);
v_isShared_2886_ = v_isSharedCheck_2890_;
goto v_resetjp_2884_;
}
v_resetjp_2884_:
{
lean_object* v___x_2888_; 
if (v_isShared_2886_ == 0)
{
v___x_2888_ = v___x_2885_;
goto v_reusejp_2887_;
}
else
{
lean_object* v_reuseFailAlloc_2889_; 
v_reuseFailAlloc_2889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2889_, 0, v_a_2883_);
v___x_2888_ = v_reuseFailAlloc_2889_;
goto v_reusejp_2887_;
}
v_reusejp_2887_:
{
return v___x_2888_;
}
}
}
}
else
{
lean_object* v_a_2891_; lean_object* v___x_2893_; uint8_t v_isShared_2894_; uint8_t v_isSharedCheck_2898_; 
lean_dec(v_a_2857_);
lean_dec_ref(v___y_2848_);
lean_dec_ref(v___y_2847_);
lean_dec_ref(v___y_2846_);
lean_dec(v___y_2842_);
lean_dec_ref(v___y_2841_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_name_2593_);
v_a_2891_ = lean_ctor_get(v___x_2858_, 0);
v_isSharedCheck_2898_ = !lean_is_exclusive(v___x_2858_);
if (v_isSharedCheck_2898_ == 0)
{
v___x_2893_ = v___x_2858_;
v_isShared_2894_ = v_isSharedCheck_2898_;
goto v_resetjp_2892_;
}
else
{
lean_inc(v_a_2891_);
lean_dec(v___x_2858_);
v___x_2893_ = lean_box(0);
v_isShared_2894_ = v_isSharedCheck_2898_;
goto v_resetjp_2892_;
}
v_resetjp_2892_:
{
lean_object* v___x_2896_; 
if (v_isShared_2894_ == 0)
{
v___x_2896_ = v___x_2893_;
goto v_reusejp_2895_;
}
else
{
lean_object* v_reuseFailAlloc_2897_; 
v_reuseFailAlloc_2897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2897_, 0, v_a_2891_);
v___x_2896_ = v_reuseFailAlloc_2897_;
goto v_reusejp_2895_;
}
v_reusejp_2895_:
{
return v___x_2896_;
}
}
}
}
else
{
lean_object* v_a_2899_; lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2906_; 
lean_dec_ref(v___y_2848_);
lean_dec_ref(v___y_2847_);
lean_dec_ref(v___y_2846_);
lean_dec(v___y_2842_);
lean_dec_ref(v___y_2841_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_name_2593_);
v_a_2899_ = lean_ctor_get(v___x_2856_, 0);
v_isSharedCheck_2906_ = !lean_is_exclusive(v___x_2856_);
if (v_isSharedCheck_2906_ == 0)
{
v___x_2901_ = v___x_2856_;
v_isShared_2902_ = v_isSharedCheck_2906_;
goto v_resetjp_2900_;
}
else
{
lean_inc(v_a_2899_);
lean_dec(v___x_2856_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2906_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
lean_object* v___x_2904_; 
if (v_isShared_2902_ == 0)
{
v___x_2904_ = v___x_2901_;
goto v_reusejp_2903_;
}
else
{
lean_object* v_reuseFailAlloc_2905_; 
v_reuseFailAlloc_2905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2905_, 0, v_a_2899_);
v___x_2904_ = v_reuseFailAlloc_2905_;
goto v_reusejp_2903_;
}
v_reusejp_2903_:
{
return v___x_2904_;
}
}
}
}
v___jp_2907_:
{
size_t v_sz_2918_; lean_object* v___x_2919_; 
v_sz_2918_ = lean_array_size(v_a_2917_);
lean_inc_ref(v_a_2917_);
v___x_2919_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__4(v_sz_2918_, v___y_2916_, v_a_2917_, v___y_2915_, v___y_2909_, v___y_2914_, v___y_2911_, v___y_2910_, v___y_2912_);
if (lean_obj_tag(v___x_2919_) == 0)
{
lean_object* v_a_2920_; lean_object* v___x_2921_; 
v_a_2920_ = lean_ctor_get(v___x_2919_, 0);
lean_inc(v_a_2920_);
lean_dec_ref_known(v___x_2919_, 1);
v___x_2921_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_snd_2595_, v___y_2911_);
if (lean_obj_tag(v___x_2921_) == 0)
{
lean_object* v_a_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; 
v_a_2922_ = lean_ctor_get(v___x_2921_, 0);
lean_inc_n(v_a_2922_, 2);
lean_dec_ref_known(v___x_2921_, 1);
lean_inc(v_a_2920_);
v___x_2923_ = lean_array_push(v_a_2920_, v_a_2922_);
v___x_2924_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(v_a_2917_, v_assignableMVars_2596_, v___x_2923_, v___y_2914_, v___y_2911_, v___y_2910_, v___y_2912_);
lean_dec_ref(v___x_2923_);
lean_dec_ref(v_a_2917_);
if (lean_obj_tag(v___x_2924_) == 0)
{
lean_object* v_a_2925_; lean_object* v___x_2926_; 
v_a_2925_ = lean_ctor_get(v___x_2924_, 0);
lean_inc(v_a_2925_);
lean_dec_ref_known(v___x_2924_, 1);
v___x_2926_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__1___redArg(v_fst_2597_, v___y_2911_);
if (lean_obj_tag(v___x_2926_) == 0)
{
lean_object* v_a_2927_; lean_object* v___x_2928_; 
v_a_2927_ = lean_ctor_get(v___x_2926_, 0);
lean_inc(v_a_2927_);
lean_dec_ref_known(v___x_2926_, 1);
lean_inc(v_a_2922_);
v___x_2928_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(v_subExpr_2598_, v_a_2922_, v___y_2914_, v___y_2911_, v___y_2910_, v___y_2912_);
if (lean_obj_tag(v___x_2928_) == 0)
{
if (lean_obj_tag(v_rwKind_2603_) == 0)
{
lean_object* v_a_2929_; uint8_t v___x_2930_; uint8_t v___x_2931_; 
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
v_a_2929_ = lean_ctor_get(v___x_2928_, 0);
lean_inc(v_a_2929_);
lean_dec_ref_known(v___x_2928_, 1);
v___x_2930_ = lean_unbox(v_a_2929_);
lean_dec(v_a_2929_);
v___x_2931_ = lean_unbox(v_a_2925_);
lean_dec(v_a_2925_);
lean_inc(v_a_2922_);
v___y_2840_ = v___x_2930_;
v___y_2841_ = v_a_2922_;
v___y_2842_ = v___y_2908_;
v___y_2843_ = v___x_2931_;
v___y_2844_ = v___y_2913_;
v___y_2845_ = v___y_2916_;
v___y_2846_ = v_a_2927_;
v___y_2847_ = v_a_2922_;
v___y_2848_ = v_a_2920_;
v_justLemmaName_2849_ = v_a_2599_;
v___y_2850_ = v___y_2915_;
v___y_2851_ = v___y_2909_;
v___y_2852_ = v___y_2914_;
v___y_2853_ = v___y_2911_;
v___y_2854_ = v___y_2910_;
v___y_2855_ = v___y_2912_;
goto v___jp_2839_;
}
else
{
lean_object* v_a_2932_; lean_object* v___x_2933_; 
v_a_2932_ = lean_ctor_get(v___x_2928_, 0);
lean_inc(v_a_2932_);
lean_dec_ref_known(v___x_2928_, 1);
v___x_2933_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__7___redArg(v_mctx_2604_, v___f_2605_, v___y_2915_, v___y_2909_, v___y_2914_, v___y_2911_, v___y_2910_, v___y_2912_);
if (lean_obj_tag(v___x_2933_) == 0)
{
lean_object* v_a_2934_; uint8_t v___x_2935_; uint8_t v___x_2936_; uint8_t v___x_2937_; 
v_a_2934_ = lean_ctor_get(v___x_2933_, 0);
lean_inc(v_a_2934_);
lean_dec_ref_known(v___x_2933_, 1);
v___x_2935_ = lean_unbox(v_a_2932_);
lean_dec(v_a_2932_);
v___x_2936_ = lean_unbox(v_a_2925_);
lean_dec(v_a_2925_);
v___x_2937_ = lean_unbox(v_a_2934_);
lean_dec(v_a_2934_);
lean_inc(v_a_2922_);
v___y_2840_ = v___x_2935_;
v___y_2841_ = v_a_2922_;
v___y_2842_ = v___y_2908_;
v___y_2843_ = v___x_2936_;
v___y_2844_ = v___y_2913_;
v___y_2845_ = v___y_2916_;
v___y_2846_ = v_a_2927_;
v___y_2847_ = v_a_2922_;
v___y_2848_ = v_a_2920_;
v_justLemmaName_2849_ = v___x_2937_;
v___y_2850_ = v___y_2915_;
v___y_2851_ = v___y_2909_;
v___y_2852_ = v___y_2914_;
v___y_2853_ = v___y_2911_;
v___y_2854_ = v___y_2910_;
v___y_2855_ = v___y_2912_;
goto v___jp_2839_;
}
else
{
lean_object* v_a_2938_; lean_object* v___x_2940_; uint8_t v_isShared_2941_; uint8_t v_isSharedCheck_2945_; 
lean_dec(v_a_2932_);
lean_dec(v_a_2927_);
lean_dec(v_a_2925_);
lean_dec(v_a_2922_);
lean_dec(v_a_2920_);
lean_dec(v___y_2908_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_name_2593_);
v_a_2938_ = lean_ctor_get(v___x_2933_, 0);
v_isSharedCheck_2945_ = !lean_is_exclusive(v___x_2933_);
if (v_isSharedCheck_2945_ == 0)
{
v___x_2940_ = v___x_2933_;
v_isShared_2941_ = v_isSharedCheck_2945_;
goto v_resetjp_2939_;
}
else
{
lean_inc(v_a_2938_);
lean_dec(v___x_2933_);
v___x_2940_ = lean_box(0);
v_isShared_2941_ = v_isSharedCheck_2945_;
goto v_resetjp_2939_;
}
v_resetjp_2939_:
{
lean_object* v___x_2943_; 
if (v_isShared_2941_ == 0)
{
v___x_2943_ = v___x_2940_;
goto v_reusejp_2942_;
}
else
{
lean_object* v_reuseFailAlloc_2944_; 
v_reuseFailAlloc_2944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2944_, 0, v_a_2938_);
v___x_2943_ = v_reuseFailAlloc_2944_;
goto v_reusejp_2942_;
}
v_reusejp_2942_:
{
return v___x_2943_;
}
}
}
}
}
else
{
lean_object* v_a_2946_; lean_object* v___x_2948_; uint8_t v_isShared_2949_; uint8_t v_isSharedCheck_2953_; 
lean_dec(v_a_2927_);
lean_dec(v_a_2925_);
lean_dec(v_a_2922_);
lean_dec(v_a_2920_);
lean_dec(v___y_2908_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_name_2593_);
v_a_2946_ = lean_ctor_get(v___x_2928_, 0);
v_isSharedCheck_2953_ = !lean_is_exclusive(v___x_2928_);
if (v_isSharedCheck_2953_ == 0)
{
v___x_2948_ = v___x_2928_;
v_isShared_2949_ = v_isSharedCheck_2953_;
goto v_resetjp_2947_;
}
else
{
lean_inc(v_a_2946_);
lean_dec(v___x_2928_);
v___x_2948_ = lean_box(0);
v_isShared_2949_ = v_isSharedCheck_2953_;
goto v_resetjp_2947_;
}
v_resetjp_2947_:
{
lean_object* v___x_2951_; 
if (v_isShared_2949_ == 0)
{
v___x_2951_ = v___x_2948_;
goto v_reusejp_2950_;
}
else
{
lean_object* v_reuseFailAlloc_2952_; 
v_reuseFailAlloc_2952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2952_, 0, v_a_2946_);
v___x_2951_ = v_reuseFailAlloc_2952_;
goto v_reusejp_2950_;
}
v_reusejp_2950_:
{
return v___x_2951_;
}
}
}
}
else
{
lean_object* v_a_2954_; lean_object* v___x_2956_; uint8_t v_isShared_2957_; uint8_t v_isSharedCheck_2961_; 
lean_dec(v_a_2925_);
lean_dec(v_a_2922_);
lean_dec(v_a_2920_);
lean_dec(v___y_2908_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_name_2593_);
v_a_2954_ = lean_ctor_get(v___x_2926_, 0);
v_isSharedCheck_2961_ = !lean_is_exclusive(v___x_2926_);
if (v_isSharedCheck_2961_ == 0)
{
v___x_2956_ = v___x_2926_;
v_isShared_2957_ = v_isSharedCheck_2961_;
goto v_resetjp_2955_;
}
else
{
lean_inc(v_a_2954_);
lean_dec(v___x_2926_);
v___x_2956_ = lean_box(0);
v_isShared_2957_ = v_isSharedCheck_2961_;
goto v_resetjp_2955_;
}
v_resetjp_2955_:
{
lean_object* v___x_2959_; 
if (v_isShared_2957_ == 0)
{
v___x_2959_ = v___x_2956_;
goto v_reusejp_2958_;
}
else
{
lean_object* v_reuseFailAlloc_2960_; 
v_reuseFailAlloc_2960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2960_, 0, v_a_2954_);
v___x_2959_ = v_reuseFailAlloc_2960_;
goto v_reusejp_2958_;
}
v_reusejp_2958_:
{
return v___x_2959_;
}
}
}
}
else
{
lean_object* v_a_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_2969_; 
lean_dec(v_a_2922_);
lean_dec(v_a_2920_);
lean_dec(v___y_2908_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_name_2593_);
v_a_2962_ = lean_ctor_get(v___x_2924_, 0);
v_isSharedCheck_2969_ = !lean_is_exclusive(v___x_2924_);
if (v_isSharedCheck_2969_ == 0)
{
v___x_2964_ = v___x_2924_;
v_isShared_2965_ = v_isSharedCheck_2969_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_a_2962_);
lean_dec(v___x_2924_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_2969_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___x_2967_; 
if (v_isShared_2965_ == 0)
{
v___x_2967_ = v___x_2964_;
goto v_reusejp_2966_;
}
else
{
lean_object* v_reuseFailAlloc_2968_; 
v_reuseFailAlloc_2968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2968_, 0, v_a_2962_);
v___x_2967_ = v_reuseFailAlloc_2968_;
goto v_reusejp_2966_;
}
v_reusejp_2966_:
{
return v___x_2967_;
}
}
}
}
else
{
lean_object* v_a_2970_; lean_object* v___x_2972_; uint8_t v_isShared_2973_; uint8_t v_isSharedCheck_2977_; 
lean_dec(v_a_2920_);
lean_dec_ref(v_a_2917_);
lean_dec(v___y_2908_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_name_2593_);
v_a_2970_ = lean_ctor_get(v___x_2921_, 0);
v_isSharedCheck_2977_ = !lean_is_exclusive(v___x_2921_);
if (v_isSharedCheck_2977_ == 0)
{
v___x_2972_ = v___x_2921_;
v_isShared_2973_ = v_isSharedCheck_2977_;
goto v_resetjp_2971_;
}
else
{
lean_inc(v_a_2970_);
lean_dec(v___x_2921_);
v___x_2972_ = lean_box(0);
v_isShared_2973_ = v_isSharedCheck_2977_;
goto v_resetjp_2971_;
}
v_resetjp_2971_:
{
lean_object* v___x_2975_; 
if (v_isShared_2973_ == 0)
{
v___x_2975_ = v___x_2972_;
goto v_reusejp_2974_;
}
else
{
lean_object* v_reuseFailAlloc_2976_; 
v_reuseFailAlloc_2976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2976_, 0, v_a_2970_);
v___x_2975_ = v_reuseFailAlloc_2976_;
goto v_reusejp_2974_;
}
v_reusejp_2974_:
{
return v___x_2975_;
}
}
}
}
else
{
lean_object* v_a_2978_; lean_object* v___x_2980_; uint8_t v_isShared_2981_; uint8_t v_isSharedCheck_2985_; 
lean_dec_ref(v_a_2917_);
lean_dec(v___y_2908_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_snd_2595_);
lean_dec_ref(v_name_2593_);
v_a_2978_ = lean_ctor_get(v___x_2919_, 0);
v_isSharedCheck_2985_ = !lean_is_exclusive(v___x_2919_);
if (v_isSharedCheck_2985_ == 0)
{
v___x_2980_ = v___x_2919_;
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
else
{
lean_inc(v_a_2978_);
lean_dec(v___x_2919_);
v___x_2980_ = lean_box(0);
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
v_resetjp_2979_:
{
lean_object* v___x_2983_; 
if (v_isShared_2981_ == 0)
{
v___x_2983_ = v___x_2980_;
goto v_reusejp_2982_;
}
else
{
lean_object* v_reuseFailAlloc_2984_; 
v_reuseFailAlloc_2984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2984_, 0, v_a_2978_);
v___x_2983_ = v_reuseFailAlloc_2984_;
goto v_reusejp_2982_;
}
v_reusejp_2982_:
{
return v___x_2983_;
}
}
}
}
v___jp_2986_:
{
if (lean_obj_tag(v___y_2996_) == 0)
{
lean_object* v_a_2997_; 
v_a_2997_ = lean_ctor_get(v___y_2996_, 0);
lean_inc(v_a_2997_);
lean_dec_ref_known(v___y_2996_, 1);
v___y_2908_ = v___y_2988_;
v___y_2909_ = v___y_2987_;
v___y_2910_ = v___y_2989_;
v___y_2911_ = v___y_2990_;
v___y_2912_ = v___y_2991_;
v___y_2913_ = v___y_2992_;
v___y_2914_ = v___y_2993_;
v___y_2915_ = v___y_2995_;
v___y_2916_ = v___y_2994_;
v_a_2917_ = v_a_2997_;
goto v___jp_2907_;
}
else
{
lean_object* v_a_2998_; lean_object* v___x_3000_; uint8_t v_isShared_3001_; uint8_t v_isSharedCheck_3005_; 
lean_dec(v___y_2988_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_snd_2595_);
lean_dec_ref(v_name_2593_);
v_a_2998_ = lean_ctor_get(v___y_2996_, 0);
v_isSharedCheck_3005_ = !lean_is_exclusive(v___y_2996_);
if (v_isSharedCheck_3005_ == 0)
{
v___x_3000_ = v___y_2996_;
v_isShared_3001_ = v_isSharedCheck_3005_;
goto v_resetjp_2999_;
}
else
{
lean_inc(v_a_2998_);
lean_dec(v___y_2996_);
v___x_3000_ = lean_box(0);
v_isShared_3001_ = v_isSharedCheck_3005_;
goto v_resetjp_2999_;
}
v_resetjp_2999_:
{
lean_object* v___x_3003_; 
if (v_isShared_3001_ == 0)
{
v___x_3003_ = v___x_3000_;
goto v_reusejp_3002_;
}
else
{
lean_object* v_reuseFailAlloc_3004_; 
v_reuseFailAlloc_3004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3004_, 0, v_a_2998_);
v___x_3003_ = v_reuseFailAlloc_3004_;
goto v_reusejp_3002_;
}
v_reusejp_3002_:
{
return v___x_3003_;
}
}
}
}
v___jp_3006_:
{
lean_object* v___x_3013_; lean_object* v___x_3014_; uint8_t v___x_3015_; lean_object* v___x_3016_; 
v___x_3013_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__3));
v___x_3014_ = lean_box(0);
v___x_3015_ = 0;
v___x_3016_ = l_Lean_Meta_synthAppInstances(v___x_3013_, v___x_3014_, v_fst_2606_, v_fst_2607_, v___x_3015_, v___x_3015_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
if (lean_obj_tag(v___x_3016_) == 0)
{
size_t v_sz_3017_; size_t v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; uint8_t v___x_3023_; 
lean_dec_ref_known(v___x_3016_, 1);
v_sz_3017_ = lean_array_size(v_fst_2606_);
v___x_3018_ = ((size_t)0ULL);
v___x_3019_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__3(v_sz_3017_, v___x_3018_, v_fst_2606_);
v___x_3020_ = lean_unsigned_to_nat(0u);
v___x_3021_ = lean_array_get_size(v___x_3019_);
v___x_3022_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___closed__4));
v___x_3023_ = lean_nat_dec_lt(v___x_3020_, v___x_3021_);
if (v___x_3023_ == 0)
{
lean_dec_ref(v___x_3019_);
v___y_2908_ = v___x_3020_;
v___y_2909_ = v___y_3008_;
v___y_2910_ = v___y_3011_;
v___y_2911_ = v___y_3010_;
v___y_2912_ = v___y_3012_;
v___y_2913_ = v___x_3015_;
v___y_2914_ = v___y_3009_;
v___y_2915_ = v___y_3007_;
v___y_2916_ = v___x_3018_;
v_a_2917_ = v___x_3022_;
goto v___jp_2907_;
}
else
{
uint8_t v___x_3024_; 
v___x_3024_ = lean_nat_dec_le(v___x_3021_, v___x_3021_);
if (v___x_3024_ == 0)
{
if (v___x_3023_ == 0)
{
lean_dec_ref(v___x_3019_);
v___y_2908_ = v___x_3020_;
v___y_2909_ = v___y_3008_;
v___y_2910_ = v___y_3011_;
v___y_2911_ = v___y_3010_;
v___y_2912_ = v___y_3012_;
v___y_2913_ = v___x_3015_;
v___y_2914_ = v___y_3009_;
v___y_2915_ = v___y_3007_;
v___y_2916_ = v___x_3018_;
v_a_2917_ = v___x_3022_;
goto v___jp_2907_;
}
else
{
size_t v___x_3025_; lean_object* v___x_3026_; 
v___x_3025_ = lean_usize_of_nat(v___x_3021_);
v___x_3026_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8(v___x_3019_, v___x_3018_, v___x_3025_, v___x_3022_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
lean_dec_ref(v___x_3019_);
v___y_2987_ = v___y_3008_;
v___y_2988_ = v___x_3020_;
v___y_2989_ = v___y_3011_;
v___y_2990_ = v___y_3010_;
v___y_2991_ = v___y_3012_;
v___y_2992_ = v___x_3015_;
v___y_2993_ = v___y_3009_;
v___y_2994_ = v___x_3018_;
v___y_2995_ = v___y_3007_;
v___y_2996_ = v___x_3026_;
goto v___jp_2986_;
}
}
else
{
size_t v___x_3027_; lean_object* v___x_3028_; 
v___x_3027_ = lean_usize_of_nat(v___x_3021_);
v___x_3028_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__8(v___x_3019_, v___x_3018_, v___x_3027_, v___x_3022_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
lean_dec_ref(v___x_3019_);
v___y_2987_ = v___y_3008_;
v___y_2988_ = v___x_3020_;
v___y_2989_ = v___y_3011_;
v___y_2990_ = v___y_3010_;
v___y_2991_ = v___y_3012_;
v___y_2992_ = v___x_3015_;
v___y_2993_ = v___y_3009_;
v___y_2994_ = v___x_3018_;
v___y_2995_ = v___y_3007_;
v___y_2996_ = v___x_3028_;
goto v___jp_2986_;
}
}
}
else
{
lean_object* v_a_3029_; lean_object* v___x_3031_; uint8_t v_isShared_3032_; uint8_t v_isSharedCheck_3036_; 
lean_dec_ref(v_fst_2606_);
lean_dec_ref(v___f_2605_);
lean_dec_ref(v_mctx_2604_);
lean_dec(v_rflTarget_x3f_2602_);
lean_dec_ref(v_i_2601_);
lean_dec_ref(v_lem_2600_);
lean_dec_ref(v_subExpr_2598_);
lean_dec_ref(v_fst_2597_);
lean_dec_ref(v_snd_2595_);
lean_dec_ref(v_name_2593_);
v_a_3029_ = lean_ctor_get(v___x_3016_, 0);
v_isSharedCheck_3036_ = !lean_is_exclusive(v___x_3016_);
if (v_isSharedCheck_3036_ == 0)
{
v___x_3031_ = v___x_3016_;
v_isShared_3032_ = v_isSharedCheck_3036_;
goto v_resetjp_3030_;
}
else
{
lean_inc(v_a_3029_);
lean_dec(v___x_3016_);
v___x_3031_ = lean_box(0);
v_isShared_3032_ = v_isSharedCheck_3036_;
goto v_resetjp_3030_;
}
v_resetjp_3030_:
{
lean_object* v___x_3034_; 
if (v_isShared_3032_ == 0)
{
v___x_3034_ = v___x_3031_;
goto v_reusejp_3033_;
}
else
{
lean_object* v_reuseFailAlloc_3035_; 
v_reuseFailAlloc_3035_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3035_, 0, v_a_3029_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3___boxed(lean_object** _args){
lean_object* v_fst_3070_ = _args[0];
lean_object* v_name_3071_ = _args[1];
lean_object* v_symm_3072_ = _args[2];
lean_object* v_snd_3073_ = _args[3];
lean_object* v_assignableMVars_3074_ = _args[4];
lean_object* v_fst_3075_ = _args[5];
lean_object* v_subExpr_3076_ = _args[6];
lean_object* v_a_3077_ = _args[7];
lean_object* v_lem_3078_ = _args[8];
lean_object* v_i_3079_ = _args[9];
lean_object* v_rflTarget_x3f_3080_ = _args[10];
lean_object* v_rwKind_3081_ = _args[11];
lean_object* v_mctx_3082_ = _args[12];
lean_object* v___f_3083_ = _args[13];
lean_object* v_fst_3084_ = _args[14];
lean_object* v_fst_3085_ = _args[15];
lean_object* v_____r_3086_ = _args[16];
lean_object* v___y_3087_ = _args[17];
lean_object* v___y_3088_ = _args[18];
lean_object* v___y_3089_ = _args[19];
lean_object* v___y_3090_ = _args[20];
lean_object* v___y_3091_ = _args[21];
lean_object* v___y_3092_ = _args[22];
lean_object* v___y_3093_ = _args[23];
_start:
{
uint8_t v_symm_boxed_3094_; uint8_t v_a_96866__boxed_3095_; lean_object* v_res_3096_; 
v_symm_boxed_3094_ = lean_unbox(v_symm_3072_);
v_a_96866__boxed_3095_ = lean_unbox(v_a_3077_);
v_res_3096_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3(v_fst_3070_, v_name_3071_, v_symm_boxed_3094_, v_snd_3073_, v_assignableMVars_3074_, v_fst_3075_, v_subExpr_3076_, v_a_96866__boxed_3095_, v_lem_3078_, v_i_3079_, v_rflTarget_x3f_3080_, v_rwKind_3081_, v_mctx_3082_, v___f_3083_, v_fst_3084_, v_fst_3085_, v_____r_3086_, v___y_3087_, v___y_3088_, v___y_3089_, v___y_3090_, v___y_3091_, v___y_3092_);
lean_dec(v___y_3092_);
lean_dec_ref(v___y_3091_);
lean_dec(v___y_3090_);
lean_dec_ref(v___y_3089_);
lean_dec(v___y_3088_);
lean_dec_ref(v___y_3087_);
lean_dec(v_rwKind_3081_);
lean_dec_ref(v_assignableMVars_3074_);
return v_res_3096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1(lean_object* v_rootExpr_3097_, lean_object* v_fst_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_){
_start:
{
lean_object* v_pos_3106_; lean_object* v___x_3107_; 
v_pos_3106_ = lean_ctor_get(v___y_3099_, 9);
v___x_3107_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_kabstractFindsPositions(v_rootExpr_3097_, v_fst_3098_, v_pos_3106_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_);
return v___x_3107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1___boxed(lean_object* v_rootExpr_3108_, lean_object* v_fst_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_){
_start:
{
lean_object* v_res_3117_; 
v_res_3117_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1(v_rootExpr_3108_, v_fst_3109_, v___y_3110_, v___y_3111_, v___y_3112_, v___y_3113_, v___y_3114_, v___y_3115_);
lean_dec(v___y_3115_);
lean_dec_ref(v___y_3114_);
lean_dec(v___y_3113_);
lean_dec_ref(v___y_3112_);
lean_dec(v___y_3111_);
lean_dec_ref(v___y_3110_);
return v_res_3117_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1(void){
_start:
{
lean_object* v___x_3119_; lean_object* v___x_3120_; 
v___x_3119_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__0));
v___x_3120_ = l_Lean_stringToMessageData(v___x_3119_);
return v___x_3120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9(lean_object* v_lem_3124_, lean_object* v___x_3125_, lean_object* v_i_3126_, lean_object* v_assignableMVars_3127_, lean_object* v_as_3128_, size_t v_sz_3129_, size_t v_i_3130_, lean_object* v_b_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
lean_object* v_a_3140_; uint8_t v___x_3144_; 
v___x_3144_ = lean_usize_dec_lt(v_i_3130_, v_sz_3129_);
if (v___x_3144_ == 0)
{
lean_object* v___x_3145_; 
lean_dec_ref(v_i_3126_);
lean_dec_ref(v___x_3125_);
lean_dec_ref(v_lem_3124_);
v___x_3145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3145_, 0, v_b_3131_);
return v___x_3145_;
}
else
{
lean_object* v_name_3146_; uint8_t v_symm_3147_; lean_object* v_relName_3148_; lean_object* v_a_3149_; lean_object* v_relName_3150_; lean_object* v_symm_x3f_3151_; lean_object* v___x_3152_; lean_object* v_a_3154_; lean_object* v___y_3159_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; lean_object* v___y_3174_; lean_object* v___y_3175_; lean_object* v___y_3176_; uint8_t v___y_3177_; lean_object* v___y_3178_; lean_object* v_fst_3179_; lean_object* v_snd_3180_; lean_object* v___x_3212_; lean_object* v___y_3214_; uint8_t v___y_3225_; uint8_t v___x_3285_; 
lean_dec_ref(v_b_3131_);
v_name_3146_ = lean_ctor_get(v_lem_3124_, 0);
v_symm_3147_ = lean_ctor_get_uint8(v_lem_3124_, sizeof(void*)*2);
v_relName_3148_ = lean_ctor_get(v_lem_3124_, 1);
v_a_3149_ = lean_array_uget_borrowed(v_as_3128_, v_i_3130_);
v_relName_3150_ = lean_ctor_get(v_a_3149_, 0);
v_symm_x3f_3151_ = lean_ctor_get(v_a_3149_, 2);
v___x_3152_ = lean_box(0);
v___x_3212_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__2));
v___x_3285_ = lean_name_eq(v_relName_3148_, v_relName_3150_);
if (v___x_3285_ == 0)
{
v___y_3225_ = v___x_3285_;
goto v___jp_3224_;
}
else
{
if (lean_obj_tag(v_symm_x3f_3151_) == 0)
{
v___y_3225_ = v___x_3285_;
goto v___jp_3224_;
}
else
{
lean_object* v_val_3286_; uint8_t v___x_3287_; 
v_val_3286_ = lean_ctor_get(v_symm_x3f_3151_, 0);
v___x_3287_ = lean_unbox(v_val_3286_);
if (v___x_3287_ == 0)
{
if (v_symm_3147_ == 0)
{
v___y_3225_ = v___x_3285_;
goto v___jp_3224_;
}
else
{
v_a_3140_ = v___x_3212_;
goto v___jp_3139_;
}
}
else
{
v___y_3225_ = v_symm_3147_;
goto v___jp_3224_;
}
}
}
v___jp_3153_:
{
lean_object* v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; 
v___x_3155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3155_, 0, v_a_3154_);
v___x_3156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3156_, 0, v___x_3155_);
lean_ctor_set(v___x_3156_, 1, v___x_3152_);
v___x_3157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3157_, 0, v___x_3156_);
return v___x_3157_;
}
v___jp_3158_:
{
if (lean_obj_tag(v___y_3159_) == 0)
{
lean_object* v_a_3160_; lean_object* v___x_3161_; 
v_a_3160_ = lean_ctor_get(v___y_3159_, 0);
lean_inc(v_a_3160_);
lean_dec_ref_known(v___y_3159_, 1);
v___x_3161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3161_, 0, v_a_3160_);
v_a_3154_ = v___x_3161_;
goto v___jp_3153_;
}
else
{
lean_object* v_a_3162_; lean_object* v___x_3164_; uint8_t v_isShared_3165_; uint8_t v_isSharedCheck_3169_; 
v_a_3162_ = lean_ctor_get(v___y_3159_, 0);
v_isSharedCheck_3169_ = !lean_is_exclusive(v___y_3159_);
if (v_isSharedCheck_3169_ == 0)
{
v___x_3164_ = v___y_3159_;
v_isShared_3165_ = v_isSharedCheck_3169_;
goto v_resetjp_3163_;
}
else
{
lean_inc(v_a_3162_);
lean_dec(v___y_3159_);
v___x_3164_ = lean_box(0);
v_isShared_3165_ = v_isSharedCheck_3169_;
goto v_resetjp_3163_;
}
v_resetjp_3163_:
{
lean_object* v___x_3167_; 
if (v_isShared_3165_ == 0)
{
v___x_3167_ = v___x_3164_;
goto v_reusejp_3166_;
}
else
{
lean_object* v_reuseFailAlloc_3168_; 
v_reuseFailAlloc_3168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3168_, 0, v_a_3162_);
v___x_3167_ = v_reuseFailAlloc_3168_;
goto v_reusejp_3166_;
}
v_reusejp_3166_:
{
return v___x_3167_;
}
}
}
}
v___jp_3170_:
{
lean_object* v___x_3181_; lean_object* v___x_3182_; 
v___x_3181_ = lean_st_ref_get(v___y_3135_);
lean_inc_ref(v_fst_3179_);
lean_inc_ref(v___y_3174_);
v___x_3182_ = l_Lean_Meta_isExprDefEq(v___y_3174_, v_fst_3179_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
if (lean_obj_tag(v___x_3182_) == 0)
{
lean_object* v_a_3183_; lean_object* v_mctx_3184_; lean_object* v___f_3185_; uint8_t v___x_3186_; 
v_a_3183_ = lean_ctor_get(v___x_3182_, 0);
lean_inc(v_a_3183_);
lean_dec_ref_known(v___x_3182_, 1);
v_mctx_3184_ = lean_ctor_get(v___x_3181_, 0);
lean_inc_ref(v_mctx_3184_);
lean_dec(v___x_3181_);
lean_inc_ref(v_fst_3179_);
v___f_3185_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__1___boxed), 9, 2);
lean_closure_set(v___f_3185_, 0, v___y_3172_);
lean_closure_set(v___f_3185_, 1, v_fst_3179_);
v___x_3186_ = lean_unbox(v_a_3183_);
lean_dec(v_a_3183_);
if (v___x_3186_ == 0)
{
lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; 
lean_inc_ref(v_fst_3179_);
v___x_3187_ = l_Lean_MessageData_ofExpr(v_fst_3179_);
v___x_3188_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__1);
v___x_3189_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3189_, 0, v___x_3187_);
lean_ctor_set(v___x_3189_, 1, v___x_3188_);
lean_inc_ref(v___y_3174_);
v___x_3190_ = l_Lean_MessageData_ofExpr(v___y_3174_);
v___x_3191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3191_, 0, v___x_3189_);
lean_ctor_set(v___x_3191_, 1, v___x_3190_);
v___x_3192_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v___x_3191_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
if (lean_obj_tag(v___x_3192_) == 0)
{
lean_object* v_a_3193_; lean_object* v___x_3194_; 
v_a_3193_ = lean_ctor_get(v___x_3192_, 0);
lean_inc(v_a_3193_);
lean_dec_ref_known(v___x_3192_, 1);
v___x_3194_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3(v_fst_3179_, v_name_3146_, v_symm_3147_, v_snd_3180_, v_assignableMVars_3127_, v___y_3173_, v___y_3174_, v___y_3177_, v_lem_3124_, v_i_3126_, v___y_3175_, v___y_3176_, v_mctx_3184_, v___f_3185_, v___y_3178_, v___y_3171_, v_a_3193_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
lean_dec(v___y_3176_);
v___y_3159_ = v___x_3194_;
goto v___jp_3158_;
}
else
{
lean_object* v_a_3195_; lean_object* v___x_3197_; uint8_t v_isShared_3198_; uint8_t v_isSharedCheck_3202_; 
lean_dec_ref(v___f_3185_);
lean_dec_ref(v_mctx_3184_);
lean_dec_ref(v_snd_3180_);
lean_dec_ref(v_fst_3179_);
lean_dec_ref(v___y_3178_);
lean_dec(v___y_3176_);
lean_dec(v___y_3175_);
lean_dec_ref(v___y_3174_);
lean_dec_ref(v___y_3173_);
lean_dec_ref(v___y_3171_);
lean_dec_ref(v_name_3146_);
lean_dec_ref(v_i_3126_);
lean_dec_ref(v_lem_3124_);
v_a_3195_ = lean_ctor_get(v___x_3192_, 0);
v_isSharedCheck_3202_ = !lean_is_exclusive(v___x_3192_);
if (v_isSharedCheck_3202_ == 0)
{
v___x_3197_ = v___x_3192_;
v_isShared_3198_ = v_isSharedCheck_3202_;
goto v_resetjp_3196_;
}
else
{
lean_inc(v_a_3195_);
lean_dec(v___x_3192_);
v___x_3197_ = lean_box(0);
v_isShared_3198_ = v_isSharedCheck_3202_;
goto v_resetjp_3196_;
}
v_resetjp_3196_:
{
lean_object* v___x_3200_; 
if (v_isShared_3198_ == 0)
{
v___x_3200_ = v___x_3197_;
goto v_reusejp_3199_;
}
else
{
lean_object* v_reuseFailAlloc_3201_; 
v_reuseFailAlloc_3201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3201_, 0, v_a_3195_);
v___x_3200_ = v_reuseFailAlloc_3201_;
goto v_reusejp_3199_;
}
v_reusejp_3199_:
{
return v___x_3200_;
}
}
}
}
else
{
lean_object* v___x_3203_; 
v___x_3203_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__3(v_fst_3179_, v_name_3146_, v_symm_3147_, v_snd_3180_, v_assignableMVars_3127_, v___y_3173_, v___y_3174_, v___y_3177_, v_lem_3124_, v_i_3126_, v___y_3175_, v___y_3176_, v_mctx_3184_, v___f_3185_, v___y_3178_, v___y_3171_, v___x_3152_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
lean_dec(v___y_3176_);
v___y_3159_ = v___x_3203_;
goto v___jp_3158_;
}
}
else
{
lean_object* v_a_3204_; lean_object* v___x_3206_; uint8_t v_isShared_3207_; uint8_t v_isSharedCheck_3211_; 
lean_dec(v___x_3181_);
lean_dec_ref(v_snd_3180_);
lean_dec_ref(v_fst_3179_);
lean_dec_ref(v___y_3178_);
lean_dec(v___y_3176_);
lean_dec(v___y_3175_);
lean_dec_ref(v___y_3174_);
lean_dec_ref(v___y_3173_);
lean_dec_ref(v___y_3172_);
lean_dec_ref(v___y_3171_);
lean_dec_ref(v_name_3146_);
lean_dec_ref(v_i_3126_);
lean_dec_ref(v_lem_3124_);
v_a_3204_ = lean_ctor_get(v___x_3182_, 0);
v_isSharedCheck_3211_ = !lean_is_exclusive(v___x_3182_);
if (v_isSharedCheck_3211_ == 0)
{
v___x_3206_ = v___x_3182_;
v_isShared_3207_ = v_isSharedCheck_3211_;
goto v_resetjp_3205_;
}
else
{
lean_inc(v_a_3204_);
lean_dec(v___x_3182_);
v___x_3206_ = lean_box(0);
v_isShared_3207_ = v_isSharedCheck_3211_;
goto v_resetjp_3205_;
}
v_resetjp_3205_:
{
lean_object* v___x_3209_; 
if (v_isShared_3207_ == 0)
{
v___x_3209_ = v___x_3206_;
goto v_reusejp_3208_;
}
else
{
lean_object* v_reuseFailAlloc_3210_; 
v_reuseFailAlloc_3210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3210_, 0, v_a_3204_);
v___x_3209_ = v_reuseFailAlloc_3210_;
goto v_reusejp_3208_;
}
v_reusejp_3208_:
{
return v___x_3209_;
}
}
}
}
v___jp_3213_:
{
if (lean_obj_tag(v___y_3214_) == 0)
{
lean_object* v_a_3215_; 
v_a_3215_ = lean_ctor_get(v___y_3214_, 0);
lean_inc(v_a_3215_);
lean_dec_ref_known(v___y_3214_, 1);
if (lean_obj_tag(v_a_3215_) == 1)
{
lean_dec_ref(v_i_3126_);
lean_dec_ref(v___x_3125_);
lean_dec_ref(v_lem_3124_);
v_a_3154_ = v_a_3215_;
goto v___jp_3153_;
}
else
{
lean_dec(v_a_3215_);
v_a_3140_ = v___x_3212_;
goto v___jp_3139_;
}
}
else
{
lean_object* v_a_3216_; lean_object* v___x_3218_; uint8_t v_isShared_3219_; uint8_t v_isSharedCheck_3223_; 
lean_dec_ref(v_i_3126_);
lean_dec_ref(v___x_3125_);
lean_dec_ref(v_lem_3124_);
v_a_3216_ = lean_ctor_get(v___y_3214_, 0);
v_isSharedCheck_3223_ = !lean_is_exclusive(v___y_3214_);
if (v_isSharedCheck_3223_ == 0)
{
v___x_3218_ = v___y_3214_;
v_isShared_3219_ = v_isSharedCheck_3223_;
goto v_resetjp_3217_;
}
else
{
lean_inc(v_a_3216_);
lean_dec(v___y_3214_);
v___x_3218_ = lean_box(0);
v_isShared_3219_ = v_isSharedCheck_3223_;
goto v_resetjp_3217_;
}
v_resetjp_3217_:
{
lean_object* v___x_3221_; 
if (v_isShared_3219_ == 0)
{
v___x_3221_ = v___x_3218_;
goto v_reusejp_3220_;
}
else
{
lean_object* v_reuseFailAlloc_3222_; 
v_reuseFailAlloc_3222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3222_, 0, v_a_3216_);
v___x_3221_ = v_reuseFailAlloc_3222_;
goto v_reusejp_3220_;
}
v_reusejp_3220_:
{
return v___x_3221_;
}
}
}
}
v___jp_3224_:
{
if (v___y_3225_ == 0)
{
v_a_3140_ = v___x_3212_;
goto v___jp_3139_;
}
else
{
lean_object* v___x_3226_; 
lean_inc_ref(v_name_3146_);
v___x_3226_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(v_name_3146_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
if (lean_obj_tag(v___x_3226_) == 0)
{
lean_object* v_a_3227_; lean_object* v_snd_3228_; lean_object* v_snd_3229_; lean_object* v_fst_3230_; lean_object* v_fst_3231_; lean_object* v_fst_3232_; lean_object* v_snd_3233_; lean_object* v___x_3234_; 
v_a_3227_ = lean_ctor_get(v___x_3226_, 0);
lean_inc(v_a_3227_);
lean_dec_ref_known(v___x_3226_, 1);
v_snd_3228_ = lean_ctor_get(v_a_3227_, 1);
lean_inc(v_snd_3228_);
v_snd_3229_ = lean_ctor_get(v_snd_3228_, 1);
lean_inc(v_snd_3229_);
v_fst_3230_ = lean_ctor_get(v_a_3227_, 0);
lean_inc(v_fst_3230_);
lean_dec(v_a_3227_);
v_fst_3231_ = lean_ctor_get(v_snd_3228_, 0);
lean_inc(v_fst_3231_);
lean_dec(v_snd_3228_);
v_fst_3232_ = lean_ctor_get(v_snd_3229_, 0);
lean_inc(v_fst_3232_);
v_snd_3233_ = lean_ctor_get(v_snd_3229_, 1);
lean_inc(v_snd_3233_);
lean_dec(v_snd_3229_);
v___x_3234_ = l_Lean_Expr_cleanupAnnotations(v_snd_3233_);
if (lean_obj_tag(v___x_3234_) == 5)
{
lean_object* v_fn_3235_; 
v_fn_3235_ = lean_ctor_get(v___x_3234_, 0);
lean_inc_ref(v_fn_3235_);
if (lean_obj_tag(v_fn_3235_) == 5)
{
lean_object* v_arg_3236_; lean_object* v_fn_3237_; lean_object* v_arg_3238_; lean_object* v_relation_3239_; lean_object* v___x_3240_; 
v_arg_3236_ = lean_ctor_get(v___x_3234_, 1);
lean_inc_ref(v_arg_3236_);
lean_dec_ref_known(v___x_3234_, 2);
v_fn_3237_ = lean_ctor_get(v_fn_3235_, 0);
lean_inc_ref(v_fn_3237_);
v_arg_3238_ = lean_ctor_get(v_fn_3235_, 1);
lean_inc_ref(v_arg_3238_);
lean_dec_ref_known(v_fn_3235_, 2);
v_relation_3239_ = lean_ctor_get(v_a_3149_, 1);
lean_inc_ref(v_relation_3239_);
v___x_3240_ = l_Lean_Meta_isExprDefEq(v_fn_3237_, v_relation_3239_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
if (lean_obj_tag(v___x_3240_) == 0)
{
lean_object* v_a_3241_; uint8_t v___x_3242_; 
v_a_3241_ = lean_ctor_get(v___x_3240_, 0);
lean_inc(v_a_3241_);
lean_dec_ref_known(v___x_3240_, 1);
v___x_3242_ = lean_unbox(v_a_3241_);
if (v___x_3242_ == 0)
{
lean_object* v___x_3243_; lean_object* v_cache_3244_; lean_object* v_zetaDeltaFVarIds_3245_; lean_object* v_postponed_3246_; lean_object* v_diag_3247_; lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3255_; 
lean_dec(v_a_3241_);
lean_dec_ref(v_arg_3238_);
lean_dec_ref(v_arg_3236_);
lean_dec(v_fst_3232_);
lean_dec(v_fst_3231_);
lean_dec(v_fst_3230_);
v___x_3243_ = lean_st_ref_take(v___y_3135_);
v_cache_3244_ = lean_ctor_get(v___x_3243_, 1);
v_zetaDeltaFVarIds_3245_ = lean_ctor_get(v___x_3243_, 2);
v_postponed_3246_ = lean_ctor_get(v___x_3243_, 3);
v_diag_3247_ = lean_ctor_get(v___x_3243_, 4);
v_isSharedCheck_3255_ = !lean_is_exclusive(v___x_3243_);
if (v_isSharedCheck_3255_ == 0)
{
lean_object* v_unused_3256_; 
v_unused_3256_ = lean_ctor_get(v___x_3243_, 0);
lean_dec(v_unused_3256_);
v___x_3249_ = v___x_3243_;
v_isShared_3250_ = v_isSharedCheck_3255_;
goto v_resetjp_3248_;
}
else
{
lean_inc(v_diag_3247_);
lean_inc(v_postponed_3246_);
lean_inc(v_zetaDeltaFVarIds_3245_);
lean_inc(v_cache_3244_);
lean_dec(v___x_3243_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3255_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
lean_object* v___x_3252_; 
lean_inc_ref(v___x_3125_);
if (v_isShared_3250_ == 0)
{
lean_ctor_set(v___x_3249_, 0, v___x_3125_);
v___x_3252_ = v___x_3249_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v___x_3125_);
lean_ctor_set(v_reuseFailAlloc_3254_, 1, v_cache_3244_);
lean_ctor_set(v_reuseFailAlloc_3254_, 2, v_zetaDeltaFVarIds_3245_);
lean_ctor_set(v_reuseFailAlloc_3254_, 3, v_postponed_3246_);
lean_ctor_set(v_reuseFailAlloc_3254_, 4, v_diag_3247_);
v___x_3252_ = v_reuseFailAlloc_3254_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
lean_object* v___x_3253_; 
v___x_3253_ = lean_st_ref_set(v___y_3135_, v___x_3252_);
v_a_3140_ = v___x_3212_;
goto v___jp_3139_;
}
}
}
else
{
lean_inc_ref(v_name_3146_);
lean_dec_ref(v___x_3125_);
if (v_symm_3147_ == 0)
{
lean_object* v_rootExpr_3257_; lean_object* v_subExpr_3258_; lean_object* v_rflTarget_x3f_3259_; lean_object* v_rwKind_3260_; uint8_t v___x_3261_; 
v_rootExpr_3257_ = lean_ctor_get(v_i_3126_, 0);
v_subExpr_3258_ = lean_ctor_get(v_i_3126_, 1);
v_rflTarget_x3f_3259_ = lean_ctor_get(v_i_3126_, 2);
v_rwKind_3260_ = lean_ctor_get(v_i_3126_, 4);
v___x_3261_ = lean_unbox(v_a_3241_);
lean_dec(v_a_3241_);
lean_inc(v_rwKind_3260_);
lean_inc(v_rflTarget_x3f_3259_);
lean_inc_ref(v_subExpr_3258_);
lean_inc_ref(v_rootExpr_3257_);
v___y_3171_ = v_fst_3232_;
v___y_3172_ = v_rootExpr_3257_;
v___y_3173_ = v_fst_3230_;
v___y_3174_ = v_subExpr_3258_;
v___y_3175_ = v_rflTarget_x3f_3259_;
v___y_3176_ = v_rwKind_3260_;
v___y_3177_ = v___x_3261_;
v___y_3178_ = v_fst_3231_;
v_fst_3179_ = v_arg_3238_;
v_snd_3180_ = v_arg_3236_;
goto v___jp_3170_;
}
else
{
lean_object* v_rootExpr_3262_; lean_object* v_subExpr_3263_; lean_object* v_rflTarget_x3f_3264_; lean_object* v_rwKind_3265_; uint8_t v___x_3266_; 
v_rootExpr_3262_ = lean_ctor_get(v_i_3126_, 0);
v_subExpr_3263_ = lean_ctor_get(v_i_3126_, 1);
v_rflTarget_x3f_3264_ = lean_ctor_get(v_i_3126_, 2);
v_rwKind_3265_ = lean_ctor_get(v_i_3126_, 4);
v___x_3266_ = lean_unbox(v_a_3241_);
lean_dec(v_a_3241_);
lean_inc(v_rwKind_3265_);
lean_inc(v_rflTarget_x3f_3264_);
lean_inc_ref(v_subExpr_3263_);
lean_inc_ref(v_rootExpr_3262_);
v___y_3171_ = v_fst_3232_;
v___y_3172_ = v_rootExpr_3262_;
v___y_3173_ = v_fst_3230_;
v___y_3174_ = v_subExpr_3263_;
v___y_3175_ = v_rflTarget_x3f_3264_;
v___y_3176_ = v_rwKind_3265_;
v___y_3177_ = v___x_3266_;
v___y_3178_ = v_fst_3231_;
v_fst_3179_ = v_arg_3236_;
v_snd_3180_ = v_arg_3238_;
goto v___jp_3170_;
}
}
}
else
{
lean_object* v_a_3267_; lean_object* v___x_3269_; uint8_t v_isShared_3270_; uint8_t v_isSharedCheck_3274_; 
lean_dec_ref(v_arg_3238_);
lean_dec_ref(v_arg_3236_);
lean_dec(v_fst_3232_);
lean_dec(v_fst_3231_);
lean_dec(v_fst_3230_);
lean_dec_ref(v_i_3126_);
lean_dec_ref(v___x_3125_);
lean_dec_ref(v_lem_3124_);
v_a_3267_ = lean_ctor_get(v___x_3240_, 0);
v_isSharedCheck_3274_ = !lean_is_exclusive(v___x_3240_);
if (v_isSharedCheck_3274_ == 0)
{
v___x_3269_ = v___x_3240_;
v_isShared_3270_ = v_isSharedCheck_3274_;
goto v_resetjp_3268_;
}
else
{
lean_inc(v_a_3267_);
lean_dec(v___x_3240_);
v___x_3269_ = lean_box(0);
v_isShared_3270_ = v_isSharedCheck_3274_;
goto v_resetjp_3268_;
}
v_resetjp_3268_:
{
lean_object* v___x_3272_; 
if (v_isShared_3270_ == 0)
{
v___x_3272_ = v___x_3269_;
goto v_reusejp_3271_;
}
else
{
lean_object* v_reuseFailAlloc_3273_; 
v_reuseFailAlloc_3273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3273_, 0, v_a_3267_);
v___x_3272_ = v_reuseFailAlloc_3273_;
goto v_reusejp_3271_;
}
v_reusejp_3271_:
{
return v___x_3272_;
}
}
}
}
else
{
lean_object* v___x_3275_; 
lean_dec_ref(v_fn_3235_);
lean_dec(v_fst_3232_);
lean_dec(v_fst_3231_);
lean_dec(v_fst_3230_);
v___x_3275_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0(v___x_3234_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
lean_dec_ref_known(v___x_3234_, 2);
v___y_3214_ = v___x_3275_;
goto v___jp_3213_;
}
}
else
{
lean_object* v___x_3276_; 
lean_dec(v_fst_3232_);
lean_dec(v_fst_3231_);
lean_dec(v_fst_3230_);
v___x_3276_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___lam__0(v___x_3234_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
lean_dec_ref(v___x_3234_);
v___y_3214_ = v___x_3276_;
goto v___jp_3213_;
}
}
else
{
lean_object* v_a_3277_; lean_object* v___x_3279_; uint8_t v_isShared_3280_; uint8_t v_isSharedCheck_3284_; 
lean_dec_ref(v_i_3126_);
lean_dec_ref(v___x_3125_);
lean_dec_ref(v_lem_3124_);
v_a_3277_ = lean_ctor_get(v___x_3226_, 0);
v_isSharedCheck_3284_ = !lean_is_exclusive(v___x_3226_);
if (v_isSharedCheck_3284_ == 0)
{
v___x_3279_ = v___x_3226_;
v_isShared_3280_ = v_isSharedCheck_3284_;
goto v_resetjp_3278_;
}
else
{
lean_inc(v_a_3277_);
lean_dec(v___x_3226_);
v___x_3279_ = lean_box(0);
v_isShared_3280_ = v_isSharedCheck_3284_;
goto v_resetjp_3278_;
}
v_resetjp_3278_:
{
lean_object* v___x_3282_; 
if (v_isShared_3280_ == 0)
{
v___x_3282_ = v___x_3279_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3283_; 
v_reuseFailAlloc_3283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3283_, 0, v_a_3277_);
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
v___jp_3139_:
{
size_t v___x_3141_; size_t v___x_3142_; 
v___x_3141_ = ((size_t)1ULL);
v___x_3142_ = lean_usize_add(v_i_3130_, v___x_3141_);
lean_inc_ref(v_a_3140_);
v_i_3130_ = v___x_3142_;
v_b_3131_ = v_a_3140_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___boxed(lean_object* v_lem_3288_, lean_object* v___x_3289_, lean_object* v_i_3290_, lean_object* v_assignableMVars_3291_, lean_object* v_as_3292_, lean_object* v_sz_3293_, lean_object* v_i_3294_, lean_object* v_b_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_, lean_object* v___y_3301_, lean_object* v___y_3302_){
_start:
{
size_t v_sz_boxed_3303_; size_t v_i_boxed_3304_; lean_object* v_res_3305_; 
v_sz_boxed_3303_ = lean_unbox_usize(v_sz_3293_);
lean_dec(v_sz_3293_);
v_i_boxed_3304_ = lean_unbox_usize(v_i_3294_);
lean_dec(v_i_3294_);
v_res_3305_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9(v_lem_3288_, v___x_3289_, v_i_3290_, v_assignableMVars_3291_, v_as_3292_, v_sz_boxed_3303_, v_i_boxed_3304_, v_b_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_, v___y_3300_, v___y_3301_);
lean_dec(v___y_3301_);
lean_dec_ref(v___y_3300_);
lean_dec(v___y_3299_);
lean_dec_ref(v___y_3298_);
lean_dec(v___y_3297_);
lean_dec_ref(v___y_3296_);
lean_dec_ref(v_as_3292_);
lean_dec_ref(v_assignableMVars_3291_);
return v_res_3305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1(void){
_start:
{
lean_object* v___x_3307_; lean_object* v___x_3308_; 
v___x_3307_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__0));
v___x_3308_ = l_Lean_stringToMessageData(v___x_3307_);
return v___x_3308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try(lean_object* v_i_3309_, lean_object* v_lem_3310_, lean_object* v_assignableMVars_3311_, lean_object* v_a_3312_, lean_object* v_a_3313_, lean_object* v_a_3314_, lean_object* v_a_3315_, lean_object* v_a_3316_, lean_object* v_a_3317_){
_start:
{
lean_object* v___x_3322_; lean_object* v_mctx_3323_; lean_object* v_gpos_3324_; lean_object* v___x_3325_; size_t v_sz_3326_; size_t v___x_3327_; lean_object* v___x_3328_; 
v___x_3322_ = lean_st_ref_get(v_a_3315_);
v_mctx_3323_ = lean_ctor_get(v___x_3322_, 0);
lean_inc_ref(v_mctx_3323_);
lean_dec(v___x_3322_);
v_gpos_3324_ = lean_ctor_get(v_i_3309_, 3);
lean_inc_ref(v_gpos_3324_);
v___x_3325_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9___closed__2));
v_sz_3326_ = lean_array_size(v_gpos_3324_);
v___x_3327_ = ((size_t)0ULL);
v___x_3328_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__9(v_lem_3310_, v_mctx_3323_, v_i_3309_, v_assignableMVars_3311_, v_gpos_3324_, v_sz_3326_, v___x_3327_, v___x_3325_, v_a_3312_, v_a_3313_, v_a_3314_, v_a_3315_, v_a_3316_, v_a_3317_);
lean_dec_ref(v_gpos_3324_);
if (lean_obj_tag(v___x_3328_) == 0)
{
lean_object* v_a_3329_; lean_object* v___x_3331_; uint8_t v_isShared_3332_; uint8_t v_isSharedCheck_3339_; 
v_a_3329_ = lean_ctor_get(v___x_3328_, 0);
v_isSharedCheck_3339_ = !lean_is_exclusive(v___x_3328_);
if (v_isSharedCheck_3339_ == 0)
{
v___x_3331_ = v___x_3328_;
v_isShared_3332_ = v_isSharedCheck_3339_;
goto v_resetjp_3330_;
}
else
{
lean_inc(v_a_3329_);
lean_dec(v___x_3328_);
v___x_3331_ = lean_box(0);
v_isShared_3332_ = v_isSharedCheck_3339_;
goto v_resetjp_3330_;
}
v_resetjp_3330_:
{
lean_object* v_fst_3333_; 
v_fst_3333_ = lean_ctor_get(v_a_3329_, 0);
lean_inc(v_fst_3333_);
lean_dec(v_a_3329_);
if (lean_obj_tag(v_fst_3333_) == 0)
{
lean_del_object(v___x_3331_);
goto v___jp_3319_;
}
else
{
lean_object* v_val_3334_; 
v_val_3334_ = lean_ctor_get(v_fst_3333_, 0);
lean_inc(v_val_3334_);
lean_dec_ref_known(v_fst_3333_, 1);
if (lean_obj_tag(v_val_3334_) == 0)
{
lean_del_object(v___x_3331_);
goto v___jp_3319_;
}
else
{
lean_object* v_val_3335_; lean_object* v___x_3337_; 
v_val_3335_ = lean_ctor_get(v_val_3334_, 0);
lean_inc(v_val_3335_);
lean_dec_ref_known(v_val_3334_, 1);
if (v_isShared_3332_ == 0)
{
lean_ctor_set(v___x_3331_, 0, v_val_3335_);
v___x_3337_ = v___x_3331_;
goto v_reusejp_3336_;
}
else
{
lean_object* v_reuseFailAlloc_3338_; 
v_reuseFailAlloc_3338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3338_, 0, v_val_3335_);
v___x_3337_ = v_reuseFailAlloc_3338_;
goto v_reusejp_3336_;
}
v_reusejp_3336_:
{
return v___x_3337_;
}
}
}
}
}
else
{
lean_object* v_a_3340_; lean_object* v___x_3342_; uint8_t v_isShared_3343_; uint8_t v_isSharedCheck_3347_; 
v_a_3340_ = lean_ctor_get(v___x_3328_, 0);
v_isSharedCheck_3347_ = !lean_is_exclusive(v___x_3328_);
if (v_isSharedCheck_3347_ == 0)
{
v___x_3342_ = v___x_3328_;
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
else
{
lean_inc(v_a_3340_);
lean_dec(v___x_3328_);
v___x_3342_ = lean_box(0);
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
v_resetjp_3341_:
{
lean_object* v___x_3345_; 
if (v_isShared_3343_ == 0)
{
v___x_3345_ = v___x_3342_;
goto v_reusejp_3344_;
}
else
{
lean_object* v_reuseFailAlloc_3346_; 
v_reuseFailAlloc_3346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3346_, 0, v_a_3340_);
v___x_3345_ = v_reuseFailAlloc_3346_;
goto v_reusejp_3344_;
}
v_reusejp_3344_:
{
return v___x_3345_;
}
}
}
v___jp_3319_:
{
lean_object* v___x_3320_; lean_object* v___x_3321_; 
v___x_3320_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___closed__1);
v___x_3321_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v___x_3320_, v_a_3314_, v_a_3315_, v_a_3316_, v_a_3317_);
return v___x_3321_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try___boxed(lean_object* v_i_3348_, lean_object* v_lem_3349_, lean_object* v_assignableMVars_3350_, lean_object* v_a_3351_, lean_object* v_a_3352_, lean_object* v_a_3353_, lean_object* v_a_3354_, lean_object* v_a_3355_, lean_object* v_a_3356_, lean_object* v_a_3357_){
_start:
{
lean_object* v_res_3358_; 
v_res_3358_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_GrwLemma_try(v_i_3348_, v_lem_3349_, v_assignableMVars_3350_, v_a_3351_, v_a_3352_, v_a_3353_, v_a_3354_, v_a_3355_, v_a_3356_);
lean_dec(v_a_3356_);
lean_dec_ref(v_a_3355_);
lean_dec(v_a_3354_);
lean_dec_ref(v_a_3353_);
lean_dec(v_a_3352_);
lean_dec_ref(v_a_3351_);
lean_dec_ref(v_assignableMVars_3350_);
return v_res_3358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0(lean_object* v_mvarId_3359_, lean_object* v___y_3360_, lean_object* v___y_3361_, lean_object* v___y_3362_, lean_object* v___y_3363_, lean_object* v___y_3364_, lean_object* v___y_3365_){
_start:
{
lean_object* v___x_3367_; 
v___x_3367_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___redArg(v_mvarId_3359_, v___y_3363_);
return v___x_3367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0___boxed(lean_object* v_mvarId_3368_, lean_object* v___y_3369_, lean_object* v___y_3370_, lean_object* v___y_3371_, lean_object* v___y_3372_, lean_object* v___y_3373_, lean_object* v___y_3374_, lean_object* v___y_3375_){
_start:
{
lean_object* v_res_3376_; 
v_res_3376_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0(v_mvarId_3368_, v___y_3369_, v___y_3370_, v___y_3371_, v___y_3372_, v___y_3373_, v___y_3374_);
lean_dec(v___y_3374_);
lean_dec_ref(v___y_3373_);
lean_dec(v___y_3372_);
lean_dec_ref(v___y_3371_);
lean_dec(v___y_3370_);
lean_dec_ref(v___y_3369_);
lean_dec(v_mvarId_3368_);
return v_res_3376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2(lean_object* v_00_u03b1_3377_, lean_object* v_msg_3378_, lean_object* v___y_3379_, lean_object* v___y_3380_, lean_object* v___y_3381_, lean_object* v___y_3382_, lean_object* v___y_3383_, lean_object* v___y_3384_){
_start:
{
lean_object* v___x_3386_; 
v___x_3386_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___redArg(v_msg_3378_, v___y_3381_, v___y_3382_, v___y_3383_, v___y_3384_);
return v___x_3386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2___boxed(lean_object* v_00_u03b1_3387_, lean_object* v_msg_3388_, lean_object* v___y_3389_, lean_object* v___y_3390_, lean_object* v___y_3391_, lean_object* v___y_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_){
_start:
{
lean_object* v_res_3396_; 
v_res_3396_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__2(v_00_u03b1_3387_, v_msg_3388_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_);
lean_dec(v___y_3394_);
lean_dec_ref(v___y_3393_);
lean_dec(v___y_3392_);
lean_dec_ref(v___y_3391_);
lean_dec(v___y_3390_);
lean_dec_ref(v___y_3389_);
return v_res_3396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5(lean_object* v_as_3397_, size_t v_sz_3398_, size_t v_i_3399_, lean_object* v_b_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_, lean_object* v___y_3403_, lean_object* v___y_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_){
_start:
{
lean_object* v___x_3408_; 
v___x_3408_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___redArg(v_as_3397_, v_sz_3398_, v_i_3399_, v_b_3400_, v___y_3403_, v___y_3404_, v___y_3405_, v___y_3406_);
return v___x_3408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5___boxed(lean_object* v_as_3409_, lean_object* v_sz_3410_, lean_object* v_i_3411_, lean_object* v_b_3412_, lean_object* v___y_3413_, lean_object* v___y_3414_, lean_object* v___y_3415_, lean_object* v___y_3416_, lean_object* v___y_3417_, lean_object* v___y_3418_, lean_object* v___y_3419_){
_start:
{
size_t v_sz_boxed_3420_; size_t v_i_boxed_3421_; lean_object* v_res_3422_; 
v_sz_boxed_3420_ = lean_unbox_usize(v_sz_3410_);
lean_dec(v_sz_3410_);
v_i_boxed_3421_ = lean_unbox_usize(v_i_3411_);
lean_dec(v_i_3411_);
v_res_3422_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__5(v_as_3409_, v_sz_boxed_3420_, v_i_boxed_3421_, v_b_3412_, v___y_3413_, v___y_3414_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_);
lean_dec(v___y_3418_);
lean_dec_ref(v___y_3417_);
lean_dec(v___y_3416_);
lean_dec_ref(v___y_3415_);
lean_dec(v___y_3414_);
lean_dec_ref(v___y_3413_);
lean_dec_ref(v_as_3409_);
return v_res_3422_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0(lean_object* v_00_u03b2_3423_, lean_object* v_x_3424_, lean_object* v_x_3425_){
_start:
{
uint8_t v___x_3426_; 
v___x_3426_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___redArg(v_x_3424_, v_x_3425_);
return v___x_3426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0___boxed(lean_object* v_00_u03b2_3427_, lean_object* v_x_3428_, lean_object* v_x_3429_){
_start:
{
uint8_t v_res_3430_; lean_object* v_r_3431_; 
v_res_3430_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0(v_00_u03b2_3427_, v_x_3428_, v_x_3429_);
lean_dec(v_x_3429_);
lean_dec_ref(v_x_3428_);
v_r_3431_ = lean_box(v_res_3430_);
return v_r_3431_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4(lean_object* v_00_u03b2_3432_, lean_object* v_x_3433_, size_t v_x_3434_, lean_object* v_x_3435_){
_start:
{
uint8_t v___x_3436_; 
v___x_3436_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___redArg(v_x_3433_, v_x_3434_, v_x_3435_);
return v___x_3436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4___boxed(lean_object* v_00_u03b2_3437_, lean_object* v_x_3438_, lean_object* v_x_3439_, lean_object* v_x_3440_){
_start:
{
size_t v_x_98300__boxed_3441_; uint8_t v_res_3442_; lean_object* v_r_3443_; 
v_x_98300__boxed_3441_ = lean_unbox_usize(v_x_3439_);
lean_dec(v_x_3439_);
v_res_3442_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4(v_00_u03b2_3437_, v_x_3438_, v_x_98300__boxed_3441_, v_x_3440_);
lean_dec(v_x_3440_);
lean_dec_ref(v_x_3438_);
v_r_3443_ = lean_box(v_res_3442_);
return v_r_3443_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11(lean_object* v_00_u03b2_3444_, lean_object* v_keys_3445_, lean_object* v_vals_3446_, lean_object* v_heq_3447_, lean_object* v_i_3448_, lean_object* v_k_3449_){
_start:
{
uint8_t v___x_3450_; 
v___x_3450_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___redArg(v_keys_3445_, v_i_3448_, v_k_3449_);
return v___x_3450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11___boxed(lean_object* v_00_u03b2_3451_, lean_object* v_keys_3452_, lean_object* v_vals_3453_, lean_object* v_heq_3454_, lean_object* v_i_3455_, lean_object* v_k_3456_){
_start:
{
uint8_t v_res_3457_; lean_object* v_r_3458_; 
v_res_3457_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_GrwLemma_try_spec__0_spec__0_spec__4_spec__11(v_00_u03b2_3451_, v_keys_3452_, v_vals_3453_, v_heq_3454_, v_i_3455_, v_k_3456_);
lean_dec(v_k_3456_);
lean_dec_ref(v_vals_3453_);
lean_dec_ref(v_keys_3452_);
v_r_3458_ = lean_box(v_res_3457_);
return v_r_3458_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey_default);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedGrwKey);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
lean_object* initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_GRewrite(builtin);
}
#ifdef __cplusplus
}
#endif
