// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.Rewrite
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.SectionState public meta import Mathlib.Control.Basic public meta import Mathlib.Tactic.ClickSuggestions.Util
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_MVarId_applyRfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*);
extern lean_object* l_Lean_pp_mvars;
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_findLocalDeclWithType_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_kabstractFindsPositions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(lean_object*);
extern lean_object* l_Lean_SubExpr_Pos_root;
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Expr_toHeadIndex(lean_object*);
uint8_t l_Lean_instBEqHeadIndex_beq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headNumArgs(lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwKey_isDuplicate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwKey_isDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strong"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "goal-vdash"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__4_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__3_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__5_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__6_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__6_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⊢ "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__9_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__9_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__2_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__7_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__10_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "Expected equation, not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Exected an equality or iff, not "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__4_value),LEAN_SCALAR_PTR_LITERAL(15, 73, 51, 51, 21, 209, 204, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__5_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " and "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = " do not match according to the head-constant indexing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " does not unify with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; uint8_t v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_2_ = l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
v___x_3_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__0));
v___x_4_ = 0;
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
lean_ctor_set(v___x_6_, 2, v___x_5_);
lean_ctor_set(v___x_6_, 3, v___x_3_);
lean_ctor_set(v___x_6_, 4, v___x_2_);
lean_ctor_set_uint8(v___x_6_, sizeof(void*)*5, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default___closed__1);
return v___x_7_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default;
return v___x_8_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0(lean_object* v_a_9_, lean_object* v_b_10_){
_start:
{
lean_object* v_numGoals_11_; uint8_t v_symm_12_; lean_object* v_nameLength_13_; lean_object* v_replacementSize_14_; lean_object* v_name_15_; lean_object* v_numGoals_16_; uint8_t v_symm_17_; lean_object* v_nameLength_18_; lean_object* v_replacementSize_19_; lean_object* v_name_20_; uint8_t v___x_31_; 
v_numGoals_11_ = lean_ctor_get(v_a_9_, 0);
v_symm_12_ = lean_ctor_get_uint8(v_a_9_, sizeof(void*)*5);
v_nameLength_13_ = lean_ctor_get(v_a_9_, 1);
v_replacementSize_14_ = lean_ctor_get(v_a_9_, 2);
v_name_15_ = lean_ctor_get(v_a_9_, 3);
v_numGoals_16_ = lean_ctor_get(v_b_10_, 0);
v_symm_17_ = lean_ctor_get_uint8(v_b_10_, sizeof(void*)*5);
v_nameLength_18_ = lean_ctor_get(v_b_10_, 1);
v_replacementSize_19_ = lean_ctor_get(v_b_10_, 2);
v_name_20_ = lean_ctor_get(v_b_10_, 3);
v___x_31_ = lean_nat_dec_lt(v_numGoals_11_, v_numGoals_16_);
if (v___x_31_ == 0)
{
uint8_t v___x_32_; 
v___x_32_ = lean_nat_dec_eq(v_numGoals_11_, v_numGoals_16_);
if (v___x_32_ == 0)
{
uint8_t v___x_33_; 
v___x_33_ = 2;
return v___x_33_;
}
else
{
if (v_symm_12_ == 0)
{
if (v_symm_17_ == 1)
{
uint8_t v___x_34_; 
v___x_34_ = 0;
return v___x_34_;
}
else
{
goto v___jp_21_;
}
}
else
{
if (v_symm_17_ == 0)
{
uint8_t v___x_35_; 
v___x_35_ = 2;
return v___x_35_;
}
else
{
goto v___jp_21_;
}
}
}
}
else
{
uint8_t v___x_36_; 
v___x_36_ = 0;
return v___x_36_;
}
v___jp_21_:
{
uint8_t v___x_22_; 
v___x_22_ = lean_nat_dec_lt(v_nameLength_13_, v_nameLength_18_);
if (v___x_22_ == 0)
{
uint8_t v___x_23_; 
v___x_23_ = lean_nat_dec_eq(v_nameLength_13_, v_nameLength_18_);
if (v___x_23_ == 0)
{
uint8_t v___x_24_; 
v___x_24_ = 2;
return v___x_24_;
}
else
{
uint8_t v___x_25_; 
v___x_25_ = lean_nat_dec_lt(v_replacementSize_14_, v_replacementSize_19_);
if (v___x_25_ == 0)
{
uint8_t v___x_26_; 
v___x_26_ = lean_nat_dec_eq(v_replacementSize_14_, v_replacementSize_19_);
if (v___x_26_ == 0)
{
uint8_t v___x_27_; 
v___x_27_ = 2;
return v___x_27_;
}
else
{
uint8_t v___x_28_; 
v___x_28_ = lean_string_compare(v_name_15_, v_name_20_);
return v___x_28_;
}
}
else
{
uint8_t v___x_29_; 
v___x_29_ = 0;
return v___x_29_;
}
}
}
else
{
uint8_t v___x_30_; 
v___x_30_ = 0;
return v___x_30_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0___boxed(lean_object* v_a_37_, lean_object* v_b_38_){
_start:
{
uint8_t v_res_39_; lean_object* v_r_40_; 
v_res_39_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdRwKey___lam__0(v_a_37_, v_b_38_);
lean_dec_ref(v_b_38_);
lean_dec_ref(v_a_37_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwKey_isDuplicate(lean_object* v_a_43_, lean_object* v_b_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_replacement_50_; lean_object* v_replacement_51_; lean_object* v_mvars_52_; lean_object* v_expr_53_; lean_object* v_mvars_54_; lean_object* v_expr_55_; lean_object* v___x_56_; lean_object* v___x_57_; uint8_t v___x_58_; 
v_replacement_50_ = lean_ctor_get(v_a_43_, 4);
lean_inc_ref(v_replacement_50_);
lean_dec_ref(v_a_43_);
v_replacement_51_ = lean_ctor_get(v_b_44_, 4);
lean_inc_ref(v_replacement_51_);
lean_dec_ref(v_b_44_);
v_mvars_52_ = lean_ctor_get(v_replacement_50_, 1);
lean_inc_ref(v_mvars_52_);
v_expr_53_ = lean_ctor_get(v_replacement_50_, 2);
lean_inc_ref(v_expr_53_);
lean_dec_ref(v_replacement_50_);
v_mvars_54_ = lean_ctor_get(v_replacement_51_, 1);
lean_inc_ref(v_mvars_54_);
v_expr_55_ = lean_ctor_get(v_replacement_51_, 2);
lean_inc_ref(v_expr_55_);
lean_dec_ref(v_replacement_51_);
v___x_56_ = lean_array_get_size(v_mvars_52_);
lean_dec_ref(v_mvars_52_);
v___x_57_ = lean_array_get_size(v_mvars_54_);
lean_dec_ref(v_mvars_54_);
v___x_58_ = lean_nat_dec_eq(v___x_56_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec_ref(v_expr_55_);
lean_dec_ref(v_expr_53_);
v___x_59_ = lean_box(v___x_58_);
v___x_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
return v___x_60_;
}
else
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(v_expr_53_, v_expr_55_, v_a_45_, v_a_46_, v_a_47_, v_a_48_);
return v___x_61_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwKey_isDuplicate___boxed(lean_object* v_a_62_, lean_object* v_b_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwKey_isDuplicate(v_a_62_, v_b_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
lean_dec(v_a_67_);
lean_dec_ref(v_a_66_);
lean_dec(v_a_65_);
lean_dec_ref(v_a_64_);
return v_res_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object* v_opts_70_, lean_object* v_opt_71_){
_start:
{
lean_object* v_name_72_; lean_object* v_defValue_73_; lean_object* v_map_74_; lean_object* v___x_75_; 
v_name_72_ = lean_ctor_get(v_opt_71_, 0);
v_defValue_73_ = lean_ctor_get(v_opt_71_, 1);
v_map_74_ = lean_ctor_get(v_opts_70_, 0);
v___x_75_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_74_, v_name_72_);
if (lean_obj_tag(v___x_75_) == 0)
{
uint8_t v___x_76_; 
v___x_76_ = lean_unbox(v_defValue_73_);
return v___x_76_;
}
else
{
lean_object* v_val_77_; 
v_val_77_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_val_77_);
lean_dec_ref_known(v___x_75_, 1);
if (lean_obj_tag(v_val_77_) == 1)
{
uint8_t v_v_78_; 
v_v_78_ = lean_ctor_get_uint8(v_val_77_, 0);
lean_dec_ref_known(v_val_77_, 0);
return v_v_78_;
}
else
{
uint8_t v___x_79_; 
lean_dec(v_val_77_);
v___x_79_ = lean_unbox(v_defValue_73_);
return v___x_79_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object* v_opts_80_, lean_object* v_opt_81_){
_start:
{
uint8_t v_res_82_; lean_object* v_r_83_; 
v_res_82_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v_opts_80_, v_opt_81_);
lean_dec_ref(v_opt_81_);
lean_dec_ref(v_opts_80_);
v_r_83_ = lean_box(v_res_82_);
return v_r_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object* v_opts_84_, lean_object* v_opt_85_){
_start:
{
lean_object* v_name_86_; lean_object* v_defValue_87_; lean_object* v_map_88_; lean_object* v___x_89_; 
v_name_86_ = lean_ctor_get(v_opt_85_, 0);
v_defValue_87_ = lean_ctor_get(v_opt_85_, 1);
v_map_88_ = lean_ctor_get(v_opts_84_, 0);
v___x_89_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_88_, v_name_86_);
if (lean_obj_tag(v___x_89_) == 0)
{
lean_inc(v_defValue_87_);
return v_defValue_87_;
}
else
{
lean_object* v_val_90_; 
v_val_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc(v_val_90_);
lean_dec_ref_known(v___x_89_, 1);
if (lean_obj_tag(v_val_90_) == 3)
{
lean_object* v_v_91_; 
v_v_91_ = lean_ctor_get(v_val_90_, 0);
lean_inc(v_v_91_);
lean_dec_ref_known(v_val_90_, 1);
return v_v_91_;
}
else
{
lean_dec(v_val_90_);
lean_inc(v_defValue_87_);
return v_defValue_87_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object* v_opts_92_, lean_object* v_opt_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v_opts_92_, v_opt_93_);
lean_dec_ref(v_opt_93_);
lean_dec_ref(v_opts_92_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object* v_o_98_, lean_object* v_k_99_, uint8_t v_v_100_){
_start:
{
lean_object* v_map_101_; uint8_t v_hasTrace_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_116_; 
v_map_101_ = lean_ctor_get(v_o_98_, 0);
v_hasTrace_102_ = lean_ctor_get_uint8(v_o_98_, sizeof(void*)*1);
v_isSharedCheck_116_ = !lean_is_exclusive(v_o_98_);
if (v_isSharedCheck_116_ == 0)
{
v___x_104_ = v_o_98_;
v_isShared_105_ = v_isSharedCheck_116_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_map_101_);
lean_dec(v_o_98_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_116_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_106_, 0, v_v_100_);
lean_inc(v_k_99_);
v___x_107_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_99_, v___x_106_, v_map_101_);
if (v_hasTrace_102_ == 0)
{
lean_object* v___x_108_; uint8_t v___x_109_; lean_object* v___x_111_; 
v___x_108_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1));
v___x_109_ = l_Lean_Name_isPrefixOf(v___x_108_, v_k_99_);
lean_dec(v_k_99_);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 0, v___x_107_);
v___x_111_ = v___x_104_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___x_107_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
lean_ctor_set_uint8(v___x_111_, sizeof(void*)*1, v___x_109_);
return v___x_111_;
}
}
else
{
lean_object* v___x_114_; 
lean_dec(v_k_99_);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 0, v___x_107_);
v___x_114_ = v___x_104_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_107_);
lean_ctor_set_uint8(v_reuseFailAlloc_115_, sizeof(void*)*1, v_hasTrace_102_);
v___x_114_ = v_reuseFailAlloc_115_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
return v___x_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object* v_o_117_, lean_object* v_k_118_, lean_object* v_v_119_){
_start:
{
uint8_t v_v_boxed_120_; lean_object* v_res_121_; 
v_v_boxed_120_ = lean_unbox(v_v_119_);
v_res_121_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_o_117_, v_k_118_, v_v_boxed_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object* v_opts_122_, lean_object* v_opt_123_, uint8_t v_val_124_){
_start:
{
lean_object* v_name_125_; lean_object* v___x_126_; 
v_name_125_ = lean_ctor_get(v_opt_123_, 0);
lean_inc(v_name_125_);
lean_dec_ref(v_opt_123_);
v___x_126_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_opts_122_, v_name_125_, v_val_124_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object* v_opts_127_, lean_object* v_opt_128_, lean_object* v_val_129_){
_start:
{
uint8_t v_val_boxed_130_; lean_object* v_res_131_; 
v_val_boxed_130_ = lean_unbox(v_val_129_);
v_res_131_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_opts_127_, v_opt_128_, v_val_boxed_130_);
return v_res_131_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0(void){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_132_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0);
v___x_134_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1);
v___x_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
lean_ctor_set(v___x_136_, 1, v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object* v_lem_137_, lean_object* v_rwKind_138_, lean_object* v_hyp_x3f_139_, lean_object* v_proof_140_, uint8_t v_justLemmaName_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_proof_148_; lean_object* v___y_149_; 
if (v_justLemmaName_141_ == 0)
{
lean_object* v___x_153_; lean_object* v_fileName_154_; lean_object* v_fileMap_155_; lean_object* v_options_156_; lean_object* v_currRecDepth_157_; lean_object* v_ref_158_; lean_object* v_currNamespace_159_; lean_object* v_openDecls_160_; lean_object* v_initHeartbeats_161_; lean_object* v_maxHeartbeats_162_; lean_object* v_quotContext_163_; lean_object* v_currMacroScope_164_; lean_object* v_cancelTk_x3f_165_; uint8_t v_suppressElabErrors_166_; lean_object* v_inheritedTraceOptions_167_; lean_object* v_env_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; uint8_t v___x_173_; lean_object* v_fileName_175_; lean_object* v_fileMap_176_; lean_object* v_currRecDepth_177_; lean_object* v_ref_178_; lean_object* v_currNamespace_179_; lean_object* v_openDecls_180_; lean_object* v_initHeartbeats_181_; lean_object* v_maxHeartbeats_182_; lean_object* v_quotContext_183_; lean_object* v_currMacroScope_184_; lean_object* v_cancelTk_x3f_185_; uint8_t v_suppressElabErrors_186_; lean_object* v_inheritedTraceOptions_187_; lean_object* v___y_188_; uint8_t v___y_195_; uint8_t v___x_216_; 
v___x_153_ = lean_st_ref_get(v_a_145_);
v_fileName_154_ = lean_ctor_get(v_a_144_, 0);
v_fileMap_155_ = lean_ctor_get(v_a_144_, 1);
v_options_156_ = lean_ctor_get(v_a_144_, 2);
v_currRecDepth_157_ = lean_ctor_get(v_a_144_, 3);
v_ref_158_ = lean_ctor_get(v_a_144_, 5);
v_currNamespace_159_ = lean_ctor_get(v_a_144_, 6);
v_openDecls_160_ = lean_ctor_get(v_a_144_, 7);
v_initHeartbeats_161_ = lean_ctor_get(v_a_144_, 8);
v_maxHeartbeats_162_ = lean_ctor_get(v_a_144_, 9);
v_quotContext_163_ = lean_ctor_get(v_a_144_, 10);
v_currMacroScope_164_ = lean_ctor_get(v_a_144_, 11);
v_cancelTk_x3f_165_ = lean_ctor_get(v_a_144_, 12);
v_suppressElabErrors_166_ = lean_ctor_get_uint8(v_a_144_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_167_ = lean_ctor_get(v_a_144_, 13);
v_env_168_ = lean_ctor_get(v___x_153_, 0);
lean_inc_ref(v_env_168_);
lean_dec(v___x_153_);
v___x_169_ = lean_box(1);
v___x_170_ = l_Lean_pp_mvars;
lean_inc_ref(v_options_156_);
v___x_171_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_options_156_, v___x_170_, v_justLemmaName_141_);
v___x_172_ = l_Lean_diagnostics;
v___x_173_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v___x_171_, v___x_172_);
v___x_216_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_168_);
lean_dec_ref(v_env_168_);
if (v___x_216_ == 0)
{
if (v___x_173_ == 0)
{
v_fileName_175_ = v_fileName_154_;
v_fileMap_176_ = v_fileMap_155_;
v_currRecDepth_177_ = v_currRecDepth_157_;
v_ref_178_ = v_ref_158_;
v_currNamespace_179_ = v_currNamespace_159_;
v_openDecls_180_ = v_openDecls_160_;
v_initHeartbeats_181_ = v_initHeartbeats_161_;
v_maxHeartbeats_182_ = v_maxHeartbeats_162_;
v_quotContext_183_ = v_quotContext_163_;
v_currMacroScope_184_ = v_currMacroScope_164_;
v_cancelTk_x3f_185_ = v_cancelTk_x3f_165_;
v_suppressElabErrors_186_ = v_suppressElabErrors_166_;
v_inheritedTraceOptions_187_ = v_inheritedTraceOptions_167_;
v___y_188_ = v_a_145_;
goto v___jp_174_;
}
else
{
v___y_195_ = v___x_216_;
goto v___jp_194_;
}
}
else
{
v___y_195_ = v___x_173_;
goto v___jp_194_;
}
v___jp_174_:
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_189_ = l_Lean_maxRecDepth;
v___x_190_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v___x_171_, v___x_189_);
lean_inc_ref(v_inheritedTraceOptions_187_);
lean_inc(v_cancelTk_x3f_185_);
lean_inc(v_currMacroScope_184_);
lean_inc(v_quotContext_183_);
lean_inc(v_maxHeartbeats_182_);
lean_inc(v_initHeartbeats_181_);
lean_inc(v_openDecls_180_);
lean_inc(v_currNamespace_179_);
lean_inc(v_ref_178_);
lean_inc(v_currRecDepth_177_);
lean_inc_ref(v_fileMap_176_);
lean_inc_ref(v_fileName_175_);
v___x_191_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_191_, 0, v_fileName_175_);
lean_ctor_set(v___x_191_, 1, v_fileMap_176_);
lean_ctor_set(v___x_191_, 2, v___x_171_);
lean_ctor_set(v___x_191_, 3, v_currRecDepth_177_);
lean_ctor_set(v___x_191_, 4, v___x_190_);
lean_ctor_set(v___x_191_, 5, v_ref_178_);
lean_ctor_set(v___x_191_, 6, v_currNamespace_179_);
lean_ctor_set(v___x_191_, 7, v_openDecls_180_);
lean_ctor_set(v___x_191_, 8, v_initHeartbeats_181_);
lean_ctor_set(v___x_191_, 9, v_maxHeartbeats_182_);
lean_ctor_set(v___x_191_, 10, v_quotContext_183_);
lean_ctor_set(v___x_191_, 11, v_currMacroScope_184_);
lean_ctor_set(v___x_191_, 12, v_cancelTk_x3f_185_);
lean_ctor_set(v___x_191_, 13, v_inheritedTraceOptions_187_);
lean_ctor_set_uint8(v___x_191_, sizeof(void*)*14, v___x_173_);
lean_ctor_set_uint8(v___x_191_, sizeof(void*)*14 + 1, v_suppressElabErrors_186_);
v___x_192_ = l_Lean_PrettyPrinter_delab(v_proof_140_, v___x_169_, v_a_142_, v_a_143_, v___x_191_, v___y_188_);
lean_dec_ref_known(v___x_191_, 14);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_a_193_);
lean_dec_ref_known(v___x_192_, 1);
v_proof_148_ = v_a_193_;
v___y_149_ = v_a_144_;
goto v___jp_147_;
}
else
{
lean_dec(v_hyp_x3f_139_);
lean_dec(v_rwKind_138_);
lean_dec_ref(v_lem_137_);
return v___x_192_;
}
}
v___jp_194_:
{
if (v___y_195_ == 0)
{
lean_object* v___x_196_; lean_object* v_env_197_; lean_object* v_nextMacroScope_198_; lean_object* v_ngen_199_; lean_object* v_auxDeclNGen_200_; lean_object* v_traceState_201_; lean_object* v_messages_202_; lean_object* v_infoState_203_; lean_object* v_snapshotTasks_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_214_; 
v___x_196_ = lean_st_ref_take(v_a_145_);
v_env_197_ = lean_ctor_get(v___x_196_, 0);
v_nextMacroScope_198_ = lean_ctor_get(v___x_196_, 1);
v_ngen_199_ = lean_ctor_get(v___x_196_, 2);
v_auxDeclNGen_200_ = lean_ctor_get(v___x_196_, 3);
v_traceState_201_ = lean_ctor_get(v___x_196_, 4);
v_messages_202_ = lean_ctor_get(v___x_196_, 6);
v_infoState_203_ = lean_ctor_get(v___x_196_, 7);
v_snapshotTasks_204_ = lean_ctor_get(v___x_196_, 8);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_214_ == 0)
{
lean_object* v_unused_215_; 
v_unused_215_ = lean_ctor_get(v___x_196_, 5);
lean_dec(v_unused_215_);
v___x_206_ = v___x_196_;
v_isShared_207_ = v_isSharedCheck_214_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_snapshotTasks_204_);
lean_inc(v_infoState_203_);
lean_inc(v_messages_202_);
lean_inc(v_traceState_201_);
lean_inc(v_auxDeclNGen_200_);
lean_inc(v_ngen_199_);
lean_inc(v_nextMacroScope_198_);
lean_inc(v_env_197_);
lean_dec(v___x_196_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_214_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_208_ = l_Lean_Kernel_enableDiag(v_env_197_, v___x_173_);
v___x_209_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2);
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 5, v___x_209_);
lean_ctor_set(v___x_206_, 0, v___x_208_);
v___x_211_ = v___x_206_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v___x_208_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v_nextMacroScope_198_);
lean_ctor_set(v_reuseFailAlloc_213_, 2, v_ngen_199_);
lean_ctor_set(v_reuseFailAlloc_213_, 3, v_auxDeclNGen_200_);
lean_ctor_set(v_reuseFailAlloc_213_, 4, v_traceState_201_);
lean_ctor_set(v_reuseFailAlloc_213_, 5, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_213_, 6, v_messages_202_);
lean_ctor_set(v_reuseFailAlloc_213_, 7, v_infoState_203_);
lean_ctor_set(v_reuseFailAlloc_213_, 8, v_snapshotTasks_204_);
v___x_211_ = v_reuseFailAlloc_213_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; 
v___x_212_ = lean_st_ref_set(v_a_145_, v___x_211_);
v_fileName_175_ = v_fileName_154_;
v_fileMap_176_ = v_fileMap_155_;
v_currRecDepth_177_ = v_currRecDepth_157_;
v_ref_178_ = v_ref_158_;
v_currNamespace_179_ = v_currNamespace_159_;
v_openDecls_180_ = v_openDecls_160_;
v_initHeartbeats_181_ = v_initHeartbeats_161_;
v_maxHeartbeats_182_ = v_maxHeartbeats_162_;
v_quotContext_183_ = v_quotContext_163_;
v_currMacroScope_184_ = v_currMacroScope_164_;
v_cancelTk_x3f_185_ = v_cancelTk_x3f_165_;
v_suppressElabErrors_186_ = v_suppressElabErrors_166_;
v_inheritedTraceOptions_187_ = v_inheritedTraceOptions_167_;
v___y_188_ = v_a_145_;
goto v___jp_174_;
}
}
}
else
{
v_fileName_175_ = v_fileName_154_;
v_fileMap_176_ = v_fileMap_155_;
v_currRecDepth_177_ = v_currRecDepth_157_;
v_ref_178_ = v_ref_158_;
v_currNamespace_179_ = v_currNamespace_159_;
v_openDecls_180_ = v_openDecls_160_;
v_initHeartbeats_181_ = v_initHeartbeats_161_;
v_maxHeartbeats_182_ = v_maxHeartbeats_162_;
v_quotContext_183_ = v_quotContext_163_;
v_currMacroScope_184_ = v_currMacroScope_164_;
v_cancelTk_x3f_185_ = v_cancelTk_x3f_165_;
v_suppressElabErrors_186_ = v_suppressElabErrors_166_;
v_inheritedTraceOptions_187_ = v_inheritedTraceOptions_167_;
v___y_188_ = v_a_145_;
goto v___jp_174_;
}
}
}
else
{
lean_object* v_name_217_; lean_object* v___x_218_; 
lean_dec_ref(v_proof_140_);
v_name_217_ = lean_ctor_get(v_lem_137_, 0);
lean_inc_ref(v_name_217_);
v___x_218_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(v_name_217_, v_a_142_, v_a_143_, v_a_144_, v_a_145_);
if (lean_obj_tag(v___x_218_) == 0)
{
lean_object* v_a_219_; lean_object* v___x_220_; 
v_a_219_ = lean_ctor_get(v___x_218_, 0);
lean_inc(v_a_219_);
lean_dec_ref_known(v___x_218_, 1);
v___x_220_ = l_Lean_mkIdent(v_a_219_);
v_proof_148_ = v___x_220_;
v___y_149_ = v_a_144_;
goto v___jp_147_;
}
else
{
lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
lean_dec(v_hyp_x3f_139_);
lean_dec(v_rwKind_138_);
lean_dec_ref(v_lem_137_);
v_a_221_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_218_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_218_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
v___jp_147_:
{
uint8_t v_symm_150_; uint8_t v___x_151_; lean_object* v___x_152_; 
v_symm_150_ = lean_ctor_get_uint8(v_lem_137_, sizeof(void*)*1);
lean_dec_ref(v_lem_137_);
v___x_151_ = 0;
v___x_152_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkRewrite___redArg(v_rwKind_138_, v_symm_150_, v_proof_148_, v_hyp_x3f_139_, v___x_151_, v___y_149_);
return v___x_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object* v_lem_229_, lean_object* v_rwKind_230_, lean_object* v_hyp_x3f_231_, lean_object* v_proof_232_, lean_object* v_justLemmaName_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_, lean_object* v_a_238_){
_start:
{
uint8_t v_justLemmaName_boxed_239_; lean_object* v_res_240_; 
v_justLemmaName_boxed_239_ = lean_unbox(v_justLemmaName_233_);
v_res_240_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_229_, v_rwKind_230_, v_hyp_x3f_231_, v_proof_232_, v_justLemmaName_boxed_239_, v_a_234_, v_a_235_, v_a_236_, v_a_237_);
lean_dec(v_a_237_);
lean_dec_ref(v_a_236_);
lean_dec(v_a_235_);
lean_dec_ref(v_a_234_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(lean_object* v_e_241_, lean_object* v___y_242_){
_start:
{
uint8_t v___x_244_; 
v___x_244_ = l_Lean_Expr_hasMVar(v_e_241_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; 
v___x_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_245_, 0, v_e_241_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v_mctx_247_; lean_object* v___x_248_; lean_object* v_fst_249_; lean_object* v_snd_250_; lean_object* v___x_251_; lean_object* v_cache_252_; lean_object* v_zetaDeltaFVarIds_253_; lean_object* v_postponed_254_; lean_object* v_diag_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_264_; 
v___x_246_ = lean_st_ref_get(v___y_242_);
v_mctx_247_ = lean_ctor_get(v___x_246_, 0);
lean_inc_ref(v_mctx_247_);
lean_dec(v___x_246_);
v___x_248_ = l_Lean_instantiateMVarsCore(v_mctx_247_, v_e_241_);
v_fst_249_ = lean_ctor_get(v___x_248_, 0);
lean_inc(v_fst_249_);
v_snd_250_ = lean_ctor_get(v___x_248_, 1);
lean_inc(v_snd_250_);
lean_dec_ref(v___x_248_);
v___x_251_ = lean_st_ref_take(v___y_242_);
v_cache_252_ = lean_ctor_get(v___x_251_, 1);
v_zetaDeltaFVarIds_253_ = lean_ctor_get(v___x_251_, 2);
v_postponed_254_ = lean_ctor_get(v___x_251_, 3);
v_diag_255_ = lean_ctor_get(v___x_251_, 4);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_264_ == 0)
{
lean_object* v_unused_265_; 
v_unused_265_ = lean_ctor_get(v___x_251_, 0);
lean_dec(v_unused_265_);
v___x_257_ = v___x_251_;
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_diag_255_);
lean_inc(v_postponed_254_);
lean_inc(v_zetaDeltaFVarIds_253_);
lean_inc(v_cache_252_);
lean_dec(v___x_251_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_260_; 
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v_snd_250_);
v___x_260_ = v___x_257_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_snd_250_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v_cache_252_);
lean_ctor_set(v_reuseFailAlloc_263_, 2, v_zetaDeltaFVarIds_253_);
lean_ctor_set(v_reuseFailAlloc_263_, 3, v_postponed_254_);
lean_ctor_set(v_reuseFailAlloc_263_, 4, v_diag_255_);
v___x_260_ = v_reuseFailAlloc_263_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = lean_st_ref_set(v___y_242_, v___x_260_);
v___x_262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_262_, 0, v_fst_249_);
return v___x_262_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg___boxed(lean_object* v_e_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v_e_266_, v___y_267_);
lean_dec(v___y_267_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3(lean_object* v_e_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v_e_270_, v___y_274_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___boxed(lean_object* v_e_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3(v_e_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0(lean_object* v_k_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_296_; 
lean_inc(v___y_290_);
lean_inc_ref(v___y_289_);
v___x_296_ = lean_apply_7(v_k_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_, lean_box(0));
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0___boxed(lean_object* v_k_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0(v_k_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg(lean_object* v_k_306_, uint8_t v_allowLevelAssignments_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___f_315_; lean_object* v___x_316_; 
lean_inc(v___y_309_);
lean_inc_ref(v___y_308_);
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_315_, 0, v_k_306_);
lean_closure_set(v___f_315_, 1, v___y_308_);
lean_closure_set(v___f_315_, 2, v___y_309_);
v___x_316_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_307_, v___f_315_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
if (lean_obj_tag(v___x_316_) == 0)
{
return v___x_316_;
}
else
{
lean_object* v_a_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_324_; 
v_a_317_ = lean_ctor_get(v___x_316_, 0);
v_isSharedCheck_324_ = !lean_is_exclusive(v___x_316_);
if (v_isSharedCheck_324_ == 0)
{
v___x_319_ = v___x_316_;
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_a_317_);
lean_dec(v___x_316_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_322_; 
if (v_isShared_320_ == 0)
{
v___x_322_ = v___x_319_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v_a_317_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
return v___x_322_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg___boxed(lean_object* v_k_325_, lean_object* v_allowLevelAssignments_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_334_; lean_object* v_res_335_; 
v_allowLevelAssignments_boxed_334_ = lean_unbox(v_allowLevelAssignments_326_);
v_res_335_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg(v_k_325_, v_allowLevelAssignments_boxed_334_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_, v___y_332_);
lean_dec(v___y_332_);
lean_dec_ref(v___y_331_);
lean_dec(v___y_330_);
lean_dec_ref(v___y_329_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4(lean_object* v_00_u03b1_336_, lean_object* v_k_337_, uint8_t v_allowLevelAssignments_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg(v_k_337_, v_allowLevelAssignments_338_, v___y_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___boxed(lean_object* v_00_u03b1_347_, lean_object* v_k_348_, lean_object* v_allowLevelAssignments_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_357_; lean_object* v_res_358_; 
v_allowLevelAssignments_boxed_357_ = lean_unbox(v_allowLevelAssignments_349_);
v_res_358_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4(v_00_u03b1_347_, v_k_348_, v_allowLevelAssignments_boxed_357_, v___y_350_, v___y_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0(lean_object* v___y_359_, lean_object* v_mctx_360_, lean_object* v_cache_361_, lean_object* v_a_x3f_362_){
_start:
{
lean_object* v___x_364_; lean_object* v_zetaDeltaFVarIds_365_; lean_object* v_postponed_366_; lean_object* v_diag_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_377_; 
v___x_364_ = lean_st_ref_take(v___y_359_);
v_zetaDeltaFVarIds_365_ = lean_ctor_get(v___x_364_, 2);
v_postponed_366_ = lean_ctor_get(v___x_364_, 3);
v_diag_367_ = lean_ctor_get(v___x_364_, 4);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_377_ == 0)
{
lean_object* v_unused_378_; lean_object* v_unused_379_; 
v_unused_378_ = lean_ctor_get(v___x_364_, 1);
lean_dec(v_unused_378_);
v_unused_379_ = lean_ctor_get(v___x_364_, 0);
lean_dec(v_unused_379_);
v___x_369_ = v___x_364_;
v_isShared_370_ = v_isSharedCheck_377_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_diag_367_);
lean_inc(v_postponed_366_);
lean_inc(v_zetaDeltaFVarIds_365_);
lean_dec(v___x_364_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_377_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___x_372_; 
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 1, v_cache_361_);
lean_ctor_set(v___x_369_, 0, v_mctx_360_);
v___x_372_ = v___x_369_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_mctx_360_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v_cache_361_);
lean_ctor_set(v_reuseFailAlloc_376_, 2, v_zetaDeltaFVarIds_365_);
lean_ctor_set(v_reuseFailAlloc_376_, 3, v_postponed_366_);
lean_ctor_set(v_reuseFailAlloc_376_, 4, v_diag_367_);
v___x_372_ = v_reuseFailAlloc_376_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_373_ = lean_st_ref_set(v___y_359_, v___x_372_);
v___x_374_ = lean_box(0);
v___x_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
return v___x_375_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0___boxed(lean_object* v___y_380_, lean_object* v_mctx_381_, lean_object* v_cache_382_, lean_object* v_a_x3f_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0(v___y_380_, v_mctx_381_, v_cache_382_, v_a_x3f_383_);
lean_dec(v_a_x3f_383_);
lean_dec(v___y_380_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg(lean_object* v_x_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_mctx_396_; lean_object* v_cache_397_; lean_object* v___x_398_; 
v___x_394_ = lean_st_ref_get(v___y_390_);
v___x_395_ = lean_st_ref_get(v___y_390_);
v_mctx_396_ = lean_ctor_get(v___x_394_, 0);
lean_inc_ref(v_mctx_396_);
lean_dec(v___x_394_);
v_cache_397_ = lean_ctor_get(v___x_395_, 1);
lean_inc_ref(v_cache_397_);
lean_dec(v___x_395_);
lean_inc(v___y_392_);
lean_inc_ref(v___y_391_);
lean_inc(v___y_390_);
lean_inc_ref(v___y_389_);
lean_inc(v___y_388_);
lean_inc_ref(v___y_387_);
v___x_398_ = lean_apply_7(v_x_386_, v___y_387_, v___y_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_, lean_box(0));
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_415_; 
v_a_399_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_415_ == 0)
{
v___x_401_ = v___x_398_;
v_isShared_402_ = v_isSharedCheck_415_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_398_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_415_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
lean_inc(v_a_399_);
if (v_isShared_402_ == 0)
{
lean_ctor_set_tag(v___x_401_, 1);
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_414_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
lean_object* v___x_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
v___x_405_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0(v___y_390_, v_mctx_396_, v_cache_397_, v___x_404_);
lean_dec_ref(v___x_404_);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_405_);
if (v_isSharedCheck_412_ == 0)
{
lean_object* v_unused_413_; 
v_unused_413_ = lean_ctor_get(v___x_405_, 0);
lean_dec(v_unused_413_);
v___x_407_ = v___x_405_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_dec(v___x_405_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 0, v_a_399_);
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_399_);
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
lean_object* v_a_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_425_; 
v_a_416_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_a_416_);
lean_dec_ref_known(v___x_398_, 1);
v___x_417_ = lean_box(0);
v___x_418_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___lam__0(v___y_390_, v_mctx_396_, v_cache_397_, v___x_417_);
v_isSharedCheck_425_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_425_ == 0)
{
lean_object* v_unused_426_; 
v_unused_426_ = lean_ctor_get(v___x_418_, 0);
lean_dec(v_unused_426_);
v___x_420_ = v___x_418_;
v_isShared_421_ = v_isSharedCheck_425_;
goto v_resetjp_419_;
}
else
{
lean_dec(v___x_418_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_425_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_423_; 
if (v_isShared_421_ == 0)
{
lean_ctor_set_tag(v___x_420_, 1);
lean_ctor_set(v___x_420_, 0, v_a_416_);
v___x_423_ = v___x_420_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_424_; 
v_reuseFailAlloc_424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_424_, 0, v_a_416_);
v___x_423_ = v_reuseFailAlloc_424_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
return v___x_423_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg___boxed(lean_object* v_x_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg(v_x_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
lean_dec(v___y_433_);
lean_dec_ref(v___y_432_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8(lean_object* v_00_u03b1_436_, lean_object* v_x_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg(v_x_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___boxed(lean_object* v_00_u03b1_446_, lean_object* v_x_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8(v_00_u03b1_446_, v_x_447_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_, v___y_453_);
lean_dec(v___y_453_);
lean_dec_ref(v___y_452_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg(lean_object* v_x_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = l_Lean_Meta_saveState___redArg(v___y_460_, v___y_462_);
if (lean_obj_tag(v___x_464_) == 0)
{
lean_object* v_a_465_; lean_object* v___x_466_; 
v_a_465_ = lean_ctor_get(v___x_464_, 0);
lean_inc(v_a_465_);
lean_dec_ref_known(v___x_464_, 1);
lean_inc(v___y_462_);
lean_inc_ref(v___y_461_);
lean_inc(v___y_460_);
lean_inc_ref(v___y_459_);
lean_inc(v___y_458_);
lean_inc_ref(v___y_457_);
v___x_466_ = lean_apply_7(v_x_456_, v___y_457_, v___y_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_, lean_box(0));
if (lean_obj_tag(v___x_466_) == 0)
{
lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_475_; 
lean_dec(v_a_465_);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_466_);
if (v_isSharedCheck_475_ == 0)
{
lean_object* v_unused_476_; 
v_unused_476_ = lean_ctor_get(v___x_466_, 0);
lean_dec(v_unused_476_);
v___x_468_ = v___x_466_;
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
else
{
lean_dec(v___x_466_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
uint8_t v___x_470_; lean_object* v___x_471_; lean_object* v___x_473_; 
v___x_470_ = 1;
v___x_471_ = lean_box(v___x_470_);
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 0, v___x_471_);
v___x_473_ = v___x_468_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v___x_471_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
}
else
{
lean_object* v_a_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_506_; 
v_a_477_ = lean_ctor_get(v___x_466_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_466_);
if (v_isSharedCheck_506_ == 0)
{
v___x_479_ = v___x_466_;
v_isShared_480_ = v_isSharedCheck_506_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_a_477_);
lean_dec(v___x_466_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_506_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_482_; 
lean_inc(v_a_477_);
if (v_isShared_480_ == 0)
{
v___x_482_ = v___x_479_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_a_477_);
v___x_482_ = v_reuseFailAlloc_505_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
uint8_t v___y_484_; uint8_t v___x_503_; 
v___x_503_ = l_Lean_Exception_isInterrupt(v_a_477_);
if (v___x_503_ == 0)
{
uint8_t v___x_504_; 
v___x_504_ = l_Lean_Exception_isRuntime(v_a_477_);
v___y_484_ = v___x_504_;
goto v___jp_483_;
}
else
{
lean_dec(v_a_477_);
v___y_484_ = v___x_503_;
goto v___jp_483_;
}
v___jp_483_:
{
if (v___y_484_ == 0)
{
lean_object* v___x_485_; 
lean_dec_ref(v___x_482_);
v___x_485_ = l_Lean_Meta_SavedState_restore___redArg(v_a_465_, v___y_460_, v___y_462_);
lean_dec(v_a_465_);
if (lean_obj_tag(v___x_485_) == 0)
{
lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_493_; 
v_isSharedCheck_493_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_493_ == 0)
{
lean_object* v_unused_494_; 
v_unused_494_ = lean_ctor_get(v___x_485_, 0);
lean_dec(v_unused_494_);
v___x_487_ = v___x_485_;
v_isShared_488_ = v_isSharedCheck_493_;
goto v_resetjp_486_;
}
else
{
lean_dec(v___x_485_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_493_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_489_; lean_object* v___x_491_; 
v___x_489_ = lean_box(v___y_484_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 0, v___x_489_);
v___x_491_ = v___x_487_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v___x_489_);
v___x_491_ = v_reuseFailAlloc_492_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
return v___x_491_;
}
}
}
else
{
lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_502_; 
v_a_495_ = lean_ctor_get(v___x_485_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_502_ == 0)
{
v___x_497_ = v___x_485_;
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_485_);
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
lean_dec(v_a_465_);
return v___x_482_;
}
}
}
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
lean_dec_ref(v_x_456_);
v_a_507_ = lean_ctor_get(v___x_464_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_464_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_464_);
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
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg___boxed(lean_object* v_x_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg(v_x_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_, v___y_520_, v___y_521_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
lean_dec(v___y_519_);
lean_dec_ref(v___y_518_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9(lean_object* v_00_u03b1_524_, lean_object* v_x_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg(v_x_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___boxed(lean_object* v_00_u03b1_534_, lean_object* v_x_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9(v_00_u03b1_534_, v_x_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec(v___y_539_);
lean_dec_ref(v___y_538_);
lean_dec(v___y_537_);
lean_dec_ref(v___y_536_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0(lean_object* v_x_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_){
_start:
{
lean_object* v___x_552_; 
lean_inc(v___y_546_);
lean_inc_ref(v___y_545_);
v___x_552_ = lean_apply_7(v_x_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, lean_box(0));
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0___boxed(lean_object* v_x_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0(v_x_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_, v___y_559_);
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg(lean_object* v_mctx_562_, lean_object* v_x_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v___f_571_; lean_object* v___x_572_; 
lean_inc(v___y_565_);
lean_inc_ref(v___y_564_);
v___f_571_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_571_, 0, v_x_563_);
lean_closure_set(v___f_571_, 1, v___y_564_);
lean_closure_set(v___f_571_, 2, v___y_565_);
v___x_572_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), v_mctx_562_, v___f_571_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_572_) == 0)
{
return v___x_572_;
}
else
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_580_; 
v_a_573_ = lean_ctor_get(v___x_572_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_572_);
if (v_isSharedCheck_580_ == 0)
{
v___x_575_ = v___x_572_;
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_572_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_578_; 
if (v_isShared_576_ == 0)
{
v___x_578_ = v___x_575_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_a_573_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg___boxed(lean_object* v_mctx_581_, lean_object* v_x_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg(v_mctx_581_, v_x_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10(lean_object* v_00_u03b1_591_, lean_object* v_mctx_592_, lean_object* v_x_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg(v_mctx_592_, v_x_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___boxed(lean_object* v_00_u03b1_602_, lean_object* v_mctx_603_, lean_object* v_x_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10(v_00_u03b1_602_, v_mctx_603_, v_x_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_);
lean_dec(v___y_610_);
lean_dec_ref(v___y_609_);
lean_dec(v___y_608_);
lean_dec_ref(v___y_607_);
lean_dec(v___y_606_);
lean_dec_ref(v___y_605_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0(lean_object* v_rootExpr_613_, lean_object* v_fst_614_, lean_object* v_pos_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_kabstractFindsPositions(v_rootExpr_613_, v_fst_614_, v_pos_615_, v___y_618_, v___y_619_, v___y_620_, v___y_621_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0___boxed(lean_object* v_rootExpr_624_, lean_object* v_fst_625_, lean_object* v_pos_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0(v_rootExpr_624_, v_fst_625_, v_pos_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_);
lean_dec(v___y_632_);
lean_dec_ref(v___y_631_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
lean_dec(v_pos_626_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1(lean_object* v_a_635_, lean_object* v_val_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = l_Lean_Meta_isExprDefEq(v_a_635_, v_val_636_, v___y_639_, v___y_640_, v___y_641_, v___y_642_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1___boxed(lean_object* v_a_645_, lean_object* v_val_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1(v_a_645_, v_val_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_, v___y_652_);
lean_dec(v___y_652_);
lean_dec_ref(v___y_651_);
lean_dec(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2(lean_object* v___x_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = l_Lean_MVarId_applyRfl(v___x_655_, v___y_658_, v___y_659_, v___y_660_, v___y_661_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2___boxed(lean_object* v___x_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2(v___x_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18___redArg(lean_object* v_x_673_, lean_object* v_x_674_, lean_object* v_x_675_, lean_object* v_x_676_){
_start:
{
lean_object* v_ks_677_; lean_object* v_vs_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_702_; 
v_ks_677_ = lean_ctor_get(v_x_673_, 0);
v_vs_678_ = lean_ctor_get(v_x_673_, 1);
v_isSharedCheck_702_ = !lean_is_exclusive(v_x_673_);
if (v_isSharedCheck_702_ == 0)
{
v___x_680_ = v_x_673_;
v_isShared_681_ = v_isSharedCheck_702_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_vs_678_);
lean_inc(v_ks_677_);
lean_dec(v_x_673_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_702_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_682_; uint8_t v___x_683_; 
v___x_682_ = lean_array_get_size(v_ks_677_);
v___x_683_ = lean_nat_dec_lt(v_x_674_, v___x_682_);
if (v___x_683_ == 0)
{
lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_687_; 
lean_dec(v_x_674_);
v___x_684_ = lean_array_push(v_ks_677_, v_x_675_);
v___x_685_ = lean_array_push(v_vs_678_, v_x_676_);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v___x_685_);
lean_ctor_set(v___x_680_, 0, v___x_684_);
v___x_687_ = v___x_680_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_684_);
lean_ctor_set(v_reuseFailAlloc_688_, 1, v___x_685_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
else
{
lean_object* v_k_x27_689_; uint8_t v___x_690_; 
v_k_x27_689_ = lean_array_fget_borrowed(v_ks_677_, v_x_674_);
v___x_690_ = l_Lean_instBEqMVarId_beq(v_x_675_, v_k_x27_689_);
if (v___x_690_ == 0)
{
lean_object* v___x_692_; 
if (v_isShared_681_ == 0)
{
v___x_692_ = v___x_680_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_ks_677_);
lean_ctor_set(v_reuseFailAlloc_696_, 1, v_vs_678_);
v___x_692_ = v_reuseFailAlloc_696_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_693_ = lean_unsigned_to_nat(1u);
v___x_694_ = lean_nat_add(v_x_674_, v___x_693_);
lean_dec(v_x_674_);
v_x_673_ = v___x_692_;
v_x_674_ = v___x_694_;
goto _start;
}
}
else
{
lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_700_; 
v___x_697_ = lean_array_fset(v_ks_677_, v_x_674_, v_x_675_);
v___x_698_ = lean_array_fset(v_vs_678_, v_x_674_, v_x_676_);
lean_dec(v_x_674_);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v___x_698_);
lean_ctor_set(v___x_680_, 0, v___x_697_);
v___x_700_ = v___x_680_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v___x_697_);
lean_ctor_set(v_reuseFailAlloc_701_, 1, v___x_698_);
v___x_700_ = v_reuseFailAlloc_701_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
return v___x_700_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17___redArg(lean_object* v_n_703_, lean_object* v_k_704_, lean_object* v_v_705_){
_start:
{
lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_706_ = lean_unsigned_to_nat(0u);
v___x_707_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18___redArg(v_n_703_, v___x_706_, v_k_704_, v_v_705_);
return v___x_707_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0(void){
_start:
{
lean_object* v___x_708_; 
v___x_708_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(lean_object* v_x_709_, size_t v_x_710_, size_t v_x_711_, lean_object* v_x_712_, lean_object* v_x_713_){
_start:
{
if (lean_obj_tag(v_x_709_) == 0)
{
lean_object* v_es_714_; size_t v___x_715_; size_t v___x_716_; lean_object* v_j_717_; lean_object* v___x_718_; uint8_t v___x_719_; 
v_es_714_ = lean_ctor_get(v_x_709_, 0);
v___x_715_ = ((size_t)31ULL);
v___x_716_ = lean_usize_land(v_x_710_, v___x_715_);
v_j_717_ = lean_usize_to_nat(v___x_716_);
v___x_718_ = lean_array_get_size(v_es_714_);
v___x_719_ = lean_nat_dec_lt(v_j_717_, v___x_718_);
if (v___x_719_ == 0)
{
lean_dec(v_j_717_);
lean_dec(v_x_713_);
lean_dec(v_x_712_);
return v_x_709_;
}
else
{
lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_758_; 
lean_inc_ref(v_es_714_);
v_isSharedCheck_758_ = !lean_is_exclusive(v_x_709_);
if (v_isSharedCheck_758_ == 0)
{
lean_object* v_unused_759_; 
v_unused_759_ = lean_ctor_get(v_x_709_, 0);
lean_dec(v_unused_759_);
v___x_721_ = v_x_709_;
v_isShared_722_ = v_isSharedCheck_758_;
goto v_resetjp_720_;
}
else
{
lean_dec(v_x_709_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_758_;
goto v_resetjp_720_;
}
v_resetjp_720_:
{
lean_object* v_v_723_; lean_object* v___x_724_; lean_object* v_xs_x27_725_; lean_object* v___y_727_; 
v_v_723_ = lean_array_fget(v_es_714_, v_j_717_);
v___x_724_ = lean_box(0);
v_xs_x27_725_ = lean_array_fset(v_es_714_, v_j_717_, v___x_724_);
switch(lean_obj_tag(v_v_723_))
{
case 0:
{
lean_object* v_key_732_; lean_object* v_val_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_743_; 
v_key_732_ = lean_ctor_get(v_v_723_, 0);
v_val_733_ = lean_ctor_get(v_v_723_, 1);
v_isSharedCheck_743_ = !lean_is_exclusive(v_v_723_);
if (v_isSharedCheck_743_ == 0)
{
v___x_735_ = v_v_723_;
v_isShared_736_ = v_isSharedCheck_743_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_val_733_);
lean_inc(v_key_732_);
lean_dec(v_v_723_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_743_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
uint8_t v___x_737_; 
v___x_737_ = l_Lean_instBEqMVarId_beq(v_x_712_, v_key_732_);
if (v___x_737_ == 0)
{
lean_object* v___x_738_; lean_object* v___x_739_; 
lean_del_object(v___x_735_);
v___x_738_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_732_, v_val_733_, v_x_712_, v_x_713_);
v___x_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
v___y_727_ = v___x_739_;
goto v___jp_726_;
}
else
{
lean_object* v___x_741_; 
lean_dec(v_val_733_);
lean_dec(v_key_732_);
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 1, v_x_713_);
lean_ctor_set(v___x_735_, 0, v_x_712_);
v___x_741_ = v___x_735_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_x_712_);
lean_ctor_set(v_reuseFailAlloc_742_, 1, v_x_713_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
v___y_727_ = v___x_741_;
goto v___jp_726_;
}
}
}
}
case 1:
{
lean_object* v_node_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_756_; 
v_node_744_ = lean_ctor_get(v_v_723_, 0);
v_isSharedCheck_756_ = !lean_is_exclusive(v_v_723_);
if (v_isSharedCheck_756_ == 0)
{
v___x_746_ = v_v_723_;
v_isShared_747_ = v_isSharedCheck_756_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_node_744_);
lean_dec(v_v_723_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_756_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
size_t v___x_748_; size_t v___x_749_; size_t v___x_750_; size_t v___x_751_; lean_object* v___x_752_; lean_object* v___x_754_; 
v___x_748_ = ((size_t)5ULL);
v___x_749_ = lean_usize_shift_right(v_x_710_, v___x_748_);
v___x_750_ = ((size_t)1ULL);
v___x_751_ = lean_usize_add(v_x_711_, v___x_750_);
v___x_752_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(v_node_744_, v___x_749_, v___x_751_, v_x_712_, v_x_713_);
if (v_isShared_747_ == 0)
{
lean_ctor_set(v___x_746_, 0, v___x_752_);
v___x_754_ = v___x_746_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v___x_752_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
v___y_727_ = v___x_754_;
goto v___jp_726_;
}
}
}
default: 
{
lean_object* v___x_757_; 
v___x_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_757_, 0, v_x_712_);
lean_ctor_set(v___x_757_, 1, v_x_713_);
v___y_727_ = v___x_757_;
goto v___jp_726_;
}
}
v___jp_726_:
{
lean_object* v___x_728_; lean_object* v___x_730_; 
v___x_728_ = lean_array_fset(v_xs_x27_725_, v_j_717_, v___y_727_);
lean_dec(v_j_717_);
if (v_isShared_722_ == 0)
{
lean_ctor_set(v___x_721_, 0, v___x_728_);
v___x_730_ = v___x_721_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v___x_728_);
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
}
else
{
lean_object* v_ks_760_; lean_object* v_vs_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_781_; 
v_ks_760_ = lean_ctor_get(v_x_709_, 0);
v_vs_761_ = lean_ctor_get(v_x_709_, 1);
v_isSharedCheck_781_ = !lean_is_exclusive(v_x_709_);
if (v_isSharedCheck_781_ == 0)
{
v___x_763_ = v_x_709_;
v_isShared_764_ = v_isSharedCheck_781_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_vs_761_);
lean_inc(v_ks_760_);
lean_dec(v_x_709_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_781_;
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
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_ks_760_);
lean_ctor_set(v_reuseFailAlloc_780_, 1, v_vs_761_);
v___x_766_ = v_reuseFailAlloc_780_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
lean_object* v_newNode_767_; uint8_t v___y_769_; size_t v___x_775_; uint8_t v___x_776_; 
v_newNode_767_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17___redArg(v___x_766_, v_x_712_, v_x_713_);
v___x_775_ = ((size_t)7ULL);
v___x_776_ = lean_usize_dec_le(v___x_775_, v_x_711_);
if (v___x_776_ == 0)
{
lean_object* v___x_777_; lean_object* v___x_778_; uint8_t v___x_779_; 
v___x_777_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_767_);
v___x_778_ = lean_unsigned_to_nat(4u);
v___x_779_ = lean_nat_dec_lt(v___x_777_, v___x_778_);
lean_dec(v___x_777_);
v___y_769_ = v___x_779_;
goto v___jp_768_;
}
else
{
v___y_769_ = v___x_776_;
goto v___jp_768_;
}
v___jp_768_:
{
if (v___y_769_ == 0)
{
lean_object* v_ks_770_; lean_object* v_vs_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v_ks_770_ = lean_ctor_get(v_newNode_767_, 0);
lean_inc_ref(v_ks_770_);
v_vs_771_ = lean_ctor_get(v_newNode_767_, 1);
lean_inc_ref(v_vs_771_);
lean_dec_ref(v_newNode_767_);
v___x_772_ = lean_unsigned_to_nat(0u);
v___x_773_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___closed__0);
v___x_774_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg(v_x_711_, v_ks_770_, v_vs_771_, v___x_772_, v___x_773_);
lean_dec_ref(v_vs_771_);
lean_dec_ref(v_ks_770_);
return v___x_774_;
}
else
{
return v_newNode_767_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg(size_t v_depth_782_, lean_object* v_keys_783_, lean_object* v_vals_784_, lean_object* v_i_785_, lean_object* v_entries_786_){
_start:
{
lean_object* v___x_787_; uint8_t v___x_788_; 
v___x_787_ = lean_array_get_size(v_keys_783_);
v___x_788_ = lean_nat_dec_lt(v_i_785_, v___x_787_);
if (v___x_788_ == 0)
{
lean_dec(v_i_785_);
return v_entries_786_;
}
else
{
lean_object* v_k_789_; lean_object* v_v_790_; uint64_t v___x_791_; size_t v_h_792_; size_t v___x_793_; lean_object* v___x_794_; size_t v___x_795_; size_t v___x_796_; size_t v___x_797_; size_t v_h_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
v_k_789_ = lean_array_fget_borrowed(v_keys_783_, v_i_785_);
v_v_790_ = lean_array_fget_borrowed(v_vals_784_, v_i_785_);
v___x_791_ = l_Lean_instHashableMVarId_hash(v_k_789_);
v_h_792_ = lean_uint64_to_usize(v___x_791_);
v___x_793_ = ((size_t)5ULL);
v___x_794_ = lean_unsigned_to_nat(1u);
v___x_795_ = ((size_t)1ULL);
v___x_796_ = lean_usize_sub(v_depth_782_, v___x_795_);
v___x_797_ = lean_usize_mul(v___x_793_, v___x_796_);
v_h_798_ = lean_usize_shift_right(v_h_792_, v___x_797_);
v___x_799_ = lean_nat_add(v_i_785_, v___x_794_);
lean_dec(v_i_785_);
lean_inc(v_v_790_);
lean_inc(v_k_789_);
v___x_800_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(v_entries_786_, v_h_798_, v_depth_782_, v_k_789_, v_v_790_);
v_i_785_ = v___x_799_;
v_entries_786_ = v___x_800_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg___boxed(lean_object* v_depth_802_, lean_object* v_keys_803_, lean_object* v_vals_804_, lean_object* v_i_805_, lean_object* v_entries_806_){
_start:
{
size_t v_depth_boxed_807_; lean_object* v_res_808_; 
v_depth_boxed_807_ = lean_unbox_usize(v_depth_802_);
lean_dec(v_depth_802_);
v_res_808_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg(v_depth_boxed_807_, v_keys_803_, v_vals_804_, v_i_805_, v_entries_806_);
lean_dec_ref(v_vals_804_);
lean_dec_ref(v_keys_803_);
return v_res_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg___boxed(lean_object* v_x_809_, lean_object* v_x_810_, lean_object* v_x_811_, lean_object* v_x_812_, lean_object* v_x_813_){
_start:
{
size_t v_x_104886__boxed_814_; size_t v_x_104887__boxed_815_; lean_object* v_res_816_; 
v_x_104886__boxed_814_ = lean_unbox_usize(v_x_810_);
lean_dec(v_x_810_);
v_x_104887__boxed_815_ = lean_unbox_usize(v_x_811_);
lean_dec(v_x_811_);
v_res_816_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(v_x_809_, v_x_104886__boxed_814_, v_x_104887__boxed_815_, v_x_812_, v_x_813_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7___redArg(lean_object* v_x_817_, lean_object* v_x_818_, lean_object* v_x_819_){
_start:
{
uint64_t v___x_820_; size_t v___x_821_; size_t v___x_822_; lean_object* v___x_823_; 
v___x_820_ = l_Lean_instHashableMVarId_hash(v_x_818_);
v___x_821_ = lean_uint64_to_usize(v___x_820_);
v___x_822_ = ((size_t)1ULL);
v___x_823_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(v_x_817_, v___x_821_, v___x_822_, v_x_818_, v_x_819_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg(lean_object* v_mvarId_824_, lean_object* v_val_825_, lean_object* v___y_826_){
_start:
{
lean_object* v___x_828_; lean_object* v_mctx_829_; lean_object* v_cache_830_; lean_object* v_zetaDeltaFVarIds_831_; lean_object* v_postponed_832_; lean_object* v_diag_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_861_; 
v___x_828_ = lean_st_ref_take(v___y_826_);
v_mctx_829_ = lean_ctor_get(v___x_828_, 0);
v_cache_830_ = lean_ctor_get(v___x_828_, 1);
v_zetaDeltaFVarIds_831_ = lean_ctor_get(v___x_828_, 2);
v_postponed_832_ = lean_ctor_get(v___x_828_, 3);
v_diag_833_ = lean_ctor_get(v___x_828_, 4);
v_isSharedCheck_861_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_861_ == 0)
{
v___x_835_ = v___x_828_;
v_isShared_836_ = v_isSharedCheck_861_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_diag_833_);
lean_inc(v_postponed_832_);
lean_inc(v_zetaDeltaFVarIds_831_);
lean_inc(v_cache_830_);
lean_inc(v_mctx_829_);
lean_dec(v___x_828_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_861_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v_depth_837_; lean_object* v_levelAssignDepth_838_; lean_object* v_lmvarCounter_839_; lean_object* v_mvarCounter_840_; lean_object* v_lDecls_841_; lean_object* v_decls_842_; lean_object* v_userNames_843_; lean_object* v_lAssignment_844_; lean_object* v_eAssignment_845_; lean_object* v_dAssignment_846_; lean_object* v___x_848_; uint8_t v_isShared_849_; uint8_t v_isSharedCheck_860_; 
v_depth_837_ = lean_ctor_get(v_mctx_829_, 0);
v_levelAssignDepth_838_ = lean_ctor_get(v_mctx_829_, 1);
v_lmvarCounter_839_ = lean_ctor_get(v_mctx_829_, 2);
v_mvarCounter_840_ = lean_ctor_get(v_mctx_829_, 3);
v_lDecls_841_ = lean_ctor_get(v_mctx_829_, 4);
v_decls_842_ = lean_ctor_get(v_mctx_829_, 5);
v_userNames_843_ = lean_ctor_get(v_mctx_829_, 6);
v_lAssignment_844_ = lean_ctor_get(v_mctx_829_, 7);
v_eAssignment_845_ = lean_ctor_get(v_mctx_829_, 8);
v_dAssignment_846_ = lean_ctor_get(v_mctx_829_, 9);
v_isSharedCheck_860_ = !lean_is_exclusive(v_mctx_829_);
if (v_isSharedCheck_860_ == 0)
{
v___x_848_ = v_mctx_829_;
v_isShared_849_ = v_isSharedCheck_860_;
goto v_resetjp_847_;
}
else
{
lean_inc(v_dAssignment_846_);
lean_inc(v_eAssignment_845_);
lean_inc(v_lAssignment_844_);
lean_inc(v_userNames_843_);
lean_inc(v_decls_842_);
lean_inc(v_lDecls_841_);
lean_inc(v_mvarCounter_840_);
lean_inc(v_lmvarCounter_839_);
lean_inc(v_levelAssignDepth_838_);
lean_inc(v_depth_837_);
lean_dec(v_mctx_829_);
v___x_848_ = lean_box(0);
v_isShared_849_ = v_isSharedCheck_860_;
goto v_resetjp_847_;
}
v_resetjp_847_:
{
lean_object* v___x_850_; lean_object* v___x_852_; 
v___x_850_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7___redArg(v_eAssignment_845_, v_mvarId_824_, v_val_825_);
if (v_isShared_849_ == 0)
{
lean_ctor_set(v___x_848_, 8, v___x_850_);
v___x_852_ = v___x_848_;
goto v_reusejp_851_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_depth_837_);
lean_ctor_set(v_reuseFailAlloc_859_, 1, v_levelAssignDepth_838_);
lean_ctor_set(v_reuseFailAlloc_859_, 2, v_lmvarCounter_839_);
lean_ctor_set(v_reuseFailAlloc_859_, 3, v_mvarCounter_840_);
lean_ctor_set(v_reuseFailAlloc_859_, 4, v_lDecls_841_);
lean_ctor_set(v_reuseFailAlloc_859_, 5, v_decls_842_);
lean_ctor_set(v_reuseFailAlloc_859_, 6, v_userNames_843_);
lean_ctor_set(v_reuseFailAlloc_859_, 7, v_lAssignment_844_);
lean_ctor_set(v_reuseFailAlloc_859_, 8, v___x_850_);
lean_ctor_set(v_reuseFailAlloc_859_, 9, v_dAssignment_846_);
v___x_852_ = v_reuseFailAlloc_859_;
goto v_reusejp_851_;
}
v_reusejp_851_:
{
lean_object* v___x_854_; 
if (v_isShared_836_ == 0)
{
lean_ctor_set(v___x_835_, 0, v___x_852_);
v___x_854_ = v___x_835_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_858_; 
v_reuseFailAlloc_858_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_858_, 0, v___x_852_);
lean_ctor_set(v_reuseFailAlloc_858_, 1, v_cache_830_);
lean_ctor_set(v_reuseFailAlloc_858_, 2, v_zetaDeltaFVarIds_831_);
lean_ctor_set(v_reuseFailAlloc_858_, 3, v_postponed_832_);
lean_ctor_set(v_reuseFailAlloc_858_, 4, v_diag_833_);
v___x_854_ = v_reuseFailAlloc_858_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_st_ref_set(v___y_826_, v___x_854_);
v___x_856_ = lean_box(0);
v___x_857_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_857_, 0, v___x_856_);
return v___x_857_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg___boxed(lean_object* v_mvarId_862_, lean_object* v_val_863_, lean_object* v___y_864_, lean_object* v___y_865_){
_start:
{
lean_object* v_res_866_; 
v_res_866_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg(v_mvarId_862_, v_val_863_, v___y_864_);
lean_dec(v___y_864_);
return v_res_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0(lean_object* v_a_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = l_Lean_Meta_findLocalDeclWithType_x3f(v_a_867_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
return v___x_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0___boxed(lean_object* v_a_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_){
_start:
{
lean_object* v_res_884_; 
v_res_884_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0(v_a_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_, v___y_881_, v___y_882_);
lean_dec(v___y_882_);
lean_dec_ref(v___y_881_);
lean_dec(v___y_880_);
lean_dec_ref(v___y_879_);
lean_dec(v___y_878_);
lean_dec_ref(v___y_877_);
return v_res_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6(lean_object* v___x_885_, lean_object* v_as_886_, size_t v_sz_887_, size_t v_i_888_, lean_object* v_b_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_){
_start:
{
lean_object* v_a_898_; uint8_t v___x_902_; 
v___x_902_ = lean_usize_dec_lt(v_i_888_, v_sz_887_);
if (v___x_902_ == 0)
{
lean_object* v___x_903_; 
v___x_903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_903_, 0, v_b_889_);
return v___x_903_;
}
else
{
lean_object* v_a_904_; lean_object* v___x_905_; 
v_a_904_ = lean_array_uget_borrowed(v_as_886_, v_i_888_);
lean_inc(v_a_904_);
v___x_905_ = l_Lean_MVarId_getType(v_a_904_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_905_) == 0)
{
lean_object* v_a_906_; lean_object* v___x_907_; 
v_a_906_ = lean_ctor_get(v___x_905_, 0);
lean_inc(v_a_906_);
lean_dec_ref_known(v___x_905_, 1);
v___x_907_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v_a_906_, v___y_893_);
if (lean_obj_tag(v___x_907_) == 0)
{
lean_object* v_a_908_; lean_object* v_fst_909_; lean_object* v_snd_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_955_; 
v_a_908_ = lean_ctor_get(v___x_907_, 0);
lean_inc(v_a_908_);
lean_dec_ref_known(v___x_907_, 1);
v_fst_909_ = lean_ctor_get(v_b_889_, 0);
v_snd_910_ = lean_ctor_get(v_b_889_, 1);
v_isSharedCheck_955_ = !lean_is_exclusive(v_b_889_);
if (v_isSharedCheck_955_ == 0)
{
v___x_912_ = v_b_889_;
v_isShared_913_ = v_isSharedCheck_955_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_snd_910_);
lean_inc(v_fst_909_);
lean_dec(v_b_889_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_955_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
if (lean_obj_tag(v___x_885_) == 1)
{
lean_object* v___x_919_; 
lean_inc(v_a_908_);
v___x_919_ = l_Lean_Meta_isProp(v_a_908_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_919_) == 0)
{
lean_object* v_a_920_; uint8_t v___x_921_; 
v_a_920_ = lean_ctor_get(v___x_919_, 0);
lean_inc(v_a_920_);
lean_dec_ref_known(v___x_919_, 1);
v___x_921_ = lean_unbox(v_a_920_);
lean_dec(v_a_920_);
if (v___x_921_ == 0)
{
goto v___jp_914_;
}
else
{
uint8_t v___x_922_; lean_object* v___f_923_; lean_object* v___x_924_; 
v___x_922_ = 0;
lean_inc(v_a_908_);
v___f_923_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___lam__0___boxed), 8, 1);
lean_closure_set(v___f_923_, 0, v_a_908_);
v___x_924_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__4___redArg(v___f_923_, v___x_922_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_924_) == 0)
{
lean_object* v_a_925_; 
v_a_925_ = lean_ctor_get(v___x_924_, 0);
lean_inc(v_a_925_);
lean_dec_ref_known(v___x_924_, 1);
if (lean_obj_tag(v_a_925_) == 1)
{
lean_object* v_val_926_; lean_object* v___x_927_; lean_object* v___x_928_; 
lean_del_object(v___x_912_);
lean_dec(v_snd_910_);
lean_dec(v_a_908_);
v_val_926_ = lean_ctor_get(v_a_925_, 0);
lean_inc(v_val_926_);
lean_dec_ref_known(v_a_925_, 1);
v___x_927_ = l_Lean_Expr_fvar___override(v_val_926_);
lean_inc(v_a_904_);
v___x_928_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg(v_a_904_, v___x_927_, v___y_893_);
if (lean_obj_tag(v___x_928_) == 0)
{
lean_object* v___x_929_; lean_object* v___x_930_; 
lean_dec_ref_known(v___x_928_, 1);
v___x_929_ = lean_box(v___x_922_);
v___x_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_930_, 0, v_fst_909_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
v_a_898_ = v___x_930_;
goto v___jp_897_;
}
else
{
lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_938_; 
lean_dec(v_fst_909_);
v_a_931_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_938_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_938_ == 0)
{
v___x_933_ = v___x_928_;
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v___x_928_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_936_; 
if (v_isShared_934_ == 0)
{
v___x_936_ = v___x_933_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v_a_931_);
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
else
{
lean_dec(v_a_925_);
goto v___jp_914_;
}
}
else
{
lean_object* v_a_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_946_; 
lean_del_object(v___x_912_);
lean_dec(v_snd_910_);
lean_dec(v_fst_909_);
lean_dec(v_a_908_);
v_a_939_ = lean_ctor_get(v___x_924_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_924_);
if (v_isSharedCheck_946_ == 0)
{
v___x_941_ = v___x_924_;
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_a_939_);
lean_dec(v___x_924_);
v___x_941_ = lean_box(0);
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
v_resetjp_940_:
{
lean_object* v___x_944_; 
if (v_isShared_942_ == 0)
{
v___x_944_ = v___x_941_;
goto v_reusejp_943_;
}
else
{
lean_object* v_reuseFailAlloc_945_; 
v_reuseFailAlloc_945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_945_, 0, v_a_939_);
v___x_944_ = v_reuseFailAlloc_945_;
goto v_reusejp_943_;
}
v_reusejp_943_:
{
return v___x_944_;
}
}
}
}
}
else
{
lean_object* v_a_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_954_; 
lean_del_object(v___x_912_);
lean_dec(v_snd_910_);
lean_dec(v_fst_909_);
lean_dec(v_a_908_);
v_a_947_ = lean_ctor_get(v___x_919_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_919_);
if (v_isSharedCheck_954_ == 0)
{
v___x_949_ = v___x_919_;
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_a_947_);
lean_dec(v___x_919_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
lean_object* v___x_952_; 
if (v_isShared_950_ == 0)
{
v___x_952_ = v___x_949_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v_a_947_);
v___x_952_ = v_reuseFailAlloc_953_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
return v___x_952_;
}
}
}
}
else
{
goto v___jp_914_;
}
v___jp_914_:
{
lean_object* v___x_915_; lean_object* v___x_917_; 
v___x_915_ = lean_array_push(v_fst_909_, v_a_908_);
if (v_isShared_913_ == 0)
{
lean_ctor_set(v___x_912_, 0, v___x_915_);
v___x_917_ = v___x_912_;
goto v_reusejp_916_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v___x_915_);
lean_ctor_set(v_reuseFailAlloc_918_, 1, v_snd_910_);
v___x_917_ = v_reuseFailAlloc_918_;
goto v_reusejp_916_;
}
v_reusejp_916_:
{
v_a_898_ = v___x_917_;
goto v___jp_897_;
}
}
}
}
else
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
lean_dec_ref(v_b_889_);
v_a_956_ = lean_ctor_get(v___x_907_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_907_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_907_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_907_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
}
else
{
lean_object* v_a_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_971_; 
lean_dec_ref(v_b_889_);
v_a_964_ = lean_ctor_get(v___x_905_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_905_);
if (v_isSharedCheck_971_ == 0)
{
v___x_966_ = v___x_905_;
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_a_964_);
lean_dec(v___x_905_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v___x_969_; 
if (v_isShared_967_ == 0)
{
v___x_969_ = v___x_966_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_a_964_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
}
}
v___jp_897_:
{
size_t v___x_899_; size_t v___x_900_; 
v___x_899_ = ((size_t)1ULL);
v___x_900_ = lean_usize_add(v_i_888_, v___x_899_);
v_i_888_ = v___x_900_;
v_b_889_ = v_a_898_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6___boxed(lean_object* v___x_972_, lean_object* v_as_973_, lean_object* v_sz_974_, lean_object* v_i_975_, lean_object* v_b_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_){
_start:
{
size_t v_sz_boxed_984_; size_t v_i_boxed_985_; lean_object* v_res_986_; 
v_sz_boxed_984_ = lean_unbox_usize(v_sz_974_);
lean_dec(v_sz_974_);
v_i_boxed_985_ = lean_unbox_usize(v_i_975_);
lean_dec(v_i_975_);
v_res_986_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6(v___x_972_, v_as_973_, v_sz_boxed_984_, v_i_boxed_985_, v_b_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec(v___y_978_);
lean_dec_ref(v___y_977_);
lean_dec_ref(v_as_973_);
lean_dec(v___x_972_);
return v_res_986_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg(lean_object* v_keys_987_, lean_object* v_i_988_, lean_object* v_k_989_){
_start:
{
lean_object* v___x_990_; uint8_t v___x_991_; 
v___x_990_ = lean_array_get_size(v_keys_987_);
v___x_991_ = lean_nat_dec_lt(v_i_988_, v___x_990_);
if (v___x_991_ == 0)
{
lean_dec(v_i_988_);
return v___x_991_;
}
else
{
lean_object* v_k_x27_992_; uint8_t v___x_993_; 
v_k_x27_992_ = lean_array_fget_borrowed(v_keys_987_, v_i_988_);
v___x_993_ = l_Lean_instBEqMVarId_beq(v_k_989_, v_k_x27_992_);
if (v___x_993_ == 0)
{
lean_object* v___x_994_; lean_object* v___x_995_; 
v___x_994_ = lean_unsigned_to_nat(1u);
v___x_995_ = lean_nat_add(v_i_988_, v___x_994_);
lean_dec(v_i_988_);
v_i_988_ = v___x_995_;
goto _start;
}
else
{
lean_dec(v_i_988_);
return v___x_993_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg___boxed(lean_object* v_keys_997_, lean_object* v_i_998_, lean_object* v_k_999_){
_start:
{
uint8_t v_res_1000_; lean_object* v_r_1001_; 
v_res_1000_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg(v_keys_997_, v_i_998_, v_k_999_);
lean_dec(v_k_999_);
lean_dec_ref(v_keys_997_);
v_r_1001_ = lean_box(v_res_1000_);
return v_r_1001_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg(lean_object* v_x_1002_, size_t v_x_1003_, lean_object* v_x_1004_){
_start:
{
if (lean_obj_tag(v_x_1002_) == 0)
{
lean_object* v_es_1005_; lean_object* v___x_1006_; size_t v___x_1007_; size_t v___x_1008_; lean_object* v_j_1009_; lean_object* v___x_1010_; 
v_es_1005_ = lean_ctor_get(v_x_1002_, 0);
v___x_1006_ = lean_box(2);
v___x_1007_ = ((size_t)31ULL);
v___x_1008_ = lean_usize_land(v_x_1003_, v___x_1007_);
v_j_1009_ = lean_usize_to_nat(v___x_1008_);
v___x_1010_ = lean_array_get_borrowed(v___x_1006_, v_es_1005_, v_j_1009_);
lean_dec(v_j_1009_);
switch(lean_obj_tag(v___x_1010_))
{
case 0:
{
lean_object* v_key_1011_; uint8_t v___x_1012_; 
v_key_1011_ = lean_ctor_get(v___x_1010_, 0);
v___x_1012_ = l_Lean_instBEqMVarId_beq(v_x_1004_, v_key_1011_);
return v___x_1012_;
}
case 1:
{
lean_object* v_node_1013_; size_t v___x_1014_; size_t v___x_1015_; 
v_node_1013_ = lean_ctor_get(v___x_1010_, 0);
v___x_1014_ = ((size_t)5ULL);
v___x_1015_ = lean_usize_shift_right(v_x_1003_, v___x_1014_);
v_x_1002_ = v_node_1013_;
v_x_1003_ = v___x_1015_;
goto _start;
}
default: 
{
uint8_t v___x_1017_; 
v___x_1017_ = 0;
return v___x_1017_;
}
}
}
else
{
lean_object* v_ks_1018_; lean_object* v___x_1019_; uint8_t v___x_1020_; 
v_ks_1018_ = lean_ctor_get(v_x_1002_, 0);
v___x_1019_ = lean_unsigned_to_nat(0u);
v___x_1020_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg(v_ks_1018_, v___x_1019_, v_x_1004_);
return v___x_1020_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg___boxed(lean_object* v_x_1021_, lean_object* v_x_1022_, lean_object* v_x_1023_){
_start:
{
size_t v_x_105309__boxed_1024_; uint8_t v_res_1025_; lean_object* v_r_1026_; 
v_x_105309__boxed_1024_ = lean_unbox_usize(v_x_1022_);
lean_dec(v_x_1022_);
v_res_1025_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg(v_x_1021_, v_x_105309__boxed_1024_, v_x_1023_);
lean_dec(v_x_1023_);
lean_dec_ref(v_x_1021_);
v_r_1026_ = lean_box(v_res_1025_);
return v_r_1026_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg(lean_object* v_x_1027_, lean_object* v_x_1028_){
_start:
{
uint64_t v___x_1029_; size_t v___x_1030_; uint8_t v___x_1031_; 
v___x_1029_ = l_Lean_instHashableMVarId_hash(v_x_1028_);
v___x_1030_ = lean_uint64_to_usize(v___x_1029_);
v___x_1031_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg(v_x_1027_, v___x_1030_, v_x_1028_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg___boxed(lean_object* v_x_1032_, lean_object* v_x_1033_){
_start:
{
uint8_t v_res_1034_; lean_object* v_r_1035_; 
v_res_1034_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg(v_x_1032_, v_x_1033_);
lean_dec(v_x_1033_);
lean_dec_ref(v_x_1032_);
v_r_1035_ = lean_box(v_res_1034_);
return v_r_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg(lean_object* v_mvarId_1036_, lean_object* v___y_1037_){
_start:
{
lean_object* v___x_1039_; lean_object* v_mctx_1040_; lean_object* v_eAssignment_1041_; uint8_t v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1039_ = lean_st_ref_get(v___y_1037_);
v_mctx_1040_ = lean_ctor_get(v___x_1039_, 0);
lean_inc_ref(v_mctx_1040_);
lean_dec(v___x_1039_);
v_eAssignment_1041_ = lean_ctor_get(v_mctx_1040_, 8);
lean_inc_ref(v_eAssignment_1041_);
lean_dec_ref(v_mctx_1040_);
v___x_1042_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg(v_eAssignment_1041_, v_mvarId_1036_);
lean_dec_ref(v_eAssignment_1041_);
v___x_1043_ = lean_box(v___x_1042_);
v___x_1044_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1044_, 0, v___x_1043_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg___boxed(lean_object* v_mvarId_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_){
_start:
{
lean_object* v_res_1048_; 
v_res_1048_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg(v_mvarId_1045_, v___y_1046_);
lean_dec(v___y_1046_);
lean_dec(v_mvarId_1045_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11(lean_object* v_as_1049_, size_t v_i_1050_, size_t v_stop_1051_, lean_object* v_b_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_){
_start:
{
lean_object* v_a_1061_; uint8_t v___x_1065_; 
v___x_1065_ = lean_usize_dec_eq(v_i_1050_, v_stop_1051_);
if (v___x_1065_ == 0)
{
lean_object* v___x_1066_; lean_object* v___x_1069_; 
v___x_1066_ = lean_array_uget_borrowed(v_as_1049_, v_i_1050_);
v___x_1069_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg(v___x_1066_, v___y_1056_);
if (lean_obj_tag(v___x_1069_) == 0)
{
lean_object* v_a_1070_; uint8_t v___x_1071_; 
v_a_1070_ = lean_ctor_get(v___x_1069_, 0);
lean_inc(v_a_1070_);
lean_dec_ref_known(v___x_1069_, 1);
v___x_1071_ = lean_unbox(v_a_1070_);
lean_dec(v_a_1070_);
if (v___x_1071_ == 0)
{
goto v___jp_1067_;
}
else
{
v_a_1061_ = v_b_1052_;
goto v___jp_1060_;
}
}
else
{
if (lean_obj_tag(v___x_1069_) == 0)
{
lean_object* v_a_1072_; uint8_t v___x_1073_; 
v_a_1072_ = lean_ctor_get(v___x_1069_, 0);
lean_inc(v_a_1072_);
lean_dec_ref_known(v___x_1069_, 1);
v___x_1073_ = lean_unbox(v_a_1072_);
lean_dec(v_a_1072_);
if (v___x_1073_ == 0)
{
v_a_1061_ = v_b_1052_;
goto v___jp_1060_;
}
else
{
goto v___jp_1067_;
}
}
else
{
lean_object* v_a_1074_; lean_object* v___x_1076_; uint8_t v_isShared_1077_; uint8_t v_isSharedCheck_1081_; 
lean_dec_ref(v_b_1052_);
v_a_1074_ = lean_ctor_get(v___x_1069_, 0);
v_isSharedCheck_1081_ = !lean_is_exclusive(v___x_1069_);
if (v_isSharedCheck_1081_ == 0)
{
v___x_1076_ = v___x_1069_;
v_isShared_1077_ = v_isSharedCheck_1081_;
goto v_resetjp_1075_;
}
else
{
lean_inc(v_a_1074_);
lean_dec(v___x_1069_);
v___x_1076_ = lean_box(0);
v_isShared_1077_ = v_isSharedCheck_1081_;
goto v_resetjp_1075_;
}
v_resetjp_1075_:
{
lean_object* v___x_1079_; 
if (v_isShared_1077_ == 0)
{
v___x_1079_ = v___x_1076_;
goto v_reusejp_1078_;
}
else
{
lean_object* v_reuseFailAlloc_1080_; 
v_reuseFailAlloc_1080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1080_, 0, v_a_1074_);
v___x_1079_ = v_reuseFailAlloc_1080_;
goto v_reusejp_1078_;
}
v_reusejp_1078_:
{
return v___x_1079_;
}
}
}
}
v___jp_1067_:
{
lean_object* v___x_1068_; 
lean_inc(v___x_1066_);
v___x_1068_ = lean_array_push(v_b_1052_, v___x_1066_);
v_a_1061_ = v___x_1068_;
goto v___jp_1060_;
}
}
else
{
lean_object* v___x_1082_; 
v___x_1082_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1082_, 0, v_b_1052_);
return v___x_1082_;
}
v___jp_1060_:
{
size_t v___x_1062_; size_t v___x_1063_; 
v___x_1062_ = ((size_t)1ULL);
v___x_1063_ = lean_usize_add(v_i_1050_, v___x_1062_);
v_i_1050_ = v___x_1063_;
v_b_1052_ = v_a_1061_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11___boxed(lean_object* v_as_1083_, lean_object* v_i_1084_, lean_object* v_stop_1085_, lean_object* v_b_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
size_t v_i_boxed_1094_; size_t v_stop_boxed_1095_; lean_object* v_res_1096_; 
v_i_boxed_1094_ = lean_unbox_usize(v_i_1084_);
lean_dec(v_i_1084_);
v_stop_boxed_1095_ = lean_unbox_usize(v_stop_1085_);
lean_dec(v_stop_1085_);
v_res_1096_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11(v_as_1083_, v_i_boxed_1094_, v_stop_boxed_1095_, v_b_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
lean_dec(v___y_1092_);
lean_dec_ref(v___y_1091_);
lean_dec(v___y_1090_);
lean_dec_ref(v___y_1089_);
lean_dec(v___y_1088_);
lean_dec_ref(v___y_1087_);
lean_dec_ref(v_as_1083_);
return v_res_1096_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12(void){
_start:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; 
v___x_1123_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__11));
v___x_1124_ = lean_unsigned_to_nat(2u);
v___x_1125_ = lean_mk_empty_array_with_capacity(v___x_1124_);
v___x_1126_ = lean_array_push(v___x_1125_, v___x_1123_);
return v___x_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg(lean_object* v_as_1127_, size_t v_sz_1128_, size_t v_i_1129_, lean_object* v_b_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_){
_start:
{
uint8_t v___x_1136_; 
v___x_1136_ = lean_usize_dec_lt(v_i_1129_, v_sz_1128_);
if (v___x_1136_ == 0)
{
lean_object* v___x_1137_; 
v___x_1137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1137_, 0, v_b_1130_);
return v___x_1137_;
}
else
{
lean_object* v_a_1138_; lean_object* v___x_1139_; 
v_a_1138_ = lean_array_uget_borrowed(v_as_1127_, v_i_1129_);
lean_inc(v_a_1138_);
v___x_1139_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_a_1138_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_);
if (lean_obj_tag(v___x_1139_) == 0)
{
lean_object* v_a_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; size_t v___x_1147_; size_t v___x_1148_; 
v_a_1140_ = lean_ctor_get(v___x_1139_, 0);
lean_inc(v_a_1140_);
lean_dec_ref_known(v___x_1139_, 1);
v___x_1141_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0));
v___x_1142_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__1));
v___x_1143_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__12);
v___x_1144_ = lean_array_push(v___x_1143_, v_a_1140_);
v___x_1145_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1145_, 0, v___x_1141_);
lean_ctor_set(v___x_1145_, 1, v___x_1142_);
lean_ctor_set(v___x_1145_, 2, v___x_1144_);
v___x_1146_ = lean_array_push(v_b_1130_, v___x_1145_);
v___x_1147_ = ((size_t)1ULL);
v___x_1148_ = lean_usize_add(v_i_1129_, v___x_1147_);
v_i_1129_ = v___x_1148_;
v_b_1130_ = v___x_1146_;
goto _start;
}
else
{
lean_object* v_a_1150_; lean_object* v___x_1152_; uint8_t v_isShared_1153_; uint8_t v_isSharedCheck_1157_; 
lean_dec_ref(v_b_1130_);
v_a_1150_ = lean_ctor_get(v___x_1139_, 0);
v_isSharedCheck_1157_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1157_ == 0)
{
v___x_1152_ = v___x_1139_;
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
else
{
lean_inc(v_a_1150_);
lean_dec(v___x_1139_);
v___x_1152_ = lean_box(0);
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
v_resetjp_1151_:
{
lean_object* v___x_1155_; 
if (v_isShared_1153_ == 0)
{
v___x_1155_ = v___x_1152_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1156_; 
v_reuseFailAlloc_1156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1156_, 0, v_a_1150_);
v___x_1155_ = v_reuseFailAlloc_1156_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
return v___x_1155_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___boxed(lean_object* v_as_1158_, lean_object* v_sz_1159_, lean_object* v_i_1160_, lean_object* v_b_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
size_t v_sz_boxed_1167_; size_t v_i_boxed_1168_; lean_object* v_res_1169_; 
v_sz_boxed_1167_ = lean_unbox_usize(v_sz_1159_);
lean_dec(v_sz_1159_);
v_i_boxed_1168_ = lean_unbox_usize(v_i_1160_);
lean_dec(v_i_1160_);
v_res_1169_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg(v_as_1158_, v_sz_boxed_1167_, v_i_boxed_1168_, v_b_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec_ref(v_as_1158_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2(size_t v_sz_1170_, size_t v_i_1171_, lean_object* v_bs_1172_){
_start:
{
uint8_t v___x_1173_; 
v___x_1173_ = lean_usize_dec_lt(v_i_1171_, v_sz_1170_);
if (v___x_1173_ == 0)
{
return v_bs_1172_;
}
else
{
lean_object* v_v_1174_; lean_object* v___x_1175_; lean_object* v_bs_x27_1176_; lean_object* v___x_1177_; size_t v___x_1178_; size_t v___x_1179_; lean_object* v___x_1180_; 
v_v_1174_ = lean_array_uget(v_bs_1172_, v_i_1171_);
v___x_1175_ = lean_unsigned_to_nat(0u);
v_bs_x27_1176_ = lean_array_uset(v_bs_1172_, v_i_1171_, v___x_1175_);
v___x_1177_ = l_Lean_Expr_mvarId_x21(v_v_1174_);
lean_dec(v_v_1174_);
v___x_1178_ = ((size_t)1ULL);
v___x_1179_ = lean_usize_add(v_i_1171_, v___x_1178_);
v___x_1180_ = lean_array_uset(v_bs_x27_1176_, v_i_1171_, v___x_1177_);
v_i_1171_ = v___x_1179_;
v_bs_1172_ = v___x_1180_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2___boxed(lean_object* v_sz_1182_, lean_object* v_i_1183_, lean_object* v_bs_1184_){
_start:
{
size_t v_sz_boxed_1185_; size_t v_i_boxed_1186_; lean_object* v_res_1187_; 
v_sz_boxed_1185_ = lean_unbox_usize(v_sz_1182_);
lean_dec(v_sz_1182_);
v_i_boxed_1186_ = lean_unbox_usize(v_i_1183_);
lean_dec(v_i_1183_);
v_res_1187_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2(v_sz_boxed_1185_, v_i_boxed_1186_, v_bs_1184_);
return v_res_1187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2(lean_object* v_msgData_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v___x_1194_; lean_object* v_env_1195_; lean_object* v___x_1196_; lean_object* v_mctx_1197_; lean_object* v_lctx_1198_; lean_object* v_options_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1194_ = lean_st_ref_get(v___y_1192_);
v_env_1195_ = lean_ctor_get(v___x_1194_, 0);
lean_inc_ref(v_env_1195_);
lean_dec(v___x_1194_);
v___x_1196_ = lean_st_ref_get(v___y_1190_);
v_mctx_1197_ = lean_ctor_get(v___x_1196_, 0);
lean_inc_ref(v_mctx_1197_);
lean_dec(v___x_1196_);
v_lctx_1198_ = lean_ctor_get(v___y_1189_, 2);
v_options_1199_ = lean_ctor_get(v___y_1191_, 2);
lean_inc_ref(v_options_1199_);
lean_inc_ref(v_lctx_1198_);
v___x_1200_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1200_, 0, v_env_1195_);
lean_ctor_set(v___x_1200_, 1, v_mctx_1197_);
lean_ctor_set(v___x_1200_, 2, v_lctx_1198_);
lean_ctor_set(v___x_1200_, 3, v_options_1199_);
v___x_1201_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
lean_ctor_set(v___x_1201_, 1, v_msgData_1188_);
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
return v___x_1202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2___boxed(lean_object* v_msgData_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_){
_start:
{
lean_object* v_res_1209_; 
v_res_1209_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2(v_msgData_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_);
lean_dec(v___y_1207_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
return v_res_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(lean_object* v_msg_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v_ref_1216_; lean_object* v___x_1217_; lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1226_; 
v_ref_1216_ = lean_ctor_get(v___y_1213_, 5);
v___x_1217_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1_spec__2(v_msg_1210_, v___y_1211_, v___y_1212_, v___y_1213_, v___y_1214_);
v_a_1218_ = lean_ctor_get(v___x_1217_, 0);
v_isSharedCheck_1226_ = !lean_is_exclusive(v___x_1217_);
if (v_isSharedCheck_1226_ == 0)
{
v___x_1220_ = v___x_1217_;
v_isShared_1221_ = v_isSharedCheck_1226_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1217_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1226_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v___x_1222_; lean_object* v___x_1224_; 
lean_inc(v_ref_1216_);
v___x_1222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1222_, 0, v_ref_1216_);
lean_ctor_set(v___x_1222_, 1, v_a_1218_);
if (v_isShared_1221_ == 0)
{
lean_ctor_set_tag(v___x_1220_, 1);
lean_ctor_set(v___x_1220_, 0, v___x_1222_);
v___x_1224_ = v___x_1220_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v___x_1222_);
v___x_1224_ = v_reuseFailAlloc_1225_;
goto v_reusejp_1223_;
}
v_reusejp_1223_:
{
return v___x_1224_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg___boxed(lean_object* v_msg_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v_msg_1227_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
return v_res_1233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1(void){
_start:
{
lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1235_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__0));
v___x_1236_ = l_Lean_stringToMessageData(v___x_1235_);
return v___x_1236_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3(void){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; 
v___x_1238_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__2));
v___x_1239_ = l_Lean_stringToMessageData(v___x_1238_);
return v___x_1239_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8(void){
_start:
{
lean_object* v___x_1246_; lean_object* v___x_1247_; 
v___x_1246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__7));
v___x_1247_ = l_Lean_stringToMessageData(v___x_1246_);
return v___x_1247_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10(void){
_start:
{
lean_object* v___x_1249_; lean_object* v___x_1250_; 
v___x_1249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__9));
v___x_1250_ = l_Lean_stringToMessageData(v___x_1249_);
return v___x_1250_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12(void){
_start:
{
lean_object* v___x_1252_; lean_object* v___x_1253_; 
v___x_1252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__11));
v___x_1253_ = l_Lean_stringToMessageData(v___x_1252_);
return v___x_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try(lean_object* v_i_1254_, lean_object* v_lem_1255_, lean_object* v_assignableMVars_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_, lean_object* v_a_1262_){
_start:
{
lean_object* v___y_1265_; lean_object* v___y_1266_; lean_object* v___y_1267_; lean_object* v_pattern_1268_; lean_object* v___y_1272_; lean_object* v___y_1273_; lean_object* v___y_1274_; lean_object* v___y_1275_; lean_object* v___y_1276_; lean_object* v___y_1277_; lean_object* v___y_1278_; lean_object* v___y_1279_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v___y_1297_; lean_object* v___y_1298_; lean_object* v___y_1299_; lean_object* v___y_1300_; lean_object* v___y_1301_; lean_object* v___y_1302_; lean_object* v_name_1313_; uint8_t v_symm_1314_; lean_object* v___y_1316_; lean_object* v___y_1317_; uint8_t v___y_1318_; lean_object* v___y_1319_; lean_object* v___y_1320_; lean_object* v___y_1321_; lean_object* v_filtered_1322_; lean_object* v___y_1323_; lean_object* v___y_1324_; lean_object* v___y_1325_; lean_object* v___y_1326_; lean_object* v___y_1327_; lean_object* v___y_1328_; lean_object* v___y_1393_; lean_object* v___y_1394_; lean_object* v___y_1395_; uint8_t v___y_1396_; lean_object* v___y_1397_; lean_object* v___y_1398_; lean_object* v___y_1399_; lean_object* v___y_1400_; lean_object* v___y_1401_; lean_object* v___y_1402_; lean_object* v___y_1403_; lean_object* v___y_1404_; lean_object* v___y_1407_; uint8_t v___y_1408_; lean_object* v___y_1409_; uint8_t v___y_1410_; uint8_t v___y_1411_; lean_object* v___y_1412_; lean_object* v___y_1413_; size_t v___y_1414_; lean_object* v___y_1415_; lean_object* v___y_1416_; lean_object* v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___y_1421_; lean_object* v___y_1463_; uint8_t v___y_1464_; lean_object* v___y_1465_; lean_object* v___y_1466_; lean_object* v___y_1467_; size_t v___y_1468_; lean_object* v___y_1469_; uint8_t v___y_1470_; lean_object* v___y_1471_; lean_object* v___y_1472_; lean_object* v___y_1473_; lean_object* v___y_1474_; lean_object* v___y_1475_; lean_object* v___y_1476_; uint8_t v_a_1477_; lean_object* v___y_1488_; uint8_t v___y_1489_; lean_object* v___y_1490_; lean_object* v___y_1491_; lean_object* v___y_1492_; size_t v___y_1493_; lean_object* v___y_1494_; uint8_t v___y_1495_; lean_object* v___y_1496_; lean_object* v___y_1497_; lean_object* v___y_1498_; lean_object* v___y_1499_; lean_object* v___y_1500_; lean_object* v___y_1501_; lean_object* v___y_1502_; lean_object* v___x_1513_; 
v_name_1313_ = lean_ctor_get(v_lem_1255_, 0);
lean_inc_ref_n(v_name_1313_, 2);
v_symm_1314_ = lean_ctor_get_uint8(v_lem_1255_, sizeof(void*)*1);
v___x_1513_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(v_name_1313_, v_a_1259_, v_a_1260_, v_a_1261_, v_a_1262_);
if (lean_obj_tag(v___x_1513_) == 0)
{
lean_object* v_a_1514_; lean_object* v_snd_1515_; lean_object* v_snd_1516_; lean_object* v_fst_1517_; lean_object* v___x_1519_; uint8_t v_isShared_1520_; uint8_t v_isSharedCheck_1883_; 
v_a_1514_ = lean_ctor_get(v___x_1513_, 0);
lean_inc(v_a_1514_);
lean_dec_ref_known(v___x_1513_, 1);
v_snd_1515_ = lean_ctor_get(v_a_1514_, 1);
lean_inc(v_snd_1515_);
v_snd_1516_ = lean_ctor_get(v_snd_1515_, 1);
lean_inc(v_snd_1516_);
v_fst_1517_ = lean_ctor_get(v_a_1514_, 0);
v_isSharedCheck_1883_ = !lean_is_exclusive(v_a_1514_);
if (v_isSharedCheck_1883_ == 0)
{
lean_object* v_unused_1884_; 
v_unused_1884_ = lean_ctor_get(v_a_1514_, 1);
lean_dec(v_unused_1884_);
v___x_1519_ = v_a_1514_;
v_isShared_1520_ = v_isSharedCheck_1883_;
goto v_resetjp_1518_;
}
else
{
lean_inc(v_fst_1517_);
lean_dec(v_a_1514_);
v___x_1519_ = lean_box(0);
v_isShared_1520_ = v_isSharedCheck_1883_;
goto v_resetjp_1518_;
}
v_resetjp_1518_:
{
lean_object* v_fst_1521_; lean_object* v___x_1523_; uint8_t v_isShared_1524_; uint8_t v_isSharedCheck_1881_; 
v_fst_1521_ = lean_ctor_get(v_snd_1515_, 0);
v_isSharedCheck_1881_ = !lean_is_exclusive(v_snd_1515_);
if (v_isSharedCheck_1881_ == 0)
{
lean_object* v_unused_1882_; 
v_unused_1882_ = lean_ctor_get(v_snd_1515_, 1);
lean_dec(v_unused_1882_);
v___x_1523_ = v_snd_1515_;
v_isShared_1524_ = v_isSharedCheck_1881_;
goto v_resetjp_1522_;
}
else
{
lean_inc(v_fst_1521_);
lean_dec(v_snd_1515_);
v___x_1523_ = lean_box(0);
v_isShared_1524_ = v_isSharedCheck_1881_;
goto v_resetjp_1522_;
}
v_resetjp_1522_:
{
lean_object* v_fst_1525_; lean_object* v_snd_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1880_; 
v_fst_1525_ = lean_ctor_get(v_snd_1516_, 0);
v_snd_1526_ = lean_ctor_get(v_snd_1516_, 1);
v_isSharedCheck_1880_ = !lean_is_exclusive(v_snd_1516_);
if (v_isSharedCheck_1880_ == 0)
{
v___x_1528_ = v_snd_1516_;
v_isShared_1529_ = v_isSharedCheck_1880_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_snd_1526_);
lean_inc(v_fst_1525_);
lean_dec(v_snd_1516_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1880_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
lean_object* v___x_1530_; 
lean_inc(v_a_1262_);
lean_inc_ref(v_a_1261_);
lean_inc(v_a_1260_);
lean_inc_ref(v_a_1259_);
lean_inc(v_snd_1526_);
v___x_1530_ = lean_whnf(v_snd_1526_, v_a_1259_, v_a_1260_, v_a_1261_, v_a_1262_);
if (lean_obj_tag(v___x_1530_) == 0)
{
lean_object* v_a_1531_; lean_object* v___y_1533_; lean_object* v___y_1534_; lean_object* v___y_1535_; lean_object* v___y_1536_; lean_object* v___y_1537_; lean_object* v___y_1538_; 
v_a_1531_ = lean_ctor_get(v___x_1530_, 0);
lean_inc(v_a_1531_);
lean_dec_ref_known(v___x_1530_, 1);
if (lean_obj_tag(v_a_1531_) == 5)
{
lean_object* v_fn_1545_; 
v_fn_1545_ = lean_ctor_get(v_a_1531_, 0);
lean_inc_ref(v_fn_1545_);
if (lean_obj_tag(v_fn_1545_) == 5)
{
lean_object* v_arg_1546_; lean_object* v_arg_1547_; lean_object* v_rootExpr_1548_; lean_object* v_subExpr_1549_; lean_object* v_rflTarget_x3f_1550_; lean_object* v_pos_1551_; lean_object* v_rwKind_1552_; uint8_t v___y_1554_; lean_object* v___y_1555_; lean_object* v___y_1556_; uint8_t v___y_1557_; uint8_t v___y_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; uint8_t v___y_1561_; size_t v___y_1562_; lean_object* v___y_1563_; uint8_t v_justLemmaName_1564_; lean_object* v_rwKind_1565_; lean_object* v___y_1566_; lean_object* v___y_1567_; lean_object* v___y_1568_; lean_object* v___y_1569_; lean_object* v___y_1570_; lean_object* v___y_1571_; lean_object* v___y_1644_; lean_object* v___y_1645_; lean_object* v___y_1646_; lean_object* v___y_1647_; lean_object* v___y_1648_; lean_object* v___y_1649_; uint8_t v___y_1650_; lean_object* v___y_1651_; lean_object* v___y_1652_; lean_object* v___y_1653_; size_t v___y_1654_; lean_object* v___y_1655_; lean_object* v_a_1656_; lean_object* v___y_1737_; lean_object* v___y_1738_; lean_object* v___y_1739_; lean_object* v___y_1740_; lean_object* v___y_1741_; lean_object* v___y_1742_; lean_object* v___y_1743_; uint8_t v___y_1744_; lean_object* v___y_1745_; lean_object* v___y_1746_; size_t v___y_1747_; lean_object* v___y_1748_; lean_object* v___y_1749_; lean_object* v___y_1760_; lean_object* v___y_1761_; lean_object* v___y_1762_; lean_object* v___y_1763_; lean_object* v___y_1764_; lean_object* v___y_1765_; lean_object* v___y_1766_; lean_object* v___y_1767_; lean_object* v___y_1768_; lean_object* v___y_1794_; lean_object* v___y_1795_; lean_object* v___y_1796_; lean_object* v___y_1797_; lean_object* v___y_1798_; lean_object* v___y_1799_; lean_object* v___y_1800_; lean_object* v___y_1801_; lean_object* v___y_1802_; lean_object* v___y_1803_; lean_object* v___y_1823_; lean_object* v___y_1824_; lean_object* v___y_1825_; lean_object* v___y_1826_; lean_object* v___y_1827_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v___y_1831_; lean_object* v___y_1832_; lean_object* v_fst_1842_; lean_object* v_snd_1843_; 
lean_dec(v_snd_1526_);
lean_del_object(v___x_1523_);
v_arg_1546_ = lean_ctor_get(v_a_1531_, 1);
lean_inc_ref(v_arg_1546_);
lean_dec_ref_known(v_a_1531_, 2);
v_arg_1547_ = lean_ctor_get(v_fn_1545_, 1);
lean_inc_ref(v_arg_1547_);
lean_dec_ref_known(v_fn_1545_, 2);
v_rootExpr_1548_ = lean_ctor_get(v_i_1254_, 0);
lean_inc_ref(v_rootExpr_1548_);
v_subExpr_1549_ = lean_ctor_get(v_i_1254_, 1);
lean_inc_ref(v_subExpr_1549_);
v_rflTarget_x3f_1550_ = lean_ctor_get(v_i_1254_, 2);
lean_inc(v_rflTarget_x3f_1550_);
v_pos_1551_ = lean_ctor_get(v_i_1254_, 3);
lean_inc(v_pos_1551_);
v_rwKind_1552_ = lean_ctor_get(v_i_1254_, 4);
lean_inc(v_rwKind_1552_);
lean_dec_ref(v_i_1254_);
if (v_symm_1314_ == 0)
{
v_fst_1842_ = v_arg_1547_;
v_snd_1843_ = v_arg_1546_;
goto v___jp_1841_;
}
else
{
v_fst_1842_ = v_arg_1546_;
v_snd_1843_ = v_arg_1547_;
goto v___jp_1841_;
}
v___jp_1553_:
{
lean_object* v___x_1572_; 
lean_inc_ref(v___y_1559_);
v___x_1572_ = l_Lean_Meta_ppExpr(v___y_1559_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1572_) == 0)
{
lean_object* v_a_1573_; lean_object* v___x_1574_; 
v_a_1573_ = lean_ctor_get(v___x_1572_, 0);
lean_inc(v_a_1573_);
lean_dec_ref_known(v___x_1572_, 1);
lean_inc_ref(v___y_1559_);
v___x_1574_ = l_Lean_Meta_abstractMVars(v___y_1559_, v___y_1561_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1574_) == 0)
{
lean_object* v_a_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; 
v_a_1575_ = lean_ctor_get(v___x_1574_, 0);
lean_inc(v_a_1575_);
lean_dec_ref_known(v___x_1574_, 1);
v___x_1576_ = l_Std_Format_defWidth;
lean_inc_n(v___y_1556_, 2);
v___x_1577_ = l_Std_Format_pretty(v_a_1573_, v___x_1576_, v___y_1556_, v___y_1556_);
v___x_1578_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x3f___redArg(v___y_1566_, v___y_1568_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1578_) == 0)
{
lean_object* v_a_1579_; lean_object* v___x_1580_; 
v_a_1579_ = lean_ctor_get(v___x_1578_, 0);
lean_inc(v_a_1579_);
lean_dec_ref_known(v___x_1578_, 1);
v___x_1580_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Rewrite_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_1255_, v_rwKind_1565_, v_a_1579_, v___y_1560_, v_justLemmaName_1564_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1580_) == 0)
{
lean_object* v_a_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; uint8_t v___x_1587_; 
v_a_1581_ = lean_ctor_get(v___x_1580_, 0);
lean_inc(v_a_1581_);
lean_dec_ref_known(v___x_1580_, 1);
lean_inc_ref_n(v_name_1313_, 2);
v___x_1582_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(v_name_1313_);
v___x_1583_ = lean_array_get_size(v___y_1563_);
v___x_1584_ = lean_string_length(v___x_1577_);
lean_dec_ref(v___x_1577_);
v___x_1585_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(v_name_1313_);
v___x_1586_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1586_, 0, v___x_1583_);
lean_ctor_set(v___x_1586_, 1, v___x_1582_);
lean_ctor_set(v___x_1586_, 2, v___x_1584_);
lean_ctor_set(v___x_1586_, 3, v___x_1585_);
lean_ctor_set(v___x_1586_, 4, v_a_1575_);
lean_ctor_set_uint8(v___x_1586_, sizeof(void*)*5, v_symm_1314_);
v___x_1587_ = lean_nat_dec_eq(v___x_1583_, v___y_1556_);
if (v___x_1587_ == 0)
{
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
v___y_1463_ = v___y_1556_;
v___y_1464_ = v___y_1554_;
v___y_1465_ = v___x_1586_;
v___y_1466_ = v___y_1571_;
v___y_1467_ = v___y_1559_;
v___y_1468_ = v___y_1562_;
v___y_1469_ = v___y_1568_;
v___y_1470_ = v___y_1557_;
v___y_1471_ = v___y_1567_;
v___y_1472_ = v___y_1570_;
v___y_1473_ = v___y_1569_;
v___y_1474_ = v_a_1581_;
v___y_1475_ = v___y_1566_;
v___y_1476_ = v___y_1563_;
v_a_1477_ = v___y_1558_;
goto v___jp_1462_;
}
else
{
if (lean_obj_tag(v_rflTarget_x3f_1550_) == 1)
{
lean_object* v_val_1588_; lean_object* v___f_1589_; lean_object* v___x_1590_; 
v_val_1588_ = lean_ctor_get(v_rflTarget_x3f_1550_, 0);
lean_inc(v_val_1588_);
lean_dec_ref_known(v_rflTarget_x3f_1550_, 1);
v___f_1589_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1589_, 0, v___y_1555_);
lean_closure_set(v___f_1589_, 1, v_val_1588_);
v___x_1590_ = lp_mathlib_Lean_Meta_withoutModifyingMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__8___redArg(v___f_1589_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
v___y_1488_ = v___y_1556_;
v___y_1489_ = v___y_1554_;
v___y_1490_ = v___x_1586_;
v___y_1491_ = v___y_1571_;
v___y_1492_ = v___y_1559_;
v___y_1493_ = v___y_1562_;
v___y_1494_ = v___y_1568_;
v___y_1495_ = v___y_1557_;
v___y_1496_ = v___y_1567_;
v___y_1497_ = v___y_1570_;
v___y_1498_ = v___y_1569_;
v___y_1499_ = v_a_1581_;
v___y_1500_ = v___y_1566_;
v___y_1501_ = v___y_1563_;
v___y_1502_ = v___x_1590_;
goto v___jp_1487_;
}
else
{
lean_object* v_hyp_x3f_1591_; lean_object* v_pos_1592_; lean_object* v___x_1593_; uint8_t v___x_1594_; 
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
v_hyp_x3f_1591_ = lean_ctor_get(v___y_1566_, 8);
v_pos_1592_ = lean_ctor_get(v___y_1566_, 9);
v___x_1593_ = l_Lean_SubExpr_Pos_root;
v___x_1594_ = lean_nat_dec_eq(v_pos_1592_, v___x_1593_);
if (v___x_1594_ == 0)
{
v___y_1463_ = v___y_1556_;
v___y_1464_ = v___y_1554_;
v___y_1465_ = v___x_1586_;
v___y_1466_ = v___y_1571_;
v___y_1467_ = v___y_1559_;
v___y_1468_ = v___y_1562_;
v___y_1469_ = v___y_1568_;
v___y_1470_ = v___y_1557_;
v___y_1471_ = v___y_1567_;
v___y_1472_ = v___y_1570_;
v___y_1473_ = v___y_1569_;
v___y_1474_ = v_a_1581_;
v___y_1475_ = v___y_1566_;
v___y_1476_ = v___y_1563_;
v_a_1477_ = v___y_1558_;
goto v___jp_1462_;
}
else
{
if (lean_obj_tag(v_hyp_x3f_1591_) == 0)
{
if (v___x_1594_ == 0)
{
v___y_1463_ = v___y_1556_;
v___y_1464_ = v___y_1554_;
v___y_1465_ = v___x_1586_;
v___y_1466_ = v___y_1571_;
v___y_1467_ = v___y_1559_;
v___y_1468_ = v___y_1562_;
v___y_1469_ = v___y_1568_;
v___y_1470_ = v___y_1557_;
v___y_1471_ = v___y_1567_;
v___y_1472_ = v___y_1570_;
v___y_1473_ = v___y_1569_;
v___y_1474_ = v_a_1581_;
v___y_1475_ = v___y_1566_;
v___y_1476_ = v___y_1563_;
v_a_1477_ = v___y_1558_;
goto v___jp_1462_;
}
else
{
lean_object* v___x_1595_; uint8_t v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; 
lean_inc_ref(v___y_1559_);
v___x_1595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1595_, 0, v___y_1559_);
v___x_1596_ = 0;
v___x_1597_ = lean_box(0);
v___x_1598_ = l_Lean_Meta_mkFreshExprMVar(v___x_1595_, v___x_1596_, v___x_1597_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1598_) == 0)
{
lean_object* v_a_1599_; lean_object* v___x_1600_; lean_object* v___f_1601_; lean_object* v___x_1602_; 
v_a_1599_ = lean_ctor_get(v___x_1598_, 0);
lean_inc(v_a_1599_);
lean_dec_ref_known(v___x_1598_, 1);
v___x_1600_ = l_Lean_Expr_mvarId_x21(v_a_1599_);
lean_dec(v_a_1599_);
v___f_1601_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__2___boxed), 8, 1);
lean_closure_set(v___f_1601_, 0, v___x_1600_);
v___x_1602_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__9___redArg(v___f_1601_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
v___y_1488_ = v___y_1556_;
v___y_1489_ = v___y_1554_;
v___y_1490_ = v___x_1586_;
v___y_1491_ = v___y_1571_;
v___y_1492_ = v___y_1559_;
v___y_1493_ = v___y_1562_;
v___y_1494_ = v___y_1568_;
v___y_1495_ = v___y_1557_;
v___y_1496_ = v___y_1567_;
v___y_1497_ = v___y_1570_;
v___y_1498_ = v___y_1569_;
v___y_1499_ = v_a_1581_;
v___y_1500_ = v___y_1566_;
v___y_1501_ = v___y_1563_;
v___y_1502_ = v___x_1602_;
goto v___jp_1487_;
}
else
{
lean_object* v_a_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1610_; 
lean_dec_ref_known(v___x_1586_, 5);
lean_dec(v_a_1581_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1556_);
lean_dec_ref(v_name_1313_);
v_a_1603_ = lean_ctor_get(v___x_1598_, 0);
v_isSharedCheck_1610_ = !lean_is_exclusive(v___x_1598_);
if (v_isSharedCheck_1610_ == 0)
{
v___x_1605_ = v___x_1598_;
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_a_1603_);
lean_dec(v___x_1598_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v___x_1608_; 
if (v_isShared_1606_ == 0)
{
v___x_1608_ = v___x_1605_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1609_; 
v_reuseFailAlloc_1609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1609_, 0, v_a_1603_);
v___x_1608_ = v_reuseFailAlloc_1609_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
return v___x_1608_;
}
}
}
}
}
else
{
v___y_1463_ = v___y_1556_;
v___y_1464_ = v___y_1554_;
v___y_1465_ = v___x_1586_;
v___y_1466_ = v___y_1571_;
v___y_1467_ = v___y_1559_;
v___y_1468_ = v___y_1562_;
v___y_1469_ = v___y_1568_;
v___y_1470_ = v___y_1557_;
v___y_1471_ = v___y_1567_;
v___y_1472_ = v___y_1570_;
v___y_1473_ = v___y_1569_;
v___y_1474_ = v_a_1581_;
v___y_1475_ = v___y_1566_;
v___y_1476_ = v___y_1563_;
v_a_1477_ = v___y_1558_;
goto v___jp_1462_;
}
}
}
}
}
else
{
lean_object* v_a_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1618_; 
lean_dec_ref(v___x_1577_);
lean_dec(v_a_1575_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
v_a_1611_ = lean_ctor_get(v___x_1580_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v___x_1580_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1613_ = v___x_1580_;
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_a_1611_);
lean_dec(v___x_1580_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
lean_object* v___x_1616_; 
if (v_isShared_1614_ == 0)
{
v___x_1616_ = v___x_1613_;
goto v_reusejp_1615_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v_a_1611_);
v___x_1616_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1615_;
}
v_reusejp_1615_:
{
return v___x_1616_;
}
}
}
}
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
lean_dec_ref(v___x_1577_);
lean_dec(v_a_1575_);
lean_dec(v_rwKind_1565_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1619_ = lean_ctor_get(v___x_1578_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1578_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1578_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1578_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
}
else
{
lean_object* v_a_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1634_; 
lean_dec(v_a_1573_);
lean_dec(v_rwKind_1565_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1627_ = lean_ctor_get(v___x_1574_, 0);
v_isSharedCheck_1634_ = !lean_is_exclusive(v___x_1574_);
if (v_isSharedCheck_1634_ == 0)
{
v___x_1629_ = v___x_1574_;
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_a_1627_);
lean_dec(v___x_1574_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___x_1632_; 
if (v_isShared_1630_ == 0)
{
v___x_1632_ = v___x_1629_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v_a_1627_);
v___x_1632_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
return v___x_1632_;
}
}
}
}
else
{
lean_object* v_a_1635_; lean_object* v___x_1637_; uint8_t v_isShared_1638_; uint8_t v_isSharedCheck_1642_; 
lean_dec(v_rwKind_1565_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1635_ = lean_ctor_get(v___x_1572_, 0);
v_isSharedCheck_1642_ = !lean_is_exclusive(v___x_1572_);
if (v_isSharedCheck_1642_ == 0)
{
v___x_1637_ = v___x_1572_;
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
else
{
lean_inc(v_a_1635_);
lean_dec(v___x_1572_);
v___x_1637_ = lean_box(0);
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
v_resetjp_1636_:
{
lean_object* v___x_1640_; 
if (v_isShared_1638_ == 0)
{
v___x_1640_ = v___x_1637_;
goto v_reusejp_1639_;
}
else
{
lean_object* v_reuseFailAlloc_1641_; 
v_reuseFailAlloc_1641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1641_, 0, v_a_1635_);
v___x_1640_ = v_reuseFailAlloc_1641_;
goto v_reusejp_1639_;
}
v_reusejp_1639_:
{
return v___x_1640_;
}
}
}
}
v___jp_1643_:
{
lean_object* v___x_1657_; uint8_t v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1661_; 
v___x_1657_ = lean_mk_empty_array_with_capacity(v___y_1645_);
v___x_1658_ = 1;
v___x_1659_ = lean_box(v___x_1658_);
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 1, v___x_1659_);
lean_ctor_set(v___x_1528_, 0, v___x_1657_);
v___x_1661_ = v___x_1528_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1735_; 
v_reuseFailAlloc_1735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1735_, 0, v___x_1657_);
lean_ctor_set(v_reuseFailAlloc_1735_, 1, v___x_1659_);
v___x_1661_ = v_reuseFailAlloc_1735_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
size_t v_sz_1662_; lean_object* v___x_1663_; 
v_sz_1662_ = lean_array_size(v_a_1656_);
v___x_1663_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__6(v_rwKind_1552_, v_a_1656_, v_sz_1662_, v___y_1654_, v___x_1661_, v___y_1649_, v___y_1652_, v___y_1655_, v___y_1653_, v___y_1648_, v___y_1647_);
if (lean_obj_tag(v___x_1663_) == 0)
{
lean_object* v_a_1664_; lean_object* v___x_1665_; lean_object* v_a_1666_; lean_object* v_fst_1667_; lean_object* v_snd_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; 
v_a_1664_ = lean_ctor_get(v___x_1663_, 0);
lean_inc(v_a_1664_);
lean_dec_ref_known(v___x_1663_, 1);
v___x_1665_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v___y_1651_, v___y_1653_);
v_a_1666_ = lean_ctor_get(v___x_1665_, 0);
lean_inc_n(v_a_1666_, 2);
lean_dec_ref(v___x_1665_);
v_fst_1667_ = lean_ctor_get(v_a_1664_, 0);
lean_inc_n(v_fst_1667_, 2);
v_snd_1668_ = lean_ctor_get(v_a_1664_, 1);
lean_inc(v_snd_1668_);
lean_dec(v_a_1664_);
v___x_1669_ = lean_array_push(v_fst_1667_, v_a_1666_);
v___x_1670_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(v_a_1656_, v_assignableMVars_1256_, v___x_1669_, v___y_1655_, v___y_1653_, v___y_1648_, v___y_1647_);
lean_dec_ref(v___x_1669_);
lean_dec_ref(v_a_1656_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v_a_1671_; lean_object* v___x_1672_; lean_object* v_a_1673_; lean_object* v___x_1674_; 
v_a_1671_ = lean_ctor_get(v___x_1670_, 0);
lean_inc(v_a_1671_);
lean_dec_ref_known(v___x_1670_, 1);
v___x_1672_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v_fst_1517_, v___y_1653_);
v_a_1673_ = lean_ctor_get(v___x_1672_, 0);
lean_inc(v_a_1673_);
lean_dec_ref(v___x_1672_);
lean_inc(v_a_1666_);
v___x_1674_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(v_subExpr_1549_, v_a_1666_, v___y_1655_, v___y_1653_, v___y_1648_, v___y_1647_);
if (lean_obj_tag(v___x_1674_) == 0)
{
if (lean_obj_tag(v_rwKind_1552_) == 1)
{
uint8_t v___x_1675_; 
v___x_1675_ = lean_unbox(v_snd_1668_);
if (v___x_1675_ == 0)
{
lean_object* v_a_1676_; uint8_t v___x_1677_; uint8_t v___x_1678_; uint8_t v___x_1679_; 
lean_dec_ref(v___y_1646_);
lean_dec_ref(v___y_1644_);
v_a_1676_ = lean_ctor_get(v___x_1674_, 0);
lean_inc(v_a_1676_);
lean_dec_ref_known(v___x_1674_, 1);
v___x_1677_ = lean_unbox(v_a_1676_);
lean_dec(v_a_1676_);
v___x_1678_ = lean_unbox(v_a_1671_);
lean_dec(v_a_1671_);
v___x_1679_ = lean_unbox(v_snd_1668_);
lean_dec(v_snd_1668_);
lean_inc(v_a_1666_);
v___y_1554_ = v___x_1677_;
v___y_1555_ = v_a_1666_;
v___y_1556_ = v___y_1645_;
v___y_1557_ = v___x_1678_;
v___y_1558_ = v___y_1650_;
v___y_1559_ = v_a_1666_;
v___y_1560_ = v_a_1673_;
v___y_1561_ = v___x_1658_;
v___y_1562_ = v___y_1654_;
v___y_1563_ = v_fst_1667_;
v_justLemmaName_1564_ = v___x_1679_;
v_rwKind_1565_ = v_rwKind_1552_;
v___y_1566_ = v___y_1649_;
v___y_1567_ = v___y_1652_;
v___y_1568_ = v___y_1655_;
v___y_1569_ = v___y_1653_;
v___y_1570_ = v___y_1648_;
v___y_1571_ = v___y_1647_;
goto v___jp_1553_;
}
else
{
lean_object* v_a_1680_; uint8_t v_motiveTypeCorrect_1681_; lean_object* v___x_1682_; 
v_a_1680_ = lean_ctor_get(v___x_1674_, 0);
lean_inc(v_a_1680_);
lean_dec_ref_known(v___x_1674_, 1);
v_motiveTypeCorrect_1681_ = lean_ctor_get_uint8(v_rwKind_1552_, sizeof(void*)*1);
v___x_1682_ = lp_mathlib_Lean_Meta_withMCtx___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__10___redArg(v___y_1646_, v___y_1644_, v___y_1649_, v___y_1652_, v___y_1655_, v___y_1653_, v___y_1648_, v___y_1647_);
if (lean_obj_tag(v___x_1682_) == 0)
{
lean_object* v_a_1683_; uint8_t v___x_1684_; 
v_a_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc(v_a_1683_);
lean_dec_ref_known(v___x_1682_, 1);
v___x_1684_ = lean_unbox(v_a_1683_);
lean_dec(v_a_1683_);
if (v___x_1684_ == 0)
{
uint8_t v___x_1685_; uint8_t v___x_1686_; 
lean_dec(v_snd_1668_);
v___x_1685_ = lean_unbox(v_a_1680_);
lean_dec(v_a_1680_);
v___x_1686_ = lean_unbox(v_a_1671_);
lean_dec(v_a_1671_);
lean_inc(v_a_1666_);
v___y_1554_ = v___x_1685_;
v___y_1555_ = v_a_1666_;
v___y_1556_ = v___y_1645_;
v___y_1557_ = v___x_1686_;
v___y_1558_ = v___y_1650_;
v___y_1559_ = v_a_1666_;
v___y_1560_ = v_a_1673_;
v___y_1561_ = v___x_1658_;
v___y_1562_ = v___y_1654_;
v___y_1563_ = v_fst_1667_;
v_justLemmaName_1564_ = v___y_1650_;
v_rwKind_1565_ = v_rwKind_1552_;
v___y_1566_ = v___y_1649_;
v___y_1567_ = v___y_1652_;
v___y_1568_ = v___y_1655_;
v___y_1569_ = v___y_1653_;
v___y_1570_ = v___y_1648_;
v___y_1571_ = v___y_1647_;
goto v___jp_1553_;
}
else
{
lean_object* v___x_1688_; uint8_t v_isShared_1689_; uint8_t v_isSharedCheck_1697_; 
v_isSharedCheck_1697_ = !lean_is_exclusive(v_rwKind_1552_);
if (v_isSharedCheck_1697_ == 0)
{
lean_object* v_unused_1698_; 
v_unused_1698_ = lean_ctor_get(v_rwKind_1552_, 0);
lean_dec(v_unused_1698_);
v___x_1688_ = v_rwKind_1552_;
v_isShared_1689_ = v_isSharedCheck_1697_;
goto v_resetjp_1687_;
}
else
{
lean_dec(v_rwKind_1552_);
v___x_1688_ = lean_box(0);
v_isShared_1689_ = v_isSharedCheck_1697_;
goto v_resetjp_1687_;
}
v_resetjp_1687_:
{
lean_object* v___x_1690_; lean_object* v___x_1692_; 
v___x_1690_ = lean_box(0);
if (v_isShared_1689_ == 0)
{
lean_ctor_set(v___x_1688_, 0, v___x_1690_);
v___x_1692_ = v___x_1688_;
goto v_reusejp_1691_;
}
else
{
lean_object* v_reuseFailAlloc_1696_; 
v_reuseFailAlloc_1696_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v_reuseFailAlloc_1696_, 0, v___x_1690_);
lean_ctor_set_uint8(v_reuseFailAlloc_1696_, sizeof(void*)*1, v_motiveTypeCorrect_1681_);
v___x_1692_ = v_reuseFailAlloc_1696_;
goto v_reusejp_1691_;
}
v_reusejp_1691_:
{
uint8_t v___x_1693_; uint8_t v___x_1694_; uint8_t v___x_1695_; 
v___x_1693_ = lean_unbox(v_a_1680_);
lean_dec(v_a_1680_);
v___x_1694_ = lean_unbox(v_a_1671_);
lean_dec(v_a_1671_);
v___x_1695_ = lean_unbox(v_snd_1668_);
lean_dec(v_snd_1668_);
lean_inc(v_a_1666_);
v___y_1554_ = v___x_1693_;
v___y_1555_ = v_a_1666_;
v___y_1556_ = v___y_1645_;
v___y_1557_ = v___x_1694_;
v___y_1558_ = v___y_1650_;
v___y_1559_ = v_a_1666_;
v___y_1560_ = v_a_1673_;
v___y_1561_ = v___x_1658_;
v___y_1562_ = v___y_1654_;
v___y_1563_ = v_fst_1667_;
v_justLemmaName_1564_ = v___x_1695_;
v_rwKind_1565_ = v___x_1692_;
v___y_1566_ = v___y_1649_;
v___y_1567_ = v___y_1652_;
v___y_1568_ = v___y_1655_;
v___y_1569_ = v___y_1653_;
v___y_1570_ = v___y_1648_;
v___y_1571_ = v___y_1647_;
goto v___jp_1553_;
}
}
}
}
else
{
lean_object* v_a_1699_; lean_object* v___x_1701_; uint8_t v_isShared_1702_; uint8_t v_isSharedCheck_1706_; 
lean_dec(v_a_1680_);
lean_dec_ref_known(v_rwKind_1552_, 1);
lean_dec(v_a_1673_);
lean_dec(v_a_1671_);
lean_dec(v_snd_1668_);
lean_dec(v_fst_1667_);
lean_dec(v_a_1666_);
lean_dec(v___y_1645_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1699_ = lean_ctor_get(v___x_1682_, 0);
v_isSharedCheck_1706_ = !lean_is_exclusive(v___x_1682_);
if (v_isSharedCheck_1706_ == 0)
{
v___x_1701_ = v___x_1682_;
v_isShared_1702_ = v_isSharedCheck_1706_;
goto v_resetjp_1700_;
}
else
{
lean_inc(v_a_1699_);
lean_dec(v___x_1682_);
v___x_1701_ = lean_box(0);
v_isShared_1702_ = v_isSharedCheck_1706_;
goto v_resetjp_1700_;
}
v_resetjp_1700_:
{
lean_object* v___x_1704_; 
if (v_isShared_1702_ == 0)
{
v___x_1704_ = v___x_1701_;
goto v_reusejp_1703_;
}
else
{
lean_object* v_reuseFailAlloc_1705_; 
v_reuseFailAlloc_1705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1705_, 0, v_a_1699_);
v___x_1704_ = v_reuseFailAlloc_1705_;
goto v_reusejp_1703_;
}
v_reusejp_1703_:
{
return v___x_1704_;
}
}
}
}
}
else
{
lean_object* v_a_1707_; uint8_t v___x_1708_; uint8_t v___x_1709_; uint8_t v___x_1710_; 
lean_dec_ref(v___y_1646_);
lean_dec_ref(v___y_1644_);
v_a_1707_ = lean_ctor_get(v___x_1674_, 0);
lean_inc(v_a_1707_);
lean_dec_ref_known(v___x_1674_, 1);
v___x_1708_ = lean_unbox(v_a_1707_);
lean_dec(v_a_1707_);
v___x_1709_ = lean_unbox(v_a_1671_);
lean_dec(v_a_1671_);
v___x_1710_ = lean_unbox(v_snd_1668_);
lean_dec(v_snd_1668_);
lean_inc(v_a_1666_);
v___y_1554_ = v___x_1708_;
v___y_1555_ = v_a_1666_;
v___y_1556_ = v___y_1645_;
v___y_1557_ = v___x_1709_;
v___y_1558_ = v___y_1650_;
v___y_1559_ = v_a_1666_;
v___y_1560_ = v_a_1673_;
v___y_1561_ = v___x_1658_;
v___y_1562_ = v___y_1654_;
v___y_1563_ = v_fst_1667_;
v_justLemmaName_1564_ = v___x_1710_;
v_rwKind_1565_ = v_rwKind_1552_;
v___y_1566_ = v___y_1649_;
v___y_1567_ = v___y_1652_;
v___y_1568_ = v___y_1655_;
v___y_1569_ = v___y_1653_;
v___y_1570_ = v___y_1648_;
v___y_1571_ = v___y_1647_;
goto v___jp_1553_;
}
}
else
{
lean_object* v_a_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1718_; 
lean_dec(v_a_1673_);
lean_dec(v_a_1671_);
lean_dec(v_snd_1668_);
lean_dec(v_fst_1667_);
lean_dec(v_a_1666_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1711_ = lean_ctor_get(v___x_1674_, 0);
v_isSharedCheck_1718_ = !lean_is_exclusive(v___x_1674_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1713_ = v___x_1674_;
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_a_1711_);
lean_dec(v___x_1674_);
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
else
{
lean_object* v_a_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1726_; 
lean_dec(v_snd_1668_);
lean_dec(v_fst_1667_);
lean_dec(v_a_1666_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_subExpr_1549_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1719_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1726_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1726_ == 0)
{
v___x_1721_ = v___x_1670_;
v_isShared_1722_ = v_isSharedCheck_1726_;
goto v_resetjp_1720_;
}
else
{
lean_inc(v_a_1719_);
lean_dec(v___x_1670_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1726_;
goto v_resetjp_1720_;
}
v_resetjp_1720_:
{
lean_object* v___x_1724_; 
if (v_isShared_1722_ == 0)
{
v___x_1724_ = v___x_1721_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v_a_1719_);
v___x_1724_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
return v___x_1724_;
}
}
}
}
else
{
lean_object* v_a_1727_; lean_object* v___x_1729_; uint8_t v_isShared_1730_; uint8_t v_isSharedCheck_1734_; 
lean_dec_ref(v_a_1656_);
lean_dec_ref(v___y_1651_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_subExpr_1549_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1727_ = lean_ctor_get(v___x_1663_, 0);
v_isSharedCheck_1734_ = !lean_is_exclusive(v___x_1663_);
if (v_isSharedCheck_1734_ == 0)
{
v___x_1729_ = v___x_1663_;
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
else
{
lean_inc(v_a_1727_);
lean_dec(v___x_1663_);
v___x_1729_ = lean_box(0);
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
v_resetjp_1728_:
{
lean_object* v___x_1732_; 
if (v_isShared_1730_ == 0)
{
v___x_1732_ = v___x_1729_;
goto v_reusejp_1731_;
}
else
{
lean_object* v_reuseFailAlloc_1733_; 
v_reuseFailAlloc_1733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1733_, 0, v_a_1727_);
v___x_1732_ = v_reuseFailAlloc_1733_;
goto v_reusejp_1731_;
}
v_reusejp_1731_:
{
return v___x_1732_;
}
}
}
}
}
v___jp_1736_:
{
if (lean_obj_tag(v___y_1749_) == 0)
{
lean_object* v_a_1750_; 
v_a_1750_ = lean_ctor_get(v___y_1749_, 0);
lean_inc(v_a_1750_);
lean_dec_ref_known(v___y_1749_, 1);
v___y_1644_ = v___y_1737_;
v___y_1645_ = v___y_1738_;
v___y_1646_ = v___y_1739_;
v___y_1647_ = v___y_1741_;
v___y_1648_ = v___y_1740_;
v___y_1649_ = v___y_1742_;
v___y_1650_ = v___y_1744_;
v___y_1651_ = v___y_1743_;
v___y_1652_ = v___y_1745_;
v___y_1653_ = v___y_1746_;
v___y_1654_ = v___y_1747_;
v___y_1655_ = v___y_1748_;
v_a_1656_ = v_a_1750_;
goto v___jp_1643_;
}
else
{
lean_object* v_a_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1758_; 
lean_dec_ref(v___y_1743_);
lean_dec_ref(v___y_1739_);
lean_dec(v___y_1738_);
lean_dec_ref(v___y_1737_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_subExpr_1549_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1751_ = lean_ctor_get(v___y_1749_, 0);
v_isSharedCheck_1758_ = !lean_is_exclusive(v___y_1749_);
if (v_isSharedCheck_1758_ == 0)
{
v___x_1753_ = v___y_1749_;
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_a_1751_);
lean_dec(v___y_1749_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v___x_1756_; 
if (v_isShared_1754_ == 0)
{
v___x_1756_ = v___x_1753_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v_a_1751_);
v___x_1756_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
return v___x_1756_;
}
}
}
}
v___jp_1759_:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; uint8_t v___x_1771_; lean_object* v___x_1772_; 
v___x_1769_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__5));
v___x_1770_ = lean_box(0);
v___x_1771_ = 0;
v___x_1772_ = l_Lean_Meta_synthAppInstances(v___x_1769_, v___x_1770_, v_fst_1521_, v_fst_1525_, v___x_1771_, v___x_1771_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
if (lean_obj_tag(v___x_1772_) == 0)
{
size_t v_sz_1773_; size_t v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; uint8_t v___x_1779_; 
lean_dec_ref_known(v___x_1772_, 1);
v_sz_1773_ = lean_array_size(v_fst_1521_);
v___x_1774_ = ((size_t)0ULL);
v___x_1775_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__2(v_sz_1773_, v___x_1774_, v_fst_1521_);
v___x_1776_ = lean_unsigned_to_nat(0u);
v___x_1777_ = lean_array_get_size(v___x_1775_);
v___x_1778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__6));
v___x_1779_ = lean_nat_dec_lt(v___x_1776_, v___x_1777_);
if (v___x_1779_ == 0)
{
lean_dec_ref(v___x_1775_);
v___y_1644_ = v___y_1760_;
v___y_1645_ = v___x_1776_;
v___y_1646_ = v___y_1761_;
v___y_1647_ = v___y_1768_;
v___y_1648_ = v___y_1767_;
v___y_1649_ = v___y_1763_;
v___y_1650_ = v___x_1771_;
v___y_1651_ = v___y_1762_;
v___y_1652_ = v___y_1764_;
v___y_1653_ = v___y_1766_;
v___y_1654_ = v___x_1774_;
v___y_1655_ = v___y_1765_;
v_a_1656_ = v___x_1778_;
goto v___jp_1643_;
}
else
{
uint8_t v___x_1780_; 
v___x_1780_ = lean_nat_dec_le(v___x_1777_, v___x_1777_);
if (v___x_1780_ == 0)
{
if (v___x_1779_ == 0)
{
lean_dec_ref(v___x_1775_);
v___y_1644_ = v___y_1760_;
v___y_1645_ = v___x_1776_;
v___y_1646_ = v___y_1761_;
v___y_1647_ = v___y_1768_;
v___y_1648_ = v___y_1767_;
v___y_1649_ = v___y_1763_;
v___y_1650_ = v___x_1771_;
v___y_1651_ = v___y_1762_;
v___y_1652_ = v___y_1764_;
v___y_1653_ = v___y_1766_;
v___y_1654_ = v___x_1774_;
v___y_1655_ = v___y_1765_;
v_a_1656_ = v___x_1778_;
goto v___jp_1643_;
}
else
{
size_t v___x_1781_; lean_object* v___x_1782_; 
v___x_1781_ = lean_usize_of_nat(v___x_1777_);
v___x_1782_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11(v___x_1775_, v___x_1774_, v___x_1781_, v___x_1778_, v___y_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
lean_dec_ref(v___x_1775_);
v___y_1737_ = v___y_1760_;
v___y_1738_ = v___x_1776_;
v___y_1739_ = v___y_1761_;
v___y_1740_ = v___y_1767_;
v___y_1741_ = v___y_1768_;
v___y_1742_ = v___y_1763_;
v___y_1743_ = v___y_1762_;
v___y_1744_ = v___x_1771_;
v___y_1745_ = v___y_1764_;
v___y_1746_ = v___y_1766_;
v___y_1747_ = v___x_1774_;
v___y_1748_ = v___y_1765_;
v___y_1749_ = v___x_1782_;
goto v___jp_1736_;
}
}
else
{
size_t v___x_1783_; lean_object* v___x_1784_; 
v___x_1783_ = lean_usize_of_nat(v___x_1777_);
v___x_1784_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__11(v___x_1775_, v___x_1774_, v___x_1783_, v___x_1778_, v___y_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
lean_dec_ref(v___x_1775_);
v___y_1737_ = v___y_1760_;
v___y_1738_ = v___x_1776_;
v___y_1739_ = v___y_1761_;
v___y_1740_ = v___y_1767_;
v___y_1741_ = v___y_1768_;
v___y_1742_ = v___y_1763_;
v___y_1743_ = v___y_1762_;
v___y_1744_ = v___x_1771_;
v___y_1745_ = v___y_1764_;
v___y_1746_ = v___y_1766_;
v___y_1747_ = v___x_1774_;
v___y_1748_ = v___y_1765_;
v___y_1749_ = v___x_1784_;
goto v___jp_1736_;
}
}
}
else
{
lean_object* v_a_1785_; lean_object* v___x_1787_; uint8_t v_isShared_1788_; uint8_t v_isSharedCheck_1792_; 
lean_dec_ref(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec_ref(v___y_1760_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_subExpr_1549_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1521_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1785_ = lean_ctor_get(v___x_1772_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1772_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1787_ = v___x_1772_;
v_isShared_1788_ = v_isSharedCheck_1792_;
goto v_resetjp_1786_;
}
else
{
lean_inc(v_a_1785_);
lean_dec(v___x_1772_);
v___x_1787_ = lean_box(0);
v_isShared_1788_ = v_isSharedCheck_1792_;
goto v_resetjp_1786_;
}
v_resetjp_1786_:
{
lean_object* v___x_1790_; 
if (v_isShared_1788_ == 0)
{
v___x_1790_ = v___x_1787_;
goto v_reusejp_1789_;
}
else
{
lean_object* v_reuseFailAlloc_1791_; 
v_reuseFailAlloc_1791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1791_, 0, v_a_1785_);
v___x_1790_ = v_reuseFailAlloc_1791_;
goto v_reusejp_1789_;
}
v_reusejp_1789_:
{
return v___x_1790_;
}
}
}
}
v___jp_1793_:
{
lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1807_; 
lean_dec_ref(v___y_1797_);
lean_dec_ref(v___y_1795_);
lean_dec_ref(v___y_1794_);
v___x_1804_ = l_Lean_MessageData_ofExpr(v___y_1803_);
v___x_1805_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__8);
if (v_isShared_1520_ == 0)
{
lean_ctor_set_tag(v___x_1519_, 7);
lean_ctor_set(v___x_1519_, 1, v___x_1805_);
lean_ctor_set(v___x_1519_, 0, v___x_1804_);
v___x_1807_ = v___x_1519_;
goto v_reusejp_1806_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v___x_1804_);
lean_ctor_set(v_reuseFailAlloc_1821_, 1, v___x_1805_);
v___x_1807_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1806_;
}
v_reusejp_1806_:
{
lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
v___x_1808_ = l_Lean_MessageData_ofExpr(v_subExpr_1549_);
v___x_1809_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1807_);
lean_ctor_set(v___x_1809_, 1, v___x_1808_);
v___x_1810_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__10);
v___x_1811_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1811_, 0, v___x_1809_);
lean_ctor_set(v___x_1811_, 1, v___x_1810_);
v___x_1812_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v___x_1811_, v___y_1800_, v___y_1798_, v___y_1796_, v___y_1799_);
v_a_1813_ = lean_ctor_get(v___x_1812_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1812_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1812_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1812_);
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
v___jp_1822_:
{
lean_object* v___x_1833_; lean_object* v_a_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; uint8_t v___x_1837_; 
v___x_1833_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__3___redArg(v___y_1826_, v___y_1830_);
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
lean_inc_n(v_a_1834_, 2);
lean_dec_ref(v___x_1833_);
v___x_1835_ = l_Lean_Expr_toHeadIndex(v_a_1834_);
lean_inc_ref(v_subExpr_1549_);
v___x_1836_ = l_Lean_Expr_toHeadIndex(v_subExpr_1549_);
v___x_1837_ = l_Lean_instBEqHeadIndex_beq(v___x_1835_, v___x_1836_);
lean_dec(v___x_1836_);
lean_dec(v___x_1835_);
if (v___x_1837_ == 0)
{
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v___y_1794_ = v___y_1824_;
v___y_1795_ = v___y_1823_;
v___y_1796_ = v___y_1831_;
v___y_1797_ = v___y_1825_;
v___y_1798_ = v___y_1830_;
v___y_1799_ = v___y_1832_;
v___y_1800_ = v___y_1829_;
v___y_1801_ = v___y_1828_;
v___y_1802_ = v___y_1827_;
v___y_1803_ = v_a_1834_;
goto v___jp_1793_;
}
else
{
lean_object* v___x_1838_; lean_object* v___x_1839_; uint8_t v___x_1840_; 
v___x_1838_ = l_Lean_Expr_headNumArgs(v_a_1834_);
v___x_1839_ = l_Lean_Expr_headNumArgs(v_subExpr_1549_);
v___x_1840_ = lean_nat_dec_eq(v___x_1838_, v___x_1839_);
lean_dec(v___x_1839_);
lean_dec(v___x_1838_);
if (v___x_1840_ == 0)
{
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v___y_1794_ = v___y_1824_;
v___y_1795_ = v___y_1823_;
v___y_1796_ = v___y_1831_;
v___y_1797_ = v___y_1825_;
v___y_1798_ = v___y_1830_;
v___y_1799_ = v___y_1832_;
v___y_1800_ = v___y_1829_;
v___y_1801_ = v___y_1828_;
v___y_1802_ = v___y_1827_;
v___y_1803_ = v_a_1834_;
goto v___jp_1793_;
}
else
{
lean_dec(v_a_1834_);
lean_del_object(v___x_1519_);
v___y_1760_ = v___y_1824_;
v___y_1761_ = v___y_1823_;
v___y_1762_ = v___y_1825_;
v___y_1763_ = v___y_1827_;
v___y_1764_ = v___y_1828_;
v___y_1765_ = v___y_1829_;
v___y_1766_ = v___y_1830_;
v___y_1767_ = v___y_1831_;
v___y_1768_ = v___y_1832_;
goto v___jp_1759_;
}
}
}
v___jp_1841_:
{
lean_object* v___x_1844_; lean_object* v___x_1845_; 
v___x_1844_ = lean_st_ref_get(v_a_1260_);
lean_inc_ref(v_subExpr_1549_);
lean_inc_ref(v_fst_1842_);
v___x_1845_ = l_Lean_Meta_isExprDefEq(v_fst_1842_, v_subExpr_1549_, v_a_1259_, v_a_1260_, v_a_1261_, v_a_1262_);
if (lean_obj_tag(v___x_1845_) == 0)
{
lean_object* v_a_1846_; lean_object* v_mctx_1847_; lean_object* v___f_1848_; uint8_t v___x_1849_; 
v_a_1846_ = lean_ctor_get(v___x_1845_, 0);
lean_inc(v_a_1846_);
lean_dec_ref_known(v___x_1845_, 1);
v_mctx_1847_ = lean_ctor_get(v___x_1844_, 0);
lean_inc_ref(v_mctx_1847_);
lean_dec(v___x_1844_);
lean_inc_ref(v_fst_1842_);
v___f_1848_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1848_, 0, v_rootExpr_1548_);
lean_closure_set(v___f_1848_, 1, v_fst_1842_);
lean_closure_set(v___f_1848_, 2, v_pos_1551_);
v___x_1849_ = lean_unbox(v_a_1846_);
lean_dec(v_a_1846_);
if (v___x_1849_ == 0)
{
lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v_a_1856_; lean_object* v___x_1858_; uint8_t v_isShared_1859_; uint8_t v_isSharedCheck_1863_; 
lean_dec_ref(v___f_1848_);
lean_dec_ref(v_mctx_1847_);
lean_dec_ref(v_snd_1843_);
lean_dec(v_rwKind_1552_);
lean_dec(v_rflTarget_x3f_1550_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_del_object(v___x_1519_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v___x_1850_ = l_Lean_MessageData_ofExpr(v_fst_1842_);
v___x_1851_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__12);
v___x_1852_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1852_, 0, v___x_1850_);
lean_ctor_set(v___x_1852_, 1, v___x_1851_);
v___x_1853_ = l_Lean_MessageData_ofExpr(v_subExpr_1549_);
v___x_1854_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1854_, 0, v___x_1852_);
lean_ctor_set(v___x_1854_, 1, v___x_1853_);
v___x_1855_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v___x_1854_, v_a_1259_, v_a_1260_, v_a_1261_, v_a_1262_);
v_a_1856_ = lean_ctor_get(v___x_1855_, 0);
v_isSharedCheck_1863_ = !lean_is_exclusive(v___x_1855_);
if (v_isSharedCheck_1863_ == 0)
{
v___x_1858_ = v___x_1855_;
v_isShared_1859_ = v_isSharedCheck_1863_;
goto v_resetjp_1857_;
}
else
{
lean_inc(v_a_1856_);
lean_dec(v___x_1855_);
v___x_1858_ = lean_box(0);
v_isShared_1859_ = v_isSharedCheck_1863_;
goto v_resetjp_1857_;
}
v_resetjp_1857_:
{
lean_object* v___x_1861_; 
if (v_isShared_1859_ == 0)
{
v___x_1861_ = v___x_1858_;
goto v_reusejp_1860_;
}
else
{
lean_object* v_reuseFailAlloc_1862_; 
v_reuseFailAlloc_1862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1862_, 0, v_a_1856_);
v___x_1861_ = v_reuseFailAlloc_1862_;
goto v_reusejp_1860_;
}
v_reusejp_1860_:
{
return v___x_1861_;
}
}
}
else
{
v___y_1823_ = v_mctx_1847_;
v___y_1824_ = v___f_1848_;
v___y_1825_ = v_snd_1843_;
v___y_1826_ = v_fst_1842_;
v___y_1827_ = v_a_1257_;
v___y_1828_ = v_a_1258_;
v___y_1829_ = v_a_1259_;
v___y_1830_ = v_a_1260_;
v___y_1831_ = v_a_1261_;
v___y_1832_ = v_a_1262_;
goto v___jp_1822_;
}
}
else
{
lean_object* v_a_1864_; lean_object* v___x_1866_; uint8_t v_isShared_1867_; uint8_t v_isSharedCheck_1871_; 
lean_dec(v___x_1844_);
lean_dec_ref(v_snd_1843_);
lean_dec_ref(v_fst_1842_);
lean_dec(v_rwKind_1552_);
lean_dec(v_pos_1551_);
lean_dec(v_rflTarget_x3f_1550_);
lean_dec_ref(v_subExpr_1549_);
lean_dec_ref(v_rootExpr_1548_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_del_object(v___x_1519_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
v_a_1864_ = lean_ctor_get(v___x_1845_, 0);
v_isSharedCheck_1871_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_1871_ == 0)
{
v___x_1866_ = v___x_1845_;
v_isShared_1867_ = v_isSharedCheck_1871_;
goto v_resetjp_1865_;
}
else
{
lean_inc(v_a_1864_);
lean_dec(v___x_1845_);
v___x_1866_ = lean_box(0);
v_isShared_1867_ = v_isSharedCheck_1871_;
goto v_resetjp_1865_;
}
v_resetjp_1865_:
{
lean_object* v___x_1869_; 
if (v_isShared_1867_ == 0)
{
v___x_1869_ = v___x_1866_;
goto v_reusejp_1868_;
}
else
{
lean_object* v_reuseFailAlloc_1870_; 
v_reuseFailAlloc_1870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1870_, 0, v_a_1864_);
v___x_1869_ = v_reuseFailAlloc_1870_;
goto v_reusejp_1868_;
}
v_reusejp_1868_:
{
return v___x_1869_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_1531_, 2);
lean_dec_ref(v_fn_1545_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_del_object(v___x_1519_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
lean_dec_ref(v_i_1254_);
v___y_1533_ = v_a_1257_;
v___y_1534_ = v_a_1258_;
v___y_1535_ = v_a_1259_;
v___y_1536_ = v_a_1260_;
v___y_1537_ = v_a_1261_;
v___y_1538_ = v_a_1262_;
goto v___jp_1532_;
}
}
else
{
lean_dec(v_a_1531_);
lean_del_object(v___x_1528_);
lean_dec(v_fst_1525_);
lean_dec(v_fst_1521_);
lean_del_object(v___x_1519_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
lean_dec_ref(v_i_1254_);
v___y_1533_ = v_a_1257_;
v___y_1534_ = v_a_1258_;
v___y_1535_ = v_a_1259_;
v___y_1536_ = v_a_1260_;
v___y_1537_ = v_a_1261_;
v___y_1538_ = v_a_1262_;
goto v___jp_1532_;
}
v___jp_1532_:
{
lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1542_; 
v___x_1539_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__3);
v___x_1540_ = l_Lean_MessageData_ofExpr(v_snd_1526_);
if (v_isShared_1524_ == 0)
{
lean_ctor_set_tag(v___x_1523_, 7);
lean_ctor_set(v___x_1523_, 1, v___x_1540_);
lean_ctor_set(v___x_1523_, 0, v___x_1539_);
v___x_1542_ = v___x_1523_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v___x_1539_);
lean_ctor_set(v_reuseFailAlloc_1544_, 1, v___x_1540_);
v___x_1542_ = v_reuseFailAlloc_1544_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
lean_object* v___x_1543_; 
v___x_1543_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v___x_1542_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_);
return v___x_1543_;
}
}
}
else
{
lean_object* v_a_1872_; lean_object* v___x_1874_; uint8_t v_isShared_1875_; uint8_t v_isSharedCheck_1879_; 
lean_del_object(v___x_1528_);
lean_dec(v_snd_1526_);
lean_dec(v_fst_1525_);
lean_del_object(v___x_1523_);
lean_dec(v_fst_1521_);
lean_del_object(v___x_1519_);
lean_dec(v_fst_1517_);
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
lean_dec_ref(v_i_1254_);
v_a_1872_ = lean_ctor_get(v___x_1530_, 0);
v_isSharedCheck_1879_ = !lean_is_exclusive(v___x_1530_);
if (v_isSharedCheck_1879_ == 0)
{
v___x_1874_ = v___x_1530_;
v_isShared_1875_ = v_isSharedCheck_1879_;
goto v_resetjp_1873_;
}
else
{
lean_inc(v_a_1872_);
lean_dec(v___x_1530_);
v___x_1874_ = lean_box(0);
v_isShared_1875_ = v_isSharedCheck_1879_;
goto v_resetjp_1873_;
}
v_resetjp_1873_:
{
lean_object* v___x_1877_; 
if (v_isShared_1875_ == 0)
{
v___x_1877_ = v___x_1874_;
goto v_reusejp_1876_;
}
else
{
lean_object* v_reuseFailAlloc_1878_; 
v_reuseFailAlloc_1878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1878_, 0, v_a_1872_);
v___x_1877_ = v_reuseFailAlloc_1878_;
goto v_reusejp_1876_;
}
v_reusejp_1876_:
{
return v___x_1877_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_1892_; 
lean_dec_ref(v_name_1313_);
lean_dec_ref(v_lem_1255_);
lean_dec_ref(v_i_1254_);
v_a_1885_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1892_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1892_ == 0)
{
v___x_1887_ = v___x_1513_;
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_a_1885_);
lean_dec(v___x_1513_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v___x_1890_; 
if (v_isShared_1888_ == 0)
{
v___x_1890_ = v___x_1887_;
goto v_reusejp_1889_;
}
else
{
lean_object* v_reuseFailAlloc_1891_; 
v_reuseFailAlloc_1891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1891_, 0, v_a_1885_);
v___x_1890_ = v_reuseFailAlloc_1891_;
goto v_reusejp_1889_;
}
v_reusejp_1889_:
{
return v___x_1890_;
}
}
}
v___jp_1264_:
{
lean_object* v___x_1269_; lean_object* v___x_1270_; 
v___x_1269_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1269_, 0, v___y_1266_);
lean_ctor_set(v___x_1269_, 1, v___y_1267_);
lean_ctor_set(v___x_1269_, 2, v___y_1265_);
lean_ctor_set(v___x_1269_, 3, v_pattern_1268_);
v___x_1270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1269_);
return v___x_1270_;
}
v___jp_1271_:
{
lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v_a_1286_; lean_object* v___x_1288_; uint8_t v_isShared_1289_; uint8_t v_isSharedCheck_1293_; 
lean_dec_ref(v___y_1274_);
lean_dec(v___y_1273_);
lean_dec_ref(v___y_1272_);
v___x_1282_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___closed__1);
v___x_1283_ = l_Lean_indentExpr(v___y_1275_);
v___x_1284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1282_);
lean_ctor_set(v___x_1284_, 1, v___x_1283_);
v___x_1285_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v___x_1284_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_);
v_a_1286_ = lean_ctor_get(v___x_1285_, 0);
v_isSharedCheck_1293_ = !lean_is_exclusive(v___x_1285_);
if (v_isSharedCheck_1293_ == 0)
{
v___x_1288_ = v___x_1285_;
v_isShared_1289_ = v_isSharedCheck_1293_;
goto v_resetjp_1287_;
}
else
{
lean_inc(v_a_1286_);
lean_dec(v___x_1285_);
v___x_1288_ = lean_box(0);
v_isShared_1289_ = v_isSharedCheck_1293_;
goto v_resetjp_1287_;
}
v_resetjp_1287_:
{
lean_object* v___x_1291_; 
if (v_isShared_1289_ == 0)
{
v___x_1291_ = v___x_1288_;
goto v_reusejp_1290_;
}
else
{
lean_object* v_reuseFailAlloc_1292_; 
v_reuseFailAlloc_1292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1292_, 0, v_a_1286_);
v___x_1291_ = v_reuseFailAlloc_1292_;
goto v_reusejp_1290_;
}
v_reusejp_1290_:
{
return v___x_1291_;
}
}
}
v___jp_1294_:
{
lean_object* v___x_1303_; 
v___x_1303_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___y_1302_, v___y_1296_, v___y_1295_, v___y_1298_, v___y_1301_);
if (lean_obj_tag(v___x_1303_) == 0)
{
lean_object* v_a_1304_; 
v_a_1304_ = lean_ctor_get(v___x_1303_, 0);
lean_inc(v_a_1304_);
lean_dec_ref_known(v___x_1303_, 1);
v___y_1265_ = v___y_1297_;
v___y_1266_ = v___y_1299_;
v___y_1267_ = v___y_1300_;
v_pattern_1268_ = v_a_1304_;
goto v___jp_1264_;
}
else
{
lean_object* v_a_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1312_; 
lean_dec_ref(v___y_1300_);
lean_dec(v___y_1299_);
lean_dec_ref(v___y_1297_);
v_a_1305_ = lean_ctor_get(v___x_1303_, 0);
v_isSharedCheck_1312_ = !lean_is_exclusive(v___x_1303_);
if (v_isSharedCheck_1312_ == 0)
{
v___x_1307_ = v___x_1303_;
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_a_1305_);
lean_dec(v___x_1303_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1310_; 
if (v_isShared_1308_ == 0)
{
v___x_1310_ = v___x_1307_;
goto v_reusejp_1309_;
}
else
{
lean_object* v_reuseFailAlloc_1311_; 
v_reuseFailAlloc_1311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1311_, 0, v_a_1305_);
v___x_1310_ = v_reuseFailAlloc_1311_;
goto v_reusejp_1309_;
}
v_reusejp_1309_:
{
return v___x_1310_;
}
}
}
}
v___jp_1315_:
{
lean_object* v___x_1329_; 
lean_inc_ref(v_name_1313_);
v___x_1329_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(v_name_1313_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
if (lean_obj_tag(v___x_1329_) == 0)
{
lean_object* v_a_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v_a_1330_ = lean_ctor_get(v___x_1329_, 0);
lean_inc(v_a_1330_);
lean_dec_ref_known(v___x_1329_, 1);
v___x_1331_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0));
v___x_1332_ = lean_mk_empty_array_with_capacity(v___y_1316_);
lean_dec(v___y_1316_);
v___x_1333_ = lean_array_push(v___y_1321_, v_a_1330_);
lean_inc_ref(v___x_1332_);
v___x_1334_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1331_);
lean_ctor_set(v___x_1334_, 1, v___x_1332_);
lean_ctor_set(v___x_1334_, 2, v___x_1333_);
v___x_1335_ = lean_array_push(v___y_1319_, v___x_1334_);
v___x_1336_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1336_, 0, v___x_1331_);
lean_ctor_set(v___x_1336_, 1, v___x_1332_);
lean_ctor_set(v___x_1336_, 2, v___x_1335_);
v___x_1337_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_1320_, v___x_1336_, v___y_1318_, v___y_1323_, v___y_1324_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
if (lean_obj_tag(v___x_1337_) == 0)
{
lean_object* v_a_1338_; lean_object* v___x_1339_; 
v_a_1338_ = lean_ctor_get(v___x_1337_, 0);
lean_inc(v_a_1338_);
lean_dec_ref_known(v___x_1337_, 1);
v___x_1339_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(v_name_1313_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
if (lean_obj_tag(v___x_1339_) == 0)
{
lean_object* v_a_1340_; lean_object* v___x_1341_; uint8_t v___x_1342_; lean_object* v___x_1343_; 
v_a_1340_ = lean_ctor_get(v___x_1339_, 0);
lean_inc(v_a_1340_);
lean_dec_ref_known(v___x_1339_, 1);
v___x_1341_ = lean_box(0);
v___x_1342_ = 0;
v___x_1343_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_1340_, v___x_1341_, v___x_1342_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
if (lean_obj_tag(v___x_1343_) == 0)
{
lean_object* v_a_1344_; lean_object* v_snd_1345_; lean_object* v_snd_1346_; lean_object* v___x_1347_; 
v_a_1344_ = lean_ctor_get(v___x_1343_, 0);
lean_inc(v_a_1344_);
lean_dec_ref_known(v___x_1343_, 1);
v_snd_1345_ = lean_ctor_get(v_a_1344_, 1);
lean_inc(v_snd_1345_);
lean_dec(v_a_1344_);
v_snd_1346_ = lean_ctor_get(v_snd_1345_, 1);
lean_inc_n(v_snd_1346_, 2);
lean_dec(v_snd_1345_);
lean_inc(v___y_1328_);
lean_inc_ref(v___y_1327_);
lean_inc(v___y_1326_);
lean_inc_ref(v___y_1325_);
v___x_1347_ = lean_whnf(v_snd_1346_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
if (lean_obj_tag(v___x_1347_) == 0)
{
lean_object* v_a_1348_; 
v_a_1348_ = lean_ctor_get(v___x_1347_, 0);
lean_inc(v_a_1348_);
lean_dec_ref_known(v___x_1347_, 1);
if (lean_obj_tag(v_a_1348_) == 5)
{
lean_object* v_fn_1349_; 
v_fn_1349_ = lean_ctor_get(v_a_1348_, 0);
if (lean_obj_tag(v_fn_1349_) == 5)
{
lean_dec(v_snd_1346_);
if (v_symm_1314_ == 0)
{
lean_object* v_arg_1350_; 
lean_inc_ref(v_fn_1349_);
lean_dec_ref_known(v_a_1348_, 2);
v_arg_1350_ = lean_ctor_get(v_fn_1349_, 1);
lean_inc_ref(v_arg_1350_);
lean_dec_ref_known(v_fn_1349_, 2);
v___y_1295_ = v___y_1326_;
v___y_1296_ = v___y_1325_;
v___y_1297_ = v___y_1317_;
v___y_1298_ = v___y_1327_;
v___y_1299_ = v_filtered_1322_;
v___y_1300_ = v_a_1338_;
v___y_1301_ = v___y_1328_;
v___y_1302_ = v_arg_1350_;
goto v___jp_1294_;
}
else
{
lean_object* v_arg_1351_; 
v_arg_1351_ = lean_ctor_get(v_a_1348_, 1);
lean_inc_ref(v_arg_1351_);
lean_dec_ref_known(v_a_1348_, 2);
v___y_1295_ = v___y_1326_;
v___y_1296_ = v___y_1325_;
v___y_1297_ = v___y_1317_;
v___y_1298_ = v___y_1327_;
v___y_1299_ = v_filtered_1322_;
v___y_1300_ = v_a_1338_;
v___y_1301_ = v___y_1328_;
v___y_1302_ = v_arg_1351_;
goto v___jp_1294_;
}
}
else
{
lean_dec_ref_known(v_a_1348_, 2);
v___y_1272_ = v___y_1317_;
v___y_1273_ = v_filtered_1322_;
v___y_1274_ = v_a_1338_;
v___y_1275_ = v_snd_1346_;
v___y_1276_ = v___y_1323_;
v___y_1277_ = v___y_1324_;
v___y_1278_ = v___y_1325_;
v___y_1279_ = v___y_1326_;
v___y_1280_ = v___y_1327_;
v___y_1281_ = v___y_1328_;
goto v___jp_1271_;
}
}
else
{
lean_dec(v_a_1348_);
v___y_1272_ = v___y_1317_;
v___y_1273_ = v_filtered_1322_;
v___y_1274_ = v_a_1338_;
v___y_1275_ = v_snd_1346_;
v___y_1276_ = v___y_1323_;
v___y_1277_ = v___y_1324_;
v___y_1278_ = v___y_1325_;
v___y_1279_ = v___y_1326_;
v___y_1280_ = v___y_1327_;
v___y_1281_ = v___y_1328_;
goto v___jp_1271_;
}
}
else
{
lean_object* v_a_1352_; lean_object* v___x_1354_; uint8_t v_isShared_1355_; uint8_t v_isSharedCheck_1359_; 
lean_dec(v_snd_1346_);
lean_dec(v_a_1338_);
lean_dec(v_filtered_1322_);
lean_dec_ref(v___y_1317_);
v_a_1352_ = lean_ctor_get(v___x_1347_, 0);
v_isSharedCheck_1359_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1359_ == 0)
{
v___x_1354_ = v___x_1347_;
v_isShared_1355_ = v_isSharedCheck_1359_;
goto v_resetjp_1353_;
}
else
{
lean_inc(v_a_1352_);
lean_dec(v___x_1347_);
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
else
{
lean_object* v_a_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1367_; 
lean_dec(v_a_1338_);
lean_dec(v_filtered_1322_);
lean_dec_ref(v___y_1317_);
v_a_1360_ = lean_ctor_get(v___x_1343_, 0);
v_isSharedCheck_1367_ = !lean_is_exclusive(v___x_1343_);
if (v_isSharedCheck_1367_ == 0)
{
v___x_1362_ = v___x_1343_;
v_isShared_1363_ = v_isSharedCheck_1367_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_a_1360_);
lean_dec(v___x_1343_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1367_;
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
lean_object* v_reuseFailAlloc_1366_; 
v_reuseFailAlloc_1366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1366_, 0, v_a_1360_);
v___x_1365_ = v_reuseFailAlloc_1366_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
return v___x_1365_;
}
}
}
}
else
{
lean_object* v_a_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1375_; 
lean_dec(v_a_1338_);
lean_dec(v_filtered_1322_);
lean_dec_ref(v___y_1317_);
v_a_1368_ = lean_ctor_get(v___x_1339_, 0);
v_isSharedCheck_1375_ = !lean_is_exclusive(v___x_1339_);
if (v_isSharedCheck_1375_ == 0)
{
v___x_1370_ = v___x_1339_;
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_a_1368_);
lean_dec(v___x_1339_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1373_; 
if (v_isShared_1371_ == 0)
{
v___x_1373_ = v___x_1370_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1374_; 
v_reuseFailAlloc_1374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1374_, 0, v_a_1368_);
v___x_1373_ = v_reuseFailAlloc_1374_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
return v___x_1373_;
}
}
}
}
else
{
lean_object* v_a_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1383_; 
lean_dec(v_filtered_1322_);
lean_dec_ref(v___y_1317_);
lean_dec_ref(v_name_1313_);
v_a_1376_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1383_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1378_ = v___x_1337_;
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_a_1376_);
lean_dec(v___x_1337_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1381_; 
if (v_isShared_1379_ == 0)
{
v___x_1381_ = v___x_1378_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v_a_1376_);
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
else
{
lean_object* v_a_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1391_; 
lean_dec(v_filtered_1322_);
lean_dec_ref(v___y_1321_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v_name_1313_);
v_a_1384_ = lean_ctor_get(v___x_1329_, 0);
v_isSharedCheck_1391_ = !lean_is_exclusive(v___x_1329_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1386_ = v___x_1329_;
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_a_1384_);
lean_dec(v___x_1329_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1389_; 
if (v_isShared_1387_ == 0)
{
v___x_1389_ = v___x_1386_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_a_1384_);
v___x_1389_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
return v___x_1389_;
}
}
}
}
v___jp_1392_:
{
lean_object* v___x_1405_; 
v___x_1405_ = lean_box(0);
v___y_1316_ = v___y_1393_;
v___y_1317_ = v___y_1395_;
v___y_1318_ = v___y_1396_;
v___y_1319_ = v___y_1402_;
v___y_1320_ = v___y_1403_;
v___y_1321_ = v___y_1399_;
v_filtered_1322_ = v___x_1405_;
v___y_1323_ = v___y_1400_;
v___y_1324_ = v___y_1404_;
v___y_1325_ = v___y_1394_;
v___y_1326_ = v___y_1397_;
v___y_1327_ = v___y_1401_;
v___y_1328_ = v___y_1398_;
goto v___jp_1315_;
}
v___jp_1406_:
{
lean_object* v___x_1422_; 
v___x_1422_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___y_1412_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_);
if (lean_obj_tag(v___x_1422_) == 0)
{
lean_object* v_a_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; size_t v_sz_1427_; lean_object* v___x_1428_; 
v_a_1423_ = lean_ctor_get(v___x_1422_, 0);
lean_inc(v_a_1423_);
lean_dec_ref_known(v___x_1422_, 1);
v___x_1424_ = lean_unsigned_to_nat(1u);
v___x_1425_ = lean_mk_empty_array_with_capacity(v___x_1424_);
lean_inc_ref(v___x_1425_);
v___x_1426_ = lean_array_push(v___x_1425_, v_a_1423_);
v_sz_1427_ = lean_array_size(v___y_1415_);
v___x_1428_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg(v___y_1415_, v_sz_1427_, v___y_1414_, v___x_1426_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_);
lean_dec(v___y_1415_);
if (lean_obj_tag(v___x_1428_) == 0)
{
if (v___y_1408_ == 0)
{
if (v___y_1410_ == 0)
{
lean_object* v_a_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; 
v_a_1429_ = lean_ctor_get(v___x_1428_, 0);
lean_inc_n(v_a_1429_, 2);
lean_dec_ref_known(v___x_1428_, 1);
v___x_1430_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg___closed__0));
v___x_1431_ = lean_mk_empty_array_with_capacity(v___y_1407_);
v___x_1432_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1432_, 0, v___x_1430_);
lean_ctor_set(v___x_1432_, 1, v___x_1431_);
lean_ctor_set(v___x_1432_, 2, v_a_1429_);
lean_inc(v___y_1413_);
v___x_1433_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_1413_, v___x_1432_, v___y_1411_, v___y_1416_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_);
if (lean_obj_tag(v___x_1433_) == 0)
{
lean_object* v_a_1434_; lean_object* v___x_1435_; 
v_a_1434_ = lean_ctor_get(v___x_1433_, 0);
lean_inc(v_a_1434_);
lean_dec_ref_known(v___x_1433_, 1);
v___x_1435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1435_, 0, v_a_1434_);
v___y_1316_ = v___y_1407_;
v___y_1317_ = v___y_1409_;
v___y_1318_ = v___y_1411_;
v___y_1319_ = v_a_1429_;
v___y_1320_ = v___y_1413_;
v___y_1321_ = v___x_1425_;
v_filtered_1322_ = v___x_1435_;
v___y_1323_ = v___y_1416_;
v___y_1324_ = v___y_1417_;
v___y_1325_ = v___y_1418_;
v___y_1326_ = v___y_1419_;
v___y_1327_ = v___y_1420_;
v___y_1328_ = v___y_1421_;
goto v___jp_1315_;
}
else
{
lean_object* v_a_1436_; lean_object* v___x_1438_; uint8_t v_isShared_1439_; uint8_t v_isSharedCheck_1443_; 
lean_dec(v_a_1429_);
lean_dec_ref(v___x_1425_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1409_);
lean_dec(v___y_1407_);
lean_dec_ref(v_name_1313_);
v_a_1436_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1438_ = v___x_1433_;
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
else
{
lean_inc(v_a_1436_);
lean_dec(v___x_1433_);
v___x_1438_ = lean_box(0);
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
v_resetjp_1437_:
{
lean_object* v___x_1441_; 
if (v_isShared_1439_ == 0)
{
v___x_1441_ = v___x_1438_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v_a_1436_);
v___x_1441_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
return v___x_1441_;
}
}
}
}
else
{
lean_object* v_a_1444_; 
v_a_1444_ = lean_ctor_get(v___x_1428_, 0);
lean_inc(v_a_1444_);
lean_dec_ref_known(v___x_1428_, 1);
v___y_1393_ = v___y_1407_;
v___y_1394_ = v___y_1418_;
v___y_1395_ = v___y_1409_;
v___y_1396_ = v___y_1411_;
v___y_1397_ = v___y_1419_;
v___y_1398_ = v___y_1421_;
v___y_1399_ = v___x_1425_;
v___y_1400_ = v___y_1416_;
v___y_1401_ = v___y_1420_;
v___y_1402_ = v_a_1444_;
v___y_1403_ = v___y_1413_;
v___y_1404_ = v___y_1417_;
goto v___jp_1392_;
}
}
else
{
lean_object* v_a_1445_; 
v_a_1445_ = lean_ctor_get(v___x_1428_, 0);
lean_inc(v_a_1445_);
lean_dec_ref_known(v___x_1428_, 1);
v___y_1393_ = v___y_1407_;
v___y_1394_ = v___y_1418_;
v___y_1395_ = v___y_1409_;
v___y_1396_ = v___y_1411_;
v___y_1397_ = v___y_1419_;
v___y_1398_ = v___y_1421_;
v___y_1399_ = v___x_1425_;
v___y_1400_ = v___y_1416_;
v___y_1401_ = v___y_1420_;
v___y_1402_ = v_a_1445_;
v___y_1403_ = v___y_1413_;
v___y_1404_ = v___y_1417_;
goto v___jp_1392_;
}
}
else
{
lean_object* v_a_1446_; lean_object* v___x_1448_; uint8_t v_isShared_1449_; uint8_t v_isSharedCheck_1453_; 
lean_dec_ref(v___x_1425_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1409_);
lean_dec(v___y_1407_);
lean_dec_ref(v_name_1313_);
v_a_1446_ = lean_ctor_get(v___x_1428_, 0);
v_isSharedCheck_1453_ = !lean_is_exclusive(v___x_1428_);
if (v_isSharedCheck_1453_ == 0)
{
v___x_1448_ = v___x_1428_;
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
else
{
lean_inc(v_a_1446_);
lean_dec(v___x_1428_);
v___x_1448_ = lean_box(0);
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
v_resetjp_1447_:
{
lean_object* v___x_1451_; 
if (v_isShared_1449_ == 0)
{
v___x_1451_ = v___x_1448_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_a_1446_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
}
}
else
{
lean_object* v_a_1454_; lean_object* v___x_1456_; uint8_t v_isShared_1457_; uint8_t v_isSharedCheck_1461_; 
lean_dec(v___y_1415_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1409_);
lean_dec(v___y_1407_);
lean_dec_ref(v_name_1313_);
v_a_1454_ = lean_ctor_get(v___x_1422_, 0);
v_isSharedCheck_1461_ = !lean_is_exclusive(v___x_1422_);
if (v_isSharedCheck_1461_ == 0)
{
v___x_1456_ = v___x_1422_;
v_isShared_1457_ = v_isSharedCheck_1461_;
goto v_resetjp_1455_;
}
else
{
lean_inc(v_a_1454_);
lean_dec(v___x_1422_);
v___x_1456_ = lean_box(0);
v_isShared_1457_ = v_isSharedCheck_1461_;
goto v_resetjp_1455_;
}
v_resetjp_1455_:
{
lean_object* v___x_1459_; 
if (v_isShared_1457_ == 0)
{
v___x_1459_ = v___x_1456_;
goto v_reusejp_1458_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v_a_1454_);
v___x_1459_ = v_reuseFailAlloc_1460_;
goto v_reusejp_1458_;
}
v_reusejp_1458_:
{
return v___x_1459_;
}
}
}
}
v___jp_1462_:
{
if (v_a_1477_ == 0)
{
v___y_1407_ = v___y_1463_;
v___y_1408_ = v___y_1464_;
v___y_1409_ = v___y_1465_;
v___y_1410_ = v___y_1470_;
v___y_1411_ = v_a_1477_;
v___y_1412_ = v___y_1467_;
v___y_1413_ = v___y_1474_;
v___y_1414_ = v___y_1468_;
v___y_1415_ = v___y_1476_;
v___y_1416_ = v___y_1475_;
v___y_1417_ = v___y_1471_;
v___y_1418_ = v___y_1469_;
v___y_1419_ = v___y_1473_;
v___y_1420_ = v___y_1472_;
v___y_1421_ = v___y_1466_;
goto v___jp_1406_;
}
else
{
lean_object* v___x_1478_; 
lean_inc(v___y_1474_);
v___x_1478_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(v___y_1474_, v___y_1475_, v___y_1471_, v___y_1469_, v___y_1473_, v___y_1472_, v___y_1466_);
if (lean_obj_tag(v___x_1478_) == 0)
{
lean_dec_ref_known(v___x_1478_, 1);
v___y_1407_ = v___y_1463_;
v___y_1408_ = v___y_1464_;
v___y_1409_ = v___y_1465_;
v___y_1410_ = v___y_1470_;
v___y_1411_ = v_a_1477_;
v___y_1412_ = v___y_1467_;
v___y_1413_ = v___y_1474_;
v___y_1414_ = v___y_1468_;
v___y_1415_ = v___y_1476_;
v___y_1416_ = v___y_1475_;
v___y_1417_ = v___y_1471_;
v___y_1418_ = v___y_1469_;
v___y_1419_ = v___y_1473_;
v___y_1420_ = v___y_1472_;
v___y_1421_ = v___y_1466_;
goto v___jp_1406_;
}
else
{
lean_object* v_a_1479_; lean_object* v___x_1481_; uint8_t v_isShared_1482_; uint8_t v_isSharedCheck_1486_; 
lean_dec(v___y_1476_);
lean_dec(v___y_1474_);
lean_dec_ref(v___y_1467_);
lean_dec_ref(v___y_1465_);
lean_dec(v___y_1463_);
lean_dec_ref(v_name_1313_);
v_a_1479_ = lean_ctor_get(v___x_1478_, 0);
v_isSharedCheck_1486_ = !lean_is_exclusive(v___x_1478_);
if (v_isSharedCheck_1486_ == 0)
{
v___x_1481_ = v___x_1478_;
v_isShared_1482_ = v_isSharedCheck_1486_;
goto v_resetjp_1480_;
}
else
{
lean_inc(v_a_1479_);
lean_dec(v___x_1478_);
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
v___jp_1487_:
{
if (lean_obj_tag(v___y_1502_) == 0)
{
lean_object* v_a_1503_; uint8_t v___x_1504_; 
v_a_1503_ = lean_ctor_get(v___y_1502_, 0);
lean_inc(v_a_1503_);
lean_dec_ref_known(v___y_1502_, 1);
v___x_1504_ = lean_unbox(v_a_1503_);
lean_dec(v_a_1503_);
v___y_1463_ = v___y_1488_;
v___y_1464_ = v___y_1489_;
v___y_1465_ = v___y_1490_;
v___y_1466_ = v___y_1491_;
v___y_1467_ = v___y_1492_;
v___y_1468_ = v___y_1493_;
v___y_1469_ = v___y_1494_;
v___y_1470_ = v___y_1495_;
v___y_1471_ = v___y_1496_;
v___y_1472_ = v___y_1497_;
v___y_1473_ = v___y_1498_;
v___y_1474_ = v___y_1499_;
v___y_1475_ = v___y_1500_;
v___y_1476_ = v___y_1501_;
v_a_1477_ = v___x_1504_;
goto v___jp_1462_;
}
else
{
lean_object* v_a_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1512_; 
lean_dec(v___y_1501_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1492_);
lean_dec_ref(v___y_1490_);
lean_dec(v___y_1488_);
lean_dec_ref(v_name_1313_);
v_a_1505_ = lean_ctor_get(v___y_1502_, 0);
v_isSharedCheck_1512_ = !lean_is_exclusive(v___y_1502_);
if (v_isSharedCheck_1512_ == 0)
{
v___x_1507_ = v___y_1502_;
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_a_1505_);
lean_dec(v___y_1502_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v___x_1510_; 
if (v_isShared_1508_ == 0)
{
v___x_1510_ = v___x_1507_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1511_; 
v_reuseFailAlloc_1511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1511_, 0, v_a_1505_);
v___x_1510_ = v_reuseFailAlloc_1511_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
return v___x_1510_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try___boxed(lean_object* v_i_1893_, lean_object* v_lem_1894_, lean_object* v_assignableMVars_1895_, lean_object* v_a_1896_, lean_object* v_a_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_, lean_object* v_a_1900_, lean_object* v_a_1901_, lean_object* v_a_1902_){
_start:
{
lean_object* v_res_1903_; 
v_res_1903_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_RwLemma_try(v_i_1893_, v_lem_1894_, v_assignableMVars_1895_, v_a_1896_, v_a_1897_, v_a_1898_, v_a_1899_, v_a_1900_, v_a_1901_);
lean_dec(v_a_1901_);
lean_dec_ref(v_a_1900_);
lean_dec(v_a_1899_);
lean_dec_ref(v_a_1898_);
lean_dec(v_a_1897_);
lean_dec_ref(v_a_1896_);
lean_dec_ref(v_assignableMVars_1895_);
return v_res_1903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0(lean_object* v_mvarId_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_){
_start:
{
lean_object* v___x_1912_; 
v___x_1912_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___redArg(v_mvarId_1904_, v___y_1908_);
return v___x_1912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0___boxed(lean_object* v_mvarId_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_){
_start:
{
lean_object* v_res_1921_; 
v_res_1921_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0(v_mvarId_1913_, v___y_1914_, v___y_1915_, v___y_1916_, v___y_1917_, v___y_1918_, v___y_1919_);
lean_dec(v___y_1919_);
lean_dec_ref(v___y_1918_);
lean_dec(v___y_1917_);
lean_dec_ref(v___y_1916_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
lean_dec(v_mvarId_1913_);
return v_res_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1(lean_object* v_00_u03b1_1922_, lean_object* v_msg_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_){
_start:
{
lean_object* v___x_1931_; 
v___x_1931_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___redArg(v_msg_1923_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_);
return v___x_1931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1___boxed(lean_object* v_00_u03b1_1932_, lean_object* v_msg_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_){
_start:
{
lean_object* v_res_1941_; 
v_res_1941_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__1(v_00_u03b1_1932_, v_msg_1933_, v___y_1934_, v___y_1935_, v___y_1936_, v___y_1937_, v___y_1938_, v___y_1939_);
lean_dec(v___y_1939_);
lean_dec_ref(v___y_1938_);
lean_dec(v___y_1937_);
lean_dec_ref(v___y_1936_);
lean_dec(v___y_1935_);
lean_dec_ref(v___y_1934_);
return v_res_1941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5(lean_object* v_mvarId_1942_, lean_object* v_val_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v___x_1951_; 
v___x_1951_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___redArg(v_mvarId_1942_, v_val_1943_, v___y_1947_);
return v___x_1951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5___boxed(lean_object* v_mvarId_1952_, lean_object* v_val_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_){
_start:
{
lean_object* v_res_1961_; 
v_res_1961_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5(v_mvarId_1952_, v_val_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_);
lean_dec(v___y_1959_);
lean_dec_ref(v___y_1958_);
lean_dec(v___y_1957_);
lean_dec_ref(v___y_1956_);
lean_dec(v___y_1955_);
lean_dec_ref(v___y_1954_);
return v_res_1961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7(lean_object* v_as_1962_, size_t v_sz_1963_, size_t v_i_1964_, lean_object* v_b_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_){
_start:
{
lean_object* v___x_1973_; 
v___x_1973_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___redArg(v_as_1962_, v_sz_1963_, v_i_1964_, v_b_1965_, v___y_1968_, v___y_1969_, v___y_1970_, v___y_1971_);
return v___x_1973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7___boxed(lean_object* v_as_1974_, lean_object* v_sz_1975_, lean_object* v_i_1976_, lean_object* v_b_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_){
_start:
{
size_t v_sz_boxed_1985_; size_t v_i_boxed_1986_; lean_object* v_res_1987_; 
v_sz_boxed_1985_ = lean_unbox_usize(v_sz_1975_);
lean_dec(v_sz_1975_);
v_i_boxed_1986_ = lean_unbox_usize(v_i_1976_);
lean_dec(v_i_1976_);
v_res_1987_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__7(v_as_1974_, v_sz_boxed_1985_, v_i_boxed_1986_, v_b_1977_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_, v___y_1982_, v___y_1983_);
lean_dec(v___y_1983_);
lean_dec_ref(v___y_1982_);
lean_dec(v___y_1981_);
lean_dec_ref(v___y_1980_);
lean_dec(v___y_1979_);
lean_dec_ref(v___y_1978_);
lean_dec_ref(v_as_1974_);
return v_res_1987_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0(lean_object* v_00_u03b2_1988_, lean_object* v_x_1989_, lean_object* v_x_1990_){
_start:
{
uint8_t v___x_1991_; 
v___x_1991_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___redArg(v_x_1989_, v_x_1990_);
return v___x_1991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1992_, lean_object* v_x_1993_, lean_object* v_x_1994_){
_start:
{
uint8_t v_res_1995_; lean_object* v_r_1996_; 
v_res_1995_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0(v_00_u03b2_1992_, v_x_1993_, v_x_1994_);
lean_dec(v_x_1994_);
lean_dec_ref(v_x_1993_);
v_r_1996_ = lean_box(v_res_1995_);
return v_r_1996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7(lean_object* v_00_u03b2_1997_, lean_object* v_x_1998_, lean_object* v_x_1999_, lean_object* v_x_2000_){
_start:
{
lean_object* v___x_2001_; 
v___x_2001_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7___redArg(v_x_1998_, v_x_1999_, v_x_2000_);
return v___x_2001_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6(lean_object* v_00_u03b2_2002_, lean_object* v_x_2003_, size_t v_x_2004_, lean_object* v_x_2005_){
_start:
{
uint8_t v___x_2006_; 
v___x_2006_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___redArg(v_x_2003_, v_x_2004_, v_x_2005_);
return v___x_2006_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6___boxed(lean_object* v_00_u03b2_2007_, lean_object* v_x_2008_, lean_object* v_x_2009_, lean_object* v_x_2010_){
_start:
{
size_t v_x_107076__boxed_2011_; uint8_t v_res_2012_; lean_object* v_r_2013_; 
v_x_107076__boxed_2011_ = lean_unbox_usize(v_x_2009_);
lean_dec(v_x_2009_);
v_res_2012_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6(v_00_u03b2_2007_, v_x_2008_, v_x_107076__boxed_2011_, v_x_2010_);
lean_dec(v_x_2010_);
lean_dec_ref(v_x_2008_);
v_r_2013_ = lean_box(v_res_2012_);
return v_r_2013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12(lean_object* v_00_u03b2_2014_, lean_object* v_x_2015_, size_t v_x_2016_, size_t v_x_2017_, lean_object* v_x_2018_, lean_object* v_x_2019_){
_start:
{
lean_object* v___x_2020_; 
v___x_2020_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___redArg(v_x_2015_, v_x_2016_, v_x_2017_, v_x_2018_, v_x_2019_);
return v___x_2020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12___boxed(lean_object* v_00_u03b2_2021_, lean_object* v_x_2022_, lean_object* v_x_2023_, lean_object* v_x_2024_, lean_object* v_x_2025_, lean_object* v_x_2026_){
_start:
{
size_t v_x_107087__boxed_2027_; size_t v_x_107088__boxed_2028_; lean_object* v_res_2029_; 
v_x_107087__boxed_2027_ = lean_unbox_usize(v_x_2023_);
lean_dec(v_x_2023_);
v_x_107088__boxed_2028_ = lean_unbox_usize(v_x_2024_);
lean_dec(v_x_2024_);
v_res_2029_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12(v_00_u03b2_2021_, v_x_2022_, v_x_107087__boxed_2027_, v_x_107088__boxed_2028_, v_x_2025_, v_x_2026_);
return v_res_2029_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14(lean_object* v_00_u03b2_2030_, lean_object* v_keys_2031_, lean_object* v_vals_2032_, lean_object* v_heq_2033_, lean_object* v_i_2034_, lean_object* v_k_2035_){
_start:
{
uint8_t v___x_2036_; 
v___x_2036_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___redArg(v_keys_2031_, v_i_2034_, v_k_2035_);
return v___x_2036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14___boxed(lean_object* v_00_u03b2_2037_, lean_object* v_keys_2038_, lean_object* v_vals_2039_, lean_object* v_heq_2040_, lean_object* v_i_2041_, lean_object* v_k_2042_){
_start:
{
uint8_t v_res_2043_; lean_object* v_r_2044_; 
v_res_2043_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__0_spec__0_spec__6_spec__14(v_00_u03b2_2037_, v_keys_2038_, v_vals_2039_, v_heq_2040_, v_i_2041_, v_k_2042_);
lean_dec(v_k_2042_);
lean_dec_ref(v_vals_2039_);
lean_dec_ref(v_keys_2038_);
v_r_2044_ = lean_box(v_res_2043_);
return v_r_2044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17(lean_object* v_00_u03b2_2045_, lean_object* v_n_2046_, lean_object* v_k_2047_, lean_object* v_v_2048_){
_start:
{
lean_object* v___x_2049_; 
v___x_2049_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17___redArg(v_n_2046_, v_k_2047_, v_v_2048_);
return v___x_2049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18(lean_object* v_00_u03b2_2050_, size_t v_depth_2051_, lean_object* v_keys_2052_, lean_object* v_vals_2053_, lean_object* v_heq_2054_, lean_object* v_i_2055_, lean_object* v_entries_2056_){
_start:
{
lean_object* v___x_2057_; 
v___x_2057_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___redArg(v_depth_2051_, v_keys_2052_, v_vals_2053_, v_i_2055_, v_entries_2056_);
return v___x_2057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18___boxed(lean_object* v_00_u03b2_2058_, lean_object* v_depth_2059_, lean_object* v_keys_2060_, lean_object* v_vals_2061_, lean_object* v_heq_2062_, lean_object* v_i_2063_, lean_object* v_entries_2064_){
_start:
{
size_t v_depth_boxed_2065_; lean_object* v_res_2066_; 
v_depth_boxed_2065_ = lean_unbox_usize(v_depth_2059_);
lean_dec(v_depth_2059_);
v_res_2066_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__18(v_00_u03b2_2058_, v_depth_boxed_2065_, v_keys_2060_, v_vals_2061_, v_heq_2062_, v_i_2063_, v_entries_2064_);
lean_dec_ref(v_vals_2061_);
lean_dec_ref(v_keys_2060_);
return v_res_2066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18(lean_object* v_00_u03b2_2067_, lean_object* v_x_2068_, lean_object* v_x_2069_, lean_object* v_x_2070_, lean_object* v_x_2071_){
_start:
{
lean_object* v___x_2072_; 
v___x_2072_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_RwLemma_try_spec__5_spec__7_spec__12_spec__17_spec__18___redArg(v_x_2068_, v_x_2069_, v_x_2070_, v_x_2071_);
return v___x_2072_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(uint8_t builtin) {
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
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey_default);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedRwKey);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Rewrite(builtin);
}
#ifdef __cplusplus
}
#endif
