// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.Apply
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.SectionState public meta import Mathlib.Tactic.ClickSuggestions.Util import all Lean.Meta.Tactic.Apply
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
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_Meta_synthAppInstances_spec__0_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_pp_mvars;
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_BinderInfo_isExplicit(uint8_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_applyN_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Apply_0__Lean_Meta_reorderGoals(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_findLocalDeclWithType_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_apply_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyKey_default___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__10_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strong"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "goal-vdash"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__4_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__3_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__5_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__6_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__6_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⊢ "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__9_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__9_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__2_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__7_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__10_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 21, .m_data = "Goal accomplished! 🎉️"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__3_value),LEAN_SCALAR_PTR_LITERAL(15, 73, 51, 51, 21, 209, 204, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__4_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " does not unify with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0(lean_object* v_a_10_, lean_object* v_b_11_){
_start:
{
lean_object* v_numGoals_12_; lean_object* v_nameLength_13_; lean_object* v_replacementSize_14_; lean_object* v_name_15_; lean_object* v_numGoals_16_; lean_object* v_nameLength_17_; lean_object* v_replacementSize_18_; lean_object* v_name_19_; uint8_t v___x_20_; 
v_numGoals_12_ = lean_ctor_get(v_a_10_, 0);
v_nameLength_13_ = lean_ctor_get(v_a_10_, 1);
v_replacementSize_14_ = lean_ctor_get(v_a_10_, 2);
v_name_15_ = lean_ctor_get(v_a_10_, 3);
v_numGoals_16_ = lean_ctor_get(v_b_11_, 0);
v_nameLength_17_ = lean_ctor_get(v_b_11_, 1);
v_replacementSize_18_ = lean_ctor_get(v_b_11_, 2);
v_name_19_ = lean_ctor_get(v_b_11_, 3);
v___x_20_ = lean_nat_dec_lt(v_numGoals_12_, v_numGoals_16_);
if (v___x_20_ == 0)
{
uint8_t v___x_21_; 
v___x_21_ = lean_nat_dec_eq(v_numGoals_12_, v_numGoals_16_);
if (v___x_21_ == 0)
{
uint8_t v___x_22_; 
v___x_22_ = 2;
return v___x_22_;
}
else
{
uint8_t v___x_23_; 
v___x_23_ = lean_nat_dec_lt(v_nameLength_13_, v_nameLength_17_);
if (v___x_23_ == 0)
{
uint8_t v___x_24_; 
v___x_24_ = lean_nat_dec_eq(v_nameLength_13_, v_nameLength_17_);
if (v___x_24_ == 0)
{
uint8_t v___x_25_; 
v___x_25_ = 2;
return v___x_25_;
}
else
{
uint8_t v___x_26_; 
v___x_26_ = lean_nat_dec_lt(v_replacementSize_14_, v_replacementSize_18_);
if (v___x_26_ == 0)
{
uint8_t v___x_27_; 
v___x_27_ = lean_nat_dec_eq(v_replacementSize_14_, v_replacementSize_18_);
if (v___x_27_ == 0)
{
uint8_t v___x_28_; 
v___x_28_ = 2;
return v___x_28_;
}
else
{
uint8_t v___x_29_; 
v___x_29_ = lean_string_compare(v_name_15_, v_name_19_);
return v___x_29_;
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
else
{
uint8_t v___x_31_; 
v___x_31_ = 0;
return v___x_31_;
}
}
}
else
{
uint8_t v___x_32_; 
v___x_32_ = 0;
return v___x_32_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0___boxed(lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
uint8_t v_res_35_; lean_object* v_r_36_; 
v_res_35_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyKey___lam__0(v_a_33_, v_b_34_);
lean_dec_ref(v_b_34_);
lean_dec_ref(v_a_33_);
v_r_36_ = lean_box(v_res_35_);
return v_r_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg(lean_object* v___x_39_, lean_object* v___x_40_, lean_object* v_n_41_, lean_object* v_i_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_zero_48_; uint8_t v_isZero_49_; 
v_zero_48_ = lean_unsigned_to_nat(0u);
v_isZero_49_ = lean_nat_dec_eq(v_i_42_, v_zero_48_);
if (v_isZero_49_ == 1)
{
lean_object* v___x_50_; lean_object* v___x_51_; 
lean_dec(v_i_42_);
v___x_50_ = lean_box(v_isZero_49_);
v___x_51_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_51_, 0, v___x_50_);
return v___x_51_;
}
else
{
lean_object* v___x_52_; lean_object* v_one_53_; lean_object* v_n_54_; lean_object* v___y_56_; uint8_t v_a_57_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v_mvars_62_; lean_object* v_expr_63_; lean_object* v___x_64_; lean_object* v_mvars_65_; lean_object* v_expr_66_; lean_object* v___x_67_; lean_object* v___x_68_; uint8_t v___x_69_; 
v___x_52_ = l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
v_one_53_ = lean_unsigned_to_nat(1u);
v_n_54_ = lean_nat_sub(v_i_42_, v_one_53_);
lean_dec(v_i_42_);
v___x_59_ = lean_nat_sub(v_n_41_, v_n_54_);
v___x_60_ = lean_nat_sub(v___x_59_, v_one_53_);
lean_dec(v___x_59_);
v___x_61_ = lean_array_get_borrowed(v___x_52_, v___x_39_, v___x_60_);
v_mvars_62_ = lean_ctor_get(v___x_61_, 1);
v_expr_63_ = lean_ctor_get(v___x_61_, 2);
v___x_64_ = lean_array_get_borrowed(v___x_52_, v___x_40_, v___x_60_);
lean_dec(v___x_60_);
v_mvars_65_ = lean_ctor_get(v___x_64_, 1);
v_expr_66_ = lean_ctor_get(v___x_64_, 2);
v___x_67_ = lean_array_get_size(v_mvars_62_);
v___x_68_ = lean_array_get_size(v_mvars_65_);
v___x_69_ = lean_nat_dec_eq(v___x_67_, v___x_68_);
if (v___x_69_ == 0)
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = lean_box(v___x_69_);
v___x_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
v___y_56_ = v___x_71_;
v_a_57_ = v___x_69_;
goto v___jp_55_;
}
else
{
lean_object* v___x_72_; 
lean_inc_ref(v_expr_66_);
lean_inc_ref(v_expr_63_);
v___x_72_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(v_expr_63_, v_expr_66_, v___y_43_, v___y_44_, v___y_45_, v___y_46_);
if (lean_obj_tag(v___x_72_) == 0)
{
lean_object* v_a_73_; uint8_t v___x_74_; 
v_a_73_ = lean_ctor_get(v___x_72_, 0);
lean_inc(v_a_73_);
v___x_74_ = lean_unbox(v_a_73_);
lean_dec(v_a_73_);
v___y_56_ = v___x_72_;
v_a_57_ = v___x_74_;
goto v___jp_55_;
}
else
{
lean_dec(v_n_54_);
return v___x_72_;
}
}
v___jp_55_:
{
if (v_a_57_ == 0)
{
lean_dec(v_n_54_);
return v___y_56_;
}
else
{
lean_dec_ref(v___y_56_);
v_i_42_ = v_n_54_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg___boxed(lean_object* v___x_75_, lean_object* v___x_76_, lean_object* v_n_77_, lean_object* v_i_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg(v___x_75_, v___x_76_, v_n_77_, v_i_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
lean_dec(v_n_77_);
lean_dec_ref(v___x_76_);
lean_dec_ref(v___x_75_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate(lean_object* v_a_85_, lean_object* v_b_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v_newGoals_92_; lean_object* v_newGoals_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v_newGoals_92_ = lean_ctor_get(v_a_85_, 4);
v_newGoals_93_ = lean_ctor_get(v_b_86_, 4);
v___x_94_ = lean_array_get_size(v_newGoals_92_);
v___x_95_ = lean_array_get_size(v_newGoals_93_);
v___x_96_ = lean_nat_dec_eq(v___x_94_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = lean_box(v___x_96_);
v___x_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; 
v___x_99_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg(v_newGoals_92_, v_newGoals_93_, v___x_94_, v___x_94_, v_a_87_, v_a_88_, v_a_89_, v_a_90_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate___boxed(lean_object* v_a_100_, lean_object* v_b_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate(v_a_100_, v_b_101_, v_a_102_, v_a_103_, v_a_104_, v_a_105_);
lean_dec(v_a_105_);
lean_dec_ref(v_a_104_);
lean_dec(v_a_103_);
lean_dec_ref(v_a_102_);
lean_dec_ref(v_b_101_);
lean_dec_ref(v_a_100_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0(lean_object* v___x_108_, lean_object* v___x_109_, lean_object* v_n_110_, lean_object* v_i_111_, lean_object* v_a_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___redArg(v___x_108_, v___x_109_, v_n_110_, v_i_111_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0___boxed(lean_object* v___x_119_, lean_object* v___x_120_, lean_object* v_n_121_, lean_object* v_i_122_, lean_object* v_a_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyKey_isDuplicate_spec__0(v___x_119_, v___x_120_, v_n_121_, v_i_122_, v_a_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
lean_dec(v_n_121_);
lean_dec_ref(v___x_120_);
lean_dec_ref(v___x_119_);
return v_res_129_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0(lean_object* v_as_130_, size_t v_i_131_, size_t v_stop_132_){
_start:
{
uint8_t v___x_133_; 
v___x_133_ = lean_usize_dec_eq(v_i_131_, v_stop_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; uint8_t v_binderInfo_135_; uint8_t v___x_136_; 
v___x_134_ = lean_array_uget_borrowed(v_as_130_, v_i_131_);
v_binderInfo_135_ = lean_ctor_get_uint8(v___x_134_, sizeof(void*)*1);
v___x_136_ = l_Lean_BinderInfo_isExplicit(v_binderInfo_135_);
if (v___x_136_ == 0)
{
size_t v___x_137_; size_t v___x_138_; 
v___x_137_ = ((size_t)1ULL);
v___x_138_ = lean_usize_add(v_i_131_, v___x_137_);
v_i_131_ = v___x_138_;
goto _start;
}
else
{
return v___x_136_;
}
}
else
{
uint8_t v___x_140_; 
v___x_140_ = 0;
return v___x_140_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0___boxed(lean_object* v_as_141_, lean_object* v_i_142_, lean_object* v_stop_143_){
_start:
{
size_t v_i_boxed_144_; size_t v_stop_boxed_145_; uint8_t v_res_146_; lean_object* v_r_147_; 
v_i_boxed_144_ = lean_unbox_usize(v_i_142_);
lean_dec(v_i_142_);
v_stop_boxed_145_ = lean_unbox_usize(v_stop_143_);
lean_dec(v_stop_143_);
v_res_146_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0(v_as_141_, v_i_boxed_144_, v_stop_boxed_145_);
lean_dec_ref(v_as_141_);
v_r_147_ = lean_box(v_res_146_);
return v_r_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs(lean_object* v_e_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = l_Lean_Expr_getAppFn(v_e_148_);
v___x_155_ = l_Lean_Expr_getAppNumArgs(v_e_148_);
v___x_156_ = l_Lean_Meta_getFunInfoNArgs(v___x_154_, v___x_155_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
if (lean_obj_tag(v___x_156_) == 0)
{
lean_object* v_a_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_177_; 
v_a_157_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_177_ == 0)
{
v___x_159_ = v___x_156_;
v_isShared_160_ = v_isSharedCheck_177_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_a_157_);
lean_dec(v___x_156_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_177_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v_paramInfo_167_; lean_object* v___x_168_; lean_object* v___x_169_; uint8_t v___x_170_; 
v_paramInfo_167_ = lean_ctor_get(v_a_157_, 0);
lean_inc_ref(v_paramInfo_167_);
lean_dec(v_a_157_);
v___x_168_ = lean_unsigned_to_nat(0u);
v___x_169_ = lean_array_get_size(v_paramInfo_167_);
v___x_170_ = lean_nat_dec_lt(v___x_168_, v___x_169_);
if (v___x_170_ == 0)
{
lean_dec_ref(v_paramInfo_167_);
goto v___jp_161_;
}
else
{
if (v___x_170_ == 0)
{
lean_dec_ref(v_paramInfo_167_);
goto v___jp_161_;
}
else
{
size_t v___x_171_; size_t v___x_172_; uint8_t v___x_173_; 
v___x_171_ = ((size_t)0ULL);
v___x_172_ = lean_usize_of_nat(v___x_169_);
v___x_173_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs_spec__0(v_paramInfo_167_, v___x_171_, v___x_172_);
lean_dec_ref(v_paramInfo_167_);
if (v___x_173_ == 0)
{
goto v___jp_161_;
}
else
{
uint8_t v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
lean_del_object(v___x_159_);
v___x_174_ = 0;
v___x_175_ = lean_box(v___x_174_);
v___x_176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
return v___x_176_;
}
}
}
v___jp_161_:
{
uint8_t v___x_162_; lean_object* v___x_163_; lean_object* v___x_165_; 
v___x_162_ = 1;
v___x_163_ = lean_box(v___x_162_);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 0, v___x_163_);
v___x_165_ = v___x_159_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
else
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_185_; 
v_a_178_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_185_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_185_ == 0)
{
v___x_180_ = v___x_156_;
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_156_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_183_; 
if (v_isShared_181_ == 0)
{
v___x_183_ = v___x_180_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v_a_178_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs___boxed(lean_object* v_e_186_, lean_object* v_a_187_, lean_object* v_a_188_, lean_object* v_a_189_, lean_object* v_a_190_, lean_object* v_a_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs(v_e_186_, v_a_187_, v_a_188_, v_a_189_, v_a_190_);
lean_dec(v_a_190_);
lean_dec_ref(v_a_189_);
lean_dec(v_a_188_);
lean_dec_ref(v_a_187_);
lean_dec_ref(v_e_186_);
return v_res_192_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(lean_object* v_opts_193_, lean_object* v_opt_194_){
_start:
{
lean_object* v_name_195_; lean_object* v_defValue_196_; lean_object* v_map_197_; lean_object* v___x_198_; 
v_name_195_ = lean_ctor_get(v_opt_194_, 0);
v_defValue_196_ = lean_ctor_get(v_opt_194_, 1);
v_map_197_ = lean_ctor_get(v_opts_193_, 0);
v___x_198_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_197_, v_name_195_);
if (lean_obj_tag(v___x_198_) == 0)
{
uint8_t v___x_199_; 
v___x_199_ = lean_unbox(v_defValue_196_);
return v___x_199_;
}
else
{
lean_object* v_val_200_; 
v_val_200_ = lean_ctor_get(v___x_198_, 0);
lean_inc(v_val_200_);
lean_dec_ref_known(v___x_198_, 1);
if (lean_obj_tag(v_val_200_) == 1)
{
uint8_t v_v_201_; 
v_v_201_ = lean_ctor_get_uint8(v_val_200_, 0);
lean_dec_ref_known(v_val_200_, 0);
return v_v_201_;
}
else
{
uint8_t v___x_202_; 
lean_dec(v_val_200_);
v___x_202_ = lean_unbox(v_defValue_196_);
return v___x_202_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1___boxed(lean_object* v_opts_203_, lean_object* v_opt_204_){
_start:
{
uint8_t v_res_205_; lean_object* v_r_206_; 
v_res_205_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v_opts_203_, v_opt_204_);
lean_dec_ref(v_opt_204_);
lean_dec_ref(v_opts_203_);
v_r_206_ = lean_box(v_res_205_);
return v_r_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(lean_object* v_opts_207_, lean_object* v_opt_208_){
_start:
{
lean_object* v_name_209_; lean_object* v_defValue_210_; lean_object* v_map_211_; lean_object* v___x_212_; 
v_name_209_ = lean_ctor_get(v_opt_208_, 0);
v_defValue_210_ = lean_ctor_get(v_opt_208_, 1);
v_map_211_ = lean_ctor_get(v_opts_207_, 0);
v___x_212_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_211_, v_name_209_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_inc(v_defValue_210_);
return v_defValue_210_;
}
else
{
lean_object* v_val_213_; 
v_val_213_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_val_213_);
lean_dec_ref_known(v___x_212_, 1);
if (lean_obj_tag(v_val_213_) == 3)
{
lean_object* v_v_214_; 
v_v_214_ = lean_ctor_get(v_val_213_, 0);
lean_inc(v_v_214_);
lean_dec_ref_known(v_val_213_, 1);
return v_v_214_;
}
else
{
lean_dec(v_val_213_);
lean_inc(v_defValue_210_);
return v_defValue_210_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2___boxed(lean_object* v_opts_215_, lean_object* v_opt_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v_opts_215_, v_opt_216_);
lean_dec_ref(v_opt_216_);
lean_dec_ref(v_opts_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(lean_object* v_o_221_, lean_object* v_k_222_, uint8_t v_v_223_){
_start:
{
lean_object* v_map_224_; uint8_t v_hasTrace_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_239_; 
v_map_224_ = lean_ctor_get(v_o_221_, 0);
v_hasTrace_225_ = lean_ctor_get_uint8(v_o_221_, sizeof(void*)*1);
v_isSharedCheck_239_ = !lean_is_exclusive(v_o_221_);
if (v_isSharedCheck_239_ == 0)
{
v___x_227_ = v_o_221_;
v_isShared_228_ = v_isSharedCheck_239_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_map_224_);
lean_dec(v_o_221_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_239_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_229_, 0, v_v_223_);
lean_inc(v_k_222_);
v___x_230_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_222_, v___x_229_, v_map_224_);
if (v_hasTrace_225_ == 0)
{
lean_object* v___x_231_; uint8_t v___x_232_; lean_object* v___x_234_; 
v___x_231_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___closed__1));
v___x_232_ = l_Lean_Name_isPrefixOf(v___x_231_, v_k_222_);
lean_dec(v_k_222_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 0, v___x_230_);
v___x_234_ = v___x_227_;
goto v_reusejp_233_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_230_);
v___x_234_ = v_reuseFailAlloc_235_;
goto v_reusejp_233_;
}
v_reusejp_233_:
{
lean_ctor_set_uint8(v___x_234_, sizeof(void*)*1, v___x_232_);
return v___x_234_;
}
}
else
{
lean_object* v___x_237_; 
lean_dec(v_k_222_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 0, v___x_230_);
v___x_237_ = v___x_227_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v___x_230_);
lean_ctor_set_uint8(v_reuseFailAlloc_238_, sizeof(void*)*1, v_hasTrace_225_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0___boxed(lean_object* v_o_240_, lean_object* v_k_241_, lean_object* v_v_242_){
_start:
{
uint8_t v_v_boxed_243_; lean_object* v_res_244_; 
v_v_boxed_243_ = lean_unbox(v_v_242_);
v_res_244_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_o_240_, v_k_241_, v_v_boxed_243_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(lean_object* v_opts_245_, lean_object* v_opt_246_, uint8_t v_val_247_){
_start:
{
lean_object* v_name_248_; lean_object* v___x_249_; 
v_name_248_ = lean_ctor_get(v_opt_246_, 0);
lean_inc(v_name_248_);
lean_dec_ref(v_opt_246_);
v___x_249_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0_spec__0(v_opts_245_, v_name_248_, v_val_247_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0___boxed(lean_object* v_opts_250_, lean_object* v_opt_251_, lean_object* v_val_252_){
_start:
{
uint8_t v_val_boxed_253_; lean_object* v_res_254_; 
v_val_boxed_253_ = lean_unbox(v_val_252_);
v_res_254_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_opts_250_, v_opt_251_, v_val_boxed_253_);
return v_res_254_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7(void){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_271_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__7);
v___x_272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
return v___x_272_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9(void){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__8);
v___x_274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_274_, 0, v___x_273_);
lean_ctor_set(v___x_274_, 1, v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object* v_lemmaName_281_, lean_object* v_proof_282_, uint8_t v_isClosing_283_, uint8_t v_justLemmaName_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
if (v_justLemmaName_284_ == 0)
{
lean_object* v___x_290_; lean_object* v_fileName_291_; lean_object* v_fileMap_292_; lean_object* v_options_293_; lean_object* v_currRecDepth_294_; lean_object* v_ref_295_; lean_object* v_currNamespace_296_; lean_object* v_openDecls_297_; lean_object* v_initHeartbeats_298_; lean_object* v_maxHeartbeats_299_; lean_object* v_quotContext_300_; lean_object* v_currMacroScope_301_; lean_object* v_cancelTk_x3f_302_; uint8_t v_suppressElabErrors_303_; lean_object* v_inheritedTraceOptions_304_; lean_object* v_env_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; uint8_t v___x_310_; lean_object* v_fileName_312_; lean_object* v_fileMap_313_; lean_object* v_currRecDepth_314_; lean_object* v_ref_315_; lean_object* v_currNamespace_316_; lean_object* v_openDecls_317_; lean_object* v_initHeartbeats_318_; lean_object* v_maxHeartbeats_319_; lean_object* v_quotContext_320_; lean_object* v_currMacroScope_321_; lean_object* v_cancelTk_x3f_322_; uint8_t v_suppressElabErrors_323_; lean_object* v_inheritedTraceOptions_324_; lean_object* v___y_325_; uint8_t v___y_357_; uint8_t v___x_378_; 
lean_dec_ref(v_lemmaName_281_);
v___x_290_ = lean_st_ref_get(v_a_288_);
v_fileName_291_ = lean_ctor_get(v_a_287_, 0);
v_fileMap_292_ = lean_ctor_get(v_a_287_, 1);
v_options_293_ = lean_ctor_get(v_a_287_, 2);
v_currRecDepth_294_ = lean_ctor_get(v_a_287_, 3);
v_ref_295_ = lean_ctor_get(v_a_287_, 5);
v_currNamespace_296_ = lean_ctor_get(v_a_287_, 6);
v_openDecls_297_ = lean_ctor_get(v_a_287_, 7);
v_initHeartbeats_298_ = lean_ctor_get(v_a_287_, 8);
v_maxHeartbeats_299_ = lean_ctor_get(v_a_287_, 9);
v_quotContext_300_ = lean_ctor_get(v_a_287_, 10);
v_currMacroScope_301_ = lean_ctor_get(v_a_287_, 11);
v_cancelTk_x3f_302_ = lean_ctor_get(v_a_287_, 12);
v_suppressElabErrors_303_ = lean_ctor_get_uint8(v_a_287_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_304_ = lean_ctor_get(v_a_287_, 13);
v_env_305_ = lean_ctor_get(v___x_290_, 0);
lean_inc_ref(v_env_305_);
lean_dec(v___x_290_);
v___x_306_ = lean_box(1);
v___x_307_ = l_Lean_pp_mvars;
lean_inc_ref(v_options_293_);
v___x_308_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__0(v_options_293_, v___x_307_, v_justLemmaName_284_);
v___x_309_ = l_Lean_diagnostics;
v___x_310_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__1(v___x_308_, v___x_309_);
v___x_378_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_305_);
lean_dec_ref(v_env_305_);
if (v___x_378_ == 0)
{
if (v___x_310_ == 0)
{
v_fileName_312_ = v_fileName_291_;
v_fileMap_313_ = v_fileMap_292_;
v_currRecDepth_314_ = v_currRecDepth_294_;
v_ref_315_ = v_ref_295_;
v_currNamespace_316_ = v_currNamespace_296_;
v_openDecls_317_ = v_openDecls_297_;
v_initHeartbeats_318_ = v_initHeartbeats_298_;
v_maxHeartbeats_319_ = v_maxHeartbeats_299_;
v_quotContext_320_ = v_quotContext_300_;
v_currMacroScope_321_ = v_currMacroScope_301_;
v_cancelTk_x3f_322_ = v_cancelTk_x3f_302_;
v_suppressElabErrors_323_ = v_suppressElabErrors_303_;
v_inheritedTraceOptions_324_ = v_inheritedTraceOptions_304_;
v___y_325_ = v_a_288_;
goto v___jp_311_;
}
else
{
v___y_357_ = v___x_378_;
goto v___jp_356_;
}
}
else
{
v___y_357_ = v___x_310_;
goto v___jp_356_;
}
v___jp_311_:
{
lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_326_ = l_Lean_maxRecDepth;
v___x_327_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_spec__2(v___x_308_, v___x_326_);
lean_inc_ref(v_inheritedTraceOptions_324_);
lean_inc(v_cancelTk_x3f_322_);
lean_inc(v_currMacroScope_321_);
lean_inc(v_quotContext_320_);
lean_inc(v_maxHeartbeats_319_);
lean_inc(v_initHeartbeats_318_);
lean_inc(v_openDecls_317_);
lean_inc(v_currNamespace_316_);
lean_inc(v_ref_315_);
lean_inc(v_currRecDepth_314_);
lean_inc_ref(v_fileMap_313_);
lean_inc_ref(v_fileName_312_);
v___x_328_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_328_, 0, v_fileName_312_);
lean_ctor_set(v___x_328_, 1, v_fileMap_313_);
lean_ctor_set(v___x_328_, 2, v___x_308_);
lean_ctor_set(v___x_328_, 3, v_currRecDepth_314_);
lean_ctor_set(v___x_328_, 4, v___x_327_);
lean_ctor_set(v___x_328_, 5, v_ref_315_);
lean_ctor_set(v___x_328_, 6, v_currNamespace_316_);
lean_ctor_set(v___x_328_, 7, v_openDecls_317_);
lean_ctor_set(v___x_328_, 8, v_initHeartbeats_318_);
lean_ctor_set(v___x_328_, 9, v_maxHeartbeats_319_);
lean_ctor_set(v___x_328_, 10, v_quotContext_320_);
lean_ctor_set(v___x_328_, 11, v_currMacroScope_321_);
lean_ctor_set(v___x_328_, 12, v_cancelTk_x3f_322_);
lean_ctor_set(v___x_328_, 13, v_inheritedTraceOptions_324_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*14, v___x_310_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*14 + 1, v_suppressElabErrors_323_);
v___x_329_ = l_Lean_PrettyPrinter_delab(v_proof_282_, v___x_306_, v_a_285_, v_a_286_, v___x_328_, v___y_325_);
lean_dec_ref_known(v___x_328_, 14);
if (lean_obj_tag(v___x_329_) == 0)
{
if (v_isClosing_283_ == 0)
{
lean_object* v_a_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_342_; 
v_a_330_ = lean_ctor_get(v___x_329_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_329_);
if (v_isSharedCheck_342_ == 0)
{
v___x_332_ = v___x_329_;
v_isShared_333_ = v_isSharedCheck_342_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_a_330_);
lean_dec(v___x_329_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_342_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_340_; 
v___x_334_ = l_Lean_SourceInfo_fromRef(v_ref_295_, v_isClosing_283_);
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3));
v___x_336_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4));
lean_inc(v___x_334_);
v___x_337_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_334_);
lean_ctor_set(v___x_337_, 1, v___x_335_);
v___x_338_ = l_Lean_Syntax_node2(v___x_334_, v___x_336_, v___x_337_, v_a_330_);
if (v_isShared_333_ == 0)
{
lean_ctor_set(v___x_332_, 0, v___x_338_);
v___x_340_ = v___x_332_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v___x_338_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_355_; 
v_a_343_ = lean_ctor_get(v___x_329_, 0);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_329_);
if (v_isSharedCheck_355_ == 0)
{
v___x_345_ = v___x_329_;
v_isShared_346_ = v_isSharedCheck_355_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_329_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_355_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_353_; 
v___x_347_ = l_Lean_SourceInfo_fromRef(v_ref_295_, v_justLemmaName_284_);
v___x_348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5));
v___x_349_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6));
lean_inc(v___x_347_);
v___x_350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_347_);
lean_ctor_set(v___x_350_, 1, v___x_348_);
v___x_351_ = l_Lean_Syntax_node2(v___x_347_, v___x_349_, v___x_350_, v_a_343_);
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 0, v___x_351_);
v___x_353_ = v___x_345_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_351_);
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
return v___x_329_;
}
}
v___jp_356_:
{
if (v___y_357_ == 0)
{
lean_object* v___x_358_; lean_object* v_env_359_; lean_object* v_nextMacroScope_360_; lean_object* v_ngen_361_; lean_object* v_auxDeclNGen_362_; lean_object* v_traceState_363_; lean_object* v_messages_364_; lean_object* v_infoState_365_; lean_object* v_snapshotTasks_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_376_; 
v___x_358_ = lean_st_ref_take(v_a_288_);
v_env_359_ = lean_ctor_get(v___x_358_, 0);
v_nextMacroScope_360_ = lean_ctor_get(v___x_358_, 1);
v_ngen_361_ = lean_ctor_get(v___x_358_, 2);
v_auxDeclNGen_362_ = lean_ctor_get(v___x_358_, 3);
v_traceState_363_ = lean_ctor_get(v___x_358_, 4);
v_messages_364_ = lean_ctor_get(v___x_358_, 6);
v_infoState_365_ = lean_ctor_get(v___x_358_, 7);
v_snapshotTasks_366_ = lean_ctor_get(v___x_358_, 8);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_358_);
if (v_isSharedCheck_376_ == 0)
{
lean_object* v_unused_377_; 
v_unused_377_ = lean_ctor_get(v___x_358_, 5);
lean_dec(v_unused_377_);
v___x_368_ = v___x_358_;
v_isShared_369_ = v_isSharedCheck_376_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_snapshotTasks_366_);
lean_inc(v_infoState_365_);
lean_inc(v_messages_364_);
lean_inc(v_traceState_363_);
lean_inc(v_auxDeclNGen_362_);
lean_inc(v_ngen_361_);
lean_inc(v_nextMacroScope_360_);
lean_inc(v_env_359_);
lean_dec(v___x_358_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_376_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_373_; 
v___x_370_ = l_Lean_Kernel_enableDiag(v_env_359_, v___x_310_);
v___x_371_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__9);
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 5, v___x_371_);
lean_ctor_set(v___x_368_, 0, v___x_370_);
v___x_373_ = v___x_368_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_370_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v_nextMacroScope_360_);
lean_ctor_set(v_reuseFailAlloc_375_, 2, v_ngen_361_);
lean_ctor_set(v_reuseFailAlloc_375_, 3, v_auxDeclNGen_362_);
lean_ctor_set(v_reuseFailAlloc_375_, 4, v_traceState_363_);
lean_ctor_set(v_reuseFailAlloc_375_, 5, v___x_371_);
lean_ctor_set(v_reuseFailAlloc_375_, 6, v_messages_364_);
lean_ctor_set(v_reuseFailAlloc_375_, 7, v_infoState_365_);
lean_ctor_set(v_reuseFailAlloc_375_, 8, v_snapshotTasks_366_);
v___x_373_ = v_reuseFailAlloc_375_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
lean_object* v___x_374_; 
v___x_374_ = lean_st_ref_set(v_a_288_, v___x_373_);
v_fileName_312_ = v_fileName_291_;
v_fileMap_313_ = v_fileMap_292_;
v_currRecDepth_314_ = v_currRecDepth_294_;
v_ref_315_ = v_ref_295_;
v_currNamespace_316_ = v_currNamespace_296_;
v_openDecls_317_ = v_openDecls_297_;
v_initHeartbeats_318_ = v_initHeartbeats_298_;
v_maxHeartbeats_319_ = v_maxHeartbeats_299_;
v_quotContext_320_ = v_quotContext_300_;
v_currMacroScope_321_ = v_currMacroScope_301_;
v_cancelTk_x3f_322_ = v_cancelTk_x3f_302_;
v_suppressElabErrors_323_ = v_suppressElabErrors_303_;
v_inheritedTraceOptions_324_ = v_inheritedTraceOptions_304_;
v___y_325_ = v_a_288_;
goto v___jp_311_;
}
}
}
else
{
v_fileName_312_ = v_fileName_291_;
v_fileMap_313_ = v_fileMap_292_;
v_currRecDepth_314_ = v_currRecDepth_294_;
v_ref_315_ = v_ref_295_;
v_currNamespace_316_ = v_currNamespace_296_;
v_openDecls_317_ = v_openDecls_297_;
v_initHeartbeats_318_ = v_initHeartbeats_298_;
v_maxHeartbeats_319_ = v_maxHeartbeats_299_;
v_quotContext_320_ = v_quotContext_300_;
v_currMacroScope_321_ = v_currMacroScope_301_;
v_cancelTk_x3f_322_ = v_cancelTk_x3f_302_;
v_suppressElabErrors_323_ = v_suppressElabErrors_303_;
v_inheritedTraceOptions_324_ = v_inheritedTraceOptions_304_;
v___y_325_ = v_a_288_;
goto v___jp_311_;
}
}
}
else
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(v_lemmaName_281_, v_a_285_, v_a_286_, v_a_287_, v_a_288_);
if (lean_obj_tag(v___x_379_) == 0)
{
lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_422_; 
v_a_380_ = lean_ctor_get(v___x_379_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_422_ == 0)
{
v___x_382_ = v___x_379_;
v_isShared_383_ = v_isSharedCheck_422_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_379_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_422_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_384_; uint8_t v_a_386_; 
v___x_384_ = l_Lean_mkIdent(v_a_380_);
if (v_isClosing_283_ == 0)
{
lean_dec_ref(v_proof_282_);
v_a_386_ = v_isClosing_283_;
goto v___jp_385_;
}
else
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax_hasOnlyImplicitArgs(v_proof_282_, v_a_285_, v_a_286_, v_a_287_, v_a_288_);
lean_dec_ref(v_proof_282_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_413_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_413_ == 0)
{
v___x_399_ = v___x_396_;
v_isShared_400_ = v_isSharedCheck_413_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_396_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_413_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
uint8_t v___x_401_; 
v___x_401_ = lean_unbox(v_a_397_);
if (v___x_401_ == 0)
{
uint8_t v___x_402_; 
lean_del_object(v___x_399_);
v___x_402_ = lean_unbox(v_a_397_);
lean_dec(v_a_397_);
v_a_386_ = v___x_402_;
goto v___jp_385_;
}
else
{
lean_object* v_ref_403_; uint8_t v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_411_; 
lean_dec(v_a_397_);
lean_del_object(v___x_382_);
v_ref_403_ = lean_ctor_get(v_a_287_, 5);
v___x_404_ = 0;
v___x_405_ = l_Lean_SourceInfo_fromRef(v_ref_403_, v___x_404_);
v___x_406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5));
v___x_407_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__6));
lean_inc(v___x_405_);
v___x_408_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_405_);
lean_ctor_set(v___x_408_, 1, v___x_406_);
v___x_409_ = l_Lean_Syntax_node2(v___x_405_, v___x_407_, v___x_408_, v___x_384_);
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 0, v___x_409_);
v___x_411_ = v___x_399_;
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
else
{
lean_object* v_a_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_421_; 
lean_dec(v___x_384_);
lean_del_object(v___x_382_);
v_a_414_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_421_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_421_ == 0)
{
v___x_416_ = v___x_396_;
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_a_414_);
lean_dec(v___x_396_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_419_; 
if (v_isShared_417_ == 0)
{
v___x_419_ = v___x_416_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v_a_414_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
}
v___jp_385_:
{
lean_object* v_ref_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_394_; 
v_ref_387_ = lean_ctor_get(v_a_287_, 5);
v___x_388_ = l_Lean_SourceInfo_fromRef(v_ref_387_, v_a_386_);
v___x_389_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__10));
v___x_390_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__11));
lean_inc(v___x_388_);
v___x_391_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_388_);
lean_ctor_set(v___x_391_, 1, v___x_389_);
v___x_392_ = l_Lean_Syntax_node2(v___x_388_, v___x_390_, v___x_391_, v___x_384_);
if (v_isShared_383_ == 0)
{
lean_ctor_set(v___x_382_, 0, v___x_392_);
v___x_394_ = v___x_382_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_392_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
}
else
{
lean_object* v_a_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_430_; 
lean_dec_ref(v_proof_282_);
v_a_423_ = lean_ctor_get(v___x_379_, 0);
v_isSharedCheck_430_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_430_ == 0)
{
v___x_425_ = v___x_379_;
v_isShared_426_ = v_isSharedCheck_430_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_a_423_);
lean_dec(v___x_379_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_430_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_428_; 
if (v_isShared_426_ == 0)
{
v___x_428_ = v___x_425_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v_a_423_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object* v_lemmaName_431_, lean_object* v_proof_432_, lean_object* v_isClosing_433_, lean_object* v_justLemmaName_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
uint8_t v_isClosing_boxed_440_; uint8_t v_justLemmaName_boxed_441_; lean_object* v_res_442_; 
v_isClosing_boxed_440_ = lean_unbox(v_isClosing_433_);
v_justLemmaName_boxed_441_ = lean_unbox(v_justLemmaName_434_);
v_res_442_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lemmaName_431_, v_proof_432_, v_isClosing_boxed_440_, v_justLemmaName_boxed_441_, v_a_435_, v_a_436_, v_a_437_, v_a_438_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(lean_object* v_e_443_, lean_object* v___y_444_){
_start:
{
uint8_t v___x_446_; 
v___x_446_ = l_Lean_Expr_hasMVar(v_e_443_);
if (v___x_446_ == 0)
{
lean_object* v___x_447_; 
v___x_447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_447_, 0, v_e_443_);
return v___x_447_;
}
else
{
lean_object* v___x_448_; lean_object* v_mctx_449_; lean_object* v___x_450_; lean_object* v_fst_451_; lean_object* v_snd_452_; lean_object* v___x_453_; lean_object* v_cache_454_; lean_object* v_zetaDeltaFVarIds_455_; lean_object* v_postponed_456_; lean_object* v_diag_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_466_; 
v___x_448_ = lean_st_ref_get(v___y_444_);
v_mctx_449_ = lean_ctor_get(v___x_448_, 0);
lean_inc_ref(v_mctx_449_);
lean_dec(v___x_448_);
v___x_450_ = l_Lean_instantiateMVarsCore(v_mctx_449_, v_e_443_);
v_fst_451_ = lean_ctor_get(v___x_450_, 0);
lean_inc(v_fst_451_);
v_snd_452_ = lean_ctor_get(v___x_450_, 1);
lean_inc(v_snd_452_);
lean_dec_ref(v___x_450_);
v___x_453_ = lean_st_ref_take(v___y_444_);
v_cache_454_ = lean_ctor_get(v___x_453_, 1);
v_zetaDeltaFVarIds_455_ = lean_ctor_get(v___x_453_, 2);
v_postponed_456_ = lean_ctor_get(v___x_453_, 3);
v_diag_457_ = lean_ctor_get(v___x_453_, 4);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_466_ == 0)
{
lean_object* v_unused_467_; 
v_unused_467_ = lean_ctor_get(v___x_453_, 0);
lean_dec(v_unused_467_);
v___x_459_ = v___x_453_;
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_diag_457_);
lean_inc(v_postponed_456_);
lean_inc(v_zetaDeltaFVarIds_455_);
lean_inc(v_cache_454_);
lean_dec(v___x_453_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 0, v_snd_452_);
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_snd_452_);
lean_ctor_set(v_reuseFailAlloc_465_, 1, v_cache_454_);
lean_ctor_set(v_reuseFailAlloc_465_, 2, v_zetaDeltaFVarIds_455_);
lean_ctor_set(v_reuseFailAlloc_465_, 3, v_postponed_456_);
lean_ctor_set(v_reuseFailAlloc_465_, 4, v_diag_457_);
v___x_462_ = v_reuseFailAlloc_465_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_463_ = lean_st_ref_set(v___y_444_, v___x_462_);
v___x_464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_464_, 0, v_fst_451_);
return v___x_464_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg___boxed(lean_object* v_e_468_, lean_object* v___y_469_, lean_object* v___y_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(v_e_468_, v___y_469_);
lean_dec(v___y_469_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1(lean_object* v_e_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(v_e_472_, v___y_476_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___boxed(lean_object* v_e_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1(v_e_481_, v___y_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec(v___y_485_);
lean_dec_ref(v___y_484_);
lean_dec(v___y_483_);
lean_dec_ref(v___y_482_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0(lean_object* v_k_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v___x_498_; 
lean_inc(v___y_492_);
lean_inc_ref(v___y_491_);
v___x_498_ = lean_apply_7(v_k_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, lean_box(0));
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0___boxed(lean_object* v_k_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0(v_k_499_, v___y_500_, v___y_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg(lean_object* v_k_508_, uint8_t v_allowLevelAssignments_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v___f_517_; lean_object* v___x_518_; 
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_517_, 0, v_k_508_);
lean_closure_set(v___f_517_, 1, v___y_510_);
lean_closure_set(v___f_517_, 2, v___y_511_);
v___x_518_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_509_, v___f_517_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
if (lean_obj_tag(v___x_518_) == 0)
{
return v___x_518_;
}
else
{
lean_object* v_a_519_; lean_object* v___x_521_; uint8_t v_isShared_522_; uint8_t v_isSharedCheck_526_; 
v_a_519_ = lean_ctor_get(v___x_518_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v___x_518_);
if (v_isSharedCheck_526_ == 0)
{
v___x_521_ = v___x_518_;
v_isShared_522_ = v_isSharedCheck_526_;
goto v_resetjp_520_;
}
else
{
lean_inc(v_a_519_);
lean_dec(v___x_518_);
v___x_521_ = lean_box(0);
v_isShared_522_ = v_isSharedCheck_526_;
goto v_resetjp_520_;
}
v_resetjp_520_:
{
lean_object* v___x_524_; 
if (v_isShared_522_ == 0)
{
v___x_524_ = v___x_521_;
goto v_reusejp_523_;
}
else
{
lean_object* v_reuseFailAlloc_525_; 
v_reuseFailAlloc_525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_525_, 0, v_a_519_);
v___x_524_ = v_reuseFailAlloc_525_;
goto v_reusejp_523_;
}
v_reusejp_523_:
{
return v___x_524_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg___boxed(lean_object* v_k_527_, lean_object* v_allowLevelAssignments_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_536_; lean_object* v_res_537_; 
v_allowLevelAssignments_boxed_536_ = lean_unbox(v_allowLevelAssignments_528_);
v_res_537_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg(v_k_527_, v_allowLevelAssignments_boxed_536_, v___y_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_);
lean_dec(v___y_534_);
lean_dec_ref(v___y_533_);
lean_dec(v___y_532_);
lean_dec_ref(v___y_531_);
lean_dec(v___y_530_);
lean_dec_ref(v___y_529_);
return v_res_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2(lean_object* v_00_u03b1_538_, lean_object* v_k_539_, uint8_t v_allowLevelAssignments_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg(v_k_539_, v_allowLevelAssignments_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___boxed(lean_object* v_00_u03b1_549_, lean_object* v_k_550_, lean_object* v_allowLevelAssignments_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_559_; lean_object* v_res_560_; 
v_allowLevelAssignments_boxed_559_ = lean_unbox(v_allowLevelAssignments_551_);
v_res_560_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2(v_00_u03b1_549_, v_k_550_, v_allowLevelAssignments_boxed_559_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec(v___y_557_);
lean_dec_ref(v___y_556_);
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
return v_res_560_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12(void){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_587_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__11));
v___x_588_ = lean_unsigned_to_nat(2u);
v___x_589_ = lean_mk_empty_array_with_capacity(v___x_588_);
v___x_590_ = lean_array_push(v___x_589_, v___x_587_);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg(lean_object* v_as_591_, size_t v_sz_592_, size_t v_i_593_, lean_object* v_b_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_){
_start:
{
uint8_t v___x_600_; 
v___x_600_ = lean_usize_dec_lt(v_i_593_, v_sz_592_);
if (v___x_600_ == 0)
{
lean_object* v___x_601_; 
v___x_601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_601_, 0, v_b_594_);
return v___x_601_;
}
else
{
lean_object* v_a_602_; lean_object* v___x_603_; 
v_a_602_ = lean_array_uget_borrowed(v_as_591_, v_i_593_);
lean_inc(v_a_602_);
v___x_603_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_a_602_, v___y_595_, v___y_596_, v___y_597_, v___y_598_);
if (lean_obj_tag(v___x_603_) == 0)
{
lean_object* v_a_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; size_t v___x_611_; size_t v___x_612_; 
v_a_604_ = lean_ctor_get(v___x_603_, 0);
lean_inc(v_a_604_);
lean_dec_ref_known(v___x_603_, 1);
v___x_605_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0));
v___x_606_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__1));
v___x_607_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__12);
v___x_608_ = lean_array_push(v___x_607_, v_a_604_);
v___x_609_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_609_, 0, v___x_605_);
lean_ctor_set(v___x_609_, 1, v___x_606_);
lean_ctor_set(v___x_609_, 2, v___x_608_);
v___x_610_ = lean_array_push(v_b_594_, v___x_609_);
v___x_611_ = ((size_t)1ULL);
v___x_612_ = lean_usize_add(v_i_593_, v___x_611_);
v_i_593_ = v___x_612_;
v_b_594_ = v___x_610_;
goto _start;
}
else
{
lean_object* v_a_614_; lean_object* v___x_616_; uint8_t v_isShared_617_; uint8_t v_isSharedCheck_621_; 
lean_dec_ref(v_b_594_);
v_a_614_ = lean_ctor_get(v___x_603_, 0);
v_isSharedCheck_621_ = !lean_is_exclusive(v___x_603_);
if (v_isSharedCheck_621_ == 0)
{
v___x_616_ = v___x_603_;
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
else
{
lean_inc(v_a_614_);
lean_dec(v___x_603_);
v___x_616_ = lean_box(0);
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
v_resetjp_615_:
{
lean_object* v___x_619_; 
if (v_isShared_617_ == 0)
{
v___x_619_ = v___x_616_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v_a_614_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___boxed(lean_object* v_as_622_, lean_object* v_sz_623_, lean_object* v_i_624_, lean_object* v_b_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
size_t v_sz_boxed_631_; size_t v_i_boxed_632_; lean_object* v_res_633_; 
v_sz_boxed_631_ = lean_unbox_usize(v_sz_623_);
lean_dec(v_sz_623_);
v_i_boxed_632_ = lean_unbox_usize(v_i_624_);
lean_dec(v_i_624_);
v_res_633_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg(v_as_622_, v_sz_boxed_631_, v_i_boxed_632_, v_b_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec_ref(v_as_622_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg(size_t v_sz_634_, size_t v_i_635_, lean_object* v_bs_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
uint8_t v___x_642_; 
v___x_642_ = lean_usize_dec_lt(v_i_635_, v_sz_634_);
if (v___x_642_ == 0)
{
lean_object* v___x_643_; 
v___x_643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_643_, 0, v_bs_636_);
return v___x_643_;
}
else
{
lean_object* v_v_644_; lean_object* v___x_645_; 
v_v_644_ = lean_array_uget_borrowed(v_bs_636_, v_i_635_);
lean_inc(v_v_644_);
v___x_645_ = l_Lean_Meta_abstractMVars(v_v_644_, v___x_642_, v___y_637_, v___y_638_, v___y_639_, v___y_640_);
if (lean_obj_tag(v___x_645_) == 0)
{
lean_object* v_a_646_; lean_object* v___x_647_; lean_object* v_bs_x27_648_; size_t v___x_649_; size_t v___x_650_; lean_object* v___x_651_; 
v_a_646_ = lean_ctor_get(v___x_645_, 0);
lean_inc(v_a_646_);
lean_dec_ref_known(v___x_645_, 1);
v___x_647_ = lean_unsigned_to_nat(0u);
v_bs_x27_648_ = lean_array_uset(v_bs_636_, v_i_635_, v___x_647_);
v___x_649_ = ((size_t)1ULL);
v___x_650_ = lean_usize_add(v_i_635_, v___x_649_);
v___x_651_ = lean_array_uset(v_bs_x27_648_, v_i_635_, v_a_646_);
v_i_635_ = v___x_650_;
v_bs_636_ = v___x_651_;
goto _start;
}
else
{
lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_660_; 
lean_dec_ref(v_bs_636_);
v_a_653_ = lean_ctor_get(v___x_645_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_645_);
if (v_isSharedCheck_660_ == 0)
{
v___x_655_ = v___x_645_;
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_645_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_658_; 
if (v_isShared_656_ == 0)
{
v___x_658_ = v___x_655_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_a_653_);
v___x_658_ = v_reuseFailAlloc_659_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
return v___x_658_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg___boxed(lean_object* v_sz_661_, lean_object* v_i_662_, lean_object* v_bs_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_){
_start:
{
size_t v_sz_boxed_669_; size_t v_i_boxed_670_; lean_object* v_res_671_; 
v_sz_boxed_669_ = lean_unbox_usize(v_sz_661_);
lean_dec(v_sz_661_);
v_i_boxed_670_ = lean_unbox_usize(v_i_662_);
lean_dec(v_i_662_);
v_res_671_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg(v_sz_boxed_669_, v_i_boxed_670_, v_bs_663_, v___y_664_, v___y_665_, v___y_666_, v___y_667_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg(lean_object* v_msg_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_){
_start:
{
lean_object* v_ref_678_; lean_object* v___x_679_; lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_688_; 
v_ref_678_ = lean_ctor_get(v___y_675_, 5);
v___x_679_ = l_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_applyN_spec__1_spec__1(v_msg_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
v_a_680_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_688_ == 0)
{
v___x_682_ = v___x_679_;
v_isShared_683_ = v_isSharedCheck_688_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_679_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_688_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_684_; lean_object* v___x_686_; 
lean_inc(v_ref_678_);
v___x_684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_684_, 0, v_ref_678_);
lean_ctor_set(v___x_684_, 1, v_a_680_);
if (v_isShared_683_ == 0)
{
lean_ctor_set_tag(v___x_682_, 1);
lean_ctor_set(v___x_682_, 0, v___x_684_);
v___x_686_ = v___x_682_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_684_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg___boxed(lean_object* v_msg_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg(v_msg_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
lean_dec(v___y_691_);
lean_dec_ref(v___y_690_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(lean_object* v_as_696_, size_t v_i_697_, size_t v_stop_698_, lean_object* v_b_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
uint8_t v___x_705_; 
v___x_705_ = lean_usize_dec_eq(v_i_697_, v_stop_698_);
if (v___x_705_ == 0)
{
lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_706_ = lean_array_uget_borrowed(v_as_696_, v_i_697_);
lean_inc(v___x_706_);
v___x_707_ = l_Lean_Meta_ppExpr(v___x_706_, v___y_700_, v___y_701_, v___y_702_, v___y_703_);
if (lean_obj_tag(v___x_707_) == 0)
{
lean_object* v_a_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; size_t v___x_714_; size_t v___x_715_; 
v_a_708_ = lean_ctor_get(v___x_707_, 0);
lean_inc(v_a_708_);
lean_dec_ref_known(v___x_707_, 1);
v___x_709_ = lean_unsigned_to_nat(0u);
v___x_710_ = l_Std_Format_defWidth;
v___x_711_ = l_Std_Format_pretty(v_a_708_, v___x_710_, v___x_709_, v___x_709_);
v___x_712_ = lean_string_length(v___x_711_);
lean_dec_ref(v___x_711_);
v___x_713_ = lean_nat_add(v___x_712_, v_b_699_);
lean_dec(v_b_699_);
v___x_714_ = ((size_t)1ULL);
v___x_715_ = lean_usize_add(v_i_697_, v___x_714_);
v_i_697_ = v___x_715_;
v_b_699_ = v___x_713_;
goto _start;
}
else
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_724_; 
lean_dec(v_b_699_);
v_a_717_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_724_ == 0)
{
v___x_719_ = v___x_707_;
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_707_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_722_; 
if (v_isShared_720_ == 0)
{
v___x_722_ = v___x_719_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v_a_717_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
}
}
else
{
lean_object* v___x_725_; 
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v_b_699_);
return v___x_725_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg___boxed(lean_object* v_as_726_, lean_object* v_i_727_, lean_object* v_stop_728_, lean_object* v_b_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_){
_start:
{
size_t v_i_boxed_735_; size_t v_stop_boxed_736_; lean_object* v_res_737_; 
v_i_boxed_735_ = lean_unbox_usize(v_i_727_);
lean_dec(v_i_727_);
v_stop_boxed_736_ = lean_unbox_usize(v_stop_728_);
lean_dec(v_stop_728_);
v_res_737_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(v_as_726_, v_i_boxed_735_, v_stop_boxed_736_, v_b_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec_ref(v_as_726_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0(lean_object* v_a_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_){
_start:
{
lean_object* v___x_746_; 
v___x_746_ = l_Lean_Meta_findLocalDeclWithType_x3f(v_a_738_, v___y_741_, v___y_742_, v___y_743_, v___y_744_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0___boxed(lean_object* v_a_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0(v_a_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
lean_dec(v___y_749_);
lean_dec_ref(v___y_748_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg(lean_object* v_mvarId_756_, lean_object* v_val_757_, lean_object* v___y_758_){
_start:
{
lean_object* v___x_760_; lean_object* v_mctx_761_; lean_object* v_cache_762_; lean_object* v_zetaDeltaFVarIds_763_; lean_object* v_postponed_764_; lean_object* v_diag_765_; lean_object* v___x_767_; uint8_t v_isShared_768_; uint8_t v_isSharedCheck_793_; 
v___x_760_ = lean_st_ref_take(v___y_758_);
v_mctx_761_ = lean_ctor_get(v___x_760_, 0);
v_cache_762_ = lean_ctor_get(v___x_760_, 1);
v_zetaDeltaFVarIds_763_ = lean_ctor_get(v___x_760_, 2);
v_postponed_764_ = lean_ctor_get(v___x_760_, 3);
v_diag_765_ = lean_ctor_get(v___x_760_, 4);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_793_ == 0)
{
v___x_767_ = v___x_760_;
v_isShared_768_ = v_isSharedCheck_793_;
goto v_resetjp_766_;
}
else
{
lean_inc(v_diag_765_);
lean_inc(v_postponed_764_);
lean_inc(v_zetaDeltaFVarIds_763_);
lean_inc(v_cache_762_);
lean_inc(v_mctx_761_);
lean_dec(v___x_760_);
v___x_767_ = lean_box(0);
v_isShared_768_ = v_isSharedCheck_793_;
goto v_resetjp_766_;
}
v_resetjp_766_:
{
lean_object* v_depth_769_; lean_object* v_levelAssignDepth_770_; lean_object* v_lmvarCounter_771_; lean_object* v_mvarCounter_772_; lean_object* v_lDecls_773_; lean_object* v_decls_774_; lean_object* v_userNames_775_; lean_object* v_lAssignment_776_; lean_object* v_eAssignment_777_; lean_object* v_dAssignment_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_792_; 
v_depth_769_ = lean_ctor_get(v_mctx_761_, 0);
v_levelAssignDepth_770_ = lean_ctor_get(v_mctx_761_, 1);
v_lmvarCounter_771_ = lean_ctor_get(v_mctx_761_, 2);
v_mvarCounter_772_ = lean_ctor_get(v_mctx_761_, 3);
v_lDecls_773_ = lean_ctor_get(v_mctx_761_, 4);
v_decls_774_ = lean_ctor_get(v_mctx_761_, 5);
v_userNames_775_ = lean_ctor_get(v_mctx_761_, 6);
v_lAssignment_776_ = lean_ctor_get(v_mctx_761_, 7);
v_eAssignment_777_ = lean_ctor_get(v_mctx_761_, 8);
v_dAssignment_778_ = lean_ctor_get(v_mctx_761_, 9);
v_isSharedCheck_792_ = !lean_is_exclusive(v_mctx_761_);
if (v_isSharedCheck_792_ == 0)
{
v___x_780_ = v_mctx_761_;
v_isShared_781_ = v_isSharedCheck_792_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_dAssignment_778_);
lean_inc(v_eAssignment_777_);
lean_inc(v_lAssignment_776_);
lean_inc(v_userNames_775_);
lean_inc(v_decls_774_);
lean_inc(v_lDecls_773_);
lean_inc(v_mvarCounter_772_);
lean_inc(v_lmvarCounter_771_);
lean_inc(v_levelAssignDepth_770_);
lean_inc(v_depth_769_);
lean_dec(v_mctx_761_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_792_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_782_; lean_object* v___x_784_; 
v___x_782_ = l_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_apply_spec__1_spec__1___redArg(v_eAssignment_777_, v_mvarId_756_, v_val_757_);
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 8, v___x_782_);
v___x_784_ = v___x_780_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v_depth_769_);
lean_ctor_set(v_reuseFailAlloc_791_, 1, v_levelAssignDepth_770_);
lean_ctor_set(v_reuseFailAlloc_791_, 2, v_lmvarCounter_771_);
lean_ctor_set(v_reuseFailAlloc_791_, 3, v_mvarCounter_772_);
lean_ctor_set(v_reuseFailAlloc_791_, 4, v_lDecls_773_);
lean_ctor_set(v_reuseFailAlloc_791_, 5, v_decls_774_);
lean_ctor_set(v_reuseFailAlloc_791_, 6, v_userNames_775_);
lean_ctor_set(v_reuseFailAlloc_791_, 7, v_lAssignment_776_);
lean_ctor_set(v_reuseFailAlloc_791_, 8, v___x_782_);
lean_ctor_set(v_reuseFailAlloc_791_, 9, v_dAssignment_778_);
v___x_784_ = v_reuseFailAlloc_791_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
lean_object* v___x_786_; 
if (v_isShared_768_ == 0)
{
lean_ctor_set(v___x_767_, 0, v___x_784_);
v___x_786_ = v___x_767_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v___x_784_);
lean_ctor_set(v_reuseFailAlloc_790_, 1, v_cache_762_);
lean_ctor_set(v_reuseFailAlloc_790_, 2, v_zetaDeltaFVarIds_763_);
lean_ctor_set(v_reuseFailAlloc_790_, 3, v_postponed_764_);
lean_ctor_set(v_reuseFailAlloc_790_, 4, v_diag_765_);
v___x_786_ = v_reuseFailAlloc_790_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; 
v___x_787_ = lean_st_ref_set(v___y_758_, v___x_786_);
v___x_788_ = lean_box(0);
v___x_789_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_789_, 0, v___x_788_);
return v___x_789_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg___boxed(lean_object* v_mvarId_794_, lean_object* v_val_795_, lean_object* v___y_796_, lean_object* v___y_797_){
_start:
{
lean_object* v_res_798_; 
v_res_798_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg(v_mvarId_794_, v_val_795_, v___y_796_);
lean_dec(v___y_796_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg(lean_object* v_as_x27_799_, lean_object* v_b_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_){
_start:
{
if (lean_obj_tag(v_as_x27_799_) == 0)
{
lean_object* v___x_808_; 
v___x_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_808_, 0, v_b_800_);
return v___x_808_;
}
else
{
lean_object* v_head_809_; lean_object* v_tail_810_; lean_object* v___x_811_; 
v_head_809_ = lean_ctor_get(v_as_x27_799_, 0);
v_tail_810_ = lean_ctor_get(v_as_x27_799_, 1);
lean_inc(v_head_809_);
v___x_811_ = l_Lean_MVarId_getType(v_head_809_, v___y_803_, v___y_804_, v___y_805_, v___y_806_);
if (lean_obj_tag(v___x_811_) == 0)
{
lean_object* v_a_812_; lean_object* v___x_813_; lean_object* v_a_814_; lean_object* v___x_815_; 
v_a_812_ = lean_ctor_get(v___x_811_, 0);
lean_inc(v_a_812_);
lean_dec_ref_known(v___x_811_, 1);
v___x_813_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(v_a_812_, v___y_804_);
v_a_814_ = lean_ctor_get(v___x_813_, 0);
lean_inc_n(v_a_814_, 2);
lean_dec_ref(v___x_813_);
v___x_815_ = l_Lean_Meta_isProp(v_a_814_, v___y_803_, v___y_804_, v___y_805_, v___y_806_);
if (lean_obj_tag(v___x_815_) == 0)
{
lean_object* v_a_816_; lean_object* v_fst_817_; lean_object* v_snd_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_847_; 
v_a_816_ = lean_ctor_get(v___x_815_, 0);
lean_inc(v_a_816_);
lean_dec_ref_known(v___x_815_, 1);
v_fst_817_ = lean_ctor_get(v_b_800_, 0);
v_snd_818_ = lean_ctor_get(v_b_800_, 1);
v_isSharedCheck_847_ = !lean_is_exclusive(v_b_800_);
if (v_isSharedCheck_847_ == 0)
{
v___x_820_ = v_b_800_;
v_isShared_821_ = v_isSharedCheck_847_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_snd_818_);
lean_inc(v_fst_817_);
lean_dec(v_b_800_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_847_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
uint8_t v___x_828_; 
v___x_828_ = lean_unbox(v_a_816_);
lean_dec(v_a_816_);
if (v___x_828_ == 0)
{
goto v___jp_822_;
}
else
{
uint8_t v___x_829_; lean_object* v___f_830_; lean_object* v___x_831_; 
v___x_829_ = 0;
lean_inc(v_a_814_);
v___f_830_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_830_, 0, v_a_814_);
v___x_831_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__2___redArg(v___f_830_, v___x_829_, v___y_801_, v___y_802_, v___y_803_, v___y_804_, v___y_805_, v___y_806_);
if (lean_obj_tag(v___x_831_) == 0)
{
lean_object* v_a_832_; 
v_a_832_ = lean_ctor_get(v___x_831_, 0);
lean_inc(v_a_832_);
lean_dec_ref_known(v___x_831_, 1);
if (lean_obj_tag(v_a_832_) == 1)
{
lean_object* v_val_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; 
lean_del_object(v___x_820_);
lean_dec(v_snd_818_);
lean_dec(v_a_814_);
v_val_833_ = lean_ctor_get(v_a_832_, 0);
lean_inc(v_val_833_);
lean_dec_ref_known(v_a_832_, 1);
v___x_834_ = l_Lean_Expr_fvar___override(v_val_833_);
lean_inc(v_head_809_);
v___x_835_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg(v_head_809_, v___x_834_, v___y_804_);
lean_dec_ref(v___x_835_);
v___x_836_ = lean_box(v___x_829_);
v___x_837_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_837_, 0, v_fst_817_);
lean_ctor_set(v___x_837_, 1, v___x_836_);
v_as_x27_799_ = v_tail_810_;
v_b_800_ = v___x_837_;
goto _start;
}
else
{
lean_dec(v_a_832_);
goto v___jp_822_;
}
}
else
{
lean_object* v_a_839_; lean_object* v___x_841_; uint8_t v_isShared_842_; uint8_t v_isSharedCheck_846_; 
lean_del_object(v___x_820_);
lean_dec(v_snd_818_);
lean_dec(v_fst_817_);
lean_dec(v_a_814_);
v_a_839_ = lean_ctor_get(v___x_831_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v___x_831_);
if (v_isSharedCheck_846_ == 0)
{
v___x_841_ = v___x_831_;
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
else
{
lean_inc(v_a_839_);
lean_dec(v___x_831_);
v___x_841_ = lean_box(0);
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
v_resetjp_840_:
{
lean_object* v___x_844_; 
if (v_isShared_842_ == 0)
{
v___x_844_ = v___x_841_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_a_839_);
v___x_844_ = v_reuseFailAlloc_845_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
return v___x_844_;
}
}
}
}
v___jp_822_:
{
lean_object* v___x_823_; lean_object* v___x_825_; 
v___x_823_ = lean_array_push(v_fst_817_, v_a_814_);
if (v_isShared_821_ == 0)
{
lean_ctor_set(v___x_820_, 0, v___x_823_);
v___x_825_ = v___x_820_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v___x_823_);
lean_ctor_set(v_reuseFailAlloc_827_, 1, v_snd_818_);
v___x_825_ = v_reuseFailAlloc_827_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
v_as_x27_799_ = v_tail_810_;
v_b_800_ = v___x_825_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_855_; 
lean_dec(v_a_814_);
lean_dec_ref(v_b_800_);
v_a_848_ = lean_ctor_get(v___x_815_, 0);
v_isSharedCheck_855_ = !lean_is_exclusive(v___x_815_);
if (v_isSharedCheck_855_ == 0)
{
v___x_850_ = v___x_815_;
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_a_848_);
lean_dec(v___x_815_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
lean_object* v___x_853_; 
if (v_isShared_851_ == 0)
{
v___x_853_ = v___x_850_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_854_; 
v_reuseFailAlloc_854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_854_, 0, v_a_848_);
v___x_853_ = v_reuseFailAlloc_854_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
return v___x_853_;
}
}
}
}
else
{
lean_object* v_a_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_863_; 
lean_dec_ref(v_b_800_);
v_a_856_ = lean_ctor_get(v___x_811_, 0);
v_isSharedCheck_863_ = !lean_is_exclusive(v___x_811_);
if (v_isSharedCheck_863_ == 0)
{
v___x_858_ = v___x_811_;
v_isShared_859_ = v_isSharedCheck_863_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_a_856_);
lean_dec(v___x_811_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_863_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v___x_861_; 
if (v_isShared_859_ == 0)
{
v___x_861_ = v___x_858_;
goto v_reusejp_860_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v_a_856_);
v___x_861_ = v_reuseFailAlloc_862_;
goto v_reusejp_860_;
}
v_reusejp_860_:
{
return v___x_861_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg___boxed(lean_object* v_as_x27_864_, lean_object* v_b_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_){
_start:
{
lean_object* v_res_873_; 
v_res_873_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg(v_as_x27_864_, v_b_865_, v___y_866_, v___y_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
lean_dec(v_as_x27_864_);
return v_res_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg(lean_object* v_mvarId_874_, lean_object* v___y_875_){
_start:
{
lean_object* v___x_877_; lean_object* v_mctx_878_; lean_object* v_eAssignment_879_; uint8_t v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; 
v___x_877_ = lean_st_ref_get(v___y_875_);
v_mctx_878_ = lean_ctor_get(v___x_877_, 0);
lean_inc_ref(v_mctx_878_);
lean_dec(v___x_877_);
v_eAssignment_879_ = lean_ctor_get(v_mctx_878_, 8);
lean_inc_ref(v_eAssignment_879_);
lean_dec_ref(v_mctx_878_);
v___x_880_ = l_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_Meta_synthAppInstances_spec__0_spec__0___redArg(v_eAssignment_879_, v_mvarId_874_);
lean_dec_ref(v_eAssignment_879_);
v___x_881_ = lean_box(v___x_880_);
v___x_882_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_882_, 0, v___x_881_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg___boxed(lean_object* v_mvarId_883_, lean_object* v___y_884_, lean_object* v___y_885_){
_start:
{
lean_object* v_res_886_; 
v_res_886_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg(v_mvarId_883_, v___y_884_);
lean_dec(v___y_884_);
lean_dec(v_mvarId_883_);
return v_res_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8(lean_object* v_as_887_, size_t v_i_888_, size_t v_stop_889_, lean_object* v_b_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_){
_start:
{
lean_object* v_a_899_; uint8_t v___x_903_; 
v___x_903_ = lean_usize_dec_eq(v_i_888_, v_stop_889_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_904_ = lean_array_uget_borrowed(v_as_887_, v_i_888_);
v___x_907_ = l_Lean_Expr_mvarId_x21(v___x_904_);
v___x_908_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg(v___x_907_, v___y_894_);
lean_dec(v___x_907_);
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_a_909_; uint8_t v___x_910_; 
v_a_909_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_a_909_);
lean_dec_ref_known(v___x_908_, 1);
v___x_910_ = lean_unbox(v_a_909_);
lean_dec(v_a_909_);
if (v___x_910_ == 0)
{
goto v___jp_905_;
}
else
{
v_a_899_ = v_b_890_;
goto v___jp_898_;
}
}
else
{
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_a_911_; uint8_t v___x_912_; 
v_a_911_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_a_911_);
lean_dec_ref_known(v___x_908_, 1);
v___x_912_ = lean_unbox(v_a_911_);
lean_dec(v_a_911_);
if (v___x_912_ == 0)
{
v_a_899_ = v_b_890_;
goto v___jp_898_;
}
else
{
goto v___jp_905_;
}
}
else
{
lean_object* v_a_913_; lean_object* v___x_915_; uint8_t v_isShared_916_; uint8_t v_isSharedCheck_920_; 
lean_dec_ref(v_b_890_);
v_a_913_ = lean_ctor_get(v___x_908_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_908_);
if (v_isSharedCheck_920_ == 0)
{
v___x_915_ = v___x_908_;
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
else
{
lean_inc(v_a_913_);
lean_dec(v___x_908_);
v___x_915_ = lean_box(0);
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
v_resetjp_914_:
{
lean_object* v___x_918_; 
if (v_isShared_916_ == 0)
{
v___x_918_ = v___x_915_;
goto v_reusejp_917_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v_a_913_);
v___x_918_ = v_reuseFailAlloc_919_;
goto v_reusejp_917_;
}
v_reusejp_917_:
{
return v___x_918_;
}
}
}
}
v___jp_905_:
{
lean_object* v___x_906_; 
lean_inc(v___x_904_);
v___x_906_ = lean_array_push(v_b_890_, v___x_904_);
v_a_899_ = v___x_906_;
goto v___jp_898_;
}
}
else
{
lean_object* v___x_921_; 
v___x_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_921_, 0, v_b_890_);
return v___x_921_;
}
v___jp_898_:
{
size_t v___x_900_; size_t v___x_901_; 
v___x_900_ = ((size_t)1ULL);
v___x_901_ = lean_usize_add(v_i_888_, v___x_900_);
v_i_888_ = v___x_901_;
v_b_890_ = v_a_899_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8___boxed(lean_object* v_as_922_, lean_object* v_i_923_, lean_object* v_stop_924_, lean_object* v_b_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
size_t v_i_boxed_933_; size_t v_stop_boxed_934_; lean_object* v_res_935_; 
v_i_boxed_933_ = lean_unbox_usize(v_i_923_);
lean_dec(v_i_923_);
v_stop_boxed_934_ = lean_unbox_usize(v_stop_924_);
lean_dec(v_stop_924_);
v_res_935_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8(v_as_922_, v_i_boxed_933_, v_stop_boxed_934_, v_b_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_);
lean_dec(v___y_931_);
lean_dec_ref(v___y_930_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
lean_dec_ref(v_as_922_);
return v_res_935_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7(void){
_start:
{
lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_949_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__6));
v___x_950_ = l_Lean_stringToMessageData(v___x_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try(lean_object* v_lem_951_, lean_object* v_assignableMVars_952_, lean_object* v_a_953_, lean_object* v_a_954_, lean_object* v_a_955_, lean_object* v_a_956_, lean_object* v_a_957_, lean_object* v_a_958_){
_start:
{
uint8_t v___y_961_; lean_object* v___y_962_; lean_object* v___y_963_; lean_object* v___y_964_; lean_object* v___y_965_; lean_object* v_filtered_966_; lean_object* v___y_967_; lean_object* v___y_968_; lean_object* v___y_969_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v___y_972_; uint8_t v___y_1044_; lean_object* v___y_1045_; lean_object* v___y_1046_; lean_object* v___y_1047_; uint8_t v___y_1048_; lean_object* v_htmls_1049_; lean_object* v___y_1050_; lean_object* v___y_1051_; lean_object* v___y_1052_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; uint8_t v___y_1072_; lean_object* v___y_1073_; lean_object* v___y_1074_; lean_object* v___y_1075_; lean_object* v___y_1076_; uint8_t v___y_1077_; lean_object* v___y_1078_; lean_object* v___y_1079_; lean_object* v___y_1080_; lean_object* v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; lean_object* v_a_1085_; uint8_t v___y_1134_; lean_object* v___y_1135_; lean_object* v___y_1136_; lean_object* v___y_1137_; lean_object* v___y_1138_; uint8_t v___y_1139_; lean_object* v___y_1140_; lean_object* v___y_1141_; lean_object* v___y_1142_; lean_object* v___y_1143_; lean_object* v___y_1144_; lean_object* v___y_1145_; lean_object* v___y_1146_; lean_object* v___y_1147_; lean_object* v___x_1157_; 
lean_inc_ref(v_lem_951_);
v___x_1157_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(v_lem_951_, v_a_955_, v_a_956_, v_a_957_, v_a_958_);
if (lean_obj_tag(v___x_1157_) == 0)
{
lean_object* v_a_1158_; lean_object* v_fst_1159_; lean_object* v_snd_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1329_; 
v_a_1158_ = lean_ctor_get(v___x_1157_, 0);
lean_inc(v_a_1158_);
lean_dec_ref_known(v___x_1157_, 1);
v_fst_1159_ = lean_ctor_get(v_a_1158_, 0);
v_snd_1160_ = lean_ctor_get(v_a_1158_, 1);
v_isSharedCheck_1329_ = !lean_is_exclusive(v_a_1158_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1162_ = v_a_1158_;
v_isShared_1163_ = v_isSharedCheck_1329_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_snd_1160_);
lean_inc(v_fst_1159_);
lean_dec(v_a_1158_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1329_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___y_1165_; lean_object* v___y_1166_; lean_object* v___y_1167_; lean_object* v___y_1168_; lean_object* v___y_1169_; lean_object* v___y_1170_; lean_object* v___y_1171_; lean_object* v_a_1172_; lean_object* v___y_1230_; lean_object* v___y_1231_; lean_object* v___y_1232_; lean_object* v___y_1233_; lean_object* v___y_1234_; lean_object* v___y_1235_; lean_object* v___y_1236_; lean_object* v___y_1237_; lean_object* v_snd_1247_; lean_object* v_fst_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1328_; 
v_snd_1247_ = lean_ctor_get(v_snd_1160_, 1);
v_fst_1248_ = lean_ctor_get(v_snd_1160_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v_snd_1160_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1250_ = v_snd_1160_;
v_isShared_1251_ = v_isSharedCheck_1328_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_snd_1247_);
lean_inc(v_fst_1248_);
lean_dec(v_snd_1160_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1328_;
goto v_resetjp_1249_;
}
v___jp_1164_:
{
uint8_t v___x_1173_; lean_object* v___x_1174_; 
v___x_1173_ = 0;
v___x_1174_ = l___private_Lean_Meta_Tactic_Apply_0__Lean_Meta_reorderGoals(v_a_1172_, v___x_1173_, v___y_1169_, v___y_1170_, v___y_1166_, v___y_1171_);
if (lean_obj_tag(v___x_1174_) == 0)
{
lean_object* v_a_1175_; lean_object* v___x_1176_; uint8_t v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1180_; 
v_a_1175_ = lean_ctor_get(v___x_1174_, 0);
lean_inc(v_a_1175_);
lean_dec_ref_known(v___x_1174_, 1);
v___x_1176_ = lean_mk_empty_array_with_capacity(v___y_1167_);
v___x_1177_ = 1;
v___x_1178_ = lean_box(v___x_1177_);
if (v_isShared_1163_ == 0)
{
lean_ctor_set(v___x_1162_, 1, v___x_1178_);
lean_ctor_set(v___x_1162_, 0, v___x_1176_);
v___x_1180_ = v___x_1162_;
goto v_reusejp_1179_;
}
else
{
lean_object* v_reuseFailAlloc_1220_; 
v_reuseFailAlloc_1220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1220_, 0, v___x_1176_);
lean_ctor_set(v_reuseFailAlloc_1220_, 1, v___x_1178_);
v___x_1180_ = v_reuseFailAlloc_1220_;
goto v_reusejp_1179_;
}
v_reusejp_1179_:
{
lean_object* v___x_1181_; 
v___x_1181_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg(v_a_1175_, v___x_1180_, v___y_1168_, v___y_1165_, v___y_1169_, v___y_1170_, v___y_1166_, v___y_1171_);
if (lean_obj_tag(v___x_1181_) == 0)
{
lean_object* v_a_1182_; lean_object* v_fst_1183_; lean_object* v_snd_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; 
v_a_1182_ = lean_ctor_get(v___x_1181_, 0);
lean_inc(v_a_1182_);
lean_dec_ref_known(v___x_1181_, 1);
v_fst_1183_ = lean_ctor_get(v_a_1182_, 0);
lean_inc(v_fst_1183_);
v_snd_1184_ = lean_ctor_get(v_a_1182_, 1);
lean_inc(v_snd_1184_);
lean_dec(v_a_1182_);
v___x_1185_ = lean_array_mk(v_a_1175_);
v___x_1186_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(v___x_1185_, v_assignableMVars_952_, v_fst_1183_, v___y_1169_, v___y_1170_, v___y_1166_, v___y_1171_);
lean_dec_ref(v___x_1185_);
if (lean_obj_tag(v___x_1186_) == 0)
{
lean_object* v_a_1187_; lean_object* v___x_1188_; lean_object* v_a_1189_; lean_object* v___x_1190_; uint8_t v___x_1191_; uint8_t v___x_1192_; 
v_a_1187_ = lean_ctor_get(v___x_1186_, 0);
lean_inc(v_a_1187_);
lean_dec_ref_known(v___x_1186_, 1);
v___x_1188_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__1___redArg(v_fst_1159_, v___y_1170_);
v_a_1189_ = lean_ctor_get(v___x_1188_, 0);
lean_inc(v_a_1189_);
lean_dec_ref(v___x_1188_);
v___x_1190_ = lean_array_get_size(v_fst_1183_);
v___x_1191_ = lean_nat_dec_eq(v___x_1190_, v___y_1167_);
v___x_1192_ = lean_nat_dec_lt(v___y_1167_, v___x_1190_);
if (v___x_1192_ == 0)
{
uint8_t v___x_1193_; 
v___x_1193_ = lean_unbox(v_a_1187_);
lean_dec(v_a_1187_);
lean_inc(v___y_1167_);
v___y_1072_ = v___x_1191_;
v___y_1073_ = v___y_1165_;
v___y_1074_ = v___y_1166_;
v___y_1075_ = v___y_1167_;
v___y_1076_ = v___y_1168_;
v___y_1077_ = v___x_1193_;
v___y_1078_ = v___y_1171_;
v___y_1079_ = v_a_1189_;
v___y_1080_ = v_fst_1183_;
v___y_1081_ = v___y_1169_;
v___y_1082_ = v___x_1190_;
v___y_1083_ = v___y_1170_;
v___y_1084_ = v_snd_1184_;
v_a_1085_ = v___y_1167_;
goto v___jp_1071_;
}
else
{
uint8_t v___x_1194_; 
v___x_1194_ = lean_nat_dec_le(v___x_1190_, v___x_1190_);
if (v___x_1194_ == 0)
{
if (v___x_1192_ == 0)
{
uint8_t v___x_1195_; 
v___x_1195_ = lean_unbox(v_a_1187_);
lean_dec(v_a_1187_);
lean_inc(v___y_1167_);
v___y_1072_ = v___x_1191_;
v___y_1073_ = v___y_1165_;
v___y_1074_ = v___y_1166_;
v___y_1075_ = v___y_1167_;
v___y_1076_ = v___y_1168_;
v___y_1077_ = v___x_1195_;
v___y_1078_ = v___y_1171_;
v___y_1079_ = v_a_1189_;
v___y_1080_ = v_fst_1183_;
v___y_1081_ = v___y_1169_;
v___y_1082_ = v___x_1190_;
v___y_1083_ = v___y_1170_;
v___y_1084_ = v_snd_1184_;
v_a_1085_ = v___y_1167_;
goto v___jp_1071_;
}
else
{
size_t v___x_1196_; size_t v___x_1197_; lean_object* v___x_1198_; uint8_t v___x_1199_; 
v___x_1196_ = ((size_t)0ULL);
v___x_1197_ = lean_usize_of_nat(v___x_1190_);
lean_inc(v___y_1167_);
v___x_1198_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(v_fst_1183_, v___x_1196_, v___x_1197_, v___y_1167_, v___y_1169_, v___y_1170_, v___y_1166_, v___y_1171_);
v___x_1199_ = lean_unbox(v_a_1187_);
lean_dec(v_a_1187_);
v___y_1134_ = v___x_1191_;
v___y_1135_ = v___y_1165_;
v___y_1136_ = v___y_1166_;
v___y_1137_ = v___y_1167_;
v___y_1138_ = v___y_1168_;
v___y_1139_ = v___x_1199_;
v___y_1140_ = v___y_1171_;
v___y_1141_ = v_a_1189_;
v___y_1142_ = v_fst_1183_;
v___y_1143_ = v___y_1169_;
v___y_1144_ = v___x_1190_;
v___y_1145_ = v___y_1170_;
v___y_1146_ = v_snd_1184_;
v___y_1147_ = v___x_1198_;
goto v___jp_1133_;
}
}
else
{
size_t v___x_1200_; size_t v___x_1201_; lean_object* v___x_1202_; uint8_t v___x_1203_; 
v___x_1200_ = ((size_t)0ULL);
v___x_1201_ = lean_usize_of_nat(v___x_1190_);
lean_inc(v___y_1167_);
v___x_1202_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(v_fst_1183_, v___x_1200_, v___x_1201_, v___y_1167_, v___y_1169_, v___y_1170_, v___y_1166_, v___y_1171_);
v___x_1203_ = lean_unbox(v_a_1187_);
lean_dec(v_a_1187_);
v___y_1134_ = v___x_1191_;
v___y_1135_ = v___y_1165_;
v___y_1136_ = v___y_1166_;
v___y_1137_ = v___y_1167_;
v___y_1138_ = v___y_1168_;
v___y_1139_ = v___x_1203_;
v___y_1140_ = v___y_1171_;
v___y_1141_ = v_a_1189_;
v___y_1142_ = v_fst_1183_;
v___y_1143_ = v___y_1169_;
v___y_1144_ = v___x_1190_;
v___y_1145_ = v___y_1170_;
v___y_1146_ = v_snd_1184_;
v___y_1147_ = v___x_1202_;
goto v___jp_1133_;
}
}
}
else
{
lean_object* v_a_1204_; lean_object* v___x_1206_; uint8_t v_isShared_1207_; uint8_t v_isSharedCheck_1211_; 
lean_dec(v_snd_1184_);
lean_dec(v_fst_1183_);
lean_dec(v___y_1167_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1204_ = lean_ctor_get(v___x_1186_, 0);
v_isSharedCheck_1211_ = !lean_is_exclusive(v___x_1186_);
if (v_isSharedCheck_1211_ == 0)
{
v___x_1206_ = v___x_1186_;
v_isShared_1207_ = v_isSharedCheck_1211_;
goto v_resetjp_1205_;
}
else
{
lean_inc(v_a_1204_);
lean_dec(v___x_1186_);
v___x_1206_ = lean_box(0);
v_isShared_1207_ = v_isSharedCheck_1211_;
goto v_resetjp_1205_;
}
v_resetjp_1205_:
{
lean_object* v___x_1209_; 
if (v_isShared_1207_ == 0)
{
v___x_1209_ = v___x_1206_;
goto v_reusejp_1208_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v_a_1204_);
v___x_1209_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1208_;
}
v_reusejp_1208_:
{
return v___x_1209_;
}
}
}
}
else
{
lean_object* v_a_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1219_; 
lean_dec(v_a_1175_);
lean_dec(v___y_1167_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1212_ = lean_ctor_get(v___x_1181_, 0);
v_isSharedCheck_1219_ = !lean_is_exclusive(v___x_1181_);
if (v_isSharedCheck_1219_ == 0)
{
v___x_1214_ = v___x_1181_;
v_isShared_1215_ = v_isSharedCheck_1219_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_a_1212_);
lean_dec(v___x_1181_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1219_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v___x_1217_; 
if (v_isShared_1215_ == 0)
{
v___x_1217_ = v___x_1214_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v_a_1212_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
}
}
}
else
{
lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1228_; 
lean_dec(v___y_1167_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1221_ = lean_ctor_get(v___x_1174_, 0);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1174_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1223_ = v___x_1174_;
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1174_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v___x_1226_; 
if (v_isShared_1224_ == 0)
{
v___x_1226_ = v___x_1223_;
goto v_reusejp_1225_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_a_1221_);
v___x_1226_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1225_;
}
v_reusejp_1225_:
{
return v___x_1226_;
}
}
}
}
v___jp_1229_:
{
if (lean_obj_tag(v___y_1237_) == 0)
{
lean_object* v_a_1238_; 
v_a_1238_ = lean_ctor_get(v___y_1237_, 0);
lean_inc(v_a_1238_);
lean_dec_ref_known(v___y_1237_, 1);
v___y_1165_ = v___y_1230_;
v___y_1166_ = v___y_1231_;
v___y_1167_ = v___y_1232_;
v___y_1168_ = v___y_1233_;
v___y_1169_ = v___y_1234_;
v___y_1170_ = v___y_1235_;
v___y_1171_ = v___y_1236_;
v_a_1172_ = v_a_1238_;
goto v___jp_1164_;
}
else
{
lean_object* v_a_1239_; lean_object* v___x_1241_; uint8_t v_isShared_1242_; uint8_t v_isSharedCheck_1246_; 
lean_dec(v___y_1232_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1239_ = lean_ctor_get(v___y_1237_, 0);
v_isSharedCheck_1246_ = !lean_is_exclusive(v___y_1237_);
if (v_isSharedCheck_1246_ == 0)
{
v___x_1241_ = v___y_1237_;
v_isShared_1242_ = v_isSharedCheck_1246_;
goto v_resetjp_1240_;
}
else
{
lean_inc(v_a_1239_);
lean_dec(v___y_1237_);
v___x_1241_ = lean_box(0);
v_isShared_1242_ = v_isSharedCheck_1246_;
goto v_resetjp_1240_;
}
v_resetjp_1240_:
{
lean_object* v___x_1244_; 
if (v_isShared_1242_ == 0)
{
v___x_1244_ = v___x_1241_;
goto v_reusejp_1243_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v_a_1239_);
v___x_1244_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1243_;
}
v_reusejp_1243_:
{
return v___x_1244_;
}
}
}
}
v_resetjp_1249_:
{
lean_object* v_fst_1252_; lean_object* v_snd_1253_; lean_object* v___x_1255_; uint8_t v_isShared_1256_; uint8_t v_isSharedCheck_1327_; 
v_fst_1252_ = lean_ctor_get(v_snd_1247_, 0);
v_snd_1253_ = lean_ctor_get(v_snd_1247_, 1);
v_isSharedCheck_1327_ = !lean_is_exclusive(v_snd_1247_);
if (v_isSharedCheck_1327_ == 0)
{
v___x_1255_ = v_snd_1247_;
v_isShared_1256_ = v_isSharedCheck_1327_;
goto v_resetjp_1254_;
}
else
{
lean_inc(v_snd_1253_);
lean_inc(v_fst_1252_);
lean_dec(v_snd_1247_);
v___x_1255_ = lean_box(0);
v_isShared_1256_ = v_isSharedCheck_1327_;
goto v_resetjp_1254_;
}
v_resetjp_1254_:
{
lean_object* v___y_1258_; lean_object* v___y_1259_; lean_object* v___y_1260_; lean_object* v___y_1261_; lean_object* v___y_1262_; lean_object* v___y_1263_; lean_object* v_goal_1287_; lean_object* v___x_1288_; 
v_goal_1287_ = lean_ctor_get(v_a_953_, 7);
lean_inc(v_goal_1287_);
v___x_1288_ = l_Lean_MVarId_getType(v_goal_1287_, v_a_955_, v_a_956_, v_a_957_, v_a_958_);
if (lean_obj_tag(v___x_1288_) == 0)
{
lean_object* v_a_1289_; lean_object* v___x_1290_; 
v_a_1289_ = lean_ctor_get(v___x_1288_, 0);
lean_inc_n(v_a_1289_, 2);
lean_dec_ref_known(v___x_1288_, 1);
lean_inc(v_snd_1253_);
v___x_1290_ = l_Lean_Meta_isExprDefEq(v_snd_1253_, v_a_1289_, v_a_955_, v_a_956_, v_a_957_, v_a_958_);
if (lean_obj_tag(v___x_1290_) == 0)
{
lean_object* v_a_1291_; uint8_t v___x_1292_; 
v_a_1291_ = lean_ctor_get(v___x_1290_, 0);
lean_inc(v_a_1291_);
lean_dec_ref_known(v___x_1290_, 1);
v___x_1292_ = lean_unbox(v_a_1291_);
lean_dec(v_a_1291_);
if (v___x_1292_ == 0)
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1296_; 
lean_dec(v_fst_1252_);
lean_dec(v_fst_1248_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v___x_1293_ = l_Lean_MessageData_ofExpr(v_snd_1253_);
v___x_1294_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__7);
if (v_isShared_1256_ == 0)
{
lean_ctor_set_tag(v___x_1255_, 7);
lean_ctor_set(v___x_1255_, 1, v___x_1294_);
lean_ctor_set(v___x_1255_, 0, v___x_1293_);
v___x_1296_ = v___x_1255_;
goto v_reusejp_1295_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v___x_1293_);
lean_ctor_set(v_reuseFailAlloc_1310_, 1, v___x_1294_);
v___x_1296_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1295_;
}
v_reusejp_1295_:
{
lean_object* v___x_1297_; lean_object* v___x_1299_; 
v___x_1297_ = l_Lean_MessageData_ofExpr(v_a_1289_);
if (v_isShared_1251_ == 0)
{
lean_ctor_set_tag(v___x_1250_, 7);
lean_ctor_set(v___x_1250_, 1, v___x_1297_);
lean_ctor_set(v___x_1250_, 0, v___x_1296_);
v___x_1299_ = v___x_1250_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1309_; 
v_reuseFailAlloc_1309_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1309_, 0, v___x_1296_);
lean_ctor_set(v_reuseFailAlloc_1309_, 1, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1309_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
lean_object* v___x_1300_; lean_object* v_a_1301_; lean_object* v___x_1303_; uint8_t v_isShared_1304_; uint8_t v_isSharedCheck_1308_; 
v___x_1300_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg(v___x_1299_, v_a_955_, v_a_956_, v_a_957_, v_a_958_);
v_a_1301_ = lean_ctor_get(v___x_1300_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v___x_1300_);
if (v_isSharedCheck_1308_ == 0)
{
v___x_1303_ = v___x_1300_;
v_isShared_1304_ = v_isSharedCheck_1308_;
goto v_resetjp_1302_;
}
else
{
lean_inc(v_a_1301_);
lean_dec(v___x_1300_);
v___x_1303_ = lean_box(0);
v_isShared_1304_ = v_isSharedCheck_1308_;
goto v_resetjp_1302_;
}
v_resetjp_1302_:
{
lean_object* v___x_1306_; 
if (v_isShared_1304_ == 0)
{
v___x_1306_ = v___x_1303_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v_a_1301_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
}
}
}
else
{
lean_dec(v_a_1289_);
lean_del_object(v___x_1255_);
lean_dec(v_snd_1253_);
lean_del_object(v___x_1250_);
v___y_1258_ = v_a_953_;
v___y_1259_ = v_a_954_;
v___y_1260_ = v_a_955_;
v___y_1261_ = v_a_956_;
v___y_1262_ = v_a_957_;
v___y_1263_ = v_a_958_;
goto v___jp_1257_;
}
}
else
{
lean_object* v_a_1311_; lean_object* v___x_1313_; uint8_t v_isShared_1314_; uint8_t v_isSharedCheck_1318_; 
lean_dec(v_a_1289_);
lean_del_object(v___x_1255_);
lean_dec(v_snd_1253_);
lean_dec(v_fst_1252_);
lean_del_object(v___x_1250_);
lean_dec(v_fst_1248_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1311_ = lean_ctor_get(v___x_1290_, 0);
v_isSharedCheck_1318_ = !lean_is_exclusive(v___x_1290_);
if (v_isSharedCheck_1318_ == 0)
{
v___x_1313_ = v___x_1290_;
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
else
{
lean_inc(v_a_1311_);
lean_dec(v___x_1290_);
v___x_1313_ = lean_box(0);
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
v_resetjp_1312_:
{
lean_object* v___x_1316_; 
if (v_isShared_1314_ == 0)
{
v___x_1316_ = v___x_1313_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_a_1311_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
}
}
else
{
lean_object* v_a_1319_; lean_object* v___x_1321_; uint8_t v_isShared_1322_; uint8_t v_isSharedCheck_1326_; 
lean_del_object(v___x_1255_);
lean_dec(v_snd_1253_);
lean_dec(v_fst_1252_);
lean_del_object(v___x_1250_);
lean_dec(v_fst_1248_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1319_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1326_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1326_ == 0)
{
v___x_1321_ = v___x_1288_;
v_isShared_1322_ = v_isSharedCheck_1326_;
goto v_resetjp_1320_;
}
else
{
lean_inc(v_a_1319_);
lean_dec(v___x_1288_);
v___x_1321_ = lean_box(0);
v_isShared_1322_ = v_isSharedCheck_1326_;
goto v_resetjp_1320_;
}
v_resetjp_1320_:
{
lean_object* v___x_1324_; 
if (v_isShared_1322_ == 0)
{
v___x_1324_ = v___x_1321_;
goto v_reusejp_1323_;
}
else
{
lean_object* v_reuseFailAlloc_1325_; 
v_reuseFailAlloc_1325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1325_, 0, v_a_1319_);
v___x_1324_ = v_reuseFailAlloc_1325_;
goto v_reusejp_1323_;
}
v_reusejp_1323_:
{
return v___x_1324_;
}
}
}
v___jp_1257_:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; uint8_t v___x_1266_; lean_object* v___x_1267_; 
v___x_1264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__4));
v___x_1265_ = lean_box(0);
v___x_1266_ = 0;
v___x_1267_ = l_Lean_Meta_synthAppInstances(v___x_1264_, v___x_1265_, v_fst_1248_, v_fst_1252_, v___x_1266_, v___x_1266_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_);
if (lean_obj_tag(v___x_1267_) == 0)
{
lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; uint8_t v___x_1271_; 
lean_dec_ref_known(v___x_1267_, 1);
v___x_1268_ = lean_unsigned_to_nat(0u);
v___x_1269_ = lean_array_get_size(v_fst_1248_);
v___x_1270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__5));
v___x_1271_ = lean_nat_dec_lt(v___x_1268_, v___x_1269_);
if (v___x_1271_ == 0)
{
lean_dec(v_fst_1248_);
v___y_1165_ = v___y_1259_;
v___y_1166_ = v___y_1262_;
v___y_1167_ = v___x_1268_;
v___y_1168_ = v___y_1258_;
v___y_1169_ = v___y_1260_;
v___y_1170_ = v___y_1261_;
v___y_1171_ = v___y_1263_;
v_a_1172_ = v___x_1270_;
goto v___jp_1164_;
}
else
{
uint8_t v___x_1272_; 
v___x_1272_ = lean_nat_dec_le(v___x_1269_, v___x_1269_);
if (v___x_1272_ == 0)
{
if (v___x_1271_ == 0)
{
lean_dec(v_fst_1248_);
v___y_1165_ = v___y_1259_;
v___y_1166_ = v___y_1262_;
v___y_1167_ = v___x_1268_;
v___y_1168_ = v___y_1258_;
v___y_1169_ = v___y_1260_;
v___y_1170_ = v___y_1261_;
v___y_1171_ = v___y_1263_;
v_a_1172_ = v___x_1270_;
goto v___jp_1164_;
}
else
{
size_t v___x_1273_; size_t v___x_1274_; lean_object* v___x_1275_; 
v___x_1273_ = ((size_t)0ULL);
v___x_1274_ = lean_usize_of_nat(v___x_1269_);
v___x_1275_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8(v_fst_1248_, v___x_1273_, v___x_1274_, v___x_1270_, v___y_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_);
lean_dec(v_fst_1248_);
v___y_1230_ = v___y_1259_;
v___y_1231_ = v___y_1262_;
v___y_1232_ = v___x_1268_;
v___y_1233_ = v___y_1258_;
v___y_1234_ = v___y_1260_;
v___y_1235_ = v___y_1261_;
v___y_1236_ = v___y_1263_;
v___y_1237_ = v___x_1275_;
goto v___jp_1229_;
}
}
else
{
size_t v___x_1276_; size_t v___x_1277_; lean_object* v___x_1278_; 
v___x_1276_ = ((size_t)0ULL);
v___x_1277_ = lean_usize_of_nat(v___x_1269_);
v___x_1278_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__8(v_fst_1248_, v___x_1276_, v___x_1277_, v___x_1270_, v___y_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_);
lean_dec(v_fst_1248_);
v___y_1230_ = v___y_1259_;
v___y_1231_ = v___y_1262_;
v___y_1232_ = v___x_1268_;
v___y_1233_ = v___y_1258_;
v___y_1234_ = v___y_1260_;
v___y_1235_ = v___y_1261_;
v___y_1236_ = v___y_1263_;
v___y_1237_ = v___x_1278_;
goto v___jp_1229_;
}
}
}
else
{
lean_object* v_a_1279_; lean_object* v___x_1281_; uint8_t v_isShared_1282_; uint8_t v_isSharedCheck_1286_; 
lean_dec(v_fst_1248_);
lean_del_object(v___x_1162_);
lean_dec(v_fst_1159_);
lean_dec_ref(v_lem_951_);
v_a_1279_ = lean_ctor_get(v___x_1267_, 0);
v_isSharedCheck_1286_ = !lean_is_exclusive(v___x_1267_);
if (v_isSharedCheck_1286_ == 0)
{
v___x_1281_ = v___x_1267_;
v_isShared_1282_ = v_isSharedCheck_1286_;
goto v_resetjp_1280_;
}
else
{
lean_inc(v_a_1279_);
lean_dec(v___x_1267_);
v___x_1281_ = lean_box(0);
v_isShared_1282_ = v_isSharedCheck_1286_;
goto v_resetjp_1280_;
}
v_resetjp_1280_:
{
lean_object* v___x_1284_; 
if (v_isShared_1282_ == 0)
{
v___x_1284_ = v___x_1281_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1285_; 
v_reuseFailAlloc_1285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1285_, 0, v_a_1279_);
v___x_1284_ = v_reuseFailAlloc_1285_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
return v___x_1284_;
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
lean_object* v_a_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1337_; 
lean_dec_ref(v_lem_951_);
v_a_1330_ = lean_ctor_get(v___x_1157_, 0);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1157_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1332_ = v___x_1157_;
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_a_1330_);
lean_dec(v___x_1157_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___x_1335_; 
if (v_isShared_1333_ == 0)
{
v___x_1335_ = v___x_1332_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_a_1330_);
v___x_1335_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
return v___x_1335_;
}
}
}
v___jp_960_:
{
lean_object* v___x_973_; 
lean_inc_ref(v_lem_951_);
v___x_973_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(v_lem_951_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
if (lean_obj_tag(v___x_973_) == 0)
{
lean_object* v_a_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v_a_974_ = lean_ctor_get(v___x_973_, 0);
lean_inc(v_a_974_);
lean_dec_ref_known(v___x_973_, 1);
v___x_975_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0));
v___x_976_ = lean_mk_empty_array_with_capacity(v___y_964_);
lean_dec(v___y_964_);
v___x_977_ = lean_unsigned_to_nat(1u);
v___x_978_ = lean_mk_empty_array_with_capacity(v___x_977_);
v___x_979_ = lean_array_push(v___x_978_, v_a_974_);
lean_inc_ref(v___x_976_);
v___x_980_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_980_, 0, v___x_975_);
lean_ctor_set(v___x_980_, 1, v___x_976_);
lean_ctor_set(v___x_980_, 2, v___x_979_);
v___x_981_ = lean_array_push(v___y_965_, v___x_980_);
v___x_982_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_982_, 0, v___x_975_);
lean_ctor_set(v___x_982_, 1, v___x_976_);
lean_ctor_set(v___x_982_, 2, v___x_981_);
v___x_983_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_962_, v___x_982_, v___y_961_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; lean_object* v___x_985_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v___x_983_, 1);
v___x_985_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(v_lem_951_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; lean_object* v___x_987_; uint8_t v___x_988_; lean_object* v___x_989_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
v___x_987_ = lean_box(0);
v___x_988_ = 0;
v___x_989_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_986_, v___x_987_, v___x_988_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v_a_990_; lean_object* v_snd_991_; lean_object* v_snd_992_; lean_object* v___x_993_; 
v_a_990_ = lean_ctor_get(v___x_989_, 0);
lean_inc(v_a_990_);
lean_dec_ref_known(v___x_989_, 1);
v_snd_991_ = lean_ctor_get(v_a_990_, 1);
lean_inc(v_snd_991_);
lean_dec(v_a_990_);
v_snd_992_ = lean_ctor_get(v_snd_991_, 1);
lean_inc(v_snd_992_);
lean_dec(v_snd_991_);
v___x_993_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_snd_992_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
if (lean_obj_tag(v___x_993_) == 0)
{
lean_object* v_a_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1002_; 
v_a_994_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_996_ = v___x_993_;
v_isShared_997_ = v_isSharedCheck_1002_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_a_994_);
lean_dec(v___x_993_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1002_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v___x_998_; lean_object* v___x_1000_; 
v___x_998_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_998_, 0, v_filtered_966_);
lean_ctor_set(v___x_998_, 1, v_a_984_);
lean_ctor_set(v___x_998_, 2, v___y_963_);
lean_ctor_set(v___x_998_, 3, v_a_994_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 0, v___x_998_);
v___x_1000_ = v___x_996_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1001_; 
v_reuseFailAlloc_1001_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1001_, 0, v___x_998_);
v___x_1000_ = v_reuseFailAlloc_1001_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
return v___x_1000_;
}
}
}
else
{
lean_object* v_a_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1010_; 
lean_dec(v_a_984_);
lean_dec(v_filtered_966_);
lean_dec_ref(v___y_963_);
v_a_1003_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1010_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1010_ == 0)
{
v___x_1005_ = v___x_993_;
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_a_1003_);
lean_dec(v___x_993_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1008_; 
if (v_isShared_1006_ == 0)
{
v___x_1008_ = v___x_1005_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_a_1003_);
v___x_1008_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1007_;
}
v_reusejp_1007_:
{
return v___x_1008_;
}
}
}
}
else
{
lean_object* v_a_1011_; lean_object* v___x_1013_; uint8_t v_isShared_1014_; uint8_t v_isSharedCheck_1018_; 
lean_dec(v_a_984_);
lean_dec(v_filtered_966_);
lean_dec_ref(v___y_963_);
v_a_1011_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1018_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1018_ == 0)
{
v___x_1013_ = v___x_989_;
v_isShared_1014_ = v_isSharedCheck_1018_;
goto v_resetjp_1012_;
}
else
{
lean_inc(v_a_1011_);
lean_dec(v___x_989_);
v___x_1013_ = lean_box(0);
v_isShared_1014_ = v_isSharedCheck_1018_;
goto v_resetjp_1012_;
}
v_resetjp_1012_:
{
lean_object* v___x_1016_; 
if (v_isShared_1014_ == 0)
{
v___x_1016_ = v___x_1013_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v_a_1011_);
v___x_1016_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
return v___x_1016_;
}
}
}
}
else
{
lean_object* v_a_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1026_; 
lean_dec(v_a_984_);
lean_dec(v_filtered_966_);
lean_dec_ref(v___y_963_);
v_a_1019_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1026_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1026_ == 0)
{
v___x_1021_ = v___x_985_;
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_a_1019_);
lean_dec(v___x_985_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v___x_1024_; 
if (v_isShared_1022_ == 0)
{
v___x_1024_ = v___x_1021_;
goto v_reusejp_1023_;
}
else
{
lean_object* v_reuseFailAlloc_1025_; 
v_reuseFailAlloc_1025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1025_, 0, v_a_1019_);
v___x_1024_ = v_reuseFailAlloc_1025_;
goto v_reusejp_1023_;
}
v_reusejp_1023_:
{
return v___x_1024_;
}
}
}
}
else
{
lean_object* v_a_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1034_; 
lean_dec(v_filtered_966_);
lean_dec_ref(v___y_963_);
lean_dec_ref(v_lem_951_);
v_a_1027_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1034_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1034_ == 0)
{
v___x_1029_ = v___x_983_;
v_isShared_1030_ = v_isSharedCheck_1034_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_a_1027_);
lean_dec(v___x_983_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1034_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1032_; 
if (v_isShared_1030_ == 0)
{
v___x_1032_ = v___x_1029_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1033_; 
v_reuseFailAlloc_1033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1033_, 0, v_a_1027_);
v___x_1032_ = v_reuseFailAlloc_1033_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
return v___x_1032_;
}
}
}
}
else
{
lean_object* v_a_1035_; lean_object* v___x_1037_; uint8_t v_isShared_1038_; uint8_t v_isSharedCheck_1042_; 
lean_dec(v_filtered_966_);
lean_dec_ref(v___y_965_);
lean_dec(v___y_964_);
lean_dec_ref(v___y_963_);
lean_dec(v___y_962_);
lean_dec_ref(v_lem_951_);
v_a_1035_ = lean_ctor_get(v___x_973_, 0);
v_isSharedCheck_1042_ = !lean_is_exclusive(v___x_973_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1037_ = v___x_973_;
v_isShared_1038_ = v_isSharedCheck_1042_;
goto v_resetjp_1036_;
}
else
{
lean_inc(v_a_1035_);
lean_dec(v___x_973_);
v___x_1037_ = lean_box(0);
v_isShared_1038_ = v_isSharedCheck_1042_;
goto v_resetjp_1036_;
}
v_resetjp_1036_:
{
lean_object* v___x_1040_; 
if (v_isShared_1038_ == 0)
{
v___x_1040_ = v___x_1037_;
goto v_reusejp_1039_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v_a_1035_);
v___x_1040_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1039_;
}
v_reusejp_1039_:
{
return v___x_1040_;
}
}
}
}
v___jp_1043_:
{
if (v___y_1048_ == 0)
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1056_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg___closed__0));
v___x_1057_ = lean_mk_empty_array_with_capacity(v___y_1047_);
lean_inc_ref(v_htmls_1049_);
v___x_1058_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1056_);
lean_ctor_set(v___x_1058_, 1, v___x_1057_);
lean_ctor_set(v___x_1058_, 2, v_htmls_1049_);
lean_inc(v___y_1045_);
v___x_1059_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_1045_, v___x_1058_, v___y_1044_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_);
if (lean_obj_tag(v___x_1059_) == 0)
{
lean_object* v_a_1060_; lean_object* v___x_1061_; 
v_a_1060_ = lean_ctor_get(v___x_1059_, 0);
lean_inc(v_a_1060_);
lean_dec_ref_known(v___x_1059_, 1);
v___x_1061_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1061_, 0, v_a_1060_);
v___y_961_ = v___y_1044_;
v___y_962_ = v___y_1045_;
v___y_963_ = v___y_1046_;
v___y_964_ = v___y_1047_;
v___y_965_ = v_htmls_1049_;
v_filtered_966_ = v___x_1061_;
v___y_967_ = v___y_1050_;
v___y_968_ = v___y_1051_;
v___y_969_ = v___y_1052_;
v___y_970_ = v___y_1053_;
v___y_971_ = v___y_1054_;
v___y_972_ = v___y_1055_;
goto v___jp_960_;
}
else
{
lean_object* v_a_1062_; lean_object* v___x_1064_; uint8_t v_isShared_1065_; uint8_t v_isSharedCheck_1069_; 
lean_dec_ref(v_htmls_1049_);
lean_dec(v___y_1047_);
lean_dec_ref(v___y_1046_);
lean_dec(v___y_1045_);
lean_dec_ref(v_lem_951_);
v_a_1062_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1069_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1064_ = v___x_1059_;
v_isShared_1065_ = v_isSharedCheck_1069_;
goto v_resetjp_1063_;
}
else
{
lean_inc(v_a_1062_);
lean_dec(v___x_1059_);
v___x_1064_ = lean_box(0);
v_isShared_1065_ = v_isSharedCheck_1069_;
goto v_resetjp_1063_;
}
v_resetjp_1063_:
{
lean_object* v___x_1067_; 
if (v_isShared_1065_ == 0)
{
v___x_1067_ = v___x_1064_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v_a_1062_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
}
}
else
{
lean_object* v___x_1070_; 
v___x_1070_ = lean_box(0);
v___y_961_ = v___y_1044_;
v___y_962_ = v___y_1045_;
v___y_963_ = v___y_1046_;
v___y_964_ = v___y_1047_;
v___y_965_ = v_htmls_1049_;
v_filtered_966_ = v___x_1070_;
v___y_967_ = v___y_1050_;
v___y_968_ = v___y_1051_;
v___y_969_ = v___y_1052_;
v___y_970_ = v___y_1053_;
v___y_971_ = v___y_1054_;
v___y_972_ = v___y_1055_;
goto v___jp_960_;
}
}
v___jp_1071_:
{
size_t v_sz_1086_; size_t v___x_1087_; lean_object* v___x_1088_; 
v_sz_1086_ = lean_array_size(v___y_1080_);
v___x_1087_ = ((size_t)0ULL);
lean_inc(v___y_1080_);
v___x_1088_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg(v_sz_1086_, v___x_1087_, v___y_1080_, v___y_1081_, v___y_1083_, v___y_1074_, v___y_1078_);
if (lean_obj_tag(v___x_1088_) == 0)
{
lean_object* v_a_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; uint8_t v___x_1092_; lean_object* v___x_1093_; 
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc(v_a_1089_);
lean_dec_ref_known(v___x_1088_, 1);
lean_inc_ref_n(v_lem_951_, 3);
v___x_1090_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(v_lem_951_);
v___x_1091_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(v_lem_951_);
v___x_1092_ = lean_unbox(v___y_1084_);
lean_dec(v___y_1084_);
v___x_1093_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_Apply_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_951_, v___y_1079_, v___y_1072_, v___x_1092_, v___y_1081_, v___y_1083_, v___y_1074_, v___y_1078_);
if (lean_obj_tag(v___x_1093_) == 0)
{
lean_object* v_a_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; 
v_a_1094_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_a_1094_);
lean_dec_ref_known(v___x_1093_, 1);
v___x_1095_ = lean_mk_empty_array_with_capacity(v___y_1075_);
v___x_1096_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg(v___y_1080_, v_sz_1086_, v___x_1087_, v___x_1095_, v___y_1081_, v___y_1083_, v___y_1074_, v___y_1078_);
lean_dec(v___y_1080_);
if (lean_obj_tag(v___x_1096_) == 0)
{
lean_object* v_a_1097_; lean_object* v___x_1098_; 
v_a_1097_ = lean_ctor_get(v___x_1096_, 0);
lean_inc(v_a_1097_);
lean_dec_ref_known(v___x_1096_, 1);
v___x_1098_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1098_, 0, v___y_1082_);
lean_ctor_set(v___x_1098_, 1, v___x_1090_);
lean_ctor_set(v___x_1098_, 2, v_a_1085_);
lean_ctor_set(v___x_1098_, 3, v___x_1091_);
lean_ctor_set(v___x_1098_, 4, v_a_1089_);
if (v___y_1072_ == 0)
{
v___y_1044_ = v___y_1072_;
v___y_1045_ = v_a_1094_;
v___y_1046_ = v___x_1098_;
v___y_1047_ = v___y_1075_;
v___y_1048_ = v___y_1077_;
v_htmls_1049_ = v_a_1097_;
v___y_1050_ = v___y_1076_;
v___y_1051_ = v___y_1073_;
v___y_1052_ = v___y_1081_;
v___y_1053_ = v___y_1083_;
v___y_1054_ = v___y_1074_;
v___y_1055_ = v___y_1078_;
goto v___jp_1043_;
}
else
{
lean_object* v___x_1099_; 
lean_dec(v_a_1097_);
lean_inc(v_a_1094_);
v___x_1099_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_addSolvedSuggestion(v_a_1094_, v___y_1076_, v___y_1073_, v___y_1081_, v___y_1083_, v___y_1074_, v___y_1078_);
if (lean_obj_tag(v___x_1099_) == 0)
{
lean_object* v___x_1100_; 
lean_dec_ref_known(v___x_1099_, 1);
v___x_1100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___closed__2));
v___y_1044_ = v___y_1072_;
v___y_1045_ = v_a_1094_;
v___y_1046_ = v___x_1098_;
v___y_1047_ = v___y_1075_;
v___y_1048_ = v___y_1077_;
v_htmls_1049_ = v___x_1100_;
v___y_1050_ = v___y_1076_;
v___y_1051_ = v___y_1073_;
v___y_1052_ = v___y_1081_;
v___y_1053_ = v___y_1083_;
v___y_1054_ = v___y_1074_;
v___y_1055_ = v___y_1078_;
goto v___jp_1043_;
}
else
{
lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1108_; 
lean_dec_ref_known(v___x_1098_, 5);
lean_dec(v_a_1094_);
lean_dec(v___y_1075_);
lean_dec_ref(v_lem_951_);
v_a_1101_ = lean_ctor_get(v___x_1099_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___x_1099_);
if (v_isSharedCheck_1108_ == 0)
{
v___x_1103_ = v___x_1099_;
v_isShared_1104_ = v_isSharedCheck_1108_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1099_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1108_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1106_; 
if (v_isShared_1104_ == 0)
{
v___x_1106_ = v___x_1103_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1107_; 
v_reuseFailAlloc_1107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1107_, 0, v_a_1101_);
v___x_1106_ = v_reuseFailAlloc_1107_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
return v___x_1106_;
}
}
}
}
}
else
{
lean_object* v_a_1109_; lean_object* v___x_1111_; uint8_t v_isShared_1112_; uint8_t v_isSharedCheck_1116_; 
lean_dec(v_a_1094_);
lean_dec_ref(v___x_1091_);
lean_dec(v___x_1090_);
lean_dec(v_a_1089_);
lean_dec(v_a_1085_);
lean_dec(v___y_1082_);
lean_dec(v___y_1075_);
lean_dec_ref(v_lem_951_);
v_a_1109_ = lean_ctor_get(v___x_1096_, 0);
v_isSharedCheck_1116_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1116_ == 0)
{
v___x_1111_ = v___x_1096_;
v_isShared_1112_ = v_isSharedCheck_1116_;
goto v_resetjp_1110_;
}
else
{
lean_inc(v_a_1109_);
lean_dec(v___x_1096_);
v___x_1111_ = lean_box(0);
v_isShared_1112_ = v_isSharedCheck_1116_;
goto v_resetjp_1110_;
}
v_resetjp_1110_:
{
lean_object* v___x_1114_; 
if (v_isShared_1112_ == 0)
{
v___x_1114_ = v___x_1111_;
goto v_reusejp_1113_;
}
else
{
lean_object* v_reuseFailAlloc_1115_; 
v_reuseFailAlloc_1115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1115_, 0, v_a_1109_);
v___x_1114_ = v_reuseFailAlloc_1115_;
goto v_reusejp_1113_;
}
v_reusejp_1113_:
{
return v___x_1114_;
}
}
}
}
else
{
lean_object* v_a_1117_; lean_object* v___x_1119_; uint8_t v_isShared_1120_; uint8_t v_isSharedCheck_1124_; 
lean_dec_ref(v___x_1091_);
lean_dec(v___x_1090_);
lean_dec(v_a_1089_);
lean_dec(v_a_1085_);
lean_dec(v___y_1082_);
lean_dec(v___y_1080_);
lean_dec(v___y_1075_);
lean_dec_ref(v_lem_951_);
v_a_1117_ = lean_ctor_get(v___x_1093_, 0);
v_isSharedCheck_1124_ = !lean_is_exclusive(v___x_1093_);
if (v_isSharedCheck_1124_ == 0)
{
v___x_1119_ = v___x_1093_;
v_isShared_1120_ = v_isSharedCheck_1124_;
goto v_resetjp_1118_;
}
else
{
lean_inc(v_a_1117_);
lean_dec(v___x_1093_);
v___x_1119_ = lean_box(0);
v_isShared_1120_ = v_isSharedCheck_1124_;
goto v_resetjp_1118_;
}
v_resetjp_1118_:
{
lean_object* v___x_1122_; 
if (v_isShared_1120_ == 0)
{
v___x_1122_ = v___x_1119_;
goto v_reusejp_1121_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v_a_1117_);
v___x_1122_ = v_reuseFailAlloc_1123_;
goto v_reusejp_1121_;
}
v_reusejp_1121_:
{
return v___x_1122_;
}
}
}
}
else
{
lean_object* v_a_1125_; lean_object* v___x_1127_; uint8_t v_isShared_1128_; uint8_t v_isSharedCheck_1132_; 
lean_dec(v_a_1085_);
lean_dec(v___y_1084_);
lean_dec(v___y_1082_);
lean_dec(v___y_1080_);
lean_dec_ref(v___y_1079_);
lean_dec(v___y_1075_);
lean_dec_ref(v_lem_951_);
v_a_1125_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1132_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1132_ == 0)
{
v___x_1127_ = v___x_1088_;
v_isShared_1128_ = v_isSharedCheck_1132_;
goto v_resetjp_1126_;
}
else
{
lean_inc(v_a_1125_);
lean_dec(v___x_1088_);
v___x_1127_ = lean_box(0);
v_isShared_1128_ = v_isSharedCheck_1132_;
goto v_resetjp_1126_;
}
v_resetjp_1126_:
{
lean_object* v___x_1130_; 
if (v_isShared_1128_ == 0)
{
v___x_1130_ = v___x_1127_;
goto v_reusejp_1129_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v_a_1125_);
v___x_1130_ = v_reuseFailAlloc_1131_;
goto v_reusejp_1129_;
}
v_reusejp_1129_:
{
return v___x_1130_;
}
}
}
}
v___jp_1133_:
{
if (lean_obj_tag(v___y_1147_) == 0)
{
lean_object* v_a_1148_; 
v_a_1148_ = lean_ctor_get(v___y_1147_, 0);
lean_inc(v_a_1148_);
lean_dec_ref_known(v___y_1147_, 1);
v___y_1072_ = v___y_1134_;
v___y_1073_ = v___y_1135_;
v___y_1074_ = v___y_1136_;
v___y_1075_ = v___y_1137_;
v___y_1076_ = v___y_1138_;
v___y_1077_ = v___y_1139_;
v___y_1078_ = v___y_1140_;
v___y_1079_ = v___y_1141_;
v___y_1080_ = v___y_1142_;
v___y_1081_ = v___y_1143_;
v___y_1082_ = v___y_1144_;
v___y_1083_ = v___y_1145_;
v___y_1084_ = v___y_1146_;
v_a_1085_ = v_a_1148_;
goto v___jp_1071_;
}
else
{
lean_object* v_a_1149_; lean_object* v___x_1151_; uint8_t v_isShared_1152_; uint8_t v_isSharedCheck_1156_; 
lean_dec(v___y_1146_);
lean_dec(v___y_1144_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
lean_dec(v___y_1137_);
lean_dec_ref(v_lem_951_);
v_a_1149_ = lean_ctor_get(v___y_1147_, 0);
v_isSharedCheck_1156_ = !lean_is_exclusive(v___y_1147_);
if (v_isSharedCheck_1156_ == 0)
{
v___x_1151_ = v___y_1147_;
v_isShared_1152_ = v_isSharedCheck_1156_;
goto v_resetjp_1150_;
}
else
{
lean_inc(v_a_1149_);
lean_dec(v___y_1147_);
v___x_1151_ = lean_box(0);
v_isShared_1152_ = v_isSharedCheck_1156_;
goto v_resetjp_1150_;
}
v_resetjp_1150_:
{
lean_object* v___x_1154_; 
if (v_isShared_1152_ == 0)
{
v___x_1154_ = v___x_1151_;
goto v_reusejp_1153_;
}
else
{
lean_object* v_reuseFailAlloc_1155_; 
v_reuseFailAlloc_1155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1155_, 0, v_a_1149_);
v___x_1154_ = v_reuseFailAlloc_1155_;
goto v_reusejp_1153_;
}
v_reusejp_1153_:
{
return v___x_1154_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try___boxed(lean_object* v_lem_1338_, lean_object* v_assignableMVars_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_){
_start:
{
lean_object* v_res_1347_; 
v_res_1347_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyLemma_try(v_lem_1338_, v_assignableMVars_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_);
lean_dec(v_a_1345_);
lean_dec_ref(v_a_1344_);
lean_dec(v_a_1343_);
lean_dec_ref(v_a_1342_);
lean_dec(v_a_1341_);
lean_dec_ref(v_a_1340_);
lean_dec_ref(v_assignableMVars_1339_);
return v_res_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0(lean_object* v_mvarId_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_){
_start:
{
lean_object* v___x_1356_; 
v___x_1356_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___redArg(v_mvarId_1348_, v___y_1352_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0___boxed(lean_object* v_mvarId_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__0(v_mvarId_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
lean_dec(v___y_1361_);
lean_dec_ref(v___y_1360_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
lean_dec(v_mvarId_1357_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3(lean_object* v_mvarId_1366_, lean_object* v_val_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_){
_start:
{
lean_object* v___x_1375_; 
v___x_1375_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___redArg(v_mvarId_1366_, v_val_1367_, v___y_1371_);
return v___x_1375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3___boxed(lean_object* v_mvarId_1376_, lean_object* v_val_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_){
_start:
{
lean_object* v_res_1385_; 
v_res_1385_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__3(v_mvarId_1376_, v_val_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_, v___y_1382_, v___y_1383_);
lean_dec(v___y_1383_);
lean_dec_ref(v___y_1382_);
lean_dec(v___y_1381_);
lean_dec_ref(v___y_1380_);
lean_dec(v___y_1379_);
lean_dec_ref(v___y_1378_);
return v_res_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4(lean_object* v_as_1386_, lean_object* v_as_x27_1387_, lean_object* v_b_1388_, lean_object* v_a_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v___x_1397_; 
v___x_1397_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___redArg(v_as_x27_1387_, v_b_1388_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_);
return v___x_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4___boxed(lean_object* v_as_1398_, lean_object* v_as_x27_1399_, lean_object* v_b_1400_, lean_object* v_a_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_){
_start:
{
lean_object* v_res_1409_; 
v_res_1409_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__4(v_as_1398_, v_as_x27_1399_, v_b_1400_, v_a_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
lean_dec(v___y_1407_);
lean_dec_ref(v___y_1406_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec(v_as_x27_1399_);
lean_dec(v_as_1398_);
return v_res_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5(size_t v_sz_1410_, size_t v_i_1411_, lean_object* v_bs_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_){
_start:
{
lean_object* v___x_1420_; 
v___x_1420_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___redArg(v_sz_1410_, v_i_1411_, v_bs_1412_, v___y_1415_, v___y_1416_, v___y_1417_, v___y_1418_);
return v___x_1420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5___boxed(lean_object* v_sz_1421_, lean_object* v_i_1422_, lean_object* v_bs_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_){
_start:
{
size_t v_sz_boxed_1431_; size_t v_i_boxed_1432_; lean_object* v_res_1433_; 
v_sz_boxed_1431_ = lean_unbox_usize(v_sz_1421_);
lean_dec(v_sz_1421_);
v_i_boxed_1432_ = lean_unbox_usize(v_i_1422_);
lean_dec(v_i_1422_);
v_res_1433_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__5(v_sz_boxed_1431_, v_i_boxed_1432_, v_bs_1423_, v___y_1424_, v___y_1425_, v___y_1426_, v___y_1427_, v___y_1428_, v___y_1429_);
lean_dec(v___y_1429_);
lean_dec_ref(v___y_1428_);
lean_dec(v___y_1427_);
lean_dec_ref(v___y_1426_);
lean_dec(v___y_1425_);
lean_dec_ref(v___y_1424_);
return v_res_1433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6(lean_object* v_as_1434_, size_t v_sz_1435_, size_t v_i_1436_, lean_object* v_b_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_){
_start:
{
lean_object* v___x_1445_; 
v___x_1445_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___redArg(v_as_1434_, v_sz_1435_, v_i_1436_, v_b_1437_, v___y_1440_, v___y_1441_, v___y_1442_, v___y_1443_);
return v___x_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6___boxed(lean_object* v_as_1446_, lean_object* v_sz_1447_, lean_object* v_i_1448_, lean_object* v_b_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_){
_start:
{
size_t v_sz_boxed_1457_; size_t v_i_boxed_1458_; lean_object* v_res_1459_; 
v_sz_boxed_1457_ = lean_unbox_usize(v_sz_1447_);
lean_dec(v_sz_1447_);
v_i_boxed_1458_ = lean_unbox_usize(v_i_1448_);
lean_dec(v_i_1448_);
v_res_1459_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__6(v_as_1446_, v_sz_boxed_1457_, v_i_boxed_1458_, v_b_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_);
lean_dec(v___y_1455_);
lean_dec_ref(v___y_1454_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
lean_dec(v___y_1451_);
lean_dec_ref(v___y_1450_);
lean_dec_ref(v_as_1446_);
return v_res_1459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7(lean_object* v_as_1460_, size_t v_i_1461_, size_t v_stop_1462_, lean_object* v_b_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_){
_start:
{
lean_object* v___x_1471_; 
v___x_1471_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___redArg(v_as_1460_, v_i_1461_, v_stop_1462_, v_b_1463_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
return v___x_1471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7___boxed(lean_object* v_as_1472_, lean_object* v_i_1473_, lean_object* v_stop_1474_, lean_object* v_b_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
size_t v_i_boxed_1483_; size_t v_stop_boxed_1484_; lean_object* v_res_1485_; 
v_i_boxed_1483_ = lean_unbox_usize(v_i_1473_);
lean_dec(v_i_1473_);
v_stop_boxed_1484_ = lean_unbox_usize(v_stop_1474_);
lean_dec(v_stop_1474_);
v_res_1485_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__7(v_as_1472_, v_i_boxed_1483_, v_stop_boxed_1484_, v_b_1475_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v___y_1477_);
lean_dec_ref(v___y_1476_);
lean_dec_ref(v_as_1472_);
return v_res_1485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9(lean_object* v_00_u03b1_1486_, lean_object* v_msg_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_){
_start:
{
lean_object* v___x_1495_; 
v___x_1495_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___redArg(v_msg_1487_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9___boxed(lean_object* v_00_u03b1_1496_, lean_object* v_msg_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_){
_start:
{
lean_object* v_res_1505_; 
v_res_1505_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyLemma_try_spec__9(v_00_u03b1_1496_, v_msg_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_, v___y_1503_);
lean_dec(v___y_1503_);
lean_dec_ref(v___y_1502_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
return v_res_1505_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(uint8_t builtin) {
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
res = runtime_initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Apply(builtin);
}
#ifdef __cplusplus
}
#endif
