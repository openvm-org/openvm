// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.ApplyAt
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.SectionState public import Mathlib.Tactic.ApplyAt public meta import Mathlib.Tactic.ClickSuggestions.Util
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_isExplicitEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedApplyAtKey_default___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tacticApply_At_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(233, 71, 158, 146, 216, 26, 31, 241)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strong"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "goal-vdash"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__4_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__3_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__5_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__6_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__6_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⊢ "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__9_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__9_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__2_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__7_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__10_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__0_value),LEAN_SCALAR_PTR_LITERAL(15, 73, 51, 51, 21, 209, 204, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " does not unify with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0(lean_object* v_a_10_, lean_object* v_b_11_){
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0___boxed(lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
uint8_t v_res_35_; lean_object* v_r_36_; 
v_res_35_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdApplyAtKey___lam__0(v_a_33_, v_b_34_);
lean_dec_ref(v_b_34_);
lean_dec_ref(v_a_33_);
v_r_36_ = lean_box(v_res_35_);
return v_r_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg(lean_object* v___x_39_, lean_object* v___x_40_, lean_object* v_n_41_, lean_object* v_i_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg___boxed(lean_object* v___x_75_, lean_object* v___x_76_, lean_object* v_n_77_, lean_object* v_i_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg(v___x_75_, v___x_76_, v_n_77_, v_i_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate(lean_object* v_a_85_, lean_object* v_b_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_){
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
v___x_99_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg(v_newGoals_92_, v_newGoals_93_, v___x_94_, v___x_94_, v_a_87_, v_a_88_, v_a_89_, v_a_90_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate___boxed(lean_object* v_a_100_, lean_object* v_b_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate(v_a_100_, v_b_101_, v_a_102_, v_a_103_, v_a_104_, v_a_105_);
lean_dec(v_a_105_);
lean_dec_ref(v_a_104_);
lean_dec(v_a_103_);
lean_dec_ref(v_a_102_);
lean_dec_ref(v_b_101_);
lean_dec_ref(v_a_100_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0(lean_object* v___x_108_, lean_object* v___x_109_, lean_object* v_n_110_, lean_object* v_i_111_, lean_object* v_a_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___redArg(v___x_108_, v___x_109_, v_n_110_, v_i_111_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0___boxed(lean_object* v___x_119_, lean_object* v___x_120_, lean_object* v_n_121_, lean_object* v_i_122_, lean_object* v_a_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib___private_Init_Data_Nat_Control_0__Nat_allM_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtKey_isDuplicate_spec__0(v___x_119_, v___x_120_, v_n_121_, v_i_122_, v_a_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(lean_object* v_lem_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_unresolveName(v_lem_139_, v_a_142_, v_a_143_, v_a_144_, v_a_145_);
if (lean_obj_tag(v___x_147_) == 0)
{
lean_object* v_a_148_; lean_object* v___x_149_; 
v_a_148_ = lean_ctor_get(v___x_147_, 0);
lean_inc(v_a_148_);
lean_dec_ref_known(v___x_147_, 1);
v___x_149_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_getHypIdent_x21(v_a_140_, v_a_141_, v_a_142_, v_a_143_, v_a_144_, v_a_145_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_167_; 
v_a_150_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_167_ == 0)
{
v___x_152_ = v___x_149_;
v_isShared_153_ = v_isSharedCheck_167_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_a_150_);
lean_dec(v___x_149_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_167_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v_ref_154_; uint8_t v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_165_; 
v_ref_154_ = lean_ctor_get(v_a_144_, 5);
v___x_155_ = 0;
v___x_156_ = l_Lean_SourceInfo_fromRef(v_ref_154_, v___x_155_);
v___x_157_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__3));
v___x_158_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__4));
lean_inc_n(v___x_156_, 2);
v___x_159_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_156_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = l_Lean_mkIdent(v_a_148_);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___closed__5));
v___x_162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_156_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = l_Lean_Syntax_node4(v___x_156_, v___x_157_, v___x_159_, v___x_160_, v___x_162_, v_a_150_);
if (v_isShared_153_ == 0)
{
lean_ctor_set(v___x_152_, 0, v___x_163_);
v___x_165_ = v___x_152_;
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
else
{
lean_dec(v_a_148_);
return v___x_149_;
}
}
else
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_175_; 
v_a_168_ = lean_ctor_get(v___x_147_, 0);
v_isSharedCheck_175_ = !lean_is_exclusive(v___x_147_);
if (v_isSharedCheck_175_ == 0)
{
v___x_170_ = v___x_147_;
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_147_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_173_; 
if (v_isShared_171_ == 0)
{
v___x_173_ = v___x_170_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v_a_168_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax___boxed(lean_object* v_lem_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_176_, v_a_177_, v_a_178_, v_a_179_, v_a_180_, v_a_181_, v_a_182_);
lean_dec(v_a_182_);
lean_dec_ref(v_a_181_);
lean_dec(v_a_180_);
lean_dec_ref(v_a_179_);
lean_dec(v_a_178_);
lean_dec_ref(v_a_177_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(lean_object* v_e_185_, lean_object* v___y_186_){
_start:
{
uint8_t v___x_188_; 
v___x_188_ = l_Lean_Expr_hasMVar(v_e_185_);
if (v___x_188_ == 0)
{
lean_object* v___x_189_; 
v___x_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_189_, 0, v_e_185_);
return v___x_189_;
}
else
{
lean_object* v___x_190_; lean_object* v_mctx_191_; lean_object* v___x_192_; lean_object* v_fst_193_; lean_object* v_snd_194_; lean_object* v___x_195_; lean_object* v_cache_196_; lean_object* v_zetaDeltaFVarIds_197_; lean_object* v_postponed_198_; lean_object* v_diag_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_208_; 
v___x_190_ = lean_st_ref_get(v___y_186_);
v_mctx_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc_ref(v_mctx_191_);
lean_dec(v___x_190_);
v___x_192_ = l_Lean_instantiateMVarsCore(v_mctx_191_, v_e_185_);
v_fst_193_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_fst_193_);
v_snd_194_ = lean_ctor_get(v___x_192_, 1);
lean_inc(v_snd_194_);
lean_dec_ref(v___x_192_);
v___x_195_ = lean_st_ref_take(v___y_186_);
v_cache_196_ = lean_ctor_get(v___x_195_, 1);
v_zetaDeltaFVarIds_197_ = lean_ctor_get(v___x_195_, 2);
v_postponed_198_ = lean_ctor_get(v___x_195_, 3);
v_diag_199_ = lean_ctor_get(v___x_195_, 4);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_195_);
if (v_isSharedCheck_208_ == 0)
{
lean_object* v_unused_209_; 
v_unused_209_ = lean_ctor_get(v___x_195_, 0);
lean_dec(v_unused_209_);
v___x_201_ = v___x_195_;
v_isShared_202_ = v_isSharedCheck_208_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_diag_199_);
lean_inc(v_postponed_198_);
lean_inc(v_zetaDeltaFVarIds_197_);
lean_inc(v_cache_196_);
lean_dec(v___x_195_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_208_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_204_; 
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 0, v_snd_194_);
v___x_204_ = v___x_201_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_snd_194_);
lean_ctor_set(v_reuseFailAlloc_207_, 1, v_cache_196_);
lean_ctor_set(v_reuseFailAlloc_207_, 2, v_zetaDeltaFVarIds_197_);
lean_ctor_set(v_reuseFailAlloc_207_, 3, v_postponed_198_);
lean_ctor_set(v_reuseFailAlloc_207_, 4, v_diag_199_);
v___x_204_ = v_reuseFailAlloc_207_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = lean_st_ref_set(v___y_186_, v___x_204_);
v___x_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_206_, 0, v_fst_193_);
return v___x_206_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg___boxed(lean_object* v_e_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(v_e_210_, v___y_211_);
lean_dec(v___y_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1(lean_object* v_e_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(v_e_214_, v___y_218_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___boxed(lean_object* v_e_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1(v_e_223_, v___y_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_);
lean_dec(v___y_229_);
lean_dec_ref(v___y_228_);
lean_dec(v___y_227_);
lean_dec_ref(v___y_226_);
lean_dec(v___y_225_);
lean_dec_ref(v___y_224_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__9(lean_object* v_msg_232_){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_233_ = lean_box(0);
v___x_234_ = lean_panic_fn_borrowed(v___x_233_, v_msg_232_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg(size_t v_sz_235_, size_t v_i_236_, lean_object* v_bs_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
uint8_t v___x_243_; 
v___x_243_ = lean_usize_dec_lt(v_i_236_, v_sz_235_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; 
v___x_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_244_, 0, v_bs_237_);
return v___x_244_;
}
else
{
lean_object* v_v_245_; lean_object* v___x_246_; 
v_v_245_ = lean_array_uget_borrowed(v_bs_237_, v_i_236_);
lean_inc(v_v_245_);
v___x_246_ = l_Lean_Meta_abstractMVars(v_v_245_, v___x_243_, v___y_238_, v___y_239_, v___y_240_, v___y_241_);
if (lean_obj_tag(v___x_246_) == 0)
{
lean_object* v_a_247_; lean_object* v___x_248_; lean_object* v_bs_x27_249_; size_t v___x_250_; size_t v___x_251_; lean_object* v___x_252_; 
v_a_247_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_a_247_);
lean_dec_ref_known(v___x_246_, 1);
v___x_248_ = lean_unsigned_to_nat(0u);
v_bs_x27_249_ = lean_array_uset(v_bs_237_, v_i_236_, v___x_248_);
v___x_250_ = ((size_t)1ULL);
v___x_251_ = lean_usize_add(v_i_236_, v___x_250_);
v___x_252_ = lean_array_uset(v_bs_x27_249_, v_i_236_, v_a_247_);
v_i_236_ = v___x_251_;
v_bs_237_ = v___x_252_;
goto _start;
}
else
{
lean_object* v_a_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_261_; 
lean_dec_ref(v_bs_237_);
v_a_254_ = lean_ctor_get(v___x_246_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_246_);
if (v_isSharedCheck_261_ == 0)
{
v___x_256_ = v___x_246_;
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_a_254_);
lean_dec(v___x_246_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg___boxed(lean_object* v_sz_262_, lean_object* v_i_263_, lean_object* v_bs_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
size_t v_sz_boxed_270_; size_t v_i_boxed_271_; lean_object* v_res_272_; 
v_sz_boxed_270_ = lean_unbox_usize(v_sz_262_);
lean_dec(v_sz_262_);
v_i_boxed_271_ = lean_unbox_usize(v_i_263_);
lean_dec(v_i_263_);
v_res_272_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg(v_sz_boxed_270_, v_i_boxed_271_, v_bs_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3(size_t v_sz_273_, size_t v_i_274_, lean_object* v_bs_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
uint8_t v___x_283_; 
v___x_283_ = lean_usize_dec_lt(v_i_274_, v_sz_273_);
if (v___x_283_ == 0)
{
lean_object* v___x_284_; 
v___x_284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_284_, 0, v_bs_275_);
return v___x_284_;
}
else
{
lean_object* v_v_285_; lean_object* v___x_286_; lean_object* v_bs_x27_287_; lean_object* v___y_289_; lean_object* v___x_303_; 
v_v_285_ = lean_array_uget(v_bs_275_, v_i_274_);
v___x_286_ = lean_unsigned_to_nat(0u);
v_bs_x27_287_ = lean_array_uset(v_bs_275_, v_i_274_, v___x_286_);
v___x_303_ = l_Lean_MVarId_getType(v_v_285_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
if (lean_obj_tag(v___x_303_) == 0)
{
lean_object* v_a_304_; lean_object* v___x_305_; 
v_a_304_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_a_304_);
lean_dec_ref_known(v___x_303_, 1);
v___x_305_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(v_a_304_, v___y_279_);
v___y_289_ = v___x_305_;
goto v___jp_288_;
}
else
{
v___y_289_ = v___x_303_;
goto v___jp_288_;
}
v___jp_288_:
{
if (lean_obj_tag(v___y_289_) == 0)
{
lean_object* v_a_290_; size_t v___x_291_; size_t v___x_292_; lean_object* v___x_293_; 
v_a_290_ = lean_ctor_get(v___y_289_, 0);
lean_inc(v_a_290_);
lean_dec_ref_known(v___y_289_, 1);
v___x_291_ = ((size_t)1ULL);
v___x_292_ = lean_usize_add(v_i_274_, v___x_291_);
v___x_293_ = lean_array_uset(v_bs_x27_287_, v_i_274_, v_a_290_);
v_i_274_ = v___x_292_;
v_bs_275_ = v___x_293_;
goto _start;
}
else
{
lean_object* v_a_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_302_; 
lean_dec_ref(v_bs_x27_287_);
v_a_295_ = lean_ctor_get(v___y_289_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___y_289_);
if (v_isSharedCheck_302_ == 0)
{
v___x_297_ = v___y_289_;
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
else
{
lean_inc(v_a_295_);
lean_dec(v___y_289_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v___x_300_; 
if (v_isShared_298_ == 0)
{
v___x_300_ = v___x_297_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_a_295_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3___boxed(lean_object* v_sz_306_, lean_object* v_i_307_, lean_object* v_bs_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_){
_start:
{
size_t v_sz_boxed_316_; size_t v_i_boxed_317_; lean_object* v_res_318_; 
v_sz_boxed_316_ = lean_unbox_usize(v_sz_306_);
lean_dec(v_sz_306_);
v_i_boxed_317_ = lean_unbox_usize(v_i_307_);
lean_dec(v_i_307_);
v_res_318_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3(v_sz_boxed_316_, v_i_boxed_317_, v_bs_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
lean_dec(v___y_314_);
lean_dec_ref(v___y_313_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9(lean_object* v_msgData_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v___x_325_; lean_object* v_env_326_; lean_object* v___x_327_; lean_object* v_mctx_328_; lean_object* v_lctx_329_; lean_object* v_options_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_325_ = lean_st_ref_get(v___y_323_);
v_env_326_ = lean_ctor_get(v___x_325_, 0);
lean_inc_ref(v_env_326_);
lean_dec(v___x_325_);
v___x_327_ = lean_st_ref_get(v___y_321_);
v_mctx_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc_ref(v_mctx_328_);
lean_dec(v___x_327_);
v_lctx_329_ = lean_ctor_get(v___y_320_, 2);
v_options_330_ = lean_ctor_get(v___y_322_, 2);
lean_inc_ref(v_options_330_);
lean_inc_ref(v_lctx_329_);
v___x_331_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_331_, 0, v_env_326_);
lean_ctor_set(v___x_331_, 1, v_mctx_328_);
lean_ctor_set(v___x_331_, 2, v_lctx_329_);
lean_ctor_set(v___x_331_, 3, v_options_330_);
v___x_332_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
lean_ctor_set(v___x_332_, 1, v_msgData_319_);
v___x_333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9___boxed(lean_object* v_msgData_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9(v_msgData_334_, v___y_335_, v___y_336_, v___y_337_, v___y_338_);
lean_dec(v___y_338_);
lean_dec_ref(v___y_337_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg(lean_object* v_msg_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_){
_start:
{
lean_object* v_ref_347_; lean_object* v___x_348_; lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_357_; 
v_ref_347_ = lean_ctor_get(v___y_344_, 5);
v___x_348_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8_spec__9(v_msg_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_);
v_a_349_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_357_ == 0)
{
v___x_351_ = v___x_348_;
v_isShared_352_ = v_isSharedCheck_357_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_348_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_357_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___x_353_; lean_object* v___x_355_; 
lean_inc(v_ref_347_);
v___x_353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_353_, 0, v_ref_347_);
lean_ctor_set(v___x_353_, 1, v_a_349_);
if (v_isShared_352_ == 0)
{
lean_ctor_set_tag(v___x_351_, 1);
lean_ctor_set(v___x_351_, 0, v___x_353_);
v___x_355_ = v___x_351_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_353_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg___boxed(lean_object* v_msg_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg(v_msg_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
lean_dec(v___y_362_);
lean_dec_ref(v___y_361_);
lean_dec(v___y_360_);
lean_dec_ref(v___y_359_);
return v_res_364_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg(lean_object* v_keys_365_, lean_object* v_i_366_, lean_object* v_k_367_){
_start:
{
lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_368_ = lean_array_get_size(v_keys_365_);
v___x_369_ = lean_nat_dec_lt(v_i_366_, v___x_368_);
if (v___x_369_ == 0)
{
lean_dec(v_i_366_);
return v___x_369_;
}
else
{
lean_object* v_k_x27_370_; uint8_t v___x_371_; 
v_k_x27_370_ = lean_array_fget_borrowed(v_keys_365_, v_i_366_);
v___x_371_ = l_Lean_instBEqMVarId_beq(v_k_367_, v_k_x27_370_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_372_ = lean_unsigned_to_nat(1u);
v___x_373_ = lean_nat_add(v_i_366_, v___x_372_);
lean_dec(v_i_366_);
v_i_366_ = v___x_373_;
goto _start;
}
else
{
lean_dec(v_i_366_);
return v___x_371_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg___boxed(lean_object* v_keys_375_, lean_object* v_i_376_, lean_object* v_k_377_){
_start:
{
uint8_t v_res_378_; lean_object* v_r_379_; 
v_res_378_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg(v_keys_375_, v_i_376_, v_k_377_);
lean_dec(v_k_377_);
lean_dec_ref(v_keys_375_);
v_r_379_ = lean_box(v_res_378_);
return v_r_379_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg(lean_object* v_x_380_, size_t v_x_381_, lean_object* v_x_382_){
_start:
{
if (lean_obj_tag(v_x_380_) == 0)
{
lean_object* v_es_383_; lean_object* v___x_384_; size_t v___x_385_; size_t v___x_386_; lean_object* v_j_387_; lean_object* v___x_388_; 
v_es_383_ = lean_ctor_get(v_x_380_, 0);
v___x_384_ = lean_box(2);
v___x_385_ = ((size_t)31ULL);
v___x_386_ = lean_usize_land(v_x_381_, v___x_385_);
v_j_387_ = lean_usize_to_nat(v___x_386_);
v___x_388_ = lean_array_get_borrowed(v___x_384_, v_es_383_, v_j_387_);
lean_dec(v_j_387_);
switch(lean_obj_tag(v___x_388_))
{
case 0:
{
lean_object* v_key_389_; uint8_t v___x_390_; 
v_key_389_ = lean_ctor_get(v___x_388_, 0);
v___x_390_ = l_Lean_instBEqMVarId_beq(v_x_382_, v_key_389_);
return v___x_390_;
}
case 1:
{
lean_object* v_node_391_; size_t v___x_392_; size_t v___x_393_; 
v_node_391_ = lean_ctor_get(v___x_388_, 0);
v___x_392_ = ((size_t)5ULL);
v___x_393_ = lean_usize_shift_right(v_x_381_, v___x_392_);
v_x_380_ = v_node_391_;
v_x_381_ = v___x_393_;
goto _start;
}
default: 
{
uint8_t v___x_395_; 
v___x_395_ = 0;
return v___x_395_;
}
}
}
else
{
lean_object* v_ks_396_; lean_object* v___x_397_; uint8_t v___x_398_; 
v_ks_396_ = lean_ctor_get(v_x_380_, 0);
v___x_397_ = lean_unsigned_to_nat(0u);
v___x_398_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg(v_ks_396_, v___x_397_, v_x_382_);
return v___x_398_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_x_399_, lean_object* v_x_400_, lean_object* v_x_401_){
_start:
{
size_t v_x_44950__boxed_402_; uint8_t v_res_403_; lean_object* v_r_404_; 
v_x_44950__boxed_402_ = lean_unbox_usize(v_x_400_);
lean_dec(v_x_400_);
v_res_403_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg(v_x_399_, v_x_44950__boxed_402_, v_x_401_);
lean_dec(v_x_401_);
lean_dec_ref(v_x_399_);
v_r_404_ = lean_box(v_res_403_);
return v_r_404_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg(lean_object* v_x_405_, lean_object* v_x_406_){
_start:
{
uint64_t v___x_407_; size_t v___x_408_; uint8_t v___x_409_; 
v___x_407_ = l_Lean_instHashableMVarId_hash(v_x_406_);
v___x_408_ = lean_uint64_to_usize(v___x_407_);
v___x_409_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg(v_x_405_, v___x_408_, v_x_406_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg___boxed(lean_object* v_x_410_, lean_object* v_x_411_){
_start:
{
uint8_t v_res_412_; lean_object* v_r_413_; 
v_res_412_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg(v_x_410_, v_x_411_);
lean_dec(v_x_411_);
lean_dec_ref(v_x_410_);
v_r_413_ = lean_box(v_res_412_);
return v_r_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg(lean_object* v_mvarId_414_, lean_object* v___y_415_){
_start:
{
lean_object* v___x_417_; lean_object* v_mctx_418_; lean_object* v_eAssignment_419_; uint8_t v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_417_ = lean_st_ref_get(v___y_415_);
v_mctx_418_ = lean_ctor_get(v___x_417_, 0);
lean_inc_ref(v_mctx_418_);
lean_dec(v___x_417_);
v_eAssignment_419_ = lean_ctor_get(v_mctx_418_, 8);
lean_inc_ref(v_eAssignment_419_);
lean_dec_ref(v_mctx_418_);
v___x_420_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg(v_eAssignment_419_, v_mvarId_414_);
lean_dec_ref(v_eAssignment_419_);
v___x_421_ = lean_box(v___x_420_);
v___x_422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg___boxed(lean_object* v_mvarId_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg(v_mvarId_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec(v_mvarId_423_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7(lean_object* v_as_427_, size_t v_i_428_, size_t v_stop_429_, lean_object* v_b_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_){
_start:
{
lean_object* v_a_439_; uint8_t v___x_443_; 
v___x_443_ = lean_usize_dec_eq(v_i_428_, v_stop_429_);
if (v___x_443_ == 0)
{
lean_object* v___x_444_; lean_object* v___x_447_; 
v___x_444_ = lean_array_uget_borrowed(v_as_427_, v_i_428_);
v___x_447_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg(v___x_444_, v___y_434_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v_a_448_; uint8_t v___x_449_; 
v_a_448_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_448_);
lean_dec_ref_known(v___x_447_, 1);
v___x_449_ = lean_unbox(v_a_448_);
lean_dec(v_a_448_);
if (v___x_449_ == 0)
{
goto v___jp_445_;
}
else
{
v_a_439_ = v_b_430_;
goto v___jp_438_;
}
}
else
{
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v_a_450_; uint8_t v___x_451_; 
v_a_450_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_447_, 1);
v___x_451_ = lean_unbox(v_a_450_);
lean_dec(v_a_450_);
if (v___x_451_ == 0)
{
v_a_439_ = v_b_430_;
goto v___jp_438_;
}
else
{
goto v___jp_445_;
}
}
else
{
lean_object* v_a_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_459_; 
lean_dec_ref(v_b_430_);
v_a_452_ = lean_ctor_get(v___x_447_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_447_);
if (v_isSharedCheck_459_ == 0)
{
v___x_454_ = v___x_447_;
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_a_452_);
lean_dec(v___x_447_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_457_; 
if (v_isShared_455_ == 0)
{
v___x_457_ = v___x_454_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v_a_452_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
v___jp_445_:
{
lean_object* v___x_446_; 
lean_inc(v___x_444_);
v___x_446_ = lean_array_push(v_b_430_, v___x_444_);
v_a_439_ = v___x_446_;
goto v___jp_438_;
}
}
else
{
lean_object* v___x_460_; 
v___x_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_460_, 0, v_b_430_);
return v___x_460_;
}
v___jp_438_:
{
size_t v___x_440_; size_t v___x_441_; 
v___x_440_ = ((size_t)1ULL);
v___x_441_ = lean_usize_add(v_i_428_, v___x_440_);
v_i_428_ = v___x_441_;
v_b_430_ = v_a_439_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7___boxed(lean_object* v_as_461_, lean_object* v_i_462_, lean_object* v_stop_463_, lean_object* v_b_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_){
_start:
{
size_t v_i_boxed_472_; size_t v_stop_boxed_473_; lean_object* v_res_474_; 
v_i_boxed_472_ = lean_unbox_usize(v_i_462_);
lean_dec(v_i_462_);
v_stop_boxed_473_ = lean_unbox_usize(v_stop_463_);
lean_dec(v_stop_463_);
v_res_474_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7(v_as_461_, v_i_boxed_472_, v_stop_boxed_473_, v_b_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec_ref(v_as_461_);
return v_res_474_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12(void){
_start:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_501_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__11));
v___x_502_ = lean_unsigned_to_nat(2u);
v___x_503_ = lean_mk_empty_array_with_capacity(v___x_502_);
v___x_504_ = lean_array_push(v___x_503_, v___x_501_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg(lean_object* v_as_505_, size_t v_sz_506_, size_t v_i_507_, lean_object* v_b_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
uint8_t v___x_514_; 
v___x_514_ = lean_usize_dec_lt(v_i_507_, v_sz_506_);
if (v___x_514_ == 0)
{
lean_object* v___x_515_; 
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v_b_508_);
return v___x_515_;
}
else
{
lean_object* v_a_516_; lean_object* v___x_517_; 
v_a_516_ = lean_array_uget_borrowed(v_as_505_, v_i_507_);
lean_inc(v_a_516_);
v___x_517_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_a_516_, v___y_509_, v___y_510_, v___y_511_, v___y_512_);
if (lean_obj_tag(v___x_517_) == 0)
{
lean_object* v_a_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; size_t v___x_525_; size_t v___x_526_; 
v_a_518_ = lean_ctor_get(v___x_517_, 0);
lean_inc(v_a_518_);
lean_dec_ref_known(v___x_517_, 1);
v___x_519_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0));
v___x_520_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__1));
v___x_521_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__12);
v___x_522_ = lean_array_push(v___x_521_, v_a_518_);
v___x_523_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_523_, 0, v___x_519_);
lean_ctor_set(v___x_523_, 1, v___x_520_);
lean_ctor_set(v___x_523_, 2, v___x_522_);
v___x_524_ = lean_array_push(v_b_508_, v___x_523_);
v___x_525_ = ((size_t)1ULL);
v___x_526_ = lean_usize_add(v_i_507_, v___x_525_);
v_i_507_ = v___x_526_;
v_b_508_ = v___x_524_;
goto _start;
}
else
{
lean_object* v_a_528_; lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_535_; 
lean_dec_ref(v_b_508_);
v_a_528_ = lean_ctor_get(v___x_517_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_517_);
if (v_isSharedCheck_535_ == 0)
{
v___x_530_ = v___x_517_;
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
else
{
lean_inc(v_a_528_);
lean_dec(v___x_517_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_533_; 
if (v_isShared_531_ == 0)
{
v___x_533_ = v___x_530_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_a_528_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___boxed(lean_object* v_as_536_, lean_object* v_sz_537_, lean_object* v_i_538_, lean_object* v_b_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_){
_start:
{
size_t v_sz_boxed_545_; size_t v_i_boxed_546_; lean_object* v_res_547_; 
v_sz_boxed_545_ = lean_unbox_usize(v_sz_537_);
lean_dec(v_sz_537_);
v_i_boxed_546_ = lean_unbox_usize(v_i_538_);
lean_dec(v_i_538_);
v_res_547_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg(v_as_536_, v_sz_boxed_545_, v_i_boxed_546_, v_b_539_, v___y_540_, v___y_541_, v___y_542_, v___y_543_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v_as_536_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(lean_object* v_as_548_, size_t v_i_549_, size_t v_stop_550_, lean_object* v_b_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
uint8_t v___x_557_; 
v___x_557_ = lean_usize_dec_eq(v_i_549_, v_stop_550_);
if (v___x_557_ == 0)
{
lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_558_ = lean_array_uget_borrowed(v_as_548_, v_i_549_);
lean_inc(v___x_558_);
v___x_559_ = l_Lean_Meta_ppExpr(v___x_558_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
if (lean_obj_tag(v___x_559_) == 0)
{
lean_object* v_a_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; size_t v___x_566_; size_t v___x_567_; 
v_a_560_ = lean_ctor_get(v___x_559_, 0);
lean_inc(v_a_560_);
lean_dec_ref_known(v___x_559_, 1);
v___x_561_ = lean_unsigned_to_nat(0u);
v___x_562_ = l_Std_Format_defWidth;
v___x_563_ = l_Std_Format_pretty(v_a_560_, v___x_562_, v___x_561_, v___x_561_);
v___x_564_ = lean_string_length(v___x_563_);
lean_dec_ref(v___x_563_);
v___x_565_ = lean_nat_add(v___x_564_, v_b_551_);
lean_dec(v_b_551_);
v___x_566_ = ((size_t)1ULL);
v___x_567_ = lean_usize_add(v_i_549_, v___x_566_);
v_i_549_ = v___x_567_;
v_b_551_ = v___x_565_;
goto _start;
}
else
{
lean_object* v_a_569_; lean_object* v___x_571_; uint8_t v_isShared_572_; uint8_t v_isSharedCheck_576_; 
lean_dec(v_b_551_);
v_a_569_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_576_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_576_ == 0)
{
v___x_571_ = v___x_559_;
v_isShared_572_ = v_isSharedCheck_576_;
goto v_resetjp_570_;
}
else
{
lean_inc(v_a_569_);
lean_dec(v___x_559_);
v___x_571_ = lean_box(0);
v_isShared_572_ = v_isSharedCheck_576_;
goto v_resetjp_570_;
}
v_resetjp_570_:
{
lean_object* v___x_574_; 
if (v_isShared_572_ == 0)
{
v___x_574_ = v___x_571_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v_a_569_);
v___x_574_ = v_reuseFailAlloc_575_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
return v___x_574_;
}
}
}
}
else
{
lean_object* v___x_577_; 
v___x_577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_577_, 0, v_b_551_);
return v___x_577_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg___boxed(lean_object* v_as_578_, lean_object* v_i_579_, lean_object* v_stop_580_, lean_object* v_b_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_){
_start:
{
size_t v_i_boxed_587_; size_t v_stop_boxed_588_; lean_object* v_res_589_; 
v_i_boxed_587_ = lean_unbox_usize(v_i_579_);
lean_dec(v_i_579_);
v_stop_boxed_588_ = lean_unbox_usize(v_stop_580_);
lean_dec(v_stop_580_);
v_res_589_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(v_as_578_, v_i_boxed_587_, v_stop_boxed_588_, v_b_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_);
lean_dec(v___y_585_);
lean_dec_ref(v___y_584_);
lean_dec(v___y_583_);
lean_dec_ref(v___y_582_);
lean_dec_ref(v_as_578_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2(size_t v_sz_590_, size_t v_i_591_, lean_object* v_bs_592_){
_start:
{
uint8_t v___x_593_; 
v___x_593_ = lean_usize_dec_lt(v_i_591_, v_sz_590_);
if (v___x_593_ == 0)
{
return v_bs_592_;
}
else
{
lean_object* v_v_594_; lean_object* v___x_595_; lean_object* v_bs_x27_596_; lean_object* v___x_597_; size_t v___x_598_; size_t v___x_599_; lean_object* v___x_600_; 
v_v_594_ = lean_array_uget(v_bs_592_, v_i_591_);
v___x_595_ = lean_unsigned_to_nat(0u);
v_bs_x27_596_ = lean_array_uset(v_bs_592_, v_i_591_, v___x_595_);
v___x_597_ = l_Lean_Expr_mvarId_x21(v_v_594_);
lean_dec(v_v_594_);
v___x_598_ = ((size_t)1ULL);
v___x_599_ = lean_usize_add(v_i_591_, v___x_598_);
v___x_600_ = lean_array_uset(v_bs_x27_596_, v_i_591_, v___x_597_);
v_i_591_ = v___x_599_;
v_bs_592_ = v___x_600_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2___boxed(lean_object* v_sz_602_, lean_object* v_i_603_, lean_object* v_bs_604_){
_start:
{
size_t v_sz_boxed_605_; size_t v_i_boxed_606_; lean_object* v_res_607_; 
v_sz_boxed_605_ = lean_unbox_usize(v_sz_602_);
lean_dec(v_sz_602_);
v_i_boxed_606_ = lean_unbox_usize(v_i_603_);
lean_dec(v_i_603_);
v_res_607_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2(v_sz_boxed_605_, v_i_boxed_606_, v_bs_604_);
return v_res_607_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4(void){
_start:
{
lean_object* v___x_614_; lean_object* v___x_615_; 
v___x_614_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__3));
v___x_615_ = l_Lean_stringToMessageData(v___x_614_);
return v___x_615_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8(void){
_start:
{
lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_619_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__7));
v___x_620_ = lean_unsigned_to_nat(14u);
v___x_621_ = lean_unsigned_to_nat(22u);
v___x_622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__6));
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__5));
v___x_624_ = l_mkPanicMessageWithDecl(v___x_623_, v___x_622_, v___x_621_, v___x_620_, v___x_619_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try(lean_object* v_lem_625_, lean_object* v_assignableMVars_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v___x_634_; 
lean_inc_ref(v_lem_625_);
v___x_634_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_forallMetaTelescopeReducing(v_lem_625_, v_a_629_, v_a_630_, v_a_631_, v_a_632_);
if (lean_obj_tag(v___x_634_) == 0)
{
lean_object* v_a_635_; lean_object* v_snd_636_; lean_object* v_snd_637_; lean_object* v_fst_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_1013_; 
v_a_635_ = lean_ctor_get(v___x_634_, 0);
lean_inc(v_a_635_);
lean_dec_ref_known(v___x_634_, 1);
v_snd_636_ = lean_ctor_get(v_a_635_, 1);
lean_inc(v_snd_636_);
lean_dec(v_a_635_);
v_snd_637_ = lean_ctor_get(v_snd_636_, 1);
v_fst_638_ = lean_ctor_get(v_snd_636_, 0);
v_isSharedCheck_1013_ = !lean_is_exclusive(v_snd_636_);
if (v_isSharedCheck_1013_ == 0)
{
v___x_640_ = v_snd_636_;
v_isShared_641_ = v_isSharedCheck_1013_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_snd_637_);
lean_inc(v_fst_638_);
lean_dec(v_snd_636_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_1013_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
lean_object* v_fst_642_; lean_object* v_snd_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_1012_; 
v_fst_642_ = lean_ctor_get(v_snd_637_, 0);
v_snd_643_ = lean_ctor_get(v_snd_637_, 1);
v_isSharedCheck_1012_ = !lean_is_exclusive(v_snd_637_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_645_ = v_snd_637_;
v_isShared_646_ = v_isSharedCheck_1012_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_snd_643_);
lean_inc(v_fst_642_);
lean_dec(v_snd_637_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_1012_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v_hyp_x3f_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___y_652_; uint8_t v___y_653_; lean_object* v___y_654_; lean_object* v___y_655_; lean_object* v___y_656_; lean_object* v___y_657_; lean_object* v_filtered_658_; lean_object* v___y_659_; lean_object* v___y_660_; lean_object* v___y_661_; lean_object* v___y_662_; lean_object* v___y_663_; lean_object* v___y_664_; lean_object* v___y_746_; size_t v___y_747_; lean_object* v___y_748_; lean_object* v___y_749_; lean_object* v___y_750_; lean_object* v___y_751_; uint8_t v___y_752_; lean_object* v___y_753_; uint8_t v___y_754_; lean_object* v___y_755_; lean_object* v___y_756_; lean_object* v___y_757_; lean_object* v___y_758_; lean_object* v_a_759_; lean_object* v___y_834_; size_t v___y_835_; lean_object* v___y_836_; lean_object* v___y_837_; lean_object* v___y_838_; lean_object* v___y_839_; uint8_t v___y_840_; lean_object* v___y_841_; uint8_t v___y_842_; lean_object* v___y_843_; lean_object* v___y_844_; lean_object* v___y_845_; lean_object* v___y_846_; lean_object* v___y_847_; uint8_t v___y_858_; lean_object* v___y_859_; lean_object* v___y_860_; lean_object* v___y_861_; size_t v___y_862_; lean_object* v___y_863_; lean_object* v___y_864_; lean_object* v___y_865_; lean_object* v___y_866_; lean_object* v_a_867_; lean_object* v___y_904_; lean_object* v___y_905_; uint8_t v___y_906_; size_t v___y_907_; lean_object* v___y_908_; lean_object* v___y_909_; lean_object* v___y_910_; lean_object* v___y_911_; lean_object* v___y_912_; lean_object* v___y_913_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___y_927_; lean_object* v___y_928_; lean_object* v___y_929_; lean_object* v___y_930_; lean_object* v___y_931_; lean_object* v___y_932_; lean_object* v___y_958_; 
v_hyp_x3f_647_ = lean_ctor_get(v_a_627_, 8);
v___x_648_ = l_Lean_instInhabitedExpr;
v___x_649_ = lean_array_get_size(v_fst_638_);
v___x_650_ = lean_unsigned_to_nat(1u);
v___x_923_ = lean_nat_sub(v___x_649_, v___x_650_);
v___x_924_ = lean_array_get(v___x_648_, v_fst_638_, v___x_923_);
lean_dec(v___x_923_);
v___x_925_ = lean_array_pop(v_fst_638_);
if (lean_obj_tag(v_hyp_x3f_647_) == 0)
{
lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1009_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__8);
v___x_1010_ = lp_mathlib_panic___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__9(v___x_1009_);
v___y_958_ = v___x_1010_;
goto v___jp_957_;
}
else
{
lean_object* v_val_1011_; 
v_val_1011_ = lean_ctor_get(v_hyp_x3f_647_, 0);
lean_inc(v_val_1011_);
v___y_958_ = v_val_1011_;
goto v___jp_957_;
}
v___jp_651_:
{
lean_object* v___x_665_; 
lean_inc_ref(v_lem_625_);
v___x_665_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(v_lem_625_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v_a_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___x_665_, 1);
v___x_667_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0));
v___x_668_ = lean_mk_empty_array_with_capacity(v___y_654_);
lean_dec(v___y_654_);
v___x_669_ = lean_array_push(v___y_656_, v_a_666_);
lean_inc_ref(v___x_668_);
v___x_670_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_670_, 0, v___x_667_);
lean_ctor_set(v___x_670_, 1, v___x_668_);
lean_ctor_set(v___x_670_, 2, v___x_669_);
v___x_671_ = lean_array_push(v___y_652_, v___x_670_);
v___x_672_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_672_, 0, v___x_667_);
lean_ctor_set(v___x_672_, 1, v___x_668_);
lean_ctor_set(v___x_672_, 2, v___x_671_);
v___x_673_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v___y_655_, v___x_672_, v___y_653_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_673_) == 0)
{
lean_object* v_a_674_; lean_object* v___x_675_; 
v_a_674_ = lean_ctor_get(v___x_673_, 0);
lean_inc(v_a_674_);
lean_dec_ref_known(v___x_673_, 1);
v___x_675_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_getType(v_lem_625_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v_a_676_; lean_object* v___x_677_; uint8_t v___x_678_; lean_object* v___x_679_; 
v_a_676_ = lean_ctor_get(v___x_675_, 0);
lean_inc(v_a_676_);
lean_dec_ref_known(v___x_675_, 1);
v___x_677_ = lean_box(0);
v___x_678_ = 0;
v___x_679_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_676_, v___x_677_, v___x_678_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_679_) == 0)
{
lean_object* v_a_680_; lean_object* v_fst_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; 
v_a_680_ = lean_ctor_get(v___x_679_, 0);
lean_inc(v_a_680_);
lean_dec_ref_known(v___x_679_, 1);
v_fst_681_ = lean_ctor_get(v_a_680_, 0);
lean_inc(v_fst_681_);
lean_dec(v_a_680_);
v___x_682_ = lean_array_get_size(v_fst_681_);
v___x_683_ = lean_nat_sub(v___x_682_, v___x_650_);
v___x_684_ = lean_array_get(v___x_648_, v_fst_681_, v___x_683_);
lean_dec(v___x_683_);
lean_dec(v_fst_681_);
lean_inc(v___y_664_);
lean_inc_ref(v___y_663_);
lean_inc(v___y_662_);
lean_inc_ref(v___y_661_);
v___x_685_ = lean_infer_type(v___x_684_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_685_) == 0)
{
lean_object* v_a_686_; lean_object* v___x_687_; 
v_a_686_ = lean_ctor_get(v___x_685_, 0);
lean_inc(v_a_686_);
lean_dec_ref_known(v___x_685_, 1);
v___x_687_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_a_686_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_696_; 
v_a_688_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_696_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_696_ == 0)
{
v___x_690_ = v___x_687_;
v_isShared_691_ = v_isSharedCheck_696_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_687_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_696_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_692_; lean_object* v___x_694_; 
v___x_692_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_692_, 0, v_filtered_658_);
lean_ctor_set(v___x_692_, 1, v_a_674_);
lean_ctor_set(v___x_692_, 2, v___y_657_);
lean_ctor_set(v___x_692_, 3, v_a_688_);
if (v_isShared_691_ == 0)
{
lean_ctor_set(v___x_690_, 0, v___x_692_);
v___x_694_ = v___x_690_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_692_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
return v___x_694_;
}
}
}
else
{
lean_object* v_a_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_704_; 
lean_dec(v_a_674_);
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
v_a_697_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_704_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_704_ == 0)
{
v___x_699_ = v___x_687_;
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_a_697_);
lean_dec(v___x_687_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_702_; 
if (v_isShared_700_ == 0)
{
v___x_702_ = v___x_699_;
goto v_reusejp_701_;
}
else
{
lean_object* v_reuseFailAlloc_703_; 
v_reuseFailAlloc_703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_703_, 0, v_a_697_);
v___x_702_ = v_reuseFailAlloc_703_;
goto v_reusejp_701_;
}
v_reusejp_701_:
{
return v___x_702_;
}
}
}
}
else
{
lean_object* v_a_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_712_; 
lean_dec(v_a_674_);
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
v_a_705_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_712_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_712_ == 0)
{
v___x_707_ = v___x_685_;
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_a_705_);
lean_dec(v___x_685_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v___x_710_; 
if (v_isShared_708_ == 0)
{
v___x_710_ = v___x_707_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_711_; 
v_reuseFailAlloc_711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_711_, 0, v_a_705_);
v___x_710_ = v_reuseFailAlloc_711_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
return v___x_710_;
}
}
}
}
else
{
lean_object* v_a_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_720_; 
lean_dec(v_a_674_);
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
v_a_713_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_720_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_720_ == 0)
{
v___x_715_ = v___x_679_;
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_a_713_);
lean_dec(v___x_679_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___x_718_; 
if (v_isShared_716_ == 0)
{
v___x_718_ = v___x_715_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v_a_713_);
v___x_718_ = v_reuseFailAlloc_719_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
return v___x_718_;
}
}
}
}
else
{
lean_object* v_a_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_728_; 
lean_dec(v_a_674_);
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
v_a_721_ = lean_ctor_get(v___x_675_, 0);
v_isSharedCheck_728_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_728_ == 0)
{
v___x_723_ = v___x_675_;
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_a_721_);
lean_dec(v___x_675_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_726_; 
if (v_isShared_724_ == 0)
{
v___x_726_ = v___x_723_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_727_; 
v_reuseFailAlloc_727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_727_, 0, v_a_721_);
v___x_726_ = v_reuseFailAlloc_727_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
return v___x_726_;
}
}
}
}
else
{
lean_object* v_a_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_736_; 
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
lean_dec_ref(v_lem_625_);
v_a_729_ = lean_ctor_get(v___x_673_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_673_);
if (v_isSharedCheck_736_ == 0)
{
v___x_731_ = v___x_673_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_a_729_);
lean_dec(v___x_673_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___x_734_; 
if (v_isShared_732_ == 0)
{
v___x_734_ = v___x_731_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v_a_729_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
}
else
{
lean_object* v_a_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_744_; 
lean_dec(v_filtered_658_);
lean_dec_ref(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_652_);
lean_dec_ref(v_lem_625_);
v_a_737_ = lean_ctor_get(v___x_665_, 0);
v_isSharedCheck_744_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_744_ == 0)
{
v___x_739_ = v___x_665_;
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_665_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v___x_742_; 
if (v_isShared_740_ == 0)
{
v___x_742_ = v___x_739_;
goto v_reusejp_741_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_a_737_);
v___x_742_ = v_reuseFailAlloc_743_;
goto v_reusejp_741_;
}
v_reusejp_741_:
{
return v___x_742_;
}
}
}
}
v___jp_745_:
{
size_t v_sz_760_; lean_object* v___x_761_; 
v_sz_760_ = lean_array_size(v___y_749_);
lean_inc_ref(v___y_749_);
v___x_761_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg(v_sz_760_, v___y_747_, v___y_749_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
if (lean_obj_tag(v___x_761_) == 0)
{
lean_object* v_a_762_; uint8_t v___x_763_; lean_object* v___x_764_; 
v_a_762_ = lean_ctor_get(v___x_761_, 0);
lean_inc(v_a_762_);
lean_dec_ref_known(v___x_761_, 1);
v___x_763_ = 1;
lean_inc_ref(v___y_751_);
v___x_764_ = l_Lean_Meta_abstractMVars(v___y_751_, v___x_763_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
if (lean_obj_tag(v___x_764_) == 0)
{
lean_object* v_a_765_; lean_object* v___x_766_; lean_object* v___x_767_; 
v_a_765_ = lean_ctor_get(v___x_764_, 0);
lean_inc(v_a_765_);
lean_dec_ref_known(v___x_764_, 1);
lean_inc_ref_n(v_lem_625_, 2);
v___x_766_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_length(v_lem_625_);
v___x_767_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_ApplyAt_0__Mathlib_Tactic_ClickSuggestions_tacticSyntax(v_lem_625_, v___y_753_, v___y_748_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
if (lean_obj_tag(v___x_767_) == 0)
{
lean_object* v_a_768_; lean_object* v___x_769_; 
v_a_768_ = lean_ctor_get(v___x_767_, 0);
lean_inc(v_a_768_);
lean_dec_ref_known(v___x_767_, 1);
v___x_769_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___y_751_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
if (lean_obj_tag(v___x_769_) == 0)
{
lean_object* v_a_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v_a_770_ = lean_ctor_get(v___x_769_, 0);
lean_inc(v_a_770_);
lean_dec_ref_known(v___x_769_, 1);
v___x_771_ = lean_mk_empty_array_with_capacity(v___x_650_);
lean_inc_ref(v___x_771_);
v___x_772_ = lean_array_push(v___x_771_, v_a_770_);
v___x_773_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg(v___y_749_, v_sz_760_, v___y_747_, v___x_772_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
lean_dec_ref(v___y_749_);
if (lean_obj_tag(v___x_773_) == 0)
{
lean_object* v_a_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v_a_774_ = lean_ctor_get(v___x_773_, 0);
lean_inc(v_a_774_);
lean_dec_ref_known(v___x_773_, 1);
lean_inc_ref(v_lem_625_);
v___x_775_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toString(v_lem_625_);
v___x_776_ = lean_array_push(v_a_762_, v_a_765_);
v___x_777_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_777_, 0, v___y_758_);
lean_ctor_set(v___x_777_, 1, v___x_766_);
lean_ctor_set(v___x_777_, 2, v_a_759_);
lean_ctor_set(v___x_777_, 3, v___x_775_);
lean_ctor_set(v___x_777_, 4, v___x_776_);
if (v___y_752_ == 0)
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
v___x_778_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg___closed__0));
v___x_779_ = lean_mk_empty_array_with_capacity(v___y_746_);
lean_inc(v_a_774_);
v___x_780_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_780_, 0, v___x_778_);
lean_ctor_set(v___x_780_, 1, v___x_779_);
lean_ctor_set(v___x_780_, 2, v_a_774_);
lean_inc(v_a_768_);
v___x_781_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_mkSuggestion(v_a_768_, v___x_780_, v___y_754_, v___y_753_, v___y_748_, v___y_755_, v___y_750_, v___y_757_, v___y_756_);
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_783_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
lean_inc(v_a_782_);
lean_dec_ref_known(v___x_781_, 1);
v___x_783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_783_, 0, v_a_782_);
v___y_652_ = v_a_774_;
v___y_653_ = v___y_754_;
v___y_654_ = v___y_746_;
v___y_655_ = v_a_768_;
v___y_656_ = v___x_771_;
v___y_657_ = v___x_777_;
v_filtered_658_ = v___x_783_;
v___y_659_ = v___y_753_;
v___y_660_ = v___y_748_;
v___y_661_ = v___y_755_;
v___y_662_ = v___y_750_;
v___y_663_ = v___y_757_;
v___y_664_ = v___y_756_;
goto v___jp_651_;
}
else
{
lean_object* v_a_784_; lean_object* v___x_786_; uint8_t v_isShared_787_; uint8_t v_isSharedCheck_791_; 
lean_dec_ref_known(v___x_777_, 5);
lean_dec(v_a_774_);
lean_dec_ref(v___x_771_);
lean_dec(v_a_768_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_784_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_791_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_791_ == 0)
{
v___x_786_ = v___x_781_;
v_isShared_787_ = v_isSharedCheck_791_;
goto v_resetjp_785_;
}
else
{
lean_inc(v_a_784_);
lean_dec(v___x_781_);
v___x_786_ = lean_box(0);
v_isShared_787_ = v_isSharedCheck_791_;
goto v_resetjp_785_;
}
v_resetjp_785_:
{
lean_object* v___x_789_; 
if (v_isShared_787_ == 0)
{
v___x_789_ = v___x_786_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v_a_784_);
v___x_789_ = v_reuseFailAlloc_790_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
return v___x_789_;
}
}
}
}
else
{
lean_object* v___x_792_; 
v___x_792_ = lean_box(0);
v___y_652_ = v_a_774_;
v___y_653_ = v___y_754_;
v___y_654_ = v___y_746_;
v___y_655_ = v_a_768_;
v___y_656_ = v___x_771_;
v___y_657_ = v___x_777_;
v_filtered_658_ = v___x_792_;
v___y_659_ = v___y_753_;
v___y_660_ = v___y_748_;
v___y_661_ = v___y_755_;
v___y_662_ = v___y_750_;
v___y_663_ = v___y_757_;
v___y_664_ = v___y_756_;
goto v___jp_651_;
}
}
else
{
lean_object* v_a_793_; lean_object* v___x_795_; uint8_t v_isShared_796_; uint8_t v_isSharedCheck_800_; 
lean_dec_ref(v___x_771_);
lean_dec(v_a_768_);
lean_dec(v___x_766_);
lean_dec(v_a_765_);
lean_dec(v_a_762_);
lean_dec(v_a_759_);
lean_dec(v___y_758_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_793_ = lean_ctor_get(v___x_773_, 0);
v_isSharedCheck_800_ = !lean_is_exclusive(v___x_773_);
if (v_isSharedCheck_800_ == 0)
{
v___x_795_ = v___x_773_;
v_isShared_796_ = v_isSharedCheck_800_;
goto v_resetjp_794_;
}
else
{
lean_inc(v_a_793_);
lean_dec(v___x_773_);
v___x_795_ = lean_box(0);
v_isShared_796_ = v_isSharedCheck_800_;
goto v_resetjp_794_;
}
v_resetjp_794_:
{
lean_object* v___x_798_; 
if (v_isShared_796_ == 0)
{
v___x_798_ = v___x_795_;
goto v_reusejp_797_;
}
else
{
lean_object* v_reuseFailAlloc_799_; 
v_reuseFailAlloc_799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_799_, 0, v_a_793_);
v___x_798_ = v_reuseFailAlloc_799_;
goto v_reusejp_797_;
}
v_reusejp_797_:
{
return v___x_798_;
}
}
}
}
else
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_808_; 
lean_dec(v_a_768_);
lean_dec(v___x_766_);
lean_dec(v_a_765_);
lean_dec(v_a_762_);
lean_dec(v_a_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_801_ = lean_ctor_get(v___x_769_, 0);
v_isSharedCheck_808_ = !lean_is_exclusive(v___x_769_);
if (v_isSharedCheck_808_ == 0)
{
v___x_803_ = v___x_769_;
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_769_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v___x_806_; 
if (v_isShared_804_ == 0)
{
v___x_806_ = v___x_803_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_a_801_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
}
}
else
{
lean_object* v_a_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_816_; 
lean_dec(v___x_766_);
lean_dec(v_a_765_);
lean_dec(v_a_762_);
lean_dec(v_a_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_751_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_809_ = lean_ctor_get(v___x_767_, 0);
v_isSharedCheck_816_ = !lean_is_exclusive(v___x_767_);
if (v_isSharedCheck_816_ == 0)
{
v___x_811_ = v___x_767_;
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_a_809_);
lean_dec(v___x_767_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_814_; 
if (v_isShared_812_ == 0)
{
v___x_814_ = v___x_811_;
goto v_reusejp_813_;
}
else
{
lean_object* v_reuseFailAlloc_815_; 
v_reuseFailAlloc_815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_815_, 0, v_a_809_);
v___x_814_ = v_reuseFailAlloc_815_;
goto v_reusejp_813_;
}
v_reusejp_813_:
{
return v___x_814_;
}
}
}
}
else
{
lean_object* v_a_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_824_; 
lean_dec(v_a_762_);
lean_dec(v_a_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_751_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_817_ = lean_ctor_get(v___x_764_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_764_);
if (v_isSharedCheck_824_ == 0)
{
v___x_819_ = v___x_764_;
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_a_817_);
lean_dec(v___x_764_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v___x_822_; 
if (v_isShared_820_ == 0)
{
v___x_822_ = v___x_819_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_a_817_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_dec(v_a_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_751_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_746_);
lean_dec_ref(v_lem_625_);
v_a_825_ = lean_ctor_get(v___x_761_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_761_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_761_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_761_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
v___jp_833_:
{
if (lean_obj_tag(v___y_847_) == 0)
{
lean_object* v_a_848_; 
v_a_848_ = lean_ctor_get(v___y_847_, 0);
lean_inc(v_a_848_);
lean_dec_ref_known(v___y_847_, 1);
v___y_746_ = v___y_834_;
v___y_747_ = v___y_835_;
v___y_748_ = v___y_836_;
v___y_749_ = v___y_837_;
v___y_750_ = v___y_838_;
v___y_751_ = v___y_839_;
v___y_752_ = v___y_840_;
v___y_753_ = v___y_841_;
v___y_754_ = v___y_842_;
v___y_755_ = v___y_843_;
v___y_756_ = v___y_844_;
v___y_757_ = v___y_845_;
v___y_758_ = v___y_846_;
v_a_759_ = v_a_848_;
goto v___jp_745_;
}
else
{
lean_object* v_a_849_; lean_object* v___x_851_; uint8_t v_isShared_852_; uint8_t v_isSharedCheck_856_; 
lean_dec(v___y_846_);
lean_dec_ref(v___y_839_);
lean_dec_ref(v___y_837_);
lean_dec(v___y_834_);
lean_dec_ref(v_lem_625_);
v_a_849_ = lean_ctor_get(v___y_847_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___y_847_);
if (v_isSharedCheck_856_ == 0)
{
v___x_851_ = v___y_847_;
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_a_849_);
lean_dec(v___y_847_);
v___x_851_ = lean_box(0);
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
v_resetjp_850_:
{
lean_object* v___x_854_; 
if (v_isShared_852_ == 0)
{
v___x_854_ = v___x_851_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_855_; 
v_reuseFailAlloc_855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_855_, 0, v_a_849_);
v___x_854_ = v_reuseFailAlloc_855_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
return v___x_854_;
}
}
}
}
v___jp_857_:
{
size_t v_sz_868_; lean_object* v___x_869_; 
v_sz_868_ = lean_array_size(v_a_867_);
lean_inc_ref(v_a_867_);
v___x_869_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__3(v_sz_868_, v___y_862_, v_a_867_, v___y_860_, v___y_864_, v___y_861_, v___y_865_, v___y_866_, v___y_863_);
if (lean_obj_tag(v___x_869_) == 0)
{
lean_object* v_a_870_; lean_object* v___x_871_; lean_object* v_a_872_; lean_object* v___x_873_; lean_object* v___x_874_; 
v_a_870_ = lean_ctor_get(v___x_869_, 0);
lean_inc_n(v_a_870_, 2);
lean_dec_ref_known(v___x_869_, 1);
v___x_871_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__1___redArg(v_snd_643_, v___y_865_);
v_a_872_ = lean_ctor_get(v___x_871_, 0);
lean_inc_n(v_a_872_, 2);
lean_dec_ref(v___x_871_);
v___x_873_ = lean_array_push(v_a_870_, v_a_872_);
v___x_874_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_hasUnhelpfulMVars(v_a_867_, v_assignableMVars_626_, v___x_873_, v___y_861_, v___y_865_, v___y_866_, v___y_863_);
lean_dec_ref(v___x_873_);
lean_dec_ref(v_a_867_);
if (lean_obj_tag(v___x_874_) == 0)
{
lean_object* v_a_875_; lean_object* v___x_876_; uint8_t v___x_877_; 
v_a_875_ = lean_ctor_get(v___x_874_, 0);
lean_inc(v_a_875_);
lean_dec_ref_known(v___x_874_, 1);
v___x_876_ = lean_array_get_size(v_a_870_);
v___x_877_ = lean_nat_dec_lt(v___y_859_, v___x_876_);
if (v___x_877_ == 0)
{
uint8_t v___x_878_; 
v___x_878_ = lean_unbox(v_a_875_);
lean_dec(v_a_875_);
lean_inc(v___y_859_);
v___y_746_ = v___y_859_;
v___y_747_ = v___y_862_;
v___y_748_ = v___y_864_;
v___y_749_ = v_a_870_;
v___y_750_ = v___y_865_;
v___y_751_ = v_a_872_;
v___y_752_ = v___x_878_;
v___y_753_ = v___y_860_;
v___y_754_ = v___y_858_;
v___y_755_ = v___y_861_;
v___y_756_ = v___y_863_;
v___y_757_ = v___y_866_;
v___y_758_ = v___x_876_;
v_a_759_ = v___y_859_;
goto v___jp_745_;
}
else
{
uint8_t v___x_879_; 
v___x_879_ = lean_nat_dec_le(v___x_876_, v___x_876_);
if (v___x_879_ == 0)
{
if (v___x_877_ == 0)
{
uint8_t v___x_880_; 
v___x_880_ = lean_unbox(v_a_875_);
lean_dec(v_a_875_);
lean_inc(v___y_859_);
v___y_746_ = v___y_859_;
v___y_747_ = v___y_862_;
v___y_748_ = v___y_864_;
v___y_749_ = v_a_870_;
v___y_750_ = v___y_865_;
v___y_751_ = v_a_872_;
v___y_752_ = v___x_880_;
v___y_753_ = v___y_860_;
v___y_754_ = v___y_858_;
v___y_755_ = v___y_861_;
v___y_756_ = v___y_863_;
v___y_757_ = v___y_866_;
v___y_758_ = v___x_876_;
v_a_759_ = v___y_859_;
goto v___jp_745_;
}
else
{
size_t v___x_881_; lean_object* v___x_882_; uint8_t v___x_883_; 
v___x_881_ = lean_usize_of_nat(v___x_876_);
lean_inc(v___y_859_);
v___x_882_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(v_a_870_, v___y_862_, v___x_881_, v___y_859_, v___y_861_, v___y_865_, v___y_866_, v___y_863_);
v___x_883_ = lean_unbox(v_a_875_);
lean_dec(v_a_875_);
v___y_834_ = v___y_859_;
v___y_835_ = v___y_862_;
v___y_836_ = v___y_864_;
v___y_837_ = v_a_870_;
v___y_838_ = v___y_865_;
v___y_839_ = v_a_872_;
v___y_840_ = v___x_883_;
v___y_841_ = v___y_860_;
v___y_842_ = v___y_858_;
v___y_843_ = v___y_861_;
v___y_844_ = v___y_863_;
v___y_845_ = v___y_866_;
v___y_846_ = v___x_876_;
v___y_847_ = v___x_882_;
goto v___jp_833_;
}
}
else
{
size_t v___x_884_; lean_object* v___x_885_; uint8_t v___x_886_; 
v___x_884_ = lean_usize_of_nat(v___x_876_);
lean_inc(v___y_859_);
v___x_885_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(v_a_870_, v___y_862_, v___x_884_, v___y_859_, v___y_861_, v___y_865_, v___y_866_, v___y_863_);
v___x_886_ = lean_unbox(v_a_875_);
lean_dec(v_a_875_);
v___y_834_ = v___y_859_;
v___y_835_ = v___y_862_;
v___y_836_ = v___y_864_;
v___y_837_ = v_a_870_;
v___y_838_ = v___y_865_;
v___y_839_ = v_a_872_;
v___y_840_ = v___x_886_;
v___y_841_ = v___y_860_;
v___y_842_ = v___y_858_;
v___y_843_ = v___y_861_;
v___y_844_ = v___y_863_;
v___y_845_ = v___y_866_;
v___y_846_ = v___x_876_;
v___y_847_ = v___x_885_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_894_; 
lean_dec(v_a_872_);
lean_dec(v_a_870_);
lean_dec(v___y_859_);
lean_dec_ref(v_lem_625_);
v_a_887_ = lean_ctor_get(v___x_874_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_874_);
if (v_isSharedCheck_894_ == 0)
{
v___x_889_ = v___x_874_;
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_874_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_894_;
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
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_a_887_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
else
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
lean_dec_ref(v_a_867_);
lean_dec(v___y_859_);
lean_dec(v_snd_643_);
lean_dec_ref(v_lem_625_);
v_a_895_ = lean_ctor_get(v___x_869_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_869_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_869_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_869_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
}
v___jp_903_:
{
if (lean_obj_tag(v___y_913_) == 0)
{
lean_object* v_a_914_; 
v_a_914_ = lean_ctor_get(v___y_913_, 0);
lean_inc(v_a_914_);
lean_dec_ref_known(v___y_913_, 1);
v___y_858_ = v___y_906_;
v___y_859_ = v___y_905_;
v___y_860_ = v___y_904_;
v___y_861_ = v___y_908_;
v___y_862_ = v___y_907_;
v___y_863_ = v___y_909_;
v___y_864_ = v___y_910_;
v___y_865_ = v___y_911_;
v___y_866_ = v___y_912_;
v_a_867_ = v_a_914_;
goto v___jp_857_;
}
else
{
lean_object* v_a_915_; lean_object* v___x_917_; uint8_t v_isShared_918_; uint8_t v_isSharedCheck_922_; 
lean_dec(v___y_905_);
lean_dec(v_snd_643_);
lean_dec_ref(v_lem_625_);
v_a_915_ = lean_ctor_get(v___y_913_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___y_913_);
if (v_isSharedCheck_922_ == 0)
{
v___x_917_ = v___y_913_;
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
else
{
lean_inc(v_a_915_);
lean_dec(v___y_913_);
v___x_917_ = lean_box(0);
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
v_resetjp_916_:
{
lean_object* v___x_920_; 
if (v_isShared_918_ == 0)
{
v___x_920_ = v___x_917_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v_a_915_);
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
v___jp_926_:
{
lean_object* v___x_933_; lean_object* v___x_934_; uint8_t v___x_935_; lean_object* v___x_936_; 
v___x_933_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__1));
v___x_934_ = lean_box(0);
v___x_935_ = 0;
v___x_936_ = l_Lean_Meta_synthAppInstances(v___x_933_, v___x_934_, v___x_925_, v_fst_642_, v___x_935_, v___x_935_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
if (lean_obj_tag(v___x_936_) == 0)
{
size_t v_sz_937_; size_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; uint8_t v___x_943_; 
lean_dec_ref_known(v___x_936_, 1);
v_sz_937_ = lean_array_size(v___x_925_);
v___x_938_ = ((size_t)0ULL);
v___x_939_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__2(v_sz_937_, v___x_938_, v___x_925_);
v___x_940_ = lean_unsigned_to_nat(0u);
v___x_941_ = lean_array_get_size(v___x_939_);
v___x_942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__2));
v___x_943_ = lean_nat_dec_lt(v___x_940_, v___x_941_);
if (v___x_943_ == 0)
{
lean_dec_ref(v___x_939_);
v___y_858_ = v___x_935_;
v___y_859_ = v___x_940_;
v___y_860_ = v___y_927_;
v___y_861_ = v___y_929_;
v___y_862_ = v___x_938_;
v___y_863_ = v___y_932_;
v___y_864_ = v___y_928_;
v___y_865_ = v___y_930_;
v___y_866_ = v___y_931_;
v_a_867_ = v___x_942_;
goto v___jp_857_;
}
else
{
uint8_t v___x_944_; 
v___x_944_ = lean_nat_dec_le(v___x_941_, v___x_941_);
if (v___x_944_ == 0)
{
if (v___x_943_ == 0)
{
lean_dec_ref(v___x_939_);
v___y_858_ = v___x_935_;
v___y_859_ = v___x_940_;
v___y_860_ = v___y_927_;
v___y_861_ = v___y_929_;
v___y_862_ = v___x_938_;
v___y_863_ = v___y_932_;
v___y_864_ = v___y_928_;
v___y_865_ = v___y_930_;
v___y_866_ = v___y_931_;
v_a_867_ = v___x_942_;
goto v___jp_857_;
}
else
{
size_t v___x_945_; lean_object* v___x_946_; 
v___x_945_ = lean_usize_of_nat(v___x_941_);
v___x_946_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7(v___x_939_, v___x_938_, v___x_945_, v___x_942_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec_ref(v___x_939_);
v___y_904_ = v___y_927_;
v___y_905_ = v___x_940_;
v___y_906_ = v___x_935_;
v___y_907_ = v___x_938_;
v___y_908_ = v___y_929_;
v___y_909_ = v___y_932_;
v___y_910_ = v___y_928_;
v___y_911_ = v___y_930_;
v___y_912_ = v___y_931_;
v___y_913_ = v___x_946_;
goto v___jp_903_;
}
}
else
{
size_t v___x_947_; lean_object* v___x_948_; 
v___x_947_ = lean_usize_of_nat(v___x_941_);
v___x_948_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__7(v___x_939_, v___x_938_, v___x_947_, v___x_942_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec_ref(v___x_939_);
v___y_904_ = v___y_927_;
v___y_905_ = v___x_940_;
v___y_906_ = v___x_935_;
v___y_907_ = v___x_938_;
v___y_908_ = v___y_929_;
v___y_909_ = v___y_932_;
v___y_910_ = v___y_928_;
v___y_911_ = v___y_930_;
v___y_912_ = v___y_931_;
v___y_913_ = v___x_948_;
goto v___jp_903_;
}
}
}
else
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_956_; 
lean_dec_ref(v___x_925_);
lean_dec(v_snd_643_);
lean_dec_ref(v_lem_625_);
v_a_949_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_956_ == 0)
{
v___x_951_ = v___x_936_;
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_936_);
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
v___jp_957_:
{
lean_object* v___x_959_; lean_object* v___x_960_; 
lean_inc(v___y_958_);
v___x_959_ = l_Lean_Expr_fvar___override(v___y_958_);
lean_inc(v___x_924_);
v___x_960_ = l_Lean_Meta_isExprDefEq(v___x_924_, v___x_959_, v_a_629_, v_a_630_, v_a_631_, v_a_632_);
if (lean_obj_tag(v___x_960_) == 0)
{
lean_object* v_a_961_; uint8_t v___x_962_; 
v_a_961_ = lean_ctor_get(v___x_960_, 0);
lean_inc(v_a_961_);
lean_dec_ref_known(v___x_960_, 1);
v___x_962_ = lean_unbox(v_a_961_);
lean_dec(v_a_961_);
if (v___x_962_ == 0)
{
lean_object* v___x_963_; 
lean_dec_ref(v___x_925_);
lean_dec(v_snd_643_);
lean_dec(v_fst_642_);
lean_dec_ref(v_lem_625_);
lean_inc(v_a_632_);
lean_inc_ref(v_a_631_);
lean_inc(v_a_630_);
lean_inc_ref(v_a_629_);
v___x_963_ = lean_infer_type(v___x_924_, v_a_629_, v_a_630_, v_a_631_, v_a_632_);
if (lean_obj_tag(v___x_963_) == 0)
{
lean_object* v_a_964_; lean_object* v___x_965_; 
v_a_964_ = lean_ctor_get(v___x_963_, 0);
lean_inc(v_a_964_);
lean_dec_ref_known(v___x_963_, 1);
v___x_965_ = l_Lean_FVarId_getType___redArg(v___y_958_, v_a_629_, v_a_631_, v_a_632_);
if (lean_obj_tag(v___x_965_) == 0)
{
lean_object* v_a_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_970_; 
v_a_966_ = lean_ctor_get(v___x_965_, 0);
lean_inc(v_a_966_);
lean_dec_ref_known(v___x_965_, 1);
v___x_967_ = l_Lean_MessageData_ofExpr(v_a_964_);
v___x_968_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___closed__4);
if (v_isShared_646_ == 0)
{
lean_ctor_set_tag(v___x_645_, 7);
lean_ctor_set(v___x_645_, 1, v___x_968_);
lean_ctor_set(v___x_645_, 0, v___x_967_);
v___x_970_ = v___x_645_;
goto v_reusejp_969_;
}
else
{
lean_object* v_reuseFailAlloc_984_; 
v_reuseFailAlloc_984_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_984_, 0, v___x_967_);
lean_ctor_set(v_reuseFailAlloc_984_, 1, v___x_968_);
v___x_970_ = v_reuseFailAlloc_984_;
goto v_reusejp_969_;
}
v_reusejp_969_:
{
lean_object* v___x_971_; lean_object* v___x_973_; 
v___x_971_ = l_Lean_MessageData_ofExpr(v_a_966_);
if (v_isShared_641_ == 0)
{
lean_ctor_set_tag(v___x_640_, 7);
lean_ctor_set(v___x_640_, 1, v___x_971_);
lean_ctor_set(v___x_640_, 0, v___x_970_);
v___x_973_ = v___x_640_;
goto v_reusejp_972_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v___x_970_);
lean_ctor_set(v_reuseFailAlloc_983_, 1, v___x_971_);
v___x_973_ = v_reuseFailAlloc_983_;
goto v_reusejp_972_;
}
v_reusejp_972_:
{
lean_object* v___x_974_; lean_object* v_a_975_; lean_object* v___x_977_; uint8_t v_isShared_978_; uint8_t v_isSharedCheck_982_; 
v___x_974_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg(v___x_973_, v_a_629_, v_a_630_, v_a_631_, v_a_632_);
v_a_975_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_982_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_982_ == 0)
{
v___x_977_ = v___x_974_;
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
else
{
lean_inc(v_a_975_);
lean_dec(v___x_974_);
v___x_977_ = lean_box(0);
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
v_resetjp_976_:
{
lean_object* v___x_980_; 
if (v_isShared_978_ == 0)
{
v___x_980_ = v___x_977_;
goto v_reusejp_979_;
}
else
{
lean_object* v_reuseFailAlloc_981_; 
v_reuseFailAlloc_981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_981_, 0, v_a_975_);
v___x_980_ = v_reuseFailAlloc_981_;
goto v_reusejp_979_;
}
v_reusejp_979_:
{
return v___x_980_;
}
}
}
}
}
else
{
lean_object* v_a_985_; lean_object* v___x_987_; uint8_t v_isShared_988_; uint8_t v_isSharedCheck_992_; 
lean_dec(v_a_964_);
lean_del_object(v___x_645_);
lean_del_object(v___x_640_);
v_a_985_ = lean_ctor_get(v___x_965_, 0);
v_isSharedCheck_992_ = !lean_is_exclusive(v___x_965_);
if (v_isSharedCheck_992_ == 0)
{
v___x_987_ = v___x_965_;
v_isShared_988_ = v_isSharedCheck_992_;
goto v_resetjp_986_;
}
else
{
lean_inc(v_a_985_);
lean_dec(v___x_965_);
v___x_987_ = lean_box(0);
v_isShared_988_ = v_isSharedCheck_992_;
goto v_resetjp_986_;
}
v_resetjp_986_:
{
lean_object* v___x_990_; 
if (v_isShared_988_ == 0)
{
v___x_990_ = v___x_987_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v_a_985_);
v___x_990_ = v_reuseFailAlloc_991_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
return v___x_990_;
}
}
}
}
else
{
lean_object* v_a_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1000_; 
lean_dec(v___y_958_);
lean_del_object(v___x_645_);
lean_del_object(v___x_640_);
v_a_993_ = lean_ctor_get(v___x_963_, 0);
v_isSharedCheck_1000_ = !lean_is_exclusive(v___x_963_);
if (v_isSharedCheck_1000_ == 0)
{
v___x_995_ = v___x_963_;
v_isShared_996_ = v_isSharedCheck_1000_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_a_993_);
lean_dec(v___x_963_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1000_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v___x_998_; 
if (v_isShared_996_ == 0)
{
v___x_998_ = v___x_995_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v_a_993_);
v___x_998_ = v_reuseFailAlloc_999_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
return v___x_998_;
}
}
}
}
else
{
lean_dec(v___y_958_);
lean_dec(v___x_924_);
lean_del_object(v___x_645_);
lean_del_object(v___x_640_);
v___y_927_ = v_a_627_;
v___y_928_ = v_a_628_;
v___y_929_ = v_a_629_;
v___y_930_ = v_a_630_;
v___y_931_ = v_a_631_;
v___y_932_ = v_a_632_;
goto v___jp_926_;
}
}
else
{
lean_object* v_a_1001_; lean_object* v___x_1003_; uint8_t v_isShared_1004_; uint8_t v_isSharedCheck_1008_; 
lean_dec(v___y_958_);
lean_dec_ref(v___x_925_);
lean_dec(v___x_924_);
lean_del_object(v___x_645_);
lean_dec(v_snd_643_);
lean_dec(v_fst_642_);
lean_del_object(v___x_640_);
lean_dec_ref(v_lem_625_);
v_a_1001_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_1008_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_1008_ == 0)
{
v___x_1003_ = v___x_960_;
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
else
{
lean_inc(v_a_1001_);
lean_dec(v___x_960_);
v___x_1003_ = lean_box(0);
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
v_resetjp_1002_:
{
lean_object* v___x_1006_; 
if (v_isShared_1004_ == 0)
{
v___x_1006_ = v___x_1003_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v_a_1001_);
v___x_1006_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
return v___x_1006_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1014_; lean_object* v___x_1016_; uint8_t v_isShared_1017_; uint8_t v_isSharedCheck_1021_; 
lean_dec_ref(v_lem_625_);
v_a_1014_ = lean_ctor_get(v___x_634_, 0);
v_isSharedCheck_1021_ = !lean_is_exclusive(v___x_634_);
if (v_isSharedCheck_1021_ == 0)
{
v___x_1016_ = v___x_634_;
v_isShared_1017_ = v_isSharedCheck_1021_;
goto v_resetjp_1015_;
}
else
{
lean_inc(v_a_1014_);
lean_dec(v___x_634_);
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
v_reuseFailAlloc_1020_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try___boxed(lean_object* v_lem_1022_, lean_object* v_assignableMVars_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v_res_1031_; 
v_res_1031_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try(v_lem_1022_, v_assignableMVars_1023_, v_a_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
lean_dec(v_a_1029_);
lean_dec_ref(v_a_1028_);
lean_dec(v_a_1027_);
lean_dec_ref(v_a_1026_);
lean_dec(v_a_1025_);
lean_dec_ref(v_a_1024_);
lean_dec_ref(v_assignableMVars_1023_);
return v_res_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0(lean_object* v_mvarId_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_){
_start:
{
lean_object* v___x_1040_; 
v___x_1040_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___redArg(v_mvarId_1032_, v___y_1036_);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0___boxed(lean_object* v_mvarId_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_){
_start:
{
lean_object* v_res_1049_; 
v_res_1049_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0(v_mvarId_1041_, v___y_1042_, v___y_1043_, v___y_1044_, v___y_1045_, v___y_1046_, v___y_1047_);
lean_dec(v___y_1047_);
lean_dec_ref(v___y_1046_);
lean_dec(v___y_1045_);
lean_dec_ref(v___y_1044_);
lean_dec(v___y_1043_);
lean_dec_ref(v___y_1042_);
lean_dec(v_mvarId_1041_);
return v_res_1049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4(size_t v_sz_1050_, size_t v_i_1051_, lean_object* v_bs_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_){
_start:
{
lean_object* v___x_1060_; 
v___x_1060_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___redArg(v_sz_1050_, v_i_1051_, v_bs_1052_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_);
return v___x_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4___boxed(lean_object* v_sz_1061_, lean_object* v_i_1062_, lean_object* v_bs_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_){
_start:
{
size_t v_sz_boxed_1071_; size_t v_i_boxed_1072_; lean_object* v_res_1073_; 
v_sz_boxed_1071_ = lean_unbox_usize(v_sz_1061_);
lean_dec(v_sz_1061_);
v_i_boxed_1072_ = lean_unbox_usize(v_i_1062_);
lean_dec(v_i_1062_);
v_res_1073_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__4(v_sz_boxed_1071_, v_i_boxed_1072_, v_bs_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_, v___y_1069_);
lean_dec(v___y_1069_);
lean_dec_ref(v___y_1068_);
lean_dec(v___y_1067_);
lean_dec_ref(v___y_1066_);
lean_dec(v___y_1065_);
lean_dec_ref(v___y_1064_);
return v_res_1073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5(lean_object* v_as_1074_, size_t v_sz_1075_, size_t v_i_1076_, lean_object* v_b_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_){
_start:
{
lean_object* v___x_1085_; 
v___x_1085_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___redArg(v_as_1074_, v_sz_1075_, v_i_1076_, v_b_1077_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5___boxed(lean_object* v_as_1086_, lean_object* v_sz_1087_, lean_object* v_i_1088_, lean_object* v_b_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
size_t v_sz_boxed_1097_; size_t v_i_boxed_1098_; lean_object* v_res_1099_; 
v_sz_boxed_1097_ = lean_unbox_usize(v_sz_1087_);
lean_dec(v_sz_1087_);
v_i_boxed_1098_ = lean_unbox_usize(v_i_1088_);
lean_dec(v_i_1088_);
v_res_1099_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__5(v_as_1086_, v_sz_boxed_1097_, v_i_boxed_1098_, v_b_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
lean_dec(v___y_1093_);
lean_dec_ref(v___y_1092_);
lean_dec(v___y_1091_);
lean_dec_ref(v___y_1090_);
lean_dec_ref(v_as_1086_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6(lean_object* v_as_1100_, size_t v_i_1101_, size_t v_stop_1102_, lean_object* v_b_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
lean_object* v___x_1111_; 
v___x_1111_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___redArg(v_as_1100_, v_i_1101_, v_stop_1102_, v_b_1103_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
return v___x_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6___boxed(lean_object* v_as_1112_, lean_object* v_i_1113_, lean_object* v_stop_1114_, lean_object* v_b_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_){
_start:
{
size_t v_i_boxed_1123_; size_t v_stop_boxed_1124_; lean_object* v_res_1125_; 
v_i_boxed_1123_ = lean_unbox_usize(v_i_1113_);
lean_dec(v_i_1113_);
v_stop_boxed_1124_ = lean_unbox_usize(v_stop_1114_);
lean_dec(v_stop_1114_);
v_res_1125_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__6(v_as_1112_, v_i_boxed_1123_, v_stop_boxed_1124_, v_b_1115_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
lean_dec(v___y_1117_);
lean_dec_ref(v___y_1116_);
lean_dec_ref(v_as_1112_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8(lean_object* v_00_u03b1_1126_, lean_object* v_msg_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_){
_start:
{
lean_object* v___x_1135_; 
v___x_1135_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___redArg(v_msg_1127_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_);
return v___x_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8___boxed(lean_object* v_00_u03b1_1136_, lean_object* v_msg_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__8(v_00_u03b1_1136_, v_msg_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
return v_res_1145_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0(lean_object* v_00_u03b2_1146_, lean_object* v_x_1147_, lean_object* v_x_1148_){
_start:
{
uint8_t v___x_1149_; 
v___x_1149_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___redArg(v_x_1147_, v_x_1148_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1150_, lean_object* v_x_1151_, lean_object* v_x_1152_){
_start:
{
uint8_t v_res_1153_; lean_object* v_r_1154_; 
v_res_1153_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0(v_00_u03b2_1150_, v_x_1151_, v_x_1152_);
lean_dec(v_x_1152_);
lean_dec_ref(v_x_1151_);
v_r_1154_ = lean_box(v_res_1153_);
return v_r_1154_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1155_, lean_object* v_x_1156_, size_t v_x_1157_, lean_object* v_x_1158_){
_start:
{
uint8_t v___x_1159_; 
v___x_1159_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___redArg(v_x_1156_, v_x_1157_, v_x_1158_);
return v___x_1159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_1160_, lean_object* v_x_1161_, lean_object* v_x_1162_, lean_object* v_x_1163_){
_start:
{
size_t v_x_46251__boxed_1164_; uint8_t v_res_1165_; lean_object* v_r_1166_; 
v_x_46251__boxed_1164_ = lean_unbox_usize(v_x_1162_);
lean_dec(v_x_1162_);
v_res_1165_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3(v_00_u03b2_1160_, v_x_1161_, v_x_46251__boxed_1164_, v_x_1163_);
lean_dec(v_x_1163_);
lean_dec_ref(v_x_1161_);
v_r_1166_ = lean_box(v_res_1165_);
return v_r_1166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12(lean_object* v_00_u03b2_1167_, lean_object* v_keys_1168_, lean_object* v_vals_1169_, lean_object* v_heq_1170_, lean_object* v_i_1171_, lean_object* v_k_1172_){
_start:
{
uint8_t v___x_1173_; 
v___x_1173_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___redArg(v_keys_1168_, v_i_1171_, v_k_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12___boxed(lean_object* v_00_u03b2_1174_, lean_object* v_keys_1175_, lean_object* v_vals_1176_, lean_object* v_heq_1177_, lean_object* v_i_1178_, lean_object* v_k_1179_){
_start:
{
uint8_t v_res_1180_; lean_object* v_r_1181_; 
v_res_1180_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_ClickSuggestions_ApplyAtLemma_try_spec__0_spec__0_spec__3_spec__12(v_00_u03b2_1174_, v_keys_1175_, v_vals_1176_, v_heq_1177_, v_i_1178_, v_k_1179_);
lean_dec(v_k_1179_);
lean_dec_ref(v_vals_1176_);
lean_dec_ref(v_keys_1175_);
v_r_1181_ = lean_box(v_res_1180_);
return v_r_1181_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_ApplyAt(builtin);
}
#ifdef __cplusplus
}
#endif
