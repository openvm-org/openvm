// Lean compiler output
// Module: Mathlib.Tactic.Translate.Reorder
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_has_loose_bvar(lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_st_mk_ref(lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_instMonadBaseIO;
lean_object* l_OptionT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_pure(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaBoundedTelescope(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_head_x3f___redArg(lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_instMonadOption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_finIdxOf_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Option_bind(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__0(lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Vector_Basic_0__Vector_mapM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Init.Data.Array.Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Array.swapAt!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "index "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " out of bounds"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permuteList_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permuteList_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_Permutation_reverse_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_reverse(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0___redArg(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_reverse(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Array_map__unattach_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Array_map__unattach_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_range(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instBEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instBEq = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instBEq___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__List_map__unattach_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__List_map__unattach_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToString___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToString = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Reorder_reverse(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "the permutation (reorder := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = ") is out of bounds, the type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nhas only "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " arguments"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = ") is out of bounds, the function"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__3___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instFunctorOption___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__7_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_bind, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__9_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Mathlib.Tactic.Translate.Reorder"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 88, .m_capacity = 88, .m_length = 87, .m_data = "_private.Mathlib.Tactic.Translate.Reorder.0.Mathlib.Tactic.Translate.guessReorder.visit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = " is out of bounds ("};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "translateReorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(4, 147, 46, 38, 55, 154, 104, 4)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(106, 114, 39, 188, 154, 41, 11, 126)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "`(translateReorder| "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(4, 147, 46, 38, 55, 154, 104, 4)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__6_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Category_translateReorder;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "reorderPart"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Translate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__3_value),LEAN_SCALAR_PTR_LITERAL(172, 43, 0, 86, 12, 33, 116, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 111, 110, 194, 80, 122, 40, 41)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__5_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__7_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__12_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_translateReorder_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderPart = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "reorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__3_value),LEAN_SCALAR_PTR_LITERAL(172, 43, 0, 86, 12, 33, 116, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(180, 75, 101, 132, 106, 186, 147, 166)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorder = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "index `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "` is out of bounds, there are only `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "` arguments"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "invalid index `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "`, arguments are counted starting from 1."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "invalid argument `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "`, it is not an argument of `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "The reorder within argument "};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = " has been set to both `"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__2 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "` and `"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__4 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "`. Please specify it only once."};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__6 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2___redArg(lean_object*);
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 136, .m_capacity = 136, .m_length = 135, .m_data = "Please remove the duplicate entries from the disjoint cycle representation.\nSee the docstring of `reorder` for how to specify reorders."};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Invalid cycle `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 102, .m_capacity = 102, .m_length = 101, .m_data = "`, a cycle must have at least 2 elements.\nSee the docstring of `reorder` for how to specify reorders."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___redArg(lean_object* v___x_1_, lean_object* v_msg_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_panic_fn_borrowed(v___x_1_, v_msg_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___redArg___boxed(lean_object* v___x_4_, lean_object* v_msg_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___redArg(v___x_4_, v_msg_5_);
lean_dec_ref(v___x_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0(lean_object* v_00_u03b1_7_, lean_object* v___x_8_, lean_object* v_msg_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_panic_fn_borrowed(v___x_8_, v_msg_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0___boxed(lean_object* v_00_u03b1_11_, lean_object* v___x_12_, lean_object* v_msg_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux_spec__0(v_00_u03b1_11_, v___x_12_, v_msg_13_);
lean_dec_ref(v___x_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg(lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v_a_21_, lean_object* v_a_22_){
_start:
{
if (lean_obj_tag(v_a_20_) == 0)
{
lean_object* v___x_23_; 
v___x_23_ = lean_array_set(v_a_19_, v_a_22_, v_a_21_);
return v___x_23_;
}
else
{
lean_object* v_head_24_; lean_object* v_tail_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_51_; 
v_head_24_ = lean_ctor_get(v_a_20_, 0);
v_tail_25_ = lean_ctor_get(v_a_20_, 1);
v_isSharedCheck_51_ = !lean_is_exclusive(v_a_20_);
if (v_isSharedCheck_51_ == 0)
{
v___x_27_ = v_a_20_;
v_isShared_28_ = v_isSharedCheck_51_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_tail_25_);
lean_inc(v_head_24_);
lean_dec(v_a_20_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_51_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; uint8_t v___x_30_; 
v___x_29_ = lean_array_get_size(v_a_19_);
v___x_30_ = lean_nat_dec_lt(v_head_24_, v___x_29_);
if (v___x_30_ == 0)
{
lean_object* v___x_32_; 
if (v_isShared_28_ == 0)
{
lean_ctor_set_tag(v___x_27_, 0);
lean_ctor_set(v___x_27_, 1, v_a_19_);
lean_ctor_set(v___x_27_, 0, v_a_21_);
v___x_32_ = v___x_27_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v_a_21_);
lean_ctor_set(v_reuseFailAlloc_47_, 1, v_a_19_);
v___x_32_ = v_reuseFailAlloc_47_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v_fst_44_; lean_object* v_snd_45_; 
v___x_33_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__0));
v___x_34_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__1));
v___x_35_ = lean_unsigned_to_nat(438u);
v___x_36_ = lean_unsigned_to_nat(4u);
v___x_37_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__2));
v___x_38_ = l_Nat_reprFast(v_head_24_);
v___x_39_ = lean_string_append(v___x_37_, v___x_38_);
lean_dec_ref(v___x_38_);
v___x_40_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__3));
v___x_41_ = lean_string_append(v___x_39_, v___x_40_);
v___x_42_ = l_mkPanicMessageWithDecl(v___x_33_, v___x_34_, v___x_35_, v___x_36_, v___x_41_);
lean_dec_ref(v___x_41_);
v___x_43_ = lean_panic_fn_borrowed(v___x_32_, v___x_42_);
lean_dec_ref(v___x_32_);
v_fst_44_ = lean_ctor_get(v___x_43_, 0);
lean_inc(v_fst_44_);
v_snd_45_ = lean_ctor_get(v___x_43_, 1);
lean_inc(v_snd_45_);
lean_dec(v___x_43_);
v_a_19_ = v_snd_45_;
v_a_20_ = v_tail_25_;
v_a_21_ = v_fst_44_;
goto _start;
}
}
else
{
lean_object* v_e_48_; lean_object* v_xs_x27_49_; 
lean_del_object(v___x_27_);
v_e_48_ = lean_array_fget(v_a_19_, v_head_24_);
v_xs_x27_49_ = lean_array_fset(v_a_19_, v_head_24_, v_a_21_);
lean_dec(v_head_24_);
v_a_19_ = v_xs_x27_49_;
v_a_20_ = v_tail_25_;
v_a_21_ = v_e_48_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___boxed(lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg(v_a_52_, v_a_53_, v_a_54_, v_a_55_);
lean_dec(v_a_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux(lean_object* v_00_u03b1_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_, lean_object* v_a_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg(v_a_58_, v_a_59_, v_a_60_, v_a_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___boxed(lean_object* v_00_u03b1_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux(v_00_u03b1_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
lean_dec(v_a_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg(lean_object* v_inst_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
if (lean_obj_tag(v_a_71_) == 0)
{
return v_a_70_;
}
else
{
lean_object* v_head_72_; lean_object* v_tail_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v_head_72_ = lean_ctor_get(v_a_71_, 0);
lean_inc(v_head_72_);
v_tail_73_ = lean_ctor_get(v_a_71_, 1);
lean_inc(v_tail_73_);
lean_dec_ref_known(v_a_71_, 2);
v___x_74_ = lean_array_get(v_inst_69_, v_a_70_, v_head_72_);
v___x_75_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg(v_a_70_, v_tail_73_, v___x_74_, v_head_72_);
lean_dec(v_head_72_);
return v___x_75_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg___boxed(lean_object* v_inst_76_, lean_object* v_a_77_, lean_object* v_a_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg(v_inst_76_, v_a_77_, v_a_78_);
lean_dec(v_inst_76_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21(lean_object* v_00_u03b1_80_, lean_object* v_inst_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg(v_inst_81_, v_a_82_, v_a_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___boxed(lean_object* v_00_u03b1_85_, lean_object* v_inst_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21(v_00_u03b1_85_, v_inst_86_, v_a_87_, v_a_88_);
lean_dec(v_inst_86_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0(lean_object* v_inst_90_, lean_object* v_x1_91_, lean_object* v_x2_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermute_x21___redArg(v_inst_90_, v_x1_91_, v_x2_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0___boxed(lean_object* v_inst_94_, lean_object* v_x1_95_, lean_object* v_x2_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0(v_inst_94_, v_x1_95_, v_x2_96_);
lean_dec(v_inst_94_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(lean_object* v_inst_98_, lean_object* v_c_99_, lean_object* v_init_100_){
_start:
{
lean_object* v___f_101_; lean_object* v___x_102_; 
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_101_, 0, v_inst_98_);
v___x_102_ = l_List_foldl___redArg(v___f_101_, v_init_100_, v_c_99_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21(lean_object* v_00_u03b1_103_, lean_object* v_inst_104_, lean_object* v_c_105_, lean_object* v_init_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(v_inst_104_, v_c_105_, v_init_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permuteList_x21___redArg(lean_object* v_inst_108_, lean_object* v_p_109_, lean_object* v_us_110_){
_start:
{
uint8_t v___x_111_; 
v___x_111_ = l_List_isEmpty___redArg(v_p_109_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_112_ = lean_array_mk(v_us_110_);
v___x_113_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(v_inst_108_, v_p_109_, v___x_112_);
v___x_114_ = lean_array_to_list(v___x_113_);
return v___x_114_;
}
else
{
lean_dec(v_p_109_);
lean_dec(v_inst_108_);
return v_us_110_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_permuteList_x21(lean_object* v_00_u03b1_115_, lean_object* v_inst_116_, lean_object* v_p_117_, lean_object* v_us_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permuteList_x21___redArg(v_inst_116_, v_p_117_, v_us_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_Permutation_reverse_spec__0(lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
if (lean_obj_tag(v_a_120_) == 0)
{
lean_object* v___x_122_; 
v___x_122_ = l_List_reverse___redArg(v_a_121_);
return v___x_122_;
}
else
{
lean_object* v_head_123_; lean_object* v_tail_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_133_; 
v_head_123_ = lean_ctor_get(v_a_120_, 0);
v_tail_124_ = lean_ctor_get(v_a_120_, 1);
v_isSharedCheck_133_ = !lean_is_exclusive(v_a_120_);
if (v_isSharedCheck_133_ == 0)
{
v___x_126_ = v_a_120_;
v_isShared_127_ = v_isSharedCheck_133_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_tail_124_);
lean_inc(v_head_123_);
lean_dec(v_a_120_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_133_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_128_; lean_object* v___x_130_; 
v___x_128_ = l_List_reverse___redArg(v_head_123_);
if (v_isShared_127_ == 0)
{
lean_ctor_set(v___x_126_, 1, v_a_121_);
lean_ctor_set(v___x_126_, 0, v___x_128_);
v___x_130_ = v___x_126_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v___x_128_);
lean_ctor_set(v_reuseFailAlloc_132_, 1, v_a_121_);
v___x_130_ = v_reuseFailAlloc_132_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
v_a_120_ = v_tail_124_;
v_a_121_ = v___x_130_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_reverse(lean_object* v_c_134_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = lean_box(0);
v___x_136_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_Permutation_reverse_spec__0(v_c_134_, v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0___redArg(lean_object* v_a_137_, lean_object* v_b_138_){
_start:
{
lean_object* v_it_u2082_139_; 
v_it_u2082_139_ = lean_ctor_get(v_a_137_, 1);
lean_inc(v_it_u2082_139_);
if (lean_obj_tag(v_it_u2082_139_) == 0)
{
lean_object* v_it_u2081_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_151_; 
v_it_u2081_140_ = lean_ctor_get(v_a_137_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v_a_137_);
if (v_isSharedCheck_151_ == 0)
{
lean_object* v_unused_152_; 
v_unused_152_ = lean_ctor_get(v_a_137_, 1);
lean_dec(v_unused_152_);
v___x_142_ = v_a_137_;
v_isShared_143_ = v_isSharedCheck_151_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_it_u2081_140_);
lean_dec(v_a_137_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_151_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
if (lean_obj_tag(v_it_u2081_140_) == 0)
{
lean_del_object(v___x_142_);
return v_b_138_;
}
else
{
lean_object* v_head_144_; lean_object* v_tail_145_; lean_object* v___x_146_; lean_object* v___x_148_; 
v_head_144_ = lean_ctor_get(v_it_u2081_140_, 0);
lean_inc(v_head_144_);
v_tail_145_ = lean_ctor_get(v_it_u2081_140_, 1);
lean_inc(v_tail_145_);
lean_dec_ref_known(v_it_u2081_140_, 2);
v___x_146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_146_, 0, v_head_144_);
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 1, v___x_146_);
lean_ctor_set(v___x_142_, 0, v_tail_145_);
v___x_148_ = v___x_142_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v_tail_145_);
lean_ctor_set(v_reuseFailAlloc_150_, 1, v___x_146_);
v___x_148_ = v_reuseFailAlloc_150_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
v_a_137_ = v___x_148_;
goto _start;
}
}
}
}
else
{
lean_object* v_val_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_187_; 
v_val_153_ = lean_ctor_get(v_it_u2082_139_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v_it_u2082_139_);
if (v_isSharedCheck_187_ == 0)
{
v___x_155_ = v_it_u2082_139_;
v_isShared_156_ = v_isSharedCheck_187_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_val_153_);
lean_dec(v_it_u2082_139_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_187_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
if (lean_obj_tag(v_val_153_) == 0)
{
lean_object* v_it_u2081_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_166_; 
lean_del_object(v___x_155_);
v_it_u2081_157_ = lean_ctor_get(v_a_137_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v_a_137_);
if (v_isSharedCheck_166_ == 0)
{
lean_object* v_unused_167_; 
v_unused_167_ = lean_ctor_get(v_a_137_, 1);
lean_dec(v_unused_167_);
v___x_159_ = v_a_137_;
v_isShared_160_ = v_isSharedCheck_166_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_it_u2081_157_);
lean_dec(v_a_137_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_166_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_161_; lean_object* v___x_163_; 
v___x_161_ = lean_box(0);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 1, v___x_161_);
v___x_163_ = v___x_159_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_it_u2081_157_);
lean_ctor_set(v_reuseFailAlloc_165_, 1, v___x_161_);
v___x_163_ = v_reuseFailAlloc_165_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
v_a_137_ = v___x_163_;
goto _start;
}
}
}
else
{
lean_object* v_it_u2081_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_185_; 
v_it_u2081_168_ = lean_ctor_get(v_a_137_, 0);
v_isSharedCheck_185_ = !lean_is_exclusive(v_a_137_);
if (v_isSharedCheck_185_ == 0)
{
lean_object* v_unused_186_; 
v_unused_186_ = lean_ctor_get(v_a_137_, 1);
lean_dec(v_unused_186_);
v___x_170_ = v_a_137_;
v_isShared_171_ = v_isSharedCheck_185_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_it_u2081_168_);
lean_dec(v_a_137_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_185_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v_head_172_; lean_object* v_tail_173_; lean_object* v___x_175_; 
v_head_172_ = lean_ctor_get(v_val_153_, 0);
lean_inc(v_head_172_);
v_tail_173_ = lean_ctor_get(v_val_153_, 1);
lean_inc(v_tail_173_);
lean_dec_ref_known(v_val_153_, 2);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 0, v_tail_173_);
v___x_175_ = v___x_155_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v_tail_173_);
v___x_175_ = v_reuseFailAlloc_184_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
lean_object* v___x_177_; 
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 1, v___x_175_);
v___x_177_ = v___x_170_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v_it_u2081_168_);
lean_ctor_set(v_reuseFailAlloc_183_, 1, v___x_175_);
v___x_177_ = v_reuseFailAlloc_183_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_178_ = lean_unsigned_to_nat(1u);
v___x_179_ = lean_nat_add(v_head_172_, v___x_178_);
lean_dec(v_head_172_);
v___x_180_ = lean_nat_dec_le(v_b_138_, v___x_179_);
if (v___x_180_ == 0)
{
lean_dec(v___x_179_);
v_a_137_ = v___x_177_;
goto _start;
}
else
{
lean_dec(v_b_138_);
v_a_137_ = v___x_177_;
v_b_138_ = v___x_179_;
goto _start;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_range(lean_object* v_p_188_){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_189_ = lean_unsigned_to_nat(0u);
v___x_190_ = lean_box(0);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v_p_188_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0___redArg(v___x_191_, v___x_189_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0(lean_object* v_inst_193_, lean_object* v_R_194_, lean_object* v_a_195_, lean_object* v_b_196_, lean_object* v_c_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0___redArg(v_a_195_, v_b_196_);
return v___x_198_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg(lean_object* v_xs_199_, lean_object* v_ys_200_, lean_object* v_x_201_){
_start:
{
lean_object* v_zero_202_; uint8_t v_isZero_203_; 
v_zero_202_ = lean_unsigned_to_nat(0u);
v_isZero_203_ = lean_nat_dec_eq(v_x_201_, v_zero_202_);
if (v_isZero_203_ == 1)
{
lean_dec(v_x_201_);
return v_isZero_203_;
}
else
{
lean_object* v_one_204_; lean_object* v_n_205_; lean_object* v___x_206_; lean_object* v___x_207_; uint8_t v___x_208_; 
v_one_204_ = lean_unsigned_to_nat(1u);
v_n_205_ = lean_nat_sub(v_x_201_, v_one_204_);
lean_dec(v_x_201_);
v___x_206_ = lean_array_fget_borrowed(v_xs_199_, v_n_205_);
v___x_207_ = lean_array_fget_borrowed(v_ys_200_, v_n_205_);
v___x_208_ = lean_nat_dec_eq(v___x_206_, v___x_207_);
if (v___x_208_ == 0)
{
lean_dec(v_n_205_);
return v___x_208_;
}
else
{
v_x_201_ = v_n_205_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg___boxed(lean_object* v_xs_210_, lean_object* v_ys_211_, lean_object* v_x_212_){
_start:
{
uint8_t v_res_213_; lean_object* v_r_214_; 
v_res_213_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg(v_xs_210_, v_ys_211_, v_x_212_);
lean_dec_ref(v_ys_211_);
lean_dec_ref(v_xs_210_);
v_r_214_ = lean_box(v_res_213_);
return v_r_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0___redArg(lean_object* v_a_215_, lean_object* v_b_216_){
_start:
{
lean_object* v_next_217_; 
v_next_217_ = lean_ctor_get(v_a_215_, 0);
lean_inc(v_next_217_);
if (lean_obj_tag(v_next_217_) == 0)
{
lean_dec_ref(v_a_215_);
return v_b_216_;
}
else
{
lean_object* v_upperBound_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_238_; 
v_upperBound_218_ = lean_ctor_get(v_a_215_, 1);
v_isSharedCheck_238_ = !lean_is_exclusive(v_a_215_);
if (v_isSharedCheck_238_ == 0)
{
lean_object* v_unused_239_; 
v_unused_239_ = lean_ctor_get(v_a_215_, 0);
lean_dec(v_unused_239_);
v___x_220_ = v_a_215_;
v_isShared_221_ = v_isSharedCheck_238_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_upperBound_218_);
lean_dec(v_a_215_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_238_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v_val_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_237_; 
v_val_222_ = lean_ctor_get(v_next_217_, 0);
v_isSharedCheck_237_ = !lean_is_exclusive(v_next_217_);
if (v_isSharedCheck_237_ == 0)
{
v___x_224_ = v_next_217_;
v_isShared_225_ = v_isSharedCheck_237_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_val_222_);
lean_dec(v_next_217_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_237_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
uint8_t v___x_226_; 
v___x_226_ = lean_nat_dec_lt(v_val_222_, v_upperBound_218_);
if (v___x_226_ == 0)
{
lean_del_object(v___x_224_);
lean_dec(v_val_222_);
lean_del_object(v___x_220_);
lean_dec(v_upperBound_218_);
return v_b_216_;
}
else
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_230_; 
v___x_227_ = lean_unsigned_to_nat(1u);
v___x_228_ = lean_nat_add(v_val_222_, v___x_227_);
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 0, v___x_228_);
v___x_230_ = v___x_224_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v___x_228_);
v___x_230_ = v_reuseFailAlloc_236_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
lean_object* v___x_232_; 
if (v_isShared_221_ == 0)
{
lean_ctor_set(v___x_220_, 0, v___x_230_);
v___x_232_ = v___x_220_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_230_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v_upperBound_218_);
v___x_232_ = v_reuseFailAlloc_235_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
lean_object* v___x_233_; 
v___x_233_ = lean_array_push(v_b_216_, v_val_222_);
v_a_215_ = v___x_232_;
v_b_216_ = v___x_233_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq(lean_object* v_p_u2081_244_, lean_object* v_p_u2082_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
lean_inc(v_p_u2081_244_);
v___x_246_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_range(v_p_u2081_244_);
lean_inc(v_p_u2082_245_);
v___x_247_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_range(v_p_u2082_245_);
v___x_248_ = lean_nat_dec_eq(v___x_246_, v___x_247_);
lean_dec(v___x_247_);
if (v___x_248_ == 0)
{
lean_dec(v___x_246_);
lean_dec(v_p_u2082_245_);
lean_dec(v_p_u2081_244_);
return v___x_248_;
}
else
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v_rangeArr_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; uint8_t v___x_258_; 
v___x_249_ = lean_unsigned_to_nat(0u);
v___x_250_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__0));
v___x_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___x_246_);
v___x_252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___closed__1));
v_rangeArr_253_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0___redArg(v___x_251_, v___x_252_);
lean_inc_ref(v_rangeArr_253_);
v___x_254_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(v___x_249_, v_p_u2081_244_, v_rangeArr_253_);
v___x_255_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(v___x_249_, v_p_u2082_245_, v_rangeArr_253_);
v___x_256_ = lean_array_get_size(v___x_254_);
v___x_257_ = lean_array_get_size(v___x_255_);
v___x_258_ = lean_nat_dec_eq(v___x_256_, v___x_257_);
if (v___x_258_ == 0)
{
lean_dec_ref(v___x_255_);
lean_dec_ref(v___x_254_);
return v___x_258_;
}
else
{
uint8_t v___x_259_; 
v___x_259_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg(v___x_254_, v___x_255_, v___x_256_);
lean_dec_ref(v___x_255_);
lean_dec_ref(v___x_254_);
return v___x_259_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq___boxed(lean_object* v_p_u2081_260_, lean_object* v_p_u2082_261_){
_start:
{
uint8_t v_res_262_; lean_object* v_r_263_; 
v_res_262_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq(v_p_u2081_260_, v_p_u2082_261_);
v_r_263_ = lean_box(v_res_262_);
return v_r_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0(lean_object* v_inst_264_, lean_object* v_R_265_, lean_object* v_a_266_, lean_object* v_b_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__0___redArg(v_a_266_, v_b_267_);
return v___x_268_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1(lean_object* v_xs_269_, lean_object* v_ys_270_, lean_object* v_hsz_271_, lean_object* v_x_272_, lean_object* v_x_273_){
_start:
{
uint8_t v___x_274_; 
v___x_274_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___redArg(v_xs_269_, v_ys_270_, v_x_272_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1___boxed(lean_object* v_xs_275_, lean_object* v_ys_276_, lean_object* v_hsz_277_, lean_object* v_x_278_, lean_object* v_x_279_){
_start:
{
uint8_t v_res_280_; lean_object* v_r_281_; 
v_res_280_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_Permutation_beq_spec__1(v_xs_275_, v_ys_276_, v_hsz_277_, v_x_278_, v_x_279_);
lean_dec_ref(v_ys_276_);
lean_dec_ref(v_xs_275_);
v_r_281_ = lean_box(v_res_280_);
return v_r_281_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty(lean_object* v_r_289_){
_start:
{
lean_object* v_perm_290_; 
v_perm_290_ = lean_ctor_get(v_r_289_, 0);
if (lean_obj_tag(v_perm_290_) == 0)
{
lean_object* v_argReorders_291_; lean_object* v___x_292_; lean_object* v___x_293_; uint8_t v___x_294_; 
v_argReorders_291_ = lean_ctor_get(v_r_289_, 1);
v___x_292_ = lean_array_get_size(v_argReorders_291_);
v___x_293_ = lean_unsigned_to_nat(0u);
v___x_294_ = lean_nat_dec_eq(v___x_292_, v___x_293_);
return v___x_294_;
}
else
{
uint8_t v___x_295_; 
v___x_295_ = 0;
return v___x_295_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty___boxed(lean_object* v_r_296_){
_start:
{
uint8_t v_res_297_; lean_object* v_r_298_; 
v_res_297_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty(v_r_296_);
lean_dec_ref(v_r_296_);
v_r_298_ = lean_box(v_res_297_);
return v_r_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(lean_object* v_inst_299_, lean_object* v_r_300_, lean_object* v_a_301_){
_start:
{
lean_object* v_perm_302_; lean_object* v___x_303_; 
v_perm_302_ = lean_ctor_get(v_r_300_, 0);
lean_inc(v_perm_302_);
lean_dec_ref(v_r_300_);
v___x_303_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_permute_x21___redArg(v_inst_299_, v_perm_302_, v_a_301_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21(lean_object* v_00_u03b1_304_, lean_object* v_inst_305_, lean_object* v_r_306_, lean_object* v_a_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(v_inst_305_, v_r_306_, v_a_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor(lean_object* v_n_309_, lean_object* v_head_310_, lean_object* v_a_311_){
_start:
{
if (lean_obj_tag(v_a_311_) == 0)
{
lean_object* v___x_312_; 
lean_dec(v_head_310_);
v___x_312_ = lean_box(0);
return v___x_312_;
}
else
{
lean_object* v_head_313_; lean_object* v_tail_314_; uint8_t v___x_315_; 
v_head_313_ = lean_ctor_get(v_a_311_, 0);
v_tail_314_ = lean_ctor_get(v_a_311_, 1);
v___x_315_ = lean_nat_dec_eq(v_head_313_, v_n_309_);
if (v___x_315_ == 0)
{
v_a_311_ = v_tail_314_;
goto _start;
}
else
{
lean_object* v___x_317_; 
v___x_317_ = l_List_head_x3f___redArg(v_tail_314_);
if (lean_obj_tag(v___x_317_) == 0)
{
lean_object* v___x_318_; 
v___x_318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_318_, 0, v_head_310_);
return v___x_318_;
}
else
{
lean_dec(v_head_310_);
return v___x_317_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor___boxed(lean_object* v_n_319_, lean_object* v_head_320_, lean_object* v_a_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor(v_n_319_, v_head_320_, v_a_321_);
lean_dec(v_a_321_);
lean_dec(v_n_319_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0(lean_object* v_n_323_, lean_object* v_x_324_){
_start:
{
if (lean_obj_tag(v_x_324_) == 0)
{
lean_object* v___x_325_; 
v___x_325_ = lean_box(0);
return v___x_325_;
}
else
{
lean_object* v_head_326_; lean_object* v_tail_327_; lean_object* v_head_328_; lean_object* v___x_329_; 
v_head_326_ = lean_ctor_get(v_x_324_, 0);
lean_inc(v_head_326_);
v_tail_327_ = lean_ctor_get(v_x_324_, 1);
lean_inc(v_tail_327_);
lean_dec_ref_known(v_x_324_, 2);
v_head_328_ = lean_ctor_get(v_head_326_, 0);
lean_inc(v_head_328_);
v___x_329_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_getCycleSuccessor(v_n_323_, v_head_328_, v_head_326_);
lean_dec(v_head_326_);
if (lean_obj_tag(v___x_329_) == 0)
{
v_x_324_ = v_tail_327_;
goto _start;
}
else
{
lean_dec(v_tail_327_);
return v___x_329_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0___boxed(lean_object* v_n_331_, lean_object* v_x_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0(v_n_331_, v_x_332_);
lean_dec(v_n_331_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle(lean_object* v_r_334_, lean_object* v_n_335_){
_start:
{
lean_object* v_perm_336_; lean_object* v___x_337_; 
v_perm_336_ = lean_ctor_get(v_r_334_, 0);
lean_inc(v_perm_336_);
lean_dec_ref(v_r_334_);
v___x_337_ = lp_mathlib_List_findSome_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle_spec__0(v_n_335_, v_perm_336_);
if (lean_obj_tag(v___x_337_) == 0)
{
lean_inc(v_n_335_);
return v_n_335_;
}
else
{
lean_object* v_val_338_; 
v_val_338_ = lean_ctor_get(v___x_337_, 0);
lean_inc(v_val_338_);
lean_dec_ref_known(v___x_337_, 1);
return v_val_338_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle___boxed(lean_object* v_r_339_, lean_object* v_n_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle(v_r_339_, v_n_340_);
lean_dec(v_n_340_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0(lean_object* v_r_342_, size_t v_sz_343_, size_t v_i_344_, lean_object* v_bs_345_){
_start:
{
uint8_t v___x_346_; 
v___x_346_ = lean_usize_dec_lt(v_i_344_, v_sz_343_);
if (v___x_346_ == 0)
{
lean_dec_ref(v_r_342_);
return v_bs_345_;
}
else
{
lean_object* v_v_347_; lean_object* v_fst_348_; lean_object* v_snd_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_364_; 
v_v_347_ = lean_array_uget(v_bs_345_, v_i_344_);
v_fst_348_ = lean_ctor_get(v_v_347_, 0);
v_snd_349_ = lean_ctor_get(v_v_347_, 1);
v_isSharedCheck_364_ = !lean_is_exclusive(v_v_347_);
if (v_isSharedCheck_364_ == 0)
{
v___x_351_ = v_v_347_;
v_isShared_352_ = v_isSharedCheck_364_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_snd_349_);
lean_inc(v_fst_348_);
lean_dec(v_v_347_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_364_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___x_353_; lean_object* v_bs_x27_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_358_; 
v___x_353_ = lean_unsigned_to_nat(0u);
v_bs_x27_354_ = lean_array_uset(v_bs_345_, v_i_344_, v___x_353_);
lean_inc_ref(v_r_342_);
v___x_355_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_ArgReorder_reverse_permuteSingle(v_r_342_, v_fst_348_);
lean_dec(v_fst_348_);
v___x_356_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_reverse(v_snd_349_);
if (v_isShared_352_ == 0)
{
lean_ctor_set(v___x_351_, 1, v___x_356_);
lean_ctor_set(v___x_351_, 0, v___x_355_);
v___x_358_ = v___x_351_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v___x_355_);
lean_ctor_set(v_reuseFailAlloc_363_, 1, v___x_356_);
v___x_358_ = v_reuseFailAlloc_363_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
size_t v___x_359_; size_t v___x_360_; lean_object* v___x_361_; 
v___x_359_ = ((size_t)1ULL);
v___x_360_ = lean_usize_add(v_i_344_, v___x_359_);
v___x_361_ = lean_array_uset(v_bs_x27_354_, v_i_344_, v___x_358_);
v_i_344_ = v___x_360_;
v_bs_345_ = v___x_361_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_reverse(lean_object* v_r_365_){
_start:
{
lean_object* v_perm_366_; lean_object* v_argReorders_367_; lean_object* v___x_368_; size_t v_sz_369_; size_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_perm_366_ = lean_ctor_get(v_r_365_, 0);
v_argReorders_367_ = lean_ctor_get(v_r_365_, 1);
lean_inc_ref(v_argReorders_367_);
lean_inc(v_perm_366_);
v___x_368_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_reverse(v_perm_366_);
v_sz_369_ = lean_array_size(v_argReorders_367_);
v___x_370_ = ((size_t)0ULL);
v___x_371_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0(v_r_365_, v_sz_369_, v___x_370_, v_argReorders_367_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_368_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0___boxed(lean_object* v_r_373_, lean_object* v_sz_374_, lean_object* v_i_375_, lean_object* v_bs_376_){
_start:
{
size_t v_sz_boxed_377_; size_t v_i_boxed_378_; lean_object* v_res_379_; 
v_sz_boxed_377_ = lean_unbox_usize(v_sz_374_);
lean_dec(v_sz_374_);
v_i_boxed_378_ = lean_unbox_usize(v_i_375_);
lean_dec(v_i_375_);
v_res_379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_reverse_spec__0(v_r_373_, v_sz_boxed_377_, v_i_boxed_378_, v_bs_376_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Array_map__unattach_match__1_splitter___redArg(lean_object* v_x_380_, lean_object* v_h__1_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lean_apply_2(v_h__1_381_, v_x_380_, lean_box(0));
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Array_map__unattach_match__1_splitter(lean_object* v_00_u03b1_383_, lean_object* v_P_384_, lean_object* v_motive_385_, lean_object* v_x_386_, lean_object* v_h__1_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lean_apply_2(v_h__1_387_, v_x_386_, lean_box(0));
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0(lean_object* v_as_389_, size_t v_i_390_, size_t v_stop_391_, lean_object* v_b_392_){
_start:
{
lean_object* v___y_394_; uint8_t v___x_398_; 
v___x_398_ = lean_usize_dec_eq(v_i_390_, v_stop_391_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; lean_object* v_fst_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_399_ = lean_array_uget_borrowed(v_as_389_, v_i_390_);
v_fst_400_ = lean_ctor_get(v___x_399_, 0);
v___x_401_ = lean_unsigned_to_nat(1u);
v___x_402_ = lean_nat_add(v_fst_400_, v___x_401_);
v___x_403_ = lean_nat_dec_le(v_b_392_, v___x_402_);
if (v___x_403_ == 0)
{
lean_dec(v___x_402_);
v___y_394_ = v_b_392_;
goto v___jp_393_;
}
else
{
lean_dec(v_b_392_);
v___y_394_ = v___x_402_;
goto v___jp_393_;
}
}
else
{
return v_b_392_;
}
v___jp_393_:
{
size_t v___x_395_; size_t v___x_396_; 
v___x_395_ = ((size_t)1ULL);
v___x_396_ = lean_usize_add(v_i_390_, v___x_395_);
v_i_390_ = v___x_396_;
v_b_392_ = v___y_394_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0___boxed(lean_object* v_as_404_, lean_object* v_i_405_, lean_object* v_stop_406_, lean_object* v_b_407_){
_start:
{
size_t v_i_boxed_408_; size_t v_stop_boxed_409_; lean_object* v_res_410_; 
v_i_boxed_408_ = lean_unbox_usize(v_i_405_);
lean_dec(v_i_405_);
v_stop_boxed_409_ = lean_unbox_usize(v_stop_406_);
lean_dec(v_stop_406_);
v_res_410_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0(v_as_404_, v_i_boxed_408_, v_stop_boxed_409_, v_b_407_);
lean_dec_ref(v_as_404_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_range(lean_object* v_r_411_){
_start:
{
lean_object* v_perm_412_; lean_object* v_argReorders_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_432_; 
v_perm_412_ = lean_ctor_get(v_r_411_, 0);
v_argReorders_413_ = lean_ctor_get(v_r_411_, 1);
v_isSharedCheck_432_ = !lean_is_exclusive(v_r_411_);
if (v_isSharedCheck_432_ == 0)
{
v___x_415_ = v_r_411_;
v_isShared_416_ = v_isSharedCheck_432_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_argReorders_413_);
lean_inc(v_perm_412_);
lean_dec(v_r_411_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_432_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_420_; 
v___x_417_ = lean_unsigned_to_nat(0u);
v___x_418_ = lean_box(0);
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 1, v___x_418_);
v___x_420_ = v___x_415_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_perm_412_);
lean_ctor_set(v_reuseFailAlloc_431_, 1, v___x_418_);
v___x_420_ = v_reuseFailAlloc_431_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
lean_object* v___x_421_; lean_object* v___x_422_; uint8_t v___x_423_; 
v___x_421_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_Permutation_range_spec__0___redArg(v___x_420_, v___x_417_);
v___x_422_ = lean_array_get_size(v_argReorders_413_);
v___x_423_ = lean_nat_dec_lt(v___x_417_, v___x_422_);
if (v___x_423_ == 0)
{
lean_dec_ref(v_argReorders_413_);
return v___x_421_;
}
else
{
uint8_t v___x_424_; 
v___x_424_ = lean_nat_dec_le(v___x_422_, v___x_422_);
if (v___x_424_ == 0)
{
if (v___x_423_ == 0)
{
lean_dec_ref(v_argReorders_413_);
return v___x_421_;
}
else
{
size_t v___x_425_; size_t v___x_426_; lean_object* v___x_427_; 
v___x_425_ = ((size_t)0ULL);
v___x_426_ = lean_usize_of_nat(v___x_422_);
v___x_427_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0(v_argReorders_413_, v___x_425_, v___x_426_, v___x_421_);
lean_dec_ref(v_argReorders_413_);
return v___x_427_;
}
}
else
{
size_t v___x_428_; size_t v___x_429_; lean_object* v___x_430_; 
v___x_428_ = ((size_t)0ULL);
v___x_429_ = lean_usize_of_nat(v___x_422_);
v___x_430_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_ArgReorder_range_spec__0(v_argReorders_413_, v___x_428_, v___x_429_, v___x_421_);
lean_dec_ref(v_argReorders_413_);
return v___x_430_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg(lean_object* v_xs_433_, lean_object* v_ys_434_, lean_object* v_x_435_){
_start:
{
lean_object* v_zero_436_; uint8_t v_isZero_437_; 
v_zero_436_ = lean_unsigned_to_nat(0u);
v_isZero_437_ = lean_nat_dec_eq(v_x_435_, v_zero_436_);
if (v_isZero_437_ == 1)
{
lean_dec(v_x_435_);
return v_isZero_437_;
}
else
{
lean_object* v_one_438_; lean_object* v_n_439_; uint8_t v___y_441_; lean_object* v___x_443_; lean_object* v_fst_444_; lean_object* v_snd_445_; lean_object* v___x_446_; lean_object* v_fst_447_; lean_object* v_snd_448_; uint8_t v___x_449_; 
v_one_438_ = lean_unsigned_to_nat(1u);
v_n_439_ = lean_nat_sub(v_x_435_, v_one_438_);
lean_dec(v_x_435_);
v___x_443_ = lean_array_fget_borrowed(v_xs_433_, v_n_439_);
v_fst_444_ = lean_ctor_get(v___x_443_, 0);
v_snd_445_ = lean_ctor_get(v___x_443_, 1);
v___x_446_ = lean_array_fget_borrowed(v_ys_434_, v_n_439_);
v_fst_447_ = lean_ctor_get(v___x_446_, 0);
v_snd_448_ = lean_ctor_get(v___x_446_, 1);
v___x_449_ = lean_nat_dec_eq(v_fst_444_, v_fst_447_);
if (v___x_449_ == 0)
{
v___y_441_ = v___x_449_;
goto v___jp_440_;
}
else
{
uint8_t v___x_450_; 
lean_inc(v_snd_448_);
lean_inc(v_snd_445_);
v___x_450_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq(v_snd_445_, v_snd_448_);
v___y_441_ = v___x_450_;
goto v___jp_440_;
}
v___jp_440_:
{
if (v___y_441_ == 0)
{
lean_dec(v_n_439_);
return v___y_441_;
}
else
{
v_x_435_ = v_n_439_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq(lean_object* v_r_u2081_451_, lean_object* v_r_u2082_452_){
_start:
{
lean_object* v_perm_453_; lean_object* v_argReorders_454_; lean_object* v_perm_455_; lean_object* v_argReorders_456_; uint8_t v___x_457_; 
v_perm_453_ = lean_ctor_get(v_r_u2081_451_, 0);
lean_inc(v_perm_453_);
v_argReorders_454_ = lean_ctor_get(v_r_u2081_451_, 1);
lean_inc_ref(v_argReorders_454_);
lean_dec_ref(v_r_u2081_451_);
v_perm_455_ = lean_ctor_get(v_r_u2082_452_, 0);
lean_inc(v_perm_455_);
v_argReorders_456_ = lean_ctor_get(v_r_u2082_452_, 1);
lean_inc_ref(v_argReorders_456_);
lean_dec_ref(v_r_u2082_452_);
v___x_457_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_beq(v_perm_453_, v_perm_455_);
if (v___x_457_ == 0)
{
lean_dec_ref(v_argReorders_456_);
lean_dec_ref(v_argReorders_454_);
return v___x_457_;
}
else
{
lean_object* v___x_458_; lean_object* v___x_459_; uint8_t v___x_460_; 
v___x_458_ = lean_array_get_size(v_argReorders_454_);
v___x_459_ = lean_array_get_size(v_argReorders_456_);
v___x_460_ = lean_nat_dec_eq(v___x_458_, v___x_459_);
if (v___x_460_ == 0)
{
lean_dec_ref(v_argReorders_456_);
lean_dec_ref(v_argReorders_454_);
return v___x_460_;
}
else
{
uint8_t v___x_461_; 
v___x_461_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg(v_argReorders_454_, v_argReorders_456_, v___x_458_);
lean_dec_ref(v_argReorders_456_);
lean_dec_ref(v_argReorders_454_);
return v___x_461_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq___boxed(lean_object* v_r_u2081_462_, lean_object* v_r_u2082_463_){
_start:
{
uint8_t v_res_464_; lean_object* v_r_465_; 
v_res_464_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_beq(v_r_u2081_462_, v_r_u2082_463_);
v_r_465_ = lean_box(v_res_464_);
return v_r_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg___boxed(lean_object* v_xs_466_, lean_object* v_ys_467_, lean_object* v_x_468_){
_start:
{
uint8_t v_res_469_; lean_object* v_r_470_; 
v_res_469_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg(v_xs_466_, v_ys_467_, v_x_468_);
lean_dec_ref(v_ys_467_);
lean_dec_ref(v_xs_466_);
v_r_470_ = lean_box(v_res_469_);
return v_r_470_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0(lean_object* v_xs_471_, lean_object* v_ys_472_, lean_object* v_hsz_473_, lean_object* v_x_474_, lean_object* v_x_475_){
_start:
{
uint8_t v___x_476_; 
v___x_476_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___redArg(v_xs_471_, v_ys_472_, v_x_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0___boxed(lean_object* v_xs_477_, lean_object* v_ys_478_, lean_object* v_hsz_479_, lean_object* v_x_480_, lean_object* v_x_481_){
_start:
{
uint8_t v_res_482_; lean_object* v_r_483_; 
v_res_482_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_Translate_ArgReorder_beq_spec__0(v_xs_477_, v_ys_478_, v_hsz_479_, v_x_480_, v_x_481_);
lean_dec_ref(v_ys_478_);
lean_dec_ref(v_xs_477_);
v_r_483_ = lean_box(v_res_482_);
return v_r_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__0(lean_object* v_a_486_, lean_object* v_a_487_){
_start:
{
if (lean_obj_tag(v_a_486_) == 0)
{
lean_object* v___x_488_; 
v___x_488_ = l_List_reverse___redArg(v_a_487_);
return v___x_488_;
}
else
{
lean_object* v_head_489_; lean_object* v_tail_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_501_; 
v_head_489_ = lean_ctor_get(v_a_486_, 0);
v_tail_490_ = lean_ctor_get(v_a_486_, 1);
v_isSharedCheck_501_ = !lean_is_exclusive(v_a_486_);
if (v_isSharedCheck_501_ == 0)
{
v___x_492_ = v_a_486_;
v_isShared_493_ = v_isSharedCheck_501_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_tail_490_);
lean_inc(v_head_489_);
lean_dec(v_a_486_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_501_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_498_; 
v___x_494_ = lean_unsigned_to_nat(1u);
v___x_495_ = lean_nat_add(v_head_489_, v___x_494_);
lean_dec(v_head_489_);
v___x_496_ = l_Nat_reprFast(v___x_495_);
if (v_isShared_493_ == 0)
{
lean_ctor_set(v___x_492_, 1, v_a_487_);
lean_ctor_set(v___x_492_, 0, v___x_496_);
v___x_498_ = v___x_492_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_496_);
lean_ctor_set(v_reuseFailAlloc_500_, 1, v_a_487_);
v___x_498_ = v_reuseFailAlloc_500_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
v_a_486_ = v_tail_490_;
v_a_487_ = v___x_498_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1(lean_object* v_a_503_, lean_object* v_a_504_){
_start:
{
if (lean_obj_tag(v_a_503_) == 0)
{
lean_object* v___x_505_; 
v___x_505_ = l_List_reverse___redArg(v_a_504_);
return v___x_505_;
}
else
{
lean_object* v_head_506_; lean_object* v_tail_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_519_; 
v_head_506_ = lean_ctor_get(v_a_503_, 0);
v_tail_507_ = lean_ctor_get(v_a_503_, 1);
v_isSharedCheck_519_ = !lean_is_exclusive(v_a_503_);
if (v_isSharedCheck_519_ == 0)
{
v___x_509_ = v_a_503_;
v_isShared_510_ = v_isSharedCheck_519_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_tail_507_);
lean_inc(v_head_506_);
lean_dec(v_a_503_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_519_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_511_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1___closed__0));
v___x_512_ = lean_box(0);
v___x_513_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__0(v_head_506_, v___x_512_);
v___x_514_ = l_String_intercalate(v___x_511_, v___x_513_);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v_a_504_);
lean_ctor_set(v___x_509_, 0, v___x_514_);
v___x_516_ = v___x_509_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v___x_514_);
lean_ctor_set(v_reuseFailAlloc_518_, 1, v_a_504_);
v___x_516_ = v_reuseFailAlloc_518_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
v_a_503_ = v_tail_507_;
v_a_504_ = v___x_516_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2(size_t v_sz_523_, size_t v_i_524_, lean_object* v_bs_525_){
_start:
{
uint8_t v___x_526_; 
v___x_526_ = lean_usize_dec_lt(v_i_524_, v_sz_523_);
if (v___x_526_ == 0)
{
return v_bs_525_;
}
else
{
lean_object* v_v_527_; lean_object* v_fst_528_; lean_object* v_snd_529_; lean_object* v___x_530_; lean_object* v_bs_x27_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; size_t v___x_541_; size_t v___x_542_; lean_object* v___x_543_; 
v_v_527_ = lean_array_uget_borrowed(v_bs_525_, v_i_524_);
v_fst_528_ = lean_ctor_get(v_v_527_, 0);
lean_inc(v_fst_528_);
v_snd_529_ = lean_ctor_get(v_v_527_, 1);
lean_inc(v_snd_529_);
v___x_530_ = lean_unsigned_to_nat(0u);
v_bs_x27_531_ = lean_array_uset(v_bs_525_, v_i_524_, v___x_530_);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_nat_add(v_fst_528_, v___x_532_);
lean_dec(v_fst_528_);
v___x_534_ = l_Nat_reprFast(v___x_533_);
v___x_535_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__0));
v___x_536_ = lean_string_append(v___x_534_, v___x_535_);
v___x_537_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_snd_529_);
v___x_538_ = lean_string_append(v___x_536_, v___x_537_);
lean_dec_ref(v___x_537_);
v___x_539_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1));
v___x_540_ = lean_string_append(v___x_538_, v___x_539_);
v___x_541_ = ((size_t)1ULL);
v___x_542_ = lean_usize_add(v_i_524_, v___x_541_);
v___x_543_ = lean_array_uset(v_bs_x27_531_, v_i_524_, v___x_540_);
v_i_524_ = v___x_542_;
v_bs_525_ = v___x_543_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(lean_object* v_r_545_){
_start:
{
lean_object* v_perm_546_; lean_object* v_argReorders_547_; lean_object* v___x_548_; lean_object* v_perm_549_; size_t v_sz_550_; size_t v___x_551_; lean_object* v_argReorders_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v_perm_546_ = lean_ctor_get(v_r_545_, 0);
lean_inc(v_perm_546_);
v_argReorders_547_ = lean_ctor_get(v_r_545_, 1);
lean_inc_ref(v_argReorders_547_);
lean_dec_ref(v_r_545_);
v___x_548_ = lean_box(0);
v_perm_549_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__1(v_perm_546_, v___x_548_);
v_sz_550_ = lean_array_size(v_argReorders_547_);
v___x_551_ = ((size_t)0ULL);
v_argReorders_552_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2(v_sz_550_, v___x_551_, v_argReorders_547_);
v___x_553_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString___closed__0));
v___x_554_ = lean_array_to_list(v_argReorders_552_);
v___x_555_ = l_List_appendTR___redArg(v_perm_549_, v___x_554_);
v___x_556_ = l_String_intercalate(v___x_553_, v___x_555_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___boxed(lean_object* v_sz_557_, lean_object* v_i_558_, lean_object* v_bs_559_){
_start:
{
size_t v_sz_boxed_560_; size_t v_i_boxed_561_; lean_object* v_res_562_; 
v_sz_boxed_560_ = lean_unbox_usize(v_sz_557_);
lean_dec(v_sz_557_);
v_i_boxed_561_ = lean_unbox_usize(v_i_558_);
lean_dec(v_i_558_);
v_res_562_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2(v_sz_boxed_560_, v_i_boxed_561_, v_bs_559_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__List_map__unattach_match__1_splitter___redArg(lean_object* v_x_563_, lean_object* v_h__1_564_){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = lean_apply_2(v_h__1_564_, v_x_563_, lean_box(0));
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__List_map__unattach_match__1_splitter(lean_object* v_00_u03b1_566_, lean_object* v_P_567_, lean_object* v_motive_568_, lean_object* v_x_569_, lean_object* v_h__1_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lean_apply_2(v_h__1_570_, v_x_569_, lean_box(0));
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_instToMessageData___lam__0(lean_object* v_x_574_){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_575_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_x_574_);
v___x_576_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_576_, 0, v___x_575_);
v___x_577_ = l_Lean_MessageData_ofFormat(v___x_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_Reorder_reverse(lean_object* v_r_580_){
_start:
{
lean_object* v_univReorder_581_; lean_object* v_reorder_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_591_; 
v_univReorder_581_ = lean_ctor_get(v_r_580_, 0);
v_reorder_582_ = lean_ctor_get(v_r_580_, 1);
v_isSharedCheck_591_ = !lean_is_exclusive(v_r_580_);
if (v_isSharedCheck_591_ == 0)
{
v___x_584_ = v_r_580_;
v_isShared_585_ = v_isSharedCheck_591_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_reorder_582_);
lean_inc(v_univReorder_581_);
lean_dec(v_r_580_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_591_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_589_; 
v___x_586_ = lp_mathlib_Mathlib_Tactic_Translate_Permutation_reverse(v_univReorder_581_);
v___x_587_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_reverse(v_reorder_582_);
if (v_isShared_585_ == 0)
{
lean_ctor_set(v___x_584_, 1, v___x_587_);
lean_ctor_set(v___x_584_, 0, v___x_586_);
v___x_589_ = v___x_584_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_590_, 0, v___x_586_);
lean_ctor_set(v_reuseFailAlloc_590_, 1, v___x_587_);
v___x_589_ = v_reuseFailAlloc_590_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
return v___x_589_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(lean_object* v_bis_592_, lean_object* v_e_593_){
_start:
{
if (lean_obj_tag(v_bis_592_) == 1)
{
switch(lean_obj_tag(v_e_593_))
{
case 7:
{
lean_object* v_head_594_; lean_object* v_tail_595_; lean_object* v_binderName_596_; lean_object* v_binderType_597_; lean_object* v_body_598_; lean_object* v___x_599_; uint8_t v___x_600_; lean_object* v___x_601_; 
v_head_594_ = lean_ctor_get(v_bis_592_, 0);
v_tail_595_ = lean_ctor_get(v_bis_592_, 1);
v_binderName_596_ = lean_ctor_get(v_e_593_, 0);
lean_inc(v_binderName_596_);
v_binderType_597_ = lean_ctor_get(v_e_593_, 1);
lean_inc_ref(v_binderType_597_);
v_body_598_ = lean_ctor_get(v_e_593_, 2);
lean_inc_ref(v_body_598_);
lean_dec_ref_known(v_e_593_, 3);
v___x_599_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(v_tail_595_, v_body_598_);
v___x_600_ = lean_unbox(v_head_594_);
v___x_601_ = l_Lean_Expr_forallE___override(v_binderName_596_, v_binderType_597_, v___x_599_, v___x_600_);
return v___x_601_;
}
case 6:
{
lean_object* v_head_602_; lean_object* v_tail_603_; lean_object* v_binderName_604_; lean_object* v_binderType_605_; lean_object* v_body_606_; lean_object* v___x_607_; uint8_t v___x_608_; lean_object* v___x_609_; 
v_head_602_ = lean_ctor_get(v_bis_592_, 0);
v_tail_603_ = lean_ctor_get(v_bis_592_, 1);
v_binderName_604_ = lean_ctor_get(v_e_593_, 0);
lean_inc(v_binderName_604_);
v_binderType_605_ = lean_ctor_get(v_e_593_, 1);
lean_inc_ref(v_binderType_605_);
v_body_606_ = lean_ctor_get(v_e_593_, 2);
lean_inc_ref(v_body_606_);
lean_dec_ref_known(v_e_593_, 3);
v___x_607_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(v_tail_603_, v_body_606_);
v___x_608_ = lean_unbox(v_head_602_);
v___x_609_ = l_Lean_Expr_lam___override(v_binderName_604_, v_binderType_605_, v___x_607_, v___x_608_);
return v___x_609_;
}
default: 
{
return v_e_593_;
}
}
}
else
{
return v_e_593_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos___boxed(lean_object* v_bis_610_, lean_object* v_e_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(v_bis_610_, v_e_611_);
lean_dec(v_bis_610_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4(lean_object* v_msgData_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_){
_start:
{
lean_object* v___x_619_; lean_object* v_env_620_; lean_object* v___x_621_; lean_object* v_mctx_622_; lean_object* v_lctx_623_; lean_object* v_options_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_619_ = lean_st_ref_get(v___y_617_);
v_env_620_ = lean_ctor_get(v___x_619_, 0);
lean_inc_ref(v_env_620_);
lean_dec(v___x_619_);
v___x_621_ = lean_st_ref_get(v___y_615_);
v_mctx_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc_ref(v_mctx_622_);
lean_dec(v___x_621_);
v_lctx_623_ = lean_ctor_get(v___y_614_, 2);
v_options_624_ = lean_ctor_get(v___y_616_, 2);
lean_inc_ref(v_options_624_);
lean_inc_ref(v_lctx_623_);
v___x_625_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_625_, 0, v_env_620_);
lean_ctor_set(v___x_625_, 1, v_mctx_622_);
lean_ctor_set(v___x_625_, 2, v_lctx_623_);
lean_ctor_set(v___x_625_, 3, v_options_624_);
v___x_626_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_625_);
lean_ctor_set(v___x_626_, 1, v_msgData_613_);
v___x_627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_627_, 0, v___x_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4___boxed(lean_object* v_msgData_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4(v_msgData_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_);
lean_dec(v___y_632_);
lean_dec_ref(v___y_631_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(lean_object* v_msg_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
lean_object* v_ref_641_; lean_object* v___x_642_; lean_object* v_a_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_651_; 
v_ref_641_ = lean_ctor_get(v___y_638_, 5);
v___x_642_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3_spec__4(v_msg_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
v_a_643_ = lean_ctor_get(v___x_642_, 0);
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_642_);
if (v_isSharedCheck_651_ == 0)
{
v___x_645_ = v___x_642_;
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_a_643_);
lean_dec(v___x_642_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_647_; lean_object* v___x_649_; 
lean_inc(v_ref_641_);
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v_ref_641_);
lean_ctor_set(v___x_647_, 1, v_a_643_);
if (v_isShared_646_ == 0)
{
lean_ctor_set_tag(v___x_645_, 1);
lean_ctor_set(v___x_645_, 0, v___x_647_);
v___x_649_ = v___x_645_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_647_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg___boxed(lean_object* v_msg_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v_msg_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_653_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8___redArg(lean_object* v_x_659_, lean_object* v_x_660_, lean_object* v_x_661_, lean_object* v_x_662_){
_start:
{
lean_object* v_ks_663_; lean_object* v_vs_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_688_; 
v_ks_663_ = lean_ctor_get(v_x_659_, 0);
v_vs_664_ = lean_ctor_get(v_x_659_, 1);
v_isSharedCheck_688_ = !lean_is_exclusive(v_x_659_);
if (v_isSharedCheck_688_ == 0)
{
v___x_666_ = v_x_659_;
v_isShared_667_ = v_isSharedCheck_688_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_vs_664_);
lean_inc(v_ks_663_);
lean_dec(v_x_659_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_688_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_668_; uint8_t v___x_669_; 
v___x_668_ = lean_array_get_size(v_ks_663_);
v___x_669_ = lean_nat_dec_lt(v_x_660_, v___x_668_);
if (v___x_669_ == 0)
{
lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_673_; 
lean_dec(v_x_660_);
v___x_670_ = lean_array_push(v_ks_663_, v_x_661_);
v___x_671_ = lean_array_push(v_vs_664_, v_x_662_);
if (v_isShared_667_ == 0)
{
lean_ctor_set(v___x_666_, 1, v___x_671_);
lean_ctor_set(v___x_666_, 0, v___x_670_);
v___x_673_ = v___x_666_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v___x_670_);
lean_ctor_set(v_reuseFailAlloc_674_, 1, v___x_671_);
v___x_673_ = v_reuseFailAlloc_674_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
return v___x_673_;
}
}
else
{
lean_object* v_k_x27_675_; uint8_t v___x_676_; 
v_k_x27_675_ = lean_array_fget_borrowed(v_ks_663_, v_x_660_);
v___x_676_ = l_Lean_instBEqMVarId_beq(v_x_661_, v_k_x27_675_);
if (v___x_676_ == 0)
{
lean_object* v___x_678_; 
if (v_isShared_667_ == 0)
{
v___x_678_ = v___x_666_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v_ks_663_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_vs_664_);
v___x_678_ = v_reuseFailAlloc_682_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_679_ = lean_unsigned_to_nat(1u);
v___x_680_ = lean_nat_add(v_x_660_, v___x_679_);
lean_dec(v_x_660_);
v_x_659_ = v___x_678_;
v_x_660_ = v___x_680_;
goto _start;
}
}
else
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_686_; 
v___x_683_ = lean_array_fset(v_ks_663_, v_x_660_, v_x_661_);
v___x_684_ = lean_array_fset(v_vs_664_, v_x_660_, v_x_662_);
lean_dec(v_x_660_);
if (v_isShared_667_ == 0)
{
lean_ctor_set(v___x_666_, 1, v___x_684_);
lean_ctor_set(v___x_666_, 0, v___x_683_);
v___x_686_ = v___x_666_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_683_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v___x_684_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7___redArg(lean_object* v_n_689_, lean_object* v_k_690_, lean_object* v_v_691_){
_start:
{
lean_object* v___x_692_; lean_object* v___x_693_; 
v___x_692_ = lean_unsigned_to_nat(0u);
v___x_693_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8___redArg(v_n_689_, v___x_692_, v_k_690_, v_v_691_);
return v___x_693_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(lean_object* v_x_695_, size_t v_x_696_, size_t v_x_697_, lean_object* v_x_698_, lean_object* v_x_699_){
_start:
{
if (lean_obj_tag(v_x_695_) == 0)
{
lean_object* v_es_700_; size_t v___x_701_; size_t v___x_702_; lean_object* v_j_703_; lean_object* v___x_704_; uint8_t v___x_705_; 
v_es_700_ = lean_ctor_get(v_x_695_, 0);
v___x_701_ = ((size_t)31ULL);
v___x_702_ = lean_usize_land(v_x_696_, v___x_701_);
v_j_703_ = lean_usize_to_nat(v___x_702_);
v___x_704_ = lean_array_get_size(v_es_700_);
v___x_705_ = lean_nat_dec_lt(v_j_703_, v___x_704_);
if (v___x_705_ == 0)
{
lean_dec(v_j_703_);
lean_dec(v_x_699_);
lean_dec(v_x_698_);
return v_x_695_;
}
else
{
lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_744_; 
lean_inc_ref(v_es_700_);
v_isSharedCheck_744_ = !lean_is_exclusive(v_x_695_);
if (v_isSharedCheck_744_ == 0)
{
lean_object* v_unused_745_; 
v_unused_745_ = lean_ctor_get(v_x_695_, 0);
lean_dec(v_unused_745_);
v___x_707_ = v_x_695_;
v_isShared_708_ = v_isSharedCheck_744_;
goto v_resetjp_706_;
}
else
{
lean_dec(v_x_695_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_744_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v_v_709_; lean_object* v___x_710_; lean_object* v_xs_x27_711_; lean_object* v___y_713_; 
v_v_709_ = lean_array_fget(v_es_700_, v_j_703_);
v___x_710_ = lean_box(0);
v_xs_x27_711_ = lean_array_fset(v_es_700_, v_j_703_, v___x_710_);
switch(lean_obj_tag(v_v_709_))
{
case 0:
{
lean_object* v_key_718_; lean_object* v_val_719_; lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_729_; 
v_key_718_ = lean_ctor_get(v_v_709_, 0);
v_val_719_ = lean_ctor_get(v_v_709_, 1);
v_isSharedCheck_729_ = !lean_is_exclusive(v_v_709_);
if (v_isSharedCheck_729_ == 0)
{
v___x_721_ = v_v_709_;
v_isShared_722_ = v_isSharedCheck_729_;
goto v_resetjp_720_;
}
else
{
lean_inc(v_val_719_);
lean_inc(v_key_718_);
lean_dec(v_v_709_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_729_;
goto v_resetjp_720_;
}
v_resetjp_720_:
{
uint8_t v___x_723_; 
v___x_723_ = l_Lean_instBEqMVarId_beq(v_x_698_, v_key_718_);
if (v___x_723_ == 0)
{
lean_object* v___x_724_; lean_object* v___x_725_; 
lean_del_object(v___x_721_);
v___x_724_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_718_, v_val_719_, v_x_698_, v_x_699_);
v___x_725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
v___y_713_ = v___x_725_;
goto v___jp_712_;
}
else
{
lean_object* v___x_727_; 
lean_dec(v_val_719_);
lean_dec(v_key_718_);
if (v_isShared_722_ == 0)
{
lean_ctor_set(v___x_721_, 1, v_x_699_);
lean_ctor_set(v___x_721_, 0, v_x_698_);
v___x_727_ = v___x_721_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v_x_698_);
lean_ctor_set(v_reuseFailAlloc_728_, 1, v_x_699_);
v___x_727_ = v_reuseFailAlloc_728_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
v___y_713_ = v___x_727_;
goto v___jp_712_;
}
}
}
}
case 1:
{
lean_object* v_node_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_742_; 
v_node_730_ = lean_ctor_get(v_v_709_, 0);
v_isSharedCheck_742_ = !lean_is_exclusive(v_v_709_);
if (v_isSharedCheck_742_ == 0)
{
v___x_732_ = v_v_709_;
v_isShared_733_ = v_isSharedCheck_742_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_node_730_);
lean_dec(v_v_709_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_742_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
size_t v___x_734_; size_t v___x_735_; size_t v___x_736_; size_t v___x_737_; lean_object* v___x_738_; lean_object* v___x_740_; 
v___x_734_ = ((size_t)5ULL);
v___x_735_ = lean_usize_shift_right(v_x_696_, v___x_734_);
v___x_736_ = ((size_t)1ULL);
v___x_737_ = lean_usize_add(v_x_697_, v___x_736_);
v___x_738_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(v_node_730_, v___x_735_, v___x_737_, v_x_698_, v_x_699_);
if (v_isShared_733_ == 0)
{
lean_ctor_set(v___x_732_, 0, v___x_738_);
v___x_740_ = v___x_732_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v___x_738_);
v___x_740_ = v_reuseFailAlloc_741_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
v___y_713_ = v___x_740_;
goto v___jp_712_;
}
}
}
default: 
{
lean_object* v___x_743_; 
v___x_743_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_743_, 0, v_x_698_);
lean_ctor_set(v___x_743_, 1, v_x_699_);
v___y_713_ = v___x_743_;
goto v___jp_712_;
}
}
v___jp_712_:
{
lean_object* v___x_714_; lean_object* v___x_716_; 
v___x_714_ = lean_array_fset(v_xs_x27_711_, v_j_703_, v___y_713_);
lean_dec(v_j_703_);
if (v_isShared_708_ == 0)
{
lean_ctor_set(v___x_707_, 0, v___x_714_);
v___x_716_ = v___x_707_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v___x_714_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
}
else
{
lean_object* v_ks_746_; lean_object* v_vs_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_767_; 
v_ks_746_ = lean_ctor_get(v_x_695_, 0);
v_vs_747_ = lean_ctor_get(v_x_695_, 1);
v_isSharedCheck_767_ = !lean_is_exclusive(v_x_695_);
if (v_isSharedCheck_767_ == 0)
{
v___x_749_ = v_x_695_;
v_isShared_750_ = v_isSharedCheck_767_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_vs_747_);
lean_inc(v_ks_746_);
lean_dec(v_x_695_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_767_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_752_; 
if (v_isShared_750_ == 0)
{
v___x_752_ = v___x_749_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_ks_746_);
lean_ctor_set(v_reuseFailAlloc_766_, 1, v_vs_747_);
v___x_752_ = v_reuseFailAlloc_766_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
lean_object* v_newNode_753_; uint8_t v___y_755_; size_t v___x_761_; uint8_t v___x_762_; 
v_newNode_753_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7___redArg(v___x_752_, v_x_698_, v_x_699_);
v___x_761_ = ((size_t)7ULL);
v___x_762_ = lean_usize_dec_le(v___x_761_, v_x_697_);
if (v___x_762_ == 0)
{
lean_object* v___x_763_; lean_object* v___x_764_; uint8_t v___x_765_; 
v___x_763_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_753_);
v___x_764_ = lean_unsigned_to_nat(4u);
v___x_765_ = lean_nat_dec_lt(v___x_763_, v___x_764_);
lean_dec(v___x_763_);
v___y_755_ = v___x_765_;
goto v___jp_754_;
}
else
{
v___y_755_ = v___x_762_;
goto v___jp_754_;
}
v___jp_754_:
{
if (v___y_755_ == 0)
{
lean_object* v_ks_756_; lean_object* v_vs_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v_ks_756_ = lean_ctor_get(v_newNode_753_, 0);
lean_inc_ref(v_ks_756_);
v_vs_757_ = lean_ctor_get(v_newNode_753_, 1);
lean_inc_ref(v_vs_757_);
lean_dec_ref(v_newNode_753_);
v___x_758_ = lean_unsigned_to_nat(0u);
v___x_759_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___closed__0);
v___x_760_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg(v_x_697_, v_ks_756_, v_vs_757_, v___x_758_, v___x_759_);
lean_dec_ref(v_vs_757_);
lean_dec_ref(v_ks_756_);
return v___x_760_;
}
else
{
return v_newNode_753_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg(size_t v_depth_768_, lean_object* v_keys_769_, lean_object* v_vals_770_, lean_object* v_i_771_, lean_object* v_entries_772_){
_start:
{
lean_object* v___x_773_; uint8_t v___x_774_; 
v___x_773_ = lean_array_get_size(v_keys_769_);
v___x_774_ = lean_nat_dec_lt(v_i_771_, v___x_773_);
if (v___x_774_ == 0)
{
lean_dec(v_i_771_);
return v_entries_772_;
}
else
{
lean_object* v_k_775_; lean_object* v_v_776_; uint64_t v___x_777_; size_t v_h_778_; size_t v___x_779_; lean_object* v___x_780_; size_t v___x_781_; size_t v___x_782_; size_t v___x_783_; size_t v_h_784_; lean_object* v___x_785_; lean_object* v___x_786_; 
v_k_775_ = lean_array_fget_borrowed(v_keys_769_, v_i_771_);
v_v_776_ = lean_array_fget_borrowed(v_vals_770_, v_i_771_);
v___x_777_ = l_Lean_instHashableMVarId_hash(v_k_775_);
v_h_778_ = lean_uint64_to_usize(v___x_777_);
v___x_779_ = ((size_t)5ULL);
v___x_780_ = lean_unsigned_to_nat(1u);
v___x_781_ = ((size_t)1ULL);
v___x_782_ = lean_usize_sub(v_depth_768_, v___x_781_);
v___x_783_ = lean_usize_mul(v___x_779_, v___x_782_);
v_h_784_ = lean_usize_shift_right(v_h_778_, v___x_783_);
v___x_785_ = lean_nat_add(v_i_771_, v___x_780_);
lean_dec(v_i_771_);
lean_inc(v_v_776_);
lean_inc(v_k_775_);
v___x_786_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(v_entries_772_, v_h_784_, v_depth_768_, v_k_775_, v_v_776_);
v_i_771_ = v___x_785_;
v_entries_772_ = v___x_786_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg___boxed(lean_object* v_depth_788_, lean_object* v_keys_789_, lean_object* v_vals_790_, lean_object* v_i_791_, lean_object* v_entries_792_){
_start:
{
size_t v_depth_boxed_793_; lean_object* v_res_794_; 
v_depth_boxed_793_ = lean_unbox_usize(v_depth_788_);
lean_dec(v_depth_788_);
v_res_794_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg(v_depth_boxed_793_, v_keys_789_, v_vals_790_, v_i_791_, v_entries_792_);
lean_dec_ref(v_vals_790_);
lean_dec_ref(v_keys_789_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_x_795_, lean_object* v_x_796_, lean_object* v_x_797_, lean_object* v_x_798_, lean_object* v_x_799_){
_start:
{
size_t v_x_4466__boxed_800_; size_t v_x_4467__boxed_801_; lean_object* v_res_802_; 
v_x_4466__boxed_800_ = lean_unbox_usize(v_x_796_);
lean_dec(v_x_796_);
v_x_4467__boxed_801_ = lean_unbox_usize(v_x_797_);
lean_dec(v_x_797_);
v_res_802_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(v_x_795_, v_x_4466__boxed_800_, v_x_4467__boxed_801_, v_x_798_, v_x_799_);
return v_res_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0___redArg(lean_object* v_x_803_, lean_object* v_x_804_, lean_object* v_x_805_){
_start:
{
uint64_t v___x_806_; size_t v___x_807_; size_t v___x_808_; lean_object* v___x_809_; 
v___x_806_ = l_Lean_instHashableMVarId_hash(v_x_804_);
v___x_807_ = lean_uint64_to_usize(v___x_806_);
v___x_808_ = ((size_t)1ULL);
v___x_809_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(v_x_803_, v___x_807_, v___x_808_, v_x_804_, v_x_805_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg(lean_object* v_mvarId_810_, lean_object* v_val_811_, lean_object* v___y_812_){
_start:
{
lean_object* v___x_814_; lean_object* v_mctx_815_; lean_object* v_cache_816_; lean_object* v_zetaDeltaFVarIds_817_; lean_object* v_postponed_818_; lean_object* v_diag_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_847_; 
v___x_814_ = lean_st_ref_take(v___y_812_);
v_mctx_815_ = lean_ctor_get(v___x_814_, 0);
v_cache_816_ = lean_ctor_get(v___x_814_, 1);
v_zetaDeltaFVarIds_817_ = lean_ctor_get(v___x_814_, 2);
v_postponed_818_ = lean_ctor_get(v___x_814_, 3);
v_diag_819_ = lean_ctor_get(v___x_814_, 4);
v_isSharedCheck_847_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_847_ == 0)
{
v___x_821_ = v___x_814_;
v_isShared_822_ = v_isSharedCheck_847_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_diag_819_);
lean_inc(v_postponed_818_);
lean_inc(v_zetaDeltaFVarIds_817_);
lean_inc(v_cache_816_);
lean_inc(v_mctx_815_);
lean_dec(v___x_814_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_847_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v_depth_823_; lean_object* v_levelAssignDepth_824_; lean_object* v_lmvarCounter_825_; lean_object* v_mvarCounter_826_; lean_object* v_lDecls_827_; lean_object* v_decls_828_; lean_object* v_userNames_829_; lean_object* v_lAssignment_830_; lean_object* v_eAssignment_831_; lean_object* v_dAssignment_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_846_; 
v_depth_823_ = lean_ctor_get(v_mctx_815_, 0);
v_levelAssignDepth_824_ = lean_ctor_get(v_mctx_815_, 1);
v_lmvarCounter_825_ = lean_ctor_get(v_mctx_815_, 2);
v_mvarCounter_826_ = lean_ctor_get(v_mctx_815_, 3);
v_lDecls_827_ = lean_ctor_get(v_mctx_815_, 4);
v_decls_828_ = lean_ctor_get(v_mctx_815_, 5);
v_userNames_829_ = lean_ctor_get(v_mctx_815_, 6);
v_lAssignment_830_ = lean_ctor_get(v_mctx_815_, 7);
v_eAssignment_831_ = lean_ctor_get(v_mctx_815_, 8);
v_dAssignment_832_ = lean_ctor_get(v_mctx_815_, 9);
v_isSharedCheck_846_ = !lean_is_exclusive(v_mctx_815_);
if (v_isSharedCheck_846_ == 0)
{
v___x_834_ = v_mctx_815_;
v_isShared_835_ = v_isSharedCheck_846_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_dAssignment_832_);
lean_inc(v_eAssignment_831_);
lean_inc(v_lAssignment_830_);
lean_inc(v_userNames_829_);
lean_inc(v_decls_828_);
lean_inc(v_lDecls_827_);
lean_inc(v_mvarCounter_826_);
lean_inc(v_lmvarCounter_825_);
lean_inc(v_levelAssignDepth_824_);
lean_inc(v_depth_823_);
lean_dec(v_mctx_815_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_846_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v___x_836_; lean_object* v___x_838_; 
v___x_836_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0___redArg(v_eAssignment_831_, v_mvarId_810_, v_val_811_);
if (v_isShared_835_ == 0)
{
lean_ctor_set(v___x_834_, 8, v___x_836_);
v___x_838_ = v___x_834_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_depth_823_);
lean_ctor_set(v_reuseFailAlloc_845_, 1, v_levelAssignDepth_824_);
lean_ctor_set(v_reuseFailAlloc_845_, 2, v_lmvarCounter_825_);
lean_ctor_set(v_reuseFailAlloc_845_, 3, v_mvarCounter_826_);
lean_ctor_set(v_reuseFailAlloc_845_, 4, v_lDecls_827_);
lean_ctor_set(v_reuseFailAlloc_845_, 5, v_decls_828_);
lean_ctor_set(v_reuseFailAlloc_845_, 6, v_userNames_829_);
lean_ctor_set(v_reuseFailAlloc_845_, 7, v_lAssignment_830_);
lean_ctor_set(v_reuseFailAlloc_845_, 8, v___x_836_);
lean_ctor_set(v_reuseFailAlloc_845_, 9, v_dAssignment_832_);
v___x_838_ = v_reuseFailAlloc_845_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
lean_object* v___x_840_; 
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 0, v___x_838_);
v___x_840_ = v___x_821_;
goto v_reusejp_839_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v___x_838_);
lean_ctor_set(v_reuseFailAlloc_844_, 1, v_cache_816_);
lean_ctor_set(v_reuseFailAlloc_844_, 2, v_zetaDeltaFVarIds_817_);
lean_ctor_set(v_reuseFailAlloc_844_, 3, v_postponed_818_);
lean_ctor_set(v_reuseFailAlloc_844_, 4, v_diag_819_);
v___x_840_ = v_reuseFailAlloc_844_;
goto v_reusejp_839_;
}
v_reusejp_839_:
{
lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_841_ = lean_st_ref_set(v___y_812_, v___x_840_);
v___x_842_ = lean_box(0);
v___x_843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_843_, 0, v___x_842_);
return v___x_843_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg___boxed(lean_object* v_mvarId_848_, lean_object* v_val_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg(v_mvarId_848_, v_val_849_, v___y_850_);
lean_dec(v___y_850_);
return v_res_852_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1(void){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_854_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__0));
v___x_855_ = l_Lean_stringToMessageData(v___x_854_);
return v___x_855_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3(void){
_start:
{
lean_object* v___x_857_; lean_object* v___x_858_; 
v___x_857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__2));
v___x_858_ = l_Lean_stringToMessageData(v___x_857_);
return v___x_858_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5(void){
_start:
{
lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__4));
v___x_861_ = l_Lean_stringToMessageData(v___x_860_);
return v___x_861_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7(void){
_start:
{
lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_863_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__6));
v___x_864_ = l_Lean_stringToMessageData(v___x_863_);
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall(lean_object* v_reorder_865_, lean_object* v_e_866_, lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_a_869_, lean_object* v_a_870_){
_start:
{
lean_object* v___x_872_; uint8_t v___x_873_; lean_object* v___x_874_; 
lean_inc_ref(v_reorder_865_);
v___x_872_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_range(v_reorder_865_);
v___x_873_ = 0;
lean_inc(v___x_872_);
v___x_874_ = l_Lean_Meta_forallMetaBoundedTelescope(v_e_866_, v___x_872_, v___x_873_, v_a_867_, v_a_868_, v_a_869_, v_a_870_);
if (lean_obj_tag(v___x_874_) == 0)
{
lean_object* v_a_875_; lean_object* v_snd_876_; lean_object* v_fst_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_951_; 
v_a_875_ = lean_ctor_get(v___x_874_, 0);
lean_inc(v_a_875_);
lean_dec_ref_known(v___x_874_, 1);
v_snd_876_ = lean_ctor_get(v_a_875_, 1);
v_fst_877_ = lean_ctor_get(v_a_875_, 0);
v_isSharedCheck_951_ = !lean_is_exclusive(v_a_875_);
if (v_isSharedCheck_951_ == 0)
{
v___x_879_ = v_a_875_;
v_isShared_880_ = v_isSharedCheck_951_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_snd_876_);
lean_inc(v_fst_877_);
lean_dec(v_a_875_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_951_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v_fst_881_; lean_object* v_snd_882_; lean_object* v___x_884_; uint8_t v_isShared_885_; uint8_t v_isSharedCheck_950_; 
v_fst_881_ = lean_ctor_get(v_snd_876_, 0);
v_snd_882_ = lean_ctor_get(v_snd_876_, 1);
v_isSharedCheck_950_ = !lean_is_exclusive(v_snd_876_);
if (v_isSharedCheck_950_ == 0)
{
v___x_884_ = v_snd_876_;
v_isShared_885_ = v_isSharedCheck_950_;
goto v_resetjp_883_;
}
else
{
lean_inc(v_snd_882_);
lean_inc(v_fst_881_);
lean_dec(v_snd_876_);
v___x_884_ = lean_box(0);
v_isShared_885_ = v_isSharedCheck_950_;
goto v_resetjp_883_;
}
v_resetjp_883_:
{
uint8_t v___x_886_; lean_object* v___y_888_; lean_object* v___y_889_; lean_object* v___y_890_; lean_object* v___y_891_; lean_object* v___x_918_; uint8_t v___x_919_; 
v___x_886_ = 0;
v___x_918_ = lean_array_get_size(v_fst_877_);
v___x_919_ = lean_nat_dec_eq(v___x_918_, v___x_872_);
lean_dec(v___x_872_);
if (v___x_919_ == 0)
{
lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_925_; 
v___x_920_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1);
lean_inc_ref(v_reorder_865_);
v___x_921_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_reorder_865_);
v___x_922_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_922_, 0, v___x_921_);
v___x_923_ = l_Lean_MessageData_ofFormat(v___x_922_);
if (v_isShared_885_ == 0)
{
lean_ctor_set_tag(v___x_884_, 7);
lean_ctor_set(v___x_884_, 1, v___x_923_);
lean_ctor_set(v___x_884_, 0, v___x_920_);
v___x_925_ = v___x_884_;
goto v_reusejp_924_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_949_, 0, v___x_920_);
lean_ctor_set(v_reuseFailAlloc_949_, 1, v___x_923_);
v___x_925_ = v_reuseFailAlloc_949_;
goto v_reusejp_924_;
}
v_reusejp_924_:
{
lean_object* v___x_926_; lean_object* v___x_928_; 
v___x_926_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__3);
if (v_isShared_880_ == 0)
{
lean_ctor_set_tag(v___x_879_, 7);
lean_ctor_set(v___x_879_, 1, v___x_926_);
lean_ctor_set(v___x_879_, 0, v___x_925_);
v___x_928_ = v___x_879_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v___x_925_);
lean_ctor_set(v_reuseFailAlloc_948_, 1, v___x_926_);
v___x_928_ = v_reuseFailAlloc_948_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; 
lean_inc(v_snd_882_);
v___x_929_ = l_Lean_indentExpr(v_snd_882_);
v___x_930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_928_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
v___x_931_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5);
v___x_932_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_932_, 0, v___x_930_);
lean_ctor_set(v___x_932_, 1, v___x_931_);
v___x_933_ = l_Nat_reprFast(v___x_918_);
v___x_934_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
v___x_935_ = l_Lean_MessageData_ofFormat(v___x_934_);
v___x_936_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_936_, 0, v___x_932_);
lean_ctor_set(v___x_936_, 1, v___x_935_);
v___x_937_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7);
v___x_938_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_938_, 0, v___x_936_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
v___x_939_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v___x_938_, v_a_867_, v_a_868_, v_a_869_, v_a_870_);
if (lean_obj_tag(v___x_939_) == 0)
{
lean_dec_ref_known(v___x_939_, 1);
v___y_888_ = v_a_867_;
v___y_889_ = v_a_868_;
v___y_890_ = v_a_869_;
v___y_891_ = v_a_870_;
goto v___jp_887_;
}
else
{
lean_object* v_a_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_947_; 
lean_dec(v_snd_882_);
lean_dec(v_fst_881_);
lean_dec(v_fst_877_);
lean_dec_ref(v_reorder_865_);
v_a_940_ = lean_ctor_get(v___x_939_, 0);
v_isSharedCheck_947_ = !lean_is_exclusive(v___x_939_);
if (v_isSharedCheck_947_ == 0)
{
v___x_942_ = v___x_939_;
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_a_940_);
lean_dec(v___x_939_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___x_945_; 
if (v_isShared_943_ == 0)
{
v___x_945_ = v___x_942_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_946_; 
v_reuseFailAlloc_946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_946_, 0, v_a_940_);
v___x_945_ = v_reuseFailAlloc_946_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
return v___x_945_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_884_);
lean_del_object(v___x_879_);
v___y_888_ = v_a_867_;
v___y_889_ = v_a_868_;
v___y_890_ = v_a_869_;
v___y_891_ = v_a_870_;
goto v___jp_887_;
}
v___jp_887_:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_892_ = lean_box(v___x_886_);
lean_inc_ref(v_reorder_865_);
v___x_893_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(v___x_892_, v_reorder_865_, v_fst_881_);
v___x_894_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars(v_fst_877_, v_reorder_865_, v___y_888_, v___y_889_, v___y_890_, v___y_891_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_object* v_a_895_; uint8_t v___x_896_; uint8_t v___x_897_; uint8_t v___x_898_; lean_object* v___x_899_; 
v_a_895_ = lean_ctor_get(v___x_894_, 0);
lean_inc(v_a_895_);
lean_dec_ref_known(v___x_894_, 1);
v___x_896_ = 0;
v___x_897_ = 1;
v___x_898_ = 1;
v___x_899_ = l_Lean_Meta_mkForallFVars(v_a_895_, v_snd_882_, v___x_896_, v___x_897_, v___x_897_, v___x_898_, v___y_888_, v___y_889_, v___y_890_, v___y_891_);
lean_dec(v_a_895_);
if (lean_obj_tag(v___x_899_) == 0)
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_909_; 
v_a_900_ = lean_ctor_get(v___x_899_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_899_);
if (v_isSharedCheck_909_ == 0)
{
v___x_902_ = v___x_899_;
v_isShared_903_ = v_isSharedCheck_909_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_899_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_909_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_907_; 
v___x_904_ = lean_array_to_list(v___x_893_);
v___x_905_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(v___x_904_, v_a_900_);
lean_dec(v___x_904_);
if (v_isShared_903_ == 0)
{
lean_ctor_set(v___x_902_, 0, v___x_905_);
v___x_907_ = v___x_902_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v___x_905_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
}
else
{
lean_dec_ref(v___x_893_);
return v___x_899_;
}
}
else
{
lean_object* v_a_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_917_; 
lean_dec_ref(v___x_893_);
lean_dec(v_snd_882_);
v_a_910_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_917_ == 0)
{
v___x_912_ = v___x_894_;
v_isShared_913_ = v_isSharedCheck_917_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_a_910_);
lean_dec(v___x_894_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_917_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
lean_object* v___x_915_; 
if (v_isShared_913_ == 0)
{
v___x_915_ = v___x_912_;
goto v_reusejp_914_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v_a_910_);
v___x_915_ = v_reuseFailAlloc_916_;
goto v_reusejp_914_;
}
v_reusejp_914_:
{
return v___x_915_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_952_; lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_959_; 
lean_dec(v___x_872_);
lean_dec_ref(v_reorder_865_);
v_a_952_ = lean_ctor_get(v___x_874_, 0);
v_isSharedCheck_959_ = !lean_is_exclusive(v___x_874_);
if (v_isSharedCheck_959_ == 0)
{
v___x_954_ = v___x_874_;
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
else
{
lean_inc(v_a_952_);
lean_dec(v___x_874_);
v___x_954_ = lean_box(0);
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
v_resetjp_953_:
{
lean_object* v___x_957_; 
if (v_isShared_955_ == 0)
{
v___x_957_ = v___x_954_;
goto v_reusejp_956_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v_a_952_);
v___x_957_ = v_reuseFailAlloc_958_;
goto v_reusejp_956_;
}
v_reusejp_956_:
{
return v___x_957_;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1(void){
_start:
{
lean_object* v___x_961_; lean_object* v___x_962_; 
v___x_961_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__0));
v___x_962_ = l_Lean_stringToMessageData(v___x_961_);
return v___x_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda(lean_object* v_reorder_963_, lean_object* v_e_964_, lean_object* v_a_965_, lean_object* v_a_966_, lean_object* v_a_967_, lean_object* v_a_968_){
_start:
{
lean_object* v___x_970_; 
lean_inc(v_a_968_);
lean_inc_ref(v_a_967_);
lean_inc(v_a_966_);
lean_inc_ref(v_a_965_);
lean_inc_ref(v_e_964_);
v___x_970_ = lean_infer_type(v_e_964_, v_a_965_, v_a_966_, v_a_967_, v_a_968_);
if (lean_obj_tag(v___x_970_) == 0)
{
lean_object* v_a_971_; lean_object* v___x_973_; uint8_t v_isShared_974_; uint8_t v_isSharedCheck_1066_; 
v_a_971_ = lean_ctor_get(v___x_970_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_970_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_973_ = v___x_970_;
v_isShared_974_ = v_isSharedCheck_1066_;
goto v_resetjp_972_;
}
else
{
lean_inc(v_a_971_);
lean_dec(v___x_970_);
v___x_973_ = lean_box(0);
v_isShared_974_ = v_isSharedCheck_1066_;
goto v_resetjp_972_;
}
v_resetjp_972_:
{
lean_object* v___x_975_; uint8_t v___x_976_; lean_object* v___x_977_; 
lean_inc_ref(v_reorder_963_);
v___x_975_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_range(v_reorder_963_);
v___x_976_ = 0;
lean_inc(v___x_975_);
v___x_977_ = l_Lean_Meta_forallMetaBoundedTelescope(v_a_971_, v___x_975_, v___x_976_, v_a_965_, v_a_966_, v_a_967_, v_a_968_);
if (lean_obj_tag(v___x_977_) == 0)
{
lean_object* v_a_978_; lean_object* v_snd_979_; lean_object* v_fst_980_; lean_object* v___x_982_; uint8_t v_isShared_983_; uint8_t v_isSharedCheck_1057_; 
v_a_978_ = lean_ctor_get(v___x_977_, 0);
lean_inc(v_a_978_);
lean_dec_ref_known(v___x_977_, 1);
v_snd_979_ = lean_ctor_get(v_a_978_, 1);
v_fst_980_ = lean_ctor_get(v_a_978_, 0);
v_isSharedCheck_1057_ = !lean_is_exclusive(v_a_978_);
if (v_isSharedCheck_1057_ == 0)
{
v___x_982_ = v_a_978_;
v_isShared_983_ = v_isSharedCheck_1057_;
goto v_resetjp_981_;
}
else
{
lean_inc(v_snd_979_);
lean_inc(v_fst_980_);
lean_dec(v_a_978_);
v___x_982_ = lean_box(0);
v_isShared_983_ = v_isSharedCheck_1057_;
goto v_resetjp_981_;
}
v_resetjp_981_:
{
lean_object* v_fst_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_1055_; 
v_fst_984_ = lean_ctor_get(v_snd_979_, 0);
v_isSharedCheck_1055_ = !lean_is_exclusive(v_snd_979_);
if (v_isSharedCheck_1055_ == 0)
{
lean_object* v_unused_1056_; 
v_unused_1056_ = lean_ctor_get(v_snd_979_, 1);
lean_dec(v_unused_1056_);
v___x_986_ = v_snd_979_;
v_isShared_987_ = v_isSharedCheck_1055_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_fst_984_);
lean_dec(v_snd_979_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_1055_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
uint8_t v___x_988_; lean_object* v___y_990_; lean_object* v___y_991_; lean_object* v___y_992_; lean_object* v___y_993_; lean_object* v___x_1021_; uint8_t v___x_1022_; 
v___x_988_ = 0;
v___x_1021_ = lean_array_get_size(v_fst_980_);
v___x_1022_ = lean_nat_dec_eq(v___x_1021_, v___x_975_);
lean_dec(v___x_975_);
if (v___x_1022_ == 0)
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1026_; 
v___x_1023_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__1);
lean_inc_ref(v_reorder_963_);
v___x_1024_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_reorder_963_);
if (v_isShared_974_ == 0)
{
lean_ctor_set_tag(v___x_973_, 3);
lean_ctor_set(v___x_973_, 0, v___x_1024_);
v___x_1026_ = v___x_973_;
goto v_reusejp_1025_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v___x_1024_);
v___x_1026_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1025_;
}
v_reusejp_1025_:
{
lean_object* v___x_1027_; lean_object* v___x_1029_; 
v___x_1027_ = l_Lean_MessageData_ofFormat(v___x_1026_);
if (v_isShared_987_ == 0)
{
lean_ctor_set_tag(v___x_986_, 7);
lean_ctor_set(v___x_986_, 1, v___x_1027_);
lean_ctor_set(v___x_986_, 0, v___x_1023_);
v___x_1029_ = v___x_986_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v___x_1023_);
lean_ctor_set(v_reuseFailAlloc_1053_, 1, v___x_1027_);
v___x_1029_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
lean_object* v___x_1030_; lean_object* v___x_1032_; 
v___x_1030_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1, &lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___closed__1);
if (v_isShared_983_ == 0)
{
lean_ctor_set_tag(v___x_982_, 7);
lean_ctor_set(v___x_982_, 1, v___x_1030_);
lean_ctor_set(v___x_982_, 0, v___x_1029_);
v___x_1032_ = v___x_982_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1052_, 1, v___x_1030_);
v___x_1032_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
lean_inc_ref(v_e_964_);
v___x_1033_ = l_Lean_indentExpr(v_e_964_);
v___x_1034_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1032_);
lean_ctor_set(v___x_1034_, 1, v___x_1033_);
v___x_1035_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__5);
v___x_1036_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1034_);
lean_ctor_set(v___x_1036_, 1, v___x_1035_);
v___x_1037_ = l_Nat_reprFast(v___x_1021_);
v___x_1038_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1037_);
v___x_1039_ = l_Lean_MessageData_ofFormat(v___x_1038_);
v___x_1040_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1036_);
lean_ctor_set(v___x_1040_, 1, v___x_1039_);
v___x_1041_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7, &lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Translate_reorderForall___closed__7);
v___x_1042_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1040_);
lean_ctor_set(v___x_1042_, 1, v___x_1041_);
v___x_1043_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v___x_1042_, v_a_965_, v_a_966_, v_a_967_, v_a_968_);
if (lean_obj_tag(v___x_1043_) == 0)
{
lean_dec_ref_known(v___x_1043_, 1);
v___y_990_ = v_a_965_;
v___y_991_ = v_a_966_;
v___y_992_ = v_a_967_;
v___y_993_ = v_a_968_;
goto v___jp_989_;
}
else
{
lean_object* v_a_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1051_; 
lean_dec(v_fst_984_);
lean_dec(v_fst_980_);
lean_dec_ref(v_e_964_);
lean_dec_ref(v_reorder_963_);
v_a_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1051_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1051_ == 0)
{
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1051_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_a_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1051_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v___x_1049_; 
if (v_isShared_1047_ == 0)
{
v___x_1049_ = v___x_1046_;
goto v_reusejp_1048_;
}
else
{
lean_object* v_reuseFailAlloc_1050_; 
v_reuseFailAlloc_1050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1050_, 0, v_a_1044_);
v___x_1049_ = v_reuseFailAlloc_1050_;
goto v_reusejp_1048_;
}
v_reusejp_1048_:
{
return v___x_1049_;
}
}
}
}
}
}
}
else
{
lean_del_object(v___x_986_);
lean_del_object(v___x_982_);
lean_del_object(v___x_973_);
v___y_990_ = v_a_965_;
v___y_991_ = v_a_966_;
v___y_992_ = v_a_967_;
v___y_993_ = v_a_968_;
goto v___jp_989_;
}
v___jp_989_:
{
lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_994_ = lean_box(v___x_988_);
lean_inc_ref(v_reorder_963_);
v___x_995_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(v___x_994_, v_reorder_963_, v_fst_984_);
lean_inc(v_fst_980_);
v___x_996_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars(v_fst_980_, v_reorder_963_, v___y_990_, v___y_991_, v___y_992_, v___y_993_);
if (lean_obj_tag(v___x_996_) == 0)
{
lean_object* v_a_997_; lean_object* v___x_998_; uint8_t v___x_999_; uint8_t v___x_1000_; uint8_t v___x_1001_; lean_object* v___x_1002_; 
v_a_997_ = lean_ctor_get(v___x_996_, 0);
lean_inc(v_a_997_);
lean_dec_ref_known(v___x_996_, 1);
v___x_998_ = l_Lean_Expr_beta(v_e_964_, v_fst_980_);
v___x_999_ = 0;
v___x_1000_ = 1;
v___x_1001_ = 1;
v___x_1002_ = l_Lean_Meta_mkLambdaFVars(v_a_997_, v___x_998_, v___x_999_, v___x_1000_, v___x_999_, v___x_1000_, v___x_1001_, v___y_990_, v___y_991_, v___y_992_, v___y_993_);
lean_dec(v_a_997_);
if (lean_obj_tag(v___x_1002_) == 0)
{
lean_object* v_a_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1012_; 
v_a_1003_ = lean_ctor_get(v___x_1002_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_1002_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_1005_ = v___x_1002_;
v_isShared_1006_ = v_isSharedCheck_1012_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_a_1003_);
lean_dec(v___x_1002_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1012_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1010_; 
v___x_1007_ = lean_array_to_list(v___x_995_);
v___x_1008_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_fixBinderInfos(v___x_1007_, v_a_1003_);
lean_dec(v___x_1007_);
if (v_isShared_1006_ == 0)
{
lean_ctor_set(v___x_1005_, 0, v___x_1008_);
v___x_1010_ = v___x_1005_;
goto v_reusejp_1009_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_1008_);
v___x_1010_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1009_;
}
v_reusejp_1009_:
{
return v___x_1010_;
}
}
}
else
{
lean_dec_ref(v___x_995_);
return v___x_1002_;
}
}
else
{
lean_object* v_a_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1020_; 
lean_dec_ref(v___x_995_);
lean_dec(v_fst_980_);
lean_dec_ref(v_e_964_);
v_a_1013_ = lean_ctor_get(v___x_996_, 0);
v_isSharedCheck_1020_ = !lean_is_exclusive(v___x_996_);
if (v_isSharedCheck_1020_ == 0)
{
v___x_1015_ = v___x_996_;
v_isShared_1016_ = v_isSharedCheck_1020_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_a_1013_);
lean_dec(v___x_996_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1020_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
lean_object* v___x_1018_; 
if (v_isShared_1016_ == 0)
{
v___x_1018_ = v___x_1015_;
goto v_reusejp_1017_;
}
else
{
lean_object* v_reuseFailAlloc_1019_; 
v_reuseFailAlloc_1019_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1019_, 0, v_a_1013_);
v___x_1018_ = v_reuseFailAlloc_1019_;
goto v_reusejp_1017_;
}
v_reusejp_1017_:
{
return v___x_1018_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1065_; 
lean_dec(v___x_975_);
lean_del_object(v___x_973_);
lean_dec_ref(v_e_964_);
lean_dec_ref(v_reorder_963_);
v_a_1058_ = lean_ctor_get(v___x_977_, 0);
v_isSharedCheck_1065_ = !lean_is_exclusive(v___x_977_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1060_ = v___x_977_;
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_a_1058_);
lean_dec(v___x_977_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v___x_1063_; 
if (v_isShared_1061_ == 0)
{
v___x_1063_ = v___x_1060_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1064_; 
v_reuseFailAlloc_1064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1064_, 0, v_a_1058_);
v___x_1063_ = v_reuseFailAlloc_1064_;
goto v_reusejp_1062_;
}
v_reusejp_1062_:
{
return v___x_1063_;
}
}
}
}
}
else
{
lean_dec_ref(v_e_964_);
lean_dec_ref(v_reorder_963_);
return v___x_970_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1(lean_object* v_as_1067_, size_t v_sz_1068_, size_t v_i_1069_, lean_object* v_b_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_){
_start:
{
uint8_t v___x_1076_; 
v___x_1076_ = lean_usize_dec_lt(v_i_1069_, v_sz_1068_);
if (v___x_1076_ == 0)
{
lean_object* v___x_1077_; 
v___x_1077_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1077_, 0, v_b_1070_);
return v___x_1077_;
}
else
{
lean_object* v_a_1078_; lean_object* v_fst_1079_; lean_object* v_snd_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; 
v_a_1078_ = lean_array_uget_borrowed(v_as_1067_, v_i_1069_);
v_fst_1079_ = lean_ctor_get(v_a_1078_, 0);
v_snd_1080_ = lean_ctor_get(v_a_1078_, 1);
v___x_1081_ = l_Lean_instInhabitedExpr;
v___x_1082_ = lean_array_get_borrowed(v___x_1081_, v_b_1070_, v_fst_1079_);
v___x_1083_ = l_Lean_Expr_mvarId_x21(v___x_1082_);
lean_inc(v___x_1083_);
v___x_1084_ = l_Lean_MVarId_getDecl(v___x_1083_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
if (lean_obj_tag(v___x_1084_) == 0)
{
lean_object* v_a_1085_; lean_object* v_userName_1086_; lean_object* v_type_1087_; lean_object* v___x_1088_; 
v_a_1085_ = lean_ctor_get(v___x_1084_, 0);
lean_inc(v_a_1085_);
lean_dec_ref_known(v___x_1084_, 1);
v_userName_1086_ = lean_ctor_get(v_a_1085_, 0);
lean_inc(v_userName_1086_);
v_type_1087_ = lean_ctor_get(v_a_1085_, 2);
lean_inc_ref(v_type_1087_);
lean_dec(v_a_1085_);
lean_inc(v_snd_1080_);
v___x_1088_ = lp_mathlib_Mathlib_Tactic_Translate_reorderForall(v_snd_1080_, v_type_1087_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
if (lean_obj_tag(v___x_1088_) == 0)
{
lean_object* v_a_1089_; lean_object* v___x_1090_; uint8_t v___x_1091_; lean_object* v___x_1092_; 
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc(v_a_1089_);
lean_dec_ref_known(v___x_1088_, 1);
v___x_1090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1090_, 0, v_a_1089_);
v___x_1091_ = 0;
v___x_1092_ = l_Lean_Meta_mkFreshExprMVar(v___x_1090_, v___x_1091_, v_userName_1086_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
if (lean_obj_tag(v___x_1092_) == 0)
{
lean_object* v_a_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v_a_1093_ = lean_ctor_get(v___x_1092_, 0);
lean_inc_n(v_a_1093_, 2);
lean_dec_ref_known(v___x_1092_, 1);
lean_inc(v_snd_1080_);
v___x_1094_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_reverse(v_snd_1080_);
v___x_1095_ = lp_mathlib_Mathlib_Tactic_Translate_reorderLambda(v___x_1094_, v_a_1093_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; lean_object* v___x_1097_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
lean_inc(v_a_1096_);
lean_dec_ref_known(v___x_1095_, 1);
v___x_1097_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg(v___x_1083_, v_a_1096_, v___y_1072_);
if (lean_obj_tag(v___x_1097_) == 0)
{
lean_object* v___x_1098_; size_t v___x_1099_; size_t v___x_1100_; 
lean_dec_ref_known(v___x_1097_, 1);
v___x_1098_ = lean_array_set(v_b_1070_, v_fst_1079_, v_a_1093_);
v___x_1099_ = ((size_t)1ULL);
v___x_1100_ = lean_usize_add(v_i_1069_, v___x_1099_);
v_i_1069_ = v___x_1100_;
v_b_1070_ = v___x_1098_;
goto _start;
}
else
{
lean_object* v_a_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1109_; 
lean_dec(v_a_1093_);
lean_dec_ref(v_b_1070_);
v_a_1102_ = lean_ctor_get(v___x_1097_, 0);
v_isSharedCheck_1109_ = !lean_is_exclusive(v___x_1097_);
if (v_isSharedCheck_1109_ == 0)
{
v___x_1104_ = v___x_1097_;
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_a_1102_);
lean_dec(v___x_1097_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1107_; 
if (v_isShared_1105_ == 0)
{
v___x_1107_ = v___x_1104_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v_a_1102_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
}
}
else
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec(v_a_1093_);
lean_dec(v___x_1083_);
lean_dec_ref(v_b_1070_);
v_a_1110_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1095_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1095_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
}
else
{
lean_object* v_a_1118_; lean_object* v___x_1120_; uint8_t v_isShared_1121_; uint8_t v_isSharedCheck_1125_; 
lean_dec(v___x_1083_);
lean_dec_ref(v_b_1070_);
v_a_1118_ = lean_ctor_get(v___x_1092_, 0);
v_isSharedCheck_1125_ = !lean_is_exclusive(v___x_1092_);
if (v_isSharedCheck_1125_ == 0)
{
v___x_1120_ = v___x_1092_;
v_isShared_1121_ = v_isSharedCheck_1125_;
goto v_resetjp_1119_;
}
else
{
lean_inc(v_a_1118_);
lean_dec(v___x_1092_);
v___x_1120_ = lean_box(0);
v_isShared_1121_ = v_isSharedCheck_1125_;
goto v_resetjp_1119_;
}
v_resetjp_1119_:
{
lean_object* v___x_1123_; 
if (v_isShared_1121_ == 0)
{
v___x_1123_ = v___x_1120_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1124_; 
v_reuseFailAlloc_1124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1124_, 0, v_a_1118_);
v___x_1123_ = v_reuseFailAlloc_1124_;
goto v_reusejp_1122_;
}
v_reusejp_1122_:
{
return v___x_1123_;
}
}
}
}
else
{
lean_object* v_a_1126_; lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1133_; 
lean_dec(v_userName_1086_);
lean_dec(v___x_1083_);
lean_dec_ref(v_b_1070_);
v_a_1126_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1133_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1133_ == 0)
{
v___x_1128_ = v___x_1088_;
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
else
{
lean_inc(v_a_1126_);
lean_dec(v___x_1088_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
lean_object* v___x_1131_; 
if (v_isShared_1129_ == 0)
{
v___x_1131_ = v___x_1128_;
goto v_reusejp_1130_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v_a_1126_);
v___x_1131_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1130_;
}
v_reusejp_1130_:
{
return v___x_1131_;
}
}
}
}
else
{
lean_object* v_a_1134_; lean_object* v___x_1136_; uint8_t v_isShared_1137_; uint8_t v_isSharedCheck_1141_; 
lean_dec(v___x_1083_);
lean_dec_ref(v_b_1070_);
v_a_1134_ = lean_ctor_get(v___x_1084_, 0);
v_isSharedCheck_1141_ = !lean_is_exclusive(v___x_1084_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1136_ = v___x_1084_;
v_isShared_1137_ = v_isSharedCheck_1141_;
goto v_resetjp_1135_;
}
else
{
lean_inc(v_a_1134_);
lean_dec(v___x_1084_);
v___x_1136_ = lean_box(0);
v_isShared_1137_ = v_isSharedCheck_1141_;
goto v_resetjp_1135_;
}
v_resetjp_1135_:
{
lean_object* v___x_1139_; 
if (v_isShared_1137_ == 0)
{
v___x_1139_ = v___x_1136_;
goto v_reusejp_1138_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v_a_1134_);
v___x_1139_ = v_reuseFailAlloc_1140_;
goto v_reusejp_1138_;
}
v_reusejp_1138_:
{
return v___x_1139_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars(lean_object* v_mvars_1142_, lean_object* v_reorder_1143_, lean_object* v_a_1144_, lean_object* v_a_1145_, lean_object* v_a_1146_, lean_object* v_a_1147_){
_start:
{
lean_object* v_argReorders_1149_; size_t v_sz_1150_; size_t v___x_1151_; lean_object* v___x_1152_; 
v_argReorders_1149_ = lean_ctor_get(v_reorder_1143_, 1);
v_sz_1150_ = lean_array_size(v_argReorders_1149_);
v___x_1151_ = ((size_t)0ULL);
v___x_1152_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1(v_argReorders_1149_, v_sz_1150_, v___x_1151_, v_mvars_1142_, v_a_1144_, v_a_1145_, v_a_1146_, v_a_1147_);
if (lean_obj_tag(v___x_1152_) == 0)
{
lean_object* v_a_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1162_; 
v_a_1153_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1155_ = v___x_1152_;
v_isShared_1156_ = v_isSharedCheck_1162_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_a_1153_);
lean_dec(v___x_1152_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1162_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1160_; 
v___x_1157_ = l_Lean_instInhabitedExpr;
v___x_1158_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_permute_x21___redArg(v___x_1157_, v_reorder_1143_, v_a_1153_);
if (v_isShared_1156_ == 0)
{
lean_ctor_set(v___x_1155_, 0, v___x_1158_);
v___x_1160_ = v___x_1155_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v___x_1158_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
else
{
lean_dec_ref(v_reorder_1143_);
return v___x_1152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars___boxed(lean_object* v_mvars_1163_, lean_object* v_reorder_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_){
_start:
{
lean_object* v_res_1170_; 
v_res_1170_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars(v_mvars_1163_, v_reorder_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_);
lean_dec(v_a_1168_);
lean_dec_ref(v_a_1167_);
lean_dec(v_a_1166_);
lean_dec_ref(v_a_1165_);
return v_res_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1___boxed(lean_object* v_as_1171_, lean_object* v_sz_1172_, lean_object* v_i_1173_, lean_object* v_b_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_){
_start:
{
size_t v_sz_boxed_1180_; size_t v_i_boxed_1181_; lean_object* v_res_1182_; 
v_sz_boxed_1180_ = lean_unbox_usize(v_sz_1172_);
lean_dec(v_sz_1172_);
v_i_boxed_1181_ = lean_unbox_usize(v_i_1173_);
lean_dec(v_i_1173_);
v_res_1182_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__1(v_as_1171_, v_sz_boxed_1180_, v_i_boxed_1181_, v_b_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
lean_dec(v___y_1176_);
lean_dec_ref(v___y_1175_);
lean_dec_ref(v_as_1171_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderForall___boxed(lean_object* v_reorder_1183_, lean_object* v_e_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_, lean_object* v_a_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_Mathlib_Tactic_Translate_reorderForall(v_reorder_1183_, v_e_1184_, v_a_1185_, v_a_1186_, v_a_1187_, v_a_1188_);
lean_dec(v_a_1188_);
lean_dec_ref(v_a_1187_);
lean_dec(v_a_1186_);
lean_dec_ref(v_a_1185_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_reorderLambda___boxed(lean_object* v_reorder_1191_, lean_object* v_e_1192_, lean_object* v_a_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_){
_start:
{
lean_object* v_res_1198_; 
v_res_1198_ = lp_mathlib_Mathlib_Tactic_Translate_reorderLambda(v_reorder_1191_, v_e_1192_, v_a_1193_, v_a_1194_, v_a_1195_, v_a_1196_);
lean_dec(v_a_1196_);
lean_dec_ref(v_a_1195_);
lean_dec(v_a_1194_);
lean_dec_ref(v_a_1193_);
return v_res_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0(lean_object* v_mvarId_1199_, lean_object* v_val_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_){
_start:
{
lean_object* v___x_1206_; 
v___x_1206_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___redArg(v_mvarId_1199_, v_val_1200_, v___y_1202_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0___boxed(lean_object* v_mvarId_1207_, lean_object* v_val_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0(v_mvarId_1207_, v_val_1208_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
lean_dec(v___y_1212_);
lean_dec_ref(v___y_1211_);
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1209_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3(lean_object* v_00_u03b1_1215_, lean_object* v_msg_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_){
_start:
{
lean_object* v___x_1222_; 
v___x_1222_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v_msg_1216_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___boxed(lean_object* v_00_u03b1_1223_, lean_object* v_msg_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3(v_00_u03b1_1223_, v_msg_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0(lean_object* v_00_u03b2_1231_, lean_object* v_x_1232_, lean_object* v_x_1233_, lean_object* v_x_1234_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0___redArg(v_x_1232_, v_x_1233_, v_x_1234_);
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1236_, lean_object* v_x_1237_, size_t v_x_1238_, size_t v_x_1239_, lean_object* v_x_1240_, lean_object* v_x_1241_){
_start:
{
lean_object* v___x_1242_; 
v___x_1242_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___redArg(v_x_1237_, v_x_1238_, v_x_1239_, v_x_1240_, v_x_1241_);
return v___x_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_1243_, lean_object* v_x_1244_, lean_object* v_x_1245_, lean_object* v_x_1246_, lean_object* v_x_1247_, lean_object* v_x_1248_){
_start:
{
size_t v_x_5328__boxed_1249_; size_t v_x_5329__boxed_1250_; lean_object* v_res_1251_; 
v_x_5328__boxed_1249_ = lean_unbox_usize(v_x_1245_);
lean_dec(v_x_1245_);
v_x_5329__boxed_1250_ = lean_unbox_usize(v_x_1246_);
lean_dec(v_x_1246_);
v_res_1251_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3(v_00_u03b2_1243_, v_x_1244_, v_x_5328__boxed_1249_, v_x_5329__boxed_1250_, v_x_1247_, v_x_1248_);
return v_res_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7(lean_object* v_00_u03b2_1252_, lean_object* v_n_1253_, lean_object* v_k_1254_, lean_object* v_v_1255_){
_start:
{
lean_object* v___x_1256_; 
v___x_1256_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7___redArg(v_n_1253_, v_k_1254_, v_v_1255_);
return v___x_1256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8(lean_object* v_00_u03b2_1257_, size_t v_depth_1258_, lean_object* v_keys_1259_, lean_object* v_vals_1260_, lean_object* v_heq_1261_, lean_object* v_i_1262_, lean_object* v_entries_1263_){
_start:
{
lean_object* v___x_1264_; 
v___x_1264_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___redArg(v_depth_1258_, v_keys_1259_, v_vals_1260_, v_i_1262_, v_entries_1263_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8___boxed(lean_object* v_00_u03b2_1265_, lean_object* v_depth_1266_, lean_object* v_keys_1267_, lean_object* v_vals_1268_, lean_object* v_heq_1269_, lean_object* v_i_1270_, lean_object* v_entries_1271_){
_start:
{
size_t v_depth_boxed_1272_; lean_object* v_res_1273_; 
v_depth_boxed_1272_ = lean_unbox_usize(v_depth_1266_);
lean_dec(v_depth_1266_);
v_res_1273_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__8(v_00_u03b2_1265_, v_depth_boxed_1272_, v_keys_1267_, v_vals_1268_, v_heq_1269_, v_i_1270_, v_entries_1271_);
lean_dec_ref(v_vals_1268_);
lean_dec_ref(v_keys_1267_);
return v_res_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8(lean_object* v_00_u03b2_1274_, lean_object* v_x_1275_, lean_object* v_x_1276_, lean_object* v_x_1277_, lean_object* v_x_1278_){
_start:
{
lean_object* v___x_1279_; 
v___x_1279_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_reorderMVars_spec__0_spec__0_spec__3_spec__7_spec__8___redArg(v_x_1275_, v_x_1276_, v_x_1277_, v_x_1278_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg(lean_object* v_next_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_snd_1284_; lean_object* v___x_1286_; uint8_t v_isShared_1287_; uint8_t v_isSharedCheck_1337_; 
v_snd_1284_ = lean_ctor_get(v_a_1283_, 1);
v_isSharedCheck_1337_ = !lean_is_exclusive(v_a_1283_);
if (v_isSharedCheck_1337_ == 0)
{
lean_object* v_unused_1338_; 
v_unused_1338_ = lean_ctor_get(v_a_1283_, 0);
lean_dec(v_unused_1338_);
v___x_1286_ = v_a_1283_;
v_isShared_1287_ = v_isSharedCheck_1337_;
goto v_resetjp_1285_;
}
else
{
lean_inc(v_snd_1284_);
lean_dec(v_a_1283_);
v___x_1286_ = lean_box(0);
v_isShared_1287_ = v_isSharedCheck_1337_;
goto v_resetjp_1285_;
}
v_resetjp_1285_:
{
lean_object* v_snd_1288_; lean_object* v_fst_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1336_; 
v_snd_1288_ = lean_ctor_get(v_snd_1284_, 1);
v_fst_1289_ = lean_ctor_get(v_snd_1284_, 0);
v_isSharedCheck_1336_ = !lean_is_exclusive(v_snd_1284_);
if (v_isSharedCheck_1336_ == 0)
{
v___x_1291_ = v_snd_1284_;
v_isShared_1292_ = v_isSharedCheck_1336_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_snd_1288_);
lean_inc(v_fst_1289_);
lean_dec(v_snd_1284_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1336_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v_fst_1293_; lean_object* v_snd_1294_; lean_object* v___x_1296_; uint8_t v_isShared_1297_; uint8_t v_isSharedCheck_1335_; 
v_fst_1293_ = lean_ctor_get(v_snd_1288_, 0);
v_snd_1294_ = lean_ctor_get(v_snd_1288_, 1);
v_isSharedCheck_1335_ = !lean_is_exclusive(v_snd_1288_);
if (v_isSharedCheck_1335_ == 0)
{
v___x_1296_ = v_snd_1288_;
v_isShared_1297_ = v_isSharedCheck_1335_;
goto v_resetjp_1295_;
}
else
{
lean_inc(v_snd_1294_);
lean_inc(v_fst_1293_);
lean_dec(v_snd_1288_);
v___x_1296_ = lean_box(0);
v_isShared_1297_ = v_isSharedCheck_1335_;
goto v_resetjp_1295_;
}
v_resetjp_1295_:
{
lean_object* v___x_1298_; 
v___x_1298_ = lean_array_fget_borrowed(v_fst_1289_, v_fst_1293_);
if (lean_obj_tag(v___x_1298_) == 1)
{
lean_object* v_val_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; uint8_t v___x_1302_; 
v_val_1299_ = lean_ctor_get(v___x_1298_, 0);
lean_inc(v_val_1299_);
v___x_1300_ = lean_box(0);
v___x_1301_ = lean_array_set(v_fst_1289_, v_fst_1293_, v___x_1300_);
v___x_1302_ = lean_nat_dec_eq(v_val_1299_, v_next_1282_);
if (v___x_1302_ == 0)
{
lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1307_; 
lean_dec(v_fst_1293_);
v___x_1303_ = lean_box(0);
lean_inc(v_val_1299_);
v___x_1304_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1304_, 0, v_val_1299_);
lean_ctor_set(v___x_1304_, 1, v___x_1303_);
v___x_1305_ = l_List_appendTR___redArg(v_snd_1294_, v___x_1304_);
if (v_isShared_1297_ == 0)
{
lean_ctor_set(v___x_1296_, 1, v___x_1305_);
lean_ctor_set(v___x_1296_, 0, v_val_1299_);
v___x_1307_ = v___x_1296_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1315_; 
v_reuseFailAlloc_1315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1315_, 0, v_val_1299_);
lean_ctor_set(v_reuseFailAlloc_1315_, 1, v___x_1305_);
v___x_1307_ = v_reuseFailAlloc_1315_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
lean_object* v___x_1309_; 
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 1, v___x_1307_);
lean_ctor_set(v___x_1291_, 0, v___x_1301_);
v___x_1309_ = v___x_1291_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v___x_1301_);
lean_ctor_set(v_reuseFailAlloc_1314_, 1, v___x_1307_);
v___x_1309_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
lean_object* v___x_1311_; 
if (v_isShared_1287_ == 0)
{
lean_ctor_set(v___x_1286_, 1, v___x_1309_);
lean_ctor_set(v___x_1286_, 0, v___x_1300_);
v___x_1311_ = v___x_1286_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1313_; 
v_reuseFailAlloc_1313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1313_, 0, v___x_1300_);
lean_ctor_set(v_reuseFailAlloc_1313_, 1, v___x_1309_);
v___x_1311_ = v_reuseFailAlloc_1313_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
v_a_1283_ = v___x_1311_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_1317_; 
lean_dec(v_val_1299_);
if (v_isShared_1297_ == 0)
{
v___x_1317_ = v___x_1296_;
goto v_reusejp_1316_;
}
else
{
lean_object* v_reuseFailAlloc_1324_; 
v_reuseFailAlloc_1324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1324_, 0, v_fst_1293_);
lean_ctor_set(v_reuseFailAlloc_1324_, 1, v_snd_1294_);
v___x_1317_ = v_reuseFailAlloc_1324_;
goto v_reusejp_1316_;
}
v_reusejp_1316_:
{
lean_object* v___x_1319_; 
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 1, v___x_1317_);
lean_ctor_set(v___x_1291_, 0, v___x_1301_);
v___x_1319_ = v___x_1291_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1323_; 
v_reuseFailAlloc_1323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1323_, 0, v___x_1301_);
lean_ctor_set(v_reuseFailAlloc_1323_, 1, v___x_1317_);
v___x_1319_ = v_reuseFailAlloc_1323_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
lean_object* v___x_1321_; 
if (v_isShared_1287_ == 0)
{
lean_ctor_set(v___x_1286_, 1, v___x_1319_);
lean_ctor_set(v___x_1286_, 0, v___x_1300_);
v___x_1321_ = v___x_1286_;
goto v_reusejp_1320_;
}
else
{
lean_object* v_reuseFailAlloc_1322_; 
v_reuseFailAlloc_1322_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1322_, 0, v___x_1300_);
lean_ctor_set(v_reuseFailAlloc_1322_, 1, v___x_1319_);
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
else
{
lean_object* v___x_1325_; lean_object* v___x_1327_; 
v___x_1325_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___closed__0));
if (v_isShared_1297_ == 0)
{
v___x_1327_ = v___x_1296_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1334_; 
v_reuseFailAlloc_1334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1334_, 0, v_fst_1293_);
lean_ctor_set(v_reuseFailAlloc_1334_, 1, v_snd_1294_);
v___x_1327_ = v_reuseFailAlloc_1334_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
lean_object* v___x_1329_; 
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 1, v___x_1327_);
v___x_1329_ = v___x_1291_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1333_; 
v_reuseFailAlloc_1333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1333_, 0, v_fst_1289_);
lean_ctor_set(v_reuseFailAlloc_1333_, 1, v___x_1327_);
v___x_1329_ = v_reuseFailAlloc_1333_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
lean_object* v___x_1331_; 
if (v_isShared_1287_ == 0)
{
lean_ctor_set(v___x_1286_, 1, v___x_1329_);
lean_ctor_set(v___x_1286_, 0, v___x_1325_);
v___x_1331_ = v___x_1286_;
goto v_reusejp_1330_;
}
else
{
lean_object* v_reuseFailAlloc_1332_; 
v_reuseFailAlloc_1332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1332_, 0, v___x_1325_);
lean_ctor_set(v_reuseFailAlloc_1332_, 1, v___x_1329_);
v___x_1331_ = v_reuseFailAlloc_1332_;
goto v_reusejp_1330_;
}
v_reusejp_1330_:
{
return v___x_1331_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg___boxed(lean_object* v_next_1339_, lean_object* v_a_1340_){
_start:
{
lean_object* v_res_1341_; 
v_res_1341_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg(v_next_1339_, v_a_1340_);
lean_dec(v_next_1339_);
return v_res_1341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg(lean_object* v_upperBound_1342_, lean_object* v_n_1343_, lean_object* v_a_1344_, lean_object* v_b_1345_){
_start:
{
lean_object* v_a_1347_; uint8_t v___x_1351_; 
v___x_1351_ = lean_nat_dec_lt(v_a_1344_, v_upperBound_1342_);
if (v___x_1351_ == 0)
{
lean_dec(v_a_1344_);
return v_b_1345_;
}
else
{
lean_object* v_snd_1352_; lean_object* v___x_1354_; uint8_t v_isShared_1355_; uint8_t v_isSharedCheck_1426_; 
v_snd_1352_ = lean_ctor_get(v_b_1345_, 1);
v_isSharedCheck_1426_ = !lean_is_exclusive(v_b_1345_);
if (v_isSharedCheck_1426_ == 0)
{
lean_object* v_unused_1427_; 
v_unused_1427_ = lean_ctor_get(v_b_1345_, 0);
lean_dec(v_unused_1427_);
v___x_1354_ = v_b_1345_;
v_isShared_1355_ = v_isSharedCheck_1426_;
goto v_resetjp_1353_;
}
else
{
lean_inc(v_snd_1352_);
lean_dec(v_b_1345_);
v___x_1354_ = lean_box(0);
v_isShared_1355_ = v_isSharedCheck_1426_;
goto v_resetjp_1353_;
}
v_resetjp_1353_:
{
lean_object* v_fst_1356_; lean_object* v_snd_1357_; lean_object* v___x_1359_; uint8_t v_isShared_1360_; uint8_t v_isSharedCheck_1425_; 
v_fst_1356_ = lean_ctor_get(v_snd_1352_, 0);
v_snd_1357_ = lean_ctor_get(v_snd_1352_, 1);
v_isSharedCheck_1425_ = !lean_is_exclusive(v_snd_1352_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1359_ = v_snd_1352_;
v_isShared_1360_ = v_isSharedCheck_1425_;
goto v_resetjp_1358_;
}
else
{
lean_inc(v_snd_1357_);
lean_inc(v_fst_1356_);
lean_dec(v_snd_1352_);
v___x_1359_ = lean_box(0);
v_isShared_1360_ = v_isSharedCheck_1425_;
goto v_resetjp_1358_;
}
v_resetjp_1358_:
{
lean_object* v___x_1361_; lean_object* v___x_1362_; 
v___x_1361_ = lean_box(0);
v___x_1362_ = lean_array_fget_borrowed(v_fst_1356_, v_a_1344_);
if (lean_obj_tag(v___x_1362_) == 1)
{
lean_object* v_val_1363_; uint8_t v___x_1364_; 
v_val_1363_ = lean_ctor_get(v___x_1362_, 0);
v___x_1364_ = lean_nat_dec_eq(v_a_1344_, v_val_1363_);
if (v___x_1364_ == 0)
{
lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1369_; 
v___x_1365_ = lean_box(0);
lean_inc_n(v_val_1363_, 2);
v___x_1366_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_val_1363_);
lean_ctor_set(v___x_1366_, 1, v___x_1365_);
lean_inc(v_a_1344_);
v___x_1367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1367_, 0, v_a_1344_);
lean_ctor_set(v___x_1367_, 1, v___x_1366_);
if (v_isShared_1360_ == 0)
{
lean_ctor_set(v___x_1359_, 1, v___x_1367_);
lean_ctor_set(v___x_1359_, 0, v_val_1363_);
v___x_1369_ = v___x_1359_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1412_; 
v_reuseFailAlloc_1412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1412_, 0, v_val_1363_);
lean_ctor_set(v_reuseFailAlloc_1412_, 1, v___x_1367_);
v___x_1369_ = v_reuseFailAlloc_1412_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
lean_object* v___x_1371_; 
if (v_isShared_1355_ == 0)
{
lean_ctor_set(v___x_1354_, 1, v___x_1369_);
lean_ctor_set(v___x_1354_, 0, v_fst_1356_);
v___x_1371_ = v___x_1354_;
goto v_reusejp_1370_;
}
else
{
lean_object* v_reuseFailAlloc_1411_; 
v_reuseFailAlloc_1411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1411_, 0, v_fst_1356_);
lean_ctor_set(v_reuseFailAlloc_1411_, 1, v___x_1369_);
v___x_1371_ = v_reuseFailAlloc_1411_;
goto v_reusejp_1370_;
}
v_reusejp_1370_:
{
lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v_snd_1374_; lean_object* v_fst_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1410_; 
v___x_1372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1372_, 0, v___x_1361_);
lean_ctor_set(v___x_1372_, 1, v___x_1371_);
v___x_1373_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg(v_a_1344_, v___x_1372_);
v_snd_1374_ = lean_ctor_get(v___x_1373_, 1);
v_fst_1375_ = lean_ctor_get(v___x_1373_, 0);
v_isSharedCheck_1410_ = !lean_is_exclusive(v___x_1373_);
if (v_isSharedCheck_1410_ == 0)
{
v___x_1377_ = v___x_1373_;
v_isShared_1378_ = v_isSharedCheck_1410_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_snd_1374_);
lean_inc(v_fst_1375_);
lean_dec(v___x_1373_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1410_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
if (lean_obj_tag(v_fst_1375_) == 0)
{
lean_object* v_snd_1379_; lean_object* v_fst_1380_; lean_object* v___x_1382_; uint8_t v_isShared_1383_; uint8_t v_isSharedCheck_1397_; 
lean_del_object(v___x_1377_);
v_snd_1379_ = lean_ctor_get(v_snd_1374_, 1);
v_fst_1380_ = lean_ctor_get(v_snd_1374_, 0);
v_isSharedCheck_1397_ = !lean_is_exclusive(v_snd_1374_);
if (v_isSharedCheck_1397_ == 0)
{
v___x_1382_ = v_snd_1374_;
v_isShared_1383_ = v_isSharedCheck_1397_;
goto v_resetjp_1381_;
}
else
{
lean_inc(v_snd_1379_);
lean_inc(v_fst_1380_);
lean_dec(v_snd_1374_);
v___x_1382_ = lean_box(0);
v_isShared_1383_ = v_isSharedCheck_1397_;
goto v_resetjp_1381_;
}
v_resetjp_1381_:
{
lean_object* v_snd_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1395_; 
v_snd_1384_ = lean_ctor_get(v_snd_1379_, 1);
v_isSharedCheck_1395_ = !lean_is_exclusive(v_snd_1379_);
if (v_isSharedCheck_1395_ == 0)
{
lean_object* v_unused_1396_; 
v_unused_1396_ = lean_ctor_get(v_snd_1379_, 0);
lean_dec(v_unused_1396_);
v___x_1386_ = v_snd_1379_;
v_isShared_1387_ = v_isSharedCheck_1395_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_snd_1384_);
lean_dec(v_snd_1379_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1395_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1388_; lean_object* v___x_1390_; 
v___x_1388_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1388_, 0, v_snd_1384_);
lean_ctor_set(v___x_1388_, 1, v_snd_1357_);
if (v_isShared_1387_ == 0)
{
lean_ctor_set(v___x_1386_, 1, v___x_1388_);
lean_ctor_set(v___x_1386_, 0, v_fst_1380_);
v___x_1390_ = v___x_1386_;
goto v_reusejp_1389_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v_fst_1380_);
lean_ctor_set(v_reuseFailAlloc_1394_, 1, v___x_1388_);
v___x_1390_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1389_;
}
v_reusejp_1389_:
{
lean_object* v___x_1392_; 
if (v_isShared_1383_ == 0)
{
lean_ctor_set(v___x_1382_, 1, v___x_1390_);
lean_ctor_set(v___x_1382_, 0, v___x_1361_);
v___x_1392_ = v___x_1382_;
goto v_reusejp_1391_;
}
else
{
lean_object* v_reuseFailAlloc_1393_; 
v_reuseFailAlloc_1393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1393_, 0, v___x_1361_);
lean_ctor_set(v_reuseFailAlloc_1393_, 1, v___x_1390_);
v___x_1392_ = v_reuseFailAlloc_1393_;
goto v_reusejp_1391_;
}
v_reusejp_1391_:
{
v_a_1347_ = v___x_1392_;
goto v___jp_1346_;
}
}
}
}
}
else
{
lean_object* v_fst_1398_; lean_object* v___x_1400_; uint8_t v_isShared_1401_; uint8_t v_isSharedCheck_1408_; 
lean_dec(v_a_1344_);
v_fst_1398_ = lean_ctor_get(v_snd_1374_, 0);
v_isSharedCheck_1408_ = !lean_is_exclusive(v_snd_1374_);
if (v_isSharedCheck_1408_ == 0)
{
lean_object* v_unused_1409_; 
v_unused_1409_ = lean_ctor_get(v_snd_1374_, 1);
lean_dec(v_unused_1409_);
v___x_1400_ = v_snd_1374_;
v_isShared_1401_ = v_isSharedCheck_1408_;
goto v_resetjp_1399_;
}
else
{
lean_inc(v_fst_1398_);
lean_dec(v_snd_1374_);
v___x_1400_ = lean_box(0);
v_isShared_1401_ = v_isSharedCheck_1408_;
goto v_resetjp_1399_;
}
v_resetjp_1399_:
{
lean_object* v___x_1403_; 
if (v_isShared_1401_ == 0)
{
lean_ctor_set(v___x_1400_, 1, v_snd_1357_);
v___x_1403_ = v___x_1400_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1407_; 
v_reuseFailAlloc_1407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1407_, 0, v_fst_1398_);
lean_ctor_set(v_reuseFailAlloc_1407_, 1, v_snd_1357_);
v___x_1403_ = v_reuseFailAlloc_1407_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
lean_object* v___x_1405_; 
if (v_isShared_1378_ == 0)
{
lean_ctor_set(v___x_1377_, 1, v___x_1403_);
v___x_1405_ = v___x_1377_;
goto v_reusejp_1404_;
}
else
{
lean_object* v_reuseFailAlloc_1406_; 
v_reuseFailAlloc_1406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1406_, 0, v_fst_1375_);
lean_ctor_set(v_reuseFailAlloc_1406_, 1, v___x_1403_);
v___x_1405_ = v_reuseFailAlloc_1406_;
goto v_reusejp_1404_;
}
v_reusejp_1404_:
{
return v___x_1405_;
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
lean_object* v___x_1414_; 
if (v_isShared_1360_ == 0)
{
v___x_1414_ = v___x_1359_;
goto v_reusejp_1413_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_fst_1356_);
lean_ctor_set(v_reuseFailAlloc_1418_, 1, v_snd_1357_);
v___x_1414_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1413_;
}
v_reusejp_1413_:
{
lean_object* v___x_1416_; 
if (v_isShared_1355_ == 0)
{
lean_ctor_set(v___x_1354_, 1, v___x_1414_);
lean_ctor_set(v___x_1354_, 0, v___x_1361_);
v___x_1416_ = v___x_1354_;
goto v_reusejp_1415_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v___x_1361_);
lean_ctor_set(v_reuseFailAlloc_1417_, 1, v___x_1414_);
v___x_1416_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1415_;
}
v_reusejp_1415_:
{
v_a_1347_ = v___x_1416_;
goto v___jp_1346_;
}
}
}
}
else
{
lean_object* v___x_1420_; 
if (v_isShared_1360_ == 0)
{
v___x_1420_ = v___x_1359_;
goto v_reusejp_1419_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v_fst_1356_);
lean_ctor_set(v_reuseFailAlloc_1424_, 1, v_snd_1357_);
v___x_1420_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1419_;
}
v_reusejp_1419_:
{
lean_object* v___x_1422_; 
if (v_isShared_1355_ == 0)
{
lean_ctor_set(v___x_1354_, 1, v___x_1420_);
lean_ctor_set(v___x_1354_, 0, v___x_1361_);
v___x_1422_ = v___x_1354_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v___x_1361_);
lean_ctor_set(v_reuseFailAlloc_1423_, 1, v___x_1420_);
v___x_1422_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
v_a_1347_ = v___x_1422_;
goto v___jp_1346_;
}
}
}
}
}
}
v___jp_1346_:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1348_ = lean_unsigned_to_nat(1u);
v___x_1349_ = lean_nat_add(v_a_1344_, v___x_1348_);
lean_dec(v_a_1344_);
v_a_1344_ = v___x_1349_;
v_b_1345_ = v_a_1347_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg___boxed(lean_object* v_upperBound_1428_, lean_object* v_n_1429_, lean_object* v_a_1430_, lean_object* v_b_1431_){
_start:
{
lean_object* v_res_1432_; 
v_res_1432_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg(v_upperBound_1428_, v_n_1429_, v_a_1430_, v_b_1431_);
lean_dec(v_n_1429_);
lean_dec(v_upperBound_1428_);
return v_res_1432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm(lean_object* v_n_1433_, lean_object* v_map_1434_){
_start:
{
lean_object* v___x_1435_; lean_object* v_perm_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v_fst_1441_; 
v___x_1435_ = lean_unsigned_to_nat(0u);
v_perm_1436_ = lean_box(0);
v___x_1437_ = lean_box(0);
v___x_1438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1438_, 0, v_map_1434_);
lean_ctor_set(v___x_1438_, 1, v_perm_1436_);
v___x_1439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1439_, 0, v___x_1437_);
lean_ctor_set(v___x_1439_, 1, v___x_1438_);
v___x_1440_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg(v_n_1433_, v_n_1433_, v___x_1435_, v___x_1439_);
v_fst_1441_ = lean_ctor_get(v___x_1440_, 0);
lean_inc(v_fst_1441_);
if (lean_obj_tag(v_fst_1441_) == 0)
{
lean_object* v_snd_1442_; lean_object* v_snd_1443_; 
v_snd_1442_ = lean_ctor_get(v___x_1440_, 1);
lean_inc(v_snd_1442_);
lean_dec_ref(v___x_1440_);
v_snd_1443_ = lean_ctor_get(v_snd_1442_, 1);
lean_inc(v_snd_1443_);
lean_dec(v_snd_1442_);
return v_snd_1443_;
}
else
{
lean_object* v_val_1444_; 
lean_dec_ref(v___x_1440_);
v_val_1444_ = lean_ctor_get(v_fst_1441_, 0);
lean_inc(v_val_1444_);
lean_dec_ref_known(v_fst_1441_, 1);
return v_val_1444_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm___boxed(lean_object* v_n_1445_, lean_object* v_map_1446_){
_start:
{
lean_object* v_res_1447_; 
v_res_1447_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm(v_n_1445_, v_map_1446_);
lean_dec(v_n_1445_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0(lean_object* v_n_1448_, lean_object* v_next_1449_, lean_object* v_inst_1450_, lean_object* v_a_1451_){
_start:
{
lean_object* v___x_1452_; 
v___x_1452_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___redArg(v_next_1449_, v_a_1451_);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0___boxed(lean_object* v_n_1453_, lean_object* v_next_1454_, lean_object* v_inst_1455_, lean_object* v_a_1456_){
_start:
{
lean_object* v_res_1457_; 
v_res_1457_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__0(v_n_1453_, v_next_1454_, v_inst_1455_, v_a_1456_);
lean_dec(v_next_1454_);
lean_dec(v_n_1453_);
return v_res_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1(lean_object* v_upperBound_1458_, lean_object* v_n_1459_, lean_object* v_inst_1460_, lean_object* v_R_1461_, lean_object* v_a_1462_, lean_object* v_b_1463_, lean_object* v_c_1464_){
_start:
{
lean_object* v___x_1465_; 
v___x_1465_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___redArg(v_upperBound_1458_, v_n_1459_, v_a_1462_, v_b_1463_);
return v___x_1465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1___boxed(lean_object* v_upperBound_1466_, lean_object* v_n_1467_, lean_object* v_inst_1468_, lean_object* v_R_1469_, lean_object* v_a_1470_, lean_object* v_b_1471_, lean_object* v_c_1472_){
_start:
{
lean_object* v_res_1473_; 
v_res_1473_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm_spec__1(v_upperBound_1466_, v_n_1467_, v_inst_1468_, v_R_1469_, v_a_1470_, v_b_1471_, v_c_1472_);
lean_dec(v_n_1467_);
lean_dec(v_upperBound_1466_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0(lean_object* v_inst_1474_, lean_object* v_tgt_1475_, lean_object* v_x_1476_){
_start:
{
lean_object* v___x_1477_; 
v___x_1477_ = l_Array_finIdxOf_x3f___redArg(v_inst_1474_, v_tgt_1475_, v_x_1476_);
if (lean_obj_tag(v___x_1477_) == 0)
{
if (lean_obj_tag(v___x_1477_) == 0)
{
lean_object* v___x_1478_; 
v___x_1478_ = lean_box(0);
return v___x_1478_;
}
else
{
lean_object* v___x_1479_; 
v___x_1479_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1479_, 0, v___x_1477_);
return v___x_1479_;
}
}
else
{
lean_object* v_val_1480_; lean_object* v___x_1482_; uint8_t v_isShared_1483_; uint8_t v_isSharedCheck_1488_; 
v_val_1480_ = lean_ctor_get(v___x_1477_, 0);
v_isSharedCheck_1488_ = !lean_is_exclusive(v___x_1477_);
if (v_isSharedCheck_1488_ == 0)
{
v___x_1482_ = v___x_1477_;
v_isShared_1483_ = v_isSharedCheck_1488_;
goto v_resetjp_1481_;
}
else
{
lean_inc(v_val_1480_);
lean_dec(v___x_1477_);
v___x_1482_ = lean_box(0);
v_isShared_1483_ = v_isSharedCheck_1488_;
goto v_resetjp_1481_;
}
v_resetjp_1481_:
{
lean_object* v___x_1485_; 
if (v_isShared_1483_ == 0)
{
v___x_1485_ = v___x_1482_;
goto v_reusejp_1484_;
}
else
{
lean_object* v_reuseFailAlloc_1487_; 
v_reuseFailAlloc_1487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1487_, 0, v_val_1480_);
v___x_1485_ = v_reuseFailAlloc_1487_;
goto v_reusejp_1484_;
}
v_reusejp_1484_:
{
lean_object* v___x_1486_; 
v___x_1486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1486_, 0, v___x_1485_);
return v___x_1486_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0___boxed(lean_object* v_inst_1489_, lean_object* v_tgt_1490_, lean_object* v_x_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0(v_inst_1489_, v_tgt_1490_, v_x_1491_);
lean_dec_ref(v_tgt_1490_);
return v_res_1492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg(lean_object* v_inst_1514_, lean_object* v_src_1515_, lean_object* v_tgt_1516_){
_start:
{
lean_object* v_n_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; 
v_n_1517_ = lean_array_get_size(v_src_1515_);
v___x_1518_ = lean_array_get_size(v_tgt_1516_);
v___x_1519_ = lean_nat_dec_eq(v_n_1517_, v___x_1518_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1520_; 
lean_dec_ref(v_tgt_1516_);
lean_dec_ref(v_src_1515_);
lean_dec_ref(v_inst_1514_);
v___x_1520_ = lean_box(0);
return v___x_1520_;
}
else
{
lean_object* v___f_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; 
v___f_1521_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1521_, 0, v_inst_1514_);
lean_closure_set(v___f_1521_, 1, v_tgt_1516_);
v___x_1522_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__9));
v___x_1523_ = lean_unsigned_to_nat(0u);
v___x_1524_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg___closed__10));
v___x_1525_ = l___private_Init_Data_Vector_Basic_0__Vector_mapM_go(lean_box(0), lean_box(0), lean_box(0), v_n_1517_, v___x_1522_, v___f_1521_, v_src_1515_, v___x_1523_, lean_box(0), v___x_1524_);
if (lean_obj_tag(v___x_1525_) == 0)
{
lean_object* v___x_1526_; 
v___x_1526_ = lean_box(0);
return v___x_1526_;
}
else
{
lean_object* v_val_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1535_; 
v_val_1527_ = lean_ctor_get(v___x_1525_, 0);
v_isSharedCheck_1535_ = !lean_is_exclusive(v___x_1525_);
if (v_isSharedCheck_1535_ == 0)
{
v___x_1529_ = v___x_1525_;
v_isShared_1530_ = v_isSharedCheck_1535_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_val_1527_);
lean_dec(v___x_1525_);
v___x_1529_ = lean_box(0);
v_isShared_1530_ = v_isSharedCheck_1535_;
goto v_resetjp_1528_;
}
v_resetjp_1528_:
{
lean_object* v___x_1531_; lean_object* v___x_1533_; 
v___x_1531_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm(v_n_1517_, v_val_1527_);
if (v_isShared_1530_ == 0)
{
lean_ctor_set(v___x_1529_, 0, v___x_1531_);
v___x_1533_ = v___x_1529_;
goto v_reusejp_1532_;
}
else
{
lean_object* v_reuseFailAlloc_1534_; 
v_reuseFailAlloc_1534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1534_, 0, v___x_1531_);
v___x_1533_ = v_reuseFailAlloc_1534_;
goto v_reusejp_1532_;
}
v_reusejp_1532_:
{
return v___x_1533_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_getPermutation(lean_object* v_00_u03b1_1536_, lean_object* v_inst_1537_, lean_object* v_src_1538_, lean_object* v_tgt_1539_){
_start:
{
lean_object* v___x_1540_; 
v___x_1540_ = lp_mathlib_Mathlib_Tactic_Translate_getPermutation___redArg(v_inst_1537_, v_src_1538_, v_tgt_1539_);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(lean_object* v_x_1541_){
_start:
{
if (lean_obj_tag(v_x_1541_) == 7)
{
lean_object* v_body_1542_; lean_object* v_d_1543_; lean_object* v___x_1547_; uint8_t v___y_1549_; uint8_t v___x_1550_; 
v_body_1542_ = lean_ctor_get(v_x_1541_, 2);
v_d_1543_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(v_body_1542_);
v___x_1547_ = lean_unsigned_to_nat(0u);
v___x_1550_ = lean_nat_dec_eq(v_d_1543_, v___x_1547_);
if (v___x_1550_ == 0)
{
v___y_1549_ = v___x_1550_;
goto v___jp_1548_;
}
else
{
uint8_t v___x_1551_; 
v___x_1551_ = lean_expr_has_loose_bvar(v_body_1542_, v___x_1547_);
if (v___x_1551_ == 0)
{
v___y_1549_ = v___x_1550_;
goto v___jp_1548_;
}
else
{
goto v___jp_1544_;
}
}
v___jp_1544_:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; 
v___x_1545_ = lean_unsigned_to_nat(1u);
v___x_1546_ = lean_nat_add(v_d_1543_, v___x_1545_);
lean_dec(v_d_1543_);
return v___x_1546_;
}
v___jp_1548_:
{
if (v___y_1549_ == 0)
{
goto v___jp_1544_;
}
else
{
lean_dec(v_d_1543_);
return v___x_1547_;
}
}
}
else
{
lean_object* v___x_1552_; 
v___x_1552_ = lean_unsigned_to_nat(0u);
return v___x_1552_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth___boxed(lean_object* v_x_1553_){
_start:
{
lean_object* v_res_1554_; 
v_res_1554_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(v_x_1553_);
lean_dec_ref(v_x_1553_);
return v_res_1554_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1555_; lean_object* v___f_1556_; 
v___x_1555_ = l_instMonadBaseIO;
v___f_1556_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_1556_, 0, v___x_1555_);
return v___f_1556_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1557_; lean_object* v___f_1558_; 
v___x_1557_ = l_instMonadBaseIO;
v___f_1558_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__3), 5, 1);
lean_closure_set(v___f_1558_, 0, v___x_1557_);
return v___f_1558_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2(void){
_start:
{
lean_object* v___x_1559_; lean_object* v___f_1560_; 
v___x_1559_ = l_instMonadBaseIO;
v___f_1560_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__6), 5, 1);
lean_closure_set(v___f_1560_, 0, v___x_1559_);
return v___f_1560_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1561_; lean_object* v___f_1562_; 
v___x_1561_ = l_instMonadBaseIO;
v___f_1562_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_1562_, 0, v___x_1561_);
return v___f_1562_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4(void){
_start:
{
lean_object* v___x_1563_; lean_object* v___f_1564_; 
v___x_1563_ = l_instMonadBaseIO;
v___f_1564_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__11), 5, 1);
lean_closure_set(v___f_1564_, 0, v___x_1563_);
return v___f_1564_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5(void){
_start:
{
lean_object* v___x_1565_; lean_object* v___x_1566_; 
v___x_1565_ = l_instMonadBaseIO;
v___x_1566_ = lean_alloc_closure((void*)(l_OptionT_pure), 4, 2);
lean_closure_set(v___x_1566_, 0, lean_box(0));
lean_closure_set(v___x_1566_, 1, v___x_1565_);
return v___x_1566_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6(void){
_start:
{
lean_object* v___x_1567_; lean_object* v___x_1568_; 
v___x_1567_ = l_instMonadBaseIO;
v___x_1568_ = lean_alloc_closure((void*)(l_OptionT_bind), 6, 2);
lean_closure_set(v___x_1568_, 0, lean_box(0));
lean_closure_set(v___x_1568_, 1, v___x_1567_);
return v___x_1568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3(lean_object* v_n_1569_, lean_object* v_msg_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
lean_object* v___f_1574_; lean_object* v___f_1575_; lean_object* v___f_1576_; lean_object* v___f_1577_; lean_object* v___f_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___f_1588_; lean_object* v___x_16777__overap_1589_; lean_object* v___x_1590_; 
v___f_1574_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__0);
v___f_1575_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__1);
v___f_1576_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__2);
v___f_1577_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__3);
v___f_1578_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__4);
v___x_1579_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1579_, 0, v___f_1574_);
lean_ctor_set(v___x_1579_, 1, v___f_1575_);
v___x_1580_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__5);
v___x_1581_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1581_, 0, v___x_1579_);
lean_ctor_set(v___x_1581_, 1, v___x_1580_);
lean_ctor_set(v___x_1581_, 2, v___f_1576_);
lean_ctor_set(v___x_1581_, 3, v___f_1577_);
lean_ctor_set(v___x_1581_, 4, v___f_1578_);
v___x_1582_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___closed__6);
v___x_1583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1581_);
lean_ctor_set(v___x_1583_, 1, v___x_1582_);
v___x_1584_ = l_StateRefT_x27_instMonad___redArg(v___x_1583_);
v___x_1585_ = lean_box(0);
v___x_1586_ = lean_mk_array(v_n_1569_, v___x_1585_);
v___x_1587_ = l_instInhabitedOfMonad___redArg(v___x_1584_, v___x_1586_);
v___f_1588_ = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1588_, 0, v___x_1587_);
v___x_16777__overap_1589_ = lean_panic_fn_borrowed(v___f_1588_, v_msg_1570_);
lean_dec_ref(v___f_1588_);
lean_inc(v___y_1572_);
lean_inc_ref(v___y_1571_);
v___x_1590_ = lean_apply_3(v___x_16777__overap_1589_, v___y_1571_, v___y_1572_, lean_box(0));
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3___boxed(lean_object* v_n_1591_, lean_object* v_msg_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_){
_start:
{
lean_object* v_res_1596_; 
v_res_1596_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3(v_n_1591_, v_msg_1592_, v___y_1593_, v___y_1594_);
lean_dec(v___y_1594_);
lean_dec_ref(v___y_1593_);
return v_res_1596_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(lean_object* v_a_1597_, lean_object* v_x_1598_){
_start:
{
if (lean_obj_tag(v_x_1598_) == 0)
{
uint8_t v___x_1599_; 
v___x_1599_ = 0;
return v___x_1599_;
}
else
{
lean_object* v_key_1600_; lean_object* v_tail_1601_; uint8_t v___y_1603_; lean_object* v_fst_1605_; lean_object* v_snd_1606_; lean_object* v_fst_1607_; lean_object* v_snd_1608_; uint8_t v___x_1609_; 
v_key_1600_ = lean_ctor_get(v_x_1598_, 0);
v_tail_1601_ = lean_ctor_get(v_x_1598_, 2);
v_fst_1605_ = lean_ctor_get(v_key_1600_, 0);
v_snd_1606_ = lean_ctor_get(v_key_1600_, 1);
v_fst_1607_ = lean_ctor_get(v_a_1597_, 0);
v_snd_1608_ = lean_ctor_get(v_a_1597_, 1);
v___x_1609_ = lean_expr_eqv(v_fst_1605_, v_fst_1607_);
if (v___x_1609_ == 0)
{
v___y_1603_ = v___x_1609_;
goto v___jp_1602_;
}
else
{
uint8_t v___x_1610_; 
v___x_1610_ = lean_expr_eqv(v_snd_1606_, v_snd_1608_);
v___y_1603_ = v___x_1610_;
goto v___jp_1602_;
}
v___jp_1602_:
{
if (v___y_1603_ == 0)
{
v_x_1598_ = v_tail_1601_;
goto _start;
}
else
{
return v___y_1603_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg___boxed(lean_object* v_a_1611_, lean_object* v_x_1612_){
_start:
{
uint8_t v_res_1613_; lean_object* v_r_1614_; 
v_res_1613_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(v_a_1611_, v_x_1612_);
lean_dec(v_x_1612_);
lean_dec_ref(v_a_1611_);
v_r_1614_ = lean_box(v_res_1613_);
return v_r_1614_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg(lean_object* v_m_1615_, lean_object* v_a_1616_){
_start:
{
lean_object* v_buckets_1617_; lean_object* v_fst_1618_; lean_object* v_snd_1619_; lean_object* v___x_1620_; uint64_t v___x_1621_; uint64_t v___x_1622_; uint64_t v___x_1623_; uint64_t v___x_1624_; uint64_t v___x_1625_; uint64_t v_fold_1626_; uint64_t v___x_1627_; uint64_t v___x_1628_; uint64_t v___x_1629_; size_t v___x_1630_; size_t v___x_1631_; size_t v___x_1632_; size_t v___x_1633_; size_t v___x_1634_; lean_object* v___x_1635_; uint8_t v___x_1636_; 
v_buckets_1617_ = lean_ctor_get(v_m_1615_, 1);
v_fst_1618_ = lean_ctor_get(v_a_1616_, 0);
v_snd_1619_ = lean_ctor_get(v_a_1616_, 1);
v___x_1620_ = lean_array_get_size(v_buckets_1617_);
v___x_1621_ = l_Lean_Expr_hash(v_fst_1618_);
v___x_1622_ = l_Lean_Expr_hash(v_snd_1619_);
v___x_1623_ = lean_uint64_mix_hash(v___x_1621_, v___x_1622_);
v___x_1624_ = 32ULL;
v___x_1625_ = lean_uint64_shift_right(v___x_1623_, v___x_1624_);
v_fold_1626_ = lean_uint64_xor(v___x_1623_, v___x_1625_);
v___x_1627_ = 16ULL;
v___x_1628_ = lean_uint64_shift_right(v_fold_1626_, v___x_1627_);
v___x_1629_ = lean_uint64_xor(v_fold_1626_, v___x_1628_);
v___x_1630_ = lean_uint64_to_usize(v___x_1629_);
v___x_1631_ = lean_usize_of_nat(v___x_1620_);
v___x_1632_ = ((size_t)1ULL);
v___x_1633_ = lean_usize_sub(v___x_1631_, v___x_1632_);
v___x_1634_ = lean_usize_land(v___x_1630_, v___x_1633_);
v___x_1635_ = lean_array_uget_borrowed(v_buckets_1617_, v___x_1634_);
v___x_1636_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(v_a_1616_, v___x_1635_);
return v___x_1636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg___boxed(lean_object* v_m_1637_, lean_object* v_a_1638_){
_start:
{
uint8_t v_res_1639_; lean_object* v_r_1640_; 
v_res_1639_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg(v_m_1637_, v_a_1638_);
lean_dec_ref(v_a_1638_);
lean_dec_ref(v_m_1637_);
v_r_1640_ = lean_box(v_res_1639_);
return v_r_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg(lean_object* v_a_1641_, lean_object* v_x_1642_){
_start:
{
if (lean_obj_tag(v_x_1642_) == 0)
{
lean_object* v___x_1643_; 
v___x_1643_ = lean_box(0);
return v___x_1643_;
}
else
{
lean_object* v_key_1644_; lean_object* v_value_1645_; lean_object* v_tail_1646_; uint8_t v___x_1647_; 
v_key_1644_ = lean_ctor_get(v_x_1642_, 0);
v_value_1645_ = lean_ctor_get(v_x_1642_, 1);
v_tail_1646_ = lean_ctor_get(v_x_1642_, 2);
v___x_1647_ = l_Lean_instBEqFVarId_beq(v_key_1644_, v_a_1641_);
if (v___x_1647_ == 0)
{
v_x_1642_ = v_tail_1646_;
goto _start;
}
else
{
lean_object* v___x_1649_; 
lean_inc(v_value_1645_);
v___x_1649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1649_, 0, v_value_1645_);
return v___x_1649_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg___boxed(lean_object* v_a_1650_, lean_object* v_x_1651_){
_start:
{
lean_object* v_res_1652_; 
v_res_1652_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg(v_a_1650_, v_x_1651_);
lean_dec(v_x_1651_);
lean_dec(v_a_1650_);
return v_res_1652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(lean_object* v_m_1653_, lean_object* v_a_1654_){
_start:
{
lean_object* v_buckets_1655_; lean_object* v___x_1656_; uint64_t v___x_1657_; uint64_t v___x_1658_; uint64_t v___x_1659_; uint64_t v_fold_1660_; uint64_t v___x_1661_; uint64_t v___x_1662_; uint64_t v___x_1663_; size_t v___x_1664_; size_t v___x_1665_; size_t v___x_1666_; size_t v___x_1667_; size_t v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; 
v_buckets_1655_ = lean_ctor_get(v_m_1653_, 1);
v___x_1656_ = lean_array_get_size(v_buckets_1655_);
v___x_1657_ = l_Lean_instHashableFVarId_hash(v_a_1654_);
v___x_1658_ = 32ULL;
v___x_1659_ = lean_uint64_shift_right(v___x_1657_, v___x_1658_);
v_fold_1660_ = lean_uint64_xor(v___x_1657_, v___x_1659_);
v___x_1661_ = 16ULL;
v___x_1662_ = lean_uint64_shift_right(v_fold_1660_, v___x_1661_);
v___x_1663_ = lean_uint64_xor(v_fold_1660_, v___x_1662_);
v___x_1664_ = lean_uint64_to_usize(v___x_1663_);
v___x_1665_ = lean_usize_of_nat(v___x_1656_);
v___x_1666_ = ((size_t)1ULL);
v___x_1667_ = lean_usize_sub(v___x_1665_, v___x_1666_);
v___x_1668_ = lean_usize_land(v___x_1664_, v___x_1667_);
v___x_1669_ = lean_array_uget_borrowed(v_buckets_1655_, v___x_1668_);
v___x_1670_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg(v_a_1654_, v___x_1669_);
return v___x_1670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg___boxed(lean_object* v_m_1671_, lean_object* v_a_1672_){
_start:
{
lean_object* v_res_1673_; 
v_res_1673_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(v_m_1671_, v_a_1672_);
lean_dec(v_a_1672_);
lean_dec_ref(v_m_1671_);
return v_res_1673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6___redArg(lean_object* v_x_1674_, lean_object* v_x_1675_){
_start:
{
if (lean_obj_tag(v_x_1675_) == 0)
{
return v_x_1674_;
}
else
{
lean_object* v_key_1676_; lean_object* v_value_1677_; lean_object* v_tail_1678_; lean_object* v___x_1680_; uint8_t v_isShared_1681_; uint8_t v_isSharedCheck_1705_; 
v_key_1676_ = lean_ctor_get(v_x_1675_, 0);
v_value_1677_ = lean_ctor_get(v_x_1675_, 1);
v_tail_1678_ = lean_ctor_get(v_x_1675_, 2);
v_isSharedCheck_1705_ = !lean_is_exclusive(v_x_1675_);
if (v_isSharedCheck_1705_ == 0)
{
v___x_1680_ = v_x_1675_;
v_isShared_1681_ = v_isSharedCheck_1705_;
goto v_resetjp_1679_;
}
else
{
lean_inc(v_tail_1678_);
lean_inc(v_value_1677_);
lean_inc(v_key_1676_);
lean_dec(v_x_1675_);
v___x_1680_ = lean_box(0);
v_isShared_1681_ = v_isSharedCheck_1705_;
goto v_resetjp_1679_;
}
v_resetjp_1679_:
{
lean_object* v_fst_1682_; lean_object* v_snd_1683_; lean_object* v___x_1684_; uint64_t v___x_1685_; uint64_t v___x_1686_; uint64_t v___x_1687_; uint64_t v___x_1688_; uint64_t v___x_1689_; uint64_t v_fold_1690_; uint64_t v___x_1691_; uint64_t v___x_1692_; uint64_t v___x_1693_; size_t v___x_1694_; size_t v___x_1695_; size_t v___x_1696_; size_t v___x_1697_; size_t v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1701_; 
v_fst_1682_ = lean_ctor_get(v_key_1676_, 0);
v_snd_1683_ = lean_ctor_get(v_key_1676_, 1);
v___x_1684_ = lean_array_get_size(v_x_1674_);
v___x_1685_ = l_Lean_Expr_hash(v_fst_1682_);
v___x_1686_ = l_Lean_Expr_hash(v_snd_1683_);
v___x_1687_ = lean_uint64_mix_hash(v___x_1685_, v___x_1686_);
v___x_1688_ = 32ULL;
v___x_1689_ = lean_uint64_shift_right(v___x_1687_, v___x_1688_);
v_fold_1690_ = lean_uint64_xor(v___x_1687_, v___x_1689_);
v___x_1691_ = 16ULL;
v___x_1692_ = lean_uint64_shift_right(v_fold_1690_, v___x_1691_);
v___x_1693_ = lean_uint64_xor(v_fold_1690_, v___x_1692_);
v___x_1694_ = lean_uint64_to_usize(v___x_1693_);
v___x_1695_ = lean_usize_of_nat(v___x_1684_);
v___x_1696_ = ((size_t)1ULL);
v___x_1697_ = lean_usize_sub(v___x_1695_, v___x_1696_);
v___x_1698_ = lean_usize_land(v___x_1694_, v___x_1697_);
v___x_1699_ = lean_array_uget_borrowed(v_x_1674_, v___x_1698_);
lean_inc(v___x_1699_);
if (v_isShared_1681_ == 0)
{
lean_ctor_set(v___x_1680_, 2, v___x_1699_);
v___x_1701_ = v___x_1680_;
goto v_reusejp_1700_;
}
else
{
lean_object* v_reuseFailAlloc_1704_; 
v_reuseFailAlloc_1704_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1704_, 0, v_key_1676_);
lean_ctor_set(v_reuseFailAlloc_1704_, 1, v_value_1677_);
lean_ctor_set(v_reuseFailAlloc_1704_, 2, v___x_1699_);
v___x_1701_ = v_reuseFailAlloc_1704_;
goto v_reusejp_1700_;
}
v_reusejp_1700_:
{
lean_object* v___x_1702_; 
v___x_1702_ = lean_array_uset(v_x_1674_, v___x_1698_, v___x_1701_);
v_x_1674_ = v___x_1702_;
v_x_1675_ = v_tail_1678_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3___redArg(lean_object* v_i_1706_, lean_object* v_source_1707_, lean_object* v_target_1708_){
_start:
{
lean_object* v___x_1709_; uint8_t v___x_1710_; 
v___x_1709_ = lean_array_get_size(v_source_1707_);
v___x_1710_ = lean_nat_dec_lt(v_i_1706_, v___x_1709_);
if (v___x_1710_ == 0)
{
lean_dec_ref(v_source_1707_);
lean_dec(v_i_1706_);
return v_target_1708_;
}
else
{
lean_object* v_es_1711_; lean_object* v___x_1712_; lean_object* v_source_1713_; lean_object* v_target_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; 
v_es_1711_ = lean_array_fget(v_source_1707_, v_i_1706_);
v___x_1712_ = lean_box(0);
v_source_1713_ = lean_array_fset(v_source_1707_, v_i_1706_, v___x_1712_);
v_target_1714_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6___redArg(v_target_1708_, v_es_1711_);
v___x_1715_ = lean_unsigned_to_nat(1u);
v___x_1716_ = lean_nat_add(v_i_1706_, v___x_1715_);
lean_dec(v_i_1706_);
v_i_1706_ = v___x_1716_;
v_source_1707_ = v_source_1713_;
v_target_1708_ = v_target_1714_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1___redArg(lean_object* v_data_1718_){
_start:
{
lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v_nbuckets_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; 
v___x_1719_ = lean_array_get_size(v_data_1718_);
v___x_1720_ = lean_unsigned_to_nat(2u);
v_nbuckets_1721_ = lean_nat_mul(v___x_1719_, v___x_1720_);
v___x_1722_ = lean_unsigned_to_nat(0u);
v___x_1723_ = lean_box(0);
v___x_1724_ = lean_mk_array(v_nbuckets_1721_, v___x_1723_);
v___x_1725_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3___redArg(v___x_1722_, v_data_1718_, v___x_1724_);
return v___x_1725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0___redArg(lean_object* v_m_1726_, lean_object* v_a_1727_, lean_object* v_b_1728_){
_start:
{
lean_object* v_size_1729_; lean_object* v_buckets_1730_; lean_object* v_fst_1731_; lean_object* v_snd_1732_; lean_object* v___x_1733_; uint64_t v___x_1734_; uint64_t v___x_1735_; uint64_t v___x_1736_; uint64_t v___x_1737_; uint64_t v___x_1738_; uint64_t v_fold_1739_; uint64_t v___x_1740_; uint64_t v___x_1741_; uint64_t v___x_1742_; size_t v___x_1743_; size_t v___x_1744_; size_t v___x_1745_; size_t v___x_1746_; size_t v___x_1747_; lean_object* v_bkt_1748_; uint8_t v___x_1749_; 
v_size_1729_ = lean_ctor_get(v_m_1726_, 0);
v_buckets_1730_ = lean_ctor_get(v_m_1726_, 1);
v_fst_1731_ = lean_ctor_get(v_a_1727_, 0);
v_snd_1732_ = lean_ctor_get(v_a_1727_, 1);
v___x_1733_ = lean_array_get_size(v_buckets_1730_);
v___x_1734_ = l_Lean_Expr_hash(v_fst_1731_);
v___x_1735_ = l_Lean_Expr_hash(v_snd_1732_);
v___x_1736_ = lean_uint64_mix_hash(v___x_1734_, v___x_1735_);
v___x_1737_ = 32ULL;
v___x_1738_ = lean_uint64_shift_right(v___x_1736_, v___x_1737_);
v_fold_1739_ = lean_uint64_xor(v___x_1736_, v___x_1738_);
v___x_1740_ = 16ULL;
v___x_1741_ = lean_uint64_shift_right(v_fold_1739_, v___x_1740_);
v___x_1742_ = lean_uint64_xor(v_fold_1739_, v___x_1741_);
v___x_1743_ = lean_uint64_to_usize(v___x_1742_);
v___x_1744_ = lean_usize_of_nat(v___x_1733_);
v___x_1745_ = ((size_t)1ULL);
v___x_1746_ = lean_usize_sub(v___x_1744_, v___x_1745_);
v___x_1747_ = lean_usize_land(v___x_1743_, v___x_1746_);
v_bkt_1748_ = lean_array_uget_borrowed(v_buckets_1730_, v___x_1747_);
v___x_1749_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(v_a_1727_, v_bkt_1748_);
if (v___x_1749_ == 0)
{
lean_object* v___x_1751_; uint8_t v_isShared_1752_; uint8_t v_isSharedCheck_1770_; 
lean_inc_ref(v_buckets_1730_);
lean_inc(v_size_1729_);
v_isSharedCheck_1770_ = !lean_is_exclusive(v_m_1726_);
if (v_isSharedCheck_1770_ == 0)
{
lean_object* v_unused_1771_; lean_object* v_unused_1772_; 
v_unused_1771_ = lean_ctor_get(v_m_1726_, 1);
lean_dec(v_unused_1771_);
v_unused_1772_ = lean_ctor_get(v_m_1726_, 0);
lean_dec(v_unused_1772_);
v___x_1751_ = v_m_1726_;
v_isShared_1752_ = v_isSharedCheck_1770_;
goto v_resetjp_1750_;
}
else
{
lean_dec(v_m_1726_);
v___x_1751_ = lean_box(0);
v_isShared_1752_ = v_isSharedCheck_1770_;
goto v_resetjp_1750_;
}
v_resetjp_1750_:
{
lean_object* v___x_1753_; lean_object* v_size_x27_1754_; lean_object* v___x_1755_; lean_object* v_buckets_x27_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; uint8_t v___x_1762_; 
v___x_1753_ = lean_unsigned_to_nat(1u);
v_size_x27_1754_ = lean_nat_add(v_size_1729_, v___x_1753_);
lean_dec(v_size_1729_);
lean_inc(v_bkt_1748_);
v___x_1755_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1755_, 0, v_a_1727_);
lean_ctor_set(v___x_1755_, 1, v_b_1728_);
lean_ctor_set(v___x_1755_, 2, v_bkt_1748_);
v_buckets_x27_1756_ = lean_array_uset(v_buckets_1730_, v___x_1747_, v___x_1755_);
v___x_1757_ = lean_unsigned_to_nat(4u);
v___x_1758_ = lean_nat_mul(v_size_x27_1754_, v___x_1757_);
v___x_1759_ = lean_unsigned_to_nat(3u);
v___x_1760_ = lean_nat_div(v___x_1758_, v___x_1759_);
lean_dec(v___x_1758_);
v___x_1761_ = lean_array_get_size(v_buckets_x27_1756_);
v___x_1762_ = lean_nat_dec_le(v___x_1760_, v___x_1761_);
lean_dec(v___x_1760_);
if (v___x_1762_ == 0)
{
lean_object* v_val_1763_; lean_object* v___x_1765_; 
v_val_1763_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1___redArg(v_buckets_x27_1756_);
if (v_isShared_1752_ == 0)
{
lean_ctor_set(v___x_1751_, 1, v_val_1763_);
lean_ctor_set(v___x_1751_, 0, v_size_x27_1754_);
v___x_1765_ = v___x_1751_;
goto v_reusejp_1764_;
}
else
{
lean_object* v_reuseFailAlloc_1766_; 
v_reuseFailAlloc_1766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1766_, 0, v_size_x27_1754_);
lean_ctor_set(v_reuseFailAlloc_1766_, 1, v_val_1763_);
v___x_1765_ = v_reuseFailAlloc_1766_;
goto v_reusejp_1764_;
}
v_reusejp_1764_:
{
return v___x_1765_;
}
}
else
{
lean_object* v___x_1768_; 
if (v_isShared_1752_ == 0)
{
lean_ctor_set(v___x_1751_, 1, v_buckets_x27_1756_);
lean_ctor_set(v___x_1751_, 0, v_size_x27_1754_);
v___x_1768_ = v___x_1751_;
goto v_reusejp_1767_;
}
else
{
lean_object* v_reuseFailAlloc_1769_; 
v_reuseFailAlloc_1769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1769_, 0, v_size_x27_1754_);
lean_ctor_set(v_reuseFailAlloc_1769_, 1, v_buckets_x27_1756_);
v___x_1768_ = v_reuseFailAlloc_1769_;
goto v_reusejp_1767_;
}
v_reusejp_1767_:
{
return v___x_1768_;
}
}
}
}
else
{
lean_dec(v_b_1728_);
lean_dec_ref(v_a_1727_);
return v_m_1726_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(lean_object* v_src_1776_, lean_object* v_tgt_1777_, lean_object* v_n_1778_, lean_object* v_map_1779_, lean_object* v_a_1780_, lean_object* v_a_1781_){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v_map_1788_; lean_object* v___y_1789_; lean_object* v___y_1796_; lean_object* v___y_1799_; lean_object* v___y_1800_; lean_object* v_d_u2081_1803_; lean_object* v_b_u2081_1804_; lean_object* v_d_u2082_1805_; lean_object* v_b_u2082_1806_; lean_object* v___y_1807_; lean_object* v___y_1808_; uint8_t v___x_1812_; 
v___x_1785_ = lean_st_ref_get(v_a_1781_);
lean_inc_ref(v_tgt_1777_);
lean_inc_ref(v_src_1776_);
v___x_1786_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1786_, 0, v_src_1776_);
lean_ctor_set(v___x_1786_, 1, v_tgt_1777_);
v___x_1812_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg(v___x_1785_, v___x_1786_);
lean_dec(v___x_1785_);
if (v___x_1812_ == 0)
{
switch(lean_obj_tag(v_src_1776_))
{
case 7:
{
if (lean_obj_tag(v_tgt_1777_) == 7)
{
lean_object* v_binderType_1813_; lean_object* v_body_1814_; lean_object* v_binderType_1815_; lean_object* v_body_1816_; 
v_binderType_1813_ = lean_ctor_get(v_src_1776_, 1);
lean_inc_ref(v_binderType_1813_);
v_body_1814_ = lean_ctor_get(v_src_1776_, 2);
lean_inc_ref(v_body_1814_);
lean_dec_ref_known(v_src_1776_, 3);
v_binderType_1815_ = lean_ctor_get(v_tgt_1777_, 1);
lean_inc_ref(v_binderType_1815_);
v_body_1816_ = lean_ctor_get(v_tgt_1777_, 2);
lean_inc_ref(v_body_1816_);
lean_dec_ref_known(v_tgt_1777_, 3);
v_d_u2081_1803_ = v_binderType_1813_;
v_b_u2081_1804_ = v_body_1814_;
v_d_u2082_1805_ = v_binderType_1815_;
v_b_u2082_1806_ = v_body_1816_;
v___y_1807_ = v_a_1780_;
v___y_1808_ = v_a_1781_;
goto v___jp_1802_;
}
else
{
lean_dec_ref_known(v_src_1776_, 3);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 6:
{
if (lean_obj_tag(v_tgt_1777_) == 6)
{
lean_object* v_binderType_1817_; lean_object* v_body_1818_; lean_object* v_binderType_1819_; lean_object* v_body_1820_; 
v_binderType_1817_ = lean_ctor_get(v_src_1776_, 1);
lean_inc_ref(v_binderType_1817_);
v_body_1818_ = lean_ctor_get(v_src_1776_, 2);
lean_inc_ref(v_body_1818_);
lean_dec_ref_known(v_src_1776_, 3);
v_binderType_1819_ = lean_ctor_get(v_tgt_1777_, 1);
lean_inc_ref(v_binderType_1819_);
v_body_1820_ = lean_ctor_get(v_tgt_1777_, 2);
lean_inc_ref(v_body_1820_);
lean_dec_ref_known(v_tgt_1777_, 3);
v_d_u2081_1803_ = v_binderType_1817_;
v_b_u2081_1804_ = v_body_1818_;
v_d_u2082_1805_ = v_binderType_1819_;
v_b_u2082_1806_ = v_body_1820_;
v___y_1807_ = v_a_1780_;
v___y_1808_ = v_a_1781_;
goto v___jp_1802_;
}
else
{
lean_dec_ref_known(v_src_1776_, 3);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 10:
{
if (lean_obj_tag(v_tgt_1777_) == 10)
{
lean_object* v_expr_1821_; lean_object* v_expr_1822_; lean_object* v___x_1823_; 
v_expr_1821_ = lean_ctor_get(v_src_1776_, 1);
lean_inc_ref(v_expr_1821_);
lean_dec_ref_known(v_src_1776_, 2);
v_expr_1822_ = lean_ctor_get(v_tgt_1777_, 1);
lean_inc_ref(v_expr_1822_);
lean_dec_ref_known(v_tgt_1777_, 2);
v___x_1823_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_expr_1821_, v_expr_1822_, v_n_1778_, v_map_1779_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1823_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___x_1823_;
}
else
{
lean_object* v_val_1824_; 
v_val_1824_ = lean_ctor_get(v___x_1823_, 0);
lean_inc(v_val_1824_);
lean_dec_ref_known(v___x_1823_, 1);
v_map_1788_ = v_val_1824_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
lean_dec_ref_known(v_src_1776_, 2);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 8:
{
if (lean_obj_tag(v_tgt_1777_) == 8)
{
lean_object* v_type_1825_; lean_object* v_value_1826_; lean_object* v_body_1827_; lean_object* v_type_1828_; lean_object* v_value_1829_; lean_object* v_body_1830_; lean_object* v___y_1832_; lean_object* v___x_1836_; 
v_type_1825_ = lean_ctor_get(v_src_1776_, 1);
lean_inc_ref(v_type_1825_);
v_value_1826_ = lean_ctor_get(v_src_1776_, 2);
lean_inc_ref(v_value_1826_);
v_body_1827_ = lean_ctor_get(v_src_1776_, 3);
lean_inc_ref(v_body_1827_);
lean_dec_ref_known(v_src_1776_, 4);
v_type_1828_ = lean_ctor_get(v_tgt_1777_, 1);
lean_inc_ref(v_type_1828_);
v_value_1829_ = lean_ctor_get(v_tgt_1777_, 2);
lean_inc_ref(v_value_1829_);
v_body_1830_ = lean_ctor_get(v_tgt_1777_, 3);
lean_inc_ref(v_body_1830_);
lean_dec_ref_known(v_tgt_1777_, 4);
lean_inc(v_n_1778_);
v___x_1836_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_type_1825_, v_type_1828_, v_n_1778_, v_map_1779_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1836_) == 0)
{
lean_dec_ref(v_value_1829_);
lean_dec_ref(v_value_1826_);
v___y_1832_ = v___x_1836_;
goto v___jp_1831_;
}
else
{
lean_object* v_val_1837_; lean_object* v___x_1838_; 
v_val_1837_ = lean_ctor_get(v___x_1836_, 0);
lean_inc(v_val_1837_);
lean_dec_ref_known(v___x_1836_, 1);
lean_inc(v_n_1778_);
v___x_1838_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_value_1826_, v_value_1829_, v_n_1778_, v_val_1837_, v_a_1780_, v_a_1781_);
v___y_1832_ = v___x_1838_;
goto v___jp_1831_;
}
v___jp_1831_:
{
if (lean_obj_tag(v___y_1832_) == 0)
{
lean_dec_ref(v_body_1830_);
lean_dec_ref(v_body_1827_);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec(v_n_1778_);
return v___y_1832_;
}
else
{
lean_object* v_val_1833_; lean_object* v___x_1834_; 
v_val_1833_ = lean_ctor_get(v___y_1832_, 0);
lean_inc(v_val_1833_);
lean_dec_ref_known(v___y_1832_, 1);
v___x_1834_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_body_1827_, v_body_1830_, v_n_1778_, v_val_1833_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1834_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___x_1834_;
}
else
{
lean_object* v_val_1835_; 
v_val_1835_ = lean_ctor_get(v___x_1834_, 0);
lean_inc(v_val_1835_);
lean_dec_ref_known(v___x_1834_, 1);
v_map_1788_ = v_val_1835_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
}
}
else
{
lean_dec_ref_known(v_src_1776_, 4);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 5:
{
if (lean_obj_tag(v_tgt_1777_) == 5)
{
lean_object* v_fn_1839_; lean_object* v_arg_1840_; lean_object* v_fn_1841_; lean_object* v_arg_1842_; lean_object* v___x_1843_; 
v_fn_1839_ = lean_ctor_get(v_src_1776_, 0);
lean_inc_ref(v_fn_1839_);
v_arg_1840_ = lean_ctor_get(v_src_1776_, 1);
lean_inc_ref(v_arg_1840_);
lean_dec_ref_known(v_src_1776_, 2);
v_fn_1841_ = lean_ctor_get(v_tgt_1777_, 0);
lean_inc_ref(v_fn_1841_);
v_arg_1842_ = lean_ctor_get(v_tgt_1777_, 1);
lean_inc_ref(v_arg_1842_);
lean_dec_ref_known(v_tgt_1777_, 2);
lean_inc(v_n_1778_);
v___x_1843_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_fn_1839_, v_fn_1841_, v_n_1778_, v_map_1779_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1843_) == 0)
{
lean_dec_ref(v_arg_1842_);
lean_dec_ref(v_arg_1840_);
lean_dec(v_n_1778_);
v___y_1796_ = v___x_1843_;
goto v___jp_1795_;
}
else
{
lean_object* v_val_1844_; lean_object* v___x_1845_; 
v_val_1844_ = lean_ctor_get(v___x_1843_, 0);
lean_inc(v_val_1844_);
lean_dec_ref_known(v___x_1843_, 1);
v___x_1845_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_arg_1840_, v_arg_1842_, v_n_1778_, v_val_1844_, v_a_1780_, v_a_1781_);
v___y_1796_ = v___x_1845_;
goto v___jp_1795_;
}
}
else
{
lean_dec_ref_known(v_src_1776_, 2);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 11:
{
if (lean_obj_tag(v_tgt_1777_) == 11)
{
lean_object* v_struct_1846_; lean_object* v_struct_1847_; lean_object* v___x_1848_; 
v_struct_1846_ = lean_ctor_get(v_src_1776_, 2);
lean_inc_ref(v_struct_1846_);
lean_dec_ref_known(v_src_1776_, 3);
v_struct_1847_ = lean_ctor_get(v_tgt_1777_, 2);
lean_inc_ref(v_struct_1847_);
lean_dec_ref_known(v_tgt_1777_, 3);
v___x_1848_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_struct_1846_, v_struct_1847_, v_n_1778_, v_map_1779_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1848_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___x_1848_;
}
else
{
lean_object* v_val_1849_; 
v_val_1849_ = lean_ctor_get(v___x_1848_, 0);
lean_inc(v_val_1849_);
lean_dec_ref_known(v___x_1848_, 1);
v_map_1788_ = v_val_1849_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
lean_dec_ref_known(v_src_1776_, 3);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 1:
{
if (lean_obj_tag(v_tgt_1777_) == 1)
{
lean_object* v_fvarId_1850_; lean_object* v_fvarId_1851_; lean_object* v_fst_1852_; lean_object* v_snd_1853_; lean_object* v___x_1854_; 
v_fvarId_1850_ = lean_ctor_get(v_src_1776_, 0);
lean_inc(v_fvarId_1850_);
lean_dec_ref_known(v_src_1776_, 1);
v_fvarId_1851_ = lean_ctor_get(v_tgt_1777_, 0);
lean_inc(v_fvarId_1851_);
lean_dec_ref_known(v_tgt_1777_, 1);
v_fst_1852_ = lean_ctor_get(v_a_1780_, 0);
v_snd_1853_ = lean_ctor_get(v_a_1780_, 1);
v___x_1854_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(v_fst_1852_, v_fvarId_1850_);
lean_dec(v_fvarId_1850_);
if (lean_obj_tag(v___x_1854_) == 1)
{
lean_object* v_val_1855_; lean_object* v___x_1856_; 
v_val_1855_ = lean_ctor_get(v___x_1854_, 0);
lean_inc(v_val_1855_);
lean_dec_ref_known(v___x_1854_, 1);
v___x_1856_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(v_snd_1853_, v_fvarId_1851_);
lean_dec(v_fvarId_1851_);
if (lean_obj_tag(v___x_1856_) == 1)
{
lean_object* v_val_1857_; lean_object* v___x_1859_; uint8_t v_isShared_1860_; uint8_t v_isSharedCheck_1891_; 
v_val_1857_ = lean_ctor_get(v___x_1856_, 0);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1856_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1859_ = v___x_1856_;
v_isShared_1860_ = v_isSharedCheck_1891_;
goto v_resetjp_1858_;
}
else
{
lean_inc(v_val_1857_);
lean_dec(v___x_1856_);
v___x_1859_ = lean_box(0);
v_isShared_1860_ = v_isSharedCheck_1891_;
goto v_resetjp_1858_;
}
v_resetjp_1858_:
{
lean_object* v___y_1862_; uint8_t v___x_1870_; 
v___x_1870_ = lean_nat_dec_lt(v_val_1857_, v_n_1778_);
if (v___x_1870_ == 0)
{
lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; 
lean_del_object(v___x_1859_);
lean_dec(v_val_1855_);
lean_dec_ref(v_map_1779_);
v___x_1871_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__0));
v___x_1872_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__1));
v___x_1873_ = lean_unsigned_to_nat(319u);
v___x_1874_ = lean_unsigned_to_nat(10u);
v___x_1875_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_Permutation_permute_x21_cyclicPermuteAux___redArg___closed__2));
v___x_1876_ = l_Nat_reprFast(v_val_1857_);
v___x_1877_ = lean_string_append(v___x_1875_, v___x_1876_);
lean_dec_ref(v___x_1876_);
v___x_1878_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___closed__2));
v___x_1879_ = lean_string_append(v___x_1877_, v___x_1878_);
lean_inc(v_n_1778_);
v___x_1880_ = l_Nat_reprFast(v_n_1778_);
v___x_1881_ = lean_string_append(v___x_1879_, v___x_1880_);
lean_dec_ref(v___x_1880_);
v___x_1882_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_ArgReorder_toString_spec__2___closed__1));
v___x_1883_ = lean_string_append(v___x_1881_, v___x_1882_);
v___x_1884_ = l_mkPanicMessageWithDecl(v___x_1871_, v___x_1872_, v___x_1873_, v___x_1874_, v___x_1883_);
lean_dec_ref(v___x_1883_);
v___x_1885_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__3(v_n_1778_, v___x_1884_, v_a_1780_, v_a_1781_);
if (lean_obj_tag(v___x_1885_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___x_1885_;
}
else
{
lean_object* v_val_1886_; 
v_val_1886_ = lean_ctor_get(v___x_1885_, 0);
lean_inc(v_val_1886_);
lean_dec_ref_known(v___x_1885_, 1);
v_map_1788_ = v_val_1886_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
uint8_t v___x_1887_; 
v___x_1887_ = lean_nat_dec_lt(v_val_1855_, v_n_1778_);
lean_dec(v_n_1778_);
if (v___x_1887_ == 0)
{
lean_object* v___x_1888_; lean_object* v___x_1889_; 
v___x_1888_ = lean_box(0);
v___x_1889_ = l_outOfBounds___redArg(v___x_1888_);
v___y_1862_ = v___x_1889_;
goto v___jp_1861_;
}
else
{
lean_object* v___x_1890_; 
v___x_1890_ = lean_array_fget_borrowed(v_map_1779_, v_val_1855_);
lean_inc(v___x_1890_);
v___y_1862_ = v___x_1890_;
goto v___jp_1861_;
}
}
v___jp_1861_:
{
if (lean_obj_tag(v___y_1862_) == 1)
{
lean_object* v_val_1863_; uint8_t v___x_1864_; 
lean_del_object(v___x_1859_);
lean_dec(v_val_1855_);
v_val_1863_ = lean_ctor_get(v___y_1862_, 0);
lean_inc(v_val_1863_);
lean_dec_ref_known(v___y_1862_, 1);
v___x_1864_ = lean_nat_dec_eq(v_val_1857_, v_val_1863_);
lean_dec(v_val_1863_);
lean_dec(v_val_1857_);
if (v___x_1864_ == 0)
{
lean_object* v___x_1865_; 
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
v___x_1865_ = lean_box(0);
return v___x_1865_;
}
else
{
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
lean_object* v___x_1867_; 
lean_dec(v___y_1862_);
if (v_isShared_1860_ == 0)
{
v___x_1867_ = v___x_1859_;
goto v_reusejp_1866_;
}
else
{
lean_object* v_reuseFailAlloc_1869_; 
v_reuseFailAlloc_1869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1869_, 0, v_val_1857_);
v___x_1867_ = v_reuseFailAlloc_1869_;
goto v_reusejp_1866_;
}
v_reusejp_1866_:
{
lean_object* v___x_1868_; 
v___x_1868_ = lean_array_set(v_map_1779_, v_val_1855_, v___x_1867_);
lean_dec(v_val_1855_);
v_map_1788_ = v___x_1868_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
}
}
}
else
{
lean_dec(v___x_1856_);
lean_dec(v_val_1855_);
lean_dec(v_n_1778_);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
lean_dec(v___x_1854_);
lean_dec(v_fvarId_1851_);
lean_dec(v_n_1778_);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
else
{
lean_dec_ref_known(v_src_1776_, 1);
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 9:
{
lean_dec_ref_known(v_src_1776_, 1);
lean_dec(v_n_1778_);
if (lean_obj_tag(v_tgt_1777_) == 9)
{
lean_dec_ref_known(v_tgt_1777_, 1);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
else
{
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 0:
{
lean_dec_ref_known(v_src_1776_, 1);
lean_dec(v_n_1778_);
if (lean_obj_tag(v_tgt_1777_) == 0)
{
lean_dec_ref_known(v_tgt_1777_, 1);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
else
{
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 3:
{
lean_dec_ref_known(v_src_1776_, 1);
lean_dec(v_n_1778_);
if (lean_obj_tag(v_tgt_1777_) == 3)
{
lean_dec_ref_known(v_tgt_1777_, 1);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
else
{
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
case 4:
{
lean_dec_ref_known(v_src_1776_, 2);
lean_dec(v_n_1778_);
if (lean_obj_tag(v_tgt_1777_) == 4)
{
lean_dec_ref_known(v_tgt_1777_, 2);
v_map_1788_ = v_map_1779_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
else
{
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec_ref(v_tgt_1777_);
goto v___jp_1783_;
}
}
default: 
{
lean_dec_ref_known(v___x_1786_, 2);
lean_dec_ref(v_map_1779_);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
lean_dec_ref(v_src_1776_);
goto v___jp_1783_;
}
}
}
else
{
lean_object* v___x_1892_; 
lean_dec_ref_known(v___x_1786_, 2);
lean_dec(v_n_1778_);
lean_dec_ref(v_tgt_1777_);
lean_dec_ref(v_src_1776_);
v___x_1892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1892_, 0, v_map_1779_);
return v___x_1892_;
}
v___jp_1783_:
{
lean_object* v___x_1784_; 
v___x_1784_ = lean_box(0);
return v___x_1784_;
}
v___jp_1787_:
{
lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; 
v___x_1790_ = lean_st_ref_take(v___y_1789_);
v___x_1791_ = lean_box(0);
v___x_1792_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0___redArg(v___x_1790_, v___x_1786_, v___x_1791_);
v___x_1793_ = lean_st_ref_set(v___y_1789_, v___x_1792_);
v___x_1794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1794_, 0, v_map_1788_);
return v___x_1794_;
}
v___jp_1795_:
{
if (lean_obj_tag(v___y_1796_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___y_1796_;
}
else
{
lean_object* v_val_1797_; 
v_val_1797_ = lean_ctor_get(v___y_1796_, 0);
lean_inc(v_val_1797_);
lean_dec_ref_known(v___y_1796_, 1);
v_map_1788_ = v_val_1797_;
v___y_1789_ = v_a_1781_;
goto v___jp_1787_;
}
}
v___jp_1798_:
{
if (lean_obj_tag(v___y_1800_) == 0)
{
lean_dec_ref_known(v___x_1786_, 2);
return v___y_1800_;
}
else
{
lean_object* v_val_1801_; 
v_val_1801_ = lean_ctor_get(v___y_1800_, 0);
lean_inc(v_val_1801_);
lean_dec_ref_known(v___y_1800_, 1);
v_map_1788_ = v_val_1801_;
v___y_1789_ = v___y_1799_;
goto v___jp_1787_;
}
}
v___jp_1802_:
{
lean_object* v___x_1809_; 
lean_inc(v_n_1778_);
v___x_1809_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_d_u2081_1803_, v_d_u2082_1805_, v_n_1778_, v_map_1779_, v___y_1807_, v___y_1808_);
if (lean_obj_tag(v___x_1809_) == 0)
{
lean_dec_ref(v_b_u2082_1806_);
lean_dec_ref(v_b_u2081_1804_);
lean_dec(v_n_1778_);
v___y_1799_ = v___y_1808_;
v___y_1800_ = v___x_1809_;
goto v___jp_1798_;
}
else
{
lean_object* v_val_1810_; lean_object* v___x_1811_; 
v_val_1810_ = lean_ctor_get(v___x_1809_, 0);
lean_inc(v_val_1810_);
lean_dec_ref_known(v___x_1809_, 1);
v___x_1811_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_b_u2081_1804_, v_b_u2082_1806_, v_n_1778_, v_val_1810_, v___y_1807_, v___y_1808_);
v___y_1799_ = v___y_1808_;
v___y_1800_ = v___x_1811_;
goto v___jp_1798_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit___boxed(lean_object* v_src_1893_, lean_object* v_tgt_1894_, lean_object* v_n_1895_, lean_object* v_map_1896_, lean_object* v_a_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_){
_start:
{
lean_object* v_res_1900_; 
v_res_1900_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_src_1893_, v_tgt_1894_, v_n_1895_, v_map_1896_, v_a_1897_, v_a_1898_);
lean_dec(v_a_1898_);
lean_dec_ref(v_a_1897_);
return v_res_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0(lean_object* v_00_u03b2_1901_, lean_object* v_m_1902_, lean_object* v_a_1903_, lean_object* v_b_1904_){
_start:
{
lean_object* v___x_1905_; 
v___x_1905_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0___redArg(v_m_1902_, v_a_1903_, v_b_1904_);
return v___x_1905_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1(lean_object* v_00_u03b2_1906_, lean_object* v_m_1907_, lean_object* v_a_1908_){
_start:
{
uint8_t v___x_1909_; 
v___x_1909_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___redArg(v_m_1907_, v_a_1908_);
return v___x_1909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1___boxed(lean_object* v_00_u03b2_1910_, lean_object* v_m_1911_, lean_object* v_a_1912_){
_start:
{
uint8_t v_res_1913_; lean_object* v_r_1914_; 
v_res_1913_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__1(v_00_u03b2_1910_, v_m_1911_, v_a_1912_);
lean_dec_ref(v_a_1912_);
lean_dec_ref(v_m_1911_);
v_r_1914_ = lean_box(v_res_1913_);
return v_r_1914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2(lean_object* v_00_u03b2_1915_, lean_object* v_m_1916_, lean_object* v_a_1917_){
_start:
{
lean_object* v___x_1918_; 
v___x_1918_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___redArg(v_m_1916_, v_a_1917_);
return v___x_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2___boxed(lean_object* v_00_u03b2_1919_, lean_object* v_m_1920_, lean_object* v_a_1921_){
_start:
{
lean_object* v_res_1922_; 
v_res_1922_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2(v_00_u03b2_1919_, v_m_1920_, v_a_1921_);
lean_dec(v_a_1921_);
lean_dec_ref(v_m_1920_);
return v_res_1922_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0(lean_object* v_00_u03b2_1923_, lean_object* v_a_1924_, lean_object* v_x_1925_){
_start:
{
uint8_t v___x_1926_; 
v___x_1926_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___redArg(v_a_1924_, v_x_1925_);
return v___x_1926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1927_, lean_object* v_a_1928_, lean_object* v_x_1929_){
_start:
{
uint8_t v_res_1930_; lean_object* v_r_1931_; 
v_res_1930_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__0(v_00_u03b2_1927_, v_a_1928_, v_x_1929_);
lean_dec(v_x_1929_);
lean_dec_ref(v_a_1928_);
v_r_1931_ = lean_box(v_res_1930_);
return v_r_1931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1(lean_object* v_00_u03b2_1932_, lean_object* v_data_1933_){
_start:
{
lean_object* v___x_1934_; 
v___x_1934_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1___redArg(v_data_1933_);
return v___x_1934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4(lean_object* v_00_u03b2_1935_, lean_object* v_a_1936_, lean_object* v_x_1937_){
_start:
{
lean_object* v___x_1938_; 
v___x_1938_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___redArg(v_a_1936_, v_x_1937_);
return v___x_1938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1939_, lean_object* v_a_1940_, lean_object* v_x_1941_){
_start:
{
lean_object* v_res_1942_; 
v_res_1942_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__2_spec__4(v_00_u03b2_1939_, v_a_1940_, v_x_1941_);
lean_dec(v_x_1941_);
lean_dec(v_a_1940_);
return v_res_1942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_1943_, lean_object* v_i_1944_, lean_object* v_source_1945_, lean_object* v_target_1946_){
_start:
{
lean_object* v___x_1947_; 
v___x_1947_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3___redArg(v_i_1944_, v_source_1945_, v_target_1946_);
return v___x_1947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6(lean_object* v_00_u03b2_1948_, lean_object* v_x_1949_, lean_object* v_x_1950_){
_start:
{
lean_object* v___x_1951_; 
v___x_1951_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit_spec__0_spec__1_spec__3_spec__6___redArg(v_x_1949_, v_x_1950_);
return v___x_1951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0(lean_object* v_k_1952_, lean_object* v_b_1953_, lean_object* v_c_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_){
_start:
{
lean_object* v___x_1960_; 
lean_inc(v___y_1958_);
lean_inc_ref(v___y_1957_);
lean_inc(v___y_1956_);
lean_inc_ref(v___y_1955_);
v___x_1960_ = lean_apply_7(v_k_1952_, v_b_1953_, v_c_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, lean_box(0));
return v___x_1960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0___boxed(lean_object* v_k_1961_, lean_object* v_b_1962_, lean_object* v_c_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_){
_start:
{
lean_object* v_res_1969_; 
v_res_1969_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0(v_k_1961_, v_b_1962_, v_c_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_);
lean_dec(v___y_1967_);
lean_dec_ref(v___y_1966_);
lean_dec(v___y_1965_);
lean_dec_ref(v___y_1964_);
return v_res_1969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(lean_object* v_type_1970_, lean_object* v_maxFVars_x3f_1971_, lean_object* v_k_1972_, uint8_t v_cleanupAnnotations_1973_, uint8_t v_whnfType_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v___f_1980_; lean_object* v___x_1981_; 
v___f_1980_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1980_, 0, v_k_1972_);
v___x_1981_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_1970_, v_maxFVars_x3f_1971_, v___f_1980_, v_cleanupAnnotations_1973_, v_whnfType_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_);
if (lean_obj_tag(v___x_1981_) == 0)
{
lean_object* v_a_1982_; lean_object* v___x_1984_; uint8_t v_isShared_1985_; uint8_t v_isSharedCheck_1989_; 
v_a_1982_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_1989_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_1989_ == 0)
{
v___x_1984_ = v___x_1981_;
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
else
{
lean_inc(v_a_1982_);
lean_dec(v___x_1981_);
v___x_1984_ = lean_box(0);
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
v_resetjp_1983_:
{
lean_object* v___x_1987_; 
if (v_isShared_1985_ == 0)
{
v___x_1987_ = v___x_1984_;
goto v_reusejp_1986_;
}
else
{
lean_object* v_reuseFailAlloc_1988_; 
v_reuseFailAlloc_1988_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1988_, 0, v_a_1982_);
v___x_1987_ = v_reuseFailAlloc_1988_;
goto v_reusejp_1986_;
}
v_reusejp_1986_:
{
return v___x_1987_;
}
}
}
else
{
lean_object* v_a_1990_; lean_object* v___x_1992_; uint8_t v_isShared_1993_; uint8_t v_isSharedCheck_1997_; 
v_a_1990_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_1997_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_1997_ == 0)
{
v___x_1992_ = v___x_1981_;
v_isShared_1993_ = v_isSharedCheck_1997_;
goto v_resetjp_1991_;
}
else
{
lean_inc(v_a_1990_);
lean_dec(v___x_1981_);
v___x_1992_ = lean_box(0);
v_isShared_1993_ = v_isSharedCheck_1997_;
goto v_resetjp_1991_;
}
v_resetjp_1991_:
{
lean_object* v___x_1995_; 
if (v_isShared_1993_ == 0)
{
v___x_1995_ = v___x_1992_;
goto v_reusejp_1994_;
}
else
{
lean_object* v_reuseFailAlloc_1996_; 
v_reuseFailAlloc_1996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1996_, 0, v_a_1990_);
v___x_1995_ = v_reuseFailAlloc_1996_;
goto v_reusejp_1994_;
}
v_reusejp_1994_:
{
return v___x_1995_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___boxed(lean_object* v_type_1998_, lean_object* v_maxFVars_x3f_1999_, lean_object* v_k_2000_, lean_object* v_cleanupAnnotations_2001_, lean_object* v_whnfType_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2008_; uint8_t v_whnfType_boxed_2009_; lean_object* v_res_2010_; 
v_cleanupAnnotations_boxed_2008_ = lean_unbox(v_cleanupAnnotations_2001_);
v_whnfType_boxed_2009_ = lean_unbox(v_whnfType_2002_);
v_res_2010_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(v_type_1998_, v_maxFVars_x3f_1999_, v_k_2000_, v_cleanupAnnotations_boxed_2008_, v_whnfType_boxed_2009_, v___y_2003_, v___y_2004_, v___y_2005_, v___y_2006_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___y_2003_);
return v_res_2010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4(lean_object* v_00_u03b1_2011_, lean_object* v_type_2012_, lean_object* v_maxFVars_x3f_2013_, lean_object* v_k_2014_, uint8_t v_cleanupAnnotations_2015_, uint8_t v_whnfType_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_){
_start:
{
lean_object* v___x_2022_; 
v___x_2022_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(v_type_2012_, v_maxFVars_x3f_2013_, v_k_2014_, v_cleanupAnnotations_2015_, v_whnfType_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_);
return v___x_2022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___boxed(lean_object* v_00_u03b1_2023_, lean_object* v_type_2024_, lean_object* v_maxFVars_x3f_2025_, lean_object* v_k_2026_, lean_object* v_cleanupAnnotations_2027_, lean_object* v_whnfType_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2034_; uint8_t v_whnfType_boxed_2035_; lean_object* v_res_2036_; 
v_cleanupAnnotations_boxed_2034_ = lean_unbox(v_cleanupAnnotations_2027_);
v_whnfType_boxed_2035_ = lean_unbox(v_whnfType_2028_);
v_res_2036_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4(v_00_u03b1_2023_, v_type_2024_, v_maxFVars_x3f_2025_, v_k_2026_, v_cleanupAnnotations_boxed_2034_, v_whnfType_boxed_2035_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_);
lean_dec(v___y_2032_);
lean_dec_ref(v___y_2031_);
lean_dec(v___y_2030_);
lean_dec_ref(v___y_2029_);
return v_res_2036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg(size_t v_sz_2037_, size_t v_i_2038_, lean_object* v_bs_2039_){
_start:
{
uint8_t v___x_2040_; 
v___x_2040_ = lean_usize_dec_lt(v_i_2038_, v_sz_2037_);
if (v___x_2040_ == 0)
{
return v_bs_2039_;
}
else
{
lean_object* v_v_2041_; lean_object* v___x_2042_; lean_object* v_bs_x27_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; size_t v___x_2047_; size_t v___x_2048_; lean_object* v___x_2049_; 
v_v_2041_ = lean_array_uget(v_bs_2039_, v_i_2038_);
v___x_2042_ = lean_unsigned_to_nat(0u);
v_bs_x27_2043_ = lean_array_uset(v_bs_2039_, v_i_2038_, v___x_2042_);
v___x_2044_ = lean_usize_to_nat(v_i_2038_);
v___x_2045_ = l_Lean_Expr_fvarId_x21(v_v_2041_);
lean_dec(v_v_2041_);
v___x_2046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2046_, 0, v___x_2045_);
lean_ctor_set(v___x_2046_, 1, v___x_2044_);
v___x_2047_ = ((size_t)1ULL);
v___x_2048_ = lean_usize_add(v_i_2038_, v___x_2047_);
v___x_2049_ = lean_array_uset(v_bs_x27_2043_, v_i_2038_, v___x_2046_);
v_i_2038_ = v___x_2048_;
v_bs_2039_ = v___x_2049_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg___boxed(lean_object* v_sz_2051_, lean_object* v_i_2052_, lean_object* v_bs_2053_){
_start:
{
size_t v_sz_boxed_2054_; size_t v_i_boxed_2055_; lean_object* v_res_2056_; 
v_sz_boxed_2054_ = lean_unbox_usize(v_sz_2051_);
lean_dec(v_sz_2051_);
v_i_boxed_2055_ = lean_unbox_usize(v_i_2052_);
lean_dec(v_i_2052_);
v_res_2056_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg(v_sz_boxed_2054_, v_i_boxed_2055_, v_bs_2053_);
return v_res_2056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2(lean_object* v_as_2057_, size_t v_sz_2058_, size_t v_i_2059_, lean_object* v_bs_2060_){
_start:
{
uint8_t v___x_2061_; 
v___x_2061_ = lean_usize_dec_lt(v_i_2059_, v_sz_2058_);
if (v___x_2061_ == 0)
{
return v_bs_2060_;
}
else
{
lean_object* v_v_2062_; lean_object* v___x_2063_; lean_object* v_bs_x27_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; size_t v___x_2068_; size_t v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; 
v_v_2062_ = lean_array_uget(v_bs_2060_, v_i_2059_);
v___x_2063_ = lean_unsigned_to_nat(0u);
v_bs_x27_2064_ = lean_array_uset(v_bs_2060_, v_i_2059_, v___x_2063_);
v___x_2065_ = lean_usize_to_nat(v_i_2059_);
v___x_2066_ = l_Lean_Expr_fvarId_x21(v_v_2062_);
lean_dec(v_v_2062_);
v___x_2067_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2067_, 0, v___x_2066_);
lean_ctor_set(v___x_2067_, 1, v___x_2065_);
v___x_2068_ = ((size_t)1ULL);
v___x_2069_ = lean_usize_add(v_i_2059_, v___x_2068_);
v___x_2070_ = lean_array_uset(v_bs_x27_2064_, v_i_2059_, v___x_2067_);
v___x_2071_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg(v_sz_2058_, v___x_2069_, v___x_2070_);
return v___x_2071_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2___boxed(lean_object* v_as_2072_, lean_object* v_sz_2073_, lean_object* v_i_2074_, lean_object* v_bs_2075_){
_start:
{
size_t v_sz_boxed_2076_; size_t v_i_boxed_2077_; lean_object* v_res_2078_; 
v_sz_boxed_2076_ = lean_unbox_usize(v_sz_2073_);
lean_dec(v_sz_2073_);
v_i_boxed_2077_ = lean_unbox_usize(v_i_2074_);
lean_dec(v_i_2074_);
v_res_2078_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2(v_as_2072_, v_sz_boxed_2076_, v_i_boxed_2077_, v_bs_2075_);
lean_dec_ref(v_as_2072_);
return v_res_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10___redArg(lean_object* v_x_2079_, lean_object* v_x_2080_){
_start:
{
if (lean_obj_tag(v_x_2080_) == 0)
{
return v_x_2079_;
}
else
{
lean_object* v_key_2081_; lean_object* v_value_2082_; lean_object* v_tail_2083_; lean_object* v___x_2085_; uint8_t v_isShared_2086_; uint8_t v_isSharedCheck_2106_; 
v_key_2081_ = lean_ctor_get(v_x_2080_, 0);
v_value_2082_ = lean_ctor_get(v_x_2080_, 1);
v_tail_2083_ = lean_ctor_get(v_x_2080_, 2);
v_isSharedCheck_2106_ = !lean_is_exclusive(v_x_2080_);
if (v_isSharedCheck_2106_ == 0)
{
v___x_2085_ = v_x_2080_;
v_isShared_2086_ = v_isSharedCheck_2106_;
goto v_resetjp_2084_;
}
else
{
lean_inc(v_tail_2083_);
lean_inc(v_value_2082_);
lean_inc(v_key_2081_);
lean_dec(v_x_2080_);
v___x_2085_ = lean_box(0);
v_isShared_2086_ = v_isSharedCheck_2106_;
goto v_resetjp_2084_;
}
v_resetjp_2084_:
{
lean_object* v___x_2087_; uint64_t v___x_2088_; uint64_t v___x_2089_; uint64_t v___x_2090_; uint64_t v_fold_2091_; uint64_t v___x_2092_; uint64_t v___x_2093_; uint64_t v___x_2094_; size_t v___x_2095_; size_t v___x_2096_; size_t v___x_2097_; size_t v___x_2098_; size_t v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2102_; 
v___x_2087_ = lean_array_get_size(v_x_2079_);
v___x_2088_ = l_Lean_instHashableFVarId_hash(v_key_2081_);
v___x_2089_ = 32ULL;
v___x_2090_ = lean_uint64_shift_right(v___x_2088_, v___x_2089_);
v_fold_2091_ = lean_uint64_xor(v___x_2088_, v___x_2090_);
v___x_2092_ = 16ULL;
v___x_2093_ = lean_uint64_shift_right(v_fold_2091_, v___x_2092_);
v___x_2094_ = lean_uint64_xor(v_fold_2091_, v___x_2093_);
v___x_2095_ = lean_uint64_to_usize(v___x_2094_);
v___x_2096_ = lean_usize_of_nat(v___x_2087_);
v___x_2097_ = ((size_t)1ULL);
v___x_2098_ = lean_usize_sub(v___x_2096_, v___x_2097_);
v___x_2099_ = lean_usize_land(v___x_2095_, v___x_2098_);
v___x_2100_ = lean_array_uget_borrowed(v_x_2079_, v___x_2099_);
lean_inc(v___x_2100_);
if (v_isShared_2086_ == 0)
{
lean_ctor_set(v___x_2085_, 2, v___x_2100_);
v___x_2102_ = v___x_2085_;
goto v_reusejp_2101_;
}
else
{
lean_object* v_reuseFailAlloc_2105_; 
v_reuseFailAlloc_2105_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2105_, 0, v_key_2081_);
lean_ctor_set(v_reuseFailAlloc_2105_, 1, v_value_2082_);
lean_ctor_set(v_reuseFailAlloc_2105_, 2, v___x_2100_);
v___x_2102_ = v_reuseFailAlloc_2105_;
goto v_reusejp_2101_;
}
v_reusejp_2101_:
{
lean_object* v___x_2103_; 
v___x_2103_ = lean_array_uset(v_x_2079_, v___x_2099_, v___x_2102_);
v_x_2079_ = v___x_2103_;
v_x_2080_ = v_tail_2083_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8___redArg(lean_object* v_i_2107_, lean_object* v_source_2108_, lean_object* v_target_2109_){
_start:
{
lean_object* v___x_2110_; uint8_t v___x_2111_; 
v___x_2110_ = lean_array_get_size(v_source_2108_);
v___x_2111_ = lean_nat_dec_lt(v_i_2107_, v___x_2110_);
if (v___x_2111_ == 0)
{
lean_dec_ref(v_source_2108_);
lean_dec(v_i_2107_);
return v_target_2109_;
}
else
{
lean_object* v_es_2112_; lean_object* v___x_2113_; lean_object* v_source_2114_; lean_object* v_target_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; 
v_es_2112_ = lean_array_fget(v_source_2108_, v_i_2107_);
v___x_2113_ = lean_box(0);
v_source_2114_ = lean_array_fset(v_source_2108_, v_i_2107_, v___x_2113_);
v_target_2115_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10___redArg(v_target_2109_, v_es_2112_);
v___x_2116_ = lean_unsigned_to_nat(1u);
v___x_2117_ = lean_nat_add(v_i_2107_, v___x_2116_);
lean_dec(v_i_2107_);
v_i_2107_ = v___x_2117_;
v_source_2108_ = v_source_2114_;
v_target_2109_ = v_target_2115_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7___redArg(lean_object* v_data_2119_){
_start:
{
lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v_nbuckets_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; 
v___x_2120_ = lean_array_get_size(v_data_2119_);
v___x_2121_ = lean_unsigned_to_nat(2u);
v_nbuckets_2122_ = lean_nat_mul(v___x_2120_, v___x_2121_);
v___x_2123_ = lean_unsigned_to_nat(0u);
v___x_2124_ = lean_box(0);
v___x_2125_ = lean_mk_array(v_nbuckets_2122_, v___x_2124_);
v___x_2126_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8___redArg(v___x_2123_, v_data_2119_, v___x_2125_);
return v___x_2126_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg(lean_object* v_a_2127_, lean_object* v_x_2128_){
_start:
{
if (lean_obj_tag(v_x_2128_) == 0)
{
uint8_t v___x_2129_; 
v___x_2129_ = 0;
return v___x_2129_;
}
else
{
lean_object* v_key_2130_; lean_object* v_tail_2131_; uint8_t v___x_2132_; 
v_key_2130_ = lean_ctor_get(v_x_2128_, 0);
v_tail_2131_ = lean_ctor_get(v_x_2128_, 2);
v___x_2132_ = l_Lean_instBEqFVarId_beq(v_key_2130_, v_a_2127_);
if (v___x_2132_ == 0)
{
v_x_2128_ = v_tail_2131_;
goto _start;
}
else
{
return v___x_2132_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg___boxed(lean_object* v_a_2134_, lean_object* v_x_2135_){
_start:
{
uint8_t v_res_2136_; lean_object* v_r_2137_; 
v_res_2136_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg(v_a_2134_, v_x_2135_);
lean_dec(v_x_2135_);
lean_dec(v_a_2134_);
v_r_2137_ = lean_box(v_res_2136_);
return v_r_2137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8___redArg(lean_object* v_a_2138_, lean_object* v_b_2139_, lean_object* v_x_2140_){
_start:
{
if (lean_obj_tag(v_x_2140_) == 0)
{
lean_dec(v_b_2139_);
lean_dec(v_a_2138_);
return v_x_2140_;
}
else
{
lean_object* v_key_2141_; lean_object* v_value_2142_; lean_object* v_tail_2143_; lean_object* v___x_2145_; uint8_t v_isShared_2146_; uint8_t v_isSharedCheck_2155_; 
v_key_2141_ = lean_ctor_get(v_x_2140_, 0);
v_value_2142_ = lean_ctor_get(v_x_2140_, 1);
v_tail_2143_ = lean_ctor_get(v_x_2140_, 2);
v_isSharedCheck_2155_ = !lean_is_exclusive(v_x_2140_);
if (v_isSharedCheck_2155_ == 0)
{
v___x_2145_ = v_x_2140_;
v_isShared_2146_ = v_isSharedCheck_2155_;
goto v_resetjp_2144_;
}
else
{
lean_inc(v_tail_2143_);
lean_inc(v_value_2142_);
lean_inc(v_key_2141_);
lean_dec(v_x_2140_);
v___x_2145_ = lean_box(0);
v_isShared_2146_ = v_isSharedCheck_2155_;
goto v_resetjp_2144_;
}
v_resetjp_2144_:
{
uint8_t v___x_2147_; 
v___x_2147_ = l_Lean_instBEqFVarId_beq(v_key_2141_, v_a_2138_);
if (v___x_2147_ == 0)
{
lean_object* v___x_2148_; lean_object* v___x_2150_; 
v___x_2148_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8___redArg(v_a_2138_, v_b_2139_, v_tail_2143_);
if (v_isShared_2146_ == 0)
{
lean_ctor_set(v___x_2145_, 2, v___x_2148_);
v___x_2150_ = v___x_2145_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2151_; 
v_reuseFailAlloc_2151_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2151_, 0, v_key_2141_);
lean_ctor_set(v_reuseFailAlloc_2151_, 1, v_value_2142_);
lean_ctor_set(v_reuseFailAlloc_2151_, 2, v___x_2148_);
v___x_2150_ = v_reuseFailAlloc_2151_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
return v___x_2150_;
}
}
else
{
lean_object* v___x_2153_; 
lean_dec(v_value_2142_);
lean_dec(v_key_2141_);
if (v_isShared_2146_ == 0)
{
lean_ctor_set(v___x_2145_, 1, v_b_2139_);
lean_ctor_set(v___x_2145_, 0, v_a_2138_);
v___x_2153_ = v___x_2145_;
goto v_reusejp_2152_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v_a_2138_);
lean_ctor_set(v_reuseFailAlloc_2154_, 1, v_b_2139_);
lean_ctor_set(v_reuseFailAlloc_2154_, 2, v_tail_2143_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4___redArg(lean_object* v_m_2156_, lean_object* v_a_2157_, lean_object* v_b_2158_){
_start:
{
lean_object* v_size_2159_; lean_object* v_buckets_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2203_; 
v_size_2159_ = lean_ctor_get(v_m_2156_, 0);
v_buckets_2160_ = lean_ctor_get(v_m_2156_, 1);
v_isSharedCheck_2203_ = !lean_is_exclusive(v_m_2156_);
if (v_isSharedCheck_2203_ == 0)
{
v___x_2162_ = v_m_2156_;
v_isShared_2163_ = v_isSharedCheck_2203_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_buckets_2160_);
lean_inc(v_size_2159_);
lean_dec(v_m_2156_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2203_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
lean_object* v___x_2164_; uint64_t v___x_2165_; uint64_t v___x_2166_; uint64_t v___x_2167_; uint64_t v_fold_2168_; uint64_t v___x_2169_; uint64_t v___x_2170_; uint64_t v___x_2171_; size_t v___x_2172_; size_t v___x_2173_; size_t v___x_2174_; size_t v___x_2175_; size_t v___x_2176_; lean_object* v_bkt_2177_; uint8_t v___x_2178_; 
v___x_2164_ = lean_array_get_size(v_buckets_2160_);
v___x_2165_ = l_Lean_instHashableFVarId_hash(v_a_2157_);
v___x_2166_ = 32ULL;
v___x_2167_ = lean_uint64_shift_right(v___x_2165_, v___x_2166_);
v_fold_2168_ = lean_uint64_xor(v___x_2165_, v___x_2167_);
v___x_2169_ = 16ULL;
v___x_2170_ = lean_uint64_shift_right(v_fold_2168_, v___x_2169_);
v___x_2171_ = lean_uint64_xor(v_fold_2168_, v___x_2170_);
v___x_2172_ = lean_uint64_to_usize(v___x_2171_);
v___x_2173_ = lean_usize_of_nat(v___x_2164_);
v___x_2174_ = ((size_t)1ULL);
v___x_2175_ = lean_usize_sub(v___x_2173_, v___x_2174_);
v___x_2176_ = lean_usize_land(v___x_2172_, v___x_2175_);
v_bkt_2177_ = lean_array_uget_borrowed(v_buckets_2160_, v___x_2176_);
v___x_2178_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg(v_a_2157_, v_bkt_2177_);
if (v___x_2178_ == 0)
{
lean_object* v___x_2179_; lean_object* v_size_x27_2180_; lean_object* v___x_2181_; lean_object* v_buckets_x27_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; uint8_t v___x_2188_; 
v___x_2179_ = lean_unsigned_to_nat(1u);
v_size_x27_2180_ = lean_nat_add(v_size_2159_, v___x_2179_);
lean_dec(v_size_2159_);
lean_inc(v_bkt_2177_);
v___x_2181_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2181_, 0, v_a_2157_);
lean_ctor_set(v___x_2181_, 1, v_b_2158_);
lean_ctor_set(v___x_2181_, 2, v_bkt_2177_);
v_buckets_x27_2182_ = lean_array_uset(v_buckets_2160_, v___x_2176_, v___x_2181_);
v___x_2183_ = lean_unsigned_to_nat(4u);
v___x_2184_ = lean_nat_mul(v_size_x27_2180_, v___x_2183_);
v___x_2185_ = lean_unsigned_to_nat(3u);
v___x_2186_ = lean_nat_div(v___x_2184_, v___x_2185_);
lean_dec(v___x_2184_);
v___x_2187_ = lean_array_get_size(v_buckets_x27_2182_);
v___x_2188_ = lean_nat_dec_le(v___x_2186_, v___x_2187_);
lean_dec(v___x_2186_);
if (v___x_2188_ == 0)
{
lean_object* v_val_2189_; lean_object* v___x_2191_; 
v_val_2189_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7___redArg(v_buckets_x27_2182_);
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 1, v_val_2189_);
lean_ctor_set(v___x_2162_, 0, v_size_x27_2180_);
v___x_2191_ = v___x_2162_;
goto v_reusejp_2190_;
}
else
{
lean_object* v_reuseFailAlloc_2192_; 
v_reuseFailAlloc_2192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2192_, 0, v_size_x27_2180_);
lean_ctor_set(v_reuseFailAlloc_2192_, 1, v_val_2189_);
v___x_2191_ = v_reuseFailAlloc_2192_;
goto v_reusejp_2190_;
}
v_reusejp_2190_:
{
return v___x_2191_;
}
}
else
{
lean_object* v___x_2194_; 
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 1, v_buckets_x27_2182_);
lean_ctor_set(v___x_2162_, 0, v_size_x27_2180_);
v___x_2194_ = v___x_2162_;
goto v_reusejp_2193_;
}
else
{
lean_object* v_reuseFailAlloc_2195_; 
v_reuseFailAlloc_2195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2195_, 0, v_size_x27_2180_);
lean_ctor_set(v_reuseFailAlloc_2195_, 1, v_buckets_x27_2182_);
v___x_2194_ = v_reuseFailAlloc_2195_;
goto v_reusejp_2193_;
}
v_reusejp_2193_:
{
return v___x_2194_;
}
}
}
else
{
lean_object* v___x_2196_; lean_object* v_buckets_x27_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2201_; 
lean_inc(v_bkt_2177_);
v___x_2196_ = lean_box(0);
v_buckets_x27_2197_ = lean_array_uset(v_buckets_2160_, v___x_2176_, v___x_2196_);
v___x_2198_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8___redArg(v_a_2157_, v_b_2158_, v_bkt_2177_);
v___x_2199_ = lean_array_uset(v_buckets_x27_2197_, v___x_2176_, v___x_2198_);
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 1, v___x_2199_);
v___x_2201_ = v___x_2162_;
goto v_reusejp_2200_;
}
else
{
lean_object* v_reuseFailAlloc_2202_; 
v_reuseFailAlloc_2202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2202_, 0, v_size_2159_);
lean_ctor_set(v_reuseFailAlloc_2202_, 1, v___x_2199_);
v___x_2201_ = v_reuseFailAlloc_2202_;
goto v_reusejp_2200_;
}
v_reusejp_2200_:
{
return v___x_2201_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5(lean_object* v_as_2204_, size_t v_sz_2205_, size_t v_i_2206_, lean_object* v_b_2207_){
_start:
{
uint8_t v___x_2208_; 
v___x_2208_ = lean_usize_dec_lt(v_i_2206_, v_sz_2205_);
if (v___x_2208_ == 0)
{
return v_b_2207_;
}
else
{
lean_object* v_a_2209_; lean_object* v_fst_2210_; lean_object* v_snd_2211_; lean_object* v_r_2212_; size_t v___x_2213_; size_t v___x_2214_; 
v_a_2209_ = lean_array_uget_borrowed(v_as_2204_, v_i_2206_);
v_fst_2210_ = lean_ctor_get(v_a_2209_, 0);
v_snd_2211_ = lean_ctor_get(v_a_2209_, 1);
lean_inc(v_snd_2211_);
lean_inc(v_fst_2210_);
v_r_2212_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4___redArg(v_b_2207_, v_fst_2210_, v_snd_2211_);
v___x_2213_ = ((size_t)1ULL);
v___x_2214_ = lean_usize_add(v_i_2206_, v___x_2213_);
v_i_2206_ = v___x_2214_;
v_b_2207_ = v_r_2212_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5___boxed(lean_object* v_as_2216_, lean_object* v_sz_2217_, lean_object* v_i_2218_, lean_object* v_b_2219_){
_start:
{
size_t v_sz_boxed_2220_; size_t v_i_boxed_2221_; lean_object* v_res_2222_; 
v_sz_boxed_2220_ = lean_unbox_usize(v_sz_2217_);
lean_dec(v_sz_2217_);
v_i_boxed_2221_ = lean_unbox_usize(v_i_2218_);
lean_dec(v_i_2218_);
v_res_2222_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5(v_as_2216_, v_sz_boxed_2220_, v_i_boxed_2221_, v_b_2219_);
lean_dec_ref(v_as_2216_);
return v_res_2222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3(lean_object* v_m_2223_, lean_object* v_l_2224_){
_start:
{
size_t v_sz_2225_; size_t v___x_2226_; lean_object* v___x_2227_; 
v_sz_2225_ = lean_array_size(v_l_2224_);
v___x_2226_ = ((size_t)0ULL);
v___x_2227_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__5(v_l_2224_, v_sz_2225_, v___x_2226_, v_m_2223_);
return v___x_2227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3___boxed(lean_object* v_m_2228_, lean_object* v_l_2229_){
_start:
{
lean_object* v_res_2230_; 
v_res_2230_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3(v_m_2228_, v_l_2229_);
lean_dec_ref(v_l_2229_);
return v_res_2230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0(lean_object* v_fst_2231_, lean_object* v_fst_2232_, lean_object* v_snd_2233_, lean_object* v_____r_2234_, lean_object* v_argReorders_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_){
_start:
{
lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; 
v___x_2241_ = l_Lean_Expr_bindingBody_x21(v_fst_2231_);
v___x_2242_ = l_Lean_Expr_bindingBody_x21(v_fst_2232_);
v___x_2243_ = lean_unsigned_to_nat(1u);
v___x_2244_ = lean_nat_add(v_snd_2233_, v___x_2243_);
v___x_2245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2245_, 0, v___x_2242_);
lean_ctor_set(v___x_2245_, 1, v___x_2244_);
v___x_2246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2246_, 0, v___x_2241_);
lean_ctor_set(v___x_2246_, 1, v___x_2245_);
v___x_2247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2247_, 0, v_argReorders_2235_);
lean_ctor_set(v___x_2247_, 1, v___x_2246_);
v___x_2248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2248_, 0, v___x_2247_);
v___x_2249_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2249_, 0, v___x_2248_);
return v___x_2249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0___boxed(lean_object* v_fst_2250_, lean_object* v_fst_2251_, lean_object* v_snd_2252_, lean_object* v_____r_2253_, lean_object* v_argReorders_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_){
_start:
{
lean_object* v_res_2260_; 
v_res_2260_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0(v_fst_2250_, v_fst_2251_, v_snd_2252_, v_____r_2253_, v_argReorders_2254_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_);
lean_dec(v___y_2258_);
lean_dec_ref(v___y_2257_);
lean_dec(v___y_2256_);
lean_dec_ref(v___y_2255_);
lean_dec(v_snd_2252_);
lean_dec(v_fst_2251_);
lean_dec(v_fst_2250_);
return v_res_2260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1___boxed(lean_object* v___x_2261_, lean_object* v_a_2262_, lean_object* v___x_2263_, lean_object* v_srcVars_2264_, lean_object* v_src_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_, lean_object* v___y_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_){
_start:
{
lean_object* v_res_2271_; 
v_res_2271_ = lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1(v___x_2261_, v_a_2262_, v___x_2263_, v_srcVars_2264_, v_src_2265_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_);
lean_dec(v___y_2269_);
lean_dec_ref(v___y_2268_);
lean_dec(v___y_2267_);
lean_dec_ref(v___y_2266_);
return v_res_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder(lean_object* v_src_2272_, lean_object* v_tgt_2273_, lean_object* v_a_2274_, lean_object* v_a_2275_, lean_object* v_a_2276_, lean_object* v_a_2277_){
_start:
{
lean_object* v_keyedConfig_2279_; uint8_t v_trackZetaDelta_2280_; lean_object* v_zetaDeltaSet_2281_; lean_object* v_lctx_2282_; lean_object* v_localInstances_2283_; lean_object* v_defEqCtx_x3f_2284_; lean_object* v_synthPendingDepth_2285_; lean_object* v_customCanUnfoldPredicate_x3f_2286_; uint8_t v_univApprox_2287_; uint8_t v_inTypeClassResolution_2288_; uint8_t v_cacheInferType_2289_; uint8_t v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; 
v_keyedConfig_2279_ = lean_ctor_get(v_a_2274_, 0);
v_trackZetaDelta_2280_ = lean_ctor_get_uint8(v_a_2274_, sizeof(void*)*7);
v_zetaDeltaSet_2281_ = lean_ctor_get(v_a_2274_, 1);
v_lctx_2282_ = lean_ctor_get(v_a_2274_, 2);
v_localInstances_2283_ = lean_ctor_get(v_a_2274_, 3);
v_defEqCtx_x3f_2284_ = lean_ctor_get(v_a_2274_, 4);
v_synthPendingDepth_2285_ = lean_ctor_get(v_a_2274_, 5);
v_customCanUnfoldPredicate_x3f_2286_ = lean_ctor_get(v_a_2274_, 6);
v_univApprox_2287_ = lean_ctor_get_uint8(v_a_2274_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2288_ = lean_ctor_get_uint8(v_a_2274_, sizeof(void*)*7 + 2);
v_cacheInferType_2289_ = lean_ctor_get_uint8(v_a_2274_, sizeof(void*)*7 + 3);
v___x_2290_ = 2;
lean_inc_ref(v_keyedConfig_2279_);
v___x_2291_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2290_, v_keyedConfig_2279_);
lean_inc(v_customCanUnfoldPredicate_x3f_2286_);
lean_inc(v_synthPendingDepth_2285_);
lean_inc(v_defEqCtx_x3f_2284_);
lean_inc_ref(v_localInstances_2283_);
lean_inc_ref(v_lctx_2282_);
lean_inc(v_zetaDeltaSet_2281_);
v___x_2292_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2292_, 0, v___x_2291_);
lean_ctor_set(v___x_2292_, 1, v_zetaDeltaSet_2281_);
lean_ctor_set(v___x_2292_, 2, v_lctx_2282_);
lean_ctor_set(v___x_2292_, 3, v_localInstances_2283_);
lean_ctor_set(v___x_2292_, 4, v_defEqCtx_x3f_2284_);
lean_ctor_set(v___x_2292_, 5, v_synthPendingDepth_2285_);
lean_ctor_set(v___x_2292_, 6, v_customCanUnfoldPredicate_x3f_2286_);
lean_ctor_set_uint8(v___x_2292_, sizeof(void*)*7, v_trackZetaDelta_2280_);
lean_ctor_set_uint8(v___x_2292_, sizeof(void*)*7 + 1, v_univApprox_2287_);
lean_ctor_set_uint8(v___x_2292_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2288_);
lean_ctor_set_uint8(v___x_2292_, sizeof(void*)*7 + 3, v_cacheInferType_2289_);
lean_inc(v_a_2277_);
lean_inc_ref(v_a_2276_);
lean_inc(v_a_2275_);
lean_inc_ref(v___x_2292_);
v___x_2293_ = lean_whnf(v_src_2272_, v___x_2292_, v_a_2275_, v_a_2276_, v_a_2277_);
if (lean_obj_tag(v___x_2293_) == 0)
{
lean_object* v_a_2294_; lean_object* v___x_2295_; 
v_a_2294_ = lean_ctor_get(v___x_2293_, 0);
lean_inc(v_a_2294_);
lean_dec_ref_known(v___x_2293_, 1);
lean_inc(v_a_2277_);
lean_inc_ref(v_a_2276_);
lean_inc(v_a_2275_);
lean_inc_ref(v___x_2292_);
v___x_2295_ = lean_whnf(v_tgt_2273_, v___x_2292_, v_a_2275_, v_a_2276_, v_a_2277_);
if (lean_obj_tag(v___x_2295_) == 0)
{
lean_object* v_a_2296_; lean_object* v___x_2298_; uint8_t v_isShared_2299_; uint8_t v_isSharedCheck_2311_; 
v_a_2296_ = lean_ctor_get(v___x_2295_, 0);
v_isSharedCheck_2311_ = !lean_is_exclusive(v___x_2295_);
if (v_isSharedCheck_2311_ == 0)
{
v___x_2298_ = v___x_2295_;
v_isShared_2299_ = v_isSharedCheck_2311_;
goto v_resetjp_2297_;
}
else
{
lean_inc(v_a_2296_);
lean_dec(v___x_2295_);
v___x_2298_ = lean_box(0);
v_isShared_2299_ = v_isSharedCheck_2311_;
goto v_resetjp_2297_;
}
v_resetjp_2297_:
{
lean_object* v___x_2300_; lean_object* v___x_2301_; uint8_t v___x_2302_; 
v___x_2300_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(v_a_2294_);
v___x_2301_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_depForallDepth(v_a_2296_);
v___x_2302_ = lean_nat_dec_eq(v___x_2300_, v___x_2301_);
lean_dec(v___x_2301_);
if (v___x_2302_ == 0)
{
lean_object* v___x_2303_; lean_object* v___x_2305_; 
lean_dec(v___x_2300_);
lean_dec(v_a_2296_);
lean_dec(v_a_2294_);
lean_dec_ref_known(v___x_2292_, 7);
v___x_2303_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__1));
if (v_isShared_2299_ == 0)
{
lean_ctor_set(v___x_2298_, 0, v___x_2303_);
v___x_2305_ = v___x_2298_;
goto v_reusejp_2304_;
}
else
{
lean_object* v_reuseFailAlloc_2306_; 
v_reuseFailAlloc_2306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2306_, 0, v___x_2303_);
v___x_2305_ = v_reuseFailAlloc_2306_;
goto v_reusejp_2304_;
}
v_reusejp_2304_:
{
return v___x_2305_;
}
}
else
{
lean_object* v___x_2307_; lean_object* v___f_2308_; uint8_t v___x_2309_; lean_object* v___x_2310_; 
lean_del_object(v___x_2298_);
lean_inc(v___x_2300_);
v___x_2307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2307_, 0, v___x_2300_);
lean_inc_ref(v___x_2307_);
v___f_2308_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1___boxed), 10, 3);
lean_closure_set(v___f_2308_, 0, v___x_2300_);
lean_closure_set(v___f_2308_, 1, v_a_2296_);
lean_closure_set(v___f_2308_, 2, v___x_2307_);
v___x_2309_ = 0;
v___x_2310_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(v_a_2294_, v___x_2307_, v___f_2308_, v___x_2309_, v___x_2309_, v___x_2292_, v_a_2275_, v_a_2276_, v_a_2277_);
lean_dec_ref_known(v___x_2292_, 7);
return v___x_2310_;
}
}
}
else
{
lean_object* v_a_2312_; lean_object* v___x_2314_; uint8_t v_isShared_2315_; uint8_t v_isSharedCheck_2319_; 
lean_dec(v_a_2294_);
lean_dec_ref_known(v___x_2292_, 7);
v_a_2312_ = lean_ctor_get(v___x_2295_, 0);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2295_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2314_ = v___x_2295_;
v_isShared_2315_ = v_isSharedCheck_2319_;
goto v_resetjp_2313_;
}
else
{
lean_inc(v_a_2312_);
lean_dec(v___x_2295_);
v___x_2314_ = lean_box(0);
v_isShared_2315_ = v_isSharedCheck_2319_;
goto v_resetjp_2313_;
}
v_resetjp_2313_:
{
lean_object* v___x_2317_; 
if (v_isShared_2315_ == 0)
{
v___x_2317_ = v___x_2314_;
goto v_reusejp_2316_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v_a_2312_);
v___x_2317_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2316_;
}
v_reusejp_2316_:
{
return v___x_2317_;
}
}
}
}
else
{
lean_object* v_a_2320_; lean_object* v___x_2322_; uint8_t v_isShared_2323_; uint8_t v_isSharedCheck_2327_; 
lean_dec_ref_known(v___x_2292_, 7);
lean_dec_ref(v_tgt_2273_);
v_a_2320_ = lean_ctor_get(v___x_2293_, 0);
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2293_);
if (v_isSharedCheck_2327_ == 0)
{
v___x_2322_ = v___x_2293_;
v_isShared_2323_ = v_isSharedCheck_2327_;
goto v_resetjp_2321_;
}
else
{
lean_inc(v_a_2320_);
lean_dec(v___x_2293_);
v___x_2322_ = lean_box(0);
v_isShared_2323_ = v_isSharedCheck_2327_;
goto v_resetjp_2321_;
}
v_resetjp_2321_:
{
lean_object* v___x_2325_; 
if (v_isShared_2323_ == 0)
{
v___x_2325_ = v___x_2322_;
goto v_reusejp_2324_;
}
else
{
lean_object* v_reuseFailAlloc_2326_; 
v_reuseFailAlloc_2326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2326_, 0, v_a_2320_);
v___x_2325_ = v_reuseFailAlloc_2326_;
goto v_reusejp_2324_;
}
v_reusejp_2324_:
{
return v___x_2325_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg(lean_object* v_upperBound_2328_, lean_object* v_srcVars_2329_, lean_object* v_tgtVars_2330_, lean_object* v_a_2331_, lean_object* v_b_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_){
_start:
{
uint8_t v___x_2338_; 
v___x_2338_ = lean_nat_dec_lt(v_a_2331_, v_upperBound_2328_);
if (v___x_2338_ == 0)
{
lean_object* v___x_2339_; 
lean_dec(v_a_2331_);
v___x_2339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2339_, 0, v_b_2332_);
return v___x_2339_;
}
else
{
lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = l_Lean_instInhabitedExpr;
v___x_2341_ = lean_array_get_borrowed(v___x_2340_, v_srcVars_2329_, v_a_2331_);
lean_inc(v___y_2336_);
lean_inc_ref(v___y_2335_);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___x_2341_);
v___x_2342_ = lean_infer_type(v___x_2341_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_);
if (lean_obj_tag(v___x_2342_) == 0)
{
lean_object* v_a_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; 
v_a_2343_ = lean_ctor_get(v___x_2342_, 0);
lean_inc(v_a_2343_);
lean_dec_ref_known(v___x_2342_, 1);
v___x_2344_ = lean_array_get_borrowed(v___x_2340_, v_tgtVars_2330_, v_a_2331_);
lean_inc(v___y_2336_);
lean_inc_ref(v___y_2335_);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___x_2344_);
v___x_2345_ = lean_infer_type(v___x_2344_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_);
if (lean_obj_tag(v___x_2345_) == 0)
{
lean_object* v_a_2346_; lean_object* v___x_2347_; 
v_a_2346_ = lean_ctor_get(v___x_2345_, 0);
lean_inc(v_a_2346_);
lean_dec_ref_known(v___x_2345_, 1);
v___x_2347_ = lp_mathlib_Mathlib_Tactic_Translate_guessReorder(v_a_2343_, v_a_2346_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_);
if (lean_obj_tag(v___x_2347_) == 0)
{
lean_object* v_a_2348_; lean_object* v_a_2350_; uint8_t v___x_2354_; 
v_a_2348_ = lean_ctor_get(v___x_2347_, 0);
lean_inc(v_a_2348_);
lean_dec_ref_known(v___x_2347_, 1);
v___x_2354_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty(v_a_2348_);
if (v___x_2354_ == 0)
{
lean_object* v___x_2355_; lean_object* v___x_2356_; 
lean_inc(v_a_2331_);
v___x_2355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2355_, 0, v_a_2331_);
lean_ctor_set(v___x_2355_, 1, v_a_2348_);
v___x_2356_ = lean_array_push(v_b_2332_, v___x_2355_);
v_a_2350_ = v___x_2356_;
goto v___jp_2349_;
}
else
{
lean_dec(v_a_2348_);
v_a_2350_ = v_b_2332_;
goto v___jp_2349_;
}
v___jp_2349_:
{
lean_object* v___x_2351_; lean_object* v___x_2352_; 
v___x_2351_ = lean_unsigned_to_nat(1u);
v___x_2352_ = lean_nat_add(v_a_2331_, v___x_2351_);
lean_dec(v_a_2331_);
v_a_2331_ = v___x_2352_;
v_b_2332_ = v_a_2350_;
goto _start;
}
}
else
{
lean_object* v_a_2357_; lean_object* v___x_2359_; uint8_t v_isShared_2360_; uint8_t v_isSharedCheck_2364_; 
lean_dec_ref(v_b_2332_);
lean_dec(v_a_2331_);
v_a_2357_ = lean_ctor_get(v___x_2347_, 0);
v_isSharedCheck_2364_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2364_ == 0)
{
v___x_2359_ = v___x_2347_;
v_isShared_2360_ = v_isSharedCheck_2364_;
goto v_resetjp_2358_;
}
else
{
lean_inc(v_a_2357_);
lean_dec(v___x_2347_);
v___x_2359_ = lean_box(0);
v_isShared_2360_ = v_isSharedCheck_2364_;
goto v_resetjp_2358_;
}
v_resetjp_2358_:
{
lean_object* v___x_2362_; 
if (v_isShared_2360_ == 0)
{
v___x_2362_ = v___x_2359_;
goto v_reusejp_2361_;
}
else
{
lean_object* v_reuseFailAlloc_2363_; 
v_reuseFailAlloc_2363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2363_, 0, v_a_2357_);
v___x_2362_ = v_reuseFailAlloc_2363_;
goto v_reusejp_2361_;
}
v_reusejp_2361_:
{
return v___x_2362_;
}
}
}
}
else
{
lean_object* v_a_2365_; lean_object* v___x_2367_; uint8_t v_isShared_2368_; uint8_t v_isSharedCheck_2372_; 
lean_dec(v_a_2343_);
lean_dec_ref(v_b_2332_);
lean_dec(v_a_2331_);
v_a_2365_ = lean_ctor_get(v___x_2345_, 0);
v_isSharedCheck_2372_ = !lean_is_exclusive(v___x_2345_);
if (v_isSharedCheck_2372_ == 0)
{
v___x_2367_ = v___x_2345_;
v_isShared_2368_ = v_isSharedCheck_2372_;
goto v_resetjp_2366_;
}
else
{
lean_inc(v_a_2365_);
lean_dec(v___x_2345_);
v___x_2367_ = lean_box(0);
v_isShared_2368_ = v_isSharedCheck_2372_;
goto v_resetjp_2366_;
}
v_resetjp_2366_:
{
lean_object* v___x_2370_; 
if (v_isShared_2368_ == 0)
{
v___x_2370_ = v___x_2367_;
goto v_reusejp_2369_;
}
else
{
lean_object* v_reuseFailAlloc_2371_; 
v_reuseFailAlloc_2371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2371_, 0, v_a_2365_);
v___x_2370_ = v_reuseFailAlloc_2371_;
goto v_reusejp_2369_;
}
v_reusejp_2369_:
{
return v___x_2370_;
}
}
}
}
else
{
lean_object* v_a_2373_; lean_object* v___x_2375_; uint8_t v_isShared_2376_; uint8_t v_isSharedCheck_2380_; 
lean_dec_ref(v_b_2332_);
lean_dec(v_a_2331_);
v_a_2373_ = lean_ctor_get(v___x_2342_, 0);
v_isSharedCheck_2380_ = !lean_is_exclusive(v___x_2342_);
if (v_isSharedCheck_2380_ == 0)
{
v___x_2375_ = v___x_2342_;
v_isShared_2376_ = v_isSharedCheck_2380_;
goto v_resetjp_2374_;
}
else
{
lean_inc(v_a_2373_);
lean_dec(v___x_2342_);
v___x_2375_ = lean_box(0);
v_isShared_2376_ = v_isSharedCheck_2380_;
goto v_resetjp_2374_;
}
v_resetjp_2374_:
{
lean_object* v___x_2378_; 
if (v_isShared_2376_ == 0)
{
v___x_2378_ = v___x_2375_;
goto v_reusejp_2377_;
}
else
{
lean_object* v_reuseFailAlloc_2379_; 
v_reuseFailAlloc_2379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2379_, 0, v_a_2373_);
v___x_2378_ = v_reuseFailAlloc_2379_;
goto v_reusejp_2377_;
}
v_reusejp_2377_:
{
return v___x_2378_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg(lean_object* v_a_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_){
_start:
{
lean_object* v___y_2388_; lean_object* v_snd_2408_; lean_object* v_snd_2409_; lean_object* v_fst_2410_; lean_object* v___x_2412_; uint8_t v_isShared_2413_; uint8_t v_isSharedCheck_2461_; 
v_snd_2408_ = lean_ctor_get(v_a_2381_, 1);
lean_inc(v_snd_2408_);
v_snd_2409_ = lean_ctor_get(v_snd_2408_, 1);
lean_inc(v_snd_2409_);
v_fst_2410_ = lean_ctor_get(v_a_2381_, 0);
v_isSharedCheck_2461_ = !lean_is_exclusive(v_a_2381_);
if (v_isSharedCheck_2461_ == 0)
{
lean_object* v_unused_2462_; 
v_unused_2462_ = lean_ctor_get(v_a_2381_, 1);
lean_dec(v_unused_2462_);
v___x_2412_ = v_a_2381_;
v_isShared_2413_ = v_isSharedCheck_2461_;
goto v_resetjp_2411_;
}
else
{
lean_inc(v_fst_2410_);
lean_dec(v_a_2381_);
v___x_2412_ = lean_box(0);
v_isShared_2413_ = v_isSharedCheck_2461_;
goto v_resetjp_2411_;
}
v___jp_2387_:
{
if (lean_obj_tag(v___y_2388_) == 0)
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2399_; 
v_a_2389_ = lean_ctor_get(v___y_2388_, 0);
v_isSharedCheck_2399_ = !lean_is_exclusive(v___y_2388_);
if (v_isSharedCheck_2399_ == 0)
{
v___x_2391_ = v___y_2388_;
v_isShared_2392_ = v_isSharedCheck_2399_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___y_2388_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2399_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
if (lean_obj_tag(v_a_2389_) == 0)
{
lean_object* v_a_2393_; lean_object* v___x_2395_; 
v_a_2393_ = lean_ctor_get(v_a_2389_, 0);
lean_inc(v_a_2393_);
lean_dec_ref_known(v_a_2389_, 1);
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v_a_2393_);
v___x_2395_ = v___x_2391_;
goto v_reusejp_2394_;
}
else
{
lean_object* v_reuseFailAlloc_2396_; 
v_reuseFailAlloc_2396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2396_, 0, v_a_2393_);
v___x_2395_ = v_reuseFailAlloc_2396_;
goto v_reusejp_2394_;
}
v_reusejp_2394_:
{
return v___x_2395_;
}
}
else
{
lean_object* v_a_2397_; 
lean_del_object(v___x_2391_);
v_a_2397_ = lean_ctor_get(v_a_2389_, 0);
lean_inc(v_a_2397_);
lean_dec_ref_known(v_a_2389_, 1);
v_a_2381_ = v_a_2397_;
goto _start;
}
}
}
else
{
lean_object* v_a_2400_; lean_object* v___x_2402_; uint8_t v_isShared_2403_; uint8_t v_isSharedCheck_2407_; 
v_a_2400_ = lean_ctor_get(v___y_2388_, 0);
v_isSharedCheck_2407_ = !lean_is_exclusive(v___y_2388_);
if (v_isSharedCheck_2407_ == 0)
{
v___x_2402_ = v___y_2388_;
v_isShared_2403_ = v_isSharedCheck_2407_;
goto v_resetjp_2401_;
}
else
{
lean_inc(v_a_2400_);
lean_dec(v___y_2388_);
v___x_2402_ = lean_box(0);
v_isShared_2403_ = v_isSharedCheck_2407_;
goto v_resetjp_2401_;
}
v_resetjp_2401_:
{
lean_object* v___x_2405_; 
if (v_isShared_2403_ == 0)
{
v___x_2405_ = v___x_2402_;
goto v_reusejp_2404_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v_a_2400_);
v___x_2405_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2404_;
}
v_reusejp_2404_:
{
return v___x_2405_;
}
}
}
}
v_resetjp_2411_:
{
lean_object* v_fst_2414_; lean_object* v___x_2416_; uint8_t v_isShared_2417_; uint8_t v_isSharedCheck_2459_; 
v_fst_2414_ = lean_ctor_get(v_snd_2408_, 0);
v_isSharedCheck_2459_ = !lean_is_exclusive(v_snd_2408_);
if (v_isSharedCheck_2459_ == 0)
{
lean_object* v_unused_2460_; 
v_unused_2460_ = lean_ctor_get(v_snd_2408_, 1);
lean_dec(v_unused_2460_);
v___x_2416_ = v_snd_2408_;
v_isShared_2417_ = v_isSharedCheck_2459_;
goto v_resetjp_2415_;
}
else
{
lean_inc(v_fst_2414_);
lean_dec(v_snd_2408_);
v___x_2416_ = lean_box(0);
v_isShared_2417_ = v_isSharedCheck_2459_;
goto v_resetjp_2415_;
}
v_resetjp_2415_:
{
lean_object* v_fst_2418_; lean_object* v_snd_2419_; lean_object* v___x_2421_; uint8_t v_isShared_2422_; uint8_t v_isSharedCheck_2458_; 
v_fst_2418_ = lean_ctor_get(v_snd_2409_, 0);
v_snd_2419_ = lean_ctor_get(v_snd_2409_, 1);
v_isSharedCheck_2458_ = !lean_is_exclusive(v_snd_2409_);
if (v_isSharedCheck_2458_ == 0)
{
v___x_2421_ = v_snd_2409_;
v_isShared_2422_ = v_isSharedCheck_2458_;
goto v_resetjp_2420_;
}
else
{
lean_inc(v_snd_2419_);
lean_inc(v_fst_2418_);
lean_dec(v_snd_2409_);
v___x_2421_ = lean_box(0);
v_isShared_2422_ = v_isSharedCheck_2458_;
goto v_resetjp_2420_;
}
v_resetjp_2420_:
{
uint8_t v___y_2424_; uint8_t v___x_2456_; 
v___x_2456_ = l_Lean_Expr_isForall(v_fst_2414_);
if (v___x_2456_ == 0)
{
v___y_2424_ = v___x_2456_;
goto v___jp_2423_;
}
else
{
uint8_t v___x_2457_; 
v___x_2457_ = l_Lean_Expr_isForall(v_fst_2418_);
v___y_2424_ = v___x_2457_;
goto v___jp_2423_;
}
v___jp_2423_:
{
if (v___y_2424_ == 0)
{
lean_object* v___x_2426_; 
if (v_isShared_2422_ == 0)
{
v___x_2426_ = v___x_2421_;
goto v_reusejp_2425_;
}
else
{
lean_object* v_reuseFailAlloc_2434_; 
v_reuseFailAlloc_2434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2434_, 0, v_fst_2418_);
lean_ctor_set(v_reuseFailAlloc_2434_, 1, v_snd_2419_);
v___x_2426_ = v_reuseFailAlloc_2434_;
goto v_reusejp_2425_;
}
v_reusejp_2425_:
{
lean_object* v___x_2428_; 
if (v_isShared_2417_ == 0)
{
lean_ctor_set(v___x_2416_, 1, v___x_2426_);
v___x_2428_ = v___x_2416_;
goto v_reusejp_2427_;
}
else
{
lean_object* v_reuseFailAlloc_2433_; 
v_reuseFailAlloc_2433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2433_, 0, v_fst_2414_);
lean_ctor_set(v_reuseFailAlloc_2433_, 1, v___x_2426_);
v___x_2428_ = v_reuseFailAlloc_2433_;
goto v_reusejp_2427_;
}
v_reusejp_2427_:
{
lean_object* v___x_2430_; 
if (v_isShared_2413_ == 0)
{
lean_ctor_set(v___x_2412_, 1, v___x_2428_);
v___x_2430_ = v___x_2412_;
goto v_reusejp_2429_;
}
else
{
lean_object* v_reuseFailAlloc_2432_; 
v_reuseFailAlloc_2432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2432_, 0, v_fst_2410_);
lean_ctor_set(v_reuseFailAlloc_2432_, 1, v___x_2428_);
v___x_2430_ = v_reuseFailAlloc_2432_;
goto v_reusejp_2429_;
}
v_reusejp_2429_:
{
lean_object* v___x_2431_; 
v___x_2431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2430_);
return v___x_2431_;
}
}
}
}
else
{
lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; 
lean_del_object(v___x_2416_);
lean_del_object(v___x_2412_);
v___x_2435_ = l_Lean_Expr_bindingDomain_x21(v_fst_2414_);
v___x_2436_ = l_Lean_Expr_bindingDomain_x21(v_fst_2418_);
v___x_2437_ = lp_mathlib_Mathlib_Tactic_Translate_guessReorder(v___x_2435_, v___x_2436_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
if (lean_obj_tag(v___x_2437_) == 0)
{
lean_object* v_a_2438_; uint8_t v___x_2439_; 
v_a_2438_ = lean_ctor_get(v___x_2437_, 0);
lean_inc(v_a_2438_);
lean_dec_ref_known(v___x_2437_, 1);
v___x_2439_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_isEmpty(v_a_2438_);
if (v___x_2439_ == 0)
{
lean_object* v___x_2441_; 
lean_inc(v_snd_2419_);
if (v_isShared_2422_ == 0)
{
lean_ctor_set(v___x_2421_, 1, v_a_2438_);
lean_ctor_set(v___x_2421_, 0, v_snd_2419_);
v___x_2441_ = v___x_2421_;
goto v_reusejp_2440_;
}
else
{
lean_object* v_reuseFailAlloc_2445_; 
v_reuseFailAlloc_2445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2445_, 0, v_snd_2419_);
lean_ctor_set(v_reuseFailAlloc_2445_, 1, v_a_2438_);
v___x_2441_ = v_reuseFailAlloc_2445_;
goto v_reusejp_2440_;
}
v_reusejp_2440_:
{
lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; 
v___x_2442_ = lean_array_push(v_fst_2410_, v___x_2441_);
v___x_2443_ = lean_box(0);
v___x_2444_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0(v_fst_2414_, v_fst_2418_, v_snd_2419_, v___x_2443_, v___x_2442_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
lean_dec(v_snd_2419_);
lean_dec(v_fst_2418_);
lean_dec(v_fst_2414_);
v___y_2388_ = v___x_2444_;
goto v___jp_2387_;
}
}
else
{
lean_object* v___x_2446_; lean_object* v___x_2447_; 
lean_dec(v_a_2438_);
lean_del_object(v___x_2421_);
v___x_2446_ = lean_box(0);
v___x_2447_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___lam__0(v_fst_2414_, v_fst_2418_, v_snd_2419_, v___x_2446_, v_fst_2410_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_);
lean_dec(v_snd_2419_);
lean_dec(v_fst_2418_);
lean_dec(v_fst_2414_);
v___y_2388_ = v___x_2447_;
goto v___jp_2387_;
}
}
else
{
lean_object* v_a_2448_; lean_object* v___x_2450_; uint8_t v_isShared_2451_; uint8_t v_isSharedCheck_2455_; 
lean_del_object(v___x_2421_);
lean_dec(v_snd_2419_);
lean_dec(v_fst_2418_);
lean_dec(v_fst_2414_);
lean_dec(v_fst_2410_);
v_a_2448_ = lean_ctor_get(v___x_2437_, 0);
v_isSharedCheck_2455_ = !lean_is_exclusive(v___x_2437_);
if (v_isSharedCheck_2455_ == 0)
{
v___x_2450_ = v___x_2437_;
v_isShared_2451_ = v_isSharedCheck_2455_;
goto v_resetjp_2449_;
}
else
{
lean_inc(v_a_2448_);
lean_dec(v___x_2437_);
v___x_2450_ = lean_box(0);
v_isShared_2451_ = v_isSharedCheck_2455_;
goto v_resetjp_2449_;
}
v_resetjp_2449_:
{
lean_object* v___x_2453_; 
if (v_isShared_2451_ == 0)
{
v___x_2453_ = v___x_2450_;
goto v_reusejp_2452_;
}
else
{
lean_object* v_reuseFailAlloc_2454_; 
v_reuseFailAlloc_2454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2454_, 0, v_a_2448_);
v___x_2453_ = v_reuseFailAlloc_2454_;
goto v_reusejp_2452_;
}
v_reusejp_2452_:
{
return v___x_2453_;
}
}
}
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; 
v___x_2463_ = lean_box(0);
v___x_2464_ = lean_unsigned_to_nat(16u);
v___x_2465_ = lean_mk_array(v___x_2464_, v___x_2463_);
return v___x_2465_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2466_; lean_object* v___x_2467_; lean_object* v___x_2468_; 
v___x_2466_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__0);
v___x_2467_ = lean_unsigned_to_nat(0u);
v___x_2468_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2468_, 0, v___x_2467_);
lean_ctor_set(v___x_2468_, 1, v___x_2466_);
return v___x_2468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0(lean_object* v_srcVars_2469_, lean_object* v___x_2470_, lean_object* v_src_2471_, lean_object* v_tgtVars_2472_, lean_object* v_tgt_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_){
_start:
{
lean_object* v___y_2480_; size_t v_sz_2522_; size_t v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; size_t v_sz_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; 
v_sz_2522_ = lean_array_size(v_srcVars_2469_);
v___x_2523_ = ((size_t)0ULL);
lean_inc_ref(v_srcVars_2469_);
v___x_2524_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2(v_srcVars_2469_, v_sz_2522_, v___x_2523_, v_srcVars_2469_);
v___x_2525_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___closed__1);
v_sz_2526_ = lean_array_size(v_tgtVars_2472_);
lean_inc_ref(v_tgtVars_2472_);
v___x_2527_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2(v_tgtVars_2472_, v_sz_2526_, v___x_2523_, v_tgtVars_2472_);
v___x_2528_ = lean_st_mk_ref(v___x_2525_);
v___x_2529_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3(v___x_2525_, v___x_2524_);
lean_dec_ref(v___x_2524_);
v___x_2530_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3(v___x_2525_, v___x_2527_);
lean_dec_ref(v___x_2527_);
v___x_2531_ = lean_box(0);
lean_inc_n(v___x_2470_, 2);
v___x_2532_ = lean_mk_array(v___x_2470_, v___x_2531_);
v___x_2533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2533_, 0, v___x_2529_);
lean_ctor_set(v___x_2533_, 1, v___x_2530_);
lean_inc_ref(v_tgt_2473_);
lean_inc_ref(v_src_2471_);
v___x_2534_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_guessReorder_visit(v_src_2471_, v_tgt_2473_, v___x_2470_, v___x_2532_, v___x_2533_, v___x_2528_);
lean_dec_ref_known(v___x_2533_, 2);
if (lean_obj_tag(v___x_2534_) == 0)
{
lean_dec(v___x_2528_);
goto v___jp_2535_;
}
else
{
lean_object* v___x_2539_; 
v___x_2539_ = lean_st_ref_get(v___x_2528_);
lean_dec(v___x_2528_);
lean_dec(v___x_2539_);
goto v___jp_2535_;
}
v___jp_2479_:
{
lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; 
v___x_2481_ = lean_unsigned_to_nat(0u);
v___x_2482_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_instInhabitedArgReorder_default___closed__0));
v___x_2483_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg(v___x_2470_, v_srcVars_2469_, v_tgtVars_2472_, v___x_2481_, v___x_2482_, v___y_2474_, v___y_2475_, v___y_2476_, v___y_2477_);
lean_dec_ref(v_tgtVars_2472_);
lean_dec_ref(v_srcVars_2469_);
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_object* v_a_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; 
v_a_2484_ = lean_ctor_get(v___x_2483_, 0);
lean_inc(v_a_2484_);
lean_dec_ref_known(v___x_2483_, 1);
v___x_2485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2485_, 0, v_tgt_2473_);
lean_ctor_set(v___x_2485_, 1, v___x_2470_);
v___x_2486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2486_, 0, v_src_2471_);
lean_ctor_set(v___x_2486_, 1, v___x_2485_);
v___x_2487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2487_, 0, v_a_2484_);
lean_ctor_set(v___x_2487_, 1, v___x_2486_);
v___x_2488_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg(v___x_2487_, v___y_2474_, v___y_2475_, v___y_2476_, v___y_2477_);
if (lean_obj_tag(v___x_2488_) == 0)
{
lean_object* v_a_2489_; lean_object* v___x_2491_; uint8_t v_isShared_2492_; uint8_t v_isSharedCheck_2505_; 
v_a_2489_ = lean_ctor_get(v___x_2488_, 0);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2488_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2491_ = v___x_2488_;
v_isShared_2492_ = v_isSharedCheck_2505_;
goto v_resetjp_2490_;
}
else
{
lean_inc(v_a_2489_);
lean_dec(v___x_2488_);
v___x_2491_ = lean_box(0);
v_isShared_2492_ = v_isSharedCheck_2505_;
goto v_resetjp_2490_;
}
v_resetjp_2490_:
{
lean_object* v_fst_2493_; lean_object* v___x_2495_; uint8_t v_isShared_2496_; uint8_t v_isSharedCheck_2503_; 
v_fst_2493_ = lean_ctor_get(v_a_2489_, 0);
v_isSharedCheck_2503_ = !lean_is_exclusive(v_a_2489_);
if (v_isSharedCheck_2503_ == 0)
{
lean_object* v_unused_2504_; 
v_unused_2504_ = lean_ctor_get(v_a_2489_, 1);
lean_dec(v_unused_2504_);
v___x_2495_ = v_a_2489_;
v_isShared_2496_ = v_isSharedCheck_2503_;
goto v_resetjp_2494_;
}
else
{
lean_inc(v_fst_2493_);
lean_dec(v_a_2489_);
v___x_2495_ = lean_box(0);
v_isShared_2496_ = v_isSharedCheck_2503_;
goto v_resetjp_2494_;
}
v_resetjp_2494_:
{
lean_object* v___x_2498_; 
if (v_isShared_2496_ == 0)
{
lean_ctor_set(v___x_2495_, 1, v_fst_2493_);
lean_ctor_set(v___x_2495_, 0, v___y_2480_);
v___x_2498_ = v___x_2495_;
goto v_reusejp_2497_;
}
else
{
lean_object* v_reuseFailAlloc_2502_; 
v_reuseFailAlloc_2502_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2502_, 0, v___y_2480_);
lean_ctor_set(v_reuseFailAlloc_2502_, 1, v_fst_2493_);
v___x_2498_ = v_reuseFailAlloc_2502_;
goto v_reusejp_2497_;
}
v_reusejp_2497_:
{
lean_object* v___x_2500_; 
if (v_isShared_2492_ == 0)
{
lean_ctor_set(v___x_2491_, 0, v___x_2498_);
v___x_2500_ = v___x_2491_;
goto v_reusejp_2499_;
}
else
{
lean_object* v_reuseFailAlloc_2501_; 
v_reuseFailAlloc_2501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2501_, 0, v___x_2498_);
v___x_2500_ = v_reuseFailAlloc_2501_;
goto v_reusejp_2499_;
}
v_reusejp_2499_:
{
return v___x_2500_;
}
}
}
}
}
else
{
lean_object* v_a_2506_; lean_object* v___x_2508_; uint8_t v_isShared_2509_; uint8_t v_isSharedCheck_2513_; 
lean_dec(v___y_2480_);
v_a_2506_ = lean_ctor_get(v___x_2488_, 0);
v_isSharedCheck_2513_ = !lean_is_exclusive(v___x_2488_);
if (v_isSharedCheck_2513_ == 0)
{
v___x_2508_ = v___x_2488_;
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
else
{
lean_inc(v_a_2506_);
lean_dec(v___x_2488_);
v___x_2508_ = lean_box(0);
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
v_resetjp_2507_:
{
lean_object* v___x_2511_; 
if (v_isShared_2509_ == 0)
{
v___x_2511_ = v___x_2508_;
goto v_reusejp_2510_;
}
else
{
lean_object* v_reuseFailAlloc_2512_; 
v_reuseFailAlloc_2512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2512_, 0, v_a_2506_);
v___x_2511_ = v_reuseFailAlloc_2512_;
goto v_reusejp_2510_;
}
v_reusejp_2510_:
{
return v___x_2511_;
}
}
}
}
else
{
lean_object* v_a_2514_; lean_object* v___x_2516_; uint8_t v_isShared_2517_; uint8_t v_isSharedCheck_2521_; 
lean_dec(v___y_2480_);
lean_dec_ref(v_tgt_2473_);
lean_dec_ref(v_src_2471_);
lean_dec(v___x_2470_);
v_a_2514_ = lean_ctor_get(v___x_2483_, 0);
v_isSharedCheck_2521_ = !lean_is_exclusive(v___x_2483_);
if (v_isSharedCheck_2521_ == 0)
{
v___x_2516_ = v___x_2483_;
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
else
{
lean_inc(v_a_2514_);
lean_dec(v___x_2483_);
v___x_2516_ = lean_box(0);
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
v_resetjp_2515_:
{
lean_object* v___x_2519_; 
if (v_isShared_2517_ == 0)
{
v___x_2519_ = v___x_2516_;
goto v_reusejp_2518_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v_a_2514_);
v___x_2519_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2518_;
}
v_reusejp_2518_:
{
return v___x_2519_;
}
}
}
}
v___jp_2535_:
{
if (lean_obj_tag(v___x_2534_) == 0)
{
lean_object* v___x_2536_; 
v___x_2536_ = lean_box(0);
v___y_2480_ = v___x_2536_;
goto v___jp_2479_;
}
else
{
lean_object* v_val_2537_; lean_object* v___x_2538_; 
v_val_2537_ = lean_ctor_get(v___x_2534_, 0);
lean_inc(v_val_2537_);
lean_dec_ref_known(v___x_2534_, 1);
v___x_2538_ = lp_mathlib___private_Mathlib_Tactic_Translate_Reorder_0__Mathlib_Tactic_Translate_decomposePerm(v___x_2470_, v_val_2537_);
v___y_2480_ = v___x_2538_;
goto v___jp_2479_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___boxed(lean_object* v_srcVars_2540_, lean_object* v___x_2541_, lean_object* v_src_2542_, lean_object* v_tgtVars_2543_, lean_object* v_tgt_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_){
_start:
{
lean_object* v_res_2550_; 
v_res_2550_ = lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0(v_srcVars_2540_, v___x_2541_, v_src_2542_, v_tgtVars_2543_, v_tgt_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
lean_dec(v___y_2548_);
lean_dec_ref(v___y_2547_);
lean_dec(v___y_2546_);
lean_dec_ref(v___y_2545_);
return v_res_2550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__1(lean_object* v___x_2551_, lean_object* v_a_2552_, lean_object* v___x_2553_, lean_object* v_srcVars_2554_, lean_object* v_src_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_, lean_object* v___y_2559_){
_start:
{
lean_object* v___f_2561_; uint8_t v___x_2562_; lean_object* v___x_2563_; 
v___f_2561_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Translate_guessReorder___lam__0___boxed), 10, 3);
lean_closure_set(v___f_2561_, 0, v_srcVars_2554_);
lean_closure_set(v___f_2561_, 1, v___x_2551_);
lean_closure_set(v___f_2561_, 2, v_src_2555_);
v___x_2562_ = 0;
v___x_2563_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg(v_a_2552_, v___x_2553_, v___f_2561_, v___x_2562_, v___x_2562_, v___y_2556_, v___y_2557_, v___y_2558_, v___y_2559_);
return v___x_2563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_guessReorder___boxed(lean_object* v_src_2564_, lean_object* v_tgt_2565_, lean_object* v_a_2566_, lean_object* v_a_2567_, lean_object* v_a_2568_, lean_object* v_a_2569_, lean_object* v_a_2570_){
_start:
{
lean_object* v_res_2571_; 
v_res_2571_ = lp_mathlib_Mathlib_Tactic_Translate_guessReorder(v_src_2564_, v_tgt_2565_, v_a_2566_, v_a_2567_, v_a_2568_, v_a_2569_);
lean_dec(v_a_2569_);
lean_dec_ref(v_a_2568_);
lean_dec(v_a_2567_);
lean_dec_ref(v_a_2566_);
return v_res_2571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg___boxed(lean_object* v_upperBound_2572_, lean_object* v_srcVars_2573_, lean_object* v_tgtVars_2574_, lean_object* v_a_2575_, lean_object* v_b_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_){
_start:
{
lean_object* v_res_2582_; 
v_res_2582_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg(v_upperBound_2572_, v_srcVars_2573_, v_tgtVars_2574_, v_a_2575_, v_b_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_);
lean_dec(v___y_2580_);
lean_dec_ref(v___y_2579_);
lean_dec(v___y_2578_);
lean_dec_ref(v___y_2577_);
lean_dec_ref(v_tgtVars_2574_);
lean_dec_ref(v_srcVars_2573_);
lean_dec(v_upperBound_2572_);
return v_res_2582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg___boxed(lean_object* v_a_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_){
_start:
{
lean_object* v_res_2589_; 
v_res_2589_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg(v_a_2583_, v___y_2584_, v___y_2585_, v___y_2586_, v___y_2587_);
lean_dec(v___y_2587_);
lean_dec_ref(v___y_2586_);
lean_dec(v___y_2585_);
lean_dec_ref(v___y_2584_);
return v_res_2589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0(lean_object* v_inst_2590_, lean_object* v_a_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_){
_start:
{
lean_object* v___x_2597_; 
v___x_2597_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___redArg(v_a_2591_, v___y_2592_, v___y_2593_, v___y_2594_, v___y_2595_);
return v___x_2597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0___boxed(lean_object* v_inst_2598_, lean_object* v_a_2599_, lean_object* v___y_2600_, lean_object* v___y_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_, lean_object* v___y_2604_){
_start:
{
lean_object* v_res_2605_; 
v_res_2605_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Translate_guessReorder_spec__0(v_inst_2598_, v_a_2599_, v___y_2600_, v___y_2601_, v___y_2602_, v___y_2603_);
lean_dec(v___y_2603_);
lean_dec_ref(v___y_2602_);
lean_dec(v___y_2601_);
lean_dec_ref(v___y_2600_);
return v_res_2605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1(lean_object* v_upperBound_2606_, lean_object* v_srcVars_2607_, lean_object* v_tgtVars_2608_, lean_object* v_inst_2609_, lean_object* v_R_2610_, lean_object* v_a_2611_, lean_object* v_b_2612_, lean_object* v_c_2613_, lean_object* v___y_2614_, lean_object* v___y_2615_, lean_object* v___y_2616_, lean_object* v___y_2617_){
_start:
{
lean_object* v___x_2619_; 
v___x_2619_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___redArg(v_upperBound_2606_, v_srcVars_2607_, v_tgtVars_2608_, v_a_2611_, v_b_2612_, v___y_2614_, v___y_2615_, v___y_2616_, v___y_2617_);
return v___x_2619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1___boxed(lean_object* v_upperBound_2620_, lean_object* v_srcVars_2621_, lean_object* v_tgtVars_2622_, lean_object* v_inst_2623_, lean_object* v_R_2624_, lean_object* v_a_2625_, lean_object* v_b_2626_, lean_object* v_c_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_, lean_object* v___y_2632_){
_start:
{
lean_object* v_res_2633_; 
v_res_2633_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_guessReorder_spec__1(v_upperBound_2620_, v_srcVars_2621_, v_tgtVars_2622_, v_inst_2623_, v_R_2624_, v_a_2625_, v_b_2626_, v_c_2627_, v___y_2628_, v___y_2629_, v___y_2630_, v___y_2631_);
lean_dec(v___y_2631_);
lean_dec_ref(v___y_2630_);
lean_dec(v___y_2629_);
lean_dec_ref(v___y_2628_);
lean_dec_ref(v_tgtVars_2622_);
lean_dec_ref(v_srcVars_2621_);
lean_dec(v_upperBound_2620_);
return v_res_2633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2(lean_object* v_as_2634_, size_t v_sz_2635_, size_t v_i_2636_, lean_object* v_bs_2637_){
_start:
{
lean_object* v___x_2638_; 
v___x_2638_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___redArg(v_sz_2635_, v_i_2636_, v_bs_2637_);
return v___x_2638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2___boxed(lean_object* v_as_2639_, lean_object* v_sz_2640_, lean_object* v_i_2641_, lean_object* v_bs_2642_){
_start:
{
size_t v_sz_boxed_2643_; size_t v_i_boxed_2644_; lean_object* v_res_2645_; 
v_sz_boxed_2643_ = lean_unbox_usize(v_sz_2640_);
lean_dec(v_sz_2640_);
v_i_boxed_2644_ = lean_unbox_usize(v_i_2641_);
lean_dec(v_i_2641_);
v_res_2645_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Translate_guessReorder_spec__2_spec__2(v_as_2639_, v_sz_boxed_2643_, v_i_boxed_2644_, v_bs_2642_);
lean_dec_ref(v_as_2639_);
return v_res_2645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4(lean_object* v_00_u03b2_2646_, lean_object* v_m_2647_, lean_object* v_a_2648_, lean_object* v_b_2649_){
_start:
{
lean_object* v___x_2650_; 
v___x_2650_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4___redArg(v_m_2647_, v_a_2648_, v_b_2649_);
return v___x_2650_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6(lean_object* v_00_u03b2_2651_, lean_object* v_a_2652_, lean_object* v_x_2653_){
_start:
{
uint8_t v___x_2654_; 
v___x_2654_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___redArg(v_a_2652_, v_x_2653_);
return v___x_2654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6___boxed(lean_object* v_00_u03b2_2655_, lean_object* v_a_2656_, lean_object* v_x_2657_){
_start:
{
uint8_t v_res_2658_; lean_object* v_r_2659_; 
v_res_2658_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__6(v_00_u03b2_2655_, v_a_2656_, v_x_2657_);
lean_dec(v_x_2657_);
lean_dec(v_a_2656_);
v_r_2659_ = lean_box(v_res_2658_);
return v_r_2659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7(lean_object* v_00_u03b2_2660_, lean_object* v_data_2661_){
_start:
{
lean_object* v___x_2662_; 
v___x_2662_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7___redArg(v_data_2661_);
return v___x_2662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8(lean_object* v_00_u03b2_2663_, lean_object* v_a_2664_, lean_object* v_b_2665_, lean_object* v_x_2666_){
_start:
{
lean_object* v___x_2667_; 
v___x_2667_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__8___redArg(v_a_2664_, v_b_2665_, v_x_2666_);
return v___x_2667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8(lean_object* v_00_u03b2_2668_, lean_object* v_i_2669_, lean_object* v_source_2670_, lean_object* v_target_2671_){
_start:
{
lean_object* v___x_2672_; 
v___x_2672_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8___redArg(v_i_2669_, v_source_2670_, v_target_2671_);
return v___x_2672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10(lean_object* v_00_u03b2_2673_, lean_object* v_x_2674_, lean_object* v_x_2675_){
_start:
{
lean_object* v___x_2676_; 
v___x_2676_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_Translate_guessReorder_spec__3_spec__4_spec__7_spec__8_spec__10___redArg(v_x_2674_, v_x_2675_);
return v___x_2676_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Category_translateReorder(void){
_start:
{
lean_object* v___x_2720_; 
v___x_2720_ = lean_box(0);
return v___x_2720_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; 
v___x_2797_ = lean_box(0);
v___x_2798_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2799_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2799_, 0, v___x_2798_);
lean_ctor_set(v___x_2799_, 1, v___x_2797_);
return v___x_2799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg(){
_start:
{
lean_object* v___x_2801_; lean_object* v___x_2802_; 
v___x_2801_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___closed__0);
v___x_2802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2802_, 0, v___x_2801_);
return v___x_2802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg___boxed(lean_object* v___y_2803_){
_start:
{
lean_object* v_res_2804_; 
v_res_2804_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
return v_res_2804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0(lean_object* v_00_u03b1_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_){
_start:
{
lean_object* v___x_2811_; 
v___x_2811_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
return v___x_2811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___boxed(lean_object* v_00_u03b1_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_){
_start:
{
lean_object* v_res_2818_; 
v_res_2818_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0(v_00_u03b1_2812_, v___y_2813_, v___y_2814_, v___y_2815_, v___y_2816_);
lean_dec(v___y_2816_);
lean_dec_ref(v___y_2815_);
lean_dec(v___y_2814_);
lean_dec_ref(v___y_2813_);
return v_res_2818_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0(uint8_t v___x_2819_, lean_object* v_x_2820_){
_start:
{
return v___x_2819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0___boxed(lean_object* v___x_2821_, lean_object* v_x_2822_){
_start:
{
uint8_t v___x_2614__boxed_2823_; uint8_t v_res_2824_; lean_object* v_r_2825_; 
v___x_2614__boxed_2823_ = lean_unbox(v___x_2821_);
v_res_2824_ = lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___lam__0(v___x_2614__boxed_2823_, v_x_2822_);
lean_dec(v_x_2822_);
v_r_2825_ = lean_box(v_res_2824_);
return v_r_2825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(lean_object* v_ref_2826_, lean_object* v_msg_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_, lean_object* v___y_2830_, lean_object* v___y_2831_){
_start:
{
lean_object* v_fileName_2833_; lean_object* v_fileMap_2834_; lean_object* v_options_2835_; lean_object* v_currRecDepth_2836_; lean_object* v_maxRecDepth_2837_; lean_object* v_ref_2838_; lean_object* v_currNamespace_2839_; lean_object* v_openDecls_2840_; lean_object* v_initHeartbeats_2841_; lean_object* v_maxHeartbeats_2842_; lean_object* v_quotContext_2843_; lean_object* v_currMacroScope_2844_; uint8_t v_diag_2845_; lean_object* v_cancelTk_x3f_2846_; uint8_t v_suppressElabErrors_2847_; lean_object* v_inheritedTraceOptions_2848_; lean_object* v_ref_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; 
v_fileName_2833_ = lean_ctor_get(v___y_2830_, 0);
v_fileMap_2834_ = lean_ctor_get(v___y_2830_, 1);
v_options_2835_ = lean_ctor_get(v___y_2830_, 2);
v_currRecDepth_2836_ = lean_ctor_get(v___y_2830_, 3);
v_maxRecDepth_2837_ = lean_ctor_get(v___y_2830_, 4);
v_ref_2838_ = lean_ctor_get(v___y_2830_, 5);
v_currNamespace_2839_ = lean_ctor_get(v___y_2830_, 6);
v_openDecls_2840_ = lean_ctor_get(v___y_2830_, 7);
v_initHeartbeats_2841_ = lean_ctor_get(v___y_2830_, 8);
v_maxHeartbeats_2842_ = lean_ctor_get(v___y_2830_, 9);
v_quotContext_2843_ = lean_ctor_get(v___y_2830_, 10);
v_currMacroScope_2844_ = lean_ctor_get(v___y_2830_, 11);
v_diag_2845_ = lean_ctor_get_uint8(v___y_2830_, sizeof(void*)*14);
v_cancelTk_x3f_2846_ = lean_ctor_get(v___y_2830_, 12);
v_suppressElabErrors_2847_ = lean_ctor_get_uint8(v___y_2830_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2848_ = lean_ctor_get(v___y_2830_, 13);
v_ref_2849_ = l_Lean_replaceRef(v_ref_2826_, v_ref_2838_);
lean_inc_ref(v_inheritedTraceOptions_2848_);
lean_inc(v_cancelTk_x3f_2846_);
lean_inc(v_currMacroScope_2844_);
lean_inc(v_quotContext_2843_);
lean_inc(v_maxHeartbeats_2842_);
lean_inc(v_initHeartbeats_2841_);
lean_inc(v_openDecls_2840_);
lean_inc(v_currNamespace_2839_);
lean_inc(v_maxRecDepth_2837_);
lean_inc(v_currRecDepth_2836_);
lean_inc_ref(v_options_2835_);
lean_inc_ref(v_fileMap_2834_);
lean_inc_ref(v_fileName_2833_);
v___x_2850_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2850_, 0, v_fileName_2833_);
lean_ctor_set(v___x_2850_, 1, v_fileMap_2834_);
lean_ctor_set(v___x_2850_, 2, v_options_2835_);
lean_ctor_set(v___x_2850_, 3, v_currRecDepth_2836_);
lean_ctor_set(v___x_2850_, 4, v_maxRecDepth_2837_);
lean_ctor_set(v___x_2850_, 5, v_ref_2849_);
lean_ctor_set(v___x_2850_, 6, v_currNamespace_2839_);
lean_ctor_set(v___x_2850_, 7, v_openDecls_2840_);
lean_ctor_set(v___x_2850_, 8, v_initHeartbeats_2841_);
lean_ctor_set(v___x_2850_, 9, v_maxHeartbeats_2842_);
lean_ctor_set(v___x_2850_, 10, v_quotContext_2843_);
lean_ctor_set(v___x_2850_, 11, v_currMacroScope_2844_);
lean_ctor_set(v___x_2850_, 12, v_cancelTk_x3f_2846_);
lean_ctor_set(v___x_2850_, 13, v_inheritedTraceOptions_2848_);
lean_ctor_set_uint8(v___x_2850_, sizeof(void*)*14, v_diag_2845_);
lean_ctor_set_uint8(v___x_2850_, sizeof(void*)*14 + 1, v_suppressElabErrors_2847_);
v___x_2851_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v_msg_2827_, v___y_2828_, v___y_2829_, v___x_2850_, v___y_2831_);
lean_dec_ref_known(v___x_2850_, 14);
return v___x_2851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg___boxed(lean_object* v_ref_2852_, lean_object* v_msg_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_, lean_object* v___y_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_){
_start:
{
lean_object* v_res_2859_; 
v_res_2859_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_ref_2852_, v_msg_2853_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_);
lean_dec(v___y_2857_);
lean_dec_ref(v___y_2856_);
lean_dec(v___y_2855_);
lean_dec_ref(v___y_2854_);
lean_dec(v_ref_2852_);
return v_res_2859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3(lean_object* v_xs_2860_, lean_object* v_v_2861_, lean_object* v_i_2862_){
_start:
{
lean_object* v___x_2863_; uint8_t v___x_2864_; 
v___x_2863_ = lean_array_get_size(v_xs_2860_);
v___x_2864_ = lean_nat_dec_lt(v_i_2862_, v___x_2863_);
if (v___x_2864_ == 0)
{
lean_object* v___x_2865_; 
lean_dec(v_i_2862_);
v___x_2865_ = lean_box(0);
return v___x_2865_;
}
else
{
lean_object* v___x_2866_; uint8_t v___x_2867_; 
v___x_2866_ = lean_array_fget_borrowed(v_xs_2860_, v_i_2862_);
v___x_2867_ = lean_name_eq(v___x_2866_, v_v_2861_);
if (v___x_2867_ == 0)
{
lean_object* v___x_2868_; lean_object* v___x_2869_; 
v___x_2868_ = lean_unsigned_to_nat(1u);
v___x_2869_ = lean_nat_add(v_i_2862_, v___x_2868_);
lean_dec(v_i_2862_);
v_i_2862_ = v___x_2869_;
goto _start;
}
else
{
lean_object* v___x_2871_; 
v___x_2871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2871_, 0, v_i_2862_);
return v___x_2871_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3___boxed(lean_object* v_xs_2872_, lean_object* v_v_2873_, lean_object* v_i_2874_){
_start:
{
lean_object* v_res_2875_; 
v_res_2875_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3(v_xs_2872_, v_v_2873_, v_i_2874_);
lean_dec(v_v_2873_);
lean_dec_ref(v_xs_2872_);
return v_res_2875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2(lean_object* v_xs_2876_, lean_object* v_v_2877_){
_start:
{
lean_object* v___x_2878_; lean_object* v___x_2879_; 
v___x_2878_ = lean_unsigned_to_nat(0u);
v___x_2879_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2_spec__3(v_xs_2876_, v_v_2877_, v___x_2878_);
return v___x_2879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2___boxed(lean_object* v_xs_2880_, lean_object* v_v_2881_){
_start:
{
lean_object* v_res_2882_; 
v_res_2882_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2(v_xs_2880_, v_v_2881_);
lean_dec(v_v_2881_);
lean_dec_ref(v_xs_2880_);
return v_res_2882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2(lean_object* v_xs_2883_, lean_object* v_v_2884_){
_start:
{
lean_object* v___x_2885_; 
v___x_2885_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2_spec__2(v_xs_2883_, v_v_2884_);
if (lean_obj_tag(v___x_2885_) == 0)
{
lean_object* v___x_2886_; 
v___x_2886_ = lean_box(0);
return v___x_2886_;
}
else
{
lean_object* v_val_2887_; lean_object* v___x_2889_; uint8_t v_isShared_2890_; uint8_t v_isSharedCheck_2894_; 
v_val_2887_ = lean_ctor_get(v___x_2885_, 0);
v_isSharedCheck_2894_ = !lean_is_exclusive(v___x_2885_);
if (v_isSharedCheck_2894_ == 0)
{
v___x_2889_ = v___x_2885_;
v_isShared_2890_ = v_isSharedCheck_2894_;
goto v_resetjp_2888_;
}
else
{
lean_inc(v_val_2887_);
lean_dec(v___x_2885_);
v___x_2889_ = lean_box(0);
v_isShared_2890_ = v_isSharedCheck_2894_;
goto v_resetjp_2888_;
}
v_resetjp_2888_:
{
lean_object* v___x_2892_; 
if (v_isShared_2890_ == 0)
{
v___x_2892_ = v___x_2889_;
goto v_reusejp_2891_;
}
else
{
lean_object* v_reuseFailAlloc_2893_; 
v_reuseFailAlloc_2893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2893_, 0, v_val_2887_);
v___x_2892_ = v_reuseFailAlloc_2893_;
goto v_reusejp_2891_;
}
v_reusejp_2891_:
{
return v___x_2892_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2___boxed(lean_object* v_xs_2895_, lean_object* v_v_2896_){
_start:
{
lean_object* v_res_2897_; 
v_res_2897_ = lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2(v_xs_2895_, v_v_2896_);
lean_dec(v_v_2896_);
lean_dec_ref(v_xs_2895_);
return v_res_2897_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5(void){
_start:
{
lean_object* v___x_2915_; lean_object* v___x_2916_; 
v___x_2915_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__4));
v___x_2916_ = l_Lean_stringToMessageData(v___x_2915_);
return v___x_2916_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7(void){
_start:
{
lean_object* v___x_2918_; lean_object* v___x_2919_; 
v___x_2918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__6));
v___x_2919_ = l_Lean_stringToMessageData(v___x_2918_);
return v___x_2919_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9(void){
_start:
{
lean_object* v___x_2921_; lean_object* v___x_2922_; 
v___x_2921_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__8));
v___x_2922_ = l_Lean_stringToMessageData(v___x_2921_);
return v___x_2922_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11(void){
_start:
{
lean_object* v___x_2924_; lean_object* v___x_2925_; 
v___x_2924_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__10));
v___x_2925_ = l_Lean_stringToMessageData(v___x_2924_);
return v___x_2925_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13(void){
_start:
{
lean_object* v___x_2927_; lean_object* v___x_2928_; 
v___x_2927_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__12));
v___x_2928_ = l_Lean_stringToMessageData(v___x_2927_);
return v___x_2928_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15(void){
_start:
{
lean_object* v___x_2930_; lean_object* v___x_2931_; 
v___x_2930_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__14));
v___x_2931_ = l_Lean_stringToMessageData(v___x_2930_);
return v___x_2931_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17(void){
_start:
{
lean_object* v___x_2933_; lean_object* v___x_2934_; 
v___x_2933_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__16));
v___x_2934_ = l_Lean_stringToMessageData(v___x_2933_);
return v___x_2934_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19(void){
_start:
{
lean_object* v___x_2936_; lean_object* v___x_2937_; 
v___x_2936_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__18));
v___x_2937_ = l_Lean_stringToMessageData(v___x_2936_);
return v___x_2937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx(lean_object* v_stx_2938_, lean_object* v_argNames_2939_, lean_object* v_fvars_2940_, lean_object* v_head_2941_, lean_object* v_a_2942_, lean_object* v_a_2943_, lean_object* v_a_2944_, lean_object* v_a_2945_){
_start:
{
lean_object* v___x_2947_; lean_object* v_n_2949_; lean_object* v___y_2950_; lean_object* v___y_2951_; lean_object* v___y_2952_; lean_object* v___y_2953_; lean_object* v___y_2981_; lean_object* v___y_2982_; lean_object* v___y_2983_; lean_object* v___y_2984_; lean_object* v___y_2989_; lean_object* v___y_2990_; lean_object* v___y_2991_; lean_object* v___y_2992_; lean_object* v___x_3016_; uint8_t v___x_3017_; 
v___x_2947_ = l_Lean_instInhabitedExpr;
v___x_3016_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__10));
lean_inc(v_stx_2938_);
v___x_3017_ = l_Lean_Syntax_isOfKind(v_stx_2938_, v___x_3016_);
if (v___x_3017_ == 0)
{
lean_object* v___x_3018_; uint8_t v___x_3019_; 
lean_dec_ref(v_head_2941_);
v___x_3018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__13));
lean_inc(v_stx_2938_);
v___x_3019_ = l_Lean_Syntax_isOfKind(v_stx_2938_, v___x_3018_);
if (v___x_3019_ == 0)
{
lean_object* v___x_3020_; 
lean_dec(v_stx_2938_);
v___x_3020_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
return v___x_3020_;
}
else
{
lean_object* v___x_3021_; lean_object* v___x_3022_; uint8_t v___x_3023_; 
v___x_3021_ = l_Lean_TSyntax_getNat(v_stx_2938_);
v___x_3022_ = lean_unsigned_to_nat(0u);
v___x_3023_ = lean_nat_dec_eq(v___x_3021_, v___x_3022_);
lean_dec(v___x_3021_);
if (v___x_3023_ == 0)
{
v___y_2989_ = v_a_2942_;
v___y_2990_ = v_a_2943_;
v___y_2991_ = v_a_2944_;
v___y_2992_ = v_a_2945_;
goto v___jp_2988_;
}
else
{
lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v_a_3030_; lean_object* v___x_3032_; uint8_t v_isShared_3033_; uint8_t v_isSharedCheck_3037_; 
v___x_3024_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__11);
lean_inc(v_stx_2938_);
v___x_3025_ = l_Lean_MessageData_ofSyntax(v_stx_2938_);
v___x_3026_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3026_, 0, v___x_3024_);
lean_ctor_set(v___x_3026_, 1, v___x_3025_);
v___x_3027_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__13);
v___x_3028_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3028_, 0, v___x_3026_);
lean_ctor_set(v___x_3028_, 1, v___x_3027_);
v___x_3029_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_stx_2938_, v___x_3028_, v_a_2942_, v_a_2943_, v_a_2944_, v_a_2945_);
lean_dec(v_stx_2938_);
v_a_3030_ = lean_ctor_get(v___x_3029_, 0);
v_isSharedCheck_3037_ = !lean_is_exclusive(v___x_3029_);
if (v_isSharedCheck_3037_ == 0)
{
v___x_3032_ = v___x_3029_;
v_isShared_3033_ = v_isSharedCheck_3037_;
goto v_resetjp_3031_;
}
else
{
lean_inc(v_a_3030_);
lean_dec(v___x_3029_);
v___x_3032_ = lean_box(0);
v_isShared_3033_ = v_isSharedCheck_3037_;
goto v_resetjp_3031_;
}
v_resetjp_3031_:
{
lean_object* v___x_3035_; 
if (v_isShared_3033_ == 0)
{
v___x_3035_ = v___x_3032_;
goto v_reusejp_3034_;
}
else
{
lean_object* v_reuseFailAlloc_3036_; 
v_reuseFailAlloc_3036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3036_, 0, v_a_3030_);
v___x_3035_ = v_reuseFailAlloc_3036_;
goto v_reusejp_3034_;
}
v_reusejp_3034_:
{
return v___x_3035_;
}
}
}
}
}
else
{
lean_object* v___x_3038_; lean_object* v___x_3039_; 
v___x_3038_ = l_Lean_TSyntax_getId(v_stx_2938_);
v___x_3039_ = lp_mathlib_Array_idxOf_x3f___at___00Mathlib_Tactic_Translate_elabArgStx_spec__2(v_argNames_2939_, v___x_3038_);
lean_dec(v___x_3038_);
if (lean_obj_tag(v___x_3039_) == 0)
{
lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; 
v___x_3040_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__15);
lean_inc(v_stx_2938_);
v___x_3041_ = l_Lean_MessageData_ofSyntax(v_stx_2938_);
v___x_3042_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3042_, 0, v___x_3040_);
lean_ctor_set(v___x_3042_, 1, v___x_3041_);
v___x_3043_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__17);
v___x_3044_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3044_, 0, v___x_3042_);
lean_ctor_set(v___x_3044_, 1, v___x_3043_);
v___x_3045_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3045_, 0, v___x_3044_);
lean_ctor_set(v___x_3045_, 1, v_head_2941_);
v___x_3046_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__19);
v___x_3047_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3047_, 0, v___x_3045_);
lean_ctor_set(v___x_3047_, 1, v___x_3046_);
v___x_3048_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_stx_2938_, v___x_3047_, v_a_2942_, v_a_2943_, v_a_2944_, v_a_2945_);
lean_dec(v_stx_2938_);
return v___x_3048_;
}
else
{
lean_object* v_val_3049_; 
lean_dec_ref(v_head_2941_);
v_val_3049_ = lean_ctor_get(v___x_3039_, 0);
lean_inc(v_val_3049_);
lean_dec_ref_known(v___x_3039_, 1);
v_n_2949_ = v_val_3049_;
v___y_2950_ = v_a_2942_;
v___y_2951_ = v_a_2943_;
v___y_2952_ = v_a_2944_;
v___y_2953_ = v_a_2945_;
goto v___jp_2948_;
}
}
v___jp_2948_:
{
lean_object* v___x_2954_; lean_object* v___x_2955_; lean_object* v___x_2956_; uint8_t v___x_2957_; lean_object* v___x_2958_; lean_object* v___x_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2963_; 
v___x_2954_ = lean_array_get_borrowed(v___x_2947_, v_fvars_2940_, v_n_2949_);
v___x_2955_ = lean_box(0);
v___x_2956_ = lean_box(0);
v___x_2957_ = 0;
v___x_2958_ = lean_box(v___x_2957_);
v___x_2959_ = lean_box(v___x_2957_);
lean_inc(v___x_2954_);
v___x_2960_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_addTermInfo_x27___boxed), 14, 7);
lean_closure_set(v___x_2960_, 0, v_stx_2938_);
lean_closure_set(v___x_2960_, 1, v___x_2954_);
lean_closure_set(v___x_2960_, 2, v___x_2955_);
lean_closure_set(v___x_2960_, 3, v___x_2955_);
lean_closure_set(v___x_2960_, 4, v___x_2956_);
lean_closure_set(v___x_2960_, 5, v___x_2958_);
lean_closure_set(v___x_2960_, 6, v___x_2959_);
v___x_2961_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__2));
v___x_2962_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__3));
v___x_2963_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_2960_, v___x_2961_, v___x_2962_, v___y_2950_, v___y_2951_, v___y_2952_, v___y_2953_);
if (lean_obj_tag(v___x_2963_) == 0)
{
lean_object* v___x_2965_; uint8_t v_isShared_2966_; uint8_t v_isSharedCheck_2970_; 
v_isSharedCheck_2970_ = !lean_is_exclusive(v___x_2963_);
if (v_isSharedCheck_2970_ == 0)
{
lean_object* v_unused_2971_; 
v_unused_2971_ = lean_ctor_get(v___x_2963_, 0);
lean_dec(v_unused_2971_);
v___x_2965_ = v___x_2963_;
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
else
{
lean_dec(v___x_2963_);
v___x_2965_ = lean_box(0);
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
v_resetjp_2964_:
{
lean_object* v___x_2968_; 
if (v_isShared_2966_ == 0)
{
lean_ctor_set(v___x_2965_, 0, v_n_2949_);
v___x_2968_ = v___x_2965_;
goto v_reusejp_2967_;
}
else
{
lean_object* v_reuseFailAlloc_2969_; 
v_reuseFailAlloc_2969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2969_, 0, v_n_2949_);
v___x_2968_ = v_reuseFailAlloc_2969_;
goto v_reusejp_2967_;
}
v_reusejp_2967_:
{
return v___x_2968_;
}
}
}
else
{
lean_object* v_a_2972_; lean_object* v___x_2974_; uint8_t v_isShared_2975_; uint8_t v_isSharedCheck_2979_; 
lean_dec(v_n_2949_);
v_a_2972_ = lean_ctor_get(v___x_2963_, 0);
v_isSharedCheck_2979_ = !lean_is_exclusive(v___x_2963_);
if (v_isSharedCheck_2979_ == 0)
{
v___x_2974_ = v___x_2963_;
v_isShared_2975_ = v_isSharedCheck_2979_;
goto v_resetjp_2973_;
}
else
{
lean_inc(v_a_2972_);
lean_dec(v___x_2963_);
v___x_2974_ = lean_box(0);
v_isShared_2975_ = v_isSharedCheck_2979_;
goto v_resetjp_2973_;
}
v_resetjp_2973_:
{
lean_object* v___x_2977_; 
if (v_isShared_2975_ == 0)
{
v___x_2977_ = v___x_2974_;
goto v_reusejp_2976_;
}
else
{
lean_object* v_reuseFailAlloc_2978_; 
v_reuseFailAlloc_2978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2978_, 0, v_a_2972_);
v___x_2977_ = v_reuseFailAlloc_2978_;
goto v_reusejp_2976_;
}
v_reusejp_2976_:
{
return v___x_2977_;
}
}
}
}
v___jp_2980_:
{
lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; 
v___x_2985_ = l_Lean_TSyntax_getNat(v_stx_2938_);
v___x_2986_ = lean_unsigned_to_nat(1u);
v___x_2987_ = lean_nat_sub(v___x_2985_, v___x_2986_);
lean_dec(v___x_2985_);
v_n_2949_ = v___x_2987_;
v___y_2950_ = v___y_2981_;
v___y_2951_ = v___y_2982_;
v___y_2952_ = v___y_2983_;
v___y_2953_ = v___y_2984_;
goto v___jp_2948_;
}
v___jp_2988_:
{
lean_object* v___x_2993_; lean_object* v___x_2994_; uint8_t v___x_2995_; 
v___x_2993_ = lean_array_get_size(v_fvars_2940_);
v___x_2994_ = l_Lean_TSyntax_getNat(v_stx_2938_);
v___x_2995_ = lean_nat_dec_lt(v___x_2993_, v___x_2994_);
lean_dec(v___x_2994_);
if (v___x_2995_ == 0)
{
v___y_2981_ = v___y_2989_;
v___y_2982_ = v___y_2990_;
v___y_2983_ = v___y_2991_;
v___y_2984_ = v___y_2992_;
goto v___jp_2980_;
}
else
{
lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v_a_3008_; lean_object* v___x_3010_; uint8_t v_isShared_3011_; uint8_t v_isSharedCheck_3015_; 
v___x_2996_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__5);
lean_inc(v_stx_2938_);
v___x_2997_ = l_Lean_MessageData_ofSyntax(v_stx_2938_);
v___x_2998_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2998_, 0, v___x_2996_);
lean_ctor_set(v___x_2998_, 1, v___x_2997_);
v___x_2999_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__7);
v___x_3000_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3000_, 0, v___x_2998_);
lean_ctor_set(v___x_3000_, 1, v___x_2999_);
v___x_3001_ = l_Nat_reprFast(v___x_2993_);
v___x_3002_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3002_, 0, v___x_3001_);
v___x_3003_ = l_Lean_MessageData_ofFormat(v___x_3002_);
v___x_3004_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3004_, 0, v___x_3000_);
lean_ctor_set(v___x_3004_, 1, v___x_3003_);
v___x_3005_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9, &lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___closed__9);
v___x_3006_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3006_, 0, v___x_3004_);
lean_ctor_set(v___x_3006_, 1, v___x_3005_);
v___x_3007_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_stx_2938_, v___x_3006_, v___y_2989_, v___y_2990_, v___y_2991_, v___y_2992_);
lean_dec(v_stx_2938_);
v_a_3008_ = lean_ctor_get(v___x_3007_, 0);
v_isSharedCheck_3015_ = !lean_is_exclusive(v___x_3007_);
if (v_isSharedCheck_3015_ == 0)
{
v___x_3010_ = v___x_3007_;
v_isShared_3011_ = v_isSharedCheck_3015_;
goto v_resetjp_3009_;
}
else
{
lean_inc(v_a_3008_);
lean_dec(v___x_3007_);
v___x_3010_ = lean_box(0);
v_isShared_3011_ = v_isSharedCheck_3015_;
goto v_resetjp_3009_;
}
v_resetjp_3009_:
{
lean_object* v___x_3013_; 
if (v_isShared_3011_ == 0)
{
v___x_3013_ = v___x_3010_;
goto v_reusejp_3012_;
}
else
{
lean_object* v_reuseFailAlloc_3014_; 
v_reuseFailAlloc_3014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3014_, 0, v_a_3008_);
v___x_3013_ = v_reuseFailAlloc_3014_;
goto v_reusejp_3012_;
}
v_reusejp_3012_:
{
return v___x_3013_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabArgStx___boxed(lean_object* v_stx_3050_, lean_object* v_argNames_3051_, lean_object* v_fvars_3052_, lean_object* v_head_3053_, lean_object* v_a_3054_, lean_object* v_a_3055_, lean_object* v_a_3056_, lean_object* v_a_3057_, lean_object* v_a_3058_){
_start:
{
lean_object* v_res_3059_; 
v_res_3059_ = lp_mathlib_Mathlib_Tactic_Translate_elabArgStx(v_stx_3050_, v_argNames_3051_, v_fvars_3052_, v_head_3053_, v_a_3054_, v_a_3055_, v_a_3056_, v_a_3057_);
lean_dec(v_a_3057_);
lean_dec_ref(v_a_3056_);
lean_dec(v_a_3055_);
lean_dec_ref(v_a_3054_);
lean_dec_ref(v_fvars_3052_);
lean_dec_ref(v_argNames_3051_);
return v_res_3059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1(lean_object* v_00_u03b1_3060_, lean_object* v_ref_3061_, lean_object* v_msg_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_){
_start:
{
lean_object* v___x_3068_; 
v___x_3068_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_ref_3061_, v_msg_3062_, v___y_3063_, v___y_3064_, v___y_3065_, v___y_3066_);
return v___x_3068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___boxed(lean_object* v_00_u03b1_3069_, lean_object* v_ref_3070_, lean_object* v_msg_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_){
_start:
{
lean_object* v_res_3077_; 
v_res_3077_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1(v_00_u03b1_3069_, v_ref_3070_, v_msg_3071_, v___y_3072_, v___y_3073_, v___y_3074_, v___y_3075_);
lean_dec(v___y_3075_);
lean_dec_ref(v___y_3074_);
lean_dec(v___y_3073_);
lean_dec_ref(v___y_3072_);
lean_dec(v_ref_3070_);
return v_res_3077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg(lean_object* v_type_3078_, lean_object* v_k_3079_, uint8_t v_cleanupAnnotations_3080_, uint8_t v_whnfType_3081_, lean_object* v___y_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_){
_start:
{
lean_object* v___f_3087_; lean_object* v___x_3088_; 
v___f_3087_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Tactic_Translate_guessReorder_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_3087_, 0, v_k_3079_);
v___x_3088_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_3078_, v___f_3087_, v_cleanupAnnotations_3080_, v_whnfType_3081_, v___y_3082_, v___y_3083_, v___y_3084_, v___y_3085_);
if (lean_obj_tag(v___x_3088_) == 0)
{
lean_object* v_a_3089_; lean_object* v___x_3091_; uint8_t v_isShared_3092_; uint8_t v_isSharedCheck_3096_; 
v_a_3089_ = lean_ctor_get(v___x_3088_, 0);
v_isSharedCheck_3096_ = !lean_is_exclusive(v___x_3088_);
if (v_isSharedCheck_3096_ == 0)
{
v___x_3091_ = v___x_3088_;
v_isShared_3092_ = v_isSharedCheck_3096_;
goto v_resetjp_3090_;
}
else
{
lean_inc(v_a_3089_);
lean_dec(v___x_3088_);
v___x_3091_ = lean_box(0);
v_isShared_3092_ = v_isSharedCheck_3096_;
goto v_resetjp_3090_;
}
v_resetjp_3090_:
{
lean_object* v___x_3094_; 
if (v_isShared_3092_ == 0)
{
v___x_3094_ = v___x_3091_;
goto v_reusejp_3093_;
}
else
{
lean_object* v_reuseFailAlloc_3095_; 
v_reuseFailAlloc_3095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3095_, 0, v_a_3089_);
v___x_3094_ = v_reuseFailAlloc_3095_;
goto v_reusejp_3093_;
}
v_reusejp_3093_:
{
return v___x_3094_;
}
}
}
else
{
lean_object* v_a_3097_; lean_object* v___x_3099_; uint8_t v_isShared_3100_; uint8_t v_isSharedCheck_3104_; 
v_a_3097_ = lean_ctor_get(v___x_3088_, 0);
v_isSharedCheck_3104_ = !lean_is_exclusive(v___x_3088_);
if (v_isSharedCheck_3104_ == 0)
{
v___x_3099_ = v___x_3088_;
v_isShared_3100_ = v_isSharedCheck_3104_;
goto v_resetjp_3098_;
}
else
{
lean_inc(v_a_3097_);
lean_dec(v___x_3088_);
v___x_3099_ = lean_box(0);
v_isShared_3100_ = v_isSharedCheck_3104_;
goto v_resetjp_3098_;
}
v_resetjp_3098_:
{
lean_object* v___x_3102_; 
if (v_isShared_3100_ == 0)
{
v___x_3102_ = v___x_3099_;
goto v_reusejp_3101_;
}
else
{
lean_object* v_reuseFailAlloc_3103_; 
v_reuseFailAlloc_3103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3103_, 0, v_a_3097_);
v___x_3102_ = v_reuseFailAlloc_3103_;
goto v_reusejp_3101_;
}
v_reusejp_3101_:
{
return v___x_3102_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg___boxed(lean_object* v_type_3105_, lean_object* v_k_3106_, lean_object* v_cleanupAnnotations_3107_, lean_object* v_whnfType_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_3114_; uint8_t v_whnfType_boxed_3115_; lean_object* v_res_3116_; 
v_cleanupAnnotations_boxed_3114_ = lean_unbox(v_cleanupAnnotations_3107_);
v_whnfType_boxed_3115_ = lean_unbox(v_whnfType_3108_);
v_res_3116_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg(v_type_3105_, v_k_3106_, v_cleanupAnnotations_boxed_3114_, v_whnfType_boxed_3115_, v___y_3109_, v___y_3110_, v___y_3111_, v___y_3112_);
lean_dec(v___y_3112_);
lean_dec_ref(v___y_3111_);
lean_dec(v___y_3110_);
lean_dec_ref(v___y_3109_);
return v_res_3116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6(lean_object* v_00_u03b1_3117_, lean_object* v_type_3118_, lean_object* v_k_3119_, uint8_t v_cleanupAnnotations_3120_, uint8_t v_whnfType_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_){
_start:
{
lean_object* v___x_3127_; 
v___x_3127_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg(v_type_3118_, v_k_3119_, v_cleanupAnnotations_3120_, v_whnfType_3121_, v___y_3122_, v___y_3123_, v___y_3124_, v___y_3125_);
return v___x_3127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___boxed(lean_object* v_00_u03b1_3128_, lean_object* v_type_3129_, lean_object* v_k_3130_, lean_object* v_cleanupAnnotations_3131_, lean_object* v_whnfType_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_3138_; uint8_t v_whnfType_boxed_3139_; lean_object* v_res_3140_; 
v_cleanupAnnotations_boxed_3138_ = lean_unbox(v_cleanupAnnotations_3131_);
v_whnfType_boxed_3139_ = lean_unbox(v_whnfType_3132_);
v_res_3140_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6(v_00_u03b1_3128_, v_type_3129_, v_k_3130_, v_cleanupAnnotations_boxed_3138_, v_whnfType_boxed_3139_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_);
lean_dec(v___y_3136_);
lean_dec_ref(v___y_3135_);
lean_dec(v___y_3134_);
lean_dec_ref(v___y_3133_);
return v_res_3140_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1(void){
_start:
{
lean_object* v___x_3142_; lean_object* v___x_3143_; 
v___x_3142_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__0));
v___x_3143_ = l_Lean_stringToMessageData(v___x_3142_);
return v___x_3143_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3(void){
_start:
{
lean_object* v___x_3145_; lean_object* v___x_3146_; 
v___x_3145_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__2));
v___x_3146_ = l_Lean_stringToMessageData(v___x_3145_);
return v___x_3146_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5(void){
_start:
{
lean_object* v___x_3148_; lean_object* v___x_3149_; 
v___x_3148_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__4));
v___x_3149_ = l_Lean_stringToMessageData(v___x_3148_);
return v___x_3149_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7(void){
_start:
{
lean_object* v___x_3151_; lean_object* v___x_3152_; 
v___x_3151_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__6));
v___x_3152_ = l_Lean_stringToMessageData(v___x_3151_);
return v___x_3152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg(lean_object* v_upperBound_3153_, lean_object* v___y_3154_, lean_object* v_a_3155_, lean_object* v_b_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_){
_start:
{
uint8_t v___x_3162_; 
v___x_3162_ = lean_nat_dec_lt(v_a_3155_, v_upperBound_3153_);
if (v___x_3162_ == 0)
{
lean_object* v___x_3163_; 
lean_dec(v_a_3155_);
v___x_3163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3163_, 0, v_b_3156_);
return v___x_3163_;
}
else
{
lean_object* v___x_3164_; lean_object* v_fst_3165_; lean_object* v_snd_3166_; lean_object* v___x_3168_; uint8_t v_isShared_3169_; uint8_t v_isSharedCheck_3208_; 
v___x_3164_ = lean_array_fget(v___y_3154_, v_a_3155_);
v_fst_3165_ = lean_ctor_get(v___x_3164_, 0);
v_snd_3166_ = lean_ctor_get(v___x_3164_, 1);
v_isSharedCheck_3208_ = !lean_is_exclusive(v___x_3164_);
if (v_isSharedCheck_3208_ == 0)
{
v___x_3168_ = v___x_3164_;
v_isShared_3169_ = v_isSharedCheck_3208_;
goto v_resetjp_3167_;
}
else
{
lean_inc(v_snd_3166_);
lean_inc(v_fst_3165_);
lean_dec(v___x_3164_);
v___x_3168_ = lean_box(0);
v_isShared_3169_ = v_isSharedCheck_3208_;
goto v_resetjp_3167_;
}
v_resetjp_3167_:
{
lean_object* v___x_3170_; lean_object* v_a_3172_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v_fst_3177_; lean_object* v_snd_3178_; lean_object* v___x_3180_; uint8_t v_isShared_3181_; uint8_t v_isSharedCheck_3207_; 
v___x_3170_ = lean_unsigned_to_nat(1u);
v___x_3175_ = lean_nat_add(v_a_3155_, v___x_3170_);
v___x_3176_ = lean_array_fget(v___y_3154_, v___x_3175_);
lean_dec(v___x_3175_);
v_fst_3177_ = lean_ctor_get(v___x_3176_, 0);
v_snd_3178_ = lean_ctor_get(v___x_3176_, 1);
v_isSharedCheck_3207_ = !lean_is_exclusive(v___x_3176_);
if (v_isSharedCheck_3207_ == 0)
{
v___x_3180_ = v___x_3176_;
v_isShared_3181_ = v_isSharedCheck_3207_;
goto v_resetjp_3179_;
}
else
{
lean_inc(v_snd_3178_);
lean_inc(v_fst_3177_);
lean_dec(v___x_3176_);
v___x_3180_ = lean_box(0);
v_isShared_3181_ = v_isSharedCheck_3207_;
goto v_resetjp_3179_;
}
v___jp_3171_:
{
lean_object* v___x_3173_; 
v___x_3173_ = lean_nat_add(v_a_3155_, v___x_3170_);
lean_dec(v_a_3155_);
v_a_3155_ = v___x_3173_;
v_b_3156_ = v_a_3172_;
goto _start;
}
v_resetjp_3179_:
{
lean_object* v___x_3182_; uint8_t v___x_3183_; 
v___x_3182_ = lean_box(0);
v___x_3183_ = lean_nat_dec_eq(v_fst_3165_, v_fst_3177_);
lean_dec(v_fst_3177_);
if (v___x_3183_ == 0)
{
lean_del_object(v___x_3180_);
lean_dec(v_snd_3178_);
lean_del_object(v___x_3168_);
lean_dec(v_snd_3166_);
lean_dec(v_fst_3165_);
v_a_3172_ = v___x_3182_;
goto v___jp_3171_;
}
else
{
lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3190_; 
v___x_3184_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__1);
v___x_3185_ = lean_nat_add(v_fst_3165_, v___x_3170_);
lean_dec(v_fst_3165_);
v___x_3186_ = l_Nat_reprFast(v___x_3185_);
v___x_3187_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3187_, 0, v___x_3186_);
v___x_3188_ = l_Lean_MessageData_ofFormat(v___x_3187_);
if (v_isShared_3181_ == 0)
{
lean_ctor_set_tag(v___x_3180_, 7);
lean_ctor_set(v___x_3180_, 1, v___x_3188_);
lean_ctor_set(v___x_3180_, 0, v___x_3184_);
v___x_3190_ = v___x_3180_;
goto v_reusejp_3189_;
}
else
{
lean_object* v_reuseFailAlloc_3206_; 
v_reuseFailAlloc_3206_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3206_, 0, v___x_3184_);
lean_ctor_set(v_reuseFailAlloc_3206_, 1, v___x_3188_);
v___x_3190_ = v_reuseFailAlloc_3206_;
goto v_reusejp_3189_;
}
v_reusejp_3189_:
{
lean_object* v___x_3191_; lean_object* v___x_3193_; 
v___x_3191_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__3);
if (v_isShared_3169_ == 0)
{
lean_ctor_set_tag(v___x_3168_, 7);
lean_ctor_set(v___x_3168_, 1, v___x_3191_);
lean_ctor_set(v___x_3168_, 0, v___x_3190_);
v___x_3193_ = v___x_3168_;
goto v_reusejp_3192_;
}
else
{
lean_object* v_reuseFailAlloc_3205_; 
v_reuseFailAlloc_3205_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3205_, 0, v___x_3190_);
lean_ctor_set(v_reuseFailAlloc_3205_, 1, v___x_3191_);
v___x_3193_ = v_reuseFailAlloc_3205_;
goto v_reusejp_3192_;
}
v_reusejp_3192_:
{
lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; 
v___x_3194_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_snd_3166_);
v___x_3195_ = l_Lean_stringToMessageData(v___x_3194_);
v___x_3196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3196_, 0, v___x_3193_);
lean_ctor_set(v___x_3196_, 1, v___x_3195_);
v___x_3197_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__5);
v___x_3198_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3198_, 0, v___x_3196_);
lean_ctor_set(v___x_3198_, 1, v___x_3197_);
v___x_3199_ = lp_mathlib_Mathlib_Tactic_Translate_ArgReorder_toString(v_snd_3178_);
v___x_3200_ = l_Lean_stringToMessageData(v___x_3199_);
v___x_3201_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3201_, 0, v___x_3198_);
lean_ctor_set(v___x_3201_, 1, v___x_3200_);
v___x_3202_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___closed__7);
v___x_3203_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3203_, 0, v___x_3201_);
lean_ctor_set(v___x_3203_, 1, v___x_3202_);
v___x_3204_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v___x_3203_, v___y_3157_, v___y_3158_, v___y_3159_, v___y_3160_);
if (lean_obj_tag(v___x_3204_) == 0)
{
lean_dec_ref_known(v___x_3204_, 1);
v_a_3172_ = v___x_3182_;
goto v___jp_3171_;
}
else
{
lean_dec(v_a_3155_);
return v___x_3204_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg___boxed(lean_object* v_upperBound_3209_, lean_object* v___y_3210_, lean_object* v_a_3211_, lean_object* v_b_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_){
_start:
{
lean_object* v_res_3218_; 
v_res_3218_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg(v_upperBound_3209_, v___y_3210_, v_a_3211_, v_b_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
lean_dec(v___y_3216_);
lean_dec_ref(v___y_3215_);
lean_dec(v___y_3214_);
lean_dec_ref(v___y_3213_);
lean_dec_ref(v___y_3210_);
lean_dec(v_upperBound_3209_);
return v_res_3218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12(uint8_t v___x_3219_, lean_object* v_as_3220_, size_t v_i_3221_, size_t v_stop_3222_, lean_object* v_b_3223_){
_start:
{
lean_object* v___y_3225_; uint8_t v___x_3229_; 
v___x_3229_ = lean_usize_dec_eq(v_i_3221_, v_stop_3222_);
if (v___x_3229_ == 0)
{
lean_object* v_fst_3230_; uint8_t v___x_3231_; 
v_fst_3230_ = lean_ctor_get(v_b_3223_, 0);
v___x_3231_ = lean_unbox(v_fst_3230_);
if (v___x_3231_ == 0)
{
lean_object* v_snd_3232_; lean_object* v___x_3234_; uint8_t v_isShared_3235_; uint8_t v_isSharedCheck_3240_; 
v_snd_3232_ = lean_ctor_get(v_b_3223_, 1);
v_isSharedCheck_3240_ = !lean_is_exclusive(v_b_3223_);
if (v_isSharedCheck_3240_ == 0)
{
lean_object* v_unused_3241_; 
v_unused_3241_ = lean_ctor_get(v_b_3223_, 0);
lean_dec(v_unused_3241_);
v___x_3234_ = v_b_3223_;
v_isShared_3235_ = v_isSharedCheck_3240_;
goto v_resetjp_3233_;
}
else
{
lean_inc(v_snd_3232_);
lean_dec(v_b_3223_);
v___x_3234_ = lean_box(0);
v_isShared_3235_ = v_isSharedCheck_3240_;
goto v_resetjp_3233_;
}
v_resetjp_3233_:
{
lean_object* v___x_3236_; lean_object* v___x_3238_; 
v___x_3236_ = lean_box(v___x_3219_);
if (v_isShared_3235_ == 0)
{
lean_ctor_set(v___x_3234_, 0, v___x_3236_);
v___x_3238_ = v___x_3234_;
goto v_reusejp_3237_;
}
else
{
lean_object* v_reuseFailAlloc_3239_; 
v_reuseFailAlloc_3239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3239_, 0, v___x_3236_);
lean_ctor_set(v_reuseFailAlloc_3239_, 1, v_snd_3232_);
v___x_3238_ = v_reuseFailAlloc_3239_;
goto v_reusejp_3237_;
}
v_reusejp_3237_:
{
v___y_3225_ = v___x_3238_;
goto v___jp_3224_;
}
}
}
else
{
lean_object* v_snd_3242_; lean_object* v___x_3244_; uint8_t v_isShared_3245_; uint8_t v_isSharedCheck_3252_; 
v_snd_3242_ = lean_ctor_get(v_b_3223_, 1);
v_isSharedCheck_3252_ = !lean_is_exclusive(v_b_3223_);
if (v_isSharedCheck_3252_ == 0)
{
lean_object* v_unused_3253_; 
v_unused_3253_ = lean_ctor_get(v_b_3223_, 0);
lean_dec(v_unused_3253_);
v___x_3244_ = v_b_3223_;
v_isShared_3245_ = v_isSharedCheck_3252_;
goto v_resetjp_3243_;
}
else
{
lean_inc(v_snd_3242_);
lean_dec(v_b_3223_);
v___x_3244_ = lean_box(0);
v_isShared_3245_ = v_isSharedCheck_3252_;
goto v_resetjp_3243_;
}
v_resetjp_3243_:
{
lean_object* v___x_3246_; lean_object* v___x_3247_; lean_object* v___x_3248_; lean_object* v___x_3250_; 
v___x_3246_ = lean_array_uget_borrowed(v_as_3220_, v_i_3221_);
lean_inc(v___x_3246_);
v___x_3247_ = lean_array_push(v_snd_3242_, v___x_3246_);
v___x_3248_ = lean_box(v___x_3229_);
if (v_isShared_3245_ == 0)
{
lean_ctor_set(v___x_3244_, 1, v___x_3247_);
lean_ctor_set(v___x_3244_, 0, v___x_3248_);
v___x_3250_ = v___x_3244_;
goto v_reusejp_3249_;
}
else
{
lean_object* v_reuseFailAlloc_3251_; 
v_reuseFailAlloc_3251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3251_, 0, v___x_3248_);
lean_ctor_set(v_reuseFailAlloc_3251_, 1, v___x_3247_);
v___x_3250_ = v_reuseFailAlloc_3251_;
goto v_reusejp_3249_;
}
v_reusejp_3249_:
{
v___y_3225_ = v___x_3250_;
goto v___jp_3224_;
}
}
}
}
else
{
return v_b_3223_;
}
v___jp_3224_:
{
size_t v___x_3226_; size_t v___x_3227_; 
v___x_3226_ = ((size_t)1ULL);
v___x_3227_ = lean_usize_add(v_i_3221_, v___x_3226_);
v_i_3221_ = v___x_3227_;
v_b_3223_ = v___y_3225_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12___boxed(lean_object* v___x_3254_, lean_object* v_as_3255_, lean_object* v_i_3256_, lean_object* v_stop_3257_, lean_object* v_b_3258_){
_start:
{
uint8_t v___x_18311__boxed_3259_; size_t v_i_boxed_3260_; size_t v_stop_boxed_3261_; lean_object* v_res_3262_; 
v___x_18311__boxed_3259_ = lean_unbox(v___x_3254_);
v_i_boxed_3260_ = lean_unbox_usize(v_i_3256_);
lean_dec(v_i_3256_);
v_stop_boxed_3261_ = lean_unbox_usize(v_stop_3257_);
lean_dec(v_stop_3257_);
v_res_3262_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12(v___x_18311__boxed_3259_, v_as_3255_, v_i_boxed_3260_, v_stop_boxed_3261_, v_b_3258_);
lean_dec_ref(v_as_3255_);
return v_res_3262_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg(lean_object* v_a_3263_, lean_object* v_x_3264_){
_start:
{
if (lean_obj_tag(v_x_3264_) == 0)
{
uint8_t v___x_3265_; 
v___x_3265_ = 0;
return v___x_3265_;
}
else
{
lean_object* v_key_3266_; lean_object* v_tail_3267_; uint8_t v___x_3268_; 
v_key_3266_ = lean_ctor_get(v_x_3264_, 0);
v_tail_3267_ = lean_ctor_get(v_x_3264_, 2);
v___x_3268_ = lean_nat_dec_eq(v_key_3266_, v_a_3263_);
if (v___x_3268_ == 0)
{
v_x_3264_ = v_tail_3267_;
goto _start;
}
else
{
return v___x_3268_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg___boxed(lean_object* v_a_3270_, lean_object* v_x_3271_){
_start:
{
uint8_t v_res_3272_; lean_object* v_r_3273_; 
v_res_3272_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg(v_a_3270_, v_x_3271_);
lean_dec(v_x_3271_);
lean_dec(v_a_3270_);
v_r_3273_ = lean_box(v_res_3272_);
return v_r_3273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4___redArg(lean_object* v_x_3274_, lean_object* v_x_3275_){
_start:
{
if (lean_obj_tag(v_x_3275_) == 0)
{
return v_x_3274_;
}
else
{
lean_object* v_key_3276_; lean_object* v_value_3277_; lean_object* v_tail_3278_; lean_object* v___x_3280_; uint8_t v_isShared_3281_; uint8_t v_isSharedCheck_3301_; 
v_key_3276_ = lean_ctor_get(v_x_3275_, 0);
v_value_3277_ = lean_ctor_get(v_x_3275_, 1);
v_tail_3278_ = lean_ctor_get(v_x_3275_, 2);
v_isSharedCheck_3301_ = !lean_is_exclusive(v_x_3275_);
if (v_isSharedCheck_3301_ == 0)
{
v___x_3280_ = v_x_3275_;
v_isShared_3281_ = v_isSharedCheck_3301_;
goto v_resetjp_3279_;
}
else
{
lean_inc(v_tail_3278_);
lean_inc(v_value_3277_);
lean_inc(v_key_3276_);
lean_dec(v_x_3275_);
v___x_3280_ = lean_box(0);
v_isShared_3281_ = v_isSharedCheck_3301_;
goto v_resetjp_3279_;
}
v_resetjp_3279_:
{
lean_object* v___x_3282_; uint64_t v___x_3283_; uint64_t v___x_3284_; uint64_t v___x_3285_; uint64_t v_fold_3286_; uint64_t v___x_3287_; uint64_t v___x_3288_; uint64_t v___x_3289_; size_t v___x_3290_; size_t v___x_3291_; size_t v___x_3292_; size_t v___x_3293_; size_t v___x_3294_; lean_object* v___x_3295_; lean_object* v___x_3297_; 
v___x_3282_ = lean_array_get_size(v_x_3274_);
v___x_3283_ = lean_uint64_of_nat(v_key_3276_);
v___x_3284_ = 32ULL;
v___x_3285_ = lean_uint64_shift_right(v___x_3283_, v___x_3284_);
v_fold_3286_ = lean_uint64_xor(v___x_3283_, v___x_3285_);
v___x_3287_ = 16ULL;
v___x_3288_ = lean_uint64_shift_right(v_fold_3286_, v___x_3287_);
v___x_3289_ = lean_uint64_xor(v_fold_3286_, v___x_3288_);
v___x_3290_ = lean_uint64_to_usize(v___x_3289_);
v___x_3291_ = lean_usize_of_nat(v___x_3282_);
v___x_3292_ = ((size_t)1ULL);
v___x_3293_ = lean_usize_sub(v___x_3291_, v___x_3292_);
v___x_3294_ = lean_usize_land(v___x_3290_, v___x_3293_);
v___x_3295_ = lean_array_uget_borrowed(v_x_3274_, v___x_3294_);
lean_inc(v___x_3295_);
if (v_isShared_3281_ == 0)
{
lean_ctor_set(v___x_3280_, 2, v___x_3295_);
v___x_3297_ = v___x_3280_;
goto v_reusejp_3296_;
}
else
{
lean_object* v_reuseFailAlloc_3300_; 
v_reuseFailAlloc_3300_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3300_, 0, v_key_3276_);
lean_ctor_set(v_reuseFailAlloc_3300_, 1, v_value_3277_);
lean_ctor_set(v_reuseFailAlloc_3300_, 2, v___x_3295_);
v___x_3297_ = v_reuseFailAlloc_3300_;
goto v_reusejp_3296_;
}
v_reusejp_3296_:
{
lean_object* v___x_3298_; 
v___x_3298_ = lean_array_uset(v_x_3274_, v___x_3294_, v___x_3297_);
v_x_3274_ = v___x_3298_;
v_x_3275_ = v_tail_3278_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2___redArg(lean_object* v_i_3302_, lean_object* v_source_3303_, lean_object* v_target_3304_){
_start:
{
lean_object* v___x_3305_; uint8_t v___x_3306_; 
v___x_3305_ = lean_array_get_size(v_source_3303_);
v___x_3306_ = lean_nat_dec_lt(v_i_3302_, v___x_3305_);
if (v___x_3306_ == 0)
{
lean_dec_ref(v_source_3303_);
lean_dec(v_i_3302_);
return v_target_3304_;
}
else
{
lean_object* v_es_3307_; lean_object* v___x_3308_; lean_object* v_source_3309_; lean_object* v_target_3310_; lean_object* v___x_3311_; lean_object* v___x_3312_; 
v_es_3307_ = lean_array_fget(v_source_3303_, v_i_3302_);
v___x_3308_ = lean_box(0);
v_source_3309_ = lean_array_fset(v_source_3303_, v_i_3302_, v___x_3308_);
v_target_3310_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4___redArg(v_target_3304_, v_es_3307_);
v___x_3311_ = lean_unsigned_to_nat(1u);
v___x_3312_ = lean_nat_add(v_i_3302_, v___x_3311_);
lean_dec(v_i_3302_);
v_i_3302_ = v___x_3312_;
v_source_3303_ = v_source_3309_;
v_target_3304_ = v_target_3310_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2___redArg(lean_object* v_data_3314_){
_start:
{
lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v_nbuckets_3317_; lean_object* v___x_3318_; lean_object* v___x_3319_; lean_object* v___x_3320_; lean_object* v___x_3321_; 
v___x_3315_ = lean_array_get_size(v_data_3314_);
v___x_3316_ = lean_unsigned_to_nat(2u);
v_nbuckets_3317_ = lean_nat_mul(v___x_3315_, v___x_3316_);
v___x_3318_ = lean_unsigned_to_nat(0u);
v___x_3319_ = lean_box(0);
v___x_3320_ = lean_mk_array(v_nbuckets_3317_, v___x_3319_);
v___x_3321_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2___redArg(v___x_3318_, v_data_3314_, v___x_3320_);
return v___x_3321_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1(void){
_start:
{
lean_object* v___x_3323_; lean_object* v___x_3324_; 
v___x_3323_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__0));
v___x_3324_ = l_Lean_stringToMessageData(v___x_3323_);
return v___x_3324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg(lean_object* v_a_3325_, lean_object* v_b_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_){
_start:
{
lean_object* v_it_u2082_3332_; 
v_it_u2082_3332_ = lean_ctor_get(v_a_3325_, 1);
lean_inc(v_it_u2082_3332_);
if (lean_obj_tag(v_it_u2082_3332_) == 0)
{
lean_object* v_it_u2081_3333_; lean_object* v___x_3335_; uint8_t v_isShared_3336_; uint8_t v_isSharedCheck_3345_; 
v_it_u2081_3333_ = lean_ctor_get(v_a_3325_, 0);
v_isSharedCheck_3345_ = !lean_is_exclusive(v_a_3325_);
if (v_isSharedCheck_3345_ == 0)
{
lean_object* v_unused_3346_; 
v_unused_3346_ = lean_ctor_get(v_a_3325_, 1);
lean_dec(v_unused_3346_);
v___x_3335_ = v_a_3325_;
v_isShared_3336_ = v_isSharedCheck_3345_;
goto v_resetjp_3334_;
}
else
{
lean_inc(v_it_u2081_3333_);
lean_dec(v_a_3325_);
v___x_3335_ = lean_box(0);
v_isShared_3336_ = v_isSharedCheck_3345_;
goto v_resetjp_3334_;
}
v_resetjp_3334_:
{
if (lean_obj_tag(v_it_u2081_3333_) == 0)
{
lean_object* v___x_3337_; 
lean_del_object(v___x_3335_);
v___x_3337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3337_, 0, v_b_3326_);
return v___x_3337_;
}
else
{
lean_object* v_head_3338_; lean_object* v_tail_3339_; lean_object* v___x_3340_; lean_object* v___x_3342_; 
v_head_3338_ = lean_ctor_get(v_it_u2081_3333_, 0);
lean_inc(v_head_3338_);
v_tail_3339_ = lean_ctor_get(v_it_u2081_3333_, 1);
lean_inc(v_tail_3339_);
lean_dec_ref_known(v_it_u2081_3333_, 2);
v___x_3340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3340_, 0, v_head_3338_);
if (v_isShared_3336_ == 0)
{
lean_ctor_set(v___x_3335_, 1, v___x_3340_);
lean_ctor_set(v___x_3335_, 0, v_tail_3339_);
v___x_3342_ = v___x_3335_;
goto v_reusejp_3341_;
}
else
{
lean_object* v_reuseFailAlloc_3344_; 
v_reuseFailAlloc_3344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3344_, 0, v_tail_3339_);
lean_ctor_set(v_reuseFailAlloc_3344_, 1, v___x_3340_);
v___x_3342_ = v_reuseFailAlloc_3344_;
goto v_reusejp_3341_;
}
v_reusejp_3341_:
{
v_a_3325_ = v___x_3342_;
goto _start;
}
}
}
}
else
{
lean_object* v_val_3347_; lean_object* v___x_3349_; uint8_t v_isShared_3350_; uint8_t v_isSharedCheck_3436_; 
v_val_3347_ = lean_ctor_get(v_it_u2082_3332_, 0);
v_isSharedCheck_3436_ = !lean_is_exclusive(v_it_u2082_3332_);
if (v_isSharedCheck_3436_ == 0)
{
v___x_3349_ = v_it_u2082_3332_;
v_isShared_3350_ = v_isSharedCheck_3436_;
goto v_resetjp_3348_;
}
else
{
lean_inc(v_val_3347_);
lean_dec(v_it_u2082_3332_);
v___x_3349_ = lean_box(0);
v_isShared_3350_ = v_isSharedCheck_3436_;
goto v_resetjp_3348_;
}
v_resetjp_3348_:
{
if (lean_obj_tag(v_val_3347_) == 0)
{
lean_object* v_it_u2081_3351_; lean_object* v___x_3353_; uint8_t v_isShared_3354_; uint8_t v_isSharedCheck_3360_; 
lean_del_object(v___x_3349_);
v_it_u2081_3351_ = lean_ctor_get(v_a_3325_, 0);
v_isSharedCheck_3360_ = !lean_is_exclusive(v_a_3325_);
if (v_isSharedCheck_3360_ == 0)
{
lean_object* v_unused_3361_; 
v_unused_3361_ = lean_ctor_get(v_a_3325_, 1);
lean_dec(v_unused_3361_);
v___x_3353_ = v_a_3325_;
v_isShared_3354_ = v_isSharedCheck_3360_;
goto v_resetjp_3352_;
}
else
{
lean_inc(v_it_u2081_3351_);
lean_dec(v_a_3325_);
v___x_3353_ = lean_box(0);
v_isShared_3354_ = v_isSharedCheck_3360_;
goto v_resetjp_3352_;
}
v_resetjp_3352_:
{
lean_object* v___x_3355_; lean_object* v___x_3357_; 
v___x_3355_ = lean_box(0);
if (v_isShared_3354_ == 0)
{
lean_ctor_set(v___x_3353_, 1, v___x_3355_);
v___x_3357_ = v___x_3353_;
goto v_reusejp_3356_;
}
else
{
lean_object* v_reuseFailAlloc_3359_; 
v_reuseFailAlloc_3359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3359_, 0, v_it_u2081_3351_);
lean_ctor_set(v_reuseFailAlloc_3359_, 1, v___x_3355_);
v___x_3357_ = v_reuseFailAlloc_3359_;
goto v_reusejp_3356_;
}
v_reusejp_3356_:
{
v_a_3325_ = v___x_3357_;
goto _start;
}
}
}
else
{
lean_object* v_it_u2081_3362_; lean_object* v___x_3364_; uint8_t v_isShared_3365_; uint8_t v_isSharedCheck_3434_; 
v_it_u2081_3362_ = lean_ctor_get(v_a_3325_, 0);
v_isSharedCheck_3434_ = !lean_is_exclusive(v_a_3325_);
if (v_isSharedCheck_3434_ == 0)
{
lean_object* v_unused_3435_; 
v_unused_3435_ = lean_ctor_get(v_a_3325_, 1);
lean_dec(v_unused_3435_);
v___x_3364_ = v_a_3325_;
v_isShared_3365_ = v_isSharedCheck_3434_;
goto v_resetjp_3363_;
}
else
{
lean_inc(v_it_u2081_3362_);
lean_dec(v_a_3325_);
v___x_3364_ = lean_box(0);
v_isShared_3365_ = v_isSharedCheck_3434_;
goto v_resetjp_3363_;
}
v_resetjp_3363_:
{
lean_object* v_head_3366_; lean_object* v_tail_3367_; lean_object* v_size_3368_; lean_object* v_buckets_3369_; lean_object* v___x_3371_; 
v_head_3366_ = lean_ctor_get(v_val_3347_, 0);
lean_inc(v_head_3366_);
v_tail_3367_ = lean_ctor_get(v_val_3347_, 1);
lean_inc(v_tail_3367_);
lean_dec_ref_known(v_val_3347_, 2);
v_size_3368_ = lean_ctor_get(v_b_3326_, 0);
v_buckets_3369_ = lean_ctor_get(v_b_3326_, 1);
if (v_isShared_3350_ == 0)
{
lean_ctor_set(v___x_3349_, 0, v_tail_3367_);
v___x_3371_ = v___x_3349_;
goto v_reusejp_3370_;
}
else
{
lean_object* v_reuseFailAlloc_3433_; 
v_reuseFailAlloc_3433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3433_, 0, v_tail_3367_);
v___x_3371_ = v_reuseFailAlloc_3433_;
goto v_reusejp_3370_;
}
v_reusejp_3370_:
{
lean_object* v___x_3373_; 
if (v_isShared_3365_ == 0)
{
lean_ctor_set(v___x_3364_, 1, v___x_3371_);
v___x_3373_ = v___x_3364_;
goto v_reusejp_3372_;
}
else
{
lean_object* v_reuseFailAlloc_3432_; 
v_reuseFailAlloc_3432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3432_, 0, v_it_u2081_3362_);
lean_ctor_set(v_reuseFailAlloc_3432_, 1, v___x_3371_);
v___x_3373_ = v_reuseFailAlloc_3432_;
goto v_reusejp_3372_;
}
v_reusejp_3372_:
{
lean_object* v_fst_3375_; lean_object* v_snd_3376_; lean_object* v___x_3390_; uint64_t v___x_3391_; uint64_t v___x_3392_; uint64_t v___x_3393_; uint64_t v_fold_3394_; uint64_t v___x_3395_; uint64_t v___x_3396_; uint64_t v___x_3397_; size_t v___x_3398_; size_t v___x_3399_; size_t v___x_3400_; size_t v___x_3401_; size_t v___x_3402_; lean_object* v_bkt_3403_; uint8_t v___x_3404_; 
v___x_3390_ = lean_array_get_size(v_buckets_3369_);
v___x_3391_ = lean_uint64_of_nat(v_head_3366_);
v___x_3392_ = 32ULL;
v___x_3393_ = lean_uint64_shift_right(v___x_3391_, v___x_3392_);
v_fold_3394_ = lean_uint64_xor(v___x_3391_, v___x_3393_);
v___x_3395_ = 16ULL;
v___x_3396_ = lean_uint64_shift_right(v_fold_3394_, v___x_3395_);
v___x_3397_ = lean_uint64_xor(v_fold_3394_, v___x_3396_);
v___x_3398_ = lean_uint64_to_usize(v___x_3397_);
v___x_3399_ = lean_usize_of_nat(v___x_3390_);
v___x_3400_ = ((size_t)1ULL);
v___x_3401_ = lean_usize_sub(v___x_3399_, v___x_3400_);
v___x_3402_ = lean_usize_land(v___x_3398_, v___x_3401_);
v_bkt_3403_ = lean_array_uget_borrowed(v_buckets_3369_, v___x_3402_);
v___x_3404_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg(v_head_3366_, v_bkt_3403_);
if (v___x_3404_ == 0)
{
lean_object* v___x_3406_; uint8_t v_isShared_3407_; uint8_t v_isSharedCheck_3428_; 
lean_inc_ref(v_buckets_3369_);
lean_inc(v_size_3368_);
v_isSharedCheck_3428_ = !lean_is_exclusive(v_b_3326_);
if (v_isSharedCheck_3428_ == 0)
{
lean_object* v_unused_3429_; lean_object* v_unused_3430_; 
v_unused_3429_ = lean_ctor_get(v_b_3326_, 1);
lean_dec(v_unused_3429_);
v_unused_3430_ = lean_ctor_get(v_b_3326_, 0);
lean_dec(v_unused_3430_);
v___x_3406_ = v_b_3326_;
v_isShared_3407_ = v_isSharedCheck_3428_;
goto v_resetjp_3405_;
}
else
{
lean_dec(v_b_3326_);
v___x_3406_ = lean_box(0);
v_isShared_3407_ = v_isSharedCheck_3428_;
goto v_resetjp_3405_;
}
v_resetjp_3405_:
{
lean_object* v___x_3408_; lean_object* v___x_3409_; lean_object* v_size_x27_3410_; lean_object* v___x_3411_; lean_object* v_buckets_x27_3412_; lean_object* v___x_3413_; lean_object* v___x_3414_; lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v___x_3417_; uint8_t v___x_3418_; 
v___x_3408_ = lean_box(0);
v___x_3409_ = lean_unsigned_to_nat(1u);
v_size_x27_3410_ = lean_nat_add(v_size_3368_, v___x_3409_);
lean_dec(v_size_3368_);
lean_inc(v_bkt_3403_);
v___x_3411_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3411_, 0, v_head_3366_);
lean_ctor_set(v___x_3411_, 1, v___x_3408_);
lean_ctor_set(v___x_3411_, 2, v_bkt_3403_);
v_buckets_x27_3412_ = lean_array_uset(v_buckets_3369_, v___x_3402_, v___x_3411_);
v___x_3413_ = lean_unsigned_to_nat(4u);
v___x_3414_ = lean_nat_mul(v_size_x27_3410_, v___x_3413_);
v___x_3415_ = lean_unsigned_to_nat(3u);
v___x_3416_ = lean_nat_div(v___x_3414_, v___x_3415_);
lean_dec(v___x_3414_);
v___x_3417_ = lean_array_get_size(v_buckets_x27_3412_);
v___x_3418_ = lean_nat_dec_le(v___x_3416_, v___x_3417_);
lean_dec(v___x_3416_);
if (v___x_3418_ == 0)
{
lean_object* v_val_3419_; lean_object* v___x_3421_; 
v_val_3419_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2___redArg(v_buckets_x27_3412_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 1, v_val_3419_);
lean_ctor_set(v___x_3406_, 0, v_size_x27_3410_);
v___x_3421_ = v___x_3406_;
goto v_reusejp_3420_;
}
else
{
lean_object* v_reuseFailAlloc_3423_; 
v_reuseFailAlloc_3423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3423_, 0, v_size_x27_3410_);
lean_ctor_set(v_reuseFailAlloc_3423_, 1, v_val_3419_);
v___x_3421_ = v_reuseFailAlloc_3423_;
goto v_reusejp_3420_;
}
v_reusejp_3420_:
{
lean_object* v___x_3422_; 
v___x_3422_ = lean_box(v___x_3404_);
v_fst_3375_ = v___x_3422_;
v_snd_3376_ = v___x_3421_;
goto v___jp_3374_;
}
}
else
{
lean_object* v___x_3425_; 
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 1, v_buckets_x27_3412_);
lean_ctor_set(v___x_3406_, 0, v_size_x27_3410_);
v___x_3425_ = v___x_3406_;
goto v_reusejp_3424_;
}
else
{
lean_object* v_reuseFailAlloc_3427_; 
v_reuseFailAlloc_3427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3427_, 0, v_size_x27_3410_);
lean_ctor_set(v_reuseFailAlloc_3427_, 1, v_buckets_x27_3412_);
v___x_3425_ = v_reuseFailAlloc_3427_;
goto v_reusejp_3424_;
}
v_reusejp_3424_:
{
lean_object* v___x_3426_; 
v___x_3426_ = lean_box(v___x_3404_);
v_fst_3375_ = v___x_3426_;
v_snd_3376_ = v___x_3425_;
goto v___jp_3374_;
}
}
}
}
else
{
lean_object* v___x_3431_; 
lean_dec(v_head_3366_);
v___x_3431_ = lean_box(v___x_3404_);
v_fst_3375_ = v___x_3431_;
v_snd_3376_ = v_b_3326_;
goto v___jp_3374_;
}
v___jp_3374_:
{
uint8_t v___x_3377_; 
v___x_3377_ = lean_unbox(v_fst_3375_);
lean_dec(v_fst_3375_);
if (v___x_3377_ == 0)
{
v_a_3325_ = v___x_3373_;
v_b_3326_ = v_snd_3376_;
goto _start;
}
else
{
lean_object* v___x_3379_; lean_object* v___x_3380_; 
v___x_3379_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___closed__1);
v___x_3380_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Translate_reorderLambda_spec__3___redArg(v___x_3379_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
if (lean_obj_tag(v___x_3380_) == 0)
{
lean_dec_ref_known(v___x_3380_, 1);
v_a_3325_ = v___x_3373_;
v_b_3326_ = v_snd_3376_;
goto _start;
}
else
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3389_; 
lean_dec(v_snd_3376_);
lean_dec_ref(v___x_3373_);
v_a_3382_ = lean_ctor_get(v___x_3380_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3380_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3384_ = v___x_3380_;
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3380_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
lean_object* v___x_3387_; 
if (v_isShared_3385_ == 0)
{
v___x_3387_ = v___x_3384_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v_a_3382_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg___boxed(lean_object* v_a_3437_, lean_object* v_b_3438_, lean_object* v___y_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_, lean_object* v___y_3442_, lean_object* v___y_3443_){
_start:
{
lean_object* v_res_3444_; 
v_res_3444_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg(v_a_3437_, v_b_3438_, v___y_3439_, v___y_3440_, v___y_3441_, v___y_3442_);
lean_dec(v___y_3442_);
lean_dec_ref(v___y_3441_);
lean_dec(v___y_3440_);
lean_dec_ref(v___y_3439_);
return v_res_3444_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(lean_object* v_x1_3445_, lean_object* v_x2_3446_){
_start:
{
lean_object* v_fst_3447_; lean_object* v_fst_3448_; uint8_t v___x_3449_; 
v_fst_3447_ = lean_ctor_get(v_x1_3445_, 0);
v_fst_3448_ = lean_ctor_get(v_x2_3446_, 0);
v___x_3449_ = lean_nat_dec_lt(v_fst_3447_, v_fst_3448_);
return v___x_3449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0___boxed(lean_object* v_x1_3450_, lean_object* v_x2_3451_){
_start:
{
uint8_t v_res_3452_; lean_object* v_r_3453_; 
v_res_3452_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(v_x1_3450_, v_x2_3451_);
lean_dec_ref(v_x2_3451_);
lean_dec_ref(v_x1_3450_);
v_r_3453_ = lean_box(v_res_3452_);
return v_r_3453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg(lean_object* v_hi_3454_, lean_object* v_pivot_3455_, lean_object* v_as_3456_, lean_object* v_i_3457_, lean_object* v_k_3458_){
_start:
{
uint8_t v___x_3459_; 
v___x_3459_ = lean_nat_dec_lt(v_k_3458_, v_hi_3454_);
if (v___x_3459_ == 0)
{
lean_object* v___x_3460_; lean_object* v___x_3461_; 
lean_dec(v_k_3458_);
v___x_3460_ = lean_array_fswap(v_as_3456_, v_i_3457_, v_hi_3454_);
v___x_3461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3461_, 0, v_i_3457_);
lean_ctor_set(v___x_3461_, 1, v___x_3460_);
return v___x_3461_;
}
else
{
lean_object* v___x_3462_; lean_object* v_fst_3463_; lean_object* v_fst_3464_; uint8_t v___x_3465_; 
v___x_3462_ = lean_array_fget_borrowed(v_as_3456_, v_k_3458_);
v_fst_3463_ = lean_ctor_get(v___x_3462_, 0);
v_fst_3464_ = lean_ctor_get(v_pivot_3455_, 0);
v___x_3465_ = lean_nat_dec_lt(v_fst_3463_, v_fst_3464_);
if (v___x_3465_ == 0)
{
lean_object* v___x_3466_; lean_object* v___x_3467_; 
v___x_3466_ = lean_unsigned_to_nat(1u);
v___x_3467_ = lean_nat_add(v_k_3458_, v___x_3466_);
lean_dec(v_k_3458_);
v_k_3458_ = v___x_3467_;
goto _start;
}
else
{
lean_object* v___x_3469_; lean_object* v___x_3470_; lean_object* v___x_3471_; lean_object* v___x_3472_; 
v___x_3469_ = lean_array_fswap(v_as_3456_, v_i_3457_, v_k_3458_);
v___x_3470_ = lean_unsigned_to_nat(1u);
v___x_3471_ = lean_nat_add(v_i_3457_, v___x_3470_);
lean_dec(v_i_3457_);
v___x_3472_ = lean_nat_add(v_k_3458_, v___x_3470_);
lean_dec(v_k_3458_);
v_as_3456_ = v___x_3469_;
v_i_3457_ = v___x_3471_;
v_k_3458_ = v___x_3472_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg___boxed(lean_object* v_hi_3474_, lean_object* v_pivot_3475_, lean_object* v_as_3476_, lean_object* v_i_3477_, lean_object* v_k_3478_){
_start:
{
lean_object* v_res_3479_; 
v_res_3479_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg(v_hi_3474_, v_pivot_3475_, v_as_3476_, v_i_3477_, v_k_3478_);
lean_dec_ref(v_pivot_3475_);
lean_dec(v_hi_3474_);
return v_res_3479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(lean_object* v_n_3480_, lean_object* v_as_3481_, lean_object* v_lo_3482_, lean_object* v_hi_3483_){
_start:
{
lean_object* v___y_3485_; uint8_t v___x_3495_; 
v___x_3495_ = lean_nat_dec_lt(v_lo_3482_, v_hi_3483_);
if (v___x_3495_ == 0)
{
lean_dec(v_lo_3482_);
return v_as_3481_;
}
else
{
lean_object* v___x_3496_; lean_object* v___x_3497_; lean_object* v_mid_3498_; lean_object* v___y_3500_; lean_object* v___y_3506_; lean_object* v___x_3511_; lean_object* v___x_3512_; uint8_t v___x_3513_; 
v___x_3496_ = lean_nat_add(v_lo_3482_, v_hi_3483_);
v___x_3497_ = lean_unsigned_to_nat(1u);
v_mid_3498_ = lean_nat_shiftr(v___x_3496_, v___x_3497_);
lean_dec(v___x_3496_);
v___x_3511_ = lean_array_fget_borrowed(v_as_3481_, v_mid_3498_);
v___x_3512_ = lean_array_fget_borrowed(v_as_3481_, v_lo_3482_);
v___x_3513_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(v___x_3511_, v___x_3512_);
if (v___x_3513_ == 0)
{
v___y_3506_ = v_as_3481_;
goto v___jp_3505_;
}
else
{
lean_object* v___x_3514_; 
v___x_3514_ = lean_array_fswap(v_as_3481_, v_lo_3482_, v_mid_3498_);
v___y_3506_ = v___x_3514_;
goto v___jp_3505_;
}
v___jp_3499_:
{
lean_object* v___x_3501_; lean_object* v___x_3502_; uint8_t v___x_3503_; 
v___x_3501_ = lean_array_fget_borrowed(v___y_3500_, v_mid_3498_);
v___x_3502_ = lean_array_fget_borrowed(v___y_3500_, v_hi_3483_);
v___x_3503_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(v___x_3501_, v___x_3502_);
if (v___x_3503_ == 0)
{
lean_dec(v_mid_3498_);
v___y_3485_ = v___y_3500_;
goto v___jp_3484_;
}
else
{
lean_object* v___x_3504_; 
v___x_3504_ = lean_array_fswap(v___y_3500_, v_mid_3498_, v_hi_3483_);
lean_dec(v_mid_3498_);
v___y_3485_ = v___x_3504_;
goto v___jp_3484_;
}
}
v___jp_3505_:
{
lean_object* v___x_3507_; lean_object* v___x_3508_; uint8_t v___x_3509_; 
v___x_3507_ = lean_array_fget_borrowed(v___y_3506_, v_hi_3483_);
v___x_3508_ = lean_array_fget_borrowed(v___y_3506_, v_lo_3482_);
v___x_3509_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___lam__0(v___x_3507_, v___x_3508_);
if (v___x_3509_ == 0)
{
v___y_3500_ = v___y_3506_;
goto v___jp_3499_;
}
else
{
lean_object* v___x_3510_; 
v___x_3510_ = lean_array_fswap(v___y_3506_, v_lo_3482_, v_hi_3483_);
v___y_3500_ = v___x_3510_;
goto v___jp_3499_;
}
}
}
v___jp_3484_:
{
lean_object* v_pivot_3486_; lean_object* v___x_3487_; lean_object* v_fst_3488_; lean_object* v_snd_3489_; uint8_t v___x_3490_; 
v_pivot_3486_ = lean_array_fget(v___y_3485_, v_hi_3483_);
lean_inc_n(v_lo_3482_, 2);
v___x_3487_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg(v_hi_3483_, v_pivot_3486_, v___y_3485_, v_lo_3482_, v_lo_3482_);
lean_dec(v_pivot_3486_);
v_fst_3488_ = lean_ctor_get(v___x_3487_, 0);
lean_inc(v_fst_3488_);
v_snd_3489_ = lean_ctor_get(v___x_3487_, 1);
lean_inc(v_snd_3489_);
lean_dec_ref(v___x_3487_);
v___x_3490_ = lean_nat_dec_le(v_hi_3483_, v_fst_3488_);
if (v___x_3490_ == 0)
{
lean_object* v___x_3491_; lean_object* v___x_3492_; lean_object* v___x_3493_; 
v___x_3491_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(v_n_3480_, v_snd_3489_, v_lo_3482_, v_fst_3488_);
v___x_3492_ = lean_unsigned_to_nat(1u);
v___x_3493_ = lean_nat_add(v_fst_3488_, v___x_3492_);
lean_dec(v_fst_3488_);
v_as_3481_ = v___x_3491_;
v_lo_3482_ = v___x_3493_;
goto _start;
}
else
{
lean_dec(v_fst_3488_);
lean_dec(v_lo_3482_);
return v_snd_3489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg___boxed(lean_object* v_n_3515_, lean_object* v_as_3516_, lean_object* v_lo_3517_, lean_object* v_hi_3518_){
_start:
{
lean_object* v_res_3519_; 
v_res_3519_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(v_n_3515_, v_as_3516_, v_lo_3517_, v_hi_3518_);
lean_dec(v_hi_3518_);
lean_dec(v_n_3515_);
return v_res_3519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0(size_t v_sz_3520_, size_t v_i_3521_, lean_object* v_bs_3522_){
_start:
{
uint8_t v___x_3523_; 
v___x_3523_ = lean_usize_dec_lt(v_i_3521_, v_sz_3520_);
if (v___x_3523_ == 0)
{
lean_object* v___x_3524_; 
v___x_3524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3524_, 0, v_bs_3522_);
return v___x_3524_;
}
else
{
lean_object* v_v_3525_; lean_object* v___x_3526_; lean_object* v_bs_x27_3527_; size_t v___x_3528_; size_t v___x_3529_; lean_object* v___x_3530_; 
v_v_3525_ = lean_array_uget(v_bs_3522_, v_i_3521_);
v___x_3526_ = lean_unsigned_to_nat(0u);
v_bs_x27_3527_ = lean_array_uset(v_bs_3522_, v_i_3521_, v___x_3526_);
v___x_3528_ = ((size_t)1ULL);
v___x_3529_ = lean_usize_add(v_i_3521_, v___x_3528_);
v___x_3530_ = lean_array_uset(v_bs_x27_3527_, v_i_3521_, v_v_3525_);
v_i_3521_ = v___x_3529_;
v_bs_3522_ = v___x_3530_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0___boxed(lean_object* v_sz_3532_, lean_object* v_i_3533_, lean_object* v_bs_3534_){
_start:
{
size_t v_sz_boxed_3535_; size_t v_i_boxed_3536_; lean_object* v_res_3537_; 
v_sz_boxed_3535_ = lean_unbox_usize(v_sz_3532_);
lean_dec(v_sz_3532_);
v_i_boxed_3536_ = lean_unbox_usize(v_i_3533_);
lean_dec(v_i_3533_);
v_res_3537_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0(v_sz_boxed_3535_, v_i_boxed_3536_, v_bs_3534_);
return v_res_3537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg(size_t v_sz_3538_, size_t v_i_3539_, lean_object* v_bs_3540_, lean_object* v___y_3541_, lean_object* v___y_3542_, lean_object* v___y_3543_){
_start:
{
uint8_t v___x_3545_; 
v___x_3545_ = lean_usize_dec_lt(v_i_3539_, v_sz_3538_);
if (v___x_3545_ == 0)
{
lean_object* v___x_3546_; 
v___x_3546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3546_, 0, v_bs_3540_);
return v___x_3546_;
}
else
{
lean_object* v_v_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; 
v_v_3547_ = lean_array_uget_borrowed(v_bs_3540_, v_i_3539_);
v___x_3548_ = l_Lean_Expr_fvarId_x21(v_v_3547_);
v___x_3549_ = l_Lean_FVarId_getUserName___redArg(v___x_3548_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3549_) == 0)
{
lean_object* v_a_3550_; lean_object* v___x_3551_; lean_object* v_bs_x27_3552_; size_t v___x_3553_; size_t v___x_3554_; lean_object* v___x_3555_; 
v_a_3550_ = lean_ctor_get(v___x_3549_, 0);
lean_inc(v_a_3550_);
lean_dec_ref_known(v___x_3549_, 1);
v___x_3551_ = lean_unsigned_to_nat(0u);
v_bs_x27_3552_ = lean_array_uset(v_bs_3540_, v_i_3539_, v___x_3551_);
v___x_3553_ = ((size_t)1ULL);
v___x_3554_ = lean_usize_add(v_i_3539_, v___x_3553_);
v___x_3555_ = lean_array_uset(v_bs_x27_3552_, v_i_3539_, v_a_3550_);
v_i_3539_ = v___x_3554_;
v_bs_3540_ = v___x_3555_;
goto _start;
}
else
{
lean_object* v_a_3557_; lean_object* v___x_3559_; uint8_t v_isShared_3560_; uint8_t v_isSharedCheck_3564_; 
lean_dec_ref(v_bs_3540_);
v_a_3557_ = lean_ctor_get(v___x_3549_, 0);
v_isSharedCheck_3564_ = !lean_is_exclusive(v___x_3549_);
if (v_isSharedCheck_3564_ == 0)
{
v___x_3559_ = v___x_3549_;
v_isShared_3560_ = v_isSharedCheck_3564_;
goto v_resetjp_3558_;
}
else
{
lean_inc(v_a_3557_);
lean_dec(v___x_3549_);
v___x_3559_ = lean_box(0);
v_isShared_3560_ = v_isSharedCheck_3564_;
goto v_resetjp_3558_;
}
v_resetjp_3558_:
{
lean_object* v___x_3562_; 
if (v_isShared_3560_ == 0)
{
v___x_3562_ = v___x_3559_;
goto v_reusejp_3561_;
}
else
{
lean_object* v_reuseFailAlloc_3563_; 
v_reuseFailAlloc_3563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3563_, 0, v_a_3557_);
v___x_3562_ = v_reuseFailAlloc_3563_;
goto v_reusejp_3561_;
}
v_reusejp_3561_:
{
return v___x_3562_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg___boxed(lean_object* v_sz_3565_, lean_object* v_i_3566_, lean_object* v_bs_3567_, lean_object* v___y_3568_, lean_object* v___y_3569_, lean_object* v___y_3570_, lean_object* v___y_3571_){
_start:
{
size_t v_sz_boxed_3572_; size_t v_i_boxed_3573_; lean_object* v_res_3574_; 
v_sz_boxed_3572_ = lean_unbox_usize(v_sz_3565_);
lean_dec(v_sz_3565_);
v_i_boxed_3573_ = lean_unbox_usize(v_i_3566_);
lean_dec(v_i_3566_);
v_res_3574_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg(v_sz_boxed_3572_, v_i_boxed_3573_, v_bs_3567_, v___y_3568_, v___y_3569_, v___y_3570_);
lean_dec(v___y_3570_);
lean_dec_ref(v___y_3569_);
lean_dec_ref(v___y_3568_);
return v_res_3574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4(lean_object* v_argNames_3575_, lean_object* v_args_3576_, lean_object* v_head_3577_, lean_object* v_x_3578_, lean_object* v_x_3579_, lean_object* v___y_3580_, lean_object* v___y_3581_, lean_object* v___y_3582_, lean_object* v___y_3583_){
_start:
{
if (lean_obj_tag(v_x_3578_) == 0)
{
lean_object* v___x_3585_; lean_object* v___x_3586_; 
lean_dec_ref(v_head_3577_);
v___x_3585_ = l_List_reverse___redArg(v_x_3579_);
v___x_3586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3586_, 0, v___x_3585_);
return v___x_3586_;
}
else
{
lean_object* v_head_3587_; lean_object* v_tail_3588_; lean_object* v___x_3590_; uint8_t v_isShared_3591_; uint8_t v_isSharedCheck_3606_; 
v_head_3587_ = lean_ctor_get(v_x_3578_, 0);
v_tail_3588_ = lean_ctor_get(v_x_3578_, 1);
v_isSharedCheck_3606_ = !lean_is_exclusive(v_x_3578_);
if (v_isSharedCheck_3606_ == 0)
{
v___x_3590_ = v_x_3578_;
v_isShared_3591_ = v_isSharedCheck_3606_;
goto v_resetjp_3589_;
}
else
{
lean_inc(v_tail_3588_);
lean_inc(v_head_3587_);
lean_dec(v_x_3578_);
v___x_3590_ = lean_box(0);
v_isShared_3591_ = v_isSharedCheck_3606_;
goto v_resetjp_3589_;
}
v_resetjp_3589_:
{
lean_object* v___x_3592_; 
lean_inc_ref(v_head_3577_);
v___x_3592_ = lp_mathlib_Mathlib_Tactic_Translate_elabArgStx(v_head_3587_, v_argNames_3575_, v_args_3576_, v_head_3577_, v___y_3580_, v___y_3581_, v___y_3582_, v___y_3583_);
if (lean_obj_tag(v___x_3592_) == 0)
{
lean_object* v_a_3593_; lean_object* v___x_3595_; 
v_a_3593_ = lean_ctor_get(v___x_3592_, 0);
lean_inc(v_a_3593_);
lean_dec_ref_known(v___x_3592_, 1);
if (v_isShared_3591_ == 0)
{
lean_ctor_set(v___x_3590_, 1, v_x_3579_);
lean_ctor_set(v___x_3590_, 0, v_a_3593_);
v___x_3595_ = v___x_3590_;
goto v_reusejp_3594_;
}
else
{
lean_object* v_reuseFailAlloc_3597_; 
v_reuseFailAlloc_3597_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3597_, 0, v_a_3593_);
lean_ctor_set(v_reuseFailAlloc_3597_, 1, v_x_3579_);
v___x_3595_ = v_reuseFailAlloc_3597_;
goto v_reusejp_3594_;
}
v_reusejp_3594_:
{
v_x_3578_ = v_tail_3588_;
v_x_3579_ = v___x_3595_;
goto _start;
}
}
else
{
lean_object* v_a_3598_; lean_object* v___x_3600_; uint8_t v_isShared_3601_; uint8_t v_isSharedCheck_3605_; 
lean_del_object(v___x_3590_);
lean_dec(v_tail_3588_);
lean_dec(v_x_3579_);
lean_dec_ref(v_head_3577_);
v_a_3598_ = lean_ctor_get(v___x_3592_, 0);
v_isSharedCheck_3605_ = !lean_is_exclusive(v___x_3592_);
if (v_isSharedCheck_3605_ == 0)
{
v___x_3600_ = v___x_3592_;
v_isShared_3601_ = v_isSharedCheck_3605_;
goto v_resetjp_3599_;
}
else
{
lean_inc(v_a_3598_);
lean_dec(v___x_3592_);
v___x_3600_ = lean_box(0);
v_isShared_3601_ = v_isSharedCheck_3605_;
goto v_resetjp_3599_;
}
v_resetjp_3599_:
{
lean_object* v___x_3603_; 
if (v_isShared_3601_ == 0)
{
v___x_3603_ = v___x_3600_;
goto v_reusejp_3602_;
}
else
{
lean_object* v_reuseFailAlloc_3604_; 
v_reuseFailAlloc_3604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3604_, 0, v_a_3598_);
v___x_3603_ = v_reuseFailAlloc_3604_;
goto v_reusejp_3602_;
}
v_reusejp_3602_:
{
return v___x_3603_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4___boxed(lean_object* v_argNames_3607_, lean_object* v_args_3608_, lean_object* v_head_3609_, lean_object* v_x_3610_, lean_object* v_x_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_){
_start:
{
lean_object* v_res_3617_; 
v_res_3617_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4(v_argNames_3607_, v_args_3608_, v_head_3609_, v_x_3610_, v_x_3611_, v___y_3612_, v___y_3613_, v___y_3614_, v___y_3615_);
lean_dec(v___y_3615_);
lean_dec_ref(v___y_3614_);
lean_dec(v___y_3613_);
lean_dec_ref(v___y_3612_);
lean_dec_ref(v_args_3608_);
lean_dec_ref(v_argNames_3607_);
return v_res_3617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3(size_t v_sz_3618_, size_t v_i_3619_, lean_object* v_bs_3620_){
_start:
{
uint8_t v___x_3621_; 
v___x_3621_ = lean_usize_dec_lt(v_i_3619_, v_sz_3618_);
if (v___x_3621_ == 0)
{
lean_object* v___x_3622_; 
v___x_3622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3622_, 0, v_bs_3620_);
return v___x_3622_;
}
else
{
lean_object* v_v_3623_; lean_object* v___x_3624_; lean_object* v_bs_x27_3625_; size_t v___x_3626_; size_t v___x_3627_; lean_object* v___x_3628_; 
v_v_3623_ = lean_array_uget(v_bs_3620_, v_i_3619_);
v___x_3624_ = lean_unsigned_to_nat(0u);
v_bs_x27_3625_ = lean_array_uset(v_bs_3620_, v_i_3619_, v___x_3624_);
v___x_3626_ = ((size_t)1ULL);
v___x_3627_ = lean_usize_add(v_i_3619_, v___x_3626_);
v___x_3628_ = lean_array_uset(v_bs_x27_3625_, v_i_3619_, v_v_3623_);
v_i_3619_ = v___x_3627_;
v_bs_3620_ = v___x_3628_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3___boxed(lean_object* v_sz_3630_, lean_object* v_i_3631_, lean_object* v_bs_3632_){
_start:
{
size_t v_sz_boxed_3633_; size_t v_i_boxed_3634_; lean_object* v_res_3635_; 
v_sz_boxed_3633_ = lean_unbox_usize(v_sz_3630_);
lean_dec(v_sz_3630_);
v_i_boxed_3634_ = lean_unbox_usize(v_i_3631_);
lean_dec(v_i_3631_);
v_res_3635_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3(v_sz_boxed_3633_, v_i_boxed_3634_, v_bs_3632_);
return v_res_3635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0___boxed(lean_object* v___x_3639_, lean_object* v_val_3640_, lean_object* v_xs_3641_, lean_object* v_x_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_){
_start:
{
lean_object* v_res_3648_; 
v_res_3648_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0(v___x_3639_, v_val_3640_, v_xs_3641_, v_x_3642_, v___y_3643_, v___y_3644_, v___y_3645_, v___y_3646_);
lean_dec(v___y_3646_);
lean_dec_ref(v___y_3645_);
lean_dec(v___y_3644_);
lean_dec_ref(v___y_3643_);
lean_dec_ref(v_x_3642_);
return v_res_3648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg(lean_object* v_args_3649_, lean_object* v_val_3650_, lean_object* v_as_x27_3651_, lean_object* v_b_3652_, lean_object* v___y_3653_, lean_object* v___y_3654_, lean_object* v___y_3655_, lean_object* v___y_3656_){
_start:
{
if (lean_obj_tag(v_as_x27_3651_) == 0)
{
lean_object* v___x_3658_; 
lean_dec(v_val_3650_);
v___x_3658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3658_, 0, v_b_3652_);
return v___x_3658_;
}
else
{
lean_object* v_head_3659_; lean_object* v_tail_3660_; lean_object* v___x_3661_; lean_object* v___x_3662_; lean_object* v___x_3663_; 
v_head_3659_ = lean_ctor_get(v_as_x27_3651_, 0);
v_tail_3660_ = lean_ctor_get(v_as_x27_3651_, 1);
v___x_3661_ = l_Lean_instInhabitedExpr;
v___x_3662_ = lean_array_get_borrowed(v___x_3661_, v_args_3649_, v_head_3659_);
lean_inc(v___y_3656_);
lean_inc_ref(v___y_3655_);
lean_inc(v___y_3654_);
lean_inc_ref(v___y_3653_);
lean_inc(v___x_3662_);
v___x_3663_ = lean_infer_type(v___x_3662_, v___y_3653_, v___y_3654_, v___y_3655_, v___y_3656_);
if (lean_obj_tag(v___x_3663_) == 0)
{
lean_object* v_a_3664_; lean_object* v_keyedConfig_3665_; uint8_t v_trackZetaDelta_3666_; lean_object* v_zetaDeltaSet_3667_; lean_object* v_lctx_3668_; lean_object* v_localInstances_3669_; lean_object* v_defEqCtx_x3f_3670_; lean_object* v_synthPendingDepth_3671_; lean_object* v_customCanUnfoldPredicate_x3f_3672_; uint8_t v_univApprox_3673_; uint8_t v_inTypeClassResolution_3674_; uint8_t v_cacheInferType_3675_; lean_object* v___f_3676_; uint8_t v___x_3677_; uint8_t v___x_3678_; lean_object* v___x_3679_; lean_object* v___x_3680_; lean_object* v___x_3681_; 
v_a_3664_ = lean_ctor_get(v___x_3663_, 0);
lean_inc(v_a_3664_);
lean_dec_ref_known(v___x_3663_, 1);
v_keyedConfig_3665_ = lean_ctor_get(v___y_3653_, 0);
v_trackZetaDelta_3666_ = lean_ctor_get_uint8(v___y_3653_, sizeof(void*)*7);
v_zetaDeltaSet_3667_ = lean_ctor_get(v___y_3653_, 1);
v_lctx_3668_ = lean_ctor_get(v___y_3653_, 2);
v_localInstances_3669_ = lean_ctor_get(v___y_3653_, 3);
v_defEqCtx_x3f_3670_ = lean_ctor_get(v___y_3653_, 4);
v_synthPendingDepth_3671_ = lean_ctor_get(v___y_3653_, 5);
v_customCanUnfoldPredicate_x3f_3672_ = lean_ctor_get(v___y_3653_, 6);
v_univApprox_3673_ = lean_ctor_get_uint8(v___y_3653_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3674_ = lean_ctor_get_uint8(v___y_3653_, sizeof(void*)*7 + 2);
v_cacheInferType_3675_ = lean_ctor_get_uint8(v___y_3653_, sizeof(void*)*7 + 3);
lean_inc(v_val_3650_);
lean_inc(v___x_3662_);
v___f_3676_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0___boxed), 9, 2);
lean_closure_set(v___f_3676_, 0, v___x_3662_);
lean_closure_set(v___f_3676_, 1, v_val_3650_);
v___x_3677_ = 0;
v___x_3678_ = 2;
lean_inc_ref(v_keyedConfig_3665_);
v___x_3679_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3678_, v_keyedConfig_3665_);
lean_inc(v_customCanUnfoldPredicate_x3f_3672_);
lean_inc(v_synthPendingDepth_3671_);
lean_inc(v_defEqCtx_x3f_3670_);
lean_inc_ref(v_localInstances_3669_);
lean_inc_ref(v_lctx_3668_);
lean_inc(v_zetaDeltaSet_3667_);
v___x_3680_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3680_, 0, v___x_3679_);
lean_ctor_set(v___x_3680_, 1, v_zetaDeltaSet_3667_);
lean_ctor_set(v___x_3680_, 2, v_lctx_3668_);
lean_ctor_set(v___x_3680_, 3, v_localInstances_3669_);
lean_ctor_set(v___x_3680_, 4, v_defEqCtx_x3f_3670_);
lean_ctor_set(v___x_3680_, 5, v_synthPendingDepth_3671_);
lean_ctor_set(v___x_3680_, 6, v_customCanUnfoldPredicate_x3f_3672_);
lean_ctor_set_uint8(v___x_3680_, sizeof(void*)*7, v_trackZetaDelta_3666_);
lean_ctor_set_uint8(v___x_3680_, sizeof(void*)*7 + 1, v_univApprox_3673_);
lean_ctor_set_uint8(v___x_3680_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3674_);
lean_ctor_set_uint8(v___x_3680_, sizeof(void*)*7 + 3, v_cacheInferType_3675_);
v___x_3681_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Translate_elabReorder_spec__6___redArg(v_a_3664_, v___f_3676_, v___x_3677_, v___x_3677_, v___x_3680_, v___y_3654_, v___y_3655_, v___y_3656_);
lean_dec_ref_known(v___x_3680_, 7);
if (lean_obj_tag(v___x_3681_) == 0)
{
lean_object* v_a_3682_; lean_object* v___x_3683_; lean_object* v___x_3684_; 
v_a_3682_ = lean_ctor_get(v___x_3681_, 0);
lean_inc(v_a_3682_);
lean_dec_ref_known(v___x_3681_, 1);
lean_inc(v_head_3659_);
v___x_3683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3683_, 0, v_head_3659_);
lean_ctor_set(v___x_3683_, 1, v_a_3682_);
v___x_3684_ = lean_array_push(v_b_3652_, v___x_3683_);
v_as_x27_3651_ = v_tail_3660_;
v_b_3652_ = v___x_3684_;
goto _start;
}
else
{
lean_object* v_a_3686_; lean_object* v___x_3688_; uint8_t v_isShared_3689_; uint8_t v_isSharedCheck_3693_; 
lean_dec_ref(v_b_3652_);
lean_dec(v_val_3650_);
v_a_3686_ = lean_ctor_get(v___x_3681_, 0);
v_isSharedCheck_3693_ = !lean_is_exclusive(v___x_3681_);
if (v_isSharedCheck_3693_ == 0)
{
v___x_3688_ = v___x_3681_;
v_isShared_3689_ = v_isSharedCheck_3693_;
goto v_resetjp_3687_;
}
else
{
lean_inc(v_a_3686_);
lean_dec(v___x_3681_);
v___x_3688_ = lean_box(0);
v_isShared_3689_ = v_isSharedCheck_3693_;
goto v_resetjp_3687_;
}
v_resetjp_3687_:
{
lean_object* v___x_3691_; 
if (v_isShared_3689_ == 0)
{
v___x_3691_ = v___x_3688_;
goto v_reusejp_3690_;
}
else
{
lean_object* v_reuseFailAlloc_3692_; 
v_reuseFailAlloc_3692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3692_, 0, v_a_3686_);
v___x_3691_ = v_reuseFailAlloc_3692_;
goto v_reusejp_3690_;
}
v_reusejp_3690_:
{
return v___x_3691_;
}
}
}
}
else
{
lean_object* v_a_3694_; lean_object* v___x_3696_; uint8_t v_isShared_3697_; uint8_t v_isSharedCheck_3701_; 
lean_dec_ref(v_b_3652_);
lean_dec(v_val_3650_);
v_a_3694_ = lean_ctor_get(v___x_3663_, 0);
v_isSharedCheck_3701_ = !lean_is_exclusive(v___x_3663_);
if (v_isSharedCheck_3701_ == 0)
{
v___x_3696_ = v___x_3663_;
v_isShared_3697_ = v_isSharedCheck_3701_;
goto v_resetjp_3695_;
}
else
{
lean_inc(v_a_3694_);
lean_dec(v___x_3663_);
v___x_3696_ = lean_box(0);
v_isShared_3697_ = v_isSharedCheck_3701_;
goto v_resetjp_3695_;
}
v_resetjp_3695_:
{
lean_object* v___x_3699_; 
if (v_isShared_3697_ == 0)
{
v___x_3699_ = v___x_3696_;
goto v_reusejp_3698_;
}
else
{
lean_object* v_reuseFailAlloc_3700_; 
v_reuseFailAlloc_3700_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3700_, 0, v_a_3694_);
v___x_3699_ = v_reuseFailAlloc_3700_;
goto v_reusejp_3698_;
}
v_reusejp_3698_:
{
return v___x_3699_;
}
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1(void){
_start:
{
lean_object* v___x_3703_; lean_object* v___x_3704_; 
v___x_3703_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__0));
v___x_3704_ = l_Lean_stringToMessageData(v___x_3703_);
return v___x_3704_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3(void){
_start:
{
lean_object* v___x_3706_; lean_object* v___x_3707_; 
v___x_3706_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__2));
v___x_3707_ = l_Lean_stringToMessageData(v___x_3706_);
return v___x_3707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8(lean_object* v_argNames_3708_, lean_object* v_args_3709_, lean_object* v_head_3710_, lean_object* v_as_3711_, size_t v_sz_3712_, size_t v_i_3713_, lean_object* v_b_3714_, lean_object* v___y_3715_, lean_object* v___y_3716_, lean_object* v___y_3717_, lean_object* v___y_3718_){
_start:
{
lean_object* v_a_3721_; uint8_t v___x_3725_; 
v___x_3725_ = lean_usize_dec_lt(v_i_3713_, v_sz_3712_);
if (v___x_3725_ == 0)
{
lean_object* v___x_3726_; 
lean_dec_ref(v_head_3710_);
v___x_3726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3726_, 0, v_b_3714_);
return v___x_3726_;
}
else
{
lean_object* v_fst_3727_; lean_object* v_snd_3728_; lean_object* v___x_3730_; uint8_t v_isShared_3731_; uint8_t v_isSharedCheck_3848_; 
v_fst_3727_ = lean_ctor_get(v_b_3714_, 0);
v_snd_3728_ = lean_ctor_get(v_b_3714_, 1);
v_isSharedCheck_3848_ = !lean_is_exclusive(v_b_3714_);
if (v_isSharedCheck_3848_ == 0)
{
v___x_3730_ = v_b_3714_;
v_isShared_3731_ = v_isSharedCheck_3848_;
goto v_resetjp_3729_;
}
else
{
lean_inc(v_snd_3728_);
lean_inc(v_fst_3727_);
lean_dec(v_b_3714_);
v___x_3730_ = lean_box(0);
v_isShared_3731_ = v_isSharedCheck_3848_;
goto v_resetjp_3729_;
}
v_resetjp_3729_:
{
lean_object* v___y_3733_; lean_object* v_val_3734_; lean_object* v_perm_3735_; lean_object* v___y_3736_; lean_object* v___y_3737_; lean_object* v___y_3738_; lean_object* v___y_3739_; lean_object* v_perm_3754_; lean_object* v_a_3756_; lean_object* v___x_3757_; uint8_t v___x_3758_; 
v_a_3756_ = lean_array_uget_borrowed(v_as_3711_, v_i_3713_);
v___x_3757_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorderPart___closed__4));
lean_inc(v_a_3756_);
v___x_3758_ = l_Lean_Syntax_isOfKind(v_a_3756_, v___x_3757_);
if (v___x_3758_ == 0)
{
lean_object* v___x_3759_; 
lean_del_object(v___x_3730_);
v___x_3759_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
if (lean_obj_tag(v___x_3759_) == 0)
{
lean_object* v___x_3760_; 
lean_dec_ref_known(v___x_3759_, 1);
v___x_3760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3760_, 0, v_fst_3727_);
lean_ctor_set(v___x_3760_, 1, v_snd_3728_);
v_a_3721_ = v___x_3760_;
goto v___jp_3720_;
}
else
{
lean_object* v_a_3761_; lean_object* v___x_3763_; uint8_t v_isShared_3764_; uint8_t v_isSharedCheck_3768_; 
lean_dec(v_snd_3728_);
lean_dec(v_fst_3727_);
lean_dec_ref(v_head_3710_);
v_a_3761_ = lean_ctor_get(v___x_3759_, 0);
v_isSharedCheck_3768_ = !lean_is_exclusive(v___x_3759_);
if (v_isSharedCheck_3768_ == 0)
{
v___x_3763_ = v___x_3759_;
v_isShared_3764_ = v_isSharedCheck_3768_;
goto v_resetjp_3762_;
}
else
{
lean_inc(v_a_3761_);
lean_dec(v___x_3759_);
v___x_3763_ = lean_box(0);
v_isShared_3764_ = v_isSharedCheck_3768_;
goto v_resetjp_3762_;
}
v_resetjp_3762_:
{
lean_object* v___x_3766_; 
if (v_isShared_3764_ == 0)
{
v___x_3766_ = v___x_3763_;
goto v_reusejp_3765_;
}
else
{
lean_object* v_reuseFailAlloc_3767_; 
v_reuseFailAlloc_3767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3767_, 0, v_a_3761_);
v___x_3766_ = v_reuseFailAlloc_3767_;
goto v_reusejp_3765_;
}
v_reusejp_3765_:
{
return v___x_3766_;
}
}
}
}
else
{
lean_object* v___x_3769_; lean_object* v___x_3770_; lean_object* v___x_3771_; size_t v_sz_3772_; size_t v___x_3773_; lean_object* v___x_3774_; 
v___x_3769_ = lean_unsigned_to_nat(0u);
v___x_3770_ = l_Lean_Syntax_getArg(v_a_3756_, v___x_3769_);
v___x_3771_ = l_Lean_Syntax_getArgs(v___x_3770_);
lean_dec(v___x_3770_);
v_sz_3772_ = lean_array_size(v___x_3771_);
v___x_3773_ = ((size_t)0ULL);
v___x_3774_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__3(v_sz_3772_, v___x_3773_, v___x_3771_);
if (lean_obj_tag(v___x_3774_) == 0)
{
lean_object* v___x_3775_; 
lean_del_object(v___x_3730_);
v___x_3775_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
if (lean_obj_tag(v___x_3775_) == 0)
{
lean_object* v___x_3776_; 
lean_dec_ref_known(v___x_3775_, 1);
v___x_3776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3776_, 0, v_fst_3727_);
lean_ctor_set(v___x_3776_, 1, v_snd_3728_);
v_a_3721_ = v___x_3776_;
goto v___jp_3720_;
}
else
{
lean_object* v_a_3777_; lean_object* v___x_3779_; uint8_t v_isShared_3780_; uint8_t v_isSharedCheck_3784_; 
lean_dec(v_snd_3728_);
lean_dec(v_fst_3727_);
lean_dec_ref(v_head_3710_);
v_a_3777_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_3784_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_3784_ == 0)
{
v___x_3779_ = v___x_3775_;
v_isShared_3780_ = v_isSharedCheck_3784_;
goto v_resetjp_3778_;
}
else
{
lean_inc(v_a_3777_);
lean_dec(v___x_3775_);
v___x_3779_ = lean_box(0);
v_isShared_3780_ = v_isSharedCheck_3784_;
goto v_resetjp_3778_;
}
v_resetjp_3778_:
{
lean_object* v___x_3782_; 
if (v_isShared_3780_ == 0)
{
v___x_3782_ = v___x_3779_;
goto v_reusejp_3781_;
}
else
{
lean_object* v_reuseFailAlloc_3783_; 
v_reuseFailAlloc_3783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3783_, 0, v_a_3777_);
v___x_3782_ = v_reuseFailAlloc_3783_;
goto v_reusejp_3781_;
}
v_reusejp_3781_:
{
return v___x_3782_;
}
}
}
}
else
{
lean_object* v_val_3785_; lean_object* v___x_3787_; uint8_t v_isShared_3788_; uint8_t v_isSharedCheck_3847_; 
v_val_3785_ = lean_ctor_get(v___x_3774_, 0);
v_isSharedCheck_3847_ = !lean_is_exclusive(v___x_3774_);
if (v_isSharedCheck_3847_ == 0)
{
v___x_3787_ = v___x_3774_;
v_isShared_3788_ = v_isSharedCheck_3847_;
goto v_resetjp_3786_;
}
else
{
lean_inc(v_val_3785_);
lean_dec(v___x_3774_);
v___x_3787_ = lean_box(0);
v_isShared_3788_ = v_isSharedCheck_3847_;
goto v_resetjp_3786_;
}
v_resetjp_3786_:
{
lean_object* v_argReorder_x3f_3790_; lean_object* v___y_3791_; lean_object* v___y_3792_; lean_object* v___y_3793_; lean_object* v___y_3794_; lean_object* v___x_3827_; lean_object* v___x_3828_; uint8_t v___x_3829_; 
v___x_3827_ = lean_unsigned_to_nat(1u);
v___x_3828_ = l_Lean_Syntax_getArg(v_a_3756_, v___x_3827_);
v___x_3829_ = l_Lean_Syntax_isNone(v___x_3828_);
if (v___x_3829_ == 0)
{
lean_object* v___x_3830_; uint8_t v___x_3831_; 
v___x_3830_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_3828_);
v___x_3831_ = l_Lean_Syntax_matchesNull(v___x_3828_, v___x_3830_);
if (v___x_3831_ == 0)
{
lean_object* v___x_3832_; 
lean_dec(v___x_3828_);
lean_del_object(v___x_3787_);
lean_dec(v_val_3785_);
lean_del_object(v___x_3730_);
v___x_3832_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
if (lean_obj_tag(v___x_3832_) == 0)
{
lean_object* v___x_3833_; 
lean_dec_ref_known(v___x_3832_, 1);
v___x_3833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3833_, 0, v_fst_3727_);
lean_ctor_set(v___x_3833_, 1, v_snd_3728_);
v_a_3721_ = v___x_3833_;
goto v___jp_3720_;
}
else
{
lean_object* v_a_3834_; lean_object* v___x_3836_; uint8_t v_isShared_3837_; uint8_t v_isSharedCheck_3841_; 
lean_dec(v_snd_3728_);
lean_dec(v_fst_3727_);
lean_dec_ref(v_head_3710_);
v_a_3834_ = lean_ctor_get(v___x_3832_, 0);
v_isSharedCheck_3841_ = !lean_is_exclusive(v___x_3832_);
if (v_isSharedCheck_3841_ == 0)
{
v___x_3836_ = v___x_3832_;
v_isShared_3837_ = v_isSharedCheck_3841_;
goto v_resetjp_3835_;
}
else
{
lean_inc(v_a_3834_);
lean_dec(v___x_3832_);
v___x_3836_ = lean_box(0);
v_isShared_3837_ = v_isSharedCheck_3841_;
goto v_resetjp_3835_;
}
v_resetjp_3835_:
{
lean_object* v___x_3839_; 
if (v_isShared_3837_ == 0)
{
v___x_3839_ = v___x_3836_;
goto v_reusejp_3838_;
}
else
{
lean_object* v_reuseFailAlloc_3840_; 
v_reuseFailAlloc_3840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3840_, 0, v_a_3834_);
v___x_3839_ = v_reuseFailAlloc_3840_;
goto v_reusejp_3838_;
}
v_reusejp_3838_:
{
return v___x_3839_;
}
}
}
}
else
{
lean_object* v___x_3842_; lean_object* v___x_3844_; 
v___x_3842_ = l_Lean_Syntax_getArg(v___x_3828_, v___x_3827_);
lean_dec(v___x_3828_);
if (v_isShared_3788_ == 0)
{
lean_ctor_set(v___x_3787_, 0, v___x_3842_);
v___x_3844_ = v___x_3787_;
goto v_reusejp_3843_;
}
else
{
lean_object* v_reuseFailAlloc_3845_; 
v_reuseFailAlloc_3845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3845_, 0, v___x_3842_);
v___x_3844_ = v_reuseFailAlloc_3845_;
goto v_reusejp_3843_;
}
v_reusejp_3843_:
{
v_argReorder_x3f_3790_ = v___x_3844_;
v___y_3791_ = v___y_3715_;
v___y_3792_ = v___y_3716_;
v___y_3793_ = v___y_3717_;
v___y_3794_ = v___y_3718_;
goto v___jp_3789_;
}
}
}
else
{
lean_object* v___x_3846_; 
lean_dec(v___x_3828_);
lean_del_object(v___x_3787_);
v___x_3846_ = lean_box(0);
v_argReorder_x3f_3790_ = v___x_3846_;
v___y_3791_ = v___y_3715_;
v___y_3792_ = v___y_3716_;
v___y_3793_ = v___y_3717_;
v___y_3794_ = v___y_3718_;
goto v___jp_3789_;
}
v___jp_3789_:
{
lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; 
v___x_3795_ = lean_array_to_list(v_val_3785_);
v___x_3796_ = lean_box(0);
lean_inc_ref(v_head_3710_);
v___x_3797_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__4(v_argNames_3708_, v_args_3709_, v_head_3710_, v___x_3795_, v___x_3796_, v___y_3791_, v___y_3792_, v___y_3793_, v___y_3794_);
if (lean_obj_tag(v___x_3797_) == 0)
{
lean_object* v_a_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; uint8_t v___x_3801_; 
v_a_3798_ = lean_ctor_get(v___x_3797_, 0);
lean_inc(v_a_3798_);
lean_dec_ref_known(v___x_3797_, 1);
v___x_3799_ = lean_unsigned_to_nat(2u);
v___x_3800_ = l_List_lengthTR___redArg(v_a_3798_);
v___x_3801_ = lean_nat_dec_le(v___x_3799_, v___x_3800_);
lean_dec(v___x_3800_);
if (v___x_3801_ == 0)
{
if (lean_obj_tag(v_argReorder_x3f_3790_) == 0)
{
lean_dec(v_a_3798_);
lean_del_object(v___x_3730_);
if (v___x_3758_ == 0)
{
v_perm_3754_ = v_fst_3727_;
goto v___jp_3753_;
}
else
{
lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3807_; 
v___x_3802_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__1);
lean_inc(v_a_3756_);
v___x_3803_ = l_Lean_MessageData_ofSyntax(v_a_3756_);
v___x_3804_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3804_, 0, v___x_3802_);
lean_ctor_set(v___x_3804_, 1, v___x_3803_);
v___x_3805_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___closed__3);
v___x_3806_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3806_, 0, v___x_3804_);
lean_ctor_set(v___x_3806_, 1, v___x_3805_);
v___x_3807_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Translate_elabArgStx_spec__1___redArg(v_a_3756_, v___x_3806_, v___y_3791_, v___y_3792_, v___y_3793_, v___y_3794_);
if (lean_obj_tag(v___x_3807_) == 0)
{
lean_dec_ref_known(v___x_3807_, 1);
v_perm_3754_ = v_fst_3727_;
goto v___jp_3753_;
}
else
{
lean_object* v_a_3808_; lean_object* v___x_3810_; uint8_t v_isShared_3811_; uint8_t v_isSharedCheck_3815_; 
lean_dec(v_snd_3728_);
lean_dec(v_fst_3727_);
lean_dec_ref(v_head_3710_);
v_a_3808_ = lean_ctor_get(v___x_3807_, 0);
v_isSharedCheck_3815_ = !lean_is_exclusive(v___x_3807_);
if (v_isSharedCheck_3815_ == 0)
{
v___x_3810_ = v___x_3807_;
v_isShared_3811_ = v_isSharedCheck_3815_;
goto v_resetjp_3809_;
}
else
{
lean_inc(v_a_3808_);
lean_dec(v___x_3807_);
v___x_3810_ = lean_box(0);
v_isShared_3811_ = v_isSharedCheck_3815_;
goto v_resetjp_3809_;
}
v_resetjp_3809_:
{
lean_object* v___x_3813_; 
if (v_isShared_3811_ == 0)
{
v___x_3813_ = v___x_3810_;
goto v_reusejp_3812_;
}
else
{
lean_object* v_reuseFailAlloc_3814_; 
v_reuseFailAlloc_3814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3814_, 0, v_a_3808_);
v___x_3813_ = v_reuseFailAlloc_3814_;
goto v_reusejp_3812_;
}
v_reusejp_3812_:
{
return v___x_3813_;
}
}
}
}
}
else
{
lean_object* v_val_3816_; 
v_val_3816_ = lean_ctor_get(v_argReorder_x3f_3790_, 0);
lean_inc(v_val_3816_);
lean_dec_ref_known(v_argReorder_x3f_3790_, 1);
v___y_3733_ = v_a_3798_;
v_val_3734_ = v_val_3816_;
v_perm_3735_ = v_fst_3727_;
v___y_3736_ = v___y_3791_;
v___y_3737_ = v___y_3792_;
v___y_3738_ = v___y_3793_;
v___y_3739_ = v___y_3794_;
goto v___jp_3732_;
}
}
else
{
lean_object* v___x_3817_; 
lean_inc(v_a_3798_);
v___x_3817_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3817_, 0, v_a_3798_);
lean_ctor_set(v___x_3817_, 1, v_fst_3727_);
if (lean_obj_tag(v_argReorder_x3f_3790_) == 1)
{
lean_object* v_val_3818_; 
v_val_3818_ = lean_ctor_get(v_argReorder_x3f_3790_, 0);
lean_inc(v_val_3818_);
lean_dec_ref_known(v_argReorder_x3f_3790_, 1);
v___y_3733_ = v_a_3798_;
v_val_3734_ = v_val_3818_;
v_perm_3735_ = v___x_3817_;
v___y_3736_ = v___y_3791_;
v___y_3737_ = v___y_3792_;
v___y_3738_ = v___y_3793_;
v___y_3739_ = v___y_3794_;
goto v___jp_3732_;
}
else
{
lean_dec(v_a_3798_);
lean_dec(v_argReorder_x3f_3790_);
lean_del_object(v___x_3730_);
v_perm_3754_ = v___x_3817_;
goto v___jp_3753_;
}
}
}
else
{
lean_object* v_a_3819_; lean_object* v___x_3821_; uint8_t v_isShared_3822_; uint8_t v_isSharedCheck_3826_; 
lean_dec(v_argReorder_x3f_3790_);
lean_del_object(v___x_3730_);
lean_dec(v_snd_3728_);
lean_dec(v_fst_3727_);
lean_dec_ref(v_head_3710_);
v_a_3819_ = lean_ctor_get(v___x_3797_, 0);
v_isSharedCheck_3826_ = !lean_is_exclusive(v___x_3797_);
if (v_isSharedCheck_3826_ == 0)
{
v___x_3821_ = v___x_3797_;
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
else
{
lean_inc(v_a_3819_);
lean_dec(v___x_3797_);
v___x_3821_ = lean_box(0);
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
v_resetjp_3820_:
{
lean_object* v___x_3824_; 
if (v_isShared_3822_ == 0)
{
v___x_3824_ = v___x_3821_;
goto v_reusejp_3823_;
}
else
{
lean_object* v_reuseFailAlloc_3825_; 
v_reuseFailAlloc_3825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3825_, 0, v_a_3819_);
v___x_3824_ = v_reuseFailAlloc_3825_;
goto v_reusejp_3823_;
}
v_reusejp_3823_:
{
return v___x_3824_;
}
}
}
}
}
}
}
v___jp_3732_:
{
lean_object* v___x_3740_; 
v___x_3740_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg(v_args_3709_, v_val_3734_, v___y_3733_, v_snd_3728_, v___y_3736_, v___y_3737_, v___y_3738_, v___y_3739_);
lean_dec(v___y_3733_);
if (lean_obj_tag(v___x_3740_) == 0)
{
lean_object* v_a_3741_; lean_object* v___x_3743_; 
v_a_3741_ = lean_ctor_get(v___x_3740_, 0);
lean_inc(v_a_3741_);
lean_dec_ref_known(v___x_3740_, 1);
if (v_isShared_3731_ == 0)
{
lean_ctor_set(v___x_3730_, 1, v_a_3741_);
lean_ctor_set(v___x_3730_, 0, v_perm_3735_);
v___x_3743_ = v___x_3730_;
goto v_reusejp_3742_;
}
else
{
lean_object* v_reuseFailAlloc_3744_; 
v_reuseFailAlloc_3744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3744_, 0, v_perm_3735_);
lean_ctor_set(v_reuseFailAlloc_3744_, 1, v_a_3741_);
v___x_3743_ = v_reuseFailAlloc_3744_;
goto v_reusejp_3742_;
}
v_reusejp_3742_:
{
v_a_3721_ = v___x_3743_;
goto v___jp_3720_;
}
}
else
{
lean_object* v_a_3745_; lean_object* v___x_3747_; uint8_t v_isShared_3748_; uint8_t v_isSharedCheck_3752_; 
lean_dec(v_perm_3735_);
lean_del_object(v___x_3730_);
lean_dec_ref(v_head_3710_);
v_a_3745_ = lean_ctor_get(v___x_3740_, 0);
v_isSharedCheck_3752_ = !lean_is_exclusive(v___x_3740_);
if (v_isSharedCheck_3752_ == 0)
{
v___x_3747_ = v___x_3740_;
v_isShared_3748_ = v_isSharedCheck_3752_;
goto v_resetjp_3746_;
}
else
{
lean_inc(v_a_3745_);
lean_dec(v___x_3740_);
v___x_3747_ = lean_box(0);
v_isShared_3748_ = v_isSharedCheck_3752_;
goto v_resetjp_3746_;
}
v_resetjp_3746_:
{
lean_object* v___x_3750_; 
if (v_isShared_3748_ == 0)
{
v___x_3750_ = v___x_3747_;
goto v_reusejp_3749_;
}
else
{
lean_object* v_reuseFailAlloc_3751_; 
v_reuseFailAlloc_3751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3751_, 0, v_a_3745_);
v___x_3750_ = v_reuseFailAlloc_3751_;
goto v_reusejp_3749_;
}
v_reusejp_3749_:
{
return v___x_3750_;
}
}
}
}
v___jp_3753_:
{
lean_object* v___x_3755_; 
v___x_3755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3755_, 0, v_perm_3754_);
lean_ctor_set(v___x_3755_, 1, v_snd_3728_);
v_a_3721_ = v___x_3755_;
goto v___jp_3720_;
}
}
}
v___jp_3720_:
{
size_t v___x_3722_; size_t v___x_3723_; 
v___x_3722_ = ((size_t)1ULL);
v___x_3723_ = lean_usize_add(v_i_3713_, v___x_3722_);
v_i_3713_ = v___x_3723_;
v_b_3714_ = v_a_3721_;
goto _start;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1(void){
_start:
{
lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; 
v___x_3849_ = lean_box(0);
v___x_3850_ = lean_unsigned_to_nat(16u);
v___x_3851_ = lean_mk_array(v___x_3850_, v___x_3849_);
return v___x_3851_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2(void){
_start:
{
lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; 
v___x_3852_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1, &lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__1);
v___x_3853_ = lean_unsigned_to_nat(0u);
v___x_3854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3854_, 0, v___x_3853_);
lean_ctor_set(v___x_3854_, 1, v___x_3852_);
return v___x_3854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder(lean_object* v_stx_3857_, lean_object* v_argNames_3858_, lean_object* v_args_3859_, lean_object* v_head_3860_, lean_object* v_a_3861_, lean_object* v_a_3862_, lean_object* v_a_3863_, lean_object* v_a_3864_){
_start:
{
lean_object* v___x_3866_; uint8_t v___x_3867_; 
v___x_3866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_reorder___closed__1));
lean_inc(v_stx_3857_);
v___x_3867_ = l_Lean_Syntax_isOfKind(v_stx_3857_, v___x_3866_);
if (v___x_3867_ == 0)
{
lean_object* v___x_3868_; 
lean_dec_ref(v_head_3860_);
lean_dec(v_stx_3857_);
v___x_3868_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
return v___x_3868_;
}
else
{
lean_object* v___x_3869_; lean_object* v___y_3871_; lean_object* v___y_3872_; lean_object* v___y_3873_; lean_object* v___y_3874_; lean_object* v___y_3897_; lean_object* v___y_3898_; lean_object* v___y_3899_; lean_object* v___y_3900_; lean_object* v___y_3901_; lean_object* v___y_3902_; lean_object* v___y_3903_; lean_object* v___y_3906_; lean_object* v___y_3907_; lean_object* v___y_3908_; lean_object* v___y_3909_; lean_object* v___y_3910_; lean_object* v___y_3911_; lean_object* v___y_3912_; lean_object* v___y_3915_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; uint8_t v___x_3980_; 
v___x_3869_ = lean_unsigned_to_nat(0u);
v___x_3976_ = l_Lean_Syntax_getArg(v_stx_3857_, v___x_3869_);
v___x_3977_ = l_Lean_Syntax_getArgs(v___x_3976_);
lean_dec(v___x_3976_);
v___x_3978_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__3));
v___x_3979_ = lean_array_get_size(v___x_3977_);
v___x_3980_ = lean_nat_dec_lt(v___x_3869_, v___x_3979_);
if (v___x_3980_ == 0)
{
lean_dec_ref(v___x_3977_);
v___y_3915_ = v___x_3978_;
goto v___jp_3914_;
}
else
{
lean_object* v___x_3981_; lean_object* v___x_3982_; uint8_t v___x_3983_; 
v___x_3981_ = lean_box(v___x_3867_);
v___x_3982_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3982_, 0, v___x_3981_);
lean_ctor_set(v___x_3982_, 1, v___x_3978_);
v___x_3983_ = lean_nat_dec_le(v___x_3979_, v___x_3979_);
if (v___x_3983_ == 0)
{
if (v___x_3980_ == 0)
{
lean_dec_ref_known(v___x_3982_, 2);
lean_dec_ref(v___x_3977_);
v___y_3915_ = v___x_3978_;
goto v___jp_3914_;
}
else
{
size_t v___x_3984_; size_t v___x_3985_; lean_object* v___x_3986_; lean_object* v_snd_3987_; 
v___x_3984_ = ((size_t)0ULL);
v___x_3985_ = lean_usize_of_nat(v___x_3979_);
v___x_3986_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12(v___x_3867_, v___x_3977_, v___x_3984_, v___x_3985_, v___x_3982_);
lean_dec_ref(v___x_3977_);
v_snd_3987_ = lean_ctor_get(v___x_3986_, 1);
lean_inc(v_snd_3987_);
lean_dec_ref(v___x_3986_);
v___y_3915_ = v_snd_3987_;
goto v___jp_3914_;
}
}
else
{
size_t v___x_3988_; size_t v___x_3989_; lean_object* v___x_3990_; lean_object* v_snd_3991_; 
v___x_3988_ = ((size_t)0ULL);
v___x_3989_ = lean_usize_of_nat(v___x_3979_);
v___x_3990_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Translate_elabReorder_spec__12(v___x_3867_, v___x_3977_, v___x_3988_, v___x_3989_, v___x_3982_);
lean_dec_ref(v___x_3977_);
v_snd_3991_ = lean_ctor_get(v___x_3990_, 1);
lean_inc(v_snd_3991_);
lean_dec_ref(v___x_3990_);
v___y_3915_ = v_snd_3991_;
goto v___jp_3914_;
}
}
v___jp_3870_:
{
lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; 
v___x_3875_ = lean_array_get_size(v___y_3874_);
v___x_3876_ = lean_nat_sub(v___x_3875_, v___y_3872_);
v___x_3877_ = lean_box(0);
v___x_3878_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg(v___x_3876_, v___y_3874_, v___x_3869_, v___x_3877_, v_a_3861_, v_a_3862_, v___y_3873_, v_a_3864_);
lean_dec_ref(v___y_3873_);
lean_dec(v___x_3876_);
if (lean_obj_tag(v___x_3878_) == 0)
{
lean_object* v___x_3880_; uint8_t v_isShared_3881_; uint8_t v_isSharedCheck_3886_; 
v_isSharedCheck_3886_ = !lean_is_exclusive(v___x_3878_);
if (v_isSharedCheck_3886_ == 0)
{
lean_object* v_unused_3887_; 
v_unused_3887_ = lean_ctor_get(v___x_3878_, 0);
lean_dec(v_unused_3887_);
v___x_3880_ = v___x_3878_;
v_isShared_3881_ = v_isSharedCheck_3886_;
goto v_resetjp_3879_;
}
else
{
lean_dec(v___x_3878_);
v___x_3880_ = lean_box(0);
v_isShared_3881_ = v_isSharedCheck_3886_;
goto v_resetjp_3879_;
}
v_resetjp_3879_:
{
lean_object* v___x_3882_; lean_object* v___x_3884_; 
v___x_3882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3882_, 0, v___y_3871_);
lean_ctor_set(v___x_3882_, 1, v___y_3874_);
if (v_isShared_3881_ == 0)
{
lean_ctor_set(v___x_3880_, 0, v___x_3882_);
v___x_3884_ = v___x_3880_;
goto v_reusejp_3883_;
}
else
{
lean_object* v_reuseFailAlloc_3885_; 
v_reuseFailAlloc_3885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3885_, 0, v___x_3882_);
v___x_3884_ = v_reuseFailAlloc_3885_;
goto v_reusejp_3883_;
}
v_reusejp_3883_:
{
return v___x_3884_;
}
}
}
else
{
lean_object* v_a_3888_; lean_object* v___x_3890_; uint8_t v_isShared_3891_; uint8_t v_isSharedCheck_3895_; 
lean_dec_ref(v___y_3874_);
lean_dec(v___y_3871_);
v_a_3888_ = lean_ctor_get(v___x_3878_, 0);
v_isSharedCheck_3895_ = !lean_is_exclusive(v___x_3878_);
if (v_isSharedCheck_3895_ == 0)
{
v___x_3890_ = v___x_3878_;
v_isShared_3891_ = v_isSharedCheck_3895_;
goto v_resetjp_3889_;
}
else
{
lean_inc(v_a_3888_);
lean_dec(v___x_3878_);
v___x_3890_ = lean_box(0);
v_isShared_3891_ = v_isSharedCheck_3895_;
goto v_resetjp_3889_;
}
v_resetjp_3889_:
{
lean_object* v___x_3893_; 
if (v_isShared_3891_ == 0)
{
v___x_3893_ = v___x_3890_;
goto v_reusejp_3892_;
}
else
{
lean_object* v_reuseFailAlloc_3894_; 
v_reuseFailAlloc_3894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3894_, 0, v_a_3888_);
v___x_3893_ = v_reuseFailAlloc_3894_;
goto v_reusejp_3892_;
}
v_reusejp_3892_:
{
return v___x_3893_;
}
}
}
}
v___jp_3896_:
{
lean_object* v___x_3904_; 
v___x_3904_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(v___y_3901_, v___y_3897_, v___y_3899_, v___y_3903_);
lean_dec(v___y_3903_);
lean_dec(v___y_3901_);
v___y_3871_ = v___y_3898_;
v___y_3872_ = v___y_3900_;
v___y_3873_ = v___y_3902_;
v___y_3874_ = v___x_3904_;
goto v___jp_3870_;
}
v___jp_3905_:
{
uint8_t v___x_3913_; 
v___x_3913_ = lean_nat_dec_le(v___y_3912_, v___y_3909_);
if (v___x_3913_ == 0)
{
lean_dec(v___y_3909_);
lean_inc(v___y_3912_);
v___y_3897_ = v___y_3906_;
v___y_3898_ = v___y_3907_;
v___y_3899_ = v___y_3912_;
v___y_3900_ = v___y_3908_;
v___y_3901_ = v___y_3910_;
v___y_3902_ = v___y_3911_;
v___y_3903_ = v___y_3912_;
goto v___jp_3896_;
}
else
{
v___y_3897_ = v___y_3906_;
v___y_3898_ = v___y_3907_;
v___y_3899_ = v___y_3912_;
v___y_3900_ = v___y_3908_;
v___y_3901_ = v___y_3910_;
v___y_3902_ = v___y_3911_;
v___y_3903_ = v___y_3909_;
goto v___jp_3896_;
}
}
v___jp_3914_:
{
size_t v_sz_3916_; size_t v___x_3917_; lean_object* v___x_3918_; 
v_sz_3916_ = lean_array_size(v___y_3915_);
v___x_3917_ = ((size_t)0ULL);
v___x_3918_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__0(v_sz_3916_, v___x_3917_, v___y_3915_);
if (lean_obj_tag(v___x_3918_) == 0)
{
lean_object* v___x_3919_; 
lean_dec_ref(v_head_3860_);
lean_dec(v_stx_3857_);
v___x_3919_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Translate_elabArgStx_spec__0___redArg();
return v___x_3919_;
}
else
{
lean_object* v_val_3920_; lean_object* v_fileName_3921_; lean_object* v_fileMap_3922_; lean_object* v_options_3923_; lean_object* v_currRecDepth_3924_; lean_object* v_maxRecDepth_3925_; lean_object* v_ref_3926_; lean_object* v_currNamespace_3927_; lean_object* v_openDecls_3928_; lean_object* v_initHeartbeats_3929_; lean_object* v_maxHeartbeats_3930_; lean_object* v_quotContext_3931_; lean_object* v_currMacroScope_3932_; uint8_t v_diag_3933_; lean_object* v_cancelTk_x3f_3934_; uint8_t v_suppressElabErrors_3935_; lean_object* v_inheritedTraceOptions_3936_; lean_object* v___x_3937_; size_t v_sz_3938_; lean_object* v_ref_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; 
v_val_3920_ = lean_ctor_get(v___x_3918_, 0);
lean_inc(v_val_3920_);
lean_dec_ref_known(v___x_3918_, 1);
v_fileName_3921_ = lean_ctor_get(v_a_3863_, 0);
v_fileMap_3922_ = lean_ctor_get(v_a_3863_, 1);
v_options_3923_ = lean_ctor_get(v_a_3863_, 2);
v_currRecDepth_3924_ = lean_ctor_get(v_a_3863_, 3);
v_maxRecDepth_3925_ = lean_ctor_get(v_a_3863_, 4);
v_ref_3926_ = lean_ctor_get(v_a_3863_, 5);
v_currNamespace_3927_ = lean_ctor_get(v_a_3863_, 6);
v_openDecls_3928_ = lean_ctor_get(v_a_3863_, 7);
v_initHeartbeats_3929_ = lean_ctor_get(v_a_3863_, 8);
v_maxHeartbeats_3930_ = lean_ctor_get(v_a_3863_, 9);
v_quotContext_3931_ = lean_ctor_get(v_a_3863_, 10);
v_currMacroScope_3932_ = lean_ctor_get(v_a_3863_, 11);
v_diag_3933_ = lean_ctor_get_uint8(v_a_3863_, sizeof(void*)*14);
v_cancelTk_x3f_3934_ = lean_ctor_get(v_a_3863_, 12);
v_suppressElabErrors_3935_ = lean_ctor_get_uint8(v_a_3863_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3936_ = lean_ctor_get(v_a_3863_, 13);
v___x_3937_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__0));
v_sz_3938_ = lean_array_size(v_val_3920_);
v_ref_3939_ = l_Lean_replaceRef(v_stx_3857_, v_ref_3926_);
lean_dec(v_stx_3857_);
lean_inc_ref(v_inheritedTraceOptions_3936_);
lean_inc(v_cancelTk_x3f_3934_);
lean_inc(v_currMacroScope_3932_);
lean_inc(v_quotContext_3931_);
lean_inc(v_maxHeartbeats_3930_);
lean_inc(v_initHeartbeats_3929_);
lean_inc(v_openDecls_3928_);
lean_inc(v_currNamespace_3927_);
lean_inc(v_maxRecDepth_3925_);
lean_inc(v_currRecDepth_3924_);
lean_inc_ref(v_options_3923_);
lean_inc_ref(v_fileMap_3922_);
lean_inc_ref(v_fileName_3921_);
v___x_3940_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3940_, 0, v_fileName_3921_);
lean_ctor_set(v___x_3940_, 1, v_fileMap_3922_);
lean_ctor_set(v___x_3940_, 2, v_options_3923_);
lean_ctor_set(v___x_3940_, 3, v_currRecDepth_3924_);
lean_ctor_set(v___x_3940_, 4, v_maxRecDepth_3925_);
lean_ctor_set(v___x_3940_, 5, v_ref_3939_);
lean_ctor_set(v___x_3940_, 6, v_currNamespace_3927_);
lean_ctor_set(v___x_3940_, 7, v_openDecls_3928_);
lean_ctor_set(v___x_3940_, 8, v_initHeartbeats_3929_);
lean_ctor_set(v___x_3940_, 9, v_maxHeartbeats_3930_);
lean_ctor_set(v___x_3940_, 10, v_quotContext_3931_);
lean_ctor_set(v___x_3940_, 11, v_currMacroScope_3932_);
lean_ctor_set(v___x_3940_, 12, v_cancelTk_x3f_3934_);
lean_ctor_set(v___x_3940_, 13, v_inheritedTraceOptions_3936_);
lean_ctor_set_uint8(v___x_3940_, sizeof(void*)*14, v_diag_3933_);
lean_ctor_set_uint8(v___x_3940_, sizeof(void*)*14 + 1, v_suppressElabErrors_3935_);
v___x_3941_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8(v_argNames_3858_, v_args_3859_, v_head_3860_, v_val_3920_, v_sz_3938_, v___x_3917_, v___x_3937_, v_a_3861_, v_a_3862_, v___x_3940_, v_a_3864_);
lean_dec(v_val_3920_);
if (lean_obj_tag(v___x_3941_) == 0)
{
lean_object* v_a_3942_; lean_object* v_fst_3943_; lean_object* v_snd_3944_; lean_object* v___x_3946_; uint8_t v_isShared_3947_; uint8_t v_isSharedCheck_3967_; 
v_a_3942_ = lean_ctor_get(v___x_3941_, 0);
lean_inc(v_a_3942_);
lean_dec_ref_known(v___x_3941_, 1);
v_fst_3943_ = lean_ctor_get(v_a_3942_, 0);
v_snd_3944_ = lean_ctor_get(v_a_3942_, 1);
v_isSharedCheck_3967_ = !lean_is_exclusive(v_a_3942_);
if (v_isSharedCheck_3967_ == 0)
{
v___x_3946_ = v_a_3942_;
v_isShared_3947_ = v_isSharedCheck_3967_;
goto v_resetjp_3945_;
}
else
{
lean_inc(v_snd_3944_);
lean_inc(v_fst_3943_);
lean_dec(v_a_3942_);
v___x_3946_ = lean_box(0);
v_isShared_3947_ = v_isSharedCheck_3967_;
goto v_resetjp_3945_;
}
v_resetjp_3945_:
{
lean_object* v___x_3948_; lean_object* v___x_3949_; lean_object* v___x_3951_; 
v___x_3948_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2, &lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Translate_elabReorder___closed__2);
v___x_3949_ = lean_box(0);
lean_inc(v_fst_3943_);
if (v_isShared_3947_ == 0)
{
lean_ctor_set(v___x_3946_, 1, v___x_3949_);
v___x_3951_ = v___x_3946_;
goto v_reusejp_3950_;
}
else
{
lean_object* v_reuseFailAlloc_3966_; 
v_reuseFailAlloc_3966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3966_, 0, v_fst_3943_);
lean_ctor_set(v_reuseFailAlloc_3966_, 1, v___x_3949_);
v___x_3951_ = v_reuseFailAlloc_3966_;
goto v_reusejp_3950_;
}
v_reusejp_3950_:
{
lean_object* v___x_3952_; 
v___x_3952_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg(v___x_3951_, v___x_3948_, v_a_3861_, v_a_3862_, v___x_3940_, v_a_3864_);
if (lean_obj_tag(v___x_3952_) == 0)
{
lean_object* v___x_3953_; lean_object* v___x_3954_; uint8_t v___x_3955_; 
lean_dec_ref_known(v___x_3952_, 1);
v___x_3953_ = lean_unsigned_to_nat(1u);
v___x_3954_ = lean_array_get_size(v_snd_3944_);
v___x_3955_ = lean_nat_dec_eq(v___x_3954_, v___x_3869_);
if (v___x_3955_ == 0)
{
lean_object* v___x_3956_; uint8_t v___x_3957_; 
v___x_3956_ = lean_nat_sub(v___x_3954_, v___x_3953_);
v___x_3957_ = lean_nat_dec_le(v___x_3869_, v___x_3956_);
if (v___x_3957_ == 0)
{
lean_inc(v___x_3956_);
v___y_3906_ = v_snd_3944_;
v___y_3907_ = v_fst_3943_;
v___y_3908_ = v___x_3953_;
v___y_3909_ = v___x_3956_;
v___y_3910_ = v___x_3954_;
v___y_3911_ = v___x_3940_;
v___y_3912_ = v___x_3956_;
goto v___jp_3905_;
}
else
{
v___y_3906_ = v_snd_3944_;
v___y_3907_ = v_fst_3943_;
v___y_3908_ = v___x_3953_;
v___y_3909_ = v___x_3956_;
v___y_3910_ = v___x_3954_;
v___y_3911_ = v___x_3940_;
v___y_3912_ = v___x_3869_;
goto v___jp_3905_;
}
}
else
{
v___y_3871_ = v_fst_3943_;
v___y_3872_ = v___x_3953_;
v___y_3873_ = v___x_3940_;
v___y_3874_ = v_snd_3944_;
goto v___jp_3870_;
}
}
else
{
lean_object* v_a_3958_; lean_object* v___x_3960_; uint8_t v_isShared_3961_; uint8_t v_isSharedCheck_3965_; 
lean_dec(v_snd_3944_);
lean_dec(v_fst_3943_);
lean_dec_ref_known(v___x_3940_, 14);
v_a_3958_ = lean_ctor_get(v___x_3952_, 0);
v_isSharedCheck_3965_ = !lean_is_exclusive(v___x_3952_);
if (v_isSharedCheck_3965_ == 0)
{
v___x_3960_ = v___x_3952_;
v_isShared_3961_ = v_isSharedCheck_3965_;
goto v_resetjp_3959_;
}
else
{
lean_inc(v_a_3958_);
lean_dec(v___x_3952_);
v___x_3960_ = lean_box(0);
v_isShared_3961_ = v_isSharedCheck_3965_;
goto v_resetjp_3959_;
}
v_resetjp_3959_:
{
lean_object* v___x_3963_; 
if (v_isShared_3961_ == 0)
{
v___x_3963_ = v___x_3960_;
goto v_reusejp_3962_;
}
else
{
lean_object* v_reuseFailAlloc_3964_; 
v_reuseFailAlloc_3964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3964_, 0, v_a_3958_);
v___x_3963_ = v_reuseFailAlloc_3964_;
goto v_reusejp_3962_;
}
v_reusejp_3962_:
{
return v___x_3963_;
}
}
}
}
}
}
else
{
lean_object* v_a_3968_; lean_object* v___x_3970_; uint8_t v_isShared_3971_; uint8_t v_isSharedCheck_3975_; 
lean_dec_ref_known(v___x_3940_, 14);
v_a_3968_ = lean_ctor_get(v___x_3941_, 0);
v_isSharedCheck_3975_ = !lean_is_exclusive(v___x_3941_);
if (v_isSharedCheck_3975_ == 0)
{
v___x_3970_ = v___x_3941_;
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
else
{
lean_inc(v_a_3968_);
lean_dec(v___x_3941_);
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
lean_ctor_set(v_reuseFailAlloc_3974_, 0, v_a_3968_);
v___x_3973_ = v_reuseFailAlloc_3974_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
return v___x_3973_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___lam__0(lean_object* v___x_3992_, lean_object* v_val_3993_, lean_object* v_xs_3994_, lean_object* v_x_3995_, lean_object* v___y_3996_, lean_object* v___y_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_){
_start:
{
size_t v_sz_4001_; size_t v___x_4002_; lean_object* v___x_4003_; 
v_sz_4001_ = lean_array_size(v_xs_3994_);
v___x_4002_ = ((size_t)0ULL);
lean_inc_ref(v_xs_3994_);
v___x_4003_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg(v_sz_4001_, v___x_4002_, v_xs_3994_, v___y_3996_, v___y_3998_, v___y_3999_);
if (lean_obj_tag(v___x_4003_) == 0)
{
lean_object* v_a_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; 
v_a_4004_ = lean_ctor_get(v___x_4003_, 0);
lean_inc(v_a_4004_);
lean_dec_ref_known(v___x_4003_, 1);
v___x_4005_ = l_Lean_MessageData_ofExpr(v___x_3992_);
v___x_4006_ = lp_mathlib_Mathlib_Tactic_Translate_elabReorder(v_val_3993_, v_a_4004_, v_xs_3994_, v___x_4005_, v___y_3996_, v___y_3997_, v___y_3998_, v___y_3999_);
lean_dec_ref(v_xs_3994_);
lean_dec(v_a_4004_);
return v___x_4006_;
}
else
{
lean_object* v_a_4007_; lean_object* v___x_4009_; uint8_t v_isShared_4010_; uint8_t v_isSharedCheck_4014_; 
lean_dec_ref(v_xs_3994_);
lean_dec(v_val_3993_);
lean_dec_ref(v___x_3992_);
v_a_4007_ = lean_ctor_get(v___x_4003_, 0);
v_isSharedCheck_4014_ = !lean_is_exclusive(v___x_4003_);
if (v_isSharedCheck_4014_ == 0)
{
v___x_4009_ = v___x_4003_;
v_isShared_4010_ = v_isSharedCheck_4014_;
goto v_resetjp_4008_;
}
else
{
lean_inc(v_a_4007_);
lean_dec(v___x_4003_);
v___x_4009_ = lean_box(0);
v_isShared_4010_ = v_isSharedCheck_4014_;
goto v_resetjp_4008_;
}
v_resetjp_4008_:
{
lean_object* v___x_4012_; 
if (v_isShared_4010_ == 0)
{
v___x_4012_ = v___x_4009_;
goto v_reusejp_4011_;
}
else
{
lean_object* v_reuseFailAlloc_4013_; 
v_reuseFailAlloc_4013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4013_, 0, v_a_4007_);
v___x_4012_ = v_reuseFailAlloc_4013_;
goto v_reusejp_4011_;
}
v_reusejp_4011_:
{
return v___x_4012_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg___boxed(lean_object* v_args_4015_, lean_object* v_val_4016_, lean_object* v_as_x27_4017_, lean_object* v_b_4018_, lean_object* v___y_4019_, lean_object* v___y_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_){
_start:
{
lean_object* v_res_4024_; 
v_res_4024_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg(v_args_4015_, v_val_4016_, v_as_x27_4017_, v_b_4018_, v___y_4019_, v___y_4020_, v___y_4021_, v___y_4022_);
lean_dec(v___y_4022_);
lean_dec_ref(v___y_4021_);
lean_dec(v___y_4020_);
lean_dec_ref(v___y_4019_);
lean_dec(v_as_x27_4017_);
lean_dec_ref(v_args_4015_);
return v_res_4024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabReorder___boxed(lean_object* v_stx_4025_, lean_object* v_argNames_4026_, lean_object* v_args_4027_, lean_object* v_head_4028_, lean_object* v_a_4029_, lean_object* v_a_4030_, lean_object* v_a_4031_, lean_object* v_a_4032_, lean_object* v_a_4033_){
_start:
{
lean_object* v_res_4034_; 
v_res_4034_ = lp_mathlib_Mathlib_Tactic_Translate_elabReorder(v_stx_4025_, v_argNames_4026_, v_args_4027_, v_head_4028_, v_a_4029_, v_a_4030_, v_a_4031_, v_a_4032_);
lean_dec(v_a_4032_);
lean_dec_ref(v_a_4031_);
lean_dec(v_a_4030_);
lean_dec_ref(v_a_4029_);
lean_dec_ref(v_args_4027_);
lean_dec_ref(v_argNames_4026_);
return v_res_4034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8___boxed(lean_object* v_argNames_4035_, lean_object* v_args_4036_, lean_object* v_head_4037_, lean_object* v_as_4038_, lean_object* v_sz_4039_, lean_object* v_i_4040_, lean_object* v_b_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_, lean_object* v___y_4046_){
_start:
{
size_t v_sz_boxed_4047_; size_t v_i_boxed_4048_; lean_object* v_res_4049_; 
v_sz_boxed_4047_ = lean_unbox_usize(v_sz_4039_);
lean_dec(v_sz_4039_);
v_i_boxed_4048_ = lean_unbox_usize(v_i_4040_);
lean_dec(v_i_4040_);
v_res_4049_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__8(v_argNames_4035_, v_args_4036_, v_head_4037_, v_as_4038_, v_sz_boxed_4047_, v_i_boxed_4048_, v_b_4041_, v___y_4042_, v___y_4043_, v___y_4044_, v___y_4045_);
lean_dec(v___y_4045_);
lean_dec_ref(v___y_4044_);
lean_dec(v___y_4043_);
lean_dec_ref(v___y_4042_);
lean_dec_ref(v_as_4038_);
lean_dec_ref(v_args_4036_);
lean_dec_ref(v_argNames_4035_);
return v_res_4049_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1(lean_object* v_00_u03b2_4050_, lean_object* v_a_4051_, lean_object* v_x_4052_){
_start:
{
uint8_t v___x_4053_; 
v___x_4053_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___redArg(v_a_4051_, v_x_4052_);
return v___x_4053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1___boxed(lean_object* v_00_u03b2_4054_, lean_object* v_a_4055_, lean_object* v_x_4056_){
_start:
{
uint8_t v_res_4057_; lean_object* v_r_4058_; 
v_res_4057_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Mathlib_Tactic_Translate_elabReorder_spec__1(v_00_u03b2_4054_, v_a_4055_, v_x_4056_);
lean_dec(v_x_4056_);
lean_dec(v_a_4055_);
v_r_4058_ = lean_box(v_res_4057_);
return v_r_4058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2(lean_object* v_00_u03b2_4059_, lean_object* v_data_4060_){
_start:
{
lean_object* v___x_4061_; 
v___x_4061_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2___redArg(v_data_4060_);
return v___x_4061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5(size_t v_sz_4062_, size_t v_i_4063_, lean_object* v_bs_4064_, lean_object* v___y_4065_, lean_object* v___y_4066_, lean_object* v___y_4067_, lean_object* v___y_4068_){
_start:
{
lean_object* v___x_4070_; 
v___x_4070_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___redArg(v_sz_4062_, v_i_4063_, v_bs_4064_, v___y_4065_, v___y_4067_, v___y_4068_);
return v___x_4070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5___boxed(lean_object* v_sz_4071_, lean_object* v_i_4072_, lean_object* v_bs_4073_, lean_object* v___y_4074_, lean_object* v___y_4075_, lean_object* v___y_4076_, lean_object* v___y_4077_, lean_object* v___y_4078_){
_start:
{
size_t v_sz_boxed_4079_; size_t v_i_boxed_4080_; lean_object* v_res_4081_; 
v_sz_boxed_4079_ = lean_unbox_usize(v_sz_4071_);
lean_dec(v_sz_4071_);
v_i_boxed_4080_ = lean_unbox_usize(v_i_4072_);
lean_dec(v_i_4072_);
v_res_4081_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Translate_elabReorder_spec__5(v_sz_boxed_4079_, v_i_boxed_4080_, v_bs_4073_, v___y_4074_, v___y_4075_, v___y_4076_, v___y_4077_);
lean_dec(v___y_4077_);
lean_dec_ref(v___y_4076_);
lean_dec(v___y_4075_);
lean_dec_ref(v___y_4074_);
return v_res_4081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7(lean_object* v_args_4082_, lean_object* v_val_4083_, lean_object* v_as_4084_, lean_object* v_as_x27_4085_, lean_object* v_b_4086_, lean_object* v_a_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_, lean_object* v___y_4090_, lean_object* v___y_4091_){
_start:
{
lean_object* v___x_4093_; 
v___x_4093_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___redArg(v_args_4082_, v_val_4083_, v_as_x27_4085_, v_b_4086_, v___y_4088_, v___y_4089_, v___y_4090_, v___y_4091_);
return v___x_4093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7___boxed(lean_object* v_args_4094_, lean_object* v_val_4095_, lean_object* v_as_4096_, lean_object* v_as_x27_4097_, lean_object* v_b_4098_, lean_object* v_a_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_){
_start:
{
lean_object* v_res_4105_; 
v_res_4105_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Translate_elabReorder_spec__7(v_args_4094_, v_val_4095_, v_as_4096_, v_as_x27_4097_, v_b_4098_, v_a_4099_, v___y_4100_, v___y_4101_, v___y_4102_, v___y_4103_);
lean_dec(v___y_4103_);
lean_dec_ref(v___y_4102_);
lean_dec(v___y_4101_);
lean_dec_ref(v___y_4100_);
lean_dec(v_as_x27_4097_);
lean_dec(v_as_4096_);
lean_dec_ref(v_args_4094_);
return v_res_4105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9(lean_object* v_inst_4106_, lean_object* v_R_4107_, lean_object* v_a_4108_, lean_object* v_b_4109_, lean_object* v_c_4110_, lean_object* v___y_4111_, lean_object* v___y_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_){
_start:
{
lean_object* v___x_4116_; 
v___x_4116_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___redArg(v_a_4108_, v_b_4109_, v___y_4111_, v___y_4112_, v___y_4113_, v___y_4114_);
return v___x_4116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9___boxed(lean_object* v_inst_4117_, lean_object* v_R_4118_, lean_object* v_a_4119_, lean_object* v_b_4120_, lean_object* v_c_4121_, lean_object* v___y_4122_, lean_object* v___y_4123_, lean_object* v___y_4124_, lean_object* v___y_4125_, lean_object* v___y_4126_){
_start:
{
lean_object* v_res_4127_; 
v_res_4127_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__9(v_inst_4117_, v_R_4118_, v_a_4119_, v_b_4120_, v_c_4121_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_);
lean_dec(v___y_4125_);
lean_dec_ref(v___y_4124_);
lean_dec(v___y_4123_);
lean_dec_ref(v___y_4122_);
return v_res_4127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10(lean_object* v_upperBound_4128_, lean_object* v___y_4129_, lean_object* v_inst_4130_, lean_object* v_R_4131_, lean_object* v_a_4132_, lean_object* v_b_4133_, lean_object* v_c_4134_, lean_object* v___y_4135_, lean_object* v___y_4136_, lean_object* v___y_4137_, lean_object* v___y_4138_){
_start:
{
lean_object* v___x_4140_; 
v___x_4140_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___redArg(v_upperBound_4128_, v___y_4129_, v_a_4132_, v_b_4133_, v___y_4135_, v___y_4136_, v___y_4137_, v___y_4138_);
return v___x_4140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10___boxed(lean_object* v_upperBound_4141_, lean_object* v___y_4142_, lean_object* v_inst_4143_, lean_object* v_R_4144_, lean_object* v_a_4145_, lean_object* v_b_4146_, lean_object* v_c_4147_, lean_object* v___y_4148_, lean_object* v___y_4149_, lean_object* v___y_4150_, lean_object* v___y_4151_, lean_object* v___y_4152_){
_start:
{
lean_object* v_res_4153_; 
v_res_4153_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00Mathlib_Tactic_Translate_elabReorder_spec__10(v_upperBound_4141_, v___y_4142_, v_inst_4143_, v_R_4144_, v_a_4145_, v_b_4146_, v_c_4147_, v___y_4148_, v___y_4149_, v___y_4150_, v___y_4151_);
lean_dec(v___y_4151_);
lean_dec_ref(v___y_4150_);
lean_dec(v___y_4149_);
lean_dec_ref(v___y_4148_);
lean_dec_ref(v___y_4142_);
lean_dec(v_upperBound_4141_);
return v_res_4153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11(lean_object* v_n_4154_, lean_object* v_as_4155_, lean_object* v_lo_4156_, lean_object* v_hi_4157_, lean_object* v_w_4158_, lean_object* v_hlo_4159_, lean_object* v_hhi_4160_){
_start:
{
lean_object* v___x_4161_; 
v___x_4161_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___redArg(v_n_4154_, v_as_4155_, v_lo_4156_, v_hi_4157_);
return v___x_4161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11___boxed(lean_object* v_n_4162_, lean_object* v_as_4163_, lean_object* v_lo_4164_, lean_object* v_hi_4165_, lean_object* v_w_4166_, lean_object* v_hlo_4167_, lean_object* v_hhi_4168_){
_start:
{
lean_object* v_res_4169_; 
v_res_4169_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11(v_n_4162_, v_as_4163_, v_lo_4164_, v_hi_4165_, v_w_4166_, v_hlo_4167_, v_hhi_4168_);
lean_dec(v_hi_4165_);
lean_dec(v_n_4162_);
return v_res_4169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2(lean_object* v_00_u03b2_4170_, lean_object* v_i_4171_, lean_object* v_source_4172_, lean_object* v_target_4173_){
_start:
{
lean_object* v___x_4174_; 
v___x_4174_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2___redArg(v_i_4171_, v_source_4172_, v_target_4173_);
return v___x_4174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12(lean_object* v_n_4175_, lean_object* v_lo_4176_, lean_object* v_hi_4177_, lean_object* v_hhi_4178_, lean_object* v_pivot_4179_, lean_object* v_as_4180_, lean_object* v_i_4181_, lean_object* v_k_4182_, lean_object* v_ilo_4183_, lean_object* v_ik_4184_, lean_object* v_w_4185_){
_start:
{
lean_object* v___x_4186_; 
v___x_4186_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___redArg(v_hi_4177_, v_pivot_4179_, v_as_4180_, v_i_4181_, v_k_4182_);
return v___x_4186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12___boxed(lean_object* v_n_4187_, lean_object* v_lo_4188_, lean_object* v_hi_4189_, lean_object* v_hhi_4190_, lean_object* v_pivot_4191_, lean_object* v_as_4192_, lean_object* v_i_4193_, lean_object* v_k_4194_, lean_object* v_ilo_4195_, lean_object* v_ik_4196_, lean_object* v_w_4197_){
_start:
{
lean_object* v_res_4198_; 
v_res_4198_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Translate_elabReorder_spec__11_spec__12(v_n_4187_, v_lo_4188_, v_hi_4189_, v_hhi_4190_, v_pivot_4191_, v_as_4192_, v_i_4193_, v_k_4194_, v_ilo_4195_, v_ik_4196_, v_w_4197_);
lean_dec_ref(v_pivot_4191_);
lean_dec(v_hi_4189_);
lean_dec(v_lo_4188_);
lean_dec(v_n_4187_);
return v_res_4198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4(lean_object* v_00_u03b2_4199_, lean_object* v_x_4200_, lean_object* v_x_4201_){
_start:
{
lean_object* v___x_4202_; 
v___x_4202_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Mathlib_Tactic_Translate_elabReorder_spec__2_spec__2_spec__4___redArg(v_x_4200_, v_x_4201_);
return v___x_4202_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_Reorder(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_Reorder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Lean_Parser_Category_translateReorder = _init_lp_mathlib_Lean_Parser_Category_translateReorder();
lean_mark_persistent(lp_mathlib_Lean_Parser_Category_translateReorder);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_Reorder(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_Reorder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_Reorder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_Reorder(builtin);
}
#ifdef __cplusplus
}
#endif
