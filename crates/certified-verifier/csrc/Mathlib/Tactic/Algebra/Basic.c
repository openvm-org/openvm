// Lean compiler output
// Module: Mathlib.Tactic.Algebra.Basic
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.NormCast public import Mathlib.Tactic.Algebra.Lemmas public import Mathlib.Tactic.Ring.RingNF public import Mathlib.Algebra.Algebra.Basic
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
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompute(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_evalPow_u2081___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Level_max___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare(lean_object*, lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_eq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_cmp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_evalAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_evalMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_NormCast_pushCastExt;
lean_object* l_Lean_Meta_SimpExtension_getTheorems___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_simpGlobalConfig;
lean_object* l_Lean_Meta_SimpTheorems_add(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_evalNeg___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_evalInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_Simp_mkDefaultMethodsCore(lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Qq_getLevelQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LOption_toOption___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Field"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__0_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__1_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 149, 205, 214, 52, 248, 155, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__8_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__11_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "DFunLike"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "coe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__13_value),LEAN_SCALAR_PTR_LITERAL(190, 234, 111, 20, 138, 131, 216, 106)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__14_value),LEAN_SCALAR_PTR_LITERAL(219, 234, 187, 54, 1, 195, 172, 225)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "RingHom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__19_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__21_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "instFunLike"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(173, 70, 168, 153, 92, 253, 144, 139)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__26_value),LEAN_SCALAR_PTR_LITERAL(138, 189, 49, 252, 33, 48, 247, 28)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__28_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__29_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__33_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__34_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__36_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__37_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "instMulZeroClassOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__39_value),LEAN_SCALAR_PTR_LITERAL(31, 133, 13, 57, 152, 228, 72, 248)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNat_eq_rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__43_value),LEAN_SCALAR_PTR_LITERAL(104, 50, 29, 181, 93, 30, 207, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_zero_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__45_value),LEAN_SCALAR_PTR_LITERAL(112, 178, 123, 211, 242, 68, 81, 60)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__47_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__49_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__51_value),LEAN_SCALAR_PTR_LITERAL(78, 130, 181, 61, 179, 129, 164, 15)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__52_value),LEAN_SCALAR_PTR_LITERAL(184, 164, 171, 191, 166, 224, 196, 206)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "isInt_negOfNat_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__54_value),LEAN_SCALAR_PTR_LITERAL(7, 110, 177, 196, 43, 244, 135, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__56_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__57_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__58_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__58_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__59_value),LEAN_SCALAR_PTR_LITERAL(121, 136, 96, 129, 245, 140, 119, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__61_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "IsNNRat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__63_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "den_nz"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__61_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__62_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__63_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__64_value),LEAN_SCALAR_PTR_LITERAL(23, 12, 45, 45, 118, 187, 101, 38)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "isNNRat_eq_rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__66_value),LEAN_SCALAR_PTR_LITERAL(247, 212, 43, 52, 173, 255, 177, 255)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__68_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__0_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__70_value),LEAN_SCALAR_PTR_LITERAL(60, 172, 238, 141, 54, 76, 141, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsRat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__61_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__62_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__72_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__64_value),LEAN_SCALAR_PTR_LITERAL(55, 69, 72, 86, 50, 41, 73, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__75_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isRat_eq_rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__78_value),LEAN_SCALAR_PTR_LITERAL(124, 10, 207, 197, 205, 187, 45, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_natCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 63, 36, 145, 29, 243, 122, 233)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_intCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(232, 44, 81, 152, 114, 235, 225, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_ratCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__4_value),LEAN_SCALAR_PTR_LITERAL(229, 236, 145, 145, 19, 218, 3, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__5_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 246}, .m_size = 3, .m_capacity = 3, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static size_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 5, 146, 135, 178, 80, 190, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toAddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__4_value),LEAN_SCALAR_PTR_LITERAL(46, 99, 3, 118, 139, 55, 86, 229)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IsScalarTower"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__6_value),LEAN_SCALAR_PTR_LITERAL(172, 105, 70, 18, 181, 76, 90, 42)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__8_value),LEAN_SCALAR_PTR_LITERAL(156, 129, 94, 7, 60, 24, 41, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__10_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__14_value),LEAN_SCALAR_PTR_LITERAL(226, 107, 25, 48, 80, 144, 236, 217)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__15_value),LEAN_SCALAR_PTR_LITERAL(23, 127, 6, 115, 121, 139, 223, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instHSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__17_value),LEAN_SCALAR_PTR_LITERAL(131, 168, 246, 170, 1, 89, 173, 16)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rec"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__20_value),LEAN_SCALAR_PTR_LITERAL(86, 17, 7, 2, 233, 148, 36, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__22_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "algebraMap_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__26_value),LEAN_SCALAR_PTR_LITERAL(128, 41, 46, 7, 239, 38, 2, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toModule"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__28_value),LEAN_SCALAR_PTR_LITERAL(228, 0, 138, 108, 112, 125, 203, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__30_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__31_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "add_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(189, 137, 222, 71, 237, 157, 94, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "add_algebraMap_isNat_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(138, 36, 164, 133, 141, 118, 113, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "of_eq_true"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(180, 216, 190, 52, 49, 30, 207, 178)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(157, 40, 198, 234, 16, 168, 79, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "congrArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(188, 17, 22, 243, 206, 91, 171, 36)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "symm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__12_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(220, 149, 144, 59, 77, 93, 25, 217)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "map_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__24_value),LEAN_SCALAR_PTR_LITERAL(24, 78, 225, 128, 167, 183, 102, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "NonUnitalRingHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toMulHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(63, 118, 11, 118, 99, 38, 4, 187)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__27_value),LEAN_SCALAR_PTR_LITERAL(139, 182, 57, 126, 109, 186, 83, 228)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "toNonUnitalNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__30_value),LEAN_SCALAR_PTR_LITERAL(53, 198, 158, 55, 3, 155, 15, 156)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "RingHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toNonUnitalRingHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__32_value),LEAN_SCALAR_PTR_LITERAL(197, 19, 195, 71, 82, 99, 107, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__33_value),LEAN_SCALAR_PTR_LITERAL(80, 72, 89, 182, 167, 7, 109, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "instRingHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__16_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__35_value),LEAN_SCALAR_PTR_LITERAL(0, 17, 89, 7, 99, 205, 182, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_self"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__37_value),LEAN_SCALAR_PTR_LITERAL(224, 148, 98, 216, 254, 239, 13, 169)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__38_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "cast_zero_smul_eq_zero_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(95, 39, 56, 71, 117, 254, 218, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "cast_smul_eq_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(124, 127, 142, 82, 69, 82, 139, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "neg_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(165, 121, 124, 40, 59, 46, 91, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "pow_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 209, 105, 41, 80, 230, 83, 2)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "inv_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 252, 165, 204, 176, 244, 119, 175)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isOne_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 99, 208, 221, 99, 60, 217, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 205, 87, 23, 59, 10, 241, 25)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__3_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__4_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__6_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__10_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__11_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 219, 71, 246, 148, 114, 208, 126)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__16_value),LEAN_SCALAR_PTR_LITERAL(56, 82, 209, 127, 228, 246, 91, 162)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__18_value),LEAN_SCALAR_PTR_LITERAL(158, 72, 22, 158, 153, 136, 145, 225)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "congrFun'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__20_value),LEAN_SCALAR_PTR_LITERAL(219, 239, 156, 219, 118, 185, 235, 192)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__22_value),LEAN_SCALAR_PTR_LITERAL(19, 237, 167, 212, 100, 179, 19, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__24_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "eq_1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 205, 87, 23, 59, 10, 241, 25)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__26_value),LEAN_SCALAR_PTR_LITERAL(99, 145, 121, 11, 126, 45, 91, 131)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "add_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__28_value),LEAN_SCALAR_PTR_LITERAL(188, 217, 59, 250, 243, 223, 216, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__30_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__31_value),LEAN_SCALAR_PTR_LITERAL(75, 217, 102, 131, 1, 241, 19, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toAddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__13_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__33_value),LEAN_SCALAR_PTR_LITERAL(231, 178, 143, 16, 208, 220, 52, 201)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "map_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__35_value),LEAN_SCALAR_PTR_LITERAL(230, 112, 180, 80, 118, 36, 255, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "MonoidHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toOneHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__37_value),LEAN_SCALAR_PTR_LITERAL(49, 160, 97, 54, 138, 209, 184, 199)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__38_value),LEAN_SCALAR_PTR_LITERAL(87, 83, 225, 96, 10, 131, 188, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MulOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMulOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__40_value),LEAN_SCALAR_PTR_LITERAL(68, 11, 146, 104, 134, 210, 88, 211)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__41_value),LEAN_SCALAR_PTR_LITERAL(137, 54, 46, 26, 85, 32, 178, 134)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "MulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toMulOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__43_value),LEAN_SCALAR_PTR_LITERAL(175, 32, 159, 62, 158, 163, 7, 102)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__44_value),LEAN_SCALAR_PTR_LITERAL(132, 127, 127, 138, 109, 6, 220, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toMulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__46_value),LEAN_SCALAR_PTR_LITERAL(76, 234, 170, 61, 33, 254, 89, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "MonoidWithZeroHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toMonoidHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__48_value),LEAN_SCALAR_PTR_LITERAL(135, 87, 79, 128, 160, 208, 165, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__49_value),LEAN_SCALAR_PTR_LITERAL(73, 42, 52, 64, 26, 165, 108, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "toMonoidWithZeroHomClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__32_value),LEAN_SCALAR_PTR_LITERAL(197, 19, 195, 71, 82, 99, 107, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__51_value),LEAN_SCALAR_PTR_LITERAL(22, 36, 108, 114, 70, 0, 234, 86)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "cast_eq_algebraMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(111, 203, 210, 132, 127, 160, 49, 148)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__6_value),LEAN_SCALAR_PTR_LITERAL(224, 111, 111, 173, 51, 211, 79, 232)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74_value),LEAN_SCALAR_PTR_LITERAL(185, 185, 111, 87, 40, 217, 230, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__6_value),LEAN_SCALAR_PTR_LITERAL(238, 225, 246, 163, 106, 75, 196, 234)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "algebraMap_eq_smul_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__9_value),LEAN_SCALAR_PTR_LITERAL(174, 203, 25, 208, 233, 244, 176, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "sub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__3_value),LEAN_SCALAR_PTR_LITERAL(203, 50, 219, 228, 204, 142, 182, 246)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__4_value),LEAN_SCALAR_PTR_LITERAL(153, 170, 154, 227, 136, 99, 108, 193)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__6_value),LEAN_SCALAR_PTR_LITERAL(155, 25, 183, 66, 31, 85, 84, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__7_value),LEAN_SCALAR_PTR_LITERAL(124, 210, 233, 157, 130, 57, 249, 157)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__9_value),LEAN_SCALAR_PTR_LITERAL(123, 91, 0, 102, 155, 93, 69, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 34, 112, 179, 66, 45, 192, 92)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__12_value),LEAN_SCALAR_PTR_LITERAL(194, 42, 228, 70, 75, 199, 255, 47)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__13_value),LEAN_SCALAR_PTR_LITERAL(89, 32, 50, 180, 214, 46, 71, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__15_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__16_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__18_value),LEAN_SCALAR_PTR_LITERAL(121, 130, 45, 212, 110, 237, 236, 233)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__19_value),LEAN_SCALAR_PTR_LITERAL(231, 253, 204, 163, 168, 77, 27, 58)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__18_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__74_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Rat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "algebra failed, algebra expressions not equal\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "algebra failed: not an equality"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 160, 187, 48, 105, 34, 141, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebra = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "algebraWith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__42_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(39, 191, 112, 1, 99, 246, 250, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 74, 108, 69, 108, 242, 205, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Algebra_algebraWith = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache(lean_object* v_u_4_, lean_object* v_A_5_, lean_object* v_sA_6_, lean_object* v_a_7_, lean_object* v_a_8_, lean_object* v_a_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_12_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_mkCache___closed__1));
v___x_13_ = lean_box(0);
lean_inc(v_u_4_);
v___x_14_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_14_, 0, v_u_4_);
lean_ctor_set(v___x_14_, 1, v___x_13_);
v___x_15_ = l_Lean_Expr_const___override(v___x_12_, v___x_14_);
lean_inc_ref(v_A_5_);
v___x_16_ = l_Lean_Expr_app___override(v___x_15_, v_A_5_);
v___x_17_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_16_, v_a_7_, v_a_8_, v_a_9_, v_a_10_);
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v_a_18_; lean_object* v___x_19_; 
v_a_18_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_a_18_);
lean_dec_ref_known(v___x_17_, 1);
v___x_19_ = lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(v_u_4_, v_A_5_, v_sA_6_, v_a_7_, v_a_8_, v_a_9_, v_a_10_);
if (lean_obj_tag(v___x_19_) == 0)
{
lean_object* v_a_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_29_; 
v_a_20_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_29_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_29_ == 0)
{
v___x_22_ = v___x_19_;
v_isShared_23_ = v_isSharedCheck_29_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_a_20_);
lean_dec(v___x_19_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_29_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_27_; 
v___x_24_ = l_Lean_LOption_toOption___redArg(v_a_18_);
v___x_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_25_, 0, v_a_20_);
lean_ctor_set(v___x_25_, 1, v___x_24_);
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 0, v___x_25_);
v___x_27_ = v___x_22_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v___x_25_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
else
{
lean_object* v_a_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_37_; 
lean_dec(v_a_18_);
v_a_30_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_37_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_37_ == 0)
{
v___x_32_ = v___x_19_;
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_a_30_);
lean_dec(v___x_19_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_35_; 
if (v_isShared_33_ == 0)
{
v___x_35_ = v___x_32_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_a_30_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
else
{
lean_object* v_a_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_45_; 
lean_dec_ref(v_sA_6_);
lean_dec_ref(v_A_5_);
lean_dec(v_u_4_);
v_a_38_ = lean_ctor_get(v___x_17_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_45_ == 0)
{
v___x_40_ = v___x_17_;
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_a_38_);
lean_dec(v___x_17_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_43_; 
if (v_isShared_41_ == 0)
{
v___x_43_ = v___x_40_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v_a_38_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_mkCache___boxed(lean_object* v_u_46_, lean_object* v_A_47_, lean_object* v_sA_48_, lean_object* v_a_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Mathlib_Tactic_Algebra_mkCache(v_u_46_, v_A_47_, v_sA_48_, v_a_49_, v_a_50_, v_a_51_, v_a_52_);
lean_dec(v_a_52_);
lean_dec_ref(v_a_51_);
lean_dec(v_a_50_);
lean_dec_ref(v_a_49_);
return v_res_54_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__31));
v___x_109_ = l_Lean_Expr_lit___override(v___x_108_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_197_ = lean_box(0);
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__76));
v___x_199_ = l_Lean_Expr_const___override(v___x_198_, v___x_197_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalCast(lean_object* v_u_206_, lean_object* v_v_207_, lean_object* v_R_208_, lean_object* v_A_209_, lean_object* v_sR_210_, lean_object* v_sA_211_, lean_object* v_sAlg_212_, lean_object* v_a_213_, lean_object* v_cR_214_, lean_object* v_cA_215_, lean_object* v_x_216_){
_start:
{
switch(lean_obj_tag(v_x_216_))
{
case 1:
{
lean_object* v_lit_217_; lean_object* v_proof_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_419_; 
lean_dec_ref(v_cA_215_);
lean_dec_ref(v_cR_214_);
v_lit_217_ = lean_ctor_get(v_x_216_, 1);
v_proof_218_ = lean_ctor_get(v_x_216_, 2);
v_isSharedCheck_419_ = !lean_is_exclusive(v_x_216_);
if (v_isSharedCheck_419_ == 0)
{
lean_object* v_unused_420_; 
v_unused_420_ = lean_ctor_get(v_x_216_, 0);
lean_dec(v_unused_420_);
v___x_220_ = v_x_216_;
v_isShared_221_ = v_isSharedCheck_419_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_proof_218_);
lean_inc(v_lit_217_);
lean_dec(v_x_216_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_419_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
if (lean_obj_tag(v_lit_217_) == 9)
{
lean_object* v_a_376_; 
v_a_376_ = lean_ctor_get(v_lit_217_, 0);
lean_inc_ref(v_a_376_);
if (lean_obj_tag(v_a_376_) == 0)
{
lean_object* v_val_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_418_; 
v_val_377_ = lean_ctor_get(v_a_376_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v_a_376_);
if (v_isSharedCheck_418_ == 0)
{
v___x_379_ = v_a_376_;
v_isShared_380_ = v_isSharedCheck_418_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_val_377_);
lean_dec(v_a_376_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_418_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_381_ = lean_unsigned_to_nat(0u);
v___x_382_ = lean_nat_dec_eq(v_val_377_, v___x_381_);
lean_dec(v_val_377_);
if (v___x_382_ == 0)
{
lean_del_object(v___x_379_);
goto v___jp_222_;
}
else
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_416_; 
lean_dec_ref_known(v_lit_217_, 1);
lean_del_object(v___x_220_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_R_208_);
lean_dec(v_u_206_);
v___x_383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_384_ = lean_box(0);
v___x_385_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_385_, 0, v_v_207_);
lean_ctor_set(v___x_385_, 1, v___x_384_);
lean_inc_ref_n(v___x_385_, 5);
v___x_386_ = l_Lean_Expr_const___override(v___x_383_, v___x_385_);
lean_inc_ref_n(v_A_209_, 5);
v___x_387_ = l_Lean_Expr_app___override(v___x_386_, v_A_209_);
v___x_388_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_389_ = l_Lean_Expr_app___override(v___x_387_, v___x_388_);
v___x_390_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_391_ = l_Lean_Expr_const___override(v___x_390_, v___x_385_);
v___x_392_ = l_Lean_Expr_app___override(v___x_391_, v_A_209_);
v___x_393_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_394_ = l_Lean_Expr_const___override(v___x_393_, v___x_385_);
v___x_395_ = l_Lean_Expr_app___override(v___x_394_, v_A_209_);
v___x_396_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_397_ = l_Lean_Expr_const___override(v___x_396_, v___x_385_);
v___x_398_ = l_Lean_Expr_app___override(v___x_397_, v_A_209_);
v___x_399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_400_ = l_Lean_Expr_const___override(v___x_399_, v___x_385_);
v___x_401_ = l_Lean_Expr_app___override(v___x_400_, v_A_209_);
lean_inc_ref(v_sA_211_);
v___x_402_ = l_Lean_Expr_app___override(v___x_401_, v_sA_211_);
v___x_403_ = l_Lean_Expr_app___override(v___x_398_, v___x_402_);
v___x_404_ = l_Lean_Expr_app___override(v___x_395_, v___x_403_);
v___x_405_ = l_Lean_Expr_app___override(v___x_392_, v___x_404_);
v___x_406_ = l_Lean_Expr_app___override(v___x_389_, v___x_405_);
v___x_407_ = lean_box(0);
v___x_408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__46));
v___x_409_ = l_Lean_Expr_const___override(v___x_408_, v___x_385_);
v___x_410_ = l_Lean_Expr_app___override(v___x_409_, v_A_209_);
v___x_411_ = l_Lean_Expr_app___override(v___x_410_, v_sA_211_);
v___x_412_ = l_Lean_Expr_app___override(v___x_411_, v_a_213_);
v___x_413_ = l_Lean_Expr_app___override(v___x_412_, v_proof_218_);
v___x_414_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_414_, 0, v___x_406_);
lean_ctor_set(v___x_414_, 1, v___x_407_);
lean_ctor_set(v___x_414_, 2, v___x_413_);
if (v_isShared_380_ == 0)
{
lean_ctor_set_tag(v___x_379_, 1);
lean_ctor_set(v___x_379_, 0, v___x_414_);
v___x_416_ = v___x_379_;
goto v_reusejp_415_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v___x_414_);
v___x_416_ = v_reuseFailAlloc_417_;
goto v_reusejp_415_;
}
v_reusejp_415_:
{
return v___x_416_;
}
}
}
}
else
{
lean_dec_ref(v_a_376_);
goto v___jp_222_;
}
}
else
{
goto v___jp_222_;
}
v___jp_222_:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v_fst_225_; lean_object* v_snd_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_375_; 
v___x_223_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_217_);
lean_inc_ref(v_sR_210_);
lean_inc_ref(v_R_208_);
lean_inc(v_u_206_);
v___x_224_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(v_u_206_, v_R_208_, v_sR_210_, v___x_223_);
v_fst_225_ = lean_ctor_get(v___x_224_, 0);
v_snd_226_ = lean_ctor_get(v___x_224_, 1);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_375_ == 0)
{
v___x_228_ = v___x_224_;
v_isShared_229_ = v_isSharedCheck_375_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_snd_226_);
lean_inc(v_fst_225_);
lean_dec(v___x_224_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_375_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_233_; 
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
v___x_231_ = lean_box(0);
lean_inc(v_v_207_);
if (v_isShared_229_ == 0)
{
lean_ctor_set_tag(v___x_228_, 1);
lean_ctor_set(v___x_228_, 1, v___x_231_);
lean_ctor_set(v___x_228_, 0, v_v_207_);
v___x_233_ = v___x_228_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_v_207_);
lean_ctor_set(v_reuseFailAlloc_374_, 1, v___x_231_);
v___x_233_ = v_reuseFailAlloc_374_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; uint8_t v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_371_; 
lean_inc_ref_n(v___x_233_, 10);
lean_inc_n(v_v_207_, 3);
v___x_234_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_234_, 0, v_v_207_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_235_, 0, v_v_207_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = l_Lean_Expr_const___override(v___x_230_, v___x_235_);
lean_inc_ref_n(v_A_209_, 17);
v___x_237_ = l_Lean_Expr_app___override(v___x_236_, v_A_209_);
v___x_238_ = l_Lean_Expr_app___override(v___x_237_, v_A_209_);
v___x_239_ = l_Lean_Expr_app___override(v___x_238_, v_A_209_);
v___x_240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
v___x_241_ = l_Lean_Expr_const___override(v___x_240_, v___x_233_);
v___x_242_ = l_Lean_Expr_app___override(v___x_241_, v_A_209_);
v___x_243_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_244_ = l_Lean_Expr_const___override(v___x_243_, v___x_233_);
v___x_245_ = l_Lean_Expr_app___override(v___x_244_, v_A_209_);
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_247_ = l_Lean_Expr_const___override(v___x_246_, v___x_233_);
v___x_248_ = l_Lean_Expr_app___override(v___x_247_, v_A_209_);
v___x_249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_250_ = l_Lean_Expr_const___override(v___x_249_, v___x_233_);
v___x_251_ = l_Lean_Expr_app___override(v___x_250_, v_A_209_);
lean_inc_ref_n(v_sA_211_, 2);
v___x_252_ = l_Lean_Expr_app___override(v___x_251_, v_sA_211_);
lean_inc_ref_n(v___x_252_, 3);
v___x_253_ = l_Lean_Expr_app___override(v___x_248_, v___x_252_);
v___x_254_ = l_Lean_Expr_app___override(v___x_245_, v___x_253_);
v___x_255_ = l_Lean_Expr_app___override(v___x_242_, v___x_254_);
v___x_256_ = l_Lean_Expr_app___override(v___x_239_, v___x_255_);
v___x_257_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_206_, 5);
v___x_258_ = l_Lean_Level_succ___override(v_u_206_);
v___x_259_ = l_Lean_Level_succ___override(v_v_207_);
lean_inc(v___x_259_);
lean_inc(v___x_258_);
v___x_260_ = l_Lean_Level_max___override(v___x_258_, v___x_259_);
v___x_261_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_259_);
lean_ctor_set(v___x_261_, 1, v___x_231_);
v___x_262_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_258_);
lean_ctor_set(v___x_262_, 1, v___x_261_);
v___x_263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_260_);
lean_ctor_set(v___x_263_, 1, v___x_262_);
v___x_264_ = l_Lean_Expr_const___override(v___x_257_, v___x_263_);
v___x_265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_266_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_266_, 0, v_u_206_);
lean_ctor_set(v___x_266_, 1, v___x_233_);
lean_inc_ref_n(v___x_266_, 3);
v___x_267_ = l_Lean_Expr_const___override(v___x_265_, v___x_266_);
lean_inc_ref_n(v_R_208_, 18);
v___x_268_ = l_Lean_Expr_app___override(v___x_267_, v_R_208_);
v___x_269_ = l_Lean_Expr_app___override(v___x_268_, v_A_209_);
v___x_270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_271_, 0, v_u_206_);
lean_ctor_set(v___x_271_, 1, v___x_231_);
lean_inc_ref_n(v___x_271_, 9);
v___x_272_ = l_Lean_Expr_const___override(v___x_270_, v___x_271_);
v___x_273_ = l_Lean_Expr_app___override(v___x_272_, v_R_208_);
v___x_274_ = l_Lean_Expr_const___override(v___x_249_, v___x_271_);
v___x_275_ = l_Lean_Expr_app___override(v___x_274_, v_R_208_);
lean_inc_ref_n(v_sR_210_, 3);
v___x_276_ = l_Lean_Expr_app___override(v___x_275_, v_sR_210_);
lean_inc_ref_n(v___x_276_, 2);
v___x_277_ = l_Lean_Expr_app___override(v___x_273_, v___x_276_);
lean_inc_ref(v___x_277_);
v___x_278_ = l_Lean_Expr_app___override(v___x_269_, v___x_277_);
v___x_279_ = l_Lean_Expr_const___override(v___x_270_, v___x_233_);
v___x_280_ = l_Lean_Expr_app___override(v___x_279_, v_A_209_);
v___x_281_ = l_Lean_Expr_app___override(v___x_280_, v___x_252_);
lean_inc_ref(v___x_281_);
v___x_282_ = l_Lean_Expr_app___override(v___x_278_, v___x_281_);
v___x_283_ = l_Lean_Expr_app___override(v___x_264_, v___x_282_);
v___x_284_ = l_Lean_Expr_app___override(v___x_283_, v_R_208_);
v___x_285_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_286_ = 0;
v___x_287_ = l_Lean_Expr_lam___override(v___x_285_, v_R_208_, v_A_209_, v___x_286_);
v___x_288_ = l_Lean_Expr_app___override(v___x_284_, v___x_287_);
v___x_289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_290_ = l_Lean_Expr_const___override(v___x_289_, v___x_266_);
v___x_291_ = l_Lean_Expr_app___override(v___x_290_, v_R_208_);
v___x_292_ = l_Lean_Expr_app___override(v___x_291_, v_A_209_);
v___x_293_ = l_Lean_Expr_app___override(v___x_292_, v___x_277_);
v___x_294_ = l_Lean_Expr_app___override(v___x_293_, v___x_281_);
v___x_295_ = l_Lean_Expr_app___override(v___x_288_, v___x_294_);
v___x_296_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_297_ = l_Lean_Expr_const___override(v___x_296_, v___x_266_);
v___x_298_ = l_Lean_Expr_app___override(v___x_297_, v_R_208_);
v___x_299_ = l_Lean_Expr_app___override(v___x_298_, v_A_209_);
v___x_300_ = l_Lean_Expr_app___override(v___x_299_, v_sR_210_);
v___x_301_ = l_Lean_Expr_app___override(v___x_300_, v___x_252_);
lean_inc_ref(v_sAlg_212_);
v___x_302_ = l_Lean_Expr_app___override(v___x_301_, v_sAlg_212_);
v___x_303_ = l_Lean_Expr_app___override(v___x_295_, v___x_302_);
v___x_304_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_304_, 0, v_u_206_);
lean_ctor_set(v___x_304_, 1, v___x_271_);
v___x_305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_305_, 0, v_u_206_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = l_Lean_Expr_const___override(v___x_230_, v___x_305_);
v___x_307_ = l_Lean_Expr_app___override(v___x_306_, v_R_208_);
v___x_308_ = l_Lean_Expr_app___override(v___x_307_, v_R_208_);
v___x_309_ = l_Lean_Expr_app___override(v___x_308_, v_R_208_);
v___x_310_ = l_Lean_Expr_const___override(v___x_240_, v___x_271_);
v___x_311_ = l_Lean_Expr_app___override(v___x_310_, v_R_208_);
v___x_312_ = l_Lean_Expr_const___override(v___x_243_, v___x_271_);
v___x_313_ = l_Lean_Expr_app___override(v___x_312_, v_R_208_);
v___x_314_ = l_Lean_Expr_const___override(v___x_246_, v___x_271_);
v___x_315_ = l_Lean_Expr_app___override(v___x_314_, v_R_208_);
v___x_316_ = l_Lean_Expr_app___override(v___x_315_, v___x_276_);
v___x_317_ = l_Lean_Expr_app___override(v___x_313_, v___x_316_);
v___x_318_ = l_Lean_Expr_app___override(v___x_311_, v___x_317_);
v___x_319_ = l_Lean_Expr_app___override(v___x_309_, v___x_318_);
lean_inc(v_fst_225_);
v___x_320_ = l_Lean_Expr_app___override(v___x_319_, v_fst_225_);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_322_ = l_Lean_Expr_const___override(v___x_321_, v___x_271_);
v___x_323_ = l_Lean_Expr_app___override(v___x_322_, v_R_208_);
v___x_324_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_325_ = l_Lean_Expr_app___override(v___x_323_, v___x_324_);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_327_ = l_Lean_Expr_const___override(v___x_326_, v___x_271_);
v___x_328_ = l_Lean_Expr_app___override(v___x_327_, v_R_208_);
v___x_329_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_330_ = l_Lean_Expr_const___override(v___x_329_, v___x_271_);
v___x_331_ = l_Lean_Expr_app___override(v___x_330_, v_R_208_);
v___x_332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_333_ = l_Lean_Expr_const___override(v___x_332_, v___x_271_);
v___x_334_ = l_Lean_Expr_app___override(v___x_333_, v_R_208_);
v___x_335_ = l_Lean_Expr_app___override(v___x_334_, v___x_276_);
v___x_336_ = l_Lean_Expr_app___override(v___x_331_, v___x_335_);
v___x_337_ = l_Lean_Expr_app___override(v___x_328_, v___x_336_);
v___x_338_ = l_Lean_Expr_app___override(v___x_325_, v___x_337_);
v___x_339_ = l_Lean_Expr_app___override(v___x_320_, v___x_338_);
lean_inc_ref(v___x_339_);
v___x_340_ = l_Lean_Expr_app___override(v___x_303_, v___x_339_);
lean_inc_ref_n(v___x_340_, 2);
v___x_341_ = l_Lean_Expr_app___override(v___x_256_, v___x_340_);
v___x_342_ = l_Lean_Expr_const___override(v___x_321_, v___x_233_);
v___x_343_ = l_Lean_Expr_app___override(v___x_342_, v_A_209_);
v___x_344_ = l_Lean_Expr_app___override(v___x_343_, v___x_324_);
v___x_345_ = l_Lean_Expr_const___override(v___x_326_, v___x_233_);
v___x_346_ = l_Lean_Expr_app___override(v___x_345_, v_A_209_);
v___x_347_ = l_Lean_Expr_const___override(v___x_329_, v___x_233_);
v___x_348_ = l_Lean_Expr_app___override(v___x_347_, v_A_209_);
v___x_349_ = l_Lean_Expr_const___override(v___x_332_, v___x_233_);
v___x_350_ = l_Lean_Expr_app___override(v___x_349_, v_A_209_);
v___x_351_ = l_Lean_Expr_app___override(v___x_350_, v___x_252_);
v___x_352_ = l_Lean_Expr_app___override(v___x_348_, v___x_351_);
v___x_353_ = l_Lean_Expr_app___override(v___x_346_, v___x_352_);
v___x_354_ = l_Lean_Expr_app___override(v___x_344_, v___x_353_);
v___x_355_ = l_Lean_Expr_app___override(v___x_341_, v___x_354_);
v___x_356_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_206_, v_R_208_, v_sR_210_, v_fst_225_, v_snd_226_);
v___x_357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_339_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v___x_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_340_);
lean_ctor_set(v___x_358_, 1, v___x_357_);
v___x_359_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_v_207_, v_A_209_, v_sA_211_, v___x_340_, v___x_358_);
v___x_360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__44));
v___x_361_ = l_Lean_Expr_const___override(v___x_360_, v___x_266_);
v___x_362_ = l_Lean_Expr_app___override(v___x_361_, v_R_208_);
v___x_363_ = l_Lean_Expr_app___override(v___x_362_, v_A_209_);
v___x_364_ = l_Lean_Expr_app___override(v___x_363_, v_sR_210_);
v___x_365_ = l_Lean_Expr_app___override(v___x_364_, v_sA_211_);
v___x_366_ = l_Lean_Expr_app___override(v___x_365_, v_sAlg_212_);
v___x_367_ = l_Lean_Expr_app___override(v___x_366_, v_a_213_);
v___x_368_ = l_Lean_Expr_app___override(v___x_367_, v_lit_217_);
v___x_369_ = l_Lean_Expr_app___override(v___x_368_, v_proof_218_);
if (v_isShared_221_ == 0)
{
lean_ctor_set_tag(v___x_220_, 0);
lean_ctor_set(v___x_220_, 2, v___x_369_);
lean_ctor_set(v___x_220_, 1, v___x_359_);
lean_ctor_set(v___x_220_, 0, v___x_355_);
v___x_371_ = v___x_220_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v___x_355_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v___x_359_);
lean_ctor_set(v_reuseFailAlloc_373_, 2, v___x_369_);
v___x_371_ = v_reuseFailAlloc_373_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___x_372_; 
v___x_372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
return v___x_372_;
}
}
}
}
}
}
case 2:
{
lean_object* v_toCache_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_620_; 
v_toCache_421_ = lean_ctor_get(v_cR_214_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v_cR_214_);
if (v_isSharedCheck_620_ == 0)
{
lean_object* v_unused_621_; 
v_unused_621_ = lean_ctor_get(v_cR_214_, 1);
lean_dec(v_unused_621_);
v___x_423_ = v_cR_214_;
v_isShared_424_ = v_isSharedCheck_620_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_toCache_421_);
lean_dec(v_cR_214_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_620_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v_r_u03b1_425_; 
v_r_u03b1_425_ = lean_ctor_get(v_toCache_421_, 0);
lean_inc(v_r_u03b1_425_);
lean_dec_ref(v_toCache_421_);
if (lean_obj_tag(v_r_u03b1_425_) == 1)
{
lean_object* v_toCache_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_617_; 
v_toCache_426_ = lean_ctor_get(v_cA_215_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v_cA_215_);
if (v_isSharedCheck_617_ == 0)
{
lean_object* v_unused_618_; 
v_unused_618_ = lean_ctor_get(v_cA_215_, 1);
lean_dec(v_unused_618_);
v___x_428_ = v_cA_215_;
v_isShared_429_ = v_isSharedCheck_617_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_toCache_426_);
lean_dec(v_cA_215_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_617_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v_r_u03b1_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_614_; 
v_r_u03b1_430_ = lean_ctor_get(v_toCache_426_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v_toCache_426_);
if (v_isSharedCheck_614_ == 0)
{
lean_object* v_unused_615_; lean_object* v_unused_616_; 
v_unused_615_ = lean_ctor_get(v_toCache_426_, 2);
lean_dec(v_unused_615_);
v_unused_616_ = lean_ctor_get(v_toCache_426_, 1);
lean_dec(v_unused_616_);
v___x_432_ = v_toCache_426_;
v_isShared_433_ = v_isSharedCheck_614_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_r_u03b1_430_);
lean_dec(v_toCache_426_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_614_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
if (lean_obj_tag(v_r_u03b1_430_) == 1)
{
lean_object* v_lit_434_; lean_object* v_proof_435_; lean_object* v_val_436_; lean_object* v_val_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_612_; 
v_lit_434_ = lean_ctor_get(v_x_216_, 1);
lean_inc_ref(v_lit_434_);
v_proof_435_ = lean_ctor_get(v_x_216_, 2);
lean_inc_ref(v_proof_435_);
lean_dec_ref_known(v_x_216_, 3);
v_val_436_ = lean_ctor_get(v_r_u03b1_425_, 0);
lean_inc(v_val_436_);
lean_dec_ref_known(v_r_u03b1_425_, 1);
v_val_437_ = lean_ctor_get(v_r_u03b1_430_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v_r_u03b1_430_);
if (v_isSharedCheck_612_ == 0)
{
v___x_439_ = v_r_u03b1_430_;
v_isShared_440_ = v_isSharedCheck_612_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_val_437_);
lean_dec(v_r_u03b1_430_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_612_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_444_; 
lean_inc_n(v_u_206_, 2);
v___x_441_ = l_Lean_Level_succ___override(v_u_206_);
v___x_442_ = lean_box(0);
if (v_isShared_429_ == 0)
{
lean_ctor_set_tag(v___x_428_, 1);
lean_ctor_set(v___x_428_, 1, v___x_442_);
lean_ctor_set(v___x_428_, 0, v_u_206_);
v___x_444_ = v___x_428_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_u_206_);
lean_ctor_set(v_reuseFailAlloc_611_, 1, v___x_442_);
v___x_444_ = v_reuseFailAlloc_611_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_449_; 
v___x_445_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__48));
lean_inc_ref(v___x_444_);
v___x_446_ = l_Lean_Expr_const___override(v___x_445_, v___x_444_);
v___x_447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50));
lean_inc(v___x_441_);
if (v_isShared_424_ == 0)
{
lean_ctor_set_tag(v___x_423_, 1);
lean_ctor_set(v___x_423_, 1, v___x_442_);
lean_ctor_set(v___x_423_, 0, v___x_441_);
v___x_449_ = v___x_423_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_441_);
lean_ctor_set(v_reuseFailAlloc_610_, 1, v___x_442_);
v___x_449_ = v_reuseFailAlloc_610_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v_fst_460_; lean_object* v_snd_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_609_; 
v___x_450_ = l_Lean_Expr_const___override(v___x_447_, v___x_449_);
lean_inc_ref_n(v_R_208_, 3);
v___x_451_ = l_Lean_Expr_app___override(v___x_446_, v_R_208_);
v___x_452_ = l_Lean_Expr_app___override(v___x_450_, v___x_451_);
v___x_453_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__53));
lean_inc_ref(v___x_444_);
v___x_454_ = l_Lean_Expr_const___override(v___x_453_, v___x_444_);
v___x_455_ = l_Lean_Expr_app___override(v___x_454_, v_R_208_);
lean_inc(v_val_436_);
v___x_456_ = l_Lean_Expr_app___override(v___x_455_, v_val_436_);
v___x_457_ = l_Lean_Expr_app___override(v___x_452_, v___x_456_);
v___x_458_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_434_);
lean_inc(v_u_206_);
v___x_459_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(v_u_206_, v_R_208_, v___x_457_, v___x_458_);
v_fst_460_ = lean_ctor_get(v___x_459_, 0);
v_snd_461_ = lean_ctor_get(v___x_459_, 1);
v_isSharedCheck_609_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_609_ == 0)
{
v___x_463_ = v___x_459_;
v_isShared_464_ = v_isSharedCheck_609_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_snd_461_);
lean_inc(v_fst_460_);
lean_dec(v___x_459_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_609_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_465_; lean_object* v___x_467_; 
v___x_465_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
lean_inc(v_v_207_);
if (v_isShared_464_ == 0)
{
lean_ctor_set_tag(v___x_463_, 1);
lean_ctor_set(v___x_463_, 1, v___x_442_);
lean_ctor_set(v___x_463_, 0, v_v_207_);
v___x_467_ = v___x_463_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_v_207_);
lean_ctor_set(v_reuseFailAlloc_608_, 1, v___x_442_);
v___x_467_ = v_reuseFailAlloc_608_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_603_; 
lean_inc_ref_n(v___x_467_, 10);
lean_inc_n(v_v_207_, 3);
v___x_468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_468_, 0, v_v_207_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
v___x_469_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_469_, 0, v_v_207_);
lean_ctor_set(v___x_469_, 1, v___x_468_);
v___x_470_ = l_Lean_Expr_const___override(v___x_465_, v___x_469_);
lean_inc_ref_n(v_A_209_, 17);
v___x_471_ = l_Lean_Expr_app___override(v___x_470_, v_A_209_);
v___x_472_ = l_Lean_Expr_app___override(v___x_471_, v_A_209_);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v_A_209_);
v___x_474_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
v___x_475_ = l_Lean_Expr_const___override(v___x_474_, v___x_467_);
v___x_476_ = l_Lean_Expr_app___override(v___x_475_, v_A_209_);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_478_ = l_Lean_Expr_const___override(v___x_477_, v___x_467_);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_A_209_);
v___x_480_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_481_ = l_Lean_Expr_const___override(v___x_480_, v___x_467_);
v___x_482_ = l_Lean_Expr_app___override(v___x_481_, v_A_209_);
v___x_483_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_484_ = l_Lean_Expr_const___override(v___x_483_, v___x_467_);
v___x_485_ = l_Lean_Expr_app___override(v___x_484_, v_A_209_);
lean_inc_ref(v_sA_211_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_sA_211_);
lean_inc_ref_n(v___x_486_, 3);
v___x_487_ = l_Lean_Expr_app___override(v___x_482_, v___x_486_);
v___x_488_ = l_Lean_Expr_app___override(v___x_479_, v___x_487_);
v___x_489_ = l_Lean_Expr_app___override(v___x_476_, v___x_488_);
v___x_490_ = l_Lean_Expr_app___override(v___x_473_, v___x_489_);
v___x_491_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
v___x_492_ = l_Lean_Level_succ___override(v_v_207_);
lean_inc(v___x_492_);
lean_inc(v___x_441_);
v___x_493_ = l_Lean_Level_max___override(v___x_441_, v___x_492_);
v___x_494_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_492_);
lean_ctor_set(v___x_494_, 1, v___x_442_);
v___x_495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_495_, 0, v___x_441_);
lean_ctor_set(v___x_495_, 1, v___x_494_);
v___x_496_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_496_, 0, v___x_493_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = l_Lean_Expr_const___override(v___x_491_, v___x_496_);
v___x_498_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_n(v_u_206_, 3);
v___x_499_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_499_, 0, v_u_206_);
lean_ctor_set(v___x_499_, 1, v___x_467_);
lean_inc_ref_n(v___x_499_, 3);
v___x_500_ = l_Lean_Expr_const___override(v___x_498_, v___x_499_);
lean_inc_ref_n(v_R_208_, 18);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v_R_208_);
v___x_502_ = l_Lean_Expr_app___override(v___x_501_, v_A_209_);
v___x_503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref_n(v___x_444_, 9);
v___x_504_ = l_Lean_Expr_const___override(v___x_503_, v___x_444_);
v___x_505_ = l_Lean_Expr_app___override(v___x_504_, v_R_208_);
v___x_506_ = l_Lean_Expr_const___override(v___x_483_, v___x_444_);
v___x_507_ = l_Lean_Expr_app___override(v___x_506_, v_R_208_);
lean_inc_ref_n(v_sR_210_, 2);
v___x_508_ = l_Lean_Expr_app___override(v___x_507_, v_sR_210_);
lean_inc_ref_n(v___x_508_, 2);
v___x_509_ = l_Lean_Expr_app___override(v___x_505_, v___x_508_);
lean_inc_ref(v___x_509_);
v___x_510_ = l_Lean_Expr_app___override(v___x_502_, v___x_509_);
v___x_511_ = l_Lean_Expr_const___override(v___x_503_, v___x_467_);
v___x_512_ = l_Lean_Expr_app___override(v___x_511_, v_A_209_);
v___x_513_ = l_Lean_Expr_app___override(v___x_512_, v___x_486_);
lean_inc_ref(v___x_513_);
v___x_514_ = l_Lean_Expr_app___override(v___x_510_, v___x_513_);
v___x_515_ = l_Lean_Expr_app___override(v___x_497_, v___x_514_);
v___x_516_ = l_Lean_Expr_app___override(v___x_515_, v_R_208_);
v___x_517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_518_ = 0;
v___x_519_ = l_Lean_Expr_lam___override(v___x_517_, v_R_208_, v_A_209_, v___x_518_);
v___x_520_ = l_Lean_Expr_app___override(v___x_516_, v___x_519_);
v___x_521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_522_ = l_Lean_Expr_const___override(v___x_521_, v___x_499_);
v___x_523_ = l_Lean_Expr_app___override(v___x_522_, v_R_208_);
v___x_524_ = l_Lean_Expr_app___override(v___x_523_, v_A_209_);
v___x_525_ = l_Lean_Expr_app___override(v___x_524_, v___x_509_);
v___x_526_ = l_Lean_Expr_app___override(v___x_525_, v___x_513_);
v___x_527_ = l_Lean_Expr_app___override(v___x_520_, v___x_526_);
v___x_528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_529_ = l_Lean_Expr_const___override(v___x_528_, v___x_499_);
v___x_530_ = l_Lean_Expr_app___override(v___x_529_, v_R_208_);
v___x_531_ = l_Lean_Expr_app___override(v___x_530_, v_A_209_);
v___x_532_ = l_Lean_Expr_app___override(v___x_531_, v_sR_210_);
v___x_533_ = l_Lean_Expr_app___override(v___x_532_, v___x_486_);
lean_inc_ref(v_sAlg_212_);
v___x_534_ = l_Lean_Expr_app___override(v___x_533_, v_sAlg_212_);
v___x_535_ = l_Lean_Expr_app___override(v___x_527_, v___x_534_);
v___x_536_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_536_, 0, v_u_206_);
lean_ctor_set(v___x_536_, 1, v___x_444_);
v___x_537_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_537_, 0, v_u_206_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = l_Lean_Expr_const___override(v___x_465_, v___x_537_);
v___x_539_ = l_Lean_Expr_app___override(v___x_538_, v_R_208_);
v___x_540_ = l_Lean_Expr_app___override(v___x_539_, v_R_208_);
v___x_541_ = l_Lean_Expr_app___override(v___x_540_, v_R_208_);
v___x_542_ = l_Lean_Expr_const___override(v___x_474_, v___x_444_);
v___x_543_ = l_Lean_Expr_app___override(v___x_542_, v_R_208_);
v___x_544_ = l_Lean_Expr_const___override(v___x_477_, v___x_444_);
v___x_545_ = l_Lean_Expr_app___override(v___x_544_, v_R_208_);
v___x_546_ = l_Lean_Expr_const___override(v___x_480_, v___x_444_);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v_R_208_);
v___x_548_ = l_Lean_Expr_app___override(v___x_547_, v___x_508_);
v___x_549_ = l_Lean_Expr_app___override(v___x_545_, v___x_548_);
v___x_550_ = l_Lean_Expr_app___override(v___x_543_, v___x_549_);
v___x_551_ = l_Lean_Expr_app___override(v___x_541_, v___x_550_);
lean_inc(v_fst_460_);
v___x_552_ = l_Lean_Expr_app___override(v___x_551_, v_fst_460_);
v___x_553_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_554_ = l_Lean_Expr_const___override(v___x_553_, v___x_444_);
v___x_555_ = l_Lean_Expr_app___override(v___x_554_, v_R_208_);
v___x_556_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_557_ = l_Lean_Expr_app___override(v___x_555_, v___x_556_);
v___x_558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_559_ = l_Lean_Expr_const___override(v___x_558_, v___x_444_);
v___x_560_ = l_Lean_Expr_app___override(v___x_559_, v_R_208_);
v___x_561_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_562_ = l_Lean_Expr_const___override(v___x_561_, v___x_444_);
v___x_563_ = l_Lean_Expr_app___override(v___x_562_, v_R_208_);
v___x_564_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_565_ = l_Lean_Expr_const___override(v___x_564_, v___x_444_);
v___x_566_ = l_Lean_Expr_app___override(v___x_565_, v_R_208_);
v___x_567_ = l_Lean_Expr_app___override(v___x_566_, v___x_508_);
v___x_568_ = l_Lean_Expr_app___override(v___x_563_, v___x_567_);
v___x_569_ = l_Lean_Expr_app___override(v___x_560_, v___x_568_);
v___x_570_ = l_Lean_Expr_app___override(v___x_557_, v___x_569_);
v___x_571_ = l_Lean_Expr_app___override(v___x_552_, v___x_570_);
lean_inc_ref(v___x_571_);
v___x_572_ = l_Lean_Expr_app___override(v___x_535_, v___x_571_);
lean_inc_ref_n(v___x_572_, 2);
v___x_573_ = l_Lean_Expr_app___override(v___x_490_, v___x_572_);
v___x_574_ = l_Lean_Expr_const___override(v___x_553_, v___x_467_);
v___x_575_ = l_Lean_Expr_app___override(v___x_574_, v_A_209_);
v___x_576_ = l_Lean_Expr_app___override(v___x_575_, v___x_556_);
v___x_577_ = l_Lean_Expr_const___override(v___x_558_, v___x_467_);
v___x_578_ = l_Lean_Expr_app___override(v___x_577_, v_A_209_);
v___x_579_ = l_Lean_Expr_const___override(v___x_561_, v___x_467_);
v___x_580_ = l_Lean_Expr_app___override(v___x_579_, v_A_209_);
v___x_581_ = l_Lean_Expr_const___override(v___x_564_, v___x_467_);
v___x_582_ = l_Lean_Expr_app___override(v___x_581_, v_A_209_);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v___x_486_);
v___x_584_ = l_Lean_Expr_app___override(v___x_580_, v___x_583_);
v___x_585_ = l_Lean_Expr_app___override(v___x_578_, v___x_584_);
v___x_586_ = l_Lean_Expr_app___override(v___x_576_, v___x_585_);
v___x_587_ = l_Lean_Expr_app___override(v___x_573_, v___x_586_);
v___x_588_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_206_, v_R_208_, v_sR_210_, v_fst_460_, v_snd_461_);
v___x_589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_571_);
lean_ctor_set(v___x_589_, 1, v___x_588_);
v___x_590_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_590_, 0, v___x_572_);
lean_ctor_set(v___x_590_, 1, v___x_589_);
v___x_591_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_v_207_, v_A_209_, v_sA_211_, v___x_572_, v___x_590_);
v___x_592_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__55));
v___x_593_ = l_Lean_Expr_const___override(v___x_592_, v___x_499_);
v___x_594_ = l_Lean_Expr_app___override(v___x_593_, v_R_208_);
v___x_595_ = l_Lean_Expr_app___override(v___x_594_, v_A_209_);
v___x_596_ = l_Lean_Expr_app___override(v___x_595_, v_val_436_);
v___x_597_ = l_Lean_Expr_app___override(v___x_596_, v_val_437_);
v___x_598_ = l_Lean_Expr_app___override(v___x_597_, v_sAlg_212_);
v___x_599_ = l_Lean_Expr_app___override(v___x_598_, v_a_213_);
v___x_600_ = l_Lean_Expr_app___override(v___x_599_, v_lit_434_);
v___x_601_ = l_Lean_Expr_app___override(v___x_600_, v_proof_435_);
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 2, v___x_601_);
lean_ctor_set(v___x_432_, 1, v___x_591_);
lean_ctor_set(v___x_432_, 0, v___x_587_);
v___x_603_ = v___x_432_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_587_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v___x_591_);
lean_ctor_set(v_reuseFailAlloc_607_, 2, v___x_601_);
v___x_603_ = v_reuseFailAlloc_607_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
lean_object* v___x_605_; 
if (v_isShared_440_ == 0)
{
lean_ctor_set(v___x_439_, 0, v___x_603_);
v___x_605_ = v___x_439_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_603_);
v___x_605_ = v_reuseFailAlloc_606_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
return v___x_605_;
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
lean_object* v___x_613_; 
lean_del_object(v___x_432_);
lean_dec(v_r_u03b1_430_);
lean_del_object(v___x_428_);
lean_dec_ref_known(v_r_u03b1_425_, 1);
lean_del_object(v___x_423_);
lean_dec_ref_known(v_x_216_, 3);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_613_ = lean_box(0);
return v___x_613_;
}
}
}
}
else
{
lean_object* v___x_619_; 
lean_dec(v_r_u03b1_425_);
lean_del_object(v___x_423_);
lean_dec_ref_known(v_x_216_, 3);
lean_dec_ref(v_cA_215_);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_619_ = lean_box(0);
return v___x_619_;
}
}
}
case 3:
{
lean_object* v_toCache_622_; lean_object* v___x_624_; uint8_t v_isShared_625_; uint8_t v_isSharedCheck_832_; 
v_toCache_622_ = lean_ctor_get(v_cR_214_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v_cR_214_);
if (v_isSharedCheck_832_ == 0)
{
lean_object* v_unused_833_; 
v_unused_833_ = lean_ctor_get(v_cR_214_, 1);
lean_dec(v_unused_833_);
v___x_624_ = v_cR_214_;
v_isShared_625_ = v_isSharedCheck_832_;
goto v_resetjp_623_;
}
else
{
lean_inc(v_toCache_622_);
lean_dec(v_cR_214_);
v___x_624_ = lean_box(0);
v_isShared_625_ = v_isSharedCheck_832_;
goto v_resetjp_623_;
}
v_resetjp_623_:
{
lean_object* v_ds_u03b1_626_; 
v_ds_u03b1_626_ = lean_ctor_get(v_toCache_622_, 1);
lean_inc(v_ds_u03b1_626_);
lean_dec_ref(v_toCache_622_);
if (lean_obj_tag(v_ds_u03b1_626_) == 1)
{
lean_object* v_toCache_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_829_; 
v_toCache_627_ = lean_ctor_get(v_cA_215_, 0);
v_isSharedCheck_829_ = !lean_is_exclusive(v_cA_215_);
if (v_isSharedCheck_829_ == 0)
{
lean_object* v_unused_830_; 
v_unused_830_ = lean_ctor_get(v_cA_215_, 1);
lean_dec(v_unused_830_);
v___x_629_ = v_cA_215_;
v_isShared_630_ = v_isSharedCheck_829_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_toCache_627_);
lean_dec(v_cA_215_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_829_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v_ds_u03b1_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_826_; 
v_ds_u03b1_631_ = lean_ctor_get(v_toCache_627_, 1);
v_isSharedCheck_826_ = !lean_is_exclusive(v_toCache_627_);
if (v_isSharedCheck_826_ == 0)
{
lean_object* v_unused_827_; lean_object* v_unused_828_; 
v_unused_827_ = lean_ctor_get(v_toCache_627_, 2);
lean_dec(v_unused_827_);
v_unused_828_ = lean_ctor_get(v_toCache_627_, 0);
lean_dec(v_unused_828_);
v___x_633_ = v_toCache_627_;
v_isShared_634_ = v_isSharedCheck_826_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_ds_u03b1_631_);
lean_dec(v_toCache_627_);
v___x_633_ = lean_box(0);
v_isShared_634_ = v_isSharedCheck_826_;
goto v_resetjp_632_;
}
v_resetjp_632_:
{
if (lean_obj_tag(v_ds_u03b1_631_) == 1)
{
lean_object* v_inst_635_; lean_object* v_q_636_; lean_object* v_n_637_; lean_object* v_d_638_; lean_object* v_proof_639_; lean_object* v_val_640_; lean_object* v_val_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_824_; 
v_inst_635_ = lean_ctor_get(v_x_216_, 0);
lean_inc_ref(v_inst_635_);
v_q_636_ = lean_ctor_get(v_x_216_, 1);
lean_inc_ref(v_q_636_);
v_n_637_ = lean_ctor_get(v_x_216_, 2);
lean_inc_ref(v_n_637_);
v_d_638_ = lean_ctor_get(v_x_216_, 3);
lean_inc_ref(v_d_638_);
v_proof_639_ = lean_ctor_get(v_x_216_, 4);
lean_inc_ref(v_proof_639_);
lean_dec_ref_known(v_x_216_, 5);
v_val_640_ = lean_ctor_get(v_ds_u03b1_626_, 0);
lean_inc(v_val_640_);
lean_dec_ref_known(v_ds_u03b1_626_, 1);
v_val_641_ = lean_ctor_get(v_ds_u03b1_631_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v_ds_u03b1_631_);
if (v_isSharedCheck_824_ == 0)
{
v___x_643_ = v_ds_u03b1_631_;
v_isShared_644_ = v_isSharedCheck_824_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_val_641_);
lean_dec(v_ds_u03b1_631_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_824_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_648_; 
lean_inc_n(v_u_206_, 2);
v___x_645_ = l_Lean_Level_succ___override(v_u_206_);
v___x_646_ = lean_box(0);
if (v_isShared_630_ == 0)
{
lean_ctor_set_tag(v___x_629_, 1);
lean_ctor_set(v___x_629_, 1, v___x_646_);
lean_ctor_set(v___x_629_, 0, v_u_206_);
v___x_648_ = v___x_629_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_u_206_);
lean_ctor_set(v_reuseFailAlloc_823_, 1, v___x_646_);
v___x_648_ = v_reuseFailAlloc_823_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_653_; 
v___x_649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__57));
lean_inc_ref(v___x_648_);
v___x_650_ = l_Lean_Expr_const___override(v___x_649_, v___x_648_);
v___x_651_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50));
lean_inc(v___x_645_);
if (v_isShared_625_ == 0)
{
lean_ctor_set_tag(v___x_624_, 1);
lean_ctor_set(v___x_624_, 1, v___x_646_);
lean_ctor_set(v___x_624_, 0, v___x_645_);
v___x_653_ = v___x_624_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v___x_645_);
lean_ctor_set(v_reuseFailAlloc_822_, 1, v___x_646_);
v___x_653_ = v_reuseFailAlloc_822_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v_fst_672_; lean_object* v_snd_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_821_; 
v___x_654_ = l_Lean_Expr_const___override(v___x_651_, v___x_653_);
lean_inc_ref_n(v_R_208_, 3);
v___x_655_ = l_Lean_Expr_app___override(v___x_650_, v_R_208_);
v___x_656_ = l_Lean_Expr_app___override(v___x_654_, v___x_655_);
v___x_657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__60));
lean_inc_ref(v___x_648_);
v___x_658_ = l_Lean_Expr_const___override(v___x_657_, v___x_648_);
v___x_659_ = l_Lean_Expr_app___override(v___x_658_, v_R_208_);
lean_inc(v_val_640_);
v___x_660_ = l_Lean_Expr_app___override(v___x_659_, v_val_640_);
v___x_661_ = l_Lean_Expr_app___override(v___x_656_, v___x_660_);
lean_inc(v_v_207_);
v___x_662_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_662_, 0, v_v_207_);
lean_ctor_set(v___x_662_, 1, v___x_646_);
v___x_663_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__65));
lean_inc_ref(v___x_662_);
v___x_664_ = l_Lean_Expr_const___override(v___x_663_, v___x_662_);
lean_inc_ref(v_A_209_);
v___x_665_ = l_Lean_Expr_app___override(v___x_664_, v_A_209_);
v___x_666_ = l_Lean_Expr_app___override(v___x_665_, v_inst_635_);
lean_inc_ref(v_a_213_);
v___x_667_ = l_Lean_Expr_app___override(v___x_666_, v_a_213_);
lean_inc_ref_n(v_n_637_, 2);
v___x_668_ = l_Lean_Expr_app___override(v___x_667_, v_n_637_);
lean_inc_ref_n(v_d_638_, 2);
v___x_669_ = l_Lean_Expr_app___override(v___x_668_, v_d_638_);
lean_inc_ref(v_proof_639_);
v___x_670_ = l_Lean_Expr_app___override(v___x_669_, v_proof_639_);
lean_inc(v_u_206_);
v___x_671_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(v_u_206_, v_R_208_, v___x_661_, v_q_636_, v_n_637_, v_d_638_, v___x_670_);
v_fst_672_ = lean_ctor_get(v___x_671_, 0);
v_snd_673_ = lean_ctor_get(v___x_671_, 1);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_671_);
if (v_isSharedCheck_821_ == 0)
{
v___x_675_ = v___x_671_;
v_isShared_676_ = v_isSharedCheck_821_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_snd_673_);
lean_inc(v_fst_672_);
lean_dec(v___x_671_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_821_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_677_; lean_object* v___x_679_; 
v___x_677_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
lean_inc_ref(v___x_662_);
lean_inc(v_v_207_);
if (v_isShared_676_ == 0)
{
lean_ctor_set_tag(v___x_675_, 1);
lean_ctor_set(v___x_675_, 1, v___x_662_);
lean_ctor_set(v___x_675_, 0, v_v_207_);
v___x_679_ = v___x_675_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_v_207_);
lean_ctor_set(v_reuseFailAlloc_820_, 1, v___x_662_);
v___x_679_ = v_reuseFailAlloc_820_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; uint8_t v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_815_; 
lean_inc_n(v_v_207_, 2);
v___x_680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_680_, 0, v_v_207_);
lean_ctor_set(v___x_680_, 1, v___x_679_);
v___x_681_ = l_Lean_Expr_const___override(v___x_677_, v___x_680_);
lean_inc_ref_n(v_A_209_, 17);
v___x_682_ = l_Lean_Expr_app___override(v___x_681_, v_A_209_);
v___x_683_ = l_Lean_Expr_app___override(v___x_682_, v_A_209_);
v___x_684_ = l_Lean_Expr_app___override(v___x_683_, v_A_209_);
v___x_685_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
lean_inc_ref_n(v___x_662_, 9);
v___x_686_ = l_Lean_Expr_const___override(v___x_685_, v___x_662_);
v___x_687_ = l_Lean_Expr_app___override(v___x_686_, v_A_209_);
v___x_688_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_689_ = l_Lean_Expr_const___override(v___x_688_, v___x_662_);
v___x_690_ = l_Lean_Expr_app___override(v___x_689_, v_A_209_);
v___x_691_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_692_ = l_Lean_Expr_const___override(v___x_691_, v___x_662_);
v___x_693_ = l_Lean_Expr_app___override(v___x_692_, v_A_209_);
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_695_ = l_Lean_Expr_const___override(v___x_694_, v___x_662_);
v___x_696_ = l_Lean_Expr_app___override(v___x_695_, v_A_209_);
lean_inc_ref(v_sA_211_);
v___x_697_ = l_Lean_Expr_app___override(v___x_696_, v_sA_211_);
lean_inc_ref_n(v___x_697_, 3);
v___x_698_ = l_Lean_Expr_app___override(v___x_693_, v___x_697_);
v___x_699_ = l_Lean_Expr_app___override(v___x_690_, v___x_698_);
v___x_700_ = l_Lean_Expr_app___override(v___x_687_, v___x_699_);
v___x_701_ = l_Lean_Expr_app___override(v___x_684_, v___x_700_);
v___x_702_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
v___x_703_ = l_Lean_Level_succ___override(v_v_207_);
lean_inc(v___x_703_);
lean_inc(v___x_645_);
v___x_704_ = l_Lean_Level_max___override(v___x_645_, v___x_703_);
v___x_705_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_703_);
lean_ctor_set(v___x_705_, 1, v___x_646_);
v___x_706_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_706_, 0, v___x_645_);
lean_ctor_set(v___x_706_, 1, v___x_705_);
v___x_707_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_707_, 0, v___x_704_);
lean_ctor_set(v___x_707_, 1, v___x_706_);
v___x_708_ = l_Lean_Expr_const___override(v___x_702_, v___x_707_);
v___x_709_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_n(v_u_206_, 3);
v___x_710_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_710_, 0, v_u_206_);
lean_ctor_set(v___x_710_, 1, v___x_662_);
lean_inc_ref_n(v___x_710_, 3);
v___x_711_ = l_Lean_Expr_const___override(v___x_709_, v___x_710_);
lean_inc_ref_n(v_R_208_, 18);
v___x_712_ = l_Lean_Expr_app___override(v___x_711_, v_R_208_);
v___x_713_ = l_Lean_Expr_app___override(v___x_712_, v_A_209_);
v___x_714_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref_n(v___x_648_, 9);
v___x_715_ = l_Lean_Expr_const___override(v___x_714_, v___x_648_);
v___x_716_ = l_Lean_Expr_app___override(v___x_715_, v_R_208_);
v___x_717_ = l_Lean_Expr_const___override(v___x_694_, v___x_648_);
v___x_718_ = l_Lean_Expr_app___override(v___x_717_, v_R_208_);
lean_inc_ref_n(v_sR_210_, 2);
v___x_719_ = l_Lean_Expr_app___override(v___x_718_, v_sR_210_);
lean_inc_ref_n(v___x_719_, 2);
v___x_720_ = l_Lean_Expr_app___override(v___x_716_, v___x_719_);
lean_inc_ref(v___x_720_);
v___x_721_ = l_Lean_Expr_app___override(v___x_713_, v___x_720_);
v___x_722_ = l_Lean_Expr_const___override(v___x_714_, v___x_662_);
v___x_723_ = l_Lean_Expr_app___override(v___x_722_, v_A_209_);
v___x_724_ = l_Lean_Expr_app___override(v___x_723_, v___x_697_);
lean_inc_ref(v___x_724_);
v___x_725_ = l_Lean_Expr_app___override(v___x_721_, v___x_724_);
v___x_726_ = l_Lean_Expr_app___override(v___x_708_, v___x_725_);
v___x_727_ = l_Lean_Expr_app___override(v___x_726_, v_R_208_);
v___x_728_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_729_ = 0;
v___x_730_ = l_Lean_Expr_lam___override(v___x_728_, v_R_208_, v_A_209_, v___x_729_);
v___x_731_ = l_Lean_Expr_app___override(v___x_727_, v___x_730_);
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_733_ = l_Lean_Expr_const___override(v___x_732_, v___x_710_);
v___x_734_ = l_Lean_Expr_app___override(v___x_733_, v_R_208_);
v___x_735_ = l_Lean_Expr_app___override(v___x_734_, v_A_209_);
v___x_736_ = l_Lean_Expr_app___override(v___x_735_, v___x_720_);
v___x_737_ = l_Lean_Expr_app___override(v___x_736_, v___x_724_);
v___x_738_ = l_Lean_Expr_app___override(v___x_731_, v___x_737_);
v___x_739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_740_ = l_Lean_Expr_const___override(v___x_739_, v___x_710_);
v___x_741_ = l_Lean_Expr_app___override(v___x_740_, v_R_208_);
v___x_742_ = l_Lean_Expr_app___override(v___x_741_, v_A_209_);
v___x_743_ = l_Lean_Expr_app___override(v___x_742_, v_sR_210_);
v___x_744_ = l_Lean_Expr_app___override(v___x_743_, v___x_697_);
lean_inc_ref(v_sAlg_212_);
v___x_745_ = l_Lean_Expr_app___override(v___x_744_, v_sAlg_212_);
v___x_746_ = l_Lean_Expr_app___override(v___x_738_, v___x_745_);
v___x_747_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_747_, 0, v_u_206_);
lean_ctor_set(v___x_747_, 1, v___x_648_);
v___x_748_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_748_, 0, v_u_206_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
v___x_749_ = l_Lean_Expr_const___override(v___x_677_, v___x_748_);
v___x_750_ = l_Lean_Expr_app___override(v___x_749_, v_R_208_);
v___x_751_ = l_Lean_Expr_app___override(v___x_750_, v_R_208_);
v___x_752_ = l_Lean_Expr_app___override(v___x_751_, v_R_208_);
v___x_753_ = l_Lean_Expr_const___override(v___x_685_, v___x_648_);
v___x_754_ = l_Lean_Expr_app___override(v___x_753_, v_R_208_);
v___x_755_ = l_Lean_Expr_const___override(v___x_688_, v___x_648_);
v___x_756_ = l_Lean_Expr_app___override(v___x_755_, v_R_208_);
v___x_757_ = l_Lean_Expr_const___override(v___x_691_, v___x_648_);
v___x_758_ = l_Lean_Expr_app___override(v___x_757_, v_R_208_);
v___x_759_ = l_Lean_Expr_app___override(v___x_758_, v___x_719_);
v___x_760_ = l_Lean_Expr_app___override(v___x_756_, v___x_759_);
v___x_761_ = l_Lean_Expr_app___override(v___x_754_, v___x_760_);
v___x_762_ = l_Lean_Expr_app___override(v___x_752_, v___x_761_);
lean_inc(v_fst_672_);
v___x_763_ = l_Lean_Expr_app___override(v___x_762_, v_fst_672_);
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_765_ = l_Lean_Expr_const___override(v___x_764_, v___x_648_);
v___x_766_ = l_Lean_Expr_app___override(v___x_765_, v_R_208_);
v___x_767_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_768_ = l_Lean_Expr_app___override(v___x_766_, v___x_767_);
v___x_769_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_770_ = l_Lean_Expr_const___override(v___x_769_, v___x_648_);
v___x_771_ = l_Lean_Expr_app___override(v___x_770_, v_R_208_);
v___x_772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_773_ = l_Lean_Expr_const___override(v___x_772_, v___x_648_);
v___x_774_ = l_Lean_Expr_app___override(v___x_773_, v_R_208_);
v___x_775_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_776_ = l_Lean_Expr_const___override(v___x_775_, v___x_648_);
v___x_777_ = l_Lean_Expr_app___override(v___x_776_, v_R_208_);
v___x_778_ = l_Lean_Expr_app___override(v___x_777_, v___x_719_);
v___x_779_ = l_Lean_Expr_app___override(v___x_774_, v___x_778_);
v___x_780_ = l_Lean_Expr_app___override(v___x_771_, v___x_779_);
v___x_781_ = l_Lean_Expr_app___override(v___x_768_, v___x_780_);
v___x_782_ = l_Lean_Expr_app___override(v___x_763_, v___x_781_);
lean_inc_ref(v___x_782_);
v___x_783_ = l_Lean_Expr_app___override(v___x_746_, v___x_782_);
lean_inc_ref_n(v___x_783_, 2);
v___x_784_ = l_Lean_Expr_app___override(v___x_701_, v___x_783_);
v___x_785_ = l_Lean_Expr_const___override(v___x_764_, v___x_662_);
v___x_786_ = l_Lean_Expr_app___override(v___x_785_, v_A_209_);
v___x_787_ = l_Lean_Expr_app___override(v___x_786_, v___x_767_);
v___x_788_ = l_Lean_Expr_const___override(v___x_769_, v___x_662_);
v___x_789_ = l_Lean_Expr_app___override(v___x_788_, v_A_209_);
v___x_790_ = l_Lean_Expr_const___override(v___x_772_, v___x_662_);
v___x_791_ = l_Lean_Expr_app___override(v___x_790_, v_A_209_);
v___x_792_ = l_Lean_Expr_const___override(v___x_775_, v___x_662_);
v___x_793_ = l_Lean_Expr_app___override(v___x_792_, v_A_209_);
v___x_794_ = l_Lean_Expr_app___override(v___x_793_, v___x_697_);
v___x_795_ = l_Lean_Expr_app___override(v___x_791_, v___x_794_);
v___x_796_ = l_Lean_Expr_app___override(v___x_789_, v___x_795_);
v___x_797_ = l_Lean_Expr_app___override(v___x_787_, v___x_796_);
v___x_798_ = l_Lean_Expr_app___override(v___x_784_, v___x_797_);
v___x_799_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_206_, v_R_208_, v_sR_210_, v_fst_672_, v_snd_673_);
v___x_800_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_800_, 0, v___x_782_);
lean_ctor_set(v___x_800_, 1, v___x_799_);
v___x_801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_801_, 0, v___x_783_);
lean_ctor_set(v___x_801_, 1, v___x_800_);
v___x_802_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_v_207_, v_A_209_, v_sA_211_, v___x_783_, v___x_801_);
v___x_803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__67));
v___x_804_ = l_Lean_Expr_const___override(v___x_803_, v___x_710_);
v___x_805_ = l_Lean_Expr_app___override(v___x_804_, v_R_208_);
v___x_806_ = l_Lean_Expr_app___override(v___x_805_, v_A_209_);
v___x_807_ = l_Lean_Expr_app___override(v___x_806_, v_val_640_);
v___x_808_ = l_Lean_Expr_app___override(v___x_807_, v_val_641_);
v___x_809_ = l_Lean_Expr_app___override(v___x_808_, v_sAlg_212_);
v___x_810_ = l_Lean_Expr_app___override(v___x_809_, v_a_213_);
v___x_811_ = l_Lean_Expr_app___override(v___x_810_, v_n_637_);
v___x_812_ = l_Lean_Expr_app___override(v___x_811_, v_d_638_);
v___x_813_ = l_Lean_Expr_app___override(v___x_812_, v_proof_639_);
if (v_isShared_634_ == 0)
{
lean_ctor_set(v___x_633_, 2, v___x_813_);
lean_ctor_set(v___x_633_, 1, v___x_802_);
lean_ctor_set(v___x_633_, 0, v___x_798_);
v___x_815_ = v___x_633_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v___x_798_);
lean_ctor_set(v_reuseFailAlloc_819_, 1, v___x_802_);
lean_ctor_set(v_reuseFailAlloc_819_, 2, v___x_813_);
v___x_815_ = v_reuseFailAlloc_819_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
lean_object* v___x_817_; 
if (v_isShared_644_ == 0)
{
lean_ctor_set(v___x_643_, 0, v___x_815_);
v___x_817_ = v___x_643_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v___x_815_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
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
lean_object* v___x_825_; 
lean_del_object(v___x_633_);
lean_dec(v_ds_u03b1_631_);
lean_del_object(v___x_629_);
lean_dec_ref_known(v_ds_u03b1_626_, 1);
lean_del_object(v___x_624_);
lean_dec_ref_known(v_x_216_, 5);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_825_ = lean_box(0);
return v___x_825_;
}
}
}
}
else
{
lean_object* v___x_831_; 
lean_dec(v_ds_u03b1_626_);
lean_del_object(v___x_624_);
lean_dec_ref_known(v_x_216_, 5);
lean_dec_ref(v_cA_215_);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_831_ = lean_box(0);
return v___x_831_;
}
}
}
case 4:
{
lean_object* v_field_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_1036_; 
v_field_834_ = lean_ctor_get(v_cR_214_, 1);
v_isSharedCheck_1036_ = !lean_is_exclusive(v_cR_214_);
if (v_isSharedCheck_1036_ == 0)
{
lean_object* v_unused_1037_; 
v_unused_1037_ = lean_ctor_get(v_cR_214_, 0);
lean_dec(v_unused_1037_);
v___x_836_ = v_cR_214_;
v_isShared_837_ = v_isSharedCheck_1036_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_field_834_);
lean_dec(v_cR_214_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_1036_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
if (lean_obj_tag(v_field_834_) == 1)
{
lean_object* v_field_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_1033_; 
v_field_838_ = lean_ctor_get(v_cA_215_, 1);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_cA_215_);
if (v_isSharedCheck_1033_ == 0)
{
lean_object* v_unused_1034_; 
v_unused_1034_ = lean_ctor_get(v_cA_215_, 0);
lean_dec(v_unused_1034_);
v___x_840_ = v_cA_215_;
v_isShared_841_ = v_isSharedCheck_1033_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_field_838_);
lean_dec(v_cA_215_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_1033_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
if (lean_obj_tag(v_field_838_) == 1)
{
lean_object* v_inst_842_; lean_object* v_q_843_; lean_object* v_n_844_; lean_object* v_d_845_; lean_object* v_proof_846_; lean_object* v_val_847_; lean_object* v_val_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_1031_; 
v_inst_842_ = lean_ctor_get(v_x_216_, 0);
lean_inc_ref(v_inst_842_);
v_q_843_ = lean_ctor_get(v_x_216_, 1);
lean_inc_ref(v_q_843_);
v_n_844_ = lean_ctor_get(v_x_216_, 2);
lean_inc_ref(v_n_844_);
v_d_845_ = lean_ctor_get(v_x_216_, 3);
lean_inc_ref(v_d_845_);
v_proof_846_ = lean_ctor_get(v_x_216_, 4);
lean_inc_ref(v_proof_846_);
lean_dec_ref_known(v_x_216_, 5);
v_val_847_ = lean_ctor_get(v_field_834_, 0);
lean_inc(v_val_847_);
lean_dec_ref_known(v_field_834_, 1);
v_val_848_ = lean_ctor_get(v_field_838_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v_field_838_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_850_ = v_field_838_;
v_isShared_851_ = v_isSharedCheck_1031_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_val_848_);
lean_dec(v_field_838_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_1031_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_855_; 
lean_inc_n(v_u_206_, 2);
v___x_852_ = l_Lean_Level_succ___override(v_u_206_);
v___x_853_ = lean_box(0);
if (v_isShared_841_ == 0)
{
lean_ctor_set_tag(v___x_840_, 1);
lean_ctor_set(v___x_840_, 1, v___x_853_);
lean_ctor_set(v___x_840_, 0, v_u_206_);
v___x_855_ = v___x_840_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v_u_206_);
lean_ctor_set(v_reuseFailAlloc_1030_, 1, v___x_853_);
v___x_855_ = v_reuseFailAlloc_1030_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_860_; 
v___x_856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__69));
lean_inc_ref(v___x_855_);
v___x_857_ = l_Lean_Expr_const___override(v___x_856_, v___x_855_);
v___x_858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__50));
lean_inc(v___x_852_);
if (v_isShared_837_ == 0)
{
lean_ctor_set_tag(v___x_836_, 1);
lean_ctor_set(v___x_836_, 1, v___x_853_);
lean_ctor_set(v___x_836_, 0, v___x_852_);
v___x_860_ = v___x_836_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_1029_; 
v_reuseFailAlloc_1029_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1029_, 0, v___x_852_);
lean_ctor_set(v_reuseFailAlloc_1029_, 1, v___x_853_);
v___x_860_ = v_reuseFailAlloc_1029_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v_fst_881_; lean_object* v_snd_882_; lean_object* v___x_884_; uint8_t v_isShared_885_; uint8_t v_isSharedCheck_1028_; 
v___x_861_ = l_Lean_Expr_const___override(v___x_858_, v___x_860_);
lean_inc_ref_n(v_R_208_, 3);
v___x_862_ = l_Lean_Expr_app___override(v___x_857_, v_R_208_);
v___x_863_ = l_Lean_Expr_app___override(v___x_861_, v___x_862_);
v___x_864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__71));
lean_inc_ref(v___x_855_);
v___x_865_ = l_Lean_Expr_const___override(v___x_864_, v___x_855_);
v___x_866_ = l_Lean_Expr_app___override(v___x_865_, v_R_208_);
lean_inc(v_val_847_);
v___x_867_ = l_Lean_Expr_app___override(v___x_866_, v_val_847_);
v___x_868_ = l_Lean_Expr_app___override(v___x_863_, v___x_867_);
lean_inc(v_v_207_);
v___x_869_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_869_, 0, v_v_207_);
lean_ctor_set(v___x_869_, 1, v___x_853_);
v___x_870_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__73));
lean_inc_ref(v___x_869_);
v___x_871_ = l_Lean_Expr_const___override(v___x_870_, v___x_869_);
lean_inc_ref(v_A_209_);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_A_209_);
v___x_873_ = l_Lean_Expr_app___override(v___x_872_, v_inst_842_);
lean_inc_ref(v_a_213_);
v___x_874_ = l_Lean_Expr_app___override(v___x_873_, v_a_213_);
v___x_875_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__77);
lean_inc_ref_n(v_n_844_, 2);
v___x_876_ = l_Lean_Expr_app___override(v___x_875_, v_n_844_);
v___x_877_ = l_Lean_Expr_app___override(v___x_874_, v___x_876_);
lean_inc_ref_n(v_d_845_, 2);
v___x_878_ = l_Lean_Expr_app___override(v___x_877_, v_d_845_);
lean_inc_ref(v_proof_846_);
v___x_879_ = l_Lean_Expr_app___override(v___x_878_, v_proof_846_);
lean_inc(v_u_206_);
v___x_880_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(v_u_206_, v_R_208_, v___x_868_, v_q_843_, v_n_844_, v_d_845_, v___x_879_);
v_fst_881_ = lean_ctor_get(v___x_880_, 0);
v_snd_882_ = lean_ctor_get(v___x_880_, 1);
v_isSharedCheck_1028_ = !lean_is_exclusive(v___x_880_);
if (v_isSharedCheck_1028_ == 0)
{
v___x_884_ = v___x_880_;
v_isShared_885_ = v_isSharedCheck_1028_;
goto v_resetjp_883_;
}
else
{
lean_inc(v_snd_882_);
lean_inc(v_fst_881_);
lean_dec(v___x_880_);
v___x_884_ = lean_box(0);
v_isShared_885_ = v_isSharedCheck_1028_;
goto v_resetjp_883_;
}
v_resetjp_883_:
{
lean_object* v___x_886_; lean_object* v___x_888_; 
v___x_886_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
lean_inc_ref(v___x_869_);
lean_inc(v_v_207_);
if (v_isShared_885_ == 0)
{
lean_ctor_set_tag(v___x_884_, 1);
lean_ctor_set(v___x_884_, 1, v___x_869_);
lean_ctor_set(v___x_884_, 0, v_v_207_);
v___x_888_ = v___x_884_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1027_, 0, v_v_207_);
lean_ctor_set(v_reuseFailAlloc_1027_, 1, v___x_869_);
v___x_888_ = v_reuseFailAlloc_1027_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; uint8_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1025_; 
lean_inc_n(v_v_207_, 2);
v___x_889_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_889_, 0, v_v_207_);
lean_ctor_set(v___x_889_, 1, v___x_888_);
v___x_890_ = l_Lean_Expr_const___override(v___x_886_, v___x_889_);
lean_inc_ref_n(v_A_209_, 17);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v_A_209_);
v___x_892_ = l_Lean_Expr_app___override(v___x_891_, v_A_209_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v_A_209_);
v___x_894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
lean_inc_ref_n(v___x_869_, 9);
v___x_895_ = l_Lean_Expr_const___override(v___x_894_, v___x_869_);
v___x_896_ = l_Lean_Expr_app___override(v___x_895_, v_A_209_);
v___x_897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_898_ = l_Lean_Expr_const___override(v___x_897_, v___x_869_);
v___x_899_ = l_Lean_Expr_app___override(v___x_898_, v_A_209_);
v___x_900_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_901_ = l_Lean_Expr_const___override(v___x_900_, v___x_869_);
v___x_902_ = l_Lean_Expr_app___override(v___x_901_, v_A_209_);
v___x_903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_904_ = l_Lean_Expr_const___override(v___x_903_, v___x_869_);
v___x_905_ = l_Lean_Expr_app___override(v___x_904_, v_A_209_);
lean_inc_ref(v_sA_211_);
v___x_906_ = l_Lean_Expr_app___override(v___x_905_, v_sA_211_);
lean_inc_ref_n(v___x_906_, 3);
v___x_907_ = l_Lean_Expr_app___override(v___x_902_, v___x_906_);
v___x_908_ = l_Lean_Expr_app___override(v___x_899_, v___x_907_);
v___x_909_ = l_Lean_Expr_app___override(v___x_896_, v___x_908_);
v___x_910_ = l_Lean_Expr_app___override(v___x_893_, v___x_909_);
v___x_911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
v___x_912_ = l_Lean_Level_succ___override(v_v_207_);
lean_inc(v___x_912_);
lean_inc(v___x_852_);
v___x_913_ = l_Lean_Level_max___override(v___x_852_, v___x_912_);
v___x_914_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_912_);
lean_ctor_set(v___x_914_, 1, v___x_853_);
v___x_915_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_915_, 0, v___x_852_);
lean_ctor_set(v___x_915_, 1, v___x_914_);
v___x_916_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_916_, 0, v___x_913_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v___x_917_ = l_Lean_Expr_const___override(v___x_911_, v___x_916_);
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_n(v_u_206_, 3);
v___x_919_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_919_, 0, v_u_206_);
lean_ctor_set(v___x_919_, 1, v___x_869_);
lean_inc_ref_n(v___x_919_, 3);
v___x_920_ = l_Lean_Expr_const___override(v___x_918_, v___x_919_);
lean_inc_ref_n(v_R_208_, 18);
v___x_921_ = l_Lean_Expr_app___override(v___x_920_, v_R_208_);
v___x_922_ = l_Lean_Expr_app___override(v___x_921_, v_A_209_);
v___x_923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref_n(v___x_855_, 9);
v___x_924_ = l_Lean_Expr_const___override(v___x_923_, v___x_855_);
v___x_925_ = l_Lean_Expr_app___override(v___x_924_, v_R_208_);
v___x_926_ = l_Lean_Expr_const___override(v___x_903_, v___x_855_);
v___x_927_ = l_Lean_Expr_app___override(v___x_926_, v_R_208_);
lean_inc_ref_n(v_sR_210_, 2);
v___x_928_ = l_Lean_Expr_app___override(v___x_927_, v_sR_210_);
lean_inc_ref_n(v___x_928_, 2);
v___x_929_ = l_Lean_Expr_app___override(v___x_925_, v___x_928_);
lean_inc_ref(v___x_929_);
v___x_930_ = l_Lean_Expr_app___override(v___x_922_, v___x_929_);
v___x_931_ = l_Lean_Expr_const___override(v___x_923_, v___x_869_);
v___x_932_ = l_Lean_Expr_app___override(v___x_931_, v_A_209_);
v___x_933_ = l_Lean_Expr_app___override(v___x_932_, v___x_906_);
lean_inc_ref(v___x_933_);
v___x_934_ = l_Lean_Expr_app___override(v___x_930_, v___x_933_);
v___x_935_ = l_Lean_Expr_app___override(v___x_917_, v___x_934_);
v___x_936_ = l_Lean_Expr_app___override(v___x_935_, v_R_208_);
v___x_937_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_938_ = 0;
v___x_939_ = l_Lean_Expr_lam___override(v___x_937_, v_R_208_, v_A_209_, v___x_938_);
v___x_940_ = l_Lean_Expr_app___override(v___x_936_, v___x_939_);
v___x_941_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_942_ = l_Lean_Expr_const___override(v___x_941_, v___x_919_);
v___x_943_ = l_Lean_Expr_app___override(v___x_942_, v_R_208_);
v___x_944_ = l_Lean_Expr_app___override(v___x_943_, v_A_209_);
v___x_945_ = l_Lean_Expr_app___override(v___x_944_, v___x_929_);
v___x_946_ = l_Lean_Expr_app___override(v___x_945_, v___x_933_);
v___x_947_ = l_Lean_Expr_app___override(v___x_940_, v___x_946_);
v___x_948_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_949_ = l_Lean_Expr_const___override(v___x_948_, v___x_919_);
v___x_950_ = l_Lean_Expr_app___override(v___x_949_, v_R_208_);
v___x_951_ = l_Lean_Expr_app___override(v___x_950_, v_A_209_);
v___x_952_ = l_Lean_Expr_app___override(v___x_951_, v_sR_210_);
v___x_953_ = l_Lean_Expr_app___override(v___x_952_, v___x_906_);
lean_inc_ref(v_sAlg_212_);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v_sAlg_212_);
v___x_955_ = l_Lean_Expr_app___override(v___x_947_, v___x_954_);
v___x_956_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_956_, 0, v_u_206_);
lean_ctor_set(v___x_956_, 1, v___x_855_);
v___x_957_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_957_, 0, v_u_206_);
lean_ctor_set(v___x_957_, 1, v___x_956_);
v___x_958_ = l_Lean_Expr_const___override(v___x_886_, v___x_957_);
v___x_959_ = l_Lean_Expr_app___override(v___x_958_, v_R_208_);
v___x_960_ = l_Lean_Expr_app___override(v___x_959_, v_R_208_);
v___x_961_ = l_Lean_Expr_app___override(v___x_960_, v_R_208_);
v___x_962_ = l_Lean_Expr_const___override(v___x_894_, v___x_855_);
v___x_963_ = l_Lean_Expr_app___override(v___x_962_, v_R_208_);
v___x_964_ = l_Lean_Expr_const___override(v___x_897_, v___x_855_);
v___x_965_ = l_Lean_Expr_app___override(v___x_964_, v_R_208_);
v___x_966_ = l_Lean_Expr_const___override(v___x_900_, v___x_855_);
v___x_967_ = l_Lean_Expr_app___override(v___x_966_, v_R_208_);
v___x_968_ = l_Lean_Expr_app___override(v___x_967_, v___x_928_);
v___x_969_ = l_Lean_Expr_app___override(v___x_965_, v___x_968_);
v___x_970_ = l_Lean_Expr_app___override(v___x_963_, v___x_969_);
v___x_971_ = l_Lean_Expr_app___override(v___x_961_, v___x_970_);
lean_inc(v_fst_881_);
v___x_972_ = l_Lean_Expr_app___override(v___x_971_, v_fst_881_);
v___x_973_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_974_ = l_Lean_Expr_const___override(v___x_973_, v___x_855_);
v___x_975_ = l_Lean_Expr_app___override(v___x_974_, v_R_208_);
v___x_976_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_977_ = l_Lean_Expr_app___override(v___x_975_, v___x_976_);
v___x_978_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_979_ = l_Lean_Expr_const___override(v___x_978_, v___x_855_);
v___x_980_ = l_Lean_Expr_app___override(v___x_979_, v_R_208_);
v___x_981_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_982_ = l_Lean_Expr_const___override(v___x_981_, v___x_855_);
v___x_983_ = l_Lean_Expr_app___override(v___x_982_, v_R_208_);
v___x_984_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_985_ = l_Lean_Expr_const___override(v___x_984_, v___x_855_);
v___x_986_ = l_Lean_Expr_app___override(v___x_985_, v_R_208_);
v___x_987_ = l_Lean_Expr_app___override(v___x_986_, v___x_928_);
v___x_988_ = l_Lean_Expr_app___override(v___x_983_, v___x_987_);
v___x_989_ = l_Lean_Expr_app___override(v___x_980_, v___x_988_);
v___x_990_ = l_Lean_Expr_app___override(v___x_977_, v___x_989_);
v___x_991_ = l_Lean_Expr_app___override(v___x_972_, v___x_990_);
lean_inc_ref(v___x_991_);
v___x_992_ = l_Lean_Expr_app___override(v___x_955_, v___x_991_);
lean_inc_ref_n(v___x_992_, 2);
v___x_993_ = l_Lean_Expr_app___override(v___x_910_, v___x_992_);
v___x_994_ = l_Lean_Expr_const___override(v___x_973_, v___x_869_);
v___x_995_ = l_Lean_Expr_app___override(v___x_994_, v_A_209_);
v___x_996_ = l_Lean_Expr_app___override(v___x_995_, v___x_976_);
v___x_997_ = l_Lean_Expr_const___override(v___x_978_, v___x_869_);
v___x_998_ = l_Lean_Expr_app___override(v___x_997_, v_A_209_);
v___x_999_ = l_Lean_Expr_const___override(v___x_981_, v___x_869_);
v___x_1000_ = l_Lean_Expr_app___override(v___x_999_, v_A_209_);
v___x_1001_ = l_Lean_Expr_const___override(v___x_984_, v___x_869_);
v___x_1002_ = l_Lean_Expr_app___override(v___x_1001_, v_A_209_);
v___x_1003_ = l_Lean_Expr_app___override(v___x_1002_, v___x_906_);
v___x_1004_ = l_Lean_Expr_app___override(v___x_1000_, v___x_1003_);
v___x_1005_ = l_Lean_Expr_app___override(v___x_998_, v___x_1004_);
v___x_1006_ = l_Lean_Expr_app___override(v___x_996_, v___x_1005_);
v___x_1007_ = l_Lean_Expr_app___override(v___x_993_, v___x_1006_);
v___x_1008_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_206_, v_R_208_, v_sR_210_, v_fst_881_, v_snd_882_);
v___x_1009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1009_, 0, v___x_991_);
lean_ctor_set(v___x_1009_, 1, v___x_1008_);
v___x_1010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1010_, 0, v___x_992_);
lean_ctor_set(v___x_1010_, 1, v___x_1009_);
v___x_1011_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_v_207_, v_A_209_, v_sA_211_, v___x_992_, v___x_1010_);
v___x_1012_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__79));
v___x_1013_ = l_Lean_Expr_const___override(v___x_1012_, v___x_919_);
v___x_1014_ = l_Lean_Expr_app___override(v___x_1013_, v_R_208_);
v___x_1015_ = l_Lean_Expr_app___override(v___x_1014_, v_A_209_);
v___x_1016_ = l_Lean_Expr_app___override(v___x_1015_, v_val_847_);
v___x_1017_ = l_Lean_Expr_app___override(v___x_1016_, v_val_848_);
v___x_1018_ = l_Lean_Expr_app___override(v___x_1017_, v_sAlg_212_);
v___x_1019_ = l_Lean_Expr_app___override(v___x_1018_, v_a_213_);
v___x_1020_ = l_Lean_Expr_app___override(v___x_1019_, v_n_844_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_d_845_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1021_, v_proof_846_);
v___x_1023_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1023_, 0, v___x_1007_);
lean_ctor_set(v___x_1023_, 1, v___x_1011_);
lean_ctor_set(v___x_1023_, 2, v___x_1022_);
if (v_isShared_851_ == 0)
{
lean_ctor_set(v___x_850_, 0, v___x_1023_);
v___x_1025_ = v___x_850_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v___x_1023_);
v___x_1025_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
return v___x_1025_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1032_; 
lean_del_object(v___x_840_);
lean_dec(v_field_838_);
lean_dec_ref_known(v_field_834_, 1);
lean_del_object(v___x_836_);
lean_dec_ref_known(v_x_216_, 5);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_1032_ = lean_box(0);
return v___x_1032_;
}
}
}
else
{
lean_object* v___x_1035_; 
lean_del_object(v___x_836_);
lean_dec(v_field_834_);
lean_dec_ref_known(v_x_216_, 5);
lean_dec_ref(v_cA_215_);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_1035_ = lean_box(0);
return v___x_1035_;
}
}
}
default: 
{
lean_object* v___x_1038_; 
lean_dec_ref(v_x_216_);
lean_dec_ref(v_cA_215_);
lean_dec_ref(v_cR_214_);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_sAlg_212_);
lean_dec_ref(v_sA_211_);
lean_dec_ref(v_sR_210_);
lean_dec_ref(v_A_209_);
lean_dec_ref(v_R_208_);
lean_dec(v_v_207_);
lean_dec(v_u_206_);
v___x_1038_ = lean_box(0);
return v___x_1038_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg(lean_object* v___y_1039_){
_start:
{
lean_object* v___x_1041_; lean_object* v_ngen_1042_; lean_object* v_namePrefix_1043_; lean_object* v_idx_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1073_; 
v___x_1041_ = lean_st_ref_get(v___y_1039_);
v_ngen_1042_ = lean_ctor_get(v___x_1041_, 2);
lean_inc_ref(v_ngen_1042_);
lean_dec(v___x_1041_);
v_namePrefix_1043_ = lean_ctor_get(v_ngen_1042_, 0);
v_idx_1044_ = lean_ctor_get(v_ngen_1042_, 1);
v_isSharedCheck_1073_ = !lean_is_exclusive(v_ngen_1042_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1046_ = v_ngen_1042_;
v_isShared_1047_ = v_isSharedCheck_1073_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_idx_1044_);
lean_inc(v_namePrefix_1043_);
lean_dec(v_ngen_1042_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1073_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v___x_1048_; lean_object* v_env_1049_; lean_object* v_nextMacroScope_1050_; lean_object* v_auxDeclNGen_1051_; lean_object* v_traceState_1052_; lean_object* v_cache_1053_; lean_object* v_messages_1054_; lean_object* v_infoState_1055_; lean_object* v_snapshotTasks_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1071_; 
v___x_1048_ = lean_st_ref_take(v___y_1039_);
v_env_1049_ = lean_ctor_get(v___x_1048_, 0);
v_nextMacroScope_1050_ = lean_ctor_get(v___x_1048_, 1);
v_auxDeclNGen_1051_ = lean_ctor_get(v___x_1048_, 3);
v_traceState_1052_ = lean_ctor_get(v___x_1048_, 4);
v_cache_1053_ = lean_ctor_get(v___x_1048_, 5);
v_messages_1054_ = lean_ctor_get(v___x_1048_, 6);
v_infoState_1055_ = lean_ctor_get(v___x_1048_, 7);
v_snapshotTasks_1056_ = lean_ctor_get(v___x_1048_, 8);
v_isSharedCheck_1071_ = !lean_is_exclusive(v___x_1048_);
if (v_isSharedCheck_1071_ == 0)
{
lean_object* v_unused_1072_; 
v_unused_1072_ = lean_ctor_get(v___x_1048_, 2);
lean_dec(v_unused_1072_);
v___x_1058_ = v___x_1048_;
v_isShared_1059_ = v_isSharedCheck_1071_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_snapshotTasks_1056_);
lean_inc(v_infoState_1055_);
lean_inc(v_messages_1054_);
lean_inc(v_cache_1053_);
lean_inc(v_traceState_1052_);
lean_inc(v_auxDeclNGen_1051_);
lean_inc(v_nextMacroScope_1050_);
lean_inc(v_env_1049_);
lean_dec(v___x_1048_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1071_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
lean_object* v_r_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1064_; 
lean_inc(v_idx_1044_);
lean_inc(v_namePrefix_1043_);
v_r_1060_ = l_Lean_Name_num___override(v_namePrefix_1043_, v_idx_1044_);
v___x_1061_ = lean_unsigned_to_nat(1u);
v___x_1062_ = lean_nat_add(v_idx_1044_, v___x_1061_);
lean_dec(v_idx_1044_);
if (v_isShared_1047_ == 0)
{
lean_ctor_set(v___x_1046_, 1, v___x_1062_);
v___x_1064_ = v___x_1046_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_namePrefix_1043_);
lean_ctor_set(v_reuseFailAlloc_1070_, 1, v___x_1062_);
v___x_1064_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
lean_object* v___x_1066_; 
if (v_isShared_1059_ == 0)
{
lean_ctor_set(v___x_1058_, 2, v___x_1064_);
v___x_1066_ = v___x_1058_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_env_1049_);
lean_ctor_set(v_reuseFailAlloc_1069_, 1, v_nextMacroScope_1050_);
lean_ctor_set(v_reuseFailAlloc_1069_, 2, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1069_, 3, v_auxDeclNGen_1051_);
lean_ctor_set(v_reuseFailAlloc_1069_, 4, v_traceState_1052_);
lean_ctor_set(v_reuseFailAlloc_1069_, 5, v_cache_1053_);
lean_ctor_set(v_reuseFailAlloc_1069_, 6, v_messages_1054_);
lean_ctor_set(v_reuseFailAlloc_1069_, 7, v_infoState_1055_);
lean_ctor_set(v_reuseFailAlloc_1069_, 8, v_snapshotTasks_1056_);
v___x_1066_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1067_ = lean_st_ref_set(v___y_1039_, v___x_1066_);
v___x_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1068_, 0, v_r_1060_);
return v___x_1068_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg___boxed(lean_object* v___y_1074_, lean_object* v___y_1075_){
_start:
{
lean_object* v_res_1076_; 
v_res_1076_ = lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg(v___y_1074_);
lean_dec(v___y_1074_);
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0(lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg(v___y_1080_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___boxed(lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v_res_1088_; 
v_res_1088_ = lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0(v___y_1083_, v___y_1084_, v___y_1085_, v___y_1086_);
lean_dec(v___y_1086_);
lean_dec_ref(v___y_1085_);
lean_dec(v___y_1084_);
lean_dec_ref(v___y_1083_);
return v_res_1088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1(lean_object* v_as_1089_, size_t v_sz_1090_, size_t v_i_1091_, lean_object* v_b_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
uint8_t v___x_1098_; 
v___x_1098_ = lean_usize_dec_lt(v_i_1091_, v_sz_1090_);
if (v___x_1098_ == 0)
{
lean_object* v___x_1099_; 
v___x_1099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1099_, 0, v_b_1092_);
return v___x_1099_;
}
else
{
lean_object* v_a_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; 
v_a_1100_ = lean_array_uget_borrowed(v_as_1089_, v_i_1091_);
v___x_1101_ = lean_box(0);
lean_inc(v_a_1100_);
v___x_1102_ = l_Lean_mkConst(v_a_1100_, v___x_1101_);
v___x_1103_ = l_Lean_Meta_abstractMVars(v___x_1102_, v___x_1098_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
if (lean_obj_tag(v___x_1103_) == 0)
{
lean_object* v_a_1104_; lean_object* v_paramNames_1105_; lean_object* v_expr_1106_; lean_object* v___x_1107_; 
v_a_1104_ = lean_ctor_get(v___x_1103_, 0);
lean_inc(v_a_1104_);
lean_dec_ref_known(v___x_1103_, 1);
v_paramNames_1105_ = lean_ctor_get(v_a_1104_, 0);
lean_inc_ref(v_paramNames_1105_);
v_expr_1106_ = lean_ctor_get(v_a_1104_, 2);
lean_inc_ref(v_expr_1106_);
lean_dec(v_a_1104_);
v___x_1107_ = lp_mathlib_Lean_mkFreshId___at___00Mathlib_Tactic_Algebra_pushCast_spec__0___redArg(v___y_1096_);
if (lean_obj_tag(v___x_1107_) == 0)
{
lean_object* v_a_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; uint8_t v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; 
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
lean_inc(v_a_1108_);
lean_dec_ref_known(v___x_1107_, 1);
v___x_1109_ = lean_box(0);
v___x_1110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1110_, 0, v_a_1108_);
lean_ctor_set(v___x_1110_, 1, v___x_1109_);
v___x_1111_ = 0;
v___x_1112_ = lean_unsigned_to_nat(1000u);
v___x_1113_ = l_Lean_Meta_simpGlobalConfig;
v___x_1114_ = l_Lean_Meta_SimpTheorems_add(v_b_1092_, v___x_1110_, v_paramNames_1105_, v_expr_1106_, v___x_1111_, v___x_1098_, v___x_1112_, v___x_1113_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
if (lean_obj_tag(v___x_1114_) == 0)
{
lean_object* v_a_1115_; size_t v___x_1116_; size_t v___x_1117_; 
v_a_1115_ = lean_ctor_get(v___x_1114_, 0);
lean_inc(v_a_1115_);
lean_dec_ref_known(v___x_1114_, 1);
v___x_1116_ = ((size_t)1ULL);
v___x_1117_ = lean_usize_add(v_i_1091_, v___x_1116_);
v_i_1091_ = v___x_1117_;
v_b_1092_ = v_a_1115_;
goto _start;
}
else
{
return v___x_1114_;
}
}
else
{
lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1126_; 
lean_dec_ref(v_expr_1106_);
lean_dec_ref(v_paramNames_1105_);
lean_dec_ref(v_b_1092_);
v_a_1119_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1121_ = v___x_1107_;
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1107_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1124_; 
if (v_isShared_1122_ == 0)
{
v___x_1124_ = v___x_1121_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v_a_1119_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
}
else
{
lean_object* v_a_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1134_; 
lean_dec_ref(v_b_1092_);
v_a_1127_ = lean_ctor_get(v___x_1103_, 0);
v_isSharedCheck_1134_ = !lean_is_exclusive(v___x_1103_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1129_ = v___x_1103_;
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_a_1127_);
lean_dec(v___x_1103_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1132_; 
if (v_isShared_1130_ == 0)
{
v___x_1132_ = v___x_1129_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1133_; 
v_reuseFailAlloc_1133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1133_, 0, v_a_1127_);
v___x_1132_ = v_reuseFailAlloc_1133_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
return v___x_1132_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1___boxed(lean_object* v_as_1135_, lean_object* v_sz_1136_, lean_object* v_i_1137_, lean_object* v_b_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_){
_start:
{
size_t v_sz_boxed_1144_; size_t v_i_boxed_1145_; lean_object* v_res_1146_; 
v_sz_boxed_1144_ = lean_unbox_usize(v_sz_1136_);
lean_dec(v_sz_1136_);
v_i_boxed_1145_ = lean_unbox_usize(v_i_1137_);
lean_dec(v_i_1137_);
v_res_1146_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1(v_as_1135_, v_sz_boxed_1144_, v_i_boxed_1145_, v_b_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec_ref(v_as_1135_);
return v_res_1146_;
}
}
static size_t _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7(void){
_start:
{
lean_object* v___x_1164_; size_t v_sz_1165_; 
v___x_1164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__6));
v_sz_1165_ = lean_array_size(v___x_1164_);
return v_sz_1165_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9(void){
_start:
{
lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; 
v___x_1173_ = lean_box(0);
v___x_1174_ = lean_unsigned_to_nat(16u);
v___x_1175_ = lean_mk_array(v___x_1174_, v___x_1173_);
return v___x_1175_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10(void){
_start:
{
lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1176_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__9);
v___x_1177_ = lean_unsigned_to_nat(0u);
v___x_1178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1177_);
lean_ctor_set(v___x_1178_, 1, v___x_1176_);
return v___x_1178_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11(void){
_start:
{
lean_object* v___x_1179_; 
v___x_1179_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1179_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12(void){
_start:
{
lean_object* v___x_1180_; lean_object* v___x_1181_; 
v___x_1180_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__11);
v___x_1181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1181_, 0, v___x_1180_);
return v___x_1181_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13(void){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; uint8_t v___x_1184_; lean_object* v___x_1185_; 
v___x_1182_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12);
v___x_1183_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10);
v___x_1184_ = 1;
v___x_1185_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1185_, 0, v___x_1183_);
lean_ctor_set(v___x_1185_, 1, v___x_1182_);
lean_ctor_set_uint8(v___x_1185_, sizeof(void*)*2, v___x_1184_);
return v___x_1185_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15(void){
_start:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; 
v___x_1188_ = lean_unsigned_to_nat(0u);
v___x_1189_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12);
v___x_1190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1189_);
lean_ctor_set(v___x_1190_, 1, v___x_1188_);
return v___x_1190_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16(void){
_start:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; 
v___x_1191_ = lean_unsigned_to_nat(32u);
v___x_1192_ = lean_mk_empty_array_with_capacity(v___x_1191_);
v___x_1193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1192_);
return v___x_1193_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17(void){
_start:
{
size_t v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1194_ = ((size_t)5ULL);
v___x_1195_ = lean_unsigned_to_nat(0u);
v___x_1196_ = lean_unsigned_to_nat(32u);
v___x_1197_ = lean_mk_empty_array_with_capacity(v___x_1196_);
v___x_1198_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__16);
v___x_1199_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1199_, 0, v___x_1198_);
lean_ctor_set(v___x_1199_, 1, v___x_1197_);
lean_ctor_set(v___x_1199_, 2, v___x_1195_);
lean_ctor_set(v___x_1199_, 3, v___x_1195_);
lean_ctor_set_usize(v___x_1199_, 4, v___x_1194_);
return v___x_1199_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18(void){
_start:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17);
v___x_1201_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__12);
v___x_1202_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
lean_ctor_set(v___x_1202_, 1, v___x_1201_);
lean_ctor_set(v___x_1202_, 2, v___x_1201_);
lean_ctor_set(v___x_1202_, 3, v___x_1200_);
return v___x_1202_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19(void){
_start:
{
lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1203_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__18);
v___x_1204_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__15);
v___x_1205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
lean_ctor_set(v___x_1205_, 1, v___x_1203_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast(lean_object* v_e_1206_, lean_object* v_a_1207_, lean_object* v_a_1208_, lean_object* v_a_1209_, lean_object* v_a_1210_){
_start:
{
lean_object* v___x_1212_; lean_object* v___x_1213_; 
v___x_1212_ = l_Lean_Meta_NormCast_pushCastExt;
v___x_1213_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v___x_1212_, v_a_1210_);
if (lean_obj_tag(v___x_1213_) == 0)
{
lean_object* v_a_1214_; lean_object* v___x_1215_; size_t v_sz_1216_; size_t v___x_1217_; lean_object* v___x_1218_; 
v_a_1214_ = lean_ctor_get(v___x_1213_, 0);
lean_inc(v_a_1214_);
lean_dec_ref_known(v___x_1213_, 1);
v___x_1215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__6));
v_sz_1216_ = lean_usize_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__7);
v___x_1217_ = ((size_t)0ULL);
v___x_1218_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Algebra_pushCast_spec__1(v___x_1215_, v_sz_1216_, v___x_1217_, v_a_1214_, v_a_1207_, v_a_1208_, v_a_1209_, v_a_1210_);
if (lean_obj_tag(v___x_1218_) == 0)
{
lean_object* v_a_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; 
v_a_1219_ = lean_ctor_get(v___x_1218_, 0);
lean_inc(v_a_1219_);
lean_dec_ref_known(v___x_1218_, 1);
v___x_1220_ = lean_box(0);
v___x_1221_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__8));
v___x_1222_ = lean_unsigned_to_nat(1u);
v___x_1223_ = lean_mk_empty_array_with_capacity(v___x_1222_);
v___x_1224_ = lean_array_push(v___x_1223_, v_a_1219_);
v___x_1225_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__13);
v___x_1226_ = l_Lean_Options_empty;
v___x_1227_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_1221_, v___x_1224_, v___x_1225_, v___x_1226_, v_a_1207_, v_a_1209_, v_a_1210_);
if (lean_obj_tag(v___x_1227_) == 0)
{
lean_object* v_a_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; 
v_a_1228_ = lean_ctor_get(v___x_1227_, 0);
lean_inc(v_a_1228_);
lean_dec_ref_known(v___x_1227_, 1);
v___x_1229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__14));
v___x_1230_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__19);
v___x_1231_ = l_Lean_Meta_simp(v_e_1206_, v_a_1228_, v___x_1229_, v___x_1220_, v___x_1230_, v_a_1207_, v_a_1208_, v_a_1209_, v_a_1210_);
if (lean_obj_tag(v___x_1231_) == 0)
{
lean_object* v_a_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1240_; 
v_a_1232_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1240_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1240_ == 0)
{
v___x_1234_ = v___x_1231_;
v_isShared_1235_ = v_isSharedCheck_1240_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_a_1232_);
lean_dec(v___x_1231_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1240_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v_fst_1236_; lean_object* v___x_1238_; 
v_fst_1236_ = lean_ctor_get(v_a_1232_, 0);
lean_inc(v_fst_1236_);
lean_dec(v_a_1232_);
if (v_isShared_1235_ == 0)
{
lean_ctor_set(v___x_1234_, 0, v_fst_1236_);
v___x_1238_ = v___x_1234_;
goto v_reusejp_1237_;
}
else
{
lean_object* v_reuseFailAlloc_1239_; 
v_reuseFailAlloc_1239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1239_, 0, v_fst_1236_);
v___x_1238_ = v_reuseFailAlloc_1239_;
goto v_reusejp_1237_;
}
v_reusejp_1237_:
{
return v___x_1238_;
}
}
}
else
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
v_a_1241_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1231_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v___x_1231_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_a_1241_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
}
else
{
lean_object* v_a_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1256_; 
lean_dec_ref(v_e_1206_);
v_a_1249_ = lean_ctor_get(v___x_1227_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v___x_1227_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v___x_1227_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v___x_1227_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1254_; 
if (v_isShared_1252_ == 0)
{
v___x_1254_ = v___x_1251_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v_a_1249_);
v___x_1254_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1253_;
}
v_reusejp_1253_:
{
return v___x_1254_;
}
}
}
}
else
{
lean_object* v_a_1257_; lean_object* v___x_1259_; uint8_t v_isShared_1260_; uint8_t v_isSharedCheck_1264_; 
lean_dec_ref(v_e_1206_);
v_a_1257_ = lean_ctor_get(v___x_1218_, 0);
v_isSharedCheck_1264_ = !lean_is_exclusive(v___x_1218_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1259_ = v___x_1218_;
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
else
{
lean_inc(v_a_1257_);
lean_dec(v___x_1218_);
v___x_1259_ = lean_box(0);
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
v_resetjp_1258_:
{
lean_object* v___x_1262_; 
if (v_isShared_1260_ == 0)
{
v___x_1262_ = v___x_1259_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v_a_1257_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
}
}
else
{
lean_object* v_a_1265_; lean_object* v___x_1267_; uint8_t v_isShared_1268_; uint8_t v_isSharedCheck_1272_; 
lean_dec_ref(v_e_1206_);
v_a_1265_ = lean_ctor_get(v___x_1213_, 0);
v_isSharedCheck_1272_ = !lean_is_exclusive(v___x_1213_);
if (v_isSharedCheck_1272_ == 0)
{
v___x_1267_ = v___x_1213_;
v_isShared_1268_ = v_isSharedCheck_1272_;
goto v_resetjp_1266_;
}
else
{
lean_inc(v_a_1265_);
lean_dec(v___x_1213_);
v___x_1267_ = lean_box(0);
v_isShared_1268_ = v_isSharedCheck_1272_;
goto v_resetjp_1266_;
}
v_resetjp_1266_:
{
lean_object* v___x_1270_; 
if (v_isShared_1268_ == 0)
{
v___x_1270_ = v___x_1267_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1271_; 
v_reuseFailAlloc_1271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1271_, 0, v_a_1265_);
v___x_1270_ = v_reuseFailAlloc_1271_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
return v___x_1270_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pushCast___boxed(lean_object* v_e_1273_, lean_object* v_a_1274_, lean_object* v_a_1275_, lean_object* v_a_1276_, lean_object* v_a_1277_, lean_object* v_a_1278_){
_start:
{
lean_object* v_res_1279_; 
v_res_1279_ = lp_mathlib_Mathlib_Tactic_Algebra_pushCast(v_e_1273_, v_a_1274_, v_a_1275_, v_a_1276_, v_a_1277_);
lean_dec(v_a_1277_);
lean_dec_ref(v_a_1276_);
lean_dec(v_a_1275_);
lean_dec_ref(v_a_1274_);
return v_res_1279_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19(void){
_start:
{
lean_object* v___x_1312_; lean_object* v___x_1313_; 
v___x_1312_ = lean_unsigned_to_nat(0u);
v___x_1313_ = l_Lean_Expr_bvar___override(v___x_1312_);
return v___x_1313_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24(void){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; 
v___x_1321_ = lean_unsigned_to_nat(1u);
v___x_1322_ = l_Lean_Expr_bvar___override(v___x_1321_);
return v___x_1322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25(void){
_start:
{
lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1323_ = lean_unsigned_to_nat(2u);
v___x_1324_ = l_Lean_Expr_bvar___override(v___x_1323_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast(lean_object* v_u_1335_, lean_object* v_u_x27_1336_, lean_object* v_v_1337_, lean_object* v_R_1338_, lean_object* v_R_x27_1339_, lean_object* v_A_1340_, lean_object* v_sR_1341_, lean_object* v_sA_1342_, lean_object* v_sAlg_1343_, lean_object* v_smul_1344_, lean_object* v_r_x27_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_, lean_object* v_a_1349_){
_start:
{
lean_object* v___x_1351_; 
lean_inc_ref(v_R_x27_1339_);
lean_inc_ref(v_R_1338_);
v___x_1351_ = l_Lean_Meta_isExprDefEq(v_R_1338_, v_R_x27_1339_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1351_) == 0)
{
lean_object* v_a_1352_; lean_object* v___x_1354_; uint8_t v_isShared_1355_; uint8_t v_isSharedCheck_1647_; 
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1647_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1647_ == 0)
{
v___x_1354_ = v___x_1351_;
v_isShared_1355_ = v_isSharedCheck_1647_;
goto v_resetjp_1353_;
}
else
{
lean_inc(v_a_1352_);
lean_dec(v___x_1351_);
v___x_1354_ = lean_box(0);
v_isShared_1355_ = v_isSharedCheck_1647_;
goto v_resetjp_1353_;
}
v_resetjp_1353_:
{
uint8_t v___x_1356_; 
v___x_1356_ = lean_unbox(v_a_1352_);
lean_dec(v_a_1352_);
if (v___x_1356_ == 0)
{
lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; 
lean_del_object(v___x_1354_);
lean_inc_n(v_u_x27_1336_, 2);
v___x_1357_ = l_Lean_Level_succ___override(v_u_x27_1336_);
v___x_1358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0));
v___x_1359_ = lean_box(0);
v___x_1360_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1360_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1360_, 1, v___x_1359_);
lean_inc_ref(v___x_1360_);
v___x_1361_ = l_Lean_Expr_const___override(v___x_1358_, v___x_1360_);
lean_inc_ref(v_R_x27_1339_);
v___x_1362_ = l_Lean_Expr_app___override(v___x_1361_, v_R_x27_1339_);
v___x_1363_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1362_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1363_) == 0)
{
lean_object* v_a_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
v_a_1364_ = lean_ctor_get(v___x_1363_, 0);
lean_inc_n(v_a_1364_, 2);
lean_dec_ref_known(v___x_1363_, 1);
lean_inc_n(v_u_1335_, 2);
v___x_1365_ = l_Lean_Level_succ___override(v_u_1335_);
v___x_1366_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1));
v___x_1367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1367_, 0, v_u_1335_);
lean_ctor_set(v___x_1367_, 1, v___x_1359_);
lean_inc_ref_n(v___x_1367_, 2);
lean_inc(v_u_x27_1336_);
v___x_1368_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1368_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1368_, 1, v___x_1367_);
lean_inc_ref(v___x_1368_);
v___x_1369_ = l_Lean_Expr_const___override(v___x_1366_, v___x_1368_);
lean_inc_ref(v_R_x27_1339_);
v___x_1370_ = l_Lean_Expr_app___override(v___x_1369_, v_R_x27_1339_);
lean_inc_ref_n(v_R_1338_, 2);
v___x_1371_ = l_Lean_Expr_app___override(v___x_1370_, v_R_1338_);
v___x_1372_ = l_Lean_Expr_app___override(v___x_1371_, v_a_1364_);
v___x_1373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_1374_ = l_Lean_Expr_const___override(v___x_1373_, v___x_1367_);
v___x_1375_ = l_Lean_Expr_app___override(v___x_1374_, v_R_1338_);
lean_inc_ref(v_sR_1341_);
v___x_1376_ = l_Lean_Expr_app___override(v___x_1375_, v_sR_1341_);
lean_inc_ref(v___x_1376_);
v___x_1377_ = l_Lean_Expr_app___override(v___x_1372_, v___x_1376_);
v___x_1378_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1377_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1378_) == 0)
{
lean_object* v_a_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; 
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc(v_a_1379_);
lean_dec_ref_known(v___x_1378_, 1);
lean_inc_n(v_v_1337_, 2);
v___x_1380_ = l_Lean_Level_succ___override(v_v_1337_);
v___x_1381_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__3));
v___x_1382_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1382_, 0, v_v_1337_);
lean_ctor_set(v___x_1382_, 1, v___x_1359_);
lean_inc_ref_n(v___x_1382_, 3);
lean_inc(v_u_x27_1336_);
v___x_1383_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1383_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1383_, 1, v___x_1382_);
lean_inc_ref(v___x_1383_);
v___x_1384_ = l_Lean_Expr_const___override(v___x_1381_, v___x_1383_);
lean_inc_ref_n(v_R_x27_1339_, 2);
v___x_1385_ = l_Lean_Expr_app___override(v___x_1384_, v_R_x27_1339_);
lean_inc_ref_n(v_A_1340_, 3);
v___x_1386_ = l_Lean_Expr_app___override(v___x_1385_, v_A_1340_);
lean_inc_ref(v___x_1360_);
v___x_1387_ = l_Lean_Expr_const___override(v___x_1373_, v___x_1360_);
v___x_1388_ = l_Lean_Expr_app___override(v___x_1387_, v_R_x27_1339_);
lean_inc(v_a_1364_);
v___x_1389_ = l_Lean_Expr_app___override(v___x_1388_, v_a_1364_);
lean_inc_ref(v___x_1389_);
v___x_1390_ = l_Lean_Expr_app___override(v___x_1386_, v___x_1389_);
v___x_1391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__5));
v___x_1392_ = l_Lean_Expr_const___override(v___x_1391_, v___x_1382_);
v___x_1393_ = l_Lean_Expr_app___override(v___x_1392_, v_A_1340_);
v___x_1394_ = l_Lean_Expr_const___override(v___x_1373_, v___x_1382_);
v___x_1395_ = l_Lean_Expr_app___override(v___x_1394_, v_A_1340_);
v___x_1396_ = l_Lean_Expr_app___override(v___x_1395_, v_sA_1342_);
lean_inc_ref(v___x_1396_);
v___x_1397_ = l_Lean_Expr_app___override(v___x_1393_, v___x_1396_);
lean_inc_ref(v___x_1397_);
v___x_1398_ = l_Lean_Expr_app___override(v___x_1390_, v___x_1397_);
v___x_1399_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1398_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1399_) == 0)
{
lean_object* v_a_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; 
v_a_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc(v_a_1400_);
lean_dec_ref_known(v___x_1399_, 1);
v___x_1401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__7));
lean_inc_ref(v___x_1382_);
lean_inc(v_u_1335_);
v___x_1402_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1402_, 0, v_u_1335_);
lean_ctor_set(v___x_1402_, 1, v___x_1382_);
lean_inc_ref_n(v___x_1402_, 2);
lean_inc(v_u_x27_1336_);
v___x_1403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1403_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1403_, 1, v___x_1402_);
lean_inc_ref(v___x_1403_);
v___x_1404_ = l_Lean_Expr_const___override(v___x_1401_, v___x_1403_);
lean_inc_ref_n(v_R_x27_1339_, 2);
v___x_1405_ = l_Lean_Expr_app___override(v___x_1404_, v_R_x27_1339_);
lean_inc_ref_n(v_R_1338_, 3);
v___x_1406_ = l_Lean_Expr_app___override(v___x_1405_, v_R_1338_);
lean_inc_ref_n(v_A_1340_, 2);
v___x_1407_ = l_Lean_Expr_app___override(v___x_1406_, v_A_1340_);
v___x_1408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9));
lean_inc_ref(v___x_1368_);
v___x_1409_ = l_Lean_Expr_const___override(v___x_1408_, v___x_1368_);
v___x_1410_ = l_Lean_Expr_app___override(v___x_1409_, v_R_x27_1339_);
v___x_1411_ = l_Lean_Expr_app___override(v___x_1410_, v_R_1338_);
lean_inc(v_a_1364_);
v___x_1412_ = l_Lean_Expr_app___override(v___x_1411_, v_a_1364_);
lean_inc_ref(v___x_1376_);
v___x_1413_ = l_Lean_Expr_app___override(v___x_1412_, v___x_1376_);
lean_inc(v_a_1379_);
v___x_1414_ = l_Lean_Expr_app___override(v___x_1413_, v_a_1379_);
v___x_1415_ = l_Lean_Expr_app___override(v___x_1407_, v___x_1414_);
v___x_1416_ = l_Lean_Expr_const___override(v___x_1408_, v___x_1402_);
v___x_1417_ = l_Lean_Expr_app___override(v___x_1416_, v_R_1338_);
v___x_1418_ = l_Lean_Expr_app___override(v___x_1417_, v_A_1340_);
lean_inc_ref(v_sR_1341_);
v___x_1419_ = l_Lean_Expr_app___override(v___x_1418_, v_sR_1341_);
lean_inc_ref(v___x_1396_);
v___x_1420_ = l_Lean_Expr_app___override(v___x_1419_, v___x_1396_);
lean_inc_ref(v_sAlg_1343_);
v___x_1421_ = l_Lean_Expr_app___override(v___x_1420_, v_sAlg_1343_);
lean_inc_ref(v___x_1421_);
v___x_1422_ = l_Lean_Expr_app___override(v___x_1415_, v___x_1421_);
lean_inc_ref(v_smul_1344_);
v___x_1423_ = l_Lean_Expr_app___override(v___x_1422_, v_smul_1344_);
v___x_1424_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1423_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1424_) == 0)
{
lean_object* v_a_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; uint8_t v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; 
v_a_1425_ = lean_ctor_get(v___x_1424_, 0);
lean_inc(v_a_1425_);
lean_dec_ref_known(v___x_1424_, 1);
lean_inc(v___x_1357_);
lean_inc(v___x_1365_);
v___x_1426_ = l_Lean_Level_max___override(v___x_1365_, v___x_1357_);
v___x_1427_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
v___x_1428_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1428_, 0, v___x_1365_);
lean_ctor_set(v___x_1428_, 1, v___x_1359_);
lean_inc_ref(v___x_1428_);
v___x_1429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1429_, 0, v___x_1357_);
lean_ctor_set(v___x_1429_, 1, v___x_1428_);
v___x_1430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1426_);
lean_ctor_set(v___x_1430_, 1, v___x_1429_);
v___x_1431_ = l_Lean_Expr_const___override(v___x_1427_, v___x_1430_);
v___x_1432_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_ref_n(v___x_1368_, 2);
v___x_1433_ = l_Lean_Expr_const___override(v___x_1432_, v___x_1368_);
lean_inc_ref_n(v_R_x27_1339_, 6);
v___x_1434_ = l_Lean_Expr_app___override(v___x_1433_, v_R_x27_1339_);
lean_inc_ref_n(v_R_1338_, 5);
v___x_1435_ = l_Lean_Expr_app___override(v___x_1434_, v_R_1338_);
v___x_1436_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_1437_ = l_Lean_Expr_const___override(v___x_1436_, v___x_1360_);
v___x_1438_ = l_Lean_Expr_app___override(v___x_1437_, v_R_x27_1339_);
v___x_1439_ = l_Lean_Expr_app___override(v___x_1438_, v___x_1389_);
lean_inc_ref(v___x_1439_);
v___x_1440_ = l_Lean_Expr_app___override(v___x_1435_, v___x_1439_);
v___x_1441_ = l_Lean_Expr_const___override(v___x_1436_, v___x_1367_);
v___x_1442_ = l_Lean_Expr_app___override(v___x_1441_, v_R_1338_);
lean_inc_ref_n(v___x_1376_, 2);
v___x_1443_ = l_Lean_Expr_app___override(v___x_1442_, v___x_1376_);
lean_inc_ref(v___x_1443_);
v___x_1444_ = l_Lean_Expr_app___override(v___x_1440_, v___x_1443_);
v___x_1445_ = l_Lean_Expr_app___override(v___x_1431_, v___x_1444_);
v___x_1446_ = l_Lean_Expr_app___override(v___x_1445_, v_R_x27_1339_);
v___x_1447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_1448_ = 0;
v___x_1449_ = l_Lean_Expr_lam___override(v___x_1447_, v_R_x27_1339_, v_R_1338_, v___x_1448_);
v___x_1450_ = l_Lean_Expr_app___override(v___x_1446_, v___x_1449_);
v___x_1451_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_1452_ = l_Lean_Expr_const___override(v___x_1451_, v___x_1368_);
v___x_1453_ = l_Lean_Expr_app___override(v___x_1452_, v_R_x27_1339_);
v___x_1454_ = l_Lean_Expr_app___override(v___x_1453_, v_R_1338_);
v___x_1455_ = l_Lean_Expr_app___override(v___x_1454_, v___x_1439_);
v___x_1456_ = l_Lean_Expr_app___override(v___x_1455_, v___x_1443_);
v___x_1457_ = l_Lean_Expr_app___override(v___x_1450_, v___x_1456_);
v___x_1458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_1459_ = l_Lean_Expr_const___override(v___x_1458_, v___x_1368_);
v___x_1460_ = l_Lean_Expr_app___override(v___x_1459_, v_R_x27_1339_);
v___x_1461_ = l_Lean_Expr_app___override(v___x_1460_, v_R_1338_);
lean_inc(v_a_1364_);
v___x_1462_ = l_Lean_Expr_app___override(v___x_1461_, v_a_1364_);
v___x_1463_ = l_Lean_Expr_app___override(v___x_1462_, v___x_1376_);
lean_inc(v_a_1379_);
v___x_1464_ = l_Lean_Expr_app___override(v___x_1463_, v_a_1379_);
v___x_1465_ = l_Lean_Expr_app___override(v___x_1457_, v___x_1464_);
lean_inc_ref(v_r_x27_1345_);
v___x_1466_ = l_Lean_Expr_app___override(v___x_1465_, v_r_x27_1345_);
lean_inc_ref(v___x_1466_);
v___x_1467_ = lp_mathlib_Mathlib_Tactic_Algebra_pushCast(v___x_1466_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1467_) == 0)
{
lean_object* v_a_1468_; lean_object* v_expr_1469_; lean_object* v___x_1470_; 
v_a_1468_ = lean_ctor_get(v___x_1467_, 0);
lean_inc(v_a_1468_);
lean_dec_ref_known(v___x_1467_, 1);
v_expr_1469_ = lean_ctor_get(v_a_1468_, 0);
lean_inc_ref(v_expr_1469_);
v___x_1470_ = l_Lean_Meta_Simp_Result_getProof(v_a_1468_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1470_) == 0)
{
lean_object* v_a_1471_; lean_object* v___x_1473_; uint8_t v_isShared_1474_; uint8_t v_isSharedCheck_1555_; 
v_a_1471_ = lean_ctor_get(v___x_1470_, 0);
v_isSharedCheck_1555_ = !lean_is_exclusive(v___x_1470_);
if (v_isSharedCheck_1555_ == 0)
{
v___x_1473_ = v___x_1470_;
v_isShared_1474_ = v_isSharedCheck_1555_;
goto v_resetjp_1472_;
}
else
{
lean_inc(v_a_1471_);
lean_dec(v___x_1470_);
v___x_1473_ = lean_box(0);
v_isShared_1474_ = v_isSharedCheck_1555_;
goto v_resetjp_1472_;
}
v_resetjp_1472_:
{
lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1553_; 
v___x_1475_ = lean_box(0);
v___x_1476_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__11));
v___x_1477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_1478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1380_);
lean_ctor_set(v___x_1478_, 1, v___x_1359_);
v___x_1479_ = l_Lean_Expr_const___override(v___x_1477_, v___x_1478_);
lean_inc_ref_n(v_A_1340_, 9);
v___x_1480_ = l_Lean_Expr_app___override(v___x_1479_, v_A_1340_);
v___x_1481_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16));
v___x_1482_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1482_, 0, v_v_1337_);
lean_ctor_set(v___x_1482_, 1, v___x_1382_);
lean_inc_ref(v___x_1482_);
v___x_1483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1483_, 0, v_u_1335_);
lean_ctor_set(v___x_1483_, 1, v___x_1482_);
v___x_1484_ = l_Lean_Expr_const___override(v___x_1481_, v___x_1483_);
lean_inc_ref_n(v_R_1338_, 6);
v___x_1485_ = l_Lean_Expr_app___override(v___x_1484_, v_R_1338_);
v___x_1486_ = l_Lean_Expr_app___override(v___x_1485_, v_A_1340_);
v___x_1487_ = l_Lean_Expr_app___override(v___x_1486_, v_A_1340_);
v___x_1488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18));
lean_inc_ref(v___x_1402_);
v___x_1489_ = l_Lean_Expr_const___override(v___x_1488_, v___x_1402_);
v___x_1490_ = l_Lean_Expr_app___override(v___x_1489_, v_R_1338_);
v___x_1491_ = l_Lean_Expr_app___override(v___x_1490_, v_A_1340_);
v___x_1492_ = l_Lean_Expr_app___override(v___x_1491_, v___x_1421_);
v___x_1493_ = l_Lean_Expr_app___override(v___x_1487_, v___x_1492_);
v___x_1494_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19);
v___x_1495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1495_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1495_, 1, v___x_1482_);
v___x_1496_ = l_Lean_Expr_const___override(v___x_1481_, v___x_1495_);
lean_inc_ref_n(v_R_x27_1339_, 2);
v___x_1497_ = l_Lean_Expr_app___override(v___x_1496_, v_R_x27_1339_);
v___x_1498_ = l_Lean_Expr_app___override(v___x_1497_, v_A_1340_);
v___x_1499_ = l_Lean_Expr_app___override(v___x_1498_, v_A_1340_);
v___x_1500_ = l_Lean_Expr_const___override(v___x_1488_, v___x_1383_);
v___x_1501_ = l_Lean_Expr_app___override(v___x_1500_, v_R_x27_1339_);
v___x_1502_ = l_Lean_Expr_app___override(v___x_1501_, v_A_1340_);
v___x_1503_ = l_Lean_Expr_app___override(v___x_1502_, v_smul_1344_);
v___x_1504_ = l_Lean_Expr_app___override(v___x_1499_, v___x_1503_);
lean_inc_ref(v_r_x27_1345_);
v___x_1505_ = l_Lean_Expr_app___override(v___x_1504_, v_r_x27_1345_);
v___x_1506_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21));
lean_inc_ref(v___x_1428_);
v___x_1507_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1507_, 0, v___x_1475_);
lean_ctor_set(v___x_1507_, 1, v___x_1428_);
v___x_1508_ = l_Lean_Expr_const___override(v___x_1506_, v___x_1507_);
v___x_1509_ = l_Lean_Expr_app___override(v___x_1508_, v_R_1338_);
lean_inc_ref(v___x_1466_);
v___x_1510_ = l_Lean_Expr_app___override(v___x_1509_, v___x_1466_);
v___x_1511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__23));
v___x_1512_ = l_Lean_Expr_const___override(v___x_1477_, v___x_1428_);
v___x_1513_ = l_Lean_Expr_app___override(v___x_1512_, v_R_1338_);
v___x_1514_ = l_Lean_Expr_app___override(v___x_1513_, v___x_1466_);
v___x_1515_ = l_Lean_Expr_app___override(v___x_1514_, v___x_1494_);
v___x_1516_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24);
v___x_1517_ = l_Lean_Expr_app___override(v___x_1493_, v___x_1516_);
v___x_1518_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__25);
v___x_1519_ = l_Lean_Expr_app___override(v___x_1517_, v___x_1518_);
v___x_1520_ = l_Lean_Expr_app___override(v___x_1480_, v___x_1519_);
v___x_1521_ = l_Lean_Expr_app___override(v___x_1505_, v___x_1518_);
v___x_1522_ = l_Lean_Expr_app___override(v___x_1520_, v___x_1521_);
v___x_1523_ = l_Lean_Expr_lam___override(v___x_1511_, v___x_1515_, v___x_1522_, v___x_1448_);
v___x_1524_ = l_Lean_Expr_lam___override(v___x_1447_, v_R_1338_, v___x_1523_, v___x_1448_);
v___x_1525_ = l_Lean_Expr_app___override(v___x_1510_, v___x_1524_);
v___x_1526_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__27));
v___x_1527_ = l_Lean_Expr_const___override(v___x_1526_, v___x_1403_);
v___x_1528_ = l_Lean_Expr_app___override(v___x_1527_, v_R_x27_1339_);
v___x_1529_ = l_Lean_Expr_app___override(v___x_1528_, v_a_1364_);
v___x_1530_ = l_Lean_Expr_app___override(v___x_1529_, v_R_1338_);
v___x_1531_ = l_Lean_Expr_app___override(v___x_1530_, v___x_1376_);
v___x_1532_ = l_Lean_Expr_app___override(v___x_1531_, v_a_1379_);
v___x_1533_ = l_Lean_Expr_app___override(v___x_1532_, v_A_1340_);
v___x_1534_ = l_Lean_Expr_app___override(v___x_1533_, v___x_1397_);
v___x_1535_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__29));
v___x_1536_ = l_Lean_Expr_const___override(v___x_1535_, v___x_1402_);
v___x_1537_ = l_Lean_Expr_app___override(v___x_1536_, v_R_1338_);
v___x_1538_ = l_Lean_Expr_app___override(v___x_1537_, v_A_1340_);
v___x_1539_ = l_Lean_Expr_app___override(v___x_1538_, v_sR_1341_);
v___x_1540_ = l_Lean_Expr_app___override(v___x_1539_, v___x_1396_);
v___x_1541_ = l_Lean_Expr_app___override(v___x_1540_, v_sAlg_1343_);
v___x_1542_ = l_Lean_Expr_app___override(v___x_1534_, v___x_1541_);
v___x_1543_ = l_Lean_Expr_app___override(v___x_1542_, v_a_1400_);
v___x_1544_ = l_Lean_Expr_app___override(v___x_1543_, v_a_1425_);
v___x_1545_ = l_Lean_Expr_app___override(v___x_1544_, v_r_x27_1345_);
v___x_1546_ = l_Lean_Expr_app___override(v___x_1545_, v___x_1494_);
v___x_1547_ = l_Lean_Expr_app___override(v___x_1525_, v___x_1546_);
lean_inc_ref(v_expr_1469_);
v___x_1548_ = l_Lean_Expr_app___override(v___x_1547_, v_expr_1469_);
v___x_1549_ = l_Lean_Expr_app___override(v___x_1548_, v_a_1471_);
v___x_1550_ = l_Lean_Expr_lam___override(v___x_1476_, v_A_1340_, v___x_1549_, v___x_1448_);
v___x_1551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1551_, 0, v_expr_1469_);
lean_ctor_set(v___x_1551_, 1, v___x_1550_);
if (v_isShared_1474_ == 0)
{
lean_ctor_set(v___x_1473_, 0, v___x_1551_);
v___x_1553_ = v___x_1473_;
goto v_reusejp_1552_;
}
else
{
lean_object* v_reuseFailAlloc_1554_; 
v_reuseFailAlloc_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1554_, 0, v___x_1551_);
v___x_1553_ = v_reuseFailAlloc_1554_;
goto v_reusejp_1552_;
}
v_reusejp_1552_:
{
return v___x_1553_;
}
}
}
else
{
lean_object* v_a_1556_; lean_object* v___x_1558_; uint8_t v_isShared_1559_; uint8_t v_isSharedCheck_1563_; 
lean_dec_ref(v_expr_1469_);
lean_dec_ref(v___x_1466_);
lean_dec_ref_known(v___x_1428_, 2);
lean_dec(v_a_1425_);
lean_dec_ref(v___x_1421_);
lean_dec_ref_known(v___x_1403_, 2);
lean_dec_ref_known(v___x_1402_, 2);
lean_dec(v_a_1400_);
lean_dec_ref(v___x_1397_);
lean_dec_ref(v___x_1396_);
lean_dec_ref_known(v___x_1383_, 2);
lean_dec_ref_known(v___x_1382_, 2);
lean_dec(v___x_1380_);
lean_dec(v_a_1379_);
lean_dec_ref(v___x_1376_);
lean_dec(v_a_1364_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1556_ = lean_ctor_get(v___x_1470_, 0);
v_isSharedCheck_1563_ = !lean_is_exclusive(v___x_1470_);
if (v_isSharedCheck_1563_ == 0)
{
v___x_1558_ = v___x_1470_;
v_isShared_1559_ = v_isSharedCheck_1563_;
goto v_resetjp_1557_;
}
else
{
lean_inc(v_a_1556_);
lean_dec(v___x_1470_);
v___x_1558_ = lean_box(0);
v_isShared_1559_ = v_isSharedCheck_1563_;
goto v_resetjp_1557_;
}
v_resetjp_1557_:
{
lean_object* v___x_1561_; 
if (v_isShared_1559_ == 0)
{
v___x_1561_ = v___x_1558_;
goto v_reusejp_1560_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_a_1556_);
v___x_1561_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1560_;
}
v_reusejp_1560_:
{
return v___x_1561_;
}
}
}
}
else
{
lean_object* v_a_1564_; lean_object* v___x_1566_; uint8_t v_isShared_1567_; uint8_t v_isSharedCheck_1571_; 
lean_dec_ref(v___x_1466_);
lean_dec_ref_known(v___x_1428_, 2);
lean_dec(v_a_1425_);
lean_dec_ref(v___x_1421_);
lean_dec_ref_known(v___x_1403_, 2);
lean_dec_ref_known(v___x_1402_, 2);
lean_dec(v_a_1400_);
lean_dec_ref(v___x_1397_);
lean_dec_ref(v___x_1396_);
lean_dec_ref_known(v___x_1383_, 2);
lean_dec_ref_known(v___x_1382_, 2);
lean_dec(v___x_1380_);
lean_dec(v_a_1379_);
lean_dec_ref(v___x_1376_);
lean_dec(v_a_1364_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1564_ = lean_ctor_get(v___x_1467_, 0);
v_isSharedCheck_1571_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1571_ == 0)
{
v___x_1566_ = v___x_1467_;
v_isShared_1567_ = v_isSharedCheck_1571_;
goto v_resetjp_1565_;
}
else
{
lean_inc(v_a_1564_);
lean_dec(v___x_1467_);
v___x_1566_ = lean_box(0);
v_isShared_1567_ = v_isSharedCheck_1571_;
goto v_resetjp_1565_;
}
v_resetjp_1565_:
{
lean_object* v___x_1569_; 
if (v_isShared_1567_ == 0)
{
v___x_1569_ = v___x_1566_;
goto v_reusejp_1568_;
}
else
{
lean_object* v_reuseFailAlloc_1570_; 
v_reuseFailAlloc_1570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1570_, 0, v_a_1564_);
v___x_1569_ = v_reuseFailAlloc_1570_;
goto v_reusejp_1568_;
}
v_reusejp_1568_:
{
return v___x_1569_;
}
}
}
}
else
{
lean_object* v_a_1572_; lean_object* v___x_1574_; uint8_t v_isShared_1575_; uint8_t v_isSharedCheck_1579_; 
lean_dec_ref(v___x_1421_);
lean_dec_ref_known(v___x_1403_, 2);
lean_dec_ref_known(v___x_1402_, 2);
lean_dec(v_a_1400_);
lean_dec_ref(v___x_1397_);
lean_dec_ref(v___x_1396_);
lean_dec_ref(v___x_1389_);
lean_dec_ref_known(v___x_1383_, 2);
lean_dec_ref_known(v___x_1382_, 2);
lean_dec(v___x_1380_);
lean_dec(v_a_1379_);
lean_dec_ref(v___x_1376_);
lean_dec_ref_known(v___x_1368_, 2);
lean_dec_ref_known(v___x_1367_, 2);
lean_dec(v___x_1365_);
lean_dec(v_a_1364_);
lean_dec_ref_known(v___x_1360_, 2);
lean_dec(v___x_1357_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1572_ = lean_ctor_get(v___x_1424_, 0);
v_isSharedCheck_1579_ = !lean_is_exclusive(v___x_1424_);
if (v_isSharedCheck_1579_ == 0)
{
v___x_1574_ = v___x_1424_;
v_isShared_1575_ = v_isSharedCheck_1579_;
goto v_resetjp_1573_;
}
else
{
lean_inc(v_a_1572_);
lean_dec(v___x_1424_);
v___x_1574_ = lean_box(0);
v_isShared_1575_ = v_isSharedCheck_1579_;
goto v_resetjp_1573_;
}
v_resetjp_1573_:
{
lean_object* v___x_1577_; 
if (v_isShared_1575_ == 0)
{
v___x_1577_ = v___x_1574_;
goto v_reusejp_1576_;
}
else
{
lean_object* v_reuseFailAlloc_1578_; 
v_reuseFailAlloc_1578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1578_, 0, v_a_1572_);
v___x_1577_ = v_reuseFailAlloc_1578_;
goto v_reusejp_1576_;
}
v_reusejp_1576_:
{
return v___x_1577_;
}
}
}
}
else
{
lean_object* v_a_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1587_; 
lean_dec_ref(v___x_1397_);
lean_dec_ref(v___x_1396_);
lean_dec_ref(v___x_1389_);
lean_dec_ref_known(v___x_1383_, 2);
lean_dec_ref_known(v___x_1382_, 2);
lean_dec(v___x_1380_);
lean_dec(v_a_1379_);
lean_dec_ref(v___x_1376_);
lean_dec_ref_known(v___x_1368_, 2);
lean_dec_ref_known(v___x_1367_, 2);
lean_dec(v___x_1365_);
lean_dec(v_a_1364_);
lean_dec_ref_known(v___x_1360_, 2);
lean_dec(v___x_1357_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1580_ = lean_ctor_get(v___x_1399_, 0);
v_isSharedCheck_1587_ = !lean_is_exclusive(v___x_1399_);
if (v_isSharedCheck_1587_ == 0)
{
v___x_1582_ = v___x_1399_;
v_isShared_1583_ = v_isSharedCheck_1587_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_a_1580_);
lean_dec(v___x_1399_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1587_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
lean_object* v___x_1585_; 
if (v_isShared_1583_ == 0)
{
v___x_1585_ = v___x_1582_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1586_; 
v_reuseFailAlloc_1586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1586_, 0, v_a_1580_);
v___x_1585_ = v_reuseFailAlloc_1586_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
return v___x_1585_;
}
}
}
}
else
{
lean_object* v_a_1588_; lean_object* v___x_1590_; uint8_t v_isShared_1591_; uint8_t v_isSharedCheck_1595_; 
lean_dec_ref(v___x_1376_);
lean_dec_ref_known(v___x_1368_, 2);
lean_dec_ref_known(v___x_1367_, 2);
lean_dec(v___x_1365_);
lean_dec(v_a_1364_);
lean_dec_ref_known(v___x_1360_, 2);
lean_dec(v___x_1357_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sA_1342_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1588_ = lean_ctor_get(v___x_1378_, 0);
v_isSharedCheck_1595_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1595_ == 0)
{
v___x_1590_ = v___x_1378_;
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
else
{
lean_inc(v_a_1588_);
lean_dec(v___x_1378_);
v___x_1590_ = lean_box(0);
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
v_resetjp_1589_:
{
lean_object* v___x_1593_; 
if (v_isShared_1591_ == 0)
{
v___x_1593_ = v___x_1590_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v_a_1588_);
v___x_1593_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
return v___x_1593_;
}
}
}
}
else
{
lean_object* v_a_1596_; lean_object* v___x_1598_; uint8_t v_isShared_1599_; uint8_t v_isSharedCheck_1603_; 
lean_dec_ref_known(v___x_1360_, 2);
lean_dec(v___x_1357_);
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sA_1342_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1596_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1603_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1603_ == 0)
{
v___x_1598_ = v___x_1363_;
v_isShared_1599_ = v_isSharedCheck_1603_;
goto v_resetjp_1597_;
}
else
{
lean_inc(v_a_1596_);
lean_dec(v___x_1363_);
v___x_1598_ = lean_box(0);
v_isShared_1599_ = v_isSharedCheck_1603_;
goto v_resetjp_1597_;
}
v_resetjp_1597_:
{
lean_object* v___x_1601_; 
if (v_isShared_1599_ == 0)
{
v___x_1601_ = v___x_1598_;
goto v_reusejp_1600_;
}
else
{
lean_object* v_reuseFailAlloc_1602_; 
v_reuseFailAlloc_1602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1602_, 0, v_a_1596_);
v___x_1601_ = v_reuseFailAlloc_1602_;
goto v_reusejp_1600_;
}
v_reusejp_1600_:
{
return v___x_1601_;
}
}
}
}
else
{
lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; uint8_t v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1645_; 
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_R_x27_1339_);
lean_dec(v_u_1335_);
lean_inc_n(v_v_1337_, 2);
v___x_1604_ = l_Lean_Level_succ___override(v_v_1337_);
v___x_1605_ = lean_box(0);
v___x_1606_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1606_, 0, v___x_1604_);
lean_ctor_set(v___x_1606_, 1, v___x_1605_);
v___x_1607_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16));
v___x_1608_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1608_, 0, v_v_1337_);
lean_ctor_set(v___x_1608_, 1, v___x_1605_);
lean_inc_ref_n(v___x_1608_, 2);
v___x_1609_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1609_, 0, v_v_1337_);
lean_ctor_set(v___x_1609_, 1, v___x_1608_);
v___x_1610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18));
v___x_1611_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__9));
v___x_1612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_1613_ = l_Lean_Expr_const___override(v___x_1612_, v___x_1608_);
lean_inc_ref_n(v_A_1340_, 6);
v___x_1614_ = l_Lean_Expr_app___override(v___x_1613_, v_A_1340_);
v___x_1615_ = l_Lean_Expr_app___override(v___x_1614_, v_sA_1342_);
v___x_1616_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19);
lean_inc(v_u_x27_1336_);
v___x_1617_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1617_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1617_, 1, v___x_1609_);
v___x_1618_ = l_Lean_Expr_const___override(v___x_1607_, v___x_1617_);
v___x_1619_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1619_, 0, v_u_x27_1336_);
lean_ctor_set(v___x_1619_, 1, v___x_1608_);
lean_inc_ref(v___x_1619_);
v___x_1620_ = l_Lean_Expr_const___override(v___x_1610_, v___x_1619_);
v___x_1621_ = 0;
v___x_1622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_1623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__31));
v___x_1624_ = l_Lean_Expr_const___override(v___x_1623_, v___x_1606_);
v___x_1625_ = l_Lean_Expr_app___override(v___x_1624_, v_A_1340_);
lean_inc_ref_n(v_R_1338_, 2);
v___x_1626_ = l_Lean_Expr_app___override(v___x_1618_, v_R_1338_);
v___x_1627_ = l_Lean_Expr_app___override(v___x_1626_, v_A_1340_);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1627_, v_A_1340_);
v___x_1629_ = l_Lean_Expr_app___override(v___x_1620_, v_R_1338_);
v___x_1630_ = l_Lean_Expr_app___override(v___x_1629_, v_A_1340_);
v___x_1631_ = l_Lean_Expr_const___override(v___x_1611_, v___x_1619_);
v___x_1632_ = l_Lean_Expr_app___override(v___x_1631_, v_R_1338_);
v___x_1633_ = l_Lean_Expr_app___override(v___x_1632_, v_A_1340_);
v___x_1634_ = l_Lean_Expr_app___override(v___x_1633_, v_sR_1341_);
v___x_1635_ = l_Lean_Expr_app___override(v___x_1634_, v___x_1615_);
v___x_1636_ = l_Lean_Expr_app___override(v___x_1635_, v_sAlg_1343_);
v___x_1637_ = l_Lean_Expr_app___override(v___x_1630_, v___x_1636_);
v___x_1638_ = l_Lean_Expr_app___override(v___x_1628_, v___x_1637_);
lean_inc_ref(v_r_x27_1345_);
v___x_1639_ = l_Lean_Expr_app___override(v___x_1638_, v_r_x27_1345_);
v___x_1640_ = l_Lean_Expr_app___override(v___x_1639_, v___x_1616_);
v___x_1641_ = l_Lean_Expr_app___override(v___x_1625_, v___x_1640_);
v___x_1642_ = l_Lean_Expr_lam___override(v___x_1622_, v_A_1340_, v___x_1641_, v___x_1621_);
v___x_1643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1643_, 0, v_r_x27_1345_);
lean_ctor_set(v___x_1643_, 1, v___x_1642_);
if (v_isShared_1355_ == 0)
{
lean_ctor_set(v___x_1354_, 0, v___x_1643_);
v___x_1645_ = v___x_1354_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1646_; 
v_reuseFailAlloc_1646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1646_, 0, v___x_1643_);
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
lean_dec_ref(v_r_x27_1345_);
lean_dec_ref(v_smul_1344_);
lean_dec_ref(v_sAlg_1343_);
lean_dec_ref(v_sA_1342_);
lean_dec_ref(v_sR_1341_);
lean_dec_ref(v_A_1340_);
lean_dec_ref(v_R_x27_1339_);
lean_dec_ref(v_R_1338_);
lean_dec(v_v_1337_);
lean_dec(v_u_x27_1336_);
lean_dec(v_u_1335_);
v_a_1648_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1351_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1351_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___boxed(lean_object* v_u_1656_, lean_object* v_u_x27_1657_, lean_object* v_v_1658_, lean_object* v_R_1659_, lean_object* v_R_x27_1660_, lean_object* v_A_1661_, lean_object* v_sR_1662_, lean_object* v_sA_1663_, lean_object* v_sAlg_1664_, lean_object* v_smul_1665_, lean_object* v_r_x27_1666_, lean_object* v_a_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_, lean_object* v_a_1670_, lean_object* v_a_1671_){
_start:
{
lean_object* v_res_1672_; 
v_res_1672_ = lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast(v_u_1656_, v_u_x27_1657_, v_v_1658_, v_R_1659_, v_R_x27_1660_, v_A_1661_, v_sR_1662_, v_sA_1663_, v_sAlg_1664_, v_smul_1665_, v_r_x27_1666_, v_a_1667_, v_a_1668_, v_a_1669_, v_a_1670_);
lean_dec(v_a_1670_);
lean_dec_ref(v_a_1669_);
lean_dec(v_a_1668_);
lean_dec_ref(v_a_1667_);
return v_res_1672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg(lean_object* v_u_1685_, lean_object* v_v_1686_, lean_object* v_R_1687_, lean_object* v_A_1688_, lean_object* v_sR_1689_, lean_object* v_sA_1690_, lean_object* v_sAlg_1691_, lean_object* v_cR_1692_, lean_object* v_za_1693_, lean_object* v_zb_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_){
_start:
{
lean_object* v_r_1700_; lean_object* v_x_1701_; lean_object* v___x_1703_; uint8_t v_isShared_1704_; uint8_t v_isSharedCheck_1908_; 
v_r_1700_ = lean_ctor_get(v_za_1693_, 0);
v_x_1701_ = lean_ctor_get(v_za_1693_, 1);
v_isSharedCheck_1908_ = !lean_is_exclusive(v_za_1693_);
if (v_isSharedCheck_1908_ == 0)
{
v___x_1703_ = v_za_1693_;
v_isShared_1704_ = v_isSharedCheck_1908_;
goto v_resetjp_1702_;
}
else
{
lean_inc(v_x_1701_);
lean_inc(v_r_1700_);
lean_dec(v_za_1693_);
v___x_1703_ = lean_box(0);
v_isShared_1704_ = v_isSharedCheck_1908_;
goto v_resetjp_1702_;
}
v_resetjp_1702_:
{
lean_object* v_r_1705_; lean_object* v_x_1706_; lean_object* v___x_1708_; uint8_t v_isShared_1709_; uint8_t v_isSharedCheck_1907_; 
v_r_1705_ = lean_ctor_get(v_zb_1694_, 0);
v_x_1706_ = lean_ctor_get(v_zb_1694_, 1);
v_isSharedCheck_1907_ = !lean_is_exclusive(v_zb_1694_);
if (v_isSharedCheck_1907_ == 0)
{
v___x_1708_ = v_zb_1694_;
v_isShared_1709_ = v_isSharedCheck_1907_;
goto v_resetjp_1707_;
}
else
{
lean_inc(v_x_1706_);
lean_inc(v_r_1705_);
lean_dec(v_zb_1694_);
v___x_1708_ = lean_box(0);
v_isShared_1709_ = v_isSharedCheck_1907_;
goto v_resetjp_1707_;
}
v_resetjp_1707_:
{
lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; 
lean_inc_ref_n(v_sR_1689_, 2);
lean_inc_ref_n(v_R_1687_, 2);
lean_inc_n(v_u_1685_, 2);
v___x_1710_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_1685_, v_R_1687_, v_sR_1689_, v_cR_1692_);
v___x_1711_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_r_1705_);
v___x_1712_ = lp_mathlib_Mathlib_Tactic_Ring_Common_evalAdd___redArg(v_u_1685_, v_R_1687_, v_sR_1689_, v___x_1710_, v___x_1711_, v_r_1705_, v_x_1701_, v_x_1706_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_);
if (lean_obj_tag(v___x_1712_) == 0)
{
lean_object* v_a_1713_; lean_object* v___x_1715_; uint8_t v_isShared_1716_; uint8_t v_isSharedCheck_1898_; 
v_a_1713_ = lean_ctor_get(v___x_1712_, 0);
v_isSharedCheck_1898_ = !lean_is_exclusive(v___x_1712_);
if (v_isSharedCheck_1898_ == 0)
{
v___x_1715_ = v___x_1712_;
v_isShared_1716_ = v_isSharedCheck_1898_;
goto v_resetjp_1714_;
}
else
{
lean_inc(v_a_1713_);
lean_dec(v___x_1712_);
v___x_1715_ = lean_box(0);
v_isShared_1716_ = v_isSharedCheck_1898_;
goto v_resetjp_1714_;
}
v_resetjp_1714_:
{
lean_object* v_val_1717_; 
v_val_1717_ = lean_ctor_get(v_a_1713_, 1);
lean_inc(v_val_1717_);
if (lean_obj_tag(v_val_1717_) == 0)
{
lean_object* v_expr_1718_; lean_object* v_proof_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1811_; 
v_expr_1718_ = lean_ctor_get(v_a_1713_, 0);
v_proof_1719_ = lean_ctor_get(v_a_1713_, 2);
v_isSharedCheck_1811_ = !lean_is_exclusive(v_a_1713_);
if (v_isSharedCheck_1811_ == 0)
{
lean_object* v_unused_1812_; 
v_unused_1812_ = lean_ctor_get(v_a_1713_, 1);
lean_dec(v_unused_1812_);
v___x_1721_ = v_a_1713_;
v_isShared_1722_ = v_isSharedCheck_1811_;
goto v_resetjp_1720_;
}
else
{
lean_inc(v_proof_1719_);
lean_inc(v_expr_1718_);
lean_dec(v_a_1713_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1811_;
goto v_resetjp_1720_;
}
v_resetjp_1720_:
{
lean_object* v___x_1723_; lean_object* v___x_1725_; 
v___x_1723_ = lean_box(0);
lean_inc(v_v_1686_);
if (v_isShared_1704_ == 0)
{
lean_ctor_set_tag(v___x_1703_, 1);
lean_ctor_set(v___x_1703_, 1, v___x_1723_);
lean_ctor_set(v___x_1703_, 0, v_v_1686_);
v___x_1725_ = v___x_1703_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1810_; 
v_reuseFailAlloc_1810_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1810_, 0, v_v_1686_);
lean_ctor_set(v_reuseFailAlloc_1810_, 1, v___x_1723_);
v___x_1725_ = v_reuseFailAlloc_1810_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; uint8_t v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1779_; 
v___x_1726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
lean_inc_ref_n(v___x_1725_, 2);
v___x_1727_ = l_Lean_Expr_const___override(v___x_1726_, v___x_1725_);
lean_inc_ref_n(v_A_1688_, 6);
v___x_1728_ = l_Lean_Expr_app___override(v___x_1727_, v_A_1688_);
lean_inc_ref(v_sA_1690_);
v___x_1729_ = l_Lean_Expr_app___override(v___x_1728_, v_sA_1690_);
v___x_1730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_1685_, 2);
v___x_1731_ = l_Lean_Level_succ___override(v_u_1685_);
v___x_1732_ = l_Lean_Level_succ___override(v_v_1686_);
lean_inc(v___x_1732_);
lean_inc(v___x_1731_);
v___x_1733_ = l_Lean_Level_max___override(v___x_1731_, v___x_1732_);
v___x_1734_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1734_, 0, v___x_1732_);
lean_ctor_set(v___x_1734_, 1, v___x_1723_);
v___x_1735_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1731_);
lean_ctor_set(v___x_1735_, 1, v___x_1734_);
v___x_1736_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1736_, 0, v___x_1733_);
lean_ctor_set(v___x_1736_, 1, v___x_1735_);
v___x_1737_ = l_Lean_Expr_const___override(v___x_1730_, v___x_1736_);
v___x_1738_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_1739_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1739_, 0, v_u_1685_);
lean_ctor_set(v___x_1739_, 1, v___x_1725_);
lean_inc_ref_n(v___x_1739_, 3);
v___x_1740_ = l_Lean_Expr_const___override(v___x_1738_, v___x_1739_);
lean_inc_ref_n(v_R_1687_, 7);
v___x_1741_ = l_Lean_Expr_app___override(v___x_1740_, v_R_1687_);
v___x_1742_ = l_Lean_Expr_app___override(v___x_1741_, v_A_1688_);
v___x_1743_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_1744_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1744_, 0, v_u_1685_);
lean_ctor_set(v___x_1744_, 1, v___x_1723_);
lean_inc_ref(v___x_1744_);
v___x_1745_ = l_Lean_Expr_const___override(v___x_1743_, v___x_1744_);
v___x_1746_ = l_Lean_Expr_app___override(v___x_1745_, v_R_1687_);
v___x_1747_ = l_Lean_Expr_const___override(v___x_1726_, v___x_1744_);
v___x_1748_ = l_Lean_Expr_app___override(v___x_1747_, v_R_1687_);
lean_inc_ref_n(v_sR_1689_, 2);
v___x_1749_ = l_Lean_Expr_app___override(v___x_1748_, v_sR_1689_);
v___x_1750_ = l_Lean_Expr_app___override(v___x_1746_, v___x_1749_);
lean_inc_ref(v___x_1750_);
v___x_1751_ = l_Lean_Expr_app___override(v___x_1742_, v___x_1750_);
v___x_1752_ = l_Lean_Expr_const___override(v___x_1743_, v___x_1725_);
v___x_1753_ = l_Lean_Expr_app___override(v___x_1752_, v_A_1688_);
lean_inc_ref(v___x_1729_);
v___x_1754_ = l_Lean_Expr_app___override(v___x_1753_, v___x_1729_);
lean_inc_ref(v___x_1754_);
v___x_1755_ = l_Lean_Expr_app___override(v___x_1751_, v___x_1754_);
v___x_1756_ = l_Lean_Expr_app___override(v___x_1737_, v___x_1755_);
v___x_1757_ = l_Lean_Expr_app___override(v___x_1756_, v_R_1687_);
v___x_1758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_1759_ = 0;
v___x_1760_ = l_Lean_Expr_lam___override(v___x_1758_, v_R_1687_, v_A_1688_, v___x_1759_);
v___x_1761_ = l_Lean_Expr_app___override(v___x_1757_, v___x_1760_);
v___x_1762_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_1763_ = l_Lean_Expr_const___override(v___x_1762_, v___x_1739_);
v___x_1764_ = l_Lean_Expr_app___override(v___x_1763_, v_R_1687_);
v___x_1765_ = l_Lean_Expr_app___override(v___x_1764_, v_A_1688_);
v___x_1766_ = l_Lean_Expr_app___override(v___x_1765_, v___x_1750_);
v___x_1767_ = l_Lean_Expr_app___override(v___x_1766_, v___x_1754_);
v___x_1768_ = l_Lean_Expr_app___override(v___x_1761_, v___x_1767_);
v___x_1769_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_1770_ = l_Lean_Expr_const___override(v___x_1769_, v___x_1739_);
v___x_1771_ = l_Lean_Expr_app___override(v___x_1770_, v_R_1687_);
v___x_1772_ = l_Lean_Expr_app___override(v___x_1771_, v_A_1688_);
v___x_1773_ = l_Lean_Expr_app___override(v___x_1772_, v_sR_1689_);
v___x_1774_ = l_Lean_Expr_app___override(v___x_1773_, v___x_1729_);
lean_inc_ref(v_sAlg_1691_);
v___x_1775_ = l_Lean_Expr_app___override(v___x_1774_, v_sAlg_1691_);
v___x_1776_ = l_Lean_Expr_app___override(v___x_1768_, v___x_1775_);
lean_inc_ref_n(v_expr_1718_, 2);
v___x_1777_ = l_Lean_Expr_app___override(v___x_1776_, v_expr_1718_);
if (v_isShared_1709_ == 0)
{
lean_ctor_set(v___x_1708_, 1, v_val_1717_);
lean_ctor_set(v___x_1708_, 0, v_expr_1718_);
v___x_1779_ = v___x_1708_;
goto v_reusejp_1778_;
}
else
{
lean_object* v_reuseFailAlloc_1809_; 
v_reuseFailAlloc_1809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1809_, 0, v_expr_1718_);
lean_ctor_set(v_reuseFailAlloc_1809_, 1, v_val_1717_);
v___x_1779_ = v_reuseFailAlloc_1809_;
goto v_reusejp_1778_;
}
v_reusejp_1778_:
{
lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1792_; 
v___x_1780_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1));
lean_inc_ref(v___x_1739_);
v___x_1781_ = l_Lean_Expr_const___override(v___x_1780_, v___x_1739_);
lean_inc_ref(v_R_1687_);
v___x_1782_ = l_Lean_Expr_app___override(v___x_1781_, v_R_1687_);
lean_inc_ref(v_A_1688_);
v___x_1783_ = l_Lean_Expr_app___override(v___x_1782_, v_A_1688_);
lean_inc_ref(v_sR_1689_);
v___x_1784_ = l_Lean_Expr_app___override(v___x_1783_, v_sR_1689_);
lean_inc_ref(v_sA_1690_);
v___x_1785_ = l_Lean_Expr_app___override(v___x_1784_, v_sA_1690_);
lean_inc_ref(v_sAlg_1691_);
v___x_1786_ = l_Lean_Expr_app___override(v___x_1785_, v_sAlg_1691_);
lean_inc_ref(v_r_1700_);
v___x_1787_ = l_Lean_Expr_app___override(v___x_1786_, v_r_1700_);
lean_inc_ref(v_r_1705_);
v___x_1788_ = l_Lean_Expr_app___override(v___x_1787_, v_r_1705_);
v___x_1789_ = l_Lean_Expr_app___override(v___x_1788_, v_expr_1718_);
lean_inc_ref(v_proof_1719_);
v___x_1790_ = l_Lean_Expr_app___override(v___x_1789_, v_proof_1719_);
if (v_isShared_1722_ == 0)
{
lean_ctor_set(v___x_1721_, 2, v___x_1790_);
lean_ctor_set(v___x_1721_, 1, v___x_1779_);
lean_ctor_set(v___x_1721_, 0, v___x_1777_);
v___x_1792_ = v___x_1721_;
goto v_reusejp_1791_;
}
else
{
lean_object* v_reuseFailAlloc_1808_; 
v_reuseFailAlloc_1808_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1808_, 0, v___x_1777_);
lean_ctor_set(v_reuseFailAlloc_1808_, 1, v___x_1779_);
lean_ctor_set(v_reuseFailAlloc_1808_, 2, v___x_1790_);
v___x_1792_ = v_reuseFailAlloc_1808_;
goto v_reusejp_1791_;
}
v_reusejp_1791_:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1806_; 
v___x_1793_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__3));
v___x_1794_ = l_Lean_Expr_const___override(v___x_1793_, v___x_1739_);
v___x_1795_ = l_Lean_Expr_app___override(v___x_1794_, v_R_1687_);
v___x_1796_ = l_Lean_Expr_app___override(v___x_1795_, v_A_1688_);
v___x_1797_ = l_Lean_Expr_app___override(v___x_1796_, v_sR_1689_);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1797_, v_sA_1690_);
v___x_1799_ = l_Lean_Expr_app___override(v___x_1798_, v_sAlg_1691_);
v___x_1800_ = l_Lean_Expr_app___override(v___x_1799_, v_r_1700_);
v___x_1801_ = l_Lean_Expr_app___override(v___x_1800_, v_r_1705_);
v___x_1802_ = l_Lean_Expr_app___override(v___x_1801_, v_proof_1719_);
v___x_1803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1803_, 0, v___x_1802_);
v___x_1804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1804_, 0, v___x_1792_);
lean_ctor_set(v___x_1804_, 1, v___x_1803_);
if (v_isShared_1716_ == 0)
{
lean_ctor_set(v___x_1715_, 0, v___x_1804_);
v___x_1806_ = v___x_1715_;
goto v_reusejp_1805_;
}
else
{
lean_object* v_reuseFailAlloc_1807_; 
v_reuseFailAlloc_1807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1807_, 0, v___x_1804_);
v___x_1806_ = v_reuseFailAlloc_1807_;
goto v_reusejp_1805_;
}
v_reusejp_1805_:
{
return v___x_1806_;
}
}
}
}
}
}
else
{
lean_object* v_expr_1813_; lean_object* v_proof_1814_; lean_object* v___x_1816_; uint8_t v_isShared_1817_; uint8_t v_isSharedCheck_1896_; 
v_expr_1813_ = lean_ctor_get(v_a_1713_, 0);
v_proof_1814_ = lean_ctor_get(v_a_1713_, 2);
v_isSharedCheck_1896_ = !lean_is_exclusive(v_a_1713_);
if (v_isSharedCheck_1896_ == 0)
{
lean_object* v_unused_1897_; 
v_unused_1897_ = lean_ctor_get(v_a_1713_, 1);
lean_dec(v_unused_1897_);
v___x_1816_ = v_a_1713_;
v_isShared_1817_ = v_isSharedCheck_1896_;
goto v_resetjp_1815_;
}
else
{
lean_inc(v_proof_1814_);
lean_inc(v_expr_1813_);
lean_dec(v_a_1713_);
v___x_1816_ = lean_box(0);
v_isShared_1817_ = v_isSharedCheck_1896_;
goto v_resetjp_1815_;
}
v_resetjp_1815_:
{
lean_object* v___x_1818_; lean_object* v___x_1820_; 
v___x_1818_ = lean_box(0);
lean_inc(v_v_1686_);
if (v_isShared_1704_ == 0)
{
lean_ctor_set_tag(v___x_1703_, 1);
lean_ctor_set(v___x_1703_, 1, v___x_1818_);
lean_ctor_set(v___x_1703_, 0, v_v_1686_);
v___x_1820_ = v___x_1703_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1895_; 
v_reuseFailAlloc_1895_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1895_, 0, v_v_1686_);
lean_ctor_set(v_reuseFailAlloc_1895_, 1, v___x_1818_);
v___x_1820_ = v_reuseFailAlloc_1895_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; uint8_t v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1874_; 
v___x_1821_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
lean_inc_ref_n(v___x_1820_, 2);
v___x_1822_ = l_Lean_Expr_const___override(v___x_1821_, v___x_1820_);
lean_inc_ref_n(v_A_1688_, 6);
v___x_1823_ = l_Lean_Expr_app___override(v___x_1822_, v_A_1688_);
lean_inc_ref(v_sA_1690_);
v___x_1824_ = l_Lean_Expr_app___override(v___x_1823_, v_sA_1690_);
v___x_1825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_1685_, 2);
v___x_1826_ = l_Lean_Level_succ___override(v_u_1685_);
v___x_1827_ = l_Lean_Level_succ___override(v_v_1686_);
lean_inc(v___x_1827_);
lean_inc(v___x_1826_);
v___x_1828_ = l_Lean_Level_max___override(v___x_1826_, v___x_1827_);
v___x_1829_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1829_, 0, v___x_1827_);
lean_ctor_set(v___x_1829_, 1, v___x_1818_);
v___x_1830_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1830_, 0, v___x_1826_);
lean_ctor_set(v___x_1830_, 1, v___x_1829_);
v___x_1831_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1831_, 0, v___x_1828_);
lean_ctor_set(v___x_1831_, 1, v___x_1830_);
v___x_1832_ = l_Lean_Expr_const___override(v___x_1825_, v___x_1831_);
v___x_1833_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_1834_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1834_, 0, v_u_1685_);
lean_ctor_set(v___x_1834_, 1, v___x_1820_);
lean_inc_ref_n(v___x_1834_, 3);
v___x_1835_ = l_Lean_Expr_const___override(v___x_1833_, v___x_1834_);
lean_inc_ref_n(v_R_1687_, 7);
v___x_1836_ = l_Lean_Expr_app___override(v___x_1835_, v_R_1687_);
v___x_1837_ = l_Lean_Expr_app___override(v___x_1836_, v_A_1688_);
v___x_1838_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_1839_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1839_, 0, v_u_1685_);
lean_ctor_set(v___x_1839_, 1, v___x_1818_);
lean_inc_ref(v___x_1839_);
v___x_1840_ = l_Lean_Expr_const___override(v___x_1838_, v___x_1839_);
v___x_1841_ = l_Lean_Expr_app___override(v___x_1840_, v_R_1687_);
v___x_1842_ = l_Lean_Expr_const___override(v___x_1821_, v___x_1839_);
v___x_1843_ = l_Lean_Expr_app___override(v___x_1842_, v_R_1687_);
lean_inc_ref_n(v_sR_1689_, 2);
v___x_1844_ = l_Lean_Expr_app___override(v___x_1843_, v_sR_1689_);
v___x_1845_ = l_Lean_Expr_app___override(v___x_1841_, v___x_1844_);
lean_inc_ref(v___x_1845_);
v___x_1846_ = l_Lean_Expr_app___override(v___x_1837_, v___x_1845_);
v___x_1847_ = l_Lean_Expr_const___override(v___x_1838_, v___x_1820_);
v___x_1848_ = l_Lean_Expr_app___override(v___x_1847_, v_A_1688_);
lean_inc_ref(v___x_1824_);
v___x_1849_ = l_Lean_Expr_app___override(v___x_1848_, v___x_1824_);
lean_inc_ref(v___x_1849_);
v___x_1850_ = l_Lean_Expr_app___override(v___x_1846_, v___x_1849_);
v___x_1851_ = l_Lean_Expr_app___override(v___x_1832_, v___x_1850_);
v___x_1852_ = l_Lean_Expr_app___override(v___x_1851_, v_R_1687_);
v___x_1853_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_1854_ = 0;
v___x_1855_ = l_Lean_Expr_lam___override(v___x_1853_, v_R_1687_, v_A_1688_, v___x_1854_);
v___x_1856_ = l_Lean_Expr_app___override(v___x_1852_, v___x_1855_);
v___x_1857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_1858_ = l_Lean_Expr_const___override(v___x_1857_, v___x_1834_);
v___x_1859_ = l_Lean_Expr_app___override(v___x_1858_, v_R_1687_);
v___x_1860_ = l_Lean_Expr_app___override(v___x_1859_, v_A_1688_);
v___x_1861_ = l_Lean_Expr_app___override(v___x_1860_, v___x_1845_);
v___x_1862_ = l_Lean_Expr_app___override(v___x_1861_, v___x_1849_);
v___x_1863_ = l_Lean_Expr_app___override(v___x_1856_, v___x_1862_);
v___x_1864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_1865_ = l_Lean_Expr_const___override(v___x_1864_, v___x_1834_);
v___x_1866_ = l_Lean_Expr_app___override(v___x_1865_, v_R_1687_);
v___x_1867_ = l_Lean_Expr_app___override(v___x_1866_, v_A_1688_);
v___x_1868_ = l_Lean_Expr_app___override(v___x_1867_, v_sR_1689_);
v___x_1869_ = l_Lean_Expr_app___override(v___x_1868_, v___x_1824_);
lean_inc_ref(v_sAlg_1691_);
v___x_1870_ = l_Lean_Expr_app___override(v___x_1869_, v_sAlg_1691_);
v___x_1871_ = l_Lean_Expr_app___override(v___x_1863_, v___x_1870_);
lean_inc_ref_n(v_expr_1813_, 2);
v___x_1872_ = l_Lean_Expr_app___override(v___x_1871_, v_expr_1813_);
if (v_isShared_1709_ == 0)
{
lean_ctor_set(v___x_1708_, 1, v_val_1717_);
lean_ctor_set(v___x_1708_, 0, v_expr_1813_);
v___x_1874_ = v___x_1708_;
goto v_reusejp_1873_;
}
else
{
lean_object* v_reuseFailAlloc_1894_; 
v_reuseFailAlloc_1894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1894_, 0, v_expr_1813_);
lean_ctor_set(v_reuseFailAlloc_1894_, 1, v_val_1717_);
v___x_1874_ = v_reuseFailAlloc_1894_;
goto v_reusejp_1873_;
}
v_reusejp_1873_:
{
lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1887_; 
v___x_1875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___closed__1));
v___x_1876_ = l_Lean_Expr_const___override(v___x_1875_, v___x_1834_);
v___x_1877_ = l_Lean_Expr_app___override(v___x_1876_, v_R_1687_);
v___x_1878_ = l_Lean_Expr_app___override(v___x_1877_, v_A_1688_);
v___x_1879_ = l_Lean_Expr_app___override(v___x_1878_, v_sR_1689_);
v___x_1880_ = l_Lean_Expr_app___override(v___x_1879_, v_sA_1690_);
v___x_1881_ = l_Lean_Expr_app___override(v___x_1880_, v_sAlg_1691_);
v___x_1882_ = l_Lean_Expr_app___override(v___x_1881_, v_r_1700_);
v___x_1883_ = l_Lean_Expr_app___override(v___x_1882_, v_r_1705_);
v___x_1884_ = l_Lean_Expr_app___override(v___x_1883_, v_expr_1813_);
v___x_1885_ = l_Lean_Expr_app___override(v___x_1884_, v_proof_1814_);
if (v_isShared_1817_ == 0)
{
lean_ctor_set(v___x_1816_, 2, v___x_1885_);
lean_ctor_set(v___x_1816_, 1, v___x_1874_);
lean_ctor_set(v___x_1816_, 0, v___x_1872_);
v___x_1887_ = v___x_1816_;
goto v_reusejp_1886_;
}
else
{
lean_object* v_reuseFailAlloc_1893_; 
v_reuseFailAlloc_1893_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1893_, 0, v___x_1872_);
lean_ctor_set(v_reuseFailAlloc_1893_, 1, v___x_1874_);
lean_ctor_set(v_reuseFailAlloc_1893_, 2, v___x_1885_);
v___x_1887_ = v_reuseFailAlloc_1893_;
goto v_reusejp_1886_;
}
v_reusejp_1886_:
{
lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1891_; 
v___x_1888_ = lean_box(0);
v___x_1889_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1889_, 0, v___x_1887_);
lean_ctor_set(v___x_1889_, 1, v___x_1888_);
if (v_isShared_1716_ == 0)
{
lean_ctor_set(v___x_1715_, 0, v___x_1889_);
v___x_1891_ = v___x_1715_;
goto v_reusejp_1890_;
}
else
{
lean_object* v_reuseFailAlloc_1892_; 
v_reuseFailAlloc_1892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1892_, 0, v___x_1889_);
v___x_1891_ = v_reuseFailAlloc_1892_;
goto v_reusejp_1890_;
}
v_reusejp_1890_:
{
return v___x_1891_;
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
lean_object* v_a_1899_; lean_object* v___x_1901_; uint8_t v_isShared_1902_; uint8_t v_isSharedCheck_1906_; 
lean_del_object(v___x_1708_);
lean_dec_ref(v_r_1705_);
lean_del_object(v___x_1703_);
lean_dec_ref(v_r_1700_);
lean_dec_ref(v_sAlg_1691_);
lean_dec_ref(v_sA_1690_);
lean_dec_ref(v_sR_1689_);
lean_dec_ref(v_A_1688_);
lean_dec_ref(v_R_1687_);
lean_dec(v_v_1686_);
lean_dec(v_u_1685_);
v_a_1899_ = lean_ctor_get(v___x_1712_, 0);
v_isSharedCheck_1906_ = !lean_is_exclusive(v___x_1712_);
if (v_isSharedCheck_1906_ == 0)
{
v___x_1901_ = v___x_1712_;
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
else
{
lean_inc(v_a_1899_);
lean_dec(v___x_1712_);
v___x_1901_ = lean_box(0);
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
v_resetjp_1900_:
{
lean_object* v___x_1904_; 
if (v_isShared_1902_ == 0)
{
v___x_1904_ = v___x_1901_;
goto v_reusejp_1903_;
}
else
{
lean_object* v_reuseFailAlloc_1905_; 
v_reuseFailAlloc_1905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1905_, 0, v_a_1899_);
v___x_1904_ = v_reuseFailAlloc_1905_;
goto v_reusejp_1903_;
}
v_reusejp_1903_:
{
return v___x_1904_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg___boxed(lean_object* v_u_1909_, lean_object* v_v_1910_, lean_object* v_R_1911_, lean_object* v_A_1912_, lean_object* v_sR_1913_, lean_object* v_sA_1914_, lean_object* v_sAlg_1915_, lean_object* v_cR_1916_, lean_object* v_za_1917_, lean_object* v_zb_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_, lean_object* v_a_1921_, lean_object* v_a_1922_, lean_object* v_a_1923_){
_start:
{
lean_object* v_res_1924_; 
v_res_1924_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg(v_u_1909_, v_v_1910_, v_R_1911_, v_A_1912_, v_sR_1913_, v_sA_1914_, v_sAlg_1915_, v_cR_1916_, v_za_1917_, v_zb_1918_, v_a_1919_, v_a_1920_, v_a_1921_, v_a_1922_);
lean_dec(v_a_1922_);
lean_dec_ref(v_a_1921_);
lean_dec(v_a_1920_);
lean_dec_ref(v_a_1919_);
return v_res_1924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add(lean_object* v_u_1925_, lean_object* v_v_1926_, lean_object* v_R_1927_, lean_object* v_A_1928_, lean_object* v_sR_1929_, lean_object* v_sA_1930_, lean_object* v_sAlg_1931_, lean_object* v_cR_1932_, lean_object* v_a_1933_, lean_object* v_b_1934_, lean_object* v_za_1935_, lean_object* v_zb_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_, lean_object* v_a_1939_, lean_object* v_a_1940_){
_start:
{
lean_object* v___x_1942_; 
v___x_1942_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___redArg(v_u_1925_, v_v_1926_, v_R_1927_, v_A_1928_, v_sR_1929_, v_sA_1930_, v_sAlg_1931_, v_cR_1932_, v_za_1935_, v_zb_1936_, v_a_1937_, v_a_1938_, v_a_1939_, v_a_1940_);
return v___x_1942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___boxed(lean_object** _args){
lean_object* v_u_1943_ = _args[0];
lean_object* v_v_1944_ = _args[1];
lean_object* v_R_1945_ = _args[2];
lean_object* v_A_1946_ = _args[3];
lean_object* v_sR_1947_ = _args[4];
lean_object* v_sA_1948_ = _args[5];
lean_object* v_sAlg_1949_ = _args[6];
lean_object* v_cR_1950_ = _args[7];
lean_object* v_a_1951_ = _args[8];
lean_object* v_b_1952_ = _args[9];
lean_object* v_za_1953_ = _args[10];
lean_object* v_zb_1954_ = _args[11];
lean_object* v_a_1955_ = _args[12];
lean_object* v_a_1956_ = _args[13];
lean_object* v_a_1957_ = _args[14];
lean_object* v_a_1958_ = _args[15];
lean_object* v_a_1959_ = _args[16];
_start:
{
lean_object* v_res_1960_; 
v_res_1960_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add(v_u_1943_, v_v_1944_, v_R_1945_, v_A_1946_, v_sR_1947_, v_sA_1948_, v_sAlg_1949_, v_cR_1950_, v_a_1951_, v_b_1952_, v_za_1953_, v_zb_1954_, v_a_1955_, v_a_1956_, v_a_1957_, v_a_1958_);
lean_dec(v_a_1958_);
lean_dec_ref(v_a_1957_);
lean_dec(v_a_1956_);
lean_dec_ref(v_a_1955_);
lean_dec_ref(v_b_1952_);
lean_dec_ref(v_a_1951_);
return v_res_1960_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9(void){
_start:
{
lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; 
v___x_1976_ = lean_box(0);
v___x_1977_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__8));
v___x_1978_ = l_Lean_Expr_const___override(v___x_1977_, v___x_1976_);
return v___x_1978_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12(void){
_start:
{
lean_object* v___x_1983_; lean_object* v___x_1984_; 
v___x_1983_ = lean_box(0);
v___x_1984_ = l_Lean_Level_succ___override(v___x_1983_);
return v___x_1984_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13(void){
_start:
{
lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1985_ = lean_box(0);
v___x_1986_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__12);
v___x_1987_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1987_, 0, v___x_1986_);
lean_ctor_set(v___x_1987_, 1, v___x_1985_);
return v___x_1987_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14(void){
_start:
{
lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; 
v___x_1988_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13);
v___x_1989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11));
v___x_1990_ = l_Lean_Expr_const___override(v___x_1989_, v___x_1988_);
return v___x_1990_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15(void){
_start:
{
lean_object* v___x_1991_; lean_object* v___x_1992_; 
v___x_1991_ = lean_box(0);
v___x_1992_ = l_Lean_Expr_sort___override(v___x_1991_);
return v___x_1992_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16(void){
_start:
{
lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; 
v___x_1993_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15);
v___x_1994_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__14);
v___x_1995_ = l_Lean_Expr_app___override(v___x_1994_, v___x_1993_);
return v___x_1995_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19(void){
_start:
{
lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; 
v___x_1999_ = lean_box(0);
v___x_2000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__18));
v___x_2001_ = l_Lean_Expr_const___override(v___x_2000_, v___x_1999_);
return v___x_2001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg(lean_object* v_u_2034_, lean_object* v_v_2035_, lean_object* v_R_2036_, lean_object* v_A_2037_, lean_object* v_sR_2038_, lean_object* v_sA_2039_, lean_object* v_sAlg_2040_, lean_object* v_cR_2041_, lean_object* v_za_2042_, lean_object* v_zb_2043_, lean_object* v_a_2044_, lean_object* v_a_2045_, lean_object* v_a_2046_, lean_object* v_a_2047_){
_start:
{
lean_object* v_r_2049_; lean_object* v_x_2050_; lean_object* v___x_2052_; uint8_t v_isShared_2053_; uint8_t v_isSharedCheck_2304_; 
v_r_2049_ = lean_ctor_get(v_za_2042_, 0);
v_x_2050_ = lean_ctor_get(v_za_2042_, 1);
v_isSharedCheck_2304_ = !lean_is_exclusive(v_za_2042_);
if (v_isSharedCheck_2304_ == 0)
{
v___x_2052_ = v_za_2042_;
v_isShared_2053_ = v_isSharedCheck_2304_;
goto v_resetjp_2051_;
}
else
{
lean_inc(v_x_2050_);
lean_inc(v_r_2049_);
lean_dec(v_za_2042_);
v___x_2052_ = lean_box(0);
v_isShared_2053_ = v_isSharedCheck_2304_;
goto v_resetjp_2051_;
}
v_resetjp_2051_:
{
lean_object* v_r_2054_; lean_object* v_x_2055_; lean_object* v___x_2057_; uint8_t v_isShared_2058_; uint8_t v_isSharedCheck_2303_; 
v_r_2054_ = lean_ctor_get(v_zb_2043_, 0);
v_x_2055_ = lean_ctor_get(v_zb_2043_, 1);
v_isSharedCheck_2303_ = !lean_is_exclusive(v_zb_2043_);
if (v_isSharedCheck_2303_ == 0)
{
v___x_2057_ = v_zb_2043_;
v_isShared_2058_ = v_isSharedCheck_2303_;
goto v_resetjp_2056_;
}
else
{
lean_inc(v_x_2055_);
lean_inc(v_r_2054_);
lean_dec(v_zb_2043_);
v___x_2057_ = lean_box(0);
v_isShared_2058_ = v_isSharedCheck_2303_;
goto v_resetjp_2056_;
}
v_resetjp_2056_:
{
lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; 
lean_inc_ref_n(v_sR_2038_, 2);
lean_inc_ref_n(v_R_2036_, 2);
lean_inc_n(v_u_2034_, 2);
v___x_2059_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_2034_, v_R_2036_, v_sR_2038_, v_cR_2041_);
v___x_2060_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_r_2054_);
v___x_2061_ = lp_mathlib_Mathlib_Tactic_Ring_Common_evalMul___redArg(v_u_2034_, v_R_2036_, v_sR_2038_, v___x_2059_, v___x_2060_, v_r_2054_, v_x_2050_, v_x_2055_, v_a_2044_, v_a_2045_, v_a_2046_, v_a_2047_);
if (lean_obj_tag(v___x_2061_) == 0)
{
lean_object* v_a_2062_; lean_object* v___x_2064_; uint8_t v_isShared_2065_; uint8_t v_isSharedCheck_2294_; 
v_a_2062_ = lean_ctor_get(v___x_2061_, 0);
v_isSharedCheck_2294_ = !lean_is_exclusive(v___x_2061_);
if (v_isSharedCheck_2294_ == 0)
{
v___x_2064_ = v___x_2061_;
v_isShared_2065_ = v_isSharedCheck_2294_;
goto v_resetjp_2063_;
}
else
{
lean_inc(v_a_2062_);
lean_dec(v___x_2061_);
v___x_2064_ = lean_box(0);
v_isShared_2065_ = v_isSharedCheck_2294_;
goto v_resetjp_2063_;
}
v_resetjp_2063_:
{
lean_object* v_expr_2066_; lean_object* v_val_2067_; lean_object* v_proof_2068_; lean_object* v___x_2070_; uint8_t v_isShared_2071_; uint8_t v_isSharedCheck_2293_; 
v_expr_2066_ = lean_ctor_get(v_a_2062_, 0);
v_val_2067_ = lean_ctor_get(v_a_2062_, 1);
v_proof_2068_ = lean_ctor_get(v_a_2062_, 2);
v_isSharedCheck_2293_ = !lean_is_exclusive(v_a_2062_);
if (v_isSharedCheck_2293_ == 0)
{
v___x_2070_ = v_a_2062_;
v_isShared_2071_ = v_isSharedCheck_2293_;
goto v_resetjp_2069_;
}
else
{
lean_inc(v_proof_2068_);
lean_inc(v_val_2067_);
lean_inc(v_expr_2066_);
lean_dec(v_a_2062_);
v___x_2070_ = lean_box(0);
v_isShared_2071_ = v_isSharedCheck_2293_;
goto v_resetjp_2069_;
}
v_resetjp_2069_:
{
lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2075_; 
v___x_2072_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2));
v___x_2073_ = lean_box(0);
lean_inc(v_v_2035_);
if (v_isShared_2053_ == 0)
{
lean_ctor_set_tag(v___x_2052_, 1);
lean_ctor_set(v___x_2052_, 1, v___x_2073_);
lean_ctor_set(v___x_2052_, 0, v_v_2035_);
v___x_2075_ = v___x_2052_;
goto v_reusejp_2074_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v_v_2035_);
lean_ctor_set(v_reuseFailAlloc_2292_, 1, v___x_2073_);
v___x_2075_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2074_;
}
v_reusejp_2074_:
{
lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; uint8_t v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2152_; 
lean_inc_ref_n(v___x_2075_, 7);
lean_inc_n(v_v_2035_, 3);
v___x_2076_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2076_, 0, v_v_2035_);
lean_ctor_set(v___x_2076_, 1, v___x_2075_);
v___x_2077_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2077_, 0, v_v_2035_);
lean_ctor_set(v___x_2077_, 1, v___x_2076_);
v___x_2078_ = l_Lean_Expr_const___override(v___x_2072_, v___x_2077_);
lean_inc_ref_n(v_A_2037_, 12);
v___x_2079_ = l_Lean_Expr_app___override(v___x_2078_, v_A_2037_);
v___x_2080_ = l_Lean_Expr_app___override(v___x_2079_, v_A_2037_);
v___x_2081_ = l_Lean_Expr_app___override(v___x_2080_, v_A_2037_);
v___x_2082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__4));
v___x_2083_ = l_Lean_Expr_const___override(v___x_2082_, v___x_2075_);
v___x_2084_ = l_Lean_Expr_app___override(v___x_2083_, v_A_2037_);
v___x_2085_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__6));
v___x_2086_ = l_Lean_Expr_const___override(v___x_2085_, v___x_2075_);
v___x_2087_ = l_Lean_Expr_app___override(v___x_2086_, v_A_2037_);
v___x_2088_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_2089_ = l_Lean_Expr_const___override(v___x_2088_, v___x_2075_);
v___x_2090_ = l_Lean_Expr_app___override(v___x_2089_, v_A_2037_);
v___x_2091_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_2092_ = l_Lean_Expr_const___override(v___x_2091_, v___x_2075_);
v___x_2093_ = l_Lean_Expr_app___override(v___x_2092_, v_A_2037_);
v___x_2094_ = l_Lean_Expr_app___override(v___x_2093_, v_sA_2039_);
lean_inc_ref_n(v___x_2094_, 2);
v___x_2095_ = l_Lean_Expr_app___override(v___x_2090_, v___x_2094_);
v___x_2096_ = l_Lean_Expr_app___override(v___x_2087_, v___x_2095_);
lean_inc_ref(v___x_2096_);
v___x_2097_ = l_Lean_Expr_app___override(v___x_2084_, v___x_2096_);
v___x_2098_ = l_Lean_Expr_app___override(v___x_2081_, v___x_2097_);
v___x_2099_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_2034_, 3);
v___x_2100_ = l_Lean_Level_succ___override(v_u_2034_);
v___x_2101_ = l_Lean_Level_succ___override(v_v_2035_);
lean_inc_n(v___x_2101_, 2);
lean_inc_n(v___x_2100_, 2);
v___x_2102_ = l_Lean_Level_max___override(v___x_2100_, v___x_2101_);
v___x_2103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2103_, 0, v___x_2101_);
lean_ctor_set(v___x_2103_, 1, v___x_2073_);
lean_inc_ref(v___x_2103_);
v___x_2104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2104_, 0, v___x_2100_);
lean_ctor_set(v___x_2104_, 1, v___x_2103_);
lean_inc_ref(v___x_2104_);
v___x_2105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2105_, 0, v___x_2102_);
lean_ctor_set(v___x_2105_, 1, v___x_2104_);
v___x_2106_ = l_Lean_Expr_const___override(v___x_2099_, v___x_2105_);
v___x_2107_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_2108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2108_, 0, v_u_2034_);
lean_ctor_set(v___x_2108_, 1, v___x_2075_);
lean_inc_ref_n(v___x_2108_, 3);
v___x_2109_ = l_Lean_Expr_const___override(v___x_2107_, v___x_2108_);
lean_inc_ref_n(v_R_2036_, 7);
v___x_2110_ = l_Lean_Expr_app___override(v___x_2109_, v_R_2036_);
v___x_2111_ = l_Lean_Expr_app___override(v___x_2110_, v_A_2037_);
v___x_2112_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_2113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2113_, 0, v_u_2034_);
lean_ctor_set(v___x_2113_, 1, v___x_2073_);
lean_inc_ref_n(v___x_2113_, 2);
v___x_2114_ = l_Lean_Expr_const___override(v___x_2112_, v___x_2113_);
v___x_2115_ = l_Lean_Expr_app___override(v___x_2114_, v_R_2036_);
v___x_2116_ = l_Lean_Expr_const___override(v___x_2091_, v___x_2113_);
v___x_2117_ = l_Lean_Expr_app___override(v___x_2116_, v_R_2036_);
lean_inc_ref(v_sR_2038_);
v___x_2118_ = l_Lean_Expr_app___override(v___x_2117_, v_sR_2038_);
lean_inc_ref(v___x_2118_);
v___x_2119_ = l_Lean_Expr_app___override(v___x_2115_, v___x_2118_);
lean_inc_ref_n(v___x_2119_, 2);
v___x_2120_ = l_Lean_Expr_app___override(v___x_2111_, v___x_2119_);
v___x_2121_ = l_Lean_Expr_const___override(v___x_2112_, v___x_2075_);
v___x_2122_ = l_Lean_Expr_app___override(v___x_2121_, v_A_2037_);
v___x_2123_ = l_Lean_Expr_app___override(v___x_2122_, v___x_2094_);
lean_inc_ref_n(v___x_2123_, 2);
v___x_2124_ = l_Lean_Expr_app___override(v___x_2120_, v___x_2123_);
lean_inc_ref(v___x_2124_);
v___x_2125_ = l_Lean_Expr_app___override(v___x_2106_, v___x_2124_);
v___x_2126_ = l_Lean_Expr_app___override(v___x_2125_, v_R_2036_);
v___x_2127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_2128_ = 0;
v___x_2129_ = l_Lean_Expr_lam___override(v___x_2127_, v_R_2036_, v_A_2037_, v___x_2128_);
lean_inc_ref(v___x_2129_);
v___x_2130_ = l_Lean_Expr_app___override(v___x_2126_, v___x_2129_);
v___x_2131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_2132_ = l_Lean_Expr_const___override(v___x_2131_, v___x_2108_);
v___x_2133_ = l_Lean_Expr_app___override(v___x_2132_, v_R_2036_);
v___x_2134_ = l_Lean_Expr_app___override(v___x_2133_, v_A_2037_);
v___x_2135_ = l_Lean_Expr_app___override(v___x_2134_, v___x_2119_);
v___x_2136_ = l_Lean_Expr_app___override(v___x_2135_, v___x_2123_);
lean_inc_ref(v___x_2136_);
v___x_2137_ = l_Lean_Expr_app___override(v___x_2130_, v___x_2136_);
v___x_2138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_2139_ = l_Lean_Expr_const___override(v___x_2138_, v___x_2108_);
v___x_2140_ = l_Lean_Expr_app___override(v___x_2139_, v_R_2036_);
v___x_2141_ = l_Lean_Expr_app___override(v___x_2140_, v_A_2037_);
v___x_2142_ = l_Lean_Expr_app___override(v___x_2141_, v_sR_2038_);
v___x_2143_ = l_Lean_Expr_app___override(v___x_2142_, v___x_2094_);
v___x_2144_ = l_Lean_Expr_app___override(v___x_2143_, v_sAlg_2040_);
lean_inc_ref(v___x_2144_);
v___x_2145_ = l_Lean_Expr_app___override(v___x_2137_, v___x_2144_);
lean_inc_ref(v_r_2049_);
lean_inc_ref_n(v___x_2145_, 3);
v___x_2146_ = l_Lean_Expr_app___override(v___x_2145_, v_r_2049_);
lean_inc_ref(v___x_2098_);
v___x_2147_ = l_Lean_Expr_app___override(v___x_2098_, v___x_2146_);
lean_inc_ref(v_r_2054_);
v___x_2148_ = l_Lean_Expr_app___override(v___x_2145_, v_r_2054_);
v___x_2149_ = l_Lean_Expr_app___override(v___x_2147_, v___x_2148_);
lean_inc_ref_n(v_expr_2066_, 2);
v___x_2150_ = l_Lean_Expr_app___override(v___x_2145_, v_expr_2066_);
if (v_isShared_2058_ == 0)
{
lean_ctor_set(v___x_2057_, 1, v_val_2067_);
lean_ctor_set(v___x_2057_, 0, v_expr_2066_);
v___x_2152_ = v___x_2057_;
goto v_reusejp_2151_;
}
else
{
lean_object* v_reuseFailAlloc_2291_; 
v_reuseFailAlloc_2291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2291_, 0, v_expr_2066_);
lean_ctor_set(v_reuseFailAlloc_2291_, 1, v_val_2067_);
v___x_2152_ = v_reuseFailAlloc_2291_;
goto v_reusejp_2151_;
}
v_reusejp_2151_:
{
lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2286_; 
v___x_2153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
lean_inc_ref_n(v___x_2103_, 2);
v___x_2154_ = l_Lean_Expr_const___override(v___x_2153_, v___x_2103_);
lean_inc_ref_n(v_A_2037_, 9);
v___x_2155_ = l_Lean_Expr_app___override(v___x_2154_, v_A_2037_);
v___x_2156_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9);
lean_inc_ref(v___x_2149_);
v___x_2157_ = l_Lean_Expr_app___override(v___x_2155_, v___x_2149_);
lean_inc_ref_n(v___x_2150_, 3);
lean_inc_ref_n(v___x_2157_, 2);
v___x_2158_ = l_Lean_Expr_app___override(v___x_2157_, v___x_2150_);
lean_inc_ref(v___x_2158_);
v___x_2159_ = l_Lean_Expr_app___override(v___x_2156_, v___x_2158_);
v___x_2160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11));
v___x_2161_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13);
v___x_2162_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15);
v___x_2163_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16);
v___x_2164_ = l_Lean_Expr_app___override(v___x_2163_, v___x_2158_);
lean_inc(v_v_2035_);
lean_inc_n(v_u_2034_, 3);
v___x_2165_ = l_Lean_Level_max___override(v_u_2034_, v_v_2035_);
lean_inc_n(v___x_2165_, 2);
v___x_2166_ = l_Lean_Level_succ___override(v___x_2165_);
lean_inc_ref(v___x_2104_);
v___x_2167_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2167_, 0, v___x_2166_);
lean_ctor_set(v___x_2167_, 1, v___x_2104_);
v___x_2168_ = l_Lean_Expr_const___override(v___x_2099_, v___x_2167_);
lean_inc_ref_n(v___x_2124_, 3);
v___x_2169_ = l_Lean_Expr_app___override(v___x_2168_, v___x_2124_);
lean_inc_ref_n(v_R_2036_, 13);
v___x_2170_ = l_Lean_Expr_app___override(v___x_2169_, v_R_2036_);
v___x_2171_ = l_Lean_Expr_app___override(v___x_2170_, v___x_2129_);
lean_inc_ref_n(v___x_2136_, 3);
v___x_2172_ = l_Lean_Expr_app___override(v___x_2171_, v___x_2136_);
lean_inc_ref(v___x_2144_);
v___x_2173_ = l_Lean_Expr_app___override(v___x_2172_, v___x_2144_);
lean_inc_ref_n(v_r_2049_, 2);
lean_inc_ref(v___x_2173_);
v___x_2174_ = l_Lean_Expr_app___override(v___x_2173_, v_r_2049_);
v___x_2175_ = l_Lean_Expr_app___override(v___x_2098_, v___x_2174_);
lean_inc_ref_n(v_r_2054_, 2);
v___x_2176_ = l_Lean_Expr_app___override(v___x_2173_, v_r_2054_);
v___x_2177_ = l_Lean_Expr_app___override(v___x_2175_, v___x_2176_);
lean_inc_ref_n(v___x_2177_, 2);
v___x_2178_ = l_Lean_Expr_app___override(v___x_2157_, v___x_2177_);
v___x_2179_ = l_Lean_Expr_app___override(v___x_2164_, v___x_2178_);
v___x_2180_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19);
v___x_2181_ = l_Lean_Expr_app___override(v___x_2179_, v___x_2180_);
v___x_2182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__21));
v___x_2183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___x_2101_);
lean_ctor_set(v___x_2183_, 1, v___x_2161_);
v___x_2184_ = l_Lean_Expr_const___override(v___x_2182_, v___x_2183_);
v___x_2185_ = l_Lean_Expr_app___override(v___x_2184_, v_A_2037_);
v___x_2186_ = l_Lean_Expr_app___override(v___x_2185_, v___x_2162_);
v___x_2187_ = l_Lean_Expr_app___override(v___x_2186_, v___x_2150_);
v___x_2188_ = l_Lean_Expr_app___override(v___x_2187_, v___x_2177_);
v___x_2189_ = l_Lean_Expr_app___override(v___x_2188_, v___x_2157_);
v___x_2190_ = l_Lean_Expr_const___override(v___x_2160_, v___x_2103_);
v___x_2191_ = l_Lean_Expr_app___override(v___x_2190_, v_A_2037_);
v___x_2192_ = l_Lean_Expr_app___override(v___x_2191_, v___x_2150_);
lean_inc_ref_n(v___x_2113_, 4);
v___x_2193_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2193_, 0, v_u_2034_);
lean_ctor_set(v___x_2193_, 1, v___x_2113_);
v___x_2194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2194_, 0, v_u_2034_);
lean_ctor_set(v___x_2194_, 1, v___x_2193_);
v___x_2195_ = l_Lean_Expr_const___override(v___x_2072_, v___x_2194_);
v___x_2196_ = l_Lean_Expr_app___override(v___x_2195_, v_R_2036_);
v___x_2197_ = l_Lean_Expr_app___override(v___x_2196_, v_R_2036_);
v___x_2198_ = l_Lean_Expr_app___override(v___x_2197_, v_R_2036_);
v___x_2199_ = l_Lean_Expr_const___override(v___x_2082_, v___x_2113_);
v___x_2200_ = l_Lean_Expr_app___override(v___x_2199_, v_R_2036_);
v___x_2201_ = l_Lean_Expr_const___override(v___x_2085_, v___x_2113_);
v___x_2202_ = l_Lean_Expr_app___override(v___x_2201_, v_R_2036_);
v___x_2203_ = l_Lean_Expr_const___override(v___x_2088_, v___x_2113_);
v___x_2204_ = l_Lean_Expr_app___override(v___x_2203_, v_R_2036_);
v___x_2205_ = l_Lean_Expr_app___override(v___x_2204_, v___x_2118_);
v___x_2206_ = l_Lean_Expr_app___override(v___x_2202_, v___x_2205_);
lean_inc_ref(v___x_2206_);
v___x_2207_ = l_Lean_Expr_app___override(v___x_2200_, v___x_2206_);
v___x_2208_ = l_Lean_Expr_app___override(v___x_2198_, v___x_2207_);
v___x_2209_ = l_Lean_Expr_app___override(v___x_2208_, v_r_2049_);
v___x_2210_ = l_Lean_Expr_app___override(v___x_2209_, v_r_2054_);
lean_inc_ref_n(v___x_2210_, 2);
lean_inc_ref(v___x_2145_);
v___x_2211_ = l_Lean_Expr_app___override(v___x_2145_, v___x_2210_);
v___x_2212_ = l_Lean_Expr_app___override(v___x_2192_, v___x_2211_);
v___x_2213_ = l_Lean_Expr_app___override(v___x_2212_, v___x_2177_);
v___x_2214_ = l_Lean_Expr_const___override(v___x_2182_, v___x_2104_);
v___x_2215_ = l_Lean_Expr_app___override(v___x_2214_, v_R_2036_);
v___x_2216_ = l_Lean_Expr_app___override(v___x_2215_, v_A_2037_);
lean_inc_ref(v_expr_2066_);
v___x_2217_ = l_Lean_Expr_app___override(v___x_2216_, v_expr_2066_);
v___x_2218_ = l_Lean_Expr_app___override(v___x_2217_, v___x_2210_);
v___x_2219_ = l_Lean_Expr_app___override(v___x_2218_, v___x_2145_);
v___x_2220_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23));
v___x_2221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2221_, 0, v___x_2100_);
lean_ctor_set(v___x_2221_, 1, v___x_2073_);
v___x_2222_ = l_Lean_Expr_const___override(v___x_2220_, v___x_2221_);
v___x_2223_ = l_Lean_Expr_app___override(v___x_2222_, v_R_2036_);
v___x_2224_ = l_Lean_Expr_app___override(v___x_2223_, v___x_2210_);
v___x_2225_ = l_Lean_Expr_app___override(v___x_2224_, v_expr_2066_);
v___x_2226_ = l_Lean_Expr_app___override(v___x_2225_, v_proof_2068_);
v___x_2227_ = l_Lean_Expr_app___override(v___x_2219_, v___x_2226_);
v___x_2228_ = l_Lean_Expr_app___override(v___x_2213_, v___x_2227_);
v___x_2229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__25));
v___x_2230_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2230_, 0, v___x_2165_);
lean_ctor_set(v___x_2230_, 1, v___x_2073_);
v___x_2231_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2231_, 0, v_v_2035_);
lean_ctor_set(v___x_2231_, 1, v___x_2230_);
v___x_2232_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2232_, 0, v_u_2034_);
lean_ctor_set(v___x_2232_, 1, v___x_2231_);
v___x_2233_ = l_Lean_Expr_const___override(v___x_2229_, v___x_2232_);
v___x_2234_ = l_Lean_Expr_app___override(v___x_2233_, v_R_2036_);
v___x_2235_ = l_Lean_Expr_app___override(v___x_2234_, v_A_2037_);
v___x_2236_ = l_Lean_Expr_app___override(v___x_2235_, v___x_2124_);
v___x_2237_ = l_Lean_Expr_app___override(v___x_2236_, v___x_2206_);
v___x_2238_ = l_Lean_Expr_app___override(v___x_2237_, v___x_2096_);
v___x_2239_ = l_Lean_Expr_app___override(v___x_2238_, v___x_2136_);
v___x_2240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__28));
lean_inc_ref(v___x_2108_);
v___x_2241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2165_);
lean_ctor_set(v___x_2241_, 1, v___x_2108_);
lean_inc_ref(v___x_2241_);
v___x_2242_ = l_Lean_Expr_const___override(v___x_2240_, v___x_2241_);
v___x_2243_ = l_Lean_Expr_app___override(v___x_2242_, v___x_2124_);
v___x_2244_ = l_Lean_Expr_app___override(v___x_2243_, v_R_2036_);
v___x_2245_ = l_Lean_Expr_app___override(v___x_2244_, v_A_2037_);
v___x_2246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__31));
v___x_2247_ = l_Lean_Expr_const___override(v___x_2246_, v___x_2113_);
v___x_2248_ = l_Lean_Expr_app___override(v___x_2247_, v_R_2036_);
lean_inc_ref_n(v___x_2119_, 2);
v___x_2249_ = l_Lean_Expr_app___override(v___x_2248_, v___x_2119_);
v___x_2250_ = l_Lean_Expr_app___override(v___x_2245_, v___x_2249_);
v___x_2251_ = l_Lean_Expr_const___override(v___x_2246_, v___x_2075_);
v___x_2252_ = l_Lean_Expr_app___override(v___x_2251_, v_A_2037_);
lean_inc_ref_n(v___x_2123_, 2);
v___x_2253_ = l_Lean_Expr_app___override(v___x_2252_, v___x_2123_);
v___x_2254_ = l_Lean_Expr_app___override(v___x_2250_, v___x_2253_);
v___x_2255_ = l_Lean_Expr_app___override(v___x_2254_, v___x_2136_);
v___x_2256_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__34));
v___x_2257_ = l_Lean_Expr_const___override(v___x_2256_, v___x_2241_);
v___x_2258_ = l_Lean_Expr_app___override(v___x_2257_, v___x_2124_);
v___x_2259_ = l_Lean_Expr_app___override(v___x_2258_, v_R_2036_);
v___x_2260_ = l_Lean_Expr_app___override(v___x_2259_, v_A_2037_);
v___x_2261_ = l_Lean_Expr_app___override(v___x_2260_, v___x_2136_);
v___x_2262_ = l_Lean_Expr_app___override(v___x_2261_, v___x_2119_);
v___x_2263_ = l_Lean_Expr_app___override(v___x_2262_, v___x_2123_);
v___x_2264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36));
v___x_2265_ = l_Lean_Expr_const___override(v___x_2264_, v___x_2108_);
v___x_2266_ = l_Lean_Expr_app___override(v___x_2265_, v_R_2036_);
v___x_2267_ = l_Lean_Expr_app___override(v___x_2266_, v_A_2037_);
v___x_2268_ = l_Lean_Expr_app___override(v___x_2267_, v___x_2119_);
v___x_2269_ = l_Lean_Expr_app___override(v___x_2268_, v___x_2123_);
v___x_2270_ = l_Lean_Expr_app___override(v___x_2263_, v___x_2269_);
v___x_2271_ = l_Lean_Expr_app___override(v___x_2255_, v___x_2270_);
v___x_2272_ = l_Lean_Expr_app___override(v___x_2239_, v___x_2271_);
v___x_2273_ = l_Lean_Expr_app___override(v___x_2272_, v___x_2144_);
v___x_2274_ = l_Lean_Expr_app___override(v___x_2273_, v_r_2049_);
v___x_2275_ = l_Lean_Expr_app___override(v___x_2274_, v_r_2054_);
v___x_2276_ = l_Lean_Expr_app___override(v___x_2228_, v___x_2275_);
v___x_2277_ = l_Lean_Expr_app___override(v___x_2189_, v___x_2276_);
v___x_2278_ = l_Lean_Expr_app___override(v___x_2181_, v___x_2277_);
v___x_2279_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__38));
v___x_2280_ = l_Lean_Expr_const___override(v___x_2279_, v___x_2103_);
v___x_2281_ = l_Lean_Expr_app___override(v___x_2280_, v_A_2037_);
v___x_2282_ = l_Lean_Expr_app___override(v___x_2281_, v___x_2149_);
v___x_2283_ = l_Lean_Expr_app___override(v___x_2278_, v___x_2282_);
v___x_2284_ = l_Lean_Expr_app___override(v___x_2159_, v___x_2283_);
if (v_isShared_2071_ == 0)
{
lean_ctor_set(v___x_2070_, 2, v___x_2284_);
lean_ctor_set(v___x_2070_, 1, v___x_2152_);
lean_ctor_set(v___x_2070_, 0, v___x_2150_);
v___x_2286_ = v___x_2070_;
goto v_reusejp_2285_;
}
else
{
lean_object* v_reuseFailAlloc_2290_; 
v_reuseFailAlloc_2290_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2290_, 0, v___x_2150_);
lean_ctor_set(v_reuseFailAlloc_2290_, 1, v___x_2152_);
lean_ctor_set(v_reuseFailAlloc_2290_, 2, v___x_2284_);
v___x_2286_ = v_reuseFailAlloc_2290_;
goto v_reusejp_2285_;
}
v_reusejp_2285_:
{
lean_object* v___x_2288_; 
if (v_isShared_2065_ == 0)
{
lean_ctor_set(v___x_2064_, 0, v___x_2286_);
v___x_2288_ = v___x_2064_;
goto v_reusejp_2287_;
}
else
{
lean_object* v_reuseFailAlloc_2289_; 
v_reuseFailAlloc_2289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2289_, 0, v___x_2286_);
v___x_2288_ = v_reuseFailAlloc_2289_;
goto v_reusejp_2287_;
}
v_reusejp_2287_:
{
return v___x_2288_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2295_; lean_object* v___x_2297_; uint8_t v_isShared_2298_; uint8_t v_isSharedCheck_2302_; 
lean_del_object(v___x_2057_);
lean_dec_ref(v_r_2054_);
lean_del_object(v___x_2052_);
lean_dec_ref(v_r_2049_);
lean_dec_ref(v_sAlg_2040_);
lean_dec_ref(v_sA_2039_);
lean_dec_ref(v_sR_2038_);
lean_dec_ref(v_A_2037_);
lean_dec_ref(v_R_2036_);
lean_dec(v_v_2035_);
lean_dec(v_u_2034_);
v_a_2295_ = lean_ctor_get(v___x_2061_, 0);
v_isSharedCheck_2302_ = !lean_is_exclusive(v___x_2061_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2297_ = v___x_2061_;
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
else
{
lean_inc(v_a_2295_);
lean_dec(v___x_2061_);
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___boxed(lean_object* v_u_2305_, lean_object* v_v_2306_, lean_object* v_R_2307_, lean_object* v_A_2308_, lean_object* v_sR_2309_, lean_object* v_sA_2310_, lean_object* v_sAlg_2311_, lean_object* v_cR_2312_, lean_object* v_za_2313_, lean_object* v_zb_2314_, lean_object* v_a_2315_, lean_object* v_a_2316_, lean_object* v_a_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_){
_start:
{
lean_object* v_res_2320_; 
v_res_2320_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg(v_u_2305_, v_v_2306_, v_R_2307_, v_A_2308_, v_sR_2309_, v_sA_2310_, v_sAlg_2311_, v_cR_2312_, v_za_2313_, v_zb_2314_, v_a_2315_, v_a_2316_, v_a_2317_, v_a_2318_);
lean_dec(v_a_2318_);
lean_dec_ref(v_a_2317_);
lean_dec(v_a_2316_);
lean_dec_ref(v_a_2315_);
return v_res_2320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul(lean_object* v_u_2321_, lean_object* v_v_2322_, lean_object* v_R_2323_, lean_object* v_A_2324_, lean_object* v_sR_2325_, lean_object* v_sA_2326_, lean_object* v_sAlg_2327_, lean_object* v_cR_2328_, lean_object* v_a_2329_, lean_object* v_b_2330_, lean_object* v_za_2331_, lean_object* v_zb_2332_, lean_object* v_a_2333_, lean_object* v_a_2334_, lean_object* v_a_2335_, lean_object* v_a_2336_){
_start:
{
lean_object* v___x_2338_; 
v___x_2338_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg(v_u_2321_, v_v_2322_, v_R_2323_, v_A_2324_, v_sR_2325_, v_sA_2326_, v_sAlg_2327_, v_cR_2328_, v_za_2331_, v_zb_2332_, v_a_2333_, v_a_2334_, v_a_2335_, v_a_2336_);
return v___x_2338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___boxed(lean_object** _args){
lean_object* v_u_2339_ = _args[0];
lean_object* v_v_2340_ = _args[1];
lean_object* v_R_2341_ = _args[2];
lean_object* v_A_2342_ = _args[3];
lean_object* v_sR_2343_ = _args[4];
lean_object* v_sA_2344_ = _args[5];
lean_object* v_sAlg_2345_ = _args[6];
lean_object* v_cR_2346_ = _args[7];
lean_object* v_a_2347_ = _args[8];
lean_object* v_b_2348_ = _args[9];
lean_object* v_za_2349_ = _args[10];
lean_object* v_zb_2350_ = _args[11];
lean_object* v_a_2351_ = _args[12];
lean_object* v_a_2352_ = _args[13];
lean_object* v_a_2353_ = _args[14];
lean_object* v_a_2354_ = _args[15];
lean_object* v_a_2355_ = _args[16];
_start:
{
lean_object* v_res_2356_; 
v_res_2356_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul(v_u_2339_, v_v_2340_, v_R_2341_, v_A_2342_, v_sR_2343_, v_sA_2344_, v_sAlg_2345_, v_cR_2346_, v_a_2347_, v_b_2348_, v_za_2349_, v_zb_2350_, v_a_2351_, v_a_2352_, v_a_2353_, v_a_2354_);
lean_dec(v_a_2354_);
lean_dec_ref(v_a_2353_);
lean_dec(v_a_2352_);
lean_dec_ref(v_a_2351_);
lean_dec_ref(v_b_2348_);
lean_dec_ref(v_a_2347_);
return v_res_2356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg(lean_object* v_u_2369_, lean_object* v_v_2370_, lean_object* v_R_2371_, lean_object* v_A_2372_, lean_object* v_sR_2373_, lean_object* v_sA_2374_, lean_object* v_sAlg_2375_, lean_object* v_cR_2376_, lean_object* v_u_x27_2377_, lean_object* v_R_x27_2378_, lean_object* v___smul_2379_, lean_object* v_r_x27_2380_, lean_object* v_a_2381_, lean_object* v_a_2382_, lean_object* v_a_2383_, lean_object* v_a_2384_, lean_object* v_a_2385_, lean_object* v_a_2386_){
_start:
{
lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; 
lean_inc_n(v_u_2369_, 4);
v___x_2388_ = l_Lean_Level_succ___override(v_u_2369_);
lean_inc_n(v_v_2370_, 3);
v___x_2389_ = l_Lean_Level_succ___override(v_v_2370_);
v___x_2390_ = lean_box(0);
v___x_2391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2391_, 0, v_u_2369_);
lean_ctor_set(v___x_2391_, 1, v___x_2390_);
v___x_2392_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2392_, 0, v_v_2370_);
lean_ctor_set(v___x_2392_, 1, v___x_2390_);
lean_inc_ref_n(v___x_2392_, 3);
v___x_2393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2393_, 0, v_u_2369_);
lean_ctor_set(v___x_2393_, 1, v___x_2392_);
v___x_2394_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_2395_ = l_Lean_Expr_const___override(v___x_2394_, v___x_2392_);
lean_inc_n(v_u_x27_2377_, 2);
v___x_2396_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2396_, 0, v_u_x27_2377_);
lean_ctor_set(v___x_2396_, 1, v___x_2392_);
lean_inc_ref(v_r_x27_2380_);
lean_inc_ref(v___smul_2379_);
lean_inc_ref(v_sAlg_2375_);
lean_inc_ref(v_sA_2374_);
lean_inc_ref(v_sR_2373_);
lean_inc_ref(v_A_2372_);
lean_inc_ref(v_R_x27_2378_);
lean_inc_ref(v_R_2371_);
v___x_2397_ = lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast(v_u_2369_, v_u_x27_2377_, v_v_2370_, v_R_2371_, v_R_x27_2378_, v_A_2372_, v_sR_2373_, v_sA_2374_, v_sAlg_2375_, v___smul_2379_, v_r_x27_2380_, v_a_2383_, v_a_2384_, v_a_2385_, v_a_2386_);
if (lean_obj_tag(v___x_2397_) == 0)
{
lean_object* v_a_2398_; lean_object* v_fst_2399_; lean_object* v_snd_2400_; lean_object* v___x_2402_; uint8_t v_isShared_2403_; uint8_t v_isSharedCheck_2602_; 
v_a_2398_ = lean_ctor_get(v___x_2397_, 0);
lean_inc(v_a_2398_);
lean_dec_ref_known(v___x_2397_, 1);
v_fst_2399_ = lean_ctor_get(v_a_2398_, 0);
v_snd_2400_ = lean_ctor_get(v_a_2398_, 1);
v_isSharedCheck_2602_ = !lean_is_exclusive(v_a_2398_);
if (v_isSharedCheck_2602_ == 0)
{
v___x_2402_ = v_a_2398_;
v_isShared_2403_ = v_isSharedCheck_2602_;
goto v_resetjp_2401_;
}
else
{
lean_inc(v_snd_2400_);
lean_inc(v_fst_2399_);
lean_dec(v_a_2398_);
v___x_2402_ = lean_box(0);
v_isShared_2403_ = v_isSharedCheck_2602_;
goto v_resetjp_2401_;
}
v_resetjp_2401_:
{
lean_object* v_toCache_2404_; lean_object* v___x_2406_; uint8_t v_isShared_2407_; uint8_t v_isSharedCheck_2600_; 
v_toCache_2404_ = lean_ctor_get(v_cR_2376_, 0);
v_isSharedCheck_2600_ = !lean_is_exclusive(v_cR_2376_);
if (v_isSharedCheck_2600_ == 0)
{
lean_object* v_unused_2601_; 
v_unused_2601_ = lean_ctor_get(v_cR_2376_, 1);
lean_dec(v_unused_2601_);
v___x_2406_ = v_cR_2376_;
v_isShared_2407_ = v_isSharedCheck_2600_;
goto v_resetjp_2405_;
}
else
{
lean_inc(v_toCache_2404_);
lean_dec(v_cR_2376_);
v___x_2406_ = lean_box(0);
v_isShared_2407_ = v_isSharedCheck_2600_;
goto v_resetjp_2405_;
}
v_resetjp_2405_:
{
lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; 
v___x_2408_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_toCache_2404_);
lean_inc_ref_n(v_sR_2373_, 2);
lean_inc_ref_n(v_R_2371_, 2);
lean_inc_n(v_u_2369_, 2);
v___x_2409_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_2369_, v_R_2371_, v_sR_2373_, v_toCache_2404_);
lean_inc(v_fst_2399_);
v___x_2410_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_2408_, v_u_2369_, v_R_2371_, v_sR_2373_, v___x_2409_, v_toCache_2404_, v_fst_2399_, v_a_2381_, v_a_2382_, v_a_2383_, v_a_2384_, v_a_2385_, v_a_2386_);
if (lean_obj_tag(v___x_2410_) == 0)
{
lean_object* v_a_2411_; lean_object* v___x_2413_; uint8_t v_isShared_2414_; uint8_t v_isSharedCheck_2591_; 
v_a_2411_ = lean_ctor_get(v___x_2410_, 0);
v_isSharedCheck_2591_ = !lean_is_exclusive(v___x_2410_);
if (v_isSharedCheck_2591_ == 0)
{
v___x_2413_ = v___x_2410_;
v_isShared_2414_ = v_isSharedCheck_2591_;
goto v_resetjp_2412_;
}
else
{
lean_inc(v_a_2411_);
lean_dec(v___x_2410_);
v___x_2413_ = lean_box(0);
v_isShared_2414_ = v_isSharedCheck_2591_;
goto v_resetjp_2412_;
}
v_resetjp_2412_:
{
lean_object* v_val_2415_; 
v_val_2415_ = lean_ctor_get(v_a_2411_, 1);
if (lean_obj_tag(v_val_2415_) == 0)
{
lean_object* v_proof_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2445_; 
lean_dec_ref_known(v___x_2393_, 2);
lean_dec_ref_known(v___x_2391_, 2);
lean_dec(v___x_2389_);
lean_dec(v___x_2388_);
v_proof_2416_ = lean_ctor_get(v_a_2411_, 2);
lean_inc_ref(v_proof_2416_);
lean_dec(v_a_2411_);
v___x_2417_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
lean_inc_ref_n(v___x_2392_, 3);
v___x_2418_ = l_Lean_Expr_const___override(v___x_2417_, v___x_2392_);
lean_inc_ref_n(v_A_2372_, 6);
v___x_2419_ = l_Lean_Expr_app___override(v___x_2418_, v_A_2372_);
v___x_2420_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_2421_ = l_Lean_Expr_app___override(v___x_2419_, v___x_2420_);
v___x_2422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_2423_ = l_Lean_Expr_const___override(v___x_2422_, v___x_2392_);
v___x_2424_ = l_Lean_Expr_app___override(v___x_2423_, v_A_2372_);
v___x_2425_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_2426_ = l_Lean_Expr_const___override(v___x_2425_, v___x_2392_);
v___x_2427_ = l_Lean_Expr_app___override(v___x_2426_, v_A_2372_);
v___x_2428_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_2429_ = l_Lean_Expr_const___override(v___x_2428_, v___x_2392_);
v___x_2430_ = l_Lean_Expr_app___override(v___x_2429_, v_A_2372_);
v___x_2431_ = l_Lean_Expr_app___override(v___x_2395_, v_A_2372_);
lean_inc_ref(v_sA_2374_);
v___x_2432_ = l_Lean_Expr_app___override(v___x_2431_, v_sA_2374_);
v___x_2433_ = l_Lean_Expr_app___override(v___x_2430_, v___x_2432_);
v___x_2434_ = l_Lean_Expr_app___override(v___x_2427_, v___x_2433_);
v___x_2435_ = l_Lean_Expr_app___override(v___x_2424_, v___x_2434_);
v___x_2436_ = l_Lean_Expr_app___override(v___x_2421_, v___x_2435_);
v___x_2437_ = lean_box(0);
v___x_2438_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18));
v___x_2439_ = l_Lean_Expr_const___override(v___x_2438_, v___x_2396_);
lean_inc_ref(v_R_x27_2378_);
v___x_2440_ = l_Lean_Expr_app___override(v___x_2439_, v_R_x27_2378_);
v___x_2441_ = l_Lean_Expr_app___override(v___x_2440_, v_A_2372_);
v___x_2442_ = l_Lean_Expr_app___override(v___x_2441_, v___smul_2379_);
v___x_2443_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__1));
if (v_isShared_2407_ == 0)
{
lean_ctor_set_tag(v___x_2406_, 1);
lean_ctor_set(v___x_2406_, 1, v___x_2390_);
lean_ctor_set(v___x_2406_, 0, v_u_x27_2377_);
v___x_2445_ = v___x_2406_;
goto v_reusejp_2444_;
}
else
{
lean_object* v_reuseFailAlloc_2467_; 
v_reuseFailAlloc_2467_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2467_, 0, v_u_x27_2377_);
lean_ctor_set(v_reuseFailAlloc_2467_, 1, v___x_2390_);
v___x_2445_ = v_reuseFailAlloc_2467_;
goto v_reusejp_2444_;
}
v_reusejp_2444_:
{
lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2462_; 
v___x_2446_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2446_, 0, v_v_2370_);
lean_ctor_set(v___x_2446_, 1, v___x_2445_);
v___x_2447_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2447_, 0, v_u_2369_);
lean_ctor_set(v___x_2447_, 1, v___x_2446_);
v___x_2448_ = l_Lean_Expr_const___override(v___x_2443_, v___x_2447_);
v___x_2449_ = l_Lean_Expr_app___override(v___x_2448_, v_R_2371_);
v___x_2450_ = l_Lean_Expr_app___override(v___x_2449_, v_A_2372_);
v___x_2451_ = l_Lean_Expr_app___override(v___x_2450_, v_sR_2373_);
v___x_2452_ = l_Lean_Expr_app___override(v___x_2451_, v_sA_2374_);
v___x_2453_ = l_Lean_Expr_app___override(v___x_2452_, v_sAlg_2375_);
v___x_2454_ = l_Lean_Expr_app___override(v___x_2453_, v_R_x27_2378_);
v___x_2455_ = l_Lean_Expr_app___override(v___x_2454_, v___x_2442_);
v___x_2456_ = l_Lean_Expr_app___override(v___x_2455_, v_r_x27_2380_);
v___x_2457_ = l_Lean_Expr_app___override(v___x_2456_, v_fst_2399_);
v___x_2458_ = l_Lean_Expr_app___override(v___x_2457_, v_proof_2416_);
v___x_2459_ = l_Lean_Expr_app___override(v___x_2458_, v_snd_2400_);
v___x_2460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2460_, 0, v___x_2437_);
lean_ctor_set(v___x_2460_, 1, v___x_2459_);
if (v_isShared_2403_ == 0)
{
lean_ctor_set(v___x_2402_, 1, v___x_2460_);
lean_ctor_set(v___x_2402_, 0, v___x_2436_);
v___x_2462_ = v___x_2402_;
goto v_reusejp_2461_;
}
else
{
lean_object* v_reuseFailAlloc_2466_; 
v_reuseFailAlloc_2466_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2466_, 0, v___x_2436_);
lean_ctor_set(v_reuseFailAlloc_2466_, 1, v___x_2460_);
v___x_2462_ = v_reuseFailAlloc_2466_;
goto v_reusejp_2461_;
}
v_reusejp_2461_:
{
lean_object* v___x_2464_; 
if (v_isShared_2414_ == 0)
{
lean_ctor_set(v___x_2413_, 0, v___x_2462_);
v___x_2464_ = v___x_2413_;
goto v_reusejp_2463_;
}
else
{
lean_object* v_reuseFailAlloc_2465_; 
v_reuseFailAlloc_2465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2465_, 0, v___x_2462_);
v___x_2464_ = v_reuseFailAlloc_2465_;
goto v_reusejp_2463_;
}
v_reusejp_2463_:
{
return v___x_2464_;
}
}
}
}
else
{
lean_object* v_expr_2468_; lean_object* v_proof_2469_; lean_object* v___x_2470_; lean_object* v___x_2472_; 
lean_inc(v_val_2415_);
v_expr_2468_ = lean_ctor_get(v_a_2411_, 0);
lean_inc_ref(v_expr_2468_);
v_proof_2469_ = lean_ctor_get(v_a_2411_, 2);
lean_inc_ref(v_proof_2469_);
lean_dec(v_a_2411_);
v___x_2470_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
lean_inc_ref(v___x_2392_);
lean_inc(v_v_2370_);
if (v_isShared_2407_ == 0)
{
lean_ctor_set_tag(v___x_2406_, 1);
lean_ctor_set(v___x_2406_, 1, v___x_2392_);
lean_ctor_set(v___x_2406_, 0, v_v_2370_);
v___x_2472_ = v___x_2406_;
goto v_reusejp_2471_;
}
else
{
lean_object* v_reuseFailAlloc_2590_; 
v_reuseFailAlloc_2590_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2590_, 0, v_v_2370_);
lean_ctor_set(v_reuseFailAlloc_2590_, 1, v___x_2392_);
v___x_2472_ = v_reuseFailAlloc_2590_;
goto v_reusejp_2471_;
}
v_reusejp_2471_:
{
lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___x_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; uint8_t v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; lean_object* v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; lean_object* v___x_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2585_; 
lean_inc(v_v_2370_);
v___x_2473_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2473_, 0, v_v_2370_);
lean_ctor_set(v___x_2473_, 1, v___x_2472_);
v___x_2474_ = l_Lean_Expr_const___override(v___x_2470_, v___x_2473_);
lean_inc_ref_n(v_A_2372_, 17);
v___x_2475_ = l_Lean_Expr_app___override(v___x_2474_, v_A_2372_);
v___x_2476_ = l_Lean_Expr_app___override(v___x_2475_, v_A_2372_);
v___x_2477_ = l_Lean_Expr_app___override(v___x_2476_, v_A_2372_);
v___x_2478_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
lean_inc_ref_n(v___x_2392_, 7);
v___x_2479_ = l_Lean_Expr_const___override(v___x_2478_, v___x_2392_);
v___x_2480_ = l_Lean_Expr_app___override(v___x_2479_, v_A_2372_);
v___x_2481_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_2482_ = l_Lean_Expr_const___override(v___x_2481_, v___x_2392_);
v___x_2483_ = l_Lean_Expr_app___override(v___x_2482_, v_A_2372_);
v___x_2484_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_2485_ = l_Lean_Expr_const___override(v___x_2484_, v___x_2392_);
v___x_2486_ = l_Lean_Expr_app___override(v___x_2485_, v_A_2372_);
v___x_2487_ = l_Lean_Expr_app___override(v___x_2395_, v_A_2372_);
lean_inc_ref(v_sA_2374_);
v___x_2488_ = l_Lean_Expr_app___override(v___x_2487_, v_sA_2374_);
lean_inc_ref_n(v___x_2488_, 3);
v___x_2489_ = l_Lean_Expr_app___override(v___x_2486_, v___x_2488_);
v___x_2490_ = l_Lean_Expr_app___override(v___x_2483_, v___x_2489_);
v___x_2491_ = l_Lean_Expr_app___override(v___x_2480_, v___x_2490_);
v___x_2492_ = l_Lean_Expr_app___override(v___x_2477_, v___x_2491_);
v___x_2493_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc(v___x_2389_);
lean_inc(v___x_2388_);
v___x_2494_ = l_Lean_Level_max___override(v___x_2388_, v___x_2389_);
v___x_2495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2495_, 0, v___x_2389_);
lean_ctor_set(v___x_2495_, 1, v___x_2390_);
v___x_2496_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2388_);
lean_ctor_set(v___x_2496_, 1, v___x_2495_);
v___x_2497_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2497_, 0, v___x_2494_);
lean_ctor_set(v___x_2497_, 1, v___x_2496_);
v___x_2498_ = l_Lean_Expr_const___override(v___x_2493_, v___x_2497_);
v___x_2499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_ref_n(v___x_2393_, 2);
v___x_2500_ = l_Lean_Expr_const___override(v___x_2499_, v___x_2393_);
lean_inc_ref_n(v_R_2371_, 7);
v___x_2501_ = l_Lean_Expr_app___override(v___x_2500_, v_R_2371_);
v___x_2502_ = l_Lean_Expr_app___override(v___x_2501_, v_A_2372_);
v___x_2503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref(v___x_2391_);
v___x_2504_ = l_Lean_Expr_const___override(v___x_2503_, v___x_2391_);
v___x_2505_ = l_Lean_Expr_app___override(v___x_2504_, v_R_2371_);
v___x_2506_ = l_Lean_Expr_const___override(v___x_2394_, v___x_2391_);
v___x_2507_ = l_Lean_Expr_app___override(v___x_2506_, v_R_2371_);
lean_inc_ref_n(v_sR_2373_, 2);
v___x_2508_ = l_Lean_Expr_app___override(v___x_2507_, v_sR_2373_);
v___x_2509_ = l_Lean_Expr_app___override(v___x_2505_, v___x_2508_);
lean_inc_ref(v___x_2509_);
v___x_2510_ = l_Lean_Expr_app___override(v___x_2502_, v___x_2509_);
v___x_2511_ = l_Lean_Expr_const___override(v___x_2503_, v___x_2392_);
v___x_2512_ = l_Lean_Expr_app___override(v___x_2511_, v_A_2372_);
v___x_2513_ = l_Lean_Expr_app___override(v___x_2512_, v___x_2488_);
lean_inc_ref(v___x_2513_);
v___x_2514_ = l_Lean_Expr_app___override(v___x_2510_, v___x_2513_);
v___x_2515_ = l_Lean_Expr_app___override(v___x_2498_, v___x_2514_);
v___x_2516_ = l_Lean_Expr_app___override(v___x_2515_, v_R_2371_);
v___x_2517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_2518_ = 0;
v___x_2519_ = l_Lean_Expr_lam___override(v___x_2517_, v_R_2371_, v_A_2372_, v___x_2518_);
v___x_2520_ = l_Lean_Expr_app___override(v___x_2516_, v___x_2519_);
v___x_2521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_2522_ = l_Lean_Expr_const___override(v___x_2521_, v___x_2393_);
v___x_2523_ = l_Lean_Expr_app___override(v___x_2522_, v_R_2371_);
v___x_2524_ = l_Lean_Expr_app___override(v___x_2523_, v_A_2372_);
v___x_2525_ = l_Lean_Expr_app___override(v___x_2524_, v___x_2509_);
v___x_2526_ = l_Lean_Expr_app___override(v___x_2525_, v___x_2513_);
v___x_2527_ = l_Lean_Expr_app___override(v___x_2520_, v___x_2526_);
v___x_2528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_2529_ = l_Lean_Expr_const___override(v___x_2528_, v___x_2393_);
v___x_2530_ = l_Lean_Expr_app___override(v___x_2529_, v_R_2371_);
v___x_2531_ = l_Lean_Expr_app___override(v___x_2530_, v_A_2372_);
v___x_2532_ = l_Lean_Expr_app___override(v___x_2531_, v_sR_2373_);
v___x_2533_ = l_Lean_Expr_app___override(v___x_2532_, v___x_2488_);
lean_inc_ref(v_sAlg_2375_);
v___x_2534_ = l_Lean_Expr_app___override(v___x_2533_, v_sAlg_2375_);
v___x_2535_ = l_Lean_Expr_app___override(v___x_2527_, v___x_2534_);
lean_inc_ref_n(v_expr_2468_, 2);
v___x_2536_ = l_Lean_Expr_app___override(v___x_2535_, v_expr_2468_);
lean_inc_ref_n(v___x_2536_, 2);
v___x_2537_ = l_Lean_Expr_app___override(v___x_2492_, v___x_2536_);
v___x_2538_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_2539_ = l_Lean_Expr_const___override(v___x_2538_, v___x_2392_);
v___x_2540_ = l_Lean_Expr_app___override(v___x_2539_, v_A_2372_);
v___x_2541_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
v___x_2542_ = l_Lean_Expr_app___override(v___x_2540_, v___x_2541_);
v___x_2543_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_2544_ = l_Lean_Expr_const___override(v___x_2543_, v___x_2392_);
v___x_2545_ = l_Lean_Expr_app___override(v___x_2544_, v_A_2372_);
v___x_2546_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_2547_ = l_Lean_Expr_const___override(v___x_2546_, v___x_2392_);
v___x_2548_ = l_Lean_Expr_app___override(v___x_2547_, v_A_2372_);
v___x_2549_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_2550_ = l_Lean_Expr_const___override(v___x_2549_, v___x_2392_);
v___x_2551_ = l_Lean_Expr_app___override(v___x_2550_, v_A_2372_);
v___x_2552_ = l_Lean_Expr_app___override(v___x_2551_, v___x_2488_);
v___x_2553_ = l_Lean_Expr_app___override(v___x_2548_, v___x_2552_);
v___x_2554_ = l_Lean_Expr_app___override(v___x_2545_, v___x_2553_);
v___x_2555_ = l_Lean_Expr_app___override(v___x_2542_, v___x_2554_);
lean_inc_ref(v___x_2555_);
v___x_2556_ = l_Lean_Expr_app___override(v___x_2537_, v___x_2555_);
v___x_2557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2557_, 0, v_expr_2468_);
lean_ctor_set(v___x_2557_, 1, v_val_2415_);
v___x_2558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2558_, 0, v___x_2536_);
lean_ctor_set(v___x_2558_, 1, v___x_2557_);
v___x_2559_ = lean_box(0);
v___x_2560_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_2560_, 0, v___x_2536_);
lean_ctor_set(v___x_2560_, 1, v___x_2555_);
lean_ctor_set(v___x_2560_, 2, v___x_2558_);
lean_ctor_set(v___x_2560_, 3, v___x_2559_);
v___x_2561_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__18));
v___x_2562_ = l_Lean_Expr_const___override(v___x_2561_, v___x_2396_);
lean_inc_ref(v_R_x27_2378_);
v___x_2563_ = l_Lean_Expr_app___override(v___x_2562_, v_R_x27_2378_);
v___x_2564_ = l_Lean_Expr_app___override(v___x_2563_, v_A_2372_);
v___x_2565_ = l_Lean_Expr_app___override(v___x_2564_, v___smul_2379_);
v___x_2566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___closed__3));
v___x_2567_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2567_, 0, v_u_x27_2377_);
lean_ctor_set(v___x_2567_, 1, v___x_2390_);
v___x_2568_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2568_, 0, v_v_2370_);
lean_ctor_set(v___x_2568_, 1, v___x_2567_);
v___x_2569_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2569_, 0, v_u_2369_);
lean_ctor_set(v___x_2569_, 1, v___x_2568_);
v___x_2570_ = l_Lean_Expr_const___override(v___x_2566_, v___x_2569_);
v___x_2571_ = l_Lean_Expr_app___override(v___x_2570_, v_R_2371_);
v___x_2572_ = l_Lean_Expr_app___override(v___x_2571_, v_A_2372_);
v___x_2573_ = l_Lean_Expr_app___override(v___x_2572_, v_sR_2373_);
v___x_2574_ = l_Lean_Expr_app___override(v___x_2573_, v_sA_2374_);
v___x_2575_ = l_Lean_Expr_app___override(v___x_2574_, v_sAlg_2375_);
v___x_2576_ = l_Lean_Expr_app___override(v___x_2575_, v_R_x27_2378_);
v___x_2577_ = l_Lean_Expr_app___override(v___x_2576_, v___x_2565_);
v___x_2578_ = l_Lean_Expr_app___override(v___x_2577_, v_r_x27_2380_);
v___x_2579_ = l_Lean_Expr_app___override(v___x_2578_, v_fst_2399_);
v___x_2580_ = l_Lean_Expr_app___override(v___x_2579_, v_expr_2468_);
v___x_2581_ = l_Lean_Expr_app___override(v___x_2580_, v_proof_2469_);
v___x_2582_ = l_Lean_Expr_app___override(v___x_2581_, v_snd_2400_);
v___x_2583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2583_, 0, v___x_2560_);
lean_ctor_set(v___x_2583_, 1, v___x_2582_);
if (v_isShared_2403_ == 0)
{
lean_ctor_set(v___x_2402_, 1, v___x_2583_);
lean_ctor_set(v___x_2402_, 0, v___x_2556_);
v___x_2585_ = v___x_2402_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2589_; 
v_reuseFailAlloc_2589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2589_, 0, v___x_2556_);
lean_ctor_set(v_reuseFailAlloc_2589_, 1, v___x_2583_);
v___x_2585_ = v_reuseFailAlloc_2589_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
lean_object* v___x_2587_; 
if (v_isShared_2414_ == 0)
{
lean_ctor_set(v___x_2413_, 0, v___x_2585_);
v___x_2587_ = v___x_2413_;
goto v_reusejp_2586_;
}
else
{
lean_object* v_reuseFailAlloc_2588_; 
v_reuseFailAlloc_2588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v___x_2585_);
v___x_2587_ = v_reuseFailAlloc_2588_;
goto v_reusejp_2586_;
}
v_reusejp_2586_:
{
return v___x_2587_;
}
}
}
}
}
}
else
{
lean_object* v_a_2592_; lean_object* v___x_2594_; uint8_t v_isShared_2595_; uint8_t v_isSharedCheck_2599_; 
lean_del_object(v___x_2406_);
lean_del_object(v___x_2402_);
lean_dec(v_snd_2400_);
lean_dec(v_fst_2399_);
lean_dec_ref_known(v___x_2396_, 2);
lean_dec_ref(v___x_2395_);
lean_dec_ref_known(v___x_2393_, 2);
lean_dec_ref_known(v___x_2392_, 2);
lean_dec_ref_known(v___x_2391_, 2);
lean_dec(v___x_2389_);
lean_dec(v___x_2388_);
lean_dec_ref(v_r_x27_2380_);
lean_dec_ref(v___smul_2379_);
lean_dec_ref(v_R_x27_2378_);
lean_dec(v_u_x27_2377_);
lean_dec_ref(v_sAlg_2375_);
lean_dec_ref(v_sA_2374_);
lean_dec_ref(v_sR_2373_);
lean_dec_ref(v_A_2372_);
lean_dec_ref(v_R_2371_);
lean_dec(v_v_2370_);
lean_dec(v_u_2369_);
v_a_2592_ = lean_ctor_get(v___x_2410_, 0);
v_isSharedCheck_2599_ = !lean_is_exclusive(v___x_2410_);
if (v_isSharedCheck_2599_ == 0)
{
v___x_2594_ = v___x_2410_;
v_isShared_2595_ = v_isSharedCheck_2599_;
goto v_resetjp_2593_;
}
else
{
lean_inc(v_a_2592_);
lean_dec(v___x_2410_);
v___x_2594_ = lean_box(0);
v_isShared_2595_ = v_isSharedCheck_2599_;
goto v_resetjp_2593_;
}
v_resetjp_2593_:
{
lean_object* v___x_2597_; 
if (v_isShared_2595_ == 0)
{
v___x_2597_ = v___x_2594_;
goto v_reusejp_2596_;
}
else
{
lean_object* v_reuseFailAlloc_2598_; 
v_reuseFailAlloc_2598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2598_, 0, v_a_2592_);
v___x_2597_ = v_reuseFailAlloc_2598_;
goto v_reusejp_2596_;
}
v_reusejp_2596_:
{
return v___x_2597_;
}
}
}
}
}
}
else
{
lean_object* v_a_2603_; lean_object* v___x_2605_; uint8_t v_isShared_2606_; uint8_t v_isSharedCheck_2610_; 
lean_dec_ref_known(v___x_2396_, 2);
lean_dec_ref(v___x_2395_);
lean_dec_ref_known(v___x_2393_, 2);
lean_dec_ref_known(v___x_2392_, 2);
lean_dec_ref_known(v___x_2391_, 2);
lean_dec(v___x_2389_);
lean_dec(v___x_2388_);
lean_dec_ref(v_r_x27_2380_);
lean_dec_ref(v___smul_2379_);
lean_dec_ref(v_R_x27_2378_);
lean_dec(v_u_x27_2377_);
lean_dec_ref(v_cR_2376_);
lean_dec_ref(v_sAlg_2375_);
lean_dec_ref(v_sA_2374_);
lean_dec_ref(v_sR_2373_);
lean_dec_ref(v_A_2372_);
lean_dec_ref(v_R_2371_);
lean_dec(v_v_2370_);
lean_dec(v_u_2369_);
v_a_2603_ = lean_ctor_get(v___x_2397_, 0);
v_isSharedCheck_2610_ = !lean_is_exclusive(v___x_2397_);
if (v_isSharedCheck_2610_ == 0)
{
v___x_2605_ = v___x_2397_;
v_isShared_2606_ = v_isSharedCheck_2610_;
goto v_resetjp_2604_;
}
else
{
lean_inc(v_a_2603_);
lean_dec(v___x_2397_);
v___x_2605_ = lean_box(0);
v_isShared_2606_ = v_isSharedCheck_2610_;
goto v_resetjp_2604_;
}
v_resetjp_2604_:
{
lean_object* v___x_2608_; 
if (v_isShared_2606_ == 0)
{
v___x_2608_ = v___x_2605_;
goto v_reusejp_2607_;
}
else
{
lean_object* v_reuseFailAlloc_2609_; 
v_reuseFailAlloc_2609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2609_, 0, v_a_2603_);
v___x_2608_ = v_reuseFailAlloc_2609_;
goto v_reusejp_2607_;
}
v_reusejp_2607_:
{
return v___x_2608_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg___boxed(lean_object** _args){
lean_object* v_u_2611_ = _args[0];
lean_object* v_v_2612_ = _args[1];
lean_object* v_R_2613_ = _args[2];
lean_object* v_A_2614_ = _args[3];
lean_object* v_sR_2615_ = _args[4];
lean_object* v_sA_2616_ = _args[5];
lean_object* v_sAlg_2617_ = _args[6];
lean_object* v_cR_2618_ = _args[7];
lean_object* v_u_x27_2619_ = _args[8];
lean_object* v_R_x27_2620_ = _args[9];
lean_object* v___smul_2621_ = _args[10];
lean_object* v_r_x27_2622_ = _args[11];
lean_object* v_a_2623_ = _args[12];
lean_object* v_a_2624_ = _args[13];
lean_object* v_a_2625_ = _args[14];
lean_object* v_a_2626_ = _args[15];
lean_object* v_a_2627_ = _args[16];
lean_object* v_a_2628_ = _args[17];
lean_object* v_a_2629_ = _args[18];
_start:
{
lean_object* v_res_2630_; 
v_res_2630_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg(v_u_2611_, v_v_2612_, v_R_2613_, v_A_2614_, v_sR_2615_, v_sA_2616_, v_sAlg_2617_, v_cR_2618_, v_u_x27_2619_, v_R_x27_2620_, v___smul_2621_, v_r_x27_2622_, v_a_2623_, v_a_2624_, v_a_2625_, v_a_2626_, v_a_2627_, v_a_2628_);
lean_dec(v_a_2628_);
lean_dec_ref(v_a_2627_);
lean_dec(v_a_2626_);
lean_dec_ref(v_a_2625_);
lean_dec(v_a_2624_);
lean_dec_ref(v_a_2623_);
return v_res_2630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast(lean_object* v_u_2631_, lean_object* v_v_2632_, lean_object* v_R_2633_, lean_object* v_A_2634_, lean_object* v_sR_2635_, lean_object* v_sA_2636_, lean_object* v_sAlg_2637_, lean_object* v_cR_2638_, lean_object* v_u_x27_2639_, lean_object* v_R_x27_2640_, lean_object* v_x_2641_, lean_object* v___smul_2642_, lean_object* v_r_x27_2643_, lean_object* v_a_2644_, lean_object* v_a_2645_, lean_object* v_a_2646_, lean_object* v_a_2647_, lean_object* v_a_2648_, lean_object* v_a_2649_){
_start:
{
lean_object* v___x_2651_; 
v___x_2651_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___redArg(v_u_2631_, v_v_2632_, v_R_2633_, v_A_2634_, v_sR_2635_, v_sA_2636_, v_sAlg_2637_, v_cR_2638_, v_u_x27_2639_, v_R_x27_2640_, v___smul_2642_, v_r_x27_2643_, v_a_2644_, v_a_2645_, v_a_2646_, v_a_2647_, v_a_2648_, v_a_2649_);
return v___x_2651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___boxed(lean_object** _args){
lean_object* v_u_2652_ = _args[0];
lean_object* v_v_2653_ = _args[1];
lean_object* v_R_2654_ = _args[2];
lean_object* v_A_2655_ = _args[3];
lean_object* v_sR_2656_ = _args[4];
lean_object* v_sA_2657_ = _args[5];
lean_object* v_sAlg_2658_ = _args[6];
lean_object* v_cR_2659_ = _args[7];
lean_object* v_u_x27_2660_ = _args[8];
lean_object* v_R_x27_2661_ = _args[9];
lean_object* v_x_2662_ = _args[10];
lean_object* v___smul_2663_ = _args[11];
lean_object* v_r_x27_2664_ = _args[12];
lean_object* v_a_2665_ = _args[13];
lean_object* v_a_2666_ = _args[14];
lean_object* v_a_2667_ = _args[15];
lean_object* v_a_2668_ = _args[16];
lean_object* v_a_2669_ = _args[17];
lean_object* v_a_2670_ = _args[18];
lean_object* v_a_2671_ = _args[19];
_start:
{
lean_object* v_res_2672_; 
v_res_2672_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast(v_u_2652_, v_v_2653_, v_R_2654_, v_A_2655_, v_sR_2656_, v_sA_2657_, v_sAlg_2658_, v_cR_2659_, v_u_x27_2660_, v_R_x27_2661_, v_x_2662_, v___smul_2663_, v_r_x27_2664_, v_a_2665_, v_a_2666_, v_a_2667_, v_a_2668_, v_a_2669_, v_a_2670_);
lean_dec(v_a_2670_);
lean_dec_ref(v_a_2669_);
lean_dec(v_a_2668_);
lean_dec_ref(v_a_2667_);
lean_dec(v_a_2666_);
lean_dec_ref(v_a_2665_);
lean_dec_ref(v_x_2662_);
return v_res_2672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0(lean_object* v_msgData_2673_, lean_object* v___y_2674_, lean_object* v___y_2675_, lean_object* v___y_2676_, lean_object* v___y_2677_){
_start:
{
lean_object* v___x_2679_; lean_object* v_env_2680_; lean_object* v___x_2681_; lean_object* v_mctx_2682_; lean_object* v_lctx_2683_; lean_object* v_options_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; 
v___x_2679_ = lean_st_ref_get(v___y_2677_);
v_env_2680_ = lean_ctor_get(v___x_2679_, 0);
lean_inc_ref(v_env_2680_);
lean_dec(v___x_2679_);
v___x_2681_ = lean_st_ref_get(v___y_2675_);
v_mctx_2682_ = lean_ctor_get(v___x_2681_, 0);
lean_inc_ref(v_mctx_2682_);
lean_dec(v___x_2681_);
v_lctx_2683_ = lean_ctor_get(v___y_2674_, 2);
v_options_2684_ = lean_ctor_get(v___y_2676_, 2);
lean_inc_ref(v_options_2684_);
lean_inc_ref(v_lctx_2683_);
v___x_2685_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2685_, 0, v_env_2680_);
lean_ctor_set(v___x_2685_, 1, v_mctx_2682_);
lean_ctor_set(v___x_2685_, 2, v_lctx_2683_);
lean_ctor_set(v___x_2685_, 3, v_options_2684_);
v___x_2686_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2686_, 0, v___x_2685_);
lean_ctor_set(v___x_2686_, 1, v_msgData_2673_);
v___x_2687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2686_);
return v___x_2687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0___boxed(lean_object* v_msgData_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_, lean_object* v___y_2692_, lean_object* v___y_2693_){
_start:
{
lean_object* v_res_2694_; 
v_res_2694_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0(v_msgData_2688_, v___y_2689_, v___y_2690_, v___y_2691_, v___y_2692_);
lean_dec(v___y_2692_);
lean_dec_ref(v___y_2691_);
lean_dec(v___y_2690_);
lean_dec_ref(v___y_2689_);
return v_res_2694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(lean_object* v_msg_2695_, lean_object* v___y_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_){
_start:
{
lean_object* v_ref_2701_; lean_object* v___x_2702_; lean_object* v_a_2703_; lean_object* v___x_2705_; uint8_t v_isShared_2706_; uint8_t v_isSharedCheck_2711_; 
v_ref_2701_ = lean_ctor_get(v___y_2698_, 5);
v___x_2702_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0(v_msg_2695_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_);
v_a_2703_ = lean_ctor_get(v___x_2702_, 0);
v_isSharedCheck_2711_ = !lean_is_exclusive(v___x_2702_);
if (v_isSharedCheck_2711_ == 0)
{
v___x_2705_ = v___x_2702_;
v_isShared_2706_ = v_isSharedCheck_2711_;
goto v_resetjp_2704_;
}
else
{
lean_inc(v_a_2703_);
lean_dec(v___x_2702_);
v___x_2705_ = lean_box(0);
v_isShared_2706_ = v_isSharedCheck_2711_;
goto v_resetjp_2704_;
}
v_resetjp_2704_:
{
lean_object* v___x_2707_; lean_object* v___x_2709_; 
lean_inc(v_ref_2701_);
v___x_2707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2707_, 0, v_ref_2701_);
lean_ctor_set(v___x_2707_, 1, v_a_2703_);
if (v_isShared_2706_ == 0)
{
lean_ctor_set_tag(v___x_2705_, 1);
lean_ctor_set(v___x_2705_, 0, v___x_2707_);
v___x_2709_ = v___x_2705_;
goto v_reusejp_2708_;
}
else
{
lean_object* v_reuseFailAlloc_2710_; 
v_reuseFailAlloc_2710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2710_, 0, v___x_2707_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg___boxed(lean_object* v_msg_2712_, lean_object* v___y_2713_, lean_object* v___y_2714_, lean_object* v___y_2715_, lean_object* v___y_2716_, lean_object* v___y_2717_){
_start:
{
lean_object* v_res_2718_; 
v_res_2718_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(v_msg_2712_, v___y_2713_, v___y_2714_, v___y_2715_, v___y_2716_);
lean_dec(v___y_2716_);
lean_dec_ref(v___y_2715_);
lean_dec(v___y_2714_);
lean_dec_ref(v___y_2713_);
return v_res_2718_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1(void){
_start:
{
lean_object* v___x_2720_; lean_object* v___x_2721_; 
v___x_2720_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__0));
v___x_2721_ = l_Lean_stringToMessageData(v___x_2720_);
return v___x_2721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg(lean_object* v_u_2728_, lean_object* v_v_2729_, lean_object* v_R_2730_, lean_object* v_A_2731_, lean_object* v_sR_2732_, lean_object* v_sA_2733_, lean_object* v_sAlg_2734_, lean_object* v_cR_2735_, lean_object* v___rA_2736_, lean_object* v_za_2737_, lean_object* v_a_2738_, lean_object* v_a_2739_, lean_object* v_a_2740_, lean_object* v_a_2741_){
_start:
{
lean_object* v_toCache_2743_; lean_object* v___x_2745_; uint8_t v_isShared_2746_; uint8_t v_isSharedCheck_2854_; 
v_toCache_2743_ = lean_ctor_get(v_cR_2735_, 0);
v_isSharedCheck_2854_ = !lean_is_exclusive(v_cR_2735_);
if (v_isSharedCheck_2854_ == 0)
{
lean_object* v_unused_2855_; 
v_unused_2855_ = lean_ctor_get(v_cR_2735_, 1);
lean_dec(v_unused_2855_);
v___x_2745_ = v_cR_2735_;
v_isShared_2746_ = v_isSharedCheck_2854_;
goto v_resetjp_2744_;
}
else
{
lean_inc(v_toCache_2743_);
lean_dec(v_cR_2735_);
v___x_2745_ = lean_box(0);
v_isShared_2746_ = v_isSharedCheck_2854_;
goto v_resetjp_2744_;
}
v_resetjp_2744_:
{
lean_object* v_r_u03b1_2747_; 
v_r_u03b1_2747_ = lean_ctor_get(v_toCache_2743_, 0);
if (lean_obj_tag(v_r_u03b1_2747_) == 0)
{
lean_object* v___x_2748_; lean_object* v___x_2749_; 
lean_del_object(v___x_2745_);
lean_dec_ref(v_toCache_2743_);
lean_dec_ref(v_za_2737_);
lean_dec_ref(v___rA_2736_);
lean_dec_ref(v_sAlg_2734_);
lean_dec_ref(v_sA_2733_);
lean_dec_ref(v_sR_2732_);
lean_dec_ref(v_A_2731_);
lean_dec_ref(v_R_2730_);
lean_dec(v_v_2729_);
lean_dec(v_u_2728_);
v___x_2748_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1);
v___x_2749_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(v___x_2748_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
return v___x_2749_;
}
else
{
lean_object* v_r_2750_; lean_object* v_x_2751_; lean_object* v___x_2753_; uint8_t v_isShared_2754_; uint8_t v_isSharedCheck_2853_; 
v_r_2750_ = lean_ctor_get(v_za_2737_, 0);
v_x_2751_ = lean_ctor_get(v_za_2737_, 1);
v_isSharedCheck_2853_ = !lean_is_exclusive(v_za_2737_);
if (v_isSharedCheck_2853_ == 0)
{
v___x_2753_ = v_za_2737_;
v_isShared_2754_ = v_isSharedCheck_2853_;
goto v_resetjp_2752_;
}
else
{
lean_inc(v_x_2751_);
lean_inc(v_r_2750_);
lean_dec(v_za_2737_);
v___x_2753_ = lean_box(0);
v_isShared_2754_ = v_isSharedCheck_2853_;
goto v_resetjp_2752_;
}
v_resetjp_2752_:
{
lean_object* v_val_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2759_; 
v_val_2755_ = lean_ctor_get(v_r_u03b1_2747_, 0);
lean_inc(v_val_2755_);
lean_inc_ref(v_sR_2732_);
lean_inc_ref(v_R_2730_);
lean_inc_n(v_u_2728_, 2);
v___x_2756_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_2728_, v_R_2730_, v_sR_2732_, v_toCache_2743_);
v___x_2757_ = lean_box(0);
if (v_isShared_2746_ == 0)
{
lean_ctor_set_tag(v___x_2745_, 1);
lean_ctor_set(v___x_2745_, 1, v___x_2757_);
lean_ctor_set(v___x_2745_, 0, v_u_2728_);
v___x_2759_ = v___x_2745_;
goto v_reusejp_2758_;
}
else
{
lean_object* v_reuseFailAlloc_2852_; 
v_reuseFailAlloc_2852_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2852_, 0, v_u_2728_);
lean_ctor_set(v_reuseFailAlloc_2852_, 1, v___x_2757_);
v___x_2759_ = v_reuseFailAlloc_2852_;
goto v_reusejp_2758_;
}
v_reusejp_2758_:
{
lean_object* v___x_2760_; 
lean_inc(v_val_2755_);
lean_inc_ref(v_sR_2732_);
lean_inc_ref(v_R_2730_);
lean_inc(v_u_2728_);
v___x_2760_ = lp_mathlib_Mathlib_Tactic_Ring_Common_evalNeg___redArg(v_u_2728_, v_R_2730_, v_sR_2732_, v___x_2756_, v_val_2755_, v_x_2751_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
if (lean_obj_tag(v___x_2760_) == 0)
{
lean_object* v_a_2761_; lean_object* v___x_2763_; uint8_t v_isShared_2764_; uint8_t v_isSharedCheck_2843_; 
v_a_2761_ = lean_ctor_get(v___x_2760_, 0);
v_isSharedCheck_2843_ = !lean_is_exclusive(v___x_2760_);
if (v_isSharedCheck_2843_ == 0)
{
v___x_2763_ = v___x_2760_;
v_isShared_2764_ = v_isSharedCheck_2843_;
goto v_resetjp_2762_;
}
else
{
lean_inc(v_a_2761_);
lean_dec(v___x_2760_);
v___x_2763_ = lean_box(0);
v_isShared_2764_ = v_isSharedCheck_2843_;
goto v_resetjp_2762_;
}
v_resetjp_2762_:
{
lean_object* v_expr_2765_; lean_object* v_val_2766_; lean_object* v_proof_2767_; lean_object* v___x_2769_; uint8_t v_isShared_2770_; uint8_t v_isSharedCheck_2842_; 
v_expr_2765_ = lean_ctor_get(v_a_2761_, 0);
v_val_2766_ = lean_ctor_get(v_a_2761_, 1);
v_proof_2767_ = lean_ctor_get(v_a_2761_, 2);
v_isSharedCheck_2842_ = !lean_is_exclusive(v_a_2761_);
if (v_isSharedCheck_2842_ == 0)
{
v___x_2769_ = v_a_2761_;
v_isShared_2770_ = v_isSharedCheck_2842_;
goto v_resetjp_2768_;
}
else
{
lean_inc(v_proof_2767_);
lean_inc(v_val_2766_);
lean_inc(v_expr_2765_);
lean_dec(v_a_2761_);
v___x_2769_ = lean_box(0);
v_isShared_2770_ = v_isSharedCheck_2842_;
goto v_resetjp_2768_;
}
v_resetjp_2768_:
{
lean_object* v___x_2771_; lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; lean_object* v___x_2793_; lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; uint8_t v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; lean_object* v___x_2819_; lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2824_; 
lean_inc(v_v_2729_);
v___x_2771_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2771_, 0, v_v_2729_);
lean_ctor_set(v___x_2771_, 1, v___x_2757_);
v___x_2772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc(v_u_2728_);
v___x_2773_ = l_Lean_Level_succ___override(v_u_2728_);
v___x_2774_ = l_Lean_Level_succ___override(v_v_2729_);
lean_inc(v___x_2774_);
lean_inc(v___x_2773_);
v___x_2775_ = l_Lean_Level_max___override(v___x_2773_, v___x_2774_);
v___x_2776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2776_, 0, v___x_2774_);
lean_ctor_set(v___x_2776_, 1, v___x_2757_);
v___x_2777_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2777_, 0, v___x_2773_);
lean_ctor_set(v___x_2777_, 1, v___x_2776_);
v___x_2778_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2778_, 0, v___x_2775_);
lean_ctor_set(v___x_2778_, 1, v___x_2777_);
v___x_2779_ = l_Lean_Expr_const___override(v___x_2772_, v___x_2778_);
v___x_2780_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_ref_n(v___x_2771_, 2);
v___x_2781_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2781_, 0, v_u_2728_);
lean_ctor_set(v___x_2781_, 1, v___x_2771_);
lean_inc_ref_n(v___x_2781_, 3);
v___x_2782_ = l_Lean_Expr_const___override(v___x_2780_, v___x_2781_);
lean_inc_ref_n(v_R_2730_, 7);
v___x_2783_ = l_Lean_Expr_app___override(v___x_2782_, v_R_2730_);
lean_inc_ref_n(v_A_2731_, 6);
v___x_2784_ = l_Lean_Expr_app___override(v___x_2783_, v_A_2731_);
v___x_2785_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref(v___x_2759_);
v___x_2786_ = l_Lean_Expr_const___override(v___x_2785_, v___x_2759_);
v___x_2787_ = l_Lean_Expr_app___override(v___x_2786_, v_R_2730_);
v___x_2788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_2789_ = l_Lean_Expr_const___override(v___x_2788_, v___x_2759_);
v___x_2790_ = l_Lean_Expr_app___override(v___x_2789_, v_R_2730_);
lean_inc_ref(v_sR_2732_);
v___x_2791_ = l_Lean_Expr_app___override(v___x_2790_, v_sR_2732_);
v___x_2792_ = l_Lean_Expr_app___override(v___x_2787_, v___x_2791_);
lean_inc_ref(v___x_2792_);
v___x_2793_ = l_Lean_Expr_app___override(v___x_2784_, v___x_2792_);
v___x_2794_ = l_Lean_Expr_const___override(v___x_2785_, v___x_2771_);
v___x_2795_ = l_Lean_Expr_app___override(v___x_2794_, v_A_2731_);
v___x_2796_ = l_Lean_Expr_const___override(v___x_2788_, v___x_2771_);
v___x_2797_ = l_Lean_Expr_app___override(v___x_2796_, v_A_2731_);
v___x_2798_ = l_Lean_Expr_app___override(v___x_2797_, v_sA_2733_);
lean_inc_ref(v___x_2798_);
v___x_2799_ = l_Lean_Expr_app___override(v___x_2795_, v___x_2798_);
lean_inc_ref(v___x_2799_);
v___x_2800_ = l_Lean_Expr_app___override(v___x_2793_, v___x_2799_);
v___x_2801_ = l_Lean_Expr_app___override(v___x_2779_, v___x_2800_);
v___x_2802_ = l_Lean_Expr_app___override(v___x_2801_, v_R_2730_);
v___x_2803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_2804_ = 0;
v___x_2805_ = l_Lean_Expr_lam___override(v___x_2803_, v_R_2730_, v_A_2731_, v___x_2804_);
v___x_2806_ = l_Lean_Expr_app___override(v___x_2802_, v___x_2805_);
v___x_2807_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_2808_ = l_Lean_Expr_const___override(v___x_2807_, v___x_2781_);
v___x_2809_ = l_Lean_Expr_app___override(v___x_2808_, v_R_2730_);
v___x_2810_ = l_Lean_Expr_app___override(v___x_2809_, v_A_2731_);
v___x_2811_ = l_Lean_Expr_app___override(v___x_2810_, v___x_2792_);
v___x_2812_ = l_Lean_Expr_app___override(v___x_2811_, v___x_2799_);
v___x_2813_ = l_Lean_Expr_app___override(v___x_2806_, v___x_2812_);
v___x_2814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_2815_ = l_Lean_Expr_const___override(v___x_2814_, v___x_2781_);
v___x_2816_ = l_Lean_Expr_app___override(v___x_2815_, v_R_2730_);
v___x_2817_ = l_Lean_Expr_app___override(v___x_2816_, v_A_2731_);
v___x_2818_ = l_Lean_Expr_app___override(v___x_2817_, v_sR_2732_);
v___x_2819_ = l_Lean_Expr_app___override(v___x_2818_, v___x_2798_);
lean_inc_ref(v_sAlg_2734_);
v___x_2820_ = l_Lean_Expr_app___override(v___x_2819_, v_sAlg_2734_);
v___x_2821_ = l_Lean_Expr_app___override(v___x_2813_, v___x_2820_);
lean_inc_ref_n(v_expr_2765_, 2);
v___x_2822_ = l_Lean_Expr_app___override(v___x_2821_, v_expr_2765_);
if (v_isShared_2754_ == 0)
{
lean_ctor_set(v___x_2753_, 1, v_val_2766_);
lean_ctor_set(v___x_2753_, 0, v_expr_2765_);
v___x_2824_ = v___x_2753_;
goto v_reusejp_2823_;
}
else
{
lean_object* v_reuseFailAlloc_2841_; 
v_reuseFailAlloc_2841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2841_, 0, v_expr_2765_);
lean_ctor_set(v_reuseFailAlloc_2841_, 1, v_val_2766_);
v___x_2824_ = v_reuseFailAlloc_2841_;
goto v_reusejp_2823_;
}
v_reusejp_2823_:
{
lean_object* v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; lean_object* v___x_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2836_; 
v___x_2825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__3));
v___x_2826_ = l_Lean_Expr_const___override(v___x_2825_, v___x_2781_);
v___x_2827_ = l_Lean_Expr_app___override(v___x_2826_, v_R_2730_);
v___x_2828_ = l_Lean_Expr_app___override(v___x_2827_, v_A_2731_);
v___x_2829_ = l_Lean_Expr_app___override(v___x_2828_, v_val_2755_);
v___x_2830_ = l_Lean_Expr_app___override(v___x_2829_, v___rA_2736_);
v___x_2831_ = l_Lean_Expr_app___override(v___x_2830_, v_sAlg_2734_);
v___x_2832_ = l_Lean_Expr_app___override(v___x_2831_, v_r_2750_);
v___x_2833_ = l_Lean_Expr_app___override(v___x_2832_, v_expr_2765_);
v___x_2834_ = l_Lean_Expr_app___override(v___x_2833_, v_proof_2767_);
if (v_isShared_2770_ == 0)
{
lean_ctor_set(v___x_2769_, 2, v___x_2834_);
lean_ctor_set(v___x_2769_, 1, v___x_2824_);
lean_ctor_set(v___x_2769_, 0, v___x_2822_);
v___x_2836_ = v___x_2769_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2840_; 
v_reuseFailAlloc_2840_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2840_, 0, v___x_2822_);
lean_ctor_set(v_reuseFailAlloc_2840_, 1, v___x_2824_);
lean_ctor_set(v_reuseFailAlloc_2840_, 2, v___x_2834_);
v___x_2836_ = v_reuseFailAlloc_2840_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
lean_object* v___x_2838_; 
if (v_isShared_2764_ == 0)
{
lean_ctor_set(v___x_2763_, 0, v___x_2836_);
v___x_2838_ = v___x_2763_;
goto v_reusejp_2837_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v___x_2836_);
v___x_2838_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2837_;
}
v_reusejp_2837_:
{
return v___x_2838_;
}
}
}
}
}
}
else
{
lean_object* v_a_2844_; lean_object* v___x_2846_; uint8_t v_isShared_2847_; uint8_t v_isSharedCheck_2851_; 
lean_dec_ref(v___x_2759_);
lean_dec(v_val_2755_);
lean_del_object(v___x_2753_);
lean_dec_ref(v_r_2750_);
lean_dec_ref(v___rA_2736_);
lean_dec_ref(v_sAlg_2734_);
lean_dec_ref(v_sA_2733_);
lean_dec_ref(v_sR_2732_);
lean_dec_ref(v_A_2731_);
lean_dec_ref(v_R_2730_);
lean_dec(v_v_2729_);
lean_dec(v_u_2728_);
v_a_2844_ = lean_ctor_get(v___x_2760_, 0);
v_isSharedCheck_2851_ = !lean_is_exclusive(v___x_2760_);
if (v_isSharedCheck_2851_ == 0)
{
v___x_2846_ = v___x_2760_;
v_isShared_2847_ = v_isSharedCheck_2851_;
goto v_resetjp_2845_;
}
else
{
lean_inc(v_a_2844_);
lean_dec(v___x_2760_);
v___x_2846_ = lean_box(0);
v_isShared_2847_ = v_isSharedCheck_2851_;
goto v_resetjp_2845_;
}
v_resetjp_2845_:
{
lean_object* v___x_2849_; 
if (v_isShared_2847_ == 0)
{
v___x_2849_ = v___x_2846_;
goto v_reusejp_2848_;
}
else
{
lean_object* v_reuseFailAlloc_2850_; 
v_reuseFailAlloc_2850_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2850_, 0, v_a_2844_);
v___x_2849_ = v_reuseFailAlloc_2850_;
goto v_reusejp_2848_;
}
v_reusejp_2848_:
{
return v___x_2849_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___boxed(lean_object* v_u_2856_, lean_object* v_v_2857_, lean_object* v_R_2858_, lean_object* v_A_2859_, lean_object* v_sR_2860_, lean_object* v_sA_2861_, lean_object* v_sAlg_2862_, lean_object* v_cR_2863_, lean_object* v___rA_2864_, lean_object* v_za_2865_, lean_object* v_a_2866_, lean_object* v_a_2867_, lean_object* v_a_2868_, lean_object* v_a_2869_, lean_object* v_a_2870_){
_start:
{
lean_object* v_res_2871_; 
v_res_2871_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg(v_u_2856_, v_v_2857_, v_R_2858_, v_A_2859_, v_sR_2860_, v_sA_2861_, v_sAlg_2862_, v_cR_2863_, v___rA_2864_, v_za_2865_, v_a_2866_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec(v_a_2869_);
lean_dec_ref(v_a_2868_);
lean_dec(v_a_2867_);
lean_dec_ref(v_a_2866_);
return v_res_2871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg(lean_object* v_u_2872_, lean_object* v_v_2873_, lean_object* v_R_2874_, lean_object* v_A_2875_, lean_object* v_sR_2876_, lean_object* v_sA_2877_, lean_object* v_sAlg_2878_, lean_object* v_cR_2879_, lean_object* v_a_2880_, lean_object* v___rA_2881_, lean_object* v_za_2882_, lean_object* v_a_2883_, lean_object* v_a_2884_, lean_object* v_a_2885_, lean_object* v_a_2886_){
_start:
{
lean_object* v___x_2888_; 
v___x_2888_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg(v_u_2872_, v_v_2873_, v_R_2874_, v_A_2875_, v_sR_2876_, v_sA_2877_, v_sAlg_2878_, v_cR_2879_, v___rA_2881_, v_za_2882_, v_a_2883_, v_a_2884_, v_a_2885_, v_a_2886_);
return v___x_2888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___boxed(lean_object* v_u_2889_, lean_object* v_v_2890_, lean_object* v_R_2891_, lean_object* v_A_2892_, lean_object* v_sR_2893_, lean_object* v_sA_2894_, lean_object* v_sAlg_2895_, lean_object* v_cR_2896_, lean_object* v_a_2897_, lean_object* v___rA_2898_, lean_object* v_za_2899_, lean_object* v_a_2900_, lean_object* v_a_2901_, lean_object* v_a_2902_, lean_object* v_a_2903_, lean_object* v_a_2904_){
_start:
{
lean_object* v_res_2905_; 
v_res_2905_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg(v_u_2889_, v_v_2890_, v_R_2891_, v_A_2892_, v_sR_2893_, v_sA_2894_, v_sAlg_2895_, v_cR_2896_, v_a_2897_, v___rA_2898_, v_za_2899_, v_a_2900_, v_a_2901_, v_a_2902_, v_a_2903_);
lean_dec(v_a_2903_);
lean_dec_ref(v_a_2902_);
lean_dec(v_a_2901_);
lean_dec_ref(v_a_2900_);
lean_dec_ref(v_a_2897_);
return v_res_2905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0(lean_object* v_00_u03b1_2906_, lean_object* v_msg_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_){
_start:
{
lean_object* v___x_2913_; 
v___x_2913_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(v_msg_2907_, v___y_2908_, v___y_2909_, v___y_2910_, v___y_2911_);
return v___x_2913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___boxed(lean_object* v_00_u03b1_2914_, lean_object* v_msg_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_){
_start:
{
lean_object* v_res_2921_; 
v_res_2921_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0(v_00_u03b1_2914_, v_msg_2915_, v___y_2916_, v___y_2917_, v___y_2918_, v___y_2919_);
lean_dec(v___y_2919_);
lean_dec_ref(v___y_2918_);
lean_dec(v___y_2917_);
lean_dec_ref(v___y_2916_);
return v_res_2921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg(lean_object* v_u_2928_, lean_object* v_v_2929_, lean_object* v_R_2930_, lean_object* v_A_2931_, lean_object* v_sR_2932_, lean_object* v_sA_2933_, lean_object* v_sAlg_2934_, lean_object* v_cR_2935_, lean_object* v_b_2936_, lean_object* v_za_2937_, lean_object* v_vb_2938_, lean_object* v_a_2939_, lean_object* v_a_2940_, lean_object* v_a_2941_, lean_object* v_a_2942_){
_start:
{
lean_object* v_r_2944_; lean_object* v_x_2945_; lean_object* v___x_2947_; uint8_t v_isShared_2948_; uint8_t v_isSharedCheck_3047_; 
v_r_2944_ = lean_ctor_get(v_za_2937_, 0);
v_x_2945_ = lean_ctor_get(v_za_2937_, 1);
v_isSharedCheck_3047_ = !lean_is_exclusive(v_za_2937_);
if (v_isSharedCheck_3047_ == 0)
{
v___x_2947_ = v_za_2937_;
v_isShared_2948_ = v_isSharedCheck_3047_;
goto v_resetjp_2946_;
}
else
{
lean_inc(v_x_2945_);
lean_inc(v_r_2944_);
lean_dec(v_za_2937_);
v___x_2947_ = lean_box(0);
v_isShared_2948_ = v_isSharedCheck_3047_;
goto v_resetjp_2946_;
}
v_resetjp_2946_:
{
lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; 
lean_inc_ref_n(v_sR_2932_, 2);
lean_inc_ref_n(v_R_2930_, 2);
lean_inc_n(v_u_2928_, 2);
v___x_2949_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_2928_, v_R_2930_, v_sR_2932_, v_cR_2935_);
v___x_2950_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_b_2936_);
lean_inc_ref(v_r_2944_);
v___x_2951_ = lp_mathlib_Mathlib_Tactic_Ring_Common_evalPow_u2081___redArg(v_u_2928_, v_R_2930_, v_sR_2932_, v___x_2949_, v___x_2950_, v_r_2944_, v_b_2936_, v_x_2945_, v_vb_2938_, v_a_2939_, v_a_2940_, v_a_2941_, v_a_2942_);
if (lean_obj_tag(v___x_2951_) == 0)
{
lean_object* v_a_2952_; lean_object* v___x_2954_; uint8_t v_isShared_2955_; uint8_t v_isSharedCheck_3038_; 
v_a_2952_ = lean_ctor_get(v___x_2951_, 0);
v_isSharedCheck_3038_ = !lean_is_exclusive(v___x_2951_);
if (v_isSharedCheck_3038_ == 0)
{
v___x_2954_ = v___x_2951_;
v_isShared_2955_ = v_isSharedCheck_3038_;
goto v_resetjp_2953_;
}
else
{
lean_inc(v_a_2952_);
lean_dec(v___x_2951_);
v___x_2954_ = lean_box(0);
v_isShared_2955_ = v_isSharedCheck_3038_;
goto v_resetjp_2953_;
}
v_resetjp_2953_:
{
lean_object* v_expr_2956_; lean_object* v_val_2957_; lean_object* v_proof_2958_; lean_object* v___x_2960_; uint8_t v_isShared_2961_; uint8_t v_isSharedCheck_3037_; 
v_expr_2956_ = lean_ctor_get(v_a_2952_, 0);
v_val_2957_ = lean_ctor_get(v_a_2952_, 1);
v_proof_2958_ = lean_ctor_get(v_a_2952_, 2);
v_isSharedCheck_3037_ = !lean_is_exclusive(v_a_2952_);
if (v_isSharedCheck_3037_ == 0)
{
v___x_2960_ = v_a_2952_;
v_isShared_2961_ = v_isSharedCheck_3037_;
goto v_resetjp_2959_;
}
else
{
lean_inc(v_proof_2958_);
lean_inc(v_val_2957_);
lean_inc(v_expr_2956_);
lean_dec(v_a_2952_);
v___x_2960_ = lean_box(0);
v_isShared_2961_ = v_isSharedCheck_3037_;
goto v_resetjp_2959_;
}
v_resetjp_2959_:
{
lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; lean_object* v___x_2965_; lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___x_2973_; lean_object* v___x_2974_; lean_object* v___x_2975_; lean_object* v___x_2976_; lean_object* v___x_2977_; lean_object* v___x_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2982_; lean_object* v___x_2983_; lean_object* v___x_2984_; lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; lean_object* v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; lean_object* v___x_2992_; lean_object* v___x_2993_; lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; uint8_t v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___x_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3017_; 
v___x_2962_ = lean_box(0);
lean_inc(v_v_2929_);
v___x_2963_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2963_, 0, v_v_2929_);
lean_ctor_set(v___x_2963_, 1, v___x_2962_);
v___x_2964_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
lean_inc_ref_n(v___x_2963_, 2);
v___x_2965_ = l_Lean_Expr_const___override(v___x_2964_, v___x_2963_);
lean_inc_ref_n(v_A_2931_, 6);
v___x_2966_ = l_Lean_Expr_app___override(v___x_2965_, v_A_2931_);
lean_inc_ref(v_sA_2933_);
v___x_2967_ = l_Lean_Expr_app___override(v___x_2966_, v_sA_2933_);
v___x_2968_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_2928_, 2);
v___x_2969_ = l_Lean_Level_succ___override(v_u_2928_);
v___x_2970_ = l_Lean_Level_succ___override(v_v_2929_);
lean_inc(v___x_2970_);
lean_inc(v___x_2969_);
v___x_2971_ = l_Lean_Level_max___override(v___x_2969_, v___x_2970_);
v___x_2972_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2972_, 0, v___x_2970_);
lean_ctor_set(v___x_2972_, 1, v___x_2962_);
v___x_2973_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2973_, 0, v___x_2969_);
lean_ctor_set(v___x_2973_, 1, v___x_2972_);
v___x_2974_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2974_, 0, v___x_2971_);
lean_ctor_set(v___x_2974_, 1, v___x_2973_);
v___x_2975_ = l_Lean_Expr_const___override(v___x_2968_, v___x_2974_);
v___x_2976_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_2977_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2977_, 0, v_u_2928_);
lean_ctor_set(v___x_2977_, 1, v___x_2963_);
lean_inc_ref_n(v___x_2977_, 3);
v___x_2978_ = l_Lean_Expr_const___override(v___x_2976_, v___x_2977_);
lean_inc_ref_n(v_R_2930_, 7);
v___x_2979_ = l_Lean_Expr_app___override(v___x_2978_, v_R_2930_);
v___x_2980_ = l_Lean_Expr_app___override(v___x_2979_, v_A_2931_);
v___x_2981_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_2982_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2982_, 0, v_u_2928_);
lean_ctor_set(v___x_2982_, 1, v___x_2962_);
lean_inc_ref(v___x_2982_);
v___x_2983_ = l_Lean_Expr_const___override(v___x_2981_, v___x_2982_);
v___x_2984_ = l_Lean_Expr_app___override(v___x_2983_, v_R_2930_);
v___x_2985_ = l_Lean_Expr_const___override(v___x_2964_, v___x_2982_);
v___x_2986_ = l_Lean_Expr_app___override(v___x_2985_, v_R_2930_);
lean_inc_ref_n(v_sR_2932_, 2);
v___x_2987_ = l_Lean_Expr_app___override(v___x_2986_, v_sR_2932_);
v___x_2988_ = l_Lean_Expr_app___override(v___x_2984_, v___x_2987_);
lean_inc_ref(v___x_2988_);
v___x_2989_ = l_Lean_Expr_app___override(v___x_2980_, v___x_2988_);
v___x_2990_ = l_Lean_Expr_const___override(v___x_2981_, v___x_2963_);
v___x_2991_ = l_Lean_Expr_app___override(v___x_2990_, v_A_2931_);
lean_inc_ref(v___x_2967_);
v___x_2992_ = l_Lean_Expr_app___override(v___x_2991_, v___x_2967_);
lean_inc_ref(v___x_2992_);
v___x_2993_ = l_Lean_Expr_app___override(v___x_2989_, v___x_2992_);
v___x_2994_ = l_Lean_Expr_app___override(v___x_2975_, v___x_2993_);
v___x_2995_ = l_Lean_Expr_app___override(v___x_2994_, v_R_2930_);
v___x_2996_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_2997_ = 0;
v___x_2998_ = l_Lean_Expr_lam___override(v___x_2996_, v_R_2930_, v_A_2931_, v___x_2997_);
v___x_2999_ = l_Lean_Expr_app___override(v___x_2995_, v___x_2998_);
v___x_3000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_3001_ = l_Lean_Expr_const___override(v___x_3000_, v___x_2977_);
v___x_3002_ = l_Lean_Expr_app___override(v___x_3001_, v_R_2930_);
v___x_3003_ = l_Lean_Expr_app___override(v___x_3002_, v_A_2931_);
v___x_3004_ = l_Lean_Expr_app___override(v___x_3003_, v___x_2988_);
v___x_3005_ = l_Lean_Expr_app___override(v___x_3004_, v___x_2992_);
v___x_3006_ = l_Lean_Expr_app___override(v___x_2999_, v___x_3005_);
v___x_3007_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_3008_ = l_Lean_Expr_const___override(v___x_3007_, v___x_2977_);
v___x_3009_ = l_Lean_Expr_app___override(v___x_3008_, v_R_2930_);
v___x_3010_ = l_Lean_Expr_app___override(v___x_3009_, v_A_2931_);
v___x_3011_ = l_Lean_Expr_app___override(v___x_3010_, v_sR_2932_);
v___x_3012_ = l_Lean_Expr_app___override(v___x_3011_, v___x_2967_);
lean_inc_ref(v_sAlg_2934_);
v___x_3013_ = l_Lean_Expr_app___override(v___x_3012_, v_sAlg_2934_);
v___x_3014_ = l_Lean_Expr_app___override(v___x_3006_, v___x_3013_);
lean_inc_ref_n(v_expr_2956_, 2);
v___x_3015_ = l_Lean_Expr_app___override(v___x_3014_, v_expr_2956_);
if (v_isShared_2948_ == 0)
{
lean_ctor_set(v___x_2947_, 1, v_val_2957_);
lean_ctor_set(v___x_2947_, 0, v_expr_2956_);
v___x_3017_ = v___x_2947_;
goto v_reusejp_3016_;
}
else
{
lean_object* v_reuseFailAlloc_3036_; 
v_reuseFailAlloc_3036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3036_, 0, v_expr_2956_);
lean_ctor_set(v_reuseFailAlloc_3036_, 1, v_val_2957_);
v___x_3017_ = v_reuseFailAlloc_3036_;
goto v_reusejp_3016_;
}
v_reusejp_3016_:
{
lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3030_; 
v___x_3018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___closed__1));
v___x_3019_ = l_Lean_Expr_const___override(v___x_3018_, v___x_2977_);
v___x_3020_ = l_Lean_Expr_app___override(v___x_3019_, v_R_2930_);
v___x_3021_ = l_Lean_Expr_app___override(v___x_3020_, v_A_2931_);
v___x_3022_ = l_Lean_Expr_app___override(v___x_3021_, v_sR_2932_);
v___x_3023_ = l_Lean_Expr_app___override(v___x_3022_, v_sA_2933_);
v___x_3024_ = l_Lean_Expr_app___override(v___x_3023_, v_sAlg_2934_);
v___x_3025_ = l_Lean_Expr_app___override(v___x_3024_, v_r_2944_);
v___x_3026_ = l_Lean_Expr_app___override(v___x_3025_, v_expr_2956_);
v___x_3027_ = l_Lean_Expr_app___override(v___x_3026_, v_b_2936_);
v___x_3028_ = l_Lean_Expr_app___override(v___x_3027_, v_proof_2958_);
if (v_isShared_2961_ == 0)
{
lean_ctor_set(v___x_2960_, 2, v___x_3028_);
lean_ctor_set(v___x_2960_, 1, v___x_3017_);
lean_ctor_set(v___x_2960_, 0, v___x_3015_);
v___x_3030_ = v___x_2960_;
goto v_reusejp_3029_;
}
else
{
lean_object* v_reuseFailAlloc_3035_; 
v_reuseFailAlloc_3035_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3035_, 0, v___x_3015_);
lean_ctor_set(v_reuseFailAlloc_3035_, 1, v___x_3017_);
lean_ctor_set(v_reuseFailAlloc_3035_, 2, v___x_3028_);
v___x_3030_ = v_reuseFailAlloc_3035_;
goto v_reusejp_3029_;
}
v_reusejp_3029_:
{
lean_object* v___x_3031_; lean_object* v___x_3033_; 
v___x_3031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3031_, 0, v___x_3030_);
if (v_isShared_2955_ == 0)
{
lean_ctor_set(v___x_2954_, 0, v___x_3031_);
v___x_3033_ = v___x_2954_;
goto v_reusejp_3032_;
}
else
{
lean_object* v_reuseFailAlloc_3034_; 
v_reuseFailAlloc_3034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3034_, 0, v___x_3031_);
v___x_3033_ = v_reuseFailAlloc_3034_;
goto v_reusejp_3032_;
}
v_reusejp_3032_:
{
return v___x_3033_;
}
}
}
}
}
}
else
{
lean_object* v_a_3039_; lean_object* v___x_3041_; uint8_t v_isShared_3042_; uint8_t v_isSharedCheck_3046_; 
lean_del_object(v___x_2947_);
lean_dec_ref(v_r_2944_);
lean_dec_ref(v_b_2936_);
lean_dec_ref(v_sAlg_2934_);
lean_dec_ref(v_sA_2933_);
lean_dec_ref(v_sR_2932_);
lean_dec_ref(v_A_2931_);
lean_dec_ref(v_R_2930_);
lean_dec(v_v_2929_);
lean_dec(v_u_2928_);
v_a_3039_ = lean_ctor_get(v___x_2951_, 0);
v_isSharedCheck_3046_ = !lean_is_exclusive(v___x_2951_);
if (v_isSharedCheck_3046_ == 0)
{
v___x_3041_ = v___x_2951_;
v_isShared_3042_ = v_isSharedCheck_3046_;
goto v_resetjp_3040_;
}
else
{
lean_inc(v_a_3039_);
lean_dec(v___x_2951_);
v___x_3041_ = lean_box(0);
v_isShared_3042_ = v_isSharedCheck_3046_;
goto v_resetjp_3040_;
}
v_resetjp_3040_:
{
lean_object* v___x_3044_; 
if (v_isShared_3042_ == 0)
{
v___x_3044_ = v___x_3041_;
goto v_reusejp_3043_;
}
else
{
lean_object* v_reuseFailAlloc_3045_; 
v_reuseFailAlloc_3045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3045_, 0, v_a_3039_);
v___x_3044_ = v_reuseFailAlloc_3045_;
goto v_reusejp_3043_;
}
v_reusejp_3043_:
{
return v___x_3044_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg___boxed(lean_object* v_u_3048_, lean_object* v_v_3049_, lean_object* v_R_3050_, lean_object* v_A_3051_, lean_object* v_sR_3052_, lean_object* v_sA_3053_, lean_object* v_sAlg_3054_, lean_object* v_cR_3055_, lean_object* v_b_3056_, lean_object* v_za_3057_, lean_object* v_vb_3058_, lean_object* v_a_3059_, lean_object* v_a_3060_, lean_object* v_a_3061_, lean_object* v_a_3062_, lean_object* v_a_3063_){
_start:
{
lean_object* v_res_3064_; 
v_res_3064_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg(v_u_3048_, v_v_3049_, v_R_3050_, v_A_3051_, v_sR_3052_, v_sA_3053_, v_sAlg_3054_, v_cR_3055_, v_b_3056_, v_za_3057_, v_vb_3058_, v_a_3059_, v_a_3060_, v_a_3061_, v_a_3062_);
lean_dec(v_a_3062_);
lean_dec_ref(v_a_3061_);
lean_dec(v_a_3060_);
lean_dec_ref(v_a_3059_);
return v_res_3064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow(lean_object* v_u_3065_, lean_object* v_v_3066_, lean_object* v_R_3067_, lean_object* v_A_3068_, lean_object* v_sR_3069_, lean_object* v_sA_3070_, lean_object* v_sAlg_3071_, lean_object* v_cR_3072_, lean_object* v_a_3073_, lean_object* v_b_3074_, lean_object* v_za_3075_, lean_object* v_vb_3076_, lean_object* v_a_3077_, lean_object* v_a_3078_, lean_object* v_a_3079_, lean_object* v_a_3080_){
_start:
{
lean_object* v___x_3082_; 
v___x_3082_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___redArg(v_u_3065_, v_v_3066_, v_R_3067_, v_A_3068_, v_sR_3069_, v_sA_3070_, v_sAlg_3071_, v_cR_3072_, v_b_3074_, v_za_3075_, v_vb_3076_, v_a_3077_, v_a_3078_, v_a_3079_, v_a_3080_);
return v___x_3082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___boxed(lean_object** _args){
lean_object* v_u_3083_ = _args[0];
lean_object* v_v_3084_ = _args[1];
lean_object* v_R_3085_ = _args[2];
lean_object* v_A_3086_ = _args[3];
lean_object* v_sR_3087_ = _args[4];
lean_object* v_sA_3088_ = _args[5];
lean_object* v_sAlg_3089_ = _args[6];
lean_object* v_cR_3090_ = _args[7];
lean_object* v_a_3091_ = _args[8];
lean_object* v_b_3092_ = _args[9];
lean_object* v_za_3093_ = _args[10];
lean_object* v_vb_3094_ = _args[11];
lean_object* v_a_3095_ = _args[12];
lean_object* v_a_3096_ = _args[13];
lean_object* v_a_3097_ = _args[14];
lean_object* v_a_3098_ = _args[15];
lean_object* v_a_3099_ = _args[16];
_start:
{
lean_object* v_res_3100_; 
v_res_3100_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow(v_u_3083_, v_v_3084_, v_R_3085_, v_A_3086_, v_sR_3087_, v_sA_3088_, v_sAlg_3089_, v_cR_3090_, v_a_3091_, v_b_3092_, v_za_3093_, v_vb_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_);
lean_dec(v_a_3098_);
lean_dec_ref(v_a_3097_);
lean_dec(v_a_3096_);
lean_dec_ref(v_a_3095_);
lean_dec_ref(v_a_3091_);
return v_res_3100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg(lean_object* v_u_3107_, lean_object* v_v_3108_, lean_object* v_R_3109_, lean_object* v_A_3110_, lean_object* v_sR_3111_, lean_object* v_sA_3112_, lean_object* v_sAlg_3113_, lean_object* v_cR_3114_, lean_object* v_fA_3115_, lean_object* v_za_3116_, lean_object* v_a_3117_, lean_object* v_a_3118_, lean_object* v_a_3119_, lean_object* v_a_3120_, lean_object* v_a_3121_, lean_object* v_a_3122_){
_start:
{
lean_object* v_toCache_3124_; lean_object* v___x_3126_; uint8_t v_isShared_3127_; uint8_t v_isSharedCheck_3244_; 
v_toCache_3124_ = lean_ctor_get(v_cR_3114_, 0);
v_isSharedCheck_3244_ = !lean_is_exclusive(v_cR_3114_);
if (v_isSharedCheck_3244_ == 0)
{
lean_object* v_unused_3245_; 
v_unused_3245_ = lean_ctor_get(v_cR_3114_, 1);
lean_dec(v_unused_3245_);
v___x_3126_ = v_cR_3114_;
v_isShared_3127_ = v_isSharedCheck_3244_;
goto v_resetjp_3125_;
}
else
{
lean_inc(v_toCache_3124_);
lean_dec(v_cR_3114_);
v___x_3126_ = lean_box(0);
v_isShared_3127_ = v_isSharedCheck_3244_;
goto v_resetjp_3125_;
}
v_resetjp_3125_:
{
lean_object* v_ds_u03b1_3128_; 
v_ds_u03b1_3128_ = lean_ctor_get(v_toCache_3124_, 1);
lean_inc(v_ds_u03b1_3128_);
if (lean_obj_tag(v_ds_u03b1_3128_) == 0)
{
lean_object* v___x_3129_; lean_object* v___x_3130_; 
lean_del_object(v___x_3126_);
lean_dec_ref(v_toCache_3124_);
lean_dec_ref(v_za_3116_);
lean_dec_ref(v_fA_3115_);
lean_dec_ref(v_sAlg_3113_);
lean_dec_ref(v_sA_3112_);
lean_dec_ref(v_sR_3111_);
lean_dec_ref(v_A_3110_);
lean_dec_ref(v_R_3109_);
lean_dec(v_v_3108_);
lean_dec(v_u_3107_);
v___x_3129_ = lean_box(0);
v___x_3130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3130_, 0, v___x_3129_);
return v___x_3130_;
}
else
{
lean_object* v_r_3131_; lean_object* v_x_3132_; lean_object* v___x_3134_; uint8_t v_isShared_3135_; uint8_t v_isSharedCheck_3243_; 
v_r_3131_ = lean_ctor_get(v_za_3116_, 0);
v_x_3132_ = lean_ctor_get(v_za_3116_, 1);
v_isSharedCheck_3243_ = !lean_is_exclusive(v_za_3116_);
if (v_isSharedCheck_3243_ == 0)
{
v___x_3134_ = v_za_3116_;
v_isShared_3135_ = v_isSharedCheck_3243_;
goto v_resetjp_3133_;
}
else
{
lean_inc(v_x_3132_);
lean_inc(v_r_3131_);
lean_dec(v_za_3116_);
v___x_3134_ = lean_box(0);
v_isShared_3135_ = v_isSharedCheck_3243_;
goto v_resetjp_3133_;
}
v_resetjp_3133_:
{
lean_object* v_cz_u03b1_3136_; lean_object* v_val_3137_; lean_object* v___x_3139_; uint8_t v_isShared_3140_; uint8_t v_isSharedCheck_3242_; 
v_cz_u03b1_3136_ = lean_ctor_get(v_toCache_3124_, 2);
lean_inc(v_cz_u03b1_3136_);
v_val_3137_ = lean_ctor_get(v_ds_u03b1_3128_, 0);
v_isSharedCheck_3242_ = !lean_is_exclusive(v_ds_u03b1_3128_);
if (v_isSharedCheck_3242_ == 0)
{
v___x_3139_ = v_ds_u03b1_3128_;
v_isShared_3140_ = v_isSharedCheck_3242_;
goto v_resetjp_3138_;
}
else
{
lean_inc(v_val_3137_);
lean_dec(v_ds_u03b1_3128_);
v___x_3139_ = lean_box(0);
v_isShared_3140_ = v_isSharedCheck_3242_;
goto v_resetjp_3138_;
}
v_resetjp_3138_:
{
lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3145_; 
lean_inc_ref(v_sR_3111_);
lean_inc_ref(v_R_3109_);
lean_inc_n(v_u_3107_, 2);
v___x_3141_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_3107_, v_R_3109_, v_sR_3111_, v_toCache_3124_);
v___x_3142_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
v___x_3143_ = lean_box(0);
if (v_isShared_3127_ == 0)
{
lean_ctor_set_tag(v___x_3126_, 1);
lean_ctor_set(v___x_3126_, 1, v___x_3143_);
lean_ctor_set(v___x_3126_, 0, v_u_3107_);
v___x_3145_ = v___x_3126_;
goto v_reusejp_3144_;
}
else
{
lean_object* v_reuseFailAlloc_3241_; 
v_reuseFailAlloc_3241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3241_, 0, v_u_3107_);
lean_ctor_set(v_reuseFailAlloc_3241_, 1, v___x_3143_);
v___x_3145_ = v_reuseFailAlloc_3241_;
goto v_reusejp_3144_;
}
v_reusejp_3144_:
{
lean_object* v___x_3146_; 
lean_inc(v_val_3137_);
lean_inc_ref(v_sR_3111_);
lean_inc_ref(v_R_3109_);
lean_inc(v_u_3107_);
v___x_3146_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_evalInv___redArg(v_u_3107_, v_R_3109_, v_sR_3111_, v___x_3141_, v___x_3142_, v_val_3137_, v_cz_u03b1_3136_, v_x_3132_, v_a_3117_, v_a_3118_, v_a_3119_, v_a_3120_, v_a_3121_, v_a_3122_);
if (lean_obj_tag(v___x_3146_) == 0)
{
lean_object* v_a_3147_; lean_object* v___x_3149_; uint8_t v_isShared_3150_; uint8_t v_isSharedCheck_3232_; 
v_a_3147_ = lean_ctor_get(v___x_3146_, 0);
v_isSharedCheck_3232_ = !lean_is_exclusive(v___x_3146_);
if (v_isSharedCheck_3232_ == 0)
{
v___x_3149_ = v___x_3146_;
v_isShared_3150_ = v_isSharedCheck_3232_;
goto v_resetjp_3148_;
}
else
{
lean_inc(v_a_3147_);
lean_dec(v___x_3146_);
v___x_3149_ = lean_box(0);
v_isShared_3150_ = v_isSharedCheck_3232_;
goto v_resetjp_3148_;
}
v_resetjp_3148_:
{
lean_object* v_expr_3151_; lean_object* v_val_3152_; lean_object* v_proof_3153_; lean_object* v___x_3155_; uint8_t v_isShared_3156_; uint8_t v_isSharedCheck_3231_; 
v_expr_3151_ = lean_ctor_get(v_a_3147_, 0);
v_val_3152_ = lean_ctor_get(v_a_3147_, 1);
v_proof_3153_ = lean_ctor_get(v_a_3147_, 2);
v_isSharedCheck_3231_ = !lean_is_exclusive(v_a_3147_);
if (v_isSharedCheck_3231_ == 0)
{
v___x_3155_ = v_a_3147_;
v_isShared_3156_ = v_isSharedCheck_3231_;
goto v_resetjp_3154_;
}
else
{
lean_inc(v_proof_3153_);
lean_inc(v_val_3152_);
lean_inc(v_expr_3151_);
lean_dec(v_a_3147_);
v___x_3155_ = lean_box(0);
v_isShared_3156_ = v_isSharedCheck_3231_;
goto v_resetjp_3154_;
}
v_resetjp_3154_:
{
lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; uint8_t v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; lean_object* v___x_3208_; lean_object* v___x_3210_; 
lean_inc(v_v_3108_);
v___x_3157_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3157_, 0, v_v_3108_);
lean_ctor_set(v___x_3157_, 1, v___x_3143_);
v___x_3158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc(v_u_3107_);
v___x_3159_ = l_Lean_Level_succ___override(v_u_3107_);
v___x_3160_ = l_Lean_Level_succ___override(v_v_3108_);
lean_inc(v___x_3160_);
lean_inc(v___x_3159_);
v___x_3161_ = l_Lean_Level_max___override(v___x_3159_, v___x_3160_);
v___x_3162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3162_, 0, v___x_3160_);
lean_ctor_set(v___x_3162_, 1, v___x_3143_);
v___x_3163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3163_, 0, v___x_3159_);
lean_ctor_set(v___x_3163_, 1, v___x_3162_);
v___x_3164_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3164_, 0, v___x_3161_);
lean_ctor_set(v___x_3164_, 1, v___x_3163_);
v___x_3165_ = l_Lean_Expr_const___override(v___x_3158_, v___x_3164_);
v___x_3166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
lean_inc_ref_n(v___x_3157_, 2);
v___x_3167_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3167_, 0, v_u_3107_);
lean_ctor_set(v___x_3167_, 1, v___x_3157_);
lean_inc_ref_n(v___x_3167_, 3);
v___x_3168_ = l_Lean_Expr_const___override(v___x_3166_, v___x_3167_);
lean_inc_ref_n(v_R_3109_, 7);
v___x_3169_ = l_Lean_Expr_app___override(v___x_3168_, v_R_3109_);
lean_inc_ref_n(v_A_3110_, 6);
v___x_3170_ = l_Lean_Expr_app___override(v___x_3169_, v_A_3110_);
v___x_3171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
lean_inc_ref(v___x_3145_);
v___x_3172_ = l_Lean_Expr_const___override(v___x_3171_, v___x_3145_);
v___x_3173_ = l_Lean_Expr_app___override(v___x_3172_, v_R_3109_);
v___x_3174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_3175_ = l_Lean_Expr_const___override(v___x_3174_, v___x_3145_);
v___x_3176_ = l_Lean_Expr_app___override(v___x_3175_, v_R_3109_);
lean_inc_ref(v_sR_3111_);
v___x_3177_ = l_Lean_Expr_app___override(v___x_3176_, v_sR_3111_);
v___x_3178_ = l_Lean_Expr_app___override(v___x_3173_, v___x_3177_);
lean_inc_ref(v___x_3178_);
v___x_3179_ = l_Lean_Expr_app___override(v___x_3170_, v___x_3178_);
v___x_3180_ = l_Lean_Expr_const___override(v___x_3171_, v___x_3157_);
v___x_3181_ = l_Lean_Expr_app___override(v___x_3180_, v_A_3110_);
v___x_3182_ = l_Lean_Expr_const___override(v___x_3174_, v___x_3157_);
v___x_3183_ = l_Lean_Expr_app___override(v___x_3182_, v_A_3110_);
v___x_3184_ = l_Lean_Expr_app___override(v___x_3183_, v_sA_3112_);
lean_inc_ref(v___x_3184_);
v___x_3185_ = l_Lean_Expr_app___override(v___x_3181_, v___x_3184_);
lean_inc_ref(v___x_3185_);
v___x_3186_ = l_Lean_Expr_app___override(v___x_3179_, v___x_3185_);
v___x_3187_ = l_Lean_Expr_app___override(v___x_3165_, v___x_3186_);
v___x_3188_ = l_Lean_Expr_app___override(v___x_3187_, v_R_3109_);
v___x_3189_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_3190_ = 0;
v___x_3191_ = l_Lean_Expr_lam___override(v___x_3189_, v_R_3109_, v_A_3110_, v___x_3190_);
v___x_3192_ = l_Lean_Expr_app___override(v___x_3188_, v___x_3191_);
v___x_3193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_3194_ = l_Lean_Expr_const___override(v___x_3193_, v___x_3167_);
v___x_3195_ = l_Lean_Expr_app___override(v___x_3194_, v_R_3109_);
v___x_3196_ = l_Lean_Expr_app___override(v___x_3195_, v_A_3110_);
v___x_3197_ = l_Lean_Expr_app___override(v___x_3196_, v___x_3178_);
v___x_3198_ = l_Lean_Expr_app___override(v___x_3197_, v___x_3185_);
v___x_3199_ = l_Lean_Expr_app___override(v___x_3192_, v___x_3198_);
v___x_3200_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_3201_ = l_Lean_Expr_const___override(v___x_3200_, v___x_3167_);
v___x_3202_ = l_Lean_Expr_app___override(v___x_3201_, v_R_3109_);
v___x_3203_ = l_Lean_Expr_app___override(v___x_3202_, v_A_3110_);
v___x_3204_ = l_Lean_Expr_app___override(v___x_3203_, v_sR_3111_);
v___x_3205_ = l_Lean_Expr_app___override(v___x_3204_, v___x_3184_);
lean_inc_ref(v_sAlg_3113_);
v___x_3206_ = l_Lean_Expr_app___override(v___x_3205_, v_sAlg_3113_);
v___x_3207_ = l_Lean_Expr_app___override(v___x_3199_, v___x_3206_);
lean_inc_ref_n(v_expr_3151_, 2);
v___x_3208_ = l_Lean_Expr_app___override(v___x_3207_, v_expr_3151_);
if (v_isShared_3135_ == 0)
{
lean_ctor_set(v___x_3134_, 1, v_val_3152_);
lean_ctor_set(v___x_3134_, 0, v_expr_3151_);
v___x_3210_ = v___x_3134_;
goto v_reusejp_3209_;
}
else
{
lean_object* v_reuseFailAlloc_3230_; 
v_reuseFailAlloc_3230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3230_, 0, v_expr_3151_);
lean_ctor_set(v_reuseFailAlloc_3230_, 1, v_val_3152_);
v___x_3210_ = v_reuseFailAlloc_3230_;
goto v_reusejp_3209_;
}
v_reusejp_3209_:
{
lean_object* v___x_3211_; lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; lean_object* v___x_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_3220_; lean_object* v___x_3222_; 
v___x_3211_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___closed__1));
v___x_3212_ = l_Lean_Expr_const___override(v___x_3211_, v___x_3167_);
v___x_3213_ = l_Lean_Expr_app___override(v___x_3212_, v_R_3109_);
v___x_3214_ = l_Lean_Expr_app___override(v___x_3213_, v_A_3110_);
v___x_3215_ = l_Lean_Expr_app___override(v___x_3214_, v_val_3137_);
v___x_3216_ = l_Lean_Expr_app___override(v___x_3215_, v_fA_3115_);
v___x_3217_ = l_Lean_Expr_app___override(v___x_3216_, v_sAlg_3113_);
v___x_3218_ = l_Lean_Expr_app___override(v___x_3217_, v_r_3131_);
v___x_3219_ = l_Lean_Expr_app___override(v___x_3218_, v_expr_3151_);
v___x_3220_ = l_Lean_Expr_app___override(v___x_3219_, v_proof_3153_);
if (v_isShared_3156_ == 0)
{
lean_ctor_set(v___x_3155_, 2, v___x_3220_);
lean_ctor_set(v___x_3155_, 1, v___x_3210_);
lean_ctor_set(v___x_3155_, 0, v___x_3208_);
v___x_3222_ = v___x_3155_;
goto v_reusejp_3221_;
}
else
{
lean_object* v_reuseFailAlloc_3229_; 
v_reuseFailAlloc_3229_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3229_, 0, v___x_3208_);
lean_ctor_set(v_reuseFailAlloc_3229_, 1, v___x_3210_);
lean_ctor_set(v_reuseFailAlloc_3229_, 2, v___x_3220_);
v___x_3222_ = v_reuseFailAlloc_3229_;
goto v_reusejp_3221_;
}
v_reusejp_3221_:
{
lean_object* v___x_3224_; 
if (v_isShared_3140_ == 0)
{
lean_ctor_set(v___x_3139_, 0, v___x_3222_);
v___x_3224_ = v___x_3139_;
goto v_reusejp_3223_;
}
else
{
lean_object* v_reuseFailAlloc_3228_; 
v_reuseFailAlloc_3228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3228_, 0, v___x_3222_);
v___x_3224_ = v_reuseFailAlloc_3228_;
goto v_reusejp_3223_;
}
v_reusejp_3223_:
{
lean_object* v___x_3226_; 
if (v_isShared_3150_ == 0)
{
lean_ctor_set(v___x_3149_, 0, v___x_3224_);
v___x_3226_ = v___x_3149_;
goto v_reusejp_3225_;
}
else
{
lean_object* v_reuseFailAlloc_3227_; 
v_reuseFailAlloc_3227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3227_, 0, v___x_3224_);
v___x_3226_ = v_reuseFailAlloc_3227_;
goto v_reusejp_3225_;
}
v_reusejp_3225_:
{
return v___x_3226_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3233_; lean_object* v___x_3235_; uint8_t v_isShared_3236_; uint8_t v_isSharedCheck_3240_; 
lean_dec_ref(v___x_3145_);
lean_del_object(v___x_3139_);
lean_dec(v_val_3137_);
lean_del_object(v___x_3134_);
lean_dec_ref(v_r_3131_);
lean_dec_ref(v_fA_3115_);
lean_dec_ref(v_sAlg_3113_);
lean_dec_ref(v_sA_3112_);
lean_dec_ref(v_sR_3111_);
lean_dec_ref(v_A_3110_);
lean_dec_ref(v_R_3109_);
lean_dec(v_v_3108_);
lean_dec(v_u_3107_);
v_a_3233_ = lean_ctor_get(v___x_3146_, 0);
v_isSharedCheck_3240_ = !lean_is_exclusive(v___x_3146_);
if (v_isSharedCheck_3240_ == 0)
{
v___x_3235_ = v___x_3146_;
v_isShared_3236_ = v_isSharedCheck_3240_;
goto v_resetjp_3234_;
}
else
{
lean_inc(v_a_3233_);
lean_dec(v___x_3146_);
v___x_3235_ = lean_box(0);
v_isShared_3236_ = v_isSharedCheck_3240_;
goto v_resetjp_3234_;
}
v_resetjp_3234_:
{
lean_object* v___x_3238_; 
if (v_isShared_3236_ == 0)
{
v___x_3238_ = v___x_3235_;
goto v_reusejp_3237_;
}
else
{
lean_object* v_reuseFailAlloc_3239_; 
v_reuseFailAlloc_3239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3239_, 0, v_a_3233_);
v___x_3238_ = v_reuseFailAlloc_3239_;
goto v_reusejp_3237_;
}
v_reusejp_3237_:
{
return v___x_3238_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg___boxed(lean_object** _args){
lean_object* v_u_3246_ = _args[0];
lean_object* v_v_3247_ = _args[1];
lean_object* v_R_3248_ = _args[2];
lean_object* v_A_3249_ = _args[3];
lean_object* v_sR_3250_ = _args[4];
lean_object* v_sA_3251_ = _args[5];
lean_object* v_sAlg_3252_ = _args[6];
lean_object* v_cR_3253_ = _args[7];
lean_object* v_fA_3254_ = _args[8];
lean_object* v_za_3255_ = _args[9];
lean_object* v_a_3256_ = _args[10];
lean_object* v_a_3257_ = _args[11];
lean_object* v_a_3258_ = _args[12];
lean_object* v_a_3259_ = _args[13];
lean_object* v_a_3260_ = _args[14];
lean_object* v_a_3261_ = _args[15];
lean_object* v_a_3262_ = _args[16];
_start:
{
lean_object* v_res_3263_; 
v_res_3263_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg(v_u_3246_, v_v_3247_, v_R_3248_, v_A_3249_, v_sR_3250_, v_sA_3251_, v_sAlg_3252_, v_cR_3253_, v_fA_3254_, v_za_3255_, v_a_3256_, v_a_3257_, v_a_3258_, v_a_3259_, v_a_3260_, v_a_3261_);
lean_dec(v_a_3261_);
lean_dec_ref(v_a_3260_);
lean_dec(v_a_3259_);
lean_dec_ref(v_a_3258_);
lean_dec(v_a_3257_);
lean_dec_ref(v_a_3256_);
return v_res_3263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv(lean_object* v_u_3264_, lean_object* v_v_3265_, lean_object* v_R_3266_, lean_object* v_A_3267_, lean_object* v_sR_3268_, lean_object* v_sA_3269_, lean_object* v_sAlg_3270_, lean_object* v_cR_3271_, lean_object* v_a_3272_, lean_object* v_x_3273_, lean_object* v_fA_3274_, lean_object* v_za_3275_, lean_object* v_a_3276_, lean_object* v_a_3277_, lean_object* v_a_3278_, lean_object* v_a_3279_, lean_object* v_a_3280_, lean_object* v_a_3281_){
_start:
{
lean_object* v___x_3283_; 
v___x_3283_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___redArg(v_u_3264_, v_v_3265_, v_R_3266_, v_A_3267_, v_sR_3268_, v_sA_3269_, v_sAlg_3270_, v_cR_3271_, v_fA_3274_, v_za_3275_, v_a_3276_, v_a_3277_, v_a_3278_, v_a_3279_, v_a_3280_, v_a_3281_);
return v___x_3283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___boxed(lean_object** _args){
lean_object* v_u_3284_ = _args[0];
lean_object* v_v_3285_ = _args[1];
lean_object* v_R_3286_ = _args[2];
lean_object* v_A_3287_ = _args[3];
lean_object* v_sR_3288_ = _args[4];
lean_object* v_sA_3289_ = _args[5];
lean_object* v_sAlg_3290_ = _args[6];
lean_object* v_cR_3291_ = _args[7];
lean_object* v_a_3292_ = _args[8];
lean_object* v_x_3293_ = _args[9];
lean_object* v_fA_3294_ = _args[10];
lean_object* v_za_3295_ = _args[11];
lean_object* v_a_3296_ = _args[12];
lean_object* v_a_3297_ = _args[13];
lean_object* v_a_3298_ = _args[14];
lean_object* v_a_3299_ = _args[15];
lean_object* v_a_3300_ = _args[16];
lean_object* v_a_3301_ = _args[17];
lean_object* v_a_3302_ = _args[18];
_start:
{
lean_object* v_res_3303_; 
v_res_3303_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv(v_u_3284_, v_v_3285_, v_R_3286_, v_A_3287_, v_sR_3288_, v_sA_3289_, v_sAlg_3290_, v_cR_3291_, v_a_3292_, v_x_3293_, v_fA_3294_, v_za_3295_, v_a_3296_, v_a_3297_, v_a_3298_, v_a_3299_, v_a_3300_, v_a_3301_);
lean_dec(v_a_3301_);
lean_dec_ref(v_a_3300_);
lean_dec(v_a_3299_);
lean_dec_ref(v_a_3298_);
lean_dec(v_a_3297_);
lean_dec_ref(v_a_3296_);
lean_dec(v_x_3293_);
lean_dec_ref(v_a_3292_);
return v_res_3303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive(lean_object* v_u_3304_, lean_object* v_v_3305_, lean_object* v_R_3306_, lean_object* v_A_3307_, lean_object* v_sR_3308_, lean_object* v_sA_3309_, lean_object* v_sAlg_3310_, lean_object* v_cR_3311_, lean_object* v_cA_3312_, lean_object* v_x_3313_, lean_object* v_a_3314_, lean_object* v_a_3315_, lean_object* v_a_3316_, lean_object* v_a_3317_){
_start:
{
lean_object* v_a_3320_; uint8_t v___x_3332_; lean_object* v___x_3333_; 
v___x_3332_ = 0;
lean_inc_ref(v_x_3313_);
lean_inc_ref(v_A_3307_);
lean_inc(v_v_3305_);
v___x_3333_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_v_3305_, v_A_3307_, v_x_3313_, v___x_3332_, v_a_3314_, v_a_3315_, v_a_3316_, v_a_3317_);
if (lean_obj_tag(v___x_3333_) == 0)
{
lean_object* v_a_3334_; lean_object* v___x_3335_; 
v_a_3334_ = lean_ctor_get(v___x_3333_, 0);
lean_inc(v_a_3334_);
lean_dec_ref_known(v___x_3333_, 1);
v___x_3335_ = lp_mathlib_Mathlib_Tactic_Algebra_evalCast(v_u_3304_, v_v_3305_, v_R_3306_, v_A_3307_, v_sR_3308_, v_sA_3309_, v_sAlg_3310_, v_x_3313_, v_cR_3311_, v_cA_3312_, v_a_3334_);
if (lean_obj_tag(v___x_3335_) == 0)
{
lean_object* v___x_3336_; lean_object* v___x_3337_; 
v___x_3336_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___redArg___closed__1);
v___x_3337_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0___redArg(v___x_3336_, v_a_3314_, v_a_3315_, v_a_3316_, v_a_3317_);
return v___x_3337_;
}
else
{
lean_object* v_val_3338_; 
v_val_3338_ = lean_ctor_get(v___x_3335_, 0);
lean_inc(v_val_3338_);
lean_dec_ref_known(v___x_3335_, 1);
v_a_3320_ = v_val_3338_;
goto v___jp_3319_;
}
}
else
{
lean_object* v_a_3339_; lean_object* v___x_3341_; uint8_t v_isShared_3342_; uint8_t v_isSharedCheck_3346_; 
lean_dec_ref(v_x_3313_);
lean_dec_ref(v_cA_3312_);
lean_dec_ref(v_cR_3311_);
lean_dec_ref(v_sAlg_3310_);
lean_dec_ref(v_sA_3309_);
lean_dec_ref(v_sR_3308_);
lean_dec_ref(v_A_3307_);
lean_dec_ref(v_R_3306_);
lean_dec(v_v_3305_);
lean_dec(v_u_3304_);
v_a_3339_ = lean_ctor_get(v___x_3333_, 0);
v_isSharedCheck_3346_ = !lean_is_exclusive(v___x_3333_);
if (v_isSharedCheck_3346_ == 0)
{
v___x_3341_ = v___x_3333_;
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
else
{
lean_inc(v_a_3339_);
lean_dec(v___x_3333_);
v___x_3341_ = lean_box(0);
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
v_resetjp_3340_:
{
lean_object* v___x_3344_; 
if (v_isShared_3342_ == 0)
{
v___x_3344_ = v___x_3341_;
goto v_reusejp_3343_;
}
else
{
lean_object* v_reuseFailAlloc_3345_; 
v_reuseFailAlloc_3345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3345_, 0, v_a_3339_);
v___x_3344_ = v_reuseFailAlloc_3345_;
goto v_reusejp_3343_;
}
v_reusejp_3343_:
{
return v___x_3344_;
}
}
}
v___jp_3319_:
{
lean_object* v_expr_3321_; lean_object* v_val_3322_; lean_object* v_proof_3323_; lean_object* v___x_3325_; uint8_t v_isShared_3326_; uint8_t v_isSharedCheck_3331_; 
v_expr_3321_ = lean_ctor_get(v_a_3320_, 0);
v_val_3322_ = lean_ctor_get(v_a_3320_, 1);
v_proof_3323_ = lean_ctor_get(v_a_3320_, 2);
v_isSharedCheck_3331_ = !lean_is_exclusive(v_a_3320_);
if (v_isSharedCheck_3331_ == 0)
{
v___x_3325_ = v_a_3320_;
v_isShared_3326_ = v_isSharedCheck_3331_;
goto v_resetjp_3324_;
}
else
{
lean_inc(v_proof_3323_);
lean_inc(v_val_3322_);
lean_inc(v_expr_3321_);
lean_dec(v_a_3320_);
v___x_3325_ = lean_box(0);
v_isShared_3326_ = v_isSharedCheck_3331_;
goto v_resetjp_3324_;
}
v_resetjp_3324_:
{
lean_object* v___x_3328_; 
if (v_isShared_3326_ == 0)
{
v___x_3328_ = v___x_3325_;
goto v_reusejp_3327_;
}
else
{
lean_object* v_reuseFailAlloc_3330_; 
v_reuseFailAlloc_3330_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3330_, 0, v_expr_3321_);
lean_ctor_set(v_reuseFailAlloc_3330_, 1, v_val_3322_);
lean_ctor_set(v_reuseFailAlloc_3330_, 2, v_proof_3323_);
v___x_3328_ = v_reuseFailAlloc_3330_;
goto v_reusejp_3327_;
}
v_reusejp_3327_:
{
lean_object* v___x_3329_; 
v___x_3329_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3329_, 0, v___x_3328_);
return v___x_3329_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive___boxed(lean_object* v_u_3347_, lean_object* v_v_3348_, lean_object* v_R_3349_, lean_object* v_A_3350_, lean_object* v_sR_3351_, lean_object* v_sA_3352_, lean_object* v_sAlg_3353_, lean_object* v_cR_3354_, lean_object* v_cA_3355_, lean_object* v_x_3356_, lean_object* v_a_3357_, lean_object* v_a_3358_, lean_object* v_a_3359_, lean_object* v_a_3360_, lean_object* v_a_3361_){
_start:
{
lean_object* v_res_3362_; 
v_res_3362_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive(v_u_3347_, v_v_3348_, v_R_3349_, v_A_3350_, v_sR_3351_, v_sA_3352_, v_sAlg_3353_, v_cR_3354_, v_cA_3355_, v_x_3356_, v_a_3357_, v_a_3358_, v_a_3359_, v_a_3360_);
lean_dec(v_a_3360_);
lean_dec_ref(v_a_3359_);
lean_dec(v_a_3358_);
lean_dec_ref(v_a_3357_);
return v_res_3362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg(lean_object* v_u_3369_, lean_object* v_v_3370_, lean_object* v_R_3371_, lean_object* v_A_3372_, lean_object* v_sR_3373_, lean_object* v_sA_3374_, lean_object* v_sAlg_3375_, lean_object* v_cR_3376_, lean_object* v_zx_3377_){
_start:
{
lean_object* v_x_3378_; 
v_x_3378_ = lean_ctor_get(v_zx_3377_, 1);
lean_inc(v_x_3378_);
lean_dec_ref(v_zx_3377_);
if (lean_obj_tag(v_x_3378_) == 0)
{
lean_object* v___x_3379_; 
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3379_ = lean_box(0);
return v___x_3379_;
}
else
{
lean_object* v_b_3380_; 
v_b_3380_ = lean_ctor_get(v_x_3378_, 1);
lean_inc_ref(v_b_3380_);
if (lean_obj_tag(v_b_3380_) == 5)
{
lean_object* v_fn_3381_; 
v_fn_3381_ = lean_ctor_get(v_b_3380_, 0);
lean_inc_ref(v_fn_3381_);
if (lean_obj_tag(v_fn_3381_) == 5)
{
lean_object* v_fn_3382_; 
v_fn_3382_ = lean_ctor_get(v_fn_3381_, 0);
if (lean_obj_tag(v_fn_3382_) == 5)
{
lean_object* v_fn_3383_; 
v_fn_3383_ = lean_ctor_get(v_fn_3382_, 0);
lean_inc_ref(v_fn_3383_);
if (lean_obj_tag(v_fn_3383_) == 4)
{
lean_object* v_declName_3384_; 
v_declName_3384_ = lean_ctor_get(v_fn_3383_, 0);
lean_inc(v_declName_3384_);
if (lean_obj_tag(v_declName_3384_) == 1)
{
lean_object* v_pre_3385_; 
v_pre_3385_ = lean_ctor_get(v_declName_3384_, 0);
lean_inc(v_pre_3385_);
if (lean_obj_tag(v_pre_3385_) == 1)
{
lean_object* v_pre_3386_; 
v_pre_3386_ = lean_ctor_get(v_pre_3385_, 0);
if (lean_obj_tag(v_pre_3386_) == 0)
{
lean_object* v_a_3387_; lean_object* v_a_3388_; lean_object* v_a_3389_; lean_object* v_arg_3390_; lean_object* v_arg_3391_; lean_object* v_us_3392_; lean_object* v_str_3393_; lean_object* v_str_3394_; lean_object* v___x_3395_; uint8_t v___x_3396_; 
v_a_3387_ = lean_ctor_get(v_x_3378_, 0);
lean_inc_ref(v_a_3387_);
v_a_3388_ = lean_ctor_get(v_x_3378_, 2);
lean_inc_ref(v_a_3388_);
v_a_3389_ = lean_ctor_get(v_x_3378_, 3);
lean_inc(v_a_3389_);
lean_dec_ref_known(v_x_3378_, 4);
v_arg_3390_ = lean_ctor_get(v_b_3380_, 1);
lean_inc_ref(v_arg_3390_);
lean_dec_ref_known(v_b_3380_, 2);
v_arg_3391_ = lean_ctor_get(v_fn_3381_, 1);
lean_inc_ref(v_arg_3391_);
lean_dec_ref_known(v_fn_3381_, 2);
v_us_3392_ = lean_ctor_get(v_fn_3383_, 1);
lean_inc(v_us_3392_);
lean_dec_ref_known(v_fn_3383_, 2);
v_str_3393_ = lean_ctor_get(v_declName_3384_, 1);
lean_inc_ref(v_str_3393_);
lean_dec_ref_known(v_declName_3384_, 2);
v_str_3394_ = lean_ctor_get(v_pre_3385_, 1);
lean_inc_ref(v_str_3394_);
lean_dec_ref_known(v_pre_3385_, 2);
v___x_3395_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__28));
v___x_3396_ = lean_string_dec_eq(v_str_3394_, v___x_3395_);
lean_dec_ref(v_str_3394_);
if (v___x_3396_ == 0)
{
lean_object* v___x_3397_; 
lean_dec_ref(v_str_3393_);
lean_dec(v_us_3392_);
lean_dec_ref(v_arg_3391_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3397_ = lean_box(0);
return v___x_3397_;
}
else
{
lean_object* v___x_3398_; uint8_t v___x_3399_; 
v___x_3398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__29));
v___x_3399_ = lean_string_dec_eq(v_str_3393_, v___x_3398_);
lean_dec_ref(v_str_3393_);
if (v___x_3399_ == 0)
{
lean_object* v___x_3400_; 
lean_dec(v_us_3392_);
lean_dec_ref(v_arg_3391_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3400_ = lean_box(0);
return v___x_3400_;
}
else
{
if (lean_obj_tag(v_us_3392_) == 1)
{
lean_object* v_tail_3401_; 
v_tail_3401_ = lean_ctor_get(v_us_3392_, 1);
lean_inc(v_tail_3401_);
lean_dec_ref_known(v_us_3392_, 2);
if (lean_obj_tag(v_tail_3401_) == 0)
{
if (lean_obj_tag(v_arg_3391_) == 9)
{
lean_object* v_a_3402_; 
v_a_3402_ = lean_ctor_get(v_arg_3391_, 0);
lean_inc_ref(v_a_3402_);
lean_dec_ref_known(v_arg_3391_, 1);
if (lean_obj_tag(v_a_3402_) == 0)
{
lean_object* v_val_3403_; lean_object* v___x_3404_; uint8_t v___x_3405_; 
v_val_3403_ = lean_ctor_get(v_a_3402_, 0);
lean_inc(v_val_3403_);
lean_dec_ref_known(v_a_3402_, 1);
v___x_3404_ = lean_unsigned_to_nat(0u);
v___x_3405_ = lean_nat_dec_eq(v_val_3403_, v___x_3404_);
lean_dec(v_val_3403_);
if (v___x_3405_ == 0)
{
lean_object* v___x_3406_; 
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3406_ = lean_box(0);
return v___x_3406_;
}
else
{
if (lean_obj_tag(v_arg_3390_) == 5)
{
lean_object* v_fn_3407_; 
v_fn_3407_ = lean_ctor_get(v_arg_3390_, 0);
if (lean_obj_tag(v_fn_3407_) == 5)
{
lean_object* v_fn_3408_; 
v_fn_3408_ = lean_ctor_get(v_fn_3407_, 0);
lean_inc_ref(v_fn_3408_);
if (lean_obj_tag(v_fn_3408_) == 4)
{
lean_object* v_declName_3409_; 
v_declName_3409_ = lean_ctor_get(v_fn_3408_, 0);
lean_inc(v_declName_3409_);
if (lean_obj_tag(v_declName_3409_) == 1)
{
lean_object* v_pre_3410_; 
v_pre_3410_ = lean_ctor_get(v_declName_3409_, 0);
lean_inc(v_pre_3410_);
if (lean_obj_tag(v_pre_3410_) == 1)
{
lean_object* v_pre_3411_; 
v_pre_3411_ = lean_ctor_get(v_pre_3410_, 0);
if (lean_obj_tag(v_pre_3411_) == 0)
{
lean_object* v_arg_3412_; lean_object* v_us_3413_; lean_object* v_str_3414_; lean_object* v_str_3415_; lean_object* v___x_3416_; uint8_t v___x_3417_; 
v_arg_3412_ = lean_ctor_get(v_arg_3390_, 1);
lean_inc_ref(v_arg_3412_);
lean_dec_ref_known(v_arg_3390_, 2);
v_us_3413_ = lean_ctor_get(v_fn_3408_, 1);
lean_inc(v_us_3413_);
lean_dec_ref_known(v_fn_3408_, 2);
v_str_3414_ = lean_ctor_get(v_declName_3409_, 1);
lean_inc_ref(v_str_3414_);
lean_dec_ref_known(v_declName_3409_, 2);
v_str_3415_ = lean_ctor_get(v_pre_3410_, 1);
lean_inc_ref(v_str_3415_);
lean_dec_ref_known(v_pre_3410_, 2);
v___x_3416_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__33));
v___x_3417_ = lean_string_dec_eq(v_str_3415_, v___x_3416_);
lean_dec_ref(v_str_3415_);
if (v___x_3417_ == 0)
{
lean_object* v___x_3418_; 
lean_dec_ref(v_str_3414_);
lean_dec(v_us_3413_);
lean_dec_ref(v_arg_3412_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3418_ = lean_box(0);
return v___x_3418_;
}
else
{
lean_object* v___x_3419_; uint8_t v___x_3420_; 
v___x_3419_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__34));
v___x_3420_ = lean_string_dec_eq(v_str_3414_, v___x_3419_);
lean_dec_ref(v_str_3414_);
if (v___x_3420_ == 0)
{
lean_object* v___x_3421_; 
lean_dec(v_us_3413_);
lean_dec_ref(v_arg_3412_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3421_ = lean_box(0);
return v___x_3421_;
}
else
{
if (lean_obj_tag(v_us_3413_) == 1)
{
lean_object* v_tail_3422_; 
v_tail_3422_ = lean_ctor_get(v_us_3413_, 1);
lean_inc(v_tail_3422_);
lean_dec_ref_known(v_us_3413_, 2);
if (lean_obj_tag(v_tail_3422_) == 0)
{
if (lean_obj_tag(v_arg_3412_) == 5)
{
lean_object* v_fn_3423_; 
v_fn_3423_ = lean_ctor_get(v_arg_3412_, 0);
if (lean_obj_tag(v_fn_3423_) == 5)
{
lean_object* v_fn_3424_; 
v_fn_3424_ = lean_ctor_get(v_fn_3423_, 0);
lean_inc_ref(v_fn_3424_);
if (lean_obj_tag(v_fn_3424_) == 4)
{
lean_object* v_declName_3425_; 
v_declName_3425_ = lean_ctor_get(v_fn_3424_, 0);
lean_inc(v_declName_3425_);
if (lean_obj_tag(v_declName_3425_) == 1)
{
lean_object* v_pre_3426_; 
v_pre_3426_ = lean_ctor_get(v_declName_3425_, 0);
lean_inc(v_pre_3426_);
if (lean_obj_tag(v_pre_3426_) == 1)
{
lean_object* v_pre_3427_; 
v_pre_3427_ = lean_ctor_get(v_pre_3426_, 0);
if (lean_obj_tag(v_pre_3427_) == 0)
{
lean_object* v_arg_3428_; lean_object* v_us_3429_; lean_object* v_str_3430_; lean_object* v_str_3431_; lean_object* v___x_3432_; uint8_t v___x_3433_; 
v_arg_3428_ = lean_ctor_get(v_arg_3412_, 1);
lean_inc_ref(v_arg_3428_);
lean_dec_ref_known(v_arg_3412_, 2);
v_us_3429_ = lean_ctor_get(v_fn_3424_, 1);
lean_inc(v_us_3429_);
lean_dec_ref_known(v_fn_3424_, 2);
v_str_3430_ = lean_ctor_get(v_declName_3425_, 1);
lean_inc_ref(v_str_3430_);
lean_dec_ref_known(v_declName_3425_, 2);
v_str_3431_ = lean_ctor_get(v_pre_3426_, 1);
lean_inc_ref(v_str_3431_);
lean_dec_ref_known(v_pre_3426_, 2);
v___x_3432_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__36));
v___x_3433_ = lean_string_dec_eq(v_str_3431_, v___x_3432_);
lean_dec_ref(v_str_3431_);
if (v___x_3433_ == 0)
{
lean_object* v___x_3434_; 
lean_dec_ref(v_str_3430_);
lean_dec(v_us_3429_);
lean_dec_ref(v_arg_3428_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3434_ = lean_box(0);
return v___x_3434_;
}
else
{
lean_object* v___x_3435_; uint8_t v___x_3436_; 
v___x_3435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__37));
v___x_3436_ = lean_string_dec_eq(v_str_3430_, v___x_3435_);
lean_dec_ref(v_str_3430_);
if (v___x_3436_ == 0)
{
lean_object* v___x_3437_; 
lean_dec(v_us_3429_);
lean_dec_ref(v_arg_3428_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3437_ = lean_box(0);
return v___x_3437_;
}
else
{
if (lean_obj_tag(v_us_3429_) == 1)
{
lean_object* v_tail_3438_; 
v_tail_3438_ = lean_ctor_get(v_us_3429_, 1);
lean_inc(v_tail_3438_);
lean_dec_ref_known(v_us_3429_, 2);
if (lean_obj_tag(v_tail_3438_) == 0)
{
if (lean_obj_tag(v_arg_3428_) == 5)
{
lean_object* v_fn_3439_; 
v_fn_3439_ = lean_ctor_get(v_arg_3428_, 0);
if (lean_obj_tag(v_fn_3439_) == 5)
{
lean_object* v_fn_3440_; 
v_fn_3440_ = lean_ctor_get(v_fn_3439_, 0);
lean_inc_ref(v_fn_3440_);
if (lean_obj_tag(v_fn_3440_) == 4)
{
lean_object* v_declName_3441_; 
v_declName_3441_ = lean_ctor_get(v_fn_3440_, 0);
lean_inc(v_declName_3441_);
if (lean_obj_tag(v_declName_3441_) == 1)
{
lean_object* v_pre_3442_; 
v_pre_3442_ = lean_ctor_get(v_declName_3441_, 0);
if (lean_obj_tag(v_pre_3442_) == 0)
{
lean_object* v_arg_3443_; lean_object* v_us_3444_; lean_object* v_str_3445_; lean_object* v___x_3446_; uint8_t v___x_3447_; 
v_arg_3443_ = lean_ctor_get(v_arg_3428_, 1);
lean_inc_ref(v_arg_3443_);
lean_dec_ref_known(v_arg_3428_, 2);
v_us_3444_ = lean_ctor_get(v_fn_3440_, 1);
lean_inc(v_us_3444_);
lean_dec_ref_known(v_fn_3440_, 2);
v_str_3445_ = lean_ctor_get(v_declName_3441_, 1);
lean_inc_ref(v_str_3445_);
lean_dec_ref_known(v_declName_3441_, 2);
v___x_3446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__39));
v___x_3447_ = lean_string_dec_eq(v_str_3445_, v___x_3446_);
lean_dec_ref(v_str_3445_);
if (v___x_3447_ == 0)
{
lean_object* v___x_3448_; 
lean_dec(v_us_3444_);
lean_dec_ref(v_arg_3443_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3448_ = lean_box(0);
return v___x_3448_;
}
else
{
if (lean_obj_tag(v_us_3444_) == 1)
{
lean_object* v_tail_3449_; lean_object* v___x_3451_; uint8_t v_isShared_3452_; uint8_t v_isSharedCheck_3511_; 
v_tail_3449_ = lean_ctor_get(v_us_3444_, 1);
v_isSharedCheck_3511_ = !lean_is_exclusive(v_us_3444_);
if (v_isSharedCheck_3511_ == 0)
{
lean_object* v_unused_3512_; 
v_unused_3512_ = lean_ctor_get(v_us_3444_, 0);
lean_dec(v_unused_3512_);
v___x_3451_ = v_us_3444_;
v_isShared_3452_ = v_isSharedCheck_3511_;
goto v_resetjp_3450_;
}
else
{
lean_inc(v_tail_3449_);
lean_dec(v_us_3444_);
v___x_3451_ = lean_box(0);
v_isShared_3452_ = v_isSharedCheck_3511_;
goto v_resetjp_3450_;
}
v_resetjp_3450_:
{
if (lean_obj_tag(v_tail_3449_) == 0)
{
if (lean_obj_tag(v_arg_3443_) == 5)
{
lean_object* v_fn_3453_; 
v_fn_3453_ = lean_ctor_get(v_arg_3443_, 0);
lean_inc_ref(v_fn_3453_);
lean_dec_ref_known(v_arg_3443_, 2);
if (lean_obj_tag(v_fn_3453_) == 5)
{
lean_object* v_fn_3454_; 
v_fn_3454_ = lean_ctor_get(v_fn_3453_, 0);
lean_inc_ref(v_fn_3454_);
lean_dec_ref_known(v_fn_3453_, 2);
if (lean_obj_tag(v_fn_3454_) == 4)
{
lean_object* v_declName_3455_; 
v_declName_3455_ = lean_ctor_get(v_fn_3454_, 0);
lean_inc(v_declName_3455_);
if (lean_obj_tag(v_declName_3455_) == 1)
{
lean_object* v_pre_3456_; 
v_pre_3456_ = lean_ctor_get(v_declName_3455_, 0);
lean_inc(v_pre_3456_);
if (lean_obj_tag(v_pre_3456_) == 1)
{
lean_object* v_pre_3457_; 
v_pre_3457_ = lean_ctor_get(v_pre_3456_, 0);
if (lean_obj_tag(v_pre_3457_) == 0)
{
lean_object* v_us_3458_; lean_object* v_str_3459_; lean_object* v_str_3460_; lean_object* v___x_3461_; uint8_t v___x_3462_; 
v_us_3458_ = lean_ctor_get(v_fn_3454_, 1);
lean_inc(v_us_3458_);
lean_dec_ref_known(v_fn_3454_, 2);
v_str_3459_ = lean_ctor_get(v_declName_3455_, 1);
lean_inc_ref(v_str_3459_);
lean_dec_ref_known(v_declName_3455_, 2);
v_str_3460_ = lean_ctor_get(v_pre_3456_, 1);
lean_inc_ref(v_str_3460_);
lean_dec_ref_known(v_pre_3456_, 2);
v___x_3461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__10));
v___x_3462_ = lean_string_dec_eq(v_str_3460_, v___x_3461_);
lean_dec_ref(v_str_3460_);
if (v___x_3462_ == 0)
{
lean_object* v___x_3463_; 
lean_dec_ref(v_str_3459_);
lean_dec(v_us_3458_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3463_ = lean_box(0);
return v___x_3463_;
}
else
{
lean_object* v___x_3464_; uint8_t v___x_3465_; 
v___x_3464_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__11));
v___x_3465_ = lean_string_dec_eq(v_str_3459_, v___x_3464_);
lean_dec_ref(v_str_3459_);
if (v___x_3465_ == 0)
{
lean_object* v___x_3466_; 
lean_dec(v_us_3458_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3466_ = lean_box(0);
return v___x_3466_;
}
else
{
if (lean_obj_tag(v_us_3458_) == 1)
{
lean_object* v_tail_3467_; lean_object* v___x_3469_; uint8_t v_isShared_3470_; uint8_t v_isSharedCheck_3501_; 
v_tail_3467_ = lean_ctor_get(v_us_3458_, 1);
v_isSharedCheck_3501_ = !lean_is_exclusive(v_us_3458_);
if (v_isSharedCheck_3501_ == 0)
{
lean_object* v_unused_3502_; 
v_unused_3502_ = lean_ctor_get(v_us_3458_, 0);
lean_dec(v_unused_3502_);
v___x_3469_ = v_us_3458_;
v_isShared_3470_ = v_isSharedCheck_3501_;
goto v_resetjp_3468_;
}
else
{
lean_inc(v_tail_3467_);
lean_dec(v_us_3458_);
v___x_3469_ = lean_box(0);
v_isShared_3470_ = v_isSharedCheck_3501_;
goto v_resetjp_3468_;
}
v_resetjp_3468_:
{
if (lean_obj_tag(v_tail_3467_) == 0)
{
if (lean_obj_tag(v_a_3388_) == 0)
{
if (lean_obj_tag(v_a_3389_) == 0)
{
lean_object* v_value_3471_; lean_object* v___x_3472_; lean_object* v_isOne_3473_; lean_object* v___x_3474_; 
v_value_3471_ = lean_ctor_get(v_a_3388_, 1);
lean_inc(v_value_3471_);
lean_dec_ref_known(v_a_3388_, 2);
lean_inc_ref(v_sR_3373_);
lean_inc_ref(v_R_3371_);
lean_inc(v_u_3369_);
v___x_3472_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_u_3369_, v_R_3371_, v_sR_3373_, v_cR_3376_);
v_isOne_3473_ = lean_ctor_get(v___x_3472_, 8);
lean_inc_ref(v_isOne_3473_);
lean_dec_ref(v___x_3472_);
lean_inc_ref(v_a_3387_);
v___x_3474_ = lean_apply_2(v_isOne_3473_, v_a_3387_, v_value_3471_);
if (lean_obj_tag(v___x_3474_) == 0)
{
lean_del_object(v___x_3469_);
lean_del_object(v___x_3451_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
return v___x_3474_;
}
else
{
lean_object* v_val_3475_; lean_object* v___x_3477_; uint8_t v_isShared_3478_; uint8_t v_isSharedCheck_3497_; 
v_val_3475_ = lean_ctor_get(v___x_3474_, 0);
v_isSharedCheck_3497_ = !lean_is_exclusive(v___x_3474_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3477_ = v___x_3474_;
v_isShared_3478_ = v_isSharedCheck_3497_;
goto v_resetjp_3476_;
}
else
{
lean_inc(v_val_3475_);
lean_dec(v___x_3474_);
v___x_3477_ = lean_box(0);
v_isShared_3478_ = v_isSharedCheck_3497_;
goto v_resetjp_3476_;
}
v_resetjp_3476_:
{
lean_object* v___x_3480_; 
if (v_isShared_3470_ == 0)
{
lean_ctor_set(v___x_3469_, 0, v_v_3370_);
v___x_3480_ = v___x_3469_;
goto v_reusejp_3479_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v_v_3370_);
lean_ctor_set(v_reuseFailAlloc_3496_, 1, v_tail_3467_);
v___x_3480_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3479_;
}
v_reusejp_3479_:
{
lean_object* v___x_3482_; 
if (v_isShared_3452_ == 0)
{
lean_ctor_set(v___x_3451_, 1, v___x_3480_);
lean_ctor_set(v___x_3451_, 0, v_u_3369_);
v___x_3482_ = v___x_3451_;
goto v_reusejp_3481_;
}
else
{
lean_object* v_reuseFailAlloc_3495_; 
v_reuseFailAlloc_3495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3495_, 0, v_u_3369_);
lean_ctor_set(v_reuseFailAlloc_3495_, 1, v___x_3480_);
v___x_3482_ = v_reuseFailAlloc_3495_;
goto v_reusejp_3481_;
}
v_reusejp_3481_:
{
lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; lean_object* v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v___x_3490_; lean_object* v___x_3491_; lean_object* v___x_3493_; 
v___x_3483_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg___closed__1));
v___x_3484_ = l_Lean_Expr_const___override(v___x_3483_, v___x_3482_);
v___x_3485_ = l_Lean_Expr_app___override(v___x_3484_, v_R_3371_);
v___x_3486_ = l_Lean_Expr_app___override(v___x_3485_, v_A_3372_);
v___x_3487_ = l_Lean_Expr_app___override(v___x_3486_, v_sR_3373_);
v___x_3488_ = l_Lean_Expr_app___override(v___x_3487_, v_sA_3374_);
v___x_3489_ = l_Lean_Expr_app___override(v___x_3488_, v_sAlg_3375_);
v___x_3490_ = l_Lean_Expr_app___override(v___x_3489_, v_a_3387_);
v___x_3491_ = l_Lean_Expr_app___override(v___x_3490_, v_val_3475_);
if (v_isShared_3478_ == 0)
{
lean_ctor_set(v___x_3477_, 0, v___x_3491_);
v___x_3493_ = v___x_3477_;
goto v_reusejp_3492_;
}
else
{
lean_object* v_reuseFailAlloc_3494_; 
v_reuseFailAlloc_3494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3494_, 0, v___x_3491_);
v___x_3493_ = v_reuseFailAlloc_3494_;
goto v_reusejp_3492_;
}
v_reusejp_3492_:
{
return v___x_3493_;
}
}
}
}
}
}
else
{
lean_object* v___x_3498_; 
lean_dec_ref_known(v_a_3388_, 2);
lean_del_object(v___x_3469_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3498_ = lean_box(0);
return v___x_3498_;
}
}
else
{
lean_object* v___x_3499_; 
lean_del_object(v___x_3469_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3499_ = lean_box(0);
return v___x_3499_;
}
}
else
{
lean_object* v___x_3500_; 
lean_del_object(v___x_3469_);
lean_dec(v_tail_3467_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3500_ = lean_box(0);
return v___x_3500_;
}
}
}
else
{
lean_object* v___x_3503_; 
lean_dec(v_us_3458_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3503_ = lean_box(0);
return v___x_3503_;
}
}
}
}
else
{
lean_object* v___x_3504_; 
lean_dec_ref_known(v_pre_3456_, 2);
lean_dec_ref_known(v_declName_3455_, 2);
lean_dec_ref_known(v_fn_3454_, 2);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3504_ = lean_box(0);
return v___x_3504_;
}
}
else
{
lean_object* v___x_3505_; 
lean_dec(v_pre_3456_);
lean_dec_ref_known(v_declName_3455_, 2);
lean_dec_ref_known(v_fn_3454_, 2);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3505_ = lean_box(0);
return v___x_3505_;
}
}
else
{
lean_object* v___x_3506_; 
lean_dec_ref_known(v_fn_3454_, 2);
lean_dec(v_declName_3455_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3506_ = lean_box(0);
return v___x_3506_;
}
}
else
{
lean_object* v___x_3507_; 
lean_dec_ref(v_fn_3454_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3507_ = lean_box(0);
return v___x_3507_;
}
}
else
{
lean_object* v___x_3508_; 
lean_dec_ref(v_fn_3453_);
lean_del_object(v___x_3451_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3508_ = lean_box(0);
return v___x_3508_;
}
}
else
{
lean_object* v___x_3509_; 
lean_del_object(v___x_3451_);
lean_dec_ref(v_arg_3443_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3509_ = lean_box(0);
return v___x_3509_;
}
}
else
{
lean_object* v___x_3510_; 
lean_del_object(v___x_3451_);
lean_dec(v_tail_3449_);
lean_dec_ref(v_arg_3443_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3510_ = lean_box(0);
return v___x_3510_;
}
}
}
else
{
lean_object* v___x_3513_; 
lean_dec(v_us_3444_);
lean_dec_ref(v_arg_3443_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3513_ = lean_box(0);
return v___x_3513_;
}
}
}
else
{
lean_object* v___x_3514_; 
lean_dec_ref_known(v_declName_3441_, 2);
lean_dec_ref_known(v_fn_3440_, 2);
lean_dec_ref_known(v_arg_3428_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3514_ = lean_box(0);
return v___x_3514_;
}
}
else
{
lean_object* v___x_3515_; 
lean_dec_ref_known(v_fn_3440_, 2);
lean_dec(v_declName_3441_);
lean_dec_ref_known(v_arg_3428_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3515_ = lean_box(0);
return v___x_3515_;
}
}
else
{
lean_object* v___x_3516_; 
lean_dec_ref(v_fn_3440_);
lean_dec_ref_known(v_arg_3428_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3516_ = lean_box(0);
return v___x_3516_;
}
}
else
{
lean_object* v___x_3517_; 
lean_dec_ref_known(v_arg_3428_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3517_ = lean_box(0);
return v___x_3517_;
}
}
else
{
lean_object* v___x_3518_; 
lean_dec_ref(v_arg_3428_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3518_ = lean_box(0);
return v___x_3518_;
}
}
else
{
lean_object* v___x_3519_; 
lean_dec(v_tail_3438_);
lean_dec_ref(v_arg_3428_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3519_ = lean_box(0);
return v___x_3519_;
}
}
else
{
lean_object* v___x_3520_; 
lean_dec(v_us_3429_);
lean_dec_ref(v_arg_3428_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3520_ = lean_box(0);
return v___x_3520_;
}
}
}
}
else
{
lean_object* v___x_3521_; 
lean_dec_ref_known(v_pre_3426_, 2);
lean_dec_ref_known(v_declName_3425_, 2);
lean_dec_ref_known(v_fn_3424_, 2);
lean_dec_ref_known(v_arg_3412_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3521_ = lean_box(0);
return v___x_3521_;
}
}
else
{
lean_object* v___x_3522_; 
lean_dec_ref_known(v_declName_3425_, 2);
lean_dec(v_pre_3426_);
lean_dec_ref_known(v_fn_3424_, 2);
lean_dec_ref_known(v_arg_3412_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3522_ = lean_box(0);
return v___x_3522_;
}
}
else
{
lean_object* v___x_3523_; 
lean_dec(v_declName_3425_);
lean_dec_ref_known(v_fn_3424_, 2);
lean_dec_ref_known(v_arg_3412_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3523_ = lean_box(0);
return v___x_3523_;
}
}
else
{
lean_object* v___x_3524_; 
lean_dec_ref(v_fn_3424_);
lean_dec_ref_known(v_arg_3412_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3524_ = lean_box(0);
return v___x_3524_;
}
}
else
{
lean_object* v___x_3525_; 
lean_dec_ref_known(v_arg_3412_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3525_ = lean_box(0);
return v___x_3525_;
}
}
else
{
lean_object* v___x_3526_; 
lean_dec_ref(v_arg_3412_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3526_ = lean_box(0);
return v___x_3526_;
}
}
else
{
lean_object* v___x_3527_; 
lean_dec(v_tail_3422_);
lean_dec_ref(v_arg_3412_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3527_ = lean_box(0);
return v___x_3527_;
}
}
else
{
lean_object* v___x_3528_; 
lean_dec(v_us_3413_);
lean_dec_ref(v_arg_3412_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3528_ = lean_box(0);
return v___x_3528_;
}
}
}
}
else
{
lean_object* v___x_3529_; 
lean_dec_ref_known(v_pre_3410_, 2);
lean_dec_ref_known(v_declName_3409_, 2);
lean_dec_ref_known(v_fn_3408_, 2);
lean_dec_ref_known(v_arg_3390_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3529_ = lean_box(0);
return v___x_3529_;
}
}
else
{
lean_object* v___x_3530_; 
lean_dec(v_pre_3410_);
lean_dec_ref_known(v_declName_3409_, 2);
lean_dec_ref_known(v_fn_3408_, 2);
lean_dec_ref_known(v_arg_3390_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3530_ = lean_box(0);
return v___x_3530_;
}
}
else
{
lean_object* v___x_3531_; 
lean_dec(v_declName_3409_);
lean_dec_ref_known(v_fn_3408_, 2);
lean_dec_ref_known(v_arg_3390_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3531_ = lean_box(0);
return v___x_3531_;
}
}
else
{
lean_object* v___x_3532_; 
lean_dec_ref(v_fn_3408_);
lean_dec_ref_known(v_arg_3390_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3532_ = lean_box(0);
return v___x_3532_;
}
}
else
{
lean_object* v___x_3533_; 
lean_dec_ref_known(v_arg_3390_, 2);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3533_ = lean_box(0);
return v___x_3533_;
}
}
else
{
lean_object* v___x_3534_; 
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3534_ = lean_box(0);
return v___x_3534_;
}
}
}
else
{
lean_object* v___x_3535_; 
lean_dec_ref(v_a_3402_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3535_ = lean_box(0);
return v___x_3535_;
}
}
else
{
lean_object* v___x_3536_; 
lean_dec_ref(v_arg_3391_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3536_ = lean_box(0);
return v___x_3536_;
}
}
else
{
lean_object* v___x_3537_; 
lean_dec(v_tail_3401_);
lean_dec_ref(v_arg_3391_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3537_ = lean_box(0);
return v___x_3537_;
}
}
else
{
lean_object* v___x_3538_; 
lean_dec(v_us_3392_);
lean_dec_ref(v_arg_3391_);
lean_dec_ref(v_arg_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
lean_dec_ref(v_a_3387_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3538_ = lean_box(0);
return v___x_3538_;
}
}
}
}
else
{
lean_object* v___x_3539_; 
lean_dec_ref_known(v_pre_3385_, 2);
lean_dec_ref_known(v_declName_3384_, 2);
lean_dec_ref_known(v_fn_3383_, 2);
lean_dec_ref_known(v_fn_3381_, 2);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3539_ = lean_box(0);
return v___x_3539_;
}
}
else
{
lean_object* v___x_3540_; 
lean_dec_ref_known(v_declName_3384_, 2);
lean_dec(v_pre_3385_);
lean_dec_ref_known(v_fn_3383_, 2);
lean_dec_ref_known(v_fn_3381_, 2);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3540_ = lean_box(0);
return v___x_3540_;
}
}
else
{
lean_object* v___x_3541_; 
lean_dec_ref_known(v_fn_3383_, 2);
lean_dec(v_declName_3384_);
lean_dec_ref_known(v_fn_3381_, 2);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3541_ = lean_box(0);
return v___x_3541_;
}
}
else
{
lean_object* v___x_3542_; 
lean_dec_ref(v_fn_3383_);
lean_dec_ref_known(v_fn_3381_, 2);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3542_ = lean_box(0);
return v___x_3542_;
}
}
else
{
lean_object* v___x_3543_; 
lean_dec_ref_known(v_fn_3381_, 2);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3543_ = lean_box(0);
return v___x_3543_;
}
}
else
{
lean_object* v___x_3544_; 
lean_dec_ref(v_fn_3381_);
lean_dec_ref_known(v_b_3380_, 2);
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3544_ = lean_box(0);
return v___x_3544_;
}
}
else
{
lean_object* v___x_3545_; 
lean_dec_ref_known(v_x_3378_, 4);
lean_dec_ref(v_b_3380_);
lean_dec_ref(v_cR_3376_);
lean_dec_ref(v_sAlg_3375_);
lean_dec_ref(v_sA_3374_);
lean_dec_ref(v_sR_3373_);
lean_dec_ref(v_A_3372_);
lean_dec_ref(v_R_3371_);
lean_dec(v_v_3370_);
lean_dec(v_u_3369_);
v___x_3545_ = lean_box(0);
return v___x_3545_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne(lean_object* v_u_3546_, lean_object* v_v_3547_, lean_object* v_R_3548_, lean_object* v_A_3549_, lean_object* v_sR_3550_, lean_object* v_sA_3551_, lean_object* v_sAlg_3552_, lean_object* v_cR_3553_, lean_object* v_x_3554_, lean_object* v_zx_3555_){
_start:
{
lean_object* v___x_3556_; 
v___x_3556_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___redArg(v_u_3546_, v_v_3547_, v_R_3548_, v_A_3549_, v_sR_3550_, v_sA_3551_, v_sAlg_3552_, v_cR_3553_, v_zx_3555_);
return v___x_3556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___boxed(lean_object* v_u_3557_, lean_object* v_v_3558_, lean_object* v_R_3559_, lean_object* v_A_3560_, lean_object* v_sR_3561_, lean_object* v_sA_3562_, lean_object* v_sAlg_3563_, lean_object* v_cR_3564_, lean_object* v_x_3565_, lean_object* v_zx_3566_){
_start:
{
lean_object* v_res_3567_; 
v_res_3567_ = lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne(v_u_3557_, v_v_3558_, v_R_3559_, v_A_3560_, v_sR_3561_, v_sA_3562_, v_sAlg_3563_, v_cR_3564_, v_x_3565_, v_zx_3566_);
lean_dec_ref(v_x_3565_);
return v_res_3567_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0(lean_object* v_u_3568_, lean_object* v_R_3569_, lean_object* v_sR_3570_, lean_object* v_x_3571_, lean_object* v_y_3572_, lean_object* v_x_3573_, lean_object* v_x_3574_){
_start:
{
lean_object* v_x_3575_; lean_object* v_x_3576_; lean_object* v___x_3577_; lean_object* v_toRingCompare_3578_; lean_object* v___x_3579_; uint8_t v___x_3580_; 
v_x_3575_ = lean_ctor_get(v_x_3573_, 1);
lean_inc(v_x_3575_);
lean_dec_ref(v_x_3573_);
v_x_3576_ = lean_ctor_get(v_x_3574_, 1);
lean_inc(v_x_3576_);
lean_dec_ref(v_x_3574_);
v___x_3577_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
v_toRingCompare_3578_ = lean_ctor_get(v___x_3577_, 0);
v___x_3579_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare(v_u_3568_, v_R_3569_);
lean_inc_ref(v_toRingCompare_3578_);
v___x_3580_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_eq___redArg(v_toRingCompare_3578_, v_u_3568_, v_R_3569_, v_sR_3570_, v___x_3579_, v_x_3575_, v_x_3576_);
return v___x_3580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0___boxed(lean_object* v_u_3581_, lean_object* v_R_3582_, lean_object* v_sR_3583_, lean_object* v_x_3584_, lean_object* v_y_3585_, lean_object* v_x_3586_, lean_object* v_x_3587_){
_start:
{
uint8_t v_res_3588_; lean_object* v_r_3589_; 
v_res_3588_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0(v_u_3581_, v_R_3582_, v_sR_3583_, v_x_3584_, v_y_3585_, v_x_3586_, v_x_3587_);
lean_dec_ref(v_y_3585_);
lean_dec_ref(v_x_3584_);
lean_dec_ref(v_sR_3583_);
lean_dec_ref(v_R_3582_);
lean_dec(v_u_3581_);
v_r_3589_ = lean_box(v_res_3588_);
return v_r_3589_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1(lean_object* v_u_3590_, lean_object* v_R_3591_, lean_object* v_sR_3592_, lean_object* v_x_3593_, lean_object* v_y_3594_, lean_object* v_x_3595_, lean_object* v_x_3596_){
_start:
{
lean_object* v_x_3597_; lean_object* v_x_3598_; lean_object* v___x_3599_; lean_object* v___x_3600_; uint8_t v___x_3601_; 
v_x_3597_ = lean_ctor_get(v_x_3595_, 1);
lean_inc(v_x_3597_);
lean_dec_ref(v_x_3595_);
v_x_3598_ = lean_ctor_get(v_x_3596_, 1);
lean_inc(v_x_3598_);
lean_dec_ref(v_x_3596_);
v___x_3599_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
v___x_3600_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare(v_u_3590_, v_R_3591_);
v___x_3601_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_cmp___redArg(v___x_3599_, v_u_3590_, v_R_3591_, v_sR_3592_, v___x_3600_, v_x_3597_, v_x_3598_);
return v___x_3601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1___boxed(lean_object* v_u_3602_, lean_object* v_R_3603_, lean_object* v_sR_3604_, lean_object* v_x_3605_, lean_object* v_y_3606_, lean_object* v_x_3607_, lean_object* v_x_3608_){
_start:
{
uint8_t v_res_3609_; lean_object* v_r_3610_; 
v_res_3609_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1(v_u_3602_, v_R_3603_, v_sR_3604_, v_x_3605_, v_y_3606_, v_x_3607_, v_x_3608_);
lean_dec_ref(v_y_3606_);
lean_dec_ref(v_x_3605_);
lean_dec_ref(v_sR_3604_);
lean_dec_ref(v_R_3603_);
lean_dec(v_u_3602_);
v_r_3610_ = lean_box(v_res_3609_);
return v_r_3610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg(lean_object* v_u_3611_, lean_object* v_R_3612_, lean_object* v_sR_3613_){
_start:
{
lean_object* v___f_3614_; lean_object* v___f_3615_; lean_object* v___x_3616_; 
lean_inc_ref(v_sR_3613_);
lean_inc_ref(v_R_3612_);
lean_inc(v_u_3611_);
v___f_3614_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_3614_, 0, v_u_3611_);
lean_closure_set(v___f_3614_, 1, v_R_3612_);
lean_closure_set(v___f_3614_, 2, v_sR_3613_);
v___f_3615_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg___lam__1___boxed), 7, 3);
lean_closure_set(v___f_3615_, 0, v_u_3611_);
lean_closure_set(v___f_3615_, 1, v_R_3612_);
lean_closure_set(v___f_3615_, 2, v_sR_3613_);
v___x_3616_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3616_, 0, v___f_3614_);
lean_ctor_set(v___x_3616_, 1, v___f_3615_);
return v___x_3616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare(lean_object* v_u_3617_, lean_object* v_v_3618_, lean_object* v_R_3619_, lean_object* v_A_3620_, lean_object* v_sR_3621_, lean_object* v_sA_3622_, lean_object* v_sAlg_3623_){
_start:
{
lean_object* v___x_3624_; 
v___x_3624_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg(v_u_3617_, v_R_3619_, v_sR_3621_);
return v___x_3624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___boxed(lean_object* v_u_3625_, lean_object* v_v_3626_, lean_object* v_R_3627_, lean_object* v_A_3628_, lean_object* v_sR_3629_, lean_object* v_sA_3630_, lean_object* v_sAlg_3631_){
_start:
{
lean_object* v_res_3632_; 
v_res_3632_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompare(v_u_3625_, v_v_3626_, v_R_3627_, v_A_3628_, v_sR_3629_, v_sA_3630_, v_sAlg_3631_);
lean_dec_ref(v_sAlg_3631_);
lean_dec_ref(v_sA_3630_);
lean_dec_ref(v_A_3628_);
lean_dec(v_v_3626_);
return v_res_3632_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9(void){
_start:
{
lean_object* v___x_3649_; lean_object* v___x_3650_; 
v___x_3649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__8));
v___x_3650_ = l_Lean_Expr_lit___override(v___x_3649_);
return v___x_3650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_ringCompute(lean_object* v_u_3727_, lean_object* v_v_3728_, lean_object* v_R_3729_, lean_object* v_A_3730_, lean_object* v_sR_3731_, lean_object* v_sA_3732_, lean_object* v_sAlg_3733_, lean_object* v_cR_3734_, lean_object* v_cA_3735_){
_start:
{
lean_object* v_toCache_3736_; lean_object* v___x_3737_; lean_object* v___x_3738_; lean_object* v___x_3739_; lean_object* v___x_3740_; lean_object* v___x_3741_; lean_object* v___x_3742_; lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v_fst_3748_; lean_object* v_snd_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_4093_; 
v_toCache_3736_ = lean_ctor_get(v_cR_3734_, 0);
lean_inc_ref_n(v_toCache_3736_, 4);
lean_inc_ref_n(v_sR_3731_, 10);
lean_inc_ref_n(v_R_3729_, 10);
lean_inc_n(v_u_3727_, 10);
v___x_3737_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompare___redArg(v_u_3727_, v_R_3729_, v_sR_3731_);
lean_inc_ref_n(v_sAlg_3733_, 8);
lean_inc_ref_n(v_sA_3732_, 8);
lean_inc_ref_n(v_A_3730_, 8);
lean_inc_n(v_v_3728_, 8);
v___x_3738_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_add___boxed), 17, 8);
lean_closure_set(v___x_3738_, 0, v_u_3727_);
lean_closure_set(v___x_3738_, 1, v_v_3728_);
lean_closure_set(v___x_3738_, 2, v_R_3729_);
lean_closure_set(v___x_3738_, 3, v_A_3730_);
lean_closure_set(v___x_3738_, 4, v_sR_3731_);
lean_closure_set(v___x_3738_, 5, v_sA_3732_);
lean_closure_set(v___x_3738_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3738_, 7, v_toCache_3736_);
v___x_3739_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___boxed), 17, 8);
lean_closure_set(v___x_3739_, 0, v_u_3727_);
lean_closure_set(v___x_3739_, 1, v_v_3728_);
lean_closure_set(v___x_3739_, 2, v_R_3729_);
lean_closure_set(v___x_3739_, 3, v_A_3730_);
lean_closure_set(v___x_3739_, 4, v_sR_3731_);
lean_closure_set(v___x_3739_, 5, v_sA_3732_);
lean_closure_set(v___x_3739_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3739_, 7, v_toCache_3736_);
lean_inc_ref_n(v_cR_3734_, 3);
v___x_3740_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_cast___boxed), 20, 8);
lean_closure_set(v___x_3740_, 0, v_u_3727_);
lean_closure_set(v___x_3740_, 1, v_v_3728_);
lean_closure_set(v___x_3740_, 2, v_R_3729_);
lean_closure_set(v___x_3740_, 3, v_A_3730_);
lean_closure_set(v___x_3740_, 4, v_sR_3731_);
lean_closure_set(v___x_3740_, 5, v_sA_3732_);
lean_closure_set(v___x_3740_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3740_, 7, v_cR_3734_);
v___x_3741_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_neg___boxed), 16, 8);
lean_closure_set(v___x_3741_, 0, v_u_3727_);
lean_closure_set(v___x_3741_, 1, v_v_3728_);
lean_closure_set(v___x_3741_, 2, v_R_3729_);
lean_closure_set(v___x_3741_, 3, v_A_3730_);
lean_closure_set(v___x_3741_, 4, v_sR_3731_);
lean_closure_set(v___x_3741_, 5, v_sA_3732_);
lean_closure_set(v___x_3741_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3741_, 7, v_cR_3734_);
v___x_3742_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_pow___boxed), 17, 8);
lean_closure_set(v___x_3742_, 0, v_u_3727_);
lean_closure_set(v___x_3742_, 1, v_v_3728_);
lean_closure_set(v___x_3742_, 2, v_R_3729_);
lean_closure_set(v___x_3742_, 3, v_A_3730_);
lean_closure_set(v___x_3742_, 4, v_sR_3731_);
lean_closure_set(v___x_3742_, 5, v_sA_3732_);
lean_closure_set(v___x_3742_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3742_, 7, v_toCache_3736_);
v___x_3743_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_inv___boxed), 19, 8);
lean_closure_set(v___x_3743_, 0, v_u_3727_);
lean_closure_set(v___x_3743_, 1, v_v_3728_);
lean_closure_set(v___x_3743_, 2, v_R_3729_);
lean_closure_set(v___x_3743_, 3, v_A_3730_);
lean_closure_set(v___x_3743_, 4, v_sR_3731_);
lean_closure_set(v___x_3743_, 5, v_sA_3732_);
lean_closure_set(v___x_3743_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3743_, 7, v_cR_3734_);
v___x_3744_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_derive___boxed), 15, 9);
lean_closure_set(v___x_3744_, 0, v_u_3727_);
lean_closure_set(v___x_3744_, 1, v_v_3728_);
lean_closure_set(v___x_3744_, 2, v_R_3729_);
lean_closure_set(v___x_3744_, 3, v_A_3730_);
lean_closure_set(v___x_3744_, 4, v_sR_3731_);
lean_closure_set(v___x_3744_, 5, v_sA_3732_);
lean_closure_set(v___x_3744_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3744_, 7, v_cR_3734_);
lean_closure_set(v___x_3744_, 8, v_cA_3735_);
v___x_3745_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_isOne___boxed), 10, 8);
lean_closure_set(v___x_3745_, 0, v_u_3727_);
lean_closure_set(v___x_3745_, 1, v_v_3728_);
lean_closure_set(v___x_3745_, 2, v_R_3729_);
lean_closure_set(v___x_3745_, 3, v_A_3730_);
lean_closure_set(v___x_3745_, 4, v_sR_3731_);
lean_closure_set(v___x_3745_, 5, v_sA_3732_);
lean_closure_set(v___x_3745_, 6, v_sAlg_3733_);
lean_closure_set(v___x_3745_, 7, v_toCache_3736_);
v___x_3746_ = lean_unsigned_to_nat(1u);
v___x_3747_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(v_u_3727_, v_R_3729_, v_sR_3731_, v___x_3746_);
v_fst_3748_ = lean_ctor_get(v___x_3747_, 0);
v_snd_3749_ = lean_ctor_get(v___x_3747_, 1);
v_isSharedCheck_4093_ = !lean_is_exclusive(v___x_3747_);
if (v_isSharedCheck_4093_ == 0)
{
v___x_3751_ = v___x_3747_;
v_isShared_3752_ = v_isSharedCheck_4093_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_snd_3749_);
lean_inc(v_fst_3748_);
lean_dec(v___x_3747_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_4093_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v___x_3753_; lean_object* v___x_3754_; lean_object* v___x_3756_; 
v___x_3753_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__2));
v___x_3754_ = lean_box(0);
lean_inc(v_v_3728_);
if (v_isShared_3752_ == 0)
{
lean_ctor_set_tag(v___x_3751_, 1);
lean_ctor_set(v___x_3751_, 1, v___x_3754_);
lean_ctor_set(v___x_3751_, 0, v_v_3728_);
v___x_3756_ = v___x_3751_;
goto v_reusejp_3755_;
}
else
{
lean_object* v_reuseFailAlloc_4092_; 
v_reuseFailAlloc_4092_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4092_, 0, v_v_3728_);
lean_ctor_set(v_reuseFailAlloc_4092_, 1, v___x_3754_);
v___x_3756_ = v_reuseFailAlloc_4092_;
goto v_reusejp_3755_;
}
v_reusejp_3755_:
{
lean_object* v___x_3757_; lean_object* v___x_3758_; lean_object* v___x_3759_; lean_object* v___x_3760_; lean_object* v___x_3761_; lean_object* v___x_3762_; lean_object* v___x_3763_; lean_object* v___x_3764_; lean_object* v___x_3765_; lean_object* v___x_3766_; lean_object* v___x_3767_; lean_object* v___x_3768_; lean_object* v___x_3769_; lean_object* v___x_3770_; lean_object* v___x_3771_; lean_object* v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_3776_; lean_object* v___x_3777_; lean_object* v___x_3778_; lean_object* v___x_3779_; lean_object* v___x_3780_; lean_object* v___x_3781_; lean_object* v___x_3782_; lean_object* v___x_3783_; lean_object* v___x_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___x_3789_; lean_object* v___x_3790_; lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v___x_3793_; lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v___x_3802_; uint8_t v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_3811_; lean_object* v___x_3812_; lean_object* v___x_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; lean_object* v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3819_; lean_object* v___x_3820_; lean_object* v___x_3821_; lean_object* v___x_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___x_3825_; lean_object* v___x_3826_; lean_object* v___x_3827_; lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3830_; lean_object* v___x_3831_; lean_object* v___x_3832_; lean_object* v___x_3833_; lean_object* v___x_3834_; lean_object* v___x_3835_; lean_object* v___x_3836_; lean_object* v___x_3837_; lean_object* v___x_3838_; lean_object* v___x_3839_; lean_object* v___x_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v___x_3844_; lean_object* v___x_3845_; lean_object* v___x_3846_; lean_object* v___x_3847_; lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_3856_; lean_object* v___x_3857_; lean_object* v___x_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3861_; lean_object* v___x_3862_; lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v___x_3865_; lean_object* v___x_3866_; lean_object* v___x_3867_; lean_object* v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; lean_object* v___x_3879_; lean_object* v___x_3880_; lean_object* v___x_3881_; lean_object* v___x_3882_; lean_object* v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; lean_object* v___x_3891_; lean_object* v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; lean_object* v___x_3938_; lean_object* v___x_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_3944_; lean_object* v___x_3945_; lean_object* v___x_3946_; lean_object* v___x_3947_; lean_object* v___x_3948_; lean_object* v___x_3949_; lean_object* v___x_3950_; lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v___x_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; lean_object* v___x_3964_; lean_object* v___x_3965_; lean_object* v___x_3966_; lean_object* v___x_3967_; lean_object* v___x_3968_; lean_object* v___x_3969_; lean_object* v___x_3970_; lean_object* v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; lean_object* v___x_3981_; lean_object* v___x_3982_; lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v___x_3985_; lean_object* v___x_3986_; lean_object* v___x_3987_; lean_object* v___x_3988_; lean_object* v___x_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; lean_object* v___x_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; lean_object* v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; lean_object* v___x_4009_; lean_object* v___x_4010_; lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4014_; lean_object* v___x_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; lean_object* v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v___x_4022_; lean_object* v___x_4023_; lean_object* v___x_4024_; lean_object* v___x_4025_; lean_object* v___x_4026_; lean_object* v___x_4027_; lean_object* v___x_4028_; lean_object* v___x_4029_; lean_object* v___x_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; lean_object* v___x_4051_; lean_object* v___x_4052_; lean_object* v___x_4053_; lean_object* v___x_4054_; lean_object* v___x_4055_; lean_object* v___x_4056_; lean_object* v___x_4057_; lean_object* v___x_4058_; lean_object* v___x_4059_; lean_object* v___x_4060_; lean_object* v___x_4061_; lean_object* v___x_4062_; lean_object* v___x_4063_; lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4066_; lean_object* v___x_4067_; lean_object* v___x_4068_; lean_object* v___x_4069_; lean_object* v___x_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; lean_object* v___x_4073_; lean_object* v___x_4074_; lean_object* v___x_4075_; lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v___x_4078_; lean_object* v___x_4079_; lean_object* v___x_4080_; lean_object* v___x_4081_; lean_object* v___x_4082_; lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v___x_4091_; 
lean_inc_ref_n(v___x_3756_, 12);
v___x_3757_ = l_Lean_Expr_const___override(v___x_3753_, v___x_3756_);
lean_inc_ref_n(v_A_3730_, 27);
v___x_3758_ = l_Lean_Expr_app___override(v___x_3757_, v_A_3730_);
v___x_3759_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__5));
v___x_3760_ = l_Lean_Expr_const___override(v___x_3759_, v___x_3756_);
v___x_3761_ = l_Lean_Expr_app___override(v___x_3760_, v_A_3730_);
v___x_3762_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__7));
v___x_3763_ = l_Lean_Expr_const___override(v___x_3762_, v___x_3756_);
v___x_3764_ = l_Lean_Expr_app___override(v___x_3763_, v_A_3730_);
v___x_3765_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__20));
v___x_3766_ = l_Lean_Expr_const___override(v___x_3765_, v___x_3756_);
v___x_3767_ = l_Lean_Expr_app___override(v___x_3766_, v_A_3730_);
v___x_3768_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_3769_ = l_Lean_Expr_const___override(v___x_3768_, v___x_3756_);
v___x_3770_ = l_Lean_Expr_app___override(v___x_3769_, v_A_3730_);
v___x_3771_ = l_Lean_Expr_app___override(v___x_3770_, v_sA_3732_);
lean_inc_ref(v___x_3771_);
v___x_3772_ = l_Lean_Expr_app___override(v___x_3767_, v___x_3771_);
lean_inc_ref_n(v___x_3772_, 5);
v___x_3773_ = l_Lean_Expr_app___override(v___x_3764_, v___x_3772_);
v___x_3774_ = l_Lean_Expr_app___override(v___x_3761_, v___x_3773_);
lean_inc_ref_n(v___x_3774_, 2);
v___x_3775_ = l_Lean_Expr_app___override(v___x_3758_, v___x_3774_);
v___x_3776_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9, &lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__9);
v___x_3777_ = l_Lean_Expr_app___override(v___x_3775_, v___x_3776_);
v___x_3778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
lean_inc_n(v_u_3727_, 7);
v___x_3779_ = l_Lean_Level_succ___override(v_u_3727_);
lean_inc_n(v_v_3728_, 2);
v___x_3780_ = l_Lean_Level_succ___override(v_v_3728_);
lean_inc_n(v___x_3780_, 3);
lean_inc_n(v___x_3779_, 3);
v___x_3781_ = l_Lean_Level_max___override(v___x_3779_, v___x_3780_);
v___x_3782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3782_, 0, v___x_3780_);
lean_ctor_set(v___x_3782_, 1, v___x_3754_);
lean_inc_ref_n(v___x_3782_, 4);
v___x_3783_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3783_, 0, v___x_3779_);
lean_ctor_set(v___x_3783_, 1, v___x_3782_);
lean_inc_ref(v___x_3783_);
v___x_3784_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3784_, 0, v___x_3781_);
lean_ctor_set(v___x_3784_, 1, v___x_3783_);
v___x_3785_ = l_Lean_Expr_const___override(v___x_3778_, v___x_3784_);
v___x_3786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__17));
v___x_3787_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3787_, 0, v_u_3727_);
lean_ctor_set(v___x_3787_, 1, v___x_3756_);
lean_inc_ref_n(v___x_3787_, 4);
v___x_3788_ = l_Lean_Expr_const___override(v___x_3786_, v___x_3787_);
lean_inc_ref_n(v_R_3729_, 44);
v___x_3789_ = l_Lean_Expr_app___override(v___x_3788_, v_R_3729_);
v___x_3790_ = l_Lean_Expr_app___override(v___x_3789_, v_A_3730_);
v___x_3791_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3791_, 0, v_u_3727_);
lean_ctor_set(v___x_3791_, 1, v___x_3754_);
lean_inc_ref_n(v___x_3791_, 24);
v___x_3792_ = l_Lean_Expr_const___override(v___x_3765_, v___x_3791_);
v___x_3793_ = l_Lean_Expr_app___override(v___x_3792_, v_R_3729_);
v___x_3794_ = l_Lean_Expr_const___override(v___x_3768_, v___x_3791_);
v___x_3795_ = l_Lean_Expr_app___override(v___x_3794_, v_R_3729_);
lean_inc_ref_n(v_sR_3731_, 2);
v___x_3796_ = l_Lean_Expr_app___override(v___x_3795_, v_sR_3731_);
lean_inc_ref_n(v___x_3796_, 2);
v___x_3797_ = l_Lean_Expr_app___override(v___x_3793_, v___x_3796_);
lean_inc_ref_n(v___x_3797_, 5);
v___x_3798_ = l_Lean_Expr_app___override(v___x_3790_, v___x_3797_);
v___x_3799_ = l_Lean_Expr_app___override(v___x_3798_, v___x_3772_);
lean_inc_ref_n(v___x_3799_, 4);
v___x_3800_ = l_Lean_Expr_app___override(v___x_3785_, v___x_3799_);
v___x_3801_ = l_Lean_Expr_app___override(v___x_3800_, v_R_3729_);
v___x_3802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_3803_ = 0;
v___x_3804_ = l_Lean_Expr_lam___override(v___x_3802_, v_R_3729_, v_A_3730_, v___x_3803_);
v___x_3805_ = l_Lean_Expr_app___override(v___x_3801_, v___x_3804_);
v___x_3806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__24));
v___x_3807_ = l_Lean_Expr_const___override(v___x_3806_, v___x_3787_);
v___x_3808_ = l_Lean_Expr_app___override(v___x_3807_, v_R_3729_);
v___x_3809_ = l_Lean_Expr_app___override(v___x_3808_, v_A_3730_);
v___x_3810_ = l_Lean_Expr_app___override(v___x_3809_, v___x_3797_);
v___x_3811_ = l_Lean_Expr_app___override(v___x_3810_, v___x_3772_);
lean_inc_ref_n(v___x_3811_, 4);
v___x_3812_ = l_Lean_Expr_app___override(v___x_3805_, v___x_3811_);
v___x_3813_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_3814_ = l_Lean_Expr_const___override(v___x_3813_, v___x_3787_);
v___x_3815_ = l_Lean_Expr_app___override(v___x_3814_, v_R_3729_);
v___x_3816_ = l_Lean_Expr_app___override(v___x_3815_, v_A_3730_);
v___x_3817_ = l_Lean_Expr_app___override(v___x_3816_, v_sR_3731_);
v___x_3818_ = l_Lean_Expr_app___override(v___x_3817_, v___x_3771_);
v___x_3819_ = l_Lean_Expr_app___override(v___x_3818_, v_sAlg_3733_);
lean_inc_ref(v___x_3819_);
v___x_3820_ = l_Lean_Expr_app___override(v___x_3812_, v___x_3819_);
v___x_3821_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
v___x_3822_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3822_, 0, v_u_3727_);
lean_ctor_set(v___x_3822_, 1, v___x_3791_);
v___x_3823_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3823_, 0, v_u_3727_);
lean_ctor_set(v___x_3823_, 1, v___x_3822_);
v___x_3824_ = l_Lean_Expr_const___override(v___x_3821_, v___x_3823_);
v___x_3825_ = l_Lean_Expr_app___override(v___x_3824_, v_R_3729_);
v___x_3826_ = l_Lean_Expr_app___override(v___x_3825_, v_R_3729_);
v___x_3827_ = l_Lean_Expr_app___override(v___x_3826_, v_R_3729_);
v___x_3828_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__4));
v___x_3829_ = l_Lean_Expr_const___override(v___x_3828_, v___x_3791_);
v___x_3830_ = l_Lean_Expr_app___override(v___x_3829_, v_R_3729_);
v___x_3831_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__7));
v___x_3832_ = l_Lean_Expr_const___override(v___x_3831_, v___x_3791_);
v___x_3833_ = l_Lean_Expr_app___override(v___x_3832_, v_R_3729_);
v___x_3834_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__9));
v___x_3835_ = l_Lean_Expr_const___override(v___x_3834_, v___x_3791_);
v___x_3836_ = l_Lean_Expr_app___override(v___x_3835_, v_R_3729_);
v___x_3837_ = l_Lean_Expr_app___override(v___x_3836_, v___x_3796_);
v___x_3838_ = l_Lean_Expr_app___override(v___x_3833_, v___x_3837_);
v___x_3839_ = l_Lean_Expr_app___override(v___x_3830_, v___x_3838_);
v___x_3840_ = l_Lean_Expr_app___override(v___x_3827_, v___x_3839_);
lean_inc_n(v_fst_3748_, 2);
lean_inc_ref_n(v___x_3840_, 2);
v___x_3841_ = l_Lean_Expr_app___override(v___x_3840_, v_fst_3748_);
v___x_3842_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__30));
v___x_3843_ = l_Lean_Expr_const___override(v___x_3842_, v___x_3791_);
v___x_3844_ = l_Lean_Expr_app___override(v___x_3843_, v_R_3729_);
v___x_3845_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32, &lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__32);
lean_inc_ref(v___x_3844_);
v___x_3846_ = l_Lean_Expr_app___override(v___x_3844_, v___x_3845_);
v___x_3847_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__35));
v___x_3848_ = l_Lean_Expr_const___override(v___x_3847_, v___x_3791_);
v___x_3849_ = l_Lean_Expr_app___override(v___x_3848_, v_R_3729_);
v___x_3850_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__38));
v___x_3851_ = l_Lean_Expr_const___override(v___x_3850_, v___x_3791_);
v___x_3852_ = l_Lean_Expr_app___override(v___x_3851_, v_R_3729_);
v___x_3853_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__40));
v___x_3854_ = l_Lean_Expr_const___override(v___x_3853_, v___x_3791_);
v___x_3855_ = l_Lean_Expr_app___override(v___x_3854_, v_R_3729_);
v___x_3856_ = l_Lean_Expr_app___override(v___x_3855_, v___x_3796_);
v___x_3857_ = l_Lean_Expr_app___override(v___x_3852_, v___x_3856_);
v___x_3858_ = l_Lean_Expr_app___override(v___x_3849_, v___x_3857_);
v___x_3859_ = l_Lean_Expr_app___override(v___x_3846_, v___x_3858_);
lean_inc_ref_n(v___x_3859_, 2);
lean_inc_ref(v___x_3841_);
v___x_3860_ = l_Lean_Expr_app___override(v___x_3841_, v___x_3859_);
lean_inc_ref_n(v___x_3860_, 3);
lean_inc_ref_n(v___x_3820_, 2);
v___x_3861_ = l_Lean_Expr_app___override(v___x_3820_, v___x_3860_);
v___x_3862_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_3727_, v_R_3729_, v_sR_3731_, v_fst_3748_, v_snd_3749_);
v___x_3863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3863_, 0, v___x_3860_);
lean_ctor_set(v___x_3863_, 1, v___x_3862_);
v___x_3864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_3865_ = l_Lean_Expr_const___override(v___x_3864_, v___x_3782_);
v___x_3866_ = l_Lean_Expr_app___override(v___x_3865_, v_A_3730_);
v___x_3867_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__9);
lean_inc_ref(v___x_3777_);
lean_inc_ref_n(v___x_3866_, 2);
v___x_3868_ = l_Lean_Expr_app___override(v___x_3866_, v___x_3777_);
lean_inc_ref_n(v___x_3861_, 3);
lean_inc_ref(v___x_3868_);
v___x_3869_ = l_Lean_Expr_app___override(v___x_3868_, v___x_3861_);
lean_inc_ref(v___x_3869_);
v___x_3870_ = l_Lean_Expr_app___override(v___x_3867_, v___x_3869_);
v___x_3871_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__11));
v___x_3872_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__13);
v___x_3873_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__15);
v___x_3874_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__16);
v___x_3875_ = l_Lean_Expr_app___override(v___x_3874_, v___x_3869_);
v___x_3876_ = l_Lean_Expr_const___override(v___x_3842_, v___x_3756_);
v___x_3877_ = l_Lean_Expr_app___override(v___x_3876_, v_A_3730_);
v___x_3878_ = l_Lean_Expr_app___override(v___x_3877_, v___x_3776_);
v___x_3879_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__12));
v___x_3880_ = l_Lean_Expr_const___override(v___x_3879_, v___x_3756_);
v___x_3881_ = l_Lean_Expr_app___override(v___x_3880_, v_A_3730_);
v___x_3882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__15));
v___x_3883_ = l_Lean_Expr_const___override(v___x_3882_, v___x_3756_);
v___x_3884_ = l_Lean_Expr_app___override(v___x_3883_, v_A_3730_);
v___x_3885_ = l_Lean_Expr_app___override(v___x_3884_, v___x_3774_);
lean_inc_ref(v___x_3885_);
v___x_3886_ = l_Lean_Expr_app___override(v___x_3881_, v___x_3885_);
v___x_3887_ = l_Lean_Expr_app___override(v___x_3878_, v___x_3886_);
lean_inc_ref_n(v___x_3887_, 5);
v___x_3888_ = l_Lean_Expr_app___override(v___x_3866_, v___x_3887_);
lean_inc_ref(v___x_3888_);
v___x_3889_ = l_Lean_Expr_app___override(v___x_3888_, v___x_3887_);
v___x_3890_ = l_Lean_Expr_app___override(v___x_3875_, v___x_3889_);
v___x_3891_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__19);
v___x_3892_ = l_Lean_Expr_app___override(v___x_3890_, v___x_3891_);
v___x_3893_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__17));
v___x_3894_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3894_, 0, v___x_3780_);
lean_ctor_set(v___x_3894_, 1, v___x_3872_);
v___x_3895_ = l_Lean_Expr_const___override(v___x_3893_, v___x_3894_);
v___x_3896_ = l_Lean_Expr_app___override(v___x_3895_, v_A_3730_);
v___x_3897_ = l_Lean_Expr_app___override(v___x_3896_, v___x_3873_);
v___x_3898_ = l_Lean_Expr_app___override(v___x_3897_, v___x_3868_);
v___x_3899_ = l_Lean_Expr_app___override(v___x_3898_, v___x_3888_);
v___x_3900_ = l_Lean_Expr_app___override(v___x_3899_, v___x_3861_);
v___x_3901_ = l_Lean_Expr_app___override(v___x_3900_, v___x_3887_);
v___x_3902_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__21));
v___x_3903_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3903_, 0, v___x_3780_);
lean_ctor_set(v___x_3903_, 1, v___x_3782_);
v___x_3904_ = l_Lean_Expr_const___override(v___x_3902_, v___x_3903_);
v___x_3905_ = l_Lean_Expr_app___override(v___x_3904_, v_A_3730_);
v___x_3906_ = lean_box(0);
v___x_3907_ = l_Lean_Expr_forallE___override(v___x_3906_, v_A_3730_, v___x_3873_, v___x_3803_);
v___x_3908_ = l_Lean_Expr_app___override(v___x_3905_, v___x_3907_);
v___x_3909_ = l_Lean_Expr_app___override(v___x_3908_, v___x_3777_);
v___x_3910_ = l_Lean_Expr_app___override(v___x_3909_, v___x_3887_);
v___x_3911_ = l_Lean_Expr_app___override(v___x_3910_, v___x_3866_);
v___x_3912_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__19));
v___x_3913_ = l_Lean_Expr_const___override(v___x_3912_, v___x_3756_);
v___x_3914_ = l_Lean_Expr_app___override(v___x_3913_, v_A_3730_);
v___x_3915_ = l_Lean_Expr_app___override(v___x_3914_, v___x_3774_);
v___x_3916_ = l_Lean_Expr_app___override(v___x_3911_, v___x_3915_);
v___x_3917_ = l_Lean_Expr_app___override(v___x_3901_, v___x_3916_);
v___x_3918_ = l_Lean_Expr_const___override(v___x_3871_, v___x_3782_);
v___x_3919_ = l_Lean_Expr_app___override(v___x_3918_, v_A_3730_);
v___x_3920_ = l_Lean_Expr_app___override(v___x_3919_, v___x_3861_);
v___x_3921_ = l_Lean_Expr_app___override(v___x_3844_, v___x_3776_);
v___x_3922_ = l_Lean_Expr_const___override(v___x_3879_, v___x_3791_);
v___x_3923_ = l_Lean_Expr_app___override(v___x_3922_, v_R_3729_);
v___x_3924_ = l_Lean_Expr_const___override(v___x_3882_, v___x_3791_);
v___x_3925_ = l_Lean_Expr_app___override(v___x_3924_, v_R_3729_);
v___x_3926_ = l_Lean_Expr_const___override(v___x_3759_, v___x_3791_);
v___x_3927_ = l_Lean_Expr_app___override(v___x_3926_, v_R_3729_);
v___x_3928_ = l_Lean_Expr_const___override(v___x_3762_, v___x_3791_);
v___x_3929_ = l_Lean_Expr_app___override(v___x_3928_, v_R_3729_);
v___x_3930_ = l_Lean_Expr_app___override(v___x_3929_, v___x_3797_);
v___x_3931_ = l_Lean_Expr_app___override(v___x_3927_, v___x_3930_);
lean_inc_ref_n(v___x_3931_, 5);
v___x_3932_ = l_Lean_Expr_app___override(v___x_3925_, v___x_3931_);
lean_inc_ref(v___x_3932_);
v___x_3933_ = l_Lean_Expr_app___override(v___x_3923_, v___x_3932_);
v___x_3934_ = l_Lean_Expr_app___override(v___x_3921_, v___x_3933_);
lean_inc_ref_n(v___x_3934_, 6);
v___x_3935_ = l_Lean_Expr_app___override(v___x_3820_, v___x_3934_);
v___x_3936_ = l_Lean_Expr_app___override(v___x_3920_, v___x_3935_);
v___x_3937_ = l_Lean_Expr_app___override(v___x_3936_, v___x_3887_);
v___x_3938_ = l_Lean_Expr_const___override(v___x_3902_, v___x_3783_);
v___x_3939_ = l_Lean_Expr_app___override(v___x_3938_, v_R_3729_);
v___x_3940_ = l_Lean_Expr_app___override(v___x_3939_, v_A_3730_);
v___x_3941_ = l_Lean_Expr_app___override(v___x_3940_, v___x_3860_);
v___x_3942_ = l_Lean_Expr_app___override(v___x_3941_, v___x_3934_);
v___x_3943_ = l_Lean_Expr_app___override(v___x_3942_, v___x_3820_);
v___x_3944_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3944_, 0, v___x_3779_);
lean_ctor_set(v___x_3944_, 1, v___x_3754_);
lean_inc_ref(v___x_3944_);
v___x_3945_ = l_Lean_Expr_const___override(v___x_3871_, v___x_3944_);
v___x_3946_ = l_Lean_Expr_app___override(v___x_3945_, v_R_3729_);
lean_inc_ref(v___x_3946_);
v___x_3947_ = l_Lean_Expr_app___override(v___x_3946_, v___x_3860_);
v___x_3948_ = l_Lean_Expr_app___override(v___x_3840_, v___x_3934_);
lean_inc_ref(v___x_3948_);
v___x_3949_ = l_Lean_Expr_app___override(v___x_3948_, v___x_3859_);
v___x_3950_ = l_Lean_Expr_app___override(v___x_3947_, v___x_3949_);
v___x_3951_ = l_Lean_Expr_app___override(v___x_3950_, v___x_3934_);
v___x_3952_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__21));
v___x_3953_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3953_, 0, v___x_3779_);
lean_ctor_set(v___x_3953_, 1, v___x_3944_);
lean_inc_ref(v___x_3953_);
v___x_3954_ = l_Lean_Expr_const___override(v___x_3952_, v___x_3953_);
v___x_3955_ = l_Lean_Expr_app___override(v___x_3954_, v_R_3729_);
v___x_3956_ = l_Lean_Expr_app___override(v___x_3955_, v_R_3729_);
v___x_3957_ = l_Lean_Expr_app___override(v___x_3956_, v___x_3841_);
v___x_3958_ = l_Lean_Expr_app___override(v___x_3957_, v___x_3948_);
v___x_3959_ = l_Lean_Expr_const___override(v___x_3902_, v___x_3953_);
v___x_3960_ = l_Lean_Expr_app___override(v___x_3959_, v_R_3729_);
v___x_3961_ = l_Lean_Expr_forallE___override(v___x_3906_, v_R_3729_, v_R_3729_, v___x_3803_);
v___x_3962_ = l_Lean_Expr_app___override(v___x_3960_, v___x_3961_);
v___x_3963_ = l_Lean_Expr_app___override(v___x_3962_, v_fst_3748_);
v___x_3964_ = l_Lean_Expr_app___override(v___x_3963_, v___x_3934_);
v___x_3965_ = l_Lean_Expr_app___override(v___x_3964_, v___x_3840_);
v___x_3966_ = l_Lean_Expr_const___override(v___x_3753_, v___x_3791_);
v___x_3967_ = l_Lean_Expr_app___override(v___x_3966_, v_R_3729_);
v___x_3968_ = l_Lean_Expr_app___override(v___x_3967_, v___x_3931_);
v___x_3969_ = l_Lean_Expr_app___override(v___x_3968_, v___x_3776_);
v___x_3970_ = l_Lean_Expr_app___override(v___x_3946_, v___x_3969_);
v___x_3971_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__23));
v___x_3972_ = l_Lean_Expr_const___override(v___x_3971_, v___x_3791_);
v___x_3973_ = l_Lean_Expr_app___override(v___x_3972_, v_R_3729_);
v___x_3974_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__25));
v___x_3975_ = l_Lean_Expr_const___override(v___x_3974_, v___x_3791_);
v___x_3976_ = l_Lean_Expr_app___override(v___x_3975_, v_R_3729_);
v___x_3977_ = l_Lean_Expr_app___override(v___x_3976_, v___x_3931_);
v___x_3978_ = l_Lean_Expr_app___override(v___x_3973_, v___x_3977_);
v___x_3979_ = l_Lean_Expr_app___override(v___x_3978_, v___x_3776_);
v___x_3980_ = l_Lean_Expr_app___override(v___x_3970_, v___x_3979_);
v___x_3981_ = l_Lean_Expr_app___override(v___x_3980_, v___x_3934_);
v___x_3982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__27));
v___x_3983_ = l_Lean_Expr_const___override(v___x_3982_, v___x_3791_);
v___x_3984_ = l_Lean_Expr_app___override(v___x_3983_, v_R_3729_);
v___x_3985_ = l_Lean_Expr_app___override(v___x_3984_, v___x_3931_);
v___x_3986_ = l_Lean_Expr_app___override(v___x_3985_, v___x_3776_);
v___x_3987_ = l_Lean_Expr_app___override(v___x_3981_, v___x_3986_);
v___x_3988_ = l_Lean_Expr_const___override(v___x_3912_, v___x_3791_);
v___x_3989_ = l_Lean_Expr_app___override(v___x_3988_, v_R_3729_);
v___x_3990_ = l_Lean_Expr_app___override(v___x_3989_, v___x_3931_);
v___x_3991_ = l_Lean_Expr_app___override(v___x_3987_, v___x_3990_);
v___x_3992_ = l_Lean_Expr_app___override(v___x_3965_, v___x_3991_);
v___x_3993_ = l_Lean_Expr_app___override(v___x_3958_, v___x_3992_);
v___x_3994_ = l_Lean_Expr_app___override(v___x_3993_, v___x_3859_);
v___x_3995_ = l_Lean_Expr_app___override(v___x_3951_, v___x_3994_);
v___x_3996_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__29));
v___x_3997_ = l_Lean_Expr_const___override(v___x_3996_, v___x_3791_);
v___x_3998_ = l_Lean_Expr_app___override(v___x_3997_, v_R_3729_);
v___x_3999_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__32));
v___x_4000_ = l_Lean_Expr_const___override(v___x_3999_, v___x_3791_);
v___x_4001_ = l_Lean_Expr_app___override(v___x_4000_, v_R_3729_);
v___x_4002_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__34));
v___x_4003_ = l_Lean_Expr_const___override(v___x_4002_, v___x_3791_);
v___x_4004_ = l_Lean_Expr_app___override(v___x_4003_, v_R_3729_);
v___x_4005_ = l_Lean_Expr_app___override(v___x_4004_, v___x_3931_);
v___x_4006_ = l_Lean_Expr_app___override(v___x_4001_, v___x_4005_);
v___x_4007_ = l_Lean_Expr_app___override(v___x_3998_, v___x_4006_);
v___x_4008_ = l_Lean_Expr_app___override(v___x_4007_, v___x_3934_);
v___x_4009_ = l_Lean_Expr_app___override(v___x_3995_, v___x_4008_);
v___x_4010_ = l_Lean_Expr_app___override(v___x_3943_, v___x_4009_);
v___x_4011_ = l_Lean_Expr_app___override(v___x_3937_, v___x_4010_);
v___x_4012_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__36));
v___x_4013_ = l_Lean_Level_max___override(v_u_3727_, v_v_3728_);
lean_inc(v___x_4013_);
v___x_4014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4014_, 0, v___x_4013_);
lean_ctor_set(v___x_4014_, 1, v___x_3754_);
v___x_4015_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4015_, 0, v_v_3728_);
lean_ctor_set(v___x_4015_, 1, v___x_4014_);
v___x_4016_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4016_, 0, v_u_3727_);
lean_ctor_set(v___x_4016_, 1, v___x_4015_);
v___x_4017_ = l_Lean_Expr_const___override(v___x_4012_, v___x_4016_);
v___x_4018_ = l_Lean_Expr_app___override(v___x_4017_, v_R_3729_);
v___x_4019_ = l_Lean_Expr_app___override(v___x_4018_, v_A_3730_);
v___x_4020_ = l_Lean_Expr_app___override(v___x_4019_, v___x_3799_);
v___x_4021_ = l_Lean_Expr_app___override(v___x_4020_, v___x_3932_);
v___x_4022_ = l_Lean_Expr_app___override(v___x_4021_, v___x_3885_);
v___x_4023_ = l_Lean_Expr_app___override(v___x_4022_, v___x_3811_);
v___x_4024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__39));
v___x_4025_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4025_, 0, v___x_4013_);
lean_ctor_set(v___x_4025_, 1, v___x_3787_);
lean_inc_ref_n(v___x_4025_, 2);
v___x_4026_ = l_Lean_Expr_const___override(v___x_4024_, v___x_4025_);
v___x_4027_ = l_Lean_Expr_app___override(v___x_4026_, v___x_3799_);
v___x_4028_ = l_Lean_Expr_app___override(v___x_4027_, v_R_3729_);
v___x_4029_ = l_Lean_Expr_app___override(v___x_4028_, v_A_3730_);
v___x_4030_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__42));
v___x_4031_ = l_Lean_Expr_const___override(v___x_4030_, v___x_3791_);
v___x_4032_ = l_Lean_Expr_app___override(v___x_4031_, v_R_3729_);
v___x_4033_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__45));
v___x_4034_ = l_Lean_Expr_const___override(v___x_4033_, v___x_3791_);
v___x_4035_ = l_Lean_Expr_app___override(v___x_4034_, v_R_3729_);
v___x_4036_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__47));
v___x_4037_ = l_Lean_Expr_const___override(v___x_4036_, v___x_3791_);
v___x_4038_ = l_Lean_Expr_app___override(v___x_4037_, v_R_3729_);
v___x_4039_ = l_Lean_Expr_app___override(v___x_4038_, v___x_3797_);
lean_inc_ref(v___x_4039_);
v___x_4040_ = l_Lean_Expr_app___override(v___x_4035_, v___x_4039_);
v___x_4041_ = l_Lean_Expr_app___override(v___x_4032_, v___x_4040_);
v___x_4042_ = l_Lean_Expr_app___override(v___x_4029_, v___x_4041_);
v___x_4043_ = l_Lean_Expr_const___override(v___x_4030_, v___x_3756_);
v___x_4044_ = l_Lean_Expr_app___override(v___x_4043_, v_A_3730_);
v___x_4045_ = l_Lean_Expr_const___override(v___x_4033_, v___x_3756_);
v___x_4046_ = l_Lean_Expr_app___override(v___x_4045_, v_A_3730_);
v___x_4047_ = l_Lean_Expr_const___override(v___x_4036_, v___x_3756_);
v___x_4048_ = l_Lean_Expr_app___override(v___x_4047_, v_A_3730_);
v___x_4049_ = l_Lean_Expr_app___override(v___x_4048_, v___x_3772_);
lean_inc_ref(v___x_4049_);
v___x_4050_ = l_Lean_Expr_app___override(v___x_4046_, v___x_4049_);
v___x_4051_ = l_Lean_Expr_app___override(v___x_4044_, v___x_4050_);
v___x_4052_ = l_Lean_Expr_app___override(v___x_4042_, v___x_4051_);
v___x_4053_ = l_Lean_Expr_app___override(v___x_4052_, v___x_3811_);
v___x_4054_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__50));
v___x_4055_ = l_Lean_Expr_const___override(v___x_4054_, v___x_4025_);
v___x_4056_ = l_Lean_Expr_app___override(v___x_4055_, v___x_3799_);
v___x_4057_ = l_Lean_Expr_app___override(v___x_4056_, v_R_3729_);
v___x_4058_ = l_Lean_Expr_app___override(v___x_4057_, v_A_3730_);
v___x_4059_ = l_Lean_Expr_app___override(v___x_4058_, v___x_4039_);
v___x_4060_ = l_Lean_Expr_app___override(v___x_4059_, v___x_4049_);
v___x_4061_ = l_Lean_Expr_app___override(v___x_4060_, v___x_3811_);
v___x_4062_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_ringCompute___closed__52));
v___x_4063_ = l_Lean_Expr_const___override(v___x_4062_, v___x_4025_);
v___x_4064_ = l_Lean_Expr_app___override(v___x_4063_, v___x_3799_);
v___x_4065_ = l_Lean_Expr_app___override(v___x_4064_, v_R_3729_);
v___x_4066_ = l_Lean_Expr_app___override(v___x_4065_, v_A_3730_);
v___x_4067_ = l_Lean_Expr_app___override(v___x_4066_, v___x_3797_);
v___x_4068_ = l_Lean_Expr_app___override(v___x_4067_, v___x_3772_);
v___x_4069_ = l_Lean_Expr_app___override(v___x_4068_, v___x_3811_);
v___x_4070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__36));
v___x_4071_ = l_Lean_Expr_const___override(v___x_4070_, v___x_3787_);
v___x_4072_ = l_Lean_Expr_app___override(v___x_4071_, v_R_3729_);
v___x_4073_ = l_Lean_Expr_app___override(v___x_4072_, v_A_3730_);
v___x_4074_ = l_Lean_Expr_app___override(v___x_4073_, v___x_3797_);
v___x_4075_ = l_Lean_Expr_app___override(v___x_4074_, v___x_3772_);
v___x_4076_ = l_Lean_Expr_app___override(v___x_4069_, v___x_4075_);
v___x_4077_ = l_Lean_Expr_app___override(v___x_4061_, v___x_4076_);
v___x_4078_ = l_Lean_Expr_app___override(v___x_4053_, v___x_4077_);
v___x_4079_ = l_Lean_Expr_app___override(v___x_4023_, v___x_4078_);
v___x_4080_ = l_Lean_Expr_app___override(v___x_4079_, v___x_3819_);
v___x_4081_ = l_Lean_Expr_app___override(v___x_4011_, v___x_4080_);
v___x_4082_ = l_Lean_Expr_app___override(v___x_3917_, v___x_4081_);
v___x_4083_ = l_Lean_Expr_app___override(v___x_3892_, v___x_4082_);
v___x_4084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__38));
v___x_4085_ = l_Lean_Expr_const___override(v___x_4084_, v___x_3782_);
v___x_4086_ = l_Lean_Expr_app___override(v___x_4085_, v_A_3730_);
v___x_4087_ = l_Lean_Expr_app___override(v___x_4086_, v___x_3887_);
v___x_4088_ = l_Lean_Expr_app___override(v___x_4083_, v___x_4087_);
v___x_4089_ = l_Lean_Expr_app___override(v___x_3870_, v___x_4088_);
v___x_4090_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4090_, 0, v___x_3861_);
lean_ctor_set(v___x_4090_, 1, v___x_3863_);
lean_ctor_set(v___x_4090_, 2, v___x_4089_);
v___x_4091_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_4091_, 0, v___x_3737_);
lean_ctor_set(v___x_4091_, 1, v___x_3738_);
lean_ctor_set(v___x_4091_, 2, v___x_3739_);
lean_ctor_set(v___x_4091_, 3, v___x_3740_);
lean_ctor_set(v___x_4091_, 4, v___x_3741_);
lean_ctor_set(v___x_4091_, 5, v___x_3742_);
lean_ctor_set(v___x_4091_, 6, v___x_3743_);
lean_ctor_set(v___x_4091_, 7, v___x_3744_);
lean_ctor_set(v___x_4091_, 8, v___x_3745_);
lean_ctor_set(v___x_4091_, 9, v___x_4090_);
return v___x_4091_;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0(void){
_start:
{
lean_object* v___x_4094_; 
v___x_4094_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4094_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1(void){
_start:
{
lean_object* v___x_4095_; lean_object* v___x_4096_; 
v___x_4095_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__0);
v___x_4096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4096_, 0, v___x_4095_);
return v___x_4096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0(lean_object* v_00_u03b2_4097_){
_start:
{
lean_object* v___x_4098_; 
v___x_4098_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0___closed__1);
return v___x_4098_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0(void){
_start:
{
lean_object* v___x_4099_; 
v___x_4099_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4099_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1(void){
_start:
{
lean_object* v___x_4100_; lean_object* v___x_4101_; 
v___x_4100_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__0);
v___x_4101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4101_, 0, v___x_4100_);
return v___x_4101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1(lean_object* v_00_u03b2_4102_){
_start:
{
lean_object* v___x_4103_; 
v___x_4103_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1___closed__1);
return v___x_4103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2(lean_object* v_x_4104_, lean_object* v_x_4105_, lean_object* v___y_4106_, lean_object* v___y_4107_, lean_object* v___y_4108_, lean_object* v___y_4109_){
_start:
{
if (lean_obj_tag(v_x_4105_) == 0)
{
lean_object* v___x_4111_; 
v___x_4111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4111_, 0, v_x_4104_);
return v___x_4111_;
}
else
{
lean_object* v_head_4112_; lean_object* v_tail_4113_; uint8_t v___x_4114_; uint8_t v___x_4115_; lean_object* v___x_4116_; lean_object* v___x_4117_; 
v_head_4112_ = lean_ctor_get(v_x_4105_, 0);
lean_inc(v_head_4112_);
v_tail_4113_ = lean_ctor_get(v_x_4105_, 1);
lean_inc(v_tail_4113_);
lean_dec_ref_known(v_x_4105_, 2);
v___x_4114_ = 1;
v___x_4115_ = 0;
v___x_4116_ = lean_unsigned_to_nat(1000u);
v___x_4117_ = l_Lean_Meta_SimpTheorems_addConst(v_x_4104_, v_head_4112_, v___x_4114_, v___x_4115_, v___x_4116_, v___y_4106_, v___y_4107_, v___y_4108_, v___y_4109_);
if (lean_obj_tag(v___x_4117_) == 0)
{
lean_object* v_a_4118_; 
v_a_4118_ = lean_ctor_get(v___x_4117_, 0);
lean_inc(v_a_4118_);
lean_dec_ref_known(v___x_4117_, 1);
v_x_4104_ = v_a_4118_;
v_x_4105_ = v_tail_4113_;
goto _start;
}
else
{
lean_dec(v_tail_4113_);
return v___x_4117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2___boxed(lean_object* v_x_4120_, lean_object* v_x_4121_, lean_object* v___y_4122_, lean_object* v___y_4123_, lean_object* v___y_4124_, lean_object* v___y_4125_, lean_object* v___y_4126_){
_start:
{
lean_object* v_res_4127_; 
v_res_4127_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2(v_x_4120_, v_x_4121_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_);
lean_dec(v___y_4125_);
lean_dec_ref(v___y_4124_);
lean_dec(v___y_4123_);
lean_dec_ref(v___y_4122_);
return v_res_4127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0(void){
_start:
{
lean_object* v___x_4128_; 
v___x_4128_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_4128_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1(void){
_start:
{
lean_object* v___x_4129_; 
v___x_4129_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__0(lean_box(0));
return v___x_4129_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2(void){
_start:
{
lean_object* v___x_4130_; 
v___x_4130_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_Algebra_preprocess_spec__1(lean_box(0));
return v___x_4130_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3(void){
_start:
{
lean_object* v___x_4131_; 
v___x_4131_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4131_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4(void){
_start:
{
lean_object* v___x_4132_; lean_object* v___x_4133_; 
v___x_4132_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__3);
v___x_4133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4133_, 0, v___x_4132_);
return v___x_4133_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5(void){
_start:
{
lean_object* v___x_4134_; lean_object* v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; lean_object* v_thms_4138_; 
v___x_4134_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4);
v___x_4135_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__2);
v___x_4136_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__1);
v___x_4137_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__0);
v_thms_4138_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_thms_4138_, 0, v___x_4137_);
lean_ctor_set(v_thms_4138_, 1, v___x_4137_);
lean_ctor_set(v_thms_4138_, 2, v___x_4136_);
lean_ctor_set(v_thms_4138_, 3, v___x_4135_);
lean_ctor_set(v_thms_4138_, 4, v___x_4136_);
lean_ctor_set(v_thms_4138_, 5, v___x_4134_);
return v_thms_4138_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14(void){
_start:
{
lean_object* v___x_4165_; lean_object* v___x_4166_; uint8_t v___x_4167_; lean_object* v___x_4168_; 
v___x_4165_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4);
v___x_4166_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__10);
v___x_4167_ = 1;
v___x_4168_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4168_, 0, v___x_4166_);
lean_ctor_set(v___x_4168_, 1, v___x_4165_);
lean_ctor_set_uint8(v___x_4168_, sizeof(void*)*2, v___x_4167_);
return v___x_4168_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15(void){
_start:
{
lean_object* v___x_4169_; lean_object* v___x_4170_; lean_object* v___x_4171_; 
v___x_4169_ = lean_unsigned_to_nat(0u);
v___x_4170_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4);
v___x_4171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4171_, 0, v___x_4170_);
lean_ctor_set(v___x_4171_, 1, v___x_4169_);
return v___x_4171_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16(void){
_start:
{
lean_object* v___x_4172_; lean_object* v___x_4173_; lean_object* v___x_4174_; 
v___x_4172_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17, &lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__17);
v___x_4173_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__4);
v___x_4174_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4174_, 0, v___x_4173_);
lean_ctor_set(v___x_4174_, 1, v___x_4173_);
lean_ctor_set(v___x_4174_, 2, v___x_4173_);
lean_ctor_set(v___x_4174_, 3, v___x_4172_);
return v___x_4174_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17(void){
_start:
{
lean_object* v___x_4175_; lean_object* v___x_4176_; lean_object* v___x_4177_; 
v___x_4175_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__16);
v___x_4176_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__15);
v___x_4177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4177_, 0, v___x_4176_);
lean_ctor_set(v___x_4177_, 1, v___x_4175_);
return v___x_4177_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19(void){
_start:
{
lean_object* v___x_4180_; lean_object* v___x_4181_; 
v___x_4180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__18));
v___x_4181_ = l_Lean_Meta_Simp_mkDefaultMethodsCore(v___x_4180_);
return v___x_4181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess(lean_object* v_e_4182_, lean_object* v_a_4183_, lean_object* v_a_4184_, lean_object* v_a_4185_, lean_object* v_a_4186_){
_start:
{
lean_object* v_thms_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; 
v_thms_4188_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__5);
v___x_4189_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__13));
v___x_4190_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_preprocess_spec__2(v_thms_4188_, v___x_4189_, v_a_4183_, v_a_4184_, v_a_4185_, v_a_4186_);
if (lean_obj_tag(v___x_4190_) == 0)
{
lean_object* v_a_4191_; lean_object* v___x_4192_; lean_object* v___x_4193_; lean_object* v___x_4194_; lean_object* v___x_4195_; lean_object* v___x_4196_; lean_object* v___x_4197_; lean_object* v___x_4198_; 
v_a_4191_ = lean_ctor_get(v___x_4190_, 0);
lean_inc(v_a_4191_);
lean_dec_ref_known(v___x_4190_, 1);
v___x_4192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pushCast___closed__8));
v___x_4193_ = lean_unsigned_to_nat(1u);
v___x_4194_ = lean_mk_empty_array_with_capacity(v___x_4193_);
v___x_4195_ = lean_array_push(v___x_4194_, v_a_4191_);
v___x_4196_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__14);
v___x_4197_ = l_Lean_Options_empty;
v___x_4198_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_4192_, v___x_4195_, v___x_4196_, v___x_4197_, v_a_4183_, v_a_4185_, v_a_4186_);
if (lean_obj_tag(v___x_4198_) == 0)
{
lean_object* v_a_4199_; lean_object* v___x_4200_; lean_object* v___x_4201_; lean_object* v___x_4202_; 
v_a_4199_ = lean_ctor_get(v___x_4198_, 0);
lean_inc(v_a_4199_);
lean_dec_ref_known(v___x_4198_, 1);
v___x_4200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__17);
v___x_4201_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_preprocess___closed__19);
v___x_4202_ = l_Lean_Meta_Simp_main(v_e_4182_, v_a_4199_, v___x_4200_, v___x_4201_, v_a_4183_, v_a_4184_, v_a_4185_, v_a_4186_);
if (lean_obj_tag(v___x_4202_) == 0)
{
lean_object* v_a_4203_; lean_object* v___x_4205_; uint8_t v_isShared_4206_; uint8_t v_isSharedCheck_4211_; 
v_a_4203_ = lean_ctor_get(v___x_4202_, 0);
v_isSharedCheck_4211_ = !lean_is_exclusive(v___x_4202_);
if (v_isSharedCheck_4211_ == 0)
{
v___x_4205_ = v___x_4202_;
v_isShared_4206_ = v_isSharedCheck_4211_;
goto v_resetjp_4204_;
}
else
{
lean_inc(v_a_4203_);
lean_dec(v___x_4202_);
v___x_4205_ = lean_box(0);
v_isShared_4206_ = v_isSharedCheck_4211_;
goto v_resetjp_4204_;
}
v_resetjp_4204_:
{
lean_object* v_fst_4207_; lean_object* v___x_4209_; 
v_fst_4207_ = lean_ctor_get(v_a_4203_, 0);
lean_inc(v_fst_4207_);
lean_dec(v_a_4203_);
if (v_isShared_4206_ == 0)
{
lean_ctor_set(v___x_4205_, 0, v_fst_4207_);
v___x_4209_ = v___x_4205_;
goto v_reusejp_4208_;
}
else
{
lean_object* v_reuseFailAlloc_4210_; 
v_reuseFailAlloc_4210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4210_, 0, v_fst_4207_);
v___x_4209_ = v_reuseFailAlloc_4210_;
goto v_reusejp_4208_;
}
v_reusejp_4208_:
{
return v___x_4209_;
}
}
}
else
{
lean_object* v_a_4212_; lean_object* v___x_4214_; uint8_t v_isShared_4215_; uint8_t v_isSharedCheck_4219_; 
v_a_4212_ = lean_ctor_get(v___x_4202_, 0);
v_isSharedCheck_4219_ = !lean_is_exclusive(v___x_4202_);
if (v_isSharedCheck_4219_ == 0)
{
v___x_4214_ = v___x_4202_;
v_isShared_4215_ = v_isSharedCheck_4219_;
goto v_resetjp_4213_;
}
else
{
lean_inc(v_a_4212_);
lean_dec(v___x_4202_);
v___x_4214_ = lean_box(0);
v_isShared_4215_ = v_isSharedCheck_4219_;
goto v_resetjp_4213_;
}
v_resetjp_4213_:
{
lean_object* v___x_4217_; 
if (v_isShared_4215_ == 0)
{
v___x_4217_ = v___x_4214_;
goto v_reusejp_4216_;
}
else
{
lean_object* v_reuseFailAlloc_4218_; 
v_reuseFailAlloc_4218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4218_, 0, v_a_4212_);
v___x_4217_ = v_reuseFailAlloc_4218_;
goto v_reusejp_4216_;
}
v_reusejp_4216_:
{
return v___x_4217_;
}
}
}
}
else
{
lean_object* v_a_4220_; lean_object* v___x_4222_; uint8_t v_isShared_4223_; uint8_t v_isSharedCheck_4227_; 
lean_dec_ref(v_e_4182_);
v_a_4220_ = lean_ctor_get(v___x_4198_, 0);
v_isSharedCheck_4227_ = !lean_is_exclusive(v___x_4198_);
if (v_isSharedCheck_4227_ == 0)
{
v___x_4222_ = v___x_4198_;
v_isShared_4223_ = v_isSharedCheck_4227_;
goto v_resetjp_4221_;
}
else
{
lean_inc(v_a_4220_);
lean_dec(v___x_4198_);
v___x_4222_ = lean_box(0);
v_isShared_4223_ = v_isSharedCheck_4227_;
goto v_resetjp_4221_;
}
v_resetjp_4221_:
{
lean_object* v___x_4225_; 
if (v_isShared_4223_ == 0)
{
v___x_4225_ = v___x_4222_;
goto v_reusejp_4224_;
}
else
{
lean_object* v_reuseFailAlloc_4226_; 
v_reuseFailAlloc_4226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4226_, 0, v_a_4220_);
v___x_4225_ = v_reuseFailAlloc_4226_;
goto v_reusejp_4224_;
}
v_reusejp_4224_:
{
return v___x_4225_;
}
}
}
}
else
{
lean_object* v_a_4228_; lean_object* v___x_4230_; uint8_t v_isShared_4231_; uint8_t v_isSharedCheck_4235_; 
lean_dec_ref(v_e_4182_);
v_a_4228_ = lean_ctor_get(v___x_4190_, 0);
v_isSharedCheck_4235_ = !lean_is_exclusive(v___x_4190_);
if (v_isSharedCheck_4235_ == 0)
{
v___x_4230_ = v___x_4190_;
v_isShared_4231_ = v_isSharedCheck_4235_;
goto v_resetjp_4229_;
}
else
{
lean_inc(v_a_4228_);
lean_dec(v___x_4190_);
v___x_4230_ = lean_box(0);
v_isShared_4231_ = v_isSharedCheck_4235_;
goto v_resetjp_4229_;
}
v_resetjp_4229_:
{
lean_object* v___x_4233_; 
if (v_isShared_4231_ == 0)
{
v___x_4233_ = v___x_4230_;
goto v_reusejp_4232_;
}
else
{
lean_object* v_reuseFailAlloc_4234_; 
v_reuseFailAlloc_4234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4234_, 0, v_a_4228_);
v___x_4233_ = v_reuseFailAlloc_4234_;
goto v_reusejp_4232_;
}
v_reusejp_4232_:
{
return v___x_4233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_preprocess___boxed(lean_object* v_e_4236_, lean_object* v_a_4237_, lean_object* v_a_4238_, lean_object* v_a_4239_, lean_object* v_a_4240_, lean_object* v_a_4241_){
_start:
{
lean_object* v_res_4242_; 
v_res_4242_ = lp_mathlib_Mathlib_Tactic_Algebra_preprocess(v_e_4236_, v_a_4237_, v_a_4238_, v_a_4239_, v_a_4240_);
lean_dec(v_a_4240_);
lean_dec_ref(v_a_4239_);
lean_dec(v_a_4238_);
lean_dec_ref(v_a_4237_);
return v_res_4242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux(lean_object* v_e_4278_, lean_object* v_a_4279_, lean_object* v_a_4280_, lean_object* v_a_4281_, lean_object* v_a_4282_, lean_object* v_a_4283_){
_start:
{
lean_object* v_a_4286_; lean_object* v_b_4287_; lean_object* v___y_4288_; lean_object* v___y_4289_; lean_object* v___y_4290_; lean_object* v___y_4291_; lean_object* v___y_4292_; lean_object* v_R_4298_; lean_object* v_a_4299_; lean_object* v___y_4300_; lean_object* v___y_4301_; lean_object* v___y_4302_; lean_object* v___y_4303_; lean_object* v___y_4304_; lean_object* v___x_4307_; 
v___x_4307_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_e_4278_, v_a_4281_);
if (lean_obj_tag(v___x_4307_) == 0)
{
lean_object* v_a_4308_; lean_object* v___x_4310_; uint8_t v_isShared_4311_; uint8_t v_isSharedCheck_4401_; 
v_a_4308_ = lean_ctor_get(v___x_4307_, 0);
v_isSharedCheck_4401_ = !lean_is_exclusive(v___x_4307_);
if (v_isSharedCheck_4401_ == 0)
{
v___x_4310_ = v___x_4307_;
v_isShared_4311_ = v_isSharedCheck_4401_;
goto v_resetjp_4309_;
}
else
{
lean_inc(v_a_4308_);
lean_dec(v___x_4307_);
v___x_4310_ = lean_box(0);
v_isShared_4311_ = v_isSharedCheck_4401_;
goto v_resetjp_4309_;
}
v_resetjp_4309_:
{
lean_object* v___y_4313_; lean_object* v___x_4319_; uint8_t v___x_4320_; 
v___x_4319_ = l_Lean_Expr_cleanupAnnotations(v_a_4308_);
v___x_4320_ = l_Lean_Expr_isApp(v___x_4319_);
if (v___x_4320_ == 0)
{
lean_dec_ref(v___x_4319_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v_arg_4321_; lean_object* v___x_4322_; uint8_t v___x_4323_; 
v_arg_4321_ = lean_ctor_get(v___x_4319_, 1);
lean_inc_ref(v_arg_4321_);
v___x_4322_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4319_);
v___x_4323_ = l_Lean_Expr_isApp(v___x_4322_);
if (v___x_4323_ == 0)
{
lean_dec_ref(v___x_4322_);
lean_dec_ref(v_arg_4321_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v_arg_4324_; lean_object* v___x_4325_; uint8_t v___x_4326_; 
v_arg_4324_ = lean_ctor_get(v___x_4322_, 1);
lean_inc_ref(v_arg_4324_);
v___x_4325_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4322_);
v___x_4326_ = l_Lean_Expr_isApp(v___x_4325_);
if (v___x_4326_ == 0)
{
lean_dec_ref(v___x_4325_);
lean_dec_ref(v_arg_4324_);
lean_dec_ref(v_arg_4321_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v___x_4327_; lean_object* v___x_4328_; uint8_t v___x_4329_; 
v___x_4327_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4325_);
v___x_4328_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__2));
v___x_4329_ = l_Lean_Expr_isConstOf(v___x_4327_, v___x_4328_);
if (v___x_4329_ == 0)
{
lean_object* v___x_4330_; uint8_t v___x_4331_; 
v___x_4330_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_4331_ = l_Lean_Expr_isConstOf(v___x_4327_, v___x_4330_);
if (v___x_4331_ == 0)
{
uint8_t v___x_4332_; 
v___x_4332_ = l_Lean_Expr_isApp(v___x_4327_);
if (v___x_4332_ == 0)
{
lean_dec_ref(v___x_4327_);
lean_dec_ref(v_arg_4324_);
lean_dec_ref(v_arg_4321_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v___x_4333_; uint8_t v___x_4334_; 
v___x_4333_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4327_);
v___x_4334_ = l_Lean_Expr_isApp(v___x_4333_);
if (v___x_4334_ == 0)
{
lean_dec_ref(v___x_4333_);
lean_dec_ref(v_arg_4324_);
lean_dec_ref(v_arg_4321_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v_arg_4335_; lean_object* v___x_4336_; lean_object* v___x_4337_; uint8_t v___x_4338_; 
v_arg_4335_ = lean_ctor_get(v___x_4333_, 1);
lean_inc_ref(v_arg_4335_);
v___x_4336_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4333_);
v___x_4337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__5));
v___x_4338_ = l_Lean_Expr_isConstOf(v___x_4336_, v___x_4337_);
if (v___x_4338_ == 0)
{
lean_object* v___x_4339_; uint8_t v___x_4340_; 
v___x_4339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__8));
v___x_4340_ = l_Lean_Expr_isConstOf(v___x_4336_, v___x_4339_);
if (v___x_4340_ == 0)
{
lean_object* v___x_4341_; uint8_t v___x_4342_; 
v___x_4341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__11));
v___x_4342_ = l_Lean_Expr_isConstOf(v___x_4336_, v___x_4341_);
if (v___x_4342_ == 0)
{
lean_object* v___x_4343_; uint8_t v___x_4344_; 
v___x_4343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__14));
v___x_4344_ = l_Lean_Expr_isConstOf(v___x_4336_, v___x_4343_);
if (v___x_4344_ == 0)
{
uint8_t v___x_4345_; 
lean_dec_ref(v_arg_4335_);
v___x_4345_ = l_Lean_Expr_isApp(v___x_4336_);
if (v___x_4345_ == 0)
{
lean_dec_ref(v___x_4336_);
lean_dec_ref(v_arg_4324_);
lean_dec_ref(v_arg_4321_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v_arg_4346_; lean_object* v___x_4347_; lean_object* v___x_4348_; uint8_t v___x_4349_; 
v_arg_4346_ = lean_ctor_get(v___x_4336_, 1);
lean_inc_ref(v_arg_4346_);
v___x_4347_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4336_);
v___x_4348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__17));
v___x_4349_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4348_);
if (v___x_4349_ == 0)
{
lean_object* v___x_4350_; uint8_t v___x_4351_; 
v___x_4350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___closed__20));
v___x_4351_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4350_);
if (v___x_4351_ == 0)
{
lean_object* v___x_4352_; uint8_t v___x_4353_; 
v___x_4352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__2));
v___x_4353_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4352_);
if (v___x_4353_ == 0)
{
lean_object* v___x_4354_; uint8_t v___x_4355_; 
v___x_4354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__2));
v___x_4355_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4354_);
if (v___x_4355_ == 0)
{
lean_object* v___x_4356_; uint8_t v___x_4357_; 
v___x_4356_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__16));
v___x_4357_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4356_);
if (v___x_4357_ == 0)
{
lean_object* v___x_4358_; uint8_t v___x_4359_; 
lean_dec_ref(v_arg_4346_);
lean_dec_ref(v_arg_4321_);
v___x_4358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__15));
v___x_4359_ = l_Lean_Expr_isConstOf(v___x_4347_, v___x_4358_);
lean_dec_ref(v___x_4347_);
if (v___x_4359_ == 0)
{
lean_dec_ref(v_arg_4324_);
v___y_4313_ = v_a_4279_;
goto v___jp_4312_;
}
else
{
lean_object* v___x_4360_; 
lean_del_object(v___x_4310_);
v___x_4360_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_arg_4324_, v_a_4281_);
if (lean_obj_tag(v___x_4360_) == 0)
{
lean_object* v_a_4361_; lean_object* v___x_4363_; uint8_t v_isShared_4364_; uint8_t v_isSharedCheck_4390_; 
v_a_4361_ = lean_ctor_get(v___x_4360_, 0);
v_isSharedCheck_4390_ = !lean_is_exclusive(v___x_4360_);
if (v_isSharedCheck_4390_ == 0)
{
v___x_4363_ = v___x_4360_;
v_isShared_4364_ = v_isSharedCheck_4390_;
goto v_resetjp_4362_;
}
else
{
lean_inc(v_a_4361_);
lean_dec(v___x_4360_);
v___x_4363_ = lean_box(0);
v_isShared_4364_ = v_isSharedCheck_4390_;
goto v_resetjp_4362_;
}
v_resetjp_4362_:
{
lean_object* v___y_4366_; lean_object* v___x_4372_; uint8_t v___x_4373_; 
v___x_4372_ = l_Lean_Expr_cleanupAnnotations(v_a_4361_);
v___x_4373_ = l_Lean_Expr_isApp(v___x_4372_);
if (v___x_4373_ == 0)
{
lean_dec_ref(v___x_4372_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v___x_4374_; uint8_t v___x_4375_; 
v___x_4374_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4372_);
v___x_4375_ = l_Lean_Expr_isApp(v___x_4374_);
if (v___x_4375_ == 0)
{
lean_dec_ref(v___x_4374_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v___x_4376_; uint8_t v___x_4377_; 
v___x_4376_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4374_);
v___x_4377_ = l_Lean_Expr_isApp(v___x_4376_);
if (v___x_4377_ == 0)
{
lean_dec_ref(v___x_4376_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v___x_4378_; uint8_t v___x_4379_; 
v___x_4378_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4376_);
v___x_4379_ = l_Lean_Expr_isApp(v___x_4378_);
if (v___x_4379_ == 0)
{
lean_dec_ref(v___x_4378_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v___x_4380_; uint8_t v___x_4381_; 
v___x_4380_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4378_);
v___x_4381_ = l_Lean_Expr_isApp(v___x_4380_);
if (v___x_4381_ == 0)
{
lean_dec_ref(v___x_4380_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v_arg_4382_; lean_object* v___x_4383_; lean_object* v___x_4384_; uint8_t v___x_4385_; 
v_arg_4382_ = lean_ctor_get(v___x_4380_, 1);
lean_inc_ref(v_arg_4382_);
v___x_4383_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4380_);
v___x_4384_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__27));
v___x_4385_ = l_Lean_Expr_isConstOf(v___x_4383_, v___x_4384_);
lean_dec_ref(v___x_4383_);
if (v___x_4385_ == 0)
{
lean_dec_ref(v_arg_4382_);
v___y_4366_ = v_a_4279_;
goto v___jp_4365_;
}
else
{
lean_object* v___x_4386_; lean_object* v___x_4387_; lean_object* v___x_4388_; lean_object* v___x_4389_; 
lean_del_object(v___x_4363_);
v___x_4386_ = lean_box(0);
v___x_4387_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4387_, 0, v_arg_4382_);
lean_ctor_set(v___x_4387_, 1, v_a_4279_);
v___x_4388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4388_, 0, v___x_4386_);
lean_ctor_set(v___x_4388_, 1, v___x_4387_);
v___x_4389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4389_, 0, v___x_4388_);
return v___x_4389_;
}
}
}
}
}
}
v___jp_4365_:
{
lean_object* v___x_4367_; lean_object* v___x_4368_; lean_object* v___x_4370_; 
v___x_4367_ = lean_box(0);
v___x_4368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4368_, 0, v___x_4367_);
lean_ctor_set(v___x_4368_, 1, v___y_4366_);
if (v_isShared_4364_ == 0)
{
lean_ctor_set(v___x_4363_, 0, v___x_4368_);
v___x_4370_ = v___x_4363_;
goto v_reusejp_4369_;
}
else
{
lean_object* v_reuseFailAlloc_4371_; 
v_reuseFailAlloc_4371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4371_, 0, v___x_4368_);
v___x_4370_ = v_reuseFailAlloc_4371_;
goto v_reusejp_4369_;
}
v_reusejp_4369_:
{
return v___x_4370_;
}
}
}
}
else
{
lean_object* v_a_4391_; lean_object* v___x_4393_; uint8_t v_isShared_4394_; uint8_t v_isSharedCheck_4398_; 
lean_dec(v_a_4279_);
v_a_4391_ = lean_ctor_get(v___x_4360_, 0);
v_isSharedCheck_4398_ = !lean_is_exclusive(v___x_4360_);
if (v_isSharedCheck_4398_ == 0)
{
v___x_4393_ = v___x_4360_;
v_isShared_4394_ = v_isSharedCheck_4398_;
goto v_resetjp_4392_;
}
else
{
lean_inc(v_a_4391_);
lean_dec(v___x_4360_);
v___x_4393_ = lean_box(0);
v_isShared_4394_ = v_isSharedCheck_4398_;
goto v_resetjp_4392_;
}
v_resetjp_4392_:
{
lean_object* v___x_4396_; 
if (v_isShared_4394_ == 0)
{
v___x_4396_ = v___x_4393_;
goto v_reusejp_4395_;
}
else
{
lean_object* v_reuseFailAlloc_4397_; 
v_reuseFailAlloc_4397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4397_, 0, v_a_4391_);
v___x_4396_ = v_reuseFailAlloc_4397_;
goto v_reusejp_4395_;
}
v_reusejp_4395_:
{
return v___x_4396_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_4347_);
lean_dec_ref(v_arg_4324_);
lean_del_object(v___x_4310_);
v_R_4298_ = v_arg_4346_;
v_a_4299_ = v_arg_4321_;
v___y_4300_ = v_a_4279_;
v___y_4301_ = v_a_4280_;
v___y_4302_ = v_a_4281_;
v___y_4303_ = v_a_4282_;
v___y_4304_ = v_a_4283_;
goto v___jp_4297_;
}
}
else
{
lean_dec_ref(v___x_4347_);
lean_dec_ref(v_arg_4346_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4347_);
lean_dec_ref(v_arg_4346_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4347_);
lean_dec_ref(v_arg_4346_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4347_);
lean_dec_ref(v_arg_4346_);
lean_dec_ref(v_arg_4321_);
lean_del_object(v___x_4310_);
v_e_4278_ = v_arg_4324_;
goto _start;
}
}
}
else
{
lean_dec_ref(v___x_4336_);
lean_dec_ref(v_arg_4324_);
lean_del_object(v___x_4310_);
v_R_4298_ = v_arg_4335_;
v_a_4299_ = v_arg_4321_;
v___y_4300_ = v_a_4279_;
v___y_4301_ = v_a_4280_;
v___y_4302_ = v_a_4281_;
v___y_4303_ = v_a_4282_;
v___y_4304_ = v_a_4283_;
goto v___jp_4297_;
}
}
else
{
lean_dec_ref(v___x_4336_);
lean_dec_ref(v_arg_4335_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4336_);
lean_dec_ref(v_arg_4335_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4336_);
lean_dec_ref(v_arg_4335_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
}
}
else
{
lean_dec_ref(v___x_4327_);
lean_del_object(v___x_4310_);
v_a_4286_ = v_arg_4324_;
v_b_4287_ = v_arg_4321_;
v___y_4288_ = v_a_4279_;
v___y_4289_ = v_a_4280_;
v___y_4290_ = v_a_4281_;
v___y_4291_ = v_a_4282_;
v___y_4292_ = v_a_4283_;
goto v___jp_4285_;
}
}
else
{
lean_dec_ref(v___x_4327_);
lean_dec_ref(v_arg_4324_);
lean_del_object(v___x_4310_);
v_e_4278_ = v_arg_4321_;
goto _start;
}
}
}
}
v___jp_4312_:
{
lean_object* v___x_4314_; lean_object* v___x_4315_; lean_object* v___x_4317_; 
v___x_4314_ = lean_box(0);
v___x_4315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4315_, 0, v___x_4314_);
lean_ctor_set(v___x_4315_, 1, v___y_4313_);
if (v_isShared_4311_ == 0)
{
lean_ctor_set(v___x_4310_, 0, v___x_4315_);
v___x_4317_ = v___x_4310_;
goto v_reusejp_4316_;
}
else
{
lean_object* v_reuseFailAlloc_4318_; 
v_reuseFailAlloc_4318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4318_, 0, v___x_4315_);
v___x_4317_ = v_reuseFailAlloc_4318_;
goto v_reusejp_4316_;
}
v_reusejp_4316_:
{
return v___x_4317_;
}
}
}
}
else
{
lean_object* v_a_4402_; lean_object* v___x_4404_; uint8_t v_isShared_4405_; uint8_t v_isSharedCheck_4409_; 
lean_dec(v_a_4279_);
v_a_4402_ = lean_ctor_get(v___x_4307_, 0);
v_isSharedCheck_4409_ = !lean_is_exclusive(v___x_4307_);
if (v_isSharedCheck_4409_ == 0)
{
v___x_4404_ = v___x_4307_;
v_isShared_4405_ = v_isSharedCheck_4409_;
goto v_resetjp_4403_;
}
else
{
lean_inc(v_a_4402_);
lean_dec(v___x_4307_);
v___x_4404_ = lean_box(0);
v_isShared_4405_ = v_isSharedCheck_4409_;
goto v_resetjp_4403_;
}
v_resetjp_4403_:
{
lean_object* v___x_4407_; 
if (v_isShared_4405_ == 0)
{
v___x_4407_ = v___x_4404_;
goto v_reusejp_4406_;
}
else
{
lean_object* v_reuseFailAlloc_4408_; 
v_reuseFailAlloc_4408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4408_, 0, v_a_4402_);
v___x_4407_ = v_reuseFailAlloc_4408_;
goto v_reusejp_4406_;
}
v_reusejp_4406_:
{
return v___x_4407_;
}
}
}
v___jp_4285_:
{
lean_object* v___x_4293_; 
v___x_4293_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux(v_a_4286_, v___y_4288_, v___y_4289_, v___y_4290_, v___y_4291_, v___y_4292_);
if (lean_obj_tag(v___x_4293_) == 0)
{
lean_object* v_a_4294_; lean_object* v_snd_4295_; 
v_a_4294_ = lean_ctor_get(v___x_4293_, 0);
lean_inc(v_a_4294_);
lean_dec_ref_known(v___x_4293_, 1);
v_snd_4295_ = lean_ctor_get(v_a_4294_, 1);
lean_inc(v_snd_4295_);
lean_dec(v_a_4294_);
v_e_4278_ = v_b_4287_;
v_a_4279_ = v_snd_4295_;
v_a_4280_ = v___y_4289_;
v_a_4281_ = v___y_4290_;
v_a_4282_ = v___y_4291_;
v_a_4283_ = v___y_4292_;
goto _start;
}
else
{
lean_dec_ref(v_b_4287_);
return v___x_4293_;
}
}
v___jp_4297_:
{
lean_object* v___x_4305_; 
v___x_4305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4305_, 0, v_R_4298_);
lean_ctor_set(v___x_4305_, 1, v___y_4300_);
v_e_4278_ = v_a_4299_;
v_a_4279_ = v___x_4305_;
v_a_4280_ = v___y_4301_;
v_a_4281_ = v___y_4302_;
v_a_4282_ = v___y_4303_;
v_a_4283_ = v___y_4304_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux___boxed(lean_object* v_e_4410_, lean_object* v_a_4411_, lean_object* v_a_4412_, lean_object* v_a_4413_, lean_object* v_a_4414_, lean_object* v_a_4415_, lean_object* v_a_4416_){
_start:
{
lean_object* v_res_4417_; 
v_res_4417_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux(v_e_4410_, v_a_4411_, v_a_4412_, v_a_4413_, v_a_4414_, v_a_4415_);
lean_dec(v_a_4415_);
lean_dec_ref(v_a_4414_);
lean_dec(v_a_4413_);
lean_dec_ref(v_a_4412_);
return v_res_4417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(lean_object* v_e_4418_, lean_object* v_a_4419_, lean_object* v_a_4420_, lean_object* v_a_4421_, lean_object* v_a_4422_){
_start:
{
lean_object* v___x_4424_; lean_object* v___x_4425_; 
v___x_4424_ = lean_box(0);
v___x_4425_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRingsAux(v_e_4418_, v___x_4424_, v_a_4419_, v_a_4420_, v_a_4421_, v_a_4422_);
if (lean_obj_tag(v___x_4425_) == 0)
{
lean_object* v_a_4426_; lean_object* v___x_4428_; uint8_t v_isShared_4429_; uint8_t v_isSharedCheck_4434_; 
v_a_4426_ = lean_ctor_get(v___x_4425_, 0);
v_isSharedCheck_4434_ = !lean_is_exclusive(v___x_4425_);
if (v_isSharedCheck_4434_ == 0)
{
v___x_4428_ = v___x_4425_;
v_isShared_4429_ = v_isSharedCheck_4434_;
goto v_resetjp_4427_;
}
else
{
lean_inc(v_a_4426_);
lean_dec(v___x_4425_);
v___x_4428_ = lean_box(0);
v_isShared_4429_ = v_isSharedCheck_4434_;
goto v_resetjp_4427_;
}
v_resetjp_4427_:
{
lean_object* v_snd_4430_; lean_object* v___x_4432_; 
v_snd_4430_ = lean_ctor_get(v_a_4426_, 1);
lean_inc(v_snd_4430_);
lean_dec(v_a_4426_);
if (v_isShared_4429_ == 0)
{
lean_ctor_set(v___x_4428_, 0, v_snd_4430_);
v___x_4432_ = v___x_4428_;
goto v_reusejp_4431_;
}
else
{
lean_object* v_reuseFailAlloc_4433_; 
v_reuseFailAlloc_4433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4433_, 0, v_snd_4430_);
v___x_4432_ = v_reuseFailAlloc_4433_;
goto v_reusejp_4431_;
}
v_reusejp_4431_:
{
return v___x_4432_;
}
}
}
else
{
lean_object* v_a_4435_; lean_object* v___x_4437_; uint8_t v_isShared_4438_; uint8_t v_isSharedCheck_4442_; 
v_a_4435_ = lean_ctor_get(v___x_4425_, 0);
v_isSharedCheck_4442_ = !lean_is_exclusive(v___x_4425_);
if (v_isSharedCheck_4442_ == 0)
{
v___x_4437_ = v___x_4425_;
v_isShared_4438_ = v_isSharedCheck_4442_;
goto v_resetjp_4436_;
}
else
{
lean_inc(v_a_4435_);
lean_dec(v___x_4425_);
v___x_4437_ = lean_box(0);
v_isShared_4438_ = v_isSharedCheck_4442_;
goto v_resetjp_4436_;
}
v_resetjp_4436_:
{
lean_object* v___x_4440_; 
if (v_isShared_4438_ == 0)
{
v___x_4440_ = v___x_4437_;
goto v_reusejp_4439_;
}
else
{
lean_object* v_reuseFailAlloc_4441_; 
v_reuseFailAlloc_4441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4441_, 0, v_a_4435_);
v___x_4440_ = v_reuseFailAlloc_4441_;
goto v_reusejp_4439_;
}
v_reusejp_4439_:
{
return v___x_4440_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings___boxed(lean_object* v_e_4443_, lean_object* v_a_4444_, lean_object* v_a_4445_, lean_object* v_a_4446_, lean_object* v_a_4447_, lean_object* v_a_4448_){
_start:
{
lean_object* v_res_4449_; 
v_res_4449_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(v_e_4443_, v_a_4444_, v_a_4445_, v_a_4446_, v_a_4447_);
lean_dec(v_a_4447_);
lean_dec_ref(v_a_4446_);
lean_dec(v_a_4445_);
lean_dec_ref(v_a_4444_);
return v_res_4449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(lean_object* v_r1_4452_, lean_object* v_r2_4453_, lean_object* v_a_4454_, lean_object* v_a_4455_, lean_object* v_a_4456_, lean_object* v_a_4457_){
_start:
{
lean_object* v___y_4460_; uint8_t v___y_4461_; lean_object* v_a_4465_; lean_object* v_fst_4468_; lean_object* v_snd_4469_; lean_object* v_fst_4470_; lean_object* v_snd_4471_; lean_object* v_keyedConfig_4472_; uint8_t v_trackZetaDelta_4473_; lean_object* v_zetaDeltaSet_4474_; lean_object* v_lctx_4475_; lean_object* v_localInstances_4476_; lean_object* v_defEqCtx_x3f_4477_; lean_object* v_synthPendingDepth_4478_; lean_object* v_customCanUnfoldPredicate_x3f_4479_; uint8_t v_univApprox_4480_; uint8_t v_inTypeClassResolution_4481_; uint8_t v_cacheInferType_4482_; uint8_t v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v___x_4486_; 
v_fst_4468_ = lean_ctor_get(v_r1_4452_, 0);
v_snd_4469_ = lean_ctor_get(v_r1_4452_, 1);
v_fst_4470_ = lean_ctor_get(v_r2_4453_, 0);
v_snd_4471_ = lean_ctor_get(v_r2_4453_, 1);
v_keyedConfig_4472_ = lean_ctor_get(v_a_4454_, 0);
v_trackZetaDelta_4473_ = lean_ctor_get_uint8(v_a_4454_, sizeof(void*)*7);
v_zetaDeltaSet_4474_ = lean_ctor_get(v_a_4454_, 1);
v_lctx_4475_ = lean_ctor_get(v_a_4454_, 2);
v_localInstances_4476_ = lean_ctor_get(v_a_4454_, 3);
v_defEqCtx_x3f_4477_ = lean_ctor_get(v_a_4454_, 4);
v_synthPendingDepth_4478_ = lean_ctor_get(v_a_4454_, 5);
v_customCanUnfoldPredicate_x3f_4479_ = lean_ctor_get(v_a_4454_, 6);
v_univApprox_4480_ = lean_ctor_get_uint8(v_a_4454_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_4481_ = lean_ctor_get_uint8(v_a_4454_, sizeof(void*)*7 + 2);
v_cacheInferType_4482_ = lean_ctor_get_uint8(v_a_4454_, sizeof(void*)*7 + 3);
v___x_4483_ = 2;
lean_inc_ref(v_keyedConfig_4472_);
v___x_4484_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_4483_, v_keyedConfig_4472_);
lean_inc(v_customCanUnfoldPredicate_x3f_4479_);
lean_inc(v_synthPendingDepth_4478_);
lean_inc(v_defEqCtx_x3f_4477_);
lean_inc_ref(v_localInstances_4476_);
lean_inc_ref(v_lctx_4475_);
lean_inc(v_zetaDeltaSet_4474_);
v___x_4485_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_4485_, 0, v___x_4484_);
lean_ctor_set(v___x_4485_, 1, v_zetaDeltaSet_4474_);
lean_ctor_set(v___x_4485_, 2, v_lctx_4475_);
lean_ctor_set(v___x_4485_, 3, v_localInstances_4476_);
lean_ctor_set(v___x_4485_, 4, v_defEqCtx_x3f_4477_);
lean_ctor_set(v___x_4485_, 5, v_synthPendingDepth_4478_);
lean_ctor_set(v___x_4485_, 6, v_customCanUnfoldPredicate_x3f_4479_);
lean_ctor_set_uint8(v___x_4485_, sizeof(void*)*7, v_trackZetaDelta_4473_);
lean_ctor_set_uint8(v___x_4485_, sizeof(void*)*7 + 1, v_univApprox_4480_);
lean_ctor_set_uint8(v___x_4485_, sizeof(void*)*7 + 2, v_inTypeClassResolution_4481_);
lean_ctor_set_uint8(v___x_4485_, sizeof(void*)*7 + 3, v_cacheInferType_4482_);
lean_inc(v_snd_4471_);
lean_inc(v_snd_4469_);
v___x_4486_ = l_Lean_Meta_isExprDefEq(v_snd_4469_, v_snd_4471_, v___x_4485_, v_a_4455_, v_a_4456_, v_a_4457_);
lean_dec_ref_known(v___x_4485_, 7);
if (lean_obj_tag(v___x_4486_) == 0)
{
lean_object* v_a_4487_; lean_object* v___x_4489_; uint8_t v_isShared_4490_; uint8_t v_isSharedCheck_4566_; 
v_a_4487_ = lean_ctor_get(v___x_4486_, 0);
v_isSharedCheck_4566_ = !lean_is_exclusive(v___x_4486_);
if (v_isSharedCheck_4566_ == 0)
{
v___x_4489_ = v___x_4486_;
v_isShared_4490_ = v_isSharedCheck_4566_;
goto v_resetjp_4488_;
}
else
{
lean_inc(v_a_4487_);
lean_dec(v___x_4486_);
v___x_4489_ = lean_box(0);
v_isShared_4490_ = v_isSharedCheck_4566_;
goto v_resetjp_4488_;
}
v_resetjp_4488_:
{
uint8_t v___x_4491_; 
v___x_4491_ = lean_unbox(v_a_4487_);
lean_dec(v_a_4487_);
if (v___x_4491_ == 0)
{
lean_object* v___x_4492_; lean_object* v___x_4493_; lean_object* v___x_4494_; lean_object* v___y_4496_; uint8_t v___y_4497_; lean_object* v_a_4531_; lean_object* v___x_4534_; lean_object* v___x_4535_; lean_object* v___x_4536_; 
v___x_4492_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0));
v___x_4493_ = lean_box(0);
lean_inc(v_fst_4468_);
v___x_4494_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4494_, 0, v_fst_4468_);
lean_ctor_set(v___x_4494_, 1, v___x_4493_);
lean_inc_ref(v___x_4494_);
v___x_4534_ = l_Lean_Expr_const___override(v___x_4492_, v___x_4494_);
lean_inc(v_snd_4469_);
v___x_4535_ = l_Lean_Expr_app___override(v___x_4534_, v_snd_4469_);
v___x_4536_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4535_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4536_) == 0)
{
lean_object* v_a_4537_; lean_object* v___x_4538_; lean_object* v___x_4539_; lean_object* v___x_4540_; lean_object* v___x_4541_; lean_object* v___x_4542_; 
v_a_4537_ = lean_ctor_get(v___x_4536_, 0);
lean_inc(v_a_4537_);
lean_dec_ref_known(v___x_4536_, 1);
v___x_4538_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___closed__0));
lean_inc(v_fst_4470_);
v___x_4539_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4539_, 0, v_fst_4470_);
lean_ctor_set(v___x_4539_, 1, v___x_4493_);
lean_inc_ref(v___x_4539_);
v___x_4540_ = l_Lean_Expr_const___override(v___x_4538_, v___x_4539_);
lean_inc(v_snd_4471_);
v___x_4541_ = l_Lean_Expr_app___override(v___x_4540_, v_snd_4471_);
v___x_4542_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4541_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4542_) == 0)
{
lean_object* v_a_4543_; lean_object* v___x_4544_; lean_object* v___x_4545_; lean_object* v___x_4546_; lean_object* v___x_4547_; lean_object* v___x_4548_; lean_object* v___x_4549_; lean_object* v___x_4550_; lean_object* v___x_4551_; 
v_a_4543_ = lean_ctor_get(v___x_4542_, 0);
lean_inc(v_a_4543_);
lean_dec_ref_known(v___x_4542_, 1);
v___x_4544_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1));
lean_inc(v_fst_4468_);
v___x_4545_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4545_, 0, v_fst_4468_);
lean_ctor_set(v___x_4545_, 1, v___x_4539_);
v___x_4546_ = l_Lean_Expr_const___override(v___x_4544_, v___x_4545_);
lean_inc(v_snd_4469_);
v___x_4547_ = l_Lean_Expr_app___override(v___x_4546_, v_snd_4469_);
lean_inc(v_snd_4471_);
v___x_4548_ = l_Lean_Expr_app___override(v___x_4547_, v_snd_4471_);
v___x_4549_ = l_Lean_Expr_app___override(v___x_4548_, v_a_4537_);
v___x_4550_ = l_Lean_Expr_app___override(v___x_4549_, v_a_4543_);
v___x_4551_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4550_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4551_) == 0)
{
lean_object* v___x_4553_; uint8_t v_isShared_4554_; uint8_t v_isSharedCheck_4558_; 
lean_dec_ref_known(v___x_4494_, 2);
lean_del_object(v___x_4489_);
lean_dec_ref(v_r1_4452_);
v_isSharedCheck_4558_ = !lean_is_exclusive(v___x_4551_);
if (v_isSharedCheck_4558_ == 0)
{
lean_object* v_unused_4559_; 
v_unused_4559_ = lean_ctor_get(v___x_4551_, 0);
lean_dec(v_unused_4559_);
v___x_4553_ = v___x_4551_;
v_isShared_4554_ = v_isSharedCheck_4558_;
goto v_resetjp_4552_;
}
else
{
lean_dec(v___x_4551_);
v___x_4553_ = lean_box(0);
v_isShared_4554_ = v_isSharedCheck_4558_;
goto v_resetjp_4552_;
}
v_resetjp_4552_:
{
lean_object* v___x_4556_; 
if (v_isShared_4554_ == 0)
{
lean_ctor_set(v___x_4553_, 0, v_r2_4453_);
v___x_4556_ = v___x_4553_;
goto v_reusejp_4555_;
}
else
{
lean_object* v_reuseFailAlloc_4557_; 
v_reuseFailAlloc_4557_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4557_, 0, v_r2_4453_);
v___x_4556_ = v_reuseFailAlloc_4557_;
goto v_reusejp_4555_;
}
v_reusejp_4555_:
{
return v___x_4556_;
}
}
}
else
{
lean_object* v_a_4560_; 
lean_inc(v_snd_4471_);
lean_inc(v_fst_4470_);
lean_dec_ref(v_r2_4453_);
v_a_4560_ = lean_ctor_get(v___x_4551_, 0);
lean_inc(v_a_4560_);
lean_dec_ref_known(v___x_4551_, 1);
v_a_4531_ = v_a_4560_;
goto v___jp_4530_;
}
}
else
{
lean_object* v_a_4561_; 
lean_inc(v_snd_4471_);
lean_inc(v_fst_4470_);
lean_dec_ref_known(v___x_4539_, 2);
lean_dec(v_a_4537_);
lean_dec_ref(v_r2_4453_);
v_a_4561_ = lean_ctor_get(v___x_4542_, 0);
lean_inc(v_a_4561_);
lean_dec_ref_known(v___x_4542_, 1);
v_a_4531_ = v_a_4561_;
goto v___jp_4530_;
}
}
else
{
lean_object* v_a_4562_; 
lean_inc(v_snd_4471_);
lean_inc(v_fst_4470_);
lean_dec_ref(v_r2_4453_);
v_a_4562_ = lean_ctor_get(v___x_4536_, 0);
lean_inc(v_a_4562_);
lean_dec_ref_known(v___x_4536_, 1);
v_a_4531_ = v_a_4562_;
goto v___jp_4530_;
}
v___jp_4495_:
{
if (v___y_4497_ == 0)
{
lean_object* v___x_4498_; lean_object* v___x_4499_; lean_object* v___x_4500_; lean_object* v___x_4501_; 
lean_dec_ref(v___y_4496_);
lean_del_object(v___x_4489_);
lean_inc(v_fst_4470_);
v___x_4498_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4498_, 0, v_fst_4470_);
lean_ctor_set(v___x_4498_, 1, v___x_4493_);
v___x_4499_ = l_Lean_Expr_const___override(v___x_4492_, v___x_4498_);
lean_inc(v_snd_4471_);
v___x_4500_ = l_Lean_Expr_app___override(v___x_4499_, v_snd_4471_);
v___x_4501_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4500_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4501_) == 0)
{
lean_object* v_a_4502_; lean_object* v___x_4503_; lean_object* v___x_4504_; lean_object* v___x_4505_; lean_object* v___x_4506_; 
v_a_4502_ = lean_ctor_get(v___x_4501_, 0);
lean_inc(v_a_4502_);
lean_dec_ref_known(v___x_4501_, 1);
v___x_4503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___closed__0));
lean_inc_ref(v___x_4494_);
v___x_4504_ = l_Lean_Expr_const___override(v___x_4503_, v___x_4494_);
lean_inc(v_snd_4469_);
v___x_4505_ = l_Lean_Expr_app___override(v___x_4504_, v_snd_4469_);
v___x_4506_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4505_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4506_) == 0)
{
lean_object* v_a_4507_; lean_object* v___x_4508_; lean_object* v___x_4509_; lean_object* v___x_4510_; lean_object* v___x_4511_; lean_object* v___x_4512_; lean_object* v___x_4513_; lean_object* v___x_4514_; lean_object* v___x_4515_; 
v_a_4507_ = lean_ctor_get(v___x_4506_, 0);
lean_inc(v_a_4507_);
lean_dec_ref_known(v___x_4506_, 1);
v___x_4508_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1));
v___x_4509_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4509_, 0, v_fst_4470_);
lean_ctor_set(v___x_4509_, 1, v___x_4494_);
v___x_4510_ = l_Lean_Expr_const___override(v___x_4508_, v___x_4509_);
v___x_4511_ = l_Lean_Expr_app___override(v___x_4510_, v_snd_4471_);
lean_inc(v_snd_4469_);
v___x_4512_ = l_Lean_Expr_app___override(v___x_4511_, v_snd_4469_);
v___x_4513_ = l_Lean_Expr_app___override(v___x_4512_, v_a_4502_);
v___x_4514_ = l_Lean_Expr_app___override(v___x_4513_, v_a_4507_);
v___x_4515_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4514_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_);
if (lean_obj_tag(v___x_4515_) == 0)
{
lean_object* v___x_4517_; uint8_t v_isShared_4518_; uint8_t v_isSharedCheck_4522_; 
v_isSharedCheck_4522_ = !lean_is_exclusive(v___x_4515_);
if (v_isSharedCheck_4522_ == 0)
{
lean_object* v_unused_4523_; 
v_unused_4523_ = lean_ctor_get(v___x_4515_, 0);
lean_dec(v_unused_4523_);
v___x_4517_ = v___x_4515_;
v_isShared_4518_ = v_isSharedCheck_4522_;
goto v_resetjp_4516_;
}
else
{
lean_dec(v___x_4515_);
v___x_4517_ = lean_box(0);
v_isShared_4518_ = v_isSharedCheck_4522_;
goto v_resetjp_4516_;
}
v_resetjp_4516_:
{
lean_object* v___x_4520_; 
if (v_isShared_4518_ == 0)
{
lean_ctor_set(v___x_4517_, 0, v_r1_4452_);
v___x_4520_ = v___x_4517_;
goto v_reusejp_4519_;
}
else
{
lean_object* v_reuseFailAlloc_4521_; 
v_reuseFailAlloc_4521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4521_, 0, v_r1_4452_);
v___x_4520_ = v_reuseFailAlloc_4521_;
goto v_reusejp_4519_;
}
v_reusejp_4519_:
{
return v___x_4520_;
}
}
}
else
{
lean_object* v_a_4524_; 
v_a_4524_ = lean_ctor_get(v___x_4515_, 0);
lean_inc(v_a_4524_);
lean_dec_ref_known(v___x_4515_, 1);
v_a_4465_ = v_a_4524_;
goto v___jp_4464_;
}
}
else
{
lean_object* v_a_4525_; 
lean_dec(v_a_4502_);
lean_dec_ref_known(v___x_4494_, 2);
lean_dec(v_snd_4471_);
lean_dec(v_fst_4470_);
v_a_4525_ = lean_ctor_get(v___x_4506_, 0);
lean_inc(v_a_4525_);
lean_dec_ref_known(v___x_4506_, 1);
v_a_4465_ = v_a_4525_;
goto v___jp_4464_;
}
}
else
{
lean_object* v_a_4526_; 
lean_dec_ref_known(v___x_4494_, 2);
lean_dec(v_snd_4471_);
lean_dec(v_fst_4470_);
v_a_4526_ = lean_ctor_get(v___x_4501_, 0);
lean_inc(v_a_4526_);
lean_dec_ref_known(v___x_4501_, 1);
v_a_4465_ = v_a_4526_;
goto v___jp_4464_;
}
}
else
{
lean_object* v___x_4528_; 
lean_dec_ref_known(v___x_4494_, 2);
lean_dec(v_snd_4471_);
lean_dec(v_fst_4470_);
lean_dec_ref(v_r1_4452_);
if (v_isShared_4490_ == 0)
{
lean_ctor_set_tag(v___x_4489_, 1);
lean_ctor_set(v___x_4489_, 0, v___y_4496_);
v___x_4528_ = v___x_4489_;
goto v_reusejp_4527_;
}
else
{
lean_object* v_reuseFailAlloc_4529_; 
v_reuseFailAlloc_4529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4529_, 0, v___y_4496_);
v___x_4528_ = v_reuseFailAlloc_4529_;
goto v_reusejp_4527_;
}
v_reusejp_4527_:
{
return v___x_4528_;
}
}
}
v___jp_4530_:
{
uint8_t v___x_4532_; 
v___x_4532_ = l_Lean_Exception_isInterrupt(v_a_4531_);
if (v___x_4532_ == 0)
{
uint8_t v___x_4533_; 
lean_inc_ref(v_a_4531_);
v___x_4533_ = l_Lean_Exception_isRuntime(v_a_4531_);
v___y_4496_ = v_a_4531_;
v___y_4497_ = v___x_4533_;
goto v___jp_4495_;
}
else
{
v___y_4496_ = v_a_4531_;
v___y_4497_ = v___x_4532_;
goto v___jp_4495_;
}
}
}
else
{
lean_object* v___x_4564_; 
lean_dec_ref(v_r2_4453_);
if (v_isShared_4490_ == 0)
{
lean_ctor_set(v___x_4489_, 0, v_r1_4452_);
v___x_4564_ = v___x_4489_;
goto v_reusejp_4563_;
}
else
{
lean_object* v_reuseFailAlloc_4565_; 
v_reuseFailAlloc_4565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4565_, 0, v_r1_4452_);
v___x_4564_ = v_reuseFailAlloc_4565_;
goto v_reusejp_4563_;
}
v_reusejp_4563_:
{
return v___x_4564_;
}
}
}
}
else
{
lean_object* v_a_4567_; lean_object* v___x_4569_; uint8_t v_isShared_4570_; uint8_t v_isSharedCheck_4574_; 
lean_dec_ref(v_r2_4453_);
lean_dec_ref(v_r1_4452_);
v_a_4567_ = lean_ctor_get(v___x_4486_, 0);
v_isSharedCheck_4574_ = !lean_is_exclusive(v___x_4486_);
if (v_isSharedCheck_4574_ == 0)
{
v___x_4569_ = v___x_4486_;
v_isShared_4570_ = v_isSharedCheck_4574_;
goto v_resetjp_4568_;
}
else
{
lean_inc(v_a_4567_);
lean_dec(v___x_4486_);
v___x_4569_ = lean_box(0);
v_isShared_4570_ = v_isSharedCheck_4574_;
goto v_resetjp_4568_;
}
v_resetjp_4568_:
{
lean_object* v___x_4572_; 
if (v_isShared_4570_ == 0)
{
v___x_4572_ = v___x_4569_;
goto v_reusejp_4571_;
}
else
{
lean_object* v_reuseFailAlloc_4573_; 
v_reuseFailAlloc_4573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4573_, 0, v_a_4567_);
v___x_4572_ = v_reuseFailAlloc_4573_;
goto v_reusejp_4571_;
}
v_reusejp_4571_:
{
return v___x_4572_;
}
}
}
v___jp_4459_:
{
if (v___y_4461_ == 0)
{
lean_object* v___x_4462_; 
lean_dec_ref(v___y_4460_);
v___x_4462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4462_, 0, v_r1_4452_);
return v___x_4462_;
}
else
{
lean_object* v___x_4463_; 
lean_dec_ref(v_r1_4452_);
v___x_4463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4463_, 0, v___y_4460_);
return v___x_4463_;
}
}
v___jp_4464_:
{
uint8_t v___x_4466_; 
v___x_4466_ = l_Lean_Exception_isInterrupt(v_a_4465_);
if (v___x_4466_ == 0)
{
uint8_t v___x_4467_; 
lean_inc_ref(v_a_4465_);
v___x_4467_ = l_Lean_Exception_isRuntime(v_a_4465_);
v___y_4460_ = v_a_4465_;
v___y_4461_ = v___x_4467_;
goto v___jp_4459_;
}
else
{
v___y_4460_ = v_a_4465_;
v___y_4461_ = v___x_4466_;
goto v___jp_4459_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing___boxed(lean_object* v_r1_4575_, lean_object* v_r2_4576_, lean_object* v_a_4577_, lean_object* v_a_4578_, lean_object* v_a_4579_, lean_object* v_a_4580_, lean_object* v_a_4581_){
_start:
{
lean_object* v_res_4582_; 
v_res_4582_ = lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(v_r1_4575_, v_r2_4576_, v_a_4577_, v_a_4578_, v_a_4579_, v_a_4580_);
lean_dec(v_a_4580_);
lean_dec_ref(v_a_4579_);
lean_dec(v_a_4578_);
lean_dec_ref(v_a_4577_);
return v_res_4582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0(lean_object* v_x_4583_, lean_object* v_x_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_, lean_object* v___y_4587_, lean_object* v___y_4588_){
_start:
{
if (lean_obj_tag(v_x_4583_) == 0)
{
lean_object* v___x_4590_; lean_object* v___x_4591_; 
v___x_4590_ = l_List_reverse___redArg(v_x_4584_);
v___x_4591_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4591_, 0, v___x_4590_);
return v___x_4591_;
}
else
{
lean_object* v_head_4592_; lean_object* v_tail_4593_; lean_object* v___x_4595_; uint8_t v_isShared_4596_; uint8_t v_isSharedCheck_4611_; 
v_head_4592_ = lean_ctor_get(v_x_4583_, 0);
v_tail_4593_ = lean_ctor_get(v_x_4583_, 1);
v_isSharedCheck_4611_ = !lean_is_exclusive(v_x_4583_);
if (v_isSharedCheck_4611_ == 0)
{
v___x_4595_ = v_x_4583_;
v_isShared_4596_ = v_isSharedCheck_4611_;
goto v_resetjp_4594_;
}
else
{
lean_inc(v_tail_4593_);
lean_inc(v_head_4592_);
lean_dec(v_x_4583_);
v___x_4595_ = lean_box(0);
v_isShared_4596_ = v_isSharedCheck_4611_;
goto v_resetjp_4594_;
}
v_resetjp_4594_:
{
lean_object* v___x_4597_; 
v___x_4597_ = lp_mathlib_Qq_getLevelQ_x27(v_head_4592_, v___y_4585_, v___y_4586_, v___y_4587_, v___y_4588_);
if (lean_obj_tag(v___x_4597_) == 0)
{
lean_object* v_a_4598_; lean_object* v___x_4600_; 
v_a_4598_ = lean_ctor_get(v___x_4597_, 0);
lean_inc(v_a_4598_);
lean_dec_ref_known(v___x_4597_, 1);
if (v_isShared_4596_ == 0)
{
lean_ctor_set(v___x_4595_, 1, v_x_4584_);
lean_ctor_set(v___x_4595_, 0, v_a_4598_);
v___x_4600_ = v___x_4595_;
goto v_reusejp_4599_;
}
else
{
lean_object* v_reuseFailAlloc_4602_; 
v_reuseFailAlloc_4602_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4602_, 0, v_a_4598_);
lean_ctor_set(v_reuseFailAlloc_4602_, 1, v_x_4584_);
v___x_4600_ = v_reuseFailAlloc_4602_;
goto v_reusejp_4599_;
}
v_reusejp_4599_:
{
v_x_4583_ = v_tail_4593_;
v_x_4584_ = v___x_4600_;
goto _start;
}
}
else
{
lean_object* v_a_4603_; lean_object* v___x_4605_; uint8_t v_isShared_4606_; uint8_t v_isSharedCheck_4610_; 
lean_del_object(v___x_4595_);
lean_dec(v_tail_4593_);
lean_dec(v_x_4584_);
v_a_4603_ = lean_ctor_get(v___x_4597_, 0);
v_isSharedCheck_4610_ = !lean_is_exclusive(v___x_4597_);
if (v_isSharedCheck_4610_ == 0)
{
v___x_4605_ = v___x_4597_;
v_isShared_4606_ = v_isSharedCheck_4610_;
goto v_resetjp_4604_;
}
else
{
lean_inc(v_a_4603_);
lean_dec(v___x_4597_);
v___x_4605_ = lean_box(0);
v_isShared_4606_ = v_isSharedCheck_4610_;
goto v_resetjp_4604_;
}
v_resetjp_4604_:
{
lean_object* v___x_4608_; 
if (v_isShared_4606_ == 0)
{
v___x_4608_ = v___x_4605_;
goto v_reusejp_4607_;
}
else
{
lean_object* v_reuseFailAlloc_4609_; 
v_reuseFailAlloc_4609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4609_, 0, v_a_4603_);
v___x_4608_ = v_reuseFailAlloc_4609_;
goto v_reusejp_4607_;
}
v_reusejp_4607_:
{
return v___x_4608_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0___boxed(lean_object* v_x_4612_, lean_object* v_x_4613_, lean_object* v___y_4614_, lean_object* v___y_4615_, lean_object* v___y_4616_, lean_object* v___y_4617_, lean_object* v___y_4618_){
_start:
{
lean_object* v_res_4619_; 
v_res_4619_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0(v_x_4612_, v_x_4613_, v___y_4614_, v___y_4615_, v___y_4616_, v___y_4617_);
lean_dec(v___y_4617_);
lean_dec_ref(v___y_4616_);
lean_dec(v___y_4615_);
lean_dec_ref(v___y_4614_);
return v_res_4619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1(lean_object* v_x_4620_, lean_object* v_x_4621_, lean_object* v___y_4622_, lean_object* v___y_4623_, lean_object* v___y_4624_, lean_object* v___y_4625_){
_start:
{
if (lean_obj_tag(v_x_4621_) == 0)
{
lean_object* v___x_4627_; 
v___x_4627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4627_, 0, v_x_4620_);
return v___x_4627_;
}
else
{
lean_object* v_head_4628_; lean_object* v_tail_4629_; lean_object* v___x_4630_; 
v_head_4628_ = lean_ctor_get(v_x_4621_, 0);
lean_inc(v_head_4628_);
v_tail_4629_ = lean_ctor_get(v_x_4621_, 1);
lean_inc(v_tail_4629_);
lean_dec_ref_known(v_x_4621_, 2);
v___x_4630_ = lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(v_x_4620_, v_head_4628_, v___y_4622_, v___y_4623_, v___y_4624_, v___y_4625_);
if (lean_obj_tag(v___x_4630_) == 0)
{
lean_object* v_a_4631_; 
v_a_4631_ = lean_ctor_get(v___x_4630_, 0);
lean_inc(v_a_4631_);
lean_dec_ref_known(v___x_4630_, 1);
v_x_4620_ = v_a_4631_;
v_x_4621_ = v_tail_4629_;
goto _start;
}
else
{
lean_dec(v_tail_4629_);
return v___x_4630_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1___boxed(lean_object* v_x_4633_, lean_object* v_x_4634_, lean_object* v___y_4635_, lean_object* v___y_4636_, lean_object* v___y_4637_, lean_object* v___y_4638_, lean_object* v___y_4639_){
_start:
{
lean_object* v_res_4640_; 
v_res_4640_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1(v_x_4633_, v_x_4634_, v___y_4635_, v___y_4636_, v___y_4637_, v___y_4638_);
lean_dec(v___y_4638_);
lean_dec_ref(v___y_4637_);
lean_dec(v___y_4636_);
lean_dec_ref(v___y_4635_);
return v_res_4640_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0(void){
_start:
{
lean_object* v___x_4641_; lean_object* v___x_4642_; 
v___x_4641_ = lean_unsigned_to_nat(0u);
v___x_4642_ = l_Lean_Level_ofNat(v___x_4641_);
return v___x_4642_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2(void){
_start:
{
lean_object* v___x_4645_; lean_object* v___x_4646_; lean_object* v___x_4647_; 
v___x_4645_ = lean_box(0);
v___x_4646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__1));
v___x_4647_ = l_Lean_Expr_const___override(v___x_4646_, v___x_4645_);
return v___x_4647_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3(void){
_start:
{
lean_object* v___x_4648_; lean_object* v___x_4649_; lean_object* v___x_4650_; 
v___x_4648_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__2);
v___x_4649_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0);
v___x_4650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4650_, 0, v___x_4649_);
lean_ctor_set(v___x_4650_, 1, v___x_4648_);
return v___x_4650_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5(void){
_start:
{
lean_object* v___x_4653_; lean_object* v___x_4654_; lean_object* v___x_4655_; 
v___x_4653_ = lean_box(0);
v___x_4654_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__4));
v___x_4655_ = l_Lean_Expr_const___override(v___x_4654_, v___x_4653_);
return v___x_4655_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6(void){
_start:
{
lean_object* v___x_4656_; lean_object* v___x_4657_; lean_object* v___x_4658_; 
v___x_4656_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__5);
v___x_4657_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0);
v___x_4658_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4658_, 0, v___x_4657_);
lean_ctor_set(v___x_4658_, 1, v___x_4656_);
return v___x_4658_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9(void){
_start:
{
lean_object* v___x_4662_; lean_object* v___x_4663_; lean_object* v___x_4664_; 
v___x_4662_ = lean_box(0);
v___x_4663_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__8));
v___x_4664_ = l_Lean_Expr_const___override(v___x_4663_, v___x_4662_);
return v___x_4664_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10(void){
_start:
{
lean_object* v___x_4665_; lean_object* v___x_4666_; lean_object* v___x_4667_; 
v___x_4665_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__9);
v___x_4666_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__0);
v___x_4667_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4667_, 0, v___x_4666_);
lean_ctor_set(v___x_4667_, 1, v___x_4665_);
return v___x_4667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg(lean_object* v_ca_4668_, lean_object* v_e_4669_, lean_object* v_a_4670_, lean_object* v_a_4671_, lean_object* v_a_4672_, lean_object* v_a_4673_){
_start:
{
lean_object* v___x_4681_; 
v___x_4681_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(v_e_4669_, v_a_4670_, v_a_4671_, v_a_4672_, v_a_4673_);
if (lean_obj_tag(v___x_4681_) == 0)
{
lean_object* v_a_4682_; lean_object* v___x_4683_; lean_object* v___x_4684_; 
v_a_4682_ = lean_ctor_get(v___x_4681_, 0);
lean_inc(v_a_4682_);
lean_dec_ref_known(v___x_4681_, 1);
v___x_4683_ = lean_box(0);
v___x_4684_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Algebra_inferBase_spec__0(v_a_4682_, v___x_4683_, v_a_4670_, v_a_4671_, v_a_4672_, v_a_4673_);
if (lean_obj_tag(v___x_4684_) == 0)
{
lean_object* v_a_4685_; lean_object* v___x_4687_; uint8_t v_isShared_4688_; uint8_t v_isSharedCheck_4702_; 
v_a_4685_ = lean_ctor_get(v___x_4684_, 0);
v_isSharedCheck_4702_ = !lean_is_exclusive(v___x_4684_);
if (v_isSharedCheck_4702_ == 0)
{
v___x_4687_ = v___x_4684_;
v_isShared_4688_ = v_isSharedCheck_4702_;
goto v_resetjp_4686_;
}
else
{
lean_inc(v_a_4685_);
lean_dec(v___x_4684_);
v___x_4687_ = lean_box(0);
v_isShared_4688_ = v_isSharedCheck_4702_;
goto v_resetjp_4686_;
}
v_resetjp_4686_:
{
if (lean_obj_tag(v_a_4685_) == 0)
{
lean_object* v_field_4689_; 
v_field_4689_ = lean_ctor_get(v_ca_4668_, 1);
if (lean_obj_tag(v_field_4689_) == 1)
{
lean_object* v_toCache_4690_; lean_object* v_cz_u03b1_4691_; 
v_toCache_4690_ = lean_ctor_get(v_ca_4668_, 0);
v_cz_u03b1_4691_ = lean_ctor_get(v_toCache_4690_, 2);
if (lean_obj_tag(v_cz_u03b1_4691_) == 1)
{
lean_object* v___x_4692_; lean_object* v___x_4694_; 
v___x_4692_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__10);
if (v_isShared_4688_ == 0)
{
lean_ctor_set(v___x_4687_, 0, v___x_4692_);
v___x_4694_ = v___x_4687_;
goto v_reusejp_4693_;
}
else
{
lean_object* v_reuseFailAlloc_4695_; 
v_reuseFailAlloc_4695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4695_, 0, v___x_4692_);
v___x_4694_ = v_reuseFailAlloc_4695_;
goto v_reusejp_4693_;
}
v_reusejp_4693_:
{
return v___x_4694_;
}
}
else
{
lean_object* v_r_u03b1_4696_; 
lean_del_object(v___x_4687_);
v_r_u03b1_4696_ = lean_ctor_get(v_toCache_4690_, 0);
if (lean_obj_tag(v_r_u03b1_4696_) == 1)
{
goto v___jp_4675_;
}
else
{
goto v___jp_4678_;
}
}
}
else
{
lean_object* v_toCache_4697_; lean_object* v_r_u03b1_4698_; 
lean_del_object(v___x_4687_);
v_toCache_4697_ = lean_ctor_get(v_ca_4668_, 0);
v_r_u03b1_4698_ = lean_ctor_get(v_toCache_4697_, 0);
if (lean_obj_tag(v_r_u03b1_4698_) == 1)
{
goto v___jp_4675_;
}
else
{
goto v___jp_4678_;
}
}
}
else
{
lean_object* v_head_4699_; lean_object* v_tail_4700_; lean_object* v___x_4701_; 
lean_del_object(v___x_4687_);
v_head_4699_ = lean_ctor_get(v_a_4685_, 0);
lean_inc(v_head_4699_);
v_tail_4700_ = lean_ctor_get(v_a_4685_, 1);
lean_inc(v_tail_4700_);
lean_dec_ref_known(v_a_4685_, 2);
v___x_4701_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Algebra_inferBase_spec__1(v_head_4699_, v_tail_4700_, v_a_4670_, v_a_4671_, v_a_4672_, v_a_4673_);
return v___x_4701_;
}
}
}
else
{
lean_object* v_a_4703_; lean_object* v___x_4705_; uint8_t v_isShared_4706_; uint8_t v_isSharedCheck_4710_; 
v_a_4703_ = lean_ctor_get(v___x_4684_, 0);
v_isSharedCheck_4710_ = !lean_is_exclusive(v___x_4684_);
if (v_isSharedCheck_4710_ == 0)
{
v___x_4705_ = v___x_4684_;
v_isShared_4706_ = v_isSharedCheck_4710_;
goto v_resetjp_4704_;
}
else
{
lean_inc(v_a_4703_);
lean_dec(v___x_4684_);
v___x_4705_ = lean_box(0);
v_isShared_4706_ = v_isSharedCheck_4710_;
goto v_resetjp_4704_;
}
v_resetjp_4704_:
{
lean_object* v___x_4708_; 
if (v_isShared_4706_ == 0)
{
v___x_4708_ = v___x_4705_;
goto v_reusejp_4707_;
}
else
{
lean_object* v_reuseFailAlloc_4709_; 
v_reuseFailAlloc_4709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4709_, 0, v_a_4703_);
v___x_4708_ = v_reuseFailAlloc_4709_;
goto v_reusejp_4707_;
}
v_reusejp_4707_:
{
return v___x_4708_;
}
}
}
}
else
{
lean_object* v_a_4711_; lean_object* v___x_4713_; uint8_t v_isShared_4714_; uint8_t v_isSharedCheck_4718_; 
v_a_4711_ = lean_ctor_get(v___x_4681_, 0);
v_isSharedCheck_4718_ = !lean_is_exclusive(v___x_4681_);
if (v_isSharedCheck_4718_ == 0)
{
v___x_4713_ = v___x_4681_;
v_isShared_4714_ = v_isSharedCheck_4718_;
goto v_resetjp_4712_;
}
else
{
lean_inc(v_a_4711_);
lean_dec(v___x_4681_);
v___x_4713_ = lean_box(0);
v_isShared_4714_ = v_isSharedCheck_4718_;
goto v_resetjp_4712_;
}
v_resetjp_4712_:
{
lean_object* v___x_4716_; 
if (v_isShared_4714_ == 0)
{
v___x_4716_ = v___x_4713_;
goto v_reusejp_4715_;
}
else
{
lean_object* v_reuseFailAlloc_4717_; 
v_reuseFailAlloc_4717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4717_, 0, v_a_4711_);
v___x_4716_ = v_reuseFailAlloc_4717_;
goto v_reusejp_4715_;
}
v_reusejp_4715_:
{
return v___x_4716_;
}
}
}
v___jp_4675_:
{
lean_object* v___x_4676_; lean_object* v___x_4677_; 
v___x_4676_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__3);
v___x_4677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4677_, 0, v___x_4676_);
return v___x_4677_;
}
v___jp_4678_:
{
lean_object* v___x_4679_; lean_object* v___x_4680_; 
v___x_4679_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6, &lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___closed__6);
v___x_4680_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4680_, 0, v___x_4679_);
return v___x_4680_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg___boxed(lean_object* v_ca_4719_, lean_object* v_e_4720_, lean_object* v_a_4721_, lean_object* v_a_4722_, lean_object* v_a_4723_, lean_object* v_a_4724_, lean_object* v_a_4725_){
_start:
{
lean_object* v_res_4726_; 
v_res_4726_ = lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg(v_ca_4719_, v_e_4720_, v_a_4721_, v_a_4722_, v_a_4723_, v_a_4724_);
lean_dec(v_a_4724_);
lean_dec_ref(v_a_4723_);
lean_dec(v_a_4722_);
lean_dec_ref(v_a_4721_);
lean_dec_ref(v_ca_4719_);
return v_res_4726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase(lean_object* v_v_4727_, lean_object* v_A_4728_, lean_object* v_sA_4729_, lean_object* v_ca_4730_, lean_object* v_e_4731_, lean_object* v_a_4732_, lean_object* v_a_4733_, lean_object* v_a_4734_, lean_object* v_a_4735_){
_start:
{
lean_object* v___x_4737_; 
v___x_4737_ = lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg(v_ca_4730_, v_e_4731_, v_a_4732_, v_a_4733_, v_a_4734_, v_a_4735_);
return v___x_4737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_inferBase___boxed(lean_object* v_v_4738_, lean_object* v_A_4739_, lean_object* v_sA_4740_, lean_object* v_ca_4741_, lean_object* v_e_4742_, lean_object* v_a_4743_, lean_object* v_a_4744_, lean_object* v_a_4745_, lean_object* v_a_4746_, lean_object* v_a_4747_){
_start:
{
lean_object* v_res_4748_; 
v_res_4748_ = lp_mathlib_Mathlib_Tactic_Algebra_inferBase(v_v_4738_, v_A_4739_, v_sA_4740_, v_ca_4741_, v_e_4742_, v_a_4743_, v_a_4744_, v_a_4745_, v_a_4746_);
lean_dec(v_a_4746_);
lean_dec_ref(v_a_4745_);
lean_dec(v_a_4744_);
lean_dec_ref(v_a_4743_);
lean_dec_ref(v_ca_4741_);
lean_dec_ref(v_sA_4740_);
lean_dec_ref(v_A_4739_);
lean_dec(v_v_4738_);
return v_res_4748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg(lean_object* v_category_4749_, lean_object* v_opts_4750_, lean_object* v_act_4751_, lean_object* v_decl_4752_, lean_object* v___y_4753_, lean_object* v___y_4754_, lean_object* v___y_4755_, lean_object* v___y_4756_, lean_object* v___y_4757_, lean_object* v___y_4758_){
_start:
{
lean_object* v___x_4760_; lean_object* v___x_4761_; 
lean_inc(v___y_4758_);
lean_inc_ref(v___y_4757_);
lean_inc(v___y_4756_);
lean_inc_ref(v___y_4755_);
lean_inc(v___y_4754_);
lean_inc_ref(v___y_4753_);
v___x_4760_ = lean_apply_6(v_act_4751_, v___y_4753_, v___y_4754_, v___y_4755_, v___y_4756_, v___y_4757_, v___y_4758_);
v___x_4761_ = l_Lean_profileitIOUnsafe___redArg(v_category_4749_, v_opts_4750_, v___x_4760_, v_decl_4752_);
return v___x_4761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg___boxed(lean_object* v_category_4762_, lean_object* v_opts_4763_, lean_object* v_act_4764_, lean_object* v_decl_4765_, lean_object* v___y_4766_, lean_object* v___y_4767_, lean_object* v___y_4768_, lean_object* v___y_4769_, lean_object* v___y_4770_, lean_object* v___y_4771_, lean_object* v___y_4772_){
_start:
{
lean_object* v_res_4773_; 
v_res_4773_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg(v_category_4762_, v_opts_4763_, v_act_4764_, v_decl_4765_, v___y_4766_, v___y_4767_, v___y_4768_, v___y_4769_, v___y_4770_, v___y_4771_);
lean_dec(v___y_4771_);
lean_dec_ref(v___y_4770_);
lean_dec(v___y_4769_);
lean_dec_ref(v___y_4768_);
lean_dec(v___y_4767_);
lean_dec_ref(v___y_4766_);
lean_dec_ref(v_opts_4763_);
lean_dec_ref(v_category_4762_);
return v_res_4773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1(lean_object* v_00_u03b1_4774_, lean_object* v_category_4775_, lean_object* v_opts_4776_, lean_object* v_act_4777_, lean_object* v_decl_4778_, lean_object* v___y_4779_, lean_object* v___y_4780_, lean_object* v___y_4781_, lean_object* v___y_4782_, lean_object* v___y_4783_, lean_object* v___y_4784_){
_start:
{
lean_object* v___x_4786_; 
v___x_4786_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg(v_category_4775_, v_opts_4776_, v_act_4777_, v_decl_4778_, v___y_4779_, v___y_4780_, v___y_4781_, v___y_4782_, v___y_4783_, v___y_4784_);
return v___x_4786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___boxed(lean_object* v_00_u03b1_4787_, lean_object* v_category_4788_, lean_object* v_opts_4789_, lean_object* v_act_4790_, lean_object* v_decl_4791_, lean_object* v___y_4792_, lean_object* v___y_4793_, lean_object* v___y_4794_, lean_object* v___y_4795_, lean_object* v___y_4796_, lean_object* v___y_4797_, lean_object* v___y_4798_){
_start:
{
lean_object* v_res_4799_; 
v_res_4799_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1(v_00_u03b1_4787_, v_category_4788_, v_opts_4789_, v_act_4790_, v_decl_4791_, v___y_4792_, v___y_4793_, v___y_4794_, v___y_4795_, v___y_4796_, v___y_4797_);
lean_dec(v___y_4797_);
lean_dec_ref(v___y_4796_);
lean_dec(v___y_4795_);
lean_dec_ref(v___y_4794_);
lean_dec(v___y_4793_);
lean_dec_ref(v___y_4792_);
lean_dec_ref(v_opts_4789_);
lean_dec_ref(v_category_4788_);
return v_res_4799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(lean_object* v_msg_4800_, lean_object* v___y_4801_, lean_object* v___y_4802_, lean_object* v___y_4803_, lean_object* v___y_4804_){
_start:
{
lean_object* v_ref_4806_; lean_object* v___x_4807_; lean_object* v_a_4808_; lean_object* v___x_4810_; uint8_t v_isShared_4811_; uint8_t v_isSharedCheck_4816_; 
v_ref_4806_ = lean_ctor_get(v___y_4803_, 5);
v___x_4807_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Algebra_RingCompute_neg_spec__0_spec__0(v_msg_4800_, v___y_4801_, v___y_4802_, v___y_4803_, v___y_4804_);
v_a_4808_ = lean_ctor_get(v___x_4807_, 0);
v_isSharedCheck_4816_ = !lean_is_exclusive(v___x_4807_);
if (v_isSharedCheck_4816_ == 0)
{
v___x_4810_ = v___x_4807_;
v_isShared_4811_ = v_isSharedCheck_4816_;
goto v_resetjp_4809_;
}
else
{
lean_inc(v_a_4808_);
lean_dec(v___x_4807_);
v___x_4810_ = lean_box(0);
v_isShared_4811_ = v_isSharedCheck_4816_;
goto v_resetjp_4809_;
}
v_resetjp_4809_:
{
lean_object* v___x_4812_; lean_object* v___x_4814_; 
lean_inc(v_ref_4806_);
v___x_4812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4812_, 0, v_ref_4806_);
lean_ctor_set(v___x_4812_, 1, v_a_4808_);
if (v_isShared_4811_ == 0)
{
lean_ctor_set_tag(v___x_4810_, 1);
lean_ctor_set(v___x_4810_, 0, v___x_4812_);
v___x_4814_ = v___x_4810_;
goto v_reusejp_4813_;
}
else
{
lean_object* v_reuseFailAlloc_4815_; 
v_reuseFailAlloc_4815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4815_, 0, v___x_4812_);
v___x_4814_ = v_reuseFailAlloc_4815_;
goto v_reusejp_4813_;
}
v_reusejp_4813_:
{
return v___x_4814_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg___boxed(lean_object* v_msg_4817_, lean_object* v___y_4818_, lean_object* v___y_4819_, lean_object* v___y_4820_, lean_object* v___y_4821_, lean_object* v___y_4822_){
_start:
{
lean_object* v_res_4823_; 
v_res_4823_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(v_msg_4817_, v___y_4818_, v___y_4819_, v___y_4820_, v___y_4821_);
lean_dec(v___y_4821_);
lean_dec_ref(v___y_4820_);
lean_dec(v___y_4819_);
lean_dec_ref(v___y_4818_);
return v_res_4823_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1(void){
_start:
{
lean_object* v___x_4825_; lean_object* v___x_4826_; 
v___x_4825_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__0));
v___x_4826_ = l_Lean_stringToMessageData(v___x_4825_);
return v___x_4826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0(lean_object* v___x_4827_, lean_object* v_v_4828_, lean_object* v_A_4829_, lean_object* v_sA_4830_, lean_object* v___x_4831_, lean_object* v_toCache_4832_, lean_object* v_e_u2081_4833_, lean_object* v_e_u2082_4834_, lean_object* v___y_4835_, lean_object* v___y_4836_, lean_object* v___y_4837_, lean_object* v___y_4838_, lean_object* v___y_4839_, lean_object* v___y_4840_){
_start:
{
lean_object* v___x_4842_; 
lean_inc_ref(v_e_u2081_4833_);
lean_inc_ref(v_toCache_4832_);
lean_inc_ref(v___x_4831_);
lean_inc_ref(v_sA_4830_);
lean_inc_ref(v_A_4829_);
lean_inc(v_v_4828_);
lean_inc_ref(v___x_4827_);
v___x_4842_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_4827_, v_v_4828_, v_A_4829_, v_sA_4830_, v___x_4831_, v_toCache_4832_, v_e_u2081_4833_, v___y_4835_, v___y_4836_, v___y_4837_, v___y_4838_, v___y_4839_, v___y_4840_);
if (lean_obj_tag(v___x_4842_) == 0)
{
lean_object* v_a_4843_; lean_object* v_expr_4844_; lean_object* v_val_4845_; lean_object* v_proof_4846_; lean_object* v___x_4847_; 
v_a_4843_ = lean_ctor_get(v___x_4842_, 0);
lean_inc(v_a_4843_);
lean_dec_ref_known(v___x_4842_, 1);
v_expr_4844_ = lean_ctor_get(v_a_4843_, 0);
lean_inc_ref(v_expr_4844_);
v_val_4845_ = lean_ctor_get(v_a_4843_, 1);
lean_inc(v_val_4845_);
v_proof_4846_ = lean_ctor_get(v_a_4843_, 2);
lean_inc_ref(v_proof_4846_);
lean_dec(v_a_4843_);
lean_inc_ref(v_e_u2082_4834_);
lean_inc_ref(v___x_4831_);
lean_inc_ref(v_sA_4830_);
lean_inc_ref(v_A_4829_);
lean_inc(v_v_4828_);
lean_inc_ref(v___x_4827_);
v___x_4847_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_4827_, v_v_4828_, v_A_4829_, v_sA_4830_, v___x_4831_, v_toCache_4832_, v_e_u2082_4834_, v___y_4835_, v___y_4836_, v___y_4837_, v___y_4838_, v___y_4839_, v___y_4840_);
if (lean_obj_tag(v___x_4847_) == 0)
{
lean_object* v_a_4848_; lean_object* v___x_4850_; uint8_t v_isShared_4851_; uint8_t v_isSharedCheck_4937_; 
v_a_4848_ = lean_ctor_get(v___x_4847_, 0);
v_isSharedCheck_4937_ = !lean_is_exclusive(v___x_4847_);
if (v_isSharedCheck_4937_ == 0)
{
v___x_4850_ = v___x_4847_;
v_isShared_4851_ = v_isSharedCheck_4937_;
goto v_resetjp_4849_;
}
else
{
lean_inc(v_a_4848_);
lean_dec(v___x_4847_);
v___x_4850_ = lean_box(0);
v_isShared_4851_ = v_isSharedCheck_4937_;
goto v_resetjp_4849_;
}
v_resetjp_4849_:
{
lean_object* v_expr_4852_; lean_object* v_val_4853_; lean_object* v_proof_4854_; lean_object* v_toRingCompare_4892_; lean_object* v_toRingCompare_4893_; uint8_t v___x_4894_; 
v_expr_4852_ = lean_ctor_get(v_a_4848_, 0);
lean_inc_ref(v_expr_4852_);
v_val_4853_ = lean_ctor_get(v_a_4848_, 1);
lean_inc(v_val_4853_);
v_proof_4854_ = lean_ctor_get(v_a_4848_, 2);
lean_inc_ref(v_proof_4854_);
lean_dec(v_a_4848_);
v_toRingCompare_4892_ = lean_ctor_get(v___x_4827_, 0);
lean_inc_ref(v_toRingCompare_4892_);
lean_dec_ref(v___x_4827_);
v_toRingCompare_4893_ = lean_ctor_get(v___x_4831_, 0);
lean_inc_ref(v_toRingCompare_4893_);
lean_dec_ref(v___x_4831_);
v___x_4894_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_eq___redArg(v_toRingCompare_4892_, v_v_4828_, v_A_4829_, v_sA_4830_, v_toRingCompare_4893_, v_val_4845_, v_val_4853_);
lean_dec_ref(v_sA_4830_);
if (v___x_4894_ == 0)
{
lean_object* v___x_4895_; lean_object* v___x_4896_; lean_object* v___x_4897_; lean_object* v___x_4898_; lean_object* v___x_4899_; lean_object* v___x_4900_; lean_object* v___x_4901_; lean_object* v___x_4902_; lean_object* v___x_4903_; lean_object* v___x_4904_; lean_object* v___x_4905_; 
lean_dec_ref(v_proof_4854_);
lean_del_object(v___x_4850_);
lean_dec_ref(v_proof_4846_);
lean_dec_ref(v_e_u2082_4834_);
lean_dec_ref(v_e_u2081_4833_);
v___x_4895_ = lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
v___x_4896_ = lean_st_ref_get(v___x_4895_);
v___x_4897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_4898_ = l_Lean_Level_succ___override(v_v_4828_);
v___x_4899_ = lean_box(0);
v___x_4900_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4900_, 0, v___x_4898_);
lean_ctor_set(v___x_4900_, 1, v___x_4899_);
v___x_4901_ = l_Lean_Expr_const___override(v___x_4897_, v___x_4900_);
v___x_4902_ = l_Lean_Expr_app___override(v___x_4901_, v_A_4829_);
v___x_4903_ = l_Lean_Expr_app___override(v___x_4902_, v_expr_4844_);
v___x_4904_ = l_Lean_Expr_app___override(v___x_4903_, v_expr_4852_);
lean_inc(v___y_4840_);
lean_inc_ref(v___y_4839_);
lean_inc(v___y_4838_);
lean_inc_ref(v___y_4837_);
v___x_4905_ = lean_apply_6(v___x_4896_, v___x_4904_, v___y_4837_, v___y_4838_, v___y_4839_, v___y_4840_, lean_box(0));
if (lean_obj_tag(v___x_4905_) == 0)
{
lean_object* v_a_4906_; lean_object* v___x_4908_; uint8_t v_isShared_4909_; uint8_t v_isSharedCheck_4936_; 
v_a_4906_ = lean_ctor_get(v___x_4905_, 0);
v_isSharedCheck_4936_ = !lean_is_exclusive(v___x_4905_);
if (v_isSharedCheck_4936_ == 0)
{
v___x_4908_ = v___x_4905_;
v_isShared_4909_ = v_isSharedCheck_4936_;
goto v_resetjp_4907_;
}
else
{
lean_inc(v_a_4906_);
lean_dec(v___x_4905_);
v___x_4908_ = lean_box(0);
v_isShared_4909_ = v_isSharedCheck_4936_;
goto v_resetjp_4907_;
}
v_resetjp_4907_:
{
lean_object* v___x_4911_; 
if (v_isShared_4909_ == 0)
{
lean_ctor_set_tag(v___x_4908_, 1);
v___x_4911_ = v___x_4908_;
goto v_reusejp_4910_;
}
else
{
lean_object* v_reuseFailAlloc_4935_; 
v_reuseFailAlloc_4935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4935_, 0, v_a_4906_);
v___x_4911_ = v_reuseFailAlloc_4935_;
goto v_reusejp_4910_;
}
v_reusejp_4910_:
{
uint8_t v___x_4912_; lean_object* v___x_4913_; lean_object* v___x_4914_; 
v___x_4912_ = 0;
v___x_4913_ = lean_box(0);
v___x_4914_ = l_Lean_Meta_mkFreshExprMVar(v___x_4911_, v___x_4912_, v___x_4913_, v___y_4837_, v___y_4838_, v___y_4839_, v___y_4840_);
if (lean_obj_tag(v___x_4914_) == 0)
{
lean_object* v_a_4915_; lean_object* v___x_4917_; uint8_t v_isShared_4918_; uint8_t v_isSharedCheck_4934_; 
v_a_4915_ = lean_ctor_get(v___x_4914_, 0);
v_isSharedCheck_4934_ = !lean_is_exclusive(v___x_4914_);
if (v_isSharedCheck_4934_ == 0)
{
v___x_4917_ = v___x_4914_;
v_isShared_4918_ = v_isSharedCheck_4934_;
goto v_resetjp_4916_;
}
else
{
lean_inc(v_a_4915_);
lean_dec(v___x_4914_);
v___x_4917_ = lean_box(0);
v_isShared_4918_ = v_isSharedCheck_4934_;
goto v_resetjp_4916_;
}
v_resetjp_4916_:
{
lean_object* v___x_4919_; lean_object* v___x_4920_; lean_object* v___x_4922_; 
v___x_4919_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___closed__1);
v___x_4920_ = l_Lean_Expr_mvarId_x21(v_a_4915_);
lean_dec(v_a_4915_);
if (v_isShared_4918_ == 0)
{
lean_ctor_set_tag(v___x_4917_, 1);
lean_ctor_set(v___x_4917_, 0, v___x_4920_);
v___x_4922_ = v___x_4917_;
goto v_reusejp_4921_;
}
else
{
lean_object* v_reuseFailAlloc_4933_; 
v_reuseFailAlloc_4933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4933_, 0, v___x_4920_);
v___x_4922_ = v_reuseFailAlloc_4933_;
goto v_reusejp_4921_;
}
v_reusejp_4921_:
{
lean_object* v___x_4923_; lean_object* v___x_4924_; lean_object* v_a_4925_; lean_object* v___x_4927_; uint8_t v_isShared_4928_; uint8_t v_isSharedCheck_4932_; 
v___x_4923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4923_, 0, v___x_4919_);
lean_ctor_set(v___x_4923_, 1, v___x_4922_);
v___x_4924_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(v___x_4923_, v___y_4837_, v___y_4838_, v___y_4839_, v___y_4840_);
v_a_4925_ = lean_ctor_get(v___x_4924_, 0);
v_isSharedCheck_4932_ = !lean_is_exclusive(v___x_4924_);
if (v_isSharedCheck_4932_ == 0)
{
v___x_4927_ = v___x_4924_;
v_isShared_4928_ = v_isSharedCheck_4932_;
goto v_resetjp_4926_;
}
else
{
lean_inc(v_a_4925_);
lean_dec(v___x_4924_);
v___x_4927_ = lean_box(0);
v_isShared_4928_ = v_isSharedCheck_4932_;
goto v_resetjp_4926_;
}
v_resetjp_4926_:
{
lean_object* v___x_4930_; 
if (v_isShared_4928_ == 0)
{
v___x_4930_ = v___x_4927_;
goto v_reusejp_4929_;
}
else
{
lean_object* v_reuseFailAlloc_4931_; 
v_reuseFailAlloc_4931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4931_, 0, v_a_4925_);
v___x_4930_ = v_reuseFailAlloc_4931_;
goto v_reusejp_4929_;
}
v_reusejp_4929_:
{
return v___x_4930_;
}
}
}
}
}
else
{
return v___x_4914_;
}
}
}
}
else
{
return v___x_4905_;
}
}
else
{
lean_dec_ref(v_expr_4844_);
goto v___jp_4855_;
}
v___jp_4855_:
{
lean_object* v___x_4856_; lean_object* v___x_4857_; lean_object* v___x_4858_; lean_object* v___x_4859_; lean_object* v___x_4860_; lean_object* v___x_4861_; lean_object* v___x_4862_; lean_object* v___x_4863_; lean_object* v___x_4864_; lean_object* v___x_4865_; lean_object* v___x_4866_; lean_object* v___x_4867_; lean_object* v___x_4868_; lean_object* v___x_4869_; lean_object* v___x_4870_; lean_object* v___x_4871_; lean_object* v___x_4872_; lean_object* v___x_4873_; lean_object* v___x_4874_; lean_object* v___x_4875_; uint8_t v___x_4876_; lean_object* v___x_4877_; lean_object* v___x_4878_; lean_object* v___x_4879_; lean_object* v___x_4880_; lean_object* v___x_4881_; lean_object* v___x_4882_; lean_object* v___x_4883_; lean_object* v___x_4884_; lean_object* v___x_4885_; lean_object* v___x_4886_; lean_object* v___x_4887_; lean_object* v___x_4888_; lean_object* v___x_4890_; 
v___x_4856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_4857_ = l_Lean_Level_succ___override(v_v_4828_);
v___x_4858_ = lean_box(0);
v___x_4859_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4859_, 0, v___x_4857_);
lean_ctor_set(v___x_4859_, 1, v___x_4858_);
lean_inc_ref_n(v___x_4859_, 2);
v___x_4860_ = l_Lean_Expr_const___override(v___x_4856_, v___x_4859_);
lean_inc_ref_n(v_A_4829_, 3);
v___x_4861_ = l_Lean_Expr_app___override(v___x_4860_, v_A_4829_);
lean_inc_ref(v___x_4861_);
v___x_4862_ = l_Lean_Expr_app___override(v___x_4861_, v_e_u2081_4833_);
v___x_4863_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__21));
v___x_4864_ = lean_box(0);
v___x_4865_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4865_, 0, v___x_4864_);
lean_ctor_set(v___x_4865_, 1, v___x_4859_);
v___x_4866_ = l_Lean_Expr_const___override(v___x_4863_, v___x_4865_);
v___x_4867_ = l_Lean_Expr_app___override(v___x_4866_, v_A_4829_);
lean_inc_ref_n(v_expr_4852_, 2);
v___x_4868_ = l_Lean_Expr_app___override(v___x_4867_, v_expr_4852_);
v___x_4869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__22));
v___x_4870_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__23));
v___x_4871_ = l_Lean_Expr_app___override(v___x_4861_, v_expr_4852_);
v___x_4872_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__19);
v___x_4873_ = l_Lean_Expr_app___override(v___x_4871_, v___x_4872_);
v___x_4874_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24, &lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__24);
v___x_4875_ = l_Lean_Expr_app___override(v___x_4862_, v___x_4874_);
v___x_4876_ = 0;
v___x_4877_ = l_Lean_Expr_lam___override(v___x_4870_, v___x_4873_, v___x_4875_, v___x_4876_);
v___x_4878_ = l_Lean_Expr_lam___override(v___x_4869_, v_A_4829_, v___x_4877_, v___x_4876_);
v___x_4879_ = l_Lean_Expr_app___override(v___x_4868_, v___x_4878_);
v___x_4880_ = l_Lean_Expr_app___override(v___x_4879_, v_proof_4846_);
lean_inc_ref(v_e_u2082_4834_);
v___x_4881_ = l_Lean_Expr_app___override(v___x_4880_, v_e_u2082_4834_);
v___x_4882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_RingCompute_mul___redArg___closed__23));
v___x_4883_ = l_Lean_Expr_const___override(v___x_4882_, v___x_4859_);
v___x_4884_ = l_Lean_Expr_app___override(v___x_4883_, v_A_4829_);
v___x_4885_ = l_Lean_Expr_app___override(v___x_4884_, v_e_u2082_4834_);
v___x_4886_ = l_Lean_Expr_app___override(v___x_4885_, v_expr_4852_);
v___x_4887_ = l_Lean_Expr_app___override(v___x_4886_, v_proof_4854_);
v___x_4888_ = l_Lean_Expr_app___override(v___x_4881_, v___x_4887_);
if (v_isShared_4851_ == 0)
{
lean_ctor_set(v___x_4850_, 0, v___x_4888_);
v___x_4890_ = v___x_4850_;
goto v_reusejp_4889_;
}
else
{
lean_object* v_reuseFailAlloc_4891_; 
v_reuseFailAlloc_4891_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4891_, 0, v___x_4888_);
v___x_4890_ = v_reuseFailAlloc_4891_;
goto v_reusejp_4889_;
}
v_reusejp_4889_:
{
return v___x_4890_;
}
}
}
}
else
{
lean_object* v_a_4938_; lean_object* v___x_4940_; uint8_t v_isShared_4941_; uint8_t v_isSharedCheck_4945_; 
lean_dec_ref(v_proof_4846_);
lean_dec(v_val_4845_);
lean_dec_ref(v_expr_4844_);
lean_dec_ref(v_e_u2082_4834_);
lean_dec_ref(v_e_u2081_4833_);
lean_dec_ref(v___x_4831_);
lean_dec_ref(v_sA_4830_);
lean_dec_ref(v_A_4829_);
lean_dec(v_v_4828_);
lean_dec_ref(v___x_4827_);
v_a_4938_ = lean_ctor_get(v___x_4847_, 0);
v_isSharedCheck_4945_ = !lean_is_exclusive(v___x_4847_);
if (v_isSharedCheck_4945_ == 0)
{
v___x_4940_ = v___x_4847_;
v_isShared_4941_ = v_isSharedCheck_4945_;
goto v_resetjp_4939_;
}
else
{
lean_inc(v_a_4938_);
lean_dec(v___x_4847_);
v___x_4940_ = lean_box(0);
v_isShared_4941_ = v_isSharedCheck_4945_;
goto v_resetjp_4939_;
}
v_resetjp_4939_:
{
lean_object* v___x_4943_; 
if (v_isShared_4941_ == 0)
{
v___x_4943_ = v___x_4940_;
goto v_reusejp_4942_;
}
else
{
lean_object* v_reuseFailAlloc_4944_; 
v_reuseFailAlloc_4944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4944_, 0, v_a_4938_);
v___x_4943_ = v_reuseFailAlloc_4944_;
goto v_reusejp_4942_;
}
v_reusejp_4942_:
{
return v___x_4943_;
}
}
}
}
else
{
lean_object* v_a_4946_; lean_object* v___x_4948_; uint8_t v_isShared_4949_; uint8_t v_isSharedCheck_4953_; 
lean_dec_ref(v_e_u2082_4834_);
lean_dec_ref(v_e_u2081_4833_);
lean_dec_ref(v_toCache_4832_);
lean_dec_ref(v___x_4831_);
lean_dec_ref(v_sA_4830_);
lean_dec_ref(v_A_4829_);
lean_dec(v_v_4828_);
lean_dec_ref(v___x_4827_);
v_a_4946_ = lean_ctor_get(v___x_4842_, 0);
v_isSharedCheck_4953_ = !lean_is_exclusive(v___x_4842_);
if (v_isSharedCheck_4953_ == 0)
{
v___x_4948_ = v___x_4842_;
v_isShared_4949_ = v_isSharedCheck_4953_;
goto v_resetjp_4947_;
}
else
{
lean_inc(v_a_4946_);
lean_dec(v___x_4842_);
v___x_4948_ = lean_box(0);
v_isShared_4949_ = v_isSharedCheck_4953_;
goto v_resetjp_4947_;
}
v_resetjp_4947_:
{
lean_object* v___x_4951_; 
if (v_isShared_4949_ == 0)
{
v___x_4951_ = v___x_4948_;
goto v_reusejp_4950_;
}
else
{
lean_object* v_reuseFailAlloc_4952_; 
v_reuseFailAlloc_4952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4952_, 0, v_a_4946_);
v___x_4951_ = v_reuseFailAlloc_4952_;
goto v_reusejp_4950_;
}
v_reusejp_4950_:
{
return v___x_4951_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___boxed(lean_object* v___x_4954_, lean_object* v_v_4955_, lean_object* v_A_4956_, lean_object* v_sA_4957_, lean_object* v___x_4958_, lean_object* v_toCache_4959_, lean_object* v_e_u2081_4960_, lean_object* v_e_u2082_4961_, lean_object* v___y_4962_, lean_object* v___y_4963_, lean_object* v___y_4964_, lean_object* v___y_4965_, lean_object* v___y_4966_, lean_object* v___y_4967_, lean_object* v___y_4968_){
_start:
{
lean_object* v_res_4969_; 
v_res_4969_ = lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0(v___x_4954_, v_v_4955_, v_A_4956_, v_sA_4957_, v___x_4958_, v_toCache_4959_, v_e_u2081_4960_, v_e_u2082_4961_, v___y_4962_, v___y_4963_, v___y_4964_, v___y_4965_, v___y_4966_, v___y_4967_);
lean_dec(v___y_4967_);
lean_dec_ref(v___y_4966_);
lean_dec(v___y_4965_);
lean_dec_ref(v___y_4964_);
lean_dec(v___y_4963_);
lean_dec_ref(v___y_4962_);
return v_res_4969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore(lean_object* v_u_4971_, lean_object* v_v_4972_, lean_object* v_R_4973_, lean_object* v_A_4974_, lean_object* v_sR_4975_, lean_object* v_sA_4976_, lean_object* v_sAlg_4977_, lean_object* v_cR_4978_, lean_object* v_cA_4979_, lean_object* v_e_u2081_4980_, lean_object* v_e_u2082_4981_, lean_object* v_a_4982_, lean_object* v_a_4983_, lean_object* v_a_4984_, lean_object* v_a_4985_, lean_object* v_a_4986_, lean_object* v_a_4987_){
_start:
{
lean_object* v_options_4989_; lean_object* v_toCache_4990_; lean_object* v___x_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; lean_object* v___f_4994_; lean_object* v___x_4995_; lean_object* v___x_4996_; 
v_options_4989_ = lean_ctor_get(v_a_4986_, 2);
v_toCache_4990_ = lean_ctor_get(v_cA_4979_, 0);
lean_inc_ref(v_toCache_4990_);
v___x_4991_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0));
v___x_4992_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_sA_4976_);
lean_inc_ref(v_A_4974_);
lean_inc(v_v_4972_);
v___x_4993_ = lp_mathlib_Mathlib_Tactic_Algebra_ringCompute(v_u_4971_, v_v_4972_, v_R_4973_, v_A_4974_, v_sR_4975_, v_sA_4976_, v_sAlg_4977_, v_cR_4978_, v_cA_4979_);
v___f_4994_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___lam__0___boxed), 15, 8);
lean_closure_set(v___f_4994_, 0, v___x_4992_);
lean_closure_set(v___f_4994_, 1, v_v_4972_);
lean_closure_set(v___f_4994_, 2, v_A_4974_);
lean_closure_set(v___f_4994_, 3, v_sA_4976_);
lean_closure_set(v___f_4994_, 4, v___x_4993_);
lean_closure_set(v___f_4994_, 5, v_toCache_4990_);
lean_closure_set(v___f_4994_, 6, v_e_u2081_4980_);
lean_closure_set(v___f_4994_, 7, v_e_u2082_4981_);
v___x_4995_ = lean_box(0);
v___x_4996_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__1___redArg(v___x_4991_, v_options_4989_, v___f_4994_, v___x_4995_, v_a_4982_, v_a_4983_, v_a_4984_, v_a_4985_, v_a_4986_, v_a_4987_);
return v___x_4996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___boxed(lean_object** _args){
lean_object* v_u_4997_ = _args[0];
lean_object* v_v_4998_ = _args[1];
lean_object* v_R_4999_ = _args[2];
lean_object* v_A_5000_ = _args[3];
lean_object* v_sR_5001_ = _args[4];
lean_object* v_sA_5002_ = _args[5];
lean_object* v_sAlg_5003_ = _args[6];
lean_object* v_cR_5004_ = _args[7];
lean_object* v_cA_5005_ = _args[8];
lean_object* v_e_u2081_5006_ = _args[9];
lean_object* v_e_u2082_5007_ = _args[10];
lean_object* v_a_5008_ = _args[11];
lean_object* v_a_5009_ = _args[12];
lean_object* v_a_5010_ = _args[13];
lean_object* v_a_5011_ = _args[14];
lean_object* v_a_5012_ = _args[15];
lean_object* v_a_5013_ = _args[16];
lean_object* v_a_5014_ = _args[17];
_start:
{
lean_object* v_res_5015_; 
v_res_5015_ = lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore(v_u_4997_, v_v_4998_, v_R_4999_, v_A_5000_, v_sR_5001_, v_sA_5002_, v_sAlg_5003_, v_cR_5004_, v_cA_5005_, v_e_u2081_5006_, v_e_u2082_5007_, v_a_5008_, v_a_5009_, v_a_5010_, v_a_5011_, v_a_5012_, v_a_5013_);
lean_dec(v_a_5013_);
lean_dec_ref(v_a_5012_);
lean_dec(v_a_5011_);
lean_dec_ref(v_a_5010_);
lean_dec(v_a_5009_);
lean_dec_ref(v_a_5008_);
return v_res_5015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0(lean_object* v_00_u03b1_5016_, lean_object* v_msg_5017_, lean_object* v___y_5018_, lean_object* v___y_5019_, lean_object* v___y_5020_, lean_object* v___y_5021_, lean_object* v___y_5022_, lean_object* v___y_5023_){
_start:
{
lean_object* v___x_5025_; 
v___x_5025_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(v_msg_5017_, v___y_5020_, v___y_5021_, v___y_5022_, v___y_5023_);
return v___x_5025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___boxed(lean_object* v_00_u03b1_5026_, lean_object* v_msg_5027_, lean_object* v___y_5028_, lean_object* v___y_5029_, lean_object* v___y_5030_, lean_object* v___y_5031_, lean_object* v___y_5032_, lean_object* v___y_5033_, lean_object* v___y_5034_){
_start:
{
lean_object* v_res_5035_; 
v_res_5035_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0(v_00_u03b1_5026_, v_msg_5027_, v___y_5028_, v___y_5029_, v___y_5030_, v___y_5031_, v___y_5032_, v___y_5033_);
lean_dec(v___y_5033_);
lean_dec_ref(v___y_5032_);
lean_dec(v___y_5031_);
lean_dec_ref(v___y_5030_);
lean_dec(v___y_5029_);
lean_dec_ref(v___y_5028_);
return v_res_5035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg(lean_object* v_e_5036_, lean_object* v___y_5037_){
_start:
{
uint8_t v___x_5039_; 
v___x_5039_ = l_Lean_Expr_hasMVar(v_e_5036_);
if (v___x_5039_ == 0)
{
lean_object* v___x_5040_; 
v___x_5040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5040_, 0, v_e_5036_);
return v___x_5040_;
}
else
{
lean_object* v___x_5041_; lean_object* v_mctx_5042_; lean_object* v___x_5043_; lean_object* v_fst_5044_; lean_object* v_snd_5045_; lean_object* v___x_5046_; lean_object* v_cache_5047_; lean_object* v_zetaDeltaFVarIds_5048_; lean_object* v_postponed_5049_; lean_object* v_diag_5050_; lean_object* v___x_5052_; uint8_t v_isShared_5053_; uint8_t v_isSharedCheck_5059_; 
v___x_5041_ = lean_st_ref_get(v___y_5037_);
v_mctx_5042_ = lean_ctor_get(v___x_5041_, 0);
lean_inc_ref(v_mctx_5042_);
lean_dec(v___x_5041_);
v___x_5043_ = l_Lean_instantiateMVarsCore(v_mctx_5042_, v_e_5036_);
v_fst_5044_ = lean_ctor_get(v___x_5043_, 0);
lean_inc(v_fst_5044_);
v_snd_5045_ = lean_ctor_get(v___x_5043_, 1);
lean_inc(v_snd_5045_);
lean_dec_ref(v___x_5043_);
v___x_5046_ = lean_st_ref_take(v___y_5037_);
v_cache_5047_ = lean_ctor_get(v___x_5046_, 1);
v_zetaDeltaFVarIds_5048_ = lean_ctor_get(v___x_5046_, 2);
v_postponed_5049_ = lean_ctor_get(v___x_5046_, 3);
v_diag_5050_ = lean_ctor_get(v___x_5046_, 4);
v_isSharedCheck_5059_ = !lean_is_exclusive(v___x_5046_);
if (v_isSharedCheck_5059_ == 0)
{
lean_object* v_unused_5060_; 
v_unused_5060_ = lean_ctor_get(v___x_5046_, 0);
lean_dec(v_unused_5060_);
v___x_5052_ = v___x_5046_;
v_isShared_5053_ = v_isSharedCheck_5059_;
goto v_resetjp_5051_;
}
else
{
lean_inc(v_diag_5050_);
lean_inc(v_postponed_5049_);
lean_inc(v_zetaDeltaFVarIds_5048_);
lean_inc(v_cache_5047_);
lean_dec(v___x_5046_);
v___x_5052_ = lean_box(0);
v_isShared_5053_ = v_isSharedCheck_5059_;
goto v_resetjp_5051_;
}
v_resetjp_5051_:
{
lean_object* v___x_5055_; 
if (v_isShared_5053_ == 0)
{
lean_ctor_set(v___x_5052_, 0, v_snd_5045_);
v___x_5055_ = v___x_5052_;
goto v_reusejp_5054_;
}
else
{
lean_object* v_reuseFailAlloc_5058_; 
v_reuseFailAlloc_5058_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5058_, 0, v_snd_5045_);
lean_ctor_set(v_reuseFailAlloc_5058_, 1, v_cache_5047_);
lean_ctor_set(v_reuseFailAlloc_5058_, 2, v_zetaDeltaFVarIds_5048_);
lean_ctor_set(v_reuseFailAlloc_5058_, 3, v_postponed_5049_);
lean_ctor_set(v_reuseFailAlloc_5058_, 4, v_diag_5050_);
v___x_5055_ = v_reuseFailAlloc_5058_;
goto v_reusejp_5054_;
}
v_reusejp_5054_:
{
lean_object* v___x_5056_; lean_object* v___x_5057_; 
v___x_5056_ = lean_st_ref_set(v___y_5037_, v___x_5055_);
v___x_5057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5057_, 0, v_fst_5044_);
return v___x_5057_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg___boxed(lean_object* v_e_5061_, lean_object* v___y_5062_, lean_object* v___y_5063_){
_start:
{
lean_object* v_res_5064_; 
v_res_5064_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg(v_e_5061_, v___y_5062_);
lean_dec(v___y_5062_);
return v_res_5064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0(lean_object* v_e_5065_, lean_object* v___y_5066_, lean_object* v___y_5067_, lean_object* v___y_5068_, lean_object* v___y_5069_, lean_object* v___y_5070_, lean_object* v___y_5071_){
_start:
{
lean_object* v___x_5073_; 
v___x_5073_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg(v_e_5065_, v___y_5069_);
return v___x_5073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___boxed(lean_object* v_e_5074_, lean_object* v___y_5075_, lean_object* v___y_5076_, lean_object* v___y_5077_, lean_object* v___y_5078_, lean_object* v___y_5079_, lean_object* v___y_5080_, lean_object* v___y_5081_){
_start:
{
lean_object* v_res_5082_; 
v_res_5082_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0(v_e_5074_, v___y_5075_, v___y_5076_, v___y_5077_, v___y_5078_, v___y_5079_, v___y_5080_);
lean_dec(v___y_5080_);
lean_dec_ref(v___y_5079_);
lean_dec(v___y_5078_);
lean_dec_ref(v___y_5077_);
lean_dec(v___y_5076_);
lean_dec_ref(v___y_5075_);
return v_res_5082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object* v_x_5083_, lean_object* v_x_5084_, lean_object* v_x_5085_, lean_object* v_x_5086_){
_start:
{
lean_object* v_ks_5087_; lean_object* v_vs_5088_; lean_object* v___x_5090_; uint8_t v_isShared_5091_; uint8_t v_isSharedCheck_5112_; 
v_ks_5087_ = lean_ctor_get(v_x_5083_, 0);
v_vs_5088_ = lean_ctor_get(v_x_5083_, 1);
v_isSharedCheck_5112_ = !lean_is_exclusive(v_x_5083_);
if (v_isSharedCheck_5112_ == 0)
{
v___x_5090_ = v_x_5083_;
v_isShared_5091_ = v_isSharedCheck_5112_;
goto v_resetjp_5089_;
}
else
{
lean_inc(v_vs_5088_);
lean_inc(v_ks_5087_);
lean_dec(v_x_5083_);
v___x_5090_ = lean_box(0);
v_isShared_5091_ = v_isSharedCheck_5112_;
goto v_resetjp_5089_;
}
v_resetjp_5089_:
{
lean_object* v___x_5092_; uint8_t v___x_5093_; 
v___x_5092_ = lean_array_get_size(v_ks_5087_);
v___x_5093_ = lean_nat_dec_lt(v_x_5084_, v___x_5092_);
if (v___x_5093_ == 0)
{
lean_object* v___x_5094_; lean_object* v___x_5095_; lean_object* v___x_5097_; 
lean_dec(v_x_5084_);
v___x_5094_ = lean_array_push(v_ks_5087_, v_x_5085_);
v___x_5095_ = lean_array_push(v_vs_5088_, v_x_5086_);
if (v_isShared_5091_ == 0)
{
lean_ctor_set(v___x_5090_, 1, v___x_5095_);
lean_ctor_set(v___x_5090_, 0, v___x_5094_);
v___x_5097_ = v___x_5090_;
goto v_reusejp_5096_;
}
else
{
lean_object* v_reuseFailAlloc_5098_; 
v_reuseFailAlloc_5098_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5098_, 0, v___x_5094_);
lean_ctor_set(v_reuseFailAlloc_5098_, 1, v___x_5095_);
v___x_5097_ = v_reuseFailAlloc_5098_;
goto v_reusejp_5096_;
}
v_reusejp_5096_:
{
return v___x_5097_;
}
}
else
{
lean_object* v_k_x27_5099_; uint8_t v___x_5100_; 
v_k_x27_5099_ = lean_array_fget_borrowed(v_ks_5087_, v_x_5084_);
v___x_5100_ = l_Lean_instBEqMVarId_beq(v_x_5085_, v_k_x27_5099_);
if (v___x_5100_ == 0)
{
lean_object* v___x_5102_; 
if (v_isShared_5091_ == 0)
{
v___x_5102_ = v___x_5090_;
goto v_reusejp_5101_;
}
else
{
lean_object* v_reuseFailAlloc_5106_; 
v_reuseFailAlloc_5106_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5106_, 0, v_ks_5087_);
lean_ctor_set(v_reuseFailAlloc_5106_, 1, v_vs_5088_);
v___x_5102_ = v_reuseFailAlloc_5106_;
goto v_reusejp_5101_;
}
v_reusejp_5101_:
{
lean_object* v___x_5103_; lean_object* v___x_5104_; 
v___x_5103_ = lean_unsigned_to_nat(1u);
v___x_5104_ = lean_nat_add(v_x_5084_, v___x_5103_);
lean_dec(v_x_5084_);
v_x_5083_ = v___x_5102_;
v_x_5084_ = v___x_5104_;
goto _start;
}
}
else
{
lean_object* v___x_5107_; lean_object* v___x_5108_; lean_object* v___x_5110_; 
v___x_5107_ = lean_array_fset(v_ks_5087_, v_x_5084_, v_x_5085_);
v___x_5108_ = lean_array_fset(v_vs_5088_, v_x_5084_, v_x_5086_);
lean_dec(v_x_5084_);
if (v_isShared_5091_ == 0)
{
lean_ctor_set(v___x_5090_, 1, v___x_5108_);
lean_ctor_set(v___x_5090_, 0, v___x_5107_);
v___x_5110_ = v___x_5090_;
goto v_reusejp_5109_;
}
else
{
lean_object* v_reuseFailAlloc_5111_; 
v_reuseFailAlloc_5111_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5111_, 0, v___x_5107_);
lean_ctor_set(v_reuseFailAlloc_5111_, 1, v___x_5108_);
v___x_5110_ = v_reuseFailAlloc_5111_;
goto v_reusejp_5109_;
}
v_reusejp_5109_:
{
return v___x_5110_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_n_5113_, lean_object* v_k_5114_, lean_object* v_v_5115_){
_start:
{
lean_object* v___x_5116_; lean_object* v___x_5117_; 
v___x_5116_ = lean_unsigned_to_nat(0u);
v___x_5117_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_n_5113_, v___x_5116_, v_k_5114_, v_v_5115_);
return v___x_5117_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_5118_; 
v___x_5118_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_5118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(lean_object* v_x_5119_, size_t v_x_5120_, size_t v_x_5121_, lean_object* v_x_5122_, lean_object* v_x_5123_){
_start:
{
if (lean_obj_tag(v_x_5119_) == 0)
{
lean_object* v_es_5124_; size_t v___x_5125_; size_t v___x_5126_; lean_object* v_j_5127_; lean_object* v___x_5128_; uint8_t v___x_5129_; 
v_es_5124_ = lean_ctor_get(v_x_5119_, 0);
v___x_5125_ = ((size_t)31ULL);
v___x_5126_ = lean_usize_land(v_x_5120_, v___x_5125_);
v_j_5127_ = lean_usize_to_nat(v___x_5126_);
v___x_5128_ = lean_array_get_size(v_es_5124_);
v___x_5129_ = lean_nat_dec_lt(v_j_5127_, v___x_5128_);
if (v___x_5129_ == 0)
{
lean_dec(v_j_5127_);
lean_dec(v_x_5123_);
lean_dec(v_x_5122_);
return v_x_5119_;
}
else
{
lean_object* v___x_5131_; uint8_t v_isShared_5132_; uint8_t v_isSharedCheck_5168_; 
lean_inc_ref(v_es_5124_);
v_isSharedCheck_5168_ = !lean_is_exclusive(v_x_5119_);
if (v_isSharedCheck_5168_ == 0)
{
lean_object* v_unused_5169_; 
v_unused_5169_ = lean_ctor_get(v_x_5119_, 0);
lean_dec(v_unused_5169_);
v___x_5131_ = v_x_5119_;
v_isShared_5132_ = v_isSharedCheck_5168_;
goto v_resetjp_5130_;
}
else
{
lean_dec(v_x_5119_);
v___x_5131_ = lean_box(0);
v_isShared_5132_ = v_isSharedCheck_5168_;
goto v_resetjp_5130_;
}
v_resetjp_5130_:
{
lean_object* v_v_5133_; lean_object* v___x_5134_; lean_object* v_xs_x27_5135_; lean_object* v___y_5137_; 
v_v_5133_ = lean_array_fget(v_es_5124_, v_j_5127_);
v___x_5134_ = lean_box(0);
v_xs_x27_5135_ = lean_array_fset(v_es_5124_, v_j_5127_, v___x_5134_);
switch(lean_obj_tag(v_v_5133_))
{
case 0:
{
lean_object* v_key_5142_; lean_object* v_val_5143_; lean_object* v___x_5145_; uint8_t v_isShared_5146_; uint8_t v_isSharedCheck_5153_; 
v_key_5142_ = lean_ctor_get(v_v_5133_, 0);
v_val_5143_ = lean_ctor_get(v_v_5133_, 1);
v_isSharedCheck_5153_ = !lean_is_exclusive(v_v_5133_);
if (v_isSharedCheck_5153_ == 0)
{
v___x_5145_ = v_v_5133_;
v_isShared_5146_ = v_isSharedCheck_5153_;
goto v_resetjp_5144_;
}
else
{
lean_inc(v_val_5143_);
lean_inc(v_key_5142_);
lean_dec(v_v_5133_);
v___x_5145_ = lean_box(0);
v_isShared_5146_ = v_isSharedCheck_5153_;
goto v_resetjp_5144_;
}
v_resetjp_5144_:
{
uint8_t v___x_5147_; 
v___x_5147_ = l_Lean_instBEqMVarId_beq(v_x_5122_, v_key_5142_);
if (v___x_5147_ == 0)
{
lean_object* v___x_5148_; lean_object* v___x_5149_; 
lean_del_object(v___x_5145_);
v___x_5148_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_5142_, v_val_5143_, v_x_5122_, v_x_5123_);
v___x_5149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5149_, 0, v___x_5148_);
v___y_5137_ = v___x_5149_;
goto v___jp_5136_;
}
else
{
lean_object* v___x_5151_; 
lean_dec(v_val_5143_);
lean_dec(v_key_5142_);
if (v_isShared_5146_ == 0)
{
lean_ctor_set(v___x_5145_, 1, v_x_5123_);
lean_ctor_set(v___x_5145_, 0, v_x_5122_);
v___x_5151_ = v___x_5145_;
goto v_reusejp_5150_;
}
else
{
lean_object* v_reuseFailAlloc_5152_; 
v_reuseFailAlloc_5152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5152_, 0, v_x_5122_);
lean_ctor_set(v_reuseFailAlloc_5152_, 1, v_x_5123_);
v___x_5151_ = v_reuseFailAlloc_5152_;
goto v_reusejp_5150_;
}
v_reusejp_5150_:
{
v___y_5137_ = v___x_5151_;
goto v___jp_5136_;
}
}
}
}
case 1:
{
lean_object* v_node_5154_; lean_object* v___x_5156_; uint8_t v_isShared_5157_; uint8_t v_isSharedCheck_5166_; 
v_node_5154_ = lean_ctor_get(v_v_5133_, 0);
v_isSharedCheck_5166_ = !lean_is_exclusive(v_v_5133_);
if (v_isSharedCheck_5166_ == 0)
{
v___x_5156_ = v_v_5133_;
v_isShared_5157_ = v_isSharedCheck_5166_;
goto v_resetjp_5155_;
}
else
{
lean_inc(v_node_5154_);
lean_dec(v_v_5133_);
v___x_5156_ = lean_box(0);
v_isShared_5157_ = v_isSharedCheck_5166_;
goto v_resetjp_5155_;
}
v_resetjp_5155_:
{
size_t v___x_5158_; size_t v___x_5159_; size_t v___x_5160_; size_t v___x_5161_; lean_object* v___x_5162_; lean_object* v___x_5164_; 
v___x_5158_ = ((size_t)5ULL);
v___x_5159_ = lean_usize_shift_right(v_x_5120_, v___x_5158_);
v___x_5160_ = ((size_t)1ULL);
v___x_5161_ = lean_usize_add(v_x_5121_, v___x_5160_);
v___x_5162_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(v_node_5154_, v___x_5159_, v___x_5161_, v_x_5122_, v_x_5123_);
if (v_isShared_5157_ == 0)
{
lean_ctor_set(v___x_5156_, 0, v___x_5162_);
v___x_5164_ = v___x_5156_;
goto v_reusejp_5163_;
}
else
{
lean_object* v_reuseFailAlloc_5165_; 
v_reuseFailAlloc_5165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5165_, 0, v___x_5162_);
v___x_5164_ = v_reuseFailAlloc_5165_;
goto v_reusejp_5163_;
}
v_reusejp_5163_:
{
v___y_5137_ = v___x_5164_;
goto v___jp_5136_;
}
}
}
default: 
{
lean_object* v___x_5167_; 
v___x_5167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5167_, 0, v_x_5122_);
lean_ctor_set(v___x_5167_, 1, v_x_5123_);
v___y_5137_ = v___x_5167_;
goto v___jp_5136_;
}
}
v___jp_5136_:
{
lean_object* v___x_5138_; lean_object* v___x_5140_; 
v___x_5138_ = lean_array_fset(v_xs_x27_5135_, v_j_5127_, v___y_5137_);
lean_dec(v_j_5127_);
if (v_isShared_5132_ == 0)
{
lean_ctor_set(v___x_5131_, 0, v___x_5138_);
v___x_5140_ = v___x_5131_;
goto v_reusejp_5139_;
}
else
{
lean_object* v_reuseFailAlloc_5141_; 
v_reuseFailAlloc_5141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5141_, 0, v___x_5138_);
v___x_5140_ = v_reuseFailAlloc_5141_;
goto v_reusejp_5139_;
}
v_reusejp_5139_:
{
return v___x_5140_;
}
}
}
}
}
else
{
lean_object* v_ks_5170_; lean_object* v_vs_5171_; lean_object* v___x_5173_; uint8_t v_isShared_5174_; uint8_t v_isSharedCheck_5191_; 
v_ks_5170_ = lean_ctor_get(v_x_5119_, 0);
v_vs_5171_ = lean_ctor_get(v_x_5119_, 1);
v_isSharedCheck_5191_ = !lean_is_exclusive(v_x_5119_);
if (v_isSharedCheck_5191_ == 0)
{
v___x_5173_ = v_x_5119_;
v_isShared_5174_ = v_isSharedCheck_5191_;
goto v_resetjp_5172_;
}
else
{
lean_inc(v_vs_5171_);
lean_inc(v_ks_5170_);
lean_dec(v_x_5119_);
v___x_5173_ = lean_box(0);
v_isShared_5174_ = v_isSharedCheck_5191_;
goto v_resetjp_5172_;
}
v_resetjp_5172_:
{
lean_object* v___x_5176_; 
if (v_isShared_5174_ == 0)
{
v___x_5176_ = v___x_5173_;
goto v_reusejp_5175_;
}
else
{
lean_object* v_reuseFailAlloc_5190_; 
v_reuseFailAlloc_5190_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5190_, 0, v_ks_5170_);
lean_ctor_set(v_reuseFailAlloc_5190_, 1, v_vs_5171_);
v___x_5176_ = v_reuseFailAlloc_5190_;
goto v_reusejp_5175_;
}
v_reusejp_5175_:
{
lean_object* v_newNode_5177_; uint8_t v___y_5179_; size_t v___x_5185_; uint8_t v___x_5186_; 
v_newNode_5177_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3___redArg(v___x_5176_, v_x_5122_, v_x_5123_);
v___x_5185_ = ((size_t)7ULL);
v___x_5186_ = lean_usize_dec_le(v___x_5185_, v_x_5121_);
if (v___x_5186_ == 0)
{
lean_object* v___x_5187_; lean_object* v___x_5188_; uint8_t v___x_5189_; 
v___x_5187_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_5177_);
v___x_5188_ = lean_unsigned_to_nat(4u);
v___x_5189_ = lean_nat_dec_lt(v___x_5187_, v___x_5188_);
lean_dec(v___x_5187_);
v___y_5179_ = v___x_5189_;
goto v___jp_5178_;
}
else
{
v___y_5179_ = v___x_5186_;
goto v___jp_5178_;
}
v___jp_5178_:
{
if (v___y_5179_ == 0)
{
lean_object* v_ks_5180_; lean_object* v_vs_5181_; lean_object* v___x_5182_; lean_object* v___x_5183_; lean_object* v___x_5184_; 
v_ks_5180_ = lean_ctor_get(v_newNode_5177_, 0);
lean_inc_ref(v_ks_5180_);
v_vs_5181_ = lean_ctor_get(v_newNode_5177_, 1);
lean_inc_ref(v_vs_5181_);
lean_dec_ref(v_newNode_5177_);
v___x_5182_ = lean_unsigned_to_nat(0u);
v___x_5183_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___closed__0);
v___x_5184_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg(v_x_5121_, v_ks_5180_, v_vs_5181_, v___x_5182_, v___x_5183_);
lean_dec_ref(v_vs_5181_);
lean_dec_ref(v_ks_5180_);
return v___x_5184_;
}
else
{
return v_newNode_5177_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg(size_t v_depth_5192_, lean_object* v_keys_5193_, lean_object* v_vals_5194_, lean_object* v_i_5195_, lean_object* v_entries_5196_){
_start:
{
lean_object* v___x_5197_; uint8_t v___x_5198_; 
v___x_5197_ = lean_array_get_size(v_keys_5193_);
v___x_5198_ = lean_nat_dec_lt(v_i_5195_, v___x_5197_);
if (v___x_5198_ == 0)
{
lean_dec(v_i_5195_);
return v_entries_5196_;
}
else
{
lean_object* v_k_5199_; lean_object* v_v_5200_; uint64_t v___x_5201_; size_t v_h_5202_; size_t v___x_5203_; lean_object* v___x_5204_; size_t v___x_5205_; size_t v___x_5206_; size_t v___x_5207_; size_t v_h_5208_; lean_object* v___x_5209_; lean_object* v___x_5210_; 
v_k_5199_ = lean_array_fget_borrowed(v_keys_5193_, v_i_5195_);
v_v_5200_ = lean_array_fget_borrowed(v_vals_5194_, v_i_5195_);
v___x_5201_ = l_Lean_instHashableMVarId_hash(v_k_5199_);
v_h_5202_ = lean_uint64_to_usize(v___x_5201_);
v___x_5203_ = ((size_t)5ULL);
v___x_5204_ = lean_unsigned_to_nat(1u);
v___x_5205_ = ((size_t)1ULL);
v___x_5206_ = lean_usize_sub(v_depth_5192_, v___x_5205_);
v___x_5207_ = lean_usize_mul(v___x_5203_, v___x_5206_);
v_h_5208_ = lean_usize_shift_right(v_h_5202_, v___x_5207_);
v___x_5209_ = lean_nat_add(v_i_5195_, v___x_5204_);
lean_dec(v_i_5195_);
lean_inc(v_v_5200_);
lean_inc(v_k_5199_);
v___x_5210_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(v_entries_5196_, v_h_5208_, v_depth_5192_, v_k_5199_, v_v_5200_);
v_i_5195_ = v___x_5209_;
v_entries_5196_ = v___x_5210_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_depth_5212_, lean_object* v_keys_5213_, lean_object* v_vals_5214_, lean_object* v_i_5215_, lean_object* v_entries_5216_){
_start:
{
size_t v_depth_boxed_5217_; lean_object* v_res_5218_; 
v_depth_boxed_5217_ = lean_unbox_usize(v_depth_5212_);
lean_dec(v_depth_5212_);
v_res_5218_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_boxed_5217_, v_keys_5213_, v_vals_5214_, v_i_5215_, v_entries_5216_);
lean_dec_ref(v_vals_5214_);
lean_dec_ref(v_keys_5213_);
return v_res_5218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_5219_, lean_object* v_x_5220_, lean_object* v_x_5221_, lean_object* v_x_5222_, lean_object* v_x_5223_){
_start:
{
size_t v_x_11413__boxed_5224_; size_t v_x_11414__boxed_5225_; lean_object* v_res_5226_; 
v_x_11413__boxed_5224_ = lean_unbox_usize(v_x_5220_);
lean_dec(v_x_5220_);
v_x_11414__boxed_5225_ = lean_unbox_usize(v_x_5221_);
lean_dec(v_x_5221_);
v_res_5226_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(v_x_5219_, v_x_11413__boxed_5224_, v_x_11414__boxed_5225_, v_x_5222_, v_x_5223_);
return v_res_5226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1___redArg(lean_object* v_x_5227_, lean_object* v_x_5228_, lean_object* v_x_5229_){
_start:
{
uint64_t v___x_5230_; size_t v___x_5231_; size_t v___x_5232_; lean_object* v___x_5233_; 
v___x_5230_ = l_Lean_instHashableMVarId_hash(v_x_5228_);
v___x_5231_ = lean_uint64_to_usize(v___x_5230_);
v___x_5232_ = ((size_t)1ULL);
v___x_5233_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(v_x_5227_, v___x_5231_, v___x_5232_, v_x_5228_, v_x_5229_);
return v___x_5233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg(lean_object* v_mvarId_5234_, lean_object* v_val_5235_, lean_object* v___y_5236_){
_start:
{
lean_object* v___x_5238_; lean_object* v_mctx_5239_; lean_object* v_cache_5240_; lean_object* v_zetaDeltaFVarIds_5241_; lean_object* v_postponed_5242_; lean_object* v_diag_5243_; lean_object* v___x_5245_; uint8_t v_isShared_5246_; uint8_t v_isSharedCheck_5271_; 
v___x_5238_ = lean_st_ref_take(v___y_5236_);
v_mctx_5239_ = lean_ctor_get(v___x_5238_, 0);
v_cache_5240_ = lean_ctor_get(v___x_5238_, 1);
v_zetaDeltaFVarIds_5241_ = lean_ctor_get(v___x_5238_, 2);
v_postponed_5242_ = lean_ctor_get(v___x_5238_, 3);
v_diag_5243_ = lean_ctor_get(v___x_5238_, 4);
v_isSharedCheck_5271_ = !lean_is_exclusive(v___x_5238_);
if (v_isSharedCheck_5271_ == 0)
{
v___x_5245_ = v___x_5238_;
v_isShared_5246_ = v_isSharedCheck_5271_;
goto v_resetjp_5244_;
}
else
{
lean_inc(v_diag_5243_);
lean_inc(v_postponed_5242_);
lean_inc(v_zetaDeltaFVarIds_5241_);
lean_inc(v_cache_5240_);
lean_inc(v_mctx_5239_);
lean_dec(v___x_5238_);
v___x_5245_ = lean_box(0);
v_isShared_5246_ = v_isSharedCheck_5271_;
goto v_resetjp_5244_;
}
v_resetjp_5244_:
{
lean_object* v_depth_5247_; lean_object* v_levelAssignDepth_5248_; lean_object* v_lmvarCounter_5249_; lean_object* v_mvarCounter_5250_; lean_object* v_lDecls_5251_; lean_object* v_decls_5252_; lean_object* v_userNames_5253_; lean_object* v_lAssignment_5254_; lean_object* v_eAssignment_5255_; lean_object* v_dAssignment_5256_; lean_object* v___x_5258_; uint8_t v_isShared_5259_; uint8_t v_isSharedCheck_5270_; 
v_depth_5247_ = lean_ctor_get(v_mctx_5239_, 0);
v_levelAssignDepth_5248_ = lean_ctor_get(v_mctx_5239_, 1);
v_lmvarCounter_5249_ = lean_ctor_get(v_mctx_5239_, 2);
v_mvarCounter_5250_ = lean_ctor_get(v_mctx_5239_, 3);
v_lDecls_5251_ = lean_ctor_get(v_mctx_5239_, 4);
v_decls_5252_ = lean_ctor_get(v_mctx_5239_, 5);
v_userNames_5253_ = lean_ctor_get(v_mctx_5239_, 6);
v_lAssignment_5254_ = lean_ctor_get(v_mctx_5239_, 7);
v_eAssignment_5255_ = lean_ctor_get(v_mctx_5239_, 8);
v_dAssignment_5256_ = lean_ctor_get(v_mctx_5239_, 9);
v_isSharedCheck_5270_ = !lean_is_exclusive(v_mctx_5239_);
if (v_isSharedCheck_5270_ == 0)
{
v___x_5258_ = v_mctx_5239_;
v_isShared_5259_ = v_isSharedCheck_5270_;
goto v_resetjp_5257_;
}
else
{
lean_inc(v_dAssignment_5256_);
lean_inc(v_eAssignment_5255_);
lean_inc(v_lAssignment_5254_);
lean_inc(v_userNames_5253_);
lean_inc(v_decls_5252_);
lean_inc(v_lDecls_5251_);
lean_inc(v_mvarCounter_5250_);
lean_inc(v_lmvarCounter_5249_);
lean_inc(v_levelAssignDepth_5248_);
lean_inc(v_depth_5247_);
lean_dec(v_mctx_5239_);
v___x_5258_ = lean_box(0);
v_isShared_5259_ = v_isSharedCheck_5270_;
goto v_resetjp_5257_;
}
v_resetjp_5257_:
{
lean_object* v___x_5260_; lean_object* v___x_5262_; 
v___x_5260_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1___redArg(v_eAssignment_5255_, v_mvarId_5234_, v_val_5235_);
if (v_isShared_5259_ == 0)
{
lean_ctor_set(v___x_5258_, 8, v___x_5260_);
v___x_5262_ = v___x_5258_;
goto v_reusejp_5261_;
}
else
{
lean_object* v_reuseFailAlloc_5269_; 
v_reuseFailAlloc_5269_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5269_, 0, v_depth_5247_);
lean_ctor_set(v_reuseFailAlloc_5269_, 1, v_levelAssignDepth_5248_);
lean_ctor_set(v_reuseFailAlloc_5269_, 2, v_lmvarCounter_5249_);
lean_ctor_set(v_reuseFailAlloc_5269_, 3, v_mvarCounter_5250_);
lean_ctor_set(v_reuseFailAlloc_5269_, 4, v_lDecls_5251_);
lean_ctor_set(v_reuseFailAlloc_5269_, 5, v_decls_5252_);
lean_ctor_set(v_reuseFailAlloc_5269_, 6, v_userNames_5253_);
lean_ctor_set(v_reuseFailAlloc_5269_, 7, v_lAssignment_5254_);
lean_ctor_set(v_reuseFailAlloc_5269_, 8, v___x_5260_);
lean_ctor_set(v_reuseFailAlloc_5269_, 9, v_dAssignment_5256_);
v___x_5262_ = v_reuseFailAlloc_5269_;
goto v_reusejp_5261_;
}
v_reusejp_5261_:
{
lean_object* v___x_5264_; 
if (v_isShared_5246_ == 0)
{
lean_ctor_set(v___x_5245_, 0, v___x_5262_);
v___x_5264_ = v___x_5245_;
goto v_reusejp_5263_;
}
else
{
lean_object* v_reuseFailAlloc_5268_; 
v_reuseFailAlloc_5268_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5268_, 0, v___x_5262_);
lean_ctor_set(v_reuseFailAlloc_5268_, 1, v_cache_5240_);
lean_ctor_set(v_reuseFailAlloc_5268_, 2, v_zetaDeltaFVarIds_5241_);
lean_ctor_set(v_reuseFailAlloc_5268_, 3, v_postponed_5242_);
lean_ctor_set(v_reuseFailAlloc_5268_, 4, v_diag_5243_);
v___x_5264_ = v_reuseFailAlloc_5268_;
goto v_reusejp_5263_;
}
v_reusejp_5263_:
{
lean_object* v___x_5265_; lean_object* v___x_5266_; lean_object* v___x_5267_; 
v___x_5265_ = lean_st_ref_set(v___y_5236_, v___x_5264_);
v___x_5266_ = lean_box(0);
v___x_5267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5267_, 0, v___x_5266_);
return v___x_5267_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg___boxed(lean_object* v_mvarId_5272_, lean_object* v_val_5273_, lean_object* v___y_5274_, lean_object* v___y_5275_){
_start:
{
lean_object* v_res_5276_; 
v_res_5276_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg(v_mvarId_5272_, v_val_5273_, v___y_5274_);
lean_dec(v___y_5274_);
return v_res_5276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1(void){
_start:
{
lean_object* v___x_5278_; lean_object* v___x_5279_; 
v___x_5278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__0));
v___x_5279_ = l_Lean_stringToMessageData(v___x_5278_);
return v___x_5279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq(lean_object* v_base_5280_, lean_object* v_g_5281_, lean_object* v_a_5282_, lean_object* v_a_5283_, lean_object* v_a_5284_, lean_object* v_a_5285_, lean_object* v_a_5286_, lean_object* v_a_5287_){
_start:
{
lean_object* v___x_5289_; 
lean_inc(v_g_5281_);
v___x_5289_ = l_Lean_MVarId_getType(v_g_5281_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5289_) == 0)
{
lean_object* v_a_5290_; lean_object* v___x_5291_; lean_object* v_a_5292_; lean_object* v___x_5293_; 
v_a_5290_ = lean_ctor_get(v___x_5289_, 0);
lean_inc(v_a_5290_);
lean_dec_ref_known(v___x_5289_, 1);
v___x_5291_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Algebra_proveEq_spec__0___redArg(v_a_5290_, v_a_5285_);
v_a_5292_ = lean_ctor_get(v___x_5291_, 0);
lean_inc(v_a_5292_);
lean_dec_ref(v___x_5291_);
v___x_5293_ = l_Lean_Meta_whnfR(v_a_5292_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5293_) == 0)
{
lean_object* v_a_5294_; lean_object* v___x_5295_; lean_object* v___x_5296_; uint8_t v___x_5297_; 
v_a_5294_ = lean_ctor_get(v___x_5293_, 0);
lean_inc(v_a_5294_);
lean_dec_ref_known(v___x_5293_, 1);
v___x_5295_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__13));
v___x_5296_ = lean_unsigned_to_nat(3u);
v___x_5297_ = l_Lean_Expr_isAppOfArity(v_a_5294_, v___x_5295_, v___x_5296_);
if (v___x_5297_ == 0)
{
lean_object* v___x_5298_; lean_object* v___x_5299_; 
lean_dec(v_a_5294_);
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v___x_5298_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1, &lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Algebra_proveEq___closed__1);
v___x_5299_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore_spec__0___redArg(v___x_5298_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
return v___x_5299_;
}
else
{
lean_object* v___x_5300_; lean_object* v___x_5301_; lean_object* v___x_5302_; lean_object* v___x_5303_; 
v___x_5300_ = l_Lean_Expr_appFn_x21(v_a_5294_);
v___x_5301_ = l_Lean_Expr_appFn_x21(v___x_5300_);
v___x_5302_ = l_Lean_Expr_appArg_x21(v___x_5301_);
lean_dec_ref(v___x_5301_);
v___x_5303_ = lp_mathlib_Qq_getLevelQ_x27(v___x_5302_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5303_) == 0)
{
lean_object* v_a_5304_; lean_object* v_fst_5305_; lean_object* v_snd_5306_; lean_object* v___x_5308_; uint8_t v_isShared_5309_; uint8_t v_isSharedCheck_5431_; 
v_a_5304_ = lean_ctor_get(v___x_5303_, 0);
lean_inc(v_a_5304_);
lean_dec_ref_known(v___x_5303_, 1);
v_fst_5305_ = lean_ctor_get(v_a_5304_, 0);
v_snd_5306_ = lean_ctor_get(v_a_5304_, 1);
v_isSharedCheck_5431_ = !lean_is_exclusive(v_a_5304_);
if (v_isSharedCheck_5431_ == 0)
{
v___x_5308_ = v_a_5304_;
v_isShared_5309_ = v_isSharedCheck_5431_;
goto v_resetjp_5307_;
}
else
{
lean_inc(v_snd_5306_);
lean_inc(v_fst_5305_);
lean_dec(v_a_5304_);
v___x_5308_ = lean_box(0);
v_isShared_5309_ = v_isSharedCheck_5431_;
goto v_resetjp_5307_;
}
v_resetjp_5307_:
{
lean_object* v___x_5310_; lean_object* v___x_5311_; lean_object* v___x_5313_; 
v___x_5310_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__0));
v___x_5311_ = lean_box(0);
lean_inc(v_fst_5305_);
if (v_isShared_5309_ == 0)
{
lean_ctor_set_tag(v___x_5308_, 1);
lean_ctor_set(v___x_5308_, 1, v___x_5311_);
v___x_5313_ = v___x_5308_;
goto v_reusejp_5312_;
}
else
{
lean_object* v_reuseFailAlloc_5430_; 
v_reuseFailAlloc_5430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5430_, 0, v_fst_5305_);
lean_ctor_set(v_reuseFailAlloc_5430_, 1, v___x_5311_);
v___x_5313_ = v_reuseFailAlloc_5430_;
goto v_reusejp_5312_;
}
v_reusejp_5312_:
{
lean_object* v___x_5314_; lean_object* v___x_5315_; lean_object* v___x_5316_; 
lean_inc_ref(v___x_5313_);
v___x_5314_ = l_Lean_Expr_const___override(v___x_5310_, v___x_5313_);
lean_inc(v_snd_5306_);
v___x_5315_ = l_Lean_Expr_app___override(v___x_5314_, v_snd_5306_);
v___x_5316_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_5315_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5316_) == 0)
{
lean_object* v_a_5317_; lean_object* v___x_5318_; 
v_a_5317_ = lean_ctor_get(v___x_5316_, 0);
lean_inc_n(v_a_5317_, 2);
lean_dec_ref_known(v___x_5316_, 1);
lean_inc(v_snd_5306_);
lean_inc(v_fst_5305_);
v___x_5318_ = lp_mathlib_Mathlib_Tactic_Algebra_mkCache(v_fst_5305_, v_snd_5306_, v_a_5317_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5318_) == 0)
{
lean_object* v_a_5319_; lean_object* v___x_5320_; lean_object* v___x_5321_; lean_object* v_____x_5323_; lean_object* v___y_5324_; lean_object* v___y_5325_; lean_object* v___y_5326_; lean_object* v___y_5327_; lean_object* v___y_5328_; lean_object* v___y_5329_; 
v_a_5319_ = lean_ctor_get(v___x_5318_, 0);
lean_inc(v_a_5319_);
lean_dec_ref_known(v___x_5318_, 1);
v___x_5320_ = l_Lean_Expr_appArg_x21(v___x_5300_);
lean_dec_ref(v___x_5300_);
v___x_5321_ = l_Lean_Expr_appArg_x21(v_a_5294_);
lean_dec(v_a_5294_);
if (lean_obj_tag(v_base_5280_) == 0)
{
lean_object* v___x_5393_; 
lean_inc(v_g_5281_);
v___x_5393_ = l_Lean_MVarId_getType(v_g_5281_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5393_) == 0)
{
lean_object* v_a_5394_; lean_object* v___x_5395_; 
v_a_5394_ = lean_ctor_get(v___x_5393_, 0);
lean_inc(v_a_5394_);
lean_dec_ref_known(v___x_5393_, 1);
v___x_5395_ = lp_mathlib_Mathlib_Tactic_Algebra_inferBase___redArg(v_a_5319_, v_a_5394_, v_a_5284_, v_a_5285_, v_a_5286_, v_a_5287_);
if (lean_obj_tag(v___x_5395_) == 0)
{
lean_object* v_a_5396_; 
v_a_5396_ = lean_ctor_get(v___x_5395_, 0);
lean_inc(v_a_5396_);
lean_dec_ref_known(v___x_5395_, 1);
v_____x_5323_ = v_a_5396_;
v___y_5324_ = v_a_5282_;
v___y_5325_ = v_a_5283_;
v___y_5326_ = v_a_5284_;
v___y_5327_ = v_a_5285_;
v___y_5328_ = v_a_5286_;
v___y_5329_ = v_a_5287_;
goto v___jp_5322_;
}
else
{
lean_object* v_a_5397_; lean_object* v___x_5399_; uint8_t v_isShared_5400_; uint8_t v_isSharedCheck_5404_; 
lean_dec_ref(v___x_5321_);
lean_dec_ref(v___x_5320_);
lean_dec(v_a_5319_);
lean_dec(v_a_5317_);
lean_dec_ref(v___x_5313_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec(v_g_5281_);
v_a_5397_ = lean_ctor_get(v___x_5395_, 0);
v_isSharedCheck_5404_ = !lean_is_exclusive(v___x_5395_);
if (v_isSharedCheck_5404_ == 0)
{
v___x_5399_ = v___x_5395_;
v_isShared_5400_ = v_isSharedCheck_5404_;
goto v_resetjp_5398_;
}
else
{
lean_inc(v_a_5397_);
lean_dec(v___x_5395_);
v___x_5399_ = lean_box(0);
v_isShared_5400_ = v_isSharedCheck_5404_;
goto v_resetjp_5398_;
}
v_resetjp_5398_:
{
lean_object* v___x_5402_; 
if (v_isShared_5400_ == 0)
{
v___x_5402_ = v___x_5399_;
goto v_reusejp_5401_;
}
else
{
lean_object* v_reuseFailAlloc_5403_; 
v_reuseFailAlloc_5403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5403_, 0, v_a_5397_);
v___x_5402_ = v_reuseFailAlloc_5403_;
goto v_reusejp_5401_;
}
v_reusejp_5401_:
{
return v___x_5402_;
}
}
}
}
else
{
lean_object* v_a_5405_; lean_object* v___x_5407_; uint8_t v_isShared_5408_; uint8_t v_isSharedCheck_5412_; 
lean_dec_ref(v___x_5321_);
lean_dec_ref(v___x_5320_);
lean_dec(v_a_5319_);
lean_dec(v_a_5317_);
lean_dec_ref(v___x_5313_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec(v_g_5281_);
v_a_5405_ = lean_ctor_get(v___x_5393_, 0);
v_isSharedCheck_5412_ = !lean_is_exclusive(v___x_5393_);
if (v_isSharedCheck_5412_ == 0)
{
v___x_5407_ = v___x_5393_;
v_isShared_5408_ = v_isSharedCheck_5412_;
goto v_resetjp_5406_;
}
else
{
lean_inc(v_a_5405_);
lean_dec(v___x_5393_);
v___x_5407_ = lean_box(0);
v_isShared_5408_ = v_isSharedCheck_5412_;
goto v_resetjp_5406_;
}
v_resetjp_5406_:
{
lean_object* v___x_5410_; 
if (v_isShared_5408_ == 0)
{
v___x_5410_ = v___x_5407_;
goto v_reusejp_5409_;
}
else
{
lean_object* v_reuseFailAlloc_5411_; 
v_reuseFailAlloc_5411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5411_, 0, v_a_5405_);
v___x_5410_ = v_reuseFailAlloc_5411_;
goto v_reusejp_5409_;
}
v_reusejp_5409_:
{
return v___x_5410_;
}
}
}
}
else
{
lean_object* v_val_5413_; 
v_val_5413_ = lean_ctor_get(v_base_5280_, 0);
lean_inc(v_val_5413_);
lean_dec_ref_known(v_base_5280_, 1);
v_____x_5323_ = v_val_5413_;
v___y_5324_ = v_a_5282_;
v___y_5325_ = v_a_5283_;
v___y_5326_ = v_a_5284_;
v___y_5327_ = v_a_5285_;
v___y_5328_ = v_a_5286_;
v___y_5329_ = v_a_5287_;
goto v___jp_5322_;
}
v___jp_5322_:
{
lean_object* v_fst_5330_; lean_object* v_snd_5331_; lean_object* v___x_5333_; uint8_t v_isShared_5334_; uint8_t v_isSharedCheck_5392_; 
v_fst_5330_ = lean_ctor_get(v_____x_5323_, 0);
v_snd_5331_ = lean_ctor_get(v_____x_5323_, 1);
v_isSharedCheck_5392_ = !lean_is_exclusive(v_____x_5323_);
if (v_isSharedCheck_5392_ == 0)
{
v___x_5333_ = v_____x_5323_;
v_isShared_5334_ = v_isSharedCheck_5392_;
goto v_resetjp_5332_;
}
else
{
lean_inc(v_snd_5331_);
lean_inc(v_fst_5330_);
lean_dec(v_____x_5323_);
v___x_5333_ = lean_box(0);
v_isShared_5334_ = v_isSharedCheck_5392_;
goto v_resetjp_5332_;
}
v_resetjp_5332_:
{
lean_object* v___x_5336_; 
lean_inc(v_fst_5330_);
if (v_isShared_5334_ == 0)
{
lean_ctor_set_tag(v___x_5333_, 1);
lean_ctor_set(v___x_5333_, 1, v___x_5311_);
v___x_5336_ = v___x_5333_;
goto v_reusejp_5335_;
}
else
{
lean_object* v_reuseFailAlloc_5391_; 
v_reuseFailAlloc_5391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5391_, 0, v_fst_5330_);
lean_ctor_set(v_reuseFailAlloc_5391_, 1, v___x_5311_);
v___x_5336_ = v_reuseFailAlloc_5391_;
goto v_reusejp_5335_;
}
v_reusejp_5335_:
{
lean_object* v___x_5337_; lean_object* v___x_5338_; lean_object* v___x_5339_; 
v___x_5337_ = l_Lean_Expr_const___override(v___x_5310_, v___x_5336_);
lean_inc(v_snd_5331_);
v___x_5338_ = l_Lean_Expr_app___override(v___x_5337_, v_snd_5331_);
v___x_5339_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_5338_, v___y_5326_, v___y_5327_, v___y_5328_, v___y_5329_);
if (lean_obj_tag(v___x_5339_) == 0)
{
lean_object* v_a_5340_; lean_object* v___x_5341_; lean_object* v___x_5342_; lean_object* v___x_5343_; lean_object* v___x_5344_; lean_object* v___x_5345_; lean_object* v___x_5346_; lean_object* v___x_5347_; lean_object* v___x_5348_; lean_object* v___x_5349_; lean_object* v___x_5350_; lean_object* v___x_5351_; lean_object* v___x_5352_; 
v_a_5340_ = lean_ctor_get(v___x_5339_, 0);
lean_inc_n(v_a_5340_, 2);
lean_dec_ref_known(v___x_5339_, 1);
v___x_5341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalSMulCast___closed__1));
lean_inc_ref(v___x_5313_);
lean_inc(v_fst_5330_);
v___x_5342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5342_, 0, v_fst_5330_);
lean_ctor_set(v___x_5342_, 1, v___x_5313_);
v___x_5343_ = l_Lean_Expr_const___override(v___x_5341_, v___x_5342_);
lean_inc(v_snd_5331_);
v___x_5344_ = l_Lean_Expr_app___override(v___x_5343_, v_snd_5331_);
lean_inc_n(v_snd_5306_, 2);
v___x_5345_ = l_Lean_Expr_app___override(v___x_5344_, v_snd_5306_);
v___x_5346_ = l_Lean_Expr_app___override(v___x_5345_, v_a_5340_);
v___x_5347_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_evalCast___closed__12));
v___x_5348_ = l_Lean_Expr_const___override(v___x_5347_, v___x_5313_);
v___x_5349_ = l_Lean_Expr_app___override(v___x_5348_, v_snd_5306_);
lean_inc(v_a_5317_);
v___x_5350_ = l_Lean_Expr_app___override(v___x_5349_, v_a_5317_);
v___x_5351_ = l_Lean_Expr_app___override(v___x_5346_, v___x_5350_);
v___x_5352_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_5351_, v___y_5326_, v___y_5327_, v___y_5328_, v___y_5329_);
if (lean_obj_tag(v___x_5352_) == 0)
{
lean_object* v_a_5353_; lean_object* v___x_5354_; 
v_a_5353_ = lean_ctor_get(v___x_5352_, 0);
lean_inc(v_a_5353_);
lean_dec_ref_known(v___x_5352_, 1);
lean_inc(v_a_5340_);
lean_inc(v_snd_5331_);
lean_inc(v_fst_5330_);
v___x_5354_ = lp_mathlib_Mathlib_Tactic_Algebra_mkCache(v_fst_5330_, v_snd_5331_, v_a_5340_, v___y_5326_, v___y_5327_, v___y_5328_, v___y_5329_);
if (lean_obj_tag(v___x_5354_) == 0)
{
lean_object* v_a_5355_; lean_object* v___x_5356_; 
v_a_5355_ = lean_ctor_get(v___x_5354_, 0);
lean_inc(v_a_5355_);
lean_dec_ref_known(v___x_5354_, 1);
v___x_5356_ = lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore(v_fst_5330_, v_fst_5305_, v_snd_5331_, v_snd_5306_, v_a_5340_, v_a_5317_, v_a_5353_, v_a_5355_, v_a_5319_, v___x_5320_, v___x_5321_, v___y_5324_, v___y_5325_, v___y_5326_, v___y_5327_, v___y_5328_, v___y_5329_);
if (lean_obj_tag(v___x_5356_) == 0)
{
lean_object* v_a_5357_; lean_object* v___x_5358_; 
v_a_5357_ = lean_ctor_get(v___x_5356_, 0);
lean_inc(v_a_5357_);
lean_dec_ref_known(v___x_5356_, 1);
v___x_5358_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg(v_g_5281_, v_a_5357_, v___y_5327_);
return v___x_5358_;
}
else
{
lean_object* v_a_5359_; lean_object* v___x_5361_; uint8_t v_isShared_5362_; uint8_t v_isSharedCheck_5366_; 
lean_dec(v_g_5281_);
v_a_5359_ = lean_ctor_get(v___x_5356_, 0);
v_isSharedCheck_5366_ = !lean_is_exclusive(v___x_5356_);
if (v_isSharedCheck_5366_ == 0)
{
v___x_5361_ = v___x_5356_;
v_isShared_5362_ = v_isSharedCheck_5366_;
goto v_resetjp_5360_;
}
else
{
lean_inc(v_a_5359_);
lean_dec(v___x_5356_);
v___x_5361_ = lean_box(0);
v_isShared_5362_ = v_isSharedCheck_5366_;
goto v_resetjp_5360_;
}
v_resetjp_5360_:
{
lean_object* v___x_5364_; 
if (v_isShared_5362_ == 0)
{
v___x_5364_ = v___x_5361_;
goto v_reusejp_5363_;
}
else
{
lean_object* v_reuseFailAlloc_5365_; 
v_reuseFailAlloc_5365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5365_, 0, v_a_5359_);
v___x_5364_ = v_reuseFailAlloc_5365_;
goto v_reusejp_5363_;
}
v_reusejp_5363_:
{
return v___x_5364_;
}
}
}
}
else
{
lean_object* v_a_5367_; lean_object* v___x_5369_; uint8_t v_isShared_5370_; uint8_t v_isSharedCheck_5374_; 
lean_dec(v_a_5353_);
lean_dec(v_a_5340_);
lean_dec(v_snd_5331_);
lean_dec(v_fst_5330_);
lean_dec_ref(v___x_5321_);
lean_dec_ref(v___x_5320_);
lean_dec(v_a_5319_);
lean_dec(v_a_5317_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec(v_g_5281_);
v_a_5367_ = lean_ctor_get(v___x_5354_, 0);
v_isSharedCheck_5374_ = !lean_is_exclusive(v___x_5354_);
if (v_isSharedCheck_5374_ == 0)
{
v___x_5369_ = v___x_5354_;
v_isShared_5370_ = v_isSharedCheck_5374_;
goto v_resetjp_5368_;
}
else
{
lean_inc(v_a_5367_);
lean_dec(v___x_5354_);
v___x_5369_ = lean_box(0);
v_isShared_5370_ = v_isSharedCheck_5374_;
goto v_resetjp_5368_;
}
v_resetjp_5368_:
{
lean_object* v___x_5372_; 
if (v_isShared_5370_ == 0)
{
v___x_5372_ = v___x_5369_;
goto v_reusejp_5371_;
}
else
{
lean_object* v_reuseFailAlloc_5373_; 
v_reuseFailAlloc_5373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5373_, 0, v_a_5367_);
v___x_5372_ = v_reuseFailAlloc_5373_;
goto v_reusejp_5371_;
}
v_reusejp_5371_:
{
return v___x_5372_;
}
}
}
}
else
{
lean_object* v_a_5375_; lean_object* v___x_5377_; uint8_t v_isShared_5378_; uint8_t v_isSharedCheck_5382_; 
lean_dec(v_a_5340_);
lean_dec(v_snd_5331_);
lean_dec(v_fst_5330_);
lean_dec_ref(v___x_5321_);
lean_dec_ref(v___x_5320_);
lean_dec(v_a_5319_);
lean_dec(v_a_5317_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec(v_g_5281_);
v_a_5375_ = lean_ctor_get(v___x_5352_, 0);
v_isSharedCheck_5382_ = !lean_is_exclusive(v___x_5352_);
if (v_isSharedCheck_5382_ == 0)
{
v___x_5377_ = v___x_5352_;
v_isShared_5378_ = v_isSharedCheck_5382_;
goto v_resetjp_5376_;
}
else
{
lean_inc(v_a_5375_);
lean_dec(v___x_5352_);
v___x_5377_ = lean_box(0);
v_isShared_5378_ = v_isSharedCheck_5382_;
goto v_resetjp_5376_;
}
v_resetjp_5376_:
{
lean_object* v___x_5380_; 
if (v_isShared_5378_ == 0)
{
v___x_5380_ = v___x_5377_;
goto v_reusejp_5379_;
}
else
{
lean_object* v_reuseFailAlloc_5381_; 
v_reuseFailAlloc_5381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5381_, 0, v_a_5375_);
v___x_5380_ = v_reuseFailAlloc_5381_;
goto v_reusejp_5379_;
}
v_reusejp_5379_:
{
return v___x_5380_;
}
}
}
}
else
{
lean_object* v_a_5383_; lean_object* v___x_5385_; uint8_t v_isShared_5386_; uint8_t v_isSharedCheck_5390_; 
lean_dec(v_snd_5331_);
lean_dec(v_fst_5330_);
lean_dec_ref(v___x_5321_);
lean_dec_ref(v___x_5320_);
lean_dec(v_a_5319_);
lean_dec(v_a_5317_);
lean_dec_ref(v___x_5313_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec(v_g_5281_);
v_a_5383_ = lean_ctor_get(v___x_5339_, 0);
v_isSharedCheck_5390_ = !lean_is_exclusive(v___x_5339_);
if (v_isSharedCheck_5390_ == 0)
{
v___x_5385_ = v___x_5339_;
v_isShared_5386_ = v_isSharedCheck_5390_;
goto v_resetjp_5384_;
}
else
{
lean_inc(v_a_5383_);
lean_dec(v___x_5339_);
v___x_5385_ = lean_box(0);
v_isShared_5386_ = v_isSharedCheck_5390_;
goto v_resetjp_5384_;
}
v_resetjp_5384_:
{
lean_object* v___x_5388_; 
if (v_isShared_5386_ == 0)
{
v___x_5388_ = v___x_5385_;
goto v_reusejp_5387_;
}
else
{
lean_object* v_reuseFailAlloc_5389_; 
v_reuseFailAlloc_5389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5389_, 0, v_a_5383_);
v___x_5388_ = v_reuseFailAlloc_5389_;
goto v_reusejp_5387_;
}
v_reusejp_5387_:
{
return v___x_5388_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_5414_; lean_object* v___x_5416_; uint8_t v_isShared_5417_; uint8_t v_isSharedCheck_5421_; 
lean_dec(v_a_5317_);
lean_dec_ref(v___x_5313_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec_ref(v___x_5300_);
lean_dec(v_a_5294_);
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v_a_5414_ = lean_ctor_get(v___x_5318_, 0);
v_isSharedCheck_5421_ = !lean_is_exclusive(v___x_5318_);
if (v_isSharedCheck_5421_ == 0)
{
v___x_5416_ = v___x_5318_;
v_isShared_5417_ = v_isSharedCheck_5421_;
goto v_resetjp_5415_;
}
else
{
lean_inc(v_a_5414_);
lean_dec(v___x_5318_);
v___x_5416_ = lean_box(0);
v_isShared_5417_ = v_isSharedCheck_5421_;
goto v_resetjp_5415_;
}
v_resetjp_5415_:
{
lean_object* v___x_5419_; 
if (v_isShared_5417_ == 0)
{
v___x_5419_ = v___x_5416_;
goto v_reusejp_5418_;
}
else
{
lean_object* v_reuseFailAlloc_5420_; 
v_reuseFailAlloc_5420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5420_, 0, v_a_5414_);
v___x_5419_ = v_reuseFailAlloc_5420_;
goto v_reusejp_5418_;
}
v_reusejp_5418_:
{
return v___x_5419_;
}
}
}
}
else
{
lean_object* v_a_5422_; lean_object* v___x_5424_; uint8_t v_isShared_5425_; uint8_t v_isSharedCheck_5429_; 
lean_dec_ref(v___x_5313_);
lean_dec(v_snd_5306_);
lean_dec(v_fst_5305_);
lean_dec_ref(v___x_5300_);
lean_dec(v_a_5294_);
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v_a_5422_ = lean_ctor_get(v___x_5316_, 0);
v_isSharedCheck_5429_ = !lean_is_exclusive(v___x_5316_);
if (v_isSharedCheck_5429_ == 0)
{
v___x_5424_ = v___x_5316_;
v_isShared_5425_ = v_isSharedCheck_5429_;
goto v_resetjp_5423_;
}
else
{
lean_inc(v_a_5422_);
lean_dec(v___x_5316_);
v___x_5424_ = lean_box(0);
v_isShared_5425_ = v_isSharedCheck_5429_;
goto v_resetjp_5423_;
}
v_resetjp_5423_:
{
lean_object* v___x_5427_; 
if (v_isShared_5425_ == 0)
{
v___x_5427_ = v___x_5424_;
goto v_reusejp_5426_;
}
else
{
lean_object* v_reuseFailAlloc_5428_; 
v_reuseFailAlloc_5428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5428_, 0, v_a_5422_);
v___x_5427_ = v_reuseFailAlloc_5428_;
goto v_reusejp_5426_;
}
v_reusejp_5426_:
{
return v___x_5427_;
}
}
}
}
}
}
else
{
lean_object* v_a_5432_; lean_object* v___x_5434_; uint8_t v_isShared_5435_; uint8_t v_isSharedCheck_5439_; 
lean_dec_ref(v___x_5300_);
lean_dec(v_a_5294_);
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v_a_5432_ = lean_ctor_get(v___x_5303_, 0);
v_isSharedCheck_5439_ = !lean_is_exclusive(v___x_5303_);
if (v_isSharedCheck_5439_ == 0)
{
v___x_5434_ = v___x_5303_;
v_isShared_5435_ = v_isSharedCheck_5439_;
goto v_resetjp_5433_;
}
else
{
lean_inc(v_a_5432_);
lean_dec(v___x_5303_);
v___x_5434_ = lean_box(0);
v_isShared_5435_ = v_isSharedCheck_5439_;
goto v_resetjp_5433_;
}
v_resetjp_5433_:
{
lean_object* v___x_5437_; 
if (v_isShared_5435_ == 0)
{
v___x_5437_ = v___x_5434_;
goto v_reusejp_5436_;
}
else
{
lean_object* v_reuseFailAlloc_5438_; 
v_reuseFailAlloc_5438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5438_, 0, v_a_5432_);
v___x_5437_ = v_reuseFailAlloc_5438_;
goto v_reusejp_5436_;
}
v_reusejp_5436_:
{
return v___x_5437_;
}
}
}
}
}
else
{
lean_object* v_a_5440_; lean_object* v___x_5442_; uint8_t v_isShared_5443_; uint8_t v_isSharedCheck_5447_; 
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v_a_5440_ = lean_ctor_get(v___x_5293_, 0);
v_isSharedCheck_5447_ = !lean_is_exclusive(v___x_5293_);
if (v_isSharedCheck_5447_ == 0)
{
v___x_5442_ = v___x_5293_;
v_isShared_5443_ = v_isSharedCheck_5447_;
goto v_resetjp_5441_;
}
else
{
lean_inc(v_a_5440_);
lean_dec(v___x_5293_);
v___x_5442_ = lean_box(0);
v_isShared_5443_ = v_isSharedCheck_5447_;
goto v_resetjp_5441_;
}
v_resetjp_5441_:
{
lean_object* v___x_5445_; 
if (v_isShared_5443_ == 0)
{
v___x_5445_ = v___x_5442_;
goto v_reusejp_5444_;
}
else
{
lean_object* v_reuseFailAlloc_5446_; 
v_reuseFailAlloc_5446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5446_, 0, v_a_5440_);
v___x_5445_ = v_reuseFailAlloc_5446_;
goto v_reusejp_5444_;
}
v_reusejp_5444_:
{
return v___x_5445_;
}
}
}
}
else
{
lean_object* v_a_5448_; lean_object* v___x_5450_; uint8_t v_isShared_5451_; uint8_t v_isSharedCheck_5455_; 
lean_dec(v_g_5281_);
lean_dec(v_base_5280_);
v_a_5448_ = lean_ctor_get(v___x_5289_, 0);
v_isSharedCheck_5455_ = !lean_is_exclusive(v___x_5289_);
if (v_isSharedCheck_5455_ == 0)
{
v___x_5450_ = v___x_5289_;
v_isShared_5451_ = v_isSharedCheck_5455_;
goto v_resetjp_5449_;
}
else
{
lean_inc(v_a_5448_);
lean_dec(v___x_5289_);
v___x_5450_ = lean_box(0);
v_isShared_5451_ = v_isSharedCheck_5455_;
goto v_resetjp_5449_;
}
v_resetjp_5449_:
{
lean_object* v___x_5453_; 
if (v_isShared_5451_ == 0)
{
v___x_5453_ = v___x_5450_;
goto v_reusejp_5452_;
}
else
{
lean_object* v_reuseFailAlloc_5454_; 
v_reuseFailAlloc_5454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5454_, 0, v_a_5448_);
v___x_5453_ = v_reuseFailAlloc_5454_;
goto v_reusejp_5452_;
}
v_reusejp_5452_:
{
return v___x_5453_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra_proveEq___boxed(lean_object* v_base_5456_, lean_object* v_g_5457_, lean_object* v_a_5458_, lean_object* v_a_5459_, lean_object* v_a_5460_, lean_object* v_a_5461_, lean_object* v_a_5462_, lean_object* v_a_5463_, lean_object* v_a_5464_){
_start:
{
lean_object* v_res_5465_; 
v_res_5465_ = lp_mathlib_Mathlib_Tactic_Algebra_proveEq(v_base_5456_, v_g_5457_, v_a_5458_, v_a_5459_, v_a_5460_, v_a_5461_, v_a_5462_, v_a_5463_);
lean_dec(v_a_5463_);
lean_dec_ref(v_a_5462_);
lean_dec(v_a_5461_);
lean_dec_ref(v_a_5460_);
lean_dec(v_a_5459_);
lean_dec_ref(v_a_5458_);
return v_res_5465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1(lean_object* v_mvarId_5466_, lean_object* v_val_5467_, lean_object* v___y_5468_, lean_object* v___y_5469_, lean_object* v___y_5470_, lean_object* v___y_5471_, lean_object* v___y_5472_, lean_object* v___y_5473_){
_start:
{
lean_object* v___x_5475_; 
v___x_5475_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___redArg(v_mvarId_5466_, v_val_5467_, v___y_5471_);
return v___x_5475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1___boxed(lean_object* v_mvarId_5476_, lean_object* v_val_5477_, lean_object* v___y_5478_, lean_object* v___y_5479_, lean_object* v___y_5480_, lean_object* v___y_5481_, lean_object* v___y_5482_, lean_object* v___y_5483_, lean_object* v___y_5484_){
_start:
{
lean_object* v_res_5485_; 
v_res_5485_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1(v_mvarId_5476_, v_val_5477_, v___y_5478_, v___y_5479_, v___y_5480_, v___y_5481_, v___y_5482_, v___y_5483_);
lean_dec(v___y_5483_);
lean_dec_ref(v___y_5482_);
lean_dec(v___y_5481_);
lean_dec_ref(v___y_5480_);
lean_dec(v___y_5479_);
lean_dec_ref(v___y_5478_);
return v_res_5485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1(lean_object* v_00_u03b2_5486_, lean_object* v_x_5487_, lean_object* v_x_5488_, lean_object* v_x_5489_){
_start:
{
lean_object* v___x_5490_; 
v___x_5490_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1___redArg(v_x_5487_, v_x_5488_, v_x_5489_);
return v___x_5490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_5491_, lean_object* v_x_5492_, size_t v_x_5493_, size_t v_x_5494_, lean_object* v_x_5495_, lean_object* v_x_5496_){
_start:
{
lean_object* v___x_5497_; 
v___x_5497_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___redArg(v_x_5492_, v_x_5493_, v_x_5494_, v_x_5495_, v_x_5496_);
return v___x_5497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_5498_, lean_object* v_x_5499_, lean_object* v_x_5500_, lean_object* v_x_5501_, lean_object* v_x_5502_, lean_object* v_x_5503_){
_start:
{
size_t v_x_12018__boxed_5504_; size_t v_x_12019__boxed_5505_; lean_object* v_res_5506_; 
v_x_12018__boxed_5504_ = lean_unbox_usize(v_x_5500_);
lean_dec(v_x_5500_);
v_x_12019__boxed_5505_ = lean_unbox_usize(v_x_5501_);
lean_dec(v_x_5501_);
v_res_5506_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2(v_00_u03b2_5498_, v_x_5499_, v_x_12018__boxed_5504_, v_x_12019__boxed_5505_, v_x_5502_, v_x_5503_);
return v_res_5506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_5507_, lean_object* v_n_5508_, lean_object* v_k_5509_, lean_object* v_v_5510_){
_start:
{
lean_object* v___x_5511_; 
v___x_5511_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3___redArg(v_n_5508_, v_k_5509_, v_v_5510_);
return v___x_5511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_5512_, size_t v_depth_5513_, lean_object* v_keys_5514_, lean_object* v_vals_5515_, lean_object* v_heq_5516_, lean_object* v_i_5517_, lean_object* v_entries_5518_){
_start:
{
lean_object* v___x_5519_; 
v___x_5519_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_5513_, v_keys_5514_, v_vals_5515_, v_i_5517_, v_entries_5518_);
return v___x_5519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b2_5520_, lean_object* v_depth_5521_, lean_object* v_keys_5522_, lean_object* v_vals_5523_, lean_object* v_heq_5524_, lean_object* v_i_5525_, lean_object* v_entries_5526_){
_start:
{
size_t v_depth_boxed_5527_; lean_object* v_res_5528_; 
v_depth_boxed_5527_ = lean_unbox_usize(v_depth_5521_);
lean_dec(v_depth_5521_);
v_res_5528_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__4(v_00_u03b2_5520_, v_depth_boxed_5527_, v_keys_5522_, v_vals_5523_, v_heq_5524_, v_i_5525_, v_entries_5526_);
lean_dec_ref(v_vals_5523_);
lean_dec_ref(v_keys_5522_);
return v_res_5528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_5529_, lean_object* v_x_5530_, lean_object* v_x_5531_, lean_object* v_x_5532_, lean_object* v_x_5533_){
_start:
{
lean_object* v___x_5534_; 
v___x_5534_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Algebra_proveEq_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_x_5530_, v_x_5531_, v_x_5532_, v_x_5533_);
return v___x_5534_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_5548_; lean_object* v___x_5549_; lean_object* v___x_5550_; 
v___x_5548_ = lean_box(0);
v___x_5549_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_5550_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5550_, 0, v___x_5549_);
lean_ctor_set(v___x_5550_, 1, v___x_5548_);
return v___x_5550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg(){
_start:
{
lean_object* v___x_5552_; lean_object* v___x_5553_; 
v___x_5552_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___closed__0);
v___x_5553_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5553_, 0, v___x_5552_);
return v___x_5553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg___boxed(lean_object* v___y_5554_){
_start:
{
lean_object* v_res_5555_; 
v_res_5555_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg();
return v_res_5555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0(lean_object* v_00_u03b1_5556_, lean_object* v___y_5557_, lean_object* v___y_5558_, lean_object* v___y_5559_, lean_object* v___y_5560_, lean_object* v___y_5561_, lean_object* v___y_5562_, lean_object* v___y_5563_, lean_object* v___y_5564_){
_start:
{
lean_object* v___x_5566_; 
v___x_5566_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg();
return v___x_5566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___boxed(lean_object* v_00_u03b1_5567_, lean_object* v___y_5568_, lean_object* v___y_5569_, lean_object* v___y_5570_, lean_object* v___y_5571_, lean_object* v___y_5572_, lean_object* v___y_5573_, lean_object* v___y_5574_, lean_object* v___y_5575_, lean_object* v___y_5576_){
_start:
{
lean_object* v_res_5577_; 
v_res_5577_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0(v_00_u03b1_5567_, v___y_5568_, v___y_5569_, v___y_5570_, v___y_5571_, v___y_5572_, v___y_5573_, v___y_5574_, v___y_5575_);
lean_dec(v___y_5575_);
lean_dec_ref(v___y_5574_);
lean_dec(v___y_5573_);
lean_dec_ref(v___y_5572_);
lean_dec(v___y_5571_);
lean_dec_ref(v___y_5570_);
lean_dec(v___y_5569_);
lean_dec_ref(v___y_5568_);
return v_res_5577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0(lean_object* v_e_5578_, lean_object* v_x_5579_, lean_object* v___y_5580_, lean_object* v___y_5581_, lean_object* v___y_5582_, lean_object* v___y_5583_){
_start:
{
lean_object* v___x_5585_; 
v___x_5585_ = lp_mathlib_Mathlib_Tactic_Algebra_preprocess(v_e_5578_, v___y_5580_, v___y_5581_, v___y_5582_, v___y_5583_);
return v___x_5585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0___boxed(lean_object* v_e_5586_, lean_object* v_x_5587_, lean_object* v___y_5588_, lean_object* v___y_5589_, lean_object* v___y_5590_, lean_object* v___y_5591_, lean_object* v___y_5592_){
_start:
{
lean_object* v_res_5593_; 
v_res_5593_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__0(v_e_5586_, v_x_5587_, v___y_5588_, v___y_5589_, v___y_5590_, v___y_5591_);
lean_dec(v___y_5591_);
lean_dec_ref(v___y_5590_);
lean_dec(v___y_5589_);
lean_dec_ref(v___y_5588_);
lean_dec_ref(v_x_5587_);
return v_res_5593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1(lean_object* v___f_5594_, lean_object* v___x_5595_, lean_object* v___y_5596_, lean_object* v___y_5597_, lean_object* v___y_5598_, lean_object* v___y_5599_, lean_object* v___y_5600_, lean_object* v___y_5601_, lean_object* v___y_5602_, lean_object* v___y_5603_){
_start:
{
lean_object* v___x_5605_; 
v___x_5605_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5597_, v___y_5600_, v___y_5601_, v___y_5602_, v___y_5603_);
if (lean_obj_tag(v___x_5605_) == 0)
{
lean_object* v_a_5606_; uint8_t v___x_5607_; lean_object* v___x_5608_; lean_object* v___x_5609_; 
v_a_5606_ = lean_ctor_get(v___x_5605_, 0);
lean_inc(v_a_5606_);
lean_dec_ref_known(v___x_5605_, 1);
v___x_5607_ = 0;
v___x_5608_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_5609_ = lp_mathlib_Mathlib_Tactic_transformAtTarget(v___f_5594_, v___x_5595_, v___x_5607_, v_a_5606_, v___x_5608_, v___y_5600_, v___y_5601_, v___y_5602_, v___y_5603_);
if (lean_obj_tag(v___x_5609_) == 0)
{
lean_object* v_a_5610_; 
v_a_5610_ = lean_ctor_get(v___x_5609_, 0);
lean_inc(v_a_5610_);
lean_dec_ref_known(v___x_5609_, 1);
if (lean_obj_tag(v_a_5610_) == 1)
{
lean_object* v_val_5611_; lean_object* v___x_5612_; lean_object* v___x_5613_; lean_object* v___x_5614_; 
v_val_5611_ = lean_ctor_get(v_a_5610_, 0);
lean_inc(v_val_5611_);
lean_dec_ref_known(v_a_5610_, 1);
v___x_5612_ = lean_box(0);
v___x_5613_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5613_, 0, v_val_5611_);
lean_ctor_set(v___x_5613_, 1, v___x_5612_);
v___x_5614_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_5613_, v___y_5597_, v___y_5600_, v___y_5601_, v___y_5602_, v___y_5603_);
return v___x_5614_;
}
else
{
lean_object* v___x_5615_; lean_object* v___x_5616_; 
lean_dec(v_a_5610_);
v___x_5615_ = lean_box(0);
v___x_5616_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_5615_, v___y_5597_, v___y_5600_, v___y_5601_, v___y_5602_, v___y_5603_);
return v___x_5616_;
}
}
else
{
lean_object* v_a_5617_; lean_object* v___x_5619_; uint8_t v_isShared_5620_; uint8_t v_isSharedCheck_5624_; 
v_a_5617_ = lean_ctor_get(v___x_5609_, 0);
v_isSharedCheck_5624_ = !lean_is_exclusive(v___x_5609_);
if (v_isSharedCheck_5624_ == 0)
{
v___x_5619_ = v___x_5609_;
v_isShared_5620_ = v_isSharedCheck_5624_;
goto v_resetjp_5618_;
}
else
{
lean_inc(v_a_5617_);
lean_dec(v___x_5609_);
v___x_5619_ = lean_box(0);
v_isShared_5620_ = v_isSharedCheck_5624_;
goto v_resetjp_5618_;
}
v_resetjp_5618_:
{
lean_object* v___x_5622_; 
if (v_isShared_5620_ == 0)
{
v___x_5622_ = v___x_5619_;
goto v_reusejp_5621_;
}
else
{
lean_object* v_reuseFailAlloc_5623_; 
v_reuseFailAlloc_5623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5623_, 0, v_a_5617_);
v___x_5622_ = v_reuseFailAlloc_5623_;
goto v_reusejp_5621_;
}
v_reusejp_5621_:
{
return v___x_5622_;
}
}
}
}
else
{
lean_object* v_a_5625_; lean_object* v___x_5627_; uint8_t v_isShared_5628_; uint8_t v_isSharedCheck_5632_; 
lean_dec_ref(v___x_5595_);
lean_dec_ref(v___f_5594_);
v_a_5625_ = lean_ctor_get(v___x_5605_, 0);
v_isSharedCheck_5632_ = !lean_is_exclusive(v___x_5605_);
if (v_isSharedCheck_5632_ == 0)
{
v___x_5627_ = v___x_5605_;
v_isShared_5628_ = v_isSharedCheck_5632_;
goto v_resetjp_5626_;
}
else
{
lean_inc(v_a_5625_);
lean_dec(v___x_5605_);
v___x_5627_ = lean_box(0);
v_isShared_5628_ = v_isSharedCheck_5632_;
goto v_resetjp_5626_;
}
v_resetjp_5626_:
{
lean_object* v___x_5630_; 
if (v_isShared_5628_ == 0)
{
v___x_5630_ = v___x_5627_;
goto v_reusejp_5629_;
}
else
{
lean_object* v_reuseFailAlloc_5631_; 
v_reuseFailAlloc_5631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5631_, 0, v_a_5625_);
v___x_5630_ = v_reuseFailAlloc_5631_;
goto v_reusejp_5629_;
}
v_reusejp_5629_:
{
return v___x_5630_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1___boxed(lean_object* v___f_5633_, lean_object* v___x_5634_, lean_object* v___y_5635_, lean_object* v___y_5636_, lean_object* v___y_5637_, lean_object* v___y_5638_, lean_object* v___y_5639_, lean_object* v___y_5640_, lean_object* v___y_5641_, lean_object* v___y_5642_, lean_object* v___y_5643_){
_start:
{
lean_object* v_res_5644_; 
v_res_5644_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__1(v___f_5633_, v___x_5634_, v___y_5635_, v___y_5636_, v___y_5637_, v___y_5638_, v___y_5639_, v___y_5640_, v___y_5641_, v___y_5642_);
lean_dec(v___y_5642_);
lean_dec_ref(v___y_5641_);
lean_dec(v___y_5640_);
lean_dec_ref(v___y_5639_);
lean_dec(v___y_5638_);
lean_dec_ref(v___y_5637_);
lean_dec(v___y_5636_);
lean_dec_ref(v___y_5635_);
return v_res_5644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2(uint8_t v___x_5645_, lean_object* v_e_5646_, lean_object* v___y_5647_, lean_object* v___y_5648_, lean_object* v___y_5649_, lean_object* v___y_5650_){
_start:
{
lean_object* v___x_5652_; lean_object* v___x_5653_; lean_object* v___x_5654_; 
v___x_5652_ = lean_box(0);
v___x_5653_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_5653_, 0, v_e_5646_);
lean_ctor_set(v___x_5653_, 1, v___x_5652_);
lean_ctor_set_uint8(v___x_5653_, sizeof(void*)*2, v___x_5645_);
v___x_5654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5654_, 0, v___x_5653_);
return v___x_5654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2___boxed(lean_object* v___x_5655_, lean_object* v_e_5656_, lean_object* v___y_5657_, lean_object* v___y_5658_, lean_object* v___y_5659_, lean_object* v___y_5660_, lean_object* v___y_5661_){
_start:
{
uint8_t v___x_1202__boxed_5662_; lean_object* v_res_5663_; 
v___x_1202__boxed_5662_ = lean_unbox(v___x_5655_);
v_res_5663_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2(v___x_1202__boxed_5662_, v_e_5656_, v___y_5657_, v___y_5658_, v___y_5659_, v___y_5660_);
lean_dec(v___y_5660_);
lean_dec_ref(v___y_5659_);
lean_dec(v___y_5658_);
lean_dec_ref(v___y_5657_);
return v_res_5663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3(lean_object* v___f_5664_, lean_object* v___f_5665_, lean_object* v___y_5666_, lean_object* v___y_5667_, lean_object* v___y_5668_, lean_object* v___y_5669_, lean_object* v___y_5670_, lean_object* v___y_5671_, lean_object* v___y_5672_, lean_object* v___y_5673_){
_start:
{
lean_object* v___x_5675_; 
v___x_5675_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5664_, v___y_5666_, v___y_5667_, v___y_5668_, v___y_5669_, v___y_5670_, v___y_5671_, v___y_5672_, v___y_5673_);
if (lean_obj_tag(v___x_5675_) == 0)
{
lean_object* v___x_5676_; 
lean_dec_ref_known(v___x_5675_, 1);
v___x_5676_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5667_, v___y_5670_, v___y_5671_, v___y_5672_, v___y_5673_);
if (lean_obj_tag(v___x_5676_) == 0)
{
lean_object* v_a_5677_; uint8_t v___x_5678_; lean_object* v___x_5679_; lean_object* v___x_5680_; lean_object* v___x_5681_; 
v_a_5677_ = lean_ctor_get(v___x_5676_, 0);
lean_inc(v_a_5677_);
lean_dec_ref_known(v___x_5676_, 1);
v___x_5678_ = 1;
v___x_5679_ = lean_box(0);
v___x_5680_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_proveEq___boxed), 9, 2);
lean_closure_set(v___x_5680_, 0, v___x_5679_);
lean_closure_set(v___x_5680_, 1, v_a_5677_);
v___x_5681_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_5678_, v___x_5680_, v___f_5665_, v___y_5670_, v___y_5671_, v___y_5672_, v___y_5673_);
return v___x_5681_;
}
else
{
lean_object* v_a_5682_; lean_object* v___x_5684_; uint8_t v_isShared_5685_; uint8_t v_isSharedCheck_5689_; 
lean_dec_ref(v___f_5665_);
v_a_5682_ = lean_ctor_get(v___x_5676_, 0);
v_isSharedCheck_5689_ = !lean_is_exclusive(v___x_5676_);
if (v_isSharedCheck_5689_ == 0)
{
v___x_5684_ = v___x_5676_;
v_isShared_5685_ = v_isSharedCheck_5689_;
goto v_resetjp_5683_;
}
else
{
lean_inc(v_a_5682_);
lean_dec(v___x_5676_);
v___x_5684_ = lean_box(0);
v_isShared_5685_ = v_isSharedCheck_5689_;
goto v_resetjp_5683_;
}
v_resetjp_5683_:
{
lean_object* v___x_5687_; 
if (v_isShared_5685_ == 0)
{
v___x_5687_ = v___x_5684_;
goto v_reusejp_5686_;
}
else
{
lean_object* v_reuseFailAlloc_5688_; 
v_reuseFailAlloc_5688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5688_, 0, v_a_5682_);
v___x_5687_ = v_reuseFailAlloc_5688_;
goto v_reusejp_5686_;
}
v_reusejp_5686_:
{
return v___x_5687_;
}
}
}
}
else
{
lean_dec_ref(v___f_5665_);
return v___x_5675_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3___boxed(lean_object* v___f_5690_, lean_object* v___f_5691_, lean_object* v___y_5692_, lean_object* v___y_5693_, lean_object* v___y_5694_, lean_object* v___y_5695_, lean_object* v___y_5696_, lean_object* v___y_5697_, lean_object* v___y_5698_, lean_object* v___y_5699_, lean_object* v___y_5700_){
_start:
{
lean_object* v_res_5701_; 
v_res_5701_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3(v___f_5690_, v___f_5691_, v___y_5692_, v___y_5693_, v___y_5694_, v___y_5695_, v___y_5696_, v___y_5697_, v___y_5698_, v___y_5699_);
lean_dec(v___y_5699_);
lean_dec_ref(v___y_5698_);
lean_dec(v___y_5697_);
lean_dec_ref(v___y_5696_);
lean_dec(v___y_5695_);
lean_dec_ref(v___y_5694_);
lean_dec(v___y_5693_);
lean_dec_ref(v___y_5692_);
return v_res_5701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1(lean_object* v_x_5706_, lean_object* v_a_5707_, lean_object* v_a_5708_, lean_object* v_a_5709_, lean_object* v_a_5710_, lean_object* v_a_5711_, lean_object* v_a_5712_, lean_object* v_a_5713_, lean_object* v_a_5714_){
_start:
{
lean_object* v___x_5716_; uint8_t v___x_5717_; 
v___x_5716_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_algebra___closed__0));
v___x_5717_ = l_Lean_Syntax_isOfKind(v_x_5706_, v___x_5716_);
if (v___x_5717_ == 0)
{
lean_object* v___x_5718_; 
v___x_5718_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg();
return v___x_5718_;
}
else
{
lean_object* v___f_5719_; lean_object* v___x_5720_; lean_object* v___f_5721_; lean_object* v___f_5722_; lean_object* v___x_5723_; 
v___f_5719_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___closed__1));
v___x_5720_ = lean_box(v___x_5717_);
v___f_5721_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__2___boxed), 7, 1);
lean_closure_set(v___f_5721_, 0, v___x_5720_);
v___f_5722_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___lam__3___boxed), 11, 2);
lean_closure_set(v___f_5722_, 0, v___f_5719_);
lean_closure_set(v___f_5722_, 1, v___f_5721_);
v___x_5723_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5722_, v_a_5707_, v_a_5708_, v_a_5709_, v_a_5710_, v_a_5711_, v_a_5712_, v_a_5713_, v_a_5714_);
return v___x_5723_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1___boxed(lean_object* v_x_5724_, lean_object* v_a_5725_, lean_object* v_a_5726_, lean_object* v_a_5727_, lean_object* v_a_5728_, lean_object* v_a_5729_, lean_object* v_a_5730_, lean_object* v_a_5731_, lean_object* v_a_5732_, lean_object* v_a_5733_){
_start:
{
lean_object* v_res_5734_; 
v_res_5734_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1(v_x_5724_, v_a_5725_, v_a_5726_, v_a_5727_, v_a_5728_, v_a_5729_, v_a_5730_, v_a_5731_, v_a_5732_);
lean_dec(v_a_5732_);
lean_dec_ref(v_a_5731_);
lean_dec(v_a_5730_);
lean_dec_ref(v_a_5729_);
lean_dec(v_a_5728_);
lean_dec_ref(v_a_5727_);
lean_dec(v_a_5726_);
lean_dec_ref(v_a_5725_);
return v_res_5734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1(lean_object* v___f_5766_, lean_object* v___y_5767_, lean_object* v___y_5768_, lean_object* v___y_5769_, lean_object* v___y_5770_, lean_object* v___y_5771_, lean_object* v___y_5772_, lean_object* v___y_5773_, lean_object* v___y_5774_){
_start:
{
lean_object* v___x_5776_; 
v___x_5776_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5768_, v___y_5771_, v___y_5772_, v___y_5773_, v___y_5774_);
if (lean_obj_tag(v___x_5776_) == 0)
{
lean_object* v_a_5777_; lean_object* v___x_5778_; uint8_t v___x_5779_; lean_object* v___x_5780_; lean_object* v___x_5781_; 
v_a_5777_ = lean_ctor_get(v___x_5776_, 0);
lean_inc(v_a_5777_);
lean_dec_ref_known(v___x_5776_, 1);
v___x_5778_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Algebra_Basic_0__Mathlib_Tactic_Algebra_proveEq_algCore___closed__0));
v___x_5779_ = 0;
v___x_5780_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_5781_ = lp_mathlib_Mathlib_Tactic_transformAtTarget(v___f_5766_, v___x_5778_, v___x_5779_, v_a_5777_, v___x_5780_, v___y_5771_, v___y_5772_, v___y_5773_, v___y_5774_);
if (lean_obj_tag(v___x_5781_) == 0)
{
lean_object* v_a_5782_; 
v_a_5782_ = lean_ctor_get(v___x_5781_, 0);
lean_inc(v_a_5782_);
lean_dec_ref_known(v___x_5781_, 1);
if (lean_obj_tag(v_a_5782_) == 1)
{
lean_object* v_val_5783_; lean_object* v___x_5784_; lean_object* v___x_5785_; lean_object* v___x_5786_; 
v_val_5783_ = lean_ctor_get(v_a_5782_, 0);
lean_inc(v_val_5783_);
lean_dec_ref_known(v_a_5782_, 1);
v___x_5784_ = lean_box(0);
v___x_5785_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5785_, 0, v_val_5783_);
lean_ctor_set(v___x_5785_, 1, v___x_5784_);
v___x_5786_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_5785_, v___y_5768_, v___y_5771_, v___y_5772_, v___y_5773_, v___y_5774_);
return v___x_5786_;
}
else
{
lean_object* v___x_5787_; lean_object* v___x_5788_; 
lean_dec(v_a_5782_);
v___x_5787_ = lean_box(0);
v___x_5788_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_5787_, v___y_5768_, v___y_5771_, v___y_5772_, v___y_5773_, v___y_5774_);
return v___x_5788_;
}
}
else
{
lean_object* v_a_5789_; lean_object* v___x_5791_; uint8_t v_isShared_5792_; uint8_t v_isSharedCheck_5796_; 
v_a_5789_ = lean_ctor_get(v___x_5781_, 0);
v_isSharedCheck_5796_ = !lean_is_exclusive(v___x_5781_);
if (v_isSharedCheck_5796_ == 0)
{
v___x_5791_ = v___x_5781_;
v_isShared_5792_ = v_isSharedCheck_5796_;
goto v_resetjp_5790_;
}
else
{
lean_inc(v_a_5789_);
lean_dec(v___x_5781_);
v___x_5791_ = lean_box(0);
v_isShared_5792_ = v_isSharedCheck_5796_;
goto v_resetjp_5790_;
}
v_resetjp_5790_:
{
lean_object* v___x_5794_; 
if (v_isShared_5792_ == 0)
{
v___x_5794_ = v___x_5791_;
goto v_reusejp_5793_;
}
else
{
lean_object* v_reuseFailAlloc_5795_; 
v_reuseFailAlloc_5795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5795_, 0, v_a_5789_);
v___x_5794_ = v_reuseFailAlloc_5795_;
goto v_reusejp_5793_;
}
v_reusejp_5793_:
{
return v___x_5794_;
}
}
}
}
else
{
lean_object* v_a_5797_; lean_object* v___x_5799_; uint8_t v_isShared_5800_; uint8_t v_isSharedCheck_5804_; 
lean_dec_ref(v___f_5766_);
v_a_5797_ = lean_ctor_get(v___x_5776_, 0);
v_isSharedCheck_5804_ = !lean_is_exclusive(v___x_5776_);
if (v_isSharedCheck_5804_ == 0)
{
v___x_5799_ = v___x_5776_;
v_isShared_5800_ = v_isSharedCheck_5804_;
goto v_resetjp_5798_;
}
else
{
lean_inc(v_a_5797_);
lean_dec(v___x_5776_);
v___x_5799_ = lean_box(0);
v_isShared_5800_ = v_isSharedCheck_5804_;
goto v_resetjp_5798_;
}
v_resetjp_5798_:
{
lean_object* v___x_5802_; 
if (v_isShared_5800_ == 0)
{
v___x_5802_ = v___x_5799_;
goto v_reusejp_5801_;
}
else
{
lean_object* v_reuseFailAlloc_5803_; 
v_reuseFailAlloc_5803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5803_, 0, v_a_5797_);
v___x_5802_ = v_reuseFailAlloc_5803_;
goto v_reusejp_5801_;
}
v_reusejp_5801_:
{
return v___x_5802_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1___boxed(lean_object* v___f_5805_, lean_object* v___y_5806_, lean_object* v___y_5807_, lean_object* v___y_5808_, lean_object* v___y_5809_, lean_object* v___y_5810_, lean_object* v___y_5811_, lean_object* v___y_5812_, lean_object* v___y_5813_, lean_object* v___y_5814_){
_start:
{
lean_object* v_res_5815_; 
v_res_5815_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__1(v___f_5805_, v___y_5806_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_, v___y_5811_, v___y_5812_, v___y_5813_);
lean_dec(v___y_5813_);
lean_dec_ref(v___y_5812_);
lean_dec(v___y_5811_);
lean_dec_ref(v___y_5810_);
lean_dec(v___y_5809_);
lean_dec_ref(v___y_5808_);
lean_dec(v___y_5807_);
lean_dec_ref(v___y_5806_);
return v_res_5815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0(lean_object* v___x_5816_, uint8_t v___x_5817_, lean_object* v_e_5818_, lean_object* v___y_5819_, lean_object* v___y_5820_, lean_object* v___y_5821_, lean_object* v___y_5822_){
_start:
{
lean_object* v___x_5824_; lean_object* v___x_5825_; 
v___x_5824_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_5824_, 0, v_e_5818_);
lean_ctor_set(v___x_5824_, 1, v___x_5816_);
lean_ctor_set_uint8(v___x_5824_, sizeof(void*)*2, v___x_5817_);
v___x_5825_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5825_, 0, v___x_5824_);
return v___x_5825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0___boxed(lean_object* v___x_5826_, lean_object* v___x_5827_, lean_object* v_e_5828_, lean_object* v___y_5829_, lean_object* v___y_5830_, lean_object* v___y_5831_, lean_object* v___y_5832_, lean_object* v___y_5833_){
_start:
{
uint8_t v___x_1399__boxed_5834_; lean_object* v_res_5835_; 
v___x_1399__boxed_5834_ = lean_unbox(v___x_5827_);
v_res_5835_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0(v___x_5826_, v___x_1399__boxed_5834_, v_e_5828_, v___y_5829_, v___y_5830_, v___y_5831_, v___y_5832_);
lean_dec(v___y_5832_);
lean_dec_ref(v___y_5831_);
lean_dec(v___y_5830_);
lean_dec_ref(v___y_5829_);
return v_res_5835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2(lean_object* v___f_5836_, lean_object* v___x_5837_, uint8_t v___x_5838_, lean_object* v___y_5839_, lean_object* v___y_5840_, lean_object* v___y_5841_, lean_object* v___y_5842_, lean_object* v___y_5843_, lean_object* v___y_5844_, lean_object* v___y_5845_, lean_object* v___y_5846_){
_start:
{
lean_object* v___x_5848_; 
v___x_5848_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5836_, v___y_5839_, v___y_5840_, v___y_5841_, v___y_5842_, v___y_5843_, v___y_5844_, v___y_5845_, v___y_5846_);
if (lean_obj_tag(v___x_5848_) == 0)
{
lean_object* v___x_5850_; uint8_t v_isShared_5851_; uint8_t v_isSharedCheck_5892_; 
v_isSharedCheck_5892_ = !lean_is_exclusive(v___x_5848_);
if (v_isSharedCheck_5892_ == 0)
{
lean_object* v_unused_5893_; 
v_unused_5893_ = lean_ctor_get(v___x_5848_, 0);
lean_dec(v_unused_5893_);
v___x_5850_ = v___x_5848_;
v_isShared_5851_ = v_isSharedCheck_5892_;
goto v_resetjp_5849_;
}
else
{
lean_dec(v___x_5848_);
v___x_5850_ = lean_box(0);
v_isShared_5851_ = v_isSharedCheck_5892_;
goto v_resetjp_5849_;
}
v_resetjp_5849_:
{
lean_object* v___x_5852_; uint8_t v___x_5853_; lean_object* v___x_5854_; 
v___x_5852_ = lean_box(0);
v___x_5853_ = 0;
v___x_5854_ = l_Lean_Elab_Tactic_elabTerm(v___x_5837_, v___x_5852_, v___x_5853_, v___y_5839_, v___y_5840_, v___y_5841_, v___y_5842_, v___y_5843_, v___y_5844_, v___y_5845_, v___y_5846_);
if (lean_obj_tag(v___x_5854_) == 0)
{
lean_object* v_a_5855_; lean_object* v___x_5856_; 
v_a_5855_ = lean_ctor_get(v___x_5854_, 0);
lean_inc(v_a_5855_);
lean_dec_ref_known(v___x_5854_, 1);
v___x_5856_ = lp_mathlib_Qq_getLevelQ_x27(v_a_5855_, v___y_5843_, v___y_5844_, v___y_5845_, v___y_5846_);
if (lean_obj_tag(v___x_5856_) == 0)
{
lean_object* v_a_5857_; lean_object* v___x_5858_; 
v_a_5857_ = lean_ctor_get(v___x_5856_, 0);
lean_inc(v_a_5857_);
lean_dec_ref_known(v___x_5856_, 1);
v___x_5858_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5840_, v___y_5843_, v___y_5844_, v___y_5845_, v___y_5846_);
if (lean_obj_tag(v___x_5858_) == 0)
{
lean_object* v_a_5859_; lean_object* v___x_5860_; lean_object* v___f_5861_; uint8_t v___x_5862_; lean_object* v___x_5864_; 
v_a_5859_ = lean_ctor_get(v___x_5858_, 0);
lean_inc(v_a_5859_);
lean_dec_ref_known(v___x_5858_, 1);
v___x_5860_ = lean_box(v___x_5838_);
v___f_5861_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__0___boxed), 8, 2);
lean_closure_set(v___f_5861_, 0, v___x_5852_);
lean_closure_set(v___f_5861_, 1, v___x_5860_);
v___x_5862_ = 1;
if (v_isShared_5851_ == 0)
{
lean_ctor_set_tag(v___x_5850_, 1);
lean_ctor_set(v___x_5850_, 0, v_a_5857_);
v___x_5864_ = v___x_5850_;
goto v_reusejp_5863_;
}
else
{
lean_object* v_reuseFailAlloc_5867_; 
v_reuseFailAlloc_5867_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5867_, 0, v_a_5857_);
v___x_5864_ = v_reuseFailAlloc_5867_;
goto v_reusejp_5863_;
}
v_reusejp_5863_:
{
lean_object* v___x_5865_; lean_object* v___x_5866_; 
v___x_5865_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra_proveEq___boxed), 9, 2);
lean_closure_set(v___x_5865_, 0, v___x_5864_);
lean_closure_set(v___x_5865_, 1, v_a_5859_);
v___x_5866_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_5862_, v___x_5865_, v___f_5861_, v___y_5843_, v___y_5844_, v___y_5845_, v___y_5846_);
return v___x_5866_;
}
}
else
{
lean_object* v_a_5868_; lean_object* v___x_5870_; uint8_t v_isShared_5871_; uint8_t v_isSharedCheck_5875_; 
lean_dec(v_a_5857_);
lean_del_object(v___x_5850_);
v_a_5868_ = lean_ctor_get(v___x_5858_, 0);
v_isSharedCheck_5875_ = !lean_is_exclusive(v___x_5858_);
if (v_isSharedCheck_5875_ == 0)
{
v___x_5870_ = v___x_5858_;
v_isShared_5871_ = v_isSharedCheck_5875_;
goto v_resetjp_5869_;
}
else
{
lean_inc(v_a_5868_);
lean_dec(v___x_5858_);
v___x_5870_ = lean_box(0);
v_isShared_5871_ = v_isSharedCheck_5875_;
goto v_resetjp_5869_;
}
v_resetjp_5869_:
{
lean_object* v___x_5873_; 
if (v_isShared_5871_ == 0)
{
v___x_5873_ = v___x_5870_;
goto v_reusejp_5872_;
}
else
{
lean_object* v_reuseFailAlloc_5874_; 
v_reuseFailAlloc_5874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5874_, 0, v_a_5868_);
v___x_5873_ = v_reuseFailAlloc_5874_;
goto v_reusejp_5872_;
}
v_reusejp_5872_:
{
return v___x_5873_;
}
}
}
}
else
{
lean_object* v_a_5876_; lean_object* v___x_5878_; uint8_t v_isShared_5879_; uint8_t v_isSharedCheck_5883_; 
lean_del_object(v___x_5850_);
v_a_5876_ = lean_ctor_get(v___x_5856_, 0);
v_isSharedCheck_5883_ = !lean_is_exclusive(v___x_5856_);
if (v_isSharedCheck_5883_ == 0)
{
v___x_5878_ = v___x_5856_;
v_isShared_5879_ = v_isSharedCheck_5883_;
goto v_resetjp_5877_;
}
else
{
lean_inc(v_a_5876_);
lean_dec(v___x_5856_);
v___x_5878_ = lean_box(0);
v_isShared_5879_ = v_isSharedCheck_5883_;
goto v_resetjp_5877_;
}
v_resetjp_5877_:
{
lean_object* v___x_5881_; 
if (v_isShared_5879_ == 0)
{
v___x_5881_ = v___x_5878_;
goto v_reusejp_5880_;
}
else
{
lean_object* v_reuseFailAlloc_5882_; 
v_reuseFailAlloc_5882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5882_, 0, v_a_5876_);
v___x_5881_ = v_reuseFailAlloc_5882_;
goto v_reusejp_5880_;
}
v_reusejp_5880_:
{
return v___x_5881_;
}
}
}
}
else
{
lean_object* v_a_5884_; lean_object* v___x_5886_; uint8_t v_isShared_5887_; uint8_t v_isSharedCheck_5891_; 
lean_del_object(v___x_5850_);
v_a_5884_ = lean_ctor_get(v___x_5854_, 0);
v_isSharedCheck_5891_ = !lean_is_exclusive(v___x_5854_);
if (v_isSharedCheck_5891_ == 0)
{
v___x_5886_ = v___x_5854_;
v_isShared_5887_ = v_isSharedCheck_5891_;
goto v_resetjp_5885_;
}
else
{
lean_inc(v_a_5884_);
lean_dec(v___x_5854_);
v___x_5886_ = lean_box(0);
v_isShared_5887_ = v_isSharedCheck_5891_;
goto v_resetjp_5885_;
}
v_resetjp_5885_:
{
lean_object* v___x_5889_; 
if (v_isShared_5887_ == 0)
{
v___x_5889_ = v___x_5886_;
goto v_reusejp_5888_;
}
else
{
lean_object* v_reuseFailAlloc_5890_; 
v_reuseFailAlloc_5890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5890_, 0, v_a_5884_);
v___x_5889_ = v_reuseFailAlloc_5890_;
goto v_reusejp_5888_;
}
v_reusejp_5888_:
{
return v___x_5889_;
}
}
}
}
}
else
{
lean_dec(v___x_5837_);
return v___x_5848_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2___boxed(lean_object* v___f_5894_, lean_object* v___x_5895_, lean_object* v___x_5896_, lean_object* v___y_5897_, lean_object* v___y_5898_, lean_object* v___y_5899_, lean_object* v___y_5900_, lean_object* v___y_5901_, lean_object* v___y_5902_, lean_object* v___y_5903_, lean_object* v___y_5904_, lean_object* v___y_5905_){
_start:
{
uint8_t v___x_1425__boxed_5906_; lean_object* v_res_5907_; 
v___x_1425__boxed_5906_ = lean_unbox(v___x_5896_);
v_res_5907_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2(v___f_5894_, v___x_5895_, v___x_1425__boxed_5906_, v___y_5897_, v___y_5898_, v___y_5899_, v___y_5900_, v___y_5901_, v___y_5902_, v___y_5903_, v___y_5904_);
lean_dec(v___y_5904_);
lean_dec_ref(v___y_5903_);
lean_dec(v___y_5902_);
lean_dec_ref(v___y_5901_);
lean_dec(v___y_5900_);
lean_dec_ref(v___y_5899_);
lean_dec(v___y_5898_);
lean_dec_ref(v___y_5897_);
return v_res_5907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1(lean_object* v_x_5910_, lean_object* v_a_5911_, lean_object* v_a_5912_, lean_object* v_a_5913_, lean_object* v_a_5914_, lean_object* v_a_5915_, lean_object* v_a_5916_, lean_object* v_a_5917_, lean_object* v_a_5918_){
_start:
{
lean_object* v___x_5920_; uint8_t v___x_5921_; 
v___x_5920_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra_algebraWith___closed__1));
lean_inc(v_x_5910_);
v___x_5921_ = l_Lean_Syntax_isOfKind(v_x_5910_, v___x_5920_);
if (v___x_5921_ == 0)
{
lean_object* v___x_5922_; 
lean_dec(v_x_5910_);
v___x_5922_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebra__1_spec__0___redArg();
return v___x_5922_;
}
else
{
lean_object* v___f_5923_; lean_object* v___x_5924_; lean_object* v___x_5925_; lean_object* v___x_5926_; lean_object* v___f_5927_; lean_object* v___x_5928_; 
v___f_5923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___closed__0));
v___x_5924_ = lean_unsigned_to_nat(2u);
v___x_5925_ = l_Lean_Syntax_getArg(v_x_5910_, v___x_5924_);
lean_dec(v_x_5910_);
v___x_5926_ = lean_box(v___x_5921_);
v___f_5927_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_5927_, 0, v___f_5923_);
lean_closure_set(v___f_5927_, 1, v___x_5925_);
lean_closure_set(v___f_5927_, 2, v___x_5926_);
v___x_5928_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5927_, v_a_5911_, v_a_5912_, v_a_5913_, v_a_5914_, v_a_5915_, v_a_5916_, v_a_5917_, v_a_5918_);
return v___x_5928_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1___boxed(lean_object* v_x_5929_, lean_object* v_a_5930_, lean_object* v_a_5931_, lean_object* v_a_5932_, lean_object* v_a_5933_, lean_object* v_a_5934_, lean_object* v_a_5935_, lean_object* v_a_5936_, lean_object* v_a_5937_, lean_object* v_a_5938_){
_start:
{
lean_object* v_res_5939_; 
v_res_5939_ = lp_mathlib_Mathlib_Tactic_Algebra___aux__Mathlib__Tactic__Algebra__Basic______elabRules__Mathlib__Tactic__Algebra__algebraWith__1(v_x_5929_, v_a_5930_, v_a_5931_, v_a_5932_, v_a_5933_, v_a_5934_, v_a_5935_, v_a_5936_, v_a_5937_);
lean_dec(v_a_5937_);
lean_dec_ref(v_a_5936_);
lean_dec(v_a_5935_);
lean_dec_ref(v_a_5934_);
lean_dec(v_a_5933_);
lean_dec_ref(v_a_5932_);
lean_dec(v_a_5931_);
lean_dec_ref(v_a_5930_);
return v_res_5939_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_RingNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_NormCast(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_NormCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_NormCast(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Algebra_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring_RingNF(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Algebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_NormCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Algebra_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring_RingNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Algebra_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
