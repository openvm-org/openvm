// Lean compiler output
// Module: Mathlib.Tactic.Ring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Ring.Common public meta import Mathlib.Algebra.Order.Ring.Unbundled.Rat public meta import Mathlib.Tactic.Ring.Common
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_Cache_nat;
extern lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_s_u2115;
lean_object* l_Lean_Level_ofNat(lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* l_Rat_neg(lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_eq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Level_dec(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rawCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 205, 87, 23, 59, 10, 241, 25)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__3_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__6_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__7_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__10_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__12_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__13_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(72, 208, 251, 102, 233, 243, 211, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NNRat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(169, 80, 136, 140, 138, 237, 112, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Rat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(218, 237, 153, 176, 238, 25, 53, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__1_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 149, 205, 214, 52, 248, 155, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__8_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__10_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__11_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__15_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__16_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__18_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__19_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "instMulZeroClassOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__21_value),LEAN_SCALAR_PTR_LITERAL(31, 133, 13, 57, 152, 228, 72, 248)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_pos"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__26_value),LEAN_SCALAR_PTR_LITERAL(103, 165, 142, 233, 52, 15, 52, 201)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "cast_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__28_value),LEAN_SCALAR_PTR_LITERAL(178, 148, 246, 180, 103, 61, 255, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__30_value),LEAN_SCALAR_PTR_LITERAL(183, 123, 209, 218, 212, 36, 212, 19)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "IsNNRat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "den_nz"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__34_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__35_value),LEAN_SCALAR_PTR_LITERAL(23, 12, 45, 45, 118, 187, 101, 38)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "cast_nnrat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__37_value),LEAN_SCALAR_PTR_LITERAL(198, 140, 195, 100, 230, 153, 222, 217)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsRat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__39_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__35_value),LEAN_SCALAR_PTR_LITERAL(55, 69, 72, 86, 50, 41, 73, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_rat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__41_value),LEAN_SCALAR_PTR_LITERAL(55, 7, 237, 185, 66, 54, 144, 42)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 237, 167, 212, 100, 179, 19, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__6_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "natCast_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(202, 251, 162, 143, 20, 106, 30, 106)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 223, 138, 62, 113, 83, 252, 93)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__2_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__7_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__9_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__10_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__14_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__17_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__18_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__20_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__21_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(86, 172, 133, 187, 121, 84, 206, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(135, 111, 68, 46, 73, 204, 94, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(151, 227, 125, 93, 132, 85, 194, 49)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "AddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(226, 82, 90, 134, 221, 253, 108, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toAddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(99, 161, 243, 168, 232, 89, 236, 229)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(42, 135, 58, 37, 72, 75, 21, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 4, 252, 84, 28, 16, 24, 6)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toIntCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__1_value),LEAN_SCALAR_PTR_LITERAL(197, 218, 149, 200, 111, 143, 71, 100)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(78, 130, 181, 61, 179, 129, 164, 15)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__4_value),LEAN_SCALAR_PTR_LITERAL(184, 164, 171, 191, 166, 224, 196, 206)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "intCast_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 181, 236, 216, 32, 218, 231, 203)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "nonexhaustive match"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "intCast_negOfNat_Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__3_value),LEAN_SCALAR_PTR_LITERAL(15, 153, 150, 9, 156, 129, 152, 232)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(186, 16, 157, 198, 21, 20, 76, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "intCast_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__7_value),LEAN_SCALAR_PTR_LITERAL(172, 72, 63, 43, 203, 93, 89, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "intCast_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(121, 80, 57, 248, 121, 90, 171, 43)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__7_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__1_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__12_value),LEAN_SCALAR_PTR_LITERAL(220, 245, 103, 140, 169, 50, 116, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__14_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 136, 96, 129, 245, 140, 119, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "InvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(120, 190, 7, 179, 62, 236, 21, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(28, 25, 248, 9, 15, 85, 72, 194)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "DivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toInvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(162, 155, 123, 0, 237, 243, 28, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(181, 224, 200, 199, 184, 130, 54, 26)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toDivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(16, 242, 184, 157, 107, 26, 18, 78)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(60, 63, 43, 77, 240, 6, 89, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toDivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(198, 76, 78, 187, 42, 89, 29, 20)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(164, 129, 71, 97, 30, 189, 214, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "raw_refl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__32_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__33_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__1_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__0_value),LEAN_SCALAR_PTR_LITERAL(163, 125, 133, 66, 231, 251, 113, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__4_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(156, 44, 233, 53, 1, 106, 24, 217)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(124, 136, 108, 160, 134, 153, 101, 8)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toNegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(135, 233, 160, 34, 207, 245, 132, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(107, 179, 145, 12, 37, 42, 18, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toSubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(203, 24, 17, 79, 61, 156, 198, 150)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(94, 234, 159, 237, 9, 124, 201, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "SubtractionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toSubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(100, 8, 183, 201, 110, 57, 85, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(203, 26, 135, 240, 118, 74, 112, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toDivisionAddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(65, 138, 55, 164, 85, 246, 87, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(121, 151, 225, 139, 113, 68, 25, 156)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompute(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__12_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "smul_eq_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(57, 189, 150, 29, 132, 113, 252, 208)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(131, 171, 90, 153, 163, 182, 66, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 94, 206, 224, 220, 134, 83, 24)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(117, 62, 87, 200, 93, 236, 80, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(221, 146, 155, 22, 128, 55, 188, 54)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "of_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 166, 34, 86, 43, 89, 101, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__5_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "ring failed, ring expressions not equal\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "ring failed: not an equality"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "CSLift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(33, 219, 41, 42, 169, 248, 253, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "CSLiftVal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__4_value),LEAN_SCALAR_PTR_LITERAL(14, 185, 245, 5, 245, 226, 54, 103)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "of_lift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__6_value),LEAN_SCALAR_PTR_LITERAL(41, 214, 127, 21, 146, 20, 227, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not a type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Mathlib.Tactic.Ring.Basic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Mathlib.Tactic.Ring.proveEq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ring1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(221, 141, 62, 226, 100, 80, 9, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Ring_ring1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticRing1!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__24_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__25_value),LEAN_SCALAR_PTR_LITERAL(167, 213, 149, 120, 143, 153, 69, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 132, 48, 112, 76, 186, 197, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ring1!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_nat_to_int(v_a_1_);
v___x_3_ = l_Rat_ofInt(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(lean_object* v_u_29_, lean_object* v_00_u03b1_30_, lean_object* v_s_u03b1_31_, lean_object* v_n_32_){
_start:
{
lean_object* v___x_33_; lean_object* v_lit_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
lean_inc(v_n_32_);
v___x_33_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_33_, 0, v_n_32_);
v_lit_34_ = l_Lean_Expr_lit___override(v___x_33_);
v___x_35_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2));
v___x_36_ = lean_box(0);
v___x_37_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_37_, 0, v_u_29_);
lean_ctor_set(v___x_37_, 1, v___x_36_);
lean_inc_ref_n(v___x_37_, 4);
v___x_38_ = l_Lean_Expr_const___override(v___x_35_, v___x_37_);
lean_inc_ref_n(v_00_u03b1_30_, 4);
v___x_39_ = l_Lean_Expr_app___override(v___x_38_, v_00_u03b1_30_);
v___x_40_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5));
v___x_41_ = l_Lean_Expr_const___override(v___x_40_, v___x_37_);
v___x_42_ = l_Lean_Expr_app___override(v___x_41_, v_00_u03b1_30_);
v___x_43_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8));
v___x_44_ = l_Lean_Expr_const___override(v___x_43_, v___x_37_);
v___x_45_ = l_Lean_Expr_app___override(v___x_44_, v_00_u03b1_30_);
v___x_46_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11));
v___x_47_ = l_Lean_Expr_const___override(v___x_46_, v___x_37_);
v___x_48_ = l_Lean_Expr_app___override(v___x_47_, v_00_u03b1_30_);
v___x_49_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_50_ = l_Lean_Expr_const___override(v___x_49_, v___x_37_);
v___x_51_ = l_Lean_Expr_app___override(v___x_50_, v_00_u03b1_30_);
v___x_52_ = l_Lean_Expr_app___override(v___x_51_, v_s_u03b1_31_);
v___x_53_ = l_Lean_Expr_app___override(v___x_48_, v___x_52_);
v___x_54_ = l_Lean_Expr_app___override(v___x_45_, v___x_53_);
v___x_55_ = l_Lean_Expr_app___override(v___x_42_, v___x_54_);
v___x_56_ = l_Lean_Expr_app___override(v___x_39_, v___x_55_);
v___x_57_ = l_Lean_Expr_app___override(v___x_56_, v_lit_34_);
v___x_58_ = lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0(v_n_32_);
v___x_59_ = lean_box(0);
v___x_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_58_);
lean_ctor_set(v___x_60_, 1, v___x_59_);
lean_inc_ref(v___x_57_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_57_);
lean_ctor_set(v___x_61_, 1, v___x_60_);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_57_);
lean_ctor_set(v___x_62_, 1, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0_spec__0(lean_object* v_a_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lean_nat_to_int(v_a_63_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = lean_box(0);
v___x_74_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3));
v___x_75_ = l_Lean_Expr_const___override(v___x_74_, v___x_73_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(lean_object* v_u_76_, lean_object* v_00_u03b1_77_, lean_object* v_x_78_, lean_object* v_n_79_){
_start:
{
lean_object* v_lit_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
lean_inc(v_n_79_);
v_lit_80_ = l_Lean_mkRawNatLit(v_n_79_);
v___x_81_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1));
v___x_82_ = lean_box(0);
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v_u_76_);
lean_ctor_set(v___x_83_, 1, v___x_82_);
v___x_84_ = l_Lean_Expr_const___override(v___x_81_, v___x_83_);
v___x_85_ = l_Lean_Expr_app___override(v___x_84_, v_00_u03b1_77_);
v___x_86_ = l_Lean_Expr_app___override(v___x_85_, v_x_78_);
v___x_87_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4);
v___x_88_ = l_Lean_Expr_app___override(v___x_87_, v_lit_80_);
v___x_89_ = l_Lean_Expr_app___override(v___x_86_, v___x_88_);
v___x_90_ = lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0(v_n_79_);
v___x_91_ = l_Rat_neg(v___x_90_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_91_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
lean_inc_ref(v___x_89_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_89_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_89_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat(lean_object* v_u_96_, lean_object* v_00_u03b1_97_, lean_object* v_s_u03b1_98_, lean_object* v_x_99_, lean_object* v_n_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(v_u_96_, v_00_u03b1_97_, v_x_99_, v_n_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___boxed(lean_object* v_u_102_, lean_object* v_00_u03b1_103_, lean_object* v_s_u03b1_104_, lean_object* v_x_105_, lean_object* v_n_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat(v_u_102_, v_00_u03b1_103_, v_s_u03b1_104_, v_x_105_, v_n_106_);
lean_dec_ref(v_s_u03b1_104_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(lean_object* v_u_112_, lean_object* v_00_u03b1_113_, lean_object* v_x_114_, lean_object* v_q_115_, lean_object* v_n_116_, lean_object* v_d_117_, lean_object* v_h_118_){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg___closed__1));
v___x_120_ = lean_box(0);
v___x_121_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_121_, 0, v_u_112_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = l_Lean_Expr_const___override(v___x_119_, v___x_121_);
v___x_123_ = l_Lean_Expr_app___override(v___x_122_, v_00_u03b1_113_);
v___x_124_ = l_Lean_Expr_app___override(v___x_123_, v_x_114_);
v___x_125_ = l_Lean_Expr_app___override(v___x_124_, v_n_116_);
v___x_126_ = l_Lean_Expr_app___override(v___x_125_, v_d_117_);
v___x_127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_127_, 0, v_h_118_);
v___x_128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_128_, 0, v_q_115_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
lean_inc_ref(v___x_126_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_126_);
lean_ctor_set(v___x_129_, 1, v___x_128_);
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_126_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat(lean_object* v_u_131_, lean_object* v_00_u03b1_132_, lean_object* v_s_u03b1_133_, lean_object* v_x_134_, lean_object* v_q_135_, lean_object* v_n_136_, lean_object* v_d_137_, lean_object* v_h_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(v_u_131_, v_00_u03b1_132_, v_x_134_, v_q_135_, v_n_136_, v_d_137_, v_h_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___boxed(lean_object* v_u_140_, lean_object* v_00_u03b1_141_, lean_object* v_s_u03b1_142_, lean_object* v_x_143_, lean_object* v_q_144_, lean_object* v_n_145_, lean_object* v_d_146_, lean_object* v_h_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat(v_u_140_, v_00_u03b1_141_, v_s_u03b1_142_, v_x_143_, v_q_144_, v_n_145_, v_d_146_, v_h_147_);
lean_dec_ref(v_s_u03b1_142_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(lean_object* v_u_153_, lean_object* v_00_u03b1_154_, lean_object* v_x_155_, lean_object* v_q_156_, lean_object* v_n_157_, lean_object* v_d_158_, lean_object* v_h_159_){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg___closed__1));
v___x_161_ = lean_box(0);
v___x_162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_162_, 0, v_u_153_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = l_Lean_Expr_const___override(v___x_160_, v___x_162_);
v___x_164_ = l_Lean_Expr_app___override(v___x_163_, v_00_u03b1_154_);
v___x_165_ = l_Lean_Expr_app___override(v___x_164_, v_x_155_);
v___x_166_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4);
v___x_167_ = l_Lean_Expr_app___override(v___x_166_, v_n_157_);
v___x_168_ = l_Lean_Expr_app___override(v___x_165_, v___x_167_);
v___x_169_ = l_Lean_Expr_app___override(v___x_168_, v_d_158_);
v___x_170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_170_, 0, v_h_159_);
v___x_171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_171_, 0, v_q_156_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
lean_inc_ref(v___x_169_);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_169_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_169_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat(lean_object* v_u_174_, lean_object* v_00_u03b1_175_, lean_object* v_s_u03b1_176_, lean_object* v_x_177_, lean_object* v_q_178_, lean_object* v_n_179_, lean_object* v_d_180_, lean_object* v_h_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(v_u_174_, v_00_u03b1_175_, v_x_177_, v_q_178_, v_n_179_, v_d_180_, v_h_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___boxed(lean_object* v_u_183_, lean_object* v_00_u03b1_184_, lean_object* v_s_u03b1_185_, lean_object* v_x_186_, lean_object* v_q_187_, lean_object* v_n_188_, lean_object* v_d_189_, lean_object* v_h_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat(v_u_183_, v_00_u03b1_184_, v_s_u03b1_185_, v_x_186_, v_q_187_, v_n_188_, v_d_189_, v_h_190_);
lean_dec_ref(v_s_u03b1_185_);
return v_res_191_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__13));
v___x_216_ = l_Lean_Expr_lit___override(v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_evalCast(lean_object* v_u_280_, lean_object* v_00_u03b1_281_, lean_object* v_s_u03b1_282_, lean_object* v_e_283_, lean_object* v_x_284_){
_start:
{
switch(lean_obj_tag(v_x_284_))
{
case 1:
{
lean_object* v_lit_285_; lean_object* v_proof_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_402_; 
v_lit_285_ = lean_ctor_get(v_x_284_, 1);
v_proof_286_ = lean_ctor_get(v_x_284_, 2);
v_isSharedCheck_402_ = !lean_is_exclusive(v_x_284_);
if (v_isSharedCheck_402_ == 0)
{
lean_object* v_unused_403_; 
v_unused_403_ = lean_ctor_get(v_x_284_, 0);
lean_dec(v_unused_403_);
v___x_288_ = v_x_284_;
v_isShared_289_ = v_isSharedCheck_402_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_proof_286_);
lean_inc(v_lit_285_);
lean_dec(v_x_284_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_402_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
if (lean_obj_tag(v_lit_285_) == 9)
{
lean_object* v_a_359_; 
v_a_359_ = lean_ctor_get(v_lit_285_, 0);
lean_inc_ref(v_a_359_);
if (lean_obj_tag(v_a_359_) == 0)
{
lean_object* v_val_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_401_; 
v_val_360_ = lean_ctor_get(v_a_359_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v_a_359_);
if (v_isSharedCheck_401_ == 0)
{
v___x_362_ = v_a_359_;
v_isShared_363_ = v_isSharedCheck_401_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_val_360_);
lean_dec(v_a_359_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_401_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_364_; uint8_t v___x_365_; 
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lean_nat_dec_eq(v_val_360_, v___x_364_);
lean_dec(v_val_360_);
if (v___x_365_ == 0)
{
lean_del_object(v___x_362_);
goto v___jp_290_;
}
else
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_399_; 
lean_dec_ref_known(v_lit_285_, 1);
lean_del_object(v___x_288_);
v___x_366_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_367_ = lean_box(0);
v___x_368_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_368_, 0, v_u_280_);
lean_ctor_set(v___x_368_, 1, v___x_367_);
lean_inc_ref_n(v___x_368_, 5);
v___x_369_ = l_Lean_Expr_const___override(v___x_366_, v___x_368_);
lean_inc_ref_n(v_00_u03b1_281_, 5);
v___x_370_ = l_Lean_Expr_app___override(v___x_369_, v_00_u03b1_281_);
v___x_371_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_372_ = l_Lean_Expr_app___override(v___x_370_, v___x_371_);
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_374_ = l_Lean_Expr_const___override(v___x_373_, v___x_368_);
v___x_375_ = l_Lean_Expr_app___override(v___x_374_, v_00_u03b1_281_);
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_377_ = l_Lean_Expr_const___override(v___x_376_, v___x_368_);
v___x_378_ = l_Lean_Expr_app___override(v___x_377_, v_00_u03b1_281_);
v___x_379_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_380_ = l_Lean_Expr_const___override(v___x_379_, v___x_368_);
v___x_381_ = l_Lean_Expr_app___override(v___x_380_, v_00_u03b1_281_);
v___x_382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_383_ = l_Lean_Expr_const___override(v___x_382_, v___x_368_);
v___x_384_ = l_Lean_Expr_app___override(v___x_383_, v_00_u03b1_281_);
lean_inc_ref(v_s_u03b1_282_);
v___x_385_ = l_Lean_Expr_app___override(v___x_384_, v_s_u03b1_282_);
v___x_386_ = l_Lean_Expr_app___override(v___x_381_, v___x_385_);
v___x_387_ = l_Lean_Expr_app___override(v___x_378_, v___x_386_);
v___x_388_ = l_Lean_Expr_app___override(v___x_375_, v___x_387_);
v___x_389_ = l_Lean_Expr_app___override(v___x_372_, v___x_388_);
v___x_390_ = lean_box(0);
v___x_391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__29));
v___x_392_ = l_Lean_Expr_const___override(v___x_391_, v___x_368_);
v___x_393_ = l_Lean_Expr_app___override(v___x_392_, v_00_u03b1_281_);
v___x_394_ = l_Lean_Expr_app___override(v___x_393_, v_s_u03b1_282_);
v___x_395_ = l_Lean_Expr_app___override(v___x_394_, v_e_283_);
v___x_396_ = l_Lean_Expr_app___override(v___x_395_, v_proof_286_);
v___x_397_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_397_, 0, v___x_389_);
lean_ctor_set(v___x_397_, 1, v___x_390_);
lean_ctor_set(v___x_397_, 2, v___x_396_);
if (v_isShared_363_ == 0)
{
lean_ctor_set_tag(v___x_362_, 1);
lean_ctor_set(v___x_362_, 0, v___x_397_);
v___x_399_ = v___x_362_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_397_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
}
else
{
lean_dec_ref(v_a_359_);
goto v___jp_290_;
}
}
else
{
goto v___jp_290_;
}
v___jp_290_:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v_fst_293_; lean_object* v_snd_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_358_; 
v___x_291_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_285_);
lean_inc_ref(v_s_u03b1_282_);
lean_inc_ref(v_00_u03b1_281_);
lean_inc(v_u_280_);
v___x_292_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat(v_u_280_, v_00_u03b1_281_, v_s_u03b1_282_, v___x_291_);
v_fst_293_ = lean_ctor_get(v___x_292_, 0);
v_snd_294_ = lean_ctor_get(v___x_292_, 1);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_358_ == 0)
{
v___x_296_ = v___x_292_;
v_isShared_297_ = v_isSharedCheck_358_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_snd_294_);
lean_inc(v_fst_293_);
lean_dec(v___x_292_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_358_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_301_; 
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_299_ = lean_box(0);
lean_inc(v_u_280_);
if (v_isShared_297_ == 0)
{
lean_ctor_set_tag(v___x_296_, 1);
lean_ctor_set(v___x_296_, 1, v___x_299_);
lean_ctor_set(v___x_296_, 0, v_u_280_);
v___x_301_ = v___x_296_;
goto v_reusejp_300_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_u_280_);
lean_ctor_set(v_reuseFailAlloc_357_, 1, v___x_299_);
v___x_301_ = v_reuseFailAlloc_357_;
goto v_reusejp_300_;
}
v_reusejp_300_:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_354_; 
lean_inc_ref_n(v___x_301_, 9);
lean_inc_n(v_u_280_, 2);
v___x_302_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_302_, 0, v_u_280_);
lean_ctor_set(v___x_302_, 1, v___x_301_);
v___x_303_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_303_, 0, v_u_280_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
v___x_304_ = l_Lean_Expr_const___override(v___x_298_, v___x_303_);
lean_inc_ref_n(v_00_u03b1_281_, 12);
v___x_305_ = l_Lean_Expr_app___override(v___x_304_, v_00_u03b1_281_);
v___x_306_ = l_Lean_Expr_app___override(v___x_305_, v_00_u03b1_281_);
v___x_307_ = l_Lean_Expr_app___override(v___x_306_, v_00_u03b1_281_);
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_309_ = l_Lean_Expr_const___override(v___x_308_, v___x_301_);
v___x_310_ = l_Lean_Expr_app___override(v___x_309_, v_00_u03b1_281_);
v___x_311_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_312_ = l_Lean_Expr_const___override(v___x_311_, v___x_301_);
v___x_313_ = l_Lean_Expr_app___override(v___x_312_, v_00_u03b1_281_);
v___x_314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_315_ = l_Lean_Expr_const___override(v___x_314_, v___x_301_);
v___x_316_ = l_Lean_Expr_app___override(v___x_315_, v_00_u03b1_281_);
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_318_ = l_Lean_Expr_const___override(v___x_317_, v___x_301_);
v___x_319_ = l_Lean_Expr_app___override(v___x_318_, v_00_u03b1_281_);
lean_inc_ref_n(v_s_u03b1_282_, 2);
v___x_320_ = l_Lean_Expr_app___override(v___x_319_, v_s_u03b1_282_);
lean_inc_ref(v___x_320_);
v___x_321_ = l_Lean_Expr_app___override(v___x_316_, v___x_320_);
v___x_322_ = l_Lean_Expr_app___override(v___x_313_, v___x_321_);
v___x_323_ = l_Lean_Expr_app___override(v___x_310_, v___x_322_);
v___x_324_ = l_Lean_Expr_app___override(v___x_307_, v___x_323_);
lean_inc(v_fst_293_);
v___x_325_ = l_Lean_Expr_app___override(v___x_324_, v_fst_293_);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_327_ = l_Lean_Expr_const___override(v___x_326_, v___x_301_);
v___x_328_ = l_Lean_Expr_app___override(v___x_327_, v_00_u03b1_281_);
v___x_329_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_330_ = l_Lean_Expr_app___override(v___x_328_, v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_332_ = l_Lean_Expr_const___override(v___x_331_, v___x_301_);
v___x_333_ = l_Lean_Expr_app___override(v___x_332_, v_00_u03b1_281_);
v___x_334_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_335_ = l_Lean_Expr_const___override(v___x_334_, v___x_301_);
v___x_336_ = l_Lean_Expr_app___override(v___x_335_, v_00_u03b1_281_);
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_338_ = l_Lean_Expr_const___override(v___x_337_, v___x_301_);
v___x_339_ = l_Lean_Expr_app___override(v___x_338_, v_00_u03b1_281_);
v___x_340_ = l_Lean_Expr_app___override(v___x_339_, v___x_320_);
v___x_341_ = l_Lean_Expr_app___override(v___x_336_, v___x_340_);
v___x_342_ = l_Lean_Expr_app___override(v___x_333_, v___x_341_);
v___x_343_ = l_Lean_Expr_app___override(v___x_330_, v___x_342_);
v___x_344_ = l_Lean_Expr_app___override(v___x_325_, v___x_343_);
v___x_345_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_280_, v_00_u03b1_281_, v_s_u03b1_282_, v_fst_293_, v_snd_294_);
v___x_346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__27));
v___x_347_ = l_Lean_Expr_const___override(v___x_346_, v___x_301_);
v___x_348_ = l_Lean_Expr_app___override(v___x_347_, v_00_u03b1_281_);
v___x_349_ = l_Lean_Expr_app___override(v___x_348_, v_s_u03b1_282_);
v___x_350_ = l_Lean_Expr_app___override(v___x_349_, v_e_283_);
v___x_351_ = l_Lean_Expr_app___override(v___x_350_, v_lit_285_);
v___x_352_ = l_Lean_Expr_app___override(v___x_351_, v_proof_286_);
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 0);
lean_ctor_set(v___x_288_, 2, v___x_352_);
lean_ctor_set(v___x_288_, 1, v___x_345_);
lean_ctor_set(v___x_288_, 0, v___x_344_);
v___x_354_ = v___x_288_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_344_);
lean_ctor_set(v_reuseFailAlloc_356_, 1, v___x_345_);
lean_ctor_set(v_reuseFailAlloc_356_, 2, v___x_352_);
v___x_354_ = v_reuseFailAlloc_356_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
lean_object* v___x_355_; 
v___x_355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
return v___x_355_;
}
}
}
}
}
}
case 2:
{
lean_object* v_inst_404_; lean_object* v_lit_405_; lean_object* v_proof_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_472_; 
v_inst_404_ = lean_ctor_get(v_x_284_, 0);
v_lit_405_ = lean_ctor_get(v_x_284_, 1);
v_proof_406_ = lean_ctor_get(v_x_284_, 2);
v_isSharedCheck_472_ = !lean_is_exclusive(v_x_284_);
if (v_isSharedCheck_472_ == 0)
{
v___x_408_ = v_x_284_;
v_isShared_409_ = v_isSharedCheck_472_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_proof_406_);
lean_inc(v_lit_405_);
lean_inc(v_inst_404_);
lean_dec(v_x_284_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_472_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v_fst_433_; lean_object* v_snd_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_411_ = lean_box(0);
lean_inc_n(v_u_280_, 4);
v___x_412_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_412_, 0, v_u_280_);
lean_ctor_set(v___x_412_, 1, v___x_411_);
lean_inc_ref_n(v___x_412_, 9);
v___x_413_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_413_, 0, v_u_280_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
v___x_414_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_414_, 0, v_u_280_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
v___x_415_ = l_Lean_Expr_const___override(v___x_410_, v___x_414_);
lean_inc_ref_n(v_00_u03b1_281_, 13);
v___x_416_ = l_Lean_Expr_app___override(v___x_415_, v_00_u03b1_281_);
v___x_417_ = l_Lean_Expr_app___override(v___x_416_, v_00_u03b1_281_);
v___x_418_ = l_Lean_Expr_app___override(v___x_417_, v_00_u03b1_281_);
v___x_419_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_420_ = l_Lean_Expr_const___override(v___x_419_, v___x_412_);
v___x_421_ = l_Lean_Expr_app___override(v___x_420_, v_00_u03b1_281_);
v___x_422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_423_ = l_Lean_Expr_const___override(v___x_422_, v___x_412_);
v___x_424_ = l_Lean_Expr_app___override(v___x_423_, v_00_u03b1_281_);
v___x_425_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_426_ = l_Lean_Expr_const___override(v___x_425_, v___x_412_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v_00_u03b1_281_);
v___x_428_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_429_ = l_Lean_Expr_const___override(v___x_428_, v___x_412_);
v___x_430_ = l_Lean_Expr_app___override(v___x_429_, v_00_u03b1_281_);
v___x_431_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_405_);
lean_inc_ref(v_inst_404_);
v___x_432_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg(v_u_280_, v_00_u03b1_281_, v_inst_404_, v___x_431_);
v_fst_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc_n(v_fst_433_, 2);
v_snd_434_ = lean_ctor_get(v___x_432_, 1);
lean_inc(v_snd_434_);
lean_dec_ref(v___x_432_);
lean_inc_ref(v_s_u03b1_282_);
v___x_435_ = l_Lean_Expr_app___override(v___x_430_, v_s_u03b1_282_);
lean_inc_ref(v___x_435_);
v___x_436_ = l_Lean_Expr_app___override(v___x_427_, v___x_435_);
v___x_437_ = l_Lean_Expr_app___override(v___x_424_, v___x_436_);
v___x_438_ = l_Lean_Expr_app___override(v___x_421_, v___x_437_);
v___x_439_ = l_Lean_Expr_app___override(v___x_418_, v___x_438_);
v___x_440_ = l_Lean_Expr_app___override(v___x_439_, v_fst_433_);
v___x_441_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_442_ = l_Lean_Expr_const___override(v___x_441_, v___x_412_);
v___x_443_ = l_Lean_Expr_app___override(v___x_442_, v_00_u03b1_281_);
v___x_444_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_445_ = l_Lean_Expr_app___override(v___x_443_, v___x_444_);
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_447_ = l_Lean_Expr_const___override(v___x_446_, v___x_412_);
v___x_448_ = l_Lean_Expr_app___override(v___x_447_, v_00_u03b1_281_);
v___x_449_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_450_ = l_Lean_Expr_const___override(v___x_449_, v___x_412_);
v___x_451_ = l_Lean_Expr_app___override(v___x_450_, v_00_u03b1_281_);
v___x_452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_453_ = l_Lean_Expr_const___override(v___x_452_, v___x_412_);
v___x_454_ = l_Lean_Expr_app___override(v___x_453_, v_00_u03b1_281_);
v___x_455_ = l_Lean_Expr_app___override(v___x_454_, v___x_435_);
v___x_456_ = l_Lean_Expr_app___override(v___x_451_, v___x_455_);
v___x_457_ = l_Lean_Expr_app___override(v___x_448_, v___x_456_);
v___x_458_ = l_Lean_Expr_app___override(v___x_445_, v___x_457_);
v___x_459_ = l_Lean_Expr_app___override(v___x_440_, v___x_458_);
v___x_460_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_280_, v_00_u03b1_281_, v_s_u03b1_282_, v_fst_433_, v_snd_434_);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__31));
v___x_462_ = l_Lean_Expr_const___override(v___x_461_, v___x_412_);
v___x_463_ = l_Lean_Expr_app___override(v___x_462_, v_lit_405_);
v___x_464_ = l_Lean_Expr_app___override(v___x_463_, v_00_u03b1_281_);
v___x_465_ = l_Lean_Expr_app___override(v___x_464_, v_inst_404_);
v___x_466_ = l_Lean_Expr_app___override(v___x_465_, v_e_283_);
v___x_467_ = l_Lean_Expr_app___override(v___x_466_, v_proof_406_);
if (v_isShared_409_ == 0)
{
lean_ctor_set_tag(v___x_408_, 0);
lean_ctor_set(v___x_408_, 2, v___x_467_);
lean_ctor_set(v___x_408_, 1, v___x_460_);
lean_ctor_set(v___x_408_, 0, v___x_459_);
v___x_469_ = v___x_408_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v___x_459_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v___x_460_);
lean_ctor_set(v_reuseFailAlloc_471_, 2, v___x_467_);
v___x_469_ = v_reuseFailAlloc_471_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
lean_object* v___x_470_; 
v___x_470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_470_, 0, v___x_469_);
return v___x_470_;
}
}
}
case 3:
{
lean_object* v_inst_473_; lean_object* v_q_474_; lean_object* v_n_475_; lean_object* v_d_476_; lean_object* v_proof_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v_fst_520_; lean_object* v_snd_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v_inst_473_ = lean_ctor_get(v_x_284_, 0);
lean_inc_ref_n(v_inst_473_, 3);
v_q_474_ = lean_ctor_get(v_x_284_, 1);
lean_inc_ref(v_q_474_);
v_n_475_ = lean_ctor_get(v_x_284_, 2);
lean_inc_ref_n(v_n_475_, 3);
v_d_476_ = lean_ctor_get(v_x_284_, 3);
lean_inc_ref_n(v_d_476_, 3);
v_proof_477_ = lean_ctor_get(v_x_284_, 4);
lean_inc_ref_n(v_proof_477_, 2);
lean_dec_ref_known(v_x_284_, 5);
v___x_478_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_479_ = lean_box(0);
lean_inc_n(v_u_280_, 4);
v___x_480_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_480_, 0, v_u_280_);
lean_ctor_set(v___x_480_, 1, v___x_479_);
lean_inc_ref_n(v___x_480_, 10);
v___x_481_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_481_, 0, v_u_280_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v___x_482_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_482_, 0, v_u_280_);
lean_ctor_set(v___x_482_, 1, v___x_481_);
v___x_483_ = l_Lean_Expr_const___override(v___x_478_, v___x_482_);
lean_inc_ref_n(v_00_u03b1_281_, 14);
v___x_484_ = l_Lean_Expr_app___override(v___x_483_, v_00_u03b1_281_);
v___x_485_ = l_Lean_Expr_app___override(v___x_484_, v_00_u03b1_281_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_00_u03b1_281_);
v___x_487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_488_ = l_Lean_Expr_const___override(v___x_487_, v___x_480_);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v_00_u03b1_281_);
v___x_490_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_491_ = l_Lean_Expr_const___override(v___x_490_, v___x_480_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_00_u03b1_281_);
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_494_ = l_Lean_Expr_const___override(v___x_493_, v___x_480_);
v___x_495_ = l_Lean_Expr_app___override(v___x_494_, v_00_u03b1_281_);
v___x_496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_497_ = l_Lean_Expr_const___override(v___x_496_, v___x_480_);
v___x_498_ = l_Lean_Expr_app___override(v___x_497_, v_00_u03b1_281_);
v___x_499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_500_ = l_Lean_Expr_const___override(v___x_499_, v___x_480_);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v_00_u03b1_281_);
v___x_502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_503_ = l_Lean_Expr_const___override(v___x_502_, v___x_480_);
v___x_504_ = l_Lean_Expr_app___override(v___x_503_, v_00_u03b1_281_);
v___x_505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_506_ = l_Lean_Expr_const___override(v___x_505_, v___x_480_);
v___x_507_ = l_Lean_Expr_app___override(v___x_506_, v_00_u03b1_281_);
v___x_508_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_509_ = l_Lean_Expr_const___override(v___x_508_, v___x_480_);
v___x_510_ = l_Lean_Expr_app___override(v___x_509_, v_00_u03b1_281_);
v___x_511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__36));
v___x_512_ = l_Lean_Expr_const___override(v___x_511_, v___x_480_);
v___x_513_ = l_Lean_Expr_app___override(v___x_512_, v_00_u03b1_281_);
v___x_514_ = l_Lean_Expr_app___override(v___x_513_, v_inst_473_);
lean_inc_ref(v_e_283_);
v___x_515_ = l_Lean_Expr_app___override(v___x_514_, v_e_283_);
v___x_516_ = l_Lean_Expr_app___override(v___x_515_, v_n_475_);
v___x_517_ = l_Lean_Expr_app___override(v___x_516_, v_d_476_);
v___x_518_ = l_Lean_Expr_app___override(v___x_517_, v_proof_477_);
v___x_519_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNNRat___redArg(v_u_280_, v_00_u03b1_281_, v_inst_473_, v_q_474_, v_n_475_, v_d_476_, v___x_518_);
v_fst_520_ = lean_ctor_get(v___x_519_, 0);
lean_inc_n(v_fst_520_, 2);
v_snd_521_ = lean_ctor_get(v___x_519_, 1);
lean_inc(v_snd_521_);
lean_dec_ref(v___x_519_);
lean_inc_ref(v_s_u03b1_282_);
v___x_522_ = l_Lean_Expr_app___override(v___x_498_, v_s_u03b1_282_);
lean_inc_ref(v___x_522_);
v___x_523_ = l_Lean_Expr_app___override(v___x_495_, v___x_522_);
v___x_524_ = l_Lean_Expr_app___override(v___x_492_, v___x_523_);
v___x_525_ = l_Lean_Expr_app___override(v___x_489_, v___x_524_);
v___x_526_ = l_Lean_Expr_app___override(v___x_486_, v___x_525_);
v___x_527_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_528_ = l_Lean_Expr_app___override(v___x_501_, v___x_527_);
v___x_529_ = l_Lean_Expr_app___override(v___x_526_, v_fst_520_);
v___x_530_ = l_Lean_Expr_app___override(v___x_510_, v___x_522_);
v___x_531_ = l_Lean_Expr_app___override(v___x_507_, v___x_530_);
v___x_532_ = l_Lean_Expr_app___override(v___x_504_, v___x_531_);
v___x_533_ = l_Lean_Expr_app___override(v___x_528_, v___x_532_);
v___x_534_ = l_Lean_Expr_app___override(v___x_529_, v___x_533_);
v___x_535_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_280_, v_00_u03b1_281_, v_s_u03b1_282_, v_fst_520_, v_snd_521_);
v___x_536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__38));
v___x_537_ = l_Lean_Expr_const___override(v___x_536_, v___x_480_);
v___x_538_ = l_Lean_Expr_app___override(v___x_537_, v_n_475_);
v___x_539_ = l_Lean_Expr_app___override(v___x_538_, v_d_476_);
v___x_540_ = l_Lean_Expr_app___override(v___x_539_, v_00_u03b1_281_);
v___x_541_ = l_Lean_Expr_app___override(v___x_540_, v_inst_473_);
v___x_542_ = l_Lean_Expr_app___override(v___x_541_, v_e_283_);
v___x_543_ = l_Lean_Expr_app___override(v___x_542_, v_proof_477_);
v___x_544_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_544_, 0, v___x_534_);
lean_ctor_set(v___x_544_, 1, v___x_535_);
lean_ctor_set(v___x_544_, 2, v___x_543_);
v___x_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
return v___x_545_;
}
case 4:
{
lean_object* v_inst_546_; lean_object* v_q_547_; lean_object* v_n_548_; lean_object* v_d_549_; lean_object* v_proof_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v_fst_595_; lean_object* v_snd_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v_inst_546_ = lean_ctor_get(v_x_284_, 0);
lean_inc_ref_n(v_inst_546_, 3);
v_q_547_ = lean_ctor_get(v_x_284_, 1);
lean_inc_ref(v_q_547_);
v_n_548_ = lean_ctor_get(v_x_284_, 2);
lean_inc_ref_n(v_n_548_, 2);
v_d_549_ = lean_ctor_get(v_x_284_, 3);
lean_inc_ref_n(v_d_549_, 3);
v_proof_550_ = lean_ctor_get(v_x_284_, 4);
lean_inc_ref_n(v_proof_550_, 2);
lean_dec_ref_known(v_x_284_, 5);
v___x_551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_552_ = lean_box(0);
lean_inc_n(v_u_280_, 4);
v___x_553_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_553_, 0, v_u_280_);
lean_ctor_set(v___x_553_, 1, v___x_552_);
lean_inc_ref_n(v___x_553_, 10);
v___x_554_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_554_, 0, v_u_280_);
lean_ctor_set(v___x_554_, 1, v___x_553_);
v___x_555_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_555_, 0, v_u_280_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
v___x_556_ = l_Lean_Expr_const___override(v___x_551_, v___x_555_);
lean_inc_ref_n(v_00_u03b1_281_, 14);
v___x_557_ = l_Lean_Expr_app___override(v___x_556_, v_00_u03b1_281_);
v___x_558_ = l_Lean_Expr_app___override(v___x_557_, v_00_u03b1_281_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v_00_u03b1_281_);
v___x_560_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_561_ = l_Lean_Expr_const___override(v___x_560_, v___x_553_);
v___x_562_ = l_Lean_Expr_app___override(v___x_561_, v_00_u03b1_281_);
v___x_563_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_564_ = l_Lean_Expr_const___override(v___x_563_, v___x_553_);
v___x_565_ = l_Lean_Expr_app___override(v___x_564_, v_00_u03b1_281_);
v___x_566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_567_ = l_Lean_Expr_const___override(v___x_566_, v___x_553_);
v___x_568_ = l_Lean_Expr_app___override(v___x_567_, v_00_u03b1_281_);
v___x_569_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_570_ = l_Lean_Expr_const___override(v___x_569_, v___x_553_);
v___x_571_ = l_Lean_Expr_app___override(v___x_570_, v_00_u03b1_281_);
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_573_ = l_Lean_Expr_const___override(v___x_572_, v___x_553_);
v___x_574_ = l_Lean_Expr_app___override(v___x_573_, v_00_u03b1_281_);
v___x_575_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_576_ = l_Lean_Expr_const___override(v___x_575_, v___x_553_);
v___x_577_ = l_Lean_Expr_app___override(v___x_576_, v_00_u03b1_281_);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_579_ = l_Lean_Expr_const___override(v___x_578_, v___x_553_);
v___x_580_ = l_Lean_Expr_app___override(v___x_579_, v_00_u03b1_281_);
v___x_581_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_582_ = l_Lean_Expr_const___override(v___x_581_, v___x_553_);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v_00_u03b1_281_);
v___x_584_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__40));
v___x_585_ = l_Lean_Expr_const___override(v___x_584_, v___x_553_);
v___x_586_ = l_Lean_Expr_app___override(v___x_585_, v_00_u03b1_281_);
v___x_587_ = l_Lean_Expr_app___override(v___x_586_, v_inst_546_);
lean_inc_ref(v_e_283_);
v___x_588_ = l_Lean_Expr_app___override(v___x_587_, v_e_283_);
v___x_589_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4);
v___x_590_ = l_Lean_Expr_app___override(v___x_589_, v_n_548_);
lean_inc_ref(v___x_590_);
v___x_591_ = l_Lean_Expr_app___override(v___x_588_, v___x_590_);
v___x_592_ = l_Lean_Expr_app___override(v___x_591_, v_d_549_);
v___x_593_ = l_Lean_Expr_app___override(v___x_592_, v_proof_550_);
v___x_594_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNNRat___redArg(v_u_280_, v_00_u03b1_281_, v_inst_546_, v_q_547_, v_n_548_, v_d_549_, v___x_593_);
v_fst_595_ = lean_ctor_get(v___x_594_, 0);
lean_inc_n(v_fst_595_, 2);
v_snd_596_ = lean_ctor_get(v___x_594_, 1);
lean_inc(v_snd_596_);
lean_dec_ref(v___x_594_);
lean_inc_ref(v_s_u03b1_282_);
v___x_597_ = l_Lean_Expr_app___override(v___x_571_, v_s_u03b1_282_);
lean_inc_ref(v___x_597_);
v___x_598_ = l_Lean_Expr_app___override(v___x_568_, v___x_597_);
v___x_599_ = l_Lean_Expr_app___override(v___x_565_, v___x_598_);
v___x_600_ = l_Lean_Expr_app___override(v___x_562_, v___x_599_);
v___x_601_ = l_Lean_Expr_app___override(v___x_559_, v___x_600_);
v___x_602_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_603_ = l_Lean_Expr_app___override(v___x_574_, v___x_602_);
v___x_604_ = l_Lean_Expr_app___override(v___x_601_, v_fst_595_);
v___x_605_ = l_Lean_Expr_app___override(v___x_583_, v___x_597_);
v___x_606_ = l_Lean_Expr_app___override(v___x_580_, v___x_605_);
v___x_607_ = l_Lean_Expr_app___override(v___x_577_, v___x_606_);
v___x_608_ = l_Lean_Expr_app___override(v___x_603_, v___x_607_);
v___x_609_ = l_Lean_Expr_app___override(v___x_604_, v___x_608_);
v___x_610_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExProd_toSum___redArg(v_u_280_, v_00_u03b1_281_, v_s_u03b1_282_, v_fst_595_, v_snd_596_);
v___x_611_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__42));
v___x_612_ = l_Lean_Expr_const___override(v___x_611_, v___x_553_);
v___x_613_ = l_Lean_Expr_app___override(v___x_612_, v___x_590_);
v___x_614_ = l_Lean_Expr_app___override(v___x_613_, v_d_549_);
v___x_615_ = l_Lean_Expr_app___override(v___x_614_, v_00_u03b1_281_);
v___x_616_ = l_Lean_Expr_app___override(v___x_615_, v_inst_546_);
v___x_617_ = l_Lean_Expr_app___override(v___x_616_, v_e_283_);
v___x_618_ = l_Lean_Expr_app___override(v___x_617_, v_proof_550_);
v___x_619_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_619_, 0, v___x_609_);
lean_ctor_set(v___x_619_, 1, v___x_610_);
lean_ctor_set(v___x_619_, 2, v___x_618_);
v___x_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
return v___x_620_;
}
default: 
{
lean_object* v___x_621_; 
lean_dec_ref(v_x_284_);
lean_dec_ref(v_e_283_);
lean_dec_ref(v_s_u03b1_282_);
lean_dec_ref(v_00_u03b1_281_);
lean_dec(v_u_280_);
v___x_621_ = lean_box(0);
return v___x_621_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13(void){
_start:
{
lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; 
v___x_667_ = lean_box(0);
v___x_668_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__12));
v___x_669_ = l_Lean_Expr_const___override(v___x_668_, v___x_667_);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast(lean_object* v_u_696_, lean_object* v_00_u03b1_697_, lean_object* v_s_u03b1_698_, lean_object* v_v_699_, lean_object* v_00_u03b2_700_, lean_object* v_s_u03b2_701_, lean_object* v_a_702_, lean_object* v_va_703_, lean_object* v_a_704_, lean_object* v_a_705_, lean_object* v_a_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_){
_start:
{
if (lean_obj_tag(v_va_703_) == 0)
{
lean_object* v_value_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_758_; 
v_value_711_ = lean_ctor_get(v_va_703_, 1);
v_isSharedCheck_758_ = !lean_is_exclusive(v_va_703_);
if (v_isSharedCheck_758_ == 0)
{
lean_object* v_unused_759_; 
v_unused_759_ = lean_ctor_get(v_va_703_, 0);
lean_dec(v_unused_759_);
v___x_713_ = v_va_703_;
v_isShared_714_ = v_isSharedCheck_758_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_value_711_);
lean_dec(v_va_703_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_758_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v_value_715_; lean_object* v_hyp_716_; lean_object* v___x_718_; uint8_t v_isShared_719_; uint8_t v_isSharedCheck_757_; 
v_value_715_ = lean_ctor_get(v_value_711_, 0);
v_hyp_716_ = lean_ctor_get(v_value_711_, 1);
v_isSharedCheck_757_ = !lean_is_exclusive(v_value_711_);
if (v_isSharedCheck_757_ == 0)
{
v___x_718_ = v_value_711_;
v_isShared_719_ = v_isSharedCheck_757_;
goto v_resetjp_717_;
}
else
{
lean_inc(v_hyp_716_);
lean_inc(v_value_715_);
lean_dec(v_value_711_);
v___x_718_ = lean_box(0);
v_isShared_719_ = v_isSharedCheck_757_;
goto v_resetjp_717_;
}
v_resetjp_717_:
{
lean_object* v_n_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_745_; 
v_n_720_ = l_Lean_Expr_appArg_x21(v_a_702_);
v___x_721_ = lean_box(0);
v___x_722_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_722_, 0, v_u_696_);
lean_ctor_set(v___x_722_, 1, v___x_721_);
v___x_723_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5));
lean_inc_ref_n(v___x_722_, 5);
v___x_724_ = l_Lean_Expr_const___override(v___x_723_, v___x_722_);
lean_inc_ref_n(v_00_u03b1_697_, 5);
v___x_725_ = l_Lean_Expr_app___override(v___x_724_, v_00_u03b1_697_);
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8));
v___x_727_ = l_Lean_Expr_const___override(v___x_726_, v___x_722_);
v___x_728_ = l_Lean_Expr_app___override(v___x_727_, v_00_u03b1_697_);
v___x_729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11));
v___x_730_ = l_Lean_Expr_const___override(v___x_729_, v___x_722_);
v___x_731_ = l_Lean_Expr_app___override(v___x_730_, v_00_u03b1_697_);
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_733_ = l_Lean_Expr_const___override(v___x_732_, v___x_722_);
v___x_734_ = l_Lean_Expr_app___override(v___x_733_, v_00_u03b1_697_);
lean_inc_ref(v_s_u03b1_698_);
v___x_735_ = l_Lean_Expr_app___override(v___x_734_, v_s_u03b1_698_);
v___x_736_ = l_Lean_Expr_app___override(v___x_731_, v___x_735_);
v___x_737_ = l_Lean_Expr_app___override(v___x_728_, v___x_736_);
v___x_738_ = l_Lean_Expr_app___override(v___x_725_, v___x_737_);
v___x_739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2));
v___x_740_ = l_Lean_Expr_const___override(v___x_739_, v___x_722_);
v___x_741_ = l_Lean_Expr_app___override(v___x_740_, v_00_u03b1_697_);
v___x_742_ = l_Lean_Expr_app___override(v___x_741_, v___x_738_);
lean_inc_ref(v_n_720_);
v___x_743_ = l_Lean_Expr_app___override(v___x_742_, v_n_720_);
if (v_isShared_719_ == 0)
{
v___x_745_ = v___x_718_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v_value_715_);
lean_ctor_set(v_reuseFailAlloc_756_, 1, v_hyp_716_);
v___x_745_ = v_reuseFailAlloc_756_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
lean_object* v___x_747_; 
lean_inc_ref(v___x_743_);
if (v_isShared_714_ == 0)
{
lean_ctor_set(v___x_713_, 1, v___x_745_);
lean_ctor_set(v___x_713_, 0, v___x_743_);
v___x_747_ = v___x_713_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v___x_743_);
lean_ctor_set(v_reuseFailAlloc_755_, 1, v___x_745_);
v___x_747_ = v_reuseFailAlloc_755_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; 
v___x_748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__1));
v___x_749_ = l_Lean_Expr_const___override(v___x_748_, v___x_722_);
v___x_750_ = l_Lean_Expr_app___override(v___x_749_, v_00_u03b1_697_);
v___x_751_ = l_Lean_Expr_app___override(v___x_750_, v_s_u03b1_698_);
v___x_752_ = l_Lean_Expr_app___override(v___x_751_, v_n_720_);
v___x_753_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_753_, 0, v___x_743_);
lean_ctor_set(v___x_753_, 1, v___x_747_);
lean_ctor_set(v___x_753_, 2, v___x_752_);
v___x_754_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_754_, 0, v___x_753_);
return v___x_754_;
}
}
}
}
}
else
{
lean_object* v_x_760_; lean_object* v_e_761_; lean_object* v_b_762_; lean_object* v_a_763_; lean_object* v_a_764_; lean_object* v_a_765_; lean_object* v___x_767_; uint8_t v_isShared_768_; uint8_t v_isSharedCheck_874_; 
v_x_760_ = lean_ctor_get(v_va_703_, 0);
v_e_761_ = lean_ctor_get(v_va_703_, 1);
v_b_762_ = lean_ctor_get(v_va_703_, 2);
v_a_763_ = lean_ctor_get(v_va_703_, 3);
v_a_764_ = lean_ctor_get(v_va_703_, 4);
v_a_765_ = lean_ctor_get(v_va_703_, 5);
v_isSharedCheck_874_ = !lean_is_exclusive(v_va_703_);
if (v_isSharedCheck_874_ == 0)
{
v___x_767_ = v_va_703_;
v_isShared_768_ = v_isSharedCheck_874_;
goto v_resetjp_766_;
}
else
{
lean_inc(v_a_765_);
lean_inc(v_a_764_);
lean_inc(v_a_763_);
lean_inc(v_b_762_);
lean_inc(v_e_761_);
lean_inc(v_x_760_);
lean_dec(v_va_703_);
v___x_767_ = lean_box(0);
v_isShared_768_ = v_isSharedCheck_874_;
goto v_resetjp_766_;
}
v_resetjp_766_:
{
lean_object* v___x_769_; 
lean_inc_ref(v_x_760_);
lean_inc_ref(v_s_u03b1_698_);
lean_inc_ref(v_00_u03b1_697_);
lean_inc(v_u_696_);
v___x_769_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast(v_u_696_, v_00_u03b1_697_, v_s_u03b1_698_, v_v_699_, v_00_u03b2_700_, v_s_u03b2_701_, v_x_760_, v_a_763_, v_a_704_, v_a_705_, v_a_706_, v_a_707_, v_a_708_, v_a_709_);
if (lean_obj_tag(v___x_769_) == 0)
{
lean_object* v_a_770_; lean_object* v_expr_771_; lean_object* v_val_772_; lean_object* v_proof_773_; lean_object* v___x_774_; 
v_a_770_ = lean_ctor_get(v___x_769_, 0);
lean_inc(v_a_770_);
lean_dec_ref_known(v___x_769_, 1);
v_expr_771_ = lean_ctor_get(v_a_770_, 0);
lean_inc_ref(v_expr_771_);
v_val_772_ = lean_ctor_get(v_a_770_, 1);
lean_inc(v_val_772_);
v_proof_773_ = lean_ctor_get(v_a_770_, 2);
lean_inc_ref(v_proof_773_);
lean_dec(v_a_770_);
lean_inc_ref(v_s_u03b1_698_);
lean_inc_ref(v_00_u03b1_697_);
lean_inc(v_u_696_);
v___x_774_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast(v_u_696_, v_00_u03b1_697_, v_s_u03b1_698_, v_v_699_, v_00_u03b2_700_, v_s_u03b2_701_, v_b_762_, v_a_765_, v_a_704_, v_a_705_, v_a_706_, v_a_707_, v_a_708_, v_a_709_);
if (lean_obj_tag(v___x_774_) == 0)
{
lean_object* v_a_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_865_; 
v_a_775_ = lean_ctor_get(v___x_774_, 0);
v_isSharedCheck_865_ = !lean_is_exclusive(v___x_774_);
if (v_isSharedCheck_865_ == 0)
{
v___x_777_ = v___x_774_;
v_isShared_778_ = v_isSharedCheck_865_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_a_775_);
lean_dec(v___x_774_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_865_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v_expr_779_; lean_object* v_val_780_; lean_object* v_proof_781_; lean_object* v___x_783_; uint8_t v_isShared_784_; uint8_t v_isSharedCheck_864_; 
v_expr_779_ = lean_ctor_get(v_a_775_, 0);
v_val_780_ = lean_ctor_get(v_a_775_, 1);
v_proof_781_ = lean_ctor_get(v_a_775_, 2);
v_isSharedCheck_864_ = !lean_is_exclusive(v_a_775_);
if (v_isSharedCheck_864_ == 0)
{
v___x_783_ = v_a_775_;
v_isShared_784_ = v_isSharedCheck_864_;
goto v_resetjp_782_;
}
else
{
lean_inc(v_proof_781_);
lean_inc(v_val_780_);
lean_inc(v_expr_779_);
lean_dec(v_a_775_);
v___x_783_ = lean_box(0);
v_isShared_784_ = v_isSharedCheck_864_;
goto v_resetjp_782_;
}
v_resetjp_782_:
{
lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_845_; 
v___x_785_ = lean_box(0);
lean_inc_n(v_u_696_, 4);
v___x_786_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_786_, 0, v_u_696_);
lean_ctor_set(v___x_786_, 1, v___x_785_);
v___x_787_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref_n(v___x_786_, 9);
v___x_788_ = l_Lean_Expr_const___override(v___x_787_, v___x_786_);
lean_inc_ref_n(v_00_u03b1_697_, 13);
v___x_789_ = l_Lean_Expr_app___override(v___x_788_, v_00_u03b1_697_);
lean_inc_ref(v_s_u03b1_698_);
v___x_790_ = l_Lean_Expr_app___override(v___x_789_, v_s_u03b1_698_);
v___x_791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4));
v___x_792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6));
v___x_793_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8));
v___x_794_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_795_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11));
v___x_796_ = lean_box(0);
v___x_797_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15));
v___x_799_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19));
v___x_801_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22));
v___x_802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24));
v___x_803_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_803_, 0, v_u_696_);
lean_ctor_set(v___x_803_, 1, v___x_786_);
v___x_804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_804_, 0, v_u_696_);
lean_ctor_set(v___x_804_, 1, v___x_803_);
v___x_805_ = l_Lean_Expr_const___override(v___x_791_, v___x_804_);
v___x_806_ = l_Lean_Expr_app___override(v___x_805_, v_00_u03b1_697_);
v___x_807_ = l_Lean_Expr_app___override(v___x_806_, v_00_u03b1_697_);
v___x_808_ = l_Lean_Expr_app___override(v___x_807_, v_00_u03b1_697_);
v___x_809_ = l_Lean_Expr_const___override(v___x_792_, v___x_786_);
v___x_810_ = l_Lean_Expr_app___override(v___x_809_, v_00_u03b1_697_);
v___x_811_ = l_Lean_Expr_const___override(v___x_793_, v___x_786_);
v___x_812_ = l_Lean_Expr_app___override(v___x_811_, v_00_u03b1_697_);
v___x_813_ = l_Lean_Expr_const___override(v___x_794_, v___x_786_);
v___x_814_ = l_Lean_Expr_app___override(v___x_813_, v_00_u03b1_697_);
lean_inc_ref(v___x_790_);
v___x_815_ = l_Lean_Expr_app___override(v___x_814_, v___x_790_);
v___x_816_ = l_Lean_Expr_app___override(v___x_812_, v___x_815_);
v___x_817_ = l_Lean_Expr_app___override(v___x_810_, v___x_816_);
v___x_818_ = l_Lean_Expr_app___override(v___x_808_, v___x_817_);
v___x_819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_796_);
lean_ctor_set(v___x_819_, 1, v___x_786_);
v___x_820_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_820_, 0, v_u_696_);
lean_ctor_set(v___x_820_, 1, v___x_819_);
v___x_821_ = l_Lean_Expr_const___override(v___x_795_, v___x_820_);
v___x_822_ = l_Lean_Expr_app___override(v___x_821_, v_00_u03b1_697_);
v___x_823_ = l_Lean_Expr_app___override(v___x_822_, v___x_797_);
v___x_824_ = l_Lean_Expr_app___override(v___x_823_, v_00_u03b1_697_);
v___x_825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_825_, 0, v_u_696_);
lean_ctor_set(v___x_825_, 1, v___x_799_);
v___x_826_ = l_Lean_Expr_const___override(v___x_798_, v___x_825_);
v___x_827_ = l_Lean_Expr_app___override(v___x_826_, v_00_u03b1_697_);
v___x_828_ = l_Lean_Expr_app___override(v___x_827_, v___x_797_);
v___x_829_ = l_Lean_Expr_const___override(v___x_800_, v___x_786_);
v___x_830_ = l_Lean_Expr_app___override(v___x_829_, v_00_u03b1_697_);
v___x_831_ = l_Lean_Expr_const___override(v___x_801_, v___x_786_);
v___x_832_ = l_Lean_Expr_app___override(v___x_831_, v_00_u03b1_697_);
v___x_833_ = l_Lean_Expr_const___override(v___x_802_, v___x_786_);
v___x_834_ = l_Lean_Expr_app___override(v___x_833_, v_00_u03b1_697_);
v___x_835_ = l_Lean_Expr_app___override(v___x_834_, v___x_790_);
v___x_836_ = l_Lean_Expr_app___override(v___x_832_, v___x_835_);
v___x_837_ = l_Lean_Expr_app___override(v___x_830_, v___x_836_);
v___x_838_ = l_Lean_Expr_app___override(v___x_828_, v___x_837_);
v___x_839_ = l_Lean_Expr_app___override(v___x_824_, v___x_838_);
lean_inc_ref_n(v_expr_771_, 2);
v___x_840_ = l_Lean_Expr_app___override(v___x_839_, v_expr_771_);
lean_inc_ref_n(v_e_761_, 2);
v___x_841_ = l_Lean_Expr_app___override(v___x_840_, v_e_761_);
v___x_842_ = l_Lean_Expr_app___override(v___x_818_, v___x_841_);
lean_inc_ref_n(v_expr_779_, 2);
v___x_843_ = l_Lean_Expr_app___override(v___x_842_, v_expr_779_);
if (v_isShared_768_ == 0)
{
lean_ctor_set(v___x_767_, 5, v_val_780_);
lean_ctor_set(v___x_767_, 3, v_val_772_);
lean_ctor_set(v___x_767_, 2, v_expr_779_);
lean_ctor_set(v___x_767_, 0, v_expr_771_);
v___x_845_ = v___x_767_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(1, 6, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_expr_771_);
lean_ctor_set(v_reuseFailAlloc_863_, 1, v_e_761_);
lean_ctor_set(v_reuseFailAlloc_863_, 2, v_expr_779_);
lean_ctor_set(v_reuseFailAlloc_863_, 3, v_val_772_);
lean_ctor_set(v_reuseFailAlloc_863_, 4, v_a_764_);
lean_ctor_set(v_reuseFailAlloc_863_, 5, v_val_780_);
v___x_845_ = v_reuseFailAlloc_863_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_858_; 
v___x_846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__26));
v___x_847_ = l_Lean_Expr_const___override(v___x_846_, v___x_786_);
v___x_848_ = l_Lean_Expr_app___override(v___x_847_, v_00_u03b1_697_);
v___x_849_ = l_Lean_Expr_app___override(v___x_848_, v_s_u03b1_698_);
v___x_850_ = l_Lean_Expr_app___override(v___x_849_, v_expr_771_);
v___x_851_ = l_Lean_Expr_app___override(v___x_850_, v_expr_779_);
v___x_852_ = l_Lean_Expr_app___override(v___x_851_, v_x_760_);
v___x_853_ = l_Lean_Expr_app___override(v___x_852_, v_b_762_);
v___x_854_ = l_Lean_Expr_app___override(v___x_853_, v_e_761_);
v___x_855_ = l_Lean_Expr_app___override(v___x_854_, v_proof_773_);
v___x_856_ = l_Lean_Expr_app___override(v___x_855_, v_proof_781_);
if (v_isShared_784_ == 0)
{
lean_ctor_set(v___x_783_, 2, v___x_856_);
lean_ctor_set(v___x_783_, 1, v___x_845_);
lean_ctor_set(v___x_783_, 0, v___x_843_);
v___x_858_ = v___x_783_;
goto v_reusejp_857_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v___x_843_);
lean_ctor_set(v_reuseFailAlloc_862_, 1, v___x_845_);
lean_ctor_set(v_reuseFailAlloc_862_, 2, v___x_856_);
v___x_858_ = v_reuseFailAlloc_862_;
goto v_reusejp_857_;
}
v_reusejp_857_:
{
lean_object* v___x_860_; 
if (v_isShared_778_ == 0)
{
lean_ctor_set(v___x_777_, 0, v___x_858_);
v___x_860_ = v___x_777_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v___x_858_);
v___x_860_ = v_reuseFailAlloc_861_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
return v___x_860_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_proof_773_);
lean_dec(v_val_772_);
lean_dec_ref(v_expr_771_);
lean_del_object(v___x_767_);
lean_dec_ref(v_a_764_);
lean_dec_ref(v_b_762_);
lean_dec_ref(v_e_761_);
lean_dec_ref(v_x_760_);
lean_dec_ref(v_s_u03b1_698_);
lean_dec_ref(v_00_u03b1_697_);
lean_dec(v_u_696_);
return v___x_774_;
}
}
else
{
lean_object* v_a_866_; lean_object* v___x_868_; uint8_t v_isShared_869_; uint8_t v_isSharedCheck_873_; 
lean_del_object(v___x_767_);
lean_dec_ref(v_a_765_);
lean_dec_ref(v_a_764_);
lean_dec_ref(v_b_762_);
lean_dec_ref(v_e_761_);
lean_dec_ref(v_x_760_);
lean_dec_ref(v_s_u03b1_698_);
lean_dec_ref(v_00_u03b1_697_);
lean_dec(v_u_696_);
v_a_866_ = lean_ctor_get(v___x_769_, 0);
v_isSharedCheck_873_ = !lean_is_exclusive(v___x_769_);
if (v_isSharedCheck_873_ == 0)
{
v___x_868_ = v___x_769_;
v_isShared_869_ = v_isSharedCheck_873_;
goto v_resetjp_867_;
}
else
{
lean_inc(v_a_866_);
lean_dec(v___x_769_);
v___x_868_ = lean_box(0);
v_isShared_869_ = v_isSharedCheck_873_;
goto v_resetjp_867_;
}
v_resetjp_867_:
{
lean_object* v___x_871_; 
if (v_isShared_869_ == 0)
{
v___x_871_ = v___x_868_;
goto v_reusejp_870_;
}
else
{
lean_object* v_reuseFailAlloc_872_; 
v_reuseFailAlloc_872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_872_, 0, v_a_866_);
v___x_871_ = v_reuseFailAlloc_872_;
goto v_reusejp_870_;
}
v_reusejp_870_:
{
return v___x_871_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(lean_object* v_u_881_, lean_object* v_00_u03b1_882_, lean_object* v_s_u03b1_883_, lean_object* v_v_884_, lean_object* v_00_u03b2_885_, lean_object* v_s_u03b2_886_, lean_object* v_va_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_){
_start:
{
if (lean_obj_tag(v_va_887_) == 0)
{
lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v___x_895_ = lean_box(0);
v___x_896_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_896_, 0, v_u_881_);
lean_ctor_set(v___x_896_, 1, v___x_895_);
v___x_897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref_n(v___x_896_, 5);
v___x_898_ = l_Lean_Expr_const___override(v___x_897_, v___x_896_);
lean_inc_ref_n(v_00_u03b1_882_, 5);
v___x_899_ = l_Lean_Expr_app___override(v___x_898_, v_00_u03b1_882_);
lean_inc_ref(v_s_u03b1_883_);
v___x_900_ = l_Lean_Expr_app___override(v___x_899_, v_s_u03b1_883_);
v___x_901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_902_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_905_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_906_ = l_Lean_Expr_const___override(v___x_901_, v___x_896_);
v___x_907_ = l_Lean_Expr_app___override(v___x_906_, v_00_u03b1_882_);
v___x_908_ = l_Lean_Expr_app___override(v___x_907_, v___x_902_);
v___x_909_ = l_Lean_Expr_const___override(v___x_903_, v___x_896_);
v___x_910_ = l_Lean_Expr_app___override(v___x_909_, v_00_u03b1_882_);
v___x_911_ = l_Lean_Expr_const___override(v___x_904_, v___x_896_);
v___x_912_ = l_Lean_Expr_app___override(v___x_911_, v_00_u03b1_882_);
v___x_913_ = l_Lean_Expr_const___override(v___x_905_, v___x_896_);
v___x_914_ = l_Lean_Expr_app___override(v___x_913_, v_00_u03b1_882_);
v___x_915_ = l_Lean_Expr_app___override(v___x_914_, v___x_900_);
v___x_916_ = l_Lean_Expr_app___override(v___x_912_, v___x_915_);
v___x_917_ = l_Lean_Expr_app___override(v___x_910_, v___x_916_);
v___x_918_ = l_Lean_Expr_app___override(v___x_908_, v___x_917_);
v___x_919_ = lean_box(0);
v___x_920_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__1));
v___x_921_ = l_Lean_Expr_const___override(v___x_920_, v___x_896_);
v___x_922_ = l_Lean_Expr_app___override(v___x_921_, v_00_u03b1_882_);
v___x_923_ = l_Lean_Expr_app___override(v___x_922_, v_s_u03b1_883_);
v___x_924_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_924_, 0, v___x_918_);
lean_ctor_set(v___x_924_, 1, v___x_919_);
lean_ctor_set(v___x_924_, 2, v___x_923_);
v___x_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
return v___x_925_;
}
else
{
lean_object* v_a_926_; lean_object* v_b_927_; lean_object* v_a_928_; lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_1006_; 
v_a_926_ = lean_ctor_get(v_va_887_, 0);
v_b_927_ = lean_ctor_get(v_va_887_, 1);
v_a_928_ = lean_ctor_get(v_va_887_, 2);
v_a_929_ = lean_ctor_get(v_va_887_, 3);
v_isSharedCheck_1006_ = !lean_is_exclusive(v_va_887_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_931_ = v_va_887_;
v_isShared_932_ = v_isSharedCheck_1006_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_inc(v_a_928_);
lean_inc(v_b_927_);
lean_inc(v_a_926_);
lean_dec(v_va_887_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_1006_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
lean_object* v___x_933_; 
lean_inc_ref(v_s_u03b1_883_);
lean_inc_ref(v_00_u03b1_882_);
lean_inc(v_u_881_);
v___x_933_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast(v_u_881_, v_00_u03b1_882_, v_s_u03b1_883_, v_v_884_, v_00_u03b2_885_, v_s_u03b2_886_, v_a_926_, v_a_928_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_);
if (lean_obj_tag(v___x_933_) == 0)
{
lean_object* v_a_934_; lean_object* v_expr_935_; lean_object* v_val_936_; lean_object* v_proof_937_; lean_object* v___x_938_; 
v_a_934_ = lean_ctor_get(v___x_933_, 0);
lean_inc(v_a_934_);
lean_dec_ref_known(v___x_933_, 1);
v_expr_935_ = lean_ctor_get(v_a_934_, 0);
lean_inc_ref(v_expr_935_);
v_val_936_ = lean_ctor_get(v_a_934_, 1);
lean_inc(v_val_936_);
v_proof_937_ = lean_ctor_get(v_a_934_, 2);
lean_inc_ref(v_proof_937_);
lean_dec(v_a_934_);
lean_inc_ref(v_s_u03b1_883_);
lean_inc_ref(v_00_u03b1_882_);
lean_inc(v_u_881_);
v___x_938_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(v_u_881_, v_00_u03b1_882_, v_s_u03b1_883_, v_v_884_, v_00_u03b2_885_, v_s_u03b2_886_, v_a_929_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_);
if (lean_obj_tag(v___x_938_) == 0)
{
lean_object* v_a_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_997_; 
v_a_939_ = lean_ctor_get(v___x_938_, 0);
v_isSharedCheck_997_ = !lean_is_exclusive(v___x_938_);
if (v_isSharedCheck_997_ == 0)
{
v___x_941_ = v___x_938_;
v_isShared_942_ = v_isSharedCheck_997_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_a_939_);
lean_dec(v___x_938_);
v___x_941_ = lean_box(0);
v_isShared_942_ = v_isSharedCheck_997_;
goto v_resetjp_940_;
}
v_resetjp_940_:
{
lean_object* v_expr_943_; lean_object* v_val_944_; lean_object* v_proof_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_996_; 
v_expr_943_ = lean_ctor_get(v_a_939_, 0);
v_val_944_ = lean_ctor_get(v_a_939_, 1);
v_proof_945_ = lean_ctor_get(v_a_939_, 2);
v_isSharedCheck_996_ = !lean_is_exclusive(v_a_939_);
if (v_isSharedCheck_996_ == 0)
{
v___x_947_ = v_a_939_;
v_isShared_948_ = v_isSharedCheck_996_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_proof_945_);
lean_inc(v_val_944_);
lean_inc(v_expr_943_);
lean_dec(v_a_939_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_996_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_978_; 
v___x_949_ = lean_box(0);
lean_inc_n(v_u_881_, 2);
v___x_950_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_950_, 0, v_u_881_);
lean_ctor_set(v___x_950_, 1, v___x_949_);
v___x_951_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref_n(v___x_950_, 5);
v___x_952_ = l_Lean_Expr_const___override(v___x_951_, v___x_950_);
lean_inc_ref_n(v_00_u03b1_882_, 7);
v___x_953_ = l_Lean_Expr_app___override(v___x_952_, v_00_u03b1_882_);
lean_inc_ref(v_s_u03b1_883_);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v_s_u03b1_883_);
v___x_955_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_956_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_957_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_958_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_959_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_959_, 0, v_u_881_);
lean_ctor_set(v___x_959_, 1, v___x_950_);
v___x_960_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_960_, 0, v_u_881_);
lean_ctor_set(v___x_960_, 1, v___x_959_);
v___x_961_ = l_Lean_Expr_const___override(v___x_955_, v___x_960_);
v___x_962_ = l_Lean_Expr_app___override(v___x_961_, v_00_u03b1_882_);
v___x_963_ = l_Lean_Expr_app___override(v___x_962_, v_00_u03b1_882_);
v___x_964_ = l_Lean_Expr_app___override(v___x_963_, v_00_u03b1_882_);
v___x_965_ = l_Lean_Expr_const___override(v___x_956_, v___x_950_);
v___x_966_ = l_Lean_Expr_app___override(v___x_965_, v_00_u03b1_882_);
v___x_967_ = l_Lean_Expr_const___override(v___x_957_, v___x_950_);
v___x_968_ = l_Lean_Expr_app___override(v___x_967_, v_00_u03b1_882_);
v___x_969_ = l_Lean_Expr_const___override(v___x_958_, v___x_950_);
v___x_970_ = l_Lean_Expr_app___override(v___x_969_, v_00_u03b1_882_);
v___x_971_ = l_Lean_Expr_app___override(v___x_970_, v___x_954_);
v___x_972_ = l_Lean_Expr_app___override(v___x_968_, v___x_971_);
v___x_973_ = l_Lean_Expr_app___override(v___x_966_, v___x_972_);
v___x_974_ = l_Lean_Expr_app___override(v___x_964_, v___x_973_);
lean_inc_ref_n(v_expr_935_, 2);
v___x_975_ = l_Lean_Expr_app___override(v___x_974_, v_expr_935_);
lean_inc_ref_n(v_expr_943_, 2);
v___x_976_ = l_Lean_Expr_app___override(v___x_975_, v_expr_943_);
if (v_isShared_932_ == 0)
{
lean_ctor_set(v___x_931_, 3, v_val_944_);
lean_ctor_set(v___x_931_, 2, v_val_936_);
lean_ctor_set(v___x_931_, 1, v_expr_943_);
lean_ctor_set(v___x_931_, 0, v_expr_935_);
v___x_978_ = v___x_931_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v_expr_935_);
lean_ctor_set(v_reuseFailAlloc_995_, 1, v_expr_943_);
lean_ctor_set(v_reuseFailAlloc_995_, 2, v_val_936_);
lean_ctor_set(v_reuseFailAlloc_995_, 3, v_val_944_);
v___x_978_ = v_reuseFailAlloc_995_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_990_; 
v___x_979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___closed__3));
v___x_980_ = l_Lean_Expr_const___override(v___x_979_, v___x_950_);
v___x_981_ = l_Lean_Expr_app___override(v___x_980_, v_00_u03b1_882_);
v___x_982_ = l_Lean_Expr_app___override(v___x_981_, v_s_u03b1_883_);
v___x_983_ = l_Lean_Expr_app___override(v___x_982_, v_expr_935_);
v___x_984_ = l_Lean_Expr_app___override(v___x_983_, v_expr_943_);
v___x_985_ = l_Lean_Expr_app___override(v___x_984_, v_a_926_);
v___x_986_ = l_Lean_Expr_app___override(v___x_985_, v_b_927_);
v___x_987_ = l_Lean_Expr_app___override(v___x_986_, v_proof_937_);
v___x_988_ = l_Lean_Expr_app___override(v___x_987_, v_proof_945_);
if (v_isShared_948_ == 0)
{
lean_ctor_set(v___x_947_, 2, v___x_988_);
lean_ctor_set(v___x_947_, 1, v___x_978_);
lean_ctor_set(v___x_947_, 0, v___x_976_);
v___x_990_ = v___x_947_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_994_; 
v_reuseFailAlloc_994_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_994_, 0, v___x_976_);
lean_ctor_set(v_reuseFailAlloc_994_, 1, v___x_978_);
lean_ctor_set(v_reuseFailAlloc_994_, 2, v___x_988_);
v___x_990_ = v_reuseFailAlloc_994_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
lean_object* v___x_992_; 
if (v_isShared_942_ == 0)
{
lean_ctor_set(v___x_941_, 0, v___x_990_);
v___x_992_ = v___x_941_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v___x_990_);
v___x_992_ = v_reuseFailAlloc_993_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
return v___x_992_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_proof_937_);
lean_dec(v_val_936_);
lean_dec_ref(v_expr_935_);
lean_del_object(v___x_931_);
lean_dec_ref(v_b_927_);
lean_dec_ref(v_a_926_);
lean_dec_ref(v_s_u03b1_883_);
lean_dec_ref(v_00_u03b1_882_);
lean_dec(v_u_881_);
return v___x_938_;
}
}
else
{
lean_object* v_a_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1005_; 
lean_del_object(v___x_931_);
lean_dec(v_a_929_);
lean_dec_ref(v_b_927_);
lean_dec_ref(v_a_926_);
lean_dec_ref(v_s_u03b1_883_);
lean_dec_ref(v_00_u03b1_882_);
lean_dec(v_u_881_);
v_a_998_ = lean_ctor_get(v___x_933_, 0);
v_isSharedCheck_1005_ = !lean_is_exclusive(v___x_933_);
if (v_isSharedCheck_1005_ == 0)
{
v___x_1000_ = v___x_933_;
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_a_998_);
lean_dec(v___x_933_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1003_; 
if (v_isShared_1001_ == 0)
{
v___x_1003_ = v___x_1000_;
goto v_reusejp_1002_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v_a_998_);
v___x_1003_ = v_reuseFailAlloc_1004_;
goto v_reusejp_1002_;
}
v_reusejp_1002_:
{
return v___x_1003_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast(lean_object* v_u_1007_, lean_object* v_00_u03b1_1008_, lean_object* v_s_u03b1_1009_, lean_object* v_v_1010_, lean_object* v_00_u03b2_1011_, lean_object* v_s_u03b2_1012_, lean_object* v_a_1013_, lean_object* v_va_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_, lean_object* v_a_1017_, lean_object* v_a_1018_, lean_object* v_a_1019_, lean_object* v_a_1020_){
_start:
{
if (lean_obj_tag(v_va_1014_) == 0)
{
lean_object* v___x_1023_; uint8_t v_isShared_1024_; uint8_t v_isSharedCheck_1087_; 
v_isSharedCheck_1087_ = !lean_is_exclusive(v_va_1014_);
if (v_isSharedCheck_1087_ == 0)
{
lean_object* v_unused_1088_; lean_object* v_unused_1089_; 
v_unused_1088_ = lean_ctor_get(v_va_1014_, 1);
lean_dec(v_unused_1088_);
v_unused_1089_ = lean_ctor_get(v_va_1014_, 0);
lean_dec(v_unused_1089_);
v___x_1023_ = v_va_1014_;
v_isShared_1024_ = v_isSharedCheck_1087_;
goto v_resetjp_1022_;
}
else
{
lean_dec(v_va_1014_);
v___x_1023_ = lean_box(0);
v_isShared_1024_ = v_isSharedCheck_1087_;
goto v_resetjp_1022_;
}
v_resetjp_1022_:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1025_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__1));
v___x_1026_ = lean_box(0);
lean_inc(v_u_1007_);
v___x_1027_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1027_, 0, v_u_1007_);
lean_ctor_set(v___x_1027_, 1, v___x_1026_);
lean_inc_ref_n(v___x_1027_, 5);
v___x_1028_ = l_Lean_Expr_const___override(v___x_1025_, v___x_1027_);
lean_inc_ref_n(v_00_u03b1_1008_, 6);
v___x_1029_ = l_Lean_Expr_app___override(v___x_1028_, v_00_u03b1_1008_);
v___x_1030_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__4));
v___x_1031_ = l_Lean_Expr_const___override(v___x_1030_, v___x_1027_);
v___x_1032_ = l_Lean_Expr_app___override(v___x_1031_, v_00_u03b1_1008_);
v___x_1033_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5));
v___x_1034_ = l_Lean_Expr_const___override(v___x_1033_, v___x_1027_);
v___x_1035_ = l_Lean_Expr_app___override(v___x_1034_, v_00_u03b1_1008_);
v___x_1036_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8));
v___x_1037_ = l_Lean_Expr_const___override(v___x_1036_, v___x_1027_);
v___x_1038_ = l_Lean_Expr_app___override(v___x_1037_, v_00_u03b1_1008_);
v___x_1039_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11));
v___x_1040_ = l_Lean_Expr_const___override(v___x_1039_, v___x_1027_);
v___x_1041_ = l_Lean_Expr_app___override(v___x_1040_, v_00_u03b1_1008_);
v___x_1042_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_1043_ = l_Lean_Expr_const___override(v___x_1042_, v___x_1027_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1043_, v_00_u03b1_1008_);
v___x_1045_ = l_Lean_Expr_app___override(v___x_1044_, v_s_u03b1_1009_);
v___x_1046_ = l_Lean_Expr_app___override(v___x_1041_, v___x_1045_);
v___x_1047_ = l_Lean_Expr_app___override(v___x_1038_, v___x_1046_);
v___x_1048_ = l_Lean_Expr_app___override(v___x_1035_, v___x_1047_);
v___x_1049_ = l_Lean_Expr_app___override(v___x_1032_, v___x_1048_);
v___x_1050_ = l_Lean_Expr_app___override(v___x_1029_, v___x_1049_);
v___x_1051_ = l_Lean_Expr_app___override(v___x_1050_, v_a_1013_);
v___x_1052_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(v___x_1051_, v_a_1015_, v_a_1016_, v_a_1017_, v_a_1018_, v_a_1019_, v_a_1020_);
if (lean_obj_tag(v___x_1052_) == 0)
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1078_; 
v_a_1053_ = lean_ctor_get(v___x_1052_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1052_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1055_ = v___x_1052_;
v_isShared_1056_ = v_isSharedCheck_1078_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_1052_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1078_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v_fst_1057_; lean_object* v_snd_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1077_; 
v_fst_1057_ = lean_ctor_get(v_a_1053_, 0);
v_snd_1058_ = lean_ctor_get(v_a_1053_, 1);
v_isSharedCheck_1077_ = !lean_is_exclusive(v_a_1053_);
if (v_isSharedCheck_1077_ == 0)
{
v___x_1060_ = v_a_1053_;
v_isShared_1061_ = v_isSharedCheck_1077_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_snd_1058_);
lean_inc(v_fst_1057_);
lean_dec(v_a_1053_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1077_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v___x_1063_; 
lean_inc(v_snd_1058_);
if (v_isShared_1024_ == 0)
{
lean_ctor_set(v___x_1023_, 1, v_fst_1057_);
lean_ctor_set(v___x_1023_, 0, v_snd_1058_);
v___x_1063_ = v___x_1023_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v_snd_1058_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_fst_1057_);
v___x_1063_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1062_;
}
v_reusejp_1062_:
{
lean_object* v___x_1064_; lean_object* v___x_1066_; 
v___x_1064_ = l_Lean_Level_succ___override(v_u_1007_);
if (v_isShared_1061_ == 0)
{
lean_ctor_set_tag(v___x_1060_, 1);
lean_ctor_set(v___x_1060_, 1, v___x_1026_);
lean_ctor_set(v___x_1060_, 0, v___x_1064_);
v___x_1066_ = v___x_1060_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1075_, 1, v___x_1026_);
v___x_1066_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1073_; 
v___x_1067_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7));
v___x_1068_ = l_Lean_Expr_const___override(v___x_1067_, v___x_1066_);
v___x_1069_ = l_Lean_Expr_app___override(v___x_1068_, v_00_u03b1_1008_);
lean_inc(v_snd_1058_);
v___x_1070_ = l_Lean_Expr_app___override(v___x_1069_, v_snd_1058_);
v___x_1071_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1071_, 0, v_snd_1058_);
lean_ctor_set(v___x_1071_, 1, v___x_1063_);
lean_ctor_set(v___x_1071_, 2, v___x_1070_);
if (v_isShared_1056_ == 0)
{
lean_ctor_set(v___x_1055_, 0, v___x_1071_);
v___x_1073_ = v___x_1055_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v___x_1071_);
v___x_1073_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
return v___x_1073_;
}
}
}
}
}
}
else
{
lean_object* v_a_1079_; lean_object* v___x_1081_; uint8_t v_isShared_1082_; uint8_t v_isSharedCheck_1086_; 
lean_del_object(v___x_1023_);
lean_dec_ref(v_00_u03b1_1008_);
lean_dec(v_u_1007_);
v_a_1079_ = lean_ctor_get(v___x_1052_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1052_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1081_ = v___x_1052_;
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
else
{
lean_inc(v_a_1079_);
lean_dec(v___x_1052_);
v___x_1081_ = lean_box(0);
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
v_resetjp_1080_:
{
lean_object* v___x_1084_; 
if (v_isShared_1082_ == 0)
{
v___x_1084_ = v___x_1081_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v_a_1079_);
v___x_1084_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
return v___x_1084_;
}
}
}
}
}
else
{
lean_object* v_x_1090_; lean_object* v___x_1092_; uint8_t v_isShared_1093_; uint8_t v_isSharedCheck_1124_; 
lean_dec_ref(v_a_1013_);
v_x_1090_ = lean_ctor_get(v_va_1014_, 1);
v_isSharedCheck_1124_ = !lean_is_exclusive(v_va_1014_);
if (v_isSharedCheck_1124_ == 0)
{
lean_object* v_unused_1125_; 
v_unused_1125_ = lean_ctor_get(v_va_1014_, 0);
lean_dec(v_unused_1125_);
v___x_1092_ = v_va_1014_;
v_isShared_1093_ = v_isSharedCheck_1124_;
goto v_resetjp_1091_;
}
else
{
lean_inc(v_x_1090_);
lean_dec(v_va_1014_);
v___x_1092_ = lean_box(0);
v_isShared_1093_ = v_isSharedCheck_1124_;
goto v_resetjp_1091_;
}
v_resetjp_1091_:
{
lean_object* v___x_1094_; 
v___x_1094_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(v_u_1007_, v_00_u03b1_1008_, v_s_u03b1_1009_, v_v_1010_, v_00_u03b2_1011_, v_s_u03b2_1012_, v_x_1090_, v_a_1015_, v_a_1016_, v_a_1017_, v_a_1018_, v_a_1019_, v_a_1020_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v_a_1095_; lean_object* v___x_1097_; uint8_t v_isShared_1098_; uint8_t v_isSharedCheck_1115_; 
v_a_1095_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1115_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1115_ == 0)
{
v___x_1097_ = v___x_1094_;
v_isShared_1098_ = v_isSharedCheck_1115_;
goto v_resetjp_1096_;
}
else
{
lean_inc(v_a_1095_);
lean_dec(v___x_1094_);
v___x_1097_ = lean_box(0);
v_isShared_1098_ = v_isSharedCheck_1115_;
goto v_resetjp_1096_;
}
v_resetjp_1096_:
{
lean_object* v_expr_1099_; lean_object* v_val_1100_; lean_object* v_proof_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1114_; 
v_expr_1099_ = lean_ctor_get(v_a_1095_, 0);
v_val_1100_ = lean_ctor_get(v_a_1095_, 1);
v_proof_1101_ = lean_ctor_get(v_a_1095_, 2);
v_isSharedCheck_1114_ = !lean_is_exclusive(v_a_1095_);
if (v_isSharedCheck_1114_ == 0)
{
v___x_1103_ = v_a_1095_;
v_isShared_1104_ = v_isSharedCheck_1114_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_proof_1101_);
lean_inc(v_val_1100_);
lean_inc(v_expr_1099_);
lean_dec(v_a_1095_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1114_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1106_; 
lean_inc_ref(v_expr_1099_);
if (v_isShared_1093_ == 0)
{
lean_ctor_set(v___x_1092_, 1, v_val_1100_);
lean_ctor_set(v___x_1092_, 0, v_expr_1099_);
v___x_1106_ = v___x_1092_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v_expr_1099_);
lean_ctor_set(v_reuseFailAlloc_1113_, 1, v_val_1100_);
v___x_1106_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
lean_object* v___x_1108_; 
if (v_isShared_1104_ == 0)
{
lean_ctor_set(v___x_1103_, 1, v___x_1106_);
v___x_1108_ = v___x_1103_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v_expr_1099_);
lean_ctor_set(v_reuseFailAlloc_1112_, 1, v___x_1106_);
lean_ctor_set(v_reuseFailAlloc_1112_, 2, v_proof_1101_);
v___x_1108_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
lean_object* v___x_1110_; 
if (v_isShared_1098_ == 0)
{
lean_ctor_set(v___x_1097_, 0, v___x_1108_);
v___x_1110_ = v___x_1097_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v___x_1108_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
}
}
}
}
}
}
else
{
lean_object* v_a_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1123_; 
lean_del_object(v___x_1092_);
v_a_1116_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1123_ == 0)
{
v___x_1118_ = v___x_1094_;
v_isShared_1119_ = v_isSharedCheck_1123_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_a_1116_);
lean_dec(v___x_1094_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1123_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
lean_object* v___x_1121_; 
if (v_isShared_1119_ == 0)
{
v___x_1121_ = v___x_1118_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1122_; 
v_reuseFailAlloc_1122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1122_, 0, v_a_1116_);
v___x_1121_ = v_reuseFailAlloc_1122_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
return v___x_1121_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___boxed(lean_object* v_u_1126_, lean_object* v_00_u03b1_1127_, lean_object* v_s_u03b1_1128_, lean_object* v_v_1129_, lean_object* v_00_u03b2_1130_, lean_object* v_s_u03b2_1131_, lean_object* v_a_1132_, lean_object* v_va_1133_, lean_object* v_a_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_){
_start:
{
lean_object* v_res_1141_; 
v_res_1141_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast(v_u_1126_, v_00_u03b1_1127_, v_s_u03b1_1128_, v_v_1129_, v_00_u03b2_1130_, v_s_u03b2_1131_, v_a_1132_, v_va_1133_, v_a_1134_, v_a_1135_, v_a_1136_, v_a_1137_, v_a_1138_, v_a_1139_);
lean_dec(v_a_1139_);
lean_dec_ref(v_a_1138_);
lean_dec(v_a_1137_);
lean_dec_ref(v_a_1136_);
lean_dec(v_a_1135_);
lean_dec_ref(v_a_1134_);
lean_dec_ref(v_s_u03b2_1131_);
lean_dec_ref(v_00_u03b2_1130_);
lean_dec(v_v_1129_);
return v_res_1141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg___boxed(lean_object* v_u_1142_, lean_object* v_00_u03b1_1143_, lean_object* v_s_u03b1_1144_, lean_object* v_v_1145_, lean_object* v_00_u03b2_1146_, lean_object* v_s_u03b2_1147_, lean_object* v_va_1148_, lean_object* v_a_1149_, lean_object* v_a_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(v_u_1142_, v_00_u03b1_1143_, v_s_u03b1_1144_, v_v_1145_, v_00_u03b2_1146_, v_s_u03b2_1147_, v_va_1148_, v_a_1149_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_);
lean_dec(v_a_1154_);
lean_dec_ref(v_a_1153_);
lean_dec(v_a_1152_);
lean_dec_ref(v_a_1151_);
lean_dec(v_a_1150_);
lean_dec_ref(v_a_1149_);
lean_dec_ref(v_s_u03b2_1147_);
lean_dec_ref(v_00_u03b2_1146_);
lean_dec(v_v_1145_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___boxed(lean_object* v_u_1157_, lean_object* v_00_u03b1_1158_, lean_object* v_s_u03b1_1159_, lean_object* v_v_1160_, lean_object* v_00_u03b2_1161_, lean_object* v_s_u03b2_1162_, lean_object* v_a_1163_, lean_object* v_va_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_, lean_object* v_a_1170_, lean_object* v_a_1171_){
_start:
{
lean_object* v_res_1172_; 
v_res_1172_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast(v_u_1157_, v_00_u03b1_1158_, v_s_u03b1_1159_, v_v_1160_, v_00_u03b2_1161_, v_s_u03b2_1162_, v_a_1163_, v_va_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_);
lean_dec(v_a_1170_);
lean_dec_ref(v_a_1169_);
lean_dec(v_a_1168_);
lean_dec_ref(v_a_1167_);
lean_dec(v_a_1166_);
lean_dec_ref(v_a_1165_);
lean_dec_ref(v_a_1163_);
lean_dec_ref(v_s_u03b2_1162_);
lean_dec_ref(v_00_u03b2_1161_);
lean_dec(v_v_1160_);
return v_res_1172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast(lean_object* v_u_1173_, lean_object* v_00_u03b1_1174_, lean_object* v_s_u03b1_1175_, lean_object* v_v_1176_, lean_object* v_00_u03b2_1177_, lean_object* v_s_u03b2_1178_, lean_object* v_a_1179_, lean_object* v_va_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_, lean_object* v_a_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_){
_start:
{
lean_object* v___x_1188_; 
v___x_1188_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(v_u_1173_, v_00_u03b1_1174_, v_s_u03b1_1175_, v_v_1176_, v_00_u03b2_1177_, v_s_u03b2_1178_, v_va_1180_, v_a_1181_, v_a_1182_, v_a_1183_, v_a_1184_, v_a_1185_, v_a_1186_);
return v___x_1188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___boxed(lean_object* v_u_1189_, lean_object* v_00_u03b1_1190_, lean_object* v_s_u03b1_1191_, lean_object* v_v_1192_, lean_object* v_00_u03b2_1193_, lean_object* v_s_u03b2_1194_, lean_object* v_a_1195_, lean_object* v_va_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_, lean_object* v_a_1199_, lean_object* v_a_1200_, lean_object* v_a_1201_, lean_object* v_a_1202_, lean_object* v_a_1203_){
_start:
{
lean_object* v_res_1204_; 
v_res_1204_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast(v_u_1189_, v_00_u03b1_1190_, v_s_u03b1_1191_, v_v_1192_, v_00_u03b2_1193_, v_s_u03b2_1194_, v_a_1195_, v_va_1196_, v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_, v_a_1201_, v_a_1202_);
lean_dec(v_a_1202_);
lean_dec_ref(v_a_1201_);
lean_dec(v_a_1200_);
lean_dec_ref(v_a_1199_);
lean_dec(v_a_1198_);
lean_dec_ref(v_a_1197_);
lean_dec_ref(v_a_1195_);
lean_dec_ref(v_s_u03b2_1194_);
lean_dec_ref(v_00_u03b2_1193_);
lean_dec(v_v_1192_);
return v_res_1204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(lean_object* v_e_1205_, lean_object* v___y_1206_){
_start:
{
uint8_t v___x_1208_; 
v___x_1208_ = l_Lean_Expr_hasMVar(v_e_1205_);
if (v___x_1208_ == 0)
{
lean_object* v___x_1209_; 
v___x_1209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1209_, 0, v_e_1205_);
return v___x_1209_;
}
else
{
lean_object* v___x_1210_; lean_object* v_mctx_1211_; lean_object* v___x_1212_; lean_object* v_fst_1213_; lean_object* v_snd_1214_; lean_object* v___x_1215_; lean_object* v_cache_1216_; lean_object* v_zetaDeltaFVarIds_1217_; lean_object* v_postponed_1218_; lean_object* v_diag_1219_; lean_object* v___x_1221_; uint8_t v_isShared_1222_; uint8_t v_isSharedCheck_1228_; 
v___x_1210_ = lean_st_ref_get(v___y_1206_);
v_mctx_1211_ = lean_ctor_get(v___x_1210_, 0);
lean_inc_ref(v_mctx_1211_);
lean_dec(v___x_1210_);
v___x_1212_ = l_Lean_instantiateMVarsCore(v_mctx_1211_, v_e_1205_);
v_fst_1213_ = lean_ctor_get(v___x_1212_, 0);
lean_inc(v_fst_1213_);
v_snd_1214_ = lean_ctor_get(v___x_1212_, 1);
lean_inc(v_snd_1214_);
lean_dec_ref(v___x_1212_);
v___x_1215_ = lean_st_ref_take(v___y_1206_);
v_cache_1216_ = lean_ctor_get(v___x_1215_, 1);
v_zetaDeltaFVarIds_1217_ = lean_ctor_get(v___x_1215_, 2);
v_postponed_1218_ = lean_ctor_get(v___x_1215_, 3);
v_diag_1219_ = lean_ctor_get(v___x_1215_, 4);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1215_);
if (v_isSharedCheck_1228_ == 0)
{
lean_object* v_unused_1229_; 
v_unused_1229_ = lean_ctor_get(v___x_1215_, 0);
lean_dec(v_unused_1229_);
v___x_1221_ = v___x_1215_;
v_isShared_1222_ = v_isSharedCheck_1228_;
goto v_resetjp_1220_;
}
else
{
lean_inc(v_diag_1219_);
lean_inc(v_postponed_1218_);
lean_inc(v_zetaDeltaFVarIds_1217_);
lean_inc(v_cache_1216_);
lean_dec(v___x_1215_);
v___x_1221_ = lean_box(0);
v_isShared_1222_ = v_isSharedCheck_1228_;
goto v_resetjp_1220_;
}
v_resetjp_1220_:
{
lean_object* v___x_1224_; 
if (v_isShared_1222_ == 0)
{
lean_ctor_set(v___x_1221_, 0, v_snd_1214_);
v___x_1224_ = v___x_1221_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_snd_1214_);
lean_ctor_set(v_reuseFailAlloc_1227_, 1, v_cache_1216_);
lean_ctor_set(v_reuseFailAlloc_1227_, 2, v_zetaDeltaFVarIds_1217_);
lean_ctor_set(v_reuseFailAlloc_1227_, 3, v_postponed_1218_);
lean_ctor_set(v_reuseFailAlloc_1227_, 4, v_diag_1219_);
v___x_1224_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1223_;
}
v_reusejp_1223_:
{
lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1225_ = lean_st_ref_set(v___y_1206_, v___x_1224_);
v___x_1226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1226_, 0, v_fst_1213_);
return v___x_1226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg___boxed(lean_object* v_e_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(v_e_1230_, v___y_1231_);
lean_dec(v___y_1231_);
return v_res_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0(lean_object* v___x_1246_, uint8_t v___x_1247_, lean_object* v___x_1248_, lean_object* v___x_1249_, lean_object* v___x_1250_, lean_object* v_00_u03b2_1251_, lean_object* v_a_1252_, uint8_t v___x_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_){
_start:
{
lean_object* v___x_1259_; 
v___x_1259_ = l_Lean_Meta_mkFreshExprMVar(v___x_1246_, v___x_1247_, v___x_1248_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_object* v_a_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v_keyedConfig_1267_; uint8_t v_trackZetaDelta_1268_; lean_object* v_zetaDeltaSet_1269_; lean_object* v_lctx_1270_; lean_object* v_localInstances_1271_; lean_object* v_defEqCtx_x3f_1272_; lean_object* v_synthPendingDepth_1273_; lean_object* v_customCanUnfoldPredicate_x3f_1274_; uint8_t v_univApprox_1275_; uint8_t v_inTypeClassResolution_1276_; uint8_t v_cacheInferType_1277_; lean_object* v___x_1279_; uint8_t v_isShared_1280_; uint8_t v_isSharedCheck_1328_; 
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_a_1260_);
lean_dec_ref_known(v___x_1259_, 1);
v___x_1261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__1));
v___x_1262_ = l_Lean_Name_mkStr2(v___x_1249_, v___x_1261_);
v___x_1263_ = lean_box(0);
lean_inc(v___x_1250_);
v___x_1264_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1263_);
lean_ctor_set(v___x_1264_, 1, v___x_1250_);
lean_inc_ref(v___x_1264_);
v___x_1265_ = l_Lean_Expr_const___override(v___x_1262_, v___x_1264_);
lean_inc_ref(v_00_u03b2_1251_);
v___x_1266_ = l_Lean_Expr_app___override(v___x_1265_, v_00_u03b2_1251_);
v_keyedConfig_1267_ = lean_ctor_get(v___y_1254_, 0);
v_trackZetaDelta_1268_ = lean_ctor_get_uint8(v___y_1254_, sizeof(void*)*7);
v_zetaDeltaSet_1269_ = lean_ctor_get(v___y_1254_, 1);
v_lctx_1270_ = lean_ctor_get(v___y_1254_, 2);
v_localInstances_1271_ = lean_ctor_get(v___y_1254_, 3);
v_defEqCtx_x3f_1272_ = lean_ctor_get(v___y_1254_, 4);
v_synthPendingDepth_1273_ = lean_ctor_get(v___y_1254_, 5);
v_customCanUnfoldPredicate_x3f_1274_ = lean_ctor_get(v___y_1254_, 6);
v_univApprox_1275_ = lean_ctor_get_uint8(v___y_1254_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1276_ = lean_ctor_get_uint8(v___y_1254_, sizeof(void*)*7 + 2);
v_cacheInferType_1277_ = lean_ctor_get_uint8(v___y_1254_, sizeof(void*)*7 + 3);
v_isSharedCheck_1328_ = !lean_is_exclusive(v___y_1254_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1279_ = v___y_1254_;
v_isShared_1280_ = v_isSharedCheck_1328_;
goto v_resetjp_1278_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1274_);
lean_inc(v_synthPendingDepth_1273_);
lean_inc(v_defEqCtx_x3f_1272_);
lean_inc(v_localInstances_1271_);
lean_inc(v_lctx_1270_);
lean_inc(v_zetaDeltaSet_1269_);
lean_inc(v_keyedConfig_1267_);
lean_dec(v___y_1254_);
v___x_1279_ = lean_box(0);
v_isShared_1280_ = v_isSharedCheck_1328_;
goto v_resetjp_1278_;
}
v_resetjp_1278_:
{
lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; uint8_t v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1296_; 
v___x_1281_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1));
lean_inc_ref(v___x_1264_);
v___x_1282_ = l_Lean_Expr_const___override(v___x_1281_, v___x_1264_);
lean_inc_ref(v_00_u03b2_1251_);
v___x_1283_ = l_Lean_Expr_app___override(v___x_1282_, v_00_u03b2_1251_);
v___x_1284_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3));
v___x_1285_ = l_Lean_Expr_const___override(v___x_1284_, v___x_1264_);
v___x_1286_ = l_Lean_Expr_app___override(v___x_1285_, v_00_u03b2_1251_);
v___x_1287_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5));
v___x_1288_ = l_Lean_Expr_const___override(v___x_1287_, v___x_1250_);
v___x_1289_ = l_Lean_Expr_app___override(v___x_1286_, v___x_1288_);
v___x_1290_ = l_Lean_Expr_app___override(v___x_1283_, v___x_1289_);
v___x_1291_ = l_Lean_Expr_app___override(v___x_1266_, v___x_1290_);
lean_inc(v_a_1260_);
v___x_1292_ = l_Lean_Expr_app___override(v___x_1291_, v_a_1260_);
v___x_1293_ = 2;
v___x_1294_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1293_, v_keyedConfig_1267_);
if (v_isShared_1280_ == 0)
{
lean_ctor_set(v___x_1279_, 0, v___x_1294_);
v___x_1296_ = v___x_1279_;
goto v_reusejp_1295_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v___x_1294_);
lean_ctor_set(v_reuseFailAlloc_1327_, 1, v_zetaDeltaSet_1269_);
lean_ctor_set(v_reuseFailAlloc_1327_, 2, v_lctx_1270_);
lean_ctor_set(v_reuseFailAlloc_1327_, 3, v_localInstances_1271_);
lean_ctor_set(v_reuseFailAlloc_1327_, 4, v_defEqCtx_x3f_1272_);
lean_ctor_set(v_reuseFailAlloc_1327_, 5, v_synthPendingDepth_1273_);
lean_ctor_set(v_reuseFailAlloc_1327_, 6, v_customCanUnfoldPredicate_x3f_1274_);
lean_ctor_set_uint8(v_reuseFailAlloc_1327_, sizeof(void*)*7, v_trackZetaDelta_1268_);
lean_ctor_set_uint8(v_reuseFailAlloc_1327_, sizeof(void*)*7 + 1, v_univApprox_1275_);
lean_ctor_set_uint8(v_reuseFailAlloc_1327_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1276_);
lean_ctor_set_uint8(v_reuseFailAlloc_1327_, sizeof(void*)*7 + 3, v_cacheInferType_1277_);
v___x_1296_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1295_;
}
v_reusejp_1295_:
{
lean_object* v___x_1297_; 
v___x_1297_ = l_Lean_Meta_isExprDefEq(v___x_1292_, v_a_1252_, v___x_1296_, v___y_1255_, v___y_1256_, v___y_1257_);
lean_dec_ref(v___x_1296_);
if (lean_obj_tag(v___x_1297_) == 0)
{
lean_object* v_a_1298_; lean_object* v___x_1300_; uint8_t v_isShared_1301_; uint8_t v_isSharedCheck_1318_; 
v_a_1298_ = lean_ctor_get(v___x_1297_, 0);
v_isSharedCheck_1318_ = !lean_is_exclusive(v___x_1297_);
if (v_isSharedCheck_1318_ == 0)
{
v___x_1300_ = v___x_1297_;
v_isShared_1301_ = v_isSharedCheck_1318_;
goto v_resetjp_1299_;
}
else
{
lean_inc(v_a_1298_);
lean_dec(v___x_1297_);
v___x_1300_ = lean_box(0);
v_isShared_1301_ = v_isSharedCheck_1318_;
goto v_resetjp_1299_;
}
v_resetjp_1299_:
{
uint8_t v___x_1302_; 
v___x_1302_ = lean_unbox(v_a_1298_);
if (v___x_1302_ == 0)
{
lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1306_; 
lean_dec(v_a_1298_);
v___x_1303_ = lean_box(v___x_1253_);
v___x_1304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1304_, 0, v_a_1260_);
lean_ctor_set(v___x_1304_, 1, v___x_1303_);
if (v_isShared_1301_ == 0)
{
lean_ctor_set(v___x_1300_, 0, v___x_1304_);
v___x_1306_ = v___x_1300_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v___x_1304_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
else
{
lean_object* v___x_1308_; lean_object* v_a_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1317_; 
lean_del_object(v___x_1300_);
v___x_1308_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(v_a_1260_, v___y_1255_);
v_a_1309_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1317_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1317_ == 0)
{
v___x_1311_ = v___x_1308_;
v_isShared_1312_ = v_isSharedCheck_1317_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_a_1309_);
lean_dec(v___x_1308_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1317_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1313_; lean_object* v___x_1315_; 
v___x_1313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1313_, 0, v_a_1309_);
lean_ctor_set(v___x_1313_, 1, v_a_1298_);
if (v_isShared_1312_ == 0)
{
lean_ctor_set(v___x_1311_, 0, v___x_1313_);
v___x_1315_ = v___x_1311_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v___x_1313_);
v___x_1315_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
return v___x_1315_;
}
}
}
}
}
else
{
lean_object* v_a_1319_; lean_object* v___x_1321_; uint8_t v_isShared_1322_; uint8_t v_isSharedCheck_1326_; 
lean_dec(v_a_1260_);
v_a_1319_ = lean_ctor_get(v___x_1297_, 0);
v_isSharedCheck_1326_ = !lean_is_exclusive(v___x_1297_);
if (v_isSharedCheck_1326_ == 0)
{
v___x_1321_ = v___x_1297_;
v_isShared_1322_ = v_isSharedCheck_1326_;
goto v_resetjp_1320_;
}
else
{
lean_inc(v_a_1319_);
lean_dec(v___x_1297_);
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
}
}
}
else
{
lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1336_; 
lean_dec_ref(v___y_1254_);
lean_dec_ref(v_a_1252_);
lean_dec_ref(v_00_u03b2_1251_);
lean_dec(v___x_1250_);
lean_dec_ref(v___x_1249_);
v_a_1329_ = lean_ctor_get(v___x_1259_, 0);
v_isSharedCheck_1336_ = !lean_is_exclusive(v___x_1259_);
if (v_isSharedCheck_1336_ == 0)
{
v___x_1331_ = v___x_1259_;
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1259_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1334_; 
if (v_isShared_1332_ == 0)
{
v___x_1334_ = v___x_1331_;
goto v_reusejp_1333_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v_a_1329_);
v___x_1334_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1333_;
}
v_reusejp_1333_:
{
return v___x_1334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___boxed(lean_object* v___x_1337_, lean_object* v___x_1338_, lean_object* v___x_1339_, lean_object* v___x_1340_, lean_object* v___x_1341_, lean_object* v_00_u03b2_1342_, lean_object* v_a_1343_, lean_object* v___x_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
uint8_t v___x_16693__boxed_1350_; uint8_t v___x_16697__boxed_1351_; lean_object* v_res_1352_; 
v___x_16693__boxed_1350_ = lean_unbox(v___x_1338_);
v___x_16697__boxed_1351_ = lean_unbox(v___x_1344_);
v_res_1352_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0(v___x_1337_, v___x_16693__boxed_1350_, v___x_1339_, v___x_1340_, v___x_1341_, v_00_u03b2_1342_, v_a_1343_, v___x_16697__boxed_1351_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_);
lean_dec(v___y_1348_);
lean_dec_ref(v___y_1347_);
lean_dec(v___y_1346_);
return v_res_1352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1(lean_object* v___x_1353_, uint8_t v___x_1354_, lean_object* v___x_1355_, lean_object* v___x_1356_, lean_object* v_00_u03b2_1357_, lean_object* v_a_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_){
_start:
{
lean_object* v___x_1364_; 
v___x_1364_ = l_Lean_Meta_mkFreshExprMVar(v___x_1353_, v___x_1354_, v___x_1355_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_);
if (lean_obj_tag(v___x_1364_) == 0)
{
lean_object* v_a_1365_; lean_object* v_keyedConfig_1366_; uint8_t v_trackZetaDelta_1367_; lean_object* v_zetaDeltaSet_1368_; lean_object* v_lctx_1369_; lean_object* v_localInstances_1370_; lean_object* v_defEqCtx_x3f_1371_; lean_object* v_synthPendingDepth_1372_; lean_object* v_customCanUnfoldPredicate_x3f_1373_; uint8_t v_univApprox_1374_; uint8_t v_inTypeClassResolution_1375_; uint8_t v_cacheInferType_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1426_; 
v_a_1365_ = lean_ctor_get(v___x_1364_, 0);
lean_inc(v_a_1365_);
lean_dec_ref_known(v___x_1364_, 1);
v_keyedConfig_1366_ = lean_ctor_get(v___y_1359_, 0);
v_trackZetaDelta_1367_ = lean_ctor_get_uint8(v___y_1359_, sizeof(void*)*7);
v_zetaDeltaSet_1368_ = lean_ctor_get(v___y_1359_, 1);
v_lctx_1369_ = lean_ctor_get(v___y_1359_, 2);
v_localInstances_1370_ = lean_ctor_get(v___y_1359_, 3);
v_defEqCtx_x3f_1371_ = lean_ctor_get(v___y_1359_, 4);
v_synthPendingDepth_1372_ = lean_ctor_get(v___y_1359_, 5);
v_customCanUnfoldPredicate_x3f_1373_ = lean_ctor_get(v___y_1359_, 6);
v_univApprox_1374_ = lean_ctor_get_uint8(v___y_1359_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1375_ = lean_ctor_get_uint8(v___y_1359_, sizeof(void*)*7 + 2);
v_cacheInferType_1376_ = lean_ctor_get_uint8(v___y_1359_, sizeof(void*)*7 + 3);
v_isSharedCheck_1426_ = !lean_is_exclusive(v___y_1359_);
if (v_isSharedCheck_1426_ == 0)
{
v___x_1378_ = v___y_1359_;
v_isShared_1379_ = v_isSharedCheck_1426_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1373_);
lean_inc(v_synthPendingDepth_1372_);
lean_inc(v_defEqCtx_x3f_1371_);
lean_inc(v_localInstances_1370_);
lean_inc(v_lctx_1369_);
lean_inc(v_zetaDeltaSet_1368_);
lean_inc(v_keyedConfig_1366_);
lean_dec(v___y_1359_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1426_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; uint8_t v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1395_; 
v___x_1380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1));
v___x_1381_ = lean_box(0);
lean_inc_n(v___x_1356_, 2);
v___x_1382_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1382_, 0, v___x_1381_);
lean_ctor_set(v___x_1382_, 1, v___x_1356_);
v___x_1383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__5));
v___x_1384_ = l_Lean_Expr_const___override(v___x_1383_, v___x_1356_);
v___x_1385_ = l_Lean_Expr_const___override(v___x_1380_, v___x_1382_);
v___x_1386_ = l_Lean_Expr_app___override(v___x_1385_, v_00_u03b2_1357_);
v___x_1387_ = l_Lean_Expr_app___override(v___x_1386_, v___x_1384_);
v___x_1388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__3));
v___x_1389_ = l_Lean_Expr_const___override(v___x_1388_, v___x_1356_);
lean_inc(v_a_1365_);
v___x_1390_ = l_Lean_Expr_app___override(v___x_1389_, v_a_1365_);
v___x_1391_ = l_Lean_Expr_app___override(v___x_1387_, v___x_1390_);
v___x_1392_ = 2;
v___x_1393_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1392_, v_keyedConfig_1366_);
if (v_isShared_1379_ == 0)
{
lean_ctor_set(v___x_1378_, 0, v___x_1393_);
v___x_1395_ = v___x_1378_;
goto v_reusejp_1394_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v___x_1393_);
lean_ctor_set(v_reuseFailAlloc_1425_, 1, v_zetaDeltaSet_1368_);
lean_ctor_set(v_reuseFailAlloc_1425_, 2, v_lctx_1369_);
lean_ctor_set(v_reuseFailAlloc_1425_, 3, v_localInstances_1370_);
lean_ctor_set(v_reuseFailAlloc_1425_, 4, v_defEqCtx_x3f_1371_);
lean_ctor_set(v_reuseFailAlloc_1425_, 5, v_synthPendingDepth_1372_);
lean_ctor_set(v_reuseFailAlloc_1425_, 6, v_customCanUnfoldPredicate_x3f_1373_);
lean_ctor_set_uint8(v_reuseFailAlloc_1425_, sizeof(void*)*7, v_trackZetaDelta_1367_);
lean_ctor_set_uint8(v_reuseFailAlloc_1425_, sizeof(void*)*7 + 1, v_univApprox_1374_);
lean_ctor_set_uint8(v_reuseFailAlloc_1425_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1375_);
lean_ctor_set_uint8(v_reuseFailAlloc_1425_, sizeof(void*)*7 + 3, v_cacheInferType_1376_);
v___x_1395_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1394_;
}
v_reusejp_1394_:
{
lean_object* v___x_1396_; 
v___x_1396_ = l_Lean_Meta_isExprDefEq(v___x_1391_, v_a_1358_, v___x_1395_, v___y_1360_, v___y_1361_, v___y_1362_);
lean_dec_ref(v___x_1395_);
if (lean_obj_tag(v___x_1396_) == 0)
{
lean_object* v_a_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1416_; 
v_a_1397_ = lean_ctor_get(v___x_1396_, 0);
v_isSharedCheck_1416_ = !lean_is_exclusive(v___x_1396_);
if (v_isSharedCheck_1416_ == 0)
{
v___x_1399_ = v___x_1396_;
v_isShared_1400_ = v_isSharedCheck_1416_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_a_1397_);
lean_dec(v___x_1396_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1416_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
uint8_t v___x_1401_; 
v___x_1401_ = lean_unbox(v_a_1397_);
if (v___x_1401_ == 0)
{
lean_object* v___x_1402_; lean_object* v___x_1404_; 
v___x_1402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1402_, 0, v_a_1365_);
lean_ctor_set(v___x_1402_, 1, v_a_1397_);
if (v_isShared_1400_ == 0)
{
lean_ctor_set(v___x_1399_, 0, v___x_1402_);
v___x_1404_ = v___x_1399_;
goto v_reusejp_1403_;
}
else
{
lean_object* v_reuseFailAlloc_1405_; 
v_reuseFailAlloc_1405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1405_, 0, v___x_1402_);
v___x_1404_ = v_reuseFailAlloc_1405_;
goto v_reusejp_1403_;
}
v_reusejp_1403_:
{
return v___x_1404_;
}
}
else
{
lean_object* v___x_1406_; lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1415_; 
lean_del_object(v___x_1399_);
v___x_1406_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(v_a_1365_, v___y_1360_);
v_a_1407_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1415_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1415_ == 0)
{
v___x_1409_ = v___x_1406_;
v_isShared_1410_ = v_isSharedCheck_1415_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___x_1406_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1415_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
lean_object* v___x_1411_; lean_object* v___x_1413_; 
v___x_1411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1411_, 0, v_a_1407_);
lean_ctor_set(v___x_1411_, 1, v_a_1397_);
if (v_isShared_1410_ == 0)
{
lean_ctor_set(v___x_1409_, 0, v___x_1411_);
v___x_1413_ = v___x_1409_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v___x_1411_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
}
}
}
else
{
lean_object* v_a_1417_; lean_object* v___x_1419_; uint8_t v_isShared_1420_; uint8_t v_isSharedCheck_1424_; 
lean_dec(v_a_1365_);
v_a_1417_ = lean_ctor_get(v___x_1396_, 0);
v_isSharedCheck_1424_ = !lean_is_exclusive(v___x_1396_);
if (v_isSharedCheck_1424_ == 0)
{
v___x_1419_ = v___x_1396_;
v_isShared_1420_ = v_isSharedCheck_1424_;
goto v_resetjp_1418_;
}
else
{
lean_inc(v_a_1417_);
lean_dec(v___x_1396_);
v___x_1419_ = lean_box(0);
v_isShared_1420_ = v_isSharedCheck_1424_;
goto v_resetjp_1418_;
}
v_resetjp_1418_:
{
lean_object* v___x_1422_; 
if (v_isShared_1420_ == 0)
{
v___x_1422_ = v___x_1419_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v_a_1417_);
v___x_1422_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
return v___x_1422_;
}
}
}
}
}
}
else
{
lean_object* v_a_1427_; lean_object* v___x_1429_; uint8_t v_isShared_1430_; uint8_t v_isSharedCheck_1434_; 
lean_dec_ref(v___y_1359_);
lean_dec_ref(v_a_1358_);
lean_dec_ref(v_00_u03b2_1357_);
lean_dec(v___x_1356_);
v_a_1427_ = lean_ctor_get(v___x_1364_, 0);
v_isSharedCheck_1434_ = !lean_is_exclusive(v___x_1364_);
if (v_isSharedCheck_1434_ == 0)
{
v___x_1429_ = v___x_1364_;
v_isShared_1430_ = v_isSharedCheck_1434_;
goto v_resetjp_1428_;
}
else
{
lean_inc(v_a_1427_);
lean_dec(v___x_1364_);
v___x_1429_ = lean_box(0);
v_isShared_1430_ = v_isSharedCheck_1434_;
goto v_resetjp_1428_;
}
v_resetjp_1428_:
{
lean_object* v___x_1432_; 
if (v_isShared_1430_ == 0)
{
v___x_1432_ = v___x_1429_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1433_; 
v_reuseFailAlloc_1433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1433_, 0, v_a_1427_);
v___x_1432_ = v_reuseFailAlloc_1433_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
return v___x_1432_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1___boxed(lean_object* v___x_1435_, lean_object* v___x_1436_, lean_object* v___x_1437_, lean_object* v___x_1438_, lean_object* v_00_u03b2_1439_, lean_object* v_a_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_){
_start:
{
uint8_t v___x_16886__boxed_1446_; lean_object* v_res_1447_; 
v___x_16886__boxed_1446_ = lean_unbox(v___x_1436_);
v_res_1447_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1(v___x_1435_, v___x_16886__boxed_1446_, v___x_1437_, v___x_1438_, v_00_u03b2_1439_, v_a_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_);
lean_dec(v___y_1444_);
lean_dec_ref(v___y_1443_);
lean_dec(v___y_1442_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(lean_object* v_k_1448_, uint8_t v_allowLevelAssignments_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_){
_start:
{
lean_object* v___x_1455_; 
v___x_1455_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1449_, v_k_1448_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
if (lean_obj_tag(v___x_1455_) == 0)
{
lean_object* v_a_1456_; lean_object* v___x_1458_; uint8_t v_isShared_1459_; uint8_t v_isSharedCheck_1463_; 
v_a_1456_ = lean_ctor_get(v___x_1455_, 0);
v_isSharedCheck_1463_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1463_ == 0)
{
v___x_1458_ = v___x_1455_;
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
else
{
lean_inc(v_a_1456_);
lean_dec(v___x_1455_);
v___x_1458_ = lean_box(0);
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
v_resetjp_1457_:
{
lean_object* v___x_1461_; 
if (v_isShared_1459_ == 0)
{
v___x_1461_ = v___x_1458_;
goto v_reusejp_1460_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v_a_1456_);
v___x_1461_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1460_;
}
v_reusejp_1460_:
{
return v___x_1461_;
}
}
}
else
{
lean_object* v_a_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1471_; 
v_a_1464_ = lean_ctor_get(v___x_1455_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1466_ = v___x_1455_;
v_isShared_1467_ = v_isSharedCheck_1471_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_a_1464_);
lean_dec(v___x_1455_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg___boxed(lean_object* v_k_1472_, lean_object* v_allowLevelAssignments_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1479_; lean_object* v_res_1480_; 
v_allowLevelAssignments_boxed_1479_ = lean_unbox(v_allowLevelAssignments_1473_);
v_res_1480_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v_k_1472_, v_allowLevelAssignments_boxed_1479_, v___y_1474_, v___y_1475_, v___y_1476_, v___y_1477_);
lean_dec(v___y_1477_);
lean_dec_ref(v___y_1476_);
lean_dec(v___y_1475_);
lean_dec_ref(v___y_1474_);
return v_res_1480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4(lean_object* v_msgData_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
lean_object* v___x_1487_; lean_object* v_env_1488_; lean_object* v___x_1489_; lean_object* v_mctx_1490_; lean_object* v_lctx_1491_; lean_object* v_options_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1487_ = lean_st_ref_get(v___y_1485_);
v_env_1488_ = lean_ctor_get(v___x_1487_, 0);
lean_inc_ref(v_env_1488_);
lean_dec(v___x_1487_);
v___x_1489_ = lean_st_ref_get(v___y_1483_);
v_mctx_1490_ = lean_ctor_get(v___x_1489_, 0);
lean_inc_ref(v_mctx_1490_);
lean_dec(v___x_1489_);
v_lctx_1491_ = lean_ctor_get(v___y_1482_, 2);
v_options_1492_ = lean_ctor_get(v___y_1484_, 2);
lean_inc_ref(v_options_1492_);
lean_inc_ref(v_lctx_1491_);
v___x_1493_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1493_, 0, v_env_1488_);
lean_ctor_set(v___x_1493_, 1, v_mctx_1490_);
lean_ctor_set(v___x_1493_, 2, v_lctx_1491_);
lean_ctor_set(v___x_1493_, 3, v_options_1492_);
v___x_1494_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1493_);
lean_ctor_set(v___x_1494_, 1, v_msgData_1481_);
v___x_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1495_, 0, v___x_1494_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4___boxed(lean_object* v_msgData_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_){
_start:
{
lean_object* v_res_1502_; 
v_res_1502_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4(v_msgData_1496_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_);
lean_dec(v___y_1500_);
lean_dec_ref(v___y_1499_);
lean_dec(v___y_1498_);
lean_dec_ref(v___y_1497_);
return v_res_1502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(lean_object* v_msg_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_){
_start:
{
lean_object* v_ref_1509_; lean_object* v___x_1510_; lean_object* v_a_1511_; lean_object* v___x_1513_; uint8_t v_isShared_1514_; uint8_t v_isSharedCheck_1519_; 
v_ref_1509_ = lean_ctor_get(v___y_1506_, 5);
v___x_1510_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4(v_msg_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_);
v_a_1511_ = lean_ctor_get(v___x_1510_, 0);
v_isSharedCheck_1519_ = !lean_is_exclusive(v___x_1510_);
if (v_isSharedCheck_1519_ == 0)
{
v___x_1513_ = v___x_1510_;
v_isShared_1514_ = v_isSharedCheck_1519_;
goto v_resetjp_1512_;
}
else
{
lean_inc(v_a_1511_);
lean_dec(v___x_1510_);
v___x_1513_ = lean_box(0);
v_isShared_1514_ = v_isSharedCheck_1519_;
goto v_resetjp_1512_;
}
v_resetjp_1512_:
{
lean_object* v___x_1515_; lean_object* v___x_1517_; 
lean_inc(v_ref_1509_);
v___x_1515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1515_, 0, v_ref_1509_);
lean_ctor_set(v___x_1515_, 1, v_a_1511_);
if (v_isShared_1514_ == 0)
{
lean_ctor_set_tag(v___x_1513_, 1);
lean_ctor_set(v___x_1513_, 0, v___x_1515_);
v___x_1517_ = v___x_1513_;
goto v_reusejp_1516_;
}
else
{
lean_object* v_reuseFailAlloc_1518_; 
v_reuseFailAlloc_1518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1518_, 0, v___x_1515_);
v___x_1517_ = v_reuseFailAlloc_1518_;
goto v_reusejp_1516_;
}
v_reusejp_1516_:
{
return v___x_1517_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg___boxed(lean_object* v_msg_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_){
_start:
{
lean_object* v_res_1526_; 
v_res_1526_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v_msg_1520_, v___y_1521_, v___y_1522_, v___y_1523_, v___y_1524_);
lean_dec(v___y_1524_);
lean_dec_ref(v___y_1523_);
lean_dec(v___y_1522_);
lean_dec_ref(v___y_1521_);
return v_res_1526_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0(void){
_start:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; 
v___x_1545_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_1546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1546_, 0, v___x_1545_);
return v___x_1546_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2(void){
_start:
{
lean_object* v___x_1548_; lean_object* v___x_1549_; 
v___x_1548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__1));
v___x_1549_ = l_Lean_stringToMessageData(v___x_1548_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast(lean_object* v_u_1568_, lean_object* v_00_u03b1_1569_, lean_object* v_s_u03b1_1570_, lean_object* v_v_1571_, lean_object* v_00_u03b2_1572_, lean_object* v_s_u03b2_1573_, lean_object* v_a_1574_, lean_object* v_r_u03b1_1575_, lean_object* v_va_1576_, lean_object* v_a_1577_, lean_object* v_a_1578_, lean_object* v_a_1579_, lean_object* v_a_1580_, lean_object* v_a_1581_, lean_object* v_a_1582_){
_start:
{
if (lean_obj_tag(v_va_1576_) == 0)
{
lean_object* v_value_1584_; lean_object* v___x_1586_; uint8_t v_isShared_1587_; uint8_t v_isSharedCheck_1715_; 
lean_dec_ref(v_s_u03b1_1570_);
v_value_1584_ = lean_ctor_get(v_va_1576_, 1);
v_isSharedCheck_1715_ = !lean_is_exclusive(v_va_1576_);
if (v_isSharedCheck_1715_ == 0)
{
lean_object* v_unused_1716_; 
v_unused_1716_ = lean_ctor_get(v_va_1576_, 0);
lean_dec(v_unused_1716_);
v___x_1586_ = v_va_1576_;
v_isShared_1587_ = v_isSharedCheck_1715_;
goto v_resetjp_1585_;
}
else
{
lean_inc(v_value_1584_);
lean_dec(v_va_1576_);
v___x_1586_ = lean_box(0);
v_isShared_1587_ = v_isSharedCheck_1715_;
goto v_resetjp_1585_;
}
v_resetjp_1585_:
{
lean_object* v_value_1588_; lean_object* v_hyp_1589_; lean_object* v___x_1591_; uint8_t v_isShared_1592_; uint8_t v_isSharedCheck_1714_; 
v_value_1588_ = lean_ctor_get(v_value_1584_, 0);
v_hyp_1589_ = lean_ctor_get(v_value_1584_, 1);
v_isSharedCheck_1714_ = !lean_is_exclusive(v_value_1584_);
if (v_isSharedCheck_1714_ == 0)
{
v___x_1591_ = v_value_1584_;
v_isShared_1592_ = v_isSharedCheck_1714_;
goto v_resetjp_1590_;
}
else
{
lean_inc(v_hyp_1589_);
lean_inc(v_value_1588_);
lean_dec(v_value_1584_);
v___x_1591_ = lean_box(0);
v_isShared_1592_ = v_isSharedCheck_1714_;
goto v_resetjp_1590_;
}
v_resetjp_1590_:
{
lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; uint8_t v___x_1596_; lean_object* v___x_1597_; uint8_t v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___f_1601_; lean_object* v___x_1602_; 
v___x_1593_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__0));
v___x_1594_ = lean_box(0);
v___x_1595_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__0);
v___x_1596_ = 0;
v___x_1597_ = lean_box(0);
v___x_1598_ = 0;
v___x_1599_ = lean_box(v___x_1596_);
v___x_1600_ = lean_box(v___x_1598_);
lean_inc_ref(v_a_1574_);
lean_inc_ref(v_00_u03b2_1572_);
v___f_1601_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___boxed), 13, 8);
lean_closure_set(v___f_1601_, 0, v___x_1595_);
lean_closure_set(v___f_1601_, 1, v___x_1599_);
lean_closure_set(v___f_1601_, 2, v___x_1597_);
lean_closure_set(v___f_1601_, 3, v___x_1593_);
lean_closure_set(v___f_1601_, 4, v___x_1594_);
lean_closure_set(v___f_1601_, 5, v_00_u03b2_1572_);
lean_closure_set(v___f_1601_, 6, v_a_1574_);
lean_closure_set(v___f_1601_, 7, v___x_1600_);
v___x_1602_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_1601_, v___x_1598_, v_a_1579_, v_a_1580_, v_a_1581_, v_a_1582_);
if (lean_obj_tag(v___x_1602_) == 0)
{
lean_object* v_a_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1705_; 
v_a_1603_ = lean_ctor_get(v___x_1602_, 0);
v_isSharedCheck_1705_ = !lean_is_exclusive(v___x_1602_);
if (v_isSharedCheck_1705_ == 0)
{
v___x_1605_ = v___x_1602_;
v_isShared_1606_ = v_isSharedCheck_1705_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_a_1603_);
lean_dec(v___x_1602_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1705_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v_snd_1607_; uint8_t v___x_1608_; 
v_snd_1607_ = lean_ctor_get(v_a_1603_, 1);
v___x_1608_ = lean_unbox(v_snd_1607_);
if (v___x_1608_ == 0)
{
lean_object* v___x_1609_; lean_object* v___f_1610_; lean_object* v___x_1611_; 
lean_del_object(v___x_1605_);
lean_dec(v_a_1603_);
v___x_1609_ = lean_box(v___x_1596_);
v___f_1610_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__1___boxed), 11, 6);
lean_closure_set(v___f_1610_, 0, v___x_1595_);
lean_closure_set(v___f_1610_, 1, v___x_1609_);
lean_closure_set(v___f_1610_, 2, v___x_1597_);
lean_closure_set(v___f_1610_, 3, v___x_1594_);
lean_closure_set(v___f_1610_, 4, v_00_u03b2_1572_);
lean_closure_set(v___f_1610_, 5, v_a_1574_);
v___x_1611_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_1610_, v___x_1598_, v_a_1579_, v_a_1580_, v_a_1581_, v_a_1582_);
if (lean_obj_tag(v___x_1611_) == 0)
{
lean_object* v_a_1612_; lean_object* v___x_1614_; uint8_t v_isShared_1615_; uint8_t v_isSharedCheck_1655_; 
v_a_1612_ = lean_ctor_get(v___x_1611_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1611_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1614_ = v___x_1611_;
v_isShared_1615_ = v_isSharedCheck_1655_;
goto v_resetjp_1613_;
}
else
{
lean_inc(v_a_1612_);
lean_dec(v___x_1611_);
v___x_1614_ = lean_box(0);
v_isShared_1615_ = v_isSharedCheck_1655_;
goto v_resetjp_1613_;
}
v_resetjp_1613_:
{
lean_object* v_snd_1616_; uint8_t v___x_1617_; 
v_snd_1616_ = lean_ctor_get(v_a_1612_, 1);
v___x_1617_ = lean_unbox(v_snd_1616_);
if (v___x_1617_ == 0)
{
lean_object* v___x_1618_; lean_object* v___x_1619_; 
lean_del_object(v___x_1614_);
lean_dec(v_a_1612_);
lean_del_object(v___x_1591_);
lean_dec(v_hyp_1589_);
lean_dec_ref(v_value_1588_);
lean_del_object(v___x_1586_);
lean_dec_ref(v_r_u03b1_1575_);
lean_dec_ref(v_00_u03b1_1569_);
lean_dec(v_u_1568_);
v___x_1618_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__2);
v___x_1619_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v___x_1618_, v_a_1579_, v_a_1580_, v_a_1581_, v_a_1582_);
return v___x_1619_;
}
else
{
lean_object* v_fst_1620_; lean_object* v___x_1622_; uint8_t v_isShared_1623_; uint8_t v_isSharedCheck_1653_; 
v_fst_1620_ = lean_ctor_get(v_a_1612_, 0);
v_isSharedCheck_1653_ = !lean_is_exclusive(v_a_1612_);
if (v_isSharedCheck_1653_ == 0)
{
lean_object* v_unused_1654_; 
v_unused_1654_ = lean_ctor_get(v_a_1612_, 1);
lean_dec(v_unused_1654_);
v___x_1622_ = v_a_1612_;
v_isShared_1623_ = v_isSharedCheck_1653_;
goto v_resetjp_1621_;
}
else
{
lean_inc(v_fst_1620_);
lean_dec(v_a_1612_);
v___x_1622_ = lean_box(0);
v_isShared_1623_ = v_isSharedCheck_1653_;
goto v_resetjp_1621_;
}
v_resetjp_1621_:
{
lean_object* v___x_1625_; 
if (v_isShared_1623_ == 0)
{
lean_ctor_set_tag(v___x_1622_, 1);
lean_ctor_set(v___x_1622_, 1, v___x_1594_);
lean_ctor_set(v___x_1622_, 0, v_u_1568_);
v___x_1625_ = v___x_1622_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1652_; 
v_reuseFailAlloc_1652_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1652_, 0, v_u_1568_);
lean_ctor_set(v_reuseFailAlloc_1652_, 1, v___x_1594_);
v___x_1625_ = v_reuseFailAlloc_1652_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1638_; 
v___x_1626_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5));
lean_inc_ref_n(v___x_1625_, 2);
v___x_1627_ = l_Lean_Expr_const___override(v___x_1626_, v___x_1625_);
lean_inc_ref_n(v_00_u03b1_1569_, 2);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1627_, v_00_u03b1_1569_);
lean_inc_ref(v_r_u03b1_1575_);
v___x_1629_ = l_Lean_Expr_app___override(v___x_1628_, v_r_u03b1_1575_);
v___x_1630_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__1));
v___x_1631_ = l_Lean_Expr_const___override(v___x_1630_, v___x_1625_);
v___x_1632_ = l_Lean_Expr_app___override(v___x_1631_, v_00_u03b1_1569_);
v___x_1633_ = l_Lean_Expr_app___override(v___x_1632_, v___x_1629_);
v___x_1634_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNegNat___redArg___closed__4);
lean_inc(v_fst_1620_);
v___x_1635_ = l_Lean_Expr_app___override(v___x_1634_, v_fst_1620_);
v___x_1636_ = l_Lean_Expr_app___override(v___x_1633_, v___x_1635_);
if (v_isShared_1592_ == 0)
{
v___x_1638_ = v___x_1591_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1651_; 
v_reuseFailAlloc_1651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1651_, 0, v_value_1588_);
lean_ctor_set(v_reuseFailAlloc_1651_, 1, v_hyp_1589_);
v___x_1638_ = v_reuseFailAlloc_1651_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
lean_object* v___x_1640_; 
lean_inc_ref(v___x_1636_);
if (v_isShared_1587_ == 0)
{
lean_ctor_set(v___x_1586_, 1, v___x_1638_);
lean_ctor_set(v___x_1586_, 0, v___x_1636_);
v___x_1640_ = v___x_1586_;
goto v_reusejp_1639_;
}
else
{
lean_object* v_reuseFailAlloc_1650_; 
v_reuseFailAlloc_1650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1650_, 0, v___x_1636_);
lean_ctor_set(v_reuseFailAlloc_1650_, 1, v___x_1638_);
v___x_1640_ = v_reuseFailAlloc_1650_;
goto v_reusejp_1639_;
}
v_reusejp_1639_:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1648_; 
v___x_1641_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__4));
v___x_1642_ = l_Lean_Expr_const___override(v___x_1641_, v___x_1625_);
v___x_1643_ = l_Lean_Expr_app___override(v___x_1642_, v_00_u03b1_1569_);
v___x_1644_ = l_Lean_Expr_app___override(v___x_1643_, v_r_u03b1_1575_);
v___x_1645_ = l_Lean_Expr_app___override(v___x_1644_, v_fst_1620_);
v___x_1646_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1636_);
lean_ctor_set(v___x_1646_, 1, v___x_1640_);
lean_ctor_set(v___x_1646_, 2, v___x_1645_);
if (v_isShared_1615_ == 0)
{
lean_ctor_set(v___x_1614_, 0, v___x_1646_);
v___x_1648_ = v___x_1614_;
goto v_reusejp_1647_;
}
else
{
lean_object* v_reuseFailAlloc_1649_; 
v_reuseFailAlloc_1649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1649_, 0, v___x_1646_);
v___x_1648_ = v_reuseFailAlloc_1649_;
goto v_reusejp_1647_;
}
v_reusejp_1647_:
{
return v___x_1648_;
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
lean_object* v_a_1656_; lean_object* v___x_1658_; uint8_t v_isShared_1659_; uint8_t v_isSharedCheck_1663_; 
lean_del_object(v___x_1591_);
lean_dec(v_hyp_1589_);
lean_dec_ref(v_value_1588_);
lean_del_object(v___x_1586_);
lean_dec_ref(v_r_u03b1_1575_);
lean_dec_ref(v_00_u03b1_1569_);
lean_dec(v_u_1568_);
v_a_1656_ = lean_ctor_get(v___x_1611_, 0);
v_isSharedCheck_1663_ = !lean_is_exclusive(v___x_1611_);
if (v_isSharedCheck_1663_ == 0)
{
v___x_1658_ = v___x_1611_;
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
else
{
lean_inc(v_a_1656_);
lean_dec(v___x_1611_);
v___x_1658_ = lean_box(0);
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
v_resetjp_1657_:
{
lean_object* v___x_1661_; 
if (v_isShared_1659_ == 0)
{
v___x_1661_ = v___x_1658_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1662_; 
v_reuseFailAlloc_1662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1662_, 0, v_a_1656_);
v___x_1661_ = v_reuseFailAlloc_1662_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
return v___x_1661_;
}
}
}
}
else
{
lean_object* v_fst_1664_; lean_object* v___x_1666_; uint8_t v_isShared_1667_; uint8_t v_isSharedCheck_1703_; 
lean_dec_ref(v_a_1574_);
lean_dec_ref(v_00_u03b2_1572_);
v_fst_1664_ = lean_ctor_get(v_a_1603_, 0);
v_isSharedCheck_1703_ = !lean_is_exclusive(v_a_1603_);
if (v_isSharedCheck_1703_ == 0)
{
lean_object* v_unused_1704_; 
v_unused_1704_ = lean_ctor_get(v_a_1603_, 1);
lean_dec(v_unused_1704_);
v___x_1666_ = v_a_1603_;
v_isShared_1667_ = v_isSharedCheck_1703_;
goto v_resetjp_1665_;
}
else
{
lean_inc(v_fst_1664_);
lean_dec(v_a_1603_);
v___x_1666_ = lean_box(0);
v_isShared_1667_ = v_isSharedCheck_1703_;
goto v_resetjp_1665_;
}
v_resetjp_1665_:
{
lean_object* v___x_1669_; 
if (v_isShared_1667_ == 0)
{
lean_ctor_set_tag(v___x_1666_, 1);
lean_ctor_set(v___x_1666_, 1, v___x_1594_);
lean_ctor_set(v___x_1666_, 0, v_u_1568_);
v___x_1669_ = v___x_1666_;
goto v_reusejp_1668_;
}
else
{
lean_object* v_reuseFailAlloc_1702_; 
v_reuseFailAlloc_1702_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1702_, 0, v_u_1568_);
lean_ctor_set(v_reuseFailAlloc_1702_, 1, v___x_1594_);
v___x_1669_ = v_reuseFailAlloc_1702_;
goto v_reusejp_1668_;
}
v_reusejp_1668_:
{
lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1688_; 
v___x_1670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3));
lean_inc_ref_n(v___x_1669_, 4);
v___x_1671_ = l_Lean_Expr_const___override(v___x_1670_, v___x_1669_);
lean_inc_ref_n(v_00_u03b1_1569_, 4);
v___x_1672_ = l_Lean_Expr_app___override(v___x_1671_, v_00_u03b1_1569_);
v___x_1673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5));
v___x_1674_ = l_Lean_Expr_const___override(v___x_1673_, v___x_1669_);
v___x_1675_ = l_Lean_Expr_app___override(v___x_1674_, v_00_u03b1_1569_);
lean_inc_ref(v_r_u03b1_1575_);
v___x_1676_ = l_Lean_Expr_app___override(v___x_1675_, v_r_u03b1_1575_);
v___x_1677_ = l_Lean_Expr_app___override(v___x_1672_, v___x_1676_);
v___x_1678_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2));
v___x_1679_ = l_Lean_Expr_const___override(v___x_1678_, v___x_1669_);
v___x_1680_ = l_Lean_Expr_app___override(v___x_1679_, v_00_u03b1_1569_);
v___x_1681_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__1));
v___x_1682_ = l_Lean_Expr_const___override(v___x_1681_, v___x_1669_);
v___x_1683_ = l_Lean_Expr_app___override(v___x_1682_, v_00_u03b1_1569_);
v___x_1684_ = l_Lean_Expr_app___override(v___x_1683_, v___x_1677_);
v___x_1685_ = l_Lean_Expr_app___override(v___x_1680_, v___x_1684_);
lean_inc(v_fst_1664_);
v___x_1686_ = l_Lean_Expr_app___override(v___x_1685_, v_fst_1664_);
if (v_isShared_1592_ == 0)
{
v___x_1688_ = v___x_1591_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1701_; 
v_reuseFailAlloc_1701_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1701_, 0, v_value_1588_);
lean_ctor_set(v_reuseFailAlloc_1701_, 1, v_hyp_1589_);
v___x_1688_ = v_reuseFailAlloc_1701_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
lean_object* v___x_1690_; 
lean_inc_ref(v___x_1686_);
if (v_isShared_1587_ == 0)
{
lean_ctor_set(v___x_1586_, 1, v___x_1688_);
lean_ctor_set(v___x_1586_, 0, v___x_1686_);
v___x_1690_ = v___x_1586_;
goto v_reusejp_1689_;
}
else
{
lean_object* v_reuseFailAlloc_1700_; 
v_reuseFailAlloc_1700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1700_, 0, v___x_1686_);
lean_ctor_set(v_reuseFailAlloc_1700_, 1, v___x_1688_);
v___x_1690_ = v_reuseFailAlloc_1700_;
goto v_reusejp_1689_;
}
v_reusejp_1689_:
{
lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1698_; 
v___x_1691_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__6));
v___x_1692_ = l_Lean_Expr_const___override(v___x_1691_, v___x_1669_);
v___x_1693_ = l_Lean_Expr_app___override(v___x_1692_, v_00_u03b1_1569_);
v___x_1694_ = l_Lean_Expr_app___override(v___x_1693_, v_r_u03b1_1575_);
v___x_1695_ = l_Lean_Expr_app___override(v___x_1694_, v_fst_1664_);
v___x_1696_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1696_, 0, v___x_1686_);
lean_ctor_set(v___x_1696_, 1, v___x_1690_);
lean_ctor_set(v___x_1696_, 2, v___x_1695_);
if (v_isShared_1606_ == 0)
{
lean_ctor_set(v___x_1605_, 0, v___x_1696_);
v___x_1698_ = v___x_1605_;
goto v_reusejp_1697_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v___x_1696_);
v___x_1698_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1697_;
}
v_reusejp_1697_:
{
return v___x_1698_;
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
lean_object* v_a_1706_; lean_object* v___x_1708_; uint8_t v_isShared_1709_; uint8_t v_isSharedCheck_1713_; 
lean_del_object(v___x_1591_);
lean_dec(v_hyp_1589_);
lean_dec_ref(v_value_1588_);
lean_del_object(v___x_1586_);
lean_dec_ref(v_r_u03b1_1575_);
lean_dec_ref(v_a_1574_);
lean_dec_ref(v_00_u03b2_1572_);
lean_dec_ref(v_00_u03b1_1569_);
lean_dec(v_u_1568_);
v_a_1706_ = lean_ctor_get(v___x_1602_, 0);
v_isSharedCheck_1713_ = !lean_is_exclusive(v___x_1602_);
if (v_isSharedCheck_1713_ == 0)
{
v___x_1708_ = v___x_1602_;
v_isShared_1709_ = v_isSharedCheck_1713_;
goto v_resetjp_1707_;
}
else
{
lean_inc(v_a_1706_);
lean_dec(v___x_1602_);
v___x_1708_ = lean_box(0);
v_isShared_1709_ = v_isSharedCheck_1713_;
goto v_resetjp_1707_;
}
v_resetjp_1707_:
{
lean_object* v___x_1711_; 
if (v_isShared_1709_ == 0)
{
v___x_1711_ = v___x_1708_;
goto v_reusejp_1710_;
}
else
{
lean_object* v_reuseFailAlloc_1712_; 
v_reuseFailAlloc_1712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1712_, 0, v_a_1706_);
v___x_1711_ = v_reuseFailAlloc_1712_;
goto v_reusejp_1710_;
}
v_reusejp_1710_:
{
return v___x_1711_;
}
}
}
}
}
}
else
{
lean_object* v_x_1717_; lean_object* v_e_1718_; lean_object* v_b_1719_; lean_object* v_a_1720_; lean_object* v_a_1721_; lean_object* v_a_1722_; lean_object* v___x_1724_; uint8_t v_isShared_1725_; uint8_t v_isSharedCheck_1831_; 
lean_dec_ref(v_a_1574_);
v_x_1717_ = lean_ctor_get(v_va_1576_, 0);
v_e_1718_ = lean_ctor_get(v_va_1576_, 1);
v_b_1719_ = lean_ctor_get(v_va_1576_, 2);
v_a_1720_ = lean_ctor_get(v_va_1576_, 3);
v_a_1721_ = lean_ctor_get(v_va_1576_, 4);
v_a_1722_ = lean_ctor_get(v_va_1576_, 5);
v_isSharedCheck_1831_ = !lean_is_exclusive(v_va_1576_);
if (v_isSharedCheck_1831_ == 0)
{
v___x_1724_ = v_va_1576_;
v_isShared_1725_ = v_isSharedCheck_1831_;
goto v_resetjp_1723_;
}
else
{
lean_inc(v_a_1722_);
lean_inc(v_a_1721_);
lean_inc(v_a_1720_);
lean_inc(v_b_1719_);
lean_inc(v_e_1718_);
lean_inc(v_x_1717_);
lean_dec(v_va_1576_);
v___x_1724_ = lean_box(0);
v_isShared_1725_ = v_isSharedCheck_1831_;
goto v_resetjp_1723_;
}
v_resetjp_1723_:
{
lean_object* v___x_1726_; 
lean_inc_ref(v_r_u03b1_1575_);
lean_inc_ref(v_x_1717_);
lean_inc_ref(v_00_u03b2_1572_);
lean_inc_ref(v_s_u03b1_1570_);
lean_inc_ref(v_00_u03b1_1569_);
lean_inc(v_u_1568_);
v___x_1726_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast(v_u_1568_, v_00_u03b1_1569_, v_s_u03b1_1570_, v_v_1571_, v_00_u03b2_1572_, v_s_u03b2_1573_, v_x_1717_, v_r_u03b1_1575_, v_a_1720_, v_a_1577_, v_a_1578_, v_a_1579_, v_a_1580_, v_a_1581_, v_a_1582_);
if (lean_obj_tag(v___x_1726_) == 0)
{
lean_object* v_a_1727_; lean_object* v_expr_1728_; lean_object* v_val_1729_; lean_object* v_proof_1730_; lean_object* v___x_1731_; 
v_a_1727_ = lean_ctor_get(v___x_1726_, 0);
lean_inc(v_a_1727_);
lean_dec_ref_known(v___x_1726_, 1);
v_expr_1728_ = lean_ctor_get(v_a_1727_, 0);
lean_inc_ref(v_expr_1728_);
v_val_1729_ = lean_ctor_get(v_a_1727_, 1);
lean_inc(v_val_1729_);
v_proof_1730_ = lean_ctor_get(v_a_1727_, 2);
lean_inc_ref(v_proof_1730_);
lean_dec(v_a_1727_);
lean_inc_ref(v_r_u03b1_1575_);
lean_inc_ref(v_b_1719_);
lean_inc_ref(v_s_u03b1_1570_);
lean_inc_ref(v_00_u03b1_1569_);
lean_inc(v_u_1568_);
v___x_1731_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast(v_u_1568_, v_00_u03b1_1569_, v_s_u03b1_1570_, v_v_1571_, v_00_u03b2_1572_, v_s_u03b2_1573_, v_b_1719_, v_r_u03b1_1575_, v_a_1722_, v_a_1577_, v_a_1578_, v_a_1579_, v_a_1580_, v_a_1581_, v_a_1582_);
if (lean_obj_tag(v___x_1731_) == 0)
{
lean_object* v_a_1732_; lean_object* v___x_1734_; uint8_t v_isShared_1735_; uint8_t v_isSharedCheck_1822_; 
v_a_1732_ = lean_ctor_get(v___x_1731_, 0);
v_isSharedCheck_1822_ = !lean_is_exclusive(v___x_1731_);
if (v_isSharedCheck_1822_ == 0)
{
v___x_1734_ = v___x_1731_;
v_isShared_1735_ = v_isSharedCheck_1822_;
goto v_resetjp_1733_;
}
else
{
lean_inc(v_a_1732_);
lean_dec(v___x_1731_);
v___x_1734_ = lean_box(0);
v_isShared_1735_ = v_isSharedCheck_1822_;
goto v_resetjp_1733_;
}
v_resetjp_1733_:
{
lean_object* v_expr_1736_; lean_object* v_val_1737_; lean_object* v_proof_1738_; lean_object* v___x_1740_; uint8_t v_isShared_1741_; uint8_t v_isSharedCheck_1821_; 
v_expr_1736_ = lean_ctor_get(v_a_1732_, 0);
v_val_1737_ = lean_ctor_get(v_a_1732_, 1);
v_proof_1738_ = lean_ctor_get(v_a_1732_, 2);
v_isSharedCheck_1821_ = !lean_is_exclusive(v_a_1732_);
if (v_isSharedCheck_1821_ == 0)
{
v___x_1740_ = v_a_1732_;
v_isShared_1741_ = v_isSharedCheck_1821_;
goto v_resetjp_1739_;
}
else
{
lean_inc(v_proof_1738_);
lean_inc(v_val_1737_);
lean_inc(v_expr_1736_);
lean_dec(v_a_1732_);
v___x_1740_ = lean_box(0);
v_isShared_1741_ = v_isSharedCheck_1821_;
goto v_resetjp_1739_;
}
v_resetjp_1739_:
{
lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1802_; 
v___x_1742_ = lean_box(0);
lean_inc_n(v_u_1568_, 4);
v___x_1743_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1743_, 0, v_u_1568_);
lean_ctor_set(v___x_1743_, 1, v___x_1742_);
v___x_1744_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4));
v___x_1745_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6));
v___x_1746_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8));
v___x_1747_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_1748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_1749_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11));
v___x_1750_ = lean_box(0);
v___x_1751_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_1752_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15));
v___x_1753_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_1754_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19));
v___x_1755_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22));
v___x_1756_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24));
lean_inc_ref_n(v___x_1743_, 9);
v___x_1757_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1757_, 0, v_u_1568_);
lean_ctor_set(v___x_1757_, 1, v___x_1743_);
v___x_1758_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1758_, 0, v_u_1568_);
lean_ctor_set(v___x_1758_, 1, v___x_1757_);
v___x_1759_ = l_Lean_Expr_const___override(v___x_1744_, v___x_1758_);
lean_inc_ref_n(v_00_u03b1_1569_, 13);
v___x_1760_ = l_Lean_Expr_app___override(v___x_1759_, v_00_u03b1_1569_);
v___x_1761_ = l_Lean_Expr_app___override(v___x_1760_, v_00_u03b1_1569_);
v___x_1762_ = l_Lean_Expr_app___override(v___x_1761_, v_00_u03b1_1569_);
v___x_1763_ = l_Lean_Expr_const___override(v___x_1745_, v___x_1743_);
v___x_1764_ = l_Lean_Expr_app___override(v___x_1763_, v_00_u03b1_1569_);
v___x_1765_ = l_Lean_Expr_const___override(v___x_1746_, v___x_1743_);
v___x_1766_ = l_Lean_Expr_app___override(v___x_1765_, v_00_u03b1_1569_);
v___x_1767_ = l_Lean_Expr_const___override(v___x_1747_, v___x_1743_);
v___x_1768_ = l_Lean_Expr_app___override(v___x_1767_, v_00_u03b1_1569_);
v___x_1769_ = l_Lean_Expr_const___override(v___x_1748_, v___x_1743_);
v___x_1770_ = l_Lean_Expr_app___override(v___x_1769_, v_00_u03b1_1569_);
v___x_1771_ = l_Lean_Expr_app___override(v___x_1770_, v_s_u03b1_1570_);
lean_inc_ref(v___x_1771_);
v___x_1772_ = l_Lean_Expr_app___override(v___x_1768_, v___x_1771_);
v___x_1773_ = l_Lean_Expr_app___override(v___x_1766_, v___x_1772_);
v___x_1774_ = l_Lean_Expr_app___override(v___x_1764_, v___x_1773_);
v___x_1775_ = l_Lean_Expr_app___override(v___x_1762_, v___x_1774_);
v___x_1776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1776_, 0, v___x_1750_);
lean_ctor_set(v___x_1776_, 1, v___x_1743_);
v___x_1777_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1777_, 0, v_u_1568_);
lean_ctor_set(v___x_1777_, 1, v___x_1776_);
v___x_1778_ = l_Lean_Expr_const___override(v___x_1749_, v___x_1777_);
v___x_1779_ = l_Lean_Expr_app___override(v___x_1778_, v_00_u03b1_1569_);
v___x_1780_ = l_Lean_Expr_app___override(v___x_1779_, v___x_1751_);
v___x_1781_ = l_Lean_Expr_app___override(v___x_1780_, v_00_u03b1_1569_);
v___x_1782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1782_, 0, v_u_1568_);
lean_ctor_set(v___x_1782_, 1, v___x_1753_);
v___x_1783_ = l_Lean_Expr_const___override(v___x_1752_, v___x_1782_);
v___x_1784_ = l_Lean_Expr_app___override(v___x_1783_, v_00_u03b1_1569_);
v___x_1785_ = l_Lean_Expr_app___override(v___x_1784_, v___x_1751_);
v___x_1786_ = l_Lean_Expr_const___override(v___x_1754_, v___x_1743_);
v___x_1787_ = l_Lean_Expr_app___override(v___x_1786_, v_00_u03b1_1569_);
v___x_1788_ = l_Lean_Expr_const___override(v___x_1755_, v___x_1743_);
v___x_1789_ = l_Lean_Expr_app___override(v___x_1788_, v_00_u03b1_1569_);
v___x_1790_ = l_Lean_Expr_const___override(v___x_1756_, v___x_1743_);
v___x_1791_ = l_Lean_Expr_app___override(v___x_1790_, v_00_u03b1_1569_);
v___x_1792_ = l_Lean_Expr_app___override(v___x_1791_, v___x_1771_);
v___x_1793_ = l_Lean_Expr_app___override(v___x_1789_, v___x_1792_);
v___x_1794_ = l_Lean_Expr_app___override(v___x_1787_, v___x_1793_);
v___x_1795_ = l_Lean_Expr_app___override(v___x_1785_, v___x_1794_);
v___x_1796_ = l_Lean_Expr_app___override(v___x_1781_, v___x_1795_);
lean_inc_ref_n(v_expr_1728_, 2);
v___x_1797_ = l_Lean_Expr_app___override(v___x_1796_, v_expr_1728_);
lean_inc_ref_n(v_e_1718_, 2);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1797_, v_e_1718_);
v___x_1799_ = l_Lean_Expr_app___override(v___x_1775_, v___x_1798_);
lean_inc_ref_n(v_expr_1736_, 2);
v___x_1800_ = l_Lean_Expr_app___override(v___x_1799_, v_expr_1736_);
if (v_isShared_1725_ == 0)
{
lean_ctor_set(v___x_1724_, 5, v_val_1737_);
lean_ctor_set(v___x_1724_, 3, v_val_1729_);
lean_ctor_set(v___x_1724_, 2, v_expr_1736_);
lean_ctor_set(v___x_1724_, 0, v_expr_1728_);
v___x_1802_ = v___x_1724_;
goto v_reusejp_1801_;
}
else
{
lean_object* v_reuseFailAlloc_1820_; 
v_reuseFailAlloc_1820_ = lean_alloc_ctor(1, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1820_, 0, v_expr_1728_);
lean_ctor_set(v_reuseFailAlloc_1820_, 1, v_e_1718_);
lean_ctor_set(v_reuseFailAlloc_1820_, 2, v_expr_1736_);
lean_ctor_set(v_reuseFailAlloc_1820_, 3, v_val_1729_);
lean_ctor_set(v_reuseFailAlloc_1820_, 4, v_a_1721_);
lean_ctor_set(v_reuseFailAlloc_1820_, 5, v_val_1737_);
v___x_1802_ = v_reuseFailAlloc_1820_;
goto v_reusejp_1801_;
}
v_reusejp_1801_:
{
lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1815_; 
v___x_1803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___closed__8));
v___x_1804_ = l_Lean_Expr_const___override(v___x_1803_, v___x_1743_);
v___x_1805_ = l_Lean_Expr_app___override(v___x_1804_, v_00_u03b1_1569_);
v___x_1806_ = l_Lean_Expr_app___override(v___x_1805_, v_r_u03b1_1575_);
v___x_1807_ = l_Lean_Expr_app___override(v___x_1806_, v_expr_1728_);
v___x_1808_ = l_Lean_Expr_app___override(v___x_1807_, v_expr_1736_);
v___x_1809_ = l_Lean_Expr_app___override(v___x_1808_, v_x_1717_);
v___x_1810_ = l_Lean_Expr_app___override(v___x_1809_, v_b_1719_);
v___x_1811_ = l_Lean_Expr_app___override(v___x_1810_, v_e_1718_);
v___x_1812_ = l_Lean_Expr_app___override(v___x_1811_, v_proof_1730_);
v___x_1813_ = l_Lean_Expr_app___override(v___x_1812_, v_proof_1738_);
if (v_isShared_1741_ == 0)
{
lean_ctor_set(v___x_1740_, 2, v___x_1813_);
lean_ctor_set(v___x_1740_, 1, v___x_1802_);
lean_ctor_set(v___x_1740_, 0, v___x_1800_);
v___x_1815_ = v___x_1740_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v___x_1800_);
lean_ctor_set(v_reuseFailAlloc_1819_, 1, v___x_1802_);
lean_ctor_set(v_reuseFailAlloc_1819_, 2, v___x_1813_);
v___x_1815_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
lean_object* v___x_1817_; 
if (v_isShared_1735_ == 0)
{
lean_ctor_set(v___x_1734_, 0, v___x_1815_);
v___x_1817_ = v___x_1734_;
goto v_reusejp_1816_;
}
else
{
lean_object* v_reuseFailAlloc_1818_; 
v_reuseFailAlloc_1818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1818_, 0, v___x_1815_);
v___x_1817_ = v_reuseFailAlloc_1818_;
goto v_reusejp_1816_;
}
v_reusejp_1816_:
{
return v___x_1817_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_proof_1730_);
lean_dec(v_val_1729_);
lean_dec_ref(v_expr_1728_);
lean_del_object(v___x_1724_);
lean_dec_ref(v_a_1721_);
lean_dec_ref(v_b_1719_);
lean_dec_ref(v_e_1718_);
lean_dec_ref(v_x_1717_);
lean_dec_ref(v_r_u03b1_1575_);
lean_dec_ref(v_s_u03b1_1570_);
lean_dec_ref(v_00_u03b1_1569_);
lean_dec(v_u_1568_);
return v___x_1731_;
}
}
else
{
lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_1830_; 
lean_del_object(v___x_1724_);
lean_dec_ref(v_a_1722_);
lean_dec_ref(v_a_1721_);
lean_dec_ref(v_b_1719_);
lean_dec_ref(v_e_1718_);
lean_dec_ref(v_x_1717_);
lean_dec_ref(v_r_u03b1_1575_);
lean_dec_ref(v_00_u03b2_1572_);
lean_dec_ref(v_s_u03b1_1570_);
lean_dec_ref(v_00_u03b1_1569_);
lean_dec(v_u_1568_);
v_a_1823_ = lean_ctor_get(v___x_1726_, 0);
v_isSharedCheck_1830_ = !lean_is_exclusive(v___x_1726_);
if (v_isSharedCheck_1830_ == 0)
{
v___x_1825_ = v___x_1726_;
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1726_);
v___x_1825_ = lean_box(0);
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
v_resetjp_1824_:
{
lean_object* v___x_1828_; 
if (v_isShared_1826_ == 0)
{
v___x_1828_ = v___x_1825_;
goto v_reusejp_1827_;
}
else
{
lean_object* v_reuseFailAlloc_1829_; 
v_reuseFailAlloc_1829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1829_, 0, v_a_1823_);
v___x_1828_ = v_reuseFailAlloc_1829_;
goto v_reusejp_1827_;
}
v_reusejp_1827_:
{
return v___x_1828_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(lean_object* v_u_1838_, lean_object* v_00_u03b1_1839_, lean_object* v_s_u03b1_1840_, lean_object* v_v_1841_, lean_object* v_00_u03b2_1842_, lean_object* v_s_u03b2_1843_, lean_object* v_r_u03b1_1844_, lean_object* v_va_1845_, lean_object* v_a_1846_, lean_object* v_a_1847_, lean_object* v_a_1848_, lean_object* v_a_1849_, lean_object* v_a_1850_, lean_object* v_a_1851_){
_start:
{
if (lean_obj_tag(v_va_1845_) == 0)
{
lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; 
lean_dec_ref(v_00_u03b2_1842_);
v___x_1853_ = lean_box(0);
v___x_1854_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1854_, 0, v_u_1838_);
lean_ctor_set(v___x_1854_, 1, v___x_1853_);
v___x_1855_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_1856_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_1857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_1858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_1859_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_1860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref_n(v___x_1854_, 5);
v___x_1861_ = l_Lean_Expr_const___override(v___x_1855_, v___x_1854_);
lean_inc_ref_n(v_00_u03b1_1839_, 5);
v___x_1862_ = l_Lean_Expr_app___override(v___x_1861_, v_00_u03b1_1839_);
v___x_1863_ = l_Lean_Expr_app___override(v___x_1862_, v___x_1856_);
v___x_1864_ = l_Lean_Expr_const___override(v___x_1857_, v___x_1854_);
v___x_1865_ = l_Lean_Expr_app___override(v___x_1864_, v_00_u03b1_1839_);
v___x_1866_ = l_Lean_Expr_const___override(v___x_1858_, v___x_1854_);
v___x_1867_ = l_Lean_Expr_app___override(v___x_1866_, v_00_u03b1_1839_);
v___x_1868_ = l_Lean_Expr_const___override(v___x_1859_, v___x_1854_);
v___x_1869_ = l_Lean_Expr_app___override(v___x_1868_, v_00_u03b1_1839_);
v___x_1870_ = l_Lean_Expr_const___override(v___x_1860_, v___x_1854_);
v___x_1871_ = l_Lean_Expr_app___override(v___x_1870_, v_00_u03b1_1839_);
v___x_1872_ = l_Lean_Expr_app___override(v___x_1871_, v_s_u03b1_1840_);
v___x_1873_ = l_Lean_Expr_app___override(v___x_1869_, v___x_1872_);
v___x_1874_ = l_Lean_Expr_app___override(v___x_1867_, v___x_1873_);
v___x_1875_ = l_Lean_Expr_app___override(v___x_1865_, v___x_1874_);
v___x_1876_ = l_Lean_Expr_app___override(v___x_1863_, v___x_1875_);
v___x_1877_ = lean_box(0);
v___x_1878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__1));
v___x_1879_ = l_Lean_Expr_const___override(v___x_1878_, v___x_1854_);
v___x_1880_ = l_Lean_Expr_app___override(v___x_1879_, v_00_u03b1_1839_);
v___x_1881_ = l_Lean_Expr_app___override(v___x_1880_, v_r_u03b1_1844_);
v___x_1882_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1882_, 0, v___x_1876_);
lean_ctor_set(v___x_1882_, 1, v___x_1877_);
lean_ctor_set(v___x_1882_, 2, v___x_1881_);
v___x_1883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1883_, 0, v___x_1882_);
return v___x_1883_;
}
else
{
lean_object* v_a_1884_; lean_object* v_b_1885_; lean_object* v_a_1886_; lean_object* v_a_1887_; lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1964_; 
v_a_1884_ = lean_ctor_get(v_va_1845_, 0);
v_b_1885_ = lean_ctor_get(v_va_1845_, 1);
v_a_1886_ = lean_ctor_get(v_va_1845_, 2);
v_a_1887_ = lean_ctor_get(v_va_1845_, 3);
v_isSharedCheck_1964_ = !lean_is_exclusive(v_va_1845_);
if (v_isSharedCheck_1964_ == 0)
{
v___x_1889_ = v_va_1845_;
v_isShared_1890_ = v_isSharedCheck_1964_;
goto v_resetjp_1888_;
}
else
{
lean_inc(v_a_1887_);
lean_inc(v_a_1886_);
lean_inc(v_b_1885_);
lean_inc(v_a_1884_);
lean_dec(v_va_1845_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1964_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1891_; 
lean_inc_ref(v_r_u03b1_1844_);
lean_inc_ref(v_a_1884_);
lean_inc_ref(v_00_u03b2_1842_);
lean_inc_ref(v_s_u03b1_1840_);
lean_inc_ref(v_00_u03b1_1839_);
lean_inc(v_u_1838_);
v___x_1891_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast(v_u_1838_, v_00_u03b1_1839_, v_s_u03b1_1840_, v_v_1841_, v_00_u03b2_1842_, v_s_u03b2_1843_, v_a_1884_, v_r_u03b1_1844_, v_a_1886_, v_a_1846_, v_a_1847_, v_a_1848_, v_a_1849_, v_a_1850_, v_a_1851_);
if (lean_obj_tag(v___x_1891_) == 0)
{
lean_object* v_a_1892_; lean_object* v_expr_1893_; lean_object* v_val_1894_; lean_object* v_proof_1895_; lean_object* v___x_1896_; 
v_a_1892_ = lean_ctor_get(v___x_1891_, 0);
lean_inc(v_a_1892_);
lean_dec_ref_known(v___x_1891_, 1);
v_expr_1893_ = lean_ctor_get(v_a_1892_, 0);
lean_inc_ref(v_expr_1893_);
v_val_1894_ = lean_ctor_get(v_a_1892_, 1);
lean_inc(v_val_1894_);
v_proof_1895_ = lean_ctor_get(v_a_1892_, 2);
lean_inc_ref(v_proof_1895_);
lean_dec(v_a_1892_);
lean_inc_ref(v_r_u03b1_1844_);
lean_inc_ref(v_s_u03b1_1840_);
lean_inc_ref(v_00_u03b1_1839_);
lean_inc(v_u_1838_);
v___x_1896_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(v_u_1838_, v_00_u03b1_1839_, v_s_u03b1_1840_, v_v_1841_, v_00_u03b2_1842_, v_s_u03b2_1843_, v_r_u03b1_1844_, v_a_1887_, v_a_1846_, v_a_1847_, v_a_1848_, v_a_1849_, v_a_1850_, v_a_1851_);
if (lean_obj_tag(v___x_1896_) == 0)
{
lean_object* v_a_1897_; lean_object* v___x_1899_; uint8_t v_isShared_1900_; uint8_t v_isSharedCheck_1955_; 
v_a_1897_ = lean_ctor_get(v___x_1896_, 0);
v_isSharedCheck_1955_ = !lean_is_exclusive(v___x_1896_);
if (v_isSharedCheck_1955_ == 0)
{
v___x_1899_ = v___x_1896_;
v_isShared_1900_ = v_isSharedCheck_1955_;
goto v_resetjp_1898_;
}
else
{
lean_inc(v_a_1897_);
lean_dec(v___x_1896_);
v___x_1899_ = lean_box(0);
v_isShared_1900_ = v_isSharedCheck_1955_;
goto v_resetjp_1898_;
}
v_resetjp_1898_:
{
lean_object* v_expr_1901_; lean_object* v_val_1902_; lean_object* v_proof_1903_; lean_object* v___x_1905_; uint8_t v_isShared_1906_; uint8_t v_isSharedCheck_1954_; 
v_expr_1901_ = lean_ctor_get(v_a_1897_, 0);
v_val_1902_ = lean_ctor_get(v_a_1897_, 1);
v_proof_1903_ = lean_ctor_get(v_a_1897_, 2);
v_isSharedCheck_1954_ = !lean_is_exclusive(v_a_1897_);
if (v_isSharedCheck_1954_ == 0)
{
v___x_1905_ = v_a_1897_;
v_isShared_1906_ = v_isSharedCheck_1954_;
goto v_resetjp_1904_;
}
else
{
lean_inc(v_proof_1903_);
lean_inc(v_val_1902_);
lean_inc(v_expr_1901_);
lean_dec(v_a_1897_);
v___x_1905_ = lean_box(0);
v_isShared_1906_ = v_isSharedCheck_1954_;
goto v_resetjp_1904_;
}
v_resetjp_1904_:
{
lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1936_; 
v___x_1907_ = lean_box(0);
lean_inc_n(v_u_1838_, 2);
v___x_1908_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1908_, 0, v_u_1838_);
lean_ctor_set(v___x_1908_, 1, v___x_1907_);
v___x_1909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_1910_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_1911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_1912_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_1913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref_n(v___x_1908_, 5);
v___x_1914_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1914_, 0, v_u_1838_);
lean_ctor_set(v___x_1914_, 1, v___x_1908_);
v___x_1915_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1915_, 0, v_u_1838_);
lean_ctor_set(v___x_1915_, 1, v___x_1914_);
v___x_1916_ = l_Lean_Expr_const___override(v___x_1909_, v___x_1915_);
lean_inc_ref_n(v_00_u03b1_1839_, 7);
v___x_1917_ = l_Lean_Expr_app___override(v___x_1916_, v_00_u03b1_1839_);
v___x_1918_ = l_Lean_Expr_app___override(v___x_1917_, v_00_u03b1_1839_);
v___x_1919_ = l_Lean_Expr_app___override(v___x_1918_, v_00_u03b1_1839_);
v___x_1920_ = l_Lean_Expr_const___override(v___x_1910_, v___x_1908_);
v___x_1921_ = l_Lean_Expr_app___override(v___x_1920_, v_00_u03b1_1839_);
v___x_1922_ = l_Lean_Expr_const___override(v___x_1911_, v___x_1908_);
v___x_1923_ = l_Lean_Expr_app___override(v___x_1922_, v_00_u03b1_1839_);
v___x_1924_ = l_Lean_Expr_const___override(v___x_1912_, v___x_1908_);
v___x_1925_ = l_Lean_Expr_app___override(v___x_1924_, v_00_u03b1_1839_);
v___x_1926_ = l_Lean_Expr_const___override(v___x_1913_, v___x_1908_);
v___x_1927_ = l_Lean_Expr_app___override(v___x_1926_, v_00_u03b1_1839_);
v___x_1928_ = l_Lean_Expr_app___override(v___x_1927_, v_s_u03b1_1840_);
v___x_1929_ = l_Lean_Expr_app___override(v___x_1925_, v___x_1928_);
v___x_1930_ = l_Lean_Expr_app___override(v___x_1923_, v___x_1929_);
v___x_1931_ = l_Lean_Expr_app___override(v___x_1921_, v___x_1930_);
v___x_1932_ = l_Lean_Expr_app___override(v___x_1919_, v___x_1931_);
lean_inc_ref_n(v_expr_1893_, 2);
v___x_1933_ = l_Lean_Expr_app___override(v___x_1932_, v_expr_1893_);
lean_inc_ref_n(v_expr_1901_, 2);
v___x_1934_ = l_Lean_Expr_app___override(v___x_1933_, v_expr_1901_);
if (v_isShared_1890_ == 0)
{
lean_ctor_set(v___x_1889_, 3, v_val_1902_);
lean_ctor_set(v___x_1889_, 2, v_val_1894_);
lean_ctor_set(v___x_1889_, 1, v_expr_1901_);
lean_ctor_set(v___x_1889_, 0, v_expr_1893_);
v___x_1936_ = v___x_1889_;
goto v_reusejp_1935_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v_expr_1893_);
lean_ctor_set(v_reuseFailAlloc_1953_, 1, v_expr_1901_);
lean_ctor_set(v_reuseFailAlloc_1953_, 2, v_val_1894_);
lean_ctor_set(v_reuseFailAlloc_1953_, 3, v_val_1902_);
v___x_1936_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1935_;
}
v_reusejp_1935_:
{
lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1948_; 
v___x_1937_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___closed__3));
v___x_1938_ = l_Lean_Expr_const___override(v___x_1937_, v___x_1908_);
v___x_1939_ = l_Lean_Expr_app___override(v___x_1938_, v_00_u03b1_1839_);
v___x_1940_ = l_Lean_Expr_app___override(v___x_1939_, v_r_u03b1_1844_);
v___x_1941_ = l_Lean_Expr_app___override(v___x_1940_, v_expr_1893_);
v___x_1942_ = l_Lean_Expr_app___override(v___x_1941_, v_expr_1901_);
v___x_1943_ = l_Lean_Expr_app___override(v___x_1942_, v_a_1884_);
v___x_1944_ = l_Lean_Expr_app___override(v___x_1943_, v_b_1885_);
v___x_1945_ = l_Lean_Expr_app___override(v___x_1944_, v_proof_1895_);
v___x_1946_ = l_Lean_Expr_app___override(v___x_1945_, v_proof_1903_);
if (v_isShared_1906_ == 0)
{
lean_ctor_set(v___x_1905_, 2, v___x_1946_);
lean_ctor_set(v___x_1905_, 1, v___x_1936_);
lean_ctor_set(v___x_1905_, 0, v___x_1934_);
v___x_1948_ = v___x_1905_;
goto v_reusejp_1947_;
}
else
{
lean_object* v_reuseFailAlloc_1952_; 
v_reuseFailAlloc_1952_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1952_, 0, v___x_1934_);
lean_ctor_set(v_reuseFailAlloc_1952_, 1, v___x_1936_);
lean_ctor_set(v_reuseFailAlloc_1952_, 2, v___x_1946_);
v___x_1948_ = v_reuseFailAlloc_1952_;
goto v_reusejp_1947_;
}
v_reusejp_1947_:
{
lean_object* v___x_1950_; 
if (v_isShared_1900_ == 0)
{
lean_ctor_set(v___x_1899_, 0, v___x_1948_);
v___x_1950_ = v___x_1899_;
goto v_reusejp_1949_;
}
else
{
lean_object* v_reuseFailAlloc_1951_; 
v_reuseFailAlloc_1951_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1951_, 0, v___x_1948_);
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
lean_dec_ref(v_proof_1895_);
lean_dec(v_val_1894_);
lean_dec_ref(v_expr_1893_);
lean_del_object(v___x_1889_);
lean_dec_ref(v_b_1885_);
lean_dec_ref(v_a_1884_);
lean_dec_ref(v_r_u03b1_1844_);
lean_dec_ref(v_s_u03b1_1840_);
lean_dec_ref(v_00_u03b1_1839_);
lean_dec(v_u_1838_);
return v___x_1896_;
}
}
else
{
lean_object* v_a_1956_; lean_object* v___x_1958_; uint8_t v_isShared_1959_; uint8_t v_isSharedCheck_1963_; 
lean_del_object(v___x_1889_);
lean_dec(v_a_1887_);
lean_dec_ref(v_b_1885_);
lean_dec_ref(v_a_1884_);
lean_dec_ref(v_r_u03b1_1844_);
lean_dec_ref(v_00_u03b2_1842_);
lean_dec_ref(v_s_u03b1_1840_);
lean_dec_ref(v_00_u03b1_1839_);
lean_dec(v_u_1838_);
v_a_1956_ = lean_ctor_get(v___x_1891_, 0);
v_isSharedCheck_1963_ = !lean_is_exclusive(v___x_1891_);
if (v_isSharedCheck_1963_ == 0)
{
v___x_1958_ = v___x_1891_;
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
else
{
lean_inc(v_a_1956_);
lean_dec(v___x_1891_);
v___x_1958_ = lean_box(0);
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
v_resetjp_1957_:
{
lean_object* v___x_1961_; 
if (v_isShared_1959_ == 0)
{
v___x_1961_ = v___x_1958_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_1962_; 
v_reuseFailAlloc_1962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1962_, 0, v_a_1956_);
v___x_1961_ = v_reuseFailAlloc_1962_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
return v___x_1961_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast(lean_object* v_u_1965_, lean_object* v_00_u03b1_1966_, lean_object* v_s_u03b1_1967_, lean_object* v_v_1968_, lean_object* v_00_u03b2_1969_, lean_object* v_s_u03b2_1970_, lean_object* v_a_1971_, lean_object* v_r_u03b1_1972_, lean_object* v_va_1973_, lean_object* v_a_1974_, lean_object* v_a_1975_, lean_object* v_a_1976_, lean_object* v_a_1977_, lean_object* v_a_1978_, lean_object* v_a_1979_){
_start:
{
if (lean_obj_tag(v_va_1973_) == 0)
{
lean_object* v___x_1982_; uint8_t v_isShared_1983_; uint8_t v_isSharedCheck_2038_; 
lean_dec_ref(v_00_u03b2_1969_);
lean_dec_ref(v_s_u03b1_1967_);
v_isSharedCheck_2038_ = !lean_is_exclusive(v_va_1973_);
if (v_isSharedCheck_2038_ == 0)
{
lean_object* v_unused_2039_; lean_object* v_unused_2040_; 
v_unused_2039_ = lean_ctor_get(v_va_1973_, 1);
lean_dec(v_unused_2039_);
v_unused_2040_ = lean_ctor_get(v_va_1973_, 0);
lean_dec(v_unused_2040_);
v___x_1982_ = v_va_1973_;
v_isShared_1983_ = v_isSharedCheck_2038_;
goto v_resetjp_1981_;
}
else
{
lean_dec(v_va_1973_);
v___x_1982_ = lean_box(0);
v_isShared_1983_ = v_isSharedCheck_2038_;
goto v_resetjp_1981_;
}
v_resetjp_1981_:
{
lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_1984_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__0));
v___x_1985_ = lean_box(0);
lean_inc(v_u_1965_);
v___x_1986_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1986_, 0, v_u_1965_);
lean_ctor_set(v___x_1986_, 1, v___x_1985_);
lean_inc_ref_n(v___x_1986_, 3);
v___x_1987_ = l_Lean_Expr_const___override(v___x_1984_, v___x_1986_);
lean_inc_ref_n(v_00_u03b1_1966_, 4);
v___x_1988_ = l_Lean_Expr_app___override(v___x_1987_, v_00_u03b1_1966_);
v___x_1989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__2));
v___x_1990_ = l_Lean_Expr_const___override(v___x_1989_, v___x_1986_);
v___x_1991_ = l_Lean_Expr_app___override(v___x_1990_, v_00_u03b1_1966_);
v___x_1992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___lam__0___closed__3));
v___x_1993_ = l_Lean_Expr_const___override(v___x_1992_, v___x_1986_);
v___x_1994_ = l_Lean_Expr_app___override(v___x_1993_, v_00_u03b1_1966_);
v___x_1995_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5));
v___x_1996_ = l_Lean_Expr_const___override(v___x_1995_, v___x_1986_);
v___x_1997_ = l_Lean_Expr_app___override(v___x_1996_, v_00_u03b1_1966_);
v___x_1998_ = l_Lean_Expr_app___override(v___x_1997_, v_r_u03b1_1972_);
v___x_1999_ = l_Lean_Expr_app___override(v___x_1994_, v___x_1998_);
v___x_2000_ = l_Lean_Expr_app___override(v___x_1991_, v___x_1999_);
v___x_2001_ = l_Lean_Expr_app___override(v___x_1988_, v___x_2000_);
v___x_2002_ = l_Lean_Expr_app___override(v___x_2001_, v_a_1971_);
v___x_2003_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(v___x_2002_, v_a_1974_, v_a_1975_, v_a_1976_, v_a_1977_, v_a_1978_, v_a_1979_);
if (lean_obj_tag(v___x_2003_) == 0)
{
lean_object* v_a_2004_; lean_object* v___x_2006_; uint8_t v_isShared_2007_; uint8_t v_isSharedCheck_2029_; 
v_a_2004_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2029_ == 0)
{
v___x_2006_ = v___x_2003_;
v_isShared_2007_ = v_isSharedCheck_2029_;
goto v_resetjp_2005_;
}
else
{
lean_inc(v_a_2004_);
lean_dec(v___x_2003_);
v___x_2006_ = lean_box(0);
v_isShared_2007_ = v_isSharedCheck_2029_;
goto v_resetjp_2005_;
}
v_resetjp_2005_:
{
lean_object* v_fst_2008_; lean_object* v_snd_2009_; lean_object* v___x_2011_; uint8_t v_isShared_2012_; uint8_t v_isSharedCheck_2028_; 
v_fst_2008_ = lean_ctor_get(v_a_2004_, 0);
v_snd_2009_ = lean_ctor_get(v_a_2004_, 1);
v_isSharedCheck_2028_ = !lean_is_exclusive(v_a_2004_);
if (v_isSharedCheck_2028_ == 0)
{
v___x_2011_ = v_a_2004_;
v_isShared_2012_ = v_isSharedCheck_2028_;
goto v_resetjp_2010_;
}
else
{
lean_inc(v_snd_2009_);
lean_inc(v_fst_2008_);
lean_dec(v_a_2004_);
v___x_2011_ = lean_box(0);
v_isShared_2012_ = v_isSharedCheck_2028_;
goto v_resetjp_2010_;
}
v_resetjp_2010_:
{
lean_object* v___x_2014_; 
lean_inc(v_snd_2009_);
if (v_isShared_1983_ == 0)
{
lean_ctor_set(v___x_1982_, 1, v_fst_2008_);
lean_ctor_set(v___x_1982_, 0, v_snd_2009_);
v___x_2014_ = v___x_1982_;
goto v_reusejp_2013_;
}
else
{
lean_object* v_reuseFailAlloc_2027_; 
v_reuseFailAlloc_2027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2027_, 0, v_snd_2009_);
lean_ctor_set(v_reuseFailAlloc_2027_, 1, v_fst_2008_);
v___x_2014_ = v_reuseFailAlloc_2027_;
goto v_reusejp_2013_;
}
v_reusejp_2013_:
{
lean_object* v___x_2015_; lean_object* v___x_2017_; 
v___x_2015_ = l_Lean_Level_succ___override(v_u_1965_);
if (v_isShared_2012_ == 0)
{
lean_ctor_set_tag(v___x_2011_, 1);
lean_ctor_set(v___x_2011_, 1, v___x_1985_);
lean_ctor_set(v___x_2011_, 0, v___x_2015_);
v___x_2017_ = v___x_2011_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2026_; 
v_reuseFailAlloc_2026_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2026_, 0, v___x_2015_);
lean_ctor_set(v_reuseFailAlloc_2026_, 1, v___x_1985_);
v___x_2017_ = v_reuseFailAlloc_2026_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2024_; 
v___x_2018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalNatCast___closed__7));
v___x_2019_ = l_Lean_Expr_const___override(v___x_2018_, v___x_2017_);
v___x_2020_ = l_Lean_Expr_app___override(v___x_2019_, v_00_u03b1_1966_);
lean_inc(v_snd_2009_);
v___x_2021_ = l_Lean_Expr_app___override(v___x_2020_, v_snd_2009_);
v___x_2022_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2022_, 0, v_snd_2009_);
lean_ctor_set(v___x_2022_, 1, v___x_2014_);
lean_ctor_set(v___x_2022_, 2, v___x_2021_);
if (v_isShared_2007_ == 0)
{
lean_ctor_set(v___x_2006_, 0, v___x_2022_);
v___x_2024_ = v___x_2006_;
goto v_reusejp_2023_;
}
else
{
lean_object* v_reuseFailAlloc_2025_; 
v_reuseFailAlloc_2025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2025_, 0, v___x_2022_);
v___x_2024_ = v_reuseFailAlloc_2025_;
goto v_reusejp_2023_;
}
v_reusejp_2023_:
{
return v___x_2024_;
}
}
}
}
}
}
else
{
lean_object* v_a_2030_; lean_object* v___x_2032_; uint8_t v_isShared_2033_; uint8_t v_isSharedCheck_2037_; 
lean_del_object(v___x_1982_);
lean_dec_ref(v_00_u03b1_1966_);
lean_dec(v_u_1965_);
v_a_2030_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2037_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2037_ == 0)
{
v___x_2032_ = v___x_2003_;
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
else
{
lean_inc(v_a_2030_);
lean_dec(v___x_2003_);
v___x_2032_ = lean_box(0);
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
v_resetjp_2031_:
{
lean_object* v___x_2035_; 
if (v_isShared_2033_ == 0)
{
v___x_2035_ = v___x_2032_;
goto v_reusejp_2034_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v_a_2030_);
v___x_2035_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2034_;
}
v_reusejp_2034_:
{
return v___x_2035_;
}
}
}
}
}
else
{
lean_object* v_x_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2075_; 
lean_dec_ref(v_a_1971_);
v_x_2041_ = lean_ctor_get(v_va_1973_, 1);
v_isSharedCheck_2075_ = !lean_is_exclusive(v_va_1973_);
if (v_isSharedCheck_2075_ == 0)
{
lean_object* v_unused_2076_; 
v_unused_2076_ = lean_ctor_get(v_va_1973_, 0);
lean_dec(v_unused_2076_);
v___x_2043_ = v_va_1973_;
v_isShared_2044_ = v_isSharedCheck_2075_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_x_2041_);
lean_dec(v_va_1973_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2075_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v___x_2045_; 
v___x_2045_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(v_u_1965_, v_00_u03b1_1966_, v_s_u03b1_1967_, v_v_1968_, v_00_u03b2_1969_, v_s_u03b2_1970_, v_r_u03b1_1972_, v_x_2041_, v_a_1974_, v_a_1975_, v_a_1976_, v_a_1977_, v_a_1978_, v_a_1979_);
if (lean_obj_tag(v___x_2045_) == 0)
{
lean_object* v_a_2046_; lean_object* v___x_2048_; uint8_t v_isShared_2049_; uint8_t v_isSharedCheck_2066_; 
v_a_2046_ = lean_ctor_get(v___x_2045_, 0);
v_isSharedCheck_2066_ = !lean_is_exclusive(v___x_2045_);
if (v_isSharedCheck_2066_ == 0)
{
v___x_2048_ = v___x_2045_;
v_isShared_2049_ = v_isSharedCheck_2066_;
goto v_resetjp_2047_;
}
else
{
lean_inc(v_a_2046_);
lean_dec(v___x_2045_);
v___x_2048_ = lean_box(0);
v_isShared_2049_ = v_isSharedCheck_2066_;
goto v_resetjp_2047_;
}
v_resetjp_2047_:
{
lean_object* v_expr_2050_; lean_object* v_val_2051_; lean_object* v_proof_2052_; lean_object* v___x_2054_; uint8_t v_isShared_2055_; uint8_t v_isSharedCheck_2065_; 
v_expr_2050_ = lean_ctor_get(v_a_2046_, 0);
v_val_2051_ = lean_ctor_get(v_a_2046_, 1);
v_proof_2052_ = lean_ctor_get(v_a_2046_, 2);
v_isSharedCheck_2065_ = !lean_is_exclusive(v_a_2046_);
if (v_isSharedCheck_2065_ == 0)
{
v___x_2054_ = v_a_2046_;
v_isShared_2055_ = v_isSharedCheck_2065_;
goto v_resetjp_2053_;
}
else
{
lean_inc(v_proof_2052_);
lean_inc(v_val_2051_);
lean_inc(v_expr_2050_);
lean_dec(v_a_2046_);
v___x_2054_ = lean_box(0);
v_isShared_2055_ = v_isSharedCheck_2065_;
goto v_resetjp_2053_;
}
v_resetjp_2053_:
{
lean_object* v___x_2057_; 
lean_inc_ref(v_expr_2050_);
if (v_isShared_2044_ == 0)
{
lean_ctor_set(v___x_2043_, 1, v_val_2051_);
lean_ctor_set(v___x_2043_, 0, v_expr_2050_);
v___x_2057_ = v___x_2043_;
goto v_reusejp_2056_;
}
else
{
lean_object* v_reuseFailAlloc_2064_; 
v_reuseFailAlloc_2064_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2064_, 0, v_expr_2050_);
lean_ctor_set(v_reuseFailAlloc_2064_, 1, v_val_2051_);
v___x_2057_ = v_reuseFailAlloc_2064_;
goto v_reusejp_2056_;
}
v_reusejp_2056_:
{
lean_object* v___x_2059_; 
if (v_isShared_2055_ == 0)
{
lean_ctor_set(v___x_2054_, 1, v___x_2057_);
v___x_2059_ = v___x_2054_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2063_; 
v_reuseFailAlloc_2063_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2063_, 0, v_expr_2050_);
lean_ctor_set(v_reuseFailAlloc_2063_, 1, v___x_2057_);
lean_ctor_set(v_reuseFailAlloc_2063_, 2, v_proof_2052_);
v___x_2059_ = v_reuseFailAlloc_2063_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
lean_object* v___x_2061_; 
if (v_isShared_2049_ == 0)
{
lean_ctor_set(v___x_2048_, 0, v___x_2059_);
v___x_2061_ = v___x_2048_;
goto v_reusejp_2060_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v___x_2059_);
v___x_2061_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2060_;
}
v_reusejp_2060_:
{
return v___x_2061_;
}
}
}
}
}
}
else
{
lean_object* v_a_2067_; lean_object* v___x_2069_; uint8_t v_isShared_2070_; uint8_t v_isSharedCheck_2074_; 
lean_del_object(v___x_2043_);
v_a_2067_ = lean_ctor_get(v___x_2045_, 0);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2045_);
if (v_isSharedCheck_2074_ == 0)
{
v___x_2069_ = v___x_2045_;
v_isShared_2070_ = v_isSharedCheck_2074_;
goto v_resetjp_2068_;
}
else
{
lean_inc(v_a_2067_);
lean_dec(v___x_2045_);
v___x_2069_ = lean_box(0);
v_isShared_2070_ = v_isSharedCheck_2074_;
goto v_resetjp_2068_;
}
v_resetjp_2068_:
{
lean_object* v___x_2072_; 
if (v_isShared_2070_ == 0)
{
v___x_2072_ = v___x_2069_;
goto v_reusejp_2071_;
}
else
{
lean_object* v_reuseFailAlloc_2073_; 
v_reuseFailAlloc_2073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2073_, 0, v_a_2067_);
v___x_2072_ = v_reuseFailAlloc_2073_;
goto v_reusejp_2071_;
}
v_reusejp_2071_:
{
return v___x_2072_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___boxed(lean_object* v_u_2077_, lean_object* v_00_u03b1_2078_, lean_object* v_s_u03b1_2079_, lean_object* v_v_2080_, lean_object* v_00_u03b2_2081_, lean_object* v_s_u03b2_2082_, lean_object* v_a_2083_, lean_object* v_r_u03b1_2084_, lean_object* v_va_2085_, lean_object* v_a_2086_, lean_object* v_a_2087_, lean_object* v_a_2088_, lean_object* v_a_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_){
_start:
{
lean_object* v_res_2093_; 
v_res_2093_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast(v_u_2077_, v_00_u03b1_2078_, v_s_u03b1_2079_, v_v_2080_, v_00_u03b2_2081_, v_s_u03b2_2082_, v_a_2083_, v_r_u03b1_2084_, v_va_2085_, v_a_2086_, v_a_2087_, v_a_2088_, v_a_2089_, v_a_2090_, v_a_2091_);
lean_dec(v_a_2091_);
lean_dec_ref(v_a_2090_);
lean_dec(v_a_2089_);
lean_dec_ref(v_a_2088_);
lean_dec(v_a_2087_);
lean_dec_ref(v_a_2086_);
lean_dec_ref(v_s_u03b2_2082_);
lean_dec(v_v_2080_);
return v_res_2093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg___boxed(lean_object* v_u_2094_, lean_object* v_00_u03b1_2095_, lean_object* v_s_u03b1_2096_, lean_object* v_v_2097_, lean_object* v_00_u03b2_2098_, lean_object* v_s_u03b2_2099_, lean_object* v_r_u03b1_2100_, lean_object* v_va_2101_, lean_object* v_a_2102_, lean_object* v_a_2103_, lean_object* v_a_2104_, lean_object* v_a_2105_, lean_object* v_a_2106_, lean_object* v_a_2107_, lean_object* v_a_2108_){
_start:
{
lean_object* v_res_2109_; 
v_res_2109_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(v_u_2094_, v_00_u03b1_2095_, v_s_u03b1_2096_, v_v_2097_, v_00_u03b2_2098_, v_s_u03b2_2099_, v_r_u03b1_2100_, v_va_2101_, v_a_2102_, v_a_2103_, v_a_2104_, v_a_2105_, v_a_2106_, v_a_2107_);
lean_dec(v_a_2107_);
lean_dec_ref(v_a_2106_);
lean_dec(v_a_2105_);
lean_dec_ref(v_a_2104_);
lean_dec(v_a_2103_);
lean_dec_ref(v_a_2102_);
lean_dec_ref(v_s_u03b2_2099_);
lean_dec(v_v_2097_);
return v_res_2109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast___boxed(lean_object* v_u_2110_, lean_object* v_00_u03b1_2111_, lean_object* v_s_u03b1_2112_, lean_object* v_v_2113_, lean_object* v_00_u03b2_2114_, lean_object* v_s_u03b2_2115_, lean_object* v_a_2116_, lean_object* v_r_u03b1_2117_, lean_object* v_va_2118_, lean_object* v_a_2119_, lean_object* v_a_2120_, lean_object* v_a_2121_, lean_object* v_a_2122_, lean_object* v_a_2123_, lean_object* v_a_2124_, lean_object* v_a_2125_){
_start:
{
lean_object* v_res_2126_; 
v_res_2126_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalIntCast(v_u_2110_, v_00_u03b1_2111_, v_s_u03b1_2112_, v_v_2113_, v_00_u03b2_2114_, v_s_u03b2_2115_, v_a_2116_, v_r_u03b1_2117_, v_va_2118_, v_a_2119_, v_a_2120_, v_a_2121_, v_a_2122_, v_a_2123_, v_a_2124_);
lean_dec(v_a_2124_);
lean_dec_ref(v_a_2123_);
lean_dec(v_a_2122_);
lean_dec_ref(v_a_2121_);
lean_dec(v_a_2120_);
lean_dec_ref(v_a_2119_);
lean_dec_ref(v_s_u03b2_2115_);
lean_dec(v_v_2113_);
return v_res_2126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast(lean_object* v_u_2127_, lean_object* v_00_u03b1_2128_, lean_object* v_s_u03b1_2129_, lean_object* v_v_2130_, lean_object* v_00_u03b2_2131_, lean_object* v_s_u03b2_2132_, lean_object* v_a_2133_, lean_object* v_r_u03b1_2134_, lean_object* v_va_2135_, lean_object* v_a_2136_, lean_object* v_a_2137_, lean_object* v_a_2138_, lean_object* v_a_2139_, lean_object* v_a_2140_, lean_object* v_a_2141_){
_start:
{
lean_object* v___x_2143_; 
v___x_2143_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(v_u_2127_, v_00_u03b1_2128_, v_s_u03b1_2129_, v_v_2130_, v_00_u03b2_2131_, v_s_u03b2_2132_, v_r_u03b1_2134_, v_va_2135_, v_a_2136_, v_a_2137_, v_a_2138_, v_a_2139_, v_a_2140_, v_a_2141_);
return v___x_2143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___boxed(lean_object* v_u_2144_, lean_object* v_00_u03b1_2145_, lean_object* v_s_u03b1_2146_, lean_object* v_v_2147_, lean_object* v_00_u03b2_2148_, lean_object* v_s_u03b2_2149_, lean_object* v_a_2150_, lean_object* v_r_u03b1_2151_, lean_object* v_va_2152_, lean_object* v_a_2153_, lean_object* v_a_2154_, lean_object* v_a_2155_, lean_object* v_a_2156_, lean_object* v_a_2157_, lean_object* v_a_2158_, lean_object* v_a_2159_){
_start:
{
lean_object* v_res_2160_; 
v_res_2160_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast(v_u_2144_, v_00_u03b1_2145_, v_s_u03b1_2146_, v_v_2147_, v_00_u03b2_2148_, v_s_u03b2_2149_, v_a_2150_, v_r_u03b1_2151_, v_va_2152_, v_a_2153_, v_a_2154_, v_a_2155_, v_a_2156_, v_a_2157_, v_a_2158_);
lean_dec(v_a_2158_);
lean_dec_ref(v_a_2157_);
lean_dec(v_a_2156_);
lean_dec_ref(v_a_2155_);
lean_dec(v_a_2154_);
lean_dec_ref(v_a_2153_);
lean_dec_ref(v_a_2150_);
lean_dec_ref(v_s_u03b2_2149_);
lean_dec(v_v_2147_);
return v_res_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2(lean_object* v_e_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_){
_start:
{
lean_object* v___x_2167_; 
v___x_2167_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___redArg(v_e_2161_, v___y_2163_);
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2___boxed(lean_object* v_e_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_){
_start:
{
lean_object* v_res_2174_; 
v_res_2174_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__2(v_e_2168_, v___y_2169_, v___y_2170_, v___y_2171_, v___y_2172_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2171_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
return v_res_2174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3(lean_object* v_00_u03b1_2175_, lean_object* v_k_2176_, uint8_t v_allowLevelAssignments_2177_, lean_object* v___y_2178_, lean_object* v___y_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_){
_start:
{
lean_object* v___x_2183_; 
v___x_2183_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v_k_2176_, v_allowLevelAssignments_2177_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_);
return v___x_2183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___boxed(lean_object* v_00_u03b1_2184_, lean_object* v_k_2185_, lean_object* v_allowLevelAssignments_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2192_; lean_object* v_res_2193_; 
v_allowLevelAssignments_boxed_2192_ = lean_unbox(v_allowLevelAssignments_2186_);
v_res_2193_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3(v_00_u03b1_2184_, v_k_2185_, v_allowLevelAssignments_boxed_2192_, v___y_2187_, v___y_2188_, v___y_2189_, v___y_2190_);
lean_dec(v___y_2190_);
lean_dec_ref(v___y_2189_);
lean_dec(v___y_2188_);
lean_dec_ref(v___y_2187_);
return v_res_2193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4(lean_object* v_00_u03b1_2194_, lean_object* v_msg_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_){
_start:
{
lean_object* v___x_2203_; 
v___x_2203_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v_msg_2195_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_);
return v___x_2203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___boxed(lean_object* v_00_u03b1_2204_, lean_object* v_msg_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_, lean_object* v___y_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_){
_start:
{
lean_object* v_res_2213_; 
v_res_2213_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4(v_00_u03b1_2204_, v_msg_2205_, v___y_2206_, v___y_2207_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_);
lean_dec(v___y_2211_);
lean_dec_ref(v___y_2210_);
lean_dec(v___y_2209_);
lean_dec_ref(v___y_2208_);
lean_dec(v___y_2207_);
lean_dec_ref(v___y_2206_);
return v_res_2213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast(lean_object* v_u_2214_, lean_object* v_00_u03b1_2215_, lean_object* v_s_u03b1_2216_, lean_object* v_v_2217_, lean_object* v_00_u03b2_2218_, lean_object* v_s_u03b2_2219_, lean_object* v_a_2220_, lean_object* v_x_2221_){
_start:
{
if (lean_obj_tag(v_x_2221_) == 0)
{
lean_object* v_value_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2239_; 
lean_dec_ref(v_s_u03b2_2219_);
lean_dec_ref(v_00_u03b2_2218_);
lean_dec(v_v_2217_);
v_value_2222_ = lean_ctor_get(v_x_2221_, 1);
v_isSharedCheck_2239_ = !lean_is_exclusive(v_x_2221_);
if (v_isSharedCheck_2239_ == 0)
{
lean_object* v_unused_2240_; 
v_unused_2240_ = lean_ctor_get(v_x_2221_, 0);
lean_dec(v_unused_2240_);
v___x_2224_ = v_x_2221_;
v_isShared_2225_ = v_isSharedCheck_2239_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_value_2222_);
lean_dec(v_x_2221_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2239_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v_value_2226_; lean_object* v_hyp_2227_; lean_object* v___x_2229_; uint8_t v_isShared_2230_; uint8_t v_isSharedCheck_2238_; 
v_value_2226_ = lean_ctor_get(v_value_2222_, 0);
v_hyp_2227_ = lean_ctor_get(v_value_2222_, 1);
v_isSharedCheck_2238_ = !lean_is_exclusive(v_value_2222_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2229_ = v_value_2222_;
v_isShared_2230_ = v_isSharedCheck_2238_;
goto v_resetjp_2228_;
}
else
{
lean_inc(v_hyp_2227_);
lean_inc(v_value_2226_);
lean_dec(v_value_2222_);
v___x_2229_ = lean_box(0);
v_isShared_2230_ = v_isSharedCheck_2238_;
goto v_resetjp_2228_;
}
v_resetjp_2228_:
{
lean_object* v___x_2232_; 
if (v_isShared_2230_ == 0)
{
v___x_2232_ = v___x_2229_;
goto v_reusejp_2231_;
}
else
{
lean_object* v_reuseFailAlloc_2237_; 
v_reuseFailAlloc_2237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2237_, 0, v_value_2226_);
lean_ctor_set(v_reuseFailAlloc_2237_, 1, v_hyp_2227_);
v___x_2232_ = v_reuseFailAlloc_2237_;
goto v_reusejp_2231_;
}
v_reusejp_2231_:
{
lean_object* v___x_2234_; 
lean_inc_ref(v_a_2220_);
if (v_isShared_2225_ == 0)
{
lean_ctor_set(v___x_2224_, 1, v___x_2232_);
lean_ctor_set(v___x_2224_, 0, v_a_2220_);
v___x_2234_ = v___x_2224_;
goto v_reusejp_2233_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_a_2220_);
lean_ctor_set(v_reuseFailAlloc_2236_, 1, v___x_2232_);
v___x_2234_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2233_;
}
v_reusejp_2233_:
{
lean_object* v___x_2235_; 
v___x_2235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2235_, 0, v_a_2220_);
lean_ctor_set(v___x_2235_, 1, v___x_2234_);
return v___x_2235_;
}
}
}
}
}
else
{
lean_object* v_x_2241_; lean_object* v_e_2242_; lean_object* v_b_2243_; lean_object* v_a_2244_; lean_object* v_a_2245_; lean_object* v_a_2246_; lean_object* v___x_2248_; uint8_t v_isShared_2249_; uint8_t v_isSharedCheck_2325_; 
lean_dec_ref(v_a_2220_);
v_x_2241_ = lean_ctor_get(v_x_2221_, 0);
v_e_2242_ = lean_ctor_get(v_x_2221_, 1);
v_b_2243_ = lean_ctor_get(v_x_2221_, 2);
v_a_2244_ = lean_ctor_get(v_x_2221_, 3);
v_a_2245_ = lean_ctor_get(v_x_2221_, 4);
v_a_2246_ = lean_ctor_get(v_x_2221_, 5);
v_isSharedCheck_2325_ = !lean_is_exclusive(v_x_2221_);
if (v_isSharedCheck_2325_ == 0)
{
v___x_2248_ = v_x_2221_;
v_isShared_2249_ = v_isSharedCheck_2325_;
goto v_resetjp_2247_;
}
else
{
lean_inc(v_a_2246_);
lean_inc(v_a_2245_);
lean_inc(v_a_2244_);
lean_inc(v_b_2243_);
lean_inc(v_e_2242_);
lean_inc(v_x_2241_);
lean_dec(v_x_2221_);
v___x_2248_ = lean_box(0);
v_isShared_2249_ = v_isSharedCheck_2325_;
goto v_resetjp_2247_;
}
v_resetjp_2247_:
{
lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v_fst_2296_; lean_object* v_snd_2297_; lean_object* v___x_2298_; lean_object* v_fst_2299_; lean_object* v_snd_2300_; lean_object* v___x_2302_; uint8_t v_isShared_2303_; uint8_t v_isSharedCheck_2324_; 
v___x_2250_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4));
v___x_2251_ = lean_box(0);
lean_inc_n(v_v_2217_, 6);
v___x_2252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2252_, 0, v_v_2217_);
lean_ctor_set(v___x_2252_, 1, v___x_2251_);
lean_inc_ref_n(v___x_2252_, 8);
v___x_2253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2253_, 0, v_v_2217_);
lean_ctor_set(v___x_2253_, 1, v___x_2252_);
v___x_2254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2254_, 0, v_v_2217_);
lean_ctor_set(v___x_2254_, 1, v___x_2253_);
v___x_2255_ = l_Lean_Expr_const___override(v___x_2250_, v___x_2254_);
lean_inc_ref_n(v_00_u03b2_2218_, 14);
v___x_2256_ = l_Lean_Expr_app___override(v___x_2255_, v_00_u03b2_2218_);
v___x_2257_ = l_Lean_Expr_app___override(v___x_2256_, v_00_u03b2_2218_);
v___x_2258_ = l_Lean_Expr_app___override(v___x_2257_, v_00_u03b2_2218_);
v___x_2259_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6));
v___x_2260_ = l_Lean_Expr_const___override(v___x_2259_, v___x_2252_);
v___x_2261_ = l_Lean_Expr_app___override(v___x_2260_, v_00_u03b2_2218_);
v___x_2262_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8));
v___x_2263_ = l_Lean_Expr_const___override(v___x_2262_, v___x_2252_);
v___x_2264_ = l_Lean_Expr_app___override(v___x_2263_, v_00_u03b2_2218_);
v___x_2265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_2266_ = l_Lean_Expr_const___override(v___x_2265_, v___x_2252_);
v___x_2267_ = l_Lean_Expr_app___override(v___x_2266_, v_00_u03b2_2218_);
v___x_2268_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_2269_ = l_Lean_Expr_const___override(v___x_2268_, v___x_2252_);
v___x_2270_ = l_Lean_Expr_app___override(v___x_2269_, v_00_u03b2_2218_);
lean_inc_ref_n(v_s_u03b2_2219_, 2);
v___x_2271_ = l_Lean_Expr_app___override(v___x_2270_, v_s_u03b2_2219_);
v___x_2272_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11));
v___x_2273_ = lean_box(0);
v___x_2274_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2274_, 0, v___x_2273_);
lean_ctor_set(v___x_2274_, 1, v___x_2252_);
v___x_2275_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2275_, 0, v_v_2217_);
lean_ctor_set(v___x_2275_, 1, v___x_2274_);
v___x_2276_ = l_Lean_Expr_const___override(v___x_2272_, v___x_2275_);
v___x_2277_ = l_Lean_Expr_app___override(v___x_2276_, v_00_u03b2_2218_);
v___x_2278_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_2279_ = l_Lean_Expr_app___override(v___x_2277_, v___x_2278_);
v___x_2280_ = l_Lean_Expr_app___override(v___x_2279_, v_00_u03b2_2218_);
v___x_2281_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15));
v___x_2282_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_2283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2283_, 0, v_v_2217_);
lean_ctor_set(v___x_2283_, 1, v___x_2282_);
v___x_2284_ = l_Lean_Expr_const___override(v___x_2281_, v___x_2283_);
v___x_2285_ = l_Lean_Expr_app___override(v___x_2284_, v_00_u03b2_2218_);
v___x_2286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19));
v___x_2287_ = l_Lean_Expr_const___override(v___x_2286_, v___x_2252_);
v___x_2288_ = l_Lean_Expr_app___override(v___x_2287_, v_00_u03b2_2218_);
v___x_2289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22));
v___x_2290_ = l_Lean_Expr_const___override(v___x_2289_, v___x_2252_);
v___x_2291_ = l_Lean_Expr_app___override(v___x_2290_, v_00_u03b2_2218_);
v___x_2292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24));
v___x_2293_ = l_Lean_Expr_const___override(v___x_2292_, v___x_2252_);
v___x_2294_ = l_Lean_Expr_app___override(v___x_2293_, v_00_u03b2_2218_);
v___x_2295_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast(v_u_2214_, v_00_u03b1_2215_, v_s_u03b1_2216_, v_v_2217_, v_00_u03b2_2218_, v_s_u03b2_2219_, v_x_2241_, v_a_2244_);
v_fst_2296_ = lean_ctor_get(v___x_2295_, 0);
lean_inc(v_fst_2296_);
v_snd_2297_ = lean_ctor_get(v___x_2295_, 1);
lean_inc(v_snd_2297_);
lean_dec_ref(v___x_2295_);
v___x_2298_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast(v_u_2214_, v_00_u03b1_2215_, v_s_u03b1_2216_, v_v_2217_, v_00_u03b2_2218_, v_s_u03b2_2219_, v_b_2243_, v_a_2246_);
v_fst_2299_ = lean_ctor_get(v___x_2298_, 0);
v_snd_2300_ = lean_ctor_get(v___x_2298_, 1);
v_isSharedCheck_2324_ = !lean_is_exclusive(v___x_2298_);
if (v_isSharedCheck_2324_ == 0)
{
v___x_2302_ = v___x_2298_;
v_isShared_2303_ = v_isSharedCheck_2324_;
goto v_resetjp_2301_;
}
else
{
lean_inc(v_snd_2300_);
lean_inc(v_fst_2299_);
lean_dec(v___x_2298_);
v___x_2302_ = lean_box(0);
v_isShared_2303_ = v_isSharedCheck_2324_;
goto v_resetjp_2301_;
}
v_resetjp_2301_:
{
lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2319_; 
lean_inc_ref(v___x_2271_);
v___x_2304_ = l_Lean_Expr_app___override(v___x_2267_, v___x_2271_);
v___x_2305_ = l_Lean_Expr_app___override(v___x_2264_, v___x_2304_);
v___x_2306_ = l_Lean_Expr_app___override(v___x_2261_, v___x_2305_);
v___x_2307_ = l_Lean_Expr_app___override(v___x_2258_, v___x_2306_);
v___x_2308_ = l_Lean_Expr_app___override(v___x_2285_, v___x_2278_);
v___x_2309_ = l_Lean_Expr_app___override(v___x_2294_, v___x_2271_);
v___x_2310_ = l_Lean_Expr_app___override(v___x_2291_, v___x_2309_);
v___x_2311_ = l_Lean_Expr_app___override(v___x_2288_, v___x_2310_);
v___x_2312_ = l_Lean_Expr_app___override(v___x_2308_, v___x_2311_);
v___x_2313_ = l_Lean_Expr_app___override(v___x_2280_, v___x_2312_);
lean_inc(v_fst_2296_);
v___x_2314_ = l_Lean_Expr_app___override(v___x_2313_, v_fst_2296_);
lean_inc_ref(v_e_2242_);
v___x_2315_ = l_Lean_Expr_app___override(v___x_2314_, v_e_2242_);
v___x_2316_ = l_Lean_Expr_app___override(v___x_2307_, v___x_2315_);
lean_inc(v_fst_2299_);
v___x_2317_ = l_Lean_Expr_app___override(v___x_2316_, v_fst_2299_);
if (v_isShared_2249_ == 0)
{
lean_ctor_set(v___x_2248_, 5, v_snd_2300_);
lean_ctor_set(v___x_2248_, 3, v_snd_2297_);
lean_ctor_set(v___x_2248_, 2, v_fst_2299_);
lean_ctor_set(v___x_2248_, 0, v_fst_2296_);
v___x_2319_ = v___x_2248_;
goto v_reusejp_2318_;
}
else
{
lean_object* v_reuseFailAlloc_2323_; 
v_reuseFailAlloc_2323_ = lean_alloc_ctor(1, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2323_, 0, v_fst_2296_);
lean_ctor_set(v_reuseFailAlloc_2323_, 1, v_e_2242_);
lean_ctor_set(v_reuseFailAlloc_2323_, 2, v_fst_2299_);
lean_ctor_set(v_reuseFailAlloc_2323_, 3, v_snd_2297_);
lean_ctor_set(v_reuseFailAlloc_2323_, 4, v_a_2245_);
lean_ctor_set(v_reuseFailAlloc_2323_, 5, v_snd_2300_);
v___x_2319_ = v_reuseFailAlloc_2323_;
goto v_reusejp_2318_;
}
v_reusejp_2318_:
{
lean_object* v___x_2321_; 
if (v_isShared_2303_ == 0)
{
lean_ctor_set(v___x_2302_, 1, v___x_2319_);
lean_ctor_set(v___x_2302_, 0, v___x_2317_);
v___x_2321_ = v___x_2302_;
goto v_reusejp_2320_;
}
else
{
lean_object* v_reuseFailAlloc_2322_; 
v_reuseFailAlloc_2322_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2322_, 0, v___x_2317_);
lean_ctor_set(v_reuseFailAlloc_2322_, 1, v___x_2319_);
v___x_2321_ = v_reuseFailAlloc_2322_;
goto v_reusejp_2320_;
}
v_reusejp_2320_:
{
return v___x_2321_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(lean_object* v_u_2326_, lean_object* v_00_u03b1_2327_, lean_object* v_s_u03b1_2328_, lean_object* v_v_2329_, lean_object* v_00_u03b2_2330_, lean_object* v_s_u03b2_2331_, lean_object* v_x_2332_){
_start:
{
if (lean_obj_tag(v_x_2332_) == 0)
{
lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; 
v___x_2333_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_2334_ = lean_box(0);
v___x_2335_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2335_, 0, v_v_2329_);
lean_ctor_set(v___x_2335_, 1, v___x_2334_);
lean_inc_ref_n(v___x_2335_, 4);
v___x_2336_ = l_Lean_Expr_const___override(v___x_2333_, v___x_2335_);
lean_inc_ref_n(v_00_u03b2_2330_, 4);
v___x_2337_ = l_Lean_Expr_app___override(v___x_2336_, v_00_u03b2_2330_);
v___x_2338_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14, &lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__14);
v___x_2339_ = l_Lean_Expr_app___override(v___x_2337_, v___x_2338_);
v___x_2340_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__17));
v___x_2341_ = l_Lean_Expr_const___override(v___x_2340_, v___x_2335_);
v___x_2342_ = l_Lean_Expr_app___override(v___x_2341_, v_00_u03b2_2330_);
v___x_2343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__20));
v___x_2344_ = l_Lean_Expr_const___override(v___x_2343_, v___x_2335_);
v___x_2345_ = l_Lean_Expr_app___override(v___x_2344_, v_00_u03b2_2330_);
v___x_2346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__22));
v___x_2347_ = l_Lean_Expr_const___override(v___x_2346_, v___x_2335_);
v___x_2348_ = l_Lean_Expr_app___override(v___x_2347_, v_00_u03b2_2330_);
v___x_2349_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_2350_ = l_Lean_Expr_const___override(v___x_2349_, v___x_2335_);
v___x_2351_ = l_Lean_Expr_app___override(v___x_2350_, v_00_u03b2_2330_);
v___x_2352_ = l_Lean_Expr_app___override(v___x_2351_, v_s_u03b2_2331_);
v___x_2353_ = l_Lean_Expr_app___override(v___x_2348_, v___x_2352_);
v___x_2354_ = l_Lean_Expr_app___override(v___x_2345_, v___x_2353_);
v___x_2355_ = l_Lean_Expr_app___override(v___x_2342_, v___x_2354_);
v___x_2356_ = l_Lean_Expr_app___override(v___x_2339_, v___x_2355_);
v___x_2357_ = lean_box(0);
v___x_2358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2358_, 0, v___x_2356_);
lean_ctor_set(v___x_2358_, 1, v___x_2357_);
return v___x_2358_;
}
else
{
lean_object* v_a_2359_; lean_object* v_a_2360_; lean_object* v_a_2361_; lean_object* v___x_2363_; uint8_t v_isShared_2364_; uint8_t v_isSharedCheck_2409_; 
v_a_2359_ = lean_ctor_get(v_x_2332_, 0);
v_a_2360_ = lean_ctor_get(v_x_2332_, 2);
v_a_2361_ = lean_ctor_get(v_x_2332_, 3);
v_isSharedCheck_2409_ = !lean_is_exclusive(v_x_2332_);
if (v_isSharedCheck_2409_ == 0)
{
lean_object* v_unused_2410_; 
v_unused_2410_ = lean_ctor_get(v_x_2332_, 1);
lean_dec(v_unused_2410_);
v___x_2363_ = v_x_2332_;
v_isShared_2364_ = v_isSharedCheck_2409_;
goto v_resetjp_2362_;
}
else
{
lean_inc(v_a_2361_);
lean_inc(v_a_2360_);
lean_inc(v_a_2359_);
lean_dec(v_x_2332_);
v___x_2363_ = lean_box(0);
v_isShared_2364_ = v_isSharedCheck_2409_;
goto v_resetjp_2362_;
}
v_resetjp_2362_:
{
lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v_fst_2388_; lean_object* v_snd_2389_; lean_object* v___x_2390_; lean_object* v_fst_2391_; lean_object* v_snd_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2408_; 
v___x_2365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
v___x_2366_ = lean_box(0);
lean_inc_n(v_v_2329_, 4);
v___x_2367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2367_, 0, v_v_2329_);
lean_ctor_set(v___x_2367_, 1, v___x_2366_);
lean_inc_ref_n(v___x_2367_, 4);
v___x_2368_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2368_, 0, v_v_2329_);
lean_ctor_set(v___x_2368_, 1, v___x_2367_);
v___x_2369_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2369_, 0, v_v_2329_);
lean_ctor_set(v___x_2369_, 1, v___x_2368_);
v___x_2370_ = l_Lean_Expr_const___override(v___x_2365_, v___x_2369_);
lean_inc_ref_n(v_00_u03b2_2330_, 8);
v___x_2371_ = l_Lean_Expr_app___override(v___x_2370_, v_00_u03b2_2330_);
v___x_2372_ = l_Lean_Expr_app___override(v___x_2371_, v_00_u03b2_2330_);
v___x_2373_ = l_Lean_Expr_app___override(v___x_2372_, v_00_u03b2_2330_);
v___x_2374_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_2375_ = l_Lean_Expr_const___override(v___x_2374_, v___x_2367_);
v___x_2376_ = l_Lean_Expr_app___override(v___x_2375_, v_00_u03b2_2330_);
v___x_2377_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
v___x_2378_ = l_Lean_Expr_const___override(v___x_2377_, v___x_2367_);
v___x_2379_ = l_Lean_Expr_app___override(v___x_2378_, v_00_u03b2_2330_);
v___x_2380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_2381_ = l_Lean_Expr_const___override(v___x_2380_, v___x_2367_);
v___x_2382_ = l_Lean_Expr_app___override(v___x_2381_, v_00_u03b2_2330_);
v___x_2383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_2384_ = l_Lean_Expr_const___override(v___x_2383_, v___x_2367_);
v___x_2385_ = l_Lean_Expr_app___override(v___x_2384_, v_00_u03b2_2330_);
lean_inc_ref_n(v_s_u03b2_2331_, 2);
v___x_2386_ = l_Lean_Expr_app___override(v___x_2385_, v_s_u03b2_2331_);
v___x_2387_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast(v_u_2326_, v_00_u03b1_2327_, v_s_u03b1_2328_, v_v_2329_, v_00_u03b2_2330_, v_s_u03b2_2331_, v_a_2359_, v_a_2360_);
v_fst_2388_ = lean_ctor_get(v___x_2387_, 0);
lean_inc(v_fst_2388_);
v_snd_2389_ = lean_ctor_get(v___x_2387_, 1);
lean_inc(v_snd_2389_);
lean_dec_ref(v___x_2387_);
v___x_2390_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(v_u_2326_, v_00_u03b1_2327_, v_s_u03b1_2328_, v_v_2329_, v_00_u03b2_2330_, v_s_u03b2_2331_, v_a_2361_);
v_fst_2391_ = lean_ctor_get(v___x_2390_, 0);
v_snd_2392_ = lean_ctor_get(v___x_2390_, 1);
v_isSharedCheck_2408_ = !lean_is_exclusive(v___x_2390_);
if (v_isSharedCheck_2408_ == 0)
{
v___x_2394_ = v___x_2390_;
v_isShared_2395_ = v_isSharedCheck_2408_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_snd_2392_);
lean_inc(v_fst_2391_);
lean_dec(v___x_2390_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2408_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v___x_2403_; 
v___x_2396_ = l_Lean_Expr_app___override(v___x_2382_, v___x_2386_);
v___x_2397_ = l_Lean_Expr_app___override(v___x_2379_, v___x_2396_);
v___x_2398_ = l_Lean_Expr_app___override(v___x_2376_, v___x_2397_);
v___x_2399_ = l_Lean_Expr_app___override(v___x_2373_, v___x_2398_);
lean_inc(v_fst_2388_);
v___x_2400_ = l_Lean_Expr_app___override(v___x_2399_, v_fst_2388_);
lean_inc(v_fst_2391_);
v___x_2401_ = l_Lean_Expr_app___override(v___x_2400_, v_fst_2391_);
if (v_isShared_2364_ == 0)
{
lean_ctor_set(v___x_2363_, 3, v_snd_2392_);
lean_ctor_set(v___x_2363_, 2, v_snd_2389_);
lean_ctor_set(v___x_2363_, 1, v_fst_2391_);
lean_ctor_set(v___x_2363_, 0, v_fst_2388_);
v___x_2403_ = v___x_2363_;
goto v_reusejp_2402_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v_fst_2388_);
lean_ctor_set(v_reuseFailAlloc_2407_, 1, v_fst_2391_);
lean_ctor_set(v_reuseFailAlloc_2407_, 2, v_snd_2389_);
lean_ctor_set(v_reuseFailAlloc_2407_, 3, v_snd_2392_);
v___x_2403_ = v_reuseFailAlloc_2407_;
goto v_reusejp_2402_;
}
v_reusejp_2402_:
{
lean_object* v___x_2405_; 
if (v_isShared_2395_ == 0)
{
lean_ctor_set(v___x_2394_, 1, v___x_2403_);
lean_ctor_set(v___x_2394_, 0, v___x_2401_);
v___x_2405_ = v___x_2394_;
goto v_reusejp_2404_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v___x_2401_);
lean_ctor_set(v_reuseFailAlloc_2406_, 1, v___x_2403_);
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast(lean_object* v_u_2411_, lean_object* v_00_u03b1_2412_, lean_object* v_s_u03b1_2413_, lean_object* v_v_2414_, lean_object* v_00_u03b2_2415_, lean_object* v_s_u03b2_2416_, lean_object* v_a_2417_, lean_object* v_x_2418_){
_start:
{
if (lean_obj_tag(v_x_2418_) == 0)
{
lean_object* v_id_2419_; lean_object* v___x_2421_; uint8_t v_isShared_2422_; uint8_t v_isSharedCheck_2427_; 
lean_dec_ref(v_s_u03b2_2416_);
lean_dec_ref(v_00_u03b2_2415_);
lean_dec(v_v_2414_);
v_id_2419_ = lean_ctor_get(v_x_2418_, 1);
v_isSharedCheck_2427_ = !lean_is_exclusive(v_x_2418_);
if (v_isSharedCheck_2427_ == 0)
{
lean_object* v_unused_2428_; 
v_unused_2428_ = lean_ctor_get(v_x_2418_, 0);
lean_dec(v_unused_2428_);
v___x_2421_ = v_x_2418_;
v_isShared_2422_ = v_isSharedCheck_2427_;
goto v_resetjp_2420_;
}
else
{
lean_inc(v_id_2419_);
lean_dec(v_x_2418_);
v___x_2421_ = lean_box(0);
v_isShared_2422_ = v_isSharedCheck_2427_;
goto v_resetjp_2420_;
}
v_resetjp_2420_:
{
lean_object* v___x_2424_; 
lean_inc_ref(v_a_2417_);
if (v_isShared_2422_ == 0)
{
lean_ctor_set(v___x_2421_, 0, v_a_2417_);
v___x_2424_ = v___x_2421_;
goto v_reusejp_2423_;
}
else
{
lean_object* v_reuseFailAlloc_2426_; 
v_reuseFailAlloc_2426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2426_, 0, v_a_2417_);
lean_ctor_set(v_reuseFailAlloc_2426_, 1, v_id_2419_);
v___x_2424_ = v_reuseFailAlloc_2426_;
goto v_reusejp_2423_;
}
v_reusejp_2423_:
{
lean_object* v___x_2425_; 
v___x_2425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2425_, 0, v_a_2417_);
lean_ctor_set(v___x_2425_, 1, v___x_2424_);
return v___x_2425_;
}
}
}
else
{
lean_object* v_x_2429_; lean_object* v___x_2431_; uint8_t v_isShared_2432_; uint8_t v_isSharedCheck_2446_; 
lean_dec_ref(v_a_2417_);
v_x_2429_ = lean_ctor_get(v_x_2418_, 1);
v_isSharedCheck_2446_ = !lean_is_exclusive(v_x_2418_);
if (v_isSharedCheck_2446_ == 0)
{
lean_object* v_unused_2447_; 
v_unused_2447_ = lean_ctor_get(v_x_2418_, 0);
lean_dec(v_unused_2447_);
v___x_2431_ = v_x_2418_;
v_isShared_2432_ = v_isSharedCheck_2446_;
goto v_resetjp_2430_;
}
else
{
lean_inc(v_x_2429_);
lean_dec(v_x_2418_);
v___x_2431_ = lean_box(0);
v_isShared_2432_ = v_isSharedCheck_2446_;
goto v_resetjp_2430_;
}
v_resetjp_2430_:
{
lean_object* v___x_2433_; lean_object* v_fst_2434_; lean_object* v_snd_2435_; lean_object* v___x_2437_; uint8_t v_isShared_2438_; uint8_t v_isSharedCheck_2445_; 
v___x_2433_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(v_u_2411_, v_00_u03b1_2412_, v_s_u03b1_2413_, v_v_2414_, v_00_u03b2_2415_, v_s_u03b2_2416_, v_x_2429_);
v_fst_2434_ = lean_ctor_get(v___x_2433_, 0);
v_snd_2435_ = lean_ctor_get(v___x_2433_, 1);
v_isSharedCheck_2445_ = !lean_is_exclusive(v___x_2433_);
if (v_isSharedCheck_2445_ == 0)
{
v___x_2437_ = v___x_2433_;
v_isShared_2438_ = v_isSharedCheck_2445_;
goto v_resetjp_2436_;
}
else
{
lean_inc(v_snd_2435_);
lean_inc(v_fst_2434_);
lean_dec(v___x_2433_);
v___x_2437_ = lean_box(0);
v_isShared_2438_ = v_isSharedCheck_2445_;
goto v_resetjp_2436_;
}
v_resetjp_2436_:
{
lean_object* v___x_2440_; 
lean_inc(v_fst_2434_);
if (v_isShared_2432_ == 0)
{
lean_ctor_set(v___x_2431_, 1, v_snd_2435_);
lean_ctor_set(v___x_2431_, 0, v_fst_2434_);
v___x_2440_ = v___x_2431_;
goto v_reusejp_2439_;
}
else
{
lean_object* v_reuseFailAlloc_2444_; 
v_reuseFailAlloc_2444_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2444_, 0, v_fst_2434_);
lean_ctor_set(v_reuseFailAlloc_2444_, 1, v_snd_2435_);
v___x_2440_ = v_reuseFailAlloc_2444_;
goto v_reusejp_2439_;
}
v_reusejp_2439_:
{
lean_object* v___x_2442_; 
if (v_isShared_2438_ == 0)
{
lean_ctor_set(v___x_2437_, 1, v___x_2440_);
v___x_2442_ = v___x_2437_;
goto v_reusejp_2441_;
}
else
{
lean_object* v_reuseFailAlloc_2443_; 
v_reuseFailAlloc_2443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2443_, 0, v_fst_2434_);
lean_ctor_set(v_reuseFailAlloc_2443_, 1, v___x_2440_);
v___x_2442_ = v_reuseFailAlloc_2443_;
goto v_reusejp_2441_;
}
v_reusejp_2441_:
{
return v___x_2442_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast___boxed(lean_object* v_u_2448_, lean_object* v_00_u03b1_2449_, lean_object* v_s_u03b1_2450_, lean_object* v_v_2451_, lean_object* v_00_u03b2_2452_, lean_object* v_s_u03b2_2453_, lean_object* v_a_2454_, lean_object* v_x_2455_){
_start:
{
lean_object* v_res_2456_; 
v_res_2456_ = lp_mathlib_Mathlib_Tactic_Ring_ExBase_cast(v_u_2448_, v_00_u03b1_2449_, v_s_u03b1_2450_, v_v_2451_, v_00_u03b2_2452_, v_s_u03b2_2453_, v_a_2454_, v_x_2455_);
lean_dec_ref(v_s_u03b1_2450_);
lean_dec_ref(v_00_u03b1_2449_);
lean_dec(v_u_2448_);
return v_res_2456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg___boxed(lean_object* v_u_2457_, lean_object* v_00_u03b1_2458_, lean_object* v_s_u03b1_2459_, lean_object* v_v_2460_, lean_object* v_00_u03b2_2461_, lean_object* v_s_u03b2_2462_, lean_object* v_x_2463_){
_start:
{
lean_object* v_res_2464_; 
v_res_2464_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(v_u_2457_, v_00_u03b1_2458_, v_s_u03b1_2459_, v_v_2460_, v_00_u03b2_2461_, v_s_u03b2_2462_, v_x_2463_);
lean_dec_ref(v_s_u03b1_2459_);
lean_dec_ref(v_00_u03b1_2458_);
lean_dec(v_u_2457_);
return v_res_2464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast___boxed(lean_object* v_u_2465_, lean_object* v_00_u03b1_2466_, lean_object* v_s_u03b1_2467_, lean_object* v_v_2468_, lean_object* v_00_u03b2_2469_, lean_object* v_s_u03b2_2470_, lean_object* v_a_2471_, lean_object* v_x_2472_){
_start:
{
lean_object* v_res_2473_; 
v_res_2473_ = lp_mathlib_Mathlib_Tactic_Ring_ExProd_cast(v_u_2465_, v_00_u03b1_2466_, v_s_u03b1_2467_, v_v_2468_, v_00_u03b2_2469_, v_s_u03b2_2470_, v_a_2471_, v_x_2472_);
lean_dec_ref(v_s_u03b1_2467_);
lean_dec_ref(v_00_u03b1_2466_);
lean_dec(v_u_2465_);
return v_res_2473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast(lean_object* v_u_2474_, lean_object* v_00_u03b1_2475_, lean_object* v_s_u03b1_2476_, lean_object* v_v_2477_, lean_object* v_00_u03b2_2478_, lean_object* v_s_u03b2_2479_, lean_object* v_a_2480_, lean_object* v_x_2481_){
_start:
{
lean_object* v___x_2482_; 
v___x_2482_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(v_u_2474_, v_00_u03b1_2475_, v_s_u03b1_2476_, v_v_2477_, v_00_u03b2_2478_, v_s_u03b2_2479_, v_x_2481_);
return v___x_2482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___boxed(lean_object* v_u_2483_, lean_object* v_00_u03b1_2484_, lean_object* v_s_u03b1_2485_, lean_object* v_v_2486_, lean_object* v_00_u03b2_2487_, lean_object* v_s_u03b2_2488_, lean_object* v_a_2489_, lean_object* v_x_2490_){
_start:
{
lean_object* v_res_2491_; 
v_res_2491_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast(v_u_2483_, v_00_u03b1_2484_, v_s_u03b1_2485_, v_v_2486_, v_00_u03b2_2487_, v_s_u03b2_2488_, v_a_2489_, v_x_2490_);
lean_dec_ref(v_a_2489_);
lean_dec_ref(v_s_u03b1_2485_);
lean_dec_ref(v_00_u03b1_2484_);
lean_dec(v_u_2483_);
return v_res_2491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(lean_object* v_u_2492_, lean_object* v_00_u03b1_2493_, lean_object* v_a_2494_, lean_object* v_x_2495_){
_start:
{
lean_object* v_value_2496_; lean_object* v_hyp_2497_; lean_object* v___x_2498_; 
v_value_2496_ = lean_ctor_get(v_x_2495_, 0);
lean_inc_ref(v_value_2496_);
v_hyp_2497_ = lean_ctor_get(v_x_2495_, 1);
lean_inc(v_hyp_2497_);
lean_dec_ref(v_x_2495_);
v___x_2498_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat(v_u_2492_, v_00_u03b1_2493_, v_value_2496_, v_a_2494_, v_hyp_2497_);
return v___x_2498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(lean_object* v_u_2499_, lean_object* v_00_u03b1_2500_, lean_object* v_a_2501_, lean_object* v_res_2502_){
_start:
{
lean_object* v___x_2503_; 
lean_inc_ref(v_res_2502_);
lean_inc_ref(v_a_2501_);
lean_inc_ref(v_00_u03b1_2500_);
lean_inc(v_u_2499_);
v___x_2503_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ(v_u_2499_, v_00_u03b1_2500_, v_a_2501_, v_res_2502_);
if (lean_obj_tag(v___x_2503_) == 0)
{
lean_object* v___x_2504_; 
lean_dec_ref(v_res_2502_);
lean_dec_ref(v_a_2501_);
lean_dec_ref(v_00_u03b1_2500_);
lean_dec(v_u_2499_);
v___x_2504_ = lean_box(0);
return v___x_2504_;
}
else
{
lean_object* v_val_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2525_; 
v_val_2505_ = lean_ctor_get(v___x_2503_, 0);
v_isSharedCheck_2525_ = !lean_is_exclusive(v___x_2503_);
if (v_isSharedCheck_2525_ == 0)
{
v___x_2507_ = v___x_2503_;
v_isShared_2508_ = v_isSharedCheck_2525_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_val_2505_);
lean_dec(v___x_2503_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2525_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v_fst_2509_; lean_object* v_snd_2510_; lean_object* v___x_2511_; lean_object* v_fst_2512_; lean_object* v_snd_2513_; lean_object* v___x_2515_; uint8_t v_isShared_2516_; uint8_t v_isSharedCheck_2524_; 
v_fst_2509_ = lean_ctor_get(v_val_2505_, 0);
lean_inc(v_fst_2509_);
v_snd_2510_ = lean_ctor_get(v_val_2505_, 1);
lean_inc(v_snd_2510_);
lean_dec(v_val_2505_);
v___x_2511_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq(v_u_2499_, v_00_u03b1_2500_, v_a_2501_, v_res_2502_);
v_fst_2512_ = lean_ctor_get(v___x_2511_, 0);
v_snd_2513_ = lean_ctor_get(v___x_2511_, 1);
v_isSharedCheck_2524_ = !lean_is_exclusive(v___x_2511_);
if (v_isSharedCheck_2524_ == 0)
{
v___x_2515_ = v___x_2511_;
v_isShared_2516_ = v_isSharedCheck_2524_;
goto v_resetjp_2514_;
}
else
{
lean_inc(v_snd_2513_);
lean_inc(v_fst_2512_);
lean_dec(v___x_2511_);
v___x_2515_ = lean_box(0);
v_isShared_2516_ = v_isSharedCheck_2524_;
goto v_resetjp_2514_;
}
v_resetjp_2514_:
{
lean_object* v___x_2518_; 
if (v_isShared_2516_ == 0)
{
lean_ctor_set(v___x_2515_, 1, v_snd_2510_);
lean_ctor_set(v___x_2515_, 0, v_fst_2509_);
v___x_2518_ = v___x_2515_;
goto v_reusejp_2517_;
}
else
{
lean_object* v_reuseFailAlloc_2523_; 
v_reuseFailAlloc_2523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2523_, 0, v_fst_2509_);
lean_ctor_set(v_reuseFailAlloc_2523_, 1, v_snd_2510_);
v___x_2518_ = v_reuseFailAlloc_2523_;
goto v_reusejp_2517_;
}
v_reusejp_2517_:
{
lean_object* v___x_2519_; lean_object* v___x_2521_; 
v___x_2519_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2519_, 0, v_fst_2512_);
lean_ctor_set(v___x_2519_, 1, v___x_2518_);
lean_ctor_set(v___x_2519_, 2, v_snd_2513_);
if (v_isShared_2508_ == 0)
{
lean_ctor_set(v___x_2507_, 0, v___x_2519_);
v___x_2521_ = v___x_2507_;
goto v_reusejp_2520_;
}
else
{
lean_object* v_reuseFailAlloc_2522_; 
v_reuseFailAlloc_2522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2522_, 0, v___x_2519_);
v___x_2521_ = v_reuseFailAlloc_2522_;
goto v_reusejp_2520_;
}
v_reusejp_2520_:
{
return v___x_2521_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(lean_object* v_msg_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_){
_start:
{
lean_object* v_ref_2532_; lean_object* v___x_2533_; lean_object* v_a_2534_; lean_object* v___x_2536_; uint8_t v_isShared_2537_; uint8_t v_isSharedCheck_2542_; 
v_ref_2532_ = lean_ctor_get(v___y_2529_, 5);
v___x_2533_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4_spec__4(v_msg_2526_, v___y_2527_, v___y_2528_, v___y_2529_, v___y_2530_);
v_a_2534_ = lean_ctor_get(v___x_2533_, 0);
v_isSharedCheck_2542_ = !lean_is_exclusive(v___x_2533_);
if (v_isSharedCheck_2542_ == 0)
{
v___x_2536_ = v___x_2533_;
v_isShared_2537_ = v_isSharedCheck_2542_;
goto v_resetjp_2535_;
}
else
{
lean_inc(v_a_2534_);
lean_dec(v___x_2533_);
v___x_2536_ = lean_box(0);
v_isShared_2537_ = v_isSharedCheck_2542_;
goto v_resetjp_2535_;
}
v_resetjp_2535_:
{
lean_object* v___x_2538_; lean_object* v___x_2540_; 
lean_inc(v_ref_2532_);
v___x_2538_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2538_, 0, v_ref_2532_);
lean_ctor_set(v___x_2538_, 1, v_a_2534_);
if (v_isShared_2537_ == 0)
{
lean_ctor_set_tag(v___x_2536_, 1);
lean_ctor_set(v___x_2536_, 0, v___x_2538_);
v___x_2540_ = v___x_2536_;
goto v_reusejp_2539_;
}
else
{
lean_object* v_reuseFailAlloc_2541_; 
v_reuseFailAlloc_2541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2541_, 0, v___x_2538_);
v___x_2540_ = v_reuseFailAlloc_2541_;
goto v_reusejp_2539_;
}
v_reusejp_2539_:
{
return v___x_2540_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg___boxed(lean_object* v_msg_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_){
_start:
{
lean_object* v_res_2549_; 
v_res_2549_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v_msg_2543_, v___y_2544_, v___y_2545_, v___y_2546_, v___y_2547_);
lean_dec(v___y_2547_);
lean_dec_ref(v___y_2546_);
lean_dec(v___y_2545_);
lean_dec_ref(v___y_2544_);
return v_res_2549_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1(void){
_start:
{
lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__0));
v___x_2552_ = l_Lean_stringToMessageData(v___x_2551_);
return v___x_2552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add(lean_object* v_u_2553_, lean_object* v_00_u03b1_2554_, lean_object* v_s_u03b1_2555_, lean_object* v_a_2556_, lean_object* v_b_2557_, lean_object* v_za_2558_, lean_object* v_zb_2559_, lean_object* v_a_2560_, lean_object* v_a_2561_, lean_object* v_a_2562_, lean_object* v_a_2563_){
_start:
{
lean_object* v___y_2566_; lean_object* v_a_2567_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; 
lean_inc_ref_n(v_a_2556_, 2);
lean_inc_ref_n(v_00_u03b1_2554_, 6);
lean_inc_n(v_u_2553_, 4);
v___x_2570_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_2553_, v_00_u03b1_2554_, v_a_2556_, v_za_2558_);
lean_inc_ref_n(v_b_2557_, 2);
v___x_2571_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_2553_, v_00_u03b1_2554_, v_b_2557_, v_zb_2559_);
v___x_2572_ = lean_box(0);
v___x_2573_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2573_, 0, v_u_2553_);
lean_ctor_set(v___x_2573_, 1, v___x_2572_);
v___x_2574_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__7));
lean_inc_ref_n(v___x_2573_, 3);
v___x_2575_ = l_Lean_Expr_const___override(v___x_2574_, v___x_2573_);
v___x_2576_ = l_Lean_Expr_app___override(v___x_2575_, v_00_u03b1_2554_);
v___x_2577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_2578_ = l_Lean_Expr_const___override(v___x_2577_, v___x_2573_);
v___x_2579_ = l_Lean_Expr_app___override(v___x_2578_, v_00_u03b1_2554_);
v___x_2580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_2581_ = l_Lean_Expr_const___override(v___x_2580_, v___x_2573_);
v___x_2582_ = l_Lean_Expr_app___override(v___x_2581_, v_00_u03b1_2554_);
v___x_2583_ = l_Lean_Expr_app___override(v___x_2582_, v_s_u03b1_2555_);
v___x_2584_ = l_Lean_Expr_app___override(v___x_2579_, v___x_2583_);
v___x_2585_ = l_Lean_Expr_app___override(v___x_2576_, v___x_2584_);
lean_inc_ref(v___x_2585_);
v___x_2586_ = lp_mathlib_Mathlib_Meta_NormNum_Result_add(v_u_2553_, v_00_u03b1_2554_, v_a_2556_, v_b_2557_, v___x_2570_, v___x_2571_, v___x_2585_, v_a_2560_, v_a_2561_, v_a_2562_, v_a_2563_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2587_; lean_object* v_isZero_2589_; lean_object* v___y_2590_; lean_object* v___y_2591_; lean_object* v___y_2592_; lean_object* v___y_2593_; 
v_a_2587_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_a_2587_);
lean_dec_ref_known(v___x_2586_, 1);
if (lean_obj_tag(v_a_2587_) == 1)
{
lean_object* v_lit_2620_; lean_object* v_proof_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; uint8_t v___x_2624_; 
v_lit_2620_ = lean_ctor_get(v_a_2587_, 1);
v_proof_2621_ = lean_ctor_get(v_a_2587_, 2);
v___x_2622_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_2620_);
v___x_2623_ = lean_unsigned_to_nat(0u);
v___x_2624_ = lean_nat_dec_eq(v___x_2622_, v___x_2623_);
lean_dec(v___x_2622_);
if (v___x_2624_ == 0)
{
lean_object* v___x_2625_; 
v___x_2625_ = lean_box(0);
v_isZero_2589_ = v___x_2625_;
v___y_2590_ = v_a_2560_;
v___y_2591_ = v_a_2561_;
v___y_2592_ = v_a_2562_;
v___y_2593_ = v_a_2563_;
goto v___jp_2588_;
}
else
{
lean_object* v___x_2626_; 
lean_inc_ref(v_proof_2621_);
v___x_2626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2626_, 0, v_proof_2621_);
v_isZero_2589_ = v___x_2626_;
v___y_2590_ = v_a_2560_;
v___y_2591_ = v_a_2561_;
v___y_2592_ = v_a_2562_;
v___y_2593_ = v_a_2563_;
goto v___jp_2588_;
}
}
else
{
lean_object* v___x_2627_; 
v___x_2627_ = lean_box(0);
v_isZero_2589_ = v___x_2627_;
v___y_2590_ = v_a_2560_;
v___y_2591_ = v_a_2561_;
v___y_2592_ = v_a_2562_;
v___y_2593_ = v_a_2563_;
goto v___jp_2588_;
}
v___jp_2588_:
{
lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; 
v___x_2594_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__2));
lean_inc_ref(v___x_2573_);
lean_inc_n(v_u_2553_, 2);
v___x_2595_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2595_, 0, v_u_2553_);
lean_ctor_set(v___x_2595_, 1, v___x_2573_);
v___x_2596_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2596_, 0, v_u_2553_);
lean_ctor_set(v___x_2596_, 1, v___x_2595_);
v___x_2597_ = l_Lean_Expr_const___override(v___x_2594_, v___x_2596_);
lean_inc_ref_n(v_00_u03b1_2554_, 4);
v___x_2598_ = l_Lean_Expr_app___override(v___x_2597_, v_00_u03b1_2554_);
v___x_2599_ = l_Lean_Expr_app___override(v___x_2598_, v_00_u03b1_2554_);
v___x_2600_ = l_Lean_Expr_app___override(v___x_2599_, v_00_u03b1_2554_);
v___x_2601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__4));
v___x_2602_ = l_Lean_Expr_const___override(v___x_2601_, v___x_2573_);
v___x_2603_ = l_Lean_Expr_app___override(v___x_2602_, v_00_u03b1_2554_);
v___x_2604_ = l_Lean_Expr_app___override(v___x_2603_, v___x_2585_);
v___x_2605_ = l_Lean_Expr_app___override(v___x_2600_, v___x_2604_);
v___x_2606_ = l_Lean_Expr_app___override(v___x_2605_, v_a_2556_);
v___x_2607_ = l_Lean_Expr_app___override(v___x_2606_, v_b_2557_);
v___x_2608_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(v_u_2553_, v_00_u03b1_2554_, v___x_2607_, v_a_2587_);
if (lean_obj_tag(v___x_2608_) == 0)
{
lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v_a_2611_; lean_object* v___x_2613_; uint8_t v_isShared_2614_; uint8_t v_isSharedCheck_2618_; 
lean_dec(v_isZero_2589_);
v___x_2609_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_2610_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_2609_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_);
v_a_2611_ = lean_ctor_get(v___x_2610_, 0);
v_isSharedCheck_2618_ = !lean_is_exclusive(v___x_2610_);
if (v_isSharedCheck_2618_ == 0)
{
v___x_2613_ = v___x_2610_;
v_isShared_2614_ = v_isSharedCheck_2618_;
goto v_resetjp_2612_;
}
else
{
lean_inc(v_a_2611_);
lean_dec(v___x_2610_);
v___x_2613_ = lean_box(0);
v_isShared_2614_ = v_isSharedCheck_2618_;
goto v_resetjp_2612_;
}
v_resetjp_2612_:
{
lean_object* v___x_2616_; 
if (v_isShared_2614_ == 0)
{
v___x_2616_ = v___x_2613_;
goto v_reusejp_2615_;
}
else
{
lean_object* v_reuseFailAlloc_2617_; 
v_reuseFailAlloc_2617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2617_, 0, v_a_2611_);
v___x_2616_ = v_reuseFailAlloc_2617_;
goto v_reusejp_2615_;
}
v_reusejp_2615_:
{
return v___x_2616_;
}
}
}
else
{
lean_object* v_val_2619_; 
v_val_2619_ = lean_ctor_get(v___x_2608_, 0);
lean_inc(v_val_2619_);
lean_dec_ref_known(v___x_2608_, 1);
v___y_2566_ = v_isZero_2589_;
v_a_2567_ = v_val_2619_;
goto v___jp_2565_;
}
}
}
else
{
lean_object* v_a_2628_; lean_object* v___x_2630_; uint8_t v_isShared_2631_; uint8_t v_isSharedCheck_2635_; 
lean_dec_ref(v___x_2585_);
lean_dec_ref_known(v___x_2573_, 2);
lean_dec_ref(v_b_2557_);
lean_dec_ref(v_a_2556_);
lean_dec_ref(v_00_u03b1_2554_);
lean_dec(v_u_2553_);
v_a_2628_ = lean_ctor_get(v___x_2586_, 0);
v_isSharedCheck_2635_ = !lean_is_exclusive(v___x_2586_);
if (v_isSharedCheck_2635_ == 0)
{
v___x_2630_ = v___x_2586_;
v_isShared_2631_ = v_isSharedCheck_2635_;
goto v_resetjp_2629_;
}
else
{
lean_inc(v_a_2628_);
lean_dec(v___x_2586_);
v___x_2630_ = lean_box(0);
v_isShared_2631_ = v_isSharedCheck_2635_;
goto v_resetjp_2629_;
}
v_resetjp_2629_:
{
lean_object* v___x_2633_; 
if (v_isShared_2631_ == 0)
{
v___x_2633_ = v___x_2630_;
goto v_reusejp_2632_;
}
else
{
lean_object* v_reuseFailAlloc_2634_; 
v_reuseFailAlloc_2634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2634_, 0, v_a_2628_);
v___x_2633_ = v_reuseFailAlloc_2634_;
goto v_reusejp_2632_;
}
v_reusejp_2632_:
{
return v___x_2633_;
}
}
}
v___jp_2565_:
{
lean_object* v___x_2568_; lean_object* v___x_2569_; 
v___x_2568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2568_, 0, v_a_2567_);
lean_ctor_set(v___x_2568_, 1, v___y_2566_);
v___x_2569_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2569_, 0, v___x_2568_);
return v___x_2569_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___boxed(lean_object* v_u_2636_, lean_object* v_00_u03b1_2637_, lean_object* v_s_u03b1_2638_, lean_object* v_a_2639_, lean_object* v_b_2640_, lean_object* v_za_2641_, lean_object* v_zb_2642_, lean_object* v_a_2643_, lean_object* v_a_2644_, lean_object* v_a_2645_, lean_object* v_a_2646_, lean_object* v_a_2647_){
_start:
{
lean_object* v_res_2648_; 
v_res_2648_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add(v_u_2636_, v_00_u03b1_2637_, v_s_u03b1_2638_, v_a_2639_, v_b_2640_, v_za_2641_, v_zb_2642_, v_a_2643_, v_a_2644_, v_a_2645_, v_a_2646_);
lean_dec(v_a_2646_);
lean_dec_ref(v_a_2645_);
lean_dec(v_a_2644_);
lean_dec_ref(v_a_2643_);
return v_res_2648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0(lean_object* v_00_u03b1_2649_, lean_object* v_msg_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_){
_start:
{
lean_object* v___x_2656_; 
v___x_2656_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v_msg_2650_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_);
return v___x_2656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___boxed(lean_object* v_00_u03b1_2657_, lean_object* v_msg_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_){
_start:
{
lean_object* v_res_2664_; 
v_res_2664_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0(v_00_u03b1_2657_, v_msg_2658_, v___y_2659_, v___y_2660_, v___y_2661_, v___y_2662_);
lean_dec(v___y_2662_);
lean_dec_ref(v___y_2661_);
lean_dec(v___y_2660_);
lean_dec_ref(v___y_2659_);
return v_res_2664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul(lean_object* v_u_2665_, lean_object* v_00_u03b1_2666_, lean_object* v_s_u03b1_2667_, lean_object* v_a_2668_, lean_object* v_b_2669_, lean_object* v_za_2670_, lean_object* v_zb_2671_, lean_object* v_a_2672_, lean_object* v_a_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_){
_start:
{
lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v___x_2679_; lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; 
lean_inc_ref_n(v_a_2668_, 2);
lean_inc_ref_n(v_00_u03b1_2666_, 4);
lean_inc_n(v_u_2665_, 4);
v___x_2677_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_2665_, v_00_u03b1_2666_, v_a_2668_, v_za_2670_);
lean_inc_ref_n(v_b_2669_, 2);
v___x_2678_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_2665_, v_00_u03b1_2666_, v_b_2669_, v_zb_2671_);
v___x_2679_ = lean_box(0);
v___x_2680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2680_, 0, v_u_2665_);
lean_ctor_set(v___x_2680_, 1, v___x_2679_);
v___x_2681_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
lean_inc_ref(v___x_2680_);
v___x_2682_ = l_Lean_Expr_const___override(v___x_2681_, v___x_2680_);
v___x_2683_ = l_Lean_Expr_app___override(v___x_2682_, v_00_u03b1_2666_);
v___x_2684_ = l_Lean_Expr_app___override(v___x_2683_, v_s_u03b1_2667_);
lean_inc_ref(v___x_2684_);
v___x_2685_ = lp_mathlib_Mathlib_Meta_NormNum_Result_mul(v_u_2665_, v_00_u03b1_2666_, v_a_2668_, v_b_2669_, v___x_2677_, v___x_2678_, v___x_2684_, v_a_2672_, v_a_2673_, v_a_2674_, v_a_2675_);
if (lean_obj_tag(v___x_2685_) == 0)
{
lean_object* v_a_2686_; lean_object* v___x_2688_; uint8_t v_isShared_2689_; uint8_t v_isSharedCheck_2719_; 
v_a_2686_ = lean_ctor_get(v___x_2685_, 0);
v_isSharedCheck_2719_ = !lean_is_exclusive(v___x_2685_);
if (v_isSharedCheck_2719_ == 0)
{
v___x_2688_ = v___x_2685_;
v_isShared_2689_ = v_isSharedCheck_2719_;
goto v_resetjp_2687_;
}
else
{
lean_inc(v_a_2686_);
lean_dec(v___x_2685_);
v___x_2688_ = lean_box(0);
v_isShared_2689_ = v_isSharedCheck_2719_;
goto v_resetjp_2687_;
}
v_resetjp_2687_:
{
lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; 
v___x_2690_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__4));
lean_inc_ref_n(v___x_2680_, 3);
lean_inc_n(v_u_2665_, 2);
v___x_2691_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2691_, 0, v_u_2665_);
lean_ctor_set(v___x_2691_, 1, v___x_2680_);
v___x_2692_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2692_, 0, v_u_2665_);
lean_ctor_set(v___x_2692_, 1, v___x_2691_);
v___x_2693_ = l_Lean_Expr_const___override(v___x_2690_, v___x_2692_);
lean_inc_ref_n(v_00_u03b1_2666_, 6);
v___x_2694_ = l_Lean_Expr_app___override(v___x_2693_, v_00_u03b1_2666_);
v___x_2695_ = l_Lean_Expr_app___override(v___x_2694_, v_00_u03b1_2666_);
v___x_2696_ = l_Lean_Expr_app___override(v___x_2695_, v_00_u03b1_2666_);
v___x_2697_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__6));
v___x_2698_ = l_Lean_Expr_const___override(v___x_2697_, v___x_2680_);
v___x_2699_ = l_Lean_Expr_app___override(v___x_2698_, v_00_u03b1_2666_);
v___x_2700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8));
v___x_2701_ = l_Lean_Expr_const___override(v___x_2700_, v___x_2680_);
v___x_2702_ = l_Lean_Expr_app___override(v___x_2701_, v_00_u03b1_2666_);
v___x_2703_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_2704_ = l_Lean_Expr_const___override(v___x_2703_, v___x_2680_);
v___x_2705_ = l_Lean_Expr_app___override(v___x_2704_, v_00_u03b1_2666_);
v___x_2706_ = l_Lean_Expr_app___override(v___x_2705_, v___x_2684_);
v___x_2707_ = l_Lean_Expr_app___override(v___x_2702_, v___x_2706_);
v___x_2708_ = l_Lean_Expr_app___override(v___x_2699_, v___x_2707_);
v___x_2709_ = l_Lean_Expr_app___override(v___x_2696_, v___x_2708_);
v___x_2710_ = l_Lean_Expr_app___override(v___x_2709_, v_a_2668_);
v___x_2711_ = l_Lean_Expr_app___override(v___x_2710_, v_b_2669_);
v___x_2712_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(v_u_2665_, v_00_u03b1_2666_, v___x_2711_, v_a_2686_);
if (lean_obj_tag(v___x_2712_) == 0)
{
lean_object* v___x_2713_; lean_object* v___x_2714_; 
lean_del_object(v___x_2688_);
v___x_2713_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_2714_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_2713_, v_a_2672_, v_a_2673_, v_a_2674_, v_a_2675_);
return v___x_2714_;
}
else
{
lean_object* v_val_2715_; lean_object* v___x_2717_; 
v_val_2715_ = lean_ctor_get(v___x_2712_, 0);
lean_inc(v_val_2715_);
lean_dec_ref_known(v___x_2712_, 1);
if (v_isShared_2689_ == 0)
{
lean_ctor_set(v___x_2688_, 0, v_val_2715_);
v___x_2717_ = v___x_2688_;
goto v_reusejp_2716_;
}
else
{
lean_object* v_reuseFailAlloc_2718_; 
v_reuseFailAlloc_2718_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2718_, 0, v_val_2715_);
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
lean_dec_ref(v___x_2684_);
lean_dec_ref_known(v___x_2680_, 2);
lean_dec_ref(v_b_2669_);
lean_dec_ref(v_a_2668_);
lean_dec_ref(v_00_u03b1_2666_);
lean_dec(v_u_2665_);
v_a_2720_ = lean_ctor_get(v___x_2685_, 0);
v_isSharedCheck_2727_ = !lean_is_exclusive(v___x_2685_);
if (v_isSharedCheck_2727_ == 0)
{
v___x_2722_ = v___x_2685_;
v_isShared_2723_ = v_isSharedCheck_2727_;
goto v_resetjp_2721_;
}
else
{
lean_inc(v_a_2720_);
lean_dec(v___x_2685_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul___boxed(lean_object* v_u_2728_, lean_object* v_00_u03b1_2729_, lean_object* v_s_u03b1_2730_, lean_object* v_a_2731_, lean_object* v_b_2732_, lean_object* v_za_2733_, lean_object* v_zb_2734_, lean_object* v_a_2735_, lean_object* v_a_2736_, lean_object* v_a_2737_, lean_object* v_a_2738_, lean_object* v_a_2739_){
_start:
{
lean_object* v_res_2740_; 
v_res_2740_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul(v_u_2728_, v_00_u03b1_2729_, v_s_u03b1_2730_, v_a_2731_, v_b_2732_, v_za_2733_, v_zb_2734_, v_a_2735_, v_a_2736_, v_a_2737_, v_a_2738_);
lean_dec(v_a_2738_);
lean_dec_ref(v_a_2737_);
lean_dec(v_a_2736_);
lean_dec_ref(v_a_2735_);
return v_res_2740_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0(void){
_start:
{
lean_object* v___x_2741_; lean_object* v___x_2742_; 
v___x_2741_ = lean_unsigned_to_nat(1u);
v___x_2742_ = lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Ring_ExProd_mkNat_spec__0(v___x_2741_);
return v___x_2742_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2(void){
_start:
{
lean_object* v___x_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; 
v___x_2744_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_2745_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__12));
v___x_2746_ = l_Lean_Expr_const___override(v___x_2745_, v___x_2744_);
return v___x_2746_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3(void){
_start:
{
lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; 
v___x_2747_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_2748_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__2);
v___x_2749_ = l_Lean_Expr_app___override(v___x_2748_, v___x_2747_);
return v___x_2749_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5(void){
_start:
{
lean_object* v___x_2752_; lean_object* v___x_2753_; 
v___x_2752_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__4));
v___x_2753_ = l_Lean_Expr_lit___override(v___x_2752_);
return v___x_2753_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6(void){
_start:
{
lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2756_; 
v___x_2754_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5);
v___x_2755_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__3);
v___x_2756_ = l_Lean_Expr_app___override(v___x_2755_, v___x_2754_);
return v___x_2756_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9(void){
_start:
{
lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; 
v___x_2760_ = lean_box(0);
v___x_2761_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__8));
v___x_2762_ = l_Lean_Expr_const___override(v___x_2761_, v___x_2760_);
return v___x_2762_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10(void){
_start:
{
lean_object* v___x_2763_; lean_object* v___x_2764_; lean_object* v___x_2765_; 
v___x_2763_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5);
v___x_2764_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__9);
v___x_2765_ = l_Lean_Expr_app___override(v___x_2764_, v___x_2763_);
return v___x_2765_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11(void){
_start:
{
lean_object* v___x_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; 
v___x_2766_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__10);
v___x_2767_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__6);
v___x_2768_ = l_Lean_Expr_app___override(v___x_2767_, v___x_2766_);
return v___x_2768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne(lean_object* v_u_2779_, lean_object* v_00_u03b1_2780_, lean_object* v_s_u03b1_2781_, lean_object* v_x_2782_, lean_object* v_zx_2783_){
_start:
{
lean_object* v_value_2784_; lean_object* v___x_2786_; uint8_t v_isShared_2787_; uint8_t v_isSharedCheck_2826_; 
v_value_2784_ = lean_ctor_get(v_zx_2783_, 0);
v_isSharedCheck_2826_ = !lean_is_exclusive(v_zx_2783_);
if (v_isSharedCheck_2826_ == 0)
{
lean_object* v_unused_2827_; 
v_unused_2827_ = lean_ctor_get(v_zx_2783_, 1);
lean_dec(v_unused_2827_);
v___x_2786_ = v_zx_2783_;
v_isShared_2787_ = v_isSharedCheck_2826_;
goto v_resetjp_2785_;
}
else
{
lean_inc(v_value_2784_);
lean_dec(v_zx_2783_);
v___x_2786_ = lean_box(0);
v_isShared_2787_ = v_isSharedCheck_2826_;
goto v_resetjp_2785_;
}
v_resetjp_2785_:
{
lean_object* v___x_2788_; uint8_t v___x_2789_; 
v___x_2788_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0);
v___x_2789_ = l_instDecidableEqRat_decEq(v_value_2784_, v___x_2788_);
lean_dec_ref(v_value_2784_);
if (v___x_2789_ == 0)
{
lean_object* v___x_2790_; 
lean_del_object(v___x_2786_);
lean_dec_ref(v_x_2782_);
lean_dec_ref(v_s_u03b1_2781_);
lean_dec_ref(v_00_u03b1_2780_);
lean_dec(v_u_2779_);
v___x_2790_ = lean_box(0);
return v___x_2790_;
}
else
{
lean_object* v___x_2791_; lean_object* v___x_2793_; 
v___x_2791_ = lean_box(0);
lean_inc(v_u_2779_);
if (v_isShared_2787_ == 0)
{
lean_ctor_set_tag(v___x_2786_, 1);
lean_ctor_set(v___x_2786_, 1, v___x_2791_);
lean_ctor_set(v___x_2786_, 0, v_u_2779_);
v___x_2793_ = v___x_2786_;
goto v_reusejp_2792_;
}
else
{
lean_object* v_reuseFailAlloc_2825_; 
v_reuseFailAlloc_2825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2825_, 0, v_u_2779_);
lean_ctor_set(v_reuseFailAlloc_2825_, 1, v___x_2791_);
v___x_2793_ = v_reuseFailAlloc_2825_;
goto v_reusejp_2792_;
}
v_reusejp_2792_:
{
lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; lean_object* v___x_2819_; lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; lean_object* v___x_2824_; 
v___x_2794_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5));
lean_inc_ref_n(v___x_2793_, 4);
v___x_2795_ = l_Lean_Expr_const___override(v___x_2794_, v___x_2793_);
lean_inc_ref_n(v_00_u03b1_2780_, 5);
v___x_2796_ = l_Lean_Expr_app___override(v___x_2795_, v_00_u03b1_2780_);
v___x_2797_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8));
v___x_2798_ = l_Lean_Expr_const___override(v___x_2797_, v___x_2793_);
v___x_2799_ = l_Lean_Expr_app___override(v___x_2798_, v_00_u03b1_2780_);
v___x_2800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11));
v___x_2801_ = l_Lean_Expr_const___override(v___x_2800_, v___x_2793_);
v___x_2802_ = l_Lean_Expr_app___override(v___x_2801_, v_00_u03b1_2780_);
v___x_2803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_2804_ = l_Lean_Expr_const___override(v___x_2803_, v___x_2793_);
v___x_2805_ = l_Lean_Expr_app___override(v___x_2804_, v_00_u03b1_2780_);
v___x_2806_ = l_Lean_Expr_app___override(v___x_2805_, v_s_u03b1_2781_);
v___x_2807_ = l_Lean_Expr_app___override(v___x_2802_, v___x_2806_);
v___x_2808_ = l_Lean_Expr_app___override(v___x_2799_, v___x_2807_);
v___x_2809_ = l_Lean_Expr_app___override(v___x_2796_, v___x_2808_);
v___x_2810_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__11);
v___x_2811_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__13));
v___x_2812_ = l_Lean_Expr_const___override(v___x_2811_, v___x_2793_);
v___x_2813_ = l_Lean_Expr_app___override(v___x_2812_, v_00_u03b1_2780_);
v___x_2814_ = l_Lean_Expr_app___override(v___x_2813_, v___x_2809_);
lean_inc_ref(v_x_2782_);
v___x_2815_ = l_Lean_Expr_app___override(v___x_2814_, v_x_2782_);
v___x_2816_ = l_Lean_Expr_app___override(v___x_2815_, v___x_2810_);
v___x_2817_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__15));
v___x_2818_ = l_Lean_Level_succ___override(v_u_2779_);
v___x_2819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2819_, 0, v___x_2818_);
lean_ctor_set(v___x_2819_, 1, v___x_2791_);
v___x_2820_ = l_Lean_Expr_const___override(v___x_2817_, v___x_2819_);
v___x_2821_ = l_Lean_Expr_app___override(v___x_2820_, v_00_u03b1_2780_);
v___x_2822_ = l_Lean_Expr_app___override(v___x_2821_, v_x_2782_);
v___x_2823_ = l_Lean_Expr_app___override(v___x_2816_, v___x_2822_);
v___x_2824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2824_, 0, v___x_2823_);
return v___x_2824_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive(lean_object* v_u_2828_, lean_object* v_00_u03b1_2829_, lean_object* v_s_u03b1_2830_, lean_object* v_x_2831_, lean_object* v_a_2832_, lean_object* v_a_2833_, lean_object* v_a_2834_, lean_object* v_a_2835_){
_start:
{
lean_object* v_a_2838_; uint8_t v___x_2850_; lean_object* v___x_2851_; 
v___x_2850_ = 0;
lean_inc_ref(v_x_2831_);
lean_inc_ref(v_00_u03b1_2829_);
lean_inc(v_u_2828_);
v___x_2851_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_2828_, v_00_u03b1_2829_, v_x_2831_, v___x_2850_, v_a_2832_, v_a_2833_, v_a_2834_, v_a_2835_);
if (lean_obj_tag(v___x_2851_) == 0)
{
lean_object* v_a_2852_; lean_object* v___x_2853_; 
v_a_2852_ = lean_ctor_get(v___x_2851_, 0);
lean_inc(v_a_2852_);
lean_dec_ref_known(v___x_2851_, 1);
v___x_2853_ = lp_mathlib_Mathlib_Tactic_Ring_evalCast(v_u_2828_, v_00_u03b1_2829_, v_s_u03b1_2830_, v_x_2831_, v_a_2852_);
if (lean_obj_tag(v___x_2853_) == 0)
{
lean_object* v___x_2854_; lean_object* v___x_2855_; 
v___x_2854_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_2855_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_2854_, v_a_2832_, v_a_2833_, v_a_2834_, v_a_2835_);
return v___x_2855_;
}
else
{
lean_object* v_val_2856_; 
v_val_2856_ = lean_ctor_get(v___x_2853_, 0);
lean_inc(v_val_2856_);
lean_dec_ref_known(v___x_2853_, 1);
v_a_2838_ = v_val_2856_;
goto v___jp_2837_;
}
}
else
{
lean_object* v_a_2857_; lean_object* v___x_2859_; uint8_t v_isShared_2860_; uint8_t v_isSharedCheck_2864_; 
lean_dec_ref(v_x_2831_);
lean_dec_ref(v_s_u03b1_2830_);
lean_dec_ref(v_00_u03b1_2829_);
lean_dec(v_u_2828_);
v_a_2857_ = lean_ctor_get(v___x_2851_, 0);
v_isSharedCheck_2864_ = !lean_is_exclusive(v___x_2851_);
if (v_isSharedCheck_2864_ == 0)
{
v___x_2859_ = v___x_2851_;
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
else
{
lean_inc(v_a_2857_);
lean_dec(v___x_2851_);
v___x_2859_ = lean_box(0);
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
v_resetjp_2858_:
{
lean_object* v___x_2862_; 
if (v_isShared_2860_ == 0)
{
v___x_2862_ = v___x_2859_;
goto v_reusejp_2861_;
}
else
{
lean_object* v_reuseFailAlloc_2863_; 
v_reuseFailAlloc_2863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2863_, 0, v_a_2857_);
v___x_2862_ = v_reuseFailAlloc_2863_;
goto v_reusejp_2861_;
}
v_reusejp_2861_:
{
return v___x_2862_;
}
}
}
v___jp_2837_:
{
lean_object* v_expr_2839_; lean_object* v_val_2840_; lean_object* v_proof_2841_; lean_object* v___x_2843_; uint8_t v_isShared_2844_; uint8_t v_isSharedCheck_2849_; 
v_expr_2839_ = lean_ctor_get(v_a_2838_, 0);
v_val_2840_ = lean_ctor_get(v_a_2838_, 1);
v_proof_2841_ = lean_ctor_get(v_a_2838_, 2);
v_isSharedCheck_2849_ = !lean_is_exclusive(v_a_2838_);
if (v_isSharedCheck_2849_ == 0)
{
v___x_2843_ = v_a_2838_;
v_isShared_2844_ = v_isSharedCheck_2849_;
goto v_resetjp_2842_;
}
else
{
lean_inc(v_proof_2841_);
lean_inc(v_val_2840_);
lean_inc(v_expr_2839_);
lean_dec(v_a_2838_);
v___x_2843_ = lean_box(0);
v_isShared_2844_ = v_isSharedCheck_2849_;
goto v_resetjp_2842_;
}
v_resetjp_2842_:
{
lean_object* v___x_2846_; 
if (v_isShared_2844_ == 0)
{
v___x_2846_ = v___x_2843_;
goto v_reusejp_2845_;
}
else
{
lean_object* v_reuseFailAlloc_2848_; 
v_reuseFailAlloc_2848_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2848_, 0, v_expr_2839_);
lean_ctor_set(v_reuseFailAlloc_2848_, 1, v_val_2840_);
lean_ctor_set(v_reuseFailAlloc_2848_, 2, v_proof_2841_);
v___x_2846_ = v_reuseFailAlloc_2848_;
goto v_reusejp_2845_;
}
v_reusejp_2845_:
{
lean_object* v___x_2847_; 
v___x_2847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2847_, 0, v___x_2846_);
return v___x_2847_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive___boxed(lean_object* v_u_2865_, lean_object* v_00_u03b1_2866_, lean_object* v_s_u03b1_2867_, lean_object* v_x_2868_, lean_object* v_a_2869_, lean_object* v_a_2870_, lean_object* v_a_2871_, lean_object* v_a_2872_, lean_object* v_a_2873_){
_start:
{
lean_object* v_res_2874_; 
v_res_2874_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive(v_u_2865_, v_00_u03b1_2866_, v_s_u03b1_2867_, v_x_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_);
lean_dec(v_a_2872_);
lean_dec_ref(v_a_2871_);
lean_dec(v_a_2870_);
lean_dec_ref(v_a_2869_);
return v_res_2874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg(lean_object* v_x_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_){
_start:
{
lean_object* v___x_2881_; 
v___x_2881_ = l_Lean_Meta_saveState___redArg(v___y_2877_, v___y_2879_);
if (lean_obj_tag(v___x_2881_) == 0)
{
lean_object* v_a_2882_; lean_object* v___x_2883_; 
v_a_2882_ = lean_ctor_get(v___x_2881_, 0);
lean_inc(v_a_2882_);
lean_dec_ref_known(v___x_2881_, 1);
lean_inc(v___y_2879_);
lean_inc_ref(v___y_2878_);
lean_inc(v___y_2877_);
lean_inc_ref(v___y_2876_);
v___x_2883_ = lean_apply_5(v_x_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_, lean_box(0));
if (lean_obj_tag(v___x_2883_) == 0)
{
lean_object* v_a_2884_; lean_object* v___x_2886_; uint8_t v_isShared_2887_; uint8_t v_isSharedCheck_2892_; 
lean_dec(v_a_2882_);
v_a_2884_ = lean_ctor_get(v___x_2883_, 0);
v_isSharedCheck_2892_ = !lean_is_exclusive(v___x_2883_);
if (v_isSharedCheck_2892_ == 0)
{
v___x_2886_ = v___x_2883_;
v_isShared_2887_ = v_isSharedCheck_2892_;
goto v_resetjp_2885_;
}
else
{
lean_inc(v_a_2884_);
lean_dec(v___x_2883_);
v___x_2886_ = lean_box(0);
v_isShared_2887_ = v_isSharedCheck_2892_;
goto v_resetjp_2885_;
}
v_resetjp_2885_:
{
lean_object* v___x_2888_; lean_object* v___x_2890_; 
v___x_2888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2888_, 0, v_a_2884_);
if (v_isShared_2887_ == 0)
{
lean_ctor_set(v___x_2886_, 0, v___x_2888_);
v___x_2890_ = v___x_2886_;
goto v_reusejp_2889_;
}
else
{
lean_object* v_reuseFailAlloc_2891_; 
v_reuseFailAlloc_2891_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2891_, 0, v___x_2888_);
v___x_2890_ = v_reuseFailAlloc_2891_;
goto v_reusejp_2889_;
}
v_reusejp_2889_:
{
return v___x_2890_;
}
}
}
else
{
lean_object* v_a_2893_; lean_object* v___x_2895_; uint8_t v_isShared_2896_; uint8_t v_isSharedCheck_2922_; 
v_a_2893_ = lean_ctor_get(v___x_2883_, 0);
v_isSharedCheck_2922_ = !lean_is_exclusive(v___x_2883_);
if (v_isSharedCheck_2922_ == 0)
{
v___x_2895_ = v___x_2883_;
v_isShared_2896_ = v_isSharedCheck_2922_;
goto v_resetjp_2894_;
}
else
{
lean_inc(v_a_2893_);
lean_dec(v___x_2883_);
v___x_2895_ = lean_box(0);
v_isShared_2896_ = v_isSharedCheck_2922_;
goto v_resetjp_2894_;
}
v_resetjp_2894_:
{
uint8_t v___y_2898_; uint8_t v___x_2920_; 
v___x_2920_ = l_Lean_Exception_isInterrupt(v_a_2893_);
if (v___x_2920_ == 0)
{
uint8_t v___x_2921_; 
lean_inc(v_a_2893_);
v___x_2921_ = l_Lean_Exception_isRuntime(v_a_2893_);
v___y_2898_ = v___x_2921_;
goto v___jp_2897_;
}
else
{
v___y_2898_ = v___x_2920_;
goto v___jp_2897_;
}
v___jp_2897_:
{
if (v___y_2898_ == 0)
{
lean_object* v___x_2899_; 
lean_del_object(v___x_2895_);
lean_dec(v_a_2893_);
v___x_2899_ = l_Lean_Meta_SavedState_restore___redArg(v_a_2882_, v___y_2877_, v___y_2879_);
lean_dec(v_a_2882_);
if (lean_obj_tag(v___x_2899_) == 0)
{
lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2907_; 
v_isSharedCheck_2907_ = !lean_is_exclusive(v___x_2899_);
if (v_isSharedCheck_2907_ == 0)
{
lean_object* v_unused_2908_; 
v_unused_2908_ = lean_ctor_get(v___x_2899_, 0);
lean_dec(v_unused_2908_);
v___x_2901_ = v___x_2899_;
v_isShared_2902_ = v_isSharedCheck_2907_;
goto v_resetjp_2900_;
}
else
{
lean_dec(v___x_2899_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2907_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
lean_object* v___x_2903_; lean_object* v___x_2905_; 
v___x_2903_ = lean_box(0);
if (v_isShared_2902_ == 0)
{
lean_ctor_set(v___x_2901_, 0, v___x_2903_);
v___x_2905_ = v___x_2901_;
goto v_reusejp_2904_;
}
else
{
lean_object* v_reuseFailAlloc_2906_; 
v_reuseFailAlloc_2906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2906_, 0, v___x_2903_);
v___x_2905_ = v_reuseFailAlloc_2906_;
goto v_reusejp_2904_;
}
v_reusejp_2904_:
{
return v___x_2905_;
}
}
}
else
{
lean_object* v_a_2909_; lean_object* v___x_2911_; uint8_t v_isShared_2912_; uint8_t v_isSharedCheck_2916_; 
v_a_2909_ = lean_ctor_get(v___x_2899_, 0);
v_isSharedCheck_2916_ = !lean_is_exclusive(v___x_2899_);
if (v_isSharedCheck_2916_ == 0)
{
v___x_2911_ = v___x_2899_;
v_isShared_2912_ = v_isSharedCheck_2916_;
goto v_resetjp_2910_;
}
else
{
lean_inc(v_a_2909_);
lean_dec(v___x_2899_);
v___x_2911_ = lean_box(0);
v_isShared_2912_ = v_isSharedCheck_2916_;
goto v_resetjp_2910_;
}
v_resetjp_2910_:
{
lean_object* v___x_2914_; 
if (v_isShared_2912_ == 0)
{
v___x_2914_ = v___x_2911_;
goto v_reusejp_2913_;
}
else
{
lean_object* v_reuseFailAlloc_2915_; 
v_reuseFailAlloc_2915_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2915_, 0, v_a_2909_);
v___x_2914_ = v_reuseFailAlloc_2915_;
goto v_reusejp_2913_;
}
v_reusejp_2913_:
{
return v___x_2914_;
}
}
}
}
else
{
lean_object* v___x_2918_; 
lean_dec(v_a_2882_);
if (v_isShared_2896_ == 0)
{
v___x_2918_ = v___x_2895_;
goto v_reusejp_2917_;
}
else
{
lean_object* v_reuseFailAlloc_2919_; 
v_reuseFailAlloc_2919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2919_, 0, v_a_2893_);
v___x_2918_ = v_reuseFailAlloc_2919_;
goto v_reusejp_2917_;
}
v_reusejp_2917_:
{
return v___x_2918_;
}
}
}
}
}
}
else
{
lean_object* v_a_2923_; lean_object* v___x_2925_; uint8_t v_isShared_2926_; uint8_t v_isSharedCheck_2930_; 
lean_dec_ref(v_x_2875_);
v_a_2923_ = lean_ctor_get(v___x_2881_, 0);
v_isSharedCheck_2930_ = !lean_is_exclusive(v___x_2881_);
if (v_isSharedCheck_2930_ == 0)
{
v___x_2925_ = v___x_2881_;
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
else
{
lean_inc(v_a_2923_);
lean_dec(v___x_2881_);
v___x_2925_ = lean_box(0);
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
v_resetjp_2924_:
{
lean_object* v___x_2928_; 
if (v_isShared_2926_ == 0)
{
v___x_2928_ = v___x_2925_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v_a_2923_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
return v___x_2928_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg___boxed(lean_object* v_x_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_, lean_object* v___y_2935_, lean_object* v___y_2936_){
_start:
{
lean_object* v_res_2937_; 
v_res_2937_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg(v_x_2931_, v___y_2932_, v___y_2933_, v___y_2934_, v___y_2935_);
lean_dec(v___y_2935_);
lean_dec_ref(v___y_2934_);
lean_dec(v___y_2933_);
lean_dec_ref(v___y_2932_);
return v_res_2937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0(lean_object* v_00_u03b1_2938_, lean_object* v_x_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_){
_start:
{
lean_object* v___x_2945_; 
v___x_2945_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg(v_x_2939_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_);
return v___x_2945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___boxed(lean_object* v_00_u03b1_2946_, lean_object* v_x_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_){
_start:
{
lean_object* v_res_2953_; 
v_res_2953_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0(v_00_u03b1_2946_, v_x_2947_, v___y_2948_, v___y_2949_, v___y_2950_, v___y_2951_);
lean_dec(v___y_2951_);
lean_dec_ref(v___y_2950_);
lean_dec(v___y_2949_);
lean_dec_ref(v___y_2948_);
return v_res_2953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg(lean_object* v_u_2989_, lean_object* v_00_u03b1_2990_, lean_object* v_a_2991_, lean_object* v_cz_u03b1_2992_, lean_object* v___sf_u03b1_2993_, lean_object* v_za_2994_, lean_object* v_a_2995_, lean_object* v_a_2996_, lean_object* v_a_2997_, lean_object* v_a_2998_){
_start:
{
lean_object* v_a_3001_; lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; 
lean_inc_ref_n(v_a_2991_, 2);
lean_inc_ref_n(v_00_u03b1_2990_, 3);
lean_inc_n(v_u_2989_, 3);
v___x_3014_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_2989_, v_00_u03b1_2990_, v_a_2991_, v_za_2994_);
v___x_3015_ = lean_box(0);
v___x_3016_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3016_, 0, v_u_2989_);
lean_ctor_set(v___x_3016_, 1, v___x_3015_);
v___x_3017_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__2));
lean_inc_ref(v___x_3016_);
v___x_3018_ = l_Lean_Expr_const___override(v___x_3017_, v___x_3016_);
v___x_3019_ = l_Lean_Expr_app___override(v___x_3018_, v_00_u03b1_2990_);
v___x_3020_ = l_Lean_Expr_app___override(v___x_3019_, v___sf_u03b1_2993_);
lean_inc_ref(v___x_3020_);
v___x_3021_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_Result_inv___boxed), 11, 6);
lean_closure_set(v___x_3021_, 0, v_u_2989_);
lean_closure_set(v___x_3021_, 1, v_00_u03b1_2990_);
lean_closure_set(v___x_3021_, 2, v_a_2991_);
lean_closure_set(v___x_3021_, 3, v___x_3014_);
lean_closure_set(v___x_3021_, 4, v___x_3020_);
lean_closure_set(v___x_3021_, 5, v_cz_u03b1_2992_);
v___x_3022_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Ring_RingCompute_inv_spec__0___redArg(v___x_3021_, v_a_2995_, v_a_2996_, v_a_2997_, v_a_2998_);
if (lean_obj_tag(v___x_3022_) == 0)
{
lean_object* v_a_3023_; lean_object* v___x_3025_; uint8_t v_isShared_3026_; uint8_t v_isSharedCheck_3069_; 
v_a_3023_ = lean_ctor_get(v___x_3022_, 0);
v_isSharedCheck_3069_ = !lean_is_exclusive(v___x_3022_);
if (v_isSharedCheck_3069_ == 0)
{
v___x_3025_ = v___x_3022_;
v_isShared_3026_ = v_isSharedCheck_3069_;
goto v_resetjp_3024_;
}
else
{
lean_inc(v_a_3023_);
lean_dec(v___x_3022_);
v___x_3025_ = lean_box(0);
v_isShared_3026_ = v_isSharedCheck_3069_;
goto v_resetjp_3024_;
}
v_resetjp_3024_:
{
if (lean_obj_tag(v_a_3023_) == 0)
{
lean_object* v___x_3027_; lean_object* v___x_3029_; 
lean_dec_ref(v___x_3020_);
lean_dec_ref_known(v___x_3016_, 2);
lean_dec_ref(v_a_2991_);
lean_dec_ref(v_00_u03b1_2990_);
lean_dec(v_u_2989_);
v___x_3027_ = lean_box(0);
if (v_isShared_3026_ == 0)
{
lean_ctor_set(v___x_3025_, 0, v___x_3027_);
v___x_3029_ = v___x_3025_;
goto v_reusejp_3028_;
}
else
{
lean_object* v_reuseFailAlloc_3030_; 
v_reuseFailAlloc_3030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3030_, 0, v___x_3027_);
v___x_3029_ = v_reuseFailAlloc_3030_;
goto v_reusejp_3028_;
}
v_reusejp_3028_:
{
return v___x_3029_;
}
}
else
{
lean_object* v_val_3031_; lean_object* v___x_3032_; lean_object* v___x_3033_; lean_object* v___x_3034_; lean_object* v___x_3035_; lean_object* v___x_3036_; lean_object* v___x_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; lean_object* v___x_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; 
lean_del_object(v___x_3025_);
v_val_3031_ = lean_ctor_get(v_a_3023_, 0);
lean_inc(v_val_3031_);
lean_dec_ref_known(v_a_3023_, 1);
v___x_3032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__6));
lean_inc_ref_n(v___x_3016_, 5);
v___x_3033_ = l_Lean_Expr_const___override(v___x_3032_, v___x_3016_);
lean_inc_ref_n(v_00_u03b1_2990_, 6);
v___x_3034_ = l_Lean_Expr_app___override(v___x_3033_, v_00_u03b1_2990_);
v___x_3035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__9));
v___x_3036_ = l_Lean_Expr_const___override(v___x_3035_, v___x_3016_);
v___x_3037_ = l_Lean_Expr_app___override(v___x_3036_, v_00_u03b1_2990_);
v___x_3038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__12));
v___x_3039_ = l_Lean_Expr_const___override(v___x_3038_, v___x_3016_);
v___x_3040_ = l_Lean_Expr_app___override(v___x_3039_, v_00_u03b1_2990_);
v___x_3041_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__15));
v___x_3042_ = l_Lean_Expr_const___override(v___x_3041_, v___x_3016_);
v___x_3043_ = l_Lean_Expr_app___override(v___x_3042_, v_00_u03b1_2990_);
v___x_3044_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__18));
v___x_3045_ = l_Lean_Expr_const___override(v___x_3044_, v___x_3016_);
v___x_3046_ = l_Lean_Expr_app___override(v___x_3045_, v_00_u03b1_2990_);
v___x_3047_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___closed__20));
v___x_3048_ = l_Lean_Expr_const___override(v___x_3047_, v___x_3016_);
v___x_3049_ = l_Lean_Expr_app___override(v___x_3048_, v_00_u03b1_2990_);
v___x_3050_ = l_Lean_Expr_app___override(v___x_3049_, v___x_3020_);
v___x_3051_ = l_Lean_Expr_app___override(v___x_3046_, v___x_3050_);
v___x_3052_ = l_Lean_Expr_app___override(v___x_3043_, v___x_3051_);
v___x_3053_ = l_Lean_Expr_app___override(v___x_3040_, v___x_3052_);
v___x_3054_ = l_Lean_Expr_app___override(v___x_3037_, v___x_3053_);
v___x_3055_ = l_Lean_Expr_app___override(v___x_3034_, v___x_3054_);
v___x_3056_ = l_Lean_Expr_app___override(v___x_3055_, v_a_2991_);
v___x_3057_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(v_u_2989_, v_00_u03b1_2990_, v___x_3056_, v_val_3031_);
if (lean_obj_tag(v___x_3057_) == 0)
{
lean_object* v___x_3058_; lean_object* v___x_3059_; lean_object* v_a_3060_; lean_object* v___x_3062_; uint8_t v_isShared_3063_; uint8_t v_isSharedCheck_3067_; 
v___x_3058_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_3059_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_3058_, v_a_2995_, v_a_2996_, v_a_2997_, v_a_2998_);
v_a_3060_ = lean_ctor_get(v___x_3059_, 0);
v_isSharedCheck_3067_ = !lean_is_exclusive(v___x_3059_);
if (v_isSharedCheck_3067_ == 0)
{
v___x_3062_ = v___x_3059_;
v_isShared_3063_ = v_isSharedCheck_3067_;
goto v_resetjp_3061_;
}
else
{
lean_inc(v_a_3060_);
lean_dec(v___x_3059_);
v___x_3062_ = lean_box(0);
v_isShared_3063_ = v_isSharedCheck_3067_;
goto v_resetjp_3061_;
}
v_resetjp_3061_:
{
lean_object* v___x_3065_; 
if (v_isShared_3063_ == 0)
{
v___x_3065_ = v___x_3062_;
goto v_reusejp_3064_;
}
else
{
lean_object* v_reuseFailAlloc_3066_; 
v_reuseFailAlloc_3066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3066_, 0, v_a_3060_);
v___x_3065_ = v_reuseFailAlloc_3066_;
goto v_reusejp_3064_;
}
v_reusejp_3064_:
{
return v___x_3065_;
}
}
}
else
{
lean_object* v_val_3068_; 
v_val_3068_ = lean_ctor_get(v___x_3057_, 0);
lean_inc(v_val_3068_);
lean_dec_ref_known(v___x_3057_, 1);
v_a_3001_ = v_val_3068_;
goto v___jp_3000_;
}
}
}
}
else
{
lean_object* v_a_3070_; lean_object* v___x_3072_; uint8_t v_isShared_3073_; uint8_t v_isSharedCheck_3077_; 
lean_dec_ref(v___x_3020_);
lean_dec_ref_known(v___x_3016_, 2);
lean_dec_ref(v_a_2991_);
lean_dec_ref(v_00_u03b1_2990_);
lean_dec(v_u_2989_);
v_a_3070_ = lean_ctor_get(v___x_3022_, 0);
v_isSharedCheck_3077_ = !lean_is_exclusive(v___x_3022_);
if (v_isSharedCheck_3077_ == 0)
{
v___x_3072_ = v___x_3022_;
v_isShared_3073_ = v_isSharedCheck_3077_;
goto v_resetjp_3071_;
}
else
{
lean_inc(v_a_3070_);
lean_dec(v___x_3022_);
v___x_3072_ = lean_box(0);
v_isShared_3073_ = v_isSharedCheck_3077_;
goto v_resetjp_3071_;
}
v_resetjp_3071_:
{
lean_object* v___x_3075_; 
if (v_isShared_3073_ == 0)
{
v___x_3075_ = v___x_3072_;
goto v_reusejp_3074_;
}
else
{
lean_object* v_reuseFailAlloc_3076_; 
v_reuseFailAlloc_3076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3076_, 0, v_a_3070_);
v___x_3075_ = v_reuseFailAlloc_3076_;
goto v_reusejp_3074_;
}
v_reusejp_3074_:
{
return v___x_3075_;
}
}
}
v___jp_3000_:
{
lean_object* v_expr_3002_; lean_object* v_val_3003_; lean_object* v_proof_3004_; lean_object* v___x_3006_; uint8_t v_isShared_3007_; uint8_t v_isSharedCheck_3013_; 
v_expr_3002_ = lean_ctor_get(v_a_3001_, 0);
v_val_3003_ = lean_ctor_get(v_a_3001_, 1);
v_proof_3004_ = lean_ctor_get(v_a_3001_, 2);
v_isSharedCheck_3013_ = !lean_is_exclusive(v_a_3001_);
if (v_isSharedCheck_3013_ == 0)
{
v___x_3006_ = v_a_3001_;
v_isShared_3007_ = v_isSharedCheck_3013_;
goto v_resetjp_3005_;
}
else
{
lean_inc(v_proof_3004_);
lean_inc(v_val_3003_);
lean_inc(v_expr_3002_);
lean_dec(v_a_3001_);
v___x_3006_ = lean_box(0);
v_isShared_3007_ = v_isSharedCheck_3013_;
goto v_resetjp_3005_;
}
v_resetjp_3005_:
{
lean_object* v___x_3009_; 
if (v_isShared_3007_ == 0)
{
v___x_3009_ = v___x_3006_;
goto v_reusejp_3008_;
}
else
{
lean_object* v_reuseFailAlloc_3012_; 
v_reuseFailAlloc_3012_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3012_, 0, v_expr_3002_);
lean_ctor_set(v_reuseFailAlloc_3012_, 1, v_val_3003_);
lean_ctor_set(v_reuseFailAlloc_3012_, 2, v_proof_3004_);
v___x_3009_ = v_reuseFailAlloc_3012_;
goto v_reusejp_3008_;
}
v_reusejp_3008_:
{
lean_object* v___x_3010_; lean_object* v___x_3011_; 
v___x_3010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3010_, 0, v___x_3009_);
v___x_3011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3011_, 0, v___x_3010_);
return v___x_3011_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg___boxed(lean_object* v_u_3078_, lean_object* v_00_u03b1_3079_, lean_object* v_a_3080_, lean_object* v_cz_u03b1_3081_, lean_object* v___sf_u03b1_3082_, lean_object* v_za_3083_, lean_object* v_a_3084_, lean_object* v_a_3085_, lean_object* v_a_3086_, lean_object* v_a_3087_, lean_object* v_a_3088_){
_start:
{
lean_object* v_res_3089_; 
v_res_3089_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg(v_u_3078_, v_00_u03b1_3079_, v_a_3080_, v_cz_u03b1_3081_, v___sf_u03b1_3082_, v_za_3083_, v_a_3084_, v_a_3085_, v_a_3086_, v_a_3087_);
lean_dec(v_a_3087_);
lean_dec_ref(v_a_3086_);
lean_dec(v_a_3085_);
lean_dec_ref(v_a_3084_);
return v_res_3089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv(lean_object* v_u_3090_, lean_object* v_00_u03b1_3091_, lean_object* v___s_u03b1_3092_, lean_object* v_a_3093_, lean_object* v_cz_u03b1_3094_, lean_object* v___sf_u03b1_3095_, lean_object* v_za_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_, lean_object* v_a_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_, lean_object* v_a_3102_){
_start:
{
lean_object* v___x_3104_; 
v___x_3104_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___redArg(v_u_3090_, v_00_u03b1_3091_, v_a_3093_, v_cz_u03b1_3094_, v___sf_u03b1_3095_, v_za_3096_, v_a_3099_, v_a_3100_, v_a_3101_, v_a_3102_);
return v___x_3104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___boxed(lean_object* v_u_3105_, lean_object* v_00_u03b1_3106_, lean_object* v___s_u03b1_3107_, lean_object* v_a_3108_, lean_object* v_cz_u03b1_3109_, lean_object* v___sf_u03b1_3110_, lean_object* v_za_3111_, lean_object* v_a_3112_, lean_object* v_a_3113_, lean_object* v_a_3114_, lean_object* v_a_3115_, lean_object* v_a_3116_, lean_object* v_a_3117_, lean_object* v_a_3118_){
_start:
{
lean_object* v_res_3119_; 
v_res_3119_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv(v_u_3105_, v_00_u03b1_3106_, v___s_u03b1_3107_, v_a_3108_, v_cz_u03b1_3109_, v___sf_u03b1_3110_, v_za_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_, v_a_3116_, v_a_3117_);
lean_dec(v_a_3117_);
lean_dec_ref(v_a_3116_);
lean_dec(v_a_3115_);
lean_dec_ref(v_a_3114_);
lean_dec(v_a_3113_);
lean_dec_ref(v_a_3112_);
lean_dec_ref(v___s_u03b1_3107_);
return v_res_3119_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2(void){
_start:
{
lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; 
v___x_3127_ = lean_box(0);
v___x_3128_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__1));
v___x_3129_ = l_Lean_Expr_const___override(v___x_3128_, v___x_3127_);
return v___x_3129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow(lean_object* v_u_3135_, lean_object* v_00_u03b1_3136_, lean_object* v_s_u03b1_3137_, lean_object* v_a_3138_, lean_object* v_b_3139_, lean_object* v_za_3140_, lean_object* v_vb_3141_, lean_object* v_a_3142_, lean_object* v_a_3143_, lean_object* v_a_3144_, lean_object* v_a_3145_){
_start:
{
lean_object* v_val_3148_; 
if (lean_obj_tag(v_vb_3141_) == 0)
{
lean_object* v___x_3162_; uint8_t v_isShared_3163_; uint8_t v_isSharedCheck_3247_; 
v_isSharedCheck_3247_ = !lean_is_exclusive(v_vb_3141_);
if (v_isSharedCheck_3247_ == 0)
{
lean_object* v_unused_3248_; lean_object* v_unused_3249_; 
v_unused_3248_ = lean_ctor_get(v_vb_3141_, 1);
lean_dec(v_unused_3248_);
v_unused_3249_ = lean_ctor_get(v_vb_3141_, 0);
lean_dec(v_unused_3249_);
v___x_3162_ = v_vb_3141_;
v_isShared_3163_ = v_isSharedCheck_3247_;
goto v_resetjp_3161_;
}
else
{
lean_dec(v_vb_3141_);
v___x_3162_ = lean_box(0);
v_isShared_3163_ = v_isSharedCheck_3247_;
goto v_resetjp_3161_;
}
v_resetjp_3161_:
{
lean_object* v_lit_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; lean_object* v___x_3170_; 
v_lit_3164_ = l_Lean_Expr_appArg_x21(v_b_3139_);
lean_inc_n(v_u_3135_, 2);
v___x_3165_ = l_Lean_Level_succ___override(v_u_3135_);
v___x_3166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__11));
v___x_3167_ = lean_box(0);
v___x_3168_ = lean_box(0);
if (v_isShared_3163_ == 0)
{
lean_ctor_set_tag(v___x_3162_, 1);
lean_ctor_set(v___x_3162_, 1, v___x_3168_);
lean_ctor_set(v___x_3162_, 0, v_u_3135_);
v___x_3170_ = v___x_3162_;
goto v_reusejp_3169_;
}
else
{
lean_object* v_reuseFailAlloc_3246_; 
v_reuseFailAlloc_3246_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3246_, 0, v_u_3135_);
lean_ctor_set(v_reuseFailAlloc_3246_, 1, v___x_3168_);
v___x_3170_ = v_reuseFailAlloc_3246_;
goto v_reusejp_3169_;
}
v_reusejp_3169_:
{
lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; lean_object* v___x_3208_; lean_object* v___x_3209_; lean_object* v___x_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; 
lean_inc_ref_n(v___x_3170_, 5);
v___x_3171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3171_, 0, v___x_3167_);
lean_ctor_set(v___x_3171_, 1, v___x_3170_);
lean_inc_n(v_u_3135_, 4);
v___x_3172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3172_, 0, v_u_3135_);
lean_ctor_set(v___x_3172_, 1, v___x_3171_);
v___x_3173_ = l_Lean_Expr_const___override(v___x_3166_, v___x_3172_);
lean_inc_ref_n(v_00_u03b1_3136_, 10);
v___x_3174_ = l_Lean_Expr_app___override(v___x_3173_, v_00_u03b1_3136_);
v___x_3175_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_3176_ = l_Lean_Expr_app___override(v___x_3174_, v___x_3175_);
v___x_3177_ = l_Lean_Expr_app___override(v___x_3176_, v_00_u03b1_3136_);
v___x_3178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__15));
v___x_3179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_3180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3180_, 0, v_u_3135_);
lean_ctor_set(v___x_3180_, 1, v___x_3179_);
v___x_3181_ = l_Lean_Expr_const___override(v___x_3178_, v___x_3180_);
v___x_3182_ = l_Lean_Expr_app___override(v___x_3181_, v_00_u03b1_3136_);
v___x_3183_ = l_Lean_Expr_app___override(v___x_3182_, v___x_3175_);
v___x_3184_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__19));
v___x_3185_ = l_Lean_Expr_const___override(v___x_3184_, v___x_3170_);
v___x_3186_ = l_Lean_Expr_app___override(v___x_3185_, v_00_u03b1_3136_);
v___x_3187_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__22));
v___x_3188_ = l_Lean_Expr_const___override(v___x_3187_, v___x_3170_);
v___x_3189_ = l_Lean_Expr_app___override(v___x_3188_, v_00_u03b1_3136_);
v___x_3190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__24));
v___x_3191_ = l_Lean_Expr_const___override(v___x_3190_, v___x_3170_);
v___x_3192_ = l_Lean_Expr_app___override(v___x_3191_, v_00_u03b1_3136_);
v___x_3193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_3194_ = l_Lean_Expr_const___override(v___x_3193_, v___x_3170_);
v___x_3195_ = l_Lean_Expr_app___override(v___x_3194_, v_00_u03b1_3136_);
v___x_3196_ = l_Lean_Expr_app___override(v___x_3195_, v_s_u03b1_3137_);
lean_inc_ref(v___x_3196_);
v___x_3197_ = l_Lean_Expr_app___override(v___x_3192_, v___x_3196_);
v___x_3198_ = l_Lean_Expr_app___override(v___x_3189_, v___x_3197_);
v___x_3199_ = l_Lean_Expr_app___override(v___x_3186_, v___x_3198_);
v___x_3200_ = l_Lean_Expr_app___override(v___x_3183_, v___x_3199_);
v___x_3201_ = l_Lean_Expr_app___override(v___x_3177_, v___x_3200_);
lean_inc_ref_n(v_a_3138_, 2);
lean_inc_ref(v___x_3201_);
v___x_3202_ = l_Lean_Expr_app___override(v___x_3201_, v_a_3138_);
lean_inc_ref_n(v_lit_3164_, 3);
v___x_3203_ = l_Lean_Expr_app___override(v___x_3202_, v_lit_3164_);
v___x_3204_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__2);
v___x_3205_ = l_Lean_Expr_app___override(v___x_3204_, v_lit_3164_);
v___x_3206_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__3));
v___x_3207_ = l_Lean_Expr_const___override(v___x_3206_, v___x_3170_);
v___x_3208_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5));
v___x_3209_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3209_, 0, v___x_3165_);
lean_ctor_set(v___x_3209_, 1, v___x_3168_);
v___x_3210_ = l_Lean_Expr_const___override(v___x_3208_, v___x_3209_);
v___x_3211_ = l_Lean_Expr_app___override(v___x_3207_, v_00_u03b1_3136_);
v___x_3212_ = l_Lean_Expr_app___override(v___x_3210_, v___x_3211_);
v___x_3213_ = l_Lean_Expr_app___override(v___x_3212_, v___x_3196_);
v___x_3214_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_3135_, v_00_u03b1_3136_, v_a_3138_, v_za_3140_);
lean_inc_ref(v___x_3203_);
v___x_3215_ = lp_mathlib_Mathlib_Meta_NormNum_evalPow_core(v_u_3135_, v_00_u03b1_3136_, v___x_3203_, v___x_3201_, v_a_3138_, v_lit_3164_, v_lit_3164_, v___x_3205_, v___x_3213_, v___x_3214_, v_a_3144_, v_a_3145_);
if (lean_obj_tag(v___x_3215_) == 0)
{
lean_object* v_a_3216_; lean_object* v___x_3218_; uint8_t v_isShared_3219_; uint8_t v_isSharedCheck_3237_; 
v_a_3216_ = lean_ctor_get(v___x_3215_, 0);
v_isSharedCheck_3237_ = !lean_is_exclusive(v___x_3215_);
if (v_isSharedCheck_3237_ == 0)
{
v___x_3218_ = v___x_3215_;
v_isShared_3219_ = v_isSharedCheck_3237_;
goto v_resetjp_3217_;
}
else
{
lean_inc(v_a_3216_);
lean_dec(v___x_3215_);
v___x_3218_ = lean_box(0);
v_isShared_3219_ = v_isSharedCheck_3237_;
goto v_resetjp_3217_;
}
v_resetjp_3217_:
{
if (lean_obj_tag(v_a_3216_) == 0)
{
lean_object* v___x_3220_; lean_object* v___x_3222_; 
lean_dec_ref(v___x_3203_);
lean_dec_ref(v_00_u03b1_3136_);
lean_dec(v_u_3135_);
v___x_3220_ = lean_box(0);
if (v_isShared_3219_ == 0)
{
lean_ctor_set(v___x_3218_, 0, v___x_3220_);
v___x_3222_ = v___x_3218_;
goto v_reusejp_3221_;
}
else
{
lean_object* v_reuseFailAlloc_3223_; 
v_reuseFailAlloc_3223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3223_, 0, v___x_3220_);
v___x_3222_ = v_reuseFailAlloc_3223_;
goto v_reusejp_3221_;
}
v_reusejp_3221_:
{
return v___x_3222_;
}
}
else
{
lean_object* v_val_3224_; lean_object* v___x_3225_; 
lean_del_object(v___x_3218_);
v_val_3224_ = lean_ctor_get(v_a_3216_, 0);
lean_inc(v_val_3224_);
lean_dec_ref_known(v_a_3216_, 1);
v___x_3225_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(v_u_3135_, v_00_u03b1_3136_, v___x_3203_, v_val_3224_);
if (lean_obj_tag(v___x_3225_) == 0)
{
lean_object* v___x_3226_; lean_object* v___x_3227_; lean_object* v_a_3228_; lean_object* v___x_3230_; uint8_t v_isShared_3231_; uint8_t v_isSharedCheck_3235_; 
v___x_3226_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_3227_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_3226_, v_a_3142_, v_a_3143_, v_a_3144_, v_a_3145_);
v_a_3228_ = lean_ctor_get(v___x_3227_, 0);
v_isSharedCheck_3235_ = !lean_is_exclusive(v___x_3227_);
if (v_isSharedCheck_3235_ == 0)
{
v___x_3230_ = v___x_3227_;
v_isShared_3231_ = v_isSharedCheck_3235_;
goto v_resetjp_3229_;
}
else
{
lean_inc(v_a_3228_);
lean_dec(v___x_3227_);
v___x_3230_ = lean_box(0);
v_isShared_3231_ = v_isSharedCheck_3235_;
goto v_resetjp_3229_;
}
v_resetjp_3229_:
{
lean_object* v___x_3233_; 
if (v_isShared_3231_ == 0)
{
v___x_3233_ = v___x_3230_;
goto v_reusejp_3232_;
}
else
{
lean_object* v_reuseFailAlloc_3234_; 
v_reuseFailAlloc_3234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3234_, 0, v_a_3228_);
v___x_3233_ = v_reuseFailAlloc_3234_;
goto v_reusejp_3232_;
}
v_reusejp_3232_:
{
return v___x_3233_;
}
}
}
else
{
lean_object* v_val_3236_; 
v_val_3236_ = lean_ctor_get(v___x_3225_, 0);
lean_inc(v_val_3236_);
lean_dec_ref_known(v___x_3225_, 1);
v_val_3148_ = v_val_3236_;
goto v___jp_3147_;
}
}
}
}
else
{
lean_object* v_a_3238_; lean_object* v___x_3240_; uint8_t v_isShared_3241_; uint8_t v_isSharedCheck_3245_; 
lean_dec_ref(v___x_3203_);
lean_dec_ref(v_00_u03b1_3136_);
lean_dec(v_u_3135_);
v_a_3238_ = lean_ctor_get(v___x_3215_, 0);
v_isSharedCheck_3245_ = !lean_is_exclusive(v___x_3215_);
if (v_isSharedCheck_3245_ == 0)
{
v___x_3240_ = v___x_3215_;
v_isShared_3241_ = v_isSharedCheck_3245_;
goto v_resetjp_3239_;
}
else
{
lean_inc(v_a_3238_);
lean_dec(v___x_3215_);
v___x_3240_ = lean_box(0);
v_isShared_3241_ = v_isSharedCheck_3245_;
goto v_resetjp_3239_;
}
v_resetjp_3239_:
{
lean_object* v___x_3243_; 
if (v_isShared_3241_ == 0)
{
v___x_3243_ = v___x_3240_;
goto v_reusejp_3242_;
}
else
{
lean_object* v_reuseFailAlloc_3244_; 
v_reuseFailAlloc_3244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3244_, 0, v_a_3238_);
v___x_3243_ = v_reuseFailAlloc_3244_;
goto v_reusejp_3242_;
}
v_reusejp_3242_:
{
return v___x_3243_;
}
}
}
}
}
}
else
{
lean_object* v___x_3250_; lean_object* v___x_3251_; 
lean_dec_ref(v_vb_3141_);
lean_dec_ref(v_za_3140_);
lean_dec_ref(v_a_3138_);
lean_dec_ref(v_s_u03b1_3137_);
lean_dec_ref(v_00_u03b1_3136_);
lean_dec(v_u_3135_);
v___x_3250_ = lean_box(0);
v___x_3251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3251_, 0, v___x_3250_);
return v___x_3251_;
}
v___jp_3147_:
{
lean_object* v_expr_3149_; lean_object* v_val_3150_; lean_object* v_proof_3151_; lean_object* v___x_3153_; uint8_t v_isShared_3154_; uint8_t v_isSharedCheck_3160_; 
v_expr_3149_ = lean_ctor_get(v_val_3148_, 0);
v_val_3150_ = lean_ctor_get(v_val_3148_, 1);
v_proof_3151_ = lean_ctor_get(v_val_3148_, 2);
v_isSharedCheck_3160_ = !lean_is_exclusive(v_val_3148_);
if (v_isSharedCheck_3160_ == 0)
{
v___x_3153_ = v_val_3148_;
v_isShared_3154_ = v_isSharedCheck_3160_;
goto v_resetjp_3152_;
}
else
{
lean_inc(v_proof_3151_);
lean_inc(v_val_3150_);
lean_inc(v_expr_3149_);
lean_dec(v_val_3148_);
v___x_3153_ = lean_box(0);
v_isShared_3154_ = v_isSharedCheck_3160_;
goto v_resetjp_3152_;
}
v_resetjp_3152_:
{
lean_object* v___x_3156_; 
if (v_isShared_3154_ == 0)
{
v___x_3156_ = v___x_3153_;
goto v_reusejp_3155_;
}
else
{
lean_object* v_reuseFailAlloc_3159_; 
v_reuseFailAlloc_3159_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3159_, 0, v_expr_3149_);
lean_ctor_set(v_reuseFailAlloc_3159_, 1, v_val_3150_);
lean_ctor_set(v_reuseFailAlloc_3159_, 2, v_proof_3151_);
v___x_3156_ = v_reuseFailAlloc_3159_;
goto v_reusejp_3155_;
}
v_reusejp_3155_:
{
lean_object* v___x_3157_; lean_object* v___x_3158_; 
v___x_3157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3157_, 0, v___x_3156_);
v___x_3158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3158_, 0, v___x_3157_);
return v___x_3158_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___boxed(lean_object* v_u_3252_, lean_object* v_00_u03b1_3253_, lean_object* v_s_u03b1_3254_, lean_object* v_a_3255_, lean_object* v_b_3256_, lean_object* v_za_3257_, lean_object* v_vb_3258_, lean_object* v_a_3259_, lean_object* v_a_3260_, lean_object* v_a_3261_, lean_object* v_a_3262_, lean_object* v_a_3263_){
_start:
{
lean_object* v_res_3264_; 
v_res_3264_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow(v_u_3252_, v_00_u03b1_3253_, v_s_u03b1_3254_, v_a_3255_, v_b_3256_, v_za_3257_, v_vb_3258_, v_a_3259_, v_a_3260_, v_a_3261_, v_a_3262_);
lean_dec(v_a_3262_);
lean_dec_ref(v_a_3261_);
lean_dec(v_a_3260_);
lean_dec_ref(v_a_3259_);
lean_dec_ref(v_b_3256_);
return v_res_3264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg(lean_object* v_u_3301_, lean_object* v_00_u03b1_3302_, lean_object* v_a_3303_, lean_object* v___cr_u03b1_3304_, lean_object* v_za_3305_, lean_object* v_a_3306_, lean_object* v_a_3307_, lean_object* v_a_3308_, lean_object* v_a_3309_){
_start:
{
lean_object* v_a_3312_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3340_; 
lean_inc_ref_n(v_a_3303_, 2);
lean_inc_ref_n(v_00_u03b1_3302_, 4);
lean_inc_n(v_u_3301_, 4);
v___x_3324_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_toResult(v_u_3301_, v_00_u03b1_3302_, v_a_3303_, v_za_3305_);
v___x_3325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__0));
v___x_3326_ = lean_box(0);
v___x_3327_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3327_, 0, v_u_3301_);
lean_ctor_set(v___x_3327_, 1, v___x_3326_);
lean_inc_ref_n(v___x_3327_, 2);
v___x_3328_ = l_Lean_Expr_const___override(v___x_3325_, v___x_3327_);
v___x_3329_ = l_Lean_Expr_app___override(v___x_3328_, v_00_u03b1_3302_);
v___x_3330_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5));
v___x_3331_ = l_Lean_Level_succ___override(v_u_3301_);
v___x_3332_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3332_, 0, v___x_3331_);
lean_ctor_set(v___x_3332_, 1, v___x_3326_);
v___x_3333_ = l_Lean_Expr_const___override(v___x_3330_, v___x_3332_);
v___x_3334_ = l_Lean_Expr_app___override(v___x_3333_, v___x_3329_);
v___x_3335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExBase_evalIntCast___closed__5));
v___x_3336_ = l_Lean_Expr_const___override(v___x_3335_, v___x_3327_);
v___x_3337_ = l_Lean_Expr_app___override(v___x_3336_, v_00_u03b1_3302_);
v___x_3338_ = l_Lean_Expr_app___override(v___x_3337_, v___cr_u03b1_3304_);
v___x_3339_ = l_Lean_Expr_app___override(v___x_3334_, v___x_3338_);
lean_inc_ref(v___x_3339_);
v___x_3340_ = lp_mathlib_Mathlib_Meta_NormNum_Result_neg(v_u_3301_, v_00_u03b1_3302_, v_a_3303_, v___x_3324_, v___x_3339_, v_a_3306_, v_a_3307_, v_a_3308_, v_a_3309_);
if (lean_obj_tag(v___x_3340_) == 0)
{
lean_object* v_a_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; lean_object* v___x_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; lean_object* v___x_3360_; lean_object* v___x_3361_; lean_object* v___x_3362_; lean_object* v___x_3363_; lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___x_3366_; lean_object* v___x_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; lean_object* v___x_3370_; lean_object* v___x_3371_; 
v_a_3341_ = lean_ctor_get(v___x_3340_, 0);
lean_inc(v_a_3341_);
lean_dec_ref_known(v___x_3340_, 1);
v___x_3342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__3));
lean_inc_ref_n(v___x_3327_, 6);
v___x_3343_ = l_Lean_Expr_const___override(v___x_3342_, v___x_3327_);
lean_inc_ref_n(v_00_u03b1_3302_, 7);
v___x_3344_ = l_Lean_Expr_app___override(v___x_3343_, v_00_u03b1_3302_);
v___x_3345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__6));
v___x_3346_ = l_Lean_Expr_const___override(v___x_3345_, v___x_3327_);
v___x_3347_ = l_Lean_Expr_app___override(v___x_3346_, v_00_u03b1_3302_);
v___x_3348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__9));
v___x_3349_ = l_Lean_Expr_const___override(v___x_3348_, v___x_3327_);
v___x_3350_ = l_Lean_Expr_app___override(v___x_3349_, v_00_u03b1_3302_);
v___x_3351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__12));
v___x_3352_ = l_Lean_Expr_const___override(v___x_3351_, v___x_3327_);
v___x_3353_ = l_Lean_Expr_app___override(v___x_3352_, v_00_u03b1_3302_);
v___x_3354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__15));
v___x_3355_ = l_Lean_Expr_const___override(v___x_3354_, v___x_3327_);
v___x_3356_ = l_Lean_Expr_app___override(v___x_3355_, v_00_u03b1_3302_);
v___x_3357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__18));
v___x_3358_ = l_Lean_Expr_const___override(v___x_3357_, v___x_3327_);
v___x_3359_ = l_Lean_Expr_app___override(v___x_3358_, v_00_u03b1_3302_);
v___x_3360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___closed__20));
v___x_3361_ = l_Lean_Expr_const___override(v___x_3360_, v___x_3327_);
v___x_3362_ = l_Lean_Expr_app___override(v___x_3361_, v_00_u03b1_3302_);
v___x_3363_ = l_Lean_Expr_app___override(v___x_3362_, v___x_3339_);
v___x_3364_ = l_Lean_Expr_app___override(v___x_3359_, v___x_3363_);
v___x_3365_ = l_Lean_Expr_app___override(v___x_3356_, v___x_3364_);
v___x_3366_ = l_Lean_Expr_app___override(v___x_3353_, v___x_3365_);
v___x_3367_ = l_Lean_Expr_app___override(v___x_3350_, v___x_3366_);
v___x_3368_ = l_Lean_Expr_app___override(v___x_3347_, v___x_3367_);
v___x_3369_ = l_Lean_Expr_app___override(v___x_3344_, v___x_3368_);
v___x_3370_ = l_Lean_Expr_app___override(v___x_3369_, v_a_3303_);
v___x_3371_ = lp_mathlib_Mathlib_Tactic_Ring_RatCoeff_ofResult(v_u_3301_, v_00_u03b1_3302_, v___x_3370_, v_a_3341_);
if (lean_obj_tag(v___x_3371_) == 0)
{
lean_object* v___x_3372_; lean_object* v___x_3373_; 
v___x_3372_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_3373_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_3372_, v_a_3306_, v_a_3307_, v_a_3308_, v_a_3309_);
return v___x_3373_;
}
else
{
lean_object* v_val_3374_; 
v_val_3374_ = lean_ctor_get(v___x_3371_, 0);
lean_inc(v_val_3374_);
lean_dec_ref_known(v___x_3371_, 1);
v_a_3312_ = v_val_3374_;
goto v___jp_3311_;
}
}
else
{
lean_object* v_a_3375_; lean_object* v___x_3377_; uint8_t v_isShared_3378_; uint8_t v_isSharedCheck_3382_; 
lean_dec_ref(v___x_3339_);
lean_dec_ref_known(v___x_3327_, 2);
lean_dec_ref(v_a_3303_);
lean_dec_ref(v_00_u03b1_3302_);
lean_dec(v_u_3301_);
v_a_3375_ = lean_ctor_get(v___x_3340_, 0);
v_isSharedCheck_3382_ = !lean_is_exclusive(v___x_3340_);
if (v_isSharedCheck_3382_ == 0)
{
v___x_3377_ = v___x_3340_;
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
else
{
lean_inc(v_a_3375_);
lean_dec(v___x_3340_);
v___x_3377_ = lean_box(0);
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
v_resetjp_3376_:
{
lean_object* v___x_3380_; 
if (v_isShared_3378_ == 0)
{
v___x_3380_ = v___x_3377_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3381_; 
v_reuseFailAlloc_3381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3381_, 0, v_a_3375_);
v___x_3380_ = v_reuseFailAlloc_3381_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
return v___x_3380_;
}
}
}
v___jp_3311_:
{
lean_object* v_expr_3313_; lean_object* v_val_3314_; lean_object* v_proof_3315_; lean_object* v___x_3317_; uint8_t v_isShared_3318_; uint8_t v_isSharedCheck_3323_; 
v_expr_3313_ = lean_ctor_get(v_a_3312_, 0);
v_val_3314_ = lean_ctor_get(v_a_3312_, 1);
v_proof_3315_ = lean_ctor_get(v_a_3312_, 2);
v_isSharedCheck_3323_ = !lean_is_exclusive(v_a_3312_);
if (v_isSharedCheck_3323_ == 0)
{
v___x_3317_ = v_a_3312_;
v_isShared_3318_ = v_isSharedCheck_3323_;
goto v_resetjp_3316_;
}
else
{
lean_inc(v_proof_3315_);
lean_inc(v_val_3314_);
lean_inc(v_expr_3313_);
lean_dec(v_a_3312_);
v___x_3317_ = lean_box(0);
v_isShared_3318_ = v_isSharedCheck_3323_;
goto v_resetjp_3316_;
}
v_resetjp_3316_:
{
lean_object* v___x_3320_; 
if (v_isShared_3318_ == 0)
{
v___x_3320_ = v___x_3317_;
goto v_reusejp_3319_;
}
else
{
lean_object* v_reuseFailAlloc_3322_; 
v_reuseFailAlloc_3322_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3322_, 0, v_expr_3313_);
lean_ctor_set(v_reuseFailAlloc_3322_, 1, v_val_3314_);
lean_ctor_set(v_reuseFailAlloc_3322_, 2, v_proof_3315_);
v___x_3320_ = v_reuseFailAlloc_3322_;
goto v_reusejp_3319_;
}
v_reusejp_3319_:
{
lean_object* v___x_3321_; 
v___x_3321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3321_, 0, v___x_3320_);
return v___x_3321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___boxed(lean_object* v_u_3383_, lean_object* v_00_u03b1_3384_, lean_object* v_a_3385_, lean_object* v___cr_u03b1_3386_, lean_object* v_za_3387_, lean_object* v_a_3388_, lean_object* v_a_3389_, lean_object* v_a_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_){
_start:
{
lean_object* v_res_3393_; 
v_res_3393_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg(v_u_3383_, v_00_u03b1_3384_, v_a_3385_, v___cr_u03b1_3386_, v_za_3387_, v_a_3388_, v_a_3389_, v_a_3390_, v_a_3391_);
lean_dec(v_a_3391_);
lean_dec_ref(v_a_3390_);
lean_dec(v_a_3389_);
lean_dec_ref(v_a_3388_);
return v_res_3393_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0(lean_object* v_x_3394_, lean_object* v_y_3395_, lean_object* v_zx_3396_, lean_object* v_zy_3397_){
_start:
{
lean_object* v_value_3398_; lean_object* v_value_3399_; uint8_t v___x_3400_; 
v_value_3398_ = lean_ctor_get(v_zx_3396_, 0);
v_value_3399_ = lean_ctor_get(v_zy_3397_, 0);
v___x_3400_ = l_instDecidableEqRat_decEq(v_value_3398_, v_value_3399_);
return v___x_3400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0___boxed(lean_object* v_x_3401_, lean_object* v_y_3402_, lean_object* v_zx_3403_, lean_object* v_zy_3404_){
_start:
{
uint8_t v_res_3405_; lean_object* v_r_3406_; 
v_res_3405_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__0(v_x_3401_, v_y_3402_, v_zx_3403_, v_zy_3404_);
lean_dec_ref(v_zy_3404_);
lean_dec_ref(v_zx_3403_);
lean_dec_ref(v_y_3402_);
lean_dec_ref(v_x_3401_);
v_r_3406_ = lean_box(v_res_3405_);
return v_r_3406_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1(lean_object* v_x_3407_, lean_object* v_y_3408_, lean_object* v_zx_3409_, lean_object* v_zy_3410_){
_start:
{
lean_object* v_value_3411_; lean_object* v_value_3412_; uint8_t v___x_3413_; 
v_value_3411_ = lean_ctor_get(v_zx_3409_, 0);
lean_inc_ref_n(v_value_3411_, 2);
lean_dec_ref(v_zx_3409_);
v_value_3412_ = lean_ctor_get(v_zy_3410_, 0);
lean_inc_ref_n(v_value_3412_, 2);
lean_dec_ref(v_zy_3410_);
v___x_3413_ = l_Rat_blt(v_value_3411_, v_value_3412_);
if (v___x_3413_ == 0)
{
uint8_t v___x_3414_; 
v___x_3414_ = l_instDecidableEqRat_decEq(v_value_3411_, v_value_3412_);
lean_dec_ref(v_value_3412_);
lean_dec_ref(v_value_3411_);
if (v___x_3414_ == 0)
{
uint8_t v___x_3415_; 
v___x_3415_ = 2;
return v___x_3415_;
}
else
{
uint8_t v___x_3416_; 
v___x_3416_ = 1;
return v___x_3416_;
}
}
else
{
uint8_t v___x_3417_; 
lean_dec_ref(v_value_3412_);
lean_dec_ref(v_value_3411_);
v___x_3417_ = 0;
return v___x_3417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1___boxed(lean_object* v_x_3418_, lean_object* v_y_3419_, lean_object* v_zx_3420_, lean_object* v_zy_3421_){
_start:
{
uint8_t v_res_3422_; lean_object* v_r_3423_; 
v_res_3422_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare___lam__1(v_x_3418_, v_y_3419_, v_zx_3420_, v_zy_3421_);
lean_dec_ref(v_y_3419_);
lean_dec_ref(v_x_3418_);
v_r_3423_ = lean_box(v_res_3422_);
return v_r_3423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare(lean_object* v_u_3429_, lean_object* v_00_u03b1_3430_){
_start:
{
lean_object* v___x_3431_; 
v___x_3431_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ringCompare___closed__2));
return v___x_3431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompare___boxed(lean_object* v_u_3432_, lean_object* v_00_u03b1_3433_){
_start:
{
lean_object* v_res_3434_; 
v_res_3434_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare(v_u_3432_, v_00_u03b1_3433_);
lean_dec_ref(v_00_u03b1_3433_);
lean_dec(v_u_3432_);
return v_res_3434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0(uint8_t v___x_3435_, lean_object* v___x_3436_, lean_object* v_00_u03b2_3437_, uint8_t v___x_3438_, lean_object* v___y_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_, lean_object* v___y_3442_){
_start:
{
lean_object* v_keyedConfig_3444_; uint8_t v_trackZetaDelta_3445_; lean_object* v_zetaDeltaSet_3446_; lean_object* v_lctx_3447_; lean_object* v_localInstances_3448_; lean_object* v_defEqCtx_x3f_3449_; lean_object* v_synthPendingDepth_3450_; lean_object* v_customCanUnfoldPredicate_x3f_3451_; uint8_t v_univApprox_3452_; uint8_t v_inTypeClassResolution_3453_; uint8_t v_cacheInferType_3454_; lean_object* v___x_3456_; uint8_t v_isShared_3457_; uint8_t v_isSharedCheck_3474_; 
v_keyedConfig_3444_ = lean_ctor_get(v___y_3439_, 0);
v_trackZetaDelta_3445_ = lean_ctor_get_uint8(v___y_3439_, sizeof(void*)*7);
v_zetaDeltaSet_3446_ = lean_ctor_get(v___y_3439_, 1);
v_lctx_3447_ = lean_ctor_get(v___y_3439_, 2);
v_localInstances_3448_ = lean_ctor_get(v___y_3439_, 3);
v_defEqCtx_x3f_3449_ = lean_ctor_get(v___y_3439_, 4);
v_synthPendingDepth_3450_ = lean_ctor_get(v___y_3439_, 5);
v_customCanUnfoldPredicate_x3f_3451_ = lean_ctor_get(v___y_3439_, 6);
v_univApprox_3452_ = lean_ctor_get_uint8(v___y_3439_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3453_ = lean_ctor_get_uint8(v___y_3439_, sizeof(void*)*7 + 2);
v_cacheInferType_3454_ = lean_ctor_get_uint8(v___y_3439_, sizeof(void*)*7 + 3);
v_isSharedCheck_3474_ = !lean_is_exclusive(v___y_3439_);
if (v_isSharedCheck_3474_ == 0)
{
v___x_3456_ = v___y_3439_;
v_isShared_3457_ = v_isSharedCheck_3474_;
goto v_resetjp_3455_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3451_);
lean_inc(v_synthPendingDepth_3450_);
lean_inc(v_defEqCtx_x3f_3449_);
lean_inc(v_localInstances_3448_);
lean_inc(v_lctx_3447_);
lean_inc(v_zetaDeltaSet_3446_);
lean_inc(v_keyedConfig_3444_);
lean_dec(v___y_3439_);
v___x_3456_ = lean_box(0);
v_isShared_3457_ = v_isSharedCheck_3474_;
goto v_resetjp_3455_;
}
v_resetjp_3455_:
{
lean_object* v___x_3458_; lean_object* v___x_3460_; 
v___x_3458_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3435_, v_keyedConfig_3444_);
if (v_isShared_3457_ == 0)
{
lean_ctor_set(v___x_3456_, 0, v___x_3458_);
v___x_3460_ = v___x_3456_;
goto v_reusejp_3459_;
}
else
{
lean_object* v_reuseFailAlloc_3473_; 
v_reuseFailAlloc_3473_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3473_, 0, v___x_3458_);
lean_ctor_set(v_reuseFailAlloc_3473_, 1, v_zetaDeltaSet_3446_);
lean_ctor_set(v_reuseFailAlloc_3473_, 2, v_lctx_3447_);
lean_ctor_set(v_reuseFailAlloc_3473_, 3, v_localInstances_3448_);
lean_ctor_set(v_reuseFailAlloc_3473_, 4, v_defEqCtx_x3f_3449_);
lean_ctor_set(v_reuseFailAlloc_3473_, 5, v_synthPendingDepth_3450_);
lean_ctor_set(v_reuseFailAlloc_3473_, 6, v_customCanUnfoldPredicate_x3f_3451_);
lean_ctor_set_uint8(v_reuseFailAlloc_3473_, sizeof(void*)*7, v_trackZetaDelta_3445_);
lean_ctor_set_uint8(v_reuseFailAlloc_3473_, sizeof(void*)*7 + 1, v_univApprox_3452_);
lean_ctor_set_uint8(v_reuseFailAlloc_3473_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3453_);
lean_ctor_set_uint8(v_reuseFailAlloc_3473_, sizeof(void*)*7 + 3, v_cacheInferType_3454_);
v___x_3460_ = v_reuseFailAlloc_3473_;
goto v_reusejp_3459_;
}
v_reusejp_3459_:
{
lean_object* v___x_3461_; 
v___x_3461_ = l_Lean_Meta_isExprDefEq(v___x_3436_, v_00_u03b2_3437_, v___x_3460_, v___y_3440_, v___y_3441_, v___y_3442_);
lean_dec_ref(v___x_3460_);
if (lean_obj_tag(v___x_3461_) == 0)
{
lean_object* v_a_3462_; uint8_t v___x_3463_; 
v_a_3462_ = lean_ctor_get(v___x_3461_, 0);
lean_inc(v_a_3462_);
v___x_3463_ = lean_unbox(v_a_3462_);
lean_dec(v_a_3462_);
if (v___x_3463_ == 0)
{
return v___x_3461_;
}
else
{
lean_object* v___x_3465_; uint8_t v_isShared_3466_; uint8_t v_isSharedCheck_3471_; 
v_isSharedCheck_3471_ = !lean_is_exclusive(v___x_3461_);
if (v_isSharedCheck_3471_ == 0)
{
lean_object* v_unused_3472_; 
v_unused_3472_ = lean_ctor_get(v___x_3461_, 0);
lean_dec(v_unused_3472_);
v___x_3465_ = v___x_3461_;
v_isShared_3466_ = v_isSharedCheck_3471_;
goto v_resetjp_3464_;
}
else
{
lean_dec(v___x_3461_);
v___x_3465_ = lean_box(0);
v_isShared_3466_ = v_isSharedCheck_3471_;
goto v_resetjp_3464_;
}
v_resetjp_3464_:
{
lean_object* v___x_3467_; lean_object* v___x_3469_; 
v___x_3467_ = lean_box(v___x_3438_);
if (v_isShared_3466_ == 0)
{
lean_ctor_set(v___x_3465_, 0, v___x_3467_);
v___x_3469_ = v___x_3465_;
goto v_reusejp_3468_;
}
else
{
lean_object* v_reuseFailAlloc_3470_; 
v_reuseFailAlloc_3470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3470_, 0, v___x_3467_);
v___x_3469_ = v_reuseFailAlloc_3470_;
goto v_reusejp_3468_;
}
v_reusejp_3468_:
{
return v___x_3469_;
}
}
}
}
else
{
return v___x_3461_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0___boxed(lean_object* v___x_3475_, lean_object* v___x_3476_, lean_object* v_00_u03b2_3477_, lean_object* v___x_3478_, lean_object* v___y_3479_, lean_object* v___y_3480_, lean_object* v___y_3481_, lean_object* v___y_3482_, lean_object* v___y_3483_){
_start:
{
uint8_t v___x_20186__boxed_3484_; uint8_t v___x_20188__boxed_3485_; lean_object* v_res_3486_; 
v___x_20186__boxed_3484_ = lean_unbox(v___x_3475_);
v___x_20188__boxed_3485_ = lean_unbox(v___x_3478_);
v_res_3486_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0(v___x_20186__boxed_3484_, v___x_3476_, v_00_u03b2_3477_, v___x_20188__boxed_3485_, v___y_3479_, v___y_3480_, v___y_3481_, v___y_3482_);
lean_dec(v___y_3482_);
lean_dec_ref(v___y_3481_);
lean_dec(v___y_3480_);
return v_res_3486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1(uint8_t v___x_3487_, lean_object* v___x_3488_, lean_object* v_s_u03b2_3489_, uint8_t v___x_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_, lean_object* v___y_3494_){
_start:
{
lean_object* v_keyedConfig_3496_; uint8_t v_trackZetaDelta_3497_; lean_object* v_zetaDeltaSet_3498_; lean_object* v_lctx_3499_; lean_object* v_localInstances_3500_; lean_object* v_defEqCtx_x3f_3501_; lean_object* v_synthPendingDepth_3502_; lean_object* v_customCanUnfoldPredicate_x3f_3503_; uint8_t v_univApprox_3504_; uint8_t v_inTypeClassResolution_3505_; uint8_t v_cacheInferType_3506_; lean_object* v___x_3508_; uint8_t v_isShared_3509_; uint8_t v_isSharedCheck_3526_; 
v_keyedConfig_3496_ = lean_ctor_get(v___y_3491_, 0);
v_trackZetaDelta_3497_ = lean_ctor_get_uint8(v___y_3491_, sizeof(void*)*7);
v_zetaDeltaSet_3498_ = lean_ctor_get(v___y_3491_, 1);
v_lctx_3499_ = lean_ctor_get(v___y_3491_, 2);
v_localInstances_3500_ = lean_ctor_get(v___y_3491_, 3);
v_defEqCtx_x3f_3501_ = lean_ctor_get(v___y_3491_, 4);
v_synthPendingDepth_3502_ = lean_ctor_get(v___y_3491_, 5);
v_customCanUnfoldPredicate_x3f_3503_ = lean_ctor_get(v___y_3491_, 6);
v_univApprox_3504_ = lean_ctor_get_uint8(v___y_3491_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3505_ = lean_ctor_get_uint8(v___y_3491_, sizeof(void*)*7 + 2);
v_cacheInferType_3506_ = lean_ctor_get_uint8(v___y_3491_, sizeof(void*)*7 + 3);
v_isSharedCheck_3526_ = !lean_is_exclusive(v___y_3491_);
if (v_isSharedCheck_3526_ == 0)
{
v___x_3508_ = v___y_3491_;
v_isShared_3509_ = v_isSharedCheck_3526_;
goto v_resetjp_3507_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3503_);
lean_inc(v_synthPendingDepth_3502_);
lean_inc(v_defEqCtx_x3f_3501_);
lean_inc(v_localInstances_3500_);
lean_inc(v_lctx_3499_);
lean_inc(v_zetaDeltaSet_3498_);
lean_inc(v_keyedConfig_3496_);
lean_dec(v___y_3491_);
v___x_3508_ = lean_box(0);
v_isShared_3509_ = v_isSharedCheck_3526_;
goto v_resetjp_3507_;
}
v_resetjp_3507_:
{
lean_object* v___x_3510_; lean_object* v___x_3512_; 
v___x_3510_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3487_, v_keyedConfig_3496_);
if (v_isShared_3509_ == 0)
{
lean_ctor_set(v___x_3508_, 0, v___x_3510_);
v___x_3512_ = v___x_3508_;
goto v_reusejp_3511_;
}
else
{
lean_object* v_reuseFailAlloc_3525_; 
v_reuseFailAlloc_3525_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3525_, 0, v___x_3510_);
lean_ctor_set(v_reuseFailAlloc_3525_, 1, v_zetaDeltaSet_3498_);
lean_ctor_set(v_reuseFailAlloc_3525_, 2, v_lctx_3499_);
lean_ctor_set(v_reuseFailAlloc_3525_, 3, v_localInstances_3500_);
lean_ctor_set(v_reuseFailAlloc_3525_, 4, v_defEqCtx_x3f_3501_);
lean_ctor_set(v_reuseFailAlloc_3525_, 5, v_synthPendingDepth_3502_);
lean_ctor_set(v_reuseFailAlloc_3525_, 6, v_customCanUnfoldPredicate_x3f_3503_);
lean_ctor_set_uint8(v_reuseFailAlloc_3525_, sizeof(void*)*7, v_trackZetaDelta_3497_);
lean_ctor_set_uint8(v_reuseFailAlloc_3525_, sizeof(void*)*7 + 1, v_univApprox_3504_);
lean_ctor_set_uint8(v_reuseFailAlloc_3525_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3505_);
lean_ctor_set_uint8(v_reuseFailAlloc_3525_, sizeof(void*)*7 + 3, v_cacheInferType_3506_);
v___x_3512_ = v_reuseFailAlloc_3525_;
goto v_reusejp_3511_;
}
v_reusejp_3511_:
{
lean_object* v___x_3513_; 
v___x_3513_ = l_Lean_Meta_isExprDefEq(v___x_3488_, v_s_u03b2_3489_, v___x_3512_, v___y_3492_, v___y_3493_, v___y_3494_);
lean_dec_ref(v___x_3512_);
if (lean_obj_tag(v___x_3513_) == 0)
{
lean_object* v_a_3514_; uint8_t v___x_3515_; 
v_a_3514_ = lean_ctor_get(v___x_3513_, 0);
lean_inc(v_a_3514_);
v___x_3515_ = lean_unbox(v_a_3514_);
lean_dec(v_a_3514_);
if (v___x_3515_ == 0)
{
return v___x_3513_;
}
else
{
lean_object* v___x_3517_; uint8_t v_isShared_3518_; uint8_t v_isSharedCheck_3523_; 
v_isSharedCheck_3523_ = !lean_is_exclusive(v___x_3513_);
if (v_isSharedCheck_3523_ == 0)
{
lean_object* v_unused_3524_; 
v_unused_3524_ = lean_ctor_get(v___x_3513_, 0);
lean_dec(v_unused_3524_);
v___x_3517_ = v___x_3513_;
v_isShared_3518_ = v_isSharedCheck_3523_;
goto v_resetjp_3516_;
}
else
{
lean_dec(v___x_3513_);
v___x_3517_ = lean_box(0);
v_isShared_3518_ = v_isSharedCheck_3523_;
goto v_resetjp_3516_;
}
v_resetjp_3516_:
{
lean_object* v___x_3519_; lean_object* v___x_3521_; 
v___x_3519_ = lean_box(v___x_3490_);
if (v_isShared_3518_ == 0)
{
lean_ctor_set(v___x_3517_, 0, v___x_3519_);
v___x_3521_ = v___x_3517_;
goto v_reusejp_3520_;
}
else
{
lean_object* v_reuseFailAlloc_3522_; 
v_reuseFailAlloc_3522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3522_, 0, v___x_3519_);
v___x_3521_ = v_reuseFailAlloc_3522_;
goto v_reusejp_3520_;
}
v_reusejp_3520_:
{
return v___x_3521_;
}
}
}
}
else
{
return v___x_3513_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1___boxed(lean_object* v___x_3527_, lean_object* v___x_3528_, lean_object* v_s_u03b2_3529_, lean_object* v___x_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_, lean_object* v___y_3535_){
_start:
{
uint8_t v___x_20251__boxed_3536_; uint8_t v___x_20253__boxed_3537_; lean_object* v_res_3538_; 
v___x_20251__boxed_3536_ = lean_unbox(v___x_3527_);
v___x_20253__boxed_3537_ = lean_unbox(v___x_3530_);
v_res_3538_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1(v___x_20251__boxed_3536_, v___x_3528_, v_s_u03b2_3529_, v___x_20253__boxed_3537_, v___y_3531_, v___y_3532_, v___y_3533_, v___y_3534_);
lean_dec(v___y_3534_);
lean_dec_ref(v___y_3533_);
lean_dec(v___y_3532_);
return v_res_3538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___boxed(lean_object* v_u_3539_, lean_object* v_00_u03b1_3540_, lean_object* v_s_u03b1_3541_, lean_object* v_c_u03b1_3542_, lean_object* v_v_3543_, lean_object* v_00_u03b2_3544_, lean_object* v_s_u03b2_3545_, lean_object* v___smul_3546_, lean_object* v_x_3547_, lean_object* v_a_3548_, lean_object* v_a_3549_, lean_object* v_a_3550_, lean_object* v_a_3551_, lean_object* v_a_3552_, lean_object* v_a_3553_, lean_object* v_a_3554_){
_start:
{
lean_object* v_res_3555_; 
v_res_3555_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast(v_u_3539_, v_00_u03b1_3540_, v_s_u03b1_3541_, v_c_u03b1_3542_, v_v_3543_, v_00_u03b2_3544_, v_s_u03b2_3545_, v___smul_3546_, v_x_3547_, v_a_3548_, v_a_3549_, v_a_3550_, v_a_3551_, v_a_3552_, v_a_3553_);
lean_dec(v_a_3553_);
lean_dec_ref(v_a_3552_);
lean_dec(v_a_3551_);
lean_dec_ref(v_a_3550_);
lean_dec(v_a_3549_);
lean_dec_ref(v_a_3548_);
lean_dec_ref(v___smul_3546_);
return v_res_3555_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0(void){
_start:
{
lean_object* v___x_3556_; lean_object* v___x_3557_; lean_object* v___x_3558_; 
v___x_3556_ = lean_box(0);
v___x_3557_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__0);
v___x_3558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3558_, 0, v___x_3557_);
lean_ctor_set(v___x_3558_, 1, v___x_3556_);
return v___x_3558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_ringCompute(lean_object* v_u_3559_, lean_object* v_00_u03b1_3560_, lean_object* v_s_u03b1_3561_, lean_object* v_c_u03b1_3562_){
_start:
{
lean_object* v___x_3563_; lean_object* v___x_3564_; lean_object* v___x_3565_; lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3568_; lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3574_; lean_object* v___x_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v___x_3585_; lean_object* v___x_3586_; lean_object* v___x_3587_; lean_object* v___x_3588_; lean_object* v___x_3589_; lean_object* v___x_3590_; lean_object* v___x_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; lean_object* v___x_3594_; lean_object* v___x_3595_; lean_object* v___x_3596_; lean_object* v___x_3597_; lean_object* v___x_3598_; lean_object* v___x_3599_; lean_object* v___x_3600_; lean_object* v___x_3601_; lean_object* v___x_3602_; lean_object* v___x_3603_; lean_object* v___x_3604_; 
v___x_3563_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompare(v_u_3559_, v_00_u03b1_3560_);
lean_inc_ref_n(v_s_u03b1_3561_, 7);
lean_inc_ref_n(v_00_u03b1_3560_, 13);
lean_inc_n(v_u_3559_, 9);
v___x_3564_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___boxed), 12, 3);
lean_closure_set(v___x_3564_, 0, v_u_3559_);
lean_closure_set(v___x_3564_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3564_, 2, v_s_u03b1_3561_);
v___x_3565_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_mul___boxed), 12, 3);
lean_closure_set(v___x_3565_, 0, v_u_3559_);
lean_closure_set(v___x_3565_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3565_, 2, v_s_u03b1_3561_);
v___x_3566_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___boxed), 16, 4);
lean_closure_set(v___x_3566_, 0, v_u_3559_);
lean_closure_set(v___x_3566_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3566_, 2, v_s_u03b1_3561_);
lean_closure_set(v___x_3566_, 3, v_c_u03b1_3562_);
v___x_3567_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_neg___boxed), 10, 2);
lean_closure_set(v___x_3567_, 0, v_u_3559_);
lean_closure_set(v___x_3567_, 1, v_00_u03b1_3560_);
v___x_3568_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___boxed), 12, 3);
lean_closure_set(v___x_3568_, 0, v_u_3559_);
lean_closure_set(v___x_3568_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3568_, 2, v_s_u03b1_3561_);
v___x_3569_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_inv___boxed), 14, 3);
lean_closure_set(v___x_3569_, 0, v_u_3559_);
lean_closure_set(v___x_3569_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3569_, 2, v_s_u03b1_3561_);
v___x_3570_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_derive___boxed), 9, 3);
lean_closure_set(v___x_3570_, 0, v_u_3559_);
lean_closure_set(v___x_3570_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3570_, 2, v_s_u03b1_3561_);
v___x_3571_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne), 5, 3);
lean_closure_set(v___x_3571_, 0, v_u_3559_);
lean_closure_set(v___x_3571_, 1, v_00_u03b1_3560_);
lean_closure_set(v___x_3571_, 2, v_s_u03b1_3561_);
v___x_3572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__2));
v___x_3573_ = lean_box(0);
v___x_3574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3574_, 0, v_u_3559_);
lean_ctor_set(v___x_3574_, 1, v___x_3573_);
lean_inc_ref_n(v___x_3574_, 4);
v___x_3575_ = l_Lean_Expr_const___override(v___x_3572_, v___x_3574_);
v___x_3576_ = l_Lean_Expr_app___override(v___x_3575_, v_00_u03b1_3560_);
v___x_3577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__5));
v___x_3578_ = l_Lean_Expr_const___override(v___x_3577_, v___x_3574_);
v___x_3579_ = l_Lean_Expr_app___override(v___x_3578_, v_00_u03b1_3560_);
v___x_3580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__8));
v___x_3581_ = l_Lean_Expr_const___override(v___x_3580_, v___x_3574_);
v___x_3582_ = l_Lean_Expr_app___override(v___x_3581_, v_00_u03b1_3560_);
v___x_3583_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__11));
v___x_3584_ = l_Lean_Expr_const___override(v___x_3583_, v___x_3574_);
v___x_3585_ = l_Lean_Expr_app___override(v___x_3584_, v_00_u03b1_3560_);
v___x_3586_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_3587_ = l_Lean_Expr_const___override(v___x_3586_, v___x_3574_);
v___x_3588_ = l_Lean_Expr_app___override(v___x_3587_, v_00_u03b1_3560_);
v___x_3589_ = l_Lean_Expr_app___override(v___x_3588_, v_s_u03b1_3561_);
v___x_3590_ = l_Lean_Expr_app___override(v___x_3585_, v___x_3589_);
v___x_3591_ = l_Lean_Expr_app___override(v___x_3582_, v___x_3590_);
v___x_3592_ = l_Lean_Expr_app___override(v___x_3579_, v___x_3591_);
v___x_3593_ = l_Lean_Expr_app___override(v___x_3576_, v___x_3592_);
v___x_3594_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__5);
v___x_3595_ = l_Lean_Expr_app___override(v___x_3593_, v___x_3594_);
v___x_3596_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0, &lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ringCompute___closed__0);
v___x_3597_ = l_Lean_Level_succ___override(v_u_3559_);
v___x_3598_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3598_, 0, v___x_3597_);
lean_ctor_set(v___x_3598_, 1, v___x_3573_);
v___x_3599_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_isOne___closed__15));
v___x_3600_ = l_Lean_Expr_const___override(v___x_3599_, v___x_3598_);
v___x_3601_ = l_Lean_Expr_app___override(v___x_3600_, v_00_u03b1_3560_);
lean_inc_ref(v___x_3595_);
v___x_3602_ = l_Lean_Expr_app___override(v___x_3601_, v___x_3595_);
v___x_3603_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3603_, 0, v___x_3595_);
lean_ctor_set(v___x_3603_, 1, v___x_3596_);
lean_ctor_set(v___x_3603_, 2, v___x_3602_);
v___x_3604_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3604_, 0, v___x_3563_);
lean_ctor_set(v___x_3604_, 1, v___x_3564_);
lean_ctor_set(v___x_3604_, 2, v___x_3565_);
lean_ctor_set(v___x_3604_, 3, v___x_3566_);
lean_ctor_set(v___x_3604_, 4, v___x_3567_);
lean_ctor_set(v___x_3604_, 5, v___x_3568_);
lean_ctor_set(v___x_3604_, 6, v___x_3569_);
lean_ctor_set(v___x_3604_, 7, v___x_3570_);
lean_ctor_set(v___x_3604_, 8, v___x_3571_);
lean_ctor_set(v___x_3604_, 9, v___x_3603_);
return v___x_3604_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1(void){
_start:
{
lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; 
v___x_3607_ = lean_box(0);
v___x_3608_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__0));
v___x_3609_ = l_Lean_Expr_const___override(v___x_3608_, v___x_3607_);
return v___x_3609_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2(void){
_start:
{
lean_object* v___x_3610_; lean_object* v___x_3611_; 
v___x_3610_ = lean_box(0);
v___x_3611_ = l_Lean_Level_succ___override(v___x_3610_);
return v___x_3611_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3(void){
_start:
{
lean_object* v___x_3612_; lean_object* v___x_3613_; lean_object* v___x_3614_; 
v___x_3612_ = lean_box(0);
v___x_3613_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__2);
v___x_3614_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3614_, 0, v___x_3613_);
lean_ctor_set(v___x_3614_, 1, v___x_3612_);
return v___x_3614_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4(void){
_start:
{
lean_object* v___x_3615_; lean_object* v___x_3616_; lean_object* v___x_3617_; 
v___x_3615_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__3);
v___x_3616_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_pow___closed__5));
v___x_3617_ = l_Lean_Expr_const___override(v___x_3616_, v___x_3615_);
return v___x_3617_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6(void){
_start:
{
lean_object* v___x_3620_; lean_object* v___x_3621_; lean_object* v___x_3622_; 
v___x_3620_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__16));
v___x_3621_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__5));
v___x_3622_ = l_Lean_Expr_const___override(v___x_3621_, v___x_3620_);
return v___x_3622_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7(void){
_start:
{
lean_object* v___x_3623_; lean_object* v___x_3624_; 
v___x_3623_ = lean_unsigned_to_nat(0u);
v___x_3624_ = l_Lean_Level_ofNat(v___x_3623_);
return v___x_3624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg(lean_object* v_u_3643_, lean_object* v_00_u03b1_3644_, lean_object* v_s_u03b1_3645_, lean_object* v_c_u03b1_3646_, lean_object* v_v_3647_, lean_object* v_00_u03b2_3648_, lean_object* v_s_u03b2_3649_, lean_object* v_x_3650_, lean_object* v_a_3651_, lean_object* v_a_3652_, lean_object* v_a_3653_, lean_object* v_a_3654_, lean_object* v_a_3655_, lean_object* v_a_3656_){
_start:
{
lean_object* v___y_3659_; lean_object* v___y_3660_; lean_object* v___y_3661_; lean_object* v___y_3662_; lean_object* v___x_3665_; 
lean_inc_ref(v_s_u03b2_3649_);
lean_inc_ref(v_00_u03b2_3648_);
lean_inc(v_v_3647_);
v___x_3665_ = lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(v_v_3647_, v_00_u03b2_3648_, v_s_u03b2_3649_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
if (lean_obj_tag(v___x_3665_) == 0)
{
lean_object* v_a_3666_; lean_object* v___x_3667_; lean_object* v___x_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; lean_object* v___x_3671_; lean_object* v___x_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; 
v_a_3666_ = lean_ctor_get(v___x_3665_, 0);
lean_inc_n(v_a_3666_, 2);
lean_dec_ref_known(v___x_3665_, 1);
v___x_3667_ = lean_box(0);
v___x_3668_ = lean_box(0);
v___x_3669_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_3670_ = lp_mathlib_Mathlib_Tactic_Ring_Common_s_u2115;
v___x_3671_ = lp_mathlib_Mathlib_Tactic_Ring_Common_Cache_nat;
v___x_3672_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v___x_3667_, v___x_3669_, v___x_3670_, v___x_3671_);
lean_inc_ref_n(v_s_u03b2_3649_, 2);
lean_inc_ref_n(v_00_u03b2_3648_, 2);
lean_inc_n(v_v_3647_, 2);
v___x_3673_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_v_3647_, v_00_u03b2_3648_, v_s_u03b2_3649_, v_a_3666_);
lean_inc_ref(v_x_3650_);
v___x_3674_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_3672_, v_v_3647_, v_00_u03b2_3648_, v_s_u03b2_3649_, v___x_3673_, v_a_3666_, v_x_3650_, v_a_3651_, v_a_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
if (lean_obj_tag(v___x_3674_) == 0)
{
lean_object* v_a_3675_; lean_object* v_expr_3676_; lean_object* v_val_3677_; lean_object* v_proof_3678_; lean_object* v___x_3679_; 
v_a_3675_ = lean_ctor_get(v___x_3674_, 0);
lean_inc(v_a_3675_);
lean_dec_ref_known(v___x_3674_, 1);
v_expr_3676_ = lean_ctor_get(v_a_3675_, 0);
lean_inc_ref(v_expr_3676_);
v_val_3677_ = lean_ctor_get(v_a_3675_, 1);
lean_inc(v_val_3677_);
v_proof_3678_ = lean_ctor_get(v_a_3675_, 2);
lean_inc_ref(v_proof_3678_);
lean_dec(v_a_3675_);
lean_inc_ref(v_s_u03b2_3649_);
lean_inc_ref(v_s_u03b1_3645_);
v___x_3679_ = l_Lean_Meta_isExprDefEq(v_s_u03b1_3645_, v_s_u03b2_3649_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
if (lean_obj_tag(v___x_3679_) == 0)
{
lean_object* v_a_3680_; lean_object* v___x_3682_; uint8_t v_isShared_3683_; uint8_t v_isSharedCheck_3868_; 
v_a_3680_ = lean_ctor_get(v___x_3679_, 0);
v_isSharedCheck_3868_ = !lean_is_exclusive(v___x_3679_);
if (v_isSharedCheck_3868_ == 0)
{
v___x_3682_ = v___x_3679_;
v_isShared_3683_ = v_isSharedCheck_3868_;
goto v_resetjp_3681_;
}
else
{
lean_inc(v_a_3680_);
lean_dec(v___x_3679_);
v___x_3682_ = lean_box(0);
v_isShared_3683_ = v_isSharedCheck_3868_;
goto v_resetjp_3681_;
}
v_resetjp_3681_:
{
uint8_t v___x_3684_; 
v___x_3684_ = lean_unbox(v_a_3680_);
if (v___x_3684_ == 0)
{
lean_object* v_r_u03b1_3685_; uint8_t v___x_3686_; lean_object* v___y_3688_; lean_object* v___y_3689_; lean_object* v___y_3690_; lean_object* v___y_3691_; lean_object* v___y_3692_; lean_object* v___y_3693_; 
lean_del_object(v___x_3682_);
v_r_u03b1_3685_ = lean_ctor_get(v_c_u03b1_3646_, 0);
lean_inc(v_r_u03b1_3685_);
lean_dec_ref(v_c_u03b1_3646_);
v___x_3686_ = 1;
if (lean_obj_tag(v_v_3647_) == 0)
{
uint8_t v___x_3765_; lean_object* v___x_3766_; lean_object* v___x_3767_; lean_object* v___f_3768_; uint8_t v___x_3769_; lean_object* v___x_3770_; 
v___x_3765_ = 2;
v___x_3766_ = lean_box(v___x_3765_);
v___x_3767_ = lean_box(v___x_3686_);
lean_inc_ref(v_00_u03b2_3648_);
v___f_3768_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_3768_, 0, v___x_3766_);
lean_closure_set(v___f_3768_, 1, v___x_3669_);
lean_closure_set(v___f_3768_, 2, v_00_u03b2_3648_);
lean_closure_set(v___f_3768_, 3, v___x_3767_);
v___x_3769_ = lean_unbox(v_a_3680_);
v___x_3770_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_3768_, v___x_3769_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
if (lean_obj_tag(v___x_3770_) == 0)
{
lean_object* v_a_3771_; uint8_t v___x_3772_; 
v_a_3771_ = lean_ctor_get(v___x_3770_, 0);
lean_inc(v_a_3771_);
lean_dec_ref_known(v___x_3770_, 1);
v___x_3772_ = lean_unbox(v_a_3771_);
lean_dec(v_a_3771_);
if (v___x_3772_ == 0)
{
v___y_3688_ = v_a_3651_;
v___y_3689_ = v_a_3652_;
v___y_3690_ = v_a_3653_;
v___y_3691_ = v_a_3654_;
v___y_3692_ = v_a_3655_;
v___y_3693_ = v_a_3656_;
goto v___jp_3687_;
}
else
{
lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_3776_; lean_object* v___x_3777_; lean_object* v___x_3778_; lean_object* v___x_3779_; lean_object* v___f_3780_; uint8_t v___x_3781_; lean_object* v___x_3782_; 
v___x_3773_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4);
v___x_3774_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6);
lean_inc_ref(v_00_u03b2_3648_);
v___x_3775_ = l_Lean_Expr_app___override(v___x_3774_, v_00_u03b2_3648_);
v___x_3776_ = l_Lean_Expr_app___override(v___x_3773_, v___x_3775_);
lean_inc_ref_n(v_s_u03b2_3649_, 2);
v___x_3777_ = l_Lean_Expr_app___override(v___x_3776_, v_s_u03b2_3649_);
v___x_3778_ = lean_box(v___x_3765_);
v___x_3779_ = lean_box(v___x_3686_);
v___f_3780_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_3780_, 0, v___x_3778_);
lean_closure_set(v___f_3780_, 1, v___x_3777_);
lean_closure_set(v___f_3780_, 2, v_s_u03b2_3649_);
lean_closure_set(v___f_3780_, 3, v___x_3779_);
v___x_3781_ = lean_unbox(v_a_3680_);
v___x_3782_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_3780_, v___x_3781_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
if (lean_obj_tag(v___x_3782_) == 0)
{
lean_object* v_a_3783_; uint8_t v___x_3784_; 
v_a_3783_ = lean_ctor_get(v___x_3782_, 0);
lean_inc(v_a_3783_);
lean_dec_ref_known(v___x_3782_, 1);
v___x_3784_ = lean_unbox(v_a_3783_);
lean_dec(v_a_3783_);
if (v___x_3784_ == 0)
{
v___y_3688_ = v_a_3651_;
v___y_3689_ = v_a_3652_;
v___y_3690_ = v_a_3653_;
v___y_3691_ = v_a_3654_;
v___y_3692_ = v_a_3655_;
v___y_3693_ = v_a_3656_;
goto v___jp_3687_;
}
else
{
lean_object* v___x_3785_; lean_object* v___x_3786_; 
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
v___x_3785_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7);
lean_inc_ref(v_s_u03b1_3645_);
lean_inc_ref(v_00_u03b1_3644_);
lean_inc(v_u_3643_);
v___x_3786_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalNatCast___redArg(v_u_3643_, v_00_u03b1_3644_, v_s_u03b1_3645_, v___x_3785_, v_00_u03b2_3648_, v_s_u03b2_3649_, v_val_3677_, v_a_3651_, v_a_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
if (lean_obj_tag(v___x_3786_) == 0)
{
lean_object* v_a_3787_; lean_object* v___x_3789_; uint8_t v_isShared_3790_; uint8_t v_isSharedCheck_3809_; 
v_a_3787_ = lean_ctor_get(v___x_3786_, 0);
v_isSharedCheck_3809_ = !lean_is_exclusive(v___x_3786_);
if (v_isSharedCheck_3809_ == 0)
{
v___x_3789_ = v___x_3786_;
v_isShared_3790_ = v_isSharedCheck_3809_;
goto v_resetjp_3788_;
}
else
{
lean_inc(v_a_3787_);
lean_dec(v___x_3786_);
v___x_3789_ = lean_box(0);
v_isShared_3790_ = v_isSharedCheck_3809_;
goto v_resetjp_3788_;
}
v_resetjp_3788_:
{
lean_object* v_expr_3791_; lean_object* v_val_3792_; lean_object* v_proof_3793_; lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3807_; 
v_expr_3791_ = lean_ctor_get(v_a_3787_, 0);
lean_inc_ref_n(v_expr_3791_, 2);
v_val_3792_ = lean_ctor_get(v_a_3787_, 1);
lean_inc(v_val_3792_);
v_proof_3793_ = lean_ctor_get(v_a_3787_, 2);
lean_inc_ref(v_proof_3793_);
lean_dec(v_a_3787_);
v___x_3794_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3794_, 0, v_u_3643_);
lean_ctor_set(v___x_3794_, 1, v___x_3668_);
v___x_3795_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__10));
v___x_3796_ = l_Lean_Expr_const___override(v___x_3795_, v___x_3794_);
v___x_3797_ = l_Lean_Expr_app___override(v___x_3796_, v_00_u03b1_3644_);
v___x_3798_ = l_Lean_Expr_app___override(v___x_3797_, v_s_u03b1_3645_);
v___x_3799_ = l_Lean_Expr_app___override(v___x_3798_, v_expr_3676_);
v___x_3800_ = l_Lean_Expr_app___override(v___x_3799_, v_x_3650_);
v___x_3801_ = l_Lean_Expr_app___override(v___x_3800_, v_expr_3791_);
v___x_3802_ = l_Lean_Expr_app___override(v___x_3801_, v_proof_3793_);
v___x_3803_ = l_Lean_Expr_app___override(v___x_3802_, v_proof_3678_);
v___x_3804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3804_, 0, v_val_3792_);
lean_ctor_set(v___x_3804_, 1, v___x_3803_);
v___x_3805_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3805_, 0, v_expr_3791_);
lean_ctor_set(v___x_3805_, 1, v___x_3804_);
if (v_isShared_3790_ == 0)
{
lean_ctor_set(v___x_3789_, 0, v___x_3805_);
v___x_3807_ = v___x_3789_;
goto v_reusejp_3806_;
}
else
{
lean_object* v_reuseFailAlloc_3808_; 
v_reuseFailAlloc_3808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3808_, 0, v___x_3805_);
v___x_3807_ = v_reuseFailAlloc_3808_;
goto v_reusejp_3806_;
}
v_reusejp_3806_:
{
return v___x_3807_;
}
}
}
else
{
lean_object* v_a_3810_; lean_object* v___x_3812_; uint8_t v_isShared_3813_; uint8_t v_isSharedCheck_3817_; 
lean_dec_ref(v_proof_3678_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3810_ = lean_ctor_get(v___x_3786_, 0);
v_isSharedCheck_3817_ = !lean_is_exclusive(v___x_3786_);
if (v_isSharedCheck_3817_ == 0)
{
v___x_3812_ = v___x_3786_;
v_isShared_3813_ = v_isSharedCheck_3817_;
goto v_resetjp_3811_;
}
else
{
lean_inc(v_a_3810_);
lean_dec(v___x_3786_);
v___x_3812_ = lean_box(0);
v_isShared_3813_ = v_isSharedCheck_3817_;
goto v_resetjp_3811_;
}
v_resetjp_3811_:
{
lean_object* v___x_3815_; 
if (v_isShared_3813_ == 0)
{
v___x_3815_ = v___x_3812_;
goto v_reusejp_3814_;
}
else
{
lean_object* v_reuseFailAlloc_3816_; 
v_reuseFailAlloc_3816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3816_, 0, v_a_3810_);
v___x_3815_ = v_reuseFailAlloc_3816_;
goto v_reusejp_3814_;
}
v_reusejp_3814_:
{
return v___x_3815_;
}
}
}
}
}
else
{
lean_object* v_a_3818_; lean_object* v___x_3820_; uint8_t v_isShared_3821_; uint8_t v_isSharedCheck_3825_; 
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3818_ = lean_ctor_get(v___x_3782_, 0);
v_isSharedCheck_3825_ = !lean_is_exclusive(v___x_3782_);
if (v_isSharedCheck_3825_ == 0)
{
v___x_3820_ = v___x_3782_;
v_isShared_3821_ = v_isSharedCheck_3825_;
goto v_resetjp_3819_;
}
else
{
lean_inc(v_a_3818_);
lean_dec(v___x_3782_);
v___x_3820_ = lean_box(0);
v_isShared_3821_ = v_isSharedCheck_3825_;
goto v_resetjp_3819_;
}
v_resetjp_3819_:
{
lean_object* v___x_3823_; 
if (v_isShared_3821_ == 0)
{
v___x_3823_ = v___x_3820_;
goto v_reusejp_3822_;
}
else
{
lean_object* v_reuseFailAlloc_3824_; 
v_reuseFailAlloc_3824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3824_, 0, v_a_3818_);
v___x_3823_ = v_reuseFailAlloc_3824_;
goto v_reusejp_3822_;
}
v_reusejp_3822_:
{
return v___x_3823_;
}
}
}
}
}
else
{
lean_object* v_a_3826_; lean_object* v___x_3828_; uint8_t v_isShared_3829_; uint8_t v_isSharedCheck_3833_; 
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3826_ = lean_ctor_get(v___x_3770_, 0);
v_isSharedCheck_3833_ = !lean_is_exclusive(v___x_3770_);
if (v_isSharedCheck_3833_ == 0)
{
v___x_3828_ = v___x_3770_;
v_isShared_3829_ = v_isSharedCheck_3833_;
goto v_resetjp_3827_;
}
else
{
lean_inc(v_a_3826_);
lean_dec(v___x_3770_);
v___x_3828_ = lean_box(0);
v_isShared_3829_ = v_isSharedCheck_3833_;
goto v_resetjp_3827_;
}
v_resetjp_3827_:
{
lean_object* v___x_3831_; 
if (v_isShared_3829_ == 0)
{
v___x_3831_ = v___x_3828_;
goto v_reusejp_3830_;
}
else
{
lean_object* v_reuseFailAlloc_3832_; 
v_reuseFailAlloc_3832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3832_, 0, v_a_3826_);
v___x_3831_ = v_reuseFailAlloc_3832_;
goto v_reusejp_3830_;
}
v_reusejp_3830_:
{
return v___x_3831_;
}
}
}
}
else
{
v___y_3688_ = v_a_3651_;
v___y_3689_ = v_a_3652_;
v___y_3690_ = v_a_3653_;
v___y_3691_ = v_a_3654_;
v___y_3692_ = v_a_3655_;
v___y_3693_ = v_a_3656_;
goto v___jp_3687_;
}
v___jp_3687_:
{
if (lean_obj_tag(v_v_3647_) == 0)
{
lean_object* v___x_3694_; uint8_t v___x_3695_; lean_object* v___x_3696_; lean_object* v___x_3697_; lean_object* v___f_3698_; uint8_t v___x_3699_; lean_object* v___x_3700_; 
v___x_3694_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__1);
v___x_3695_ = 2;
v___x_3696_ = lean_box(v___x_3695_);
v___x_3697_ = lean_box(v___x_3686_);
lean_inc_ref(v_00_u03b2_3648_);
v___f_3698_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_3698_, 0, v___x_3696_);
lean_closure_set(v___f_3698_, 1, v___x_3694_);
lean_closure_set(v___f_3698_, 2, v_00_u03b2_3648_);
lean_closure_set(v___f_3698_, 3, v___x_3697_);
v___x_3699_ = lean_unbox(v_a_3680_);
v___x_3700_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_3698_, v___x_3699_, v___y_3690_, v___y_3691_, v___y_3692_, v___y_3693_);
if (lean_obj_tag(v___x_3700_) == 0)
{
lean_object* v_a_3701_; uint8_t v___x_3702_; 
v_a_3701_ = lean_ctor_get(v___x_3700_, 0);
lean_inc(v_a_3701_);
lean_dec_ref_known(v___x_3700_, 1);
v___x_3702_ = lean_unbox(v_a_3701_);
lean_dec(v_a_3701_);
if (v___x_3702_ == 0)
{
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v___y_3659_ = v___y_3690_;
v___y_3660_ = v___y_3691_;
v___y_3661_ = v___y_3692_;
v___y_3662_ = v___y_3693_;
goto v___jp_3658_;
}
else
{
lean_object* v___x_3703_; lean_object* v___x_3704_; lean_object* v___x_3705_; lean_object* v___x_3706_; lean_object* v___x_3707_; lean_object* v___x_3708_; lean_object* v___x_3709_; lean_object* v___f_3710_; uint8_t v___x_3711_; lean_object* v___x_3712_; 
v___x_3703_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__4);
v___x_3704_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__6);
lean_inc_ref(v_00_u03b2_3648_);
v___x_3705_ = l_Lean_Expr_app___override(v___x_3704_, v_00_u03b2_3648_);
v___x_3706_ = l_Lean_Expr_app___override(v___x_3703_, v___x_3705_);
lean_inc_ref_n(v_s_u03b2_3649_, 2);
v___x_3707_ = l_Lean_Expr_app___override(v___x_3706_, v_s_u03b2_3649_);
v___x_3708_ = lean_box(v___x_3695_);
v___x_3709_ = lean_box(v___x_3686_);
v___f_3710_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_3710_, 0, v___x_3708_);
lean_closure_set(v___f_3710_, 1, v___x_3707_);
lean_closure_set(v___f_3710_, 2, v_s_u03b2_3649_);
lean_closure_set(v___f_3710_, 3, v___x_3709_);
v___x_3711_ = lean_unbox(v_a_3680_);
lean_dec(v_a_3680_);
v___x_3712_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__3___redArg(v___f_3710_, v___x_3711_, v___y_3690_, v___y_3691_, v___y_3692_, v___y_3693_);
if (lean_obj_tag(v___x_3712_) == 0)
{
lean_object* v_a_3713_; uint8_t v___x_3714_; 
v_a_3713_ = lean_ctor_get(v___x_3712_, 0);
lean_inc(v_a_3713_);
lean_dec_ref_known(v___x_3712_, 1);
v___x_3714_ = lean_unbox(v_a_3713_);
lean_dec(v_a_3713_);
if (v___x_3714_ == 0)
{
lean_dec(v_r_u03b1_3685_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v___y_3659_ = v___y_3690_;
v___y_3660_ = v___y_3691_;
v___y_3661_ = v___y_3692_;
v___y_3662_ = v___y_3693_;
goto v___jp_3658_;
}
else
{
if (lean_obj_tag(v_r_u03b1_3685_) == 1)
{
lean_object* v_val_3715_; lean_object* v___x_3716_; lean_object* v___x_3717_; 
v_val_3715_ = lean_ctor_get(v_r_u03b1_3685_, 0);
lean_inc_n(v_val_3715_, 2);
lean_dec_ref_known(v_r_u03b1_3685_, 1);
v___x_3716_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7);
lean_inc_ref(v_00_u03b1_3644_);
lean_inc(v_u_3643_);
v___x_3717_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_evalIntCast___redArg(v_u_3643_, v_00_u03b1_3644_, v_s_u03b1_3645_, v___x_3716_, v_00_u03b2_3648_, v_s_u03b2_3649_, v_val_3715_, v_val_3677_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_, v___y_3692_, v___y_3693_);
lean_dec_ref(v_s_u03b2_3649_);
if (lean_obj_tag(v___x_3717_) == 0)
{
lean_object* v_a_3718_; lean_object* v___x_3720_; uint8_t v_isShared_3721_; uint8_t v_isSharedCheck_3740_; 
v_a_3718_ = lean_ctor_get(v___x_3717_, 0);
v_isSharedCheck_3740_ = !lean_is_exclusive(v___x_3717_);
if (v_isSharedCheck_3740_ == 0)
{
v___x_3720_ = v___x_3717_;
v_isShared_3721_ = v_isSharedCheck_3740_;
goto v_resetjp_3719_;
}
else
{
lean_inc(v_a_3718_);
lean_dec(v___x_3717_);
v___x_3720_ = lean_box(0);
v_isShared_3721_ = v_isSharedCheck_3740_;
goto v_resetjp_3719_;
}
v_resetjp_3719_:
{
lean_object* v_expr_3722_; lean_object* v_val_3723_; lean_object* v_proof_3724_; lean_object* v___x_3725_; lean_object* v___x_3726_; lean_object* v___x_3727_; lean_object* v___x_3728_; lean_object* v___x_3729_; lean_object* v___x_3730_; lean_object* v___x_3731_; lean_object* v___x_3732_; lean_object* v___x_3733_; lean_object* v___x_3734_; lean_object* v___x_3735_; lean_object* v___x_3736_; lean_object* v___x_3738_; 
v_expr_3722_ = lean_ctor_get(v_a_3718_, 0);
lean_inc_ref_n(v_expr_3722_, 2);
v_val_3723_ = lean_ctor_get(v_a_3718_, 1);
lean_inc(v_val_3723_);
v_proof_3724_ = lean_ctor_get(v_a_3718_, 2);
lean_inc_ref(v_proof_3724_);
lean_dec(v_a_3718_);
v___x_3725_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3725_, 0, v_u_3643_);
lean_ctor_set(v___x_3725_, 1, v___x_3668_);
v___x_3726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__9));
v___x_3727_ = l_Lean_Expr_const___override(v___x_3726_, v___x_3725_);
v___x_3728_ = l_Lean_Expr_app___override(v___x_3727_, v_00_u03b1_3644_);
v___x_3729_ = l_Lean_Expr_app___override(v___x_3728_, v_expr_3676_);
v___x_3730_ = l_Lean_Expr_app___override(v___x_3729_, v_x_3650_);
v___x_3731_ = l_Lean_Expr_app___override(v___x_3730_, v_expr_3722_);
v___x_3732_ = l_Lean_Expr_app___override(v___x_3731_, v_val_3715_);
v___x_3733_ = l_Lean_Expr_app___override(v___x_3732_, v_proof_3724_);
v___x_3734_ = l_Lean_Expr_app___override(v___x_3733_, v_proof_3678_);
v___x_3735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3735_, 0, v_val_3723_);
lean_ctor_set(v___x_3735_, 1, v___x_3734_);
v___x_3736_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3736_, 0, v_expr_3722_);
lean_ctor_set(v___x_3736_, 1, v___x_3735_);
if (v_isShared_3721_ == 0)
{
lean_ctor_set(v___x_3720_, 0, v___x_3736_);
v___x_3738_ = v___x_3720_;
goto v_reusejp_3737_;
}
else
{
lean_object* v_reuseFailAlloc_3739_; 
v_reuseFailAlloc_3739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3739_, 0, v___x_3736_);
v___x_3738_ = v_reuseFailAlloc_3739_;
goto v_reusejp_3737_;
}
v_reusejp_3737_:
{
return v___x_3738_;
}
}
}
else
{
lean_object* v_a_3741_; lean_object* v___x_3743_; uint8_t v_isShared_3744_; uint8_t v_isSharedCheck_3748_; 
lean_dec(v_val_3715_);
lean_dec_ref(v_proof_3678_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3741_ = lean_ctor_get(v___x_3717_, 0);
v_isSharedCheck_3748_ = !lean_is_exclusive(v___x_3717_);
if (v_isSharedCheck_3748_ == 0)
{
v___x_3743_ = v___x_3717_;
v_isShared_3744_ = v_isSharedCheck_3748_;
goto v_resetjp_3742_;
}
else
{
lean_inc(v_a_3741_);
lean_dec(v___x_3717_);
v___x_3743_ = lean_box(0);
v_isShared_3744_ = v_isSharedCheck_3748_;
goto v_resetjp_3742_;
}
v_resetjp_3742_:
{
lean_object* v___x_3746_; 
if (v_isShared_3744_ == 0)
{
v___x_3746_ = v___x_3743_;
goto v_reusejp_3745_;
}
else
{
lean_object* v_reuseFailAlloc_3747_; 
v_reuseFailAlloc_3747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3747_, 0, v_a_3741_);
v___x_3746_ = v_reuseFailAlloc_3747_;
goto v_reusejp_3745_;
}
v_reusejp_3745_:
{
return v___x_3746_;
}
}
}
}
else
{
lean_dec(v_r_u03b1_3685_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v___y_3659_ = v___y_3690_;
v___y_3660_ = v___y_3691_;
v___y_3661_ = v___y_3692_;
v___y_3662_ = v___y_3693_;
goto v___jp_3658_;
}
}
}
else
{
lean_object* v_a_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3756_; 
lean_dec(v_r_u03b1_3685_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3749_ = lean_ctor_get(v___x_3712_, 0);
v_isSharedCheck_3756_ = !lean_is_exclusive(v___x_3712_);
if (v_isSharedCheck_3756_ == 0)
{
v___x_3751_ = v___x_3712_;
v_isShared_3752_ = v_isSharedCheck_3756_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_a_3749_);
lean_dec(v___x_3712_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3756_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v___x_3754_; 
if (v_isShared_3752_ == 0)
{
v___x_3754_ = v___x_3751_;
goto v_reusejp_3753_;
}
else
{
lean_object* v_reuseFailAlloc_3755_; 
v_reuseFailAlloc_3755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3755_, 0, v_a_3749_);
v___x_3754_ = v_reuseFailAlloc_3755_;
goto v_reusejp_3753_;
}
v_reusejp_3753_:
{
return v___x_3754_;
}
}
}
}
}
else
{
lean_object* v_a_3757_; lean_object* v___x_3759_; uint8_t v_isShared_3760_; uint8_t v_isSharedCheck_3764_; 
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3757_ = lean_ctor_get(v___x_3700_, 0);
v_isSharedCheck_3764_ = !lean_is_exclusive(v___x_3700_);
if (v_isSharedCheck_3764_ == 0)
{
v___x_3759_ = v___x_3700_;
v_isShared_3760_ = v_isSharedCheck_3764_;
goto v_resetjp_3758_;
}
else
{
lean_inc(v_a_3757_);
lean_dec(v___x_3700_);
v___x_3759_ = lean_box(0);
v_isShared_3760_ = v_isSharedCheck_3764_;
goto v_resetjp_3758_;
}
v_resetjp_3758_:
{
lean_object* v___x_3762_; 
if (v_isShared_3760_ == 0)
{
v___x_3762_ = v___x_3759_;
goto v_reusejp_3761_;
}
else
{
lean_object* v_reuseFailAlloc_3763_; 
v_reuseFailAlloc_3763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3763_, 0, v_a_3757_);
v___x_3762_ = v_reuseFailAlloc_3763_;
goto v_reusejp_3761_;
}
v_reusejp_3761_:
{
return v___x_3762_;
}
}
}
}
else
{
lean_dec(v_r_u03b1_3685_);
lean_dec(v_a_3680_);
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec(v_v_3647_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v___y_3659_ = v___y_3690_;
v___y_3660_ = v___y_3691_;
v___y_3661_ = v___y_3692_;
v___y_3662_ = v___y_3693_;
goto v___jp_3658_;
}
}
}
else
{
lean_object* v___x_3834_; lean_object* v_fst_3835_; lean_object* v_snd_3836_; lean_object* v___x_3838_; uint8_t v_isShared_3839_; uint8_t v_isSharedCheck_3867_; 
lean_dec(v_a_3680_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_c_u03b1_3646_);
lean_inc_ref(v_s_u03b1_3645_);
lean_inc_ref(v_00_u03b1_3644_);
v___x_3834_ = lp_mathlib_Mathlib_Tactic_Ring_ExSum_cast___redArg(v_v_3647_, v_00_u03b2_3648_, v_s_u03b2_3649_, v_u_3643_, v_00_u03b1_3644_, v_s_u03b1_3645_, v_val_3677_);
lean_dec_ref(v_s_u03b2_3649_);
v_fst_3835_ = lean_ctor_get(v___x_3834_, 0);
v_snd_3836_ = lean_ctor_get(v___x_3834_, 1);
v_isSharedCheck_3867_ = !lean_is_exclusive(v___x_3834_);
if (v_isSharedCheck_3867_ == 0)
{
v___x_3838_ = v___x_3834_;
v_isShared_3839_ = v_isSharedCheck_3867_;
goto v_resetjp_3837_;
}
else
{
lean_inc(v_snd_3836_);
lean_inc(v_fst_3835_);
lean_dec(v___x_3834_);
v___x_3838_ = lean_box(0);
v_isShared_3839_ = v_isSharedCheck_3867_;
goto v_resetjp_3837_;
}
v_resetjp_3837_:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v___x_3844_; lean_object* v___x_3845_; lean_object* v___x_3846_; lean_object* v___x_3847_; lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_3856_; lean_object* v___x_3857_; lean_object* v___x_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3862_; 
v___x_3840_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__8));
v___x_3841_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_evalCast___closed__9));
v___x_3842_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ExProd_mkNat___closed__14));
v___x_3843_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__11));
v___x_3844_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3844_, 0, v_v_3647_);
lean_ctor_set(v___x_3844_, 1, v___x_3668_);
lean_inc_ref_n(v___x_3844_, 3);
v___x_3845_ = l_Lean_Expr_const___override(v___x_3843_, v___x_3844_);
v___x_3846_ = l_Lean_Expr_app___override(v___x_3845_, v_00_u03b1_3644_);
v___x_3847_ = l_Lean_Expr_const___override(v___x_3840_, v___x_3844_);
lean_inc_ref_n(v_00_u03b2_3648_, 2);
v___x_3848_ = l_Lean_Expr_app___override(v___x_3847_, v_00_u03b2_3648_);
v___x_3849_ = l_Lean_Expr_const___override(v___x_3841_, v___x_3844_);
v___x_3850_ = l_Lean_Expr_app___override(v___x_3849_, v_00_u03b2_3648_);
v___x_3851_ = l_Lean_Expr_const___override(v___x_3842_, v___x_3844_);
v___x_3852_ = l_Lean_Expr_app___override(v___x_3851_, v_00_u03b2_3648_);
v___x_3853_ = l_Lean_Expr_app___override(v___x_3852_, v_s_u03b1_3645_);
v___x_3854_ = l_Lean_Expr_app___override(v___x_3850_, v___x_3853_);
v___x_3855_ = l_Lean_Expr_app___override(v___x_3848_, v___x_3854_);
v___x_3856_ = l_Lean_Expr_app___override(v___x_3846_, v___x_3855_);
v___x_3857_ = l_Lean_Expr_app___override(v___x_3856_, v_x_3650_);
lean_inc(v_fst_3835_);
v___x_3858_ = l_Lean_Expr_app___override(v___x_3857_, v_fst_3835_);
v___x_3859_ = l_Lean_Expr_app___override(v___x_3858_, v_proof_3678_);
v___x_3860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3860_, 0, v_snd_3836_);
lean_ctor_set(v___x_3860_, 1, v___x_3859_);
if (v_isShared_3839_ == 0)
{
lean_ctor_set(v___x_3838_, 1, v___x_3860_);
v___x_3862_ = v___x_3838_;
goto v_reusejp_3861_;
}
else
{
lean_object* v_reuseFailAlloc_3866_; 
v_reuseFailAlloc_3866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3866_, 0, v_fst_3835_);
lean_ctor_set(v_reuseFailAlloc_3866_, 1, v___x_3860_);
v___x_3862_ = v_reuseFailAlloc_3866_;
goto v_reusejp_3861_;
}
v_reusejp_3861_:
{
lean_object* v___x_3864_; 
if (v_isShared_3683_ == 0)
{
lean_ctor_set(v___x_3682_, 0, v___x_3862_);
v___x_3864_ = v___x_3682_;
goto v_reusejp_3863_;
}
else
{
lean_object* v_reuseFailAlloc_3865_; 
v_reuseFailAlloc_3865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3865_, 0, v___x_3862_);
v___x_3864_ = v_reuseFailAlloc_3865_;
goto v_reusejp_3863_;
}
v_reusejp_3863_:
{
return v___x_3864_;
}
}
}
}
}
}
else
{
lean_object* v_a_3869_; lean_object* v___x_3871_; uint8_t v_isShared_3872_; uint8_t v_isSharedCheck_3876_; 
lean_dec_ref(v_proof_3678_);
lean_dec(v_val_3677_);
lean_dec_ref(v_expr_3676_);
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec(v_v_3647_);
lean_dec_ref(v_c_u03b1_3646_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3869_ = lean_ctor_get(v___x_3679_, 0);
v_isSharedCheck_3876_ = !lean_is_exclusive(v___x_3679_);
if (v_isSharedCheck_3876_ == 0)
{
v___x_3871_ = v___x_3679_;
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
else
{
lean_inc(v_a_3869_);
lean_dec(v___x_3679_);
v___x_3871_ = lean_box(0);
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
v_resetjp_3870_:
{
lean_object* v___x_3874_; 
if (v_isShared_3872_ == 0)
{
v___x_3874_ = v___x_3871_;
goto v_reusejp_3873_;
}
else
{
lean_object* v_reuseFailAlloc_3875_; 
v_reuseFailAlloc_3875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3875_, 0, v_a_3869_);
v___x_3874_ = v_reuseFailAlloc_3875_;
goto v_reusejp_3873_;
}
v_reusejp_3873_:
{
return v___x_3874_;
}
}
}
}
else
{
lean_object* v_a_3877_; lean_object* v___x_3879_; uint8_t v_isShared_3880_; uint8_t v_isSharedCheck_3884_; 
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec(v_v_3647_);
lean_dec_ref(v_c_u03b1_3646_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3877_ = lean_ctor_get(v___x_3674_, 0);
v_isSharedCheck_3884_ = !lean_is_exclusive(v___x_3674_);
if (v_isSharedCheck_3884_ == 0)
{
v___x_3879_ = v___x_3674_;
v_isShared_3880_ = v_isSharedCheck_3884_;
goto v_resetjp_3878_;
}
else
{
lean_inc(v_a_3877_);
lean_dec(v___x_3674_);
v___x_3879_ = lean_box(0);
v_isShared_3880_ = v_isSharedCheck_3884_;
goto v_resetjp_3878_;
}
v_resetjp_3878_:
{
lean_object* v___x_3882_; 
if (v_isShared_3880_ == 0)
{
v___x_3882_ = v___x_3879_;
goto v_reusejp_3881_;
}
else
{
lean_object* v_reuseFailAlloc_3883_; 
v_reuseFailAlloc_3883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3883_, 0, v_a_3877_);
v___x_3882_ = v_reuseFailAlloc_3883_;
goto v_reusejp_3881_;
}
v_reusejp_3881_:
{
return v___x_3882_;
}
}
}
}
else
{
lean_object* v_a_3885_; lean_object* v___x_3887_; uint8_t v_isShared_3888_; uint8_t v_isSharedCheck_3892_; 
lean_dec_ref(v_x_3650_);
lean_dec_ref(v_s_u03b2_3649_);
lean_dec_ref(v_00_u03b2_3648_);
lean_dec(v_v_3647_);
lean_dec_ref(v_c_u03b1_3646_);
lean_dec_ref(v_s_u03b1_3645_);
lean_dec_ref(v_00_u03b1_3644_);
lean_dec(v_u_3643_);
v_a_3885_ = lean_ctor_get(v___x_3665_, 0);
v_isSharedCheck_3892_ = !lean_is_exclusive(v___x_3665_);
if (v_isSharedCheck_3892_ == 0)
{
v___x_3887_ = v___x_3665_;
v_isShared_3888_ = v_isSharedCheck_3892_;
goto v_resetjp_3886_;
}
else
{
lean_inc(v_a_3885_);
lean_dec(v___x_3665_);
v___x_3887_ = lean_box(0);
v_isShared_3888_ = v_isSharedCheck_3892_;
goto v_resetjp_3886_;
}
v_resetjp_3886_:
{
lean_object* v___x_3890_; 
if (v_isShared_3888_ == 0)
{
v___x_3890_ = v___x_3887_;
goto v_reusejp_3889_;
}
else
{
lean_object* v_reuseFailAlloc_3891_; 
v_reuseFailAlloc_3891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3891_, 0, v_a_3885_);
v___x_3890_ = v_reuseFailAlloc_3891_;
goto v_reusejp_3889_;
}
v_reusejp_3889_:
{
return v___x_3890_;
}
}
}
v___jp_3658_:
{
lean_object* v___x_3663_; lean_object* v___x_3664_; 
v___x_3663_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_3664_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_3663_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
return v___x_3664_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast(lean_object* v_u_3893_, lean_object* v_00_u03b1_3894_, lean_object* v_s_u03b1_3895_, lean_object* v_c_u03b1_3896_, lean_object* v_v_3897_, lean_object* v_00_u03b2_3898_, lean_object* v_s_u03b2_3899_, lean_object* v___smul_3900_, lean_object* v_x_3901_, lean_object* v_a_3902_, lean_object* v_a_3903_, lean_object* v_a_3904_, lean_object* v_a_3905_, lean_object* v_a_3906_, lean_object* v_a_3907_){
_start:
{
lean_object* v___x_3909_; 
v___x_3909_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg(v_u_3893_, v_00_u03b1_3894_, v_s_u03b1_3895_, v_c_u03b1_3896_, v_v_3897_, v_00_u03b2_3898_, v_s_u03b2_3899_, v_x_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_, v_a_3906_, v_a_3907_);
return v___x_3909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___boxed(lean_object* v_u_3910_, lean_object* v_00_u03b1_3911_, lean_object* v_s_u03b1_3912_, lean_object* v_c_u03b1_3913_, lean_object* v_v_3914_, lean_object* v_00_u03b2_3915_, lean_object* v_s_u03b2_3916_, lean_object* v_x_3917_, lean_object* v_a_3918_, lean_object* v_a_3919_, lean_object* v_a_3920_, lean_object* v_a_3921_, lean_object* v_a_3922_, lean_object* v_a_3923_, lean_object* v_a_3924_){
_start:
{
lean_object* v_res_3925_; 
v_res_3925_ = lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg(v_u_3910_, v_00_u03b1_3911_, v_s_u03b1_3912_, v_c_u03b1_3913_, v_v_3914_, v_00_u03b2_3915_, v_s_u03b2_3916_, v_x_3917_, v_a_3918_, v_a_3919_, v_a_3920_, v_a_3921_, v_a_3922_, v_a_3923_);
lean_dec(v_a_3923_);
lean_dec_ref(v_a_3922_);
lean_dec(v_a_3921_);
lean_dec_ref(v_a_3920_);
lean_dec(v_a_3919_);
lean_dec_ref(v_a_3918_);
return v_res_3925_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0(void){
_start:
{
lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; 
v___x_3926_ = lp_mathlib_Mathlib_Tactic_Ring_Common_Cache_nat;
v___x_3927_ = lp_mathlib_Mathlib_Tactic_Ring_Common_s_u2115;
v___x_3928_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_ExProd_evalNatCast___closed__13);
v___x_3929_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__7);
v___x_3930_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v___x_3929_, v___x_3928_, v___x_3927_, v___x_3926_);
return v___x_3930_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_rc_u2115(void){
_start:
{
lean_object* v___x_3931_; 
v___x_3931_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0, &lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Ring_rc_u2115___closed__0);
return v___x_3931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_(lean_object* v___y_3932_, lean_object* v___y_3933_, lean_object* v___y_3934_, lean_object* v___y_3935_, lean_object* v___y_3936_){
_start:
{
lean_object* v___x_3938_; 
v___x_3938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3938_, 0, v___y_3932_);
return v___x_3938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2____boxed(lean_object* v___y_3939_, lean_object* v___y_3940_, lean_object* v___y_3941_, lean_object* v___y_3942_, lean_object* v___y_3943_, lean_object* v___y_3944_){
_start:
{
lean_object* v_res_3945_; 
v_res_3945_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___lam__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_(v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
lean_dec(v___y_3943_);
lean_dec_ref(v___y_3942_);
lean_dec(v___y_3941_);
lean_dec_ref(v___y_3940_);
return v_res_3945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_3948_; lean_object* v___x_3949_; lean_object* v___x_3950_; 
v___f_3948_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn___closed__0_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_));
v___x_3949_ = lean_st_mk_ref(v___f_3948_);
v___x_3950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3950_, 0, v___x_3949_);
return v___x_3950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2____boxed(lean_object* v_a_3951_){
_start:
{
lean_object* v_res_3952_; 
v_res_3952_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_();
return v_res_3952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg(lean_object* v_category_3953_, lean_object* v_opts_3954_, lean_object* v_act_3955_, lean_object* v_decl_3956_, lean_object* v___y_3957_, lean_object* v___y_3958_, lean_object* v___y_3959_, lean_object* v___y_3960_, lean_object* v___y_3961_, lean_object* v___y_3962_){
_start:
{
lean_object* v___x_3964_; lean_object* v___x_3965_; 
lean_inc(v___y_3962_);
lean_inc_ref(v___y_3961_);
lean_inc(v___y_3960_);
lean_inc_ref(v___y_3959_);
lean_inc(v___y_3958_);
lean_inc_ref(v___y_3957_);
v___x_3964_ = lean_apply_6(v_act_3955_, v___y_3957_, v___y_3958_, v___y_3959_, v___y_3960_, v___y_3961_, v___y_3962_);
v___x_3965_ = l_Lean_profileitIOUnsafe___redArg(v_category_3953_, v_opts_3954_, v___x_3964_, v_decl_3956_);
return v___x_3965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg___boxed(lean_object* v_category_3966_, lean_object* v_opts_3967_, lean_object* v_act_3968_, lean_object* v_decl_3969_, lean_object* v___y_3970_, lean_object* v___y_3971_, lean_object* v___y_3972_, lean_object* v___y_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_){
_start:
{
lean_object* v_res_3977_; 
v_res_3977_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg(v_category_3966_, v_opts_3967_, v_act_3968_, v_decl_3969_, v___y_3970_, v___y_3971_, v___y_3972_, v___y_3973_, v___y_3974_, v___y_3975_);
lean_dec(v___y_3975_);
lean_dec_ref(v___y_3974_);
lean_dec(v___y_3973_);
lean_dec_ref(v___y_3972_);
lean_dec(v___y_3971_);
lean_dec_ref(v___y_3970_);
lean_dec_ref(v_opts_3967_);
lean_dec_ref(v_category_3966_);
return v_res_3977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0(lean_object* v_00_u03b1_3978_, lean_object* v_category_3979_, lean_object* v_opts_3980_, lean_object* v_act_3981_, lean_object* v_decl_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_, lean_object* v___y_3987_, lean_object* v___y_3988_){
_start:
{
lean_object* v___x_3990_; 
v___x_3990_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg(v_category_3979_, v_opts_3980_, v_act_3981_, v_decl_3982_, v___y_3983_, v___y_3984_, v___y_3985_, v___y_3986_, v___y_3987_, v___y_3988_);
return v___x_3990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___boxed(lean_object* v_00_u03b1_3991_, lean_object* v_category_3992_, lean_object* v_opts_3993_, lean_object* v_act_3994_, lean_object* v_decl_3995_, lean_object* v___y_3996_, lean_object* v___y_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_, lean_object* v___y_4002_){
_start:
{
lean_object* v_res_4003_; 
v_res_4003_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0(v_00_u03b1_3991_, v_category_3992_, v_opts_3993_, v_act_3994_, v_decl_3995_, v___y_3996_, v___y_3997_, v___y_3998_, v___y_3999_, v___y_4000_, v___y_4001_);
lean_dec(v___y_4001_);
lean_dec_ref(v___y_4000_);
lean_dec(v___y_3999_);
lean_dec_ref(v___y_3998_);
lean_dec(v___y_3997_);
lean_dec_ref(v___y_3996_);
lean_dec_ref(v_opts_3993_);
lean_dec_ref(v_category_3992_);
return v_res_4003_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4(void){
_start:
{
lean_object* v___x_4013_; lean_object* v___x_4014_; 
v___x_4013_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__3));
v___x_4014_ = l_Lean_stringToMessageData(v___x_4013_);
return v___x_4014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0(lean_object* v___x_4015_, lean_object* v_v_4016_, lean_object* v_00_u03b1_4017_, lean_object* v_s_u03b1_4018_, lean_object* v___x_4019_, lean_object* v_a_4020_, lean_object* v_e_u2081_4021_, lean_object* v_e_u2082_4022_, lean_object* v___y_4023_, lean_object* v___y_4024_, lean_object* v___y_4025_, lean_object* v___y_4026_, lean_object* v___y_4027_, lean_object* v___y_4028_){
_start:
{
lean_object* v___x_4030_; 
lean_inc_ref(v_e_u2081_4021_);
lean_inc_ref(v_a_4020_);
lean_inc_ref(v___x_4019_);
lean_inc_ref(v_s_u03b1_4018_);
lean_inc_ref(v_00_u03b1_4017_);
lean_inc(v_v_4016_);
lean_inc_ref(v___x_4015_);
v___x_4030_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_4015_, v_v_4016_, v_00_u03b1_4017_, v_s_u03b1_4018_, v___x_4019_, v_a_4020_, v_e_u2081_4021_, v___y_4023_, v___y_4024_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_);
if (lean_obj_tag(v___x_4030_) == 0)
{
lean_object* v_a_4031_; lean_object* v_expr_4032_; lean_object* v_val_4033_; lean_object* v_proof_4034_; lean_object* v___x_4035_; 
v_a_4031_ = lean_ctor_get(v___x_4030_, 0);
lean_inc(v_a_4031_);
lean_dec_ref_known(v___x_4030_, 1);
v_expr_4032_ = lean_ctor_get(v_a_4031_, 0);
lean_inc_ref(v_expr_4032_);
v_val_4033_ = lean_ctor_get(v_a_4031_, 1);
lean_inc(v_val_4033_);
v_proof_4034_ = lean_ctor_get(v_a_4031_, 2);
lean_inc_ref(v_proof_4034_);
lean_dec(v_a_4031_);
lean_inc_ref(v_e_u2082_4022_);
lean_inc_ref(v___x_4019_);
lean_inc_ref(v_s_u03b1_4018_);
lean_inc_ref(v_00_u03b1_4017_);
lean_inc(v_v_4016_);
lean_inc_ref(v___x_4015_);
v___x_4035_ = lp_mathlib_Mathlib_Tactic_Ring_Common_eval___redArg(v___x_4015_, v_v_4016_, v_00_u03b1_4017_, v_s_u03b1_4018_, v___x_4019_, v_a_4020_, v_e_u2082_4022_, v___y_4023_, v___y_4024_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_);
if (lean_obj_tag(v___x_4035_) == 0)
{
lean_object* v_a_4036_; lean_object* v___x_4038_; uint8_t v_isShared_4039_; uint8_t v_isSharedCheck_4103_; 
v_a_4036_ = lean_ctor_get(v___x_4035_, 0);
v_isSharedCheck_4103_ = !lean_is_exclusive(v___x_4035_);
if (v_isSharedCheck_4103_ == 0)
{
v___x_4038_ = v___x_4035_;
v_isShared_4039_ = v_isSharedCheck_4103_;
goto v_resetjp_4037_;
}
else
{
lean_inc(v_a_4036_);
lean_dec(v___x_4035_);
v___x_4038_ = lean_box(0);
v_isShared_4039_ = v_isSharedCheck_4103_;
goto v_resetjp_4037_;
}
v_resetjp_4037_:
{
lean_object* v_expr_4040_; lean_object* v_val_4041_; lean_object* v_proof_4042_; lean_object* v_toRingCompare_4058_; lean_object* v_toRingCompare_4059_; uint8_t v___x_4060_; 
v_expr_4040_ = lean_ctor_get(v_a_4036_, 0);
lean_inc_ref(v_expr_4040_);
v_val_4041_ = lean_ctor_get(v_a_4036_, 1);
lean_inc(v_val_4041_);
v_proof_4042_ = lean_ctor_get(v_a_4036_, 2);
lean_inc_ref(v_proof_4042_);
lean_dec(v_a_4036_);
v_toRingCompare_4058_ = lean_ctor_get(v___x_4015_, 0);
lean_inc_ref(v_toRingCompare_4058_);
lean_dec_ref(v___x_4015_);
v_toRingCompare_4059_ = lean_ctor_get(v___x_4019_, 0);
lean_inc_ref(v_toRingCompare_4059_);
lean_dec_ref(v___x_4019_);
v___x_4060_ = lp_mathlib_Mathlib_Tactic_Ring_Common_ExSum_eq___redArg(v_toRingCompare_4058_, v_v_4016_, v_00_u03b1_4017_, v_s_u03b1_4018_, v_toRingCompare_4059_, v_val_4033_, v_val_4041_);
lean_dec_ref(v_s_u03b1_4018_);
if (v___x_4060_ == 0)
{
lean_object* v___x_4061_; lean_object* v___x_4062_; lean_object* v___x_4063_; lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4066_; lean_object* v___x_4067_; lean_object* v___x_4068_; lean_object* v___x_4069_; lean_object* v___x_4070_; lean_object* v___x_4071_; 
lean_dec_ref(v_proof_4042_);
lean_del_object(v___x_4038_);
lean_dec_ref(v_proof_4034_);
lean_dec_ref(v_e_u2082_4022_);
lean_dec_ref(v_e_u2081_4021_);
v___x_4061_ = lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef;
v___x_4062_ = lean_st_ref_get(v___x_4061_);
v___x_4063_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__2));
v___x_4064_ = l_Lean_Level_succ___override(v_v_4016_);
v___x_4065_ = lean_box(0);
v___x_4066_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4066_, 0, v___x_4064_);
lean_ctor_set(v___x_4066_, 1, v___x_4065_);
v___x_4067_ = l_Lean_Expr_const___override(v___x_4063_, v___x_4066_);
v___x_4068_ = l_Lean_Expr_app___override(v___x_4067_, v_00_u03b1_4017_);
v___x_4069_ = l_Lean_Expr_app___override(v___x_4068_, v_expr_4032_);
v___x_4070_ = l_Lean_Expr_app___override(v___x_4069_, v_expr_4040_);
lean_inc(v___y_4028_);
lean_inc_ref(v___y_4027_);
lean_inc(v___y_4026_);
lean_inc_ref(v___y_4025_);
v___x_4071_ = lean_apply_6(v___x_4062_, v___x_4070_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_, lean_box(0));
if (lean_obj_tag(v___x_4071_) == 0)
{
lean_object* v_a_4072_; lean_object* v___x_4074_; uint8_t v_isShared_4075_; uint8_t v_isSharedCheck_4102_; 
v_a_4072_ = lean_ctor_get(v___x_4071_, 0);
v_isSharedCheck_4102_ = !lean_is_exclusive(v___x_4071_);
if (v_isSharedCheck_4102_ == 0)
{
v___x_4074_ = v___x_4071_;
v_isShared_4075_ = v_isSharedCheck_4102_;
goto v_resetjp_4073_;
}
else
{
lean_inc(v_a_4072_);
lean_dec(v___x_4071_);
v___x_4074_ = lean_box(0);
v_isShared_4075_ = v_isSharedCheck_4102_;
goto v_resetjp_4073_;
}
v_resetjp_4073_:
{
lean_object* v___x_4077_; 
if (v_isShared_4075_ == 0)
{
lean_ctor_set_tag(v___x_4074_, 1);
v___x_4077_ = v___x_4074_;
goto v_reusejp_4076_;
}
else
{
lean_object* v_reuseFailAlloc_4101_; 
v_reuseFailAlloc_4101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4101_, 0, v_a_4072_);
v___x_4077_ = v_reuseFailAlloc_4101_;
goto v_reusejp_4076_;
}
v_reusejp_4076_:
{
uint8_t v___x_4078_; lean_object* v___x_4079_; lean_object* v___x_4080_; 
v___x_4078_ = 0;
v___x_4079_ = lean_box(0);
v___x_4080_ = l_Lean_Meta_mkFreshExprMVar(v___x_4077_, v___x_4078_, v___x_4079_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_);
if (lean_obj_tag(v___x_4080_) == 0)
{
lean_object* v_a_4081_; lean_object* v___x_4083_; uint8_t v_isShared_4084_; uint8_t v_isSharedCheck_4100_; 
v_a_4081_ = lean_ctor_get(v___x_4080_, 0);
v_isSharedCheck_4100_ = !lean_is_exclusive(v___x_4080_);
if (v_isSharedCheck_4100_ == 0)
{
v___x_4083_ = v___x_4080_;
v_isShared_4084_ = v_isSharedCheck_4100_;
goto v_resetjp_4082_;
}
else
{
lean_inc(v_a_4081_);
lean_dec(v___x_4080_);
v___x_4083_ = lean_box(0);
v_isShared_4084_ = v_isSharedCheck_4100_;
goto v_resetjp_4082_;
}
v_resetjp_4082_:
{
lean_object* v___x_4085_; lean_object* v___x_4086_; lean_object* v___x_4088_; 
v___x_4085_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__4);
v___x_4086_ = l_Lean_Expr_mvarId_x21(v_a_4081_);
lean_dec(v_a_4081_);
if (v_isShared_4084_ == 0)
{
lean_ctor_set_tag(v___x_4083_, 1);
lean_ctor_set(v___x_4083_, 0, v___x_4086_);
v___x_4088_ = v___x_4083_;
goto v_reusejp_4087_;
}
else
{
lean_object* v_reuseFailAlloc_4099_; 
v_reuseFailAlloc_4099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4099_, 0, v___x_4086_);
v___x_4088_ = v_reuseFailAlloc_4099_;
goto v_reusejp_4087_;
}
v_reusejp_4087_:
{
lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v_a_4091_; lean_object* v___x_4093_; uint8_t v_isShared_4094_; uint8_t v_isSharedCheck_4098_; 
v___x_4089_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4089_, 0, v___x_4085_);
lean_ctor_set(v___x_4089_, 1, v___x_4088_);
v___x_4090_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v___x_4089_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_);
v_a_4091_ = lean_ctor_get(v___x_4090_, 0);
v_isSharedCheck_4098_ = !lean_is_exclusive(v___x_4090_);
if (v_isSharedCheck_4098_ == 0)
{
v___x_4093_ = v___x_4090_;
v_isShared_4094_ = v_isSharedCheck_4098_;
goto v_resetjp_4092_;
}
else
{
lean_inc(v_a_4091_);
lean_dec(v___x_4090_);
v___x_4093_ = lean_box(0);
v_isShared_4094_ = v_isSharedCheck_4098_;
goto v_resetjp_4092_;
}
v_resetjp_4092_:
{
lean_object* v___x_4096_; 
if (v_isShared_4094_ == 0)
{
v___x_4096_ = v___x_4093_;
goto v_reusejp_4095_;
}
else
{
lean_object* v_reuseFailAlloc_4097_; 
v_reuseFailAlloc_4097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4097_, 0, v_a_4091_);
v___x_4096_ = v_reuseFailAlloc_4097_;
goto v_reusejp_4095_;
}
v_reusejp_4095_:
{
return v___x_4096_;
}
}
}
}
}
else
{
return v___x_4080_;
}
}
}
}
else
{
return v___x_4071_;
}
}
else
{
lean_dec_ref(v_expr_4040_);
goto v___jp_4043_;
}
v___jp_4043_:
{
lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; lean_object* v___x_4051_; lean_object* v___x_4052_; lean_object* v___x_4053_; lean_object* v___x_4054_; lean_object* v___x_4056_; 
v___x_4044_ = l_Lean_Level_succ___override(v_v_4016_);
v___x_4045_ = lean_box(0);
v___x_4046_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4046_, 0, v___x_4044_);
lean_ctor_set(v___x_4046_, 1, v___x_4045_);
v___x_4047_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__1));
v___x_4048_ = l_Lean_Expr_const___override(v___x_4047_, v___x_4046_);
v___x_4049_ = l_Lean_Expr_app___override(v___x_4048_, v_00_u03b1_4017_);
v___x_4050_ = l_Lean_Expr_app___override(v___x_4049_, v_e_u2081_4021_);
v___x_4051_ = l_Lean_Expr_app___override(v___x_4050_, v_e_u2082_4022_);
v___x_4052_ = l_Lean_Expr_app___override(v___x_4051_, v_expr_4032_);
v___x_4053_ = l_Lean_Expr_app___override(v___x_4052_, v_proof_4034_);
v___x_4054_ = l_Lean_Expr_app___override(v___x_4053_, v_proof_4042_);
if (v_isShared_4039_ == 0)
{
lean_ctor_set(v___x_4038_, 0, v___x_4054_);
v___x_4056_ = v___x_4038_;
goto v_reusejp_4055_;
}
else
{
lean_object* v_reuseFailAlloc_4057_; 
v_reuseFailAlloc_4057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4057_, 0, v___x_4054_);
v___x_4056_ = v_reuseFailAlloc_4057_;
goto v_reusejp_4055_;
}
v_reusejp_4055_:
{
return v___x_4056_;
}
}
}
}
else
{
lean_object* v_a_4104_; lean_object* v___x_4106_; uint8_t v_isShared_4107_; uint8_t v_isSharedCheck_4111_; 
lean_dec_ref(v_proof_4034_);
lean_dec(v_val_4033_);
lean_dec_ref(v_expr_4032_);
lean_dec_ref(v_e_u2082_4022_);
lean_dec_ref(v_e_u2081_4021_);
lean_dec_ref(v___x_4019_);
lean_dec_ref(v_s_u03b1_4018_);
lean_dec_ref(v_00_u03b1_4017_);
lean_dec(v_v_4016_);
lean_dec_ref(v___x_4015_);
v_a_4104_ = lean_ctor_get(v___x_4035_, 0);
v_isSharedCheck_4111_ = !lean_is_exclusive(v___x_4035_);
if (v_isSharedCheck_4111_ == 0)
{
v___x_4106_ = v___x_4035_;
v_isShared_4107_ = v_isSharedCheck_4111_;
goto v_resetjp_4105_;
}
else
{
lean_inc(v_a_4104_);
lean_dec(v___x_4035_);
v___x_4106_ = lean_box(0);
v_isShared_4107_ = v_isSharedCheck_4111_;
goto v_resetjp_4105_;
}
v_resetjp_4105_:
{
lean_object* v___x_4109_; 
if (v_isShared_4107_ == 0)
{
v___x_4109_ = v___x_4106_;
goto v_reusejp_4108_;
}
else
{
lean_object* v_reuseFailAlloc_4110_; 
v_reuseFailAlloc_4110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4110_, 0, v_a_4104_);
v___x_4109_ = v_reuseFailAlloc_4110_;
goto v_reusejp_4108_;
}
v_reusejp_4108_:
{
return v___x_4109_;
}
}
}
}
else
{
lean_object* v_a_4112_; lean_object* v___x_4114_; uint8_t v_isShared_4115_; uint8_t v_isSharedCheck_4119_; 
lean_dec_ref(v_e_u2082_4022_);
lean_dec_ref(v_e_u2081_4021_);
lean_dec_ref(v_a_4020_);
lean_dec_ref(v___x_4019_);
lean_dec_ref(v_s_u03b1_4018_);
lean_dec_ref(v_00_u03b1_4017_);
lean_dec(v_v_4016_);
lean_dec_ref(v___x_4015_);
v_a_4112_ = lean_ctor_get(v___x_4030_, 0);
v_isSharedCheck_4119_ = !lean_is_exclusive(v___x_4030_);
if (v_isSharedCheck_4119_ == 0)
{
v___x_4114_ = v___x_4030_;
v_isShared_4115_ = v_isSharedCheck_4119_;
goto v_resetjp_4113_;
}
else
{
lean_inc(v_a_4112_);
lean_dec(v___x_4030_);
v___x_4114_ = lean_box(0);
v_isShared_4115_ = v_isSharedCheck_4119_;
goto v_resetjp_4113_;
}
v_resetjp_4113_:
{
lean_object* v___x_4117_; 
if (v_isShared_4115_ == 0)
{
v___x_4117_ = v___x_4114_;
goto v_reusejp_4116_;
}
else
{
lean_object* v_reuseFailAlloc_4118_; 
v_reuseFailAlloc_4118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4118_, 0, v_a_4112_);
v___x_4117_ = v_reuseFailAlloc_4118_;
goto v_reusejp_4116_;
}
v_reusejp_4116_:
{
return v___x_4117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___boxed(lean_object* v___x_4120_, lean_object* v_v_4121_, lean_object* v_00_u03b1_4122_, lean_object* v_s_u03b1_4123_, lean_object* v___x_4124_, lean_object* v_a_4125_, lean_object* v_e_u2081_4126_, lean_object* v_e_u2082_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_, lean_object* v___y_4132_, lean_object* v___y_4133_, lean_object* v___y_4134_){
_start:
{
lean_object* v_res_4135_; 
v_res_4135_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0(v___x_4120_, v_v_4121_, v_00_u03b1_4122_, v_s_u03b1_4123_, v___x_4124_, v_a_4125_, v_e_u2081_4126_, v_e_u2082_4127_, v___y_4128_, v___y_4129_, v___y_4130_, v___y_4131_, v___y_4132_, v___y_4133_);
lean_dec(v___y_4133_);
lean_dec_ref(v___y_4132_);
lean_dec(v___y_4131_);
lean_dec_ref(v___y_4130_);
lean_dec(v___y_4129_);
lean_dec_ref(v___y_4128_);
return v_res_4135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore(lean_object* v_v_4137_, lean_object* v_00_u03b1_4138_, lean_object* v_s_u03b1_4139_, lean_object* v_e_u2081_4140_, lean_object* v_e_u2082_4141_, lean_object* v_a_4142_, lean_object* v_a_4143_, lean_object* v_a_4144_, lean_object* v_a_4145_, lean_object* v_a_4146_, lean_object* v_a_4147_){
_start:
{
lean_object* v___x_4149_; 
lean_inc_ref(v_s_u03b1_4139_);
lean_inc_ref(v_00_u03b1_4138_);
lean_inc(v_v_4137_);
v___x_4149_ = lp_mathlib_Mathlib_Tactic_Ring_Common_mkCache(v_v_4137_, v_00_u03b1_4138_, v_s_u03b1_4139_, v_a_4144_, v_a_4145_, v_a_4146_, v_a_4147_);
if (lean_obj_tag(v___x_4149_) == 0)
{
lean_object* v_a_4150_; lean_object* v_options_4151_; lean_object* v___x_4152_; lean_object* v___x_4153_; lean_object* v___x_4154_; lean_object* v___f_4155_; lean_object* v___x_4156_; lean_object* v___x_4157_; 
v_a_4150_ = lean_ctor_get(v___x_4149_, 0);
lean_inc_n(v_a_4150_, 2);
lean_dec_ref_known(v___x_4149_, 1);
v_options_4151_ = lean_ctor_get(v_a_4146_, 2);
v___x_4152_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___closed__0));
v___x_4153_ = lp_mathlib_Mathlib_Tactic_Ring_rc_u2115;
lean_inc_ref(v_s_u03b1_4139_);
lean_inc_ref(v_00_u03b1_4138_);
lean_inc(v_v_4137_);
v___x_4154_ = lp_mathlib_Mathlib_Tactic_Ring_ringCompute(v_v_4137_, v_00_u03b1_4138_, v_s_u03b1_4139_, v_a_4150_);
v___f_4155_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___boxed), 15, 8);
lean_closure_set(v___f_4155_, 0, v___x_4153_);
lean_closure_set(v___f_4155_, 1, v_v_4137_);
lean_closure_set(v___f_4155_, 2, v_00_u03b1_4138_);
lean_closure_set(v___f_4155_, 3, v_s_u03b1_4139_);
lean_closure_set(v___f_4155_, 4, v___x_4154_);
lean_closure_set(v___f_4155_, 5, v_a_4150_);
lean_closure_set(v___f_4155_, 6, v_e_u2081_4140_);
lean_closure_set(v___f_4155_, 7, v_e_u2082_4141_);
v___x_4156_ = lean_box(0);
v___x_4157_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore_spec__0___redArg(v___x_4152_, v_options_4151_, v___f_4155_, v___x_4156_, v_a_4142_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, v_a_4147_);
return v___x_4157_;
}
else
{
lean_object* v_a_4158_; lean_object* v___x_4160_; uint8_t v_isShared_4161_; uint8_t v_isSharedCheck_4165_; 
lean_dec_ref(v_e_u2082_4141_);
lean_dec_ref(v_e_u2081_4140_);
lean_dec_ref(v_s_u03b1_4139_);
lean_dec_ref(v_00_u03b1_4138_);
lean_dec(v_v_4137_);
v_a_4158_ = lean_ctor_get(v___x_4149_, 0);
v_isSharedCheck_4165_ = !lean_is_exclusive(v___x_4149_);
if (v_isSharedCheck_4165_ == 0)
{
v___x_4160_ = v___x_4149_;
v_isShared_4161_ = v_isSharedCheck_4165_;
goto v_resetjp_4159_;
}
else
{
lean_inc(v_a_4158_);
lean_dec(v___x_4149_);
v___x_4160_ = lean_box(0);
v_isShared_4161_ = v_isSharedCheck_4165_;
goto v_resetjp_4159_;
}
v_resetjp_4159_:
{
lean_object* v___x_4163_; 
if (v_isShared_4161_ == 0)
{
v___x_4163_ = v___x_4160_;
goto v_reusejp_4162_;
}
else
{
lean_object* v_reuseFailAlloc_4164_; 
v_reuseFailAlloc_4164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4164_, 0, v_a_4158_);
v___x_4163_ = v_reuseFailAlloc_4164_;
goto v_reusejp_4162_;
}
v_reusejp_4162_:
{
return v___x_4163_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___boxed(lean_object* v_v_4166_, lean_object* v_00_u03b1_4167_, lean_object* v_s_u03b1_4168_, lean_object* v_e_u2081_4169_, lean_object* v_e_u2082_4170_, lean_object* v_a_4171_, lean_object* v_a_4172_, lean_object* v_a_4173_, lean_object* v_a_4174_, lean_object* v_a_4175_, lean_object* v_a_4176_, lean_object* v_a_4177_){
_start:
{
lean_object* v_res_4178_; 
v_res_4178_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore(v_v_4166_, v_00_u03b1_4167_, v_s_u03b1_4168_, v_e_u2081_4169_, v_e_u2082_4170_, v_a_4171_, v_a_4172_, v_a_4173_, v_a_4174_, v_a_4175_, v_a_4176_);
lean_dec(v_a_4176_);
lean_dec_ref(v_a_4175_);
lean_dec(v_a_4174_);
lean_dec_ref(v_a_4173_);
lean_dec(v_a_4172_);
lean_dec_ref(v_a_4171_);
return v_res_4178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg(lean_object* v_e_4179_, lean_object* v___y_4180_){
_start:
{
uint8_t v___x_4182_; 
v___x_4182_ = l_Lean_Expr_hasMVar(v_e_4179_);
if (v___x_4182_ == 0)
{
lean_object* v___x_4183_; 
v___x_4183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4183_, 0, v_e_4179_);
return v___x_4183_;
}
else
{
lean_object* v___x_4184_; lean_object* v_mctx_4185_; lean_object* v___x_4186_; lean_object* v_fst_4187_; lean_object* v_snd_4188_; lean_object* v___x_4189_; lean_object* v_cache_4190_; lean_object* v_zetaDeltaFVarIds_4191_; lean_object* v_postponed_4192_; lean_object* v_diag_4193_; lean_object* v___x_4195_; uint8_t v_isShared_4196_; uint8_t v_isSharedCheck_4202_; 
v___x_4184_ = lean_st_ref_get(v___y_4180_);
v_mctx_4185_ = lean_ctor_get(v___x_4184_, 0);
lean_inc_ref(v_mctx_4185_);
lean_dec(v___x_4184_);
v___x_4186_ = l_Lean_instantiateMVarsCore(v_mctx_4185_, v_e_4179_);
v_fst_4187_ = lean_ctor_get(v___x_4186_, 0);
lean_inc(v_fst_4187_);
v_snd_4188_ = lean_ctor_get(v___x_4186_, 1);
lean_inc(v_snd_4188_);
lean_dec_ref(v___x_4186_);
v___x_4189_ = lean_st_ref_take(v___y_4180_);
v_cache_4190_ = lean_ctor_get(v___x_4189_, 1);
v_zetaDeltaFVarIds_4191_ = lean_ctor_get(v___x_4189_, 2);
v_postponed_4192_ = lean_ctor_get(v___x_4189_, 3);
v_diag_4193_ = lean_ctor_get(v___x_4189_, 4);
v_isSharedCheck_4202_ = !lean_is_exclusive(v___x_4189_);
if (v_isSharedCheck_4202_ == 0)
{
lean_object* v_unused_4203_; 
v_unused_4203_ = lean_ctor_get(v___x_4189_, 0);
lean_dec(v_unused_4203_);
v___x_4195_ = v___x_4189_;
v_isShared_4196_ = v_isSharedCheck_4202_;
goto v_resetjp_4194_;
}
else
{
lean_inc(v_diag_4193_);
lean_inc(v_postponed_4192_);
lean_inc(v_zetaDeltaFVarIds_4191_);
lean_inc(v_cache_4190_);
lean_dec(v___x_4189_);
v___x_4195_ = lean_box(0);
v_isShared_4196_ = v_isSharedCheck_4202_;
goto v_resetjp_4194_;
}
v_resetjp_4194_:
{
lean_object* v___x_4198_; 
if (v_isShared_4196_ == 0)
{
lean_ctor_set(v___x_4195_, 0, v_snd_4188_);
v___x_4198_ = v___x_4195_;
goto v_reusejp_4197_;
}
else
{
lean_object* v_reuseFailAlloc_4201_; 
v_reuseFailAlloc_4201_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4201_, 0, v_snd_4188_);
lean_ctor_set(v_reuseFailAlloc_4201_, 1, v_cache_4190_);
lean_ctor_set(v_reuseFailAlloc_4201_, 2, v_zetaDeltaFVarIds_4191_);
lean_ctor_set(v_reuseFailAlloc_4201_, 3, v_postponed_4192_);
lean_ctor_set(v_reuseFailAlloc_4201_, 4, v_diag_4193_);
v___x_4198_ = v_reuseFailAlloc_4201_;
goto v_reusejp_4197_;
}
v_reusejp_4197_:
{
lean_object* v___x_4199_; lean_object* v___x_4200_; 
v___x_4199_ = lean_st_ref_set(v___y_4180_, v___x_4198_);
v___x_4200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4200_, 0, v_fst_4187_);
return v___x_4200_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg___boxed(lean_object* v_e_4204_, lean_object* v___y_4205_, lean_object* v___y_4206_){
_start:
{
lean_object* v_res_4207_; 
v_res_4207_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg(v_e_4204_, v___y_4205_);
lean_dec(v___y_4205_);
return v_res_4207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0(lean_object* v_e_4208_, lean_object* v___y_4209_, lean_object* v___y_4210_, lean_object* v___y_4211_, lean_object* v___y_4212_, lean_object* v___y_4213_, lean_object* v___y_4214_){
_start:
{
lean_object* v___x_4216_; 
v___x_4216_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg(v_e_4208_, v___y_4212_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___boxed(lean_object* v_e_4217_, lean_object* v___y_4218_, lean_object* v___y_4219_, lean_object* v___y_4220_, lean_object* v___y_4221_, lean_object* v___y_4222_, lean_object* v___y_4223_, lean_object* v___y_4224_){
_start:
{
lean_object* v_res_4225_; 
v_res_4225_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0(v_e_4217_, v___y_4218_, v___y_4219_, v___y_4220_, v___y_4221_, v___y_4222_, v___y_4223_);
lean_dec(v___y_4223_);
lean_dec_ref(v___y_4222_);
lean_dec(v___y_4221_);
lean_dec_ref(v___y_4220_);
lean_dec(v___y_4219_);
lean_dec_ref(v___y_4218_);
return v_res_4225_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0(void){
_start:
{
lean_object* v___x_4226_; 
v___x_4226_ = l_instMonadEIO(lean_box(0));
return v___x_4226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2(lean_object* v_msg_4231_, lean_object* v___y_4232_, lean_object* v___y_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_){
_start:
{
lean_object* v___x_4239_; lean_object* v___x_4240_; lean_object* v_toApplicative_4241_; lean_object* v___x_4243_; uint8_t v_isShared_4244_; uint8_t v_isSharedCheck_4304_; 
v___x_4239_ = lean_obj_once(&lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0, &lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__0);
v___x_4240_ = l_StateRefT_x27_instMonad___redArg(v___x_4239_);
v_toApplicative_4241_ = lean_ctor_get(v___x_4240_, 0);
v_isSharedCheck_4304_ = !lean_is_exclusive(v___x_4240_);
if (v_isSharedCheck_4304_ == 0)
{
lean_object* v_unused_4305_; 
v_unused_4305_ = lean_ctor_get(v___x_4240_, 1);
lean_dec(v_unused_4305_);
v___x_4243_ = v___x_4240_;
v_isShared_4244_ = v_isSharedCheck_4304_;
goto v_resetjp_4242_;
}
else
{
lean_inc(v_toApplicative_4241_);
lean_dec(v___x_4240_);
v___x_4243_ = lean_box(0);
v_isShared_4244_ = v_isSharedCheck_4304_;
goto v_resetjp_4242_;
}
v_resetjp_4242_:
{
lean_object* v_toFunctor_4245_; lean_object* v_toSeq_4246_; lean_object* v_toSeqLeft_4247_; lean_object* v_toSeqRight_4248_; lean_object* v___x_4250_; uint8_t v_isShared_4251_; uint8_t v_isSharedCheck_4302_; 
v_toFunctor_4245_ = lean_ctor_get(v_toApplicative_4241_, 0);
v_toSeq_4246_ = lean_ctor_get(v_toApplicative_4241_, 2);
v_toSeqLeft_4247_ = lean_ctor_get(v_toApplicative_4241_, 3);
v_toSeqRight_4248_ = lean_ctor_get(v_toApplicative_4241_, 4);
v_isSharedCheck_4302_ = !lean_is_exclusive(v_toApplicative_4241_);
if (v_isSharedCheck_4302_ == 0)
{
lean_object* v_unused_4303_; 
v_unused_4303_ = lean_ctor_get(v_toApplicative_4241_, 1);
lean_dec(v_unused_4303_);
v___x_4250_ = v_toApplicative_4241_;
v_isShared_4251_ = v_isSharedCheck_4302_;
goto v_resetjp_4249_;
}
else
{
lean_inc(v_toSeqRight_4248_);
lean_inc(v_toSeqLeft_4247_);
lean_inc(v_toSeq_4246_);
lean_inc(v_toFunctor_4245_);
lean_dec(v_toApplicative_4241_);
v___x_4250_ = lean_box(0);
v_isShared_4251_ = v_isSharedCheck_4302_;
goto v_resetjp_4249_;
}
v_resetjp_4249_:
{
lean_object* v___f_4252_; lean_object* v___f_4253_; lean_object* v___f_4254_; lean_object* v___f_4255_; lean_object* v___x_4256_; lean_object* v___f_4257_; lean_object* v___f_4258_; lean_object* v___f_4259_; lean_object* v___x_4261_; 
v___f_4252_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__1));
v___f_4253_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__2));
lean_inc_ref(v_toFunctor_4245_);
v___f_4254_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4254_, 0, v_toFunctor_4245_);
v___f_4255_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4255_, 0, v_toFunctor_4245_);
v___x_4256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4256_, 0, v___f_4254_);
lean_ctor_set(v___x_4256_, 1, v___f_4255_);
v___f_4257_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4257_, 0, v_toSeqRight_4248_);
v___f_4258_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4258_, 0, v_toSeqLeft_4247_);
v___f_4259_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4259_, 0, v_toSeq_4246_);
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 4, v___f_4257_);
lean_ctor_set(v___x_4250_, 3, v___f_4258_);
lean_ctor_set(v___x_4250_, 2, v___f_4259_);
lean_ctor_set(v___x_4250_, 1, v___f_4252_);
lean_ctor_set(v___x_4250_, 0, v___x_4256_);
v___x_4261_ = v___x_4250_;
goto v_reusejp_4260_;
}
else
{
lean_object* v_reuseFailAlloc_4301_; 
v_reuseFailAlloc_4301_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4301_, 0, v___x_4256_);
lean_ctor_set(v_reuseFailAlloc_4301_, 1, v___f_4252_);
lean_ctor_set(v_reuseFailAlloc_4301_, 2, v___f_4259_);
lean_ctor_set(v_reuseFailAlloc_4301_, 3, v___f_4258_);
lean_ctor_set(v_reuseFailAlloc_4301_, 4, v___f_4257_);
v___x_4261_ = v_reuseFailAlloc_4301_;
goto v_reusejp_4260_;
}
v_reusejp_4260_:
{
lean_object* v___x_4263_; 
if (v_isShared_4244_ == 0)
{
lean_ctor_set(v___x_4243_, 1, v___f_4253_);
lean_ctor_set(v___x_4243_, 0, v___x_4261_);
v___x_4263_ = v___x_4243_;
goto v_reusejp_4262_;
}
else
{
lean_object* v_reuseFailAlloc_4300_; 
v_reuseFailAlloc_4300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4300_, 0, v___x_4261_);
lean_ctor_set(v_reuseFailAlloc_4300_, 1, v___f_4253_);
v___x_4263_ = v_reuseFailAlloc_4300_;
goto v_reusejp_4262_;
}
v_reusejp_4262_:
{
lean_object* v___x_4264_; lean_object* v_toApplicative_4265_; lean_object* v___x_4267_; uint8_t v_isShared_4268_; uint8_t v_isSharedCheck_4298_; 
v___x_4264_ = l_StateRefT_x27_instMonad___redArg(v___x_4263_);
v_toApplicative_4265_ = lean_ctor_get(v___x_4264_, 0);
v_isSharedCheck_4298_ = !lean_is_exclusive(v___x_4264_);
if (v_isSharedCheck_4298_ == 0)
{
lean_object* v_unused_4299_; 
v_unused_4299_ = lean_ctor_get(v___x_4264_, 1);
lean_dec(v_unused_4299_);
v___x_4267_ = v___x_4264_;
v_isShared_4268_ = v_isSharedCheck_4298_;
goto v_resetjp_4266_;
}
else
{
lean_inc(v_toApplicative_4265_);
lean_dec(v___x_4264_);
v___x_4267_ = lean_box(0);
v_isShared_4268_ = v_isSharedCheck_4298_;
goto v_resetjp_4266_;
}
v_resetjp_4266_:
{
lean_object* v_toFunctor_4269_; lean_object* v_toSeq_4270_; lean_object* v_toSeqLeft_4271_; lean_object* v_toSeqRight_4272_; lean_object* v___x_4274_; uint8_t v_isShared_4275_; uint8_t v_isSharedCheck_4296_; 
v_toFunctor_4269_ = lean_ctor_get(v_toApplicative_4265_, 0);
v_toSeq_4270_ = lean_ctor_get(v_toApplicative_4265_, 2);
v_toSeqLeft_4271_ = lean_ctor_get(v_toApplicative_4265_, 3);
v_toSeqRight_4272_ = lean_ctor_get(v_toApplicative_4265_, 4);
v_isSharedCheck_4296_ = !lean_is_exclusive(v_toApplicative_4265_);
if (v_isSharedCheck_4296_ == 0)
{
lean_object* v_unused_4297_; 
v_unused_4297_ = lean_ctor_get(v_toApplicative_4265_, 1);
lean_dec(v_unused_4297_);
v___x_4274_ = v_toApplicative_4265_;
v_isShared_4275_ = v_isSharedCheck_4296_;
goto v_resetjp_4273_;
}
else
{
lean_inc(v_toSeqRight_4272_);
lean_inc(v_toSeqLeft_4271_);
lean_inc(v_toSeq_4270_);
lean_inc(v_toFunctor_4269_);
lean_dec(v_toApplicative_4265_);
v___x_4274_ = lean_box(0);
v_isShared_4275_ = v_isSharedCheck_4296_;
goto v_resetjp_4273_;
}
v_resetjp_4273_:
{
lean_object* v___f_4276_; lean_object* v___f_4277_; lean_object* v___f_4278_; lean_object* v___f_4279_; lean_object* v___x_4280_; lean_object* v___f_4281_; lean_object* v___f_4282_; lean_object* v___f_4283_; lean_object* v___x_4285_; 
v___f_4276_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__3));
v___f_4277_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___closed__4));
lean_inc_ref(v_toFunctor_4269_);
v___f_4278_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4278_, 0, v_toFunctor_4269_);
v___f_4279_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4279_, 0, v_toFunctor_4269_);
v___x_4280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4280_, 0, v___f_4278_);
lean_ctor_set(v___x_4280_, 1, v___f_4279_);
v___f_4281_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4281_, 0, v_toSeqRight_4272_);
v___f_4282_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4282_, 0, v_toSeqLeft_4271_);
v___f_4283_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4283_, 0, v_toSeq_4270_);
if (v_isShared_4275_ == 0)
{
lean_ctor_set(v___x_4274_, 4, v___f_4281_);
lean_ctor_set(v___x_4274_, 3, v___f_4282_);
lean_ctor_set(v___x_4274_, 2, v___f_4283_);
lean_ctor_set(v___x_4274_, 1, v___f_4276_);
lean_ctor_set(v___x_4274_, 0, v___x_4280_);
v___x_4285_ = v___x_4274_;
goto v_reusejp_4284_;
}
else
{
lean_object* v_reuseFailAlloc_4295_; 
v_reuseFailAlloc_4295_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4295_, 0, v___x_4280_);
lean_ctor_set(v_reuseFailAlloc_4295_, 1, v___f_4276_);
lean_ctor_set(v_reuseFailAlloc_4295_, 2, v___f_4283_);
lean_ctor_set(v_reuseFailAlloc_4295_, 3, v___f_4282_);
lean_ctor_set(v_reuseFailAlloc_4295_, 4, v___f_4281_);
v___x_4285_ = v_reuseFailAlloc_4295_;
goto v_reusejp_4284_;
}
v_reusejp_4284_:
{
lean_object* v___x_4287_; 
if (v_isShared_4268_ == 0)
{
lean_ctor_set(v___x_4267_, 1, v___f_4277_);
lean_ctor_set(v___x_4267_, 0, v___x_4285_);
v___x_4287_ = v___x_4267_;
goto v_reusejp_4286_;
}
else
{
lean_object* v_reuseFailAlloc_4294_; 
v_reuseFailAlloc_4294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4294_, 0, v___x_4285_);
lean_ctor_set(v_reuseFailAlloc_4294_, 1, v___f_4277_);
v___x_4287_ = v_reuseFailAlloc_4294_;
goto v_reusejp_4286_;
}
v_reusejp_4286_:
{
lean_object* v___x_4288_; lean_object* v___x_4289_; lean_object* v___x_4290_; lean_object* v___f_4291_; lean_object* v___x_27702__overap_4292_; lean_object* v___x_4293_; 
v___x_4288_ = l_StateRefT_x27_instMonad___redArg(v___x_4287_);
v___x_4289_ = lean_box(0);
v___x_4290_ = l_instInhabitedOfMonad___redArg(v___x_4288_, v___x_4289_);
v___f_4291_ = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_4291_, 0, v___x_4290_);
v___x_27702__overap_4292_ = lean_panic_fn_borrowed(v___f_4291_, v_msg_4231_);
lean_dec_ref(v___f_4291_);
lean_inc(v___y_4237_);
lean_inc_ref(v___y_4236_);
lean_inc(v___y_4235_);
lean_inc_ref(v___y_4234_);
lean_inc(v___y_4233_);
lean_inc_ref(v___y_4232_);
v___x_4293_ = lean_apply_7(v___x_27702__overap_4292_, v___y_4232_, v___y_4233_, v___y_4234_, v___y_4235_, v___y_4236_, v___y_4237_, lean_box(0));
return v___x_4293_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2___boxed(lean_object* v_msg_4306_, lean_object* v___y_4307_, lean_object* v___y_4308_, lean_object* v___y_4309_, lean_object* v___y_4310_, lean_object* v___y_4311_, lean_object* v___y_4312_, lean_object* v___y_4313_){
_start:
{
lean_object* v_res_4314_; 
v_res_4314_ = lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2(v_msg_4306_, v___y_4307_, v___y_4308_, v___y_4309_, v___y_4310_, v___y_4311_, v___y_4312_);
lean_dec(v___y_4312_);
lean_dec_ref(v___y_4311_);
lean_dec(v___y_4310_);
lean_dec_ref(v___y_4309_);
lean_dec(v___y_4308_);
lean_dec_ref(v___y_4307_);
return v_res_4314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_x_4315_, lean_object* v_x_4316_, lean_object* v_x_4317_, lean_object* v_x_4318_){
_start:
{
lean_object* v_ks_4319_; lean_object* v_vs_4320_; lean_object* v___x_4322_; uint8_t v_isShared_4323_; uint8_t v_isSharedCheck_4344_; 
v_ks_4319_ = lean_ctor_get(v_x_4315_, 0);
v_vs_4320_ = lean_ctor_get(v_x_4315_, 1);
v_isSharedCheck_4344_ = !lean_is_exclusive(v_x_4315_);
if (v_isSharedCheck_4344_ == 0)
{
v___x_4322_ = v_x_4315_;
v_isShared_4323_ = v_isSharedCheck_4344_;
goto v_resetjp_4321_;
}
else
{
lean_inc(v_vs_4320_);
lean_inc(v_ks_4319_);
lean_dec(v_x_4315_);
v___x_4322_ = lean_box(0);
v_isShared_4323_ = v_isSharedCheck_4344_;
goto v_resetjp_4321_;
}
v_resetjp_4321_:
{
lean_object* v___x_4324_; uint8_t v___x_4325_; 
v___x_4324_ = lean_array_get_size(v_ks_4319_);
v___x_4325_ = lean_nat_dec_lt(v_x_4316_, v___x_4324_);
if (v___x_4325_ == 0)
{
lean_object* v___x_4326_; lean_object* v___x_4327_; lean_object* v___x_4329_; 
lean_dec(v_x_4316_);
v___x_4326_ = lean_array_push(v_ks_4319_, v_x_4317_);
v___x_4327_ = lean_array_push(v_vs_4320_, v_x_4318_);
if (v_isShared_4323_ == 0)
{
lean_ctor_set(v___x_4322_, 1, v___x_4327_);
lean_ctor_set(v___x_4322_, 0, v___x_4326_);
v___x_4329_ = v___x_4322_;
goto v_reusejp_4328_;
}
else
{
lean_object* v_reuseFailAlloc_4330_; 
v_reuseFailAlloc_4330_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4330_, 0, v___x_4326_);
lean_ctor_set(v_reuseFailAlloc_4330_, 1, v___x_4327_);
v___x_4329_ = v_reuseFailAlloc_4330_;
goto v_reusejp_4328_;
}
v_reusejp_4328_:
{
return v___x_4329_;
}
}
else
{
lean_object* v_k_x27_4331_; uint8_t v___x_4332_; 
v_k_x27_4331_ = lean_array_fget_borrowed(v_ks_4319_, v_x_4316_);
v___x_4332_ = l_Lean_instBEqMVarId_beq(v_x_4317_, v_k_x27_4331_);
if (v___x_4332_ == 0)
{
lean_object* v___x_4334_; 
if (v_isShared_4323_ == 0)
{
v___x_4334_ = v___x_4322_;
goto v_reusejp_4333_;
}
else
{
lean_object* v_reuseFailAlloc_4338_; 
v_reuseFailAlloc_4338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4338_, 0, v_ks_4319_);
lean_ctor_set(v_reuseFailAlloc_4338_, 1, v_vs_4320_);
v___x_4334_ = v_reuseFailAlloc_4338_;
goto v_reusejp_4333_;
}
v_reusejp_4333_:
{
lean_object* v___x_4335_; lean_object* v___x_4336_; 
v___x_4335_ = lean_unsigned_to_nat(1u);
v___x_4336_ = lean_nat_add(v_x_4316_, v___x_4335_);
lean_dec(v_x_4316_);
v_x_4315_ = v___x_4334_;
v_x_4316_ = v___x_4336_;
goto _start;
}
}
else
{
lean_object* v___x_4339_; lean_object* v___x_4340_; lean_object* v___x_4342_; 
v___x_4339_ = lean_array_fset(v_ks_4319_, v_x_4316_, v_x_4317_);
v___x_4340_ = lean_array_fset(v_vs_4320_, v_x_4316_, v_x_4318_);
lean_dec(v_x_4316_);
if (v_isShared_4323_ == 0)
{
lean_ctor_set(v___x_4322_, 1, v___x_4340_);
lean_ctor_set(v___x_4322_, 0, v___x_4339_);
v___x_4342_ = v___x_4322_;
goto v_reusejp_4341_;
}
else
{
lean_object* v_reuseFailAlloc_4343_; 
v_reuseFailAlloc_4343_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4343_, 0, v___x_4339_);
lean_ctor_set(v_reuseFailAlloc_4343_, 1, v___x_4340_);
v___x_4342_ = v_reuseFailAlloc_4343_;
goto v_reusejp_4341_;
}
v_reusejp_4341_:
{
return v___x_4342_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4___redArg(lean_object* v_n_4345_, lean_object* v_k_4346_, lean_object* v_v_4347_){
_start:
{
lean_object* v___x_4348_; lean_object* v___x_4349_; 
v___x_4348_ = lean_unsigned_to_nat(0u);
v___x_4349_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_n_4345_, v___x_4348_, v_k_4346_, v_v_4347_);
return v___x_4349_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_4350_; 
v___x_4350_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_4350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(lean_object* v_x_4351_, size_t v_x_4352_, size_t v_x_4353_, lean_object* v_x_4354_, lean_object* v_x_4355_){
_start:
{
if (lean_obj_tag(v_x_4351_) == 0)
{
lean_object* v_es_4356_; size_t v___x_4357_; size_t v___x_4358_; lean_object* v_j_4359_; lean_object* v___x_4360_; uint8_t v___x_4361_; 
v_es_4356_ = lean_ctor_get(v_x_4351_, 0);
v___x_4357_ = ((size_t)31ULL);
v___x_4358_ = lean_usize_land(v_x_4352_, v___x_4357_);
v_j_4359_ = lean_usize_to_nat(v___x_4358_);
v___x_4360_ = lean_array_get_size(v_es_4356_);
v___x_4361_ = lean_nat_dec_lt(v_j_4359_, v___x_4360_);
if (v___x_4361_ == 0)
{
lean_dec(v_j_4359_);
lean_dec(v_x_4355_);
lean_dec(v_x_4354_);
return v_x_4351_;
}
else
{
lean_object* v___x_4363_; uint8_t v_isShared_4364_; uint8_t v_isSharedCheck_4400_; 
lean_inc_ref(v_es_4356_);
v_isSharedCheck_4400_ = !lean_is_exclusive(v_x_4351_);
if (v_isSharedCheck_4400_ == 0)
{
lean_object* v_unused_4401_; 
v_unused_4401_ = lean_ctor_get(v_x_4351_, 0);
lean_dec(v_unused_4401_);
v___x_4363_ = v_x_4351_;
v_isShared_4364_ = v_isSharedCheck_4400_;
goto v_resetjp_4362_;
}
else
{
lean_dec(v_x_4351_);
v___x_4363_ = lean_box(0);
v_isShared_4364_ = v_isSharedCheck_4400_;
goto v_resetjp_4362_;
}
v_resetjp_4362_:
{
lean_object* v_v_4365_; lean_object* v___x_4366_; lean_object* v_xs_x27_4367_; lean_object* v___y_4369_; 
v_v_4365_ = lean_array_fget(v_es_4356_, v_j_4359_);
v___x_4366_ = lean_box(0);
v_xs_x27_4367_ = lean_array_fset(v_es_4356_, v_j_4359_, v___x_4366_);
switch(lean_obj_tag(v_v_4365_))
{
case 0:
{
lean_object* v_key_4374_; lean_object* v_val_4375_; lean_object* v___x_4377_; uint8_t v_isShared_4378_; uint8_t v_isSharedCheck_4385_; 
v_key_4374_ = lean_ctor_get(v_v_4365_, 0);
v_val_4375_ = lean_ctor_get(v_v_4365_, 1);
v_isSharedCheck_4385_ = !lean_is_exclusive(v_v_4365_);
if (v_isSharedCheck_4385_ == 0)
{
v___x_4377_ = v_v_4365_;
v_isShared_4378_ = v_isSharedCheck_4385_;
goto v_resetjp_4376_;
}
else
{
lean_inc(v_val_4375_);
lean_inc(v_key_4374_);
lean_dec(v_v_4365_);
v___x_4377_ = lean_box(0);
v_isShared_4378_ = v_isSharedCheck_4385_;
goto v_resetjp_4376_;
}
v_resetjp_4376_:
{
uint8_t v___x_4379_; 
v___x_4379_ = l_Lean_instBEqMVarId_beq(v_x_4354_, v_key_4374_);
if (v___x_4379_ == 0)
{
lean_object* v___x_4380_; lean_object* v___x_4381_; 
lean_del_object(v___x_4377_);
v___x_4380_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_4374_, v_val_4375_, v_x_4354_, v_x_4355_);
v___x_4381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4381_, 0, v___x_4380_);
v___y_4369_ = v___x_4381_;
goto v___jp_4368_;
}
else
{
lean_object* v___x_4383_; 
lean_dec(v_val_4375_);
lean_dec(v_key_4374_);
if (v_isShared_4378_ == 0)
{
lean_ctor_set(v___x_4377_, 1, v_x_4355_);
lean_ctor_set(v___x_4377_, 0, v_x_4354_);
v___x_4383_ = v___x_4377_;
goto v_reusejp_4382_;
}
else
{
lean_object* v_reuseFailAlloc_4384_; 
v_reuseFailAlloc_4384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4384_, 0, v_x_4354_);
lean_ctor_set(v_reuseFailAlloc_4384_, 1, v_x_4355_);
v___x_4383_ = v_reuseFailAlloc_4384_;
goto v_reusejp_4382_;
}
v_reusejp_4382_:
{
v___y_4369_ = v___x_4383_;
goto v___jp_4368_;
}
}
}
}
case 1:
{
lean_object* v_node_4386_; lean_object* v___x_4388_; uint8_t v_isShared_4389_; uint8_t v_isSharedCheck_4398_; 
v_node_4386_ = lean_ctor_get(v_v_4365_, 0);
v_isSharedCheck_4398_ = !lean_is_exclusive(v_v_4365_);
if (v_isSharedCheck_4398_ == 0)
{
v___x_4388_ = v_v_4365_;
v_isShared_4389_ = v_isSharedCheck_4398_;
goto v_resetjp_4387_;
}
else
{
lean_inc(v_node_4386_);
lean_dec(v_v_4365_);
v___x_4388_ = lean_box(0);
v_isShared_4389_ = v_isSharedCheck_4398_;
goto v_resetjp_4387_;
}
v_resetjp_4387_:
{
size_t v___x_4390_; size_t v___x_4391_; size_t v___x_4392_; size_t v___x_4393_; lean_object* v___x_4394_; lean_object* v___x_4396_; 
v___x_4390_ = ((size_t)5ULL);
v___x_4391_ = lean_usize_shift_right(v_x_4352_, v___x_4390_);
v___x_4392_ = ((size_t)1ULL);
v___x_4393_ = lean_usize_add(v_x_4353_, v___x_4392_);
v___x_4394_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(v_node_4386_, v___x_4391_, v___x_4393_, v_x_4354_, v_x_4355_);
if (v_isShared_4389_ == 0)
{
lean_ctor_set(v___x_4388_, 0, v___x_4394_);
v___x_4396_ = v___x_4388_;
goto v_reusejp_4395_;
}
else
{
lean_object* v_reuseFailAlloc_4397_; 
v_reuseFailAlloc_4397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4397_, 0, v___x_4394_);
v___x_4396_ = v_reuseFailAlloc_4397_;
goto v_reusejp_4395_;
}
v_reusejp_4395_:
{
v___y_4369_ = v___x_4396_;
goto v___jp_4368_;
}
}
}
default: 
{
lean_object* v___x_4399_; 
v___x_4399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4399_, 0, v_x_4354_);
lean_ctor_set(v___x_4399_, 1, v_x_4355_);
v___y_4369_ = v___x_4399_;
goto v___jp_4368_;
}
}
v___jp_4368_:
{
lean_object* v___x_4370_; lean_object* v___x_4372_; 
v___x_4370_ = lean_array_fset(v_xs_x27_4367_, v_j_4359_, v___y_4369_);
lean_dec(v_j_4359_);
if (v_isShared_4364_ == 0)
{
lean_ctor_set(v___x_4363_, 0, v___x_4370_);
v___x_4372_ = v___x_4363_;
goto v_reusejp_4371_;
}
else
{
lean_object* v_reuseFailAlloc_4373_; 
v_reuseFailAlloc_4373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4373_, 0, v___x_4370_);
v___x_4372_ = v_reuseFailAlloc_4373_;
goto v_reusejp_4371_;
}
v_reusejp_4371_:
{
return v___x_4372_;
}
}
}
}
}
else
{
lean_object* v_ks_4402_; lean_object* v_vs_4403_; lean_object* v___x_4405_; uint8_t v_isShared_4406_; uint8_t v_isSharedCheck_4423_; 
v_ks_4402_ = lean_ctor_get(v_x_4351_, 0);
v_vs_4403_ = lean_ctor_get(v_x_4351_, 1);
v_isSharedCheck_4423_ = !lean_is_exclusive(v_x_4351_);
if (v_isSharedCheck_4423_ == 0)
{
v___x_4405_ = v_x_4351_;
v_isShared_4406_ = v_isSharedCheck_4423_;
goto v_resetjp_4404_;
}
else
{
lean_inc(v_vs_4403_);
lean_inc(v_ks_4402_);
lean_dec(v_x_4351_);
v___x_4405_ = lean_box(0);
v_isShared_4406_ = v_isSharedCheck_4423_;
goto v_resetjp_4404_;
}
v_resetjp_4404_:
{
lean_object* v___x_4408_; 
if (v_isShared_4406_ == 0)
{
v___x_4408_ = v___x_4405_;
goto v_reusejp_4407_;
}
else
{
lean_object* v_reuseFailAlloc_4422_; 
v_reuseFailAlloc_4422_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4422_, 0, v_ks_4402_);
lean_ctor_set(v_reuseFailAlloc_4422_, 1, v_vs_4403_);
v___x_4408_ = v_reuseFailAlloc_4422_;
goto v_reusejp_4407_;
}
v_reusejp_4407_:
{
lean_object* v_newNode_4409_; uint8_t v___y_4411_; size_t v___x_4417_; uint8_t v___x_4418_; 
v_newNode_4409_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4___redArg(v___x_4408_, v_x_4354_, v_x_4355_);
v___x_4417_ = ((size_t)7ULL);
v___x_4418_ = lean_usize_dec_le(v___x_4417_, v_x_4353_);
if (v___x_4418_ == 0)
{
lean_object* v___x_4419_; lean_object* v___x_4420_; uint8_t v___x_4421_; 
v___x_4419_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_4409_);
v___x_4420_ = lean_unsigned_to_nat(4u);
v___x_4421_ = lean_nat_dec_lt(v___x_4419_, v___x_4420_);
lean_dec(v___x_4419_);
v___y_4411_ = v___x_4421_;
goto v___jp_4410_;
}
else
{
v___y_4411_ = v___x_4418_;
goto v___jp_4410_;
}
v___jp_4410_:
{
if (v___y_4411_ == 0)
{
lean_object* v_ks_4412_; lean_object* v_vs_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4416_; 
v_ks_4412_ = lean_ctor_get(v_newNode_4409_, 0);
lean_inc_ref(v_ks_4412_);
v_vs_4413_ = lean_ctor_get(v_newNode_4409_, 1);
lean_inc_ref(v_vs_4413_);
lean_dec_ref(v_newNode_4409_);
v___x_4414_ = lean_unsigned_to_nat(0u);
v___x_4415_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_4416_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg(v_x_4353_, v_ks_4412_, v_vs_4413_, v___x_4414_, v___x_4415_);
lean_dec_ref(v_vs_4413_);
lean_dec_ref(v_ks_4412_);
return v___x_4416_;
}
else
{
return v_newNode_4409_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg(size_t v_depth_4424_, lean_object* v_keys_4425_, lean_object* v_vals_4426_, lean_object* v_i_4427_, lean_object* v_entries_4428_){
_start:
{
lean_object* v___x_4429_; uint8_t v___x_4430_; 
v___x_4429_ = lean_array_get_size(v_keys_4425_);
v___x_4430_ = lean_nat_dec_lt(v_i_4427_, v___x_4429_);
if (v___x_4430_ == 0)
{
lean_dec(v_i_4427_);
return v_entries_4428_;
}
else
{
lean_object* v_k_4431_; lean_object* v_v_4432_; uint64_t v___x_4433_; size_t v_h_4434_; size_t v___x_4435_; lean_object* v___x_4436_; size_t v___x_4437_; size_t v___x_4438_; size_t v___x_4439_; size_t v_h_4440_; lean_object* v___x_4441_; lean_object* v___x_4442_; 
v_k_4431_ = lean_array_fget_borrowed(v_keys_4425_, v_i_4427_);
v_v_4432_ = lean_array_fget_borrowed(v_vals_4426_, v_i_4427_);
v___x_4433_ = l_Lean_instHashableMVarId_hash(v_k_4431_);
v_h_4434_ = lean_uint64_to_usize(v___x_4433_);
v___x_4435_ = ((size_t)5ULL);
v___x_4436_ = lean_unsigned_to_nat(1u);
v___x_4437_ = ((size_t)1ULL);
v___x_4438_ = lean_usize_sub(v_depth_4424_, v___x_4437_);
v___x_4439_ = lean_usize_mul(v___x_4435_, v___x_4438_);
v_h_4440_ = lean_usize_shift_right(v_h_4434_, v___x_4439_);
v___x_4441_ = lean_nat_add(v_i_4427_, v___x_4436_);
lean_dec(v_i_4427_);
lean_inc(v_v_4432_);
lean_inc(v_k_4431_);
v___x_4442_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(v_entries_4428_, v_h_4440_, v_depth_4424_, v_k_4431_, v_v_4432_);
v_i_4427_ = v___x_4441_;
v_entries_4428_ = v___x_4442_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_depth_4444_, lean_object* v_keys_4445_, lean_object* v_vals_4446_, lean_object* v_i_4447_, lean_object* v_entries_4448_){
_start:
{
size_t v_depth_boxed_4449_; lean_object* v_res_4450_; 
v_depth_boxed_4449_ = lean_unbox_usize(v_depth_4444_);
lean_dec(v_depth_4444_);
v_res_4450_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_boxed_4449_, v_keys_4445_, v_vals_4446_, v_i_4447_, v_entries_4448_);
lean_dec_ref(v_vals_4446_);
lean_dec_ref(v_keys_4445_);
return v_res_4450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_x_4451_, lean_object* v_x_4452_, lean_object* v_x_4453_, lean_object* v_x_4454_, lean_object* v_x_4455_){
_start:
{
size_t v_x_28306__boxed_4456_; size_t v_x_28307__boxed_4457_; lean_object* v_res_4458_; 
v_x_28306__boxed_4456_ = lean_unbox_usize(v_x_4452_);
lean_dec(v_x_4452_);
v_x_28307__boxed_4457_ = lean_unbox_usize(v_x_4453_);
lean_dec(v_x_4453_);
v_res_4458_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(v_x_4451_, v_x_28306__boxed_4456_, v_x_28307__boxed_4457_, v_x_4454_, v_x_4455_);
return v_res_4458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1___redArg(lean_object* v_x_4459_, lean_object* v_x_4460_, lean_object* v_x_4461_){
_start:
{
uint64_t v___x_4462_; size_t v___x_4463_; size_t v___x_4464_; lean_object* v___x_4465_; 
v___x_4462_ = l_Lean_instHashableMVarId_hash(v_x_4460_);
v___x_4463_ = lean_uint64_to_usize(v___x_4462_);
v___x_4464_ = ((size_t)1ULL);
v___x_4465_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(v_x_4459_, v___x_4463_, v___x_4464_, v_x_4460_, v_x_4461_);
return v___x_4465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(lean_object* v_mvarId_4466_, lean_object* v_val_4467_, lean_object* v___y_4468_){
_start:
{
lean_object* v___x_4470_; lean_object* v_mctx_4471_; lean_object* v_cache_4472_; lean_object* v_zetaDeltaFVarIds_4473_; lean_object* v_postponed_4474_; lean_object* v_diag_4475_; lean_object* v___x_4477_; uint8_t v_isShared_4478_; uint8_t v_isSharedCheck_4503_; 
v___x_4470_ = lean_st_ref_take(v___y_4468_);
v_mctx_4471_ = lean_ctor_get(v___x_4470_, 0);
v_cache_4472_ = lean_ctor_get(v___x_4470_, 1);
v_zetaDeltaFVarIds_4473_ = lean_ctor_get(v___x_4470_, 2);
v_postponed_4474_ = lean_ctor_get(v___x_4470_, 3);
v_diag_4475_ = lean_ctor_get(v___x_4470_, 4);
v_isSharedCheck_4503_ = !lean_is_exclusive(v___x_4470_);
if (v_isSharedCheck_4503_ == 0)
{
v___x_4477_ = v___x_4470_;
v_isShared_4478_ = v_isSharedCheck_4503_;
goto v_resetjp_4476_;
}
else
{
lean_inc(v_diag_4475_);
lean_inc(v_postponed_4474_);
lean_inc(v_zetaDeltaFVarIds_4473_);
lean_inc(v_cache_4472_);
lean_inc(v_mctx_4471_);
lean_dec(v___x_4470_);
v___x_4477_ = lean_box(0);
v_isShared_4478_ = v_isSharedCheck_4503_;
goto v_resetjp_4476_;
}
v_resetjp_4476_:
{
lean_object* v_depth_4479_; lean_object* v_levelAssignDepth_4480_; lean_object* v_lmvarCounter_4481_; lean_object* v_mvarCounter_4482_; lean_object* v_lDecls_4483_; lean_object* v_decls_4484_; lean_object* v_userNames_4485_; lean_object* v_lAssignment_4486_; lean_object* v_eAssignment_4487_; lean_object* v_dAssignment_4488_; lean_object* v___x_4490_; uint8_t v_isShared_4491_; uint8_t v_isSharedCheck_4502_; 
v_depth_4479_ = lean_ctor_get(v_mctx_4471_, 0);
v_levelAssignDepth_4480_ = lean_ctor_get(v_mctx_4471_, 1);
v_lmvarCounter_4481_ = lean_ctor_get(v_mctx_4471_, 2);
v_mvarCounter_4482_ = lean_ctor_get(v_mctx_4471_, 3);
v_lDecls_4483_ = lean_ctor_get(v_mctx_4471_, 4);
v_decls_4484_ = lean_ctor_get(v_mctx_4471_, 5);
v_userNames_4485_ = lean_ctor_get(v_mctx_4471_, 6);
v_lAssignment_4486_ = lean_ctor_get(v_mctx_4471_, 7);
v_eAssignment_4487_ = lean_ctor_get(v_mctx_4471_, 8);
v_dAssignment_4488_ = lean_ctor_get(v_mctx_4471_, 9);
v_isSharedCheck_4502_ = !lean_is_exclusive(v_mctx_4471_);
if (v_isSharedCheck_4502_ == 0)
{
v___x_4490_ = v_mctx_4471_;
v_isShared_4491_ = v_isSharedCheck_4502_;
goto v_resetjp_4489_;
}
else
{
lean_inc(v_dAssignment_4488_);
lean_inc(v_eAssignment_4487_);
lean_inc(v_lAssignment_4486_);
lean_inc(v_userNames_4485_);
lean_inc(v_decls_4484_);
lean_inc(v_lDecls_4483_);
lean_inc(v_mvarCounter_4482_);
lean_inc(v_lmvarCounter_4481_);
lean_inc(v_levelAssignDepth_4480_);
lean_inc(v_depth_4479_);
lean_dec(v_mctx_4471_);
v___x_4490_ = lean_box(0);
v_isShared_4491_ = v_isSharedCheck_4502_;
goto v_resetjp_4489_;
}
v_resetjp_4489_:
{
lean_object* v___x_4492_; lean_object* v___x_4494_; 
v___x_4492_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1___redArg(v_eAssignment_4487_, v_mvarId_4466_, v_val_4467_);
if (v_isShared_4491_ == 0)
{
lean_ctor_set(v___x_4490_, 8, v___x_4492_);
v___x_4494_ = v___x_4490_;
goto v_reusejp_4493_;
}
else
{
lean_object* v_reuseFailAlloc_4501_; 
v_reuseFailAlloc_4501_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_4501_, 0, v_depth_4479_);
lean_ctor_set(v_reuseFailAlloc_4501_, 1, v_levelAssignDepth_4480_);
lean_ctor_set(v_reuseFailAlloc_4501_, 2, v_lmvarCounter_4481_);
lean_ctor_set(v_reuseFailAlloc_4501_, 3, v_mvarCounter_4482_);
lean_ctor_set(v_reuseFailAlloc_4501_, 4, v_lDecls_4483_);
lean_ctor_set(v_reuseFailAlloc_4501_, 5, v_decls_4484_);
lean_ctor_set(v_reuseFailAlloc_4501_, 6, v_userNames_4485_);
lean_ctor_set(v_reuseFailAlloc_4501_, 7, v_lAssignment_4486_);
lean_ctor_set(v_reuseFailAlloc_4501_, 8, v___x_4492_);
lean_ctor_set(v_reuseFailAlloc_4501_, 9, v_dAssignment_4488_);
v___x_4494_ = v_reuseFailAlloc_4501_;
goto v_reusejp_4493_;
}
v_reusejp_4493_:
{
lean_object* v___x_4496_; 
if (v_isShared_4478_ == 0)
{
lean_ctor_set(v___x_4477_, 0, v___x_4494_);
v___x_4496_ = v___x_4477_;
goto v_reusejp_4495_;
}
else
{
lean_object* v_reuseFailAlloc_4500_; 
v_reuseFailAlloc_4500_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4500_, 0, v___x_4494_);
lean_ctor_set(v_reuseFailAlloc_4500_, 1, v_cache_4472_);
lean_ctor_set(v_reuseFailAlloc_4500_, 2, v_zetaDeltaFVarIds_4473_);
lean_ctor_set(v_reuseFailAlloc_4500_, 3, v_postponed_4474_);
lean_ctor_set(v_reuseFailAlloc_4500_, 4, v_diag_4475_);
v___x_4496_ = v_reuseFailAlloc_4500_;
goto v_reusejp_4495_;
}
v_reusejp_4495_:
{
lean_object* v___x_4497_; lean_object* v___x_4498_; lean_object* v___x_4499_; 
v___x_4497_ = lean_st_ref_set(v___y_4468_, v___x_4496_);
v___x_4498_ = lean_box(0);
v___x_4499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4499_, 0, v___x_4498_);
return v___x_4499_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg___boxed(lean_object* v_mvarId_4504_, lean_object* v_val_4505_, lean_object* v___y_4506_, lean_object* v___y_4507_){
_start:
{
lean_object* v_res_4508_; 
v_res_4508_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(v_mvarId_4504_, v_val_4505_, v___y_4506_);
lean_dec(v___y_4506_);
return v_res_4508_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1(void){
_start:
{
lean_object* v___x_4510_; lean_object* v___x_4511_; 
v___x_4510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__0));
v___x_4511_ = l_Lean_stringToMessageData(v___x_4510_);
return v___x_4511_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9(void){
_start:
{
lean_object* v___x_4531_; lean_object* v___x_4532_; 
v___x_4531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__8));
v___x_4532_ = l_Lean_stringToMessageData(v___x_4531_);
return v___x_4532_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13(void){
_start:
{
lean_object* v___x_4536_; lean_object* v___x_4537_; lean_object* v___x_4538_; lean_object* v___x_4539_; lean_object* v___x_4540_; lean_object* v___x_4541_; 
v___x_4536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__12));
v___x_4537_ = lean_unsigned_to_nat(39u);
v___x_4538_ = lean_unsigned_to_nat(551u);
v___x_4539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__11));
v___x_4540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__10));
v___x_4541_ = l_mkPanicMessageWithDecl(v___x_4540_, v___x_4539_, v___x_4538_, v___x_4537_, v___x_4536_);
return v___x_4541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq(lean_object* v_g_4542_, lean_object* v_a_4543_, lean_object* v_a_4544_, lean_object* v_a_4545_, lean_object* v_a_4546_, lean_object* v_a_4547_, lean_object* v_a_4548_){
_start:
{
lean_object* v___y_4551_; lean_object* v___y_4552_; uint8_t v___y_4553_; lean_object* v___y_4557_; lean_object* v_a_4558_; lean_object* v___x_4561_; 
lean_inc(v_g_4542_);
v___x_4561_ = l_Lean_MVarId_getType(v_g_4542_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4561_) == 0)
{
lean_object* v_a_4562_; lean_object* v___x_4563_; lean_object* v_a_4564_; lean_object* v___x_4565_; 
v_a_4562_ = lean_ctor_get(v___x_4561_, 0);
lean_inc(v_a_4562_);
lean_dec_ref_known(v___x_4561_, 1);
v___x_4563_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Ring_proveEq_spec__0___redArg(v_a_4562_, v_a_4546_);
v_a_4564_ = lean_ctor_get(v___x_4563_, 0);
lean_inc(v_a_4564_);
lean_dec_ref(v___x_4563_);
v___x_4565_ = l_Lean_Meta_whnfR(v_a_4564_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4565_) == 0)
{
lean_object* v_a_4566_; lean_object* v___x_4567_; lean_object* v___x_4568_; uint8_t v___x_4569_; 
v_a_4566_ = lean_ctor_get(v___x_4565_, 0);
lean_inc(v_a_4566_);
lean_dec_ref_known(v___x_4565_, 1);
v___x_4567_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore___lam__0___closed__2));
v___x_4568_ = lean_unsigned_to_nat(3u);
v___x_4569_ = l_Lean_Expr_isAppOfArity(v_a_4566_, v___x_4567_, v___x_4568_);
if (v___x_4569_ == 0)
{
lean_object* v___x_4570_; lean_object* v___x_4571_; 
lean_dec(v_a_4566_);
lean_dec(v_g_4542_);
v___x_4570_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__1);
v___x_4571_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v___x_4570_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
return v___x_4571_;
}
else
{
lean_object* v___x_4572_; lean_object* v___x_4573_; lean_object* v___x_4574_; lean_object* v___x_4575_; 
v___x_4572_ = l_Lean_Expr_appFn_x21(v_a_4566_);
v___x_4573_ = l_Lean_Expr_appFn_x21(v___x_4572_);
v___x_4574_ = l_Lean_Expr_appArg_x21(v___x_4573_);
lean_dec_ref(v___x_4573_);
lean_inc(v_a_4548_);
lean_inc_ref(v_a_4547_);
lean_inc(v_a_4546_);
lean_inc_ref(v_a_4545_);
lean_inc_ref(v___x_4574_);
v___x_4575_ = lean_infer_type(v___x_4574_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4575_) == 0)
{
lean_object* v_a_4576_; lean_object* v___x_4577_; 
v_a_4576_ = lean_ctor_get(v___x_4575_, 0);
lean_inc(v_a_4576_);
lean_dec_ref_known(v___x_4575_, 1);
lean_inc(v_a_4548_);
lean_inc_ref(v_a_4547_);
lean_inc(v_a_4546_);
lean_inc_ref(v_a_4545_);
v___x_4577_ = lean_whnf(v_a_4576_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4577_) == 0)
{
lean_object* v_a_4578_; lean_object* v___x_4580_; uint8_t v_isShared_4581_; uint8_t v_isSharedCheck_4734_; 
v_a_4578_ = lean_ctor_get(v___x_4577_, 0);
v_isSharedCheck_4734_ = !lean_is_exclusive(v___x_4577_);
if (v_isSharedCheck_4734_ == 0)
{
v___x_4580_ = v___x_4577_;
v_isShared_4581_ = v_isSharedCheck_4734_;
goto v_resetjp_4579_;
}
else
{
lean_inc(v_a_4578_);
lean_dec(v___x_4577_);
v___x_4580_ = lean_box(0);
v_isShared_4581_ = v_isSharedCheck_4734_;
goto v_resetjp_4579_;
}
v_resetjp_4579_:
{
if (lean_obj_tag(v_a_4578_) == 3)
{
lean_object* v_u_4582_; lean_object* v___x_4583_; lean_object* v___x_4584_; lean_object* v___y_4586_; lean_object* v___y_4587_; lean_object* v___y_4588_; lean_object* v___y_4589_; lean_object* v___y_4590_; uint8_t v___y_4591_; lean_object* v_a_4680_; lean_object* v___x_4704_; 
v_u_4582_ = lean_ctor_get(v_a_4578_, 0);
lean_inc(v_u_4582_);
lean_dec_ref_known(v_a_4578_, 1);
v___x_4583_ = l_Lean_Expr_appArg_x21(v___x_4572_);
lean_dec_ref(v___x_4572_);
v___x_4584_ = l_Lean_Expr_appArg_x21(v_a_4566_);
lean_dec(v_a_4566_);
v___x_4704_ = l_Lean_Level_dec(v_u_4582_);
lean_dec(v_u_4582_);
if (lean_obj_tag(v___x_4704_) == 0)
{
lean_object* v___x_4705_; lean_object* v___x_4706_; lean_object* v_a_4707_; lean_object* v___x_4709_; uint8_t v_isShared_4710_; uint8_t v_isSharedCheck_4730_; 
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_del_object(v___x_4580_);
lean_dec(v_g_4542_);
v___x_4705_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1, &lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Ring_RingCompute_add___closed__1);
v___x_4706_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_RingCompute_add_spec__0___redArg(v___x_4705_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
v_a_4707_ = lean_ctor_get(v___x_4706_, 0);
v_isSharedCheck_4730_ = !lean_is_exclusive(v___x_4706_);
if (v_isSharedCheck_4730_ == 0)
{
v___x_4709_ = v___x_4706_;
v_isShared_4710_ = v_isSharedCheck_4730_;
goto v_resetjp_4708_;
}
else
{
lean_inc(v_a_4707_);
lean_dec(v___x_4706_);
v___x_4709_ = lean_box(0);
v_isShared_4710_ = v_isSharedCheck_4730_;
goto v_resetjp_4708_;
}
v_resetjp_4708_:
{
uint8_t v___y_4712_; uint8_t v___x_4728_; 
v___x_4728_ = l_Lean_Exception_isInterrupt(v_a_4707_);
if (v___x_4728_ == 0)
{
uint8_t v___x_4729_; 
lean_inc(v_a_4707_);
v___x_4729_ = l_Lean_Exception_isRuntime(v_a_4707_);
v___y_4712_ = v___x_4729_;
goto v___jp_4711_;
}
else
{
v___y_4712_ = v___x_4728_;
goto v___jp_4711_;
}
v___jp_4711_:
{
if (v___y_4712_ == 0)
{
lean_object* v___x_4713_; lean_object* v___x_4714_; lean_object* v___x_4715_; lean_object* v___x_4716_; lean_object* v_a_4717_; lean_object* v___x_4719_; uint8_t v_isShared_4720_; uint8_t v_isSharedCheck_4724_; 
lean_del_object(v___x_4709_);
lean_dec(v_a_4707_);
v___x_4713_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9, &lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__9);
v___x_4714_ = l_Lean_indentExpr(v___x_4574_);
v___x_4715_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4715_, 0, v___x_4713_);
lean_ctor_set(v___x_4715_, 1, v___x_4714_);
v___x_4716_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Ring_ExProd_evalIntCast_spec__4___redArg(v___x_4715_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
v_a_4717_ = lean_ctor_get(v___x_4716_, 0);
v_isSharedCheck_4724_ = !lean_is_exclusive(v___x_4716_);
if (v_isSharedCheck_4724_ == 0)
{
v___x_4719_ = v___x_4716_;
v_isShared_4720_ = v_isSharedCheck_4724_;
goto v_resetjp_4718_;
}
else
{
lean_inc(v_a_4717_);
lean_dec(v___x_4716_);
v___x_4719_ = lean_box(0);
v_isShared_4720_ = v_isSharedCheck_4724_;
goto v_resetjp_4718_;
}
v_resetjp_4718_:
{
lean_object* v___x_4722_; 
if (v_isShared_4720_ == 0)
{
v___x_4722_ = v___x_4719_;
goto v_reusejp_4721_;
}
else
{
lean_object* v_reuseFailAlloc_4723_; 
v_reuseFailAlloc_4723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4723_, 0, v_a_4717_);
v___x_4722_ = v_reuseFailAlloc_4723_;
goto v_reusejp_4721_;
}
v_reusejp_4721_:
{
return v___x_4722_;
}
}
}
else
{
lean_object* v___x_4726_; 
lean_dec_ref(v___x_4574_);
if (v_isShared_4710_ == 0)
{
v___x_4726_ = v___x_4709_;
goto v_reusejp_4725_;
}
else
{
lean_object* v_reuseFailAlloc_4727_; 
v_reuseFailAlloc_4727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4727_, 0, v_a_4707_);
v___x_4726_ = v_reuseFailAlloc_4727_;
goto v_reusejp_4725_;
}
v_reusejp_4725_:
{
return v___x_4726_;
}
}
}
}
}
else
{
lean_object* v_val_4731_; 
v_val_4731_ = lean_ctor_get(v___x_4704_, 0);
lean_inc(v_val_4731_);
lean_dec_ref_known(v___x_4704_, 1);
v_a_4680_ = v_val_4731_;
goto v___jp_4679_;
}
v___jp_4585_:
{
if (v___y_4591_ == 0)
{
uint8_t v___x_4592_; lean_object* v___x_4593_; lean_object* v___x_4594_; 
lean_del_object(v___x_4580_);
v___x_4592_ = 0;
v___x_4593_ = lean_box(0);
v___x_4594_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v___y_4586_, v___x_4592_, v___x_4593_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4594_) == 0)
{
lean_object* v_a_4595_; lean_object* v___x_4596_; 
v_a_4595_ = lean_ctor_get(v___x_4594_, 0);
lean_inc_n(v_a_4595_, 2);
lean_dec_ref_known(v___x_4594_, 1);
v___x_4596_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v_a_4595_, v___x_4592_, v___x_4593_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4596_) == 0)
{
lean_object* v_a_4597_; lean_object* v___x_4598_; 
v_a_4597_ = lean_ctor_get(v___x_4596_, 0);
lean_inc(v_a_4597_);
lean_dec_ref_known(v___x_4596_, 1);
lean_inc(v_a_4595_);
v___x_4598_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v_a_4595_, v___x_4592_, v___x_4593_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4598_) == 0)
{
lean_object* v_a_4599_; lean_object* v___x_4600_; lean_object* v___x_4601_; lean_object* v___x_4602_; lean_object* v___x_4603_; lean_object* v___x_4604_; 
v_a_4599_ = lean_ctor_get(v___x_4598_, 0);
lean_inc(v_a_4599_);
lean_dec_ref_known(v___x_4598_, 1);
v___x_4600_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__3));
lean_inc(v___y_4588_);
v___x_4601_ = l_Lean_Expr_const___override(v___x_4600_, v___y_4588_);
lean_inc_ref(v___x_4574_);
v___x_4602_ = l_Lean_Expr_app___override(v___x_4601_, v___x_4574_);
lean_inc(v_a_4595_);
v___x_4603_ = l_Lean_Expr_app___override(v___x_4602_, v_a_4595_);
v___x_4604_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4603_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4604_) == 0)
{
lean_object* v_a_4605_; lean_object* v___x_4606_; lean_object* v___x_4607_; 
v_a_4605_ = lean_ctor_get(v___x_4604_, 0);
lean_inc(v_a_4605_);
lean_dec_ref_known(v___x_4604_, 1);
lean_inc(v_a_4595_);
v___x_4606_ = l_Lean_Expr_app___override(v___y_4587_, v_a_4595_);
v___x_4607_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4606_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4607_) == 0)
{
lean_object* v_a_4608_; lean_object* v___x_4609_; lean_object* v___x_4610_; lean_object* v___x_4611_; lean_object* v___x_4612_; lean_object* v___x_4613_; lean_object* v___x_4614_; lean_object* v___x_4615_; lean_object* v___x_4616_; 
v_a_4608_ = lean_ctor_get(v___x_4607_, 0);
lean_inc(v_a_4608_);
lean_dec_ref_known(v___x_4607_, 1);
v___x_4609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__5));
lean_inc(v___y_4588_);
v___x_4610_ = l_Lean_Expr_const___override(v___x_4609_, v___y_4588_);
lean_inc_ref(v___x_4574_);
v___x_4611_ = l_Lean_Expr_app___override(v___x_4610_, v___x_4574_);
lean_inc(v_a_4595_);
v___x_4612_ = l_Lean_Expr_app___override(v___x_4611_, v_a_4595_);
lean_inc(v_a_4605_);
v___x_4613_ = l_Lean_Expr_app___override(v___x_4612_, v_a_4605_);
lean_inc_ref(v___x_4583_);
lean_inc_ref(v___x_4613_);
v___x_4614_ = l_Lean_Expr_app___override(v___x_4613_, v___x_4583_);
lean_inc(v_a_4597_);
v___x_4615_ = l_Lean_Expr_app___override(v___x_4614_, v_a_4597_);
v___x_4616_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4615_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4616_) == 0)
{
lean_object* v_a_4617_; lean_object* v___x_4618_; lean_object* v___x_4619_; lean_object* v___x_4620_; 
v_a_4617_ = lean_ctor_get(v___x_4616_, 0);
lean_inc(v_a_4617_);
lean_dec_ref_known(v___x_4616_, 1);
lean_inc_ref(v___x_4584_);
v___x_4618_ = l_Lean_Expr_app___override(v___x_4613_, v___x_4584_);
lean_inc(v_a_4599_);
v___x_4619_ = l_Lean_Expr_app___override(v___x_4618_, v_a_4599_);
v___x_4620_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4619_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4620_) == 0)
{
lean_object* v_a_4621_; lean_object* v___x_4622_; lean_object* v___x_4623_; lean_object* v___x_4624_; lean_object* v___x_4625_; lean_object* v___x_4626_; lean_object* v___x_4627_; lean_object* v___x_4628_; lean_object* v___x_4629_; lean_object* v___x_4630_; lean_object* v___x_4631_; 
lean_dec_ref(v___y_4589_);
v_a_4621_ = lean_ctor_get(v___x_4620_, 0);
lean_inc(v_a_4621_);
lean_dec_ref_known(v___x_4620_, 1);
v___x_4622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__7));
v___x_4623_ = l_Lean_Expr_const___override(v___x_4622_, v___y_4588_);
v___x_4624_ = l_Lean_Expr_app___override(v___x_4623_, v___x_4574_);
lean_inc(v_a_4595_);
v___x_4625_ = l_Lean_Expr_app___override(v___x_4624_, v_a_4595_);
v___x_4626_ = l_Lean_Expr_app___override(v___x_4625_, v_a_4605_);
v___x_4627_ = l_Lean_Expr_app___override(v___x_4626_, v___x_4583_);
v___x_4628_ = l_Lean_Expr_app___override(v___x_4627_, v___x_4584_);
lean_inc(v_a_4597_);
v___x_4629_ = l_Lean_Expr_app___override(v___x_4628_, v_a_4597_);
lean_inc(v_a_4599_);
v___x_4630_ = l_Lean_Expr_app___override(v___x_4629_, v_a_4599_);
v___x_4631_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore(v___y_4590_, v_a_4595_, v_a_4608_, v_a_4597_, v_a_4599_, v_a_4543_, v_a_4544_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4631_) == 0)
{
lean_object* v_a_4632_; lean_object* v___x_4633_; lean_object* v___x_4634_; lean_object* v___x_4635_; lean_object* v___x_4636_; lean_object* v___x_4637_; lean_object* v___x_4638_; lean_object* v___x_4639_; 
v_a_4632_ = lean_ctor_get(v___x_4631_, 0);
lean_inc(v_a_4632_);
lean_dec_ref_known(v___x_4631_, 1);
v___x_4633_ = l_Lean_Expr_app___override(v___x_4630_, v_a_4617_);
v___x_4634_ = l_Lean_Expr_app___override(v___x_4633_, v_a_4621_);
v___x_4635_ = lean_box(0);
v___x_4636_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4636_, 0, v_a_4632_);
lean_ctor_set(v___x_4636_, 1, v___x_4635_);
v___x_4637_ = lean_array_mk(v___x_4636_);
v___x_4638_ = l_Lean_Expr_betaRev(v___x_4634_, v___x_4637_, v___y_4591_, v___y_4591_);
lean_dec_ref(v___x_4637_);
v___x_4639_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(v_g_4542_, v___x_4638_, v_a_4546_);
return v___x_4639_;
}
else
{
lean_object* v_a_4640_; lean_object* v___x_4642_; uint8_t v_isShared_4643_; uint8_t v_isSharedCheck_4647_; 
lean_dec_ref(v___x_4630_);
lean_dec(v_a_4621_);
lean_dec(v_a_4617_);
lean_dec(v_g_4542_);
v_a_4640_ = lean_ctor_get(v___x_4631_, 0);
v_isSharedCheck_4647_ = !lean_is_exclusive(v___x_4631_);
if (v_isSharedCheck_4647_ == 0)
{
v___x_4642_ = v___x_4631_;
v_isShared_4643_ = v_isSharedCheck_4647_;
goto v_resetjp_4641_;
}
else
{
lean_inc(v_a_4640_);
lean_dec(v___x_4631_);
v___x_4642_ = lean_box(0);
v_isShared_4643_ = v_isSharedCheck_4647_;
goto v_resetjp_4641_;
}
v_resetjp_4641_:
{
lean_object* v___x_4645_; 
if (v_isShared_4643_ == 0)
{
v___x_4645_ = v___x_4642_;
goto v_reusejp_4644_;
}
else
{
lean_object* v_reuseFailAlloc_4646_; 
v_reuseFailAlloc_4646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4646_, 0, v_a_4640_);
v___x_4645_ = v_reuseFailAlloc_4646_;
goto v_reusejp_4644_;
}
v_reusejp_4644_:
{
return v___x_4645_;
}
}
}
}
else
{
lean_object* v_a_4648_; 
lean_dec(v_a_4617_);
lean_dec(v_a_4608_);
lean_dec(v_a_4605_);
lean_dec(v_a_4599_);
lean_dec(v_a_4597_);
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec(v___y_4588_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4648_ = lean_ctor_get(v___x_4620_, 0);
lean_inc(v_a_4648_);
lean_dec_ref_known(v___x_4620_, 1);
v___y_4557_ = v___y_4589_;
v_a_4558_ = v_a_4648_;
goto v___jp_4556_;
}
}
else
{
lean_object* v_a_4649_; 
lean_dec_ref(v___x_4613_);
lean_dec(v_a_4608_);
lean_dec(v_a_4605_);
lean_dec(v_a_4599_);
lean_dec(v_a_4597_);
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec(v___y_4588_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4649_ = lean_ctor_get(v___x_4616_, 0);
lean_inc(v_a_4649_);
lean_dec_ref_known(v___x_4616_, 1);
v___y_4557_ = v___y_4589_;
v_a_4558_ = v_a_4649_;
goto v___jp_4556_;
}
}
else
{
lean_object* v_a_4650_; 
lean_dec(v_a_4605_);
lean_dec(v_a_4599_);
lean_dec(v_a_4597_);
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec(v___y_4588_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4650_ = lean_ctor_get(v___x_4607_, 0);
lean_inc(v_a_4650_);
lean_dec_ref_known(v___x_4607_, 1);
v___y_4557_ = v___y_4589_;
v_a_4558_ = v_a_4650_;
goto v___jp_4556_;
}
}
else
{
lean_object* v_a_4651_; 
lean_dec(v_a_4599_);
lean_dec(v_a_4597_);
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4651_ = lean_ctor_get(v___x_4604_, 0);
lean_inc(v_a_4651_);
lean_dec_ref_known(v___x_4604_, 1);
v___y_4557_ = v___y_4589_;
v_a_4558_ = v_a_4651_;
goto v___jp_4556_;
}
}
else
{
lean_object* v_a_4652_; lean_object* v___x_4654_; uint8_t v_isShared_4655_; uint8_t v_isSharedCheck_4659_; 
lean_dec(v_a_4597_);
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec_ref(v___y_4589_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4652_ = lean_ctor_get(v___x_4598_, 0);
v_isSharedCheck_4659_ = !lean_is_exclusive(v___x_4598_);
if (v_isSharedCheck_4659_ == 0)
{
v___x_4654_ = v___x_4598_;
v_isShared_4655_ = v_isSharedCheck_4659_;
goto v_resetjp_4653_;
}
else
{
lean_inc(v_a_4652_);
lean_dec(v___x_4598_);
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
}
else
{
lean_object* v_a_4660_; lean_object* v___x_4662_; uint8_t v_isShared_4663_; uint8_t v_isSharedCheck_4667_; 
lean_dec(v_a_4595_);
lean_dec(v___y_4590_);
lean_dec_ref(v___y_4589_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4660_ = lean_ctor_get(v___x_4596_, 0);
v_isSharedCheck_4667_ = !lean_is_exclusive(v___x_4596_);
if (v_isSharedCheck_4667_ == 0)
{
v___x_4662_ = v___x_4596_;
v_isShared_4663_ = v_isSharedCheck_4667_;
goto v_resetjp_4661_;
}
else
{
lean_inc(v_a_4660_);
lean_dec(v___x_4596_);
v___x_4662_ = lean_box(0);
v_isShared_4663_ = v_isSharedCheck_4667_;
goto v_resetjp_4661_;
}
v_resetjp_4661_:
{
lean_object* v___x_4665_; 
if (v_isShared_4663_ == 0)
{
v___x_4665_ = v___x_4662_;
goto v_reusejp_4664_;
}
else
{
lean_object* v_reuseFailAlloc_4666_; 
v_reuseFailAlloc_4666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4666_, 0, v_a_4660_);
v___x_4665_ = v_reuseFailAlloc_4666_;
goto v_reusejp_4664_;
}
v_reusejp_4664_:
{
return v___x_4665_;
}
}
}
}
else
{
lean_object* v_a_4668_; lean_object* v___x_4670_; uint8_t v_isShared_4671_; uint8_t v_isSharedCheck_4675_; 
lean_dec(v___y_4590_);
lean_dec_ref(v___y_4589_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
v_a_4668_ = lean_ctor_get(v___x_4594_, 0);
v_isSharedCheck_4675_ = !lean_is_exclusive(v___x_4594_);
if (v_isSharedCheck_4675_ == 0)
{
v___x_4670_ = v___x_4594_;
v_isShared_4671_ = v_isSharedCheck_4675_;
goto v_resetjp_4669_;
}
else
{
lean_inc(v_a_4668_);
lean_dec(v___x_4594_);
v___x_4670_ = lean_box(0);
v_isShared_4671_ = v_isSharedCheck_4675_;
goto v_resetjp_4669_;
}
v_resetjp_4669_:
{
lean_object* v___x_4673_; 
if (v_isShared_4671_ == 0)
{
v___x_4673_ = v___x_4670_;
goto v_reusejp_4672_;
}
else
{
lean_object* v_reuseFailAlloc_4674_; 
v_reuseFailAlloc_4674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4674_, 0, v_a_4668_);
v___x_4673_ = v_reuseFailAlloc_4674_;
goto v_reusejp_4672_;
}
v_reusejp_4672_:
{
return v___x_4673_;
}
}
}
}
else
{
lean_object* v___x_4677_; 
lean_dec(v___y_4590_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec_ref(v___y_4586_);
lean_dec_ref(v___x_4584_);
lean_dec_ref(v___x_4583_);
lean_dec_ref(v___x_4574_);
lean_dec(v_g_4542_);
if (v_isShared_4581_ == 0)
{
lean_ctor_set_tag(v___x_4580_, 1);
lean_ctor_set(v___x_4580_, 0, v___y_4589_);
v___x_4677_ = v___x_4580_;
goto v_reusejp_4676_;
}
else
{
lean_object* v_reuseFailAlloc_4678_; 
v_reuseFailAlloc_4678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4678_, 0, v___y_4589_);
v___x_4677_ = v_reuseFailAlloc_4678_;
goto v_reusejp_4676_;
}
v_reusejp_4676_:
{
return v___x_4677_;
}
}
}
v___jp_4679_:
{
lean_object* v___x_4681_; lean_object* v___x_4682_; lean_object* v___x_4683_; lean_object* v___x_4684_; lean_object* v___x_4685_; lean_object* v___x_4686_; lean_object* v___x_4687_; 
lean_inc_n(v_a_4680_, 2);
v___x_4681_ = l_Lean_Level_succ___override(v_a_4680_);
v___x_4682_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_RingCompute_cast___redArg___closed__5));
v___x_4683_ = lean_box(0);
v___x_4684_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4684_, 0, v_a_4680_);
lean_ctor_set(v___x_4684_, 1, v___x_4683_);
lean_inc_ref(v___x_4684_);
v___x_4685_ = l_Lean_Expr_const___override(v___x_4682_, v___x_4684_);
lean_inc_ref(v___x_4574_);
lean_inc_ref(v___x_4685_);
v___x_4686_ = l_Lean_Expr_app___override(v___x_4685_, v___x_4574_);
v___x_4687_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_4686_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4687_) == 0)
{
lean_object* v_a_4688_; lean_object* v___x_4689_; 
lean_dec_ref(v___x_4685_);
lean_dec_ref_known(v___x_4684_, 2);
lean_dec(v___x_4681_);
lean_del_object(v___x_4580_);
v_a_4688_ = lean_ctor_get(v___x_4687_, 0);
lean_inc(v_a_4688_);
lean_dec_ref_known(v___x_4687_, 1);
v___x_4689_ = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_proveEq_ringCore(v_a_4680_, v___x_4574_, v_a_4688_, v___x_4583_, v___x_4584_, v_a_4543_, v_a_4544_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
if (lean_obj_tag(v___x_4689_) == 0)
{
lean_object* v_a_4690_; lean_object* v___x_4691_; 
v_a_4690_ = lean_ctor_get(v___x_4689_, 0);
lean_inc(v_a_4690_);
lean_dec_ref_known(v___x_4689_, 1);
v___x_4691_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(v_g_4542_, v_a_4690_, v_a_4546_);
return v___x_4691_;
}
else
{
lean_object* v_a_4692_; lean_object* v___x_4694_; uint8_t v_isShared_4695_; uint8_t v_isSharedCheck_4699_; 
lean_dec(v_g_4542_);
v_a_4692_ = lean_ctor_get(v___x_4689_, 0);
v_isSharedCheck_4699_ = !lean_is_exclusive(v___x_4689_);
if (v_isSharedCheck_4699_ == 0)
{
v___x_4694_ = v___x_4689_;
v_isShared_4695_ = v_isSharedCheck_4699_;
goto v_resetjp_4693_;
}
else
{
lean_inc(v_a_4692_);
lean_dec(v___x_4689_);
v___x_4694_ = lean_box(0);
v_isShared_4695_ = v_isSharedCheck_4699_;
goto v_resetjp_4693_;
}
v_resetjp_4693_:
{
lean_object* v___x_4697_; 
if (v_isShared_4695_ == 0)
{
v___x_4697_ = v___x_4694_;
goto v_reusejp_4696_;
}
else
{
lean_object* v_reuseFailAlloc_4698_; 
v_reuseFailAlloc_4698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4698_, 0, v_a_4692_);
v___x_4697_ = v_reuseFailAlloc_4698_;
goto v_reusejp_4696_;
}
v_reusejp_4696_:
{
return v___x_4697_;
}
}
}
}
else
{
lean_object* v_a_4700_; lean_object* v___x_4701_; uint8_t v___x_4702_; 
v_a_4700_ = lean_ctor_get(v___x_4687_, 0);
lean_inc(v_a_4700_);
lean_dec_ref_known(v___x_4687_, 1);
v___x_4701_ = l_Lean_Expr_sort___override(v___x_4681_);
v___x_4702_ = l_Lean_Exception_isInterrupt(v_a_4700_);
if (v___x_4702_ == 0)
{
uint8_t v___x_4703_; 
lean_inc(v_a_4700_);
v___x_4703_ = l_Lean_Exception_isRuntime(v_a_4700_);
v___y_4586_ = v___x_4701_;
v___y_4587_ = v___x_4685_;
v___y_4588_ = v___x_4684_;
v___y_4589_ = v_a_4700_;
v___y_4590_ = v_a_4680_;
v___y_4591_ = v___x_4703_;
goto v___jp_4585_;
}
else
{
v___y_4586_ = v___x_4701_;
v___y_4587_ = v___x_4685_;
v___y_4588_ = v___x_4684_;
v___y_4589_ = v_a_4700_;
v___y_4590_ = v_a_4680_;
v___y_4591_ = v___x_4702_;
goto v___jp_4585_;
}
}
}
}
else
{
lean_object* v___x_4732_; lean_object* v___x_4733_; 
lean_del_object(v___x_4580_);
lean_dec(v_a_4578_);
lean_dec_ref(v___x_4574_);
lean_dec_ref(v___x_4572_);
lean_dec(v_a_4566_);
lean_dec(v_g_4542_);
v___x_4732_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13, &lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Ring_proveEq___closed__13);
v___x_4733_ = lp_mathlib_panic___at___00Mathlib_Tactic_Ring_proveEq_spec__2(v___x_4732_, v_a_4543_, v_a_4544_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_);
return v___x_4733_;
}
}
}
else
{
lean_object* v_a_4735_; lean_object* v___x_4737_; uint8_t v_isShared_4738_; uint8_t v_isSharedCheck_4742_; 
lean_dec_ref(v___x_4574_);
lean_dec_ref(v___x_4572_);
lean_dec(v_a_4566_);
lean_dec(v_g_4542_);
v_a_4735_ = lean_ctor_get(v___x_4577_, 0);
v_isSharedCheck_4742_ = !lean_is_exclusive(v___x_4577_);
if (v_isSharedCheck_4742_ == 0)
{
v___x_4737_ = v___x_4577_;
v_isShared_4738_ = v_isSharedCheck_4742_;
goto v_resetjp_4736_;
}
else
{
lean_inc(v_a_4735_);
lean_dec(v___x_4577_);
v___x_4737_ = lean_box(0);
v_isShared_4738_ = v_isSharedCheck_4742_;
goto v_resetjp_4736_;
}
v_resetjp_4736_:
{
lean_object* v___x_4740_; 
if (v_isShared_4738_ == 0)
{
v___x_4740_ = v___x_4737_;
goto v_reusejp_4739_;
}
else
{
lean_object* v_reuseFailAlloc_4741_; 
v_reuseFailAlloc_4741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4741_, 0, v_a_4735_);
v___x_4740_ = v_reuseFailAlloc_4741_;
goto v_reusejp_4739_;
}
v_reusejp_4739_:
{
return v___x_4740_;
}
}
}
}
else
{
lean_object* v_a_4743_; lean_object* v___x_4745_; uint8_t v_isShared_4746_; uint8_t v_isSharedCheck_4750_; 
lean_dec_ref(v___x_4574_);
lean_dec_ref(v___x_4572_);
lean_dec(v_a_4566_);
lean_dec(v_g_4542_);
v_a_4743_ = lean_ctor_get(v___x_4575_, 0);
v_isSharedCheck_4750_ = !lean_is_exclusive(v___x_4575_);
if (v_isSharedCheck_4750_ == 0)
{
v___x_4745_ = v___x_4575_;
v_isShared_4746_ = v_isSharedCheck_4750_;
goto v_resetjp_4744_;
}
else
{
lean_inc(v_a_4743_);
lean_dec(v___x_4575_);
v___x_4745_ = lean_box(0);
v_isShared_4746_ = v_isSharedCheck_4750_;
goto v_resetjp_4744_;
}
v_resetjp_4744_:
{
lean_object* v___x_4748_; 
if (v_isShared_4746_ == 0)
{
v___x_4748_ = v___x_4745_;
goto v_reusejp_4747_;
}
else
{
lean_object* v_reuseFailAlloc_4749_; 
v_reuseFailAlloc_4749_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4749_, 0, v_a_4743_);
v___x_4748_ = v_reuseFailAlloc_4749_;
goto v_reusejp_4747_;
}
v_reusejp_4747_:
{
return v___x_4748_;
}
}
}
}
}
else
{
lean_object* v_a_4751_; lean_object* v___x_4753_; uint8_t v_isShared_4754_; uint8_t v_isSharedCheck_4758_; 
lean_dec(v_g_4542_);
v_a_4751_ = lean_ctor_get(v___x_4565_, 0);
v_isSharedCheck_4758_ = !lean_is_exclusive(v___x_4565_);
if (v_isSharedCheck_4758_ == 0)
{
v___x_4753_ = v___x_4565_;
v_isShared_4754_ = v_isSharedCheck_4758_;
goto v_resetjp_4752_;
}
else
{
lean_inc(v_a_4751_);
lean_dec(v___x_4565_);
v___x_4753_ = lean_box(0);
v_isShared_4754_ = v_isSharedCheck_4758_;
goto v_resetjp_4752_;
}
v_resetjp_4752_:
{
lean_object* v___x_4756_; 
if (v_isShared_4754_ == 0)
{
v___x_4756_ = v___x_4753_;
goto v_reusejp_4755_;
}
else
{
lean_object* v_reuseFailAlloc_4757_; 
v_reuseFailAlloc_4757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4757_, 0, v_a_4751_);
v___x_4756_ = v_reuseFailAlloc_4757_;
goto v_reusejp_4755_;
}
v_reusejp_4755_:
{
return v___x_4756_;
}
}
}
}
else
{
lean_object* v_a_4759_; lean_object* v___x_4761_; uint8_t v_isShared_4762_; uint8_t v_isSharedCheck_4766_; 
lean_dec(v_g_4542_);
v_a_4759_ = lean_ctor_get(v___x_4561_, 0);
v_isSharedCheck_4766_ = !lean_is_exclusive(v___x_4561_);
if (v_isSharedCheck_4766_ == 0)
{
v___x_4761_ = v___x_4561_;
v_isShared_4762_ = v_isSharedCheck_4766_;
goto v_resetjp_4760_;
}
else
{
lean_inc(v_a_4759_);
lean_dec(v___x_4561_);
v___x_4761_ = lean_box(0);
v_isShared_4762_ = v_isSharedCheck_4766_;
goto v_resetjp_4760_;
}
v_resetjp_4760_:
{
lean_object* v___x_4764_; 
if (v_isShared_4762_ == 0)
{
v___x_4764_ = v___x_4761_;
goto v_reusejp_4763_;
}
else
{
lean_object* v_reuseFailAlloc_4765_; 
v_reuseFailAlloc_4765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4765_, 0, v_a_4759_);
v___x_4764_ = v_reuseFailAlloc_4765_;
goto v_reusejp_4763_;
}
v_reusejp_4763_:
{
return v___x_4764_;
}
}
}
v___jp_4550_:
{
if (v___y_4553_ == 0)
{
lean_object* v___x_4554_; 
lean_dec_ref(v___y_4551_);
v___x_4554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4554_, 0, v___y_4552_);
return v___x_4554_;
}
else
{
lean_object* v___x_4555_; 
lean_dec_ref(v___y_4552_);
v___x_4555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4555_, 0, v___y_4551_);
return v___x_4555_;
}
}
v___jp_4556_:
{
uint8_t v___x_4559_; 
v___x_4559_ = l_Lean_Exception_isInterrupt(v_a_4558_);
if (v___x_4559_ == 0)
{
uint8_t v___x_4560_; 
lean_inc_ref(v_a_4558_);
v___x_4560_ = l_Lean_Exception_isRuntime(v_a_4558_);
v___y_4551_ = v_a_4558_;
v___y_4552_ = v___y_4557_;
v___y_4553_ = v___x_4560_;
goto v___jp_4550_;
}
else
{
v___y_4551_ = v_a_4558_;
v___y_4552_ = v___y_4557_;
v___y_4553_ = v___x_4559_;
goto v___jp_4550_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring_proveEq___boxed(lean_object* v_g_4767_, lean_object* v_a_4768_, lean_object* v_a_4769_, lean_object* v_a_4770_, lean_object* v_a_4771_, lean_object* v_a_4772_, lean_object* v_a_4773_, lean_object* v_a_4774_){
_start:
{
lean_object* v_res_4775_; 
v_res_4775_ = lp_mathlib_Mathlib_Tactic_Ring_proveEq(v_g_4767_, v_a_4768_, v_a_4769_, v_a_4770_, v_a_4771_, v_a_4772_, v_a_4773_);
lean_dec(v_a_4773_);
lean_dec_ref(v_a_4772_);
lean_dec(v_a_4771_);
lean_dec_ref(v_a_4770_);
lean_dec(v_a_4769_);
lean_dec_ref(v_a_4768_);
return v_res_4775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1(lean_object* v_mvarId_4776_, lean_object* v_val_4777_, lean_object* v___y_4778_, lean_object* v___y_4779_, lean_object* v___y_4780_, lean_object* v___y_4781_, lean_object* v___y_4782_, lean_object* v___y_4783_){
_start:
{
lean_object* v___x_4785_; 
v___x_4785_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___redArg(v_mvarId_4776_, v_val_4777_, v___y_4781_);
return v___x_4785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1___boxed(lean_object* v_mvarId_4786_, lean_object* v_val_4787_, lean_object* v___y_4788_, lean_object* v___y_4789_, lean_object* v___y_4790_, lean_object* v___y_4791_, lean_object* v___y_4792_, lean_object* v___y_4793_, lean_object* v___y_4794_){
_start:
{
lean_object* v_res_4795_; 
v_res_4795_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1(v_mvarId_4786_, v_val_4787_, v___y_4788_, v___y_4789_, v___y_4790_, v___y_4791_, v___y_4792_, v___y_4793_);
lean_dec(v___y_4793_);
lean_dec_ref(v___y_4792_);
lean_dec(v___y_4791_);
lean_dec_ref(v___y_4790_);
lean_dec(v___y_4789_);
lean_dec_ref(v___y_4788_);
return v_res_4795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1(lean_object* v_00_u03b2_4796_, lean_object* v_x_4797_, lean_object* v_x_4798_, lean_object* v_x_4799_){
_start:
{
lean_object* v___x_4800_; 
v___x_4800_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1___redArg(v_x_4797_, v_x_4798_, v_x_4799_);
return v___x_4800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_4801_, lean_object* v_x_4802_, size_t v_x_4803_, size_t v_x_4804_, lean_object* v_x_4805_, lean_object* v_x_4806_){
_start:
{
lean_object* v___x_4807_; 
v___x_4807_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___redArg(v_x_4802_, v_x_4803_, v_x_4804_, v_x_4805_, v_x_4806_);
return v___x_4807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_4808_, lean_object* v_x_4809_, lean_object* v_x_4810_, lean_object* v_x_4811_, lean_object* v_x_4812_, lean_object* v_x_4813_){
_start:
{
size_t v_x_29102__boxed_4814_; size_t v_x_29103__boxed_4815_; lean_object* v_res_4816_; 
v_x_29102__boxed_4814_ = lean_unbox_usize(v_x_4810_);
lean_dec(v_x_4810_);
v_x_29103__boxed_4815_ = lean_unbox_usize(v_x_4811_);
lean_dec(v_x_4811_);
v_res_4816_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3(v_00_u03b2_4808_, v_x_4809_, v_x_29102__boxed_4814_, v_x_29103__boxed_4815_, v_x_4812_, v_x_4813_);
return v_res_4816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4(lean_object* v_00_u03b2_4817_, lean_object* v_n_4818_, lean_object* v_k_4819_, lean_object* v_v_4820_){
_start:
{
lean_object* v___x_4821_; 
v___x_4821_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4___redArg(v_n_4818_, v_k_4819_, v_v_4820_);
return v___x_4821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_4822_, size_t v_depth_4823_, lean_object* v_keys_4824_, lean_object* v_vals_4825_, lean_object* v_heq_4826_, lean_object* v_i_4827_, lean_object* v_entries_4828_){
_start:
{
lean_object* v___x_4829_; 
v___x_4829_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_4823_, v_keys_4824_, v_vals_4825_, v_i_4827_, v_entries_4828_);
return v___x_4829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03b2_4830_, lean_object* v_depth_4831_, lean_object* v_keys_4832_, lean_object* v_vals_4833_, lean_object* v_heq_4834_, lean_object* v_i_4835_, lean_object* v_entries_4836_){
_start:
{
size_t v_depth_boxed_4837_; lean_object* v_res_4838_; 
v_depth_boxed_4837_ = lean_unbox_usize(v_depth_4831_);
lean_dec(v_depth_4831_);
v_res_4838_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__5(v_00_u03b2_4830_, v_depth_boxed_4837_, v_keys_4832_, v_vals_4833_, v_heq_4834_, v_i_4835_, v_entries_4836_);
lean_dec_ref(v_vals_4833_);
lean_dec_ref(v_keys_4832_);
return v_res_4838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_4839_, lean_object* v_x_4840_, lean_object* v_x_4841_, lean_object* v_x_4842_, lean_object* v_x_4843_){
_start:
{
lean_object* v___x_4844_; 
v___x_4844_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Ring_proveEq_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_x_4840_, v_x_4841_, v_x_4842_, v_x_4843_);
return v___x_4844_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4875_; lean_object* v___x_4876_; lean_object* v___x_4877_; 
v___x_4875_ = lean_box(0);
v___x_4876_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4877_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4877_, 0, v___x_4876_);
lean_ctor_set(v___x_4877_, 1, v___x_4875_);
return v___x_4877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg(){
_start:
{
lean_object* v___x_4879_; lean_object* v___x_4880_; 
v___x_4879_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___closed__0);
v___x_4880_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4880_, 0, v___x_4879_);
return v___x_4880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg___boxed(lean_object* v___y_4881_){
_start:
{
lean_object* v_res_4882_; 
v_res_4882_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg();
return v_res_4882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0(lean_object* v_00_u03b1_4883_, lean_object* v___y_4884_, lean_object* v___y_4885_, lean_object* v___y_4886_, lean_object* v___y_4887_, lean_object* v___y_4888_, lean_object* v___y_4889_, lean_object* v___y_4890_, lean_object* v___y_4891_){
_start:
{
lean_object* v___x_4893_; 
v___x_4893_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg();
return v___x_4893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___boxed(lean_object* v_00_u03b1_4894_, lean_object* v___y_4895_, lean_object* v___y_4896_, lean_object* v___y_4897_, lean_object* v___y_4898_, lean_object* v___y_4899_, lean_object* v___y_4900_, lean_object* v___y_4901_, lean_object* v___y_4902_, lean_object* v___y_4903_){
_start:
{
lean_object* v_res_4904_; 
v_res_4904_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0(v_00_u03b1_4894_, v___y_4895_, v___y_4896_, v___y_4897_, v___y_4898_, v___y_4899_, v___y_4900_, v___y_4901_, v___y_4902_);
lean_dec(v___y_4902_);
lean_dec_ref(v___y_4901_);
lean_dec(v___y_4900_);
lean_dec_ref(v___y_4899_);
lean_dec(v___y_4898_);
lean_dec_ref(v___y_4897_);
lean_dec(v___y_4896_);
lean_dec_ref(v___y_4895_);
return v_res_4904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0(uint8_t v___x_4905_, lean_object* v_e_4906_, lean_object* v___y_4907_, lean_object* v___y_4908_, lean_object* v___y_4909_, lean_object* v___y_4910_){
_start:
{
lean_object* v___x_4912_; lean_object* v___x_4913_; lean_object* v___x_4914_; 
v___x_4912_ = lean_box(0);
v___x_4913_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4913_, 0, v_e_4906_);
lean_ctor_set(v___x_4913_, 1, v___x_4912_);
lean_ctor_set_uint8(v___x_4913_, sizeof(void*)*2, v___x_4905_);
v___x_4914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4914_, 0, v___x_4913_);
return v___x_4914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0___boxed(lean_object* v___x_4915_, lean_object* v_e_4916_, lean_object* v___y_4917_, lean_object* v___y_4918_, lean_object* v___y_4919_, lean_object* v___y_4920_, lean_object* v___y_4921_){
_start:
{
uint8_t v___x_767__boxed_4922_; lean_object* v_res_4923_; 
v___x_767__boxed_4922_ = lean_unbox(v___x_4915_);
v_res_4923_ = lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0(v___x_767__boxed_4922_, v_e_4916_, v___y_4917_, v___y_4918_, v___y_4919_, v___y_4920_);
lean_dec(v___y_4920_);
lean_dec_ref(v___y_4919_);
lean_dec(v___y_4918_);
lean_dec_ref(v___y_4917_);
return v_res_4923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1(lean_object* v___f_4924_, lean_object* v___y_4925_, uint8_t v___x_4926_, lean_object* v___y_4927_, lean_object* v___y_4928_, lean_object* v___y_4929_, lean_object* v___y_4930_, lean_object* v___y_4931_, lean_object* v___y_4932_, lean_object* v___y_4933_, lean_object* v___y_4934_){
_start:
{
lean_object* v___x_4936_; 
v___x_4936_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4928_, v___y_4931_, v___y_4932_, v___y_4933_, v___y_4934_);
if (lean_obj_tag(v___x_4936_) == 0)
{
lean_object* v_a_4937_; uint8_t v___y_4939_; 
v_a_4937_ = lean_ctor_get(v___x_4936_, 0);
lean_inc(v_a_4937_);
lean_dec_ref_known(v___x_4936_, 1);
if (lean_obj_tag(v___y_4925_) == 0)
{
goto v___jp_4942_;
}
else
{
if (v___x_4926_ == 0)
{
goto v___jp_4942_;
}
else
{
uint8_t v___x_4944_; 
v___x_4944_ = 1;
v___y_4939_ = v___x_4944_;
goto v___jp_4938_;
}
}
v___jp_4938_:
{
lean_object* v___x_4940_; lean_object* v___x_4941_; 
v___x_4940_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring_proveEq___boxed), 8, 1);
lean_closure_set(v___x_4940_, 0, v_a_4937_);
v___x_4941_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___y_4939_, v___x_4940_, v___f_4924_, v___y_4931_, v___y_4932_, v___y_4933_, v___y_4934_);
return v___x_4941_;
}
v___jp_4942_:
{
uint8_t v___x_4943_; 
v___x_4943_ = 2;
v___y_4939_ = v___x_4943_;
goto v___jp_4938_;
}
}
else
{
lean_object* v_a_4945_; lean_object* v___x_4947_; uint8_t v_isShared_4948_; uint8_t v_isSharedCheck_4952_; 
lean_dec_ref(v___f_4924_);
v_a_4945_ = lean_ctor_get(v___x_4936_, 0);
v_isSharedCheck_4952_ = !lean_is_exclusive(v___x_4936_);
if (v_isSharedCheck_4952_ == 0)
{
v___x_4947_ = v___x_4936_;
v_isShared_4948_ = v_isSharedCheck_4952_;
goto v_resetjp_4946_;
}
else
{
lean_inc(v_a_4945_);
lean_dec(v___x_4936_);
v___x_4947_ = lean_box(0);
v_isShared_4948_ = v_isSharedCheck_4952_;
goto v_resetjp_4946_;
}
v_resetjp_4946_:
{
lean_object* v___x_4950_; 
if (v_isShared_4948_ == 0)
{
v___x_4950_ = v___x_4947_;
goto v_reusejp_4949_;
}
else
{
lean_object* v_reuseFailAlloc_4951_; 
v_reuseFailAlloc_4951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4951_, 0, v_a_4945_);
v___x_4950_ = v_reuseFailAlloc_4951_;
goto v_reusejp_4949_;
}
v_reusejp_4949_:
{
return v___x_4950_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1___boxed(lean_object* v___f_4953_, lean_object* v___y_4954_, lean_object* v___x_4955_, lean_object* v___y_4956_, lean_object* v___y_4957_, lean_object* v___y_4958_, lean_object* v___y_4959_, lean_object* v___y_4960_, lean_object* v___y_4961_, lean_object* v___y_4962_, lean_object* v___y_4963_, lean_object* v___y_4964_){
_start:
{
uint8_t v___x_793__boxed_4965_; lean_object* v_res_4966_; 
v___x_793__boxed_4965_ = lean_unbox(v___x_4955_);
v_res_4966_ = lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1(v___f_4953_, v___y_4954_, v___x_793__boxed_4965_, v___y_4956_, v___y_4957_, v___y_4958_, v___y_4959_, v___y_4960_, v___y_4961_, v___y_4962_, v___y_4963_);
lean_dec(v___y_4963_);
lean_dec_ref(v___y_4962_);
lean_dec(v___y_4961_);
lean_dec_ref(v___y_4960_);
lean_dec(v___y_4959_);
lean_dec_ref(v___y_4958_);
lean_dec(v___y_4957_);
lean_dec_ref(v___y_4956_);
lean_dec(v___y_4954_);
return v_res_4966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1(lean_object* v_x_4967_, lean_object* v_a_4968_, lean_object* v_a_4969_, lean_object* v_a_4970_, lean_object* v_a_4971_, lean_object* v_a_4972_, lean_object* v_a_4973_, lean_object* v_a_4974_, lean_object* v_a_4975_){
_start:
{
lean_object* v___x_4977_; uint8_t v___x_4978_; 
v___x_4977_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1));
lean_inc(v_x_4967_);
v___x_4978_ = l_Lean_Syntax_isOfKind(v_x_4967_, v___x_4977_);
if (v___x_4978_ == 0)
{
lean_object* v___x_4979_; 
lean_dec(v_x_4967_);
v___x_4979_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1_spec__0___redArg();
return v___x_4979_;
}
else
{
lean_object* v___x_4980_; lean_object* v___f_4981_; lean_object* v___y_4983_; lean_object* v___x_4987_; lean_object* v___x_4988_; lean_object* v___x_4989_; 
v___x_4980_ = lean_box(v___x_4978_);
v___f_4981_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__0___boxed), 7, 1);
lean_closure_set(v___f_4981_, 0, v___x_4980_);
v___x_4987_ = lean_unsigned_to_nat(1u);
v___x_4988_ = l_Lean_Syntax_getArg(v_x_4967_, v___x_4987_);
lean_dec(v_x_4967_);
v___x_4989_ = l_Lean_Syntax_getOptional_x3f(v___x_4988_);
lean_dec(v___x_4988_);
if (lean_obj_tag(v___x_4989_) == 0)
{
lean_object* v___x_4990_; 
v___x_4990_ = lean_box(0);
v___y_4983_ = v___x_4990_;
goto v___jp_4982_;
}
else
{
lean_object* v_val_4991_; lean_object* v___x_4993_; uint8_t v_isShared_4994_; uint8_t v_isSharedCheck_4998_; 
v_val_4991_ = lean_ctor_get(v___x_4989_, 0);
v_isSharedCheck_4998_ = !lean_is_exclusive(v___x_4989_);
if (v_isSharedCheck_4998_ == 0)
{
v___x_4993_ = v___x_4989_;
v_isShared_4994_ = v_isSharedCheck_4998_;
goto v_resetjp_4992_;
}
else
{
lean_inc(v_val_4991_);
lean_dec(v___x_4989_);
v___x_4993_ = lean_box(0);
v_isShared_4994_ = v_isSharedCheck_4998_;
goto v_resetjp_4992_;
}
v_resetjp_4992_:
{
lean_object* v___x_4996_; 
if (v_isShared_4994_ == 0)
{
v___x_4996_ = v___x_4993_;
goto v_reusejp_4995_;
}
else
{
lean_object* v_reuseFailAlloc_4997_; 
v_reuseFailAlloc_4997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4997_, 0, v_val_4991_);
v___x_4996_ = v_reuseFailAlloc_4997_;
goto v_reusejp_4995_;
}
v_reusejp_4995_:
{
v___y_4983_ = v___x_4996_;
goto v___jp_4982_;
}
}
}
v___jp_4982_:
{
lean_object* v___x_4984_; lean_object* v___f_4985_; lean_object* v___x_4986_; 
v___x_4984_ = lean_box(v___x_4978_);
v___f_4985_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___lam__1___boxed), 12, 3);
lean_closure_set(v___f_4985_, 0, v___f_4981_);
lean_closure_set(v___f_4985_, 1, v___y_4983_);
lean_closure_set(v___f_4985_, 2, v___x_4984_);
v___x_4986_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4985_, v_a_4968_, v_a_4969_, v_a_4970_, v_a_4971_, v_a_4972_, v_a_4973_, v_a_4974_, v_a_4975_);
return v___x_4986_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1___boxed(lean_object* v_x_4999_, lean_object* v_a_5000_, lean_object* v_a_5001_, lean_object* v_a_5002_, lean_object* v_a_5003_, lean_object* v_a_5004_, lean_object* v_a_5005_, lean_object* v_a_5006_, lean_object* v_a_5007_, lean_object* v_a_5008_){
_start:
{
lean_object* v_res_5009_; 
v_res_5009_ = lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______elabRules__Mathlib__Tactic__Ring__ring1__1(v_x_4999_, v_a_5000_, v_a_5001_, v_a_5002_, v_a_5003_, v_a_5004_, v_a_5005_, v_a_5006_, v_a_5007_);
lean_dec(v_a_5007_);
lean_dec_ref(v_a_5006_);
lean_dec(v_a_5005_);
lean_dec_ref(v_a_5004_);
lean_dec(v_a_5003_);
lean_dec_ref(v_a_5002_);
lean_dec(v_a_5001_);
lean_dec_ref(v_a_5000_);
return v_res_5009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1(lean_object* v_x_5028_, lean_object* v_a_5029_, lean_object* v_a_5030_){
_start:
{
lean_object* v___x_5031_; uint8_t v___x_5032_; 
v___x_5031_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_tacticRing1_x21___closed__1));
v___x_5032_ = l_Lean_Syntax_isOfKind(v_x_5028_, v___x_5031_);
if (v___x_5032_ == 0)
{
lean_object* v___x_5033_; lean_object* v___x_5034_; 
v___x_5033_ = lean_box(1);
v___x_5034_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5034_, 0, v___x_5033_);
lean_ctor_set(v___x_5034_, 1, v_a_5030_);
return v___x_5034_;
}
else
{
lean_object* v_ref_5035_; uint8_t v___x_5036_; lean_object* v___x_5037_; lean_object* v___x_5038_; lean_object* v___x_5039_; lean_object* v___x_5040_; lean_object* v___x_5041_; lean_object* v___x_5042_; lean_object* v___x_5043_; lean_object* v___x_5044_; lean_object* v___x_5045_; lean_object* v___x_5046_; 
v_ref_5035_ = lean_ctor_get(v_a_5029_, 5);
v___x_5036_ = 0;
v___x_5037_ = l_Lean_SourceInfo_fromRef(v_ref_5035_, v___x_5036_);
v___x_5038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__0));
v___x_5039_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__1));
lean_inc_n(v___x_5037_, 3);
v___x_5040_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5040_, 0, v___x_5037_);
lean_ctor_set(v___x_5040_, 1, v___x_5038_);
v___x_5041_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___closed__1));
v___x_5042_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Ring_ring1___closed__7));
v___x_5043_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5043_, 0, v___x_5037_);
lean_ctor_set(v___x_5043_, 1, v___x_5042_);
v___x_5044_ = l_Lean_Syntax_node1(v___x_5037_, v___x_5041_, v___x_5043_);
v___x_5045_ = l_Lean_Syntax_node2(v___x_5037_, v___x_5039_, v___x_5040_, v___x_5044_);
v___x_5046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5046_, 0, v___x_5045_);
lean_ctor_set(v___x_5046_, 1, v_a_5030_);
return v___x_5046_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1___boxed(lean_object* v_x_5047_, lean_object* v_a_5048_, lean_object* v_a_5049_){
_start:
{
lean_object* v_res_5050_; 
v_res_5050_ = lp_mathlib_Mathlib_Tactic_Ring___aux__Mathlib__Tactic__Ring__Basic______macroRules__Mathlib__Tactic__Ring__tacticRing1_x21__1(v_x_5047_, v_a_5048_, v_a_5049_);
lean_dec_ref(v_a_5048_);
return v_res_5050_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_Common(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring_Common(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Ring_rc_u2115 = _init_lp_mathlib_Mathlib_Tactic_Ring_rc_u2115();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Ring_rc_u2115);
res = lp_mathlib___private_Mathlib_Tactic_Ring_Basic_0__Mathlib_Tactic_Ring_initFn_00___x40_Mathlib_Tactic_Ring_Basic_3588287011____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Ring_ringCleanupRef);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring_Common(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Ring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
