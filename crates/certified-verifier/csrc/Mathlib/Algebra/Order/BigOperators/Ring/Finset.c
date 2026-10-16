// Lean compiler output
// Module: Mathlib.Algebra.Order.BigOperators.Ring.Finset
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Ring.Finset public import Mathlib.Algebra.Order.AbsoluteValue.Basic public import Mathlib.Algebra.Order.BigOperators.Group.Finset public import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset public import Mathlib.Algebra.Order.BigOperators.Ring.Multiset public import Mathlib.Tactic.Ring
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_Qq_Qq_assertDefEqQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lp_mathlib_Mathlib_Meta_Positivity_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lp_mathlib_Mathlib_Meta_Positivity_Strictness_toNonzero___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_Positivity_Strictness_toNonneg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "CommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(244, 39, 115, 67, 109, 198, 49, 224)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prod"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(247, 66, 46, 56, 151, 61, 191, 120)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "nonexhaustive match"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "CommMonoidWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 241, 165, 40, 39, 172, 206, 155)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Nontrivial"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__4_value),LEAN_SCALAR_PTR_LITERAL(122, 234, 164, 90, 175, 175, 198, 2)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "NoZeroDivisors"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__6_value),LEAN_SCALAR_PTR_LITERAL(48, 220, 238, 111, 161, 3, 30, 139)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__8_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__9_value),LEAN_SCALAR_PTR_LITERAL(14, 61, 8, 113, 227, 149, 226, 31)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "MulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toMulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__11_value),LEAN_SCALAR_PTR_LITERAL(175, 32, 159, 62, 158, 163, 7, 102)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__12_value),LEAN_SCALAR_PTR_LITERAL(173, 172, 244, 29, 179, 60, 222, 203)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "MonoidWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toMulZeroOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__14_value),LEAN_SCALAR_PTR_LITERAL(31, 173, 37, 182, 97, 174, 117, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__15_value),LEAN_SCALAR_PTR_LITERAL(49, 123, 33, 227, 129, 160, 155, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toMonoidWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 241, 165, 40, 39, 172, 206, 155)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__17_value),LEAN_SCALAR_PTR_LITERAL(95, 99, 11, 4, 77, 14, 157, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__8_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__19_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "toCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 241, 165, 40, 39, 172, 206, 155)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__21_value),LEAN_SCALAR_PTR_LITERAL(38, 131, 203, 49, 19, 16, 87, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Positivity"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "prod_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__23_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__24_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__25_value),LEAN_SCALAR_PTR_LITERAL(32, 187, 114, 53, 6, 76, 120, 30)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__26_value),LEAN_SCALAR_PTR_LITERAL(167, 97, 110, 223, 36, 255, 97, 148)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "i"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__28_value),LEAN_SCALAR_PTR_LITERAL(14, 215, 4, 153, 96, 18, 167, 14)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__30_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__32_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__33_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "SetLike"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instMembership"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__35_value),LEAN_SCALAR_PTR_LITERAL(146, 248, 10, 158, 176, 176, 178, 2)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__36_value),LEAN_SCALAR_PTR_LITERAL(142, 227, 62, 155, 191, 114, 172, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "instSetLike"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__38_value),LEAN_SCALAR_PTR_LITERAL(126, 101, 65, 213, 106, 37, 148, 147)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__44_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ZeroLEOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__46_value),LEAN_SCALAR_PTR_LITERAL(129, 104, 125, 173, 67, 61, 80, 90)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MulOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__48_value),LEAN_SCALAR_PTR_LITERAL(164, 62, 57, 171, 247, 8, 21, 201)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__49_value),LEAN_SCALAR_PTR_LITERAL(5, 117, 31, 141, 230, 176, 88, 11)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MulOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMulOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__51_value),LEAN_SCALAR_PTR_LITERAL(68, 11, 146, 104, 134, 210, 88, 211)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__52_value),LEAN_SCALAR_PTR_LITERAL(137, 54, 46, 26, 85, 32, 178, 134)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toMulOneClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__11_value),LEAN_SCALAR_PTR_LITERAL(175, 32, 159, 62, 158, 163, 7, 102)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__54_value),LEAN_SCALAR_PTR_LITERAL(132, 127, 127, 138, 109, 6, 220, 119)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__56_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "toLE"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__56_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__57_value),LEAN_SCALAR_PTR_LITERAL(142, 143, 218, 41, 228, 103, 236, 64)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__59_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toPreorder"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__59_value),LEAN_SCALAR_PTR_LITERAL(47, 196, 146, 225, 179, 207, 152, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__60_value),LEAN_SCALAR_PTR_LITERAL(3, 6, 195, 109, 53, 169, 118, 52)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "PosMulMono"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__62_value),LEAN_SCALAR_PTR_LITERAL(227, 80, 252, 46, 132, 209, 148, 142)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__63_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "prod_nonneg"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__64_value),LEAN_SCALAR_PTR_LITERAL(226, 196, 84, 20, 197, 48, 129, 240)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "PosMulStrictMono"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__66_value),LEAN_SCALAR_PTR_LITERAL(168, 94, 13, 218, 246, 146, 111, 175)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__67_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "prod_pos"};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__68_value),LEAN_SCALAR_PTR_LITERAL(65, 65, 151, 139, 17, 163, 235, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd = (const lean_object*)&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg(lean_object* v_l_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_4_; lean_object* v_mctx_5_; lean_object* v___x_6_; lean_object* v_fst_7_; lean_object* v_snd_8_; lean_object* v___x_9_; lean_object* v_cache_10_; lean_object* v_zetaDeltaFVarIds_11_; lean_object* v_postponed_12_; lean_object* v_diag_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_22_; 
v___x_4_ = lean_st_ref_get(v___y_2_);
v_mctx_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc_ref(v_mctx_5_);
lean_dec(v___x_4_);
v___x_6_ = lean_instantiate_level_mvars(v_mctx_5_, v_l_1_);
v_fst_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_fst_7_);
v_snd_8_ = lean_ctor_get(v___x_6_, 1);
lean_inc(v_snd_8_);
lean_dec_ref(v___x_6_);
v___x_9_ = lean_st_ref_take(v___y_2_);
v_cache_10_ = lean_ctor_get(v___x_9_, 1);
v_zetaDeltaFVarIds_11_ = lean_ctor_get(v___x_9_, 2);
v_postponed_12_ = lean_ctor_get(v___x_9_, 3);
v_diag_13_ = lean_ctor_get(v___x_9_, 4);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_22_ == 0)
{
lean_object* v_unused_23_; 
v_unused_23_ = lean_ctor_get(v___x_9_, 0);
lean_dec(v_unused_23_);
v___x_15_ = v___x_9_;
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_diag_13_);
lean_inc(v_postponed_12_);
lean_inc(v_zetaDeltaFVarIds_11_);
lean_inc(v_cache_10_);
lean_dec(v___x_9_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_18_; 
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v_fst_7_);
v___x_18_ = v___x_15_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v_fst_7_);
lean_ctor_set(v_reuseFailAlloc_21_, 1, v_cache_10_);
lean_ctor_set(v_reuseFailAlloc_21_, 2, v_zetaDeltaFVarIds_11_);
lean_ctor_set(v_reuseFailAlloc_21_, 3, v_postponed_12_);
lean_ctor_set(v_reuseFailAlloc_21_, 4, v_diag_13_);
v___x_18_ = v_reuseFailAlloc_21_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_st_ref_set(v___y_2_, v___x_18_);
v___x_20_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_20_, 0, v_snd_8_);
return v___x_20_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg___boxed(lean_object* v_l_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg(v_l_24_, v___y_25_);
lean_dec(v___y_25_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0(lean_object* v_l_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg(v_l_28_, v___y_30_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___boxed(lean_object* v_l_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0(v_l_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
lean_dec(v___y_37_);
lean_dec_ref(v___y_36_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(lean_object* v_e_42_, lean_object* v___y_43_){
_start:
{
uint8_t v___x_45_; 
v___x_45_ = l_Lean_Expr_hasMVar(v_e_42_);
if (v___x_45_ == 0)
{
lean_object* v___x_46_; 
v___x_46_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_46_, 0, v_e_42_);
return v___x_46_;
}
else
{
lean_object* v___x_47_; lean_object* v_mctx_48_; lean_object* v___x_49_; lean_object* v_fst_50_; lean_object* v_snd_51_; lean_object* v___x_52_; lean_object* v_cache_53_; lean_object* v_zetaDeltaFVarIds_54_; lean_object* v_postponed_55_; lean_object* v_diag_56_; lean_object* v___x_58_; uint8_t v_isShared_59_; uint8_t v_isSharedCheck_65_; 
v___x_47_ = lean_st_ref_get(v___y_43_);
v_mctx_48_ = lean_ctor_get(v___x_47_, 0);
lean_inc_ref(v_mctx_48_);
lean_dec(v___x_47_);
v___x_49_ = l_Lean_instantiateMVarsCore(v_mctx_48_, v_e_42_);
v_fst_50_ = lean_ctor_get(v___x_49_, 0);
lean_inc(v_fst_50_);
v_snd_51_ = lean_ctor_get(v___x_49_, 1);
lean_inc(v_snd_51_);
lean_dec_ref(v___x_49_);
v___x_52_ = lean_st_ref_take(v___y_43_);
v_cache_53_ = lean_ctor_get(v___x_52_, 1);
v_zetaDeltaFVarIds_54_ = lean_ctor_get(v___x_52_, 2);
v_postponed_55_ = lean_ctor_get(v___x_52_, 3);
v_diag_56_ = lean_ctor_get(v___x_52_, 4);
v_isSharedCheck_65_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_65_ == 0)
{
lean_object* v_unused_66_; 
v_unused_66_ = lean_ctor_get(v___x_52_, 0);
lean_dec(v_unused_66_);
v___x_58_ = v___x_52_;
v_isShared_59_ = v_isSharedCheck_65_;
goto v_resetjp_57_;
}
else
{
lean_inc(v_diag_56_);
lean_inc(v_postponed_55_);
lean_inc(v_zetaDeltaFVarIds_54_);
lean_inc(v_cache_53_);
lean_dec(v___x_52_);
v___x_58_ = lean_box(0);
v_isShared_59_ = v_isSharedCheck_65_;
goto v_resetjp_57_;
}
v_resetjp_57_:
{
lean_object* v___x_61_; 
if (v_isShared_59_ == 0)
{
lean_ctor_set(v___x_58_, 0, v_snd_51_);
v___x_61_ = v___x_58_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v_snd_51_);
lean_ctor_set(v_reuseFailAlloc_64_, 1, v_cache_53_);
lean_ctor_set(v_reuseFailAlloc_64_, 2, v_zetaDeltaFVarIds_54_);
lean_ctor_set(v_reuseFailAlloc_64_, 3, v_postponed_55_);
lean_ctor_set(v_reuseFailAlloc_64_, 4, v_diag_56_);
v___x_61_ = v_reuseFailAlloc_64_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_st_ref_set(v___y_43_, v___x_61_);
v___x_63_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_63_, 0, v_fst_50_);
return v___x_63_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg___boxed(lean_object* v_e_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_e_67_, v___y_68_);
lean_dec(v___y_68_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1(lean_object* v_e_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_e_71_, v___y_73_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___boxed(lean_object* v_e_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1(v_e_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(lean_object* v_k_85_, uint8_t v_allowLevelAssignments_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_86_, v_k_85_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
if (lean_obj_tag(v___x_92_) == 0)
{
lean_object* v_a_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_100_; 
v_a_93_ = lean_ctor_get(v___x_92_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v___x_92_);
if (v_isSharedCheck_100_ == 0)
{
v___x_95_ = v___x_92_;
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_a_93_);
lean_dec(v___x_92_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_98_; 
if (v_isShared_96_ == 0)
{
v___x_98_ = v___x_95_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_a_93_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
else
{
lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_108_; 
v_a_101_ = lean_ctor_get(v___x_92_, 0);
v_isSharedCheck_108_ = !lean_is_exclusive(v___x_92_);
if (v_isSharedCheck_108_ == 0)
{
v___x_103_ = v___x_92_;
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v___x_92_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_106_; 
if (v_isShared_104_ == 0)
{
v___x_106_ = v___x_103_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v_a_101_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg___boxed(lean_object* v_k_109_, lean_object* v_allowLevelAssignments_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_116_; lean_object* v_res_117_; 
v_allowLevelAssignments_boxed_116_ = lean_unbox(v_allowLevelAssignments_110_);
v_res_117_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v_k_109_, v_allowLevelAssignments_boxed_116_, v___y_111_, v___y_112_, v___y_113_, v___y_114_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2(lean_object* v_00_u03b1_118_, lean_object* v_k_119_, uint8_t v_allowLevelAssignments_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v_k_119_, v_allowLevelAssignments_120_, v___y_121_, v___y_122_, v___y_123_, v___y_124_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___boxed(lean_object* v_00_u03b1_127_, lean_object* v_k_128_, lean_object* v_allowLevelAssignments_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_135_; lean_object* v_res_136_; 
v_allowLevelAssignments_boxed_135_ = lean_unbox(v_allowLevelAssignments_129_);
v_res_136_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2(v_00_u03b1_127_, v_k_128_, v_allowLevelAssignments_boxed_135_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0(lean_object* v_u_147_, lean_object* v_00_u03b1_148_, lean_object* v_e_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = l_Lean_Meta_mkFreshLevelMVar(v___y_150_, v___y_151_, v___y_152_, v___y_153_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v_a_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; uint8_t v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
v_a_156_ = lean_ctor_get(v___x_155_, 0);
lean_inc_n(v_a_156_, 2);
lean_dec_ref_known(v___x_155_, 1);
v___x_157_ = l_Lean_Level_succ___override(v_a_156_);
v___x_158_ = l_Lean_Expr_sort___override(v___x_157_);
v___x_159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
v___x_160_ = 0;
v___x_161_ = lean_box(0);
v___x_162_ = l_Lean_Meta_mkFreshExprMVar(v___x_159_, v___x_160_, v___x_161_, v___y_150_, v___y_151_, v___y_152_, v___y_153_);
if (lean_obj_tag(v___x_162_) == 0)
{
lean_object* v_a_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v_a_163_ = lean_ctor_get(v___x_162_, 0);
lean_inc(v_a_163_);
lean_dec_ref_known(v___x_162_, 1);
v___x_164_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__1));
v___x_165_ = lean_box(0);
v___x_166_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_166_, 0, v_u_147_);
lean_ctor_set(v___x_166_, 1, v___x_165_);
lean_inc_ref(v___x_166_);
v___x_167_ = l_Lean_Expr_const___override(v___x_164_, v___x_166_);
lean_inc_ref(v_00_u03b1_148_);
v___x_168_ = l_Lean_Expr_app___override(v___x_167_, v_00_u03b1_148_);
v___x_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_169_, 0, v___x_168_);
v___x_170_ = l_Lean_Meta_mkFreshExprMVar(v___x_169_, v___x_160_, v___x_161_, v___y_150_, v___y_151_, v___y_152_, v___y_153_);
if (lean_obj_tag(v___x_170_) == 0)
{
lean_object* v_a_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v_a_171_ = lean_ctor_get(v___x_170_, 0);
lean_inc(v_a_171_);
lean_dec_ref_known(v___x_170_, 1);
v___x_172_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3));
lean_inc(v_a_156_);
v___x_173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_173_, 0, v_a_156_);
lean_ctor_set(v___x_173_, 1, v___x_165_);
v___x_174_ = l_Lean_Expr_const___override(v___x_172_, v___x_173_);
lean_inc(v_a_163_);
v___x_175_ = l_Lean_Expr_app___override(v___x_174_, v_a_163_);
v___x_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
v___x_177_ = l_Lean_Meta_mkFreshExprMVar(v___x_176_, v___x_160_, v___x_161_, v___y_150_, v___y_151_, v___y_152_, v___y_153_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v_a_178_; uint8_t v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_a_178_ = lean_ctor_get(v___x_177_, 0);
lean_inc(v_a_178_);
lean_dec_ref_known(v___x_177_, 1);
v___x_179_ = 0;
lean_inc_ref(v_00_u03b1_148_);
lean_inc(v_a_163_);
v___x_180_ = l_Lean_Expr_forallE___override(v___x_161_, v_a_163_, v_00_u03b1_148_, v___x_179_);
v___x_181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
v___x_182_ = l_Lean_Meta_mkFreshExprMVar(v___x_181_, v___x_160_, v___x_161_, v___y_150_, v___y_151_, v___y_152_, v___y_153_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v_a_183_; lean_object* v_keyedConfig_184_; uint8_t v_trackZetaDelta_185_; lean_object* v_zetaDeltaSet_186_; lean_object* v_lctx_187_; lean_object* v_localInstances_188_; lean_object* v_defEqCtx_x3f_189_; lean_object* v_synthPendingDepth_190_; lean_object* v_customCanUnfoldPredicate_x3f_191_; uint8_t v_univApprox_192_; uint8_t v_inTypeClassResolution_193_; uint8_t v_cacheInferType_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_256_; 
v_a_183_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_a_183_);
lean_dec_ref_known(v___x_182_, 1);
v_keyedConfig_184_ = lean_ctor_get(v___y_150_, 0);
v_trackZetaDelta_185_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*7);
v_zetaDeltaSet_186_ = lean_ctor_get(v___y_150_, 1);
v_lctx_187_ = lean_ctor_get(v___y_150_, 2);
v_localInstances_188_ = lean_ctor_get(v___y_150_, 3);
v_defEqCtx_x3f_189_ = lean_ctor_get(v___y_150_, 4);
v_synthPendingDepth_190_ = lean_ctor_get(v___y_150_, 5);
v_customCanUnfoldPredicate_x3f_191_ = lean_ctor_get(v___y_150_, 6);
v_univApprox_192_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_193_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*7 + 2);
v_cacheInferType_194_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*7 + 3);
v_isSharedCheck_256_ = !lean_is_exclusive(v___y_150_);
if (v_isSharedCheck_256_ == 0)
{
v___x_196_ = v___y_150_;
v_isShared_197_ = v_isSharedCheck_256_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_191_);
lean_inc(v_synthPendingDepth_190_);
lean_inc(v_defEqCtx_x3f_189_);
lean_inc(v_localInstances_188_);
lean_inc(v_lctx_187_);
lean_inc(v_zetaDeltaSet_186_);
lean_inc(v_keyedConfig_184_);
lean_dec(v___y_150_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_256_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; lean_object* v___x_207_; lean_object* v___x_209_; 
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__5));
lean_inc(v_a_156_);
v___x_199_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_199_, 0, v_a_156_);
lean_ctor_set(v___x_199_, 1, v___x_166_);
v___x_200_ = l_Lean_Expr_const___override(v___x_198_, v___x_199_);
lean_inc(v_a_163_);
v___x_201_ = l_Lean_Expr_app___override(v___x_200_, v_a_163_);
v___x_202_ = l_Lean_Expr_app___override(v___x_201_, v_00_u03b1_148_);
lean_inc(v_a_171_);
v___x_203_ = l_Lean_Expr_app___override(v___x_202_, v_a_171_);
lean_inc(v_a_178_);
v___x_204_ = l_Lean_Expr_app___override(v___x_203_, v_a_178_);
lean_inc(v_a_183_);
v___x_205_ = l_Lean_Expr_app___override(v___x_204_, v_a_183_);
v___x_206_ = 2;
v___x_207_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_206_, v_keyedConfig_184_);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 0, v___x_207_);
v___x_209_ = v___x_196_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v___x_207_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v_zetaDeltaSet_186_);
lean_ctor_set(v_reuseFailAlloc_255_, 2, v_lctx_187_);
lean_ctor_set(v_reuseFailAlloc_255_, 3, v_localInstances_188_);
lean_ctor_set(v_reuseFailAlloc_255_, 4, v_defEqCtx_x3f_189_);
lean_ctor_set(v_reuseFailAlloc_255_, 5, v_synthPendingDepth_190_);
lean_ctor_set(v_reuseFailAlloc_255_, 6, v_customCanUnfoldPredicate_x3f_191_);
lean_ctor_set_uint8(v_reuseFailAlloc_255_, sizeof(void*)*7, v_trackZetaDelta_185_);
lean_ctor_set_uint8(v_reuseFailAlloc_255_, sizeof(void*)*7 + 1, v_univApprox_192_);
lean_ctor_set_uint8(v_reuseFailAlloc_255_, sizeof(void*)*7 + 2, v_inTypeClassResolution_193_);
lean_ctor_set_uint8(v_reuseFailAlloc_255_, sizeof(void*)*7 + 3, v_cacheInferType_194_);
v___x_209_ = v_reuseFailAlloc_255_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
lean_object* v___x_210_; 
v___x_210_ = l_Lean_Meta_isExprDefEq(v___x_205_, v_e_149_, v___x_209_, v___y_151_, v___y_152_, v___y_153_);
lean_dec_ref(v___x_209_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v_a_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_246_; 
v_a_211_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_246_ == 0)
{
v___x_213_ = v___x_210_;
v_isShared_214_ = v_isSharedCheck_246_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_a_211_);
lean_dec(v___x_210_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_246_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
uint8_t v___x_215_; 
v___x_215_ = lean_unbox(v_a_211_);
if (v___x_215_ == 0)
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_222_; 
v___x_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_216_, 0, v_a_183_);
lean_ctor_set(v___x_216_, 1, v_a_211_);
v___x_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_217_, 0, v_a_178_);
lean_ctor_set(v___x_217_, 1, v___x_216_);
v___x_218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_218_, 0, v_a_171_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
v___x_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_219_, 0, v_a_163_);
lean_ctor_set(v___x_219_, 1, v___x_218_);
v___x_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_220_, 0, v_a_156_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
if (v_isShared_214_ == 0)
{
lean_ctor_set(v___x_213_, 0, v___x_220_);
v___x_222_ = v___x_213_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_220_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
else
{
lean_object* v___x_224_; lean_object* v_a_225_; lean_object* v___x_226_; lean_object* v_a_227_; lean_object* v___x_228_; lean_object* v_a_229_; lean_object* v___x_230_; lean_object* v_a_231_; lean_object* v___x_232_; lean_object* v_a_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_245_; 
lean_del_object(v___x_213_);
v___x_224_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__0___redArg(v_a_156_, v___y_151_);
v_a_225_ = lean_ctor_get(v___x_224_, 0);
lean_inc(v_a_225_);
lean_dec_ref(v___x_224_);
v___x_226_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_a_163_, v___y_151_);
v_a_227_ = lean_ctor_get(v___x_226_, 0);
lean_inc(v_a_227_);
lean_dec_ref(v___x_226_);
v___x_228_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_a_171_, v___y_151_);
v_a_229_ = lean_ctor_get(v___x_228_, 0);
lean_inc(v_a_229_);
lean_dec_ref(v___x_228_);
v___x_230_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_a_178_, v___y_151_);
v_a_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_a_231_);
lean_dec_ref(v___x_230_);
v___x_232_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__1___redArg(v_a_183_, v___y_151_);
v_a_233_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_245_ == 0)
{
v___x_235_ = v___x_232_;
v_isShared_236_ = v_isSharedCheck_245_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_a_233_);
lean_dec(v___x_232_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_245_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_243_; 
v___x_237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_237_, 0, v_a_233_);
lean_ctor_set(v___x_237_, 1, v_a_211_);
v___x_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_238_, 0, v_a_231_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_239_, 0, v_a_229_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_240_, 0, v_a_227_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v_a_225_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 0, v___x_241_);
v___x_243_ = v___x_235_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_241_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
}
else
{
lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_254_; 
lean_dec(v_a_183_);
lean_dec(v_a_178_);
lean_dec(v_a_171_);
lean_dec(v_a_163_);
lean_dec(v_a_156_);
v_a_247_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_254_ == 0)
{
v___x_249_ = v___x_210_;
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_210_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_252_; 
if (v_isShared_250_ == 0)
{
v___x_252_ = v___x_249_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v_a_247_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
}
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec(v_a_178_);
lean_dec(v_a_171_);
lean_dec_ref_known(v___x_166_, 2);
lean_dec(v_a_163_);
lean_dec(v_a_156_);
lean_dec_ref(v___y_150_);
lean_dec_ref(v_e_149_);
lean_dec_ref(v_00_u03b1_148_);
v_a_257_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_182_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_182_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_a_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
else
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
lean_dec(v_a_171_);
lean_dec_ref_known(v___x_166_, 2);
lean_dec(v_a_163_);
lean_dec(v_a_156_);
lean_dec_ref(v___y_150_);
lean_dec_ref(v_e_149_);
lean_dec_ref(v_00_u03b1_148_);
v_a_265_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_177_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_177_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_a_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
}
else
{
lean_object* v_a_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_280_; 
lean_dec_ref_known(v___x_166_, 2);
lean_dec(v_a_163_);
lean_dec(v_a_156_);
lean_dec_ref(v___y_150_);
lean_dec_ref(v_e_149_);
lean_dec_ref(v_00_u03b1_148_);
v_a_273_ = lean_ctor_get(v___x_170_, 0);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_280_ == 0)
{
v___x_275_ = v___x_170_;
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_a_273_);
lean_dec(v___x_170_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_278_; 
if (v_isShared_276_ == 0)
{
v___x_278_ = v___x_275_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v_a_273_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
else
{
lean_object* v_a_281_; lean_object* v___x_283_; uint8_t v_isShared_284_; uint8_t v_isSharedCheck_288_; 
lean_dec(v_a_156_);
lean_dec_ref(v___y_150_);
lean_dec_ref(v_e_149_);
lean_dec_ref(v_00_u03b1_148_);
lean_dec(v_u_147_);
v_a_281_ = lean_ctor_get(v___x_162_, 0);
v_isSharedCheck_288_ = !lean_is_exclusive(v___x_162_);
if (v_isSharedCheck_288_ == 0)
{
v___x_283_ = v___x_162_;
v_isShared_284_ = v_isSharedCheck_288_;
goto v_resetjp_282_;
}
else
{
lean_inc(v_a_281_);
lean_dec(v___x_162_);
v___x_283_ = lean_box(0);
v_isShared_284_ = v_isSharedCheck_288_;
goto v_resetjp_282_;
}
v_resetjp_282_:
{
lean_object* v___x_286_; 
if (v_isShared_284_ == 0)
{
v___x_286_ = v___x_283_;
goto v_reusejp_285_;
}
else
{
lean_object* v_reuseFailAlloc_287_; 
v_reuseFailAlloc_287_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_287_, 0, v_a_281_);
v___x_286_ = v_reuseFailAlloc_287_;
goto v_reusejp_285_;
}
v_reusejp_285_:
{
return v___x_286_;
}
}
}
}
else
{
lean_object* v_a_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_296_; 
lean_dec_ref(v___y_150_);
lean_dec_ref(v_e_149_);
lean_dec_ref(v_00_u03b1_148_);
lean_dec(v_u_147_);
v_a_289_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_296_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_296_ == 0)
{
v___x_291_ = v___x_155_;
v_isShared_292_ = v_isSharedCheck_296_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_a_289_);
lean_dec(v___x_155_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_296_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v___x_294_; 
if (v_isShared_292_ == 0)
{
v___x_294_ = v___x_291_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v_a_289_);
v___x_294_ = v_reuseFailAlloc_295_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
return v___x_294_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___boxed(lean_object* v_u_297_, lean_object* v_00_u03b1_298_, lean_object* v_e_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0(v_u_297_, v_00_u03b1_298_, v_e_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1(uint8_t v___x_306_, lean_object* v_z_u03b1_307_, lean_object* v___x_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
lean_object* v_keyedConfig_314_; uint8_t v_trackZetaDelta_315_; lean_object* v_zetaDeltaSet_316_; lean_object* v_lctx_317_; lean_object* v_localInstances_318_; lean_object* v_defEqCtx_x3f_319_; lean_object* v_synthPendingDepth_320_; lean_object* v_customCanUnfoldPredicate_x3f_321_; uint8_t v_univApprox_322_; uint8_t v_inTypeClassResolution_323_; uint8_t v_cacheInferType_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_333_; 
v_keyedConfig_314_ = lean_ctor_get(v___y_309_, 0);
v_trackZetaDelta_315_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*7);
v_zetaDeltaSet_316_ = lean_ctor_get(v___y_309_, 1);
v_lctx_317_ = lean_ctor_get(v___y_309_, 2);
v_localInstances_318_ = lean_ctor_get(v___y_309_, 3);
v_defEqCtx_x3f_319_ = lean_ctor_get(v___y_309_, 4);
v_synthPendingDepth_320_ = lean_ctor_get(v___y_309_, 5);
v_customCanUnfoldPredicate_x3f_321_ = lean_ctor_get(v___y_309_, 6);
v_univApprox_322_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_323_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*7 + 2);
v_cacheInferType_324_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*7 + 3);
v_isSharedCheck_333_ = !lean_is_exclusive(v___y_309_);
if (v_isSharedCheck_333_ == 0)
{
v___x_326_ = v___y_309_;
v_isShared_327_ = v_isSharedCheck_333_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_321_);
lean_inc(v_synthPendingDepth_320_);
lean_inc(v_defEqCtx_x3f_319_);
lean_inc(v_localInstances_318_);
lean_inc(v_lctx_317_);
lean_inc(v_zetaDeltaSet_316_);
lean_inc(v_keyedConfig_314_);
lean_dec(v___y_309_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_333_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_328_; lean_object* v___x_330_; 
v___x_328_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_306_, v_keyedConfig_314_);
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_328_);
v___x_330_ = v___x_326_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v_zetaDeltaSet_316_);
lean_ctor_set(v_reuseFailAlloc_332_, 2, v_lctx_317_);
lean_ctor_set(v_reuseFailAlloc_332_, 3, v_localInstances_318_);
lean_ctor_set(v_reuseFailAlloc_332_, 4, v_defEqCtx_x3f_319_);
lean_ctor_set(v_reuseFailAlloc_332_, 5, v_synthPendingDepth_320_);
lean_ctor_set(v_reuseFailAlloc_332_, 6, v_customCanUnfoldPredicate_x3f_321_);
lean_ctor_set_uint8(v_reuseFailAlloc_332_, sizeof(void*)*7, v_trackZetaDelta_315_);
lean_ctor_set_uint8(v_reuseFailAlloc_332_, sizeof(void*)*7 + 1, v_univApprox_322_);
lean_ctor_set_uint8(v_reuseFailAlloc_332_, sizeof(void*)*7 + 2, v_inTypeClassResolution_323_);
lean_ctor_set_uint8(v_reuseFailAlloc_332_, sizeof(void*)*7 + 3, v_cacheInferType_324_);
v___x_330_ = v_reuseFailAlloc_332_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
lean_object* v___x_331_; 
v___x_331_ = lp_Qq_Qq_assertDefEqQ___redArg(v_z_u03b1_307_, v___x_308_, v___x_330_, v___y_310_, v___y_311_, v___y_312_);
lean_dec_ref(v___x_330_);
return v___x_331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1___boxed(lean_object* v___x_334_, lean_object* v_z_u03b1_335_, lean_object* v___x_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
uint8_t v___x_26105__boxed_342_; lean_object* v_res_343_; 
v___x_26105__boxed_342_ = lean_unbox(v___x_334_);
v_res_343_ = lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1(v___x_26105__boxed_342_, v_z_u03b1_335_, v___x_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
lean_dec(v___y_340_);
lean_dec_ref(v___y_339_);
lean_dec(v___y_338_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2(uint8_t v___x_344_, lean_object* v_fst_345_, lean_object* v___x_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v_keyedConfig_352_; uint8_t v_trackZetaDelta_353_; lean_object* v_zetaDeltaSet_354_; lean_object* v_lctx_355_; lean_object* v_localInstances_356_; lean_object* v_defEqCtx_x3f_357_; lean_object* v_synthPendingDepth_358_; lean_object* v_customCanUnfoldPredicate_x3f_359_; uint8_t v_univApprox_360_; uint8_t v_inTypeClassResolution_361_; uint8_t v_cacheInferType_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_371_; 
v_keyedConfig_352_ = lean_ctor_get(v___y_347_, 0);
v_trackZetaDelta_353_ = lean_ctor_get_uint8(v___y_347_, sizeof(void*)*7);
v_zetaDeltaSet_354_ = lean_ctor_get(v___y_347_, 1);
v_lctx_355_ = lean_ctor_get(v___y_347_, 2);
v_localInstances_356_ = lean_ctor_get(v___y_347_, 3);
v_defEqCtx_x3f_357_ = lean_ctor_get(v___y_347_, 4);
v_synthPendingDepth_358_ = lean_ctor_get(v___y_347_, 5);
v_customCanUnfoldPredicate_x3f_359_ = lean_ctor_get(v___y_347_, 6);
v_univApprox_360_ = lean_ctor_get_uint8(v___y_347_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_361_ = lean_ctor_get_uint8(v___y_347_, sizeof(void*)*7 + 2);
v_cacheInferType_362_ = lean_ctor_get_uint8(v___y_347_, sizeof(void*)*7 + 3);
v_isSharedCheck_371_ = !lean_is_exclusive(v___y_347_);
if (v_isSharedCheck_371_ == 0)
{
v___x_364_ = v___y_347_;
v_isShared_365_ = v_isSharedCheck_371_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_359_);
lean_inc(v_synthPendingDepth_358_);
lean_inc(v_defEqCtx_x3f_357_);
lean_inc(v_localInstances_356_);
lean_inc(v_lctx_355_);
lean_inc(v_zetaDeltaSet_354_);
lean_inc(v_keyedConfig_352_);
lean_dec(v___y_347_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_371_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_366_; lean_object* v___x_368_; 
v___x_366_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_344_, v_keyedConfig_352_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 0, v___x_366_);
v___x_368_ = v___x_364_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v___x_366_);
lean_ctor_set(v_reuseFailAlloc_370_, 1, v_zetaDeltaSet_354_);
lean_ctor_set(v_reuseFailAlloc_370_, 2, v_lctx_355_);
lean_ctor_set(v_reuseFailAlloc_370_, 3, v_localInstances_356_);
lean_ctor_set(v_reuseFailAlloc_370_, 4, v_defEqCtx_x3f_357_);
lean_ctor_set(v_reuseFailAlloc_370_, 5, v_synthPendingDepth_358_);
lean_ctor_set(v_reuseFailAlloc_370_, 6, v_customCanUnfoldPredicate_x3f_359_);
lean_ctor_set_uint8(v_reuseFailAlloc_370_, sizeof(void*)*7, v_trackZetaDelta_353_);
lean_ctor_set_uint8(v_reuseFailAlloc_370_, sizeof(void*)*7 + 1, v_univApprox_360_);
lean_ctor_set_uint8(v_reuseFailAlloc_370_, sizeof(void*)*7 + 2, v_inTypeClassResolution_361_);
lean_ctor_set_uint8(v_reuseFailAlloc_370_, sizeof(void*)*7 + 3, v_cacheInferType_362_);
v___x_368_ = v_reuseFailAlloc_370_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
lean_object* v___x_369_; 
v___x_369_ = lp_Qq_Qq_assertDefEqQ___redArg(v_fst_345_, v___x_346_, v___x_368_, v___y_348_, v___y_349_, v___y_350_);
lean_dec_ref(v___x_368_);
return v___x_369_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2___boxed(lean_object* v___x_372_, lean_object* v_fst_373_, lean_object* v___x_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
uint8_t v___x_26144__boxed_380_; lean_object* v_res_381_; 
v___x_26144__boxed_380_ = lean_unbox(v___x_372_);
v_res_381_ = lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2(v___x_26144__boxed_380_, v_fst_373_, v___x_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3(lean_object* v_msgData_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v___x_388_; lean_object* v_env_389_; lean_object* v___x_390_; lean_object* v_mctx_391_; lean_object* v_lctx_392_; lean_object* v_options_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_388_ = lean_st_ref_get(v___y_386_);
v_env_389_ = lean_ctor_get(v___x_388_, 0);
lean_inc_ref(v_env_389_);
lean_dec(v___x_388_);
v___x_390_ = lean_st_ref_get(v___y_384_);
v_mctx_391_ = lean_ctor_get(v___x_390_, 0);
lean_inc_ref(v_mctx_391_);
lean_dec(v___x_390_);
v_lctx_392_ = lean_ctor_get(v___y_383_, 2);
v_options_393_ = lean_ctor_get(v___y_385_, 2);
lean_inc_ref(v_options_393_);
lean_inc_ref(v_lctx_392_);
v___x_394_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_394_, 0, v_env_389_);
lean_ctor_set(v___x_394_, 1, v_mctx_391_);
lean_ctor_set(v___x_394_, 2, v_lctx_392_);
lean_ctor_set(v___x_394_, 3, v_options_393_);
v___x_395_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v_msgData_382_);
v___x_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3___boxed(lean_object* v_msgData_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3(v_msgData_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(lean_object* v_msg_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_){
_start:
{
lean_object* v_ref_410_; lean_object* v___x_411_; lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_420_; 
v_ref_410_ = lean_ctor_get(v___y_407_, 5);
v___x_411_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3_spec__3(v_msg_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_);
v_a_412_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_420_ == 0)
{
v___x_414_ = v___x_411_;
v_isShared_415_ = v_isSharedCheck_420_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_411_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_420_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_416_; lean_object* v___x_418_; 
lean_inc(v_ref_410_);
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v_ref_410_);
lean_ctor_set(v___x_416_, 1, v_a_412_);
if (v_isShared_415_ == 0)
{
lean_ctor_set_tag(v___x_414_, 1);
lean_ctor_set(v___x_414_, 0, v___x_416_);
v___x_418_ = v___x_414_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_416_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg___boxed(lean_object* v_msg_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(v_msg_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
lean_dec(v___y_423_);
lean_dec_ref(v___y_422_);
return v_res_427_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1(void){
_start:
{
lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_429_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__0));
v___x_430_ = l_Lean_stringToMessageData(v___x_429_);
return v___x_430_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40(void){
_start:
{
lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_496_ = lean_unsigned_to_nat(0u);
v___x_497_ = l_Lean_Expr_bvar___override(v___x_496_);
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = lean_unsigned_to_nat(1u);
v___x_499_ = l_Lean_Expr_bvar___override(v___x_498_);
return v___x_499_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42(void){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; 
v___x_500_ = lean_box(0);
v___x_501_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__41);
v___x_502_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_501_);
lean_ctor_set(v___x_502_, 1, v___x_500_);
return v___x_502_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43(void){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_503_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__42);
v___x_504_ = lean_array_mk(v___x_503_);
return v___x_504_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_506_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__44));
v___x_507_ = l_Lean_stringToMessageData(v___x_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7(lean_object* v_u_549_, lean_object* v_00_u03b1_550_, lean_object* v_z_u03b1_551_, lean_object* v_p_u03b1_x3f_552_, lean_object* v_e_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
if (lean_obj_tag(v_p_u03b1_x3f_552_) == 0)
{
lean_object* v___x_559_; lean_object* v___x_560_; 
lean_dec_ref(v_e_553_);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v___x_559_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_559_, 0, v_p_u03b1_x3f_552_);
v___x_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_560_, 0, v___x_559_);
return v___x_560_;
}
else
{
lean_object* v_val_561_; lean_object* v___f_562_; uint8_t v___x_563_; lean_object* v___x_564_; 
v_val_561_ = lean_ctor_get(v_p_u03b1_x3f_552_, 0);
lean_inc_ref(v_00_u03b1_550_);
lean_inc(v_u_549_);
v___f_562_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___boxed), 8, 3);
lean_closure_set(v___f_562_, 0, v_u_549_);
lean_closure_set(v___f_562_, 1, v_00_u03b1_550_);
lean_closure_set(v___f_562_, 2, v_e_553_);
v___x_563_ = 0;
v___x_564_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_562_, v___x_563_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v_a_565_; lean_object* v_snd_566_; lean_object* v_snd_567_; lean_object* v_snd_568_; lean_object* v_snd_569_; lean_object* v_snd_570_; uint8_t v___x_571_; 
v_a_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_565_);
lean_dec_ref_known(v___x_564_, 1);
v_snd_566_ = lean_ctor_get(v_a_565_, 1);
lean_inc(v_snd_566_);
v_snd_567_ = lean_ctor_get(v_snd_566_, 1);
lean_inc(v_snd_567_);
v_snd_568_ = lean_ctor_get(v_snd_567_, 1);
lean_inc(v_snd_568_);
v_snd_569_ = lean_ctor_get(v_snd_568_, 1);
lean_inc(v_snd_569_);
v_snd_570_ = lean_ctor_get(v_snd_569_, 1);
lean_inc(v_snd_570_);
v___x_571_ = lean_unbox(v_snd_570_);
if (v___x_571_ == 0)
{
lean_object* v___x_572_; lean_object* v___x_573_; 
lean_dec(v_snd_570_);
lean_dec(v_snd_569_);
lean_dec(v_snd_568_);
lean_dec(v_snd_567_);
lean_dec(v_snd_566_);
lean_dec(v_a_565_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v___x_572_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__1);
v___x_573_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(v___x_572_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
return v___x_573_;
}
else
{
lean_object* v_fst_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_1157_; 
v_fst_574_ = lean_ctor_get(v_a_565_, 0);
v_isSharedCheck_1157_ = !lean_is_exclusive(v_a_565_);
if (v_isSharedCheck_1157_ == 0)
{
lean_object* v_unused_1158_; 
v_unused_1158_ = lean_ctor_get(v_a_565_, 1);
lean_dec(v_unused_1158_);
v___x_576_ = v_a_565_;
v_isShared_577_ = v_isSharedCheck_1157_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_fst_574_);
lean_dec(v_a_565_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_1157_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v_fst_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_1155_; 
v_fst_578_ = lean_ctor_get(v_snd_566_, 0);
v_isSharedCheck_1155_ = !lean_is_exclusive(v_snd_566_);
if (v_isSharedCheck_1155_ == 0)
{
lean_object* v_unused_1156_; 
v_unused_1156_ = lean_ctor_get(v_snd_566_, 1);
lean_dec(v_unused_1156_);
v___x_580_ = v_snd_566_;
v_isShared_581_ = v_isSharedCheck_1155_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_fst_578_);
lean_dec(v_snd_566_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_1155_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v_fst_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_1153_; 
v_fst_582_ = lean_ctor_get(v_snd_567_, 0);
v_isSharedCheck_1153_ = !lean_is_exclusive(v_snd_567_);
if (v_isSharedCheck_1153_ == 0)
{
lean_object* v_unused_1154_; 
v_unused_1154_ = lean_ctor_get(v_snd_567_, 1);
lean_dec(v_unused_1154_);
v___x_584_ = v_snd_567_;
v_isShared_585_ = v_isSharedCheck_1153_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_fst_582_);
lean_dec(v_snd_567_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_1153_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v_fst_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_1151_; 
v_fst_586_ = lean_ctor_get(v_snd_568_, 0);
v_isSharedCheck_1151_ = !lean_is_exclusive(v_snd_568_);
if (v_isSharedCheck_1151_ == 0)
{
lean_object* v_unused_1152_; 
v_unused_1152_ = lean_ctor_get(v_snd_568_, 1);
lean_dec(v_unused_1152_);
v___x_588_ = v_snd_568_;
v_isShared_589_ = v_isSharedCheck_1151_;
goto v_resetjp_587_;
}
else
{
lean_inc(v_fst_586_);
lean_dec(v_snd_568_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_1151_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v_fst_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_1149_; 
v_fst_590_ = lean_ctor_get(v_snd_569_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v_snd_569_);
if (v_isSharedCheck_1149_ == 0)
{
lean_object* v_unused_1150_; 
v_unused_1150_ = lean_ctor_get(v_snd_569_, 1);
lean_dec(v_unused_1150_);
v___x_592_ = v_snd_569_;
v_isShared_593_ = v_isSharedCheck_1149_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_fst_590_);
lean_dec(v_snd_569_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_1149_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
uint8_t v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_594_ = 2;
v___x_595_ = lean_box(0);
lean_inc(v_fst_578_);
v___x_596_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v_fst_578_, v___x_594_, v___x_595_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_a_597_);
lean_dec_ref_known(v___x_596_, 1);
v___x_598_ = lean_unsigned_to_nat(1u);
v___x_599_ = lean_mk_empty_array_with_capacity(v___x_598_);
v___x_600_ = lean_array_push(v___x_599_, v_a_597_);
lean_inc(v_fst_590_);
v___x_601_ = l_Lean_Expr_betaRev(v_fst_590_, v___x_600_, v___x_563_, v___x_563_);
lean_inc_ref(v___x_601_);
lean_inc_ref(v_p_u03b1_x3f_552_);
lean_inc_ref(v_z_u03b1_551_);
lean_inc_ref(v_00_u03b1_550_);
lean_inc(v_u_549_);
v___x_602_ = lp_mathlib_Mathlib_Meta_Positivity_core(v_u_549_, v_00_u03b1_550_, v_z_u03b1_551_, v_p_u03b1_x3f_552_, v___x_601_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_object* v_a_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_607_; 
v_a_603_ = lean_ctor_get(v___x_602_, 0);
lean_inc(v_a_603_);
lean_dec_ref_known(v___x_602_, 1);
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__3));
v___x_605_ = lean_box(0);
lean_inc(v_u_549_);
if (v_isShared_593_ == 0)
{
lean_ctor_set_tag(v___x_592_, 1);
lean_ctor_set(v___x_592_, 1, v___x_605_);
lean_ctor_set(v___x_592_, 0, v_u_549_);
v___x_607_ = v___x_592_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v_u_549_);
lean_ctor_set(v_reuseFailAlloc_1140_, 1, v___x_605_);
v___x_607_ = v_reuseFailAlloc_1140_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
lean_inc_ref(v___x_607_);
v___x_608_ = l_Lean_Expr_const___override(v___x_604_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_609_ = l_Lean_Expr_app___override(v___x_608_, v_00_u03b1_550_);
v___x_610_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_609_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_610_) == 0)
{
lean_object* v_a_611_; lean_object* v___y_613_; lean_object* v___y_614_; lean_object* v___y_615_; lean_object* v___y_616_; lean_object* v_a_617_; lean_object* v___y_762_; lean_object* v___y_763_; lean_object* v___y_764_; lean_object* v___y_765_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___y_781_; lean_object* v___y_782_; 
v_a_611_ = lean_ctor_get(v___x_610_, 0);
lean_inc(v_a_611_);
lean_dec_ref_known(v___x_610_, 1);
if (lean_obj_tag(v_a_603_) == 0)
{
lean_object* v_pf_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; 
v_pf_947_ = lean_ctor_get(v_a_603_, 1);
v___x_948_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__47));
lean_inc_ref_n(v___x_607_, 8);
v___x_949_ = l_Lean_Expr_const___override(v___x_948_, v___x_607_);
lean_inc_ref_n(v_00_u03b1_550_, 8);
v___x_950_ = l_Lean_Expr_app___override(v___x_949_, v_00_u03b1_550_);
lean_inc_ref(v_z_u03b1_551_);
v___x_951_ = l_Lean_Expr_app___override(v___x_950_, v_z_u03b1_551_);
v___x_952_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50));
v___x_953_ = l_Lean_Expr_const___override(v___x_952_, v___x_607_);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v_00_u03b1_550_);
v___x_955_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53));
v___x_956_ = l_Lean_Expr_const___override(v___x_955_, v___x_607_);
v___x_957_ = l_Lean_Expr_app___override(v___x_956_, v_00_u03b1_550_);
v___x_958_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55));
v___x_959_ = l_Lean_Expr_const___override(v___x_958_, v___x_607_);
v___x_960_ = l_Lean_Expr_app___override(v___x_959_, v_00_u03b1_550_);
v___x_961_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16));
v___x_962_ = l_Lean_Expr_const___override(v___x_961_, v___x_607_);
v___x_963_ = l_Lean_Expr_app___override(v___x_962_, v_00_u03b1_550_);
v___x_964_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18));
v___x_965_ = l_Lean_Expr_const___override(v___x_964_, v___x_607_);
v___x_966_ = l_Lean_Expr_app___override(v___x_965_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_967_ = l_Lean_Expr_app___override(v___x_966_, v_a_611_);
v___x_968_ = l_Lean_Expr_app___override(v___x_963_, v___x_967_);
lean_inc_ref(v___x_968_);
v___x_969_ = l_Lean_Expr_app___override(v___x_960_, v___x_968_);
v___x_970_ = l_Lean_Expr_app___override(v___x_957_, v___x_969_);
v___x_971_ = l_Lean_Expr_app___override(v___x_954_, v___x_970_);
v___x_972_ = l_Lean_Expr_app___override(v___x_951_, v___x_971_);
v___x_973_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58));
v___x_974_ = l_Lean_Expr_const___override(v___x_973_, v___x_607_);
v___x_975_ = l_Lean_Expr_app___override(v___x_974_, v_00_u03b1_550_);
v___x_976_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61));
v___x_977_ = l_Lean_Expr_const___override(v___x_976_, v___x_607_);
v___x_978_ = l_Lean_Expr_app___override(v___x_977_, v_00_u03b1_550_);
lean_inc(v_val_561_);
v___x_979_ = l_Lean_Expr_app___override(v___x_978_, v_val_561_);
lean_inc_ref(v___x_979_);
v___x_980_ = l_Lean_Expr_app___override(v___x_975_, v___x_979_);
v___x_981_ = l_Lean_Expr_app___override(v___x_972_, v___x_980_);
v___x_982_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_981_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_982_) == 0)
{
lean_object* v_a_983_; 
v_a_983_ = lean_ctor_get(v___x_982_, 0);
lean_inc(v_a_983_);
lean_dec_ref_known(v___x_982_, 1);
if (lean_obj_tag(v_a_983_) == 1)
{
lean_object* v_a_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; 
v_a_984_ = lean_ctor_get(v_a_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v_a_983_, 1);
v___x_985_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__67));
lean_inc_ref_n(v___x_607_, 3);
v___x_986_ = l_Lean_Expr_const___override(v___x_985_, v___x_607_);
lean_inc_ref_n(v_00_u03b1_550_, 3);
v___x_987_ = l_Lean_Expr_app___override(v___x_986_, v_00_u03b1_550_);
v___x_988_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10));
v___x_989_ = l_Lean_Expr_const___override(v___x_988_, v___x_607_);
v___x_990_ = l_Lean_Expr_app___override(v___x_989_, v_00_u03b1_550_);
v___x_991_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13));
v___x_992_ = l_Lean_Expr_const___override(v___x_991_, v___x_607_);
v___x_993_ = l_Lean_Expr_app___override(v___x_992_, v_00_u03b1_550_);
v___x_994_ = l_Lean_Expr_app___override(v___x_993_, v___x_968_);
lean_inc_ref(v___x_994_);
v___x_995_ = l_Lean_Expr_app___override(v___x_990_, v___x_994_);
v___x_996_ = l_Lean_Expr_app___override(v___x_987_, v___x_995_);
lean_inc_ref(v_z_u03b1_551_);
v___x_997_ = l_Lean_Expr_app___override(v___x_996_, v_z_u03b1_551_);
v___x_998_ = l_Lean_Expr_app___override(v___x_997_, v___x_979_);
v___x_999_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_998_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_999_) == 0)
{
lean_object* v_a_1000_; 
v_a_1000_ = lean_ctor_get(v___x_999_, 0);
lean_inc(v_a_1000_);
lean_dec_ref_known(v___x_999_, 1);
if (lean_obj_tag(v_a_1000_) == 1)
{
lean_object* v_a_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; 
v_a_1001_ = lean_ctor_get(v_a_1000_, 0);
lean_inc(v_a_1001_);
lean_dec_ref_known(v_a_1000_, 1);
v___x_1002_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__5));
lean_inc_ref(v___x_607_);
v___x_1003_ = l_Lean_Expr_const___override(v___x_1002_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_1004_ = l_Lean_Expr_app___override(v___x_1003_, v_00_u03b1_550_);
v___x_1005_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_1004_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_1005_) == 0)
{
lean_object* v_a_1006_; 
v_a_1006_ = lean_ctor_get(v___x_1005_, 0);
lean_inc(v_a_1006_);
lean_dec_ref_known(v___x_1005_, 1);
if (lean_obj_tag(v_a_1006_) == 1)
{
lean_object* v___x_1008_; uint8_t v_isShared_1009_; uint8_t v_isSharedCheck_1105_; 
lean_inc_ref(v_pf_947_);
lean_inc(v_val_561_);
lean_dec_ref(v___x_601_);
lean_del_object(v___x_588_);
lean_del_object(v___x_584_);
lean_del_object(v___x_580_);
lean_del_object(v___x_576_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec(v_u_549_);
v_isSharedCheck_1105_ = !lean_is_exclusive(v_a_603_);
if (v_isSharedCheck_1105_ == 0)
{
lean_object* v_unused_1106_; lean_object* v_unused_1107_; 
v_unused_1106_ = lean_ctor_get(v_a_603_, 1);
lean_dec(v_unused_1106_);
v_unused_1107_ = lean_ctor_get(v_a_603_, 0);
lean_dec(v_unused_1107_);
v___x_1008_ = v_a_603_;
v_isShared_1009_ = v_isSharedCheck_1105_;
goto v_resetjp_1007_;
}
else
{
lean_dec(v_a_603_);
v___x_1008_ = lean_box(0);
v_isShared_1009_ = v_isSharedCheck_1105_;
goto v_resetjp_1007_;
}
v_resetjp_1007_:
{
lean_object* v_a_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; uint8_t v___x_1015_; lean_object* v___x_1016_; lean_object* v___f_1017_; lean_object* v___x_1018_; 
v_a_1010_ = lean_ctor_get(v_a_1006_, 0);
lean_inc(v_a_1010_);
lean_dec_ref_known(v_a_1006_, 1);
v___x_1011_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20));
lean_inc_ref(v___x_607_);
v___x_1012_ = l_Lean_Expr_const___override(v___x_1011_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_1013_ = l_Lean_Expr_app___override(v___x_1012_, v_00_u03b1_550_);
v___x_1014_ = l_Lean_Expr_app___override(v___x_1013_, v___x_994_);
v___x_1015_ = 1;
v___x_1016_ = lean_box(v___x_1015_);
v___f_1017_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1___boxed), 8, 3);
lean_closure_set(v___f_1017_, 0, v___x_1016_);
lean_closure_set(v___f_1017_, 1, v_z_u03b1_551_);
lean_closure_set(v___f_1017_, 2, v___x_1014_);
v___x_1018_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_1017_, v___x_563_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_1018_) == 0)
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___f_1024_; lean_object* v___x_1025_; 
lean_dec_ref_known(v___x_1018_, 1);
v___x_1019_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22));
lean_inc_ref(v___x_607_);
v___x_1020_ = l_Lean_Expr_const___override(v___x_1019_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1021_, v_a_611_);
v___x_1023_ = lean_box(v___x_1015_);
v___f_1024_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2___boxed), 8, 3);
lean_closure_set(v___f_1024_, 0, v___x_1023_);
lean_closure_set(v___f_1024_, 1, v_fst_582_);
lean_closure_set(v___f_1024_, 2, v___x_1022_);
v___x_1025_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_1024_, v___x_563_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_1025_) == 0)
{
uint8_t v___x_1026_; uint8_t v___x_1027_; uint8_t v___x_1028_; lean_object* v___x_1029_; 
lean_dec_ref_known(v___x_1025_, 1);
v___x_1026_ = 0;
v___x_1027_ = lean_unbox(v_snd_570_);
v___x_1028_ = lean_unbox(v_snd_570_);
lean_dec(v_snd_570_);
v___x_1029_ = l_Lean_Meta_mkLambdaFVars(v___x_600_, v_pf_947_, v___x_563_, v___x_1027_, v___x_563_, v___x_1028_, v___x_1026_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___x_600_);
if (lean_obj_tag(v___x_1029_) == 0)
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1080_; 
v_a_1030_ = lean_ctor_get(v___x_1029_, 0);
v_isSharedCheck_1080_ = !lean_is_exclusive(v___x_1029_);
if (v_isSharedCheck_1080_ == 0)
{
v___x_1032_ = v___x_1029_;
v_isShared_1033_ = v_isSharedCheck_1080_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_1029_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1080_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1075_; 
v___x_1034_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__69));
lean_inc_n(v_fst_574_, 2);
v___x_1035_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1035_, 0, v_fst_574_);
lean_ctor_set(v___x_1035_, 1, v___x_607_);
v___x_1036_ = l_Lean_Expr_const___override(v___x_1034_, v___x_1035_);
lean_inc_n(v_fst_578_, 5);
v___x_1037_ = l_Lean_Expr_app___override(v___x_1036_, v_fst_578_);
v___x_1038_ = l_Lean_Expr_app___override(v___x_1037_, v_00_u03b1_550_);
v___x_1039_ = l_Lean_Expr_app___override(v___x_1038_, v_a_611_);
lean_inc(v_val_561_);
v___x_1040_ = l_Lean_Expr_app___override(v___x_1039_, v_val_561_);
v___x_1041_ = l_Lean_Expr_app___override(v___x_1040_, v_a_984_);
v___x_1042_ = l_Lean_Expr_app___override(v___x_1041_, v_a_1001_);
v___x_1043_ = l_Lean_Expr_app___override(v___x_1042_, v_a_1010_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1043_, v_fst_590_);
lean_inc(v_fst_586_);
v___x_1045_ = l_Lean_Expr_app___override(v___x_1044_, v_fst_586_);
v___x_1046_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29));
v___x_1047_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31));
v___x_1048_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34));
v___x_1049_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1049_, 0, v_fst_574_);
lean_ctor_set(v___x_1049_, 1, v___x_605_);
lean_inc_ref_n(v___x_1049_, 2);
v___x_1050_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1050_, 0, v_fst_574_);
lean_ctor_set(v___x_1050_, 1, v___x_1049_);
lean_inc_ref(v___x_1050_);
v___x_1051_ = l_Lean_Expr_const___override(v___x_1048_, v___x_1050_);
v___x_1052_ = l_Lean_Expr_app___override(v___x_1051_, v_fst_578_);
v___x_1053_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3));
v___x_1054_ = l_Lean_Expr_const___override(v___x_1053_, v___x_1049_);
v___x_1055_ = l_Lean_Expr_app___override(v___x_1054_, v_fst_578_);
lean_inc_ref(v___x_1055_);
v___x_1056_ = l_Lean_Expr_app___override(v___x_1052_, v___x_1055_);
v___x_1057_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37));
v___x_1058_ = l_Lean_Expr_const___override(v___x_1057_, v___x_1050_);
v___x_1059_ = l_Lean_Expr_app___override(v___x_1058_, v___x_1055_);
v___x_1060_ = l_Lean_Expr_app___override(v___x_1059_, v_fst_578_);
v___x_1061_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39));
v___x_1062_ = l_Lean_Expr_const___override(v___x_1061_, v___x_1049_);
v___x_1063_ = l_Lean_Expr_app___override(v___x_1062_, v_fst_578_);
v___x_1064_ = l_Lean_Expr_app___override(v___x_1060_, v___x_1063_);
v___x_1065_ = l_Lean_Expr_app___override(v___x_1056_, v___x_1064_);
v___x_1066_ = l_Lean_Expr_app___override(v___x_1065_, v_fst_586_);
v___x_1067_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40);
v___x_1068_ = l_Lean_Expr_app___override(v___x_1066_, v___x_1067_);
v___x_1069_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43);
v___x_1070_ = l_Lean_Expr_betaRev(v_a_1030_, v___x_1069_, v___x_563_, v___x_563_);
v___x_1071_ = l_Lean_Expr_lam___override(v___x_1047_, v___x_1068_, v___x_1070_, v___x_1026_);
v___x_1072_ = l_Lean_Expr_lam___override(v___x_1046_, v_fst_578_, v___x_1071_, v___x_1026_);
v___x_1073_ = l_Lean_Expr_app___override(v___x_1045_, v___x_1072_);
if (v_isShared_1009_ == 0)
{
lean_ctor_set(v___x_1008_, 1, v___x_1073_);
lean_ctor_set(v___x_1008_, 0, v_val_561_);
v___x_1075_ = v___x_1008_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1079_; 
v_reuseFailAlloc_1079_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1079_, 0, v_val_561_);
lean_ctor_set(v_reuseFailAlloc_1079_, 1, v___x_1073_);
v___x_1075_ = v_reuseFailAlloc_1079_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
lean_object* v___x_1077_; 
if (v_isShared_1033_ == 0)
{
lean_ctor_set(v___x_1032_, 0, v___x_1075_);
v___x_1077_ = v___x_1032_;
goto v_reusejp_1076_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v___x_1075_);
v___x_1077_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1076_;
}
v_reusejp_1076_:
{
return v___x_1077_;
}
}
}
}
else
{
lean_object* v_a_1081_; lean_object* v___x_1083_; uint8_t v_isShared_1084_; uint8_t v_isSharedCheck_1088_; 
lean_dec(v_a_1010_);
lean_del_object(v___x_1008_);
lean_dec(v_a_1001_);
lean_dec(v_a_984_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_dec(v_fst_586_);
lean_dec(v_fst_578_);
lean_dec(v_fst_574_);
lean_dec(v_val_561_);
lean_dec_ref(v_00_u03b1_550_);
v_a_1081_ = lean_ctor_get(v___x_1029_, 0);
v_isSharedCheck_1088_ = !lean_is_exclusive(v___x_1029_);
if (v_isSharedCheck_1088_ == 0)
{
v___x_1083_ = v___x_1029_;
v_isShared_1084_ = v_isSharedCheck_1088_;
goto v_resetjp_1082_;
}
else
{
lean_inc(v_a_1081_);
lean_dec(v___x_1029_);
v___x_1083_ = lean_box(0);
v_isShared_1084_ = v_isSharedCheck_1088_;
goto v_resetjp_1082_;
}
v_resetjp_1082_:
{
lean_object* v___x_1086_; 
if (v_isShared_1084_ == 0)
{
v___x_1086_ = v___x_1083_;
goto v_reusejp_1085_;
}
else
{
lean_object* v_reuseFailAlloc_1087_; 
v_reuseFailAlloc_1087_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1087_, 0, v_a_1081_);
v___x_1086_ = v_reuseFailAlloc_1087_;
goto v_reusejp_1085_;
}
v_reusejp_1085_:
{
return v___x_1086_;
}
}
}
}
else
{
lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1096_; 
lean_dec(v_a_1010_);
lean_del_object(v___x_1008_);
lean_dec(v_a_1001_);
lean_dec(v_a_984_);
lean_dec_ref(v_pf_947_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_dec(v_fst_586_);
lean_dec(v_fst_578_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec(v_val_561_);
lean_dec_ref(v_00_u03b1_550_);
v_a_1089_ = lean_ctor_get(v___x_1025_, 0);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___x_1025_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1091_ = v___x_1025_;
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_dec(v___x_1025_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1094_; 
if (v_isShared_1092_ == 0)
{
v___x_1094_ = v___x_1091_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v_a_1089_);
v___x_1094_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
return v___x_1094_;
}
}
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1104_; 
lean_dec(v_a_1010_);
lean_del_object(v___x_1008_);
lean_dec(v_a_1001_);
lean_dec(v_a_984_);
lean_dec_ref(v_pf_947_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_dec(v_fst_586_);
lean_dec(v_fst_582_);
lean_dec(v_fst_578_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec(v_val_561_);
lean_dec_ref(v_00_u03b1_550_);
v_a_1097_ = lean_ctor_get(v___x_1018_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1018_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1099_ = v___x_1018_;
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1018_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v___x_1102_; 
if (v_isShared_1100_ == 0)
{
v___x_1102_ = v___x_1099_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1103_; 
v_reuseFailAlloc_1103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1103_, 0, v_a_1097_);
v___x_1102_ = v_reuseFailAlloc_1103_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
return v___x_1102_;
}
}
}
}
}
else
{
lean_dec(v_a_1006_);
lean_dec(v_a_1001_);
lean_dec_ref(v___x_994_);
lean_dec(v_a_984_);
v___y_779_ = v___y_554_;
v___y_780_ = v___y_555_;
v___y_781_ = v___y_556_;
v___y_782_ = v___y_557_;
goto v___jp_778_;
}
}
else
{
lean_object* v_a_1108_; lean_object* v___x_1110_; uint8_t v_isShared_1111_; uint8_t v_isSharedCheck_1115_; 
lean_dec(v_a_1001_);
lean_dec_ref(v___x_994_);
lean_dec(v_a_984_);
lean_dec_ref_known(v_a_603_, 2);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1108_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1115_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1115_ == 0)
{
v___x_1110_ = v___x_1005_;
v_isShared_1111_ = v_isSharedCheck_1115_;
goto v_resetjp_1109_;
}
else
{
lean_inc(v_a_1108_);
lean_dec(v___x_1005_);
v___x_1110_ = lean_box(0);
v_isShared_1111_ = v_isSharedCheck_1115_;
goto v_resetjp_1109_;
}
v_resetjp_1109_:
{
lean_object* v___x_1113_; 
if (v_isShared_1111_ == 0)
{
v___x_1113_ = v___x_1110_;
goto v_reusejp_1112_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v_a_1108_);
v___x_1113_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1112_;
}
v_reusejp_1112_:
{
return v___x_1113_;
}
}
}
}
else
{
lean_dec(v_a_1000_);
lean_dec_ref(v___x_994_);
lean_dec(v_a_984_);
v___y_779_ = v___y_554_;
v___y_780_ = v___y_555_;
v___y_781_ = v___y_556_;
v___y_782_ = v___y_557_;
goto v___jp_778_;
}
}
else
{
lean_object* v_a_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1123_; 
lean_dec_ref(v___x_994_);
lean_dec(v_a_984_);
lean_dec_ref_known(v_a_603_, 2);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1116_ = lean_ctor_get(v___x_999_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v___x_999_);
if (v_isSharedCheck_1123_ == 0)
{
v___x_1118_ = v___x_999_;
v_isShared_1119_ = v_isSharedCheck_1123_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_a_1116_);
lean_dec(v___x_999_);
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
else
{
lean_dec(v_a_983_);
lean_dec_ref(v___x_979_);
lean_dec_ref(v___x_968_);
v___y_779_ = v___y_554_;
v___y_780_ = v___y_555_;
v___y_781_ = v___y_556_;
v___y_782_ = v___y_557_;
goto v___jp_778_;
}
}
else
{
lean_object* v_a_1124_; lean_object* v___x_1126_; uint8_t v_isShared_1127_; uint8_t v_isSharedCheck_1131_; 
lean_dec_ref(v___x_979_);
lean_dec_ref(v___x_968_);
lean_dec_ref_known(v_a_603_, 2);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1124_ = lean_ctor_get(v___x_982_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_982_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1126_ = v___x_982_;
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
else
{
lean_inc(v_a_1124_);
lean_dec(v___x_982_);
v___x_1126_ = lean_box(0);
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
v_resetjp_1125_:
{
lean_object* v___x_1129_; 
if (v_isShared_1127_ == 0)
{
v___x_1129_ = v___x_1126_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_a_1124_);
v___x_1129_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
return v___x_1129_;
}
}
}
}
else
{
v___y_779_ = v___y_554_;
v___y_780_ = v___y_555_;
v___y_781_ = v___y_556_;
v___y_782_ = v___y_557_;
goto v___jp_778_;
}
v___jp_612_:
{
uint8_t v___x_618_; uint8_t v___x_619_; uint8_t v___x_620_; lean_object* v___x_621_; 
v___x_618_ = 0;
v___x_619_ = lean_unbox(v_snd_570_);
v___x_620_ = lean_unbox(v_snd_570_);
lean_dec(v_snd_570_);
v___x_621_ = l_Lean_Meta_mkLambdaFVars(v___x_600_, v_a_617_, v___x_563_, v___x_619_, v___x_563_, v___x_620_, v___x_618_, v___y_614_, v___y_615_, v___y_616_, v___y_613_);
lean_dec_ref(v___x_600_);
if (lean_obj_tag(v___x_621_) == 0)
{
lean_object* v_a_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v_a_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc(v_a_622_);
lean_dec_ref_known(v___x_621_, 1);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__5));
lean_inc_ref(v___x_607_);
v___x_624_ = l_Lean_Expr_const___override(v___x_623_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_625_ = l_Lean_Expr_app___override(v___x_624_, v_00_u03b1_550_);
v___x_626_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_625_, v___y_614_, v___y_615_, v___y_616_, v___y_613_);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_object* v_a_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v_a_627_ = lean_ctor_get(v___x_626_, 0);
lean_inc(v_a_627_);
lean_dec_ref_known(v___x_626_, 1);
v___x_628_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__7));
lean_inc_ref_n(v___x_607_, 5);
v___x_629_ = l_Lean_Expr_const___override(v___x_628_, v___x_607_);
lean_inc_ref_n(v_00_u03b1_550_, 5);
v___x_630_ = l_Lean_Expr_app___override(v___x_629_, v_00_u03b1_550_);
v___x_631_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10));
v___x_632_ = l_Lean_Expr_const___override(v___x_631_, v___x_607_);
v___x_633_ = l_Lean_Expr_app___override(v___x_632_, v_00_u03b1_550_);
v___x_634_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13));
v___x_635_ = l_Lean_Expr_const___override(v___x_634_, v___x_607_);
v___x_636_ = l_Lean_Expr_app___override(v___x_635_, v_00_u03b1_550_);
v___x_637_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16));
v___x_638_ = l_Lean_Expr_const___override(v___x_637_, v___x_607_);
v___x_639_ = l_Lean_Expr_app___override(v___x_638_, v_00_u03b1_550_);
v___x_640_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18));
v___x_641_ = l_Lean_Expr_const___override(v___x_640_, v___x_607_);
v___x_642_ = l_Lean_Expr_app___override(v___x_641_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_643_ = l_Lean_Expr_app___override(v___x_642_, v_a_611_);
v___x_644_ = l_Lean_Expr_app___override(v___x_639_, v___x_643_);
v___x_645_ = l_Lean_Expr_app___override(v___x_636_, v___x_644_);
lean_inc_ref(v___x_645_);
v___x_646_ = l_Lean_Expr_app___override(v___x_633_, v___x_645_);
v___x_647_ = l_Lean_Expr_app___override(v___x_630_, v___x_646_);
lean_inc_ref(v_z_u03b1_551_);
v___x_648_ = l_Lean_Expr_app___override(v___x_647_, v_z_u03b1_551_);
v___x_649_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_648_, v___y_614_, v___y_615_, v___y_616_, v___y_613_);
if (lean_obj_tag(v___x_649_) == 0)
{
lean_object* v_a_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; uint8_t v___x_655_; lean_object* v___x_656_; lean_object* v___f_657_; lean_object* v___x_658_; 
v_a_650_ = lean_ctor_get(v___x_649_, 0);
lean_inc(v_a_650_);
lean_dec_ref_known(v___x_649_, 1);
v___x_651_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20));
lean_inc_ref(v___x_607_);
v___x_652_ = l_Lean_Expr_const___override(v___x_651_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_653_ = l_Lean_Expr_app___override(v___x_652_, v_00_u03b1_550_);
v___x_654_ = l_Lean_Expr_app___override(v___x_653_, v___x_645_);
v___x_655_ = 1;
v___x_656_ = lean_box(v___x_655_);
v___f_657_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1___boxed), 8, 3);
lean_closure_set(v___f_657_, 0, v___x_656_);
lean_closure_set(v___f_657_, 1, v_z_u03b1_551_);
lean_closure_set(v___f_657_, 2, v___x_654_);
v___x_658_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_657_, v___x_563_, v___y_614_, v___y_615_, v___y_616_, v___y_613_);
if (lean_obj_tag(v___x_658_) == 0)
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___f_664_; lean_object* v___x_665_; 
lean_dec_ref_known(v___x_658_, 1);
v___x_659_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22));
lean_inc_ref(v___x_607_);
v___x_660_ = l_Lean_Expr_const___override(v___x_659_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_661_ = l_Lean_Expr_app___override(v___x_660_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_662_ = l_Lean_Expr_app___override(v___x_661_, v_a_611_);
v___x_663_ = lean_box(v___x_655_);
v___f_664_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2___boxed), 8, 3);
lean_closure_set(v___f_664_, 0, v___x_663_);
lean_closure_set(v___f_664_, 1, v_fst_582_);
lean_closure_set(v___f_664_, 2, v___x_662_);
v___x_665_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_664_, v___x_563_, v___y_614_, v___y_615_, v___y_616_, v___y_613_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v___x_667_; uint8_t v_isShared_668_; uint8_t v_isSharedCheck_719_; 
v_isSharedCheck_719_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_719_ == 0)
{
lean_object* v_unused_720_; 
v_unused_720_ = lean_ctor_get(v___x_665_, 0);
lean_dec(v_unused_720_);
v___x_667_ = v___x_665_;
v_isShared_668_ = v_isSharedCheck_719_;
goto v_resetjp_666_;
}
else
{
lean_dec(v___x_665_);
v___x_667_ = lean_box(0);
v_isShared_668_ = v_isSharedCheck_719_;
goto v_resetjp_666_;
}
v_resetjp_666_:
{
lean_object* v___x_669_; lean_object* v___x_671_; 
v___x_669_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__27));
lean_inc(v_fst_574_);
if (v_isShared_589_ == 0)
{
lean_ctor_set_tag(v___x_588_, 1);
lean_ctor_set(v___x_588_, 1, v___x_607_);
lean_ctor_set(v___x_588_, 0, v_fst_574_);
v___x_671_ = v___x_588_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_718_; 
v_reuseFailAlloc_718_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_718_, 0, v_fst_574_);
lean_ctor_set(v_reuseFailAlloc_718_, 1, v___x_607_);
v___x_671_ = v_reuseFailAlloc_718_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_684_; 
v___x_672_ = l_Lean_Expr_const___override(v___x_669_, v___x_671_);
lean_inc(v_fst_578_);
v___x_673_ = l_Lean_Expr_app___override(v___x_672_, v_fst_578_);
v___x_674_ = l_Lean_Expr_app___override(v___x_673_, v_00_u03b1_550_);
v___x_675_ = l_Lean_Expr_app___override(v___x_674_, v_a_611_);
v___x_676_ = l_Lean_Expr_app___override(v___x_675_, v_fst_590_);
lean_inc(v_fst_586_);
v___x_677_ = l_Lean_Expr_app___override(v___x_676_, v_fst_586_);
v___x_678_ = l_Lean_Expr_app___override(v___x_677_, v_a_627_);
v___x_679_ = l_Lean_Expr_app___override(v___x_678_, v_a_650_);
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29));
v___x_681_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31));
v___x_682_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34));
lean_inc(v_fst_574_);
if (v_isShared_585_ == 0)
{
lean_ctor_set_tag(v___x_584_, 1);
lean_ctor_set(v___x_584_, 1, v___x_605_);
lean_ctor_set(v___x_584_, 0, v_fst_574_);
v___x_684_ = v___x_584_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_fst_574_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v___x_605_);
v___x_684_ = v_reuseFailAlloc_717_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
lean_object* v___x_686_; 
lean_inc_ref(v___x_684_);
if (v_isShared_581_ == 0)
{
lean_ctor_set_tag(v___x_580_, 1);
lean_ctor_set(v___x_580_, 1, v___x_684_);
lean_ctor_set(v___x_580_, 0, v_fst_574_);
v___x_686_ = v___x_580_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v_fst_574_);
lean_ctor_set(v_reuseFailAlloc_716_, 1, v___x_684_);
v___x_686_ = v_reuseFailAlloc_716_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_711_; 
lean_inc_ref(v___x_686_);
v___x_687_ = l_Lean_Expr_const___override(v___x_682_, v___x_686_);
lean_inc_n(v_fst_578_, 4);
v___x_688_ = l_Lean_Expr_app___override(v___x_687_, v_fst_578_);
v___x_689_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3));
lean_inc_ref(v___x_684_);
v___x_690_ = l_Lean_Expr_const___override(v___x_689_, v___x_684_);
v___x_691_ = l_Lean_Expr_app___override(v___x_690_, v_fst_578_);
lean_inc_ref(v___x_691_);
v___x_692_ = l_Lean_Expr_app___override(v___x_688_, v___x_691_);
v___x_693_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37));
v___x_694_ = l_Lean_Expr_const___override(v___x_693_, v___x_686_);
v___x_695_ = l_Lean_Expr_app___override(v___x_694_, v___x_691_);
v___x_696_ = l_Lean_Expr_app___override(v___x_695_, v_fst_578_);
v___x_697_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39));
v___x_698_ = l_Lean_Expr_const___override(v___x_697_, v___x_684_);
v___x_699_ = l_Lean_Expr_app___override(v___x_698_, v_fst_578_);
v___x_700_ = l_Lean_Expr_app___override(v___x_696_, v___x_699_);
v___x_701_ = l_Lean_Expr_app___override(v___x_692_, v___x_700_);
v___x_702_ = l_Lean_Expr_app___override(v___x_701_, v_fst_586_);
v___x_703_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40);
v___x_704_ = l_Lean_Expr_app___override(v___x_702_, v___x_703_);
v___x_705_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43);
v___x_706_ = l_Lean_Expr_betaRev(v_a_622_, v___x_705_, v___x_563_, v___x_563_);
v___x_707_ = l_Lean_Expr_lam___override(v___x_681_, v___x_704_, v___x_706_, v___x_618_);
v___x_708_ = l_Lean_Expr_lam___override(v___x_680_, v_fst_578_, v___x_707_, v___x_618_);
v___x_709_ = l_Lean_Expr_app___override(v___x_679_, v___x_708_);
if (v_isShared_577_ == 0)
{
lean_ctor_set_tag(v___x_576_, 2);
lean_ctor_set(v___x_576_, 1, v___x_709_);
lean_ctor_set(v___x_576_, 0, v_p_u03b1_x3f_552_);
v___x_711_ = v___x_576_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_p_u03b1_x3f_552_);
lean_ctor_set(v_reuseFailAlloc_715_, 1, v___x_709_);
v___x_711_ = v_reuseFailAlloc_715_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
lean_object* v___x_713_; 
if (v_isShared_668_ == 0)
{
lean_ctor_set(v___x_667_, 0, v___x_711_);
v___x_713_ = v___x_667_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v___x_711_);
v___x_713_ = v_reuseFailAlloc_714_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
return v___x_713_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_728_; 
lean_dec(v_a_650_);
lean_dec(v_a_627_);
lean_dec(v_a_622_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_00_u03b1_550_);
v_a_721_ = lean_ctor_get(v___x_665_, 0);
v_isSharedCheck_728_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_728_ == 0)
{
v___x_723_ = v___x_665_;
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_a_721_);
lean_dec(v___x_665_);
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
lean_dec(v_a_650_);
lean_dec(v_a_627_);
lean_dec(v_a_622_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_00_u03b1_550_);
v_a_729_ = lean_ctor_get(v___x_658_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_736_ == 0)
{
v___x_731_ = v___x_658_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_a_729_);
lean_dec(v___x_658_);
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
lean_dec_ref(v___x_645_);
lean_dec(v_a_627_);
lean_dec(v_a_622_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
v_a_737_ = lean_ctor_get(v___x_649_, 0);
v_isSharedCheck_744_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_744_ == 0)
{
v___x_739_ = v___x_649_;
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_649_);
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
else
{
lean_object* v_a_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_752_; 
lean_dec(v_a_622_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
v_a_745_ = lean_ctor_get(v___x_626_, 0);
v_isSharedCheck_752_ = !lean_is_exclusive(v___x_626_);
if (v_isSharedCheck_752_ == 0)
{
v___x_747_ = v___x_626_;
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_a_745_);
lean_dec(v___x_626_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_750_; 
if (v_isShared_748_ == 0)
{
v___x_750_ = v___x_747_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v_a_745_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
else
{
lean_object* v_a_753_; lean_object* v___x_755_; uint8_t v_isShared_756_; uint8_t v_isSharedCheck_760_; 
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
v_a_753_ = lean_ctor_get(v___x_621_, 0);
v_isSharedCheck_760_ = !lean_is_exclusive(v___x_621_);
if (v_isSharedCheck_760_ == 0)
{
v___x_755_ = v___x_621_;
v_isShared_756_ = v_isSharedCheck_760_;
goto v_resetjp_754_;
}
else
{
lean_inc(v_a_753_);
lean_dec(v___x_621_);
v___x_755_ = lean_box(0);
v_isShared_756_ = v_isSharedCheck_760_;
goto v_resetjp_754_;
}
v_resetjp_754_:
{
lean_object* v___x_758_; 
if (v_isShared_756_ == 0)
{
v___x_758_ = v___x_755_;
goto v_reusejp_757_;
}
else
{
lean_object* v_reuseFailAlloc_759_; 
v_reuseFailAlloc_759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_759_, 0, v_a_753_);
v___x_758_ = v_reuseFailAlloc_759_;
goto v_reusejp_757_;
}
v_reusejp_757_:
{
return v___x_758_;
}
}
}
}
v___jp_761_:
{
lean_object* v___x_766_; 
lean_inc_ref(v_z_u03b1_551_);
lean_inc_ref(v_00_u03b1_550_);
v___x_766_ = lp_mathlib_Mathlib_Meta_Positivity_Strictness_toNonzero___redArg(v_u_549_, v_00_u03b1_550_, v_z_u03b1_551_, v___x_601_, v_a_603_);
if (lean_obj_tag(v___x_766_) == 0)
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v_a_769_; lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_776_; 
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
v___x_767_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__45);
v___x_768_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(v___x_767_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
v_a_769_ = lean_ctor_get(v___x_768_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_776_ == 0)
{
v___x_771_ = v___x_768_;
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
else
{
lean_inc(v_a_769_);
lean_dec(v___x_768_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v___x_774_; 
if (v_isShared_772_ == 0)
{
v___x_774_ = v___x_771_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_a_769_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
else
{
lean_object* v_val_777_; 
v_val_777_ = lean_ctor_get(v___x_766_, 0);
lean_inc(v_val_777_);
lean_dec_ref_known(v___x_766_, 1);
v___y_613_ = v___y_765_;
v___y_614_ = v___y_762_;
v___y_615_ = v___y_763_;
v___y_616_ = v___y_764_;
v_a_617_ = v_val_777_;
goto v___jp_612_;
}
}
v___jp_778_:
{
lean_object* v___x_783_; 
lean_inc(v_a_603_);
lean_inc(v_val_561_);
lean_inc_ref(v___x_601_);
lean_inc_ref(v_z_u03b1_551_);
lean_inc_ref(v_00_u03b1_550_);
lean_inc(v_u_549_);
v___x_783_ = lp_mathlib_Mathlib_Meta_Positivity_Strictness_toNonneg(v_u_549_, v_00_u03b1_550_, v_z_u03b1_551_, v___x_601_, v_val_561_, v_a_603_);
if (lean_obj_tag(v___x_783_) == 1)
{
lean_object* v_val_784_; uint8_t v___x_785_; uint8_t v___x_786_; uint8_t v___x_787_; lean_object* v___x_788_; 
v_val_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_val_784_);
lean_dec_ref_known(v___x_783_, 1);
v___x_785_ = 0;
v___x_786_ = lean_unbox(v_snd_570_);
v___x_787_ = lean_unbox(v_snd_570_);
v___x_788_ = l_Lean_Meta_mkLambdaFVars(v___x_600_, v_val_784_, v___x_563_, v___x_786_, v___x_563_, v___x_787_, v___x_785_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_788_) == 0)
{
lean_object* v_a_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v_a_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_a_789_);
lean_dec_ref_known(v___x_788_, 1);
v___x_790_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__47));
lean_inc_ref_n(v___x_607_, 8);
v___x_791_ = l_Lean_Expr_const___override(v___x_790_, v___x_607_);
lean_inc_ref_n(v_00_u03b1_550_, 8);
v___x_792_ = l_Lean_Expr_app___override(v___x_791_, v_00_u03b1_550_);
lean_inc_ref(v_z_u03b1_551_);
v___x_793_ = l_Lean_Expr_app___override(v___x_792_, v_z_u03b1_551_);
v___x_794_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__50));
v___x_795_ = l_Lean_Expr_const___override(v___x_794_, v___x_607_);
v___x_796_ = l_Lean_Expr_app___override(v___x_795_, v_00_u03b1_550_);
v___x_797_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__53));
v___x_798_ = l_Lean_Expr_const___override(v___x_797_, v___x_607_);
v___x_799_ = l_Lean_Expr_app___override(v___x_798_, v_00_u03b1_550_);
v___x_800_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__55));
v___x_801_ = l_Lean_Expr_const___override(v___x_800_, v___x_607_);
v___x_802_ = l_Lean_Expr_app___override(v___x_801_, v_00_u03b1_550_);
v___x_803_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__16));
v___x_804_ = l_Lean_Expr_const___override(v___x_803_, v___x_607_);
v___x_805_ = l_Lean_Expr_app___override(v___x_804_, v_00_u03b1_550_);
v___x_806_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__18));
v___x_807_ = l_Lean_Expr_const___override(v___x_806_, v___x_607_);
v___x_808_ = l_Lean_Expr_app___override(v___x_807_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_809_ = l_Lean_Expr_app___override(v___x_808_, v_a_611_);
v___x_810_ = l_Lean_Expr_app___override(v___x_805_, v___x_809_);
lean_inc_ref(v___x_810_);
v___x_811_ = l_Lean_Expr_app___override(v___x_802_, v___x_810_);
v___x_812_ = l_Lean_Expr_app___override(v___x_799_, v___x_811_);
v___x_813_ = l_Lean_Expr_app___override(v___x_796_, v___x_812_);
v___x_814_ = l_Lean_Expr_app___override(v___x_793_, v___x_813_);
v___x_815_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__58));
v___x_816_ = l_Lean_Expr_const___override(v___x_815_, v___x_607_);
v___x_817_ = l_Lean_Expr_app___override(v___x_816_, v_00_u03b1_550_);
v___x_818_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__61));
v___x_819_ = l_Lean_Expr_const___override(v___x_818_, v___x_607_);
v___x_820_ = l_Lean_Expr_app___override(v___x_819_, v_00_u03b1_550_);
lean_inc(v_val_561_);
v___x_821_ = l_Lean_Expr_app___override(v___x_820_, v_val_561_);
lean_inc_ref(v___x_821_);
v___x_822_ = l_Lean_Expr_app___override(v___x_817_, v___x_821_);
v___x_823_ = l_Lean_Expr_app___override(v___x_814_, v___x_822_);
v___x_824_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_823_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_824_) == 0)
{
lean_object* v_a_825_; 
v_a_825_ = lean_ctor_get(v___x_824_, 0);
lean_inc(v_a_825_);
lean_dec_ref_known(v___x_824_, 1);
if (lean_obj_tag(v_a_825_) == 1)
{
lean_object* v_a_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; 
v_a_826_ = lean_ctor_get(v_a_825_, 0);
lean_inc(v_a_826_);
lean_dec_ref_known(v_a_825_, 1);
v___x_827_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__63));
lean_inc_ref_n(v___x_607_, 3);
v___x_828_ = l_Lean_Expr_const___override(v___x_827_, v___x_607_);
lean_inc_ref_n(v_00_u03b1_550_, 3);
v___x_829_ = l_Lean_Expr_app___override(v___x_828_, v_00_u03b1_550_);
v___x_830_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__10));
v___x_831_ = l_Lean_Expr_const___override(v___x_830_, v___x_607_);
v___x_832_ = l_Lean_Expr_app___override(v___x_831_, v_00_u03b1_550_);
v___x_833_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__13));
v___x_834_ = l_Lean_Expr_const___override(v___x_833_, v___x_607_);
v___x_835_ = l_Lean_Expr_app___override(v___x_834_, v_00_u03b1_550_);
v___x_836_ = l_Lean_Expr_app___override(v___x_835_, v___x_810_);
lean_inc_ref(v___x_836_);
v___x_837_ = l_Lean_Expr_app___override(v___x_832_, v___x_836_);
v___x_838_ = l_Lean_Expr_app___override(v___x_829_, v___x_837_);
lean_inc_ref(v_z_u03b1_551_);
v___x_839_ = l_Lean_Expr_app___override(v___x_838_, v_z_u03b1_551_);
lean_inc_ref(v___x_821_);
v___x_840_ = l_Lean_Expr_app___override(v___x_839_, v___x_821_);
v___x_841_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_840_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_841_) == 0)
{
lean_object* v_a_842_; 
v_a_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc(v_a_842_);
lean_dec_ref_known(v___x_841_, 1);
if (lean_obj_tag(v_a_842_) == 1)
{
lean_object* v_a_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; uint8_t v___x_848_; lean_object* v___x_849_; lean_object* v___f_850_; lean_object* v___x_851_; 
lean_inc(v_val_561_);
lean_dec(v_a_603_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_del_object(v___x_588_);
lean_del_object(v___x_584_);
lean_del_object(v___x_580_);
lean_del_object(v___x_576_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec(v_u_549_);
v_a_843_ = lean_ctor_get(v_a_842_, 0);
lean_inc(v_a_843_);
lean_dec_ref_known(v_a_842_, 1);
v___x_844_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__20));
lean_inc_ref(v___x_607_);
v___x_845_ = l_Lean_Expr_const___override(v___x_844_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_846_ = l_Lean_Expr_app___override(v___x_845_, v_00_u03b1_550_);
v___x_847_ = l_Lean_Expr_app___override(v___x_846_, v___x_836_);
v___x_848_ = 1;
v___x_849_ = lean_box(v___x_848_);
v___f_850_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__1___boxed), 8, 3);
lean_closure_set(v___f_850_, 0, v___x_849_);
lean_closure_set(v___f_850_, 1, v_z_u03b1_551_);
lean_closure_set(v___f_850_, 2, v___x_847_);
v___x_851_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_850_, v___x_563_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_851_) == 0)
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___f_857_; lean_object* v___x_858_; 
lean_dec_ref_known(v___x_851_, 1);
v___x_852_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__22));
lean_inc_ref(v___x_607_);
v___x_853_ = l_Lean_Expr_const___override(v___x_852_, v___x_607_);
lean_inc_ref(v_00_u03b1_550_);
v___x_854_ = l_Lean_Expr_app___override(v___x_853_, v_00_u03b1_550_);
lean_inc(v_a_611_);
v___x_855_ = l_Lean_Expr_app___override(v___x_854_, v_a_611_);
v___x_856_ = lean_box(v___x_848_);
v___f_857_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__2___boxed), 8, 3);
lean_closure_set(v___f_857_, 0, v___x_856_);
lean_closure_set(v___f_857_, 1, v_fst_582_);
lean_closure_set(v___f_857_, 2, v___x_855_);
v___x_858_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__2___redArg(v___f_857_, v___x_563_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_858_) == 0)
{
lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_905_; 
v_isSharedCheck_905_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_905_ == 0)
{
lean_object* v_unused_906_; 
v_unused_906_ = lean_ctor_get(v___x_858_, 0);
lean_dec(v_unused_906_);
v___x_860_ = v___x_858_;
v_isShared_861_ = v_isSharedCheck_905_;
goto v_resetjp_859_;
}
else
{
lean_dec(v___x_858_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_905_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_903_; 
v___x_862_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__65));
lean_inc_n(v_fst_574_, 2);
v___x_863_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_863_, 0, v_fst_574_);
lean_ctor_set(v___x_863_, 1, v___x_607_);
v___x_864_ = l_Lean_Expr_const___override(v___x_862_, v___x_863_);
lean_inc_n(v_fst_578_, 5);
v___x_865_ = l_Lean_Expr_app___override(v___x_864_, v_fst_578_);
v___x_866_ = l_Lean_Expr_app___override(v___x_865_, v_00_u03b1_550_);
v___x_867_ = l_Lean_Expr_app___override(v___x_866_, v_a_611_);
v___x_868_ = l_Lean_Expr_app___override(v___x_867_, v___x_821_);
v___x_869_ = l_Lean_Expr_app___override(v___x_868_, v_a_826_);
v___x_870_ = l_Lean_Expr_app___override(v___x_869_, v_a_843_);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_fst_590_);
lean_inc(v_fst_586_);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_fst_586_);
v___x_873_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__29));
v___x_874_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__31));
v___x_875_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__34));
v___x_876_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_876_, 0, v_fst_574_);
lean_ctor_set(v___x_876_, 1, v___x_605_);
lean_inc_ref_n(v___x_876_, 2);
v___x_877_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_877_, 0, v_fst_574_);
lean_ctor_set(v___x_877_, 1, v___x_876_);
lean_inc_ref(v___x_877_);
v___x_878_ = l_Lean_Expr_const___override(v___x_875_, v___x_877_);
v___x_879_ = l_Lean_Expr_app___override(v___x_878_, v_fst_578_);
v___x_880_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__0___closed__3));
v___x_881_ = l_Lean_Expr_const___override(v___x_880_, v___x_876_);
v___x_882_ = l_Lean_Expr_app___override(v___x_881_, v_fst_578_);
lean_inc_ref(v___x_882_);
v___x_883_ = l_Lean_Expr_app___override(v___x_879_, v___x_882_);
v___x_884_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__37));
v___x_885_ = l_Lean_Expr_const___override(v___x_884_, v___x_877_);
v___x_886_ = l_Lean_Expr_app___override(v___x_885_, v___x_882_);
v___x_887_ = l_Lean_Expr_app___override(v___x_886_, v_fst_578_);
v___x_888_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__39));
v___x_889_ = l_Lean_Expr_const___override(v___x_888_, v___x_876_);
v___x_890_ = l_Lean_Expr_app___override(v___x_889_, v_fst_578_);
v___x_891_ = l_Lean_Expr_app___override(v___x_887_, v___x_890_);
v___x_892_ = l_Lean_Expr_app___override(v___x_883_, v___x_891_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v_fst_586_);
v___x_894_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__40);
v___x_895_ = l_Lean_Expr_app___override(v___x_893_, v___x_894_);
v___x_896_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43, &lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43_once, _init_lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___closed__43);
v___x_897_ = l_Lean_Expr_betaRev(v_a_789_, v___x_896_, v___x_563_, v___x_563_);
v___x_898_ = l_Lean_Expr_lam___override(v___x_874_, v___x_895_, v___x_897_, v___x_785_);
v___x_899_ = l_Lean_Expr_lam___override(v___x_873_, v_fst_578_, v___x_898_, v___x_785_);
v___x_900_ = l_Lean_Expr_app___override(v___x_872_, v___x_899_);
v___x_901_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_901_, 0, v_val_561_);
lean_ctor_set(v___x_901_, 1, v___x_900_);
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 0, v___x_901_);
v___x_903_ = v___x_860_;
goto v_reusejp_902_;
}
else
{
lean_object* v_reuseFailAlloc_904_; 
v_reuseFailAlloc_904_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_904_, 0, v___x_901_);
v___x_903_ = v_reuseFailAlloc_904_;
goto v_reusejp_902_;
}
v_reusejp_902_:
{
return v___x_903_;
}
}
}
else
{
lean_object* v_a_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_914_; 
lean_dec(v_a_843_);
lean_dec(v_a_826_);
lean_dec_ref(v___x_821_);
lean_dec(v_a_789_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_dec(v_fst_586_);
lean_dec(v_fst_578_);
lean_dec(v_fst_574_);
lean_dec(v_val_561_);
lean_dec_ref(v_00_u03b1_550_);
v_a_907_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_914_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_914_ == 0)
{
v___x_909_ = v___x_858_;
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_a_907_);
lean_dec(v___x_858_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
lean_object* v___x_912_; 
if (v_isShared_910_ == 0)
{
v___x_912_ = v___x_909_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_a_907_);
v___x_912_ = v_reuseFailAlloc_913_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
return v___x_912_;
}
}
}
}
else
{
lean_object* v_a_915_; lean_object* v___x_917_; uint8_t v_isShared_918_; uint8_t v_isSharedCheck_922_; 
lean_dec(v_a_843_);
lean_dec(v_a_826_);
lean_dec_ref(v___x_821_);
lean_dec(v_a_789_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_fst_590_);
lean_dec(v_fst_586_);
lean_dec(v_fst_582_);
lean_dec(v_fst_578_);
lean_dec(v_fst_574_);
lean_dec(v_val_561_);
lean_dec_ref(v_00_u03b1_550_);
v_a_915_ = lean_ctor_get(v___x_851_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___x_851_);
if (v_isSharedCheck_922_ == 0)
{
v___x_917_ = v___x_851_;
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
else
{
lean_inc(v_a_915_);
lean_dec(v___x_851_);
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
else
{
lean_dec(v_a_842_);
lean_dec_ref(v___x_836_);
lean_dec(v_a_826_);
lean_dec_ref(v___x_821_);
lean_dec(v_a_789_);
v___y_762_ = v___y_779_;
v___y_763_ = v___y_780_;
v___y_764_ = v___y_781_;
v___y_765_ = v___y_782_;
goto v___jp_761_;
}
}
else
{
lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_930_; 
lean_dec_ref(v___x_836_);
lean_dec(v_a_826_);
lean_dec_ref(v___x_821_);
lean_dec(v_a_789_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_a_603_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_923_ = lean_ctor_get(v___x_841_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_841_);
if (v_isSharedCheck_930_ == 0)
{
v___x_925_ = v___x_841_;
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_841_);
v___x_925_ = lean_box(0);
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
v_resetjp_924_:
{
lean_object* v___x_928_; 
if (v_isShared_926_ == 0)
{
v___x_928_ = v___x_925_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v_a_923_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
}
else
{
lean_dec(v_a_825_);
lean_dec_ref(v___x_821_);
lean_dec_ref(v___x_810_);
lean_dec(v_a_789_);
v___y_762_ = v___y_779_;
v___y_763_ = v___y_780_;
v___y_764_ = v___y_781_;
v___y_765_ = v___y_782_;
goto v___jp_761_;
}
}
else
{
lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_938_; 
lean_dec_ref(v___x_821_);
lean_dec_ref(v___x_810_);
lean_dec(v_a_789_);
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_a_603_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_931_ = lean_ctor_get(v___x_824_, 0);
v_isSharedCheck_938_ = !lean_is_exclusive(v___x_824_);
if (v_isSharedCheck_938_ == 0)
{
v___x_933_ = v___x_824_;
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v___x_824_);
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
lean_object* v_a_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_946_; 
lean_dec(v_a_611_);
lean_dec_ref(v___x_607_);
lean_dec(v_a_603_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_939_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_946_ == 0)
{
v___x_941_ = v___x_788_;
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_a_939_);
lean_dec(v___x_788_);
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
else
{
lean_dec(v___x_783_);
v___y_762_ = v___y_779_;
v___y_763_ = v___y_780_;
v___y_764_ = v___y_781_;
v___y_765_ = v___y_782_;
goto v___jp_761_;
}
}
}
else
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1139_; 
lean_dec_ref(v___x_607_);
lean_dec(v_a_603_);
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1132_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_1139_ == 0)
{
v___x_1134_ = v___x_610_;
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_610_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1137_; 
if (v_isShared_1135_ == 0)
{
v___x_1137_ = v___x_1134_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_a_1132_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_601_);
lean_dec_ref(v___x_600_);
lean_del_object(v___x_592_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
return v___x_602_;
}
}
else
{
lean_object* v_a_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1148_; 
lean_del_object(v___x_592_);
lean_dec(v_fst_590_);
lean_del_object(v___x_588_);
lean_dec(v_fst_586_);
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_del_object(v___x_576_);
lean_dec(v_fst_574_);
lean_dec(v_snd_570_);
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1141_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_1148_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_1148_ == 0)
{
v___x_1143_ = v___x_596_;
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_a_1141_);
lean_dec(v___x_596_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1148_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1146_; 
if (v_isShared_1144_ == 0)
{
v___x_1146_ = v___x_1143_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1147_; 
v_reuseFailAlloc_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1147_, 0, v_a_1141_);
v___x_1146_ = v_reuseFailAlloc_1147_;
goto v_reusejp_1145_;
}
v_reusejp_1145_:
{
return v___x_1146_;
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
else
{
lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1166_; 
lean_dec_ref_known(v_p_u03b1_x3f_552_, 1);
lean_dec_ref(v_z_u03b1_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v_u_549_);
v_a_1159_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_1166_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_1166_ == 0)
{
v___x_1161_ = v___x_564_;
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v___x_564_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
lean_object* v___x_1164_; 
if (v_isShared_1162_ == 0)
{
v___x_1164_ = v___x_1161_;
goto v_reusejp_1163_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v_a_1159_);
v___x_1164_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1163_;
}
v_reusejp_1163_:
{
return v___x_1164_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7___boxed(lean_object* v_u_1167_, lean_object* v_00_u03b1_1168_, lean_object* v_z_u03b1_1169_, lean_object* v_p_u03b1_x3f_1170_, lean_object* v_e_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v_res_1177_; 
v_res_1177_ = lp_mathlib_Mathlib_Meta_Positivity_evalFinsetProd___lam__7(v_u_1167_, v_00_u03b1_1168_, v_z_u03b1_1169_, v_p_u03b1_x3f_1170_, v_e_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
lean_dec(v___y_1173_);
lean_dec_ref(v___y_1172_);
return v_res_1177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3(lean_object* v_00_u03b1_1180_, lean_object* v_msg_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___redArg(v_msg_1181_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3___boxed(lean_object* v_00_u03b1_1188_, lean_object* v_msg_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
lean_object* v_res_1195_; 
v_res_1195_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_Positivity_evalFinsetProd_spec__3(v_00_u03b1_1188_, v_msg_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_);
lean_dec(v___y_1193_);
lean_dec_ref(v___y_1192_);
lean_dec(v___y_1191_);
lean_dec_ref(v___y_1190_);
return v_res_1195_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(builtin);
}
#ifdef __cplusplus
}
#endif
