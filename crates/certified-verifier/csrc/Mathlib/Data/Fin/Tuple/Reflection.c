// Lean compiler output
// Module: Mathlib.Data.Fin.Tuple.Reflection
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.VecNotation public import Mathlib.Algebra.BigOperators.Fin
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_inferTypeQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_nat_x3f(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lp_Qq_Qq_assertDefEqQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_vecTail___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_FinVec_etaExpand___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FinVec_etaExpand___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FinVec_etaExpand___redArg___closed__0 = (const lean_object*)&lp_mathlib_FinVec_etaExpand___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FinVec_sum___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_sum___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__3_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__5_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MulOne"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__7_value),LEAN_SCALAR_PTR_LITERAL(164, 62, 57, 171, 247, 8, 21, 201)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__8_value),LEAN_SCALAR_PTR_LITERAL(34, 151, 122, 170, 63, 107, 53, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MulOneClass"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMulOne"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__10_value),LEAN_SCALAR_PTR_LITERAL(68, 11, 146, 104, 134, 210, 88, 211)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__11_value),LEAN_SCALAR_PTR_LITERAL(137, 54, 46, 26, 85, 32, 178, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toMulOneClass"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__14_value),LEAN_SCALAR_PTR_LITERAL(109, 62, 35, 77, 93, 105, 196, 34)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "CommMonoid"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__16_value),LEAN_SCALAR_PTR_LITERAL(244, 39, 115, 67, 109, 198, 49, 224)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__17_value),LEAN_SCALAR_PTR_LITERAL(219, 153, 45, 180, 54, 166, 4, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__19_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__20_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instOfNat"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__27_value),LEAN_SCALAR_PTR_LITERAL(92, 84, 52, 176, 228, 163, 228, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__30_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__0 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__0_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__1;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__2 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__2_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__3 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__3_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__4_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__3_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__4 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__4_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__5 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__5_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__7_value),LEAN_SCALAR_PTR_LITERAL(164, 62, 57, 171, 247, 8, 21, 201)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__6_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__5_value),LEAN_SCALAR_PTR_LITERAL(5, 117, 31, 141, 230, 176, 88, 11)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__6 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__6_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "prod_univ_zero"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__7 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__7_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__8_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__7_value),LEAN_SCALAR_PTR_LITERAL(161, 177, 159, 197, 20, 126, 45, 72)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__8 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__8_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NeZero"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__9 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__9_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__10 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__10_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__10_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__11 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__11_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__12;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__13 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__13_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__14 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__14_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__13_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__15_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__14_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__15 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__15_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__16;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__17;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "instMulZeroClass"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__18 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__18_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__10_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__19_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__18_value),LEAN_SCALAR_PTR_LITERAL(106, 151, 89, 42, 8, 210, 160, 13)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__19 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__19_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__20;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__21;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__22 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__22_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__23 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__23_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__22_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__24_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__23_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__24 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__24_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22_value)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__25 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__25_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__25_value)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__26 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__26_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__27;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__28;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__29;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__30;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__31 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__31_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__31_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__32 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__32_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__33;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__34;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instAddNat"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__35 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__35_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__35_value),LEAN_SCALAR_PTR_LITERAL(228, 164, 175, 25, 228, 165, 175, 183)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__36 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__36_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__37;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__38;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__39;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__40;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__41;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__42 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__42_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__42_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__43 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__43_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__44;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__45;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__46;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__47 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__47_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__9_value),LEAN_SCALAR_PTR_LITERAL(82, 249, 173, 83, 51, 144, 28, 211)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__48_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__47_value),LEAN_SCALAR_PTR_LITERAL(18, 211, 45, 226, 175, 66, 42, 16)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__48 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__48_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__49;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__50;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__51;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "succ_ne_zero"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__52 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__52_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__10_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__53_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__52_value),LEAN_SCALAR_PTR_LITERAL(87, 138, 114, 183, 27, 252, 104, 142)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__53 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__53_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__54;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__55 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__55_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__56 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__56_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prod"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__57 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__57_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__56_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__58_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__57_value),LEAN_SCALAR_PTR_LITERAL(247, 66, 46, 56, 151, 61, 191, 120)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__58 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__58_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "univ"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__59 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__59_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__56_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__60_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__59_value),LEAN_SCALAR_PTR_LITERAL(177, 108, 234, 69, 25, 31, 35, 26)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__60 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__60_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__61;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "fintype"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__62 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__62_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__63_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__62_value),LEAN_SCALAR_PTR_LITERAL(169, 78, 208, 237, 66, 0, 118, 117)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__63 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__63_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__64;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "i"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__65 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__65_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__65_value),LEAN_SCALAR_PTR_LITERAL(14, 215, 4, 153, 96, 18, 167, 14)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__66 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__66_value;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__67;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__68;
static lean_once_cell_t lp_mathlib_FinVec_mkProdEqQ___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__69;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "symm"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__70 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__70_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__55_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__71_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__70_value),LEAN_SCALAR_PTR_LITERAL(220, 149, 144, 59, 77, 93, 25, 217)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__71 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__71_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "FinVec"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__72 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__72_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__72_value),LEAN_SCALAR_PTR_LITERAL(186, 160, 243, 208, 38, 208, 197, 213)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__73_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__57_value),LEAN_SCALAR_PTR_LITERAL(238, 67, 202, 95, 154, 212, 159, 108)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__73 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__73_value;
static const lean_string_object lp_mathlib_FinVec_mkProdEqQ___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "prod_eq"};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__74 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__74_value;
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__75_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__72_value),LEAN_SCALAR_PTR_LITERAL(186, 160, 243, 208, 38, 208, 197, 213)}};
static const lean_ctor_object lp_mathlib_FinVec_mkProdEqQ___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__75_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__74_value),LEAN_SCALAR_PTR_LITERAL(46, 189, 171, 198, 207, 50, 158, 15)}};
static const lean_object* lp_mathlib_FinVec_mkProdEqQ___closed__75 = (const lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__75_value;
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkProdEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkProdEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommMagma"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toAdd"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 189, 50, 33, 47, 29, 60, 68)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(44, 66, 253, 250, 14, 60, 133, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddCommSemigroup"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommMagma"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__3_value),LEAN_SCALAR_PTR_LITERAL(151, 206, 252, 2, 83, 210, 212, 206)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__4_value),LEAN_SCALAR_PTR_LITERAL(1, 93, 103, 246, 91, 23, 231, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "AddCommMonoid"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddCommSemigroup"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 119, 180, 1, 34, 115, 27, 117)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__7_value),LEAN_SCALAR_PTR_LITERAL(221, 41, 82, 179, 101, 248, 30, 217)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__0 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__0_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__1 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__1_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__2_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__1_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__2 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__2_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "AddZero"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__3 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__3_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__3_value),LEAN_SCALAR_PTR_LITERAL(171, 135, 49, 0, 6, 244, 57, 130)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__4_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__14_value),LEAN_SCALAR_PTR_LITERAL(87, 27, 84, 210, 142, 102, 48, 129)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__4 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__4_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddZeroClass"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__5 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__5_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toAddZero"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__6 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__6_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__5_value),LEAN_SCALAR_PTR_LITERAL(157, 204, 59, 233, 207, 78, 141, 136)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__7_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(64, 236, 134, 119, 35, 182, 73, 75)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__7 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__7_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddMonoid"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__8 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__8_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddZeroClass"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__9 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__9_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__10_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__9_value),LEAN_SCALAR_PTR_LITERAL(75, 217, 102, 131, 1, 241, 19, 50)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__10 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__10_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toAddMonoid"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__11 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__11_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 119, 180, 1, 34, 115, 27, 117)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__12_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__11_value),LEAN_SCALAR_PTR_LITERAL(201, 177, 114, 202, 140, 168, 124, 193)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__12 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__12_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "sum_univ_zero"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__13 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__13_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__24_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__14_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__13_value),LEAN_SCALAR_PTR_LITERAL(192, 101, 143, 161, 71, 159, 140, 133)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__14 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__14_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "sum"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__15 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__15_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__56_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__16_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__15_value),LEAN_SCALAR_PTR_LITERAL(102, 91, 243, 213, 138, 7, 6, 233)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__16 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__16_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__72_value),LEAN_SCALAR_PTR_LITERAL(186, 160, 243, 208, 38, 208, 197, 213)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__17_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__15_value),LEAN_SCALAR_PTR_LITERAL(223, 240, 87, 147, 111, 9, 144, 119)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__17 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__17_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__3_value),LEAN_SCALAR_PTR_LITERAL(171, 135, 49, 0, 6, 244, 57, 130)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 246, 176, 40, 235, 19, 153, 185)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__18 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__18_value;
static const lean_string_object lp_mathlib_FinVec_mkSumEqQ___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sum_eq"};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__19 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__19_value;
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinVec_mkProdEqQ___closed__72_value),LEAN_SCALAR_PTR_LITERAL(186, 160, 243, 208, 38, 208, 197, 213)}};
static const lean_ctor_object lp_mathlib_FinVec_mkSumEqQ___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__20_value_aux_0),((lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__19_value),LEAN_SCALAR_PTR_LITERAL(20, 159, 30, 5, 84, 67, 189, 77)}};
static const lean_object* lp_mathlib_FinVec_mkSumEqQ___closed__20 = (const lean_object*)&lp_mathlib_FinVec_mkSumEqQ___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkSumEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkSumEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__16_value),LEAN_SCALAR_PTR_LITERAL(244, 39, 115, 67, 109, 198, 49, 224)}};
static const lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Fintype"};
static const lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(146, 129, 114, 60, 203, 137, 135, 143)}};
static const lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 119, 180, 1, 34, 115, 27, 117)}};
static const lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___redArg___boxed(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_x_3_, lean_object* v_a_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_FinVec_seq___redArg(v_x_1_, v_x_2_, v_x_3_, v_a_4_);
lean_dec(v_a_4_);
lean_dec(v_x_1_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___redArg(lean_object* v_x_6_, lean_object* v_x_7_, lean_object* v_x_8_, lean_object* v_a_9_){
_start:
{
lean_object* v_zero_10_; uint8_t v_isZero_11_; lean_object* v_one_12_; lean_object* v_n_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_zero_10_ = lean_unsigned_to_nat(0u);
v_isZero_11_ = lean_nat_dec_eq(v_x_6_, v_zero_10_);
v_one_12_ = lean_unsigned_to_nat(1u);
v_n_13_ = lean_nat_sub(v_x_6_, v_one_12_);
v___x_14_ = lean_nat_add(v_n_13_, v_one_12_);
v___x_15_ = lean_nat_mod(v_zero_10_, v___x_14_);
lean_dec(v___x_14_);
lean_inc(v_x_8_);
lean_inc(v___x_15_);
v___x_16_ = lean_apply_1(v_x_8_, v___x_15_);
lean_inc(v_x_7_);
v___x_17_ = lean_apply_2(v_x_7_, v___x_15_, v___x_16_);
lean_inc_n(v_n_13_, 2);
v___x_18_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecTail___boxed), 4, 3);
lean_closure_set(v___x_18_, 0, lean_box(0));
lean_closure_set(v___x_18_, 1, v_n_13_);
lean_closure_set(v___x_18_, 2, v_x_7_);
v___x_19_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecTail___boxed), 4, 3);
lean_closure_set(v___x_19_, 0, lean_box(0));
lean_closure_set(v___x_19_, 1, v_n_13_);
lean_closure_set(v___x_19_, 2, v_x_8_);
v___x_20_ = lean_alloc_closure((void*)(lp_mathlib_FinVec_seq___redArg___boxed), 4, 3);
lean_closure_set(v___x_20_, 0, v_n_13_);
lean_closure_set(v___x_20_, 1, v___x_18_);
lean_closure_set(v___x_20_, 2, v___x_19_);
v___x_21_ = l_Fin_cases___redArg(v___x_17_, v___x_20_, v_a_9_);
lean_dec(v___x_17_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq(lean_object* v_00_u03b1_22_, lean_object* v_00_u03b2_23_, lean_object* v_x_24_, lean_object* v_x_25_, lean_object* v_x_26_, lean_object* v_a_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_FinVec_seq___redArg(v_x_24_, v_x_25_, v_x_26_, v_a_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_seq___boxed(lean_object* v_00_u03b1_29_, lean_object* v_00_u03b2_30_, lean_object* v_x_31_, lean_object* v_x_32_, lean_object* v_x_33_, lean_object* v_a_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_FinVec_seq(v_00_u03b1_29_, v_00_u03b2_30_, v_x_31_, v_x_32_, v_x_33_, v_a_34_);
lean_dec(v_a_34_);
lean_dec(v_x_31_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___redArg(lean_object* v_x_36_, lean_object* v_x_37_, lean_object* v_x_38_, lean_object* v_h__1_39_, lean_object* v_h__2_40_){
_start:
{
lean_object* v_zero_41_; uint8_t v_isZero_42_; 
v_zero_41_ = lean_unsigned_to_nat(0u);
v_isZero_42_ = lean_nat_dec_eq(v_x_36_, v_zero_41_);
if (v_isZero_42_ == 1)
{
lean_object* v___x_43_; 
lean_dec(v_h__2_40_);
v___x_43_ = lean_apply_2(v_h__1_39_, v_x_37_, v_x_38_);
return v___x_43_;
}
else
{
lean_object* v_one_44_; lean_object* v_n_45_; lean_object* v___x_46_; 
lean_dec(v_h__1_39_);
v_one_44_ = lean_unsigned_to_nat(1u);
v_n_45_ = lean_nat_sub(v_x_36_, v_one_44_);
v___x_46_ = lean_apply_3(v_h__2_40_, v_n_45_, v_x_37_, v_x_38_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___redArg___boxed(lean_object* v_x_47_, lean_object* v_x_48_, lean_object* v_x_49_, lean_object* v_h__1_50_, lean_object* v_h__2_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___redArg(v_x_47_, v_x_48_, v_x_49_, v_h__1_50_, v_h__2_51_);
lean_dec(v_x_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter(lean_object* v_00_u03b1_53_, lean_object* v_00_u03b2_54_, lean_object* v_motive_55_, lean_object* v_x_56_, lean_object* v_x_57_, lean_object* v_x_58_, lean_object* v_h__1_59_, lean_object* v_h__2_60_){
_start:
{
lean_object* v_zero_61_; uint8_t v_isZero_62_; 
v_zero_61_ = lean_unsigned_to_nat(0u);
v_isZero_62_ = lean_nat_dec_eq(v_x_56_, v_zero_61_);
if (v_isZero_62_ == 1)
{
lean_object* v___x_63_; 
lean_dec(v_h__2_60_);
v___x_63_ = lean_apply_2(v_h__1_59_, v_x_57_, v_x_58_);
return v___x_63_;
}
else
{
lean_object* v_one_64_; lean_object* v_n_65_; lean_object* v___x_66_; 
lean_dec(v_h__1_59_);
v_one_64_ = lean_unsigned_to_nat(1u);
v_n_65_ = lean_nat_sub(v_x_56_, v_one_64_);
v___x_66_ = lean_apply_3(v_h__2_60_, v_n_65_, v_x_57_, v_x_58_);
return v___x_66_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter___boxed(lean_object* v_00_u03b1_67_, lean_object* v_00_u03b2_68_, lean_object* v_motive_69_, lean_object* v_x_70_, lean_object* v_x_71_, lean_object* v_x_72_, lean_object* v_h__1_73_, lean_object* v_h__2_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_seq_match__1_splitter(v_00_u03b1_67_, v_00_u03b2_68_, v_motive_69_, v_x_70_, v_x_71_, v_x_72_, v_h__1_73_, v_h__2_74_);
lean_dec(v_x_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___lam__0(lean_object* v_f_76_, lean_object* v_x_77_, lean_object* v___y_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lean_apply_1(v_f_76_, v___y_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___lam__0___boxed(lean_object* v_f_80_, lean_object* v_x_81_, lean_object* v___y_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_FinVec_map___redArg___lam__0(v_f_80_, v_x_81_, v___y_82_);
lean_dec(v_x_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg(lean_object* v_f_84_, lean_object* v_m_85_, lean_object* v_a_86_, lean_object* v_a_87_){
_start:
{
lean_object* v___f_88_; lean_object* v___x_89_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_FinVec_map___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_88_, 0, v_f_84_);
v___x_89_ = lp_mathlib_FinVec_seq___redArg(v_m_85_, v___f_88_, v_a_86_, v_a_87_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___redArg___boxed(lean_object* v_f_90_, lean_object* v_m_91_, lean_object* v_a_92_, lean_object* v_a_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_FinVec_map___redArg(v_f_90_, v_m_91_, v_a_92_, v_a_93_);
lean_dec(v_a_93_);
lean_dec(v_m_91_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map(lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_f_97_, lean_object* v_m_98_, lean_object* v_a_99_, lean_object* v_a_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_FinVec_map___redArg(v_f_97_, v_m_98_, v_a_99_, v_a_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_map___boxed(lean_object* v_00_u03b1_102_, lean_object* v_00_u03b2_103_, lean_object* v_f_104_, lean_object* v_m_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_FinVec_map(v_00_u03b1_102_, v_00_u03b2_103_, v_f_104_, v_m_105_, v_a_106_, v_a_107_);
lean_dec(v_a_107_);
lean_dec(v_m_105_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___lam__0(lean_object* v___y_109_){
_start:
{
lean_inc(v___y_109_);
return v___y_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___lam__0___boxed(lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_FinVec_etaExpand___redArg___lam__0(v___y_110_);
lean_dec(v___y_110_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg(lean_object* v_m_113_, lean_object* v_v_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___f_116_; lean_object* v___x_117_; 
v___f_116_ = ((lean_object*)(lp_mathlib_FinVec_etaExpand___redArg___closed__0));
v___x_117_ = lp_mathlib_FinVec_map___redArg(v___f_116_, v_m_113_, v_v_114_, v_a_115_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___redArg___boxed(lean_object* v_m_118_, lean_object* v_v_119_, lean_object* v_a_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_FinVec_etaExpand___redArg(v_m_118_, v_v_119_, v_a_120_);
lean_dec(v_a_120_);
lean_dec(v_m_118_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand(lean_object* v_00_u03b1_122_, lean_object* v_m_123_, lean_object* v_v_124_, lean_object* v_a_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_FinVec_etaExpand___redArg(v_m_123_, v_v_124_, v_a_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_etaExpand___boxed(lean_object* v_00_u03b1_127_, lean_object* v_m_128_, lean_object* v_v_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_FinVec_etaExpand(v_00_u03b1_127_, v_m_128_, v_v_129_, v_a_130_);
lean_dec(v_a_130_);
lean_dec(v_m_128_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___redArg(lean_object* v_x_132_, lean_object* v_h__1_133_, lean_object* v_h__2_134_){
_start:
{
lean_object* v_zero_135_; uint8_t v_isZero_136_; 
v_zero_135_ = lean_unsigned_to_nat(0u);
v_isZero_136_ = lean_nat_dec_eq(v_x_132_, v_zero_135_);
if (v_isZero_136_ == 1)
{
lean_object* v___x_137_; 
lean_dec(v_h__2_134_);
v___x_137_ = lean_apply_1(v_h__1_133_, lean_box(0));
return v___x_137_;
}
else
{
lean_object* v_one_138_; lean_object* v_n_139_; lean_object* v___x_140_; 
lean_dec(v_h__1_133_);
v_one_138_ = lean_unsigned_to_nat(1u);
v_n_139_ = lean_nat_sub(v_x_132_, v_one_138_);
v___x_140_ = lean_apply_2(v_h__2_134_, v_n_139_, lean_box(0));
return v___x_140_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___redArg___boxed(lean_object* v_x_141_, lean_object* v_h__1_142_, lean_object* v_h__2_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___redArg(v_x_141_, v_h__1_142_, v_h__2_143_);
lean_dec(v_x_141_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter(lean_object* v_00_u03b1_145_, lean_object* v_motive_146_, lean_object* v_x_147_, lean_object* v_x_148_, lean_object* v_h__1_149_, lean_object* v_h__2_150_){
_start:
{
lean_object* v_zero_151_; uint8_t v_isZero_152_; 
v_zero_151_ = lean_unsigned_to_nat(0u);
v_isZero_152_ = lean_nat_dec_eq(v_x_147_, v_zero_151_);
if (v_isZero_152_ == 1)
{
lean_object* v___x_153_; 
lean_dec(v_h__2_150_);
v___x_153_ = lean_apply_1(v_h__1_149_, lean_box(0));
return v___x_153_;
}
else
{
lean_object* v_one_154_; lean_object* v_n_155_; lean_object* v___x_156_; 
lean_dec(v_h__1_149_);
v_one_154_ = lean_unsigned_to_nat(1u);
v_n_155_ = lean_nat_sub(v_x_147_, v_one_154_);
v___x_156_ = lean_apply_2(v_h__2_150_, v_n_155_, lean_box(0));
return v___x_156_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter___boxed(lean_object* v_00_u03b1_157_, lean_object* v_motive_158_, lean_object* v_x_159_, lean_object* v_x_160_, lean_object* v_h__1_161_, lean_object* v_h__2_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_Forall_match__1_splitter(v_00_u03b1_157_, v_motive_158_, v_x_159_, v_x_160_, v_h__1_161_, v_h__2_162_);
lean_dec(v_x_159_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg___lam__0(lean_object* v_x_164_, lean_object* v_i_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_apply_1(v_x_164_, v_i_165_);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_FinVec_sum___redArg___closed__0(void){
_start:
{
lean_object* v_one_167_; lean_object* v_zero_168_; lean_object* v___x_169_; 
v_one_167_ = lean_unsigned_to_nat(1u);
v_zero_168_ = lean_unsigned_to_nat(0u);
v___x_169_ = lean_nat_mod(v_zero_168_, v_one_167_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg(lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_x_172_, lean_object* v_x_173_){
_start:
{
lean_object* v_zero_174_; uint8_t v_isZero_175_; 
v_zero_174_ = lean_unsigned_to_nat(0u);
v_isZero_175_ = lean_nat_dec_eq(v_x_172_, v_zero_174_);
if (v_isZero_175_ == 1)
{
lean_dec(v_x_173_);
lean_dec(v_inst_170_);
lean_inc(v_inst_171_);
return v_inst_171_;
}
else
{
lean_object* v_one_176_; lean_object* v_n_177_; uint8_t v_isZero_178_; 
v_one_176_ = lean_unsigned_to_nat(1u);
v_n_177_ = lean_nat_sub(v_x_172_, v_one_176_);
v_isZero_178_ = lean_nat_dec_eq(v_n_177_, v_zero_174_);
if (v_isZero_178_ == 1)
{
lean_object* v___x_179_; lean_object* v___x_180_; 
lean_dec(v_n_177_);
lean_dec(v_inst_170_);
v___x_179_ = lean_obj_once(&lp_mathlib_FinVec_sum___redArg___closed__0, &lp_mathlib_FinVec_sum___redArg___closed__0_once, _init_lp_mathlib_FinVec_sum___redArg___closed__0);
v___x_180_ = lean_apply_1(v_x_173_, v___x_179_);
return v___x_180_;
}
else
{
lean_object* v___f_181_; lean_object* v_n_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
lean_inc(v_x_173_);
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_FinVec_sum___redArg___lam__0), 2, 1);
lean_closure_set(v___f_181_, 0, v_x_173_);
v_n_182_ = lean_nat_sub(v_n_177_, v_one_176_);
lean_dec(v_n_177_);
v___x_183_ = lean_nat_add(v_n_182_, v_one_176_);
lean_dec(v_n_182_);
lean_inc(v_inst_170_);
v___x_184_ = lp_mathlib_FinVec_sum___redArg(v_inst_170_, v_inst_171_, v___x_183_, v___f_181_);
v___x_185_ = lean_apply_1(v_x_173_, v___x_183_);
v___x_186_ = lean_apply_2(v_inst_170_, v___x_184_, v___x_185_);
return v___x_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___redArg___boxed(lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_x_189_, lean_object* v_x_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_FinVec_sum___redArg(v_inst_187_, v_inst_188_, v_x_189_, v_x_190_);
lean_dec(v_x_189_);
lean_dec(v_inst_188_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_x_195_, lean_object* v_x_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_FinVec_sum___redArg(v_inst_193_, v_inst_194_, v_x_195_, v_x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_sum___boxed(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_x_201_, lean_object* v_x_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_FinVec_sum(v_00_u03b1_198_, v_inst_199_, v_inst_200_, v_x_201_, v_x_202_);
lean_dec(v_x_201_);
lean_dec(v_inst_200_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___redArg(lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_x_206_, lean_object* v_x_207_){
_start:
{
lean_object* v_zero_208_; uint8_t v_isZero_209_; 
v_zero_208_ = lean_unsigned_to_nat(0u);
v_isZero_209_ = lean_nat_dec_eq(v_x_206_, v_zero_208_);
if (v_isZero_209_ == 1)
{
lean_dec(v_x_207_);
lean_dec(v_inst_204_);
lean_inc(v_inst_205_);
return v_inst_205_;
}
else
{
lean_object* v_one_210_; lean_object* v_n_211_; uint8_t v_isZero_212_; 
v_one_210_ = lean_unsigned_to_nat(1u);
v_n_211_ = lean_nat_sub(v_x_206_, v_one_210_);
v_isZero_212_ = lean_nat_dec_eq(v_n_211_, v_zero_208_);
if (v_isZero_212_ == 1)
{
lean_object* v___x_213_; lean_object* v___x_214_; 
lean_dec(v_n_211_);
lean_dec(v_inst_204_);
v___x_213_ = lean_obj_once(&lp_mathlib_FinVec_sum___redArg___closed__0, &lp_mathlib_FinVec_sum___redArg___closed__0_once, _init_lp_mathlib_FinVec_sum___redArg___closed__0);
v___x_214_ = lean_apply_1(v_x_207_, v___x_213_);
return v___x_214_;
}
else
{
lean_object* v___f_215_; lean_object* v_n_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
lean_inc(v_x_207_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_FinVec_sum___redArg___lam__0), 2, 1);
lean_closure_set(v___f_215_, 0, v_x_207_);
v_n_216_ = lean_nat_sub(v_n_211_, v_one_210_);
lean_dec(v_n_211_);
v___x_217_ = lean_nat_add(v_n_216_, v_one_210_);
lean_dec(v_n_216_);
lean_inc(v_inst_204_);
v___x_218_ = lp_mathlib_FinVec_prod___redArg(v_inst_204_, v_inst_205_, v___x_217_, v___f_215_);
v___x_219_ = lean_apply_1(v_x_207_, v___x_217_);
v___x_220_ = lean_apply_2(v_inst_204_, v___x_218_, v___x_219_);
return v___x_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___redArg___boxed(lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_x_223_, lean_object* v_x_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_FinVec_prod___redArg(v_inst_221_, v_inst_222_, v_x_223_, v_x_224_);
lean_dec(v_x_223_);
lean_dec(v_inst_222_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod(lean_object* v_00_u03b1_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_x_229_, lean_object* v_x_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_FinVec_prod___redArg(v_inst_227_, v_inst_228_, v_x_229_, v_x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_prod___boxed(lean_object* v_00_u03b1_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_x_235_, lean_object* v_x_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_FinVec_prod(v_00_u03b1_232_, v_inst_233_, v_inst_234_, v_x_235_, v_x_236_);
lean_dec(v_x_235_);
lean_dec(v_inst_234_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___redArg(lean_object* v_x_238_, lean_object* v_x_239_, lean_object* v_h__1_240_, lean_object* v_h__2_241_, lean_object* v_h__3_242_){
_start:
{
lean_object* v_zero_243_; uint8_t v_isZero_244_; 
v_zero_243_ = lean_unsigned_to_nat(0u);
v_isZero_244_ = lean_nat_dec_eq(v_x_238_, v_zero_243_);
if (v_isZero_244_ == 1)
{
lean_object* v___x_245_; 
lean_dec(v_h__3_242_);
lean_dec(v_h__2_241_);
v___x_245_ = lean_apply_1(v_h__1_240_, v_x_239_);
return v___x_245_;
}
else
{
lean_object* v_one_246_; lean_object* v_n_247_; uint8_t v_isZero_248_; 
lean_dec(v_h__1_240_);
v_one_246_ = lean_unsigned_to_nat(1u);
v_n_247_ = lean_nat_sub(v_x_238_, v_one_246_);
v_isZero_248_ = lean_nat_dec_eq(v_n_247_, v_zero_243_);
if (v_isZero_248_ == 1)
{
lean_object* v___x_249_; 
lean_dec(v_n_247_);
lean_dec(v_h__3_242_);
v___x_249_ = lean_apply_1(v_h__2_241_, v_x_239_);
return v___x_249_;
}
else
{
lean_object* v_n_250_; lean_object* v___x_251_; 
lean_dec(v_h__2_241_);
v_n_250_ = lean_nat_sub(v_n_247_, v_one_246_);
lean_dec(v_n_247_);
v___x_251_ = lean_apply_2(v_h__3_242_, v_n_250_, v_x_239_);
return v___x_251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___redArg___boxed(lean_object* v_x_252_, lean_object* v_x_253_, lean_object* v_h__1_254_, lean_object* v_h__2_255_, lean_object* v_h__3_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___redArg(v_x_252_, v_x_253_, v_h__1_254_, v_h__2_255_, v_h__3_256_);
lean_dec(v_x_252_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter(lean_object* v_00_u03b1_258_, lean_object* v_motive_259_, lean_object* v_x_260_, lean_object* v_x_261_, lean_object* v_h__1_262_, lean_object* v_h__2_263_, lean_object* v_h__3_264_){
_start:
{
lean_object* v_zero_265_; uint8_t v_isZero_266_; 
v_zero_265_ = lean_unsigned_to_nat(0u);
v_isZero_266_ = lean_nat_dec_eq(v_x_260_, v_zero_265_);
if (v_isZero_266_ == 1)
{
lean_object* v___x_267_; 
lean_dec(v_h__3_264_);
lean_dec(v_h__2_263_);
v___x_267_ = lean_apply_1(v_h__1_262_, v_x_261_);
return v___x_267_;
}
else
{
lean_object* v_one_268_; lean_object* v_n_269_; uint8_t v_isZero_270_; 
lean_dec(v_h__1_262_);
v_one_268_ = lean_unsigned_to_nat(1u);
v_n_269_ = lean_nat_sub(v_x_260_, v_one_268_);
v_isZero_270_ = lean_nat_dec_eq(v_n_269_, v_zero_265_);
if (v_isZero_270_ == 1)
{
lean_object* v___x_271_; 
lean_dec(v_n_269_);
lean_dec(v_h__3_264_);
v___x_271_ = lean_apply_1(v_h__2_263_, v_x_261_);
return v___x_271_;
}
else
{
lean_object* v_n_272_; lean_object* v___x_273_; 
lean_dec(v_h__2_263_);
v_n_272_ = lean_nat_sub(v_n_269_, v_one_268_);
lean_dec(v_n_269_);
v___x_273_ = lean_apply_2(v_h__3_264_, v_n_272_, v_x_261_);
return v___x_273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter___boxed(lean_object* v_00_u03b1_274_, lean_object* v_motive_275_, lean_object* v_x_276_, lean_object* v_x_277_, lean_object* v_h__1_278_, lean_object* v_h__2_279_, lean_object* v_h__3_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_sum_match__1_splitter(v_00_u03b1_274_, v_motive_275_, v_x_276_, v_x_277_, v_h__1_278_, v_h__2_279_, v_h__3_280_);
lean_dec(v_x_276_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0(lean_object* v_msgData_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; lean_object* v_env_289_; lean_object* v___x_290_; lean_object* v_mctx_291_; lean_object* v_lctx_292_; lean_object* v_options_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_288_ = lean_st_ref_get(v___y_286_);
v_env_289_ = lean_ctor_get(v___x_288_, 0);
lean_inc_ref(v_env_289_);
lean_dec(v___x_288_);
v___x_290_ = lean_st_ref_get(v___y_284_);
v_mctx_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc_ref(v_mctx_291_);
lean_dec(v___x_290_);
v_lctx_292_ = lean_ctor_get(v___y_283_, 2);
v_options_293_ = lean_ctor_get(v___y_285_, 2);
lean_inc_ref(v_options_293_);
lean_inc_ref(v_lctx_292_);
v___x_294_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_294_, 0, v_env_289_);
lean_ctor_set(v___x_294_, 1, v_mctx_291_);
lean_ctor_set(v___x_294_, 2, v_lctx_292_);
lean_ctor_set(v___x_294_, 3, v_options_293_);
v___x_295_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
lean_ctor_set(v___x_295_, 1, v_msgData_282_);
v___x_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0___boxed(lean_object* v_msgData_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0(v_msgData_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(lean_object* v_msg_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_ref_310_; lean_object* v___x_311_; lean_object* v_a_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_320_; 
v_ref_310_ = lean_ctor_get(v___y_307_, 5);
v___x_311_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0_spec__0(v_msg_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_);
v_a_312_ = lean_ctor_get(v___x_311_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v___x_311_);
if (v_isSharedCheck_320_ == 0)
{
v___x_314_ = v___x_311_;
v_isShared_315_ = v_isSharedCheck_320_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_a_312_);
lean_dec(v___x_311_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_320_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___x_316_; lean_object* v___x_318_; 
lean_inc(v_ref_310_);
v___x_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_316_, 0, v_ref_310_);
lean_ctor_set(v___x_316_, 1, v_a_312_);
if (v_isShared_315_ == 0)
{
lean_ctor_set_tag(v___x_314_, 1);
lean_ctor_set(v___x_314_, 0, v___x_316_);
v___x_318_ = v___x_314_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v___x_316_);
v___x_318_ = v_reuseFailAlloc_319_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
return v___x_318_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg___boxed(lean_object* v_msg_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(v_msg_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
return v_res_327_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1(void){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_329_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__0));
v___x_330_ = l_Lean_stringToMessageData(v___x_329_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23(void){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_367_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22));
v___x_368_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21));
v___x_369_ = l_Lean_Expr_const___override(v___x_368_, v___x_367_);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_373_ = lean_box(0);
v___x_374_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25));
v___x_375_ = l_Lean_Expr_const___override(v___x_374_, v___x_373_);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29(void){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_380_ = lean_box(0);
v___x_381_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__28));
v___x_382_ = l_Lean_Expr_const___override(v___x_381_, v___x_380_);
return v___x_382_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_385_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__30));
v___x_386_ = l_Lean_Expr_lit___override(v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS(lean_object* v_u_387_, lean_object* v_00_u03b1_388_, lean_object* v_inst_389_, lean_object* v_n_390_, lean_object* v_f_391_, lean_object* v_nezero_392_, lean_object* v_k_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_zero_399_; uint8_t v_isZero_400_; 
v_zero_399_ = lean_unsigned_to_nat(0u);
v_isZero_400_ = lean_nat_dec_eq(v_k_393_, v_zero_399_);
if (v_isZero_400_ == 1)
{
lean_object* v___x_401_; lean_object* v___x_402_; 
lean_dec_ref(v_nezero_392_);
lean_dec_ref(v_f_391_);
lean_dec(v_n_390_);
lean_dec_ref(v_inst_389_);
lean_dec_ref(v_00_u03b1_388_);
lean_dec(v_u_387_);
v___x_401_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1);
v___x_402_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(v___x_401_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
return v___x_402_;
}
else
{
lean_object* v_one_403_; lean_object* v_n_404_; uint8_t v___x_405_; 
v_one_403_ = lean_unsigned_to_nat(1u);
v_n_404_ = lean_nat_sub(v_k_393_, v_one_403_);
v___x_405_ = lean_nat_dec_eq(v_n_404_, v_zero_399_);
if (v___x_405_ == 0)
{
lean_object* v___x_406_; 
lean_inc_ref(v_nezero_392_);
lean_inc_ref(v_f_391_);
lean_inc(v_n_390_);
lean_inc_ref(v_inst_389_);
lean_inc_ref(v_00_u03b1_388_);
lean_inc(v_u_387_);
v___x_406_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS(v_u_387_, v_00_u03b1_388_, v_inst_389_, v_n_390_, v_f_391_, v_nezero_392_, v_n_404_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
if (lean_obj_tag(v___x_406_) == 0)
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_461_; 
v_a_407_ = lean_ctor_get(v___x_406_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_406_);
if (v_isSharedCheck_461_ == 0)
{
v___x_409_ = v___x_406_;
v_isShared_410_ = v_isSharedCheck_461_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_406_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_461_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_459_; 
v___x_411_ = l_Lean_mkRawNatLit(v_n_404_);
v___x_412_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__4));
v___x_413_ = lean_box(0);
lean_inc_n(v_u_387_, 2);
v___x_414_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_414_, 0, v_u_387_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
lean_inc_ref_n(v___x_414_, 5);
v___x_415_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_415_, 0, v_u_387_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
v___x_416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_416_, 0, v_u_387_);
lean_ctor_set(v___x_416_, 1, v___x_415_);
v___x_417_ = l_Lean_Expr_const___override(v___x_412_, v___x_416_);
lean_inc_ref_n(v_00_u03b1_388_, 7);
v___x_418_ = l_Lean_Expr_app___override(v___x_417_, v_00_u03b1_388_);
v___x_419_ = l_Lean_Expr_app___override(v___x_418_, v_00_u03b1_388_);
v___x_420_ = l_Lean_Expr_app___override(v___x_419_, v_00_u03b1_388_);
v___x_421_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__6));
v___x_422_ = l_Lean_Expr_const___override(v___x_421_, v___x_414_);
v___x_423_ = l_Lean_Expr_app___override(v___x_422_, v_00_u03b1_388_);
v___x_424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9));
v___x_425_ = l_Lean_Expr_const___override(v___x_424_, v___x_414_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v_00_u03b1_388_);
v___x_427_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12));
v___x_428_ = l_Lean_Expr_const___override(v___x_427_, v___x_414_);
v___x_429_ = l_Lean_Expr_app___override(v___x_428_, v_00_u03b1_388_);
v___x_430_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15));
v___x_431_ = l_Lean_Expr_const___override(v___x_430_, v___x_414_);
v___x_432_ = l_Lean_Expr_app___override(v___x_431_, v_00_u03b1_388_);
v___x_433_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18));
v___x_434_ = l_Lean_Expr_const___override(v___x_433_, v___x_414_);
v___x_435_ = l_Lean_Expr_app___override(v___x_434_, v_00_u03b1_388_);
v___x_436_ = l_Lean_Expr_app___override(v___x_435_, v_inst_389_);
v___x_437_ = l_Lean_Expr_app___override(v___x_432_, v___x_436_);
v___x_438_ = l_Lean_Expr_app___override(v___x_429_, v___x_437_);
v___x_439_ = l_Lean_Expr_app___override(v___x_426_, v___x_438_);
v___x_440_ = l_Lean_Expr_app___override(v___x_423_, v___x_439_);
v___x_441_ = l_Lean_Expr_app___override(v___x_420_, v___x_440_);
v___x_442_ = l_Lean_Expr_app___override(v___x_441_, v_a_407_);
v___x_443_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23);
v___x_444_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_445_ = l_Lean_mkNatLit(v_n_390_);
lean_inc_ref(v___x_445_);
v___x_446_ = l_Lean_Expr_app___override(v___x_444_, v___x_445_);
v___x_447_ = l_Lean_Expr_app___override(v___x_443_, v___x_446_);
lean_inc_ref(v___x_411_);
v___x_448_ = l_Lean_Expr_app___override(v___x_447_, v___x_411_);
v___x_449_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29);
v___x_450_ = l_Lean_Expr_app___override(v___x_449_, v___x_445_);
v___x_451_ = l_Lean_Expr_app___override(v___x_450_, v_nezero_392_);
v___x_452_ = l_Lean_Expr_app___override(v___x_451_, v___x_411_);
v___x_453_ = l_Lean_Expr_app___override(v___x_448_, v___x_452_);
v___x_454_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_454_, 0, v___x_453_);
lean_ctor_set(v___x_454_, 1, v___x_413_);
v___x_455_ = lean_array_mk(v___x_454_);
v___x_456_ = l_Lean_Expr_betaRev(v_f_391_, v___x_455_, v___x_405_, v___x_405_);
lean_dec_ref(v___x_455_);
v___x_457_ = l_Lean_Expr_app___override(v___x_442_, v___x_456_);
if (v_isShared_410_ == 0)
{
lean_ctor_set(v___x_409_, 0, v___x_457_);
v___x_459_ = v___x_409_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v___x_457_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
else
{
lean_dec(v_n_404_);
lean_dec_ref(v_nezero_392_);
lean_dec_ref(v_f_391_);
lean_dec(v_n_390_);
lean_dec_ref(v_inst_389_);
lean_dec_ref(v_00_u03b1_388_);
lean_dec(v_u_387_);
return v___x_406_;
}
}
else
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; 
lean_dec(v_n_404_);
lean_dec_ref(v_inst_389_);
lean_dec_ref(v_00_u03b1_388_);
lean_dec(v_u_387_);
v___x_462_ = lean_box(0);
v___x_463_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23);
v___x_464_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_465_ = l_Lean_mkNatLit(v_n_390_);
lean_inc_ref(v___x_465_);
v___x_466_ = l_Lean_Expr_app___override(v___x_464_, v___x_465_);
v___x_467_ = l_Lean_Expr_app___override(v___x_463_, v___x_466_);
v___x_468_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31);
v___x_469_ = l_Lean_Expr_app___override(v___x_467_, v___x_468_);
v___x_470_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29);
v___x_471_ = l_Lean_Expr_app___override(v___x_470_, v___x_465_);
v___x_472_ = l_Lean_Expr_app___override(v___x_471_, v_nezero_392_);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v___x_468_);
v___x_474_ = l_Lean_Expr_app___override(v___x_469_, v___x_473_);
v___x_475_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v___x_462_);
v___x_476_ = lean_array_mk(v___x_475_);
v___x_477_ = l_Lean_Expr_betaRev(v_f_391_, v___x_476_, v_isZero_400_, v_isZero_400_);
lean_dec_ref(v___x_476_);
v___x_478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
return v___x_478_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___boxed(lean_object* v_u_479_, lean_object* v_00_u03b1_480_, lean_object* v_inst_481_, lean_object* v_n_482_, lean_object* v_f_483_, lean_object* v_nezero_484_, lean_object* v_k_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_, lean_object* v_a_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS(v_u_479_, v_00_u03b1_480_, v_inst_481_, v_n_482_, v_f_483_, v_nezero_484_, v_k_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
lean_dec(v_a_489_);
lean_dec_ref(v_a_488_);
lean_dec(v_a_487_);
lean_dec_ref(v_a_486_);
lean_dec(v_k_485_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0(lean_object* v_00_u03b1_492_, lean_object* v_msg_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(v_msg_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___boxed(lean_object* v_00_u03b1_500_, lean_object* v_msg_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0(v_00_u03b1_500_, v_msg_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_);
lean_dec(v___y_505_);
lean_dec_ref(v___y_504_);
lean_dec(v___y_503_);
lean_dec_ref(v___y_502_);
return v_res_507_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__1(void){
_start:
{
lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_510_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__0));
v___x_511_ = l_Lean_Expr_lit___override(v___x_510_);
return v___x_511_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__12(void){
_start:
{
lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_529_ = lean_box(0);
v___x_530_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__11));
v___x_531_ = l_Lean_Expr_const___override(v___x_530_, v___x_529_);
return v___x_531_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__16(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_537_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22));
v___x_538_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__15));
v___x_539_ = l_Lean_Expr_const___override(v___x_538_, v___x_537_);
return v___x_539_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__17(void){
_start:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_540_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_541_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__16, &lp_mathlib_FinVec_mkProdEqQ___closed__16_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__16);
v___x_542_ = l_Lean_Expr_app___override(v___x_541_, v___x_540_);
return v___x_542_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__20(void){
_start:
{
lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; 
v___x_547_ = lean_box(0);
v___x_548_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__19));
v___x_549_ = l_Lean_Expr_const___override(v___x_548_, v___x_547_);
return v___x_549_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__21(void){
_start:
{
lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_550_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__20, &lp_mathlib_FinVec_mkProdEqQ___closed__20_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__20);
v___x_551_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__17, &lp_mathlib_FinVec_mkProdEqQ___closed__17_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__17);
v___x_552_ = l_Lean_Expr_app___override(v___x_551_, v___x_550_);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__27(void){
_start:
{
lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; 
v___x_564_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__26));
v___x_565_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__24));
v___x_566_ = l_Lean_Expr_const___override(v___x_565_, v___x_564_);
return v___x_566_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__28(void){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_567_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_568_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__27, &lp_mathlib_FinVec_mkProdEqQ___closed__27_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__27);
v___x_569_ = l_Lean_Expr_app___override(v___x_568_, v___x_567_);
return v___x_569_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__29(void){
_start:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_570_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_571_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__28, &lp_mathlib_FinVec_mkProdEqQ___closed__28_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__28);
v___x_572_ = l_Lean_Expr_app___override(v___x_571_, v___x_570_);
return v___x_572_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__30(void){
_start:
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; 
v___x_573_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_574_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__29, &lp_mathlib_FinVec_mkProdEqQ___closed__29_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__29);
v___x_575_ = l_Lean_Expr_app___override(v___x_574_, v___x_573_);
return v___x_575_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__33(void){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_579_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22));
v___x_580_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__32));
v___x_581_ = l_Lean_Expr_const___override(v___x_580_, v___x_579_);
return v___x_581_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__34(void){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_582_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_583_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__33, &lp_mathlib_FinVec_mkProdEqQ___closed__33_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__33);
v___x_584_ = l_Lean_Expr_app___override(v___x_583_, v___x_582_);
return v___x_584_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__37(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_588_ = lean_box(0);
v___x_589_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__36));
v___x_590_ = l_Lean_Expr_const___override(v___x_589_, v___x_588_);
return v___x_590_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__38(void){
_start:
{
lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_591_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__37, &lp_mathlib_FinVec_mkProdEqQ___closed__37_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__37);
v___x_592_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__34, &lp_mathlib_FinVec_mkProdEqQ___closed__34_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__34);
v___x_593_ = l_Lean_Expr_app___override(v___x_592_, v___x_591_);
return v___x_593_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__39(void){
_start:
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_594_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__38, &lp_mathlib_FinVec_mkProdEqQ___closed__38_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__38);
v___x_595_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__30, &lp_mathlib_FinVec_mkProdEqQ___closed__30_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__30);
v___x_596_ = l_Lean_Expr_app___override(v___x_595_, v___x_594_);
return v___x_596_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__40(void){
_start:
{
lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; 
v___x_597_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_598_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23);
v___x_599_ = l_Lean_Expr_app___override(v___x_598_, v___x_597_);
return v___x_599_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__41(void){
_start:
{
lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v___x_600_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__1, &lp_mathlib_FinVec_mkProdEqQ___closed__1_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__1);
v___x_601_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__40, &lp_mathlib_FinVec_mkProdEqQ___closed__40_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__40);
v___x_602_ = l_Lean_Expr_app___override(v___x_601_, v___x_600_);
return v___x_602_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__44(void){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_606_ = lean_box(0);
v___x_607_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__43));
v___x_608_ = l_Lean_Expr_const___override(v___x_607_, v___x_606_);
return v___x_608_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__45(void){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__1, &lp_mathlib_FinVec_mkProdEqQ___closed__1_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__1);
v___x_610_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__44, &lp_mathlib_FinVec_mkProdEqQ___closed__44_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__44);
v___x_611_ = l_Lean_Expr_app___override(v___x_610_, v___x_609_);
return v___x_611_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__46(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_612_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__45, &lp_mathlib_FinVec_mkProdEqQ___closed__45_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__45);
v___x_613_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__41, &lp_mathlib_FinVec_mkProdEqQ___closed__41_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__41);
v___x_614_ = l_Lean_Expr_app___override(v___x_613_, v___x_612_);
return v___x_614_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__49(void){
_start:
{
lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_619_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22));
v___x_620_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__48));
v___x_621_ = l_Lean_Expr_const___override(v___x_620_, v___x_619_);
return v___x_621_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__50(void){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_622_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_623_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__49, &lp_mathlib_FinVec_mkProdEqQ___closed__49_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__49);
v___x_624_ = l_Lean_Expr_app___override(v___x_623_, v___x_622_);
return v___x_624_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__51(void){
_start:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_625_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__21, &lp_mathlib_FinVec_mkProdEqQ___closed__21_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__21);
v___x_626_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__50, &lp_mathlib_FinVec_mkProdEqQ___closed__50_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__50);
v___x_627_ = l_Lean_Expr_app___override(v___x_626_, v___x_625_);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__54(void){
_start:
{
lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_632_ = lean_box(0);
v___x_633_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__53));
v___x_634_ = l_Lean_Expr_const___override(v___x_633_, v___x_632_);
return v___x_634_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__61(void){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_645_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__22));
v___x_646_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__60));
v___x_647_ = l_Lean_Expr_const___override(v___x_646_, v___x_645_);
return v___x_647_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__64(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_box(0);
v___x_653_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__63));
v___x_654_ = l_Lean_Expr_const___override(v___x_653_, v___x_652_);
return v___x_654_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__67(void){
_start:
{
lean_object* v_zero_658_; lean_object* v___x_659_; 
v_zero_658_ = lean_unsigned_to_nat(0u);
v___x_659_ = l_Lean_Expr_bvar___override(v_zero_658_);
return v___x_659_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__68(void){
_start:
{
lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_660_ = lean_box(0);
v___x_661_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__67, &lp_mathlib_FinVec_mkProdEqQ___closed__67_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__67);
v___x_662_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_662_, 0, v___x_661_);
lean_ctor_set(v___x_662_, 1, v___x_660_);
return v___x_662_;
}
}
static lean_object* _init_lp_mathlib_FinVec_mkProdEqQ___closed__69(void){
_start:
{
lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_663_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__68, &lp_mathlib_FinVec_mkProdEqQ___closed__68_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__68);
v___x_664_ = lean_array_mk(v___x_663_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkProdEqQ(lean_object* v_u_677_, lean_object* v_00_u03b1_678_, lean_object* v_inst_679_, lean_object* v_n_680_, lean_object* v_f_681_, lean_object* v_a_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_){
_start:
{
lean_object* v_zero_687_; uint8_t v_isZero_688_; 
v_zero_687_ = lean_unsigned_to_nat(0u);
v_isZero_688_ = lean_nat_dec_eq(v_n_680_, v_zero_687_);
if (v_isZero_688_ == 1)
{
lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_689_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21));
v___x_690_ = lean_box(0);
v___x_691_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_691_, 0, v_u_677_);
lean_ctor_set(v___x_691_, 1, v___x_690_);
lean_inc_ref_n(v___x_691_, 6);
v___x_692_ = l_Lean_Expr_const___override(v___x_689_, v___x_691_);
lean_inc_ref_n(v_00_u03b1_678_, 6);
v___x_693_ = l_Lean_Expr_app___override(v___x_692_, v_00_u03b1_678_);
v___x_694_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__1, &lp_mathlib_FinVec_mkProdEqQ___closed__1_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__1);
v___x_695_ = l_Lean_Expr_app___override(v___x_693_, v___x_694_);
v___x_696_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__4));
v___x_697_ = l_Lean_Expr_const___override(v___x_696_, v___x_691_);
v___x_698_ = l_Lean_Expr_app___override(v___x_697_, v_00_u03b1_678_);
v___x_699_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__6));
v___x_700_ = l_Lean_Expr_const___override(v___x_699_, v___x_691_);
v___x_701_ = l_Lean_Expr_app___override(v___x_700_, v_00_u03b1_678_);
v___x_702_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12));
v___x_703_ = l_Lean_Expr_const___override(v___x_702_, v___x_691_);
v___x_704_ = l_Lean_Expr_app___override(v___x_703_, v_00_u03b1_678_);
v___x_705_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15));
v___x_706_ = l_Lean_Expr_const___override(v___x_705_, v___x_691_);
v___x_707_ = l_Lean_Expr_app___override(v___x_706_, v_00_u03b1_678_);
v___x_708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18));
v___x_709_ = l_Lean_Expr_const___override(v___x_708_, v___x_691_);
v___x_710_ = l_Lean_Expr_app___override(v___x_709_, v_00_u03b1_678_);
lean_inc_ref(v_inst_679_);
v___x_711_ = l_Lean_Expr_app___override(v___x_710_, v_inst_679_);
v___x_712_ = l_Lean_Expr_app___override(v___x_707_, v___x_711_);
v___x_713_ = l_Lean_Expr_app___override(v___x_704_, v___x_712_);
v___x_714_ = l_Lean_Expr_app___override(v___x_701_, v___x_713_);
v___x_715_ = l_Lean_Expr_app___override(v___x_698_, v___x_714_);
v___x_716_ = l_Lean_Expr_app___override(v___x_695_, v___x_715_);
v___x_717_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__8));
v___x_718_ = l_Lean_Expr_const___override(v___x_717_, v___x_691_);
v___x_719_ = l_Lean_Expr_app___override(v___x_718_, v_00_u03b1_678_);
v___x_720_ = l_Lean_Expr_app___override(v___x_719_, v_inst_679_);
v___x_721_ = l_Lean_Expr_app___override(v___x_720_, v_f_681_);
v___x_722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_722_, 0, v___x_716_);
lean_ctor_set(v___x_722_, 1, v___x_721_);
v___x_723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_723_, 0, v___x_722_);
return v___x_723_;
}
else
{
lean_object* v_one_724_; lean_object* v_n_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v_nezero_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v_one_724_ = lean_unsigned_to_nat(1u);
v_n_725_ = lean_nat_sub(v_n_680_, v_one_724_);
v___x_726_ = lean_box(0);
v___x_727_ = lean_box(0);
v___x_728_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__39, &lp_mathlib_FinVec_mkProdEqQ___closed__39_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__39);
lean_inc(v_n_725_);
v___x_729_ = l_Lean_mkNatLit(v_n_725_);
lean_inc_ref(v___x_729_);
v___x_730_ = l_Lean_Expr_app___override(v___x_728_, v___x_729_);
v___x_731_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__46, &lp_mathlib_FinVec_mkProdEqQ___closed__46_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__46);
v___x_732_ = l_Lean_Expr_app___override(v___x_730_, v___x_731_);
v___x_733_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__51, &lp_mathlib_FinVec_mkProdEqQ___closed__51_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__51);
lean_inc_ref(v___x_732_);
v___x_734_ = l_Lean_Expr_app___override(v___x_733_, v___x_732_);
v___x_735_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__54, &lp_mathlib_FinVec_mkProdEqQ___closed__54_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__54);
v___x_736_ = l_Lean_Expr_app___override(v___x_735_, v___x_729_);
v_nezero_737_ = l_Lean_Expr_app___override(v___x_734_, v___x_736_);
v___x_738_ = lean_nat_add(v_n_725_, v_one_724_);
lean_dec(v_n_725_);
lean_inc_ref(v_f_681_);
lean_inc(v___x_738_);
lean_inc_ref(v_inst_679_);
lean_inc_ref(v_00_u03b1_678_);
lean_inc(v_u_677_);
v___x_739_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS(v_u_677_, v_00_u03b1_678_, v_inst_679_, v___x_738_, v_f_681_, v_nezero_737_, v___x_738_, v_a_682_, v_a_683_, v_a_684_, v_a_685_);
lean_dec(v___x_738_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_810_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_810_ == 0)
{
v___x_742_ = v___x_739_;
v_isShared_743_ = v_isSharedCheck_810_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_dec(v___x_739_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_810_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; uint8_t v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_808_; 
lean_inc(v_u_677_);
v___x_744_ = l_Lean_Level_succ___override(v_u_677_);
v___x_745_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set(v___x_745_, 1, v___x_727_);
v___x_746_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__58));
v___x_747_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_747_, 0, v_u_677_);
lean_ctor_set(v___x_747_, 1, v___x_727_);
lean_inc_ref_n(v___x_747_, 7);
v___x_748_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_748_, 0, v___x_726_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
v___x_749_ = l_Lean_Expr_const___override(v___x_746_, v___x_748_);
v___x_750_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_751_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__61, &lp_mathlib_FinVec_mkProdEqQ___closed__61_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__61);
v___x_752_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__64, &lp_mathlib_FinVec_mkProdEqQ___closed__64_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__64);
v___x_753_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__66));
v___x_754_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__69, &lp_mathlib_FinVec_mkProdEqQ___closed__69_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__69);
lean_inc_ref_n(v_f_681_, 2);
v___x_755_ = l_Lean_Expr_betaRev(v_f_681_, v___x_754_, v_isZero_688_, v_isZero_688_);
v___x_756_ = 0;
v___x_757_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__71));
v___x_758_ = l_Lean_Expr_const___override(v___x_757_, v___x_745_);
lean_inc_ref_n(v_00_u03b1_678_, 8);
v___x_759_ = l_Lean_Expr_app___override(v___x_758_, v_00_u03b1_678_);
v___x_760_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__73));
v___x_761_ = l_Lean_Expr_const___override(v___x_760_, v___x_747_);
v___x_762_ = l_Lean_Expr_app___override(v___x_761_, v_00_u03b1_678_);
v___x_763_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__9));
v___x_764_ = l_Lean_Expr_const___override(v___x_763_, v___x_747_);
v___x_765_ = l_Lean_Expr_app___override(v___x_764_, v_00_u03b1_678_);
v___x_766_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__12));
v___x_767_ = l_Lean_Expr_const___override(v___x_766_, v___x_747_);
v___x_768_ = l_Lean_Expr_app___override(v___x_767_, v_00_u03b1_678_);
v___x_769_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__15));
v___x_770_ = l_Lean_Expr_const___override(v___x_769_, v___x_747_);
v___x_771_ = l_Lean_Expr_app___override(v___x_770_, v_00_u03b1_678_);
v___x_772_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__18));
v___x_773_ = l_Lean_Expr_const___override(v___x_772_, v___x_747_);
v___x_774_ = l_Lean_Expr_app___override(v___x_773_, v_00_u03b1_678_);
lean_inc_ref_n(v_inst_679_, 2);
v___x_775_ = l_Lean_Expr_app___override(v___x_774_, v_inst_679_);
v___x_776_ = l_Lean_Expr_app___override(v___x_771_, v___x_775_);
v___x_777_ = l_Lean_Expr_app___override(v___x_768_, v___x_776_);
lean_inc_ref(v___x_777_);
v___x_778_ = l_Lean_Expr_app___override(v___x_765_, v___x_777_);
v___x_779_ = l_Lean_Expr_app___override(v___x_762_, v___x_778_);
v___x_780_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__6));
v___x_781_ = l_Lean_Expr_const___override(v___x_780_, v___x_747_);
v___x_782_ = l_Lean_Expr_app___override(v___x_781_, v_00_u03b1_678_);
v___x_783_ = l_Lean_Expr_app___override(v___x_782_, v___x_777_);
v___x_784_ = l_Lean_Expr_app___override(v___x_779_, v___x_783_);
lean_inc_ref_n(v___x_732_, 3);
v___x_785_ = l_Lean_Expr_app___override(v___x_784_, v___x_732_);
v___x_786_ = l_Lean_Expr_app___override(v___x_785_, v_f_681_);
v___x_787_ = l_Lean_Expr_app___override(v___x_759_, v___x_786_);
v___x_788_ = l_Lean_Expr_app___override(v___x_750_, v___x_732_);
lean_inc_ref_n(v___x_788_, 2);
v___x_789_ = l_Lean_Expr_app___override(v___x_749_, v___x_788_);
v___x_790_ = l_Lean_Expr_app___override(v___x_789_, v_00_u03b1_678_);
v___x_791_ = l_Lean_Expr_app___override(v___x_790_, v_inst_679_);
v___x_792_ = l_Lean_Expr_app___override(v___x_751_, v___x_788_);
v___x_793_ = l_Lean_Expr_app___override(v___x_752_, v___x_732_);
v___x_794_ = l_Lean_Expr_app___override(v___x_792_, v___x_793_);
v___x_795_ = l_Lean_Expr_app___override(v___x_791_, v___x_794_);
v___x_796_ = l_Lean_Expr_lam___override(v___x_753_, v___x_788_, v___x_755_, v___x_756_);
v___x_797_ = l_Lean_Expr_app___override(v___x_795_, v___x_796_);
v___x_798_ = l_Lean_Expr_app___override(v___x_787_, v___x_797_);
v___x_799_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__75));
v___x_800_ = l_Lean_Expr_const___override(v___x_799_, v___x_747_);
v___x_801_ = l_Lean_Expr_app___override(v___x_800_, v_00_u03b1_678_);
v___x_802_ = l_Lean_Expr_app___override(v___x_801_, v_inst_679_);
v___x_803_ = l_Lean_Expr_app___override(v___x_802_, v___x_732_);
v___x_804_ = l_Lean_Expr_app___override(v___x_803_, v_f_681_);
v___x_805_ = l_Lean_Expr_app___override(v___x_798_, v___x_804_);
v___x_806_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_806_, 0, v_a_740_);
lean_ctor_set(v___x_806_, 1, v___x_805_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 0, v___x_806_);
v___x_808_ = v___x_742_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v___x_806_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
else
{
lean_object* v_a_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_818_; 
lean_dec_ref(v___x_732_);
lean_dec_ref(v_f_681_);
lean_dec_ref(v_inst_679_);
lean_dec_ref(v_00_u03b1_678_);
lean_dec(v_u_677_);
v_a_811_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_818_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_818_ == 0)
{
v___x_813_ = v___x_739_;
v_isShared_814_ = v_isSharedCheck_818_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_a_811_);
lean_dec(v___x_739_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_818_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v___x_816_; 
if (v_isShared_814_ == 0)
{
v___x_816_ = v___x_813_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v_a_811_);
v___x_816_ = v_reuseFailAlloc_817_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
return v___x_816_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkProdEqQ___boxed(lean_object* v_u_819_, lean_object* v_00_u03b1_820_, lean_object* v_inst_821_, lean_object* v_n_822_, lean_object* v_f_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_mathlib_FinVec_mkProdEqQ(v_u_819_, v_00_u03b1_820_, v_inst_821_, v_n_822_, v_f_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_);
lean_dec(v_a_827_);
lean_dec_ref(v_a_826_);
lean_dec(v_a_825_);
lean_dec_ref(v_a_824_);
lean_dec(v_n_822_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS(lean_object* v_u_845_, lean_object* v_00_u03b1_846_, lean_object* v_inst_847_, lean_object* v_n_848_, lean_object* v_f_849_, lean_object* v_nezero_850_, lean_object* v_k_851_, lean_object* v_a_852_, lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_){
_start:
{
lean_object* v_zero_857_; uint8_t v_isZero_858_; 
v_zero_857_ = lean_unsigned_to_nat(0u);
v_isZero_858_ = lean_nat_dec_eq(v_k_851_, v_zero_857_);
if (v_isZero_858_ == 1)
{
lean_object* v___x_859_; lean_object* v___x_860_; 
lean_dec_ref(v_nezero_850_);
lean_dec_ref(v_f_849_);
lean_dec(v_n_848_);
lean_dec_ref(v_inst_847_);
lean_dec_ref(v_00_u03b1_846_);
lean_dec(v_u_845_);
v___x_859_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__1);
v___x_860_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS_spec__0___redArg(v___x_859_, v_a_852_, v_a_853_, v_a_854_, v_a_855_);
return v___x_860_;
}
else
{
lean_object* v_one_861_; lean_object* v_n_862_; uint8_t v___x_863_; 
v_one_861_ = lean_unsigned_to_nat(1u);
v_n_862_ = lean_nat_sub(v_k_851_, v_one_861_);
v___x_863_ = lean_nat_dec_eq(v_n_862_, v_zero_857_);
if (v___x_863_ == 0)
{
lean_object* v___x_864_; 
lean_inc_ref(v_nezero_850_);
lean_inc_ref(v_f_849_);
lean_inc(v_n_848_);
lean_inc_ref(v_inst_847_);
lean_inc_ref(v_00_u03b1_846_);
lean_inc(v_u_845_);
v___x_864_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS(v_u_845_, v_00_u03b1_846_, v_inst_847_, v_n_848_, v_f_849_, v_nezero_850_, v_n_862_, v_a_852_, v_a_853_, v_a_854_, v_a_855_);
if (lean_obj_tag(v___x_864_) == 0)
{
lean_object* v_a_865_; lean_object* v___x_867_; uint8_t v_isShared_868_; uint8_t v_isSharedCheck_915_; 
v_a_865_ = lean_ctor_get(v___x_864_, 0);
v_isSharedCheck_915_ = !lean_is_exclusive(v___x_864_);
if (v_isSharedCheck_915_ == 0)
{
v___x_867_ = v___x_864_;
v_isShared_868_ = v_isSharedCheck_915_;
goto v_resetjp_866_;
}
else
{
lean_inc(v_a_865_);
lean_dec(v___x_864_);
v___x_867_ = lean_box(0);
v_isShared_868_ = v_isSharedCheck_915_;
goto v_resetjp_866_;
}
v_resetjp_866_:
{
lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_913_; 
v___x_869_ = l_Lean_mkRawNatLit(v_n_862_);
v___x_870_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__24));
v___x_871_ = lean_box(0);
lean_inc_n(v_u_845_, 2);
v___x_872_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_872_, 0, v_u_845_);
lean_ctor_set(v___x_872_, 1, v___x_871_);
lean_inc_ref_n(v___x_872_, 4);
v___x_873_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_873_, 0, v_u_845_);
lean_ctor_set(v___x_873_, 1, v___x_872_);
v___x_874_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_874_, 0, v_u_845_);
lean_ctor_set(v___x_874_, 1, v___x_873_);
v___x_875_ = l_Lean_Expr_const___override(v___x_870_, v___x_874_);
lean_inc_ref_n(v_00_u03b1_846_, 6);
v___x_876_ = l_Lean_Expr_app___override(v___x_875_, v_00_u03b1_846_);
v___x_877_ = l_Lean_Expr_app___override(v___x_876_, v_00_u03b1_846_);
v___x_878_ = l_Lean_Expr_app___override(v___x_877_, v_00_u03b1_846_);
v___x_879_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__32));
v___x_880_ = l_Lean_Expr_const___override(v___x_879_, v___x_872_);
v___x_881_ = l_Lean_Expr_app___override(v___x_880_, v_00_u03b1_846_);
v___x_882_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__2));
v___x_883_ = l_Lean_Expr_const___override(v___x_882_, v___x_872_);
v___x_884_ = l_Lean_Expr_app___override(v___x_883_, v_00_u03b1_846_);
v___x_885_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__5));
v___x_886_ = l_Lean_Expr_const___override(v___x_885_, v___x_872_);
v___x_887_ = l_Lean_Expr_app___override(v___x_886_, v_00_u03b1_846_);
v___x_888_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___closed__8));
v___x_889_ = l_Lean_Expr_const___override(v___x_888_, v___x_872_);
v___x_890_ = l_Lean_Expr_app___override(v___x_889_, v_00_u03b1_846_);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v_inst_847_);
v___x_892_ = l_Lean_Expr_app___override(v___x_887_, v___x_891_);
v___x_893_ = l_Lean_Expr_app___override(v___x_884_, v___x_892_);
v___x_894_ = l_Lean_Expr_app___override(v___x_881_, v___x_893_);
v___x_895_ = l_Lean_Expr_app___override(v___x_878_, v___x_894_);
v___x_896_ = l_Lean_Expr_app___override(v___x_895_, v_a_865_);
v___x_897_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23);
v___x_898_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_899_ = l_Lean_mkNatLit(v_n_848_);
lean_inc_ref(v___x_899_);
v___x_900_ = l_Lean_Expr_app___override(v___x_898_, v___x_899_);
v___x_901_ = l_Lean_Expr_app___override(v___x_897_, v___x_900_);
lean_inc_ref(v___x_869_);
v___x_902_ = l_Lean_Expr_app___override(v___x_901_, v___x_869_);
v___x_903_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29);
v___x_904_ = l_Lean_Expr_app___override(v___x_903_, v___x_899_);
v___x_905_ = l_Lean_Expr_app___override(v___x_904_, v_nezero_850_);
v___x_906_ = l_Lean_Expr_app___override(v___x_905_, v___x_869_);
v___x_907_ = l_Lean_Expr_app___override(v___x_902_, v___x_906_);
v___x_908_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v___x_871_);
v___x_909_ = lean_array_mk(v___x_908_);
v___x_910_ = l_Lean_Expr_betaRev(v_f_849_, v___x_909_, v___x_863_, v___x_863_);
lean_dec_ref(v___x_909_);
v___x_911_ = l_Lean_Expr_app___override(v___x_896_, v___x_910_);
if (v_isShared_868_ == 0)
{
lean_ctor_set(v___x_867_, 0, v___x_911_);
v___x_913_ = v___x_867_;
goto v_reusejp_912_;
}
else
{
lean_object* v_reuseFailAlloc_914_; 
v_reuseFailAlloc_914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_914_, 0, v___x_911_);
v___x_913_ = v_reuseFailAlloc_914_;
goto v_reusejp_912_;
}
v_reusejp_912_:
{
return v___x_913_;
}
}
}
else
{
lean_dec(v_n_862_);
lean_dec_ref(v_nezero_850_);
lean_dec_ref(v_f_849_);
lean_dec(v_n_848_);
lean_dec_ref(v_inst_847_);
lean_dec_ref(v_00_u03b1_846_);
lean_dec(v_u_845_);
return v___x_864_;
}
}
else
{
lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
lean_dec(v_n_862_);
lean_dec_ref(v_inst_847_);
lean_dec_ref(v_00_u03b1_846_);
lean_dec(v_u_845_);
v___x_916_ = lean_box(0);
v___x_917_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__23);
v___x_918_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_919_ = l_Lean_mkNatLit(v_n_848_);
lean_inc_ref(v___x_919_);
v___x_920_ = l_Lean_Expr_app___override(v___x_918_, v___x_919_);
v___x_921_ = l_Lean_Expr_app___override(v___x_917_, v___x_920_);
v___x_922_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31);
v___x_923_ = l_Lean_Expr_app___override(v___x_921_, v___x_922_);
v___x_924_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__29);
v___x_925_ = l_Lean_Expr_app___override(v___x_924_, v___x_919_);
v___x_926_ = l_Lean_Expr_app___override(v___x_925_, v_nezero_850_);
v___x_927_ = l_Lean_Expr_app___override(v___x_926_, v___x_922_);
v___x_928_ = l_Lean_Expr_app___override(v___x_923_, v___x_927_);
v___x_929_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_929_, 0, v___x_928_);
lean_ctor_set(v___x_929_, 1, v___x_916_);
v___x_930_ = lean_array_mk(v___x_929_);
v___x_931_ = l_Lean_Expr_betaRev(v_f_849_, v___x_930_, v_isZero_858_, v_isZero_858_);
lean_dec_ref(v___x_930_);
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v___x_931_);
return v___x_932_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS___boxed(lean_object* v_u_933_, lean_object* v_00_u03b1_934_, lean_object* v_inst_935_, lean_object* v_n_936_, lean_object* v_f_937_, lean_object* v_nezero_938_, lean_object* v_k_939_, lean_object* v_a_940_, lean_object* v_a_941_, lean_object* v_a_942_, lean_object* v_a_943_, lean_object* v_a_944_){
_start:
{
lean_object* v_res_945_; 
v_res_945_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS(v_u_933_, v_00_u03b1_934_, v_inst_935_, v_n_936_, v_f_937_, v_nezero_938_, v_k_939_, v_a_940_, v_a_941_, v_a_942_, v_a_943_);
lean_dec(v_a_943_);
lean_dec_ref(v_a_942_);
lean_dec(v_a_941_);
lean_dec_ref(v_a_940_);
lean_dec(v_k_939_);
return v_res_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkSumEqQ(lean_object* v_u_987_, lean_object* v_00_u03b1_988_, lean_object* v_inst_989_, lean_object* v_n_990_, lean_object* v_f_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_){
_start:
{
lean_object* v_zero_997_; uint8_t v_isZero_998_; 
v_zero_997_ = lean_unsigned_to_nat(0u);
v_isZero_998_ = lean_nat_dec_eq(v_n_990_, v_zero_997_);
if (v_isZero_998_ == 1)
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; 
v___x_999_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__21));
v___x_1000_ = lean_box(0);
v___x_1001_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1001_, 0, v_u_987_);
lean_ctor_set(v___x_1001_, 1, v___x_1000_);
lean_inc_ref_n(v___x_1001_, 6);
v___x_1002_ = l_Lean_Expr_const___override(v___x_999_, v___x_1001_);
lean_inc_ref_n(v_00_u03b1_988_, 6);
v___x_1003_ = l_Lean_Expr_app___override(v___x_1002_, v_00_u03b1_988_);
v___x_1004_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__31);
v___x_1005_ = l_Lean_Expr_app___override(v___x_1003_, v___x_1004_);
v___x_1006_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__2));
v___x_1007_ = l_Lean_Expr_const___override(v___x_1006_, v___x_1001_);
v___x_1008_ = l_Lean_Expr_app___override(v___x_1007_, v_00_u03b1_988_);
v___x_1009_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__4));
v___x_1010_ = l_Lean_Expr_const___override(v___x_1009_, v___x_1001_);
v___x_1011_ = l_Lean_Expr_app___override(v___x_1010_, v_00_u03b1_988_);
v___x_1012_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__7));
v___x_1013_ = l_Lean_Expr_const___override(v___x_1012_, v___x_1001_);
v___x_1014_ = l_Lean_Expr_app___override(v___x_1013_, v_00_u03b1_988_);
v___x_1015_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__10));
v___x_1016_ = l_Lean_Expr_const___override(v___x_1015_, v___x_1001_);
v___x_1017_ = l_Lean_Expr_app___override(v___x_1016_, v_00_u03b1_988_);
v___x_1018_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__12));
v___x_1019_ = l_Lean_Expr_const___override(v___x_1018_, v___x_1001_);
v___x_1020_ = l_Lean_Expr_app___override(v___x_1019_, v_00_u03b1_988_);
lean_inc_ref(v_inst_989_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_inst_989_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1017_, v___x_1021_);
v___x_1023_ = l_Lean_Expr_app___override(v___x_1014_, v___x_1022_);
v___x_1024_ = l_Lean_Expr_app___override(v___x_1011_, v___x_1023_);
v___x_1025_ = l_Lean_Expr_app___override(v___x_1008_, v___x_1024_);
v___x_1026_ = l_Lean_Expr_app___override(v___x_1005_, v___x_1025_);
v___x_1027_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__14));
v___x_1028_ = l_Lean_Expr_const___override(v___x_1027_, v___x_1001_);
v___x_1029_ = l_Lean_Expr_app___override(v___x_1028_, v_00_u03b1_988_);
v___x_1030_ = l_Lean_Expr_app___override(v___x_1029_, v_inst_989_);
v___x_1031_ = l_Lean_Expr_app___override(v___x_1030_, v_f_991_);
v___x_1032_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1026_);
lean_ctor_set(v___x_1032_, 1, v___x_1031_);
v___x_1033_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1032_);
return v___x_1033_;
}
else
{
lean_object* v_one_1034_; lean_object* v_n_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v_nezero_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; 
v_one_1034_ = lean_unsigned_to_nat(1u);
v_n_1035_ = lean_nat_sub(v_n_990_, v_one_1034_);
v___x_1036_ = lean_box(0);
v___x_1037_ = lean_box(0);
v___x_1038_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__39, &lp_mathlib_FinVec_mkProdEqQ___closed__39_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__39);
lean_inc(v_n_1035_);
v___x_1039_ = l_Lean_mkNatLit(v_n_1035_);
lean_inc_ref(v___x_1039_);
v___x_1040_ = l_Lean_Expr_app___override(v___x_1038_, v___x_1039_);
v___x_1041_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__46, &lp_mathlib_FinVec_mkProdEqQ___closed__46_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__46);
v___x_1042_ = l_Lean_Expr_app___override(v___x_1040_, v___x_1041_);
v___x_1043_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__51, &lp_mathlib_FinVec_mkProdEqQ___closed__51_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__51);
lean_inc_ref(v___x_1042_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1043_, v___x_1042_);
v___x_1045_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__54, &lp_mathlib_FinVec_mkProdEqQ___closed__54_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__54);
v___x_1046_ = l_Lean_Expr_app___override(v___x_1045_, v___x_1039_);
v_nezero_1047_ = l_Lean_Expr_app___override(v___x_1044_, v___x_1046_);
v___x_1048_ = lean_nat_add(v_n_1035_, v_one_1034_);
lean_dec(v_n_1035_);
lean_inc_ref(v_f_991_);
lean_inc(v___x_1048_);
lean_inc_ref(v_inst_989_);
lean_inc_ref(v_00_u03b1_988_);
lean_inc(v_u_987_);
v___x_1049_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkSumEqQ_makeRHS(v_u_987_, v_00_u03b1_988_, v_inst_989_, v___x_1048_, v_f_991_, v_nezero_1047_, v___x_1048_, v_a_992_, v_a_993_, v_a_994_, v_a_995_);
lean_dec(v___x_1048_);
if (lean_obj_tag(v___x_1049_) == 0)
{
lean_object* v_a_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1120_; 
v_a_1050_ = lean_ctor_get(v___x_1049_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1049_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1052_ = v___x_1049_;
v_isShared_1053_ = v_isSharedCheck_1120_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_a_1050_);
lean_dec(v___x_1049_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1120_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; uint8_t v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1118_; 
lean_inc(v_u_987_);
v___x_1054_ = l_Lean_Level_succ___override(v_u_987_);
v___x_1055_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1055_, 0, v___x_1054_);
lean_ctor_set(v___x_1055_, 1, v___x_1037_);
v___x_1056_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__16));
v___x_1057_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1057_, 0, v_u_987_);
lean_ctor_set(v___x_1057_, 1, v___x_1037_);
lean_inc_ref_n(v___x_1057_, 7);
v___x_1058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1036_);
lean_ctor_set(v___x_1058_, 1, v___x_1057_);
v___x_1059_ = l_Lean_Expr_const___override(v___x_1056_, v___x_1058_);
v___x_1060_ = lean_obj_once(&lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26, &lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26_once, _init_lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__26);
v___x_1061_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__61, &lp_mathlib_FinVec_mkProdEqQ___closed__61_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__61);
v___x_1062_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__64, &lp_mathlib_FinVec_mkProdEqQ___closed__64_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__64);
v___x_1063_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__66));
v___x_1064_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__69, &lp_mathlib_FinVec_mkProdEqQ___closed__69_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__69);
lean_inc_ref_n(v_f_991_, 2);
v___x_1065_ = l_Lean_Expr_betaRev(v_f_991_, v___x_1064_, v_isZero_998_, v_isZero_998_);
v___x_1066_ = 0;
v___x_1067_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__71));
v___x_1068_ = l_Lean_Expr_const___override(v___x_1067_, v___x_1055_);
lean_inc_ref_n(v_00_u03b1_988_, 8);
v___x_1069_ = l_Lean_Expr_app___override(v___x_1068_, v_00_u03b1_988_);
v___x_1070_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__17));
v___x_1071_ = l_Lean_Expr_const___override(v___x_1070_, v___x_1057_);
v___x_1072_ = l_Lean_Expr_app___override(v___x_1071_, v_00_u03b1_988_);
v___x_1073_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__18));
v___x_1074_ = l_Lean_Expr_const___override(v___x_1073_, v___x_1057_);
v___x_1075_ = l_Lean_Expr_app___override(v___x_1074_, v_00_u03b1_988_);
v___x_1076_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__7));
v___x_1077_ = l_Lean_Expr_const___override(v___x_1076_, v___x_1057_);
v___x_1078_ = l_Lean_Expr_app___override(v___x_1077_, v_00_u03b1_988_);
v___x_1079_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__10));
v___x_1080_ = l_Lean_Expr_const___override(v___x_1079_, v___x_1057_);
v___x_1081_ = l_Lean_Expr_app___override(v___x_1080_, v_00_u03b1_988_);
v___x_1082_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__12));
v___x_1083_ = l_Lean_Expr_const___override(v___x_1082_, v___x_1057_);
v___x_1084_ = l_Lean_Expr_app___override(v___x_1083_, v_00_u03b1_988_);
lean_inc_ref_n(v_inst_989_, 2);
v___x_1085_ = l_Lean_Expr_app___override(v___x_1084_, v_inst_989_);
v___x_1086_ = l_Lean_Expr_app___override(v___x_1081_, v___x_1085_);
v___x_1087_ = l_Lean_Expr_app___override(v___x_1078_, v___x_1086_);
lean_inc_ref(v___x_1087_);
v___x_1088_ = l_Lean_Expr_app___override(v___x_1075_, v___x_1087_);
v___x_1089_ = l_Lean_Expr_app___override(v___x_1072_, v___x_1088_);
v___x_1090_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__4));
v___x_1091_ = l_Lean_Expr_const___override(v___x_1090_, v___x_1057_);
v___x_1092_ = l_Lean_Expr_app___override(v___x_1091_, v_00_u03b1_988_);
v___x_1093_ = l_Lean_Expr_app___override(v___x_1092_, v___x_1087_);
v___x_1094_ = l_Lean_Expr_app___override(v___x_1089_, v___x_1093_);
lean_inc_ref_n(v___x_1042_, 3);
v___x_1095_ = l_Lean_Expr_app___override(v___x_1094_, v___x_1042_);
v___x_1096_ = l_Lean_Expr_app___override(v___x_1095_, v_f_991_);
v___x_1097_ = l_Lean_Expr_app___override(v___x_1069_, v___x_1096_);
v___x_1098_ = l_Lean_Expr_app___override(v___x_1060_, v___x_1042_);
lean_inc_ref_n(v___x_1098_, 2);
v___x_1099_ = l_Lean_Expr_app___override(v___x_1059_, v___x_1098_);
v___x_1100_ = l_Lean_Expr_app___override(v___x_1099_, v_00_u03b1_988_);
v___x_1101_ = l_Lean_Expr_app___override(v___x_1100_, v_inst_989_);
v___x_1102_ = l_Lean_Expr_app___override(v___x_1061_, v___x_1098_);
v___x_1103_ = l_Lean_Expr_app___override(v___x_1062_, v___x_1042_);
v___x_1104_ = l_Lean_Expr_app___override(v___x_1102_, v___x_1103_);
v___x_1105_ = l_Lean_Expr_app___override(v___x_1101_, v___x_1104_);
v___x_1106_ = l_Lean_Expr_lam___override(v___x_1063_, v___x_1098_, v___x_1065_, v___x_1066_);
v___x_1107_ = l_Lean_Expr_app___override(v___x_1105_, v___x_1106_);
v___x_1108_ = l_Lean_Expr_app___override(v___x_1097_, v___x_1107_);
v___x_1109_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__20));
v___x_1110_ = l_Lean_Expr_const___override(v___x_1109_, v___x_1057_);
v___x_1111_ = l_Lean_Expr_app___override(v___x_1110_, v_00_u03b1_988_);
v___x_1112_ = l_Lean_Expr_app___override(v___x_1111_, v_inst_989_);
v___x_1113_ = l_Lean_Expr_app___override(v___x_1112_, v___x_1042_);
v___x_1114_ = l_Lean_Expr_app___override(v___x_1113_, v_f_991_);
v___x_1115_ = l_Lean_Expr_app___override(v___x_1108_, v___x_1114_);
v___x_1116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1116_, 0, v_a_1050_);
lean_ctor_set(v___x_1116_, 1, v___x_1115_);
if (v_isShared_1053_ == 0)
{
lean_ctor_set(v___x_1052_, 0, v___x_1116_);
v___x_1118_ = v___x_1052_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v___x_1116_);
v___x_1118_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
return v___x_1118_;
}
}
}
else
{
lean_object* v_a_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1128_; 
lean_dec_ref(v___x_1042_);
lean_dec_ref(v_f_991_);
lean_dec_ref(v_inst_989_);
lean_dec_ref(v_00_u03b1_988_);
lean_dec(v_u_987_);
v_a_1121_ = lean_ctor_get(v___x_1049_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_1049_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1123_ = v___x_1049_;
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_a_1121_);
lean_dec(v___x_1049_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1126_; 
if (v_isShared_1124_ == 0)
{
v___x_1126_ = v___x_1123_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_a_1121_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
return v___x_1126_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinVec_mkSumEqQ___boxed(lean_object* v_u_1129_, lean_object* v_00_u03b1_1130_, lean_object* v_inst_1131_, lean_object* v_n_1132_, lean_object* v_f_1133_, lean_object* v_a_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_){
_start:
{
lean_object* v_res_1139_; 
v_res_1139_ = lp_mathlib_FinVec_mkSumEqQ(v_u_1129_, v_00_u03b1_1130_, v_inst_1131_, v_n_1132_, v_f_1133_, v_a_1134_, v_a_1135_, v_a_1136_, v_a_1137_);
lean_dec(v_a_1137_);
lean_dec_ref(v_a_1136_);
lean_dec(v_a_1135_);
lean_dec_ref(v_a_1134_);
lean_dec(v_n_1132_);
return v_res_1139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(lean_object* v_e_1140_, lean_object* v___y_1141_){
_start:
{
uint8_t v___x_1143_; 
v___x_1143_ = l_Lean_Expr_hasMVar(v_e_1140_);
if (v___x_1143_ == 0)
{
lean_object* v___x_1144_; 
v___x_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1144_, 0, v_e_1140_);
return v___x_1144_;
}
else
{
lean_object* v___x_1145_; lean_object* v_mctx_1146_; lean_object* v___x_1147_; lean_object* v_fst_1148_; lean_object* v_snd_1149_; lean_object* v___x_1150_; lean_object* v_cache_1151_; lean_object* v_zetaDeltaFVarIds_1152_; lean_object* v_postponed_1153_; lean_object* v_diag_1154_; lean_object* v___x_1156_; uint8_t v_isShared_1157_; uint8_t v_isSharedCheck_1163_; 
v___x_1145_ = lean_st_ref_get(v___y_1141_);
v_mctx_1146_ = lean_ctor_get(v___x_1145_, 0);
lean_inc_ref(v_mctx_1146_);
lean_dec(v___x_1145_);
v___x_1147_ = l_Lean_instantiateMVarsCore(v_mctx_1146_, v_e_1140_);
v_fst_1148_ = lean_ctor_get(v___x_1147_, 0);
lean_inc(v_fst_1148_);
v_snd_1149_ = lean_ctor_get(v___x_1147_, 1);
lean_inc(v_snd_1149_);
lean_dec_ref(v___x_1147_);
v___x_1150_ = lean_st_ref_take(v___y_1141_);
v_cache_1151_ = lean_ctor_get(v___x_1150_, 1);
v_zetaDeltaFVarIds_1152_ = lean_ctor_get(v___x_1150_, 2);
v_postponed_1153_ = lean_ctor_get(v___x_1150_, 3);
v_diag_1154_ = lean_ctor_get(v___x_1150_, 4);
v_isSharedCheck_1163_ = !lean_is_exclusive(v___x_1150_);
if (v_isSharedCheck_1163_ == 0)
{
lean_object* v_unused_1164_; 
v_unused_1164_ = lean_ctor_get(v___x_1150_, 0);
lean_dec(v_unused_1164_);
v___x_1156_ = v___x_1150_;
v_isShared_1157_ = v_isSharedCheck_1163_;
goto v_resetjp_1155_;
}
else
{
lean_inc(v_diag_1154_);
lean_inc(v_postponed_1153_);
lean_inc(v_zetaDeltaFVarIds_1152_);
lean_inc(v_cache_1151_);
lean_dec(v___x_1150_);
v___x_1156_ = lean_box(0);
v_isShared_1157_ = v_isSharedCheck_1163_;
goto v_resetjp_1155_;
}
v_resetjp_1155_:
{
lean_object* v___x_1159_; 
if (v_isShared_1157_ == 0)
{
lean_ctor_set(v___x_1156_, 0, v_snd_1149_);
v___x_1159_ = v___x_1156_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1162_; 
v_reuseFailAlloc_1162_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1162_, 0, v_snd_1149_);
lean_ctor_set(v_reuseFailAlloc_1162_, 1, v_cache_1151_);
lean_ctor_set(v_reuseFailAlloc_1162_, 2, v_zetaDeltaFVarIds_1152_);
lean_ctor_set(v_reuseFailAlloc_1162_, 3, v_postponed_1153_);
lean_ctor_set(v_reuseFailAlloc_1162_, 4, v_diag_1154_);
v___x_1159_ = v_reuseFailAlloc_1162_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; 
v___x_1160_ = lean_st_ref_set(v___y_1141_, v___x_1159_);
v___x_1161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1161_, 0, v_fst_1148_);
return v___x_1161_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg___boxed(lean_object* v_e_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_){
_start:
{
lean_object* v_res_1168_; 
v_res_1168_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_e_1165_, v___y_1166_);
lean_dec(v___y_1166_);
return v_res_1168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0(lean_object* v_e_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_){
_start:
{
lean_object* v___x_1175_; 
v___x_1175_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_e_1169_, v___y_1171_);
return v___x_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___boxed(lean_object* v_e_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0(v_e_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(lean_object* v_k_1183_, uint8_t v_allowLevelAssignments_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1184_, v_k_1183_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
if (lean_obj_tag(v___x_1190_) == 0)
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
v_a_1191_ = lean_ctor_get(v___x_1190_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1190_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1190_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1190_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
else
{
lean_object* v_a_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1206_; 
v_a_1199_ = lean_ctor_get(v___x_1190_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1190_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1201_ = v___x_1190_;
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_a_1199_);
lean_dec(v___x_1190_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1204_; 
if (v_isShared_1202_ == 0)
{
v___x_1204_ = v___x_1201_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v_a_1199_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg___boxed(lean_object* v_k_1207_, lean_object* v_allowLevelAssignments_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1214_; lean_object* v_res_1215_; 
v_allowLevelAssignments_boxed_1214_ = lean_unbox(v_allowLevelAssignments_1208_);
v_res_1215_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(v_k_1207_, v_allowLevelAssignments_boxed_1214_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
lean_dec(v___y_1212_);
lean_dec_ref(v___y_1211_);
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1209_);
return v_res_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1(lean_object* v_00_u03b1_1216_, lean_object* v_k_1217_, uint8_t v_allowLevelAssignments_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_){
_start:
{
lean_object* v___x_1224_; 
v___x_1224_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(v_k_1217_, v_allowLevelAssignments_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_);
return v___x_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___boxed(lean_object* v_00_u03b1_1225_, lean_object* v_k_1226_, lean_object* v_allowLevelAssignments_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1233_; lean_object* v_res_1234_; 
v_allowLevelAssignments_boxed_1233_ = lean_unbox(v_allowLevelAssignments_1227_);
v_res_1234_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1(v_00_u03b1_1225_, v_k_1226_, v_allowLevelAssignments_boxed_1233_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
return v_res_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0(lean_object* v___x_1240_, uint8_t v___x_1241_, lean_object* v___x_1242_, lean_object* v_a_1243_, lean_object* v___x_1244_, lean_object* v_fst_1245_, lean_object* v_snd_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
lean_object* v___x_1252_; 
lean_inc(v___x_1242_);
v___x_1252_ = l_Lean_Meta_mkFreshExprMVar(v___x_1240_, v___x_1241_, v___x_1242_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
if (lean_obj_tag(v___x_1252_) == 0)
{
lean_object* v_a_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v_a_1253_ = lean_ctor_get(v___x_1252_, 0);
lean_inc(v_a_1253_);
lean_dec_ref_known(v___x_1252_, 1);
v___x_1254_ = ((lean_object*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__0));
lean_inc(v___x_1244_);
v___x_1255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1255_, 0, v_a_1243_);
lean_ctor_set(v___x_1255_, 1, v___x_1244_);
lean_inc_ref(v___x_1255_);
v___x_1256_ = l_Lean_Expr_const___override(v___x_1254_, v___x_1255_);
lean_inc_ref(v_fst_1245_);
v___x_1257_ = l_Lean_Expr_app___override(v___x_1256_, v_fst_1245_);
v___x_1258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1258_, 0, v___x_1257_);
lean_inc(v___x_1242_);
v___x_1259_ = l_Lean_Meta_mkFreshExprMVar(v___x_1258_, v___x_1241_, v___x_1242_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_object* v_a_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; 
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_a_1260_);
lean_dec_ref_known(v___x_1259_, 1);
v___x_1261_ = ((lean_object*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__2));
v___x_1262_ = lean_box(0);
lean_inc(v___x_1244_);
v___x_1263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
lean_ctor_set(v___x_1263_, 1, v___x_1244_);
lean_inc_ref(v___x_1263_);
v___x_1264_ = l_Lean_Expr_const___override(v___x_1261_, v___x_1263_);
v___x_1265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25));
v___x_1266_ = l_Lean_Expr_const___override(v___x_1265_, v___x_1244_);
lean_inc(v_a_1253_);
v___x_1267_ = l_Lean_Expr_app___override(v___x_1266_, v_a_1253_);
lean_inc_ref(v___x_1267_);
v___x_1268_ = l_Lean_Expr_app___override(v___x_1264_, v___x_1267_);
v___x_1269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1269_, 0, v___x_1268_);
lean_inc(v___x_1242_);
v___x_1270_ = l_Lean_Meta_mkFreshExprMVar(v___x_1269_, v___x_1241_, v___x_1242_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
if (lean_obj_tag(v___x_1270_) == 0)
{
lean_object* v_a_1271_; uint8_t v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; 
v_a_1271_ = lean_ctor_get(v___x_1270_, 0);
lean_inc(v_a_1271_);
lean_dec_ref_known(v___x_1270_, 1);
v___x_1272_ = 0;
lean_inc_ref(v_fst_1245_);
lean_inc_ref(v___x_1267_);
lean_inc(v___x_1242_);
v___x_1273_ = l_Lean_Expr_forallE___override(v___x_1242_, v___x_1267_, v_fst_1245_, v___x_1272_);
v___x_1274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1273_);
v___x_1275_ = l_Lean_Meta_mkFreshExprMVar(v___x_1274_, v___x_1241_, v___x_1242_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
if (lean_obj_tag(v___x_1275_) == 0)
{
lean_object* v_a_1276_; lean_object* v_keyedConfig_1277_; uint8_t v_trackZetaDelta_1278_; lean_object* v_zetaDeltaSet_1279_; lean_object* v_lctx_1280_; lean_object* v_localInstances_1281_; lean_object* v_defEqCtx_x3f_1282_; lean_object* v_synthPendingDepth_1283_; lean_object* v_customCanUnfoldPredicate_x3f_1284_; uint8_t v_univApprox_1285_; uint8_t v_inTypeClassResolution_1286_; uint8_t v_cacheInferType_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1349_; 
v_a_1276_ = lean_ctor_get(v___x_1275_, 0);
lean_inc(v_a_1276_);
lean_dec_ref_known(v___x_1275_, 1);
v_keyedConfig_1277_ = lean_ctor_get(v___y_1247_, 0);
v_trackZetaDelta_1278_ = lean_ctor_get_uint8(v___y_1247_, sizeof(void*)*7);
v_zetaDeltaSet_1279_ = lean_ctor_get(v___y_1247_, 1);
v_lctx_1280_ = lean_ctor_get(v___y_1247_, 2);
v_localInstances_1281_ = lean_ctor_get(v___y_1247_, 3);
v_defEqCtx_x3f_1282_ = lean_ctor_get(v___y_1247_, 4);
v_synthPendingDepth_1283_ = lean_ctor_get(v___y_1247_, 5);
v_customCanUnfoldPredicate_x3f_1284_ = lean_ctor_get(v___y_1247_, 6);
v_univApprox_1285_ = lean_ctor_get_uint8(v___y_1247_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1286_ = lean_ctor_get_uint8(v___y_1247_, sizeof(void*)*7 + 2);
v_cacheInferType_1287_ = lean_ctor_get_uint8(v___y_1247_, sizeof(void*)*7 + 3);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___y_1247_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1289_ = v___y_1247_;
v_isShared_1290_ = v_isSharedCheck_1349_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1284_);
lean_inc(v_synthPendingDepth_1283_);
lean_inc(v_defEqCtx_x3f_1282_);
lean_inc(v_localInstances_1281_);
lean_inc(v_lctx_1280_);
lean_inc(v_zetaDeltaSet_1279_);
lean_inc(v_keyedConfig_1277_);
lean_dec(v___y_1247_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1349_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; uint8_t v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1306_; 
v___x_1291_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__58));
v___x_1292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1262_);
lean_ctor_set(v___x_1292_, 1, v___x_1255_);
v___x_1293_ = l_Lean_Expr_const___override(v___x_1291_, v___x_1292_);
lean_inc_ref(v___x_1267_);
v___x_1294_ = l_Lean_Expr_app___override(v___x_1293_, v___x_1267_);
v___x_1295_ = l_Lean_Expr_app___override(v___x_1294_, v_fst_1245_);
lean_inc(v_a_1260_);
v___x_1296_ = l_Lean_Expr_app___override(v___x_1295_, v_a_1260_);
v___x_1297_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__60));
v___x_1298_ = l_Lean_Expr_const___override(v___x_1297_, v___x_1263_);
v___x_1299_ = l_Lean_Expr_app___override(v___x_1298_, v___x_1267_);
lean_inc(v_a_1271_);
v___x_1300_ = l_Lean_Expr_app___override(v___x_1299_, v_a_1271_);
v___x_1301_ = l_Lean_Expr_app___override(v___x_1296_, v___x_1300_);
lean_inc(v_a_1276_);
v___x_1302_ = l_Lean_Expr_app___override(v___x_1301_, v_a_1276_);
v___x_1303_ = 2;
v___x_1304_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1303_, v_keyedConfig_1277_);
if (v_isShared_1290_ == 0)
{
lean_ctor_set(v___x_1289_, 0, v___x_1304_);
v___x_1306_ = v___x_1289_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v___x_1304_);
lean_ctor_set(v_reuseFailAlloc_1348_, 1, v_zetaDeltaSet_1279_);
lean_ctor_set(v_reuseFailAlloc_1348_, 2, v_lctx_1280_);
lean_ctor_set(v_reuseFailAlloc_1348_, 3, v_localInstances_1281_);
lean_ctor_set(v_reuseFailAlloc_1348_, 4, v_defEqCtx_x3f_1282_);
lean_ctor_set(v_reuseFailAlloc_1348_, 5, v_synthPendingDepth_1283_);
lean_ctor_set(v_reuseFailAlloc_1348_, 6, v_customCanUnfoldPredicate_x3f_1284_);
lean_ctor_set_uint8(v_reuseFailAlloc_1348_, sizeof(void*)*7, v_trackZetaDelta_1278_);
lean_ctor_set_uint8(v_reuseFailAlloc_1348_, sizeof(void*)*7 + 1, v_univApprox_1285_);
lean_ctor_set_uint8(v_reuseFailAlloc_1348_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1286_);
lean_ctor_set_uint8(v_reuseFailAlloc_1348_, sizeof(void*)*7 + 3, v_cacheInferType_1287_);
v___x_1306_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
lean_object* v___x_1307_; 
v___x_1307_ = l_Lean_Meta_isExprDefEq(v___x_1302_, v_snd_1246_, v___x_1306_, v___y_1248_, v___y_1249_, v___y_1250_);
lean_dec_ref(v___x_1306_);
if (lean_obj_tag(v___x_1307_) == 0)
{
lean_object* v_a_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1339_; 
v_a_1308_ = lean_ctor_get(v___x_1307_, 0);
v_isSharedCheck_1339_ = !lean_is_exclusive(v___x_1307_);
if (v_isSharedCheck_1339_ == 0)
{
v___x_1310_ = v___x_1307_;
v_isShared_1311_ = v_isSharedCheck_1339_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_a_1308_);
lean_dec(v___x_1307_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1339_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
uint8_t v___x_1312_; 
v___x_1312_ = lean_unbox(v_a_1308_);
if (v___x_1312_ == 0)
{
lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1318_; 
v___x_1313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1313_, 0, v_a_1276_);
lean_ctor_set(v___x_1313_, 1, v_a_1308_);
v___x_1314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1314_, 0, v_a_1271_);
lean_ctor_set(v___x_1314_, 1, v___x_1313_);
v___x_1315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1315_, 0, v_a_1260_);
lean_ctor_set(v___x_1315_, 1, v___x_1314_);
v___x_1316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1316_, 0, v_a_1253_);
lean_ctor_set(v___x_1316_, 1, v___x_1315_);
if (v_isShared_1311_ == 0)
{
lean_ctor_set(v___x_1310_, 0, v___x_1316_);
v___x_1318_ = v___x_1310_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v___x_1316_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
return v___x_1318_;
}
}
else
{
lean_object* v___x_1320_; lean_object* v_a_1321_; lean_object* v___x_1322_; lean_object* v_a_1323_; lean_object* v___x_1324_; lean_object* v_a_1325_; lean_object* v___x_1326_; lean_object* v_a_1327_; lean_object* v___x_1329_; uint8_t v_isShared_1330_; uint8_t v_isSharedCheck_1338_; 
lean_del_object(v___x_1310_);
v___x_1320_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1253_, v___y_1248_);
v_a_1321_ = lean_ctor_get(v___x_1320_, 0);
lean_inc(v_a_1321_);
lean_dec_ref(v___x_1320_);
v___x_1322_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1260_, v___y_1248_);
v_a_1323_ = lean_ctor_get(v___x_1322_, 0);
lean_inc(v_a_1323_);
lean_dec_ref(v___x_1322_);
v___x_1324_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1271_, v___y_1248_);
v_a_1325_ = lean_ctor_get(v___x_1324_, 0);
lean_inc(v_a_1325_);
lean_dec_ref(v___x_1324_);
v___x_1326_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1276_, v___y_1248_);
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_1338_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_1338_ == 0)
{
v___x_1329_ = v___x_1326_;
v_isShared_1330_ = v_isSharedCheck_1338_;
goto v_resetjp_1328_;
}
else
{
lean_inc(v_a_1327_);
lean_dec(v___x_1326_);
v___x_1329_ = lean_box(0);
v_isShared_1330_ = v_isSharedCheck_1338_;
goto v_resetjp_1328_;
}
v_resetjp_1328_:
{
lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1336_; 
v___x_1331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1331_, 0, v_a_1327_);
lean_ctor_set(v___x_1331_, 1, v_a_1308_);
v___x_1332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1332_, 0, v_a_1325_);
lean_ctor_set(v___x_1332_, 1, v___x_1331_);
v___x_1333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1333_, 0, v_a_1323_);
lean_ctor_set(v___x_1333_, 1, v___x_1332_);
v___x_1334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1334_, 0, v_a_1321_);
lean_ctor_set(v___x_1334_, 1, v___x_1333_);
if (v_isShared_1330_ == 0)
{
lean_ctor_set(v___x_1329_, 0, v___x_1334_);
v___x_1336_ = v___x_1329_;
goto v_reusejp_1335_;
}
else
{
lean_object* v_reuseFailAlloc_1337_; 
v_reuseFailAlloc_1337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1337_, 0, v___x_1334_);
v___x_1336_ = v_reuseFailAlloc_1337_;
goto v_reusejp_1335_;
}
v_reusejp_1335_:
{
return v___x_1336_;
}
}
}
}
}
else
{
lean_object* v_a_1340_; lean_object* v___x_1342_; uint8_t v_isShared_1343_; uint8_t v_isSharedCheck_1347_; 
lean_dec(v_a_1276_);
lean_dec(v_a_1271_);
lean_dec(v_a_1260_);
lean_dec(v_a_1253_);
v_a_1340_ = lean_ctor_get(v___x_1307_, 0);
v_isSharedCheck_1347_ = !lean_is_exclusive(v___x_1307_);
if (v_isSharedCheck_1347_ == 0)
{
v___x_1342_ = v___x_1307_;
v_isShared_1343_ = v_isSharedCheck_1347_;
goto v_resetjp_1341_;
}
else
{
lean_inc(v_a_1340_);
lean_dec(v___x_1307_);
v___x_1342_ = lean_box(0);
v_isShared_1343_ = v_isSharedCheck_1347_;
goto v_resetjp_1341_;
}
v_resetjp_1341_:
{
lean_object* v___x_1345_; 
if (v_isShared_1343_ == 0)
{
v___x_1345_ = v___x_1342_;
goto v_reusejp_1344_;
}
else
{
lean_object* v_reuseFailAlloc_1346_; 
v_reuseFailAlloc_1346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1346_, 0, v_a_1340_);
v___x_1345_ = v_reuseFailAlloc_1346_;
goto v_reusejp_1344_;
}
v_reusejp_1344_:
{
return v___x_1345_;
}
}
}
}
}
}
else
{
lean_object* v_a_1350_; lean_object* v___x_1352_; uint8_t v_isShared_1353_; uint8_t v_isSharedCheck_1357_; 
lean_dec(v_a_1271_);
lean_dec_ref(v___x_1267_);
lean_dec_ref_known(v___x_1263_, 2);
lean_dec(v_a_1260_);
lean_dec_ref_known(v___x_1255_, 2);
lean_dec(v_a_1253_);
lean_dec_ref(v___y_1247_);
lean_dec_ref(v_snd_1246_);
lean_dec_ref(v_fst_1245_);
v_a_1350_ = lean_ctor_get(v___x_1275_, 0);
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1275_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1352_ = v___x_1275_;
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
else
{
lean_inc(v_a_1350_);
lean_dec(v___x_1275_);
v___x_1352_ = lean_box(0);
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
v_resetjp_1351_:
{
lean_object* v___x_1355_; 
if (v_isShared_1353_ == 0)
{
v___x_1355_ = v___x_1352_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v_a_1350_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
return v___x_1355_;
}
}
}
}
else
{
lean_object* v_a_1358_; lean_object* v___x_1360_; uint8_t v_isShared_1361_; uint8_t v_isSharedCheck_1365_; 
lean_dec_ref(v___x_1267_);
lean_dec_ref_known(v___x_1263_, 2);
lean_dec(v_a_1260_);
lean_dec_ref_known(v___x_1255_, 2);
lean_dec(v_a_1253_);
lean_dec_ref(v___y_1247_);
lean_dec_ref(v_snd_1246_);
lean_dec_ref(v_fst_1245_);
lean_dec(v___x_1242_);
v_a_1358_ = lean_ctor_get(v___x_1270_, 0);
v_isSharedCheck_1365_ = !lean_is_exclusive(v___x_1270_);
if (v_isSharedCheck_1365_ == 0)
{
v___x_1360_ = v___x_1270_;
v_isShared_1361_ = v_isSharedCheck_1365_;
goto v_resetjp_1359_;
}
else
{
lean_inc(v_a_1358_);
lean_dec(v___x_1270_);
v___x_1360_ = lean_box(0);
v_isShared_1361_ = v_isSharedCheck_1365_;
goto v_resetjp_1359_;
}
v_resetjp_1359_:
{
lean_object* v___x_1363_; 
if (v_isShared_1361_ == 0)
{
v___x_1363_ = v___x_1360_;
goto v_reusejp_1362_;
}
else
{
lean_object* v_reuseFailAlloc_1364_; 
v_reuseFailAlloc_1364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1364_, 0, v_a_1358_);
v___x_1363_ = v_reuseFailAlloc_1364_;
goto v_reusejp_1362_;
}
v_reusejp_1362_:
{
return v___x_1363_;
}
}
}
}
else
{
lean_object* v_a_1366_; lean_object* v___x_1368_; uint8_t v_isShared_1369_; uint8_t v_isSharedCheck_1373_; 
lean_dec_ref_known(v___x_1255_, 2);
lean_dec(v_a_1253_);
lean_dec_ref(v___y_1247_);
lean_dec_ref(v_snd_1246_);
lean_dec_ref(v_fst_1245_);
lean_dec(v___x_1244_);
lean_dec(v___x_1242_);
v_a_1366_ = lean_ctor_get(v___x_1259_, 0);
v_isSharedCheck_1373_ = !lean_is_exclusive(v___x_1259_);
if (v_isSharedCheck_1373_ == 0)
{
v___x_1368_ = v___x_1259_;
v_isShared_1369_ = v_isSharedCheck_1373_;
goto v_resetjp_1367_;
}
else
{
lean_inc(v_a_1366_);
lean_dec(v___x_1259_);
v___x_1368_ = lean_box(0);
v_isShared_1369_ = v_isSharedCheck_1373_;
goto v_resetjp_1367_;
}
v_resetjp_1367_:
{
lean_object* v___x_1371_; 
if (v_isShared_1369_ == 0)
{
v___x_1371_ = v___x_1368_;
goto v_reusejp_1370_;
}
else
{
lean_object* v_reuseFailAlloc_1372_; 
v_reuseFailAlloc_1372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1372_, 0, v_a_1366_);
v___x_1371_ = v_reuseFailAlloc_1372_;
goto v_reusejp_1370_;
}
v_reusejp_1370_:
{
return v___x_1371_;
}
}
}
}
else
{
lean_object* v_a_1374_; lean_object* v___x_1376_; uint8_t v_isShared_1377_; uint8_t v_isSharedCheck_1381_; 
lean_dec_ref(v___y_1247_);
lean_dec_ref(v_snd_1246_);
lean_dec_ref(v_fst_1245_);
lean_dec(v___x_1244_);
lean_dec(v_a_1243_);
lean_dec(v___x_1242_);
v_a_1374_ = lean_ctor_get(v___x_1252_, 0);
v_isSharedCheck_1381_ = !lean_is_exclusive(v___x_1252_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1376_ = v___x_1252_;
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
else
{
lean_inc(v_a_1374_);
lean_dec(v___x_1252_);
v___x_1376_ = lean_box(0);
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
v_resetjp_1375_:
{
lean_object* v___x_1379_; 
if (v_isShared_1377_ == 0)
{
v___x_1379_ = v___x_1376_;
goto v_reusejp_1378_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_a_1374_);
v___x_1379_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1378_;
}
v_reusejp_1378_:
{
return v___x_1379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___boxed(lean_object* v___x_1382_, lean_object* v___x_1383_, lean_object* v___x_1384_, lean_object* v_a_1385_, lean_object* v___x_1386_, lean_object* v_fst_1387_, lean_object* v_snd_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_){
_start:
{
uint8_t v___x_6719__boxed_1394_; lean_object* v_res_1395_; 
v___x_6719__boxed_1394_ = lean_unbox(v___x_1383_);
v_res_1395_ = lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0(v___x_1382_, v___x_6719__boxed_1394_, v___x_1384_, v_a_1385_, v___x_1386_, v_fst_1387_, v_snd_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v___y_1390_);
return v_res_1395_;
}
}
static lean_object* _init_lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1(void){
_start:
{
lean_object* v___x_1398_; lean_object* v___x_1399_; 
v___x_1398_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__12, &lp_mathlib_FinVec_mkProdEqQ___closed__12_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__12);
v___x_1399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1399_, 0, v___x_1398_);
return v___x_1399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg(lean_object* v_a_1400_, lean_object* v_a_1401_, lean_object* v_a_1402_, lean_object* v_a_1403_, lean_object* v_a_1404_){
_start:
{
lean_object* v___x_1406_; 
v___x_1406_ = lp_Qq_Qq_inferTypeQ(v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_);
if (lean_obj_tag(v___x_1406_) == 0)
{
lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1499_; 
v_a_1407_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1499_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1499_ == 0)
{
v___x_1409_ = v___x_1406_;
v_isShared_1410_ = v_isSharedCheck_1499_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___x_1406_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1499_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
lean_object* v_snd_1411_; lean_object* v_fst_1412_; lean_object* v_fst_1413_; lean_object* v_snd_1414_; lean_object* v___x_1415_; 
v_snd_1411_ = lean_ctor_get(v_a_1407_, 1);
lean_inc(v_snd_1411_);
v_fst_1412_ = lean_ctor_get(v_a_1407_, 0);
lean_inc(v_fst_1412_);
lean_dec(v_a_1407_);
v_fst_1413_ = lean_ctor_get(v_snd_1411_, 0);
lean_inc(v_fst_1413_);
v_snd_1414_ = lean_ctor_get(v_snd_1411_, 1);
lean_inc(v_snd_1414_);
lean_dec(v_snd_1411_);
v___x_1415_ = ((lean_object*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__0));
if (lean_obj_tag(v_fst_1412_) == 1)
{
lean_object* v_a_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; uint8_t v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___f_1422_; uint8_t v___x_1423_; lean_object* v___x_1424_; 
lean_del_object(v___x_1409_);
v_a_1416_ = lean_ctor_get(v_fst_1412_, 0);
lean_inc_n(v_a_1416_, 2);
lean_dec_ref_known(v_fst_1412_, 1);
v___x_1417_ = lean_box(0);
v___x_1418_ = lean_obj_once(&lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1, &lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1_once, _init_lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1);
v___x_1419_ = 0;
v___x_1420_ = lean_box(0);
v___x_1421_ = lean_box(v___x_1419_);
lean_inc(v_fst_1413_);
v___f_1422_ = lean_alloc_closure((void*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___boxed), 12, 7);
lean_closure_set(v___f_1422_, 0, v___x_1418_);
lean_closure_set(v___f_1422_, 1, v___x_1421_);
lean_closure_set(v___f_1422_, 2, v___x_1420_);
lean_closure_set(v___f_1422_, 3, v_a_1416_);
lean_closure_set(v___f_1422_, 4, v___x_1417_);
lean_closure_set(v___f_1422_, 5, v_fst_1413_);
lean_closure_set(v___f_1422_, 6, v_snd_1414_);
v___x_1423_ = 0;
v___x_1424_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(v___f_1422_, v___x_1423_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_);
if (lean_obj_tag(v___x_1424_) == 0)
{
lean_object* v_a_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1487_; 
v_a_1425_ = lean_ctor_get(v___x_1424_, 0);
v_isSharedCheck_1487_ = !lean_is_exclusive(v___x_1424_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1427_ = v___x_1424_;
v_isShared_1428_ = v_isSharedCheck_1487_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_a_1425_);
lean_dec(v___x_1424_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1487_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v_snd_1429_; lean_object* v_snd_1430_; lean_object* v_snd_1431_; lean_object* v_snd_1432_; uint8_t v___x_1433_; 
v_snd_1429_ = lean_ctor_get(v_a_1425_, 1);
lean_inc(v_snd_1429_);
v_snd_1430_ = lean_ctor_get(v_snd_1429_, 1);
lean_inc(v_snd_1430_);
v_snd_1431_ = lean_ctor_get(v_snd_1430_, 1);
lean_inc(v_snd_1431_);
v_snd_1432_ = lean_ctor_get(v_snd_1431_, 1);
lean_inc(v_snd_1432_);
v___x_1433_ = lean_unbox(v_snd_1432_);
if (v___x_1433_ == 0)
{
lean_object* v___x_1435_; 
lean_dec(v_snd_1432_);
lean_dec(v_snd_1431_);
lean_dec(v_snd_1430_);
lean_dec(v_snd_1429_);
lean_dec(v_a_1425_);
lean_dec(v_a_1416_);
lean_dec(v_fst_1413_);
if (v_isShared_1428_ == 0)
{
lean_ctor_set(v___x_1427_, 0, v___x_1415_);
v___x_1435_ = v___x_1427_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v___x_1415_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
else
{
lean_object* v_fst_1437_; lean_object* v_fst_1438_; lean_object* v_fst_1439_; lean_object* v_fst_1440_; lean_object* v___x_1441_; 
v_fst_1437_ = lean_ctor_get(v_a_1425_, 0);
lean_inc_n(v_fst_1437_, 2);
lean_dec(v_a_1425_);
v_fst_1438_ = lean_ctor_get(v_snd_1429_, 0);
lean_inc(v_fst_1438_);
lean_dec(v_snd_1429_);
v_fst_1439_ = lean_ctor_get(v_snd_1430_, 0);
lean_inc(v_fst_1439_);
lean_dec(v_snd_1430_);
v_fst_1440_ = lean_ctor_get(v_snd_1431_, 0);
lean_inc(v_fst_1440_);
lean_dec(v_snd_1431_);
v___x_1441_ = l_Lean_Expr_nat_x3f(v_fst_1437_);
if (lean_obj_tag(v___x_1441_) == 0)
{
lean_object* v___x_1443_; 
lean_dec(v_fst_1440_);
lean_dec(v_fst_1439_);
lean_dec(v_fst_1438_);
lean_dec(v_fst_1437_);
lean_dec(v_snd_1432_);
lean_dec(v_a_1416_);
lean_dec(v_fst_1413_);
if (v_isShared_1428_ == 0)
{
lean_ctor_set(v___x_1427_, 0, v___x_1415_);
v___x_1443_ = v___x_1427_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1444_; 
v_reuseFailAlloc_1444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1444_, 0, v___x_1415_);
v___x_1443_ = v_reuseFailAlloc_1444_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
return v___x_1443_;
}
}
else
{
lean_object* v_val_1445_; lean_object* v___x_1447_; uint8_t v_isShared_1448_; uint8_t v_isSharedCheck_1486_; 
lean_del_object(v___x_1427_);
v_val_1445_ = lean_ctor_get(v___x_1441_, 0);
v_isSharedCheck_1486_ = !lean_is_exclusive(v___x_1441_);
if (v_isSharedCheck_1486_ == 0)
{
v___x_1447_ = v___x_1441_;
v_isShared_1448_ = v_isSharedCheck_1486_;
goto v_resetjp_1446_;
}
else
{
lean_inc(v_val_1445_);
lean_dec(v___x_1441_);
v___x_1447_ = lean_box(0);
v_isShared_1448_ = v_isSharedCheck_1486_;
goto v_resetjp_1446_;
}
v_resetjp_1446_:
{
lean_object* v___x_1449_; 
v___x_1449_ = lp_mathlib_FinVec_mkProdEqQ(v_a_1416_, v_fst_1413_, v_fst_1438_, v_val_1445_, v_fst_1440_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_);
lean_dec(v_val_1445_);
if (lean_obj_tag(v___x_1449_) == 0)
{
lean_object* v_a_1450_; lean_object* v_fst_1451_; lean_object* v_snd_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; 
v_a_1450_ = lean_ctor_get(v___x_1449_, 0);
lean_inc(v_a_1450_);
lean_dec_ref_known(v___x_1449_, 1);
v_fst_1451_ = lean_ctor_get(v_a_1450_, 0);
lean_inc(v_fst_1451_);
v_snd_1452_ = lean_ctor_get(v_a_1450_, 1);
lean_inc(v_snd_1452_);
lean_dec(v_a_1450_);
v___x_1453_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__64, &lp_mathlib_FinVec_mkProdEqQ___closed__64_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__64);
v___x_1454_ = l_Lean_Expr_app___override(v___x_1453_, v_fst_1437_);
v___x_1455_ = lp_Qq_Qq_assertDefEqQ___redArg(v_fst_1439_, v___x_1454_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_);
if (lean_obj_tag(v___x_1455_) == 0)
{
lean_object* v___x_1457_; uint8_t v_isShared_1458_; uint8_t v_isSharedCheck_1468_; 
v_isSharedCheck_1468_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1468_ == 0)
{
lean_object* v_unused_1469_; 
v_unused_1469_ = lean_ctor_get(v___x_1455_, 0);
lean_dec(v_unused_1469_);
v___x_1457_ = v___x_1455_;
v_isShared_1458_ = v_isSharedCheck_1468_;
goto v_resetjp_1456_;
}
else
{
lean_dec(v___x_1455_);
v___x_1457_ = lean_box(0);
v_isShared_1458_ = v_isSharedCheck_1468_;
goto v_resetjp_1456_;
}
v_resetjp_1456_:
{
lean_object* v___x_1460_; 
if (v_isShared_1448_ == 0)
{
lean_ctor_set(v___x_1447_, 0, v_snd_1452_);
v___x_1460_ = v___x_1447_;
goto v_reusejp_1459_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v_snd_1452_);
v___x_1460_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1459_;
}
v_reusejp_1459_:
{
lean_object* v___x_1461_; uint8_t v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1465_; 
v___x_1461_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1461_, 0, v_fst_1451_);
lean_ctor_set(v___x_1461_, 1, v___x_1460_);
v___x_1462_ = lean_unbox(v_snd_1432_);
lean_dec(v_snd_1432_);
lean_ctor_set_uint8(v___x_1461_, sizeof(void*)*2, v___x_1462_);
v___x_1463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1463_, 0, v___x_1461_);
if (v_isShared_1458_ == 0)
{
lean_ctor_set(v___x_1457_, 0, v___x_1463_);
v___x_1465_ = v___x_1457_;
goto v_reusejp_1464_;
}
else
{
lean_object* v_reuseFailAlloc_1466_; 
v_reuseFailAlloc_1466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1466_, 0, v___x_1463_);
v___x_1465_ = v_reuseFailAlloc_1466_;
goto v_reusejp_1464_;
}
v_reusejp_1464_:
{
return v___x_1465_;
}
}
}
}
else
{
lean_object* v_a_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1477_; 
lean_dec(v_snd_1452_);
lean_dec(v_fst_1451_);
lean_del_object(v___x_1447_);
lean_dec(v_snd_1432_);
v_a_1470_ = lean_ctor_get(v___x_1455_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1472_ = v___x_1455_;
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_a_1470_);
lean_dec(v___x_1455_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1475_; 
if (v_isShared_1473_ == 0)
{
v___x_1475_ = v___x_1472_;
goto v_reusejp_1474_;
}
else
{
lean_object* v_reuseFailAlloc_1476_; 
v_reuseFailAlloc_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1476_, 0, v_a_1470_);
v___x_1475_ = v_reuseFailAlloc_1476_;
goto v_reusejp_1474_;
}
v_reusejp_1474_:
{
return v___x_1475_;
}
}
}
}
else
{
lean_object* v_a_1478_; lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1485_; 
lean_del_object(v___x_1447_);
lean_dec(v_fst_1439_);
lean_dec(v_fst_1437_);
lean_dec(v_snd_1432_);
v_a_1478_ = lean_ctor_get(v___x_1449_, 0);
v_isSharedCheck_1485_ = !lean_is_exclusive(v___x_1449_);
if (v_isSharedCheck_1485_ == 0)
{
v___x_1480_ = v___x_1449_;
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
else
{
lean_inc(v_a_1478_);
lean_dec(v___x_1449_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
lean_object* v___x_1483_; 
if (v_isShared_1481_ == 0)
{
v___x_1483_ = v___x_1480_;
goto v_reusejp_1482_;
}
else
{
lean_object* v_reuseFailAlloc_1484_; 
v_reuseFailAlloc_1484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1484_, 0, v_a_1478_);
v___x_1483_ = v_reuseFailAlloc_1484_;
goto v_reusejp_1482_;
}
v_reusejp_1482_:
{
return v___x_1483_;
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
lean_object* v_a_1488_; lean_object* v___x_1490_; uint8_t v_isShared_1491_; uint8_t v_isSharedCheck_1495_; 
lean_dec(v_a_1416_);
lean_dec(v_fst_1413_);
v_a_1488_ = lean_ctor_get(v___x_1424_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1424_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1490_ = v___x_1424_;
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_a_1488_);
lean_dec(v___x_1424_);
v___x_1490_ = lean_box(0);
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
v_resetjp_1489_:
{
lean_object* v___x_1493_; 
if (v_isShared_1491_ == 0)
{
v___x_1493_ = v___x_1490_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v_a_1488_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
else
{
lean_object* v___x_1497_; 
lean_dec(v_snd_1414_);
lean_dec(v_fst_1413_);
lean_dec(v_fst_1412_);
if (v_isShared_1410_ == 0)
{
lean_ctor_set(v___x_1409_, 0, v___x_1415_);
v___x_1497_ = v___x_1409_;
goto v_reusejp_1496_;
}
else
{
lean_object* v_reuseFailAlloc_1498_; 
v_reuseFailAlloc_1498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1498_, 0, v___x_1415_);
v___x_1497_ = v_reuseFailAlloc_1498_;
goto v_reusejp_1496_;
}
v_reusejp_1496_:
{
return v___x_1497_;
}
}
}
}
else
{
lean_object* v_a_1500_; lean_object* v___x_1502_; uint8_t v_isShared_1503_; uint8_t v_isSharedCheck_1507_; 
v_a_1500_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1507_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1507_ == 0)
{
v___x_1502_ = v___x_1406_;
v_isShared_1503_ = v_isSharedCheck_1507_;
goto v_resetjp_1501_;
}
else
{
lean_inc(v_a_1500_);
lean_dec(v___x_1406_);
v___x_1502_ = lean_box(0);
v_isShared_1503_ = v_isSharedCheck_1507_;
goto v_resetjp_1501_;
}
v_resetjp_1501_:
{
lean_object* v___x_1505_; 
if (v_isShared_1503_ == 0)
{
v___x_1505_ = v___x_1502_;
goto v_reusejp_1504_;
}
else
{
lean_object* v_reuseFailAlloc_1506_; 
v_reuseFailAlloc_1506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1506_, 0, v_a_1500_);
v___x_1505_ = v_reuseFailAlloc_1506_;
goto v_reusejp_1504_;
}
v_reusejp_1504_:
{
return v___x_1505_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___redArg___boxed(lean_object* v_a_1508_, lean_object* v_a_1509_, lean_object* v_a_1510_, lean_object* v_a_1511_, lean_object* v_a_1512_, lean_object* v_a_1513_){
_start:
{
lean_object* v_res_1514_; 
v_res_1514_ = lp_mathlib_Fin_prod__univ__ofNat___redArg(v_a_1508_, v_a_1509_, v_a_1510_, v_a_1511_, v_a_1512_);
lean_dec(v_a_1512_);
lean_dec_ref(v_a_1511_);
lean_dec(v_a_1510_);
lean_dec_ref(v_a_1509_);
return v_res_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat(lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_){
_start:
{
lean_object* v___x_1524_; 
v___x_1524_ = lp_mathlib_Fin_prod__univ__ofNat___redArg(v_a_1515_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_prod__univ__ofNat___boxed(lean_object* v_a_1525_, lean_object* v_a_1526_, lean_object* v_a_1527_, lean_object* v_a_1528_, lean_object* v_a_1529_, lean_object* v_a_1530_, lean_object* v_a_1531_, lean_object* v_a_1532_, lean_object* v_a_1533_){
_start:
{
lean_object* v_res_1534_; 
v_res_1534_ = lp_mathlib_Fin_prod__univ__ofNat(v_a_1525_, v_a_1526_, v_a_1527_, v_a_1528_, v_a_1529_, v_a_1530_, v_a_1531_, v_a_1532_);
lean_dec(v_a_1532_);
lean_dec_ref(v_a_1531_);
lean_dec(v_a_1530_);
lean_dec_ref(v_a_1529_);
lean_dec(v_a_1528_);
lean_dec_ref(v_a_1527_);
lean_dec(v_a_1526_);
return v_res_1534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0(lean_object* v___x_1537_, uint8_t v___x_1538_, lean_object* v___x_1539_, lean_object* v_a_1540_, lean_object* v___x_1541_, lean_object* v_fst_1542_, lean_object* v_snd_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; 
lean_inc(v___x_1539_);
v___x_1549_ = l_Lean_Meta_mkFreshExprMVar(v___x_1537_, v___x_1538_, v___x_1539_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
if (lean_obj_tag(v___x_1549_) == 0)
{
lean_object* v_a_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; 
v_a_1550_ = lean_ctor_get(v___x_1549_, 0);
lean_inc(v_a_1550_);
lean_dec_ref_known(v___x_1549_, 1);
v___x_1551_ = ((lean_object*)(lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___closed__0));
lean_inc(v___x_1541_);
v___x_1552_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1552_, 0, v_a_1540_);
lean_ctor_set(v___x_1552_, 1, v___x_1541_);
lean_inc_ref(v___x_1552_);
v___x_1553_ = l_Lean_Expr_const___override(v___x_1551_, v___x_1552_);
lean_inc_ref(v_fst_1542_);
v___x_1554_ = l_Lean_Expr_app___override(v___x_1553_, v_fst_1542_);
v___x_1555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1555_, 0, v___x_1554_);
lean_inc(v___x_1539_);
v___x_1556_ = l_Lean_Meta_mkFreshExprMVar(v___x_1555_, v___x_1538_, v___x_1539_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
if (lean_obj_tag(v___x_1556_) == 0)
{
lean_object* v_a_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v_a_1557_ = lean_ctor_get(v___x_1556_, 0);
lean_inc(v_a_1557_);
lean_dec_ref_known(v___x_1556_, 1);
v___x_1558_ = ((lean_object*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___lam__0___closed__2));
v___x_1559_ = lean_box(0);
lean_inc(v___x_1541_);
v___x_1560_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1560_, 0, v___x_1559_);
lean_ctor_set(v___x_1560_, 1, v___x_1541_);
lean_inc_ref(v___x_1560_);
v___x_1561_ = l_Lean_Expr_const___override(v___x_1558_, v___x_1560_);
v___x_1562_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Fin_Tuple_Reflection_0__FinVec_mkProdEqQ_makeRHS___closed__25));
v___x_1563_ = l_Lean_Expr_const___override(v___x_1562_, v___x_1541_);
lean_inc(v_a_1550_);
v___x_1564_ = l_Lean_Expr_app___override(v___x_1563_, v_a_1550_);
lean_inc_ref(v___x_1564_);
v___x_1565_ = l_Lean_Expr_app___override(v___x_1561_, v___x_1564_);
v___x_1566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1566_, 0, v___x_1565_);
lean_inc(v___x_1539_);
v___x_1567_ = l_Lean_Meta_mkFreshExprMVar(v___x_1566_, v___x_1538_, v___x_1539_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
if (lean_obj_tag(v___x_1567_) == 0)
{
lean_object* v_a_1568_; uint8_t v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; 
v_a_1568_ = lean_ctor_get(v___x_1567_, 0);
lean_inc(v_a_1568_);
lean_dec_ref_known(v___x_1567_, 1);
v___x_1569_ = 0;
lean_inc_ref(v_fst_1542_);
lean_inc_ref(v___x_1564_);
lean_inc(v___x_1539_);
v___x_1570_ = l_Lean_Expr_forallE___override(v___x_1539_, v___x_1564_, v_fst_1542_, v___x_1569_);
v___x_1571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1571_, 0, v___x_1570_);
v___x_1572_ = l_Lean_Meta_mkFreshExprMVar(v___x_1571_, v___x_1538_, v___x_1539_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
if (lean_obj_tag(v___x_1572_) == 0)
{
lean_object* v_a_1573_; lean_object* v_keyedConfig_1574_; uint8_t v_trackZetaDelta_1575_; lean_object* v_zetaDeltaSet_1576_; lean_object* v_lctx_1577_; lean_object* v_localInstances_1578_; lean_object* v_defEqCtx_x3f_1579_; lean_object* v_synthPendingDepth_1580_; lean_object* v_customCanUnfoldPredicate_x3f_1581_; uint8_t v_univApprox_1582_; uint8_t v_inTypeClassResolution_1583_; uint8_t v_cacheInferType_1584_; lean_object* v___x_1586_; uint8_t v_isShared_1587_; uint8_t v_isSharedCheck_1646_; 
v_a_1573_ = lean_ctor_get(v___x_1572_, 0);
lean_inc(v_a_1573_);
lean_dec_ref_known(v___x_1572_, 1);
v_keyedConfig_1574_ = lean_ctor_get(v___y_1544_, 0);
v_trackZetaDelta_1575_ = lean_ctor_get_uint8(v___y_1544_, sizeof(void*)*7);
v_zetaDeltaSet_1576_ = lean_ctor_get(v___y_1544_, 1);
v_lctx_1577_ = lean_ctor_get(v___y_1544_, 2);
v_localInstances_1578_ = lean_ctor_get(v___y_1544_, 3);
v_defEqCtx_x3f_1579_ = lean_ctor_get(v___y_1544_, 4);
v_synthPendingDepth_1580_ = lean_ctor_get(v___y_1544_, 5);
v_customCanUnfoldPredicate_x3f_1581_ = lean_ctor_get(v___y_1544_, 6);
v_univApprox_1582_ = lean_ctor_get_uint8(v___y_1544_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1583_ = lean_ctor_get_uint8(v___y_1544_, sizeof(void*)*7 + 2);
v_cacheInferType_1584_ = lean_ctor_get_uint8(v___y_1544_, sizeof(void*)*7 + 3);
v_isSharedCheck_1646_ = !lean_is_exclusive(v___y_1544_);
if (v_isSharedCheck_1646_ == 0)
{
v___x_1586_ = v___y_1544_;
v_isShared_1587_ = v_isSharedCheck_1646_;
goto v_resetjp_1585_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1581_);
lean_inc(v_synthPendingDepth_1580_);
lean_inc(v_defEqCtx_x3f_1579_);
lean_inc(v_localInstances_1578_);
lean_inc(v_lctx_1577_);
lean_inc(v_zetaDeltaSet_1576_);
lean_inc(v_keyedConfig_1574_);
lean_dec(v___y_1544_);
v___x_1586_ = lean_box(0);
v_isShared_1587_ = v_isSharedCheck_1646_;
goto v_resetjp_1585_;
}
v_resetjp_1585_:
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; uint8_t v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1603_; 
v___x_1588_ = ((lean_object*)(lp_mathlib_FinVec_mkSumEqQ___closed__16));
v___x_1589_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1589_, 0, v___x_1559_);
lean_ctor_set(v___x_1589_, 1, v___x_1552_);
v___x_1590_ = l_Lean_Expr_const___override(v___x_1588_, v___x_1589_);
lean_inc_ref(v___x_1564_);
v___x_1591_ = l_Lean_Expr_app___override(v___x_1590_, v___x_1564_);
v___x_1592_ = l_Lean_Expr_app___override(v___x_1591_, v_fst_1542_);
lean_inc(v_a_1557_);
v___x_1593_ = l_Lean_Expr_app___override(v___x_1592_, v_a_1557_);
v___x_1594_ = ((lean_object*)(lp_mathlib_FinVec_mkProdEqQ___closed__60));
v___x_1595_ = l_Lean_Expr_const___override(v___x_1594_, v___x_1560_);
v___x_1596_ = l_Lean_Expr_app___override(v___x_1595_, v___x_1564_);
lean_inc(v_a_1568_);
v___x_1597_ = l_Lean_Expr_app___override(v___x_1596_, v_a_1568_);
v___x_1598_ = l_Lean_Expr_app___override(v___x_1593_, v___x_1597_);
lean_inc(v_a_1573_);
v___x_1599_ = l_Lean_Expr_app___override(v___x_1598_, v_a_1573_);
v___x_1600_ = 2;
v___x_1601_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1600_, v_keyedConfig_1574_);
if (v_isShared_1587_ == 0)
{
lean_ctor_set(v___x_1586_, 0, v___x_1601_);
v___x_1603_ = v___x_1586_;
goto v_reusejp_1602_;
}
else
{
lean_object* v_reuseFailAlloc_1645_; 
v_reuseFailAlloc_1645_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1645_, 0, v___x_1601_);
lean_ctor_set(v_reuseFailAlloc_1645_, 1, v_zetaDeltaSet_1576_);
lean_ctor_set(v_reuseFailAlloc_1645_, 2, v_lctx_1577_);
lean_ctor_set(v_reuseFailAlloc_1645_, 3, v_localInstances_1578_);
lean_ctor_set(v_reuseFailAlloc_1645_, 4, v_defEqCtx_x3f_1579_);
lean_ctor_set(v_reuseFailAlloc_1645_, 5, v_synthPendingDepth_1580_);
lean_ctor_set(v_reuseFailAlloc_1645_, 6, v_customCanUnfoldPredicate_x3f_1581_);
lean_ctor_set_uint8(v_reuseFailAlloc_1645_, sizeof(void*)*7, v_trackZetaDelta_1575_);
lean_ctor_set_uint8(v_reuseFailAlloc_1645_, sizeof(void*)*7 + 1, v_univApprox_1582_);
lean_ctor_set_uint8(v_reuseFailAlloc_1645_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1583_);
lean_ctor_set_uint8(v_reuseFailAlloc_1645_, sizeof(void*)*7 + 3, v_cacheInferType_1584_);
v___x_1603_ = v_reuseFailAlloc_1645_;
goto v_reusejp_1602_;
}
v_reusejp_1602_:
{
lean_object* v___x_1604_; 
v___x_1604_ = l_Lean_Meta_isExprDefEq(v___x_1599_, v_snd_1543_, v___x_1603_, v___y_1545_, v___y_1546_, v___y_1547_);
lean_dec_ref(v___x_1603_);
if (lean_obj_tag(v___x_1604_) == 0)
{
lean_object* v_a_1605_; lean_object* v___x_1607_; uint8_t v_isShared_1608_; uint8_t v_isSharedCheck_1636_; 
v_a_1605_ = lean_ctor_get(v___x_1604_, 0);
v_isSharedCheck_1636_ = !lean_is_exclusive(v___x_1604_);
if (v_isSharedCheck_1636_ == 0)
{
v___x_1607_ = v___x_1604_;
v_isShared_1608_ = v_isSharedCheck_1636_;
goto v_resetjp_1606_;
}
else
{
lean_inc(v_a_1605_);
lean_dec(v___x_1604_);
v___x_1607_ = lean_box(0);
v_isShared_1608_ = v_isSharedCheck_1636_;
goto v_resetjp_1606_;
}
v_resetjp_1606_:
{
uint8_t v___x_1609_; 
v___x_1609_ = lean_unbox(v_a_1605_);
if (v___x_1609_ == 0)
{
lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1615_; 
v___x_1610_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1610_, 0, v_a_1573_);
lean_ctor_set(v___x_1610_, 1, v_a_1605_);
v___x_1611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1611_, 0, v_a_1568_);
lean_ctor_set(v___x_1611_, 1, v___x_1610_);
v___x_1612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1612_, 0, v_a_1557_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
v___x_1613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1613_, 0, v_a_1550_);
lean_ctor_set(v___x_1613_, 1, v___x_1612_);
if (v_isShared_1608_ == 0)
{
lean_ctor_set(v___x_1607_, 0, v___x_1613_);
v___x_1615_ = v___x_1607_;
goto v_reusejp_1614_;
}
else
{
lean_object* v_reuseFailAlloc_1616_; 
v_reuseFailAlloc_1616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1616_, 0, v___x_1613_);
v___x_1615_ = v_reuseFailAlloc_1616_;
goto v_reusejp_1614_;
}
v_reusejp_1614_:
{
return v___x_1615_;
}
}
else
{
lean_object* v___x_1617_; lean_object* v_a_1618_; lean_object* v___x_1619_; lean_object* v_a_1620_; lean_object* v___x_1621_; lean_object* v_a_1622_; lean_object* v___x_1623_; lean_object* v_a_1624_; lean_object* v___x_1626_; uint8_t v_isShared_1627_; uint8_t v_isSharedCheck_1635_; 
lean_del_object(v___x_1607_);
v___x_1617_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1550_, v___y_1545_);
v_a_1618_ = lean_ctor_get(v___x_1617_, 0);
lean_inc(v_a_1618_);
lean_dec_ref(v___x_1617_);
v___x_1619_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1557_, v___y_1545_);
v_a_1620_ = lean_ctor_get(v___x_1619_, 0);
lean_inc(v_a_1620_);
lean_dec_ref(v___x_1619_);
v___x_1621_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1568_, v___y_1545_);
v_a_1622_ = lean_ctor_get(v___x_1621_, 0);
lean_inc(v_a_1622_);
lean_dec_ref(v___x_1621_);
v___x_1623_ = lp_mathlib_Lean_instantiateMVars___at___00Fin_prod__univ__ofNat_spec__0___redArg(v_a_1573_, v___y_1545_);
v_a_1624_ = lean_ctor_get(v___x_1623_, 0);
v_isSharedCheck_1635_ = !lean_is_exclusive(v___x_1623_);
if (v_isSharedCheck_1635_ == 0)
{
v___x_1626_ = v___x_1623_;
v_isShared_1627_ = v_isSharedCheck_1635_;
goto v_resetjp_1625_;
}
else
{
lean_inc(v_a_1624_);
lean_dec(v___x_1623_);
v___x_1626_ = lean_box(0);
v_isShared_1627_ = v_isSharedCheck_1635_;
goto v_resetjp_1625_;
}
v_resetjp_1625_:
{
lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1633_; 
v___x_1628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1628_, 0, v_a_1624_);
lean_ctor_set(v___x_1628_, 1, v_a_1605_);
v___x_1629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1629_, 0, v_a_1622_);
lean_ctor_set(v___x_1629_, 1, v___x_1628_);
v___x_1630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1630_, 0, v_a_1620_);
lean_ctor_set(v___x_1630_, 1, v___x_1629_);
v___x_1631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1631_, 0, v_a_1618_);
lean_ctor_set(v___x_1631_, 1, v___x_1630_);
if (v_isShared_1627_ == 0)
{
lean_ctor_set(v___x_1626_, 0, v___x_1631_);
v___x_1633_ = v___x_1626_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1634_; 
v_reuseFailAlloc_1634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1634_, 0, v___x_1631_);
v___x_1633_ = v_reuseFailAlloc_1634_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
return v___x_1633_;
}
}
}
}
}
else
{
lean_object* v_a_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1644_; 
lean_dec(v_a_1573_);
lean_dec(v_a_1568_);
lean_dec(v_a_1557_);
lean_dec(v_a_1550_);
v_a_1637_ = lean_ctor_get(v___x_1604_, 0);
v_isSharedCheck_1644_ = !lean_is_exclusive(v___x_1604_);
if (v_isSharedCheck_1644_ == 0)
{
v___x_1639_ = v___x_1604_;
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_a_1637_);
lean_dec(v___x_1604_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1642_; 
if (v_isShared_1640_ == 0)
{
v___x_1642_ = v___x_1639_;
goto v_reusejp_1641_;
}
else
{
lean_object* v_reuseFailAlloc_1643_; 
v_reuseFailAlloc_1643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1643_, 0, v_a_1637_);
v___x_1642_ = v_reuseFailAlloc_1643_;
goto v_reusejp_1641_;
}
v_reusejp_1641_:
{
return v___x_1642_;
}
}
}
}
}
}
else
{
lean_object* v_a_1647_; lean_object* v___x_1649_; uint8_t v_isShared_1650_; uint8_t v_isSharedCheck_1654_; 
lean_dec(v_a_1568_);
lean_dec_ref(v___x_1564_);
lean_dec_ref_known(v___x_1560_, 2);
lean_dec(v_a_1557_);
lean_dec_ref_known(v___x_1552_, 2);
lean_dec(v_a_1550_);
lean_dec_ref(v___y_1544_);
lean_dec_ref(v_snd_1543_);
lean_dec_ref(v_fst_1542_);
v_a_1647_ = lean_ctor_get(v___x_1572_, 0);
v_isSharedCheck_1654_ = !lean_is_exclusive(v___x_1572_);
if (v_isSharedCheck_1654_ == 0)
{
v___x_1649_ = v___x_1572_;
v_isShared_1650_ = v_isSharedCheck_1654_;
goto v_resetjp_1648_;
}
else
{
lean_inc(v_a_1647_);
lean_dec(v___x_1572_);
v___x_1649_ = lean_box(0);
v_isShared_1650_ = v_isSharedCheck_1654_;
goto v_resetjp_1648_;
}
v_resetjp_1648_:
{
lean_object* v___x_1652_; 
if (v_isShared_1650_ == 0)
{
v___x_1652_ = v___x_1649_;
goto v_reusejp_1651_;
}
else
{
lean_object* v_reuseFailAlloc_1653_; 
v_reuseFailAlloc_1653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1653_, 0, v_a_1647_);
v___x_1652_ = v_reuseFailAlloc_1653_;
goto v_reusejp_1651_;
}
v_reusejp_1651_:
{
return v___x_1652_;
}
}
}
}
else
{
lean_object* v_a_1655_; lean_object* v___x_1657_; uint8_t v_isShared_1658_; uint8_t v_isSharedCheck_1662_; 
lean_dec_ref(v___x_1564_);
lean_dec_ref_known(v___x_1560_, 2);
lean_dec(v_a_1557_);
lean_dec_ref_known(v___x_1552_, 2);
lean_dec(v_a_1550_);
lean_dec_ref(v___y_1544_);
lean_dec_ref(v_snd_1543_);
lean_dec_ref(v_fst_1542_);
lean_dec(v___x_1539_);
v_a_1655_ = lean_ctor_get(v___x_1567_, 0);
v_isSharedCheck_1662_ = !lean_is_exclusive(v___x_1567_);
if (v_isSharedCheck_1662_ == 0)
{
v___x_1657_ = v___x_1567_;
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
else
{
lean_inc(v_a_1655_);
lean_dec(v___x_1567_);
v___x_1657_ = lean_box(0);
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
v_resetjp_1656_:
{
lean_object* v___x_1660_; 
if (v_isShared_1658_ == 0)
{
v___x_1660_ = v___x_1657_;
goto v_reusejp_1659_;
}
else
{
lean_object* v_reuseFailAlloc_1661_; 
v_reuseFailAlloc_1661_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1661_, 0, v_a_1655_);
v___x_1660_ = v_reuseFailAlloc_1661_;
goto v_reusejp_1659_;
}
v_reusejp_1659_:
{
return v___x_1660_;
}
}
}
}
else
{
lean_object* v_a_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1670_; 
lean_dec_ref_known(v___x_1552_, 2);
lean_dec(v_a_1550_);
lean_dec_ref(v___y_1544_);
lean_dec_ref(v_snd_1543_);
lean_dec_ref(v_fst_1542_);
lean_dec(v___x_1541_);
lean_dec(v___x_1539_);
v_a_1663_ = lean_ctor_get(v___x_1556_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1556_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1665_ = v___x_1556_;
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_a_1663_);
lean_dec(v___x_1556_);
v___x_1665_ = lean_box(0);
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
v_resetjp_1664_:
{
lean_object* v___x_1668_; 
if (v_isShared_1666_ == 0)
{
v___x_1668_ = v___x_1665_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v_a_1663_);
v___x_1668_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
return v___x_1668_;
}
}
}
}
else
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1678_; 
lean_dec_ref(v___y_1544_);
lean_dec_ref(v_snd_1543_);
lean_dec_ref(v_fst_1542_);
lean_dec(v___x_1541_);
lean_dec(v_a_1540_);
lean_dec(v___x_1539_);
v_a_1671_ = lean_ctor_get(v___x_1549_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1549_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___x_1549_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1549_);
v___x_1673_ = lean_box(0);
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
v_resetjp_1672_:
{
lean_object* v___x_1676_; 
if (v_isShared_1674_ == 0)
{
v___x_1676_ = v___x_1673_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v_a_1671_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___boxed(lean_object* v___x_1679_, lean_object* v___x_1680_, lean_object* v___x_1681_, lean_object* v_a_1682_, lean_object* v___x_1683_, lean_object* v_fst_1684_, lean_object* v_snd_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_){
_start:
{
uint8_t v___x_5933__boxed_1691_; lean_object* v_res_1692_; 
v___x_5933__boxed_1691_ = lean_unbox(v___x_1680_);
v_res_1692_ = lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0(v___x_1679_, v___x_5933__boxed_1691_, v___x_1681_, v_a_1682_, v___x_1683_, v_fst_1684_, v_snd_1685_, v___y_1686_, v___y_1687_, v___y_1688_, v___y_1689_);
lean_dec(v___y_1689_);
lean_dec_ref(v___y_1688_);
lean_dec(v___y_1687_);
return v_res_1692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg(lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_){
_start:
{
lean_object* v___x_1699_; 
v___x_1699_ = lp_Qq_Qq_inferTypeQ(v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
if (lean_obj_tag(v___x_1699_) == 0)
{
lean_object* v_a_1700_; lean_object* v___x_1702_; uint8_t v_isShared_1703_; uint8_t v_isSharedCheck_1792_; 
v_a_1700_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1702_ = v___x_1699_;
v_isShared_1703_ = v_isSharedCheck_1792_;
goto v_resetjp_1701_;
}
else
{
lean_inc(v_a_1700_);
lean_dec(v___x_1699_);
v___x_1702_ = lean_box(0);
v_isShared_1703_ = v_isSharedCheck_1792_;
goto v_resetjp_1701_;
}
v_resetjp_1701_:
{
lean_object* v_snd_1704_; lean_object* v_fst_1705_; lean_object* v_fst_1706_; lean_object* v_snd_1707_; lean_object* v___x_1708_; 
v_snd_1704_ = lean_ctor_get(v_a_1700_, 1);
lean_inc(v_snd_1704_);
v_fst_1705_ = lean_ctor_get(v_a_1700_, 0);
lean_inc(v_fst_1705_);
lean_dec(v_a_1700_);
v_fst_1706_ = lean_ctor_get(v_snd_1704_, 0);
lean_inc(v_fst_1706_);
v_snd_1707_ = lean_ctor_get(v_snd_1704_, 1);
lean_inc(v_snd_1707_);
lean_dec(v_snd_1704_);
v___x_1708_ = ((lean_object*)(lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__0));
if (lean_obj_tag(v_fst_1705_) == 1)
{
lean_object* v_a_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; uint8_t v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___f_1715_; uint8_t v___x_1716_; lean_object* v___x_1717_; 
lean_del_object(v___x_1702_);
v_a_1709_ = lean_ctor_get(v_fst_1705_, 0);
lean_inc_n(v_a_1709_, 2);
lean_dec_ref_known(v_fst_1705_, 1);
v___x_1710_ = lean_box(0);
v___x_1711_ = lean_obj_once(&lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1, &lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1_once, _init_lp_mathlib_Fin_prod__univ__ofNat___redArg___closed__1);
v___x_1712_ = 0;
v___x_1713_ = lean_box(0);
v___x_1714_ = lean_box(v___x_1712_);
lean_inc(v_fst_1706_);
v___f_1715_ = lean_alloc_closure((void*)(lp_mathlib_Fin_sum__univ__ofNat___redArg___lam__0___boxed), 12, 7);
lean_closure_set(v___f_1715_, 0, v___x_1711_);
lean_closure_set(v___f_1715_, 1, v___x_1714_);
lean_closure_set(v___f_1715_, 2, v___x_1713_);
lean_closure_set(v___f_1715_, 3, v_a_1709_);
lean_closure_set(v___f_1715_, 4, v___x_1710_);
lean_closure_set(v___f_1715_, 5, v_fst_1706_);
lean_closure_set(v___f_1715_, 6, v_snd_1707_);
v___x_1716_ = 0;
v___x_1717_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Fin_prod__univ__ofNat_spec__1___redArg(v___f_1715_, v___x_1716_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
if (lean_obj_tag(v___x_1717_) == 0)
{
lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1780_; 
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1780_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1780_ == 0)
{
v___x_1720_ = v___x_1717_;
v_isShared_1721_ = v_isSharedCheck_1780_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1717_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1780_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
lean_object* v_snd_1722_; lean_object* v_snd_1723_; lean_object* v_snd_1724_; lean_object* v_snd_1725_; uint8_t v___x_1726_; 
v_snd_1722_ = lean_ctor_get(v_a_1718_, 1);
lean_inc(v_snd_1722_);
v_snd_1723_ = lean_ctor_get(v_snd_1722_, 1);
lean_inc(v_snd_1723_);
v_snd_1724_ = lean_ctor_get(v_snd_1723_, 1);
lean_inc(v_snd_1724_);
v_snd_1725_ = lean_ctor_get(v_snd_1724_, 1);
lean_inc(v_snd_1725_);
v___x_1726_ = lean_unbox(v_snd_1725_);
if (v___x_1726_ == 0)
{
lean_object* v___x_1728_; 
lean_dec(v_snd_1725_);
lean_dec(v_snd_1724_);
lean_dec(v_snd_1723_);
lean_dec(v_snd_1722_);
lean_dec(v_a_1718_);
lean_dec(v_a_1709_);
lean_dec(v_fst_1706_);
if (v_isShared_1721_ == 0)
{
lean_ctor_set(v___x_1720_, 0, v___x_1708_);
v___x_1728_ = v___x_1720_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v___x_1708_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
else
{
lean_object* v_fst_1730_; lean_object* v_fst_1731_; lean_object* v_fst_1732_; lean_object* v_fst_1733_; lean_object* v___x_1734_; 
v_fst_1730_ = lean_ctor_get(v_a_1718_, 0);
lean_inc_n(v_fst_1730_, 2);
lean_dec(v_a_1718_);
v_fst_1731_ = lean_ctor_get(v_snd_1722_, 0);
lean_inc(v_fst_1731_);
lean_dec(v_snd_1722_);
v_fst_1732_ = lean_ctor_get(v_snd_1723_, 0);
lean_inc(v_fst_1732_);
lean_dec(v_snd_1723_);
v_fst_1733_ = lean_ctor_get(v_snd_1724_, 0);
lean_inc(v_fst_1733_);
lean_dec(v_snd_1724_);
v___x_1734_ = l_Lean_Expr_nat_x3f(v_fst_1730_);
if (lean_obj_tag(v___x_1734_) == 0)
{
lean_object* v___x_1736_; 
lean_dec(v_fst_1733_);
lean_dec(v_fst_1732_);
lean_dec(v_fst_1731_);
lean_dec(v_fst_1730_);
lean_dec(v_snd_1725_);
lean_dec(v_a_1709_);
lean_dec(v_fst_1706_);
if (v_isShared_1721_ == 0)
{
lean_ctor_set(v___x_1720_, 0, v___x_1708_);
v___x_1736_ = v___x_1720_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v___x_1708_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
else
{
lean_object* v_val_1738_; lean_object* v___x_1740_; uint8_t v_isShared_1741_; uint8_t v_isSharedCheck_1779_; 
lean_del_object(v___x_1720_);
v_val_1738_ = lean_ctor_get(v___x_1734_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1734_);
if (v_isSharedCheck_1779_ == 0)
{
v___x_1740_ = v___x_1734_;
v_isShared_1741_ = v_isSharedCheck_1779_;
goto v_resetjp_1739_;
}
else
{
lean_inc(v_val_1738_);
lean_dec(v___x_1734_);
v___x_1740_ = lean_box(0);
v_isShared_1741_ = v_isSharedCheck_1779_;
goto v_resetjp_1739_;
}
v_resetjp_1739_:
{
lean_object* v___x_1742_; 
v___x_1742_ = lp_mathlib_FinVec_mkSumEqQ(v_a_1709_, v_fst_1706_, v_fst_1731_, v_val_1738_, v_fst_1733_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
lean_dec(v_val_1738_);
if (lean_obj_tag(v___x_1742_) == 0)
{
lean_object* v_a_1743_; lean_object* v_fst_1744_; lean_object* v_snd_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; 
v_a_1743_ = lean_ctor_get(v___x_1742_, 0);
lean_inc(v_a_1743_);
lean_dec_ref_known(v___x_1742_, 1);
v_fst_1744_ = lean_ctor_get(v_a_1743_, 0);
lean_inc(v_fst_1744_);
v_snd_1745_ = lean_ctor_get(v_a_1743_, 1);
lean_inc(v_snd_1745_);
lean_dec(v_a_1743_);
v___x_1746_ = lean_obj_once(&lp_mathlib_FinVec_mkProdEqQ___closed__64, &lp_mathlib_FinVec_mkProdEqQ___closed__64_once, _init_lp_mathlib_FinVec_mkProdEqQ___closed__64);
v___x_1747_ = l_Lean_Expr_app___override(v___x_1746_, v_fst_1730_);
v___x_1748_ = lp_Qq_Qq_assertDefEqQ___redArg(v_fst_1732_, v___x_1747_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
if (lean_obj_tag(v___x_1748_) == 0)
{
lean_object* v___x_1750_; uint8_t v_isShared_1751_; uint8_t v_isSharedCheck_1761_; 
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1748_);
if (v_isSharedCheck_1761_ == 0)
{
lean_object* v_unused_1762_; 
v_unused_1762_ = lean_ctor_get(v___x_1748_, 0);
lean_dec(v_unused_1762_);
v___x_1750_ = v___x_1748_;
v_isShared_1751_ = v_isSharedCheck_1761_;
goto v_resetjp_1749_;
}
else
{
lean_dec(v___x_1748_);
v___x_1750_ = lean_box(0);
v_isShared_1751_ = v_isSharedCheck_1761_;
goto v_resetjp_1749_;
}
v_resetjp_1749_:
{
lean_object* v___x_1753_; 
if (v_isShared_1741_ == 0)
{
lean_ctor_set(v___x_1740_, 0, v_snd_1745_);
v___x_1753_ = v___x_1740_;
goto v_reusejp_1752_;
}
else
{
lean_object* v_reuseFailAlloc_1760_; 
v_reuseFailAlloc_1760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1760_, 0, v_snd_1745_);
v___x_1753_ = v_reuseFailAlloc_1760_;
goto v_reusejp_1752_;
}
v_reusejp_1752_:
{
lean_object* v___x_1754_; uint8_t v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1758_; 
v___x_1754_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1754_, 0, v_fst_1744_);
lean_ctor_set(v___x_1754_, 1, v___x_1753_);
v___x_1755_ = lean_unbox(v_snd_1725_);
lean_dec(v_snd_1725_);
lean_ctor_set_uint8(v___x_1754_, sizeof(void*)*2, v___x_1755_);
v___x_1756_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1756_, 0, v___x_1754_);
if (v_isShared_1751_ == 0)
{
lean_ctor_set(v___x_1750_, 0, v___x_1756_);
v___x_1758_ = v___x_1750_;
goto v_reusejp_1757_;
}
else
{
lean_object* v_reuseFailAlloc_1759_; 
v_reuseFailAlloc_1759_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1759_, 0, v___x_1756_);
v___x_1758_ = v_reuseFailAlloc_1759_;
goto v_reusejp_1757_;
}
v_reusejp_1757_:
{
return v___x_1758_;
}
}
}
}
else
{
lean_object* v_a_1763_; lean_object* v___x_1765_; uint8_t v_isShared_1766_; uint8_t v_isSharedCheck_1770_; 
lean_dec(v_snd_1745_);
lean_dec(v_fst_1744_);
lean_del_object(v___x_1740_);
lean_dec(v_snd_1725_);
v_a_1763_ = lean_ctor_get(v___x_1748_, 0);
v_isSharedCheck_1770_ = !lean_is_exclusive(v___x_1748_);
if (v_isSharedCheck_1770_ == 0)
{
v___x_1765_ = v___x_1748_;
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
else
{
lean_inc(v_a_1763_);
lean_dec(v___x_1748_);
v___x_1765_ = lean_box(0);
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
v_resetjp_1764_:
{
lean_object* v___x_1768_; 
if (v_isShared_1766_ == 0)
{
v___x_1768_ = v___x_1765_;
goto v_reusejp_1767_;
}
else
{
lean_object* v_reuseFailAlloc_1769_; 
v_reuseFailAlloc_1769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1769_, 0, v_a_1763_);
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
lean_object* v_a_1771_; lean_object* v___x_1773_; uint8_t v_isShared_1774_; uint8_t v_isSharedCheck_1778_; 
lean_del_object(v___x_1740_);
lean_dec(v_fst_1732_);
lean_dec(v_fst_1730_);
lean_dec(v_snd_1725_);
v_a_1771_ = lean_ctor_get(v___x_1742_, 0);
v_isSharedCheck_1778_ = !lean_is_exclusive(v___x_1742_);
if (v_isSharedCheck_1778_ == 0)
{
v___x_1773_ = v___x_1742_;
v_isShared_1774_ = v_isSharedCheck_1778_;
goto v_resetjp_1772_;
}
else
{
lean_inc(v_a_1771_);
lean_dec(v___x_1742_);
v___x_1773_ = lean_box(0);
v_isShared_1774_ = v_isSharedCheck_1778_;
goto v_resetjp_1772_;
}
v_resetjp_1772_:
{
lean_object* v___x_1776_; 
if (v_isShared_1774_ == 0)
{
v___x_1776_ = v___x_1773_;
goto v_reusejp_1775_;
}
else
{
lean_object* v_reuseFailAlloc_1777_; 
v_reuseFailAlloc_1777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1777_, 0, v_a_1771_);
v___x_1776_ = v_reuseFailAlloc_1777_;
goto v_reusejp_1775_;
}
v_reusejp_1775_:
{
return v___x_1776_;
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
lean_object* v_a_1781_; lean_object* v___x_1783_; uint8_t v_isShared_1784_; uint8_t v_isSharedCheck_1788_; 
lean_dec(v_a_1709_);
lean_dec(v_fst_1706_);
v_a_1781_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1788_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1788_ == 0)
{
v___x_1783_ = v___x_1717_;
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
else
{
lean_inc(v_a_1781_);
lean_dec(v___x_1717_);
v___x_1783_ = lean_box(0);
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
v_resetjp_1782_:
{
lean_object* v___x_1786_; 
if (v_isShared_1784_ == 0)
{
v___x_1786_ = v___x_1783_;
goto v_reusejp_1785_;
}
else
{
lean_object* v_reuseFailAlloc_1787_; 
v_reuseFailAlloc_1787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1787_, 0, v_a_1781_);
v___x_1786_ = v_reuseFailAlloc_1787_;
goto v_reusejp_1785_;
}
v_reusejp_1785_:
{
return v___x_1786_;
}
}
}
}
else
{
lean_object* v___x_1790_; 
lean_dec(v_snd_1707_);
lean_dec(v_fst_1706_);
lean_dec(v_fst_1705_);
if (v_isShared_1703_ == 0)
{
lean_ctor_set(v___x_1702_, 0, v___x_1708_);
v___x_1790_ = v___x_1702_;
goto v_reusejp_1789_;
}
else
{
lean_object* v_reuseFailAlloc_1791_; 
v_reuseFailAlloc_1791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1791_, 0, v___x_1708_);
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
else
{
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1800_; 
v_a_1793_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1800_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1800_ == 0)
{
v___x_1795_ = v___x_1699_;
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1699_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1798_; 
if (v_isShared_1796_ == 0)
{
v___x_1798_ = v___x_1795_;
goto v_reusejp_1797_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v_a_1793_);
v___x_1798_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1797_;
}
v_reusejp_1797_:
{
return v___x_1798_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___redArg___boxed(lean_object* v_a_1801_, lean_object* v_a_1802_, lean_object* v_a_1803_, lean_object* v_a_1804_, lean_object* v_a_1805_, lean_object* v_a_1806_){
_start:
{
lean_object* v_res_1807_; 
v_res_1807_ = lp_mathlib_Fin_sum__univ__ofNat___redArg(v_a_1801_, v_a_1802_, v_a_1803_, v_a_1804_, v_a_1805_);
lean_dec(v_a_1805_);
lean_dec_ref(v_a_1804_);
lean_dec(v_a_1803_);
lean_dec_ref(v_a_1802_);
return v_res_1807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat(lean_object* v_a_1808_, lean_object* v_a_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_, lean_object* v_a_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_, lean_object* v_a_1815_){
_start:
{
lean_object* v___x_1817_; 
v___x_1817_ = lp_mathlib_Fin_sum__univ__ofNat___redArg(v_a_1808_, v_a_1812_, v_a_1813_, v_a_1814_, v_a_1815_);
return v___x_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_sum__univ__ofNat___boxed(lean_object* v_a_1818_, lean_object* v_a_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_, lean_object* v_a_1822_, lean_object* v_a_1823_, lean_object* v_a_1824_, lean_object* v_a_1825_, lean_object* v_a_1826_){
_start:
{
lean_object* v_res_1827_; 
v_res_1827_ = lp_mathlib_Fin_sum__univ__ofNat(v_a_1818_, v_a_1819_, v_a_1820_, v_a_1821_, v_a_1822_, v_a_1823_, v_a_1824_, v_a_1825_);
lean_dec(v_a_1825_);
lean_dec_ref(v_a_1824_);
lean_dec(v_a_1823_);
lean_dec_ref(v_a_1822_);
lean_dec(v_a_1821_);
lean_dec_ref(v_a_1820_);
lean_dec(v_a_1819_);
return v_res_1827_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(builtin);
}
#ifdef __cplusplus
}
#endif
