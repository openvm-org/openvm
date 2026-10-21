// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Abs
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Cast.Order.Ring public import Mathlib.Tactic.NormNum.Basic
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Rat_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "AddGroup"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(211, 76, 74, 39, 69, 162, 229, 135)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "abs"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(11, 180, 28, 55, 197, 20, 206, 35)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Lattice"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "nonexhaustive match"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IsOrderedRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(189, 49, 135, 47, 140, 15, 71, 35)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(236, 38, 194, 105, 137, 30, 136, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toPartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(232, 130, 7, 36, 8, 188, 120, 72)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toSemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(164, 130, 80, 139, 133, 157, 146, 245)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNat_abs_nonneg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(30, 178, 71, 30, 114, 121, 225, 252)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "AddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(226, 82, 90, 134, 221, 253, 108, 55)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toAddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(99, 161, 243, 168, 232, 89, 236, 229)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_abs_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(29, 97, 240, 33, 137, 2, 158, 237)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "IsStrictOrderedRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(91, 31, 27, 198, 71, 31, 228, 59)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "isNNRat_abs_nonneg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(197, 47, 89, 0, 168, 14, 52, 142)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(66, 176, 51, 228, 18, 108, 54, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNNRat_abs_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(117, 77, 230, 9, 8, 229, 139, 50)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "evalAbs"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(104, 14, 62, 153, 10, 53, 246, 113)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(v_e_30_, v___y_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___boxed(lean_object* v_e_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0(v_e_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg(lean_object* v_k_44_, uint8_t v_allowLevelAssignments_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg___boxed(lean_object* v_k_68_, lean_object* v_allowLevelAssignments_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_75_; lean_object* v_res_76_; 
v_allowLevelAssignments_boxed_75_ = lean_unbox(v_allowLevelAssignments_69_);
v_res_76_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg(v_k_68_, v_allowLevelAssignments_boxed_75_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1(lean_object* v_00_u03b1_77_, lean_object* v_k_78_, uint8_t v_allowLevelAssignments_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg(v_k_78_, v_allowLevelAssignments_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___boxed(lean_object* v_00_u03b1_86_, lean_object* v_k_87_, lean_object* v_allowLevelAssignments_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_94_; lean_object* v_res_95_; 
v_allowLevelAssignments_boxed_94_ = lean_unbox(v_allowLevelAssignments_88_);
v_res_95_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1(v_00_u03b1_86_, v_k_87_, v_allowLevelAssignments_boxed_94_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0(lean_object* v___x_102_, uint8_t v___x_103_, lean_object* v___x_104_, lean_object* v___x_105_, lean_object* v_00_u03b1_106_, lean_object* v_x_107_, uint8_t v___x_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v___x_114_; 
lean_inc(v___x_104_);
v___x_114_ = l_Lean_Meta_mkFreshExprMVar(v___x_102_, v___x_103_, v___x_104_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_a_115_);
lean_dec_ref_known(v___x_114_, 1);
v___x_116_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__1));
lean_inc(v___x_105_);
v___x_117_ = l_Lean_Expr_const___override(v___x_116_, v___x_105_);
lean_inc_ref(v_00_u03b1_106_);
v___x_118_ = l_Lean_Expr_app___override(v___x_117_, v_00_u03b1_106_);
v___x_119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_inc(v___x_104_);
v___x_120_ = l_Lean_Meta_mkFreshExprMVar(v___x_119_, v___x_103_, v___x_104_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
if (lean_obj_tag(v___x_120_) == 0)
{
lean_object* v_a_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_a_121_ = lean_ctor_get(v___x_120_, 0);
lean_inc(v_a_121_);
lean_dec_ref_known(v___x_120_, 1);
lean_inc_ref(v_00_u03b1_106_);
v___x_122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_122_, 0, v_00_u03b1_106_);
v___x_123_ = l_Lean_Meta_mkFreshExprMVar(v___x_122_, v___x_103_, v___x_104_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_object* v_a_124_; lean_object* v_keyedConfig_125_; uint8_t v_trackZetaDelta_126_; lean_object* v_zetaDeltaSet_127_; lean_object* v_lctx_128_; lean_object* v_localInstances_129_; lean_object* v_defEqCtx_x3f_130_; lean_object* v_synthPendingDepth_131_; lean_object* v_customCanUnfoldPredicate_x3f_132_; uint8_t v_univApprox_133_; uint8_t v_inTypeClassResolution_134_; uint8_t v_cacheInferType_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_188_; 
v_a_124_ = lean_ctor_get(v___x_123_, 0);
lean_inc(v_a_124_);
lean_dec_ref_known(v___x_123_, 1);
v_keyedConfig_125_ = lean_ctor_get(v___y_109_, 0);
v_trackZetaDelta_126_ = lean_ctor_get_uint8(v___y_109_, sizeof(void*)*7);
v_zetaDeltaSet_127_ = lean_ctor_get(v___y_109_, 1);
v_lctx_128_ = lean_ctor_get(v___y_109_, 2);
v_localInstances_129_ = lean_ctor_get(v___y_109_, 3);
v_defEqCtx_x3f_130_ = lean_ctor_get(v___y_109_, 4);
v_synthPendingDepth_131_ = lean_ctor_get(v___y_109_, 5);
v_customCanUnfoldPredicate_x3f_132_ = lean_ctor_get(v___y_109_, 6);
v_univApprox_133_ = lean_ctor_get_uint8(v___y_109_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_134_ = lean_ctor_get_uint8(v___y_109_, sizeof(void*)*7 + 2);
v_cacheInferType_135_ = lean_ctor_get_uint8(v___y_109_, sizeof(void*)*7 + 3);
v_isSharedCheck_188_ = !lean_is_exclusive(v___y_109_);
if (v_isSharedCheck_188_ == 0)
{
v___x_137_ = v___y_109_;
v_isShared_138_ = v_isSharedCheck_188_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_132_);
lean_inc(v_synthPendingDepth_131_);
lean_inc(v_defEqCtx_x3f_130_);
lean_inc(v_localInstances_129_);
lean_inc(v_lctx_128_);
lean_inc(v_zetaDeltaSet_127_);
lean_inc(v_keyedConfig_125_);
lean_dec(v___y_109_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_188_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; uint8_t v___x_145_; lean_object* v___x_146_; lean_object* v___x_148_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___closed__3));
v___x_140_ = l_Lean_Expr_const___override(v___x_139_, v___x_105_);
v___x_141_ = l_Lean_Expr_app___override(v___x_140_, v_00_u03b1_106_);
lean_inc(v_a_115_);
v___x_142_ = l_Lean_Expr_app___override(v___x_141_, v_a_115_);
lean_inc(v_a_121_);
v___x_143_ = l_Lean_Expr_app___override(v___x_142_, v_a_121_);
lean_inc(v_a_124_);
v___x_144_ = l_Lean_Expr_app___override(v___x_143_, v_a_124_);
v___x_145_ = 2;
v___x_146_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_145_, v_keyedConfig_125_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 0, v___x_146_);
v___x_148_ = v___x_137_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_146_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_zetaDeltaSet_127_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_lctx_128_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_localInstances_129_);
lean_ctor_set(v_reuseFailAlloc_187_, 4, v_defEqCtx_x3f_130_);
lean_ctor_set(v_reuseFailAlloc_187_, 5, v_synthPendingDepth_131_);
lean_ctor_set(v_reuseFailAlloc_187_, 6, v_customCanUnfoldPredicate_x3f_132_);
lean_ctor_set_uint8(v_reuseFailAlloc_187_, sizeof(void*)*7, v_trackZetaDelta_126_);
lean_ctor_set_uint8(v_reuseFailAlloc_187_, sizeof(void*)*7 + 1, v_univApprox_133_);
lean_ctor_set_uint8(v_reuseFailAlloc_187_, sizeof(void*)*7 + 2, v_inTypeClassResolution_134_);
lean_ctor_set_uint8(v_reuseFailAlloc_187_, sizeof(void*)*7 + 3, v_cacheInferType_135_);
v___x_148_ = v_reuseFailAlloc_187_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
lean_object* v___x_149_; 
v___x_149_ = l_Lean_Meta_isExprDefEq(v___x_144_, v_x_107_, v___x_148_, v___y_110_, v___y_111_, v___y_112_);
lean_dec_ref(v___x_148_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_178_; 
v_a_150_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_178_ == 0)
{
v___x_152_ = v___x_149_;
v_isShared_153_ = v_isSharedCheck_178_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_a_150_);
lean_dec(v___x_149_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_178_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
uint8_t v___x_154_; 
v___x_154_ = lean_unbox(v_a_150_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_159_; 
v___x_155_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_155_, 0, v_a_124_);
lean_ctor_set(v___x_155_, 1, v_a_150_);
v___x_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_156_, 0, v_a_121_);
lean_ctor_set(v___x_156_, 1, v___x_155_);
v___x_157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_157_, 0, v_a_115_);
lean_ctor_set(v___x_157_, 1, v___x_156_);
if (v_isShared_153_ == 0)
{
lean_ctor_set(v___x_152_, 0, v___x_157_);
v___x_159_ = v___x_152_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v___x_157_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
else
{
lean_object* v___x_161_; lean_object* v_a_162_; lean_object* v___x_163_; lean_object* v_a_164_; lean_object* v___x_165_; lean_object* v_a_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_177_; 
lean_del_object(v___x_152_);
lean_dec(v_a_150_);
v___x_161_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(v_a_115_, v___y_110_);
v_a_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_a_162_);
lean_dec_ref(v___x_161_);
v___x_163_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(v_a_121_, v___y_110_);
v_a_164_ = lean_ctor_get(v___x_163_, 0);
lean_inc(v_a_164_);
lean_dec_ref(v___x_163_);
v___x_165_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_NormNum_evalAbs_spec__0___redArg(v_a_124_, v___y_110_);
v_a_166_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_177_ == 0)
{
v___x_168_ = v___x_165_;
v_isShared_169_ = v_isSharedCheck_177_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_a_166_);
lean_dec(v___x_165_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_177_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_175_; 
v___x_170_ = lean_box(v___x_108_);
v___x_171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_171_, 0, v_a_166_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v_a_164_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v_a_162_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 0, v___x_173_);
v___x_175_ = v___x_168_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_173_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
}
else
{
lean_object* v_a_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_186_; 
lean_dec(v_a_124_);
lean_dec(v_a_121_);
lean_dec(v_a_115_);
v_a_179_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_186_ == 0)
{
v___x_181_ = v___x_149_;
v_isShared_182_ = v_isSharedCheck_186_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_a_179_);
lean_dec(v___x_149_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_186_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_184_; 
if (v_isShared_182_ == 0)
{
v___x_184_ = v___x_181_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v_a_179_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
}
}
else
{
lean_object* v_a_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_196_; 
lean_dec(v_a_121_);
lean_dec(v_a_115_);
lean_dec_ref(v___y_109_);
lean_dec_ref(v_x_107_);
lean_dec_ref(v_00_u03b1_106_);
lean_dec(v___x_105_);
v_a_189_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_196_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_196_ == 0)
{
v___x_191_ = v___x_123_;
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_a_189_);
lean_dec(v___x_123_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_194_; 
if (v_isShared_192_ == 0)
{
v___x_194_ = v___x_191_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v_a_189_);
v___x_194_ = v_reuseFailAlloc_195_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
return v___x_194_;
}
}
}
}
else
{
lean_object* v_a_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_204_; 
lean_dec(v_a_115_);
lean_dec_ref(v___y_109_);
lean_dec_ref(v_x_107_);
lean_dec_ref(v_00_u03b1_106_);
lean_dec(v___x_105_);
lean_dec(v___x_104_);
v_a_197_ = lean_ctor_get(v___x_120_, 0);
v_isSharedCheck_204_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_204_ == 0)
{
v___x_199_ = v___x_120_;
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_a_197_);
lean_dec(v___x_120_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_202_; 
if (v_isShared_200_ == 0)
{
v___x_202_ = v___x_199_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v_a_197_);
v___x_202_ = v_reuseFailAlloc_203_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
return v___x_202_;
}
}
}
}
else
{
lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_212_; 
lean_dec_ref(v___y_109_);
lean_dec_ref(v_x_107_);
lean_dec_ref(v_00_u03b1_106_);
lean_dec(v___x_105_);
lean_dec(v___x_104_);
v_a_205_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_212_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_212_ == 0)
{
v___x_207_ = v___x_114_;
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_dec(v___x_114_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_210_; 
if (v_isShared_208_ == 0)
{
v___x_210_ = v___x_207_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_a_205_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___boxed(lean_object* v___x_213_, lean_object* v___x_214_, lean_object* v___x_215_, lean_object* v___x_216_, lean_object* v_00_u03b1_217_, lean_object* v_x_218_, lean_object* v___x_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
uint8_t v___x_7879__boxed_225_; uint8_t v___x_7883__boxed_226_; lean_object* v_res_227_; 
v___x_7879__boxed_225_ = lean_unbox(v___x_214_);
v___x_7883__boxed_226_ = lean_unbox(v___x_219_);
v_res_227_ = lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0(v___x_213_, v___x_7879__boxed_225_, v___x_215_, v___x_216_, v_00_u03b1_217_, v_x_218_, v___x_7883__boxed_226_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2(lean_object* v_msgData_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_){
_start:
{
lean_object* v___x_234_; lean_object* v_env_235_; lean_object* v___x_236_; lean_object* v_mctx_237_; lean_object* v_lctx_238_; lean_object* v_options_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_234_ = lean_st_ref_get(v___y_232_);
v_env_235_ = lean_ctor_get(v___x_234_, 0);
lean_inc_ref(v_env_235_);
lean_dec(v___x_234_);
v___x_236_ = lean_st_ref_get(v___y_230_);
v_mctx_237_ = lean_ctor_get(v___x_236_, 0);
lean_inc_ref(v_mctx_237_);
lean_dec(v___x_236_);
v_lctx_238_ = lean_ctor_get(v___y_229_, 2);
v_options_239_ = lean_ctor_get(v___y_231_, 2);
lean_inc_ref(v_options_239_);
lean_inc_ref(v_lctx_238_);
v___x_240_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_240_, 0, v_env_235_);
lean_ctor_set(v___x_240_, 1, v_mctx_237_);
lean_ctor_set(v___x_240_, 2, v_lctx_238_);
lean_ctor_set(v___x_240_, 3, v_options_239_);
v___x_241_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_msgData_228_);
v___x_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2___boxed(lean_object* v_msgData_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2(v_msgData_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(lean_object* v_msg_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v_ref_256_; lean_object* v___x_257_; lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_266_; 
v_ref_256_ = lean_ctor_get(v___y_253_, 5);
v___x_257_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2_spec__2(v_msg_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_);
v_a_258_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_266_ == 0)
{
v___x_260_ = v___x_257_;
v_isShared_261_ = v_isSharedCheck_266_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_a_258_);
lean_dec(v___x_257_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_266_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_262_; lean_object* v___x_264_; 
lean_inc(v_ref_256_);
v___x_262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_262_, 0, v_ref_256_);
lean_ctor_set(v___x_262_, 1, v_a_258_);
if (v_isShared_261_ == 0)
{
lean_ctor_set_tag(v___x_260_, 1);
lean_ctor_set(v___x_260_, 0, v___x_262_);
v___x_264_ = v___x_260_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v___x_262_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg___boxed(lean_object* v_msg_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(v_msg_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
return v_res_273_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__2));
v___x_279_ = l_Lean_stringToMessageData(v___x_278_);
return v___x_279_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__4));
v___x_282_ = l_Lean_stringToMessageData(v___x_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1(uint8_t v___x_355_, lean_object* v_u_356_, lean_object* v_00_u03b1_357_, lean_object* v_x_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; uint8_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___f_374_; uint8_t v___x_375_; lean_object* v___x_376_; 
v___x_364_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__1));
v___x_365_ = lean_box(0);
lean_inc(v_u_356_);
v___x_366_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_366_, 0, v_u_356_);
lean_ctor_set(v___x_366_, 1, v___x_365_);
lean_inc_ref_n(v___x_366_, 2);
v___x_367_ = l_Lean_Expr_const___override(v___x_364_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 2);
v___x_368_ = l_Lean_Expr_app___override(v___x_367_, v_00_u03b1_357_);
v___x_369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
v___x_370_ = 0;
v___x_371_ = lean_box(0);
v___x_372_ = lean_box(v___x_370_);
v___x_373_ = lean_box(v___x_355_);
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__0___boxed), 12, 7);
lean_closure_set(v___f_374_, 0, v___x_369_);
lean_closure_set(v___f_374_, 1, v___x_372_);
lean_closure_set(v___f_374_, 2, v___x_371_);
lean_closure_set(v___f_374_, 3, v___x_366_);
lean_closure_set(v___f_374_, 4, v_00_u03b1_357_);
lean_closure_set(v___f_374_, 5, v_x_358_);
lean_closure_set(v___f_374_, 6, v___x_373_);
v___x_375_ = 0;
v___x_376_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalAbs_spec__1___redArg(v___f_374_, v___x_375_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v_snd_378_; lean_object* v_snd_379_; lean_object* v_snd_380_; uint8_t v___x_381_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_a_377_);
lean_dec_ref_known(v___x_376_, 1);
v_snd_378_ = lean_ctor_get(v_a_377_, 1);
v_snd_379_ = lean_ctor_get(v_snd_378_, 1);
lean_inc(v_snd_379_);
v_snd_380_ = lean_ctor_get(v_snd_379_, 1);
v___x_381_ = lean_unbox(v_snd_380_);
if (v___x_381_ == 0)
{
lean_object* v___x_382_; lean_object* v___x_383_; 
lean_dec(v_snd_379_);
lean_dec(v_a_377_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
lean_dec(v_u_356_);
v___x_382_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__3);
v___x_383_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(v___x_382_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
return v___x_383_;
}
else
{
lean_object* v_fst_384_; lean_object* v_fst_385_; lean_object* v___x_386_; 
v_fst_384_ = lean_ctor_get(v_a_377_, 0);
lean_inc(v_fst_384_);
lean_dec(v_a_377_);
v_fst_385_ = lean_ctor_get(v_snd_379_, 0);
lean_inc_n(v_fst_385_, 2);
lean_dec(v_snd_379_);
lean_inc_ref(v_00_u03b1_357_);
v___x_386_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_356_, v_00_u03b1_357_, v_fst_385_, v___x_375_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_object* v_a_387_; 
v_a_387_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_a_387_);
lean_dec_ref_known(v___x_386_, 1);
switch(lean_obj_tag(v_a_387_))
{
case 0:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
lean_dec_ref_known(v_a_387_, 1);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v___x_388_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__5);
v___x_389_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(v___x_388_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
return v___x_389_;
}
case 1:
{
lean_object* v_inst_390_; lean_object* v_lit_391_; lean_object* v_proof_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_455_; 
v_inst_390_ = lean_ctor_get(v_a_387_, 0);
v_lit_391_ = lean_ctor_get(v_a_387_, 1);
v_proof_392_ = lean_ctor_get(v_a_387_, 2);
v_isSharedCheck_455_ = !lean_is_exclusive(v_a_387_);
if (v_isSharedCheck_455_ == 0)
{
v___x_394_ = v_a_387_;
v_isShared_395_ = v_isSharedCheck_455_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_proof_392_);
lean_inc(v_lit_391_);
lean_inc(v_inst_390_);
lean_dec(v_a_387_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_455_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_396_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__7));
lean_inc_ref(v___x_366_);
v___x_397_ = l_Lean_Expr_const___override(v___x_396_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_398_ = l_Lean_Expr_app___override(v___x_397_, v_00_u03b1_357_);
v___x_399_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_398_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
lean_inc_n(v_a_400_, 2);
lean_dec_ref_known(v___x_399_, 1);
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__9));
lean_inc_ref_n(v___x_366_, 4);
v___x_402_ = l_Lean_Expr_const___override(v___x_401_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 4);
v___x_403_ = l_Lean_Expr_app___override(v___x_402_, v_00_u03b1_357_);
v___x_404_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11));
v___x_405_ = l_Lean_Expr_const___override(v___x_404_, v___x_366_);
v___x_406_ = l_Lean_Expr_app___override(v___x_405_, v_00_u03b1_357_);
v___x_407_ = l_Lean_Expr_app___override(v___x_406_, v_a_400_);
v___x_408_ = l_Lean_Expr_app___override(v___x_403_, v___x_407_);
v___x_409_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14));
v___x_410_ = l_Lean_Expr_const___override(v___x_409_, v___x_366_);
v___x_411_ = l_Lean_Expr_app___override(v___x_410_, v_00_u03b1_357_);
v___x_412_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16));
v___x_413_ = l_Lean_Expr_const___override(v___x_412_, v___x_366_);
v___x_414_ = l_Lean_Expr_app___override(v___x_413_, v_00_u03b1_357_);
lean_inc(v_fst_384_);
v___x_415_ = l_Lean_Expr_app___override(v___x_414_, v_fst_384_);
v___x_416_ = l_Lean_Expr_app___override(v___x_411_, v___x_415_);
v___x_417_ = l_Lean_Expr_app___override(v___x_408_, v___x_416_);
v___x_418_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_417_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_438_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_438_ == 0)
{
v___x_421_ = v___x_418_;
v_isShared_422_ = v_isSharedCheck_438_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_418_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_438_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_433_; 
v___x_423_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__21));
v___x_424_ = l_Lean_Expr_const___override(v___x_423_, v___x_366_);
v___x_425_ = l_Lean_Expr_app___override(v___x_424_, v_00_u03b1_357_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v_a_400_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v_fst_384_);
v___x_428_ = l_Lean_Expr_app___override(v___x_427_, v_a_419_);
v___x_429_ = l_Lean_Expr_app___override(v___x_428_, v_fst_385_);
lean_inc_ref(v_lit_391_);
v___x_430_ = l_Lean_Expr_app___override(v___x_429_, v_lit_391_);
v___x_431_ = l_Lean_Expr_app___override(v___x_430_, v_proof_392_);
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 2, v___x_431_);
v___x_433_ = v___x_394_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v_inst_390_);
lean_ctor_set(v_reuseFailAlloc_437_, 1, v_lit_391_);
lean_ctor_set(v_reuseFailAlloc_437_, 2, v___x_431_);
v___x_433_ = v_reuseFailAlloc_437_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
lean_object* v___x_435_; 
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 0, v___x_433_);
v___x_435_ = v___x_421_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v___x_433_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
else
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_446_; 
lean_dec(v_a_400_);
lean_del_object(v___x_394_);
lean_dec_ref(v_proof_392_);
lean_dec_ref(v_lit_391_);
lean_dec_ref(v_inst_390_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_439_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_446_ == 0)
{
v___x_441_ = v___x_418_;
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_418_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___x_444_; 
if (v_isShared_442_ == 0)
{
v___x_444_ = v___x_441_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_a_439_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
}
else
{
lean_object* v_a_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_454_; 
lean_del_object(v___x_394_);
lean_dec_ref(v_proof_392_);
lean_dec_ref(v_lit_391_);
lean_dec_ref(v_inst_390_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_447_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_454_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_454_ == 0)
{
v___x_449_ = v___x_399_;
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_a_447_);
lean_dec(v___x_399_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v___x_452_; 
if (v_isShared_450_ == 0)
{
v___x_452_ = v___x_449_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_a_447_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
}
}
case 2:
{
lean_object* v_inst_456_; lean_object* v_lit_457_; lean_object* v_proof_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_529_; 
v_inst_456_ = lean_ctor_get(v_a_387_, 0);
v_lit_457_ = lean_ctor_get(v_a_387_, 1);
v_proof_458_ = lean_ctor_get(v_a_387_, 2);
v_isSharedCheck_529_ = !lean_is_exclusive(v_a_387_);
if (v_isSharedCheck_529_ == 0)
{
v___x_460_ = v_a_387_;
v_isShared_461_ = v_isSharedCheck_529_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_proof_458_);
lean_inc(v_lit_457_);
lean_inc(v_inst_456_);
lean_dec(v_a_387_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_529_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_462_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__7));
lean_inc_ref(v___x_366_);
v___x_463_ = l_Lean_Expr_const___override(v___x_462_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_464_ = l_Lean_Expr_app___override(v___x_463_, v_00_u03b1_357_);
v___x_465_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_464_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v_a_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
v_a_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc(v_a_466_);
lean_dec_ref_known(v___x_465_, 1);
v___x_467_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__9));
lean_inc_ref_n(v___x_366_, 4);
v___x_468_ = l_Lean_Expr_const___override(v___x_467_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 4);
v___x_469_ = l_Lean_Expr_app___override(v___x_468_, v_00_u03b1_357_);
v___x_470_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11));
v___x_471_ = l_Lean_Expr_const___override(v___x_470_, v___x_366_);
v___x_472_ = l_Lean_Expr_app___override(v___x_471_, v_00_u03b1_357_);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v_a_466_);
v___x_474_ = l_Lean_Expr_app___override(v___x_469_, v___x_473_);
v___x_475_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14));
v___x_476_ = l_Lean_Expr_const___override(v___x_475_, v___x_366_);
v___x_477_ = l_Lean_Expr_app___override(v___x_476_, v_00_u03b1_357_);
v___x_478_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16));
v___x_479_ = l_Lean_Expr_const___override(v___x_478_, v___x_366_);
v___x_480_ = l_Lean_Expr_app___override(v___x_479_, v_00_u03b1_357_);
lean_inc(v_fst_384_);
v___x_481_ = l_Lean_Expr_app___override(v___x_480_, v_fst_384_);
v___x_482_ = l_Lean_Expr_app___override(v___x_477_, v___x_481_);
v___x_483_ = l_Lean_Expr_app___override(v___x_474_, v___x_482_);
v___x_484_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_483_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_484_) == 0)
{
lean_object* v_a_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_512_; 
v_a_485_ = lean_ctor_get(v___x_484_, 0);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_512_ == 0)
{
v___x_487_ = v___x_484_;
v_isShared_488_ = v_isSharedCheck_512_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_a_485_);
lean_dec(v___x_484_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_512_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_507_; 
v___x_489_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__24));
lean_inc_ref_n(v___x_366_, 2);
v___x_490_ = l_Lean_Expr_const___override(v___x_489_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 2);
v___x_491_ = l_Lean_Expr_app___override(v___x_490_, v_00_u03b1_357_);
v___x_492_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__26));
v___x_493_ = l_Lean_Expr_const___override(v___x_492_, v___x_366_);
v___x_494_ = l_Lean_Expr_app___override(v___x_493_, v_00_u03b1_357_);
lean_inc_ref(v_inst_456_);
v___x_495_ = l_Lean_Expr_app___override(v___x_494_, v_inst_456_);
v___x_496_ = l_Lean_Expr_app___override(v___x_491_, v___x_495_);
v___x_497_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__28));
v___x_498_ = l_Lean_Expr_const___override(v___x_497_, v___x_366_);
v___x_499_ = l_Lean_Expr_app___override(v___x_498_, v_00_u03b1_357_);
v___x_500_ = l_Lean_Expr_app___override(v___x_499_, v_inst_456_);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v_fst_384_);
v___x_502_ = l_Lean_Expr_app___override(v___x_501_, v_a_485_);
v___x_503_ = l_Lean_Expr_app___override(v___x_502_, v_fst_385_);
lean_inc_ref(v_lit_457_);
v___x_504_ = l_Lean_Expr_app___override(v___x_503_, v_lit_457_);
v___x_505_ = l_Lean_Expr_app___override(v___x_504_, v_proof_458_);
if (v_isShared_461_ == 0)
{
lean_ctor_set_tag(v___x_460_, 1);
lean_ctor_set(v___x_460_, 2, v___x_505_);
lean_ctor_set(v___x_460_, 0, v___x_496_);
v___x_507_ = v___x_460_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v___x_496_);
lean_ctor_set(v_reuseFailAlloc_511_, 1, v_lit_457_);
lean_ctor_set(v_reuseFailAlloc_511_, 2, v___x_505_);
v___x_507_ = v_reuseFailAlloc_511_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
lean_object* v___x_509_; 
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 0, v___x_507_);
v___x_509_ = v___x_487_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v___x_507_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
else
{
lean_object* v_a_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_520_; 
lean_del_object(v___x_460_);
lean_dec_ref(v_proof_458_);
lean_dec_ref(v_lit_457_);
lean_dec_ref(v_inst_456_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_513_ = lean_ctor_get(v___x_484_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_520_ == 0)
{
v___x_515_ = v___x_484_;
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_a_513_);
lean_dec(v___x_484_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_518_; 
if (v_isShared_516_ == 0)
{
v___x_518_ = v___x_515_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_a_513_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
}
}
else
{
lean_object* v_a_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_528_; 
lean_del_object(v___x_460_);
lean_dec_ref(v_proof_458_);
lean_dec_ref(v_lit_457_);
lean_dec_ref(v_inst_456_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_521_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_528_ == 0)
{
v___x_523_ = v___x_465_;
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_a_521_);
lean_dec(v___x_465_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_524_ == 0)
{
v___x_526_ = v___x_523_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_a_521_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
}
case 3:
{
lean_object* v_inst_530_; lean_object* v_q_531_; lean_object* v_n_532_; lean_object* v_d_533_; lean_object* v_proof_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_615_; 
v_inst_530_ = lean_ctor_get(v_a_387_, 0);
v_q_531_ = lean_ctor_get(v_a_387_, 1);
v_n_532_ = lean_ctor_get(v_a_387_, 2);
v_d_533_ = lean_ctor_get(v_a_387_, 3);
v_proof_534_ = lean_ctor_get(v_a_387_, 4);
v_isSharedCheck_615_ = !lean_is_exclusive(v_a_387_);
if (v_isSharedCheck_615_ == 0)
{
v___x_536_ = v_a_387_;
v_isShared_537_ = v_isSharedCheck_615_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_proof_534_);
lean_inc(v_d_533_);
lean_inc(v_n_532_);
lean_inc(v_q_531_);
lean_inc(v_inst_530_);
lean_dec(v_a_387_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_615_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v___x_538_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__30));
lean_inc_ref(v___x_366_);
v___x_539_ = l_Lean_Expr_const___override(v___x_538_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_540_ = l_Lean_Expr_app___override(v___x_539_, v_00_u03b1_357_);
v___x_541_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_540_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_541_) == 0)
{
lean_object* v_a_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
v_a_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc(v_a_542_);
lean_dec_ref_known(v___x_541_, 1);
v___x_543_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__32));
lean_inc_ref(v___x_366_);
v___x_544_ = l_Lean_Expr_const___override(v___x_543_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_545_ = l_Lean_Expr_app___override(v___x_544_, v_00_u03b1_357_);
v___x_546_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_545_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_546_) == 0)
{
lean_object* v_a_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v_a_547_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_a_547_);
lean_dec_ref_known(v___x_546_, 1);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__34));
lean_inc_ref_n(v___x_366_, 5);
v___x_549_ = l_Lean_Expr_const___override(v___x_548_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 5);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v_00_u03b1_357_);
v___x_551_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11));
v___x_552_ = l_Lean_Expr_const___override(v___x_551_, v___x_366_);
v___x_553_ = l_Lean_Expr_app___override(v___x_552_, v_00_u03b1_357_);
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36));
v___x_555_ = l_Lean_Expr_const___override(v___x_554_, v___x_366_);
v___x_556_ = l_Lean_Expr_app___override(v___x_555_, v_00_u03b1_357_);
lean_inc(v_a_542_);
v___x_557_ = l_Lean_Expr_app___override(v___x_556_, v_a_542_);
v___x_558_ = l_Lean_Expr_app___override(v___x_553_, v___x_557_);
v___x_559_ = l_Lean_Expr_app___override(v___x_550_, v___x_558_);
v___x_560_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14));
v___x_561_ = l_Lean_Expr_const___override(v___x_560_, v___x_366_);
v___x_562_ = l_Lean_Expr_app___override(v___x_561_, v_00_u03b1_357_);
v___x_563_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16));
v___x_564_ = l_Lean_Expr_const___override(v___x_563_, v___x_366_);
v___x_565_ = l_Lean_Expr_app___override(v___x_564_, v_00_u03b1_357_);
v___x_566_ = l_Lean_Expr_app___override(v___x_565_, v_fst_384_);
v___x_567_ = l_Lean_Expr_app___override(v___x_562_, v___x_566_);
v___x_568_ = l_Lean_Expr_app___override(v___x_559_, v___x_567_);
v___x_569_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_568_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_569_) == 0)
{
lean_object* v_a_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_590_; 
v_a_570_ = lean_ctor_get(v___x_569_, 0);
v_isSharedCheck_590_ = !lean_is_exclusive(v___x_569_);
if (v_isSharedCheck_590_ == 0)
{
v___x_572_ = v___x_569_;
v_isShared_573_ = v_isSharedCheck_590_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_a_570_);
lean_dec(v___x_569_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_590_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_585_; 
v___x_574_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__38));
v___x_575_ = l_Lean_Expr_const___override(v___x_574_, v___x_366_);
v___x_576_ = l_Lean_Expr_app___override(v___x_575_, v_00_u03b1_357_);
v___x_577_ = l_Lean_Expr_app___override(v___x_576_, v_a_542_);
v___x_578_ = l_Lean_Expr_app___override(v___x_577_, v_a_547_);
v___x_579_ = l_Lean_Expr_app___override(v___x_578_, v_a_570_);
v___x_580_ = l_Lean_Expr_app___override(v___x_579_, v_fst_385_);
lean_inc_ref(v_n_532_);
v___x_581_ = l_Lean_Expr_app___override(v___x_580_, v_n_532_);
lean_inc_ref(v_d_533_);
v___x_582_ = l_Lean_Expr_app___override(v___x_581_, v_d_533_);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v_proof_534_);
if (v_isShared_537_ == 0)
{
lean_ctor_set(v___x_536_, 4, v___x_583_);
v___x_585_ = v___x_536_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v_inst_530_);
lean_ctor_set(v_reuseFailAlloc_589_, 1, v_q_531_);
lean_ctor_set(v_reuseFailAlloc_589_, 2, v_n_532_);
lean_ctor_set(v_reuseFailAlloc_589_, 3, v_d_533_);
lean_ctor_set(v_reuseFailAlloc_589_, 4, v___x_583_);
v___x_585_ = v_reuseFailAlloc_589_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
lean_object* v___x_587_; 
if (v_isShared_573_ == 0)
{
lean_ctor_set(v___x_572_, 0, v___x_585_);
v___x_587_ = v___x_572_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v___x_585_);
v___x_587_ = v_reuseFailAlloc_588_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
return v___x_587_;
}
}
}
}
else
{
lean_object* v_a_591_; lean_object* v___x_593_; uint8_t v_isShared_594_; uint8_t v_isSharedCheck_598_; 
lean_dec(v_a_547_);
lean_dec(v_a_542_);
lean_del_object(v___x_536_);
lean_dec_ref(v_proof_534_);
lean_dec_ref(v_d_533_);
lean_dec_ref(v_n_532_);
lean_dec_ref(v_q_531_);
lean_dec_ref(v_inst_530_);
lean_dec(v_fst_385_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_591_ = lean_ctor_get(v___x_569_, 0);
v_isSharedCheck_598_ = !lean_is_exclusive(v___x_569_);
if (v_isSharedCheck_598_ == 0)
{
v___x_593_ = v___x_569_;
v_isShared_594_ = v_isSharedCheck_598_;
goto v_resetjp_592_;
}
else
{
lean_inc(v_a_591_);
lean_dec(v___x_569_);
v___x_593_ = lean_box(0);
v_isShared_594_ = v_isSharedCheck_598_;
goto v_resetjp_592_;
}
v_resetjp_592_:
{
lean_object* v___x_596_; 
if (v_isShared_594_ == 0)
{
v___x_596_ = v___x_593_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_597_; 
v_reuseFailAlloc_597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_597_, 0, v_a_591_);
v___x_596_ = v_reuseFailAlloc_597_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
return v___x_596_;
}
}
}
}
else
{
lean_object* v_a_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_606_; 
lean_dec(v_a_542_);
lean_del_object(v___x_536_);
lean_dec_ref(v_proof_534_);
lean_dec_ref(v_d_533_);
lean_dec_ref(v_n_532_);
lean_dec_ref(v_q_531_);
lean_dec_ref(v_inst_530_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_599_ = lean_ctor_get(v___x_546_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_546_);
if (v_isSharedCheck_606_ == 0)
{
v___x_601_ = v___x_546_;
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_a_599_);
lean_dec(v___x_546_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_604_; 
if (v_isShared_602_ == 0)
{
v___x_604_ = v___x_601_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_a_599_);
v___x_604_ = v_reuseFailAlloc_605_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
return v___x_604_;
}
}
}
}
else
{
lean_object* v_a_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_614_; 
lean_del_object(v___x_536_);
lean_dec_ref(v_proof_534_);
lean_dec_ref(v_d_533_);
lean_dec_ref(v_n_532_);
lean_dec_ref(v_q_531_);
lean_dec_ref(v_inst_530_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_607_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_614_ == 0)
{
v___x_609_ = v___x_541_;
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_a_607_);
lean_dec(v___x_541_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v___x_612_; 
if (v_isShared_610_ == 0)
{
v___x_612_ = v___x_609_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_a_607_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
}
default: 
{
lean_object* v_inst_616_; lean_object* v_q_617_; lean_object* v_n_618_; lean_object* v_d_619_; lean_object* v_proof_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_693_; 
v_inst_616_ = lean_ctor_get(v_a_387_, 0);
v_q_617_ = lean_ctor_get(v_a_387_, 1);
v_n_618_ = lean_ctor_get(v_a_387_, 2);
v_d_619_ = lean_ctor_get(v_a_387_, 3);
v_proof_620_ = lean_ctor_get(v_a_387_, 4);
v_isSharedCheck_693_ = !lean_is_exclusive(v_a_387_);
if (v_isSharedCheck_693_ == 0)
{
v___x_622_ = v_a_387_;
v_isShared_623_ = v_isSharedCheck_693_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_proof_620_);
lean_inc(v_d_619_);
lean_inc(v_n_618_);
lean_inc(v_q_617_);
lean_inc(v_inst_616_);
lean_dec(v_a_387_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_693_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_624_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__32));
lean_inc_ref(v___x_366_);
v___x_625_ = l_Lean_Expr_const___override(v___x_624_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_626_ = l_Lean_Expr_app___override(v___x_625_, v_00_u03b1_357_);
v___x_627_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_626_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_a_628_);
lean_dec_ref_known(v___x_627_, 1);
v___x_629_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__34));
lean_inc_ref_n(v___x_366_, 5);
v___x_630_ = l_Lean_Expr_const___override(v___x_629_, v___x_366_);
lean_inc_ref_n(v_00_u03b1_357_, 5);
v___x_631_ = l_Lean_Expr_app___override(v___x_630_, v_00_u03b1_357_);
v___x_632_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__11));
v___x_633_ = l_Lean_Expr_const___override(v___x_632_, v___x_366_);
v___x_634_ = l_Lean_Expr_app___override(v___x_633_, v_00_u03b1_357_);
v___x_635_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__36));
v___x_636_ = l_Lean_Expr_const___override(v___x_635_, v___x_366_);
v___x_637_ = l_Lean_Expr_app___override(v___x_636_, v_00_u03b1_357_);
lean_inc_ref(v_inst_616_);
v___x_638_ = l_Lean_Expr_app___override(v___x_637_, v_inst_616_);
v___x_639_ = l_Lean_Expr_app___override(v___x_634_, v___x_638_);
v___x_640_ = l_Lean_Expr_app___override(v___x_631_, v___x_639_);
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__14));
v___x_642_ = l_Lean_Expr_const___override(v___x_641_, v___x_366_);
v___x_643_ = l_Lean_Expr_app___override(v___x_642_, v_00_u03b1_357_);
v___x_644_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__16));
v___x_645_ = l_Lean_Expr_const___override(v___x_644_, v___x_366_);
v___x_646_ = l_Lean_Expr_app___override(v___x_645_, v_00_u03b1_357_);
v___x_647_ = l_Lean_Expr_app___override(v___x_646_, v_fst_384_);
v___x_648_ = l_Lean_Expr_app___override(v___x_643_, v___x_647_);
v___x_649_ = l_Lean_Expr_app___override(v___x_640_, v___x_648_);
v___x_650_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_649_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_650_) == 0)
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_676_; 
v_a_651_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_676_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_676_ == 0)
{
v___x_653_ = v___x_650_;
v_isShared_654_ = v_isSharedCheck_676_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_650_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_676_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_671_; 
v___x_655_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__40));
lean_inc_ref(v___x_366_);
v___x_656_ = l_Lean_Expr_const___override(v___x_655_, v___x_366_);
lean_inc_ref(v_00_u03b1_357_);
v___x_657_ = l_Lean_Expr_app___override(v___x_656_, v_00_u03b1_357_);
lean_inc_ref(v_inst_616_);
v___x_658_ = l_Lean_Expr_app___override(v___x_657_, v_inst_616_);
v___x_659_ = l_Rat_neg(v_q_617_);
v___x_660_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___closed__42));
v___x_661_ = l_Lean_Expr_const___override(v___x_660_, v___x_366_);
v___x_662_ = l_Lean_Expr_app___override(v___x_661_, v_00_u03b1_357_);
v___x_663_ = l_Lean_Expr_app___override(v___x_662_, v_inst_616_);
v___x_664_ = l_Lean_Expr_app___override(v___x_663_, v_a_628_);
v___x_665_ = l_Lean_Expr_app___override(v___x_664_, v_a_651_);
v___x_666_ = l_Lean_Expr_app___override(v___x_665_, v_fst_385_);
lean_inc_ref(v_n_618_);
v___x_667_ = l_Lean_Expr_app___override(v___x_666_, v_n_618_);
lean_inc_ref(v_d_619_);
v___x_668_ = l_Lean_Expr_app___override(v___x_667_, v_d_619_);
v___x_669_ = l_Lean_Expr_app___override(v___x_668_, v_proof_620_);
if (v_isShared_623_ == 0)
{
lean_ctor_set_tag(v___x_622_, 3);
lean_ctor_set(v___x_622_, 4, v___x_669_);
lean_ctor_set(v___x_622_, 1, v___x_659_);
lean_ctor_set(v___x_622_, 0, v___x_658_);
v___x_671_ = v___x_622_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_675_; 
v_reuseFailAlloc_675_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_675_, 0, v___x_658_);
lean_ctor_set(v_reuseFailAlloc_675_, 1, v___x_659_);
lean_ctor_set(v_reuseFailAlloc_675_, 2, v_n_618_);
lean_ctor_set(v_reuseFailAlloc_675_, 3, v_d_619_);
lean_ctor_set(v_reuseFailAlloc_675_, 4, v___x_669_);
v___x_671_ = v_reuseFailAlloc_675_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
lean_object* v___x_673_; 
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_671_);
v___x_673_ = v___x_653_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v___x_671_);
v___x_673_ = v_reuseFailAlloc_674_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
return v___x_673_;
}
}
}
}
else
{
lean_object* v_a_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_684_; 
lean_dec(v_a_628_);
lean_del_object(v___x_622_);
lean_dec_ref(v_proof_620_);
lean_dec_ref(v_d_619_);
lean_dec_ref(v_n_618_);
lean_dec_ref(v_q_617_);
lean_dec_ref(v_inst_616_);
lean_dec(v_fst_385_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_677_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_684_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_684_ == 0)
{
v___x_679_ = v___x_650_;
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_a_677_);
lean_dec(v___x_650_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_682_; 
if (v_isShared_680_ == 0)
{
v___x_682_ = v___x_679_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v_a_677_);
v___x_682_ = v_reuseFailAlloc_683_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
return v___x_682_;
}
}
}
}
else
{
lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_692_; 
lean_del_object(v___x_622_);
lean_dec_ref(v_proof_620_);
lean_dec_ref(v_d_619_);
lean_dec_ref(v_n_618_);
lean_dec_ref(v_q_617_);
lean_dec_ref(v_inst_616_);
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
v_a_685_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_692_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_692_ == 0)
{
v___x_687_ = v___x_627_;
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_627_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_690_; 
if (v_isShared_688_ == 0)
{
v___x_690_ = v___x_687_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_a_685_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
}
}
}
}
else
{
lean_dec(v_fst_385_);
lean_dec(v_fst_384_);
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
return v___x_386_;
}
}
}
else
{
lean_object* v_a_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_701_; 
lean_dec_ref_known(v___x_366_, 2);
lean_dec_ref(v_00_u03b1_357_);
lean_dec(v_u_356_);
v_a_694_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_701_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_701_ == 0)
{
v___x_696_ = v___x_376_;
v_isShared_697_ = v_isSharedCheck_701_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_a_694_);
lean_dec(v___x_376_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_701_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_699_; 
if (v_isShared_697_ == 0)
{
v___x_699_ = v___x_696_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v_a_694_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1___boxed(lean_object* v___x_702_, lean_object* v_u_703_, lean_object* v_00_u03b1_704_, lean_object* v_x_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
uint8_t v___x_8386__boxed_711_; lean_object* v_res_712_; 
v___x_8386__boxed_711_ = lean_unbox(v___x_702_);
v_res_712_ = lp_mathlib_Mathlib_Meta_NormNum_evalAbs___lam__1(v___x_8386__boxed_711_, v_u_703_, v_00_u03b1_704_, v_x_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2(lean_object* v_00_u03b1_727_, lean_object* v_msg_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
lean_object* v___x_734_; 
v___x_734_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___redArg(v_msg_728_, v___y_729_, v___y_730_, v___y_731_, v___y_732_);
return v___x_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2___boxed(lean_object* v_00_u03b1_735_, lean_object* v_msg_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_){
_start:
{
lean_object* v_res_742_; 
v_res_742_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_evalAbs_spec__2(v_00_u03b1_735_, v_msg_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_);
lean_dec(v___y_740_);
lean_dec_ref(v___y_739_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
return v_res_742_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Abs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Abs(builtin);
}
#ifdef __cplusplus
}
#endif
