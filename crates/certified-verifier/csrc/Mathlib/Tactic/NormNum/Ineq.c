// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Ineq
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Invertible public import Mathlib.Algebra.Order.Ring.Cast public import Mathlib.Tactic.NormNum.Eq public meta import Aesop public meta import Mathlib.Tactic.ToAdditive
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "not an ordered semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__2_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__4_value),LEAN_SCALAR_PTR_LITERAL(47, 196, 146, 225, 179, 207, 152, 76)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IsOrderedRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__6_value),LEAN_SCALAR_PTR_LITERAL(189, 49, 135, 47, 140, 15, 71, 35)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "not an ordered ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__2_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__2_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4_value),LEAN_SCALAR_PTR_LITERAL(236, 38, 194, 105, 137, 30, 136, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "not a linear ordered semifield"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__4_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "IsStrictOrderedRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__6_value),LEAN_SCALAR_PTR_LITERAL(91, 31, 27, 198, 71, 31, 228, 59)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__8_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toCommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__10_value),LEAN_SCALAR_PTR_LITERAL(134, 142, 86, 147, 34, 154, 178, 196)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toPartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__12_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__13_value),LEAN_SCALAR_PTR_LITERAL(232, 130, 7, 36, 8, 188, 120, 72)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Lattice"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toSemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__15_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__16_value),LEAN_SCALAR_PTR_LITERAL(164, 130, 80, 139, 133, 157, 146, 245)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DistribLattice"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toLattice"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__18_value),LEAN_SCALAR_PTR_LITERAL(211, 66, 65, 127, 78, 58, 2, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__19_value),LEAN_SCALAR_PTR_LITERAL(176, 217, 83, 125, 106, 180, 122, 79)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "instDistribLatticeOfLinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__21_value),LEAN_SCALAR_PTR_LITERAL(204, 88, 224, 186, 247, 116, 10, 234)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "not a linear ordered field"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Field"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toSemifield"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__4_value),LEAN_SCALAR_PTR_LITERAL(104, 104, 95, 86, 153, 146, 86, 112)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Nontrivial"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(122, 234, 164, 90, 175, 175, 198, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_le_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 76, 97, 43, 0, 109, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__21_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isInt_le_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__23_value),LEAN_SCALAR_PTR_LITERAL(157, 125, 254, 173, 172, 82, 248, 238)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivisionRing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__2_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 172, 238, 141, 54, 76, 141, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRat_le_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(174, 197, 167, 230, 7, 170, 155, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isRat_le_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(240, 59, 23, 121, 145, 167, 203, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CharZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(244, 4, 154, 112, 35, 213, 145, 119)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_le_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(165, 119, 97, 84, 171, 182, 55, 73)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_le_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(196, 59, 45, 145, 4, 46, 164, 234)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "evalLE"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__1_value),LEAN_SCALAR_PTR_LITERAL(116, 110, 151, 215, 188, 1, 31, 80)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLE___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isInt_lt_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 231, 74, 196, 111, 223, 154, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isInt_lt_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(46, 129, 106, 90, 35, 173, 168, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 136, 96, 129, 245, 140, 119, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isNNRat_lt_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 181, 249, 132, 216, 214, 198, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__4_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isNNRat_lt_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(50, 232, 130, 198, 135, 203, 207, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRat_lt_false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(145, 244, 51, 69, 214, 41, 161, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isRat_lt_true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 108, 239, 197, 188, 209, 115, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isNat_lt_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(38, 219, 43, 194, 255, 213, 70, 169)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_lt_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(216, 34, 85, 38, 188, 87, 6, 197)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "evalLT"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__1_value),LEAN_SCALAR_PTR_LITERAL(171, 1, 148, 21, 30, 138, 152, 89)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_evalLT___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__0));
v___x_49_ = l_Lean_stringToMessageData(v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring(lean_object* v_u_59_, lean_object* v_00_u03b1_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = l_Lean_Meta_saveState___redArg(v_a_62_, v_a_64_);
if (lean_obj_tag(v___x_66_) == 0)
{
lean_object* v_a_67_; lean_object* v___y_69_; uint8_t v___y_70_; lean_object* v___y_83_; lean_object* v_a_84_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v_a_67_ = lean_ctor_get(v___x_66_, 0);
lean_inc(v_a_67_);
lean_dec_ref_known(v___x_66_, 1);
v___x_87_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__3));
v___x_88_ = lean_box(0);
v___x_89_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_89_, 0, v_u_59_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
lean_inc_ref(v___x_89_);
v___x_90_ = l_Lean_Expr_const___override(v___x_87_, v___x_89_);
lean_inc_ref(v_00_u03b1_60_);
v___x_91_ = l_Lean_Expr_app___override(v___x_90_, v_00_u03b1_60_);
v___x_92_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_91_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
if (lean_obj_tag(v___x_92_) == 0)
{
lean_object* v_a_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v_a_93_ = lean_ctor_get(v___x_92_, 0);
lean_inc(v_a_93_);
lean_dec_ref_known(v___x_92_, 1);
v___x_94_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__5));
lean_inc_ref(v___x_89_);
v___x_95_ = l_Lean_Expr_const___override(v___x_94_, v___x_89_);
lean_inc_ref(v_00_u03b1_60_);
v___x_96_ = l_Lean_Expr_app___override(v___x_95_, v_00_u03b1_60_);
v___x_97_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_96_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
if (lean_obj_tag(v___x_97_) == 0)
{
lean_object* v_a_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v_a_98_ = lean_ctor_get(v___x_97_, 0);
lean_inc_n(v_a_98_, 2);
lean_dec_ref_known(v___x_97_, 1);
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__7));
v___x_100_ = l_Lean_Expr_const___override(v___x_99_, v___x_89_);
v___x_101_ = l_Lean_Expr_app___override(v___x_100_, v_00_u03b1_60_);
lean_inc(v_a_93_);
v___x_102_ = l_Lean_Expr_app___override(v___x_101_, v_a_93_);
v___x_103_ = l_Lean_Expr_app___override(v___x_102_, v_a_98_);
v___x_104_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_103_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
if (lean_obj_tag(v___x_104_) == 0)
{
lean_object* v_a_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_114_; 
lean_dec(v_a_67_);
v_a_105_ = lean_ctor_get(v___x_104_, 0);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_104_);
if (v_isSharedCheck_114_ == 0)
{
v___x_107_ = v___x_104_;
v_isShared_108_ = v_isSharedCheck_114_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_a_105_);
lean_dec(v___x_104_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_114_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_112_; 
v___x_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_109_, 0, v_a_98_);
lean_ctor_set(v___x_109_, 1, v_a_105_);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v_a_93_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 0, v___x_110_);
v___x_112_ = v___x_107_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v___x_110_);
v___x_112_ = v_reuseFailAlloc_113_;
goto v_reusejp_111_;
}
v_reusejp_111_:
{
return v___x_112_;
}
}
}
else
{
lean_object* v_a_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_122_; 
lean_dec(v_a_98_);
lean_dec(v_a_93_);
v_a_115_ = lean_ctor_get(v___x_104_, 0);
v_isSharedCheck_122_ = !lean_is_exclusive(v___x_104_);
if (v_isSharedCheck_122_ == 0)
{
v___x_117_ = v___x_104_;
v_isShared_118_ = v_isSharedCheck_122_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_a_115_);
lean_dec(v___x_104_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_122_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___x_120_; 
lean_inc(v_a_115_);
if (v_isShared_118_ == 0)
{
v___x_120_ = v___x_117_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_121_; 
v_reuseFailAlloc_121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_121_, 0, v_a_115_);
v___x_120_ = v_reuseFailAlloc_121_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
v___y_83_ = v___x_120_;
v_a_84_ = v_a_115_;
goto v___jp_82_;
}
}
}
}
else
{
lean_object* v_a_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_130_; 
lean_dec(v_a_93_);
lean_dec_ref_known(v___x_89_, 2);
lean_dec_ref(v_00_u03b1_60_);
v_a_123_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_130_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_130_ == 0)
{
v___x_125_ = v___x_97_;
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_a_123_);
lean_dec(v___x_97_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_128_; 
lean_inc(v_a_123_);
if (v_isShared_126_ == 0)
{
v___x_128_ = v___x_125_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v_a_123_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
v___y_83_ = v___x_128_;
v_a_84_ = v_a_123_;
goto v___jp_82_;
}
}
}
}
else
{
lean_object* v_a_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_138_; 
lean_dec_ref_known(v___x_89_, 2);
lean_dec_ref(v_00_u03b1_60_);
v_a_131_ = lean_ctor_get(v___x_92_, 0);
v_isSharedCheck_138_ = !lean_is_exclusive(v___x_92_);
if (v_isSharedCheck_138_ == 0)
{
v___x_133_ = v___x_92_;
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_a_131_);
lean_dec(v___x_92_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_136_; 
lean_inc(v_a_131_);
if (v_isShared_134_ == 0)
{
v___x_136_ = v___x_133_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_a_131_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
v___y_83_ = v___x_136_;
v_a_84_ = v_a_131_;
goto v___jp_82_;
}
}
}
v___jp_68_:
{
if (v___y_70_ == 0)
{
lean_object* v___x_71_; 
lean_dec_ref(v___y_69_);
v___x_71_ = l_Lean_Meta_SavedState_restore___redArg(v_a_67_, v_a_62_, v_a_64_);
lean_dec(v_a_67_);
if (lean_obj_tag(v___x_71_) == 0)
{
lean_object* v___x_72_; lean_object* v___x_73_; 
lean_dec_ref_known(v___x_71_, 1);
v___x_72_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__1);
v___x_73_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_72_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
return v___x_73_;
}
else
{
lean_object* v_a_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_81_; 
v_a_74_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_81_ == 0)
{
v___x_76_ = v___x_71_;
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_a_74_);
lean_dec(v___x_71_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_79_; 
if (v_isShared_77_ == 0)
{
v___x_79_ = v___x_76_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_a_74_);
v___x_79_ = v_reuseFailAlloc_80_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
return v___x_79_;
}
}
}
}
else
{
lean_dec(v_a_67_);
return v___y_69_;
}
}
v___jp_82_:
{
uint8_t v___x_85_; 
v___x_85_ = l_Lean_Exception_isInterrupt(v_a_84_);
if (v___x_85_ == 0)
{
uint8_t v___x_86_; 
v___x_86_ = l_Lean_Exception_isRuntime(v_a_84_);
v___y_69_ = v___y_83_;
v___y_70_ = v___x_86_;
goto v___jp_68_;
}
else
{
lean_dec_ref(v_a_84_);
v___y_69_ = v___y_83_;
v___y_70_ = v___x_85_;
goto v___jp_68_;
}
}
}
else
{
lean_object* v_a_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_146_; 
lean_dec_ref(v_00_u03b1_60_);
lean_dec(v_u_59_);
v_a_139_ = lean_ctor_get(v___x_66_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_146_ == 0)
{
v___x_141_ = v___x_66_;
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_a_139_);
lean_dec(v___x_66_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
lean_object* v___x_144_; 
if (v_isShared_142_ == 0)
{
v___x_144_ = v___x_141_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v_a_139_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___boxed(lean_object* v_u_147_, lean_object* v_00_u03b1_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring(v_u_147_, v_00_u03b1_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
lean_dec(v_a_152_);
lean_dec_ref(v_a_151_);
lean_dec(v_a_150_);
lean_dec_ref(v_a_149_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0(lean_object* v_00_u03b1_155_, lean_object* v_msg_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v_msg_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___boxed(lean_object* v_00_u03b1_163_, lean_object* v_msg_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0(v_00_u03b1_163_, v_msg_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
return v_res_170_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__0));
v___x_173_ = l_Lean_stringToMessageData(v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing(lean_object* v_u_181_, lean_object* v_00_u03b1_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = l_Lean_Meta_saveState___redArg(v_a_184_, v_a_186_);
if (lean_obj_tag(v___x_188_) == 0)
{
lean_object* v_a_189_; lean_object* v___y_191_; uint8_t v___y_192_; lean_object* v___y_205_; lean_object* v_a_206_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v_a_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc(v_a_189_);
lean_dec_ref_known(v___x_188_, 1);
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__3));
v___x_210_ = lean_box(0);
v___x_211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_211_, 0, v_u_181_);
lean_ctor_set(v___x_211_, 1, v___x_210_);
lean_inc_ref(v___x_211_);
v___x_212_ = l_Lean_Expr_const___override(v___x_209_, v___x_211_);
lean_inc_ref(v_00_u03b1_182_);
v___x_213_ = l_Lean_Expr_app___override(v___x_212_, v_00_u03b1_182_);
v___x_214_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_213_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_214_) == 0)
{
lean_object* v_a_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_a_215_ = lean_ctor_get(v___x_214_, 0);
lean_inc(v_a_215_);
lean_dec_ref_known(v___x_214_, 1);
v___x_216_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__5));
lean_inc_ref(v___x_211_);
v___x_217_ = l_Lean_Expr_const___override(v___x_216_, v___x_211_);
lean_inc_ref(v_00_u03b1_182_);
v___x_218_ = l_Lean_Expr_app___override(v___x_217_, v_00_u03b1_182_);
v___x_219_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_218_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_219_) == 0)
{
lean_object* v_a_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_a_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc_n(v_a_220_, 2);
lean_dec_ref_known(v___x_219_, 1);
v___x_221_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring___closed__7));
lean_inc_ref(v___x_211_);
v___x_222_ = l_Lean_Expr_const___override(v___x_221_, v___x_211_);
lean_inc_ref(v_00_u03b1_182_);
v___x_223_ = l_Lean_Expr_app___override(v___x_222_, v_00_u03b1_182_);
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__5));
v___x_225_ = l_Lean_Expr_const___override(v___x_224_, v___x_211_);
v___x_226_ = l_Lean_Expr_app___override(v___x_225_, v_00_u03b1_182_);
lean_inc(v_a_215_);
v___x_227_ = l_Lean_Expr_app___override(v___x_226_, v_a_215_);
v___x_228_ = l_Lean_Expr_app___override(v___x_223_, v___x_227_);
v___x_229_ = l_Lean_Expr_app___override(v___x_228_, v_a_220_);
v___x_230_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_229_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_230_) == 0)
{
lean_object* v_a_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_240_; 
lean_dec(v_a_189_);
v_a_231_ = lean_ctor_get(v___x_230_, 0);
v_isSharedCheck_240_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_240_ == 0)
{
v___x_233_ = v___x_230_;
v_isShared_234_ = v_isSharedCheck_240_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_a_231_);
lean_dec(v___x_230_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_240_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_238_; 
v___x_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_235_, 0, v_a_220_);
lean_ctor_set(v___x_235_, 1, v_a_231_);
v___x_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_236_, 0, v_a_215_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 0, v___x_236_);
v___x_238_ = v___x_233_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v___x_236_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
else
{
lean_object* v_a_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
lean_dec(v_a_220_);
lean_dec(v_a_215_);
v_a_241_ = lean_ctor_get(v___x_230_, 0);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_248_ == 0)
{
v___x_243_ = v___x_230_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_a_241_);
lean_dec(v___x_230_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
lean_inc(v_a_241_);
if (v_isShared_244_ == 0)
{
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_241_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
v___y_205_ = v___x_246_;
v_a_206_ = v_a_241_;
goto v___jp_204_;
}
}
}
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
lean_dec(v_a_215_);
lean_dec_ref_known(v___x_211_, 2);
lean_dec_ref(v_00_u03b1_182_);
v_a_249_ = lean_ctor_get(v___x_219_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_219_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_219_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_219_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
lean_inc(v_a_249_);
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
v___y_205_ = v___x_254_;
v_a_206_ = v_a_249_;
goto v___jp_204_;
}
}
}
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec_ref_known(v___x_211_, 2);
lean_dec_ref(v_00_u03b1_182_);
v_a_257_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_214_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_214_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
lean_inc(v_a_257_);
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
v___y_205_ = v___x_262_;
v_a_206_ = v_a_257_;
goto v___jp_204_;
}
}
}
v___jp_190_:
{
if (v___y_192_ == 0)
{
lean_object* v___x_193_; 
lean_dec_ref(v___y_191_);
v___x_193_ = l_Lean_Meta_SavedState_restore___redArg(v_a_189_, v_a_184_, v_a_186_);
lean_dec(v_a_189_);
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec_ref_known(v___x_193_, 1);
v___x_194_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___closed__1);
v___x_195_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_194_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
return v___x_195_;
}
else
{
lean_object* v_a_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_203_; 
v_a_196_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_203_ == 0)
{
v___x_198_ = v___x_193_;
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_a_196_);
lean_dec(v___x_193_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v___x_201_; 
if (v_isShared_199_ == 0)
{
v___x_201_ = v___x_198_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v_a_196_);
v___x_201_ = v_reuseFailAlloc_202_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
return v___x_201_;
}
}
}
}
else
{
lean_dec(v_a_189_);
return v___y_191_;
}
}
v___jp_204_:
{
uint8_t v___x_207_; 
v___x_207_ = l_Lean_Exception_isInterrupt(v_a_206_);
if (v___x_207_ == 0)
{
uint8_t v___x_208_; 
v___x_208_ = l_Lean_Exception_isRuntime(v_a_206_);
v___y_191_ = v___y_205_;
v___y_192_ = v___x_208_;
goto v___jp_190_;
}
else
{
lean_dec_ref(v_a_206_);
v___y_191_ = v___y_205_;
v___y_192_ = v___x_207_;
goto v___jp_190_;
}
}
}
else
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
lean_dec_ref(v_00_u03b1_182_);
lean_dec(v_u_181_);
v_a_265_ = lean_ctor_get(v___x_188_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_188_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_188_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_188_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing___boxed(lean_object* v_u_273_, lean_object* v_00_u03b1_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing(v_u_273_, v_00_u03b1_274_, v_a_275_, v_a_276_, v_a_277_, v_a_278_);
lean_dec(v_a_278_);
lean_dec_ref(v_a_277_);
lean_dec(v_a_276_);
lean_dec_ref(v_a_275_);
return v_res_280_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__0));
v___x_283_ = l_Lean_stringToMessageData(v___x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield(lean_object* v_u_319_, lean_object* v_00_u03b1_320_, lean_object* v_a_321_, lean_object* v_a_322_, lean_object* v_a_323_, lean_object* v_a_324_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = l_Lean_Meta_saveState___redArg(v_a_322_, v_a_324_);
if (lean_obj_tag(v___x_326_) == 0)
{
lean_object* v_a_327_; lean_object* v___y_329_; uint8_t v___y_330_; lean_object* v___y_343_; lean_object* v_a_344_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v_a_327_ = lean_ctor_get(v___x_326_, 0);
lean_inc(v_a_327_);
lean_dec_ref_known(v___x_326_, 1);
v___x_347_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__3));
v___x_348_ = lean_box(0);
v___x_349_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_349_, 0, v_u_319_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
lean_inc_ref(v___x_349_);
v___x_350_ = l_Lean_Expr_const___override(v___x_347_, v___x_349_);
lean_inc_ref(v_00_u03b1_320_);
v___x_351_ = l_Lean_Expr_app___override(v___x_350_, v_00_u03b1_320_);
v___x_352_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_351_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
if (lean_obj_tag(v___x_352_) == 0)
{
lean_object* v_a_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v_a_353_ = lean_ctor_get(v___x_352_, 0);
lean_inc(v_a_353_);
lean_dec_ref_known(v___x_352_, 1);
v___x_354_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__5));
lean_inc_ref(v___x_349_);
v___x_355_ = l_Lean_Expr_const___override(v___x_354_, v___x_349_);
lean_inc_ref(v_00_u03b1_320_);
v___x_356_ = l_Lean_Expr_app___override(v___x_355_, v_00_u03b1_320_);
v___x_357_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_356_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
if (lean_obj_tag(v___x_357_) == 0)
{
lean_object* v_a_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; 
v_a_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc_n(v_a_358_, 2);
lean_dec_ref_known(v___x_357_, 1);
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__7));
lean_inc_ref_n(v___x_349_, 6);
v___x_360_ = l_Lean_Expr_const___override(v___x_359_, v___x_349_);
lean_inc_ref_n(v_00_u03b1_320_, 6);
v___x_361_ = l_Lean_Expr_app___override(v___x_360_, v_00_u03b1_320_);
v___x_362_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9));
v___x_363_ = l_Lean_Expr_const___override(v___x_362_, v___x_349_);
v___x_364_ = l_Lean_Expr_app___override(v___x_363_, v_00_u03b1_320_);
v___x_365_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11));
v___x_366_ = l_Lean_Expr_const___override(v___x_365_, v___x_349_);
v___x_367_ = l_Lean_Expr_app___override(v___x_366_, v_00_u03b1_320_);
lean_inc(v_a_353_);
v___x_368_ = l_Lean_Expr_app___override(v___x_367_, v_a_353_);
v___x_369_ = l_Lean_Expr_app___override(v___x_364_, v___x_368_);
v___x_370_ = l_Lean_Expr_app___override(v___x_361_, v___x_369_);
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14));
v___x_372_ = l_Lean_Expr_const___override(v___x_371_, v___x_349_);
v___x_373_ = l_Lean_Expr_app___override(v___x_372_, v_00_u03b1_320_);
v___x_374_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17));
v___x_375_ = l_Lean_Expr_const___override(v___x_374_, v___x_349_);
v___x_376_ = l_Lean_Expr_app___override(v___x_375_, v_00_u03b1_320_);
v___x_377_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20));
v___x_378_ = l_Lean_Expr_const___override(v___x_377_, v___x_349_);
v___x_379_ = l_Lean_Expr_app___override(v___x_378_, v_00_u03b1_320_);
v___x_380_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__22));
v___x_381_ = l_Lean_Expr_const___override(v___x_380_, v___x_349_);
v___x_382_ = l_Lean_Expr_app___override(v___x_381_, v_00_u03b1_320_);
v___x_383_ = l_Lean_Expr_app___override(v___x_382_, v_a_358_);
v___x_384_ = l_Lean_Expr_app___override(v___x_379_, v___x_383_);
v___x_385_ = l_Lean_Expr_app___override(v___x_376_, v___x_384_);
v___x_386_ = l_Lean_Expr_app___override(v___x_373_, v___x_385_);
v___x_387_ = l_Lean_Expr_app___override(v___x_370_, v___x_386_);
v___x_388_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_387_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
if (lean_obj_tag(v___x_388_) == 0)
{
lean_object* v_a_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_398_; 
lean_dec(v_a_327_);
v_a_389_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_398_ == 0)
{
v___x_391_ = v___x_388_;
v_isShared_392_ = v_isSharedCheck_398_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_a_389_);
lean_dec(v___x_388_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_398_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_396_; 
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_a_358_);
lean_ctor_set(v___x_393_, 1, v_a_389_);
v___x_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_394_, 0, v_a_353_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
if (v_isShared_392_ == 0)
{
lean_ctor_set(v___x_391_, 0, v___x_394_);
v___x_396_ = v___x_391_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v___x_394_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_dec(v_a_358_);
lean_dec(v_a_353_);
v_a_399_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_388_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_388_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
lean_inc(v_a_399_);
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
v___y_343_ = v___x_404_;
v_a_344_ = v_a_399_;
goto v___jp_342_;
}
}
}
}
else
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_414_; 
lean_dec(v_a_353_);
lean_dec_ref_known(v___x_349_, 2);
lean_dec_ref(v_00_u03b1_320_);
v_a_407_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_414_ == 0)
{
v___x_409_ = v___x_357_;
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_357_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_412_; 
lean_inc(v_a_407_);
if (v_isShared_410_ == 0)
{
v___x_412_ = v___x_409_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v_a_407_);
v___x_412_ = v_reuseFailAlloc_413_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
v___y_343_ = v___x_412_;
v_a_344_ = v_a_407_;
goto v___jp_342_;
}
}
}
}
else
{
lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_422_; 
lean_dec_ref_known(v___x_349_, 2);
lean_dec_ref(v_00_u03b1_320_);
v_a_415_ = lean_ctor_get(v___x_352_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_422_ == 0)
{
v___x_417_ = v___x_352_;
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_352_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_420_; 
lean_inc(v_a_415_);
if (v_isShared_418_ == 0)
{
v___x_420_ = v___x_417_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_a_415_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
v___y_343_ = v___x_420_;
v_a_344_ = v_a_415_;
goto v___jp_342_;
}
}
}
v___jp_328_:
{
if (v___y_330_ == 0)
{
lean_object* v___x_331_; 
lean_dec_ref(v___y_329_);
v___x_331_ = l_Lean_Meta_SavedState_restore___redArg(v_a_327_, v_a_322_, v_a_324_);
lean_dec(v_a_327_);
if (lean_obj_tag(v___x_331_) == 0)
{
lean_object* v___x_332_; lean_object* v___x_333_; 
lean_dec_ref_known(v___x_331_, 1);
v___x_332_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__1);
v___x_333_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_332_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
return v___x_333_;
}
else
{
lean_object* v_a_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_341_; 
v_a_334_ = lean_ctor_get(v___x_331_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_331_);
if (v_isSharedCheck_341_ == 0)
{
v___x_336_ = v___x_331_;
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_a_334_);
lean_dec(v___x_331_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_339_; 
if (v_isShared_337_ == 0)
{
v___x_339_ = v___x_336_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_a_334_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
else
{
lean_dec(v_a_327_);
return v___y_329_;
}
}
v___jp_342_:
{
uint8_t v___x_345_; 
v___x_345_ = l_Lean_Exception_isInterrupt(v_a_344_);
if (v___x_345_ == 0)
{
uint8_t v___x_346_; 
v___x_346_ = l_Lean_Exception_isRuntime(v_a_344_);
v___y_329_ = v___y_343_;
v___y_330_ = v___x_346_;
goto v___jp_328_;
}
else
{
lean_dec_ref(v_a_344_);
v___y_329_ = v___y_343_;
v___y_330_ = v___x_345_;
goto v___jp_328_;
}
}
}
else
{
lean_object* v_a_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_430_; 
lean_dec_ref(v_00_u03b1_320_);
lean_dec(v_u_319_);
v_a_423_ = lean_ctor_get(v___x_326_, 0);
v_isSharedCheck_430_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_430_ == 0)
{
v___x_425_ = v___x_326_;
v_isShared_426_ = v_isSharedCheck_430_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_a_423_);
lean_dec(v___x_326_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___boxed(lean_object* v_u_431_, lean_object* v_00_u03b1_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield(v_u_431_, v_00_u03b1_432_, v_a_433_, v_a_434_, v_a_435_, v_a_436_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
lean_dec(v_a_434_);
lean_dec_ref(v_a_433_);
return v_res_438_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1(void){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__0));
v___x_441_ = l_Lean_stringToMessageData(v___x_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField(lean_object* v_u_449_, lean_object* v_00_u03b1_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = l_Lean_Meta_saveState___redArg(v_a_452_, v_a_454_);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_object* v_a_457_; lean_object* v___y_459_; uint8_t v___y_460_; lean_object* v___y_473_; lean_object* v_a_474_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; 
v_a_457_ = lean_ctor_get(v___x_456_, 0);
lean_inc(v_a_457_);
lean_dec_ref_known(v___x_456_, 1);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__3));
v___x_478_ = lean_box(0);
v___x_479_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_479_, 0, v_u_449_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
lean_inc_ref(v___x_479_);
v___x_480_ = l_Lean_Expr_const___override(v___x_477_, v___x_479_);
lean_inc_ref(v_00_u03b1_450_);
v___x_481_ = l_Lean_Expr_app___override(v___x_480_, v_00_u03b1_450_);
v___x_482_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_481_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
if (lean_obj_tag(v___x_482_) == 0)
{
lean_object* v_a_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v_a_483_ = lean_ctor_get(v___x_482_, 0);
lean_inc(v_a_483_);
lean_dec_ref_known(v___x_482_, 1);
v___x_484_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__5));
lean_inc_ref(v___x_479_);
v___x_485_ = l_Lean_Expr_const___override(v___x_484_, v___x_479_);
lean_inc_ref(v_00_u03b1_450_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_00_u03b1_450_);
v___x_487_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_486_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
if (lean_obj_tag(v___x_487_) == 0)
{
lean_object* v_a_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v_a_488_ = lean_ctor_get(v___x_487_, 0);
lean_inc_n(v_a_488_, 2);
lean_dec_ref_known(v___x_487_, 1);
v___x_489_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__7));
lean_inc_ref_n(v___x_479_, 7);
v___x_490_ = l_Lean_Expr_const___override(v___x_489_, v___x_479_);
lean_inc_ref_n(v_00_u03b1_450_, 7);
v___x_491_ = l_Lean_Expr_app___override(v___x_490_, v_00_u03b1_450_);
v___x_492_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__9));
v___x_493_ = l_Lean_Expr_const___override(v___x_492_, v___x_479_);
v___x_494_ = l_Lean_Expr_app___override(v___x_493_, v_00_u03b1_450_);
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__11));
v___x_496_ = l_Lean_Expr_const___override(v___x_495_, v___x_479_);
v___x_497_ = l_Lean_Expr_app___override(v___x_496_, v_00_u03b1_450_);
v___x_498_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__5));
v___x_499_ = l_Lean_Expr_const___override(v___x_498_, v___x_479_);
v___x_500_ = l_Lean_Expr_app___override(v___x_499_, v_00_u03b1_450_);
lean_inc(v_a_483_);
v___x_501_ = l_Lean_Expr_app___override(v___x_500_, v_a_483_);
v___x_502_ = l_Lean_Expr_app___override(v___x_497_, v___x_501_);
v___x_503_ = l_Lean_Expr_app___override(v___x_494_, v___x_502_);
v___x_504_ = l_Lean_Expr_app___override(v___x_491_, v___x_503_);
v___x_505_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__14));
v___x_506_ = l_Lean_Expr_const___override(v___x_505_, v___x_479_);
v___x_507_ = l_Lean_Expr_app___override(v___x_506_, v_00_u03b1_450_);
v___x_508_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__17));
v___x_509_ = l_Lean_Expr_const___override(v___x_508_, v___x_479_);
v___x_510_ = l_Lean_Expr_app___override(v___x_509_, v_00_u03b1_450_);
v___x_511_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__20));
v___x_512_ = l_Lean_Expr_const___override(v___x_511_, v___x_479_);
v___x_513_ = l_Lean_Expr_app___override(v___x_512_, v_00_u03b1_450_);
v___x_514_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield___closed__22));
v___x_515_ = l_Lean_Expr_const___override(v___x_514_, v___x_479_);
v___x_516_ = l_Lean_Expr_app___override(v___x_515_, v_00_u03b1_450_);
v___x_517_ = l_Lean_Expr_app___override(v___x_516_, v_a_488_);
v___x_518_ = l_Lean_Expr_app___override(v___x_513_, v___x_517_);
v___x_519_ = l_Lean_Expr_app___override(v___x_510_, v___x_518_);
v___x_520_ = l_Lean_Expr_app___override(v___x_507_, v___x_519_);
v___x_521_ = l_Lean_Expr_app___override(v___x_504_, v___x_520_);
v___x_522_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_521_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
if (lean_obj_tag(v___x_522_) == 0)
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_532_; 
lean_dec(v_a_457_);
v_a_523_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_532_ == 0)
{
v___x_525_ = v___x_522_;
v_isShared_526_ = v_isSharedCheck_532_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_522_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_532_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_530_; 
v___x_527_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_527_, 0, v_a_488_);
lean_ctor_set(v___x_527_, 1, v_a_523_);
v___x_528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_528_, 0, v_a_483_);
lean_ctor_set(v___x_528_, 1, v___x_527_);
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 0, v___x_528_);
v___x_530_ = v___x_525_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v___x_528_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
return v___x_530_;
}
}
}
else
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_540_; 
lean_dec(v_a_488_);
lean_dec(v_a_483_);
v_a_533_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_540_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_540_ == 0)
{
v___x_535_ = v___x_522_;
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_522_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v___x_538_; 
lean_inc(v_a_533_);
if (v_isShared_536_ == 0)
{
v___x_538_ = v___x_535_;
goto v_reusejp_537_;
}
else
{
lean_object* v_reuseFailAlloc_539_; 
v_reuseFailAlloc_539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_539_, 0, v_a_533_);
v___x_538_ = v_reuseFailAlloc_539_;
goto v_reusejp_537_;
}
v_reusejp_537_:
{
v___y_473_ = v___x_538_;
v_a_474_ = v_a_533_;
goto v___jp_472_;
}
}
}
}
else
{
lean_object* v_a_541_; lean_object* v___x_543_; uint8_t v_isShared_544_; uint8_t v_isSharedCheck_548_; 
lean_dec(v_a_483_);
lean_dec_ref_known(v___x_479_, 2);
lean_dec_ref(v_00_u03b1_450_);
v_a_541_ = lean_ctor_get(v___x_487_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v___x_487_);
if (v_isSharedCheck_548_ == 0)
{
v___x_543_ = v___x_487_;
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
else
{
lean_inc(v_a_541_);
lean_dec(v___x_487_);
v___x_543_ = lean_box(0);
v_isShared_544_ = v_isSharedCheck_548_;
goto v_resetjp_542_;
}
v_resetjp_542_:
{
lean_object* v___x_546_; 
lean_inc(v_a_541_);
if (v_isShared_544_ == 0)
{
v___x_546_ = v___x_543_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v_a_541_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
v___y_473_ = v___x_546_;
v_a_474_ = v_a_541_;
goto v___jp_472_;
}
}
}
}
else
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec_ref_known(v___x_479_, 2);
lean_dec_ref(v_00_u03b1_450_);
v_a_549_ = lean_ctor_get(v___x_482_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_482_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_482_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___x_482_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
lean_inc(v_a_549_);
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_549_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
v___y_473_ = v___x_554_;
v_a_474_ = v_a_549_;
goto v___jp_472_;
}
}
}
v___jp_458_:
{
if (v___y_460_ == 0)
{
lean_object* v___x_461_; 
lean_dec_ref(v___y_459_);
v___x_461_ = l_Lean_Meta_SavedState_restore___redArg(v_a_457_, v_a_452_, v_a_454_);
lean_dec(v_a_457_);
if (lean_obj_tag(v___x_461_) == 0)
{
lean_object* v___x_462_; lean_object* v___x_463_; 
lean_dec_ref_known(v___x_461_, 1);
v___x_462_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___closed__1);
v___x_463_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_462_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
return v___x_463_;
}
else
{
lean_object* v_a_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_471_; 
v_a_464_ = lean_ctor_get(v___x_461_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_461_);
if (v_isSharedCheck_471_ == 0)
{
v___x_466_ = v___x_461_;
v_isShared_467_ = v_isSharedCheck_471_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_a_464_);
lean_dec(v___x_461_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_471_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v___x_469_; 
if (v_isShared_467_ == 0)
{
v___x_469_ = v___x_466_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_a_464_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
}
}
else
{
lean_dec(v_a_457_);
return v___y_459_;
}
}
v___jp_472_:
{
uint8_t v___x_475_; 
v___x_475_ = l_Lean_Exception_isInterrupt(v_a_474_);
if (v___x_475_ == 0)
{
uint8_t v___x_476_; 
v___x_476_ = l_Lean_Exception_isRuntime(v_a_474_);
v___y_459_ = v___y_473_;
v___y_460_ = v___x_476_;
goto v___jp_458_;
}
else
{
lean_dec_ref(v_a_474_);
v___y_459_ = v___y_473_;
v___y_460_ = v___x_475_;
goto v___jp_458_;
}
}
}
else
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_564_; 
lean_dec_ref(v_00_u03b1_450_);
lean_dec(v_u_449_);
v_a_557_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_564_ == 0)
{
v___x_559_ = v___x_456_;
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___x_456_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v___x_562_; 
if (v_isShared_560_ == 0)
{
v___x_562_ = v___x_559_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_a_557_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField___boxed(lean_object* v_u_565_, lean_object* v_00_u03b1_566_, lean_object* v_a_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_a_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField(v_u_565_, v_00_u03b1_566_, v_a_567_, v_a_568_, v_a_569_, v_a_570_);
lean_dec(v_a_570_);
lean_dec_ref(v_a_569_);
lean_dec(v_a_568_);
lean_dec_ref(v_a_567_);
return v_res_572_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0(void){
_start:
{
lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_573_ = lean_box(0);
v___x_574_ = l_Lean_Level_succ___override(v___x_573_);
return v___x_574_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4(void){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_579_ = lean_box(0);
v___x_580_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__0);
v___x_581_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
lean_ctor_set(v___x_581_, 1, v___x_579_);
return v___x_581_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
v___x_585_ = lean_box(0);
v___x_586_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__6));
v___x_587_ = l_Lean_Expr_const___override(v___x_586_, v___x_585_);
return v___x_587_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10(void){
_start:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_592_ = lean_box(0);
v___x_593_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__9));
v___x_594_ = l_Lean_Expr_const___override(v___x_593_, v___x_592_);
return v___x_594_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13(void){
_start:
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; 
v___x_599_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__4);
v___x_600_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__12));
v___x_601_ = l_Lean_Expr_const___override(v___x_600_, v___x_599_);
return v___x_601_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14(void){
_start:
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_602_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__7);
v___x_603_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__13);
v___x_604_ = l_Lean_Expr_app___override(v___x_603_, v___x_602_);
return v___x_604_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15(void){
_start:
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v___x_605_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__10);
v___x_606_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14);
v___x_607_ = l_Lean_Expr_app___override(v___x_606_, v___x_605_);
return v___x_607_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22(void){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; 
v___x_618_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__21));
v___x_619_ = l_Lean_stringToMessageData(v___x_618_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(lean_object* v_u_626_, lean_object* v_00_u03b1_627_, lean_object* v_a_628_, lean_object* v_b_629_, lean_object* v_ra_630_, lean_object* v_rb_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_){
_start:
{
lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v___x_637_ = lean_box(0);
lean_inc_n(v_u_626_, 2);
v___x_638_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_638_, 0, v_u_626_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
lean_inc_ref(v_00_u03b1_627_);
v___x_639_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing(v_u_626_, v_00_u03b1_627_, v_a_632_, v_a_633_, v_a_634_, v_a_635_);
if (lean_obj_tag(v___x_639_) == 0)
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_745_; 
v_a_640_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_745_ == 0)
{
v___x_642_ = v___x_639_;
v_isShared_643_ = v_isSharedCheck_745_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_639_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_745_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v_snd_644_; lean_object* v_fst_645_; lean_object* v_fst_646_; lean_object* v_snd_647_; lean_object* v___y_649_; lean_object* v___y_650_; lean_object* v___y_651_; lean_object* v_a_652_; lean_object* v_a_716_; lean_object* v___x_733_; 
v_snd_644_ = lean_ctor_get(v_a_640_, 1);
lean_inc(v_snd_644_);
v_fst_645_ = lean_ctor_get(v_a_640_, 0);
lean_inc_n(v_fst_645_, 2);
lean_dec(v_a_640_);
v_fst_646_ = lean_ctor_get(v_snd_644_, 0);
lean_inc(v_fst_646_);
v_snd_647_ = lean_ctor_get(v_snd_644_, 1);
lean_inc(v_snd_647_);
lean_dec(v_snd_644_);
lean_inc_ref(v_a_628_);
lean_inc_ref(v_00_u03b1_627_);
lean_inc(v_u_626_);
v___x_733_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_626_, v_00_u03b1_627_, v_a_628_, v_fst_645_, v_ra_630_);
if (lean_obj_tag(v___x_733_) == 0)
{
lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_743_; 
lean_dec(v_snd_647_);
lean_dec(v_fst_646_);
lean_dec(v_fst_645_);
lean_del_object(v___x_642_);
lean_dec_ref_known(v___x_638_, 2);
lean_dec_ref(v_rb_631_);
lean_dec_ref(v_b_629_);
lean_dec_ref(v_a_628_);
lean_dec_ref(v_00_u03b1_627_);
lean_dec(v_u_626_);
v___x_734_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_735_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_734_, v_a_632_, v_a_633_, v_a_634_, v_a_635_);
v_a_736_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_735_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_735_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_741_; 
if (v_isShared_739_ == 0)
{
v___x_741_ = v___x_738_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_736_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
else
{
lean_object* v_val_744_; 
v_val_744_ = lean_ctor_get(v___x_733_, 0);
lean_inc(v_val_744_);
lean_dec_ref_known(v___x_733_, 1);
v_a_716_ = v_val_744_;
goto v___jp_715_;
}
v___jp_648_:
{
lean_object* v_snd_653_; lean_object* v_fst_654_; lean_object* v_fst_655_; lean_object* v_snd_656_; uint8_t v___x_657_; 
v_snd_653_ = lean_ctor_get(v_a_652_, 1);
lean_inc(v_snd_653_);
v_fst_654_ = lean_ctor_get(v_a_652_, 0);
lean_inc(v_fst_654_);
lean_dec_ref(v_a_652_);
v_fst_655_ = lean_ctor_get(v_snd_653_, 0);
lean_inc(v_fst_655_);
v_snd_656_ = lean_ctor_get(v_snd_653_, 1);
lean_inc(v_snd_656_);
lean_dec(v_snd_653_);
v___x_657_ = lean_int_dec_le(v___y_651_, v_fst_654_);
lean_dec(v_fst_654_);
lean_dec(v___y_651_);
if (v___x_657_ == 0)
{
lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
lean_del_object(v___x_642_);
v___x_658_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__2));
lean_inc_ref(v___x_638_);
v___x_659_ = l_Lean_Expr_const___override(v___x_658_, v___x_638_);
lean_inc_ref(v_00_u03b1_627_);
v___x_660_ = l_Lean_Expr_app___override(v___x_659_, v_00_u03b1_627_);
v___x_661_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_660_, v_a_632_, v_a_633_, v_a_634_, v_a_635_);
if (lean_obj_tag(v___x_661_) == 0)
{
lean_object* v_a_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_688_; 
v_a_662_ = lean_ctor_get(v___x_661_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_661_);
if (v_isSharedCheck_688_ == 0)
{
v___x_664_ = v___x_661_;
v_isShared_665_ = v_isSharedCheck_688_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_a_662_);
lean_dec(v___x_661_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_688_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
if (lean_obj_tag(v_a_662_) == 1)
{
lean_object* v_a_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_684_; 
v_a_666_ = lean_ctor_get(v_a_662_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v_a_662_, 1);
v___x_667_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_668_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__20));
v___x_669_ = l_Lean_Expr_const___override(v___x_668_, v___x_638_);
v___x_670_ = l_Lean_Expr_app___override(v___x_669_, v_00_u03b1_627_);
v___x_671_ = l_Lean_Expr_app___override(v___x_670_, v_fst_645_);
v___x_672_ = l_Lean_Expr_app___override(v___x_671_, v_fst_646_);
v___x_673_ = l_Lean_Expr_app___override(v___x_672_, v_snd_647_);
v___x_674_ = l_Lean_Expr_app___override(v___x_673_, v_a_666_);
v___x_675_ = l_Lean_Expr_app___override(v___x_674_, v_a_628_);
v___x_676_ = l_Lean_Expr_app___override(v___x_675_, v_b_629_);
v___x_677_ = l_Lean_Expr_app___override(v___x_676_, v___y_650_);
v___x_678_ = l_Lean_Expr_app___override(v___x_677_, v_fst_655_);
v___x_679_ = l_Lean_Expr_app___override(v___x_678_, v___y_649_);
v___x_680_ = l_Lean_Expr_app___override(v___x_679_, v_snd_656_);
v___x_681_ = l_Lean_Expr_app___override(v___x_680_, v___x_667_);
v___x_682_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_682_, 0, v___x_681_);
lean_ctor_set_uint8(v___x_682_, sizeof(void*)*1, v___x_657_);
if (v_isShared_665_ == 0)
{
lean_ctor_set(v___x_664_, 0, v___x_682_);
v___x_684_ = v___x_664_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_685_; 
v_reuseFailAlloc_685_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_685_, 0, v___x_682_);
v___x_684_ = v_reuseFailAlloc_685_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
return v___x_684_;
}
}
else
{
lean_object* v___x_686_; lean_object* v___x_687_; 
lean_del_object(v___x_664_);
lean_dec(v_a_662_);
lean_dec(v_snd_656_);
lean_dec(v_fst_655_);
lean_dec_ref(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v_snd_647_);
lean_dec(v_fst_646_);
lean_dec(v_fst_645_);
lean_dec_ref_known(v___x_638_, 2);
lean_dec_ref(v_b_629_);
lean_dec_ref(v_a_628_);
lean_dec_ref(v_00_u03b1_627_);
v___x_686_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_687_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_686_, v_a_632_, v_a_633_, v_a_634_, v_a_635_);
return v___x_687_;
}
}
}
else
{
lean_object* v_a_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_696_; 
lean_dec(v_snd_656_);
lean_dec(v_fst_655_);
lean_dec_ref(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v_snd_647_);
lean_dec(v_fst_646_);
lean_dec(v_fst_645_);
lean_dec_ref_known(v___x_638_, 2);
lean_dec_ref(v_b_629_);
lean_dec_ref(v_a_628_);
lean_dec_ref(v_00_u03b1_627_);
v_a_689_ = lean_ctor_get(v___x_661_, 0);
v_isSharedCheck_696_ = !lean_is_exclusive(v___x_661_);
if (v_isSharedCheck_696_ == 0)
{
v___x_691_ = v___x_661_;
v_isShared_692_ = v_isSharedCheck_696_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_a_689_);
lean_dec(v___x_661_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_696_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___x_694_; 
if (v_isShared_692_ == 0)
{
v___x_694_ = v___x_691_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v_a_689_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
return v___x_694_;
}
}
}
}
else
{
lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_713_; 
v___x_697_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_698_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__24));
v___x_699_ = l_Lean_Expr_const___override(v___x_698_, v___x_638_);
v___x_700_ = l_Lean_Expr_app___override(v___x_699_, v_00_u03b1_627_);
v___x_701_ = l_Lean_Expr_app___override(v___x_700_, v_fst_645_);
v___x_702_ = l_Lean_Expr_app___override(v___x_701_, v_fst_646_);
v___x_703_ = l_Lean_Expr_app___override(v___x_702_, v_snd_647_);
v___x_704_ = l_Lean_Expr_app___override(v___x_703_, v_a_628_);
v___x_705_ = l_Lean_Expr_app___override(v___x_704_, v_b_629_);
v___x_706_ = l_Lean_Expr_app___override(v___x_705_, v___y_650_);
v___x_707_ = l_Lean_Expr_app___override(v___x_706_, v_fst_655_);
v___x_708_ = l_Lean_Expr_app___override(v___x_707_, v___y_649_);
v___x_709_ = l_Lean_Expr_app___override(v___x_708_, v_snd_656_);
v___x_710_ = l_Lean_Expr_app___override(v___x_709_, v___x_697_);
v___x_711_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_711_, 0, v___x_710_);
lean_ctor_set_uint8(v___x_711_, sizeof(void*)*1, v___x_657_);
if (v_isShared_643_ == 0)
{
lean_ctor_set(v___x_642_, 0, v___x_711_);
v___x_713_ = v___x_642_;
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
v___jp_715_:
{
lean_object* v_snd_717_; lean_object* v_fst_718_; lean_object* v_fst_719_; lean_object* v_snd_720_; lean_object* v___x_721_; 
v_snd_717_ = lean_ctor_get(v_a_716_, 1);
lean_inc(v_snd_717_);
v_fst_718_ = lean_ctor_get(v_a_716_, 0);
lean_inc(v_fst_718_);
lean_dec_ref(v_a_716_);
v_fst_719_ = lean_ctor_get(v_snd_717_, 0);
lean_inc(v_fst_719_);
v_snd_720_ = lean_ctor_get(v_snd_717_, 1);
lean_inc(v_snd_720_);
lean_dec(v_snd_717_);
lean_inc(v_fst_645_);
lean_inc_ref(v_b_629_);
lean_inc_ref(v_00_u03b1_627_);
v___x_721_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_626_, v_00_u03b1_627_, v_b_629_, v_fst_645_, v_rb_631_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v_a_724_; lean_object* v___x_726_; uint8_t v_isShared_727_; uint8_t v_isSharedCheck_731_; 
lean_dec(v_snd_720_);
lean_dec(v_fst_719_);
lean_dec(v_fst_718_);
lean_dec(v_snd_647_);
lean_dec(v_fst_646_);
lean_dec(v_fst_645_);
lean_del_object(v___x_642_);
lean_dec_ref_known(v___x_638_, 2);
lean_dec_ref(v_b_629_);
lean_dec_ref(v_a_628_);
lean_dec_ref(v_00_u03b1_627_);
v___x_722_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_723_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_722_, v_a_632_, v_a_633_, v_a_634_, v_a_635_);
v_a_724_ = lean_ctor_get(v___x_723_, 0);
v_isSharedCheck_731_ = !lean_is_exclusive(v___x_723_);
if (v_isSharedCheck_731_ == 0)
{
v___x_726_ = v___x_723_;
v_isShared_727_ = v_isSharedCheck_731_;
goto v_resetjp_725_;
}
else
{
lean_inc(v_a_724_);
lean_dec(v___x_723_);
v___x_726_ = lean_box(0);
v_isShared_727_ = v_isSharedCheck_731_;
goto v_resetjp_725_;
}
v_resetjp_725_:
{
lean_object* v___x_729_; 
if (v_isShared_727_ == 0)
{
v___x_729_ = v___x_726_;
goto v_reusejp_728_;
}
else
{
lean_object* v_reuseFailAlloc_730_; 
v_reuseFailAlloc_730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_730_, 0, v_a_724_);
v___x_729_ = v_reuseFailAlloc_730_;
goto v_reusejp_728_;
}
v_reusejp_728_:
{
return v___x_729_;
}
}
}
else
{
lean_object* v_val_732_; 
v_val_732_ = lean_ctor_get(v___x_721_, 0);
lean_inc(v_val_732_);
lean_dec_ref_known(v___x_721_, 1);
v___y_649_ = v_snd_720_;
v___y_650_ = v_fst_719_;
v___y_651_ = v_fst_718_;
v_a_652_ = v_val_732_;
goto v___jp_648_;
}
}
}
}
else
{
lean_object* v_a_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_753_; 
lean_dec_ref_known(v___x_638_, 2);
lean_dec_ref(v_rb_631_);
lean_dec_ref(v_ra_630_);
lean_dec_ref(v_b_629_);
lean_dec_ref(v_a_628_);
lean_dec_ref(v_00_u03b1_627_);
lean_dec(v_u_626_);
v_a_746_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_753_ == 0)
{
v___x_748_ = v___x_639_;
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_a_746_);
lean_dec(v___x_639_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_751_; 
if (v_isShared_749_ == 0)
{
v___x_751_ = v___x_748_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_a_746_);
v___x_751_ = v_reuseFailAlloc_752_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
return v___x_751_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___boxed(lean_object* v_u_754_, lean_object* v_00_u03b1_755_, lean_object* v_a_756_, lean_object* v_b_757_, lean_object* v_ra_758_, lean_object* v_rb_759_, lean_object* v_a_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v_a_763_, lean_object* v_a_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_754_, v_00_u03b1_755_, v_a_756_, v_b_757_, v_ra_758_, v_rb_759_, v_a_760_, v_a_761_, v_a_762_, v_a_763_);
lean_dec(v_a_763_);
lean_dec_ref(v_a_762_);
lean_dec(v_a_761_);
lean_dec_ref(v_a_760_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm(lean_object* v_u_766_, lean_object* v_00_u03b1_767_, lean_object* v_l_u03b1_768_, lean_object* v_a_769_, lean_object* v_b_770_, lean_object* v_ra_771_, lean_object* v_rb_772_, lean_object* v_a_773_, lean_object* v_a_774_, lean_object* v_a_775_, lean_object* v_a_776_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_766_, v_00_u03b1_767_, v_a_769_, v_b_770_, v_ra_771_, v_rb_772_, v_a_773_, v_a_774_, v_a_775_, v_a_776_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___boxed(lean_object* v_u_779_, lean_object* v_00_u03b1_780_, lean_object* v_l_u03b1_781_, lean_object* v_a_782_, lean_object* v_b_783_, lean_object* v_ra_784_, lean_object* v_rb_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_){
_start:
{
lean_object* v_res_791_; 
v_res_791_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm(v_u_779_, v_00_u03b1_780_, v_l_u03b1_781_, v_a_782_, v_b_783_, v_ra_784_, v_rb_785_, v_a_786_, v_a_787_, v_a_788_, v_a_789_);
lean_dec(v_a_789_);
lean_dec_ref(v_a_788_);
lean_dec(v_a_787_);
lean_dec_ref(v_a_786_);
lean_dec_ref(v_l_u03b1_781_);
return v_res_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(lean_object* v_u_813_, lean_object* v_00_u03b1_814_, lean_object* v_a_815_, lean_object* v_b_816_, lean_object* v_ra_817_, lean_object* v_rb_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_){
_start:
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_824_ = lean_box(0);
lean_inc_n(v_u_813_, 2);
v___x_825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_825_, 0, v_u_813_);
lean_ctor_set(v___x_825_, 1, v___x_824_);
lean_inc_ref(v_00_u03b1_814_);
v___x_826_ = lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField(v_u_813_, v_00_u03b1_814_, v_a_819_, v_a_820_, v_a_821_, v_a_822_);
if (lean_obj_tag(v___x_826_) == 0)
{
lean_object* v_a_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_932_; 
v_a_827_ = lean_ctor_get(v___x_826_, 0);
v_isSharedCheck_932_ = !lean_is_exclusive(v___x_826_);
if (v_isSharedCheck_932_ == 0)
{
v___x_829_ = v___x_826_;
v_isShared_830_ = v_isSharedCheck_932_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_a_827_);
lean_dec(v___x_826_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_932_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v_snd_831_; lean_object* v_fst_832_; lean_object* v_fst_833_; lean_object* v_snd_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___y_840_; lean_object* v___y_841_; lean_object* v___y_842_; lean_object* v___y_843_; lean_object* v_a_844_; lean_object* v_a_901_; lean_object* v___x_920_; 
v_snd_831_ = lean_ctor_get(v_a_827_, 1);
lean_inc(v_snd_831_);
v_fst_832_ = lean_ctor_get(v_a_827_, 0);
lean_inc(v_fst_832_);
lean_dec(v_a_827_);
v_fst_833_ = lean_ctor_get(v_snd_831_, 0);
lean_inc(v_fst_833_);
v_snd_834_ = lean_ctor_get(v_snd_831_, 1);
lean_inc(v_snd_834_);
lean_dec(v_snd_831_);
v___x_835_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2));
lean_inc_ref(v___x_825_);
v___x_836_ = l_Lean_Expr_const___override(v___x_835_, v___x_825_);
lean_inc_ref_n(v_00_u03b1_814_, 2);
v___x_837_ = l_Lean_Expr_app___override(v___x_836_, v_00_u03b1_814_);
v___x_838_ = l_Lean_Expr_app___override(v___x_837_, v_fst_832_);
lean_inc_ref(v___x_838_);
lean_inc_ref(v_a_815_);
lean_inc(v_u_813_);
v___x_920_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_813_, v_00_u03b1_814_, v_a_815_, v___x_838_, v_ra_817_);
if (lean_obj_tag(v___x_920_) == 0)
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_930_; 
lean_dec_ref(v___x_838_);
lean_dec(v_snd_834_);
lean_dec(v_fst_833_);
lean_del_object(v___x_829_);
lean_dec_ref_known(v___x_825_, 2);
lean_dec_ref(v_rb_818_);
lean_dec_ref(v_b_816_);
lean_dec_ref(v_a_815_);
lean_dec_ref(v_00_u03b1_814_);
lean_dec(v_u_813_);
v___x_921_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_922_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_921_, v_a_819_, v_a_820_, v_a_821_, v_a_822_);
v_a_923_ = lean_ctor_get(v___x_922_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_930_ == 0)
{
v___x_925_ = v___x_922_;
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_922_);
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
else
{
lean_object* v_val_931_; 
v_val_931_ = lean_ctor_get(v___x_920_, 0);
lean_inc(v_val_931_);
lean_dec_ref_known(v___x_920_, 1);
v_a_901_ = v_val_931_;
goto v___jp_900_;
}
v___jp_839_:
{
lean_object* v_snd_845_; lean_object* v_snd_846_; lean_object* v_fst_847_; lean_object* v_fst_848_; lean_object* v_fst_849_; lean_object* v_snd_850_; uint8_t v___x_851_; 
v_snd_845_ = lean_ctor_get(v_a_844_, 1);
lean_inc(v_snd_845_);
v_snd_846_ = lean_ctor_get(v_snd_845_, 1);
lean_inc(v_snd_846_);
v_fst_847_ = lean_ctor_get(v_a_844_, 0);
lean_inc(v_fst_847_);
lean_dec_ref(v_a_844_);
v_fst_848_ = lean_ctor_get(v_snd_845_, 0);
lean_inc(v_fst_848_);
lean_dec(v_snd_845_);
v_fst_849_ = lean_ctor_get(v_snd_846_, 0);
lean_inc(v_fst_849_);
v_snd_850_ = lean_ctor_get(v_snd_846_, 1);
lean_inc(v_snd_850_);
lean_dec(v_snd_846_);
v___x_851_ = l_Rat_instDecidableLe(v___y_843_, v_fst_847_);
if (v___x_851_ == 0)
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_874_; 
v___x_852_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_853_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__4));
lean_inc_ref(v___x_825_);
v___x_854_ = l_Lean_Expr_const___override(v___x_853_, v___x_825_);
lean_inc_ref(v_00_u03b1_814_);
v___x_855_ = l_Lean_Expr_app___override(v___x_854_, v_00_u03b1_814_);
v___x_856_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6));
v___x_857_ = l_Lean_Expr_const___override(v___x_856_, v___x_825_);
v___x_858_ = l_Lean_Expr_app___override(v___x_857_, v_00_u03b1_814_);
v___x_859_ = l_Lean_Expr_app___override(v___x_858_, v___x_838_);
v___x_860_ = l_Lean_Expr_app___override(v___x_855_, v___x_859_);
v___x_861_ = l_Lean_Expr_app___override(v___x_860_, v_fst_833_);
v___x_862_ = l_Lean_Expr_app___override(v___x_861_, v_snd_834_);
v___x_863_ = l_Lean_Expr_app___override(v___x_862_, v_a_815_);
v___x_864_ = l_Lean_Expr_app___override(v___x_863_, v_b_816_);
v___x_865_ = l_Lean_Expr_app___override(v___x_864_, v___y_842_);
v___x_866_ = l_Lean_Expr_app___override(v___x_865_, v_fst_848_);
v___x_867_ = l_Lean_Expr_app___override(v___x_866_, v___y_840_);
v___x_868_ = l_Lean_Expr_app___override(v___x_867_, v_fst_849_);
v___x_869_ = l_Lean_Expr_app___override(v___x_868_, v___y_841_);
v___x_870_ = l_Lean_Expr_app___override(v___x_869_, v_snd_850_);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v___x_852_);
v___x_872_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_872_, 0, v___x_871_);
lean_ctor_set_uint8(v___x_872_, sizeof(void*)*1, v___x_851_);
if (v_isShared_830_ == 0)
{
lean_ctor_set(v___x_829_, 0, v___x_872_);
v___x_874_ = v___x_829_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v___x_872_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
return v___x_874_;
}
}
else
{
lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_898_; 
v___x_876_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_877_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__8));
lean_inc_ref(v___x_825_);
v___x_878_ = l_Lean_Expr_const___override(v___x_877_, v___x_825_);
lean_inc_ref(v_00_u03b1_814_);
v___x_879_ = l_Lean_Expr_app___override(v___x_878_, v_00_u03b1_814_);
v___x_880_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6));
v___x_881_ = l_Lean_Expr_const___override(v___x_880_, v___x_825_);
v___x_882_ = l_Lean_Expr_app___override(v___x_881_, v_00_u03b1_814_);
v___x_883_ = l_Lean_Expr_app___override(v___x_882_, v___x_838_);
v___x_884_ = l_Lean_Expr_app___override(v___x_879_, v___x_883_);
v___x_885_ = l_Lean_Expr_app___override(v___x_884_, v_fst_833_);
v___x_886_ = l_Lean_Expr_app___override(v___x_885_, v_snd_834_);
v___x_887_ = l_Lean_Expr_app___override(v___x_886_, v_a_815_);
v___x_888_ = l_Lean_Expr_app___override(v___x_887_, v_b_816_);
v___x_889_ = l_Lean_Expr_app___override(v___x_888_, v___y_842_);
v___x_890_ = l_Lean_Expr_app___override(v___x_889_, v_fst_848_);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v___y_840_);
v___x_892_ = l_Lean_Expr_app___override(v___x_891_, v_fst_849_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v___y_841_);
v___x_894_ = l_Lean_Expr_app___override(v___x_893_, v_snd_850_);
v___x_895_ = l_Lean_Expr_app___override(v___x_894_, v___x_876_);
v___x_896_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_896_, 0, v___x_895_);
lean_ctor_set_uint8(v___x_896_, sizeof(void*)*1, v___x_851_);
if (v_isShared_830_ == 0)
{
lean_ctor_set(v___x_829_, 0, v___x_896_);
v___x_898_ = v___x_829_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v___x_896_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
v___jp_900_:
{
lean_object* v_snd_902_; lean_object* v_snd_903_; lean_object* v_fst_904_; lean_object* v_fst_905_; lean_object* v_fst_906_; lean_object* v_snd_907_; lean_object* v___x_908_; 
v_snd_902_ = lean_ctor_get(v_a_901_, 1);
lean_inc(v_snd_902_);
v_snd_903_ = lean_ctor_get(v_snd_902_, 1);
lean_inc(v_snd_903_);
v_fst_904_ = lean_ctor_get(v_a_901_, 0);
lean_inc(v_fst_904_);
lean_dec_ref(v_a_901_);
v_fst_905_ = lean_ctor_get(v_snd_902_, 0);
lean_inc(v_fst_905_);
lean_dec(v_snd_902_);
v_fst_906_ = lean_ctor_get(v_snd_903_, 0);
lean_inc(v_fst_906_);
v_snd_907_ = lean_ctor_get(v_snd_903_, 1);
lean_inc(v_snd_907_);
lean_dec(v_snd_903_);
lean_inc_ref(v___x_838_);
lean_inc_ref(v_b_816_);
lean_inc_ref(v_00_u03b1_814_);
v___x_908_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_813_, v_00_u03b1_814_, v_b_816_, v___x_838_, v_rb_818_);
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v_a_911_; lean_object* v___x_913_; uint8_t v_isShared_914_; uint8_t v_isSharedCheck_918_; 
lean_dec(v_snd_907_);
lean_dec(v_fst_906_);
lean_dec(v_fst_905_);
lean_dec(v_fst_904_);
lean_dec_ref(v___x_838_);
lean_dec(v_snd_834_);
lean_dec(v_fst_833_);
lean_del_object(v___x_829_);
lean_dec_ref_known(v___x_825_, 2);
lean_dec_ref(v_b_816_);
lean_dec_ref(v_a_815_);
lean_dec_ref(v_00_u03b1_814_);
v___x_909_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_910_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_909_, v_a_819_, v_a_820_, v_a_821_, v_a_822_);
v_a_911_ = lean_ctor_get(v___x_910_, 0);
v_isSharedCheck_918_ = !lean_is_exclusive(v___x_910_);
if (v_isSharedCheck_918_ == 0)
{
v___x_913_ = v___x_910_;
v_isShared_914_ = v_isSharedCheck_918_;
goto v_resetjp_912_;
}
else
{
lean_inc(v_a_911_);
lean_dec(v___x_910_);
v___x_913_ = lean_box(0);
v_isShared_914_ = v_isSharedCheck_918_;
goto v_resetjp_912_;
}
v_resetjp_912_:
{
lean_object* v___x_916_; 
if (v_isShared_914_ == 0)
{
v___x_916_ = v___x_913_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_917_; 
v_reuseFailAlloc_917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_917_, 0, v_a_911_);
v___x_916_ = v_reuseFailAlloc_917_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
return v___x_916_;
}
}
}
else
{
lean_object* v_val_919_; 
v_val_919_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_val_919_);
lean_dec_ref_known(v___x_908_, 1);
v___y_840_ = v_fst_906_;
v___y_841_ = v_snd_907_;
v___y_842_ = v_fst_905_;
v___y_843_ = v_fst_904_;
v_a_844_ = v_val_919_;
goto v___jp_839_;
}
}
}
}
else
{
lean_object* v_a_933_; lean_object* v___x_935_; uint8_t v_isShared_936_; uint8_t v_isSharedCheck_940_; 
lean_dec_ref_known(v___x_825_, 2);
lean_dec_ref(v_rb_818_);
lean_dec_ref(v_ra_817_);
lean_dec_ref(v_b_816_);
lean_dec_ref(v_a_815_);
lean_dec_ref(v_00_u03b1_814_);
lean_dec(v_u_813_);
v_a_933_ = lean_ctor_get(v___x_826_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_826_);
if (v_isSharedCheck_940_ == 0)
{
v___x_935_ = v___x_826_;
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
else
{
lean_inc(v_a_933_);
lean_dec(v___x_826_);
v___x_935_ = lean_box(0);
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
v_resetjp_934_:
{
lean_object* v___x_938_; 
if (v_isShared_936_ == 0)
{
v___x_938_ = v___x_935_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v_a_933_);
v___x_938_ = v_reuseFailAlloc_939_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
return v___x_938_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___boxed(lean_object* v_u_941_, lean_object* v_00_u03b1_942_, lean_object* v_a_943_, lean_object* v_b_944_, lean_object* v_ra_945_, lean_object* v_rb_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_941_, v_00_u03b1_942_, v_a_943_, v_b_944_, v_ra_945_, v_rb_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_);
lean_dec(v_a_950_);
lean_dec_ref(v_a_949_);
lean_dec(v_a_948_);
lean_dec_ref(v_a_947_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm(lean_object* v_u_953_, lean_object* v_00_u03b1_954_, lean_object* v_l_u03b1_955_, lean_object* v_a_956_, lean_object* v_b_957_, lean_object* v_ra_958_, lean_object* v_rb_959_, lean_object* v_a_960_, lean_object* v_a_961_, lean_object* v_a_962_, lean_object* v_a_963_){
_start:
{
lean_object* v___x_965_; 
v___x_965_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_953_, v_00_u03b1_954_, v_a_956_, v_b_957_, v_ra_958_, v_rb_959_, v_a_960_, v_a_961_, v_a_962_, v_a_963_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___boxed(lean_object* v_u_966_, lean_object* v_00_u03b1_967_, lean_object* v_l_u03b1_968_, lean_object* v_a_969_, lean_object* v_b_970_, lean_object* v_ra_971_, lean_object* v_rb_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm(v_u_966_, v_00_u03b1_967_, v_l_u03b1_968_, v_a_969_, v_b_970_, v_ra_971_, v_rb_972_, v_a_973_, v_a_974_, v_a_975_, v_a_976_);
lean_dec(v_a_976_);
lean_dec_ref(v_a_975_);
lean_dec(v_a_974_);
lean_dec_ref(v_a_973_);
lean_dec_ref(v_l_u03b1_968_);
return v_res_978_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4(void){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_986_ = lean_box(0);
v___x_987_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__3));
v___x_988_ = l_Lean_Expr_const___override(v___x_987_, v___x_986_);
return v___x_988_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5(void){
_start:
{
lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_989_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__4);
v___x_990_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__14);
v___x_991_ = l_Lean_Expr_app___override(v___x_990_, v___x_989_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg(lean_object* v_u_1004_, lean_object* v_00_u03b1_1005_, lean_object* v_a_1006_, lean_object* v_b_1007_, lean_object* v_ra_1008_, lean_object* v_rb_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_){
_start:
{
lean_object* v___y_1016_; lean_object* v___y_1017_; lean_object* v___y_1018_; lean_object* v___y_1019_; 
switch(lean_obj_tag(v_ra_1008_))
{
case 0:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; 
lean_dec_ref_known(v_ra_1008_, 1);
lean_dec_ref(v_rb_1009_);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v___x_1022_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1023_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1022_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1023_;
}
case 1:
{
switch(lean_obj_tag(v_rb_1009_))
{
case 0:
{
lean_dec_ref_known(v_rb_1009_, 1);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v___y_1016_ = v_a_1010_;
v___y_1017_ = v_a_1011_;
v___y_1018_ = v_a_1012_;
v___y_1019_ = v_a_1013_;
goto v___jp_1015_;
}
case 1:
{
lean_object* v_lit_1024_; lean_object* v_proof_1025_; lean_object* v_inst_1026_; lean_object* v_lit_1027_; lean_object* v_proof_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v_lit_1024_ = lean_ctor_get(v_ra_1008_, 1);
v_proof_1025_ = lean_ctor_get(v_ra_1008_, 2);
v_inst_1026_ = lean_ctor_get(v_rb_1009_, 0);
v_lit_1027_ = lean_ctor_get(v_rb_1009_, 1);
v_proof_1028_ = lean_ctor_get(v_rb_1009_, 2);
v___x_1029_ = lean_box(0);
lean_inc_n(v_u_1004_, 2);
v___x_1030_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1030_, 0, v_u_1004_);
lean_ctor_set(v___x_1030_, 1, v___x_1029_);
lean_inc_ref(v_00_u03b1_1005_);
v___x_1031_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring(v_u_1004_, v_00_u03b1_1005_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
if (lean_obj_tag(v___x_1031_) == 0)
{
lean_object* v_a_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1100_; 
v_a_1032_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1100_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1100_ == 0)
{
v___x_1034_ = v___x_1031_;
v_isShared_1035_ = v_isSharedCheck_1100_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_a_1032_);
lean_dec(v___x_1031_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1100_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v_snd_1036_; lean_object* v_fst_1037_; lean_object* v_fst_1038_; lean_object* v_snd_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; uint8_t v___x_1042_; 
v_snd_1036_ = lean_ctor_get(v_a_1032_, 1);
lean_inc(v_snd_1036_);
v_fst_1037_ = lean_ctor_get(v_a_1032_, 0);
lean_inc(v_fst_1037_);
lean_dec(v_a_1032_);
v_fst_1038_ = lean_ctor_get(v_snd_1036_, 0);
lean_inc(v_fst_1038_);
v_snd_1039_ = lean_ctor_get(v_snd_1036_, 1);
lean_inc(v_snd_1039_);
lean_dec(v_snd_1036_);
v___x_1040_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1024_);
v___x_1041_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1027_);
v___x_1042_ = lean_nat_dec_le(v___x_1040_, v___x_1041_);
lean_dec(v___x_1041_);
lean_dec(v___x_1040_);
if (v___x_1042_ == 0)
{
lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; 
lean_del_object(v___x_1034_);
v___x_1043_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__1));
lean_inc_ref(v___x_1030_);
v___x_1044_ = l_Lean_Expr_const___override(v___x_1043_, v___x_1030_);
lean_inc_ref(v_00_u03b1_1005_);
v___x_1045_ = l_Lean_Expr_app___override(v___x_1044_, v_00_u03b1_1005_);
lean_inc_ref(v_inst_1026_);
v___x_1046_ = l_Lean_Expr_app___override(v___x_1045_, v_inst_1026_);
v___x_1047_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_1046_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
if (lean_obj_tag(v___x_1047_) == 0)
{
lean_object* v_a_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1073_; 
v_a_1048_ = lean_ctor_get(v___x_1047_, 0);
v_isSharedCheck_1073_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1050_ = v___x_1047_;
v_isShared_1051_ = v_isSharedCheck_1073_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_a_1048_);
lean_dec(v___x_1047_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1073_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
if (lean_obj_tag(v_a_1048_) == 1)
{
lean_object* v_a_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1070_; 
lean_inc_ref(v_proof_1028_);
lean_inc_ref(v_lit_1027_);
lean_inc_ref(v_proof_1025_);
lean_inc_ref(v_lit_1024_);
lean_dec_ref_known(v_rb_1009_, 3);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec(v_u_1004_);
v_a_1052_ = lean_ctor_get(v_a_1048_, 0);
lean_inc(v_a_1052_);
lean_dec_ref_known(v_a_1048_, 1);
v___x_1053_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5);
v___x_1054_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__7));
v___x_1055_ = l_Lean_Expr_const___override(v___x_1054_, v___x_1030_);
v___x_1056_ = l_Lean_Expr_app___override(v___x_1055_, v_00_u03b1_1005_);
v___x_1057_ = l_Lean_Expr_app___override(v___x_1056_, v_fst_1037_);
v___x_1058_ = l_Lean_Expr_app___override(v___x_1057_, v_fst_1038_);
v___x_1059_ = l_Lean_Expr_app___override(v___x_1058_, v_snd_1039_);
v___x_1060_ = l_Lean_Expr_app___override(v___x_1059_, v_a_1052_);
v___x_1061_ = l_Lean_Expr_app___override(v___x_1060_, v_a_1006_);
v___x_1062_ = l_Lean_Expr_app___override(v___x_1061_, v_b_1007_);
v___x_1063_ = l_Lean_Expr_app___override(v___x_1062_, v_lit_1024_);
v___x_1064_ = l_Lean_Expr_app___override(v___x_1063_, v_lit_1027_);
v___x_1065_ = l_Lean_Expr_app___override(v___x_1064_, v_proof_1025_);
v___x_1066_ = l_Lean_Expr_app___override(v___x_1065_, v_proof_1028_);
v___x_1067_ = l_Lean_Expr_app___override(v___x_1066_, v___x_1053_);
v___x_1068_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1068_, 0, v___x_1067_);
lean_ctor_set_uint8(v___x_1068_, sizeof(void*)*1, v___x_1042_);
if (v_isShared_1051_ == 0)
{
lean_ctor_set(v___x_1050_, 0, v___x_1068_);
v___x_1070_ = v___x_1050_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v___x_1068_);
v___x_1070_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
return v___x_1070_;
}
}
else
{
lean_object* v___x_1072_; 
lean_del_object(v___x_1050_);
lean_dec(v_a_1048_);
lean_dec(v_snd_1039_);
lean_dec(v_fst_1038_);
lean_dec(v_fst_1037_);
lean_dec_ref_known(v___x_1030_, 2);
v___x_1072_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1072_;
}
}
}
else
{
lean_object* v_a_1074_; lean_object* v___x_1076_; uint8_t v_isShared_1077_; uint8_t v_isSharedCheck_1081_; 
lean_dec(v_snd_1039_);
lean_dec(v_fst_1038_);
lean_dec(v_fst_1037_);
lean_dec_ref_known(v___x_1030_, 2);
lean_dec_ref_known(v_rb_1009_, 3);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v_a_1074_ = lean_ctor_get(v___x_1047_, 0);
v_isSharedCheck_1081_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1081_ == 0)
{
v___x_1076_ = v___x_1047_;
v_isShared_1077_ = v_isSharedCheck_1081_;
goto v_resetjp_1075_;
}
else
{
lean_inc(v_a_1074_);
lean_dec(v___x_1047_);
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
else
{
lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1098_; 
lean_inc_ref(v_proof_1028_);
lean_inc_ref(v_lit_1027_);
lean_inc_ref(v_proof_1025_);
lean_inc_ref(v_lit_1024_);
lean_dec_ref_known(v_rb_1009_, 3);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec(v_u_1004_);
v___x_1082_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1083_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__9));
v___x_1084_ = l_Lean_Expr_const___override(v___x_1083_, v___x_1030_);
v___x_1085_ = l_Lean_Expr_app___override(v___x_1084_, v_00_u03b1_1005_);
v___x_1086_ = l_Lean_Expr_app___override(v___x_1085_, v_fst_1037_);
v___x_1087_ = l_Lean_Expr_app___override(v___x_1086_, v_fst_1038_);
v___x_1088_ = l_Lean_Expr_app___override(v___x_1087_, v_snd_1039_);
v___x_1089_ = l_Lean_Expr_app___override(v___x_1088_, v_a_1006_);
v___x_1090_ = l_Lean_Expr_app___override(v___x_1089_, v_b_1007_);
v___x_1091_ = l_Lean_Expr_app___override(v___x_1090_, v_lit_1024_);
v___x_1092_ = l_Lean_Expr_app___override(v___x_1091_, v_lit_1027_);
v___x_1093_ = l_Lean_Expr_app___override(v___x_1092_, v_proof_1025_);
v___x_1094_ = l_Lean_Expr_app___override(v___x_1093_, v_proof_1028_);
v___x_1095_ = l_Lean_Expr_app___override(v___x_1094_, v___x_1082_);
v___x_1096_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1096_, 0, v___x_1095_);
lean_ctor_set_uint8(v___x_1096_, sizeof(void*)*1, v___x_1042_);
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1096_);
v___x_1098_ = v___x_1034_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1099_; 
v_reuseFailAlloc_1099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1099_, 0, v___x_1096_);
v___x_1098_ = v_reuseFailAlloc_1099_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
return v___x_1098_;
}
}
}
}
else
{
lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1108_; 
lean_dec_ref_known(v___x_1030_, 2);
lean_dec_ref_known(v_rb_1009_, 3);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v_a_1101_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1108_ == 0)
{
v___x_1103_ = v___x_1031_;
v_isShared_1104_ = v_isSharedCheck_1108_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1031_);
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
case 2:
{
lean_object* v___x_1109_; 
v___x_1109_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1109_;
}
default: 
{
lean_object* v___x_1110_; 
v___x_1110_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1110_;
}
}
}
case 2:
{
switch(lean_obj_tag(v_rb_1009_))
{
case 0:
{
lean_dec_ref_known(v_rb_1009_, 1);
lean_dec_ref_known(v_ra_1008_, 3);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v___y_1016_ = v_a_1010_;
v___y_1017_ = v_a_1011_;
v___y_1018_ = v_a_1012_;
v___y_1019_ = v_a_1013_;
goto v___jp_1015_;
}
case 3:
{
lean_object* v___x_1111_; 
v___x_1111_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1111_;
}
case 4:
{
lean_object* v___x_1112_; 
v___x_1112_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1112_;
}
case 2:
{
lean_object* v___x_1113_; 
v___x_1113_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1113_;
}
default: 
{
lean_object* v___x_1114_; 
v___x_1114_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1114_;
}
}
}
default: 
{
switch(lean_obj_tag(v_rb_1009_))
{
case 0:
{
lean_dec_ref_known(v_rb_1009_, 1);
lean_dec_ref(v_ra_1008_);
lean_dec_ref(v_b_1007_);
lean_dec_ref(v_a_1006_);
lean_dec_ref(v_00_u03b1_1005_);
lean_dec(v_u_1004_);
v___y_1016_ = v_a_1010_;
v___y_1017_ = v_a_1011_;
v___y_1018_ = v_a_1012_;
v___y_1019_ = v_a_1013_;
goto v___jp_1015_;
}
case 3:
{
lean_object* v___x_1115_; 
v___x_1115_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1115_;
}
case 4:
{
lean_object* v___x_1116_; 
v___x_1116_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1116_;
}
case 2:
{
lean_object* v___x_1117_; 
v___x_1117_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1117_;
}
default: 
{
lean_object* v___x_1118_; 
v___x_1118_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg(v_u_1004_, v_00_u03b1_1005_, v_a_1006_, v_b_1007_, v_ra_1008_, v_rb_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_);
return v___x_1118_;
}
}
}
}
v___jp_1015_:
{
lean_object* v___x_1020_; lean_object* v___x_1021_; 
v___x_1020_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1021_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1020_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
return v___x_1021_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___boxed(lean_object* v_u_1119_, lean_object* v_00_u03b1_1120_, lean_object* v_a_1121_, lean_object* v_b_1122_, lean_object* v_ra_1123_, lean_object* v_rb_1124_, lean_object* v_a_1125_, lean_object* v_a_1126_, lean_object* v_a_1127_, lean_object* v_a_1128_, lean_object* v_a_1129_){
_start:
{
lean_object* v_res_1130_; 
v_res_1130_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg(v_u_1119_, v_00_u03b1_1120_, v_a_1121_, v_b_1122_, v_ra_1123_, v_rb_1124_, v_a_1125_, v_a_1126_, v_a_1127_, v_a_1128_);
lean_dec(v_a_1128_);
lean_dec_ref(v_a_1127_);
lean_dec(v_a_1126_);
lean_dec_ref(v_a_1125_);
return v_res_1130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core(lean_object* v_u_1131_, lean_object* v_00_u03b1_1132_, lean_object* v_l_u03b1_1133_, lean_object* v_a_1134_, lean_object* v_b_1135_, lean_object* v_ra_1136_, lean_object* v_rb_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_, lean_object* v_a_1141_){
_start:
{
lean_object* v___x_1143_; 
v___x_1143_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg(v_u_1131_, v_00_u03b1_1132_, v_a_1134_, v_b_1135_, v_ra_1136_, v_rb_1137_, v_a_1138_, v_a_1139_, v_a_1140_, v_a_1141_);
return v___x_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___boxed(lean_object* v_u_1144_, lean_object* v_00_u03b1_1145_, lean_object* v_l_u03b1_1146_, lean_object* v_a_1147_, lean_object* v_b_1148_, lean_object* v_ra_1149_, lean_object* v_rb_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE_core(v_u_1144_, v_00_u03b1_1145_, v_l_u03b1_1146_, v_a_1147_, v_b_1148_, v_ra_1149_, v_rb_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_);
lean_dec(v_a_1154_);
lean_dec_ref(v_a_1153_);
lean_dec(v_a_1152_);
lean_dec_ref(v_a_1151_);
lean_dec_ref(v_l_u03b1_1146_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(lean_object* v_k_1157_, uint8_t v_allowLevelAssignments_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1158_, v_k_1157_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_);
if (lean_obj_tag(v___x_1164_) == 0)
{
lean_object* v_a_1165_; lean_object* v___x_1167_; uint8_t v_isShared_1168_; uint8_t v_isSharedCheck_1172_; 
v_a_1165_ = lean_ctor_get(v___x_1164_, 0);
v_isSharedCheck_1172_ = !lean_is_exclusive(v___x_1164_);
if (v_isSharedCheck_1172_ == 0)
{
v___x_1167_ = v___x_1164_;
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
else
{
lean_inc(v_a_1165_);
lean_dec(v___x_1164_);
v___x_1167_ = lean_box(0);
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
v_resetjp_1166_:
{
lean_object* v___x_1170_; 
if (v_isShared_1168_ == 0)
{
v___x_1170_ = v___x_1167_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_a_1165_);
v___x_1170_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
return v___x_1170_;
}
}
}
else
{
lean_object* v_a_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1180_; 
v_a_1173_ = lean_ctor_get(v___x_1164_, 0);
v_isSharedCheck_1180_ = !lean_is_exclusive(v___x_1164_);
if (v_isSharedCheck_1180_ == 0)
{
v___x_1175_ = v___x_1164_;
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_a_1173_);
lean_dec(v___x_1164_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v___x_1178_; 
if (v_isShared_1176_ == 0)
{
v___x_1178_ = v___x_1175_;
goto v_reusejp_1177_;
}
else
{
lean_object* v_reuseFailAlloc_1179_; 
v_reuseFailAlloc_1179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1179_, 0, v_a_1173_);
v___x_1178_ = v_reuseFailAlloc_1179_;
goto v_reusejp_1177_;
}
v_reusejp_1177_:
{
return v___x_1178_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg___boxed(lean_object* v_k_1181_, lean_object* v_allowLevelAssignments_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1188_; lean_object* v_res_1189_; 
v_allowLevelAssignments_boxed_1188_ = lean_unbox(v_allowLevelAssignments_1182_);
v_res_1189_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(v_k_1181_, v_allowLevelAssignments_boxed_1188_, v___y_1183_, v___y_1184_, v___y_1185_, v___y_1186_);
lean_dec(v___y_1186_);
lean_dec_ref(v___y_1185_);
lean_dec(v___y_1184_);
lean_dec_ref(v___y_1183_);
return v_res_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0(lean_object* v_00_u03b1_1190_, lean_object* v_k_1191_, uint8_t v_allowLevelAssignments_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_){
_start:
{
lean_object* v___x_1198_; 
v___x_1198_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(v_k_1191_, v_allowLevelAssignments_1192_, v___y_1193_, v___y_1194_, v___y_1195_, v___y_1196_);
return v___x_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___boxed(lean_object* v_00_u03b1_1199_, lean_object* v_k_1200_, lean_object* v_allowLevelAssignments_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1207_; lean_object* v_res_1208_; 
v_allowLevelAssignments_boxed_1207_ = lean_unbox(v_allowLevelAssignments_1201_);
v_res_1208_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0(v_00_u03b1_1199_, v_k_1200_, v_allowLevelAssignments_boxed_1207_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
return v_res_1208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0(lean_object* v_fn_1209_, lean_object* v___x_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v___x_1216_; 
v___x_1216_ = l_Lean_Meta_isExprDefEq(v_fn_1209_, v___x_1210_, v___y_1211_, v___y_1212_, v___y_1213_, v___y_1214_);
return v___x_1216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0___boxed(lean_object* v_fn_1217_, lean_object* v___x_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_){
_start:
{
lean_object* v_res_1224_; 
v_res_1224_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0(v_fn_1217_, v___x_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_);
lean_dec(v___y_1222_);
lean_dec_ref(v___y_1221_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
return v_res_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1(lean_object* v_v_1232_, lean_object* v_00_u03b2_1233_, lean_object* v_e_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___x_1240_; 
v___x_1240_ = l_Lean_Meta_whnfR(v_e_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1240_) == 0)
{
lean_object* v_a_1241_; lean_object* v___y_1243_; lean_object* v___y_1244_; lean_object* v___y_1245_; lean_object* v___y_1246_; 
v_a_1241_ = lean_ctor_get(v___x_1240_, 0);
lean_inc(v_a_1241_);
lean_dec_ref_known(v___x_1240_, 1);
if (lean_obj_tag(v_a_1241_) == 5)
{
lean_object* v_fn_1249_; 
v_fn_1249_ = lean_ctor_get(v_a_1241_, 0);
lean_inc_ref(v_fn_1249_);
if (lean_obj_tag(v_fn_1249_) == 5)
{
lean_object* v_arg_1250_; lean_object* v_fn_1251_; lean_object* v_arg_1252_; lean_object* v___x_1253_; 
v_arg_1250_ = lean_ctor_get(v_a_1241_, 1);
lean_inc_ref(v_arg_1250_);
lean_dec_ref_known(v_a_1241_, 2);
v_fn_1251_ = lean_ctor_get(v_fn_1249_, 0);
lean_inc_ref(v_fn_1251_);
v_arg_1252_ = lean_ctor_get(v_fn_1249_, 1);
lean_inc_ref(v_arg_1252_);
lean_dec_ref_known(v_fn_1249_, 2);
v___x_1253_ = lp_mathlib_Qq_inferTypeQ_x27(v_arg_1252_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1253_) == 0)
{
lean_object* v_a_1254_; lean_object* v_snd_1255_; lean_object* v_fst_1256_; lean_object* v_fst_1257_; lean_object* v_snd_1258_; lean_object* v___x_1260_; uint8_t v_isShared_1261_; uint8_t v_isSharedCheck_1311_; 
v_a_1254_ = lean_ctor_get(v___x_1253_, 0);
lean_inc(v_a_1254_);
lean_dec_ref_known(v___x_1253_, 1);
v_snd_1255_ = lean_ctor_get(v_a_1254_, 1);
lean_inc(v_snd_1255_);
v_fst_1256_ = lean_ctor_get(v_a_1254_, 0);
lean_inc(v_fst_1256_);
lean_dec(v_a_1254_);
v_fst_1257_ = lean_ctor_get(v_snd_1255_, 0);
v_snd_1258_ = lean_ctor_get(v_snd_1255_, 1);
v_isSharedCheck_1311_ = !lean_is_exclusive(v_snd_1255_);
if (v_isSharedCheck_1311_ == 0)
{
v___x_1260_ = v_snd_1255_;
v_isShared_1261_ = v_isSharedCheck_1311_;
goto v_resetjp_1259_;
}
else
{
lean_inc(v_snd_1258_);
lean_inc(v_fst_1257_);
lean_dec(v_snd_1255_);
v___x_1260_ = lean_box(0);
v_isShared_1261_ = v_isSharedCheck_1311_;
goto v_resetjp_1259_;
}
v_resetjp_1259_:
{
uint8_t v___x_1262_; lean_object* v___x_1263_; 
v___x_1262_ = 0;
lean_inc(v_snd_1258_);
lean_inc(v_fst_1257_);
lean_inc(v_fst_1256_);
v___x_1263_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_1256_, v_fst_1257_, v_snd_1258_, v___x_1262_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1263_) == 0)
{
lean_object* v_a_1264_; lean_object* v___x_1265_; 
v_a_1264_ = lean_ctor_get(v___x_1263_, 0);
lean_inc(v_a_1264_);
lean_dec_ref_known(v___x_1263_, 1);
lean_inc_ref(v_arg_1250_);
lean_inc(v_fst_1257_);
lean_inc(v_fst_1256_);
v___x_1265_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_1256_, v_fst_1257_, v_arg_1250_, v___x_1262_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1265_) == 0)
{
lean_object* v_a_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1270_; 
v_a_1266_ = lean_ctor_get(v___x_1265_, 0);
lean_inc(v_a_1266_);
lean_dec_ref_known(v___x_1265_, 1);
v___x_1267_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__1));
v___x_1268_ = lean_box(0);
lean_inc(v_fst_1256_);
if (v_isShared_1261_ == 0)
{
lean_ctor_set_tag(v___x_1260_, 1);
lean_ctor_set(v___x_1260_, 1, v___x_1268_);
lean_ctor_set(v___x_1260_, 0, v_fst_1256_);
v___x_1270_ = v___x_1260_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v_fst_1256_);
lean_ctor_set(v_reuseFailAlloc_1310_, 1, v___x_1268_);
v___x_1270_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; 
lean_inc_ref(v___x_1270_);
v___x_1271_ = l_Lean_Expr_const___override(v___x_1267_, v___x_1270_);
lean_inc(v_fst_1257_);
v___x_1272_ = l_Lean_Expr_app___override(v___x_1271_, v_fst_1257_);
v___x_1273_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1272_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1273_) == 0)
{
lean_object* v_a_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___f_1279_; lean_object* v___x_1280_; 
v_a_1274_ = lean_ctor_get(v___x_1273_, 0);
lean_inc(v_a_1274_);
lean_dec_ref_known(v___x_1273_, 1);
v___x_1275_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___closed__3));
v___x_1276_ = l_Lean_Expr_const___override(v___x_1275_, v___x_1270_);
lean_inc(v_fst_1257_);
v___x_1277_ = l_Lean_Expr_app___override(v___x_1276_, v_fst_1257_);
v___x_1278_ = l_Lean_Expr_app___override(v___x_1277_, v_a_1274_);
v___f_1279_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1279_, 0, v_fn_1251_);
lean_closure_set(v___f_1279_, 1, v___x_1278_);
v___x_1280_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(v___f_1279_, v___x_1262_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1280_) == 0)
{
lean_object* v_a_1281_; uint8_t v___x_1282_; 
v_a_1281_ = lean_ctor_get(v___x_1280_, 0);
lean_inc(v_a_1281_);
lean_dec_ref_known(v___x_1280_, 1);
v___x_1282_ = lean_unbox(v_a_1281_);
lean_dec(v_a_1281_);
if (v___x_1282_ == 0)
{
lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v_a_1285_; lean_object* v___x_1287_; uint8_t v_isShared_1288_; uint8_t v_isSharedCheck_1292_; 
lean_dec(v_a_1266_);
lean_dec(v_a_1264_);
lean_dec(v_snd_1258_);
lean_dec(v_fst_1257_);
lean_dec(v_fst_1256_);
lean_dec_ref(v_arg_1250_);
v___x_1283_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1284_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1283_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
v_a_1285_ = lean_ctor_get(v___x_1284_, 0);
v_isSharedCheck_1292_ = !lean_is_exclusive(v___x_1284_);
if (v_isSharedCheck_1292_ == 0)
{
v___x_1287_ = v___x_1284_;
v_isShared_1288_ = v_isSharedCheck_1292_;
goto v_resetjp_1286_;
}
else
{
lean_inc(v_a_1285_);
lean_dec(v___x_1284_);
v___x_1287_ = lean_box(0);
v_isShared_1288_ = v_isSharedCheck_1292_;
goto v_resetjp_1286_;
}
v_resetjp_1286_:
{
lean_object* v___x_1290_; 
if (v_isShared_1288_ == 0)
{
v___x_1290_ = v___x_1287_;
goto v_reusejp_1289_;
}
else
{
lean_object* v_reuseFailAlloc_1291_; 
v_reuseFailAlloc_1291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1291_, 0, v_a_1285_);
v___x_1290_ = v_reuseFailAlloc_1291_;
goto v_reusejp_1289_;
}
v_reusejp_1289_:
{
return v___x_1290_;
}
}
}
else
{
lean_object* v___x_1293_; 
v___x_1293_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg(v_fst_1256_, v_fst_1257_, v_snd_1258_, v_arg_1250_, v_a_1264_, v_a_1266_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
return v___x_1293_;
}
}
else
{
lean_object* v_a_1294_; lean_object* v___x_1296_; uint8_t v_isShared_1297_; uint8_t v_isSharedCheck_1301_; 
lean_dec(v_a_1266_);
lean_dec(v_a_1264_);
lean_dec(v_snd_1258_);
lean_dec(v_fst_1257_);
lean_dec(v_fst_1256_);
lean_dec_ref(v_arg_1250_);
v_a_1294_ = lean_ctor_get(v___x_1280_, 0);
v_isSharedCheck_1301_ = !lean_is_exclusive(v___x_1280_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1296_ = v___x_1280_;
v_isShared_1297_ = v_isSharedCheck_1301_;
goto v_resetjp_1295_;
}
else
{
lean_inc(v_a_1294_);
lean_dec(v___x_1280_);
v___x_1296_ = lean_box(0);
v_isShared_1297_ = v_isSharedCheck_1301_;
goto v_resetjp_1295_;
}
v_resetjp_1295_:
{
lean_object* v___x_1299_; 
if (v_isShared_1297_ == 0)
{
v___x_1299_ = v___x_1296_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_a_1294_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
}
}
else
{
lean_object* v_a_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1309_; 
lean_dec_ref(v___x_1270_);
lean_dec(v_a_1266_);
lean_dec(v_a_1264_);
lean_dec(v_snd_1258_);
lean_dec(v_fst_1257_);
lean_dec(v_fst_1256_);
lean_dec_ref(v_fn_1251_);
lean_dec_ref(v_arg_1250_);
v_a_1302_ = lean_ctor_get(v___x_1273_, 0);
v_isSharedCheck_1309_ = !lean_is_exclusive(v___x_1273_);
if (v_isSharedCheck_1309_ == 0)
{
v___x_1304_ = v___x_1273_;
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_a_1302_);
lean_dec(v___x_1273_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1307_; 
if (v_isShared_1305_ == 0)
{
v___x_1307_ = v___x_1304_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1308_; 
v_reuseFailAlloc_1308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1308_, 0, v_a_1302_);
v___x_1307_ = v_reuseFailAlloc_1308_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
return v___x_1307_;
}
}
}
}
}
else
{
lean_dec(v_a_1264_);
lean_del_object(v___x_1260_);
lean_dec(v_snd_1258_);
lean_dec(v_fst_1257_);
lean_dec(v_fst_1256_);
lean_dec_ref(v_fn_1251_);
lean_dec_ref(v_arg_1250_);
return v___x_1265_;
}
}
else
{
lean_del_object(v___x_1260_);
lean_dec(v_snd_1258_);
lean_dec(v_fst_1257_);
lean_dec(v_fst_1256_);
lean_dec_ref(v_fn_1251_);
lean_dec_ref(v_arg_1250_);
return v___x_1263_;
}
}
}
else
{
lean_object* v_a_1312_; lean_object* v___x_1314_; uint8_t v_isShared_1315_; uint8_t v_isSharedCheck_1319_; 
lean_dec_ref(v_fn_1251_);
lean_dec_ref(v_arg_1250_);
v_a_1312_ = lean_ctor_get(v___x_1253_, 0);
v_isSharedCheck_1319_ = !lean_is_exclusive(v___x_1253_);
if (v_isSharedCheck_1319_ == 0)
{
v___x_1314_ = v___x_1253_;
v_isShared_1315_ = v_isSharedCheck_1319_;
goto v_resetjp_1313_;
}
else
{
lean_inc(v_a_1312_);
lean_dec(v___x_1253_);
v___x_1314_ = lean_box(0);
v_isShared_1315_ = v_isSharedCheck_1319_;
goto v_resetjp_1313_;
}
v_resetjp_1313_:
{
lean_object* v___x_1317_; 
if (v_isShared_1315_ == 0)
{
v___x_1317_ = v___x_1314_;
goto v_reusejp_1316_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v_a_1312_);
v___x_1317_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1316_;
}
v_reusejp_1316_:
{
return v___x_1317_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_1241_, 2);
lean_dec_ref(v_fn_1249_);
v___y_1243_ = v___y_1235_;
v___y_1244_ = v___y_1236_;
v___y_1245_ = v___y_1237_;
v___y_1246_ = v___y_1238_;
goto v___jp_1242_;
}
}
else
{
lean_dec(v_a_1241_);
v___y_1243_ = v___y_1235_;
v___y_1244_ = v___y_1236_;
v___y_1245_ = v___y_1237_;
v___y_1246_ = v___y_1238_;
goto v___jp_1242_;
}
v___jp_1242_:
{
lean_object* v___x_1247_; lean_object* v___x_1248_; 
v___x_1247_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1248_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1247_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_);
return v___x_1248_;
}
}
else
{
lean_object* v_a_1320_; lean_object* v___x_1322_; uint8_t v_isShared_1323_; uint8_t v_isSharedCheck_1327_; 
v_a_1320_ = lean_ctor_get(v___x_1240_, 0);
v_isSharedCheck_1327_ = !lean_is_exclusive(v___x_1240_);
if (v_isSharedCheck_1327_ == 0)
{
v___x_1322_ = v___x_1240_;
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
else
{
lean_inc(v_a_1320_);
lean_dec(v___x_1240_);
v___x_1322_ = lean_box(0);
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
v_resetjp_1321_:
{
lean_object* v___x_1325_; 
if (v_isShared_1323_ == 0)
{
v___x_1325_ = v___x_1322_;
goto v_reusejp_1324_;
}
else
{
lean_object* v_reuseFailAlloc_1326_; 
v_reuseFailAlloc_1326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1326_, 0, v_a_1320_);
v___x_1325_ = v_reuseFailAlloc_1326_;
goto v_reusejp_1324_;
}
v_reusejp_1324_:
{
return v___x_1325_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1___boxed(lean_object* v_v_1328_, lean_object* v_00_u03b2_1329_, lean_object* v_e_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_){
_start:
{
lean_object* v_res_1336_; 
v_res_1336_ = lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__1(v_v_1328_, v_00_u03b2_1329_, v_e_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
lean_dec_ref(v_00_u03b2_1329_);
lean_dec(v_v_1328_);
return v_res_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(lean_object* v_u_1361_, lean_object* v_00_u03b1_1362_, lean_object* v_a_1363_, lean_object* v_b_1364_, lean_object* v_ra_1365_, lean_object* v_rb_1366_, lean_object* v_a_1367_, lean_object* v_a_1368_, lean_object* v_a_1369_, lean_object* v_a_1370_){
_start:
{
lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1372_ = lean_box(0);
lean_inc_n(v_u_1361_, 2);
v___x_1373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1373_, 0, v_u_1361_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
lean_inc_ref(v_00_u03b1_1362_);
v___x_1374_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedRing(v_u_1361_, v_00_u03b1_1362_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1480_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1480_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1480_ == 0)
{
v___x_1377_ = v___x_1374_;
v_isShared_1378_ = v_isSharedCheck_1480_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_a_1375_);
lean_dec(v___x_1374_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1480_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
lean_object* v_snd_1379_; lean_object* v_fst_1380_; lean_object* v_fst_1381_; lean_object* v_snd_1382_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v_a_1387_; lean_object* v_a_1451_; lean_object* v___x_1468_; 
v_snd_1379_ = lean_ctor_get(v_a_1375_, 1);
lean_inc(v_snd_1379_);
v_fst_1380_ = lean_ctor_get(v_a_1375_, 0);
lean_inc_n(v_fst_1380_, 2);
lean_dec(v_a_1375_);
v_fst_1381_ = lean_ctor_get(v_snd_1379_, 0);
lean_inc(v_fst_1381_);
v_snd_1382_ = lean_ctor_get(v_snd_1379_, 1);
lean_inc(v_snd_1382_);
lean_dec(v_snd_1379_);
lean_inc_ref(v_a_1363_);
lean_inc_ref(v_00_u03b1_1362_);
lean_inc(v_u_1361_);
v___x_1468_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_1361_, v_00_u03b1_1362_, v_a_1363_, v_fst_1380_, v_ra_1365_);
if (lean_obj_tag(v___x_1468_) == 0)
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v_a_1471_; lean_object* v___x_1473_; uint8_t v_isShared_1474_; uint8_t v_isSharedCheck_1478_; 
lean_dec(v_snd_1382_);
lean_dec(v_fst_1381_);
lean_dec(v_fst_1380_);
lean_del_object(v___x_1377_);
lean_dec_ref_known(v___x_1373_, 2);
lean_dec_ref(v_rb_1366_);
lean_dec_ref(v_b_1364_);
lean_dec_ref(v_a_1363_);
lean_dec_ref(v_00_u03b1_1362_);
lean_dec(v_u_1361_);
v___x_1469_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1470_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1469_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_);
v_a_1471_ = lean_ctor_get(v___x_1470_, 0);
v_isSharedCheck_1478_ = !lean_is_exclusive(v___x_1470_);
if (v_isSharedCheck_1478_ == 0)
{
v___x_1473_ = v___x_1470_;
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
else
{
lean_inc(v_a_1471_);
lean_dec(v___x_1470_);
v___x_1473_ = lean_box(0);
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
v_resetjp_1472_:
{
lean_object* v___x_1476_; 
if (v_isShared_1474_ == 0)
{
v___x_1476_ = v___x_1473_;
goto v_reusejp_1475_;
}
else
{
lean_object* v_reuseFailAlloc_1477_; 
v_reuseFailAlloc_1477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1477_, 0, v_a_1471_);
v___x_1476_ = v_reuseFailAlloc_1477_;
goto v_reusejp_1475_;
}
v_reusejp_1475_:
{
return v___x_1476_;
}
}
}
else
{
lean_object* v_val_1479_; 
v_val_1479_ = lean_ctor_get(v___x_1468_, 0);
lean_inc(v_val_1479_);
lean_dec_ref_known(v___x_1468_, 1);
v_a_1451_ = v_val_1479_;
goto v___jp_1450_;
}
v___jp_1383_:
{
lean_object* v_snd_1388_; lean_object* v_fst_1389_; lean_object* v_fst_1390_; lean_object* v_snd_1391_; uint8_t v___x_1392_; 
v_snd_1388_ = lean_ctor_get(v_a_1387_, 1);
lean_inc(v_snd_1388_);
v_fst_1389_ = lean_ctor_get(v_a_1387_, 0);
lean_inc(v_fst_1389_);
lean_dec_ref(v_a_1387_);
v_fst_1390_ = lean_ctor_get(v_snd_1388_, 0);
lean_inc(v_fst_1390_);
v_snd_1391_ = lean_ctor_get(v_snd_1388_, 1);
lean_inc(v_snd_1391_);
lean_dec(v_snd_1388_);
v___x_1392_ = lean_int_dec_lt(v___y_1385_, v_fst_1389_);
lean_dec(v_fst_1389_);
lean_dec(v___y_1385_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1409_; 
v___x_1393_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1394_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__1));
v___x_1395_ = l_Lean_Expr_const___override(v___x_1394_, v___x_1373_);
v___x_1396_ = l_Lean_Expr_app___override(v___x_1395_, v_00_u03b1_1362_);
v___x_1397_ = l_Lean_Expr_app___override(v___x_1396_, v_fst_1380_);
v___x_1398_ = l_Lean_Expr_app___override(v___x_1397_, v_fst_1381_);
v___x_1399_ = l_Lean_Expr_app___override(v___x_1398_, v_snd_1382_);
v___x_1400_ = l_Lean_Expr_app___override(v___x_1399_, v_a_1363_);
v___x_1401_ = l_Lean_Expr_app___override(v___x_1400_, v_b_1364_);
v___x_1402_ = l_Lean_Expr_app___override(v___x_1401_, v___y_1386_);
v___x_1403_ = l_Lean_Expr_app___override(v___x_1402_, v_fst_1390_);
v___x_1404_ = l_Lean_Expr_app___override(v___x_1403_, v___y_1384_);
v___x_1405_ = l_Lean_Expr_app___override(v___x_1404_, v_snd_1391_);
v___x_1406_ = l_Lean_Expr_app___override(v___x_1405_, v___x_1393_);
v___x_1407_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1407_, 0, v___x_1406_);
lean_ctor_set_uint8(v___x_1407_, sizeof(void*)*1, v___x_1392_);
if (v_isShared_1378_ == 0)
{
lean_ctor_set(v___x_1377_, 0, v___x_1407_);
v___x_1409_ = v___x_1377_;
goto v_reusejp_1408_;
}
else
{
lean_object* v_reuseFailAlloc_1410_; 
v_reuseFailAlloc_1410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1410_, 0, v___x_1407_);
v___x_1409_ = v_reuseFailAlloc_1410_;
goto v_reusejp_1408_;
}
v_reusejp_1408_:
{
return v___x_1409_;
}
}
else
{
lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; 
lean_del_object(v___x_1377_);
v___x_1411_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__2));
lean_inc_ref(v___x_1373_);
v___x_1412_ = l_Lean_Expr_const___override(v___x_1411_, v___x_1373_);
lean_inc_ref(v_00_u03b1_1362_);
v___x_1413_ = l_Lean_Expr_app___override(v___x_1412_, v_00_u03b1_1362_);
v___x_1414_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_1413_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_);
if (lean_obj_tag(v___x_1414_) == 0)
{
lean_object* v_a_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1441_; 
v_a_1415_ = lean_ctor_get(v___x_1414_, 0);
v_isSharedCheck_1441_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1441_ == 0)
{
v___x_1417_ = v___x_1414_;
v_isShared_1418_ = v_isSharedCheck_1441_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_a_1415_);
lean_dec(v___x_1414_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1441_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
if (lean_obj_tag(v_a_1415_) == 1)
{
lean_object* v_a_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1437_; 
v_a_1419_ = lean_ctor_get(v_a_1415_, 0);
lean_inc(v_a_1419_);
lean_dec_ref_known(v_a_1415_, 1);
v___x_1420_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1421_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___closed__3));
v___x_1422_ = l_Lean_Expr_const___override(v___x_1421_, v___x_1373_);
v___x_1423_ = l_Lean_Expr_app___override(v___x_1422_, v_00_u03b1_1362_);
v___x_1424_ = l_Lean_Expr_app___override(v___x_1423_, v_fst_1380_);
v___x_1425_ = l_Lean_Expr_app___override(v___x_1424_, v_fst_1381_);
v___x_1426_ = l_Lean_Expr_app___override(v___x_1425_, v_snd_1382_);
v___x_1427_ = l_Lean_Expr_app___override(v___x_1426_, v_a_1419_);
v___x_1428_ = l_Lean_Expr_app___override(v___x_1427_, v_a_1363_);
v___x_1429_ = l_Lean_Expr_app___override(v___x_1428_, v_b_1364_);
v___x_1430_ = l_Lean_Expr_app___override(v___x_1429_, v___y_1386_);
v___x_1431_ = l_Lean_Expr_app___override(v___x_1430_, v_fst_1390_);
v___x_1432_ = l_Lean_Expr_app___override(v___x_1431_, v___y_1384_);
v___x_1433_ = l_Lean_Expr_app___override(v___x_1432_, v_snd_1391_);
v___x_1434_ = l_Lean_Expr_app___override(v___x_1433_, v___x_1420_);
v___x_1435_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1435_, 0, v___x_1434_);
lean_ctor_set_uint8(v___x_1435_, sizeof(void*)*1, v___x_1392_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set(v___x_1417_, 0, v___x_1435_);
v___x_1437_ = v___x_1417_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v___x_1435_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
else
{
lean_object* v___x_1439_; lean_object* v___x_1440_; 
lean_del_object(v___x_1417_);
lean_dec(v_a_1415_);
lean_dec(v_snd_1391_);
lean_dec(v_fst_1390_);
lean_dec_ref(v___y_1386_);
lean_dec_ref(v___y_1384_);
lean_dec(v_snd_1382_);
lean_dec(v_fst_1381_);
lean_dec(v_fst_1380_);
lean_dec_ref_known(v___x_1373_, 2);
lean_dec_ref(v_b_1364_);
lean_dec_ref(v_a_1363_);
lean_dec_ref(v_00_u03b1_1362_);
v___x_1439_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1440_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1439_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_);
return v___x_1440_;
}
}
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec(v_snd_1391_);
lean_dec(v_fst_1390_);
lean_dec_ref(v___y_1386_);
lean_dec_ref(v___y_1384_);
lean_dec(v_snd_1382_);
lean_dec(v_fst_1381_);
lean_dec(v_fst_1380_);
lean_dec_ref_known(v___x_1373_, 2);
lean_dec_ref(v_b_1364_);
lean_dec_ref(v_a_1363_);
lean_dec_ref(v_00_u03b1_1362_);
v_a_1442_ = lean_ctor_get(v___x_1414_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1414_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1414_);
v___x_1444_ = lean_box(0);
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
v_resetjp_1443_:
{
lean_object* v___x_1447_; 
if (v_isShared_1445_ == 0)
{
v___x_1447_ = v___x_1444_;
goto v_reusejp_1446_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v_a_1442_);
v___x_1447_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1446_;
}
v_reusejp_1446_:
{
return v___x_1447_;
}
}
}
}
}
v___jp_1450_:
{
lean_object* v_snd_1452_; lean_object* v_fst_1453_; lean_object* v_fst_1454_; lean_object* v_snd_1455_; lean_object* v___x_1456_; 
v_snd_1452_ = lean_ctor_get(v_a_1451_, 1);
lean_inc(v_snd_1452_);
v_fst_1453_ = lean_ctor_get(v_a_1451_, 0);
lean_inc(v_fst_1453_);
lean_dec_ref(v_a_1451_);
v_fst_1454_ = lean_ctor_get(v_snd_1452_, 0);
lean_inc(v_fst_1454_);
v_snd_1455_ = lean_ctor_get(v_snd_1452_, 1);
lean_inc(v_snd_1455_);
lean_dec(v_snd_1452_);
lean_inc(v_fst_1380_);
lean_inc_ref(v_b_1364_);
lean_inc_ref(v_00_u03b1_1362_);
v___x_1456_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_1361_, v_00_u03b1_1362_, v_b_1364_, v_fst_1380_, v_rb_1366_);
if (lean_obj_tag(v___x_1456_) == 0)
{
lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v_a_1459_; lean_object* v___x_1461_; uint8_t v_isShared_1462_; uint8_t v_isSharedCheck_1466_; 
lean_dec(v_snd_1455_);
lean_dec(v_fst_1454_);
lean_dec(v_fst_1453_);
lean_dec(v_snd_1382_);
lean_dec(v_fst_1381_);
lean_dec(v_fst_1380_);
lean_del_object(v___x_1377_);
lean_dec_ref_known(v___x_1373_, 2);
lean_dec_ref(v_b_1364_);
lean_dec_ref(v_a_1363_);
lean_dec_ref(v_00_u03b1_1362_);
v___x_1457_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1458_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1457_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_);
v_a_1459_ = lean_ctor_get(v___x_1458_, 0);
v_isSharedCheck_1466_ = !lean_is_exclusive(v___x_1458_);
if (v_isSharedCheck_1466_ == 0)
{
v___x_1461_ = v___x_1458_;
v_isShared_1462_ = v_isSharedCheck_1466_;
goto v_resetjp_1460_;
}
else
{
lean_inc(v_a_1459_);
lean_dec(v___x_1458_);
v___x_1461_ = lean_box(0);
v_isShared_1462_ = v_isSharedCheck_1466_;
goto v_resetjp_1460_;
}
v_resetjp_1460_:
{
lean_object* v___x_1464_; 
if (v_isShared_1462_ == 0)
{
v___x_1464_ = v___x_1461_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v_a_1459_);
v___x_1464_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
return v___x_1464_;
}
}
}
else
{
lean_object* v_val_1467_; 
v_val_1467_ = lean_ctor_get(v___x_1456_, 0);
lean_inc(v_val_1467_);
lean_dec_ref_known(v___x_1456_, 1);
v___y_1384_ = v_snd_1455_;
v___y_1385_ = v_fst_1453_;
v___y_1386_ = v_fst_1454_;
v_a_1387_ = v_val_1467_;
goto v___jp_1383_;
}
}
}
}
else
{
lean_object* v_a_1481_; lean_object* v___x_1483_; uint8_t v_isShared_1484_; uint8_t v_isSharedCheck_1488_; 
lean_dec_ref_known(v___x_1373_, 2);
lean_dec_ref(v_rb_1366_);
lean_dec_ref(v_ra_1365_);
lean_dec_ref(v_b_1364_);
lean_dec_ref(v_a_1363_);
lean_dec_ref(v_00_u03b1_1362_);
lean_dec(v_u_1361_);
v_a_1481_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1488_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1488_ == 0)
{
v___x_1483_ = v___x_1374_;
v_isShared_1484_ = v_isSharedCheck_1488_;
goto v_resetjp_1482_;
}
else
{
lean_inc(v_a_1481_);
lean_dec(v___x_1374_);
v___x_1483_ = lean_box(0);
v_isShared_1484_ = v_isSharedCheck_1488_;
goto v_resetjp_1482_;
}
v_resetjp_1482_:
{
lean_object* v___x_1486_; 
if (v_isShared_1484_ == 0)
{
v___x_1486_ = v___x_1483_;
goto v_reusejp_1485_;
}
else
{
lean_object* v_reuseFailAlloc_1487_; 
v_reuseFailAlloc_1487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1487_, 0, v_a_1481_);
v___x_1486_ = v_reuseFailAlloc_1487_;
goto v_reusejp_1485_;
}
v_reusejp_1485_:
{
return v___x_1486_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg___boxed(lean_object* v_u_1489_, lean_object* v_00_u03b1_1490_, lean_object* v_a_1491_, lean_object* v_b_1492_, lean_object* v_ra_1493_, lean_object* v_rb_1494_, lean_object* v_a_1495_, lean_object* v_a_1496_, lean_object* v_a_1497_, lean_object* v_a_1498_, lean_object* v_a_1499_){
_start:
{
lean_object* v_res_1500_; 
v_res_1500_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1489_, v_00_u03b1_1490_, v_a_1491_, v_b_1492_, v_ra_1493_, v_rb_1494_, v_a_1495_, v_a_1496_, v_a_1497_, v_a_1498_);
lean_dec(v_a_1498_);
lean_dec_ref(v_a_1497_);
lean_dec(v_a_1496_);
lean_dec_ref(v_a_1495_);
return v_res_1500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm(lean_object* v_u_1501_, lean_object* v_00_u03b1_1502_, lean_object* v_l_u03b1_1503_, lean_object* v_a_1504_, lean_object* v_b_1505_, lean_object* v_ra_1506_, lean_object* v_rb_1507_, lean_object* v_a_1508_, lean_object* v_a_1509_, lean_object* v_a_1510_, lean_object* v_a_1511_){
_start:
{
lean_object* v___x_1513_; 
v___x_1513_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1501_, v_00_u03b1_1502_, v_a_1504_, v_b_1505_, v_ra_1506_, v_rb_1507_, v_a_1508_, v_a_1509_, v_a_1510_, v_a_1511_);
return v___x_1513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___boxed(lean_object* v_u_1514_, lean_object* v_00_u03b1_1515_, lean_object* v_l_u03b1_1516_, lean_object* v_a_1517_, lean_object* v_b_1518_, lean_object* v_ra_1519_, lean_object* v_rb_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_, lean_object* v_a_1524_, lean_object* v_a_1525_){
_start:
{
lean_object* v_res_1526_; 
v_res_1526_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm(v_u_1514_, v_00_u03b1_1515_, v_l_u03b1_1516_, v_a_1517_, v_b_1518_, v_ra_1519_, v_rb_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
lean_dec(v_a_1524_);
lean_dec_ref(v_a_1523_);
lean_dec(v_a_1522_);
lean_dec_ref(v_a_1521_);
lean_dec_ref(v_l_u03b1_1516_);
return v_res_1526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(lean_object* v_u_1547_, lean_object* v_00_u03b1_1548_, lean_object* v_a_1549_, lean_object* v_b_1550_, lean_object* v_ra_1551_, lean_object* v_rb_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_, lean_object* v_a_1555_, lean_object* v_a_1556_){
_start:
{
lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; 
v___x_1558_ = lean_box(0);
lean_inc_n(v_u_1547_, 2);
v___x_1559_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1559_, 0, v_u_1547_);
lean_ctor_set(v___x_1559_, 1, v___x_1558_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1560_ = lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedSemifield(v_u_1547_, v_00_u03b1_1548_, v_a_1553_, v_a_1554_, v_a_1555_, v_a_1556_);
if (lean_obj_tag(v___x_1560_) == 0)
{
lean_object* v_a_1561_; lean_object* v___x_1563_; uint8_t v_isShared_1564_; uint8_t v_isSharedCheck_1666_; 
v_a_1561_ = lean_ctor_get(v___x_1560_, 0);
v_isSharedCheck_1666_ = !lean_is_exclusive(v___x_1560_);
if (v_isSharedCheck_1666_ == 0)
{
v___x_1563_ = v___x_1560_;
v_isShared_1564_ = v_isSharedCheck_1666_;
goto v_resetjp_1562_;
}
else
{
lean_inc(v_a_1561_);
lean_dec(v___x_1560_);
v___x_1563_ = lean_box(0);
v_isShared_1564_ = v_isSharedCheck_1666_;
goto v_resetjp_1562_;
}
v_resetjp_1562_:
{
lean_object* v_snd_1565_; lean_object* v_fst_1566_; lean_object* v_fst_1567_; lean_object* v_snd_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___y_1574_; lean_object* v___y_1575_; lean_object* v___y_1576_; lean_object* v___y_1577_; lean_object* v_a_1578_; lean_object* v_a_1635_; lean_object* v___x_1654_; 
v_snd_1565_ = lean_ctor_get(v_a_1561_, 1);
lean_inc(v_snd_1565_);
v_fst_1566_ = lean_ctor_get(v_a_1561_, 0);
lean_inc(v_fst_1566_);
lean_dec(v_a_1561_);
v_fst_1567_ = lean_ctor_get(v_snd_1565_, 0);
lean_inc(v_fst_1567_);
v_snd_1568_ = lean_ctor_get(v_snd_1565_, 1);
lean_inc(v_snd_1568_);
lean_dec(v_snd_1565_);
v___x_1569_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__2));
lean_inc_ref(v___x_1559_);
v___x_1570_ = l_Lean_Expr_const___override(v___x_1569_, v___x_1559_);
lean_inc_ref_n(v_00_u03b1_1548_, 2);
v___x_1571_ = l_Lean_Expr_app___override(v___x_1570_, v_00_u03b1_1548_);
v___x_1572_ = l_Lean_Expr_app___override(v___x_1571_, v_fst_1566_);
lean_inc_ref(v___x_1572_);
lean_inc_ref(v_a_1549_);
lean_inc(v_u_1547_);
v___x_1654_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_1547_, v_00_u03b1_1548_, v_a_1549_, v___x_1572_, v_ra_1551_);
if (lean_obj_tag(v___x_1654_) == 0)
{
lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v_a_1657_; lean_object* v___x_1659_; uint8_t v_isShared_1660_; uint8_t v_isSharedCheck_1664_; 
lean_dec_ref(v___x_1572_);
lean_dec(v_snd_1568_);
lean_dec(v_fst_1567_);
lean_del_object(v___x_1563_);
lean_dec_ref_known(v___x_1559_, 2);
lean_dec_ref(v_rb_1552_);
lean_dec_ref(v_b_1550_);
lean_dec_ref(v_a_1549_);
lean_dec_ref(v_00_u03b1_1548_);
lean_dec(v_u_1547_);
v___x_1655_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1656_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1655_, v_a_1553_, v_a_1554_, v_a_1555_, v_a_1556_);
v_a_1657_ = lean_ctor_get(v___x_1656_, 0);
v_isSharedCheck_1664_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1664_ == 0)
{
v___x_1659_ = v___x_1656_;
v_isShared_1660_ = v_isSharedCheck_1664_;
goto v_resetjp_1658_;
}
else
{
lean_inc(v_a_1657_);
lean_dec(v___x_1656_);
v___x_1659_ = lean_box(0);
v_isShared_1660_ = v_isSharedCheck_1664_;
goto v_resetjp_1658_;
}
v_resetjp_1658_:
{
lean_object* v___x_1662_; 
if (v_isShared_1660_ == 0)
{
v___x_1662_ = v___x_1659_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1663_; 
v_reuseFailAlloc_1663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1663_, 0, v_a_1657_);
v___x_1662_ = v_reuseFailAlloc_1663_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
return v___x_1662_;
}
}
}
else
{
lean_object* v_val_1665_; 
v_val_1665_ = lean_ctor_get(v___x_1654_, 0);
lean_inc(v_val_1665_);
lean_dec_ref_known(v___x_1654_, 1);
v_a_1635_ = v_val_1665_;
goto v___jp_1634_;
}
v___jp_1573_:
{
lean_object* v_snd_1579_; lean_object* v_snd_1580_; lean_object* v_fst_1581_; lean_object* v_fst_1582_; lean_object* v_fst_1583_; lean_object* v_snd_1584_; uint8_t v___x_1585_; 
v_snd_1579_ = lean_ctor_get(v_a_1578_, 1);
lean_inc(v_snd_1579_);
v_snd_1580_ = lean_ctor_get(v_snd_1579_, 1);
lean_inc(v_snd_1580_);
v_fst_1581_ = lean_ctor_get(v_a_1578_, 0);
lean_inc(v_fst_1581_);
lean_dec_ref(v_a_1578_);
v_fst_1582_ = lean_ctor_get(v_snd_1579_, 0);
lean_inc(v_fst_1582_);
lean_dec(v_snd_1579_);
v_fst_1583_ = lean_ctor_get(v_snd_1580_, 0);
lean_inc(v_fst_1583_);
v_snd_1584_ = lean_ctor_get(v_snd_1580_, 1);
lean_inc(v_snd_1584_);
lean_dec(v_snd_1580_);
v___x_1585_ = l_Rat_blt(v___y_1577_, v_fst_1581_);
if (v___x_1585_ == 0)
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1608_; 
v___x_1586_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1587_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__4));
lean_inc_ref(v___x_1559_);
v___x_1588_ = l_Lean_Expr_const___override(v___x_1587_, v___x_1559_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1589_ = l_Lean_Expr_app___override(v___x_1588_, v_00_u03b1_1548_);
v___x_1590_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5));
v___x_1591_ = l_Lean_Expr_const___override(v___x_1590_, v___x_1559_);
v___x_1592_ = l_Lean_Expr_app___override(v___x_1591_, v_00_u03b1_1548_);
v___x_1593_ = l_Lean_Expr_app___override(v___x_1592_, v___x_1572_);
v___x_1594_ = l_Lean_Expr_app___override(v___x_1589_, v___x_1593_);
v___x_1595_ = l_Lean_Expr_app___override(v___x_1594_, v_fst_1567_);
v___x_1596_ = l_Lean_Expr_app___override(v___x_1595_, v_snd_1568_);
v___x_1597_ = l_Lean_Expr_app___override(v___x_1596_, v_a_1549_);
v___x_1598_ = l_Lean_Expr_app___override(v___x_1597_, v_b_1550_);
v___x_1599_ = l_Lean_Expr_app___override(v___x_1598_, v___y_1575_);
v___x_1600_ = l_Lean_Expr_app___override(v___x_1599_, v_fst_1582_);
v___x_1601_ = l_Lean_Expr_app___override(v___x_1600_, v___y_1574_);
v___x_1602_ = l_Lean_Expr_app___override(v___x_1601_, v_fst_1583_);
v___x_1603_ = l_Lean_Expr_app___override(v___x_1602_, v___y_1576_);
v___x_1604_ = l_Lean_Expr_app___override(v___x_1603_, v_snd_1584_);
v___x_1605_ = l_Lean_Expr_app___override(v___x_1604_, v___x_1586_);
v___x_1606_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1606_, 0, v___x_1605_);
lean_ctor_set_uint8(v___x_1606_, sizeof(void*)*1, v___x_1585_);
if (v_isShared_1564_ == 0)
{
lean_ctor_set(v___x_1563_, 0, v___x_1606_);
v___x_1608_ = v___x_1563_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1609_; 
v_reuseFailAlloc_1609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1609_, 0, v___x_1606_);
v___x_1608_ = v_reuseFailAlloc_1609_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
return v___x_1608_;
}
}
else
{
lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1632_; 
v___x_1610_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1611_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__7));
lean_inc_ref(v___x_1559_);
v___x_1612_ = l_Lean_Expr_const___override(v___x_1611_, v___x_1559_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1613_ = l_Lean_Expr_app___override(v___x_1612_, v_00_u03b1_1548_);
v___x_1614_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___closed__5));
v___x_1615_ = l_Lean_Expr_const___override(v___x_1614_, v___x_1559_);
v___x_1616_ = l_Lean_Expr_app___override(v___x_1615_, v_00_u03b1_1548_);
v___x_1617_ = l_Lean_Expr_app___override(v___x_1616_, v___x_1572_);
v___x_1618_ = l_Lean_Expr_app___override(v___x_1613_, v___x_1617_);
v___x_1619_ = l_Lean_Expr_app___override(v___x_1618_, v_fst_1567_);
v___x_1620_ = l_Lean_Expr_app___override(v___x_1619_, v_snd_1568_);
v___x_1621_ = l_Lean_Expr_app___override(v___x_1620_, v_a_1549_);
v___x_1622_ = l_Lean_Expr_app___override(v___x_1621_, v_b_1550_);
v___x_1623_ = l_Lean_Expr_app___override(v___x_1622_, v___y_1575_);
v___x_1624_ = l_Lean_Expr_app___override(v___x_1623_, v_fst_1582_);
v___x_1625_ = l_Lean_Expr_app___override(v___x_1624_, v___y_1574_);
v___x_1626_ = l_Lean_Expr_app___override(v___x_1625_, v_fst_1583_);
v___x_1627_ = l_Lean_Expr_app___override(v___x_1626_, v___y_1576_);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1627_, v_snd_1584_);
v___x_1629_ = l_Lean_Expr_app___override(v___x_1628_, v___x_1610_);
v___x_1630_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1630_, 0, v___x_1629_);
lean_ctor_set_uint8(v___x_1630_, sizeof(void*)*1, v___x_1585_);
if (v_isShared_1564_ == 0)
{
lean_ctor_set(v___x_1563_, 0, v___x_1630_);
v___x_1632_ = v___x_1563_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v___x_1630_);
v___x_1632_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
return v___x_1632_;
}
}
}
v___jp_1634_:
{
lean_object* v_snd_1636_; lean_object* v_snd_1637_; lean_object* v_fst_1638_; lean_object* v_fst_1639_; lean_object* v_fst_1640_; lean_object* v_snd_1641_; lean_object* v___x_1642_; 
v_snd_1636_ = lean_ctor_get(v_a_1635_, 1);
lean_inc(v_snd_1636_);
v_snd_1637_ = lean_ctor_get(v_snd_1636_, 1);
lean_inc(v_snd_1637_);
v_fst_1638_ = lean_ctor_get(v_a_1635_, 0);
lean_inc(v_fst_1638_);
lean_dec_ref(v_a_1635_);
v_fst_1639_ = lean_ctor_get(v_snd_1636_, 0);
lean_inc(v_fst_1639_);
lean_dec(v_snd_1636_);
v_fst_1640_ = lean_ctor_get(v_snd_1637_, 0);
lean_inc(v_fst_1640_);
v_snd_1641_ = lean_ctor_get(v_snd_1637_, 1);
lean_inc(v_snd_1641_);
lean_dec(v_snd_1637_);
lean_inc_ref(v___x_1572_);
lean_inc_ref(v_b_1550_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1642_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(v_u_1547_, v_00_u03b1_1548_, v_b_1550_, v___x_1572_, v_rb_1552_);
if (lean_obj_tag(v___x_1642_) == 0)
{
lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v_a_1645_; lean_object* v___x_1647_; uint8_t v_isShared_1648_; uint8_t v_isSharedCheck_1652_; 
lean_dec(v_snd_1641_);
lean_dec(v_fst_1640_);
lean_dec(v_fst_1639_);
lean_dec(v_fst_1638_);
lean_dec_ref(v___x_1572_);
lean_dec(v_snd_1568_);
lean_dec(v_fst_1567_);
lean_del_object(v___x_1563_);
lean_dec_ref_known(v___x_1559_, 2);
lean_dec_ref(v_b_1550_);
lean_dec_ref(v_a_1549_);
lean_dec_ref(v_00_u03b1_1548_);
v___x_1643_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1644_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1643_, v_a_1553_, v_a_1554_, v_a_1555_, v_a_1556_);
v_a_1645_ = lean_ctor_get(v___x_1644_, 0);
v_isSharedCheck_1652_ = !lean_is_exclusive(v___x_1644_);
if (v_isSharedCheck_1652_ == 0)
{
v___x_1647_ = v___x_1644_;
v_isShared_1648_ = v_isSharedCheck_1652_;
goto v_resetjp_1646_;
}
else
{
lean_inc(v_a_1645_);
lean_dec(v___x_1644_);
v___x_1647_ = lean_box(0);
v_isShared_1648_ = v_isSharedCheck_1652_;
goto v_resetjp_1646_;
}
v_resetjp_1646_:
{
lean_object* v___x_1650_; 
if (v_isShared_1648_ == 0)
{
v___x_1650_ = v___x_1647_;
goto v_reusejp_1649_;
}
else
{
lean_object* v_reuseFailAlloc_1651_; 
v_reuseFailAlloc_1651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1651_, 0, v_a_1645_);
v___x_1650_ = v_reuseFailAlloc_1651_;
goto v_reusejp_1649_;
}
v_reusejp_1649_:
{
return v___x_1650_;
}
}
}
else
{
lean_object* v_val_1653_; 
v_val_1653_ = lean_ctor_get(v___x_1642_, 0);
lean_inc(v_val_1653_);
lean_dec_ref_known(v___x_1642_, 1);
v___y_1574_ = v_fst_1640_;
v___y_1575_ = v_fst_1639_;
v___y_1576_ = v_snd_1641_;
v___y_1577_ = v_fst_1638_;
v_a_1578_ = v_val_1653_;
goto v___jp_1573_;
}
}
}
}
else
{
lean_object* v_a_1667_; lean_object* v___x_1669_; uint8_t v_isShared_1670_; uint8_t v_isSharedCheck_1674_; 
lean_dec_ref_known(v___x_1559_, 2);
lean_dec_ref(v_rb_1552_);
lean_dec_ref(v_ra_1551_);
lean_dec_ref(v_b_1550_);
lean_dec_ref(v_a_1549_);
lean_dec_ref(v_00_u03b1_1548_);
lean_dec(v_u_1547_);
v_a_1667_ = lean_ctor_get(v___x_1560_, 0);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1560_);
if (v_isSharedCheck_1674_ == 0)
{
v___x_1669_ = v___x_1560_;
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
else
{
lean_inc(v_a_1667_);
lean_dec(v___x_1560_);
v___x_1669_ = lean_box(0);
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
v_resetjp_1668_:
{
lean_object* v___x_1672_; 
if (v_isShared_1670_ == 0)
{
v___x_1672_ = v___x_1669_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_a_1667_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg___boxed(lean_object* v_u_1675_, lean_object* v_00_u03b1_1676_, lean_object* v_a_1677_, lean_object* v_b_1678_, lean_object* v_ra_1679_, lean_object* v_rb_1680_, lean_object* v_a_1681_, lean_object* v_a_1682_, lean_object* v_a_1683_, lean_object* v_a_1684_, lean_object* v_a_1685_){
_start:
{
lean_object* v_res_1686_; 
v_res_1686_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(v_u_1675_, v_00_u03b1_1676_, v_a_1677_, v_b_1678_, v_ra_1679_, v_rb_1680_, v_a_1681_, v_a_1682_, v_a_1683_, v_a_1684_);
lean_dec(v_a_1684_);
lean_dec_ref(v_a_1683_);
lean_dec(v_a_1682_);
lean_dec_ref(v_a_1681_);
return v_res_1686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm(lean_object* v_u_1687_, lean_object* v_00_u03b1_1688_, lean_object* v_l_u03b1_1689_, lean_object* v_a_1690_, lean_object* v_b_1691_, lean_object* v_ra_1692_, lean_object* v_rb_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_){
_start:
{
lean_object* v___x_1699_; 
v___x_1699_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(v_u_1687_, v_00_u03b1_1688_, v_a_1690_, v_b_1691_, v_ra_1692_, v_rb_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
return v___x_1699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___boxed(lean_object* v_u_1700_, lean_object* v_00_u03b1_1701_, lean_object* v_l_u03b1_1702_, lean_object* v_a_1703_, lean_object* v_b_1704_, lean_object* v_ra_1705_, lean_object* v_rb_1706_, lean_object* v_a_1707_, lean_object* v_a_1708_, lean_object* v_a_1709_, lean_object* v_a_1710_, lean_object* v_a_1711_){
_start:
{
lean_object* v_res_1712_; 
v_res_1712_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm(v_u_1700_, v_00_u03b1_1701_, v_l_u03b1_1702_, v_a_1703_, v_b_1704_, v_ra_1705_, v_rb_1706_, v_a_1707_, v_a_1708_, v_a_1709_, v_a_1710_);
lean_dec(v_a_1710_);
lean_dec_ref(v_a_1709_);
lean_dec(v_a_1708_);
lean_dec_ref(v_a_1707_);
lean_dec_ref(v_l_u03b1_1702_);
return v_res_1712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(lean_object* v_u_1725_, lean_object* v_00_u03b1_1726_, lean_object* v_a_1727_, lean_object* v_b_1728_, lean_object* v_ra_1729_, lean_object* v_rb_1730_, lean_object* v_a_1731_, lean_object* v_a_1732_, lean_object* v_a_1733_, lean_object* v_a_1734_){
_start:
{
lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
v___x_1736_ = lean_box(0);
lean_inc_n(v_u_1725_, 2);
v___x_1737_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1737_, 0, v_u_1725_);
lean_ctor_set(v___x_1737_, 1, v___x_1736_);
lean_inc_ref(v_00_u03b1_1726_);
v___x_1738_ = lp_mathlib_Mathlib_Meta_NormNum_inferLinearOrderedField(v_u_1725_, v_00_u03b1_1726_, v_a_1731_, v_a_1732_, v_a_1733_, v_a_1734_);
if (lean_obj_tag(v___x_1738_) == 0)
{
lean_object* v_a_1739_; lean_object* v___x_1741_; uint8_t v_isShared_1742_; uint8_t v_isSharedCheck_1844_; 
v_a_1739_ = lean_ctor_get(v___x_1738_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1738_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1741_ = v___x_1738_;
v_isShared_1742_ = v_isSharedCheck_1844_;
goto v_resetjp_1740_;
}
else
{
lean_inc(v_a_1739_);
lean_dec(v___x_1738_);
v___x_1741_ = lean_box(0);
v_isShared_1742_ = v_isSharedCheck_1844_;
goto v_resetjp_1740_;
}
v_resetjp_1740_:
{
lean_object* v_snd_1743_; lean_object* v_fst_1744_; lean_object* v_fst_1745_; lean_object* v_snd_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___y_1752_; lean_object* v___y_1753_; lean_object* v___y_1754_; lean_object* v___y_1755_; lean_object* v_a_1756_; lean_object* v_a_1813_; lean_object* v___x_1832_; 
v_snd_1743_ = lean_ctor_get(v_a_1739_, 1);
lean_inc(v_snd_1743_);
v_fst_1744_ = lean_ctor_get(v_a_1739_, 0);
lean_inc(v_fst_1744_);
lean_dec(v_a_1739_);
v_fst_1745_ = lean_ctor_get(v_snd_1743_, 0);
lean_inc(v_fst_1745_);
v_snd_1746_ = lean_ctor_get(v_snd_1743_, 1);
lean_inc(v_snd_1746_);
lean_dec(v_snd_1743_);
v___x_1747_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__2));
lean_inc_ref(v___x_1737_);
v___x_1748_ = l_Lean_Expr_const___override(v___x_1747_, v___x_1737_);
lean_inc_ref_n(v_00_u03b1_1726_, 2);
v___x_1749_ = l_Lean_Expr_app___override(v___x_1748_, v_00_u03b1_1726_);
v___x_1750_ = l_Lean_Expr_app___override(v___x_1749_, v_fst_1744_);
lean_inc_ref(v___x_1750_);
lean_inc_ref(v_a_1727_);
lean_inc(v_u_1725_);
v___x_1832_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_1725_, v_00_u03b1_1726_, v_a_1727_, v___x_1750_, v_ra_1729_);
if (lean_obj_tag(v___x_1832_) == 0)
{
lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v_a_1835_; lean_object* v___x_1837_; uint8_t v_isShared_1838_; uint8_t v_isSharedCheck_1842_; 
lean_dec_ref(v___x_1750_);
lean_dec(v_snd_1746_);
lean_dec(v_fst_1745_);
lean_del_object(v___x_1741_);
lean_dec_ref_known(v___x_1737_, 2);
lean_dec_ref(v_rb_1730_);
lean_dec_ref(v_b_1728_);
lean_dec_ref(v_a_1727_);
lean_dec_ref(v_00_u03b1_1726_);
lean_dec(v_u_1725_);
v___x_1833_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1834_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1833_, v_a_1731_, v_a_1732_, v_a_1733_, v_a_1734_);
v_a_1835_ = lean_ctor_get(v___x_1834_, 0);
v_isSharedCheck_1842_ = !lean_is_exclusive(v___x_1834_);
if (v_isSharedCheck_1842_ == 0)
{
v___x_1837_ = v___x_1834_;
v_isShared_1838_ = v_isSharedCheck_1842_;
goto v_resetjp_1836_;
}
else
{
lean_inc(v_a_1835_);
lean_dec(v___x_1834_);
v___x_1837_ = lean_box(0);
v_isShared_1838_ = v_isSharedCheck_1842_;
goto v_resetjp_1836_;
}
v_resetjp_1836_:
{
lean_object* v___x_1840_; 
if (v_isShared_1838_ == 0)
{
v___x_1840_ = v___x_1837_;
goto v_reusejp_1839_;
}
else
{
lean_object* v_reuseFailAlloc_1841_; 
v_reuseFailAlloc_1841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1841_, 0, v_a_1835_);
v___x_1840_ = v_reuseFailAlloc_1841_;
goto v_reusejp_1839_;
}
v_reusejp_1839_:
{
return v___x_1840_;
}
}
}
else
{
lean_object* v_val_1843_; 
v_val_1843_ = lean_ctor_get(v___x_1832_, 0);
lean_inc(v_val_1843_);
lean_dec_ref_known(v___x_1832_, 1);
v_a_1813_ = v_val_1843_;
goto v___jp_1812_;
}
v___jp_1751_:
{
lean_object* v_snd_1757_; lean_object* v_snd_1758_; lean_object* v_fst_1759_; lean_object* v_fst_1760_; lean_object* v_fst_1761_; lean_object* v_snd_1762_; uint8_t v___x_1763_; 
v_snd_1757_ = lean_ctor_get(v_a_1756_, 1);
lean_inc(v_snd_1757_);
v_snd_1758_ = lean_ctor_get(v_snd_1757_, 1);
lean_inc(v_snd_1758_);
v_fst_1759_ = lean_ctor_get(v_a_1756_, 0);
lean_inc(v_fst_1759_);
lean_dec_ref(v_a_1756_);
v_fst_1760_ = lean_ctor_get(v_snd_1757_, 0);
lean_inc(v_fst_1760_);
lean_dec(v_snd_1757_);
v_fst_1761_ = lean_ctor_get(v_snd_1758_, 0);
lean_inc(v_fst_1761_);
v_snd_1762_ = lean_ctor_get(v_snd_1758_, 1);
lean_inc(v_snd_1762_);
lean_dec(v_snd_1758_);
v___x_1763_ = l_Rat_blt(v___y_1753_, v_fst_1759_);
if (v___x_1763_ == 0)
{
lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1786_; 
v___x_1764_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1765_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__1));
lean_inc_ref(v___x_1737_);
v___x_1766_ = l_Lean_Expr_const___override(v___x_1765_, v___x_1737_);
lean_inc_ref(v_00_u03b1_1726_);
v___x_1767_ = l_Lean_Expr_app___override(v___x_1766_, v_00_u03b1_1726_);
v___x_1768_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6));
v___x_1769_ = l_Lean_Expr_const___override(v___x_1768_, v___x_1737_);
v___x_1770_ = l_Lean_Expr_app___override(v___x_1769_, v_00_u03b1_1726_);
v___x_1771_ = l_Lean_Expr_app___override(v___x_1770_, v___x_1750_);
v___x_1772_ = l_Lean_Expr_app___override(v___x_1767_, v___x_1771_);
v___x_1773_ = l_Lean_Expr_app___override(v___x_1772_, v_fst_1745_);
v___x_1774_ = l_Lean_Expr_app___override(v___x_1773_, v_snd_1746_);
v___x_1775_ = l_Lean_Expr_app___override(v___x_1774_, v_a_1727_);
v___x_1776_ = l_Lean_Expr_app___override(v___x_1775_, v_b_1728_);
v___x_1777_ = l_Lean_Expr_app___override(v___x_1776_, v___y_1755_);
v___x_1778_ = l_Lean_Expr_app___override(v___x_1777_, v_fst_1760_);
v___x_1779_ = l_Lean_Expr_app___override(v___x_1778_, v___y_1754_);
v___x_1780_ = l_Lean_Expr_app___override(v___x_1779_, v_fst_1761_);
v___x_1781_ = l_Lean_Expr_app___override(v___x_1780_, v___y_1752_);
v___x_1782_ = l_Lean_Expr_app___override(v___x_1781_, v_snd_1762_);
v___x_1783_ = l_Lean_Expr_app___override(v___x_1782_, v___x_1764_);
v___x_1784_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1784_, 0, v___x_1783_);
lean_ctor_set_uint8(v___x_1784_, sizeof(void*)*1, v___x_1763_);
if (v_isShared_1742_ == 0)
{
lean_ctor_set(v___x_1741_, 0, v___x_1784_);
v___x_1786_ = v___x_1741_;
goto v_reusejp_1785_;
}
else
{
lean_object* v_reuseFailAlloc_1787_; 
v_reuseFailAlloc_1787_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1787_, 0, v___x_1784_);
v___x_1786_ = v_reuseFailAlloc_1787_;
goto v_reusejp_1785_;
}
v_reusejp_1785_:
{
return v___x_1786_;
}
}
else
{
lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1810_; 
v___x_1788_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1789_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___closed__3));
lean_inc_ref(v___x_1737_);
v___x_1790_ = l_Lean_Expr_const___override(v___x_1789_, v___x_1737_);
lean_inc_ref(v_00_u03b1_1726_);
v___x_1791_ = l_Lean_Expr_app___override(v___x_1790_, v_00_u03b1_1726_);
v___x_1792_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_ratArm___redArg___closed__6));
v___x_1793_ = l_Lean_Expr_const___override(v___x_1792_, v___x_1737_);
v___x_1794_ = l_Lean_Expr_app___override(v___x_1793_, v_00_u03b1_1726_);
v___x_1795_ = l_Lean_Expr_app___override(v___x_1794_, v___x_1750_);
v___x_1796_ = l_Lean_Expr_app___override(v___x_1791_, v___x_1795_);
v___x_1797_ = l_Lean_Expr_app___override(v___x_1796_, v_fst_1745_);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1797_, v_snd_1746_);
v___x_1799_ = l_Lean_Expr_app___override(v___x_1798_, v_a_1727_);
v___x_1800_ = l_Lean_Expr_app___override(v___x_1799_, v_b_1728_);
v___x_1801_ = l_Lean_Expr_app___override(v___x_1800_, v___y_1755_);
v___x_1802_ = l_Lean_Expr_app___override(v___x_1801_, v_fst_1760_);
v___x_1803_ = l_Lean_Expr_app___override(v___x_1802_, v___y_1754_);
v___x_1804_ = l_Lean_Expr_app___override(v___x_1803_, v_fst_1761_);
v___x_1805_ = l_Lean_Expr_app___override(v___x_1804_, v___y_1752_);
v___x_1806_ = l_Lean_Expr_app___override(v___x_1805_, v_snd_1762_);
v___x_1807_ = l_Lean_Expr_app___override(v___x_1806_, v___x_1788_);
v___x_1808_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1808_, 0, v___x_1807_);
lean_ctor_set_uint8(v___x_1808_, sizeof(void*)*1, v___x_1763_);
if (v_isShared_1742_ == 0)
{
lean_ctor_set(v___x_1741_, 0, v___x_1808_);
v___x_1810_ = v___x_1741_;
goto v_reusejp_1809_;
}
else
{
lean_object* v_reuseFailAlloc_1811_; 
v_reuseFailAlloc_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1811_, 0, v___x_1808_);
v___x_1810_ = v_reuseFailAlloc_1811_;
goto v_reusejp_1809_;
}
v_reusejp_1809_:
{
return v___x_1810_;
}
}
}
v___jp_1812_:
{
lean_object* v_snd_1814_; lean_object* v_snd_1815_; lean_object* v_fst_1816_; lean_object* v_fst_1817_; lean_object* v_fst_1818_; lean_object* v_snd_1819_; lean_object* v___x_1820_; 
v_snd_1814_ = lean_ctor_get(v_a_1813_, 1);
lean_inc(v_snd_1814_);
v_snd_1815_ = lean_ctor_get(v_snd_1814_, 1);
lean_inc(v_snd_1815_);
v_fst_1816_ = lean_ctor_get(v_a_1813_, 0);
lean_inc(v_fst_1816_);
lean_dec_ref(v_a_1813_);
v_fst_1817_ = lean_ctor_get(v_snd_1814_, 0);
lean_inc(v_fst_1817_);
lean_dec(v_snd_1814_);
v_fst_1818_ = lean_ctor_get(v_snd_1815_, 0);
lean_inc(v_fst_1818_);
v_snd_1819_ = lean_ctor_get(v_snd_1815_, 1);
lean_inc(v_snd_1819_);
lean_dec(v_snd_1815_);
lean_inc_ref(v___x_1750_);
lean_inc_ref(v_b_1728_);
lean_inc_ref(v_00_u03b1_1726_);
v___x_1820_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(v_u_1725_, v_00_u03b1_1726_, v_b_1728_, v___x_1750_, v_rb_1730_);
if (lean_obj_tag(v___x_1820_) == 0)
{
lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_1830_; 
lean_dec(v_snd_1819_);
lean_dec(v_fst_1818_);
lean_dec(v_fst_1817_);
lean_dec(v_fst_1816_);
lean_dec_ref(v___x_1750_);
lean_dec(v_snd_1746_);
lean_dec(v_fst_1745_);
lean_del_object(v___x_1741_);
lean_dec_ref_known(v___x_1737_, 2);
lean_dec_ref(v_b_1728_);
lean_dec_ref(v_a_1727_);
lean_dec_ref(v_00_u03b1_1726_);
v___x_1821_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1822_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1821_, v_a_1731_, v_a_1732_, v_a_1733_, v_a_1734_);
v_a_1823_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_1830_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_1830_ == 0)
{
v___x_1825_ = v___x_1822_;
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1822_);
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
else
{
lean_object* v_val_1831_; 
v_val_1831_ = lean_ctor_get(v___x_1820_, 0);
lean_inc(v_val_1831_);
lean_dec_ref_known(v___x_1820_, 1);
v___y_1752_ = v_snd_1819_;
v___y_1753_ = v_fst_1816_;
v___y_1754_ = v_fst_1818_;
v___y_1755_ = v_fst_1817_;
v_a_1756_ = v_val_1831_;
goto v___jp_1751_;
}
}
}
}
else
{
lean_object* v_a_1845_; lean_object* v___x_1847_; uint8_t v_isShared_1848_; uint8_t v_isSharedCheck_1852_; 
lean_dec_ref_known(v___x_1737_, 2);
lean_dec_ref(v_rb_1730_);
lean_dec_ref(v_ra_1729_);
lean_dec_ref(v_b_1728_);
lean_dec_ref(v_a_1727_);
lean_dec_ref(v_00_u03b1_1726_);
lean_dec(v_u_1725_);
v_a_1845_ = lean_ctor_get(v___x_1738_, 0);
v_isSharedCheck_1852_ = !lean_is_exclusive(v___x_1738_);
if (v_isSharedCheck_1852_ == 0)
{
v___x_1847_ = v___x_1738_;
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
else
{
lean_inc(v_a_1845_);
lean_dec(v___x_1738_);
v___x_1847_ = lean_box(0);
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
v_resetjp_1846_:
{
lean_object* v___x_1850_; 
if (v_isShared_1848_ == 0)
{
v___x_1850_ = v___x_1847_;
goto v_reusejp_1849_;
}
else
{
lean_object* v_reuseFailAlloc_1851_; 
v_reuseFailAlloc_1851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1851_, 0, v_a_1845_);
v___x_1850_ = v_reuseFailAlloc_1851_;
goto v_reusejp_1849_;
}
v_reusejp_1849_:
{
return v___x_1850_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg___boxed(lean_object* v_u_1853_, lean_object* v_00_u03b1_1854_, lean_object* v_a_1855_, lean_object* v_b_1856_, lean_object* v_ra_1857_, lean_object* v_rb_1858_, lean_object* v_a_1859_, lean_object* v_a_1860_, lean_object* v_a_1861_, lean_object* v_a_1862_, lean_object* v_a_1863_){
_start:
{
lean_object* v_res_1864_; 
v_res_1864_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1853_, v_00_u03b1_1854_, v_a_1855_, v_b_1856_, v_ra_1857_, v_rb_1858_, v_a_1859_, v_a_1860_, v_a_1861_, v_a_1862_);
lean_dec(v_a_1862_);
lean_dec_ref(v_a_1861_);
lean_dec(v_a_1860_);
lean_dec_ref(v_a_1859_);
return v_res_1864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm(lean_object* v_u_1865_, lean_object* v_00_u03b1_1866_, lean_object* v_l_u03b1_1867_, lean_object* v_a_1868_, lean_object* v_b_1869_, lean_object* v_ra_1870_, lean_object* v_rb_1871_, lean_object* v_a_1872_, lean_object* v_a_1873_, lean_object* v_a_1874_, lean_object* v_a_1875_){
_start:
{
lean_object* v___x_1877_; 
v___x_1877_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1865_, v_00_u03b1_1866_, v_a_1868_, v_b_1869_, v_ra_1870_, v_rb_1871_, v_a_1872_, v_a_1873_, v_a_1874_, v_a_1875_);
return v___x_1877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___boxed(lean_object* v_u_1878_, lean_object* v_00_u03b1_1879_, lean_object* v_l_u03b1_1880_, lean_object* v_a_1881_, lean_object* v_b_1882_, lean_object* v_ra_1883_, lean_object* v_rb_1884_, lean_object* v_a_1885_, lean_object* v_a_1886_, lean_object* v_a_1887_, lean_object* v_a_1888_, lean_object* v_a_1889_){
_start:
{
lean_object* v_res_1890_; 
v_res_1890_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm(v_u_1878_, v_00_u03b1_1879_, v_l_u03b1_1880_, v_a_1881_, v_b_1882_, v_ra_1883_, v_rb_1884_, v_a_1885_, v_a_1886_, v_a_1887_, v_a_1888_);
lean_dec(v_a_1888_);
lean_dec_ref(v_a_1887_);
lean_dec(v_a_1886_);
lean_dec_ref(v_a_1885_);
lean_dec_ref(v_l_u03b1_1880_);
return v_res_1890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg(lean_object* v_u_1903_, lean_object* v_00_u03b1_1904_, lean_object* v_a_1905_, lean_object* v_b_1906_, lean_object* v_ra_1907_, lean_object* v_rb_1908_, lean_object* v_a_1909_, lean_object* v_a_1910_, lean_object* v_a_1911_, lean_object* v_a_1912_){
_start:
{
lean_object* v___y_1915_; lean_object* v___y_1916_; lean_object* v___y_1917_; lean_object* v___y_1918_; 
switch(lean_obj_tag(v_ra_1907_))
{
case 0:
{
lean_object* v___x_1921_; lean_object* v___x_1922_; 
lean_dec_ref_known(v_ra_1907_, 1);
lean_dec_ref(v_rb_1908_);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v___x_1921_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1922_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1921_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_1922_;
}
case 1:
{
switch(lean_obj_tag(v_rb_1908_))
{
case 0:
{
lean_dec_ref_known(v_rb_1908_, 1);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v___y_1915_ = v_a_1909_;
v___y_1916_ = v_a_1910_;
v___y_1917_ = v_a_1911_;
v___y_1918_ = v_a_1912_;
goto v___jp_1914_;
}
case 1:
{
lean_object* v_lit_1923_; lean_object* v_proof_1924_; lean_object* v_inst_1925_; lean_object* v_lit_1926_; lean_object* v_proof_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; 
v_lit_1923_ = lean_ctor_get(v_ra_1907_, 1);
v_proof_1924_ = lean_ctor_get(v_ra_1907_, 2);
v_inst_1925_ = lean_ctor_get(v_rb_1908_, 0);
v_lit_1926_ = lean_ctor_get(v_rb_1908_, 1);
v_proof_1927_ = lean_ctor_get(v_rb_1908_, 2);
v___x_1928_ = lean_box(0);
lean_inc_n(v_u_1903_, 2);
v___x_1929_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1929_, 0, v_u_1903_);
lean_ctor_set(v___x_1929_, 1, v___x_1928_);
lean_inc_ref(v_00_u03b1_1904_);
v___x_1930_ = lp_mathlib_Mathlib_Meta_NormNum_inferOrderedSemiring(v_u_1903_, v_00_u03b1_1904_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
if (lean_obj_tag(v___x_1930_) == 0)
{
lean_object* v_a_1931_; lean_object* v___x_1933_; uint8_t v_isShared_1934_; uint8_t v_isSharedCheck_1999_; 
v_a_1931_ = lean_ctor_get(v___x_1930_, 0);
v_isSharedCheck_1999_ = !lean_is_exclusive(v___x_1930_);
if (v_isSharedCheck_1999_ == 0)
{
v___x_1933_ = v___x_1930_;
v_isShared_1934_ = v_isSharedCheck_1999_;
goto v_resetjp_1932_;
}
else
{
lean_inc(v_a_1931_);
lean_dec(v___x_1930_);
v___x_1933_ = lean_box(0);
v_isShared_1934_ = v_isSharedCheck_1999_;
goto v_resetjp_1932_;
}
v_resetjp_1932_:
{
lean_object* v_snd_1935_; lean_object* v_fst_1936_; lean_object* v_fst_1937_; lean_object* v_snd_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; uint8_t v___x_1941_; 
v_snd_1935_ = lean_ctor_get(v_a_1931_, 1);
lean_inc(v_snd_1935_);
v_fst_1936_ = lean_ctor_get(v_a_1931_, 0);
lean_inc(v_fst_1936_);
lean_dec(v_a_1931_);
v_fst_1937_ = lean_ctor_get(v_snd_1935_, 0);
lean_inc(v_fst_1937_);
v_snd_1938_ = lean_ctor_get(v_snd_1935_, 1);
lean_inc(v_snd_1938_);
lean_dec(v_snd_1935_);
v___x_1939_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1923_);
v___x_1940_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1926_);
v___x_1941_ = lean_nat_dec_lt(v___x_1939_, v___x_1940_);
lean_dec(v___x_1940_);
lean_dec(v___x_1939_);
if (v___x_1941_ == 0)
{
lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1958_; 
lean_inc_ref(v_proof_1927_);
lean_inc_ref(v_lit_1926_);
lean_inc_ref(v_proof_1924_);
lean_inc_ref(v_lit_1923_);
lean_dec_ref_known(v_rb_1908_, 3);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec(v_u_1903_);
v___x_1942_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__15);
v___x_1943_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__1));
v___x_1944_ = l_Lean_Expr_const___override(v___x_1943_, v___x_1929_);
v___x_1945_ = l_Lean_Expr_app___override(v___x_1944_, v_00_u03b1_1904_);
v___x_1946_ = l_Lean_Expr_app___override(v___x_1945_, v_fst_1936_);
v___x_1947_ = l_Lean_Expr_app___override(v___x_1946_, v_fst_1937_);
v___x_1948_ = l_Lean_Expr_app___override(v___x_1947_, v_snd_1938_);
v___x_1949_ = l_Lean_Expr_app___override(v___x_1948_, v_a_1905_);
v___x_1950_ = l_Lean_Expr_app___override(v___x_1949_, v_b_1906_);
v___x_1951_ = l_Lean_Expr_app___override(v___x_1950_, v_lit_1923_);
v___x_1952_ = l_Lean_Expr_app___override(v___x_1951_, v_lit_1926_);
v___x_1953_ = l_Lean_Expr_app___override(v___x_1952_, v_proof_1924_);
v___x_1954_ = l_Lean_Expr_app___override(v___x_1953_, v_proof_1927_);
v___x_1955_ = l_Lean_Expr_app___override(v___x_1954_, v___x_1942_);
v___x_1956_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1956_, 0, v___x_1955_);
lean_ctor_set_uint8(v___x_1956_, sizeof(void*)*1, v___x_1941_);
if (v_isShared_1934_ == 0)
{
lean_ctor_set(v___x_1933_, 0, v___x_1956_);
v___x_1958_ = v___x_1933_;
goto v_reusejp_1957_;
}
else
{
lean_object* v_reuseFailAlloc_1959_; 
v_reuseFailAlloc_1959_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1959_, 0, v___x_1956_);
v___x_1958_ = v_reuseFailAlloc_1959_;
goto v_reusejp_1957_;
}
v_reusejp_1957_:
{
return v___x_1958_;
}
}
else
{
lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; 
lean_del_object(v___x_1933_);
v___x_1960_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__1));
lean_inc_ref(v___x_1929_);
v___x_1961_ = l_Lean_Expr_const___override(v___x_1960_, v___x_1929_);
lean_inc_ref(v_00_u03b1_1904_);
v___x_1962_ = l_Lean_Expr_app___override(v___x_1961_, v_00_u03b1_1904_);
lean_inc_ref(v_inst_1925_);
v___x_1963_ = l_Lean_Expr_app___override(v___x_1962_, v_inst_1925_);
v___x_1964_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_1963_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
if (lean_obj_tag(v___x_1964_) == 0)
{
lean_object* v_a_1965_; lean_object* v___x_1967_; uint8_t v_isShared_1968_; uint8_t v_isSharedCheck_1990_; 
v_a_1965_ = lean_ctor_get(v___x_1964_, 0);
v_isSharedCheck_1990_ = !lean_is_exclusive(v___x_1964_);
if (v_isSharedCheck_1990_ == 0)
{
v___x_1967_ = v___x_1964_;
v_isShared_1968_ = v_isSharedCheck_1990_;
goto v_resetjp_1966_;
}
else
{
lean_inc(v_a_1965_);
lean_dec(v___x_1964_);
v___x_1967_ = lean_box(0);
v_isShared_1968_ = v_isSharedCheck_1990_;
goto v_resetjp_1966_;
}
v_resetjp_1966_:
{
if (lean_obj_tag(v_a_1965_) == 1)
{
lean_object* v_a_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1987_; 
lean_inc_ref(v_proof_1927_);
lean_inc_ref(v_lit_1926_);
lean_inc_ref(v_proof_1924_);
lean_inc_ref(v_lit_1923_);
lean_dec_ref_known(v_rb_1908_, 3);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec(v_u_1903_);
v_a_1969_ = lean_ctor_get(v_a_1965_, 0);
lean_inc(v_a_1969_);
lean_dec_ref_known(v_a_1965_, 1);
v___x_1970_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_evalLE_core___redArg___closed__5);
v___x_1971_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___closed__3));
v___x_1972_ = l_Lean_Expr_const___override(v___x_1971_, v___x_1929_);
v___x_1973_ = l_Lean_Expr_app___override(v___x_1972_, v_00_u03b1_1904_);
v___x_1974_ = l_Lean_Expr_app___override(v___x_1973_, v_fst_1936_);
v___x_1975_ = l_Lean_Expr_app___override(v___x_1974_, v_fst_1937_);
v___x_1976_ = l_Lean_Expr_app___override(v___x_1975_, v_snd_1938_);
v___x_1977_ = l_Lean_Expr_app___override(v___x_1976_, v_a_1969_);
v___x_1978_ = l_Lean_Expr_app___override(v___x_1977_, v_a_1905_);
v___x_1979_ = l_Lean_Expr_app___override(v___x_1978_, v_b_1906_);
v___x_1980_ = l_Lean_Expr_app___override(v___x_1979_, v_lit_1923_);
v___x_1981_ = l_Lean_Expr_app___override(v___x_1980_, v_lit_1926_);
v___x_1982_ = l_Lean_Expr_app___override(v___x_1981_, v_proof_1924_);
v___x_1983_ = l_Lean_Expr_app___override(v___x_1982_, v_proof_1927_);
v___x_1984_ = l_Lean_Expr_app___override(v___x_1983_, v___x_1970_);
v___x_1985_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1985_, 0, v___x_1984_);
lean_ctor_set_uint8(v___x_1985_, sizeof(void*)*1, v___x_1941_);
if (v_isShared_1968_ == 0)
{
lean_ctor_set(v___x_1967_, 0, v___x_1985_);
v___x_1987_ = v___x_1967_;
goto v_reusejp_1986_;
}
else
{
lean_object* v_reuseFailAlloc_1988_; 
v_reuseFailAlloc_1988_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1988_, 0, v___x_1985_);
v___x_1987_ = v_reuseFailAlloc_1988_;
goto v_reusejp_1986_;
}
v_reusejp_1986_:
{
return v___x_1987_;
}
}
else
{
lean_object* v___x_1989_; 
lean_del_object(v___x_1967_);
lean_dec(v_a_1965_);
lean_dec(v_snd_1938_);
lean_dec(v_fst_1937_);
lean_dec(v_fst_1936_);
lean_dec_ref_known(v___x_1929_, 2);
v___x_1989_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_1989_;
}
}
}
else
{
lean_object* v_a_1991_; lean_object* v___x_1993_; uint8_t v_isShared_1994_; uint8_t v_isSharedCheck_1998_; 
lean_dec(v_snd_1938_);
lean_dec(v_fst_1937_);
lean_dec(v_fst_1936_);
lean_dec_ref_known(v___x_1929_, 2);
lean_dec_ref_known(v_rb_1908_, 3);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v_a_1991_ = lean_ctor_get(v___x_1964_, 0);
v_isSharedCheck_1998_ = !lean_is_exclusive(v___x_1964_);
if (v_isSharedCheck_1998_ == 0)
{
v___x_1993_ = v___x_1964_;
v_isShared_1994_ = v_isSharedCheck_1998_;
goto v_resetjp_1992_;
}
else
{
lean_inc(v_a_1991_);
lean_dec(v___x_1964_);
v___x_1993_ = lean_box(0);
v_isShared_1994_ = v_isSharedCheck_1998_;
goto v_resetjp_1992_;
}
v_resetjp_1992_:
{
lean_object* v___x_1996_; 
if (v_isShared_1994_ == 0)
{
v___x_1996_ = v___x_1993_;
goto v_reusejp_1995_;
}
else
{
lean_object* v_reuseFailAlloc_1997_; 
v_reuseFailAlloc_1997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1997_, 0, v_a_1991_);
v___x_1996_ = v_reuseFailAlloc_1997_;
goto v_reusejp_1995_;
}
v_reusejp_1995_:
{
return v___x_1996_;
}
}
}
}
}
}
else
{
lean_object* v_a_2000_; lean_object* v___x_2002_; uint8_t v_isShared_2003_; uint8_t v_isSharedCheck_2007_; 
lean_dec_ref_known(v___x_1929_, 2);
lean_dec_ref_known(v_rb_1908_, 3);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v_a_2000_ = lean_ctor_get(v___x_1930_, 0);
v_isSharedCheck_2007_ = !lean_is_exclusive(v___x_1930_);
if (v_isSharedCheck_2007_ == 0)
{
v___x_2002_ = v___x_1930_;
v_isShared_2003_ = v_isSharedCheck_2007_;
goto v_resetjp_2001_;
}
else
{
lean_inc(v_a_2000_);
lean_dec(v___x_1930_);
v___x_2002_ = lean_box(0);
v_isShared_2003_ = v_isSharedCheck_2007_;
goto v_resetjp_2001_;
}
v_resetjp_2001_:
{
lean_object* v___x_2005_; 
if (v_isShared_2003_ == 0)
{
v___x_2005_ = v___x_2002_;
goto v_reusejp_2004_;
}
else
{
lean_object* v_reuseFailAlloc_2006_; 
v_reuseFailAlloc_2006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2006_, 0, v_a_2000_);
v___x_2005_ = v_reuseFailAlloc_2006_;
goto v_reusejp_2004_;
}
v_reusejp_2004_:
{
return v___x_2005_;
}
}
}
}
case 2:
{
lean_object* v___x_2008_; 
v___x_2008_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2008_;
}
case 3:
{
lean_object* v___x_2009_; 
v___x_2009_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2009_;
}
default: 
{
lean_object* v___x_2010_; 
v___x_2010_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2010_;
}
}
}
case 2:
{
switch(lean_obj_tag(v_rb_1908_))
{
case 0:
{
lean_dec_ref_known(v_rb_1908_, 1);
lean_dec_ref_known(v_ra_1907_, 3);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v___y_1915_ = v_a_1909_;
v___y_1916_ = v_a_1910_;
v___y_1917_ = v_a_1911_;
v___y_1918_ = v_a_1912_;
goto v___jp_1914_;
}
case 4:
{
lean_object* v___x_2011_; 
v___x_2011_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2011_;
}
case 3:
{
lean_object* v___x_2012_; 
v___x_2012_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2012_;
}
case 2:
{
lean_object* v___x_2013_; 
v___x_2013_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2013_;
}
default: 
{
lean_object* v___x_2014_; 
v___x_2014_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_intArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2014_;
}
}
}
case 3:
{
switch(lean_obj_tag(v_rb_1908_))
{
case 0:
{
lean_dec_ref_known(v_rb_1908_, 1);
lean_dec_ref_known(v_ra_1907_, 5);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v___y_1915_ = v_a_1909_;
v___y_1916_ = v_a_1910_;
v___y_1917_ = v_a_1911_;
v___y_1918_ = v_a_1912_;
goto v___jp_1914_;
}
case 4:
{
lean_object* v___x_2015_; 
v___x_2015_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2015_;
}
case 2:
{
lean_object* v___x_2016_; 
v___x_2016_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2016_;
}
case 3:
{
lean_object* v___x_2017_; 
v___x_2017_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2017_;
}
default: 
{
lean_object* v___x_2018_; 
v___x_2018_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_nnratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2018_;
}
}
}
default: 
{
switch(lean_obj_tag(v_rb_1908_))
{
case 0:
{
lean_dec_ref_known(v_rb_1908_, 1);
lean_dec_ref_known(v_ra_1907_, 5);
lean_dec_ref(v_b_1906_);
lean_dec_ref(v_a_1905_);
lean_dec_ref(v_00_u03b1_1904_);
lean_dec(v_u_1903_);
v___y_1915_ = v_a_1909_;
v___y_1916_ = v_a_1910_;
v___y_1917_ = v_a_1911_;
v___y_1918_ = v_a_1912_;
goto v___jp_1914_;
}
case 4:
{
lean_object* v___x_2019_; 
v___x_2019_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2019_;
}
case 3:
{
lean_object* v___x_2020_; 
v___x_2020_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2020_;
}
case 2:
{
lean_object* v___x_2021_; 
v___x_2021_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2021_;
}
default: 
{
lean_object* v___x_2022_; 
v___x_2022_ = lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLT_core_ratArm___redArg(v_u_1903_, v_00_u03b1_1904_, v_a_1905_, v_b_1906_, v_ra_1907_, v_rb_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
return v___x_2022_;
}
}
}
}
v___jp_1914_:
{
lean_object* v___x_1919_; lean_object* v___x_1920_; 
v___x_1919_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_1920_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_1919_, v___y_1915_, v___y_1916_, v___y_1917_, v___y_1918_);
return v___x_1920_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg___boxed(lean_object* v_u_2023_, lean_object* v_00_u03b1_2024_, lean_object* v_a_2025_, lean_object* v_b_2026_, lean_object* v_ra_2027_, lean_object* v_rb_2028_, lean_object* v_a_2029_, lean_object* v_a_2030_, lean_object* v_a_2031_, lean_object* v_a_2032_, lean_object* v_a_2033_){
_start:
{
lean_object* v_res_2034_; 
v_res_2034_ = lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg(v_u_2023_, v_00_u03b1_2024_, v_a_2025_, v_b_2026_, v_ra_2027_, v_rb_2028_, v_a_2029_, v_a_2030_, v_a_2031_, v_a_2032_);
lean_dec(v_a_2032_);
lean_dec_ref(v_a_2031_);
lean_dec(v_a_2030_);
lean_dec_ref(v_a_2029_);
return v_res_2034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core(lean_object* v_u_2035_, lean_object* v_00_u03b1_2036_, lean_object* v_l_u03b1_2037_, lean_object* v_a_2038_, lean_object* v_b_2039_, lean_object* v_ra_2040_, lean_object* v_rb_2041_, lean_object* v_a_2042_, lean_object* v_a_2043_, lean_object* v_a_2044_, lean_object* v_a_2045_){
_start:
{
lean_object* v___x_2047_; 
v___x_2047_ = lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg(v_u_2035_, v_00_u03b1_2036_, v_a_2038_, v_b_2039_, v_ra_2040_, v_rb_2041_, v_a_2042_, v_a_2043_, v_a_2044_, v_a_2045_);
return v___x_2047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___boxed(lean_object* v_u_2048_, lean_object* v_00_u03b1_2049_, lean_object* v_l_u03b1_2050_, lean_object* v_a_2051_, lean_object* v_b_2052_, lean_object* v_ra_2053_, lean_object* v_rb_2054_, lean_object* v_a_2055_, lean_object* v_a_2056_, lean_object* v_a_2057_, lean_object* v_a_2058_, lean_object* v_a_2059_){
_start:
{
lean_object* v_res_2060_; 
v_res_2060_ = lp_mathlib_Mathlib_Meta_NormNum_evalLT_core(v_u_2048_, v_00_u03b1_2049_, v_l_u03b1_2050_, v_a_2051_, v_b_2052_, v_ra_2053_, v_rb_2054_, v_a_2055_, v_a_2056_, v_a_2057_, v_a_2058_);
lean_dec(v_a_2058_);
lean_dec_ref(v_a_2057_);
lean_dec(v_a_2056_);
lean_dec_ref(v_a_2055_);
lean_dec_ref(v_l_u03b1_2050_);
return v_res_2060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1(lean_object* v_v_2068_, lean_object* v_00_u03b2_2069_, lean_object* v_e_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_){
_start:
{
lean_object* v___x_2076_; 
v___x_2076_ = l_Lean_Meta_whnfR(v_e_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2076_) == 0)
{
lean_object* v_a_2077_; lean_object* v___y_2079_; lean_object* v___y_2080_; lean_object* v___y_2081_; lean_object* v___y_2082_; 
v_a_2077_ = lean_ctor_get(v___x_2076_, 0);
lean_inc(v_a_2077_);
lean_dec_ref_known(v___x_2076_, 1);
if (lean_obj_tag(v_a_2077_) == 5)
{
lean_object* v_fn_2085_; 
v_fn_2085_ = lean_ctor_get(v_a_2077_, 0);
lean_inc_ref(v_fn_2085_);
if (lean_obj_tag(v_fn_2085_) == 5)
{
lean_object* v_arg_2086_; lean_object* v_fn_2087_; lean_object* v_arg_2088_; lean_object* v___x_2089_; 
v_arg_2086_ = lean_ctor_get(v_a_2077_, 1);
lean_inc_ref(v_arg_2086_);
lean_dec_ref_known(v_a_2077_, 2);
v_fn_2087_ = lean_ctor_get(v_fn_2085_, 0);
lean_inc_ref(v_fn_2087_);
v_arg_2088_ = lean_ctor_get(v_fn_2085_, 1);
lean_inc_ref(v_arg_2088_);
lean_dec_ref_known(v_fn_2085_, 2);
v___x_2089_ = lp_mathlib_Qq_inferTypeQ_x27(v_arg_2088_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_object* v_a_2090_; lean_object* v_snd_2091_; lean_object* v_fst_2092_; lean_object* v_fst_2093_; lean_object* v_snd_2094_; lean_object* v___x_2096_; uint8_t v_isShared_2097_; uint8_t v_isSharedCheck_2147_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc(v_a_2090_);
lean_dec_ref_known(v___x_2089_, 1);
v_snd_2091_ = lean_ctor_get(v_a_2090_, 1);
lean_inc(v_snd_2091_);
v_fst_2092_ = lean_ctor_get(v_a_2090_, 0);
lean_inc(v_fst_2092_);
lean_dec(v_a_2090_);
v_fst_2093_ = lean_ctor_get(v_snd_2091_, 0);
v_snd_2094_ = lean_ctor_get(v_snd_2091_, 1);
v_isSharedCheck_2147_ = !lean_is_exclusive(v_snd_2091_);
if (v_isSharedCheck_2147_ == 0)
{
v___x_2096_ = v_snd_2091_;
v_isShared_2097_ = v_isSharedCheck_2147_;
goto v_resetjp_2095_;
}
else
{
lean_inc(v_snd_2094_);
lean_inc(v_fst_2093_);
lean_dec(v_snd_2091_);
v___x_2096_ = lean_box(0);
v_isShared_2097_ = v_isSharedCheck_2147_;
goto v_resetjp_2095_;
}
v_resetjp_2095_:
{
uint8_t v___x_2098_; lean_object* v___x_2099_; 
v___x_2098_ = 0;
lean_inc(v_snd_2094_);
lean_inc(v_fst_2093_);
lean_inc(v_fst_2092_);
v___x_2099_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_2092_, v_fst_2093_, v_snd_2094_, v___x_2098_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2099_) == 0)
{
lean_object* v_a_2100_; lean_object* v___x_2101_; 
v_a_2100_ = lean_ctor_get(v___x_2099_, 0);
lean_inc(v_a_2100_);
lean_dec_ref_known(v___x_2099_, 1);
lean_inc_ref(v_arg_2086_);
lean_inc(v_fst_2093_);
lean_inc(v_fst_2092_);
v___x_2101_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_fst_2092_, v_fst_2093_, v_arg_2086_, v___x_2098_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2101_) == 0)
{
lean_object* v_a_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2106_; 
v_a_2102_ = lean_ctor_get(v___x_2101_, 0);
lean_inc(v_a_2102_);
lean_dec_ref_known(v___x_2101_, 1);
v___x_2103_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__1));
v___x_2104_ = lean_box(0);
lean_inc(v_fst_2092_);
if (v_isShared_2097_ == 0)
{
lean_ctor_set_tag(v___x_2096_, 1);
lean_ctor_set(v___x_2096_, 1, v___x_2104_);
lean_ctor_set(v___x_2096_, 0, v_fst_2092_);
v___x_2106_ = v___x_2096_;
goto v_reusejp_2105_;
}
else
{
lean_object* v_reuseFailAlloc_2146_; 
v_reuseFailAlloc_2146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2146_, 0, v_fst_2092_);
lean_ctor_set(v_reuseFailAlloc_2146_, 1, v___x_2104_);
v___x_2106_ = v_reuseFailAlloc_2146_;
goto v_reusejp_2105_;
}
v_reusejp_2105_:
{
lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; 
lean_inc_ref(v___x_2106_);
v___x_2107_ = l_Lean_Expr_const___override(v___x_2103_, v___x_2106_);
lean_inc(v_fst_2093_);
v___x_2108_ = l_Lean_Expr_app___override(v___x_2107_, v_fst_2093_);
v___x_2109_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_2108_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2109_) == 0)
{
lean_object* v_a_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___f_2115_; lean_object* v___x_2116_; 
v_a_2110_ = lean_ctor_get(v___x_2109_, 0);
lean_inc(v_a_2110_);
lean_dec_ref_known(v___x_2109_, 1);
v___x_2111_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___closed__3));
v___x_2112_ = l_Lean_Expr_const___override(v___x_2111_, v___x_2106_);
lean_inc(v_fst_2093_);
v___x_2113_ = l_Lean_Expr_app___override(v___x_2112_, v_fst_2093_);
v___x_2114_ = l_Lean_Expr_app___override(v___x_2113_, v_a_2110_);
v___f_2115_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_evalLE___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2115_, 0, v_fn_2087_);
lean_closure_set(v___f_2115_, 1, v___x_2114_);
v___x_2116_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Meta_NormNum_evalLE_spec__0___redArg(v___f_2115_, v___x_2098_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
if (lean_obj_tag(v___x_2116_) == 0)
{
lean_object* v_a_2117_; uint8_t v___x_2118_; 
v_a_2117_ = lean_ctor_get(v___x_2116_, 0);
lean_inc(v_a_2117_);
lean_dec_ref_known(v___x_2116_, 1);
v___x_2118_ = lean_unbox(v_a_2117_);
lean_dec(v_a_2117_);
if (v___x_2118_ == 0)
{
lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v_a_2121_; lean_object* v___x_2123_; uint8_t v_isShared_2124_; uint8_t v_isSharedCheck_2128_; 
lean_dec(v_a_2102_);
lean_dec(v_a_2100_);
lean_dec(v_snd_2094_);
lean_dec(v_fst_2093_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_arg_2086_);
v___x_2119_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_2120_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_2119_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
v_a_2121_ = lean_ctor_get(v___x_2120_, 0);
v_isSharedCheck_2128_ = !lean_is_exclusive(v___x_2120_);
if (v_isSharedCheck_2128_ == 0)
{
v___x_2123_ = v___x_2120_;
v_isShared_2124_ = v_isSharedCheck_2128_;
goto v_resetjp_2122_;
}
else
{
lean_inc(v_a_2121_);
lean_dec(v___x_2120_);
v___x_2123_ = lean_box(0);
v_isShared_2124_ = v_isSharedCheck_2128_;
goto v_resetjp_2122_;
}
v_resetjp_2122_:
{
lean_object* v___x_2126_; 
if (v_isShared_2124_ == 0)
{
v___x_2126_ = v___x_2123_;
goto v_reusejp_2125_;
}
else
{
lean_object* v_reuseFailAlloc_2127_; 
v_reuseFailAlloc_2127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2127_, 0, v_a_2121_);
v___x_2126_ = v_reuseFailAlloc_2127_;
goto v_reusejp_2125_;
}
v_reusejp_2125_:
{
return v___x_2126_;
}
}
}
else
{
lean_object* v___x_2129_; 
v___x_2129_ = lp_mathlib_Mathlib_Meta_NormNum_evalLT_core___redArg(v_fst_2092_, v_fst_2093_, v_snd_2094_, v_arg_2086_, v_a_2100_, v_a_2102_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
return v___x_2129_;
}
}
else
{
lean_object* v_a_2130_; lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2137_; 
lean_dec(v_a_2102_);
lean_dec(v_a_2100_);
lean_dec(v_snd_2094_);
lean_dec(v_fst_2093_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_arg_2086_);
v_a_2130_ = lean_ctor_get(v___x_2116_, 0);
v_isSharedCheck_2137_ = !lean_is_exclusive(v___x_2116_);
if (v_isSharedCheck_2137_ == 0)
{
v___x_2132_ = v___x_2116_;
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
else
{
lean_inc(v_a_2130_);
lean_dec(v___x_2116_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
lean_object* v___x_2135_; 
if (v_isShared_2133_ == 0)
{
v___x_2135_ = v___x_2132_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2136_; 
v_reuseFailAlloc_2136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2136_, 0, v_a_2130_);
v___x_2135_ = v_reuseFailAlloc_2136_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
return v___x_2135_;
}
}
}
}
else
{
lean_object* v_a_2138_; lean_object* v___x_2140_; uint8_t v_isShared_2141_; uint8_t v_isSharedCheck_2145_; 
lean_dec_ref(v___x_2106_);
lean_dec(v_a_2102_);
lean_dec(v_a_2100_);
lean_dec(v_snd_2094_);
lean_dec(v_fst_2093_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_fn_2087_);
lean_dec_ref(v_arg_2086_);
v_a_2138_ = lean_ctor_get(v___x_2109_, 0);
v_isSharedCheck_2145_ = !lean_is_exclusive(v___x_2109_);
if (v_isSharedCheck_2145_ == 0)
{
v___x_2140_ = v___x_2109_;
v_isShared_2141_ = v_isSharedCheck_2145_;
goto v_resetjp_2139_;
}
else
{
lean_inc(v_a_2138_);
lean_dec(v___x_2109_);
v___x_2140_ = lean_box(0);
v_isShared_2141_ = v_isSharedCheck_2145_;
goto v_resetjp_2139_;
}
v_resetjp_2139_:
{
lean_object* v___x_2143_; 
if (v_isShared_2141_ == 0)
{
v___x_2143_ = v___x_2140_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2144_; 
v_reuseFailAlloc_2144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2144_, 0, v_a_2138_);
v___x_2143_ = v_reuseFailAlloc_2144_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
return v___x_2143_;
}
}
}
}
}
else
{
lean_dec(v_a_2100_);
lean_del_object(v___x_2096_);
lean_dec(v_snd_2094_);
lean_dec(v_fst_2093_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_fn_2087_);
lean_dec_ref(v_arg_2086_);
return v___x_2101_;
}
}
else
{
lean_del_object(v___x_2096_);
lean_dec(v_snd_2094_);
lean_dec(v_fst_2093_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_fn_2087_);
lean_dec_ref(v_arg_2086_);
return v___x_2099_;
}
}
}
else
{
lean_object* v_a_2148_; lean_object* v___x_2150_; uint8_t v_isShared_2151_; uint8_t v_isSharedCheck_2155_; 
lean_dec_ref(v_fn_2087_);
lean_dec_ref(v_arg_2086_);
v_a_2148_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2155_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2155_ == 0)
{
v___x_2150_ = v___x_2089_;
v_isShared_2151_ = v_isSharedCheck_2155_;
goto v_resetjp_2149_;
}
else
{
lean_inc(v_a_2148_);
lean_dec(v___x_2089_);
v___x_2150_ = lean_box(0);
v_isShared_2151_ = v_isSharedCheck_2155_;
goto v_resetjp_2149_;
}
v_resetjp_2149_:
{
lean_object* v___x_2153_; 
if (v_isShared_2151_ == 0)
{
v___x_2153_ = v___x_2150_;
goto v_reusejp_2152_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v_a_2148_);
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
else
{
lean_dec_ref(v_fn_2085_);
lean_dec_ref_known(v_a_2077_, 2);
v___y_2079_ = v___y_2071_;
v___y_2080_ = v___y_2072_;
v___y_2081_ = v___y_2073_;
v___y_2082_ = v___y_2074_;
goto v___jp_2078_;
}
}
else
{
lean_dec(v_a_2077_);
v___y_2079_ = v___y_2071_;
v___y_2080_ = v___y_2072_;
v___y_2081_ = v___y_2073_;
v___y_2082_ = v___y_2074_;
goto v___jp_2078_;
}
v___jp_2078_:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; 
v___x_2083_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22, &lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_NormNum_Ineq_0__Mathlib_Meta_NormNum_evalLE_core_intArm___redArg___closed__22);
v___x_2084_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferOrderedSemiring_spec__0___redArg(v___x_2083_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_);
return v___x_2084_;
}
}
else
{
lean_object* v_a_2156_; lean_object* v___x_2158_; uint8_t v_isShared_2159_; uint8_t v_isSharedCheck_2163_; 
v_a_2156_ = lean_ctor_get(v___x_2076_, 0);
v_isSharedCheck_2163_ = !lean_is_exclusive(v___x_2076_);
if (v_isSharedCheck_2163_ == 0)
{
v___x_2158_ = v___x_2076_;
v_isShared_2159_ = v_isSharedCheck_2163_;
goto v_resetjp_2157_;
}
else
{
lean_inc(v_a_2156_);
lean_dec(v___x_2076_);
v___x_2158_ = lean_box(0);
v_isShared_2159_ = v_isSharedCheck_2163_;
goto v_resetjp_2157_;
}
v_resetjp_2157_:
{
lean_object* v___x_2161_; 
if (v_isShared_2159_ == 0)
{
v___x_2161_ = v___x_2158_;
goto v_reusejp_2160_;
}
else
{
lean_object* v_reuseFailAlloc_2162_; 
v_reuseFailAlloc_2162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2162_, 0, v_a_2156_);
v___x_2161_ = v_reuseFailAlloc_2162_;
goto v_reusejp_2160_;
}
v_reusejp_2160_:
{
return v___x_2161_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1___boxed(lean_object* v_v_2164_, lean_object* v_00_u03b2_2165_, lean_object* v_e_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_){
_start:
{
lean_object* v_res_2172_; 
v_res_2172_ = lp_mathlib_Mathlib_Meta_NormNum_evalLT___lam__1(v_v_2164_, v_00_u03b2_2165_, v_e_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v_00_u03b2_2165_);
lean_dec(v_v_2164_);
return v_res_2172_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Invertible(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Cast(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Cast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Cast(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Cast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
}
#ifdef __cplusplus
}
#endif
